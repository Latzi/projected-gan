# training_loop.py
# -----------------------------------------------------------------------------
# Overhauled version that:
#   1) Forcibly selects random subsets for snapshots (ignoring label grouping).
#   2) Expects dataset[i] -> (img, label, bb_mask).
#   3) If 'img' has 4 channels, slices it to 3 for snapshot visualization.
#   4) Returns (grid_size, images, labels) from setup_snapshot_image_grid_NEW.
#   5) Re-uses that grid_size later for saving fakes so that dimensions match.
# -----------------------------------------------------------------------------

import os
import traceback
import time
import copy
import json
import dill
import psutil
import PIL.Image
import numpy as np
import torch
import torch.nn.functional as F
import dnnlib
import pickle

from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from metrics import metric_main


# ----------------------------------------------------------------------------
def setup_snapshot_image_grid_NEW(training_set, random_seed=0):
    """
    Creates a random grid of real images for snapshot visualization.
      - Ignores any label grouping.
      - Expects dataset[i] -> (img, label, bb_mask).
      - If 'img' has 4 channels, slices to 3 for snapshot visualization.
    Returns:
       grid_size: a tuple (gw, gh)
       images: numpy array of shape [N, 3, H, W]
       labels: numpy array of shape [N, ...] (or empty if no labels)
    """
    #print("\n[DEBUG] Using the NEW setup_snapshot_image_grid_NEW function!", flush=True)
    #traceback.print_stack(limit=25)
    
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)
    all_indices = list(range(len(training_set)))
    rnd.shuffle(all_indices)
    grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]
    
    images_list, labels_list = [], []
    for idx in grid_indices:
        img, lbl, bb = training_set[idx]  # Expect 3 items per sample.
        if img.shape[0] == 4:
            img = img[:3]
        images_list.append(img)
        labels_list.append(lbl)
        # (Ignore bb_mask for visualization.)
    
    images = np.stack(images_list)
    labels = np.stack(labels_list)
    grid_size = (gw, gh)
    return grid_size, images, labels


# ----------------------------------------------------------------------------
def save_image_grid(img, fname, drange, grid_size):
    """
    Saves a grid of images to disk.
    Expects:
      - img: numpy array of shape [N, C, H, W] where C is 1 or 3.
      - grid_size: a tuple (gw, gh).
    """
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size  # Must be a 2-tuple.
    _N, C, H, W = img.shape

    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])
    assert C in [1, 3], f"Expected image with 1 or 3 channels, got {C}"
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    else:
        PIL.Image.fromarray(img, 'RGB').save(fname)


# ----------------------------------------------------------------------------
def training_loop(
    run_dir='.',
    training_set_kwargs=None,
    data_loader_kwargs=None,
    G_kwargs=None,
    D_kwargs=None,
    G_opt_kwargs=None,
    D_opt_kwargs=None,
    loss_kwargs=None,
    metrics=None,
    random_seed=0,
    num_gpus=1,
    rank=0,
    batch_size=4,
    batch_gpu=4,
    ema_kimg=10,
    ema_rampup=0.05,
    G_reg_interval=None,
    D_reg_interval=16,
    total_kimg=25000,
    kimg_per_tick=4,
    image_snapshot_ticks=50,
    network_snapshot_ticks=50,
    resume_pkl=None,
    resume_kimg=0,
    cudnn_benchmark=True,
    abort_fn=None,
    progress_fn=None,
    restart_every=-1,
):
    if training_set_kwargs is None:
        training_set_kwargs = {}
    if data_loader_kwargs is None:
        data_loader_kwargs = {}
    if G_kwargs is None:
        G_kwargs = {}
    if D_kwargs is None:
        D_kwargs = {}
    if G_opt_kwargs is None:
        G_opt_kwargs = {}
    if D_opt_kwargs is None:
        D_opt_kwargs = {}
    if loss_kwargs is None:
        loss_kwargs = {}
    if metrics is None:
        metrics = []

    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conv2d_gradfix.enabled = True
    grid_sample_gradfix.enabled = True

    __RESTART__ = torch.tensor(0., device=device)
    __CUR_NIMG__ = torch.tensor(resume_kimg * 1000, dtype=torch.long, device=device)
    __CUR_TICK__ = torch.tensor(0, dtype=torch.long, device=device)
    __BATCH_IDX__ = torch.tensor(0, dtype=torch.long, device=device)
    __PL_MEAN__ = torch.zeros([], device=device)
    best_fid = 9999

    if rank == 0:
        print('Loading training set...', flush=True)
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(
        dataset=training_set,
        sampler=training_set_sampler,
        batch_size=batch_size // num_gpus,
        **data_loader_kwargs
    ))
    if rank == 0:
        print()
        print(f'Num images: {len(training_set)}', flush=True)
        print(f'Image shape: {training_set.image_shape}', flush=True)
        print(f'Label shape: {training_set.label_shape}', flush=True)
        print()

    if rank == 0:
        print('Constructing networks...', flush=True)
    common_kwargs = dict(
        c_dim=training_set.label_dim,
        img_resolution=training_set.resolution,
        img_channels=training_set.num_channels
    )
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    G_ema = copy.deepcopy(G).eval()

    ckpt_pkl = None
    if restart_every > 0 and os.path.isfile(misc.get_ckpt_path(run_dir)):
        ckpt_pkl = resume_pkl = misc.get_ckpt_path(run_dir)
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"', flush=True)
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
        if ckpt_pkl is not None:
            __CUR_NIMG__ = resume_data['progress']['cur_nimg'].to(device)
            __CUR_TICK__ = resume_data['progress']['cur_tick'].to(device)
            __BATCH_IDX__ = resume_data['progress']['batch_idx'].to(device)
            __PL_MEAN__ = resume_data['progress'].get('pl_mean', torch.zeros([])).to(device)
            best_fid = resume_data['progress']['best_fid']
        del resume_data

    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        img = misc.print_module_summary(G, [z, c])
        misc.print_module_summary(D, [img, c])

    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...', flush=True)
    for module in [G, D, G_ema]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)

    if rank == 0:
        print('Setting up training phases...', flush=True)
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, G_ema=G_ema, D=D, **loss_kwargs)
    phases = []
    for name, module, opt_kwargs, reg_interval in [
        ('G', G, G_opt_kwargs, G_reg_interval),
        ('D', D, D_opt_kwargs, D_reg_interval),
    ]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs)
            phases.append(dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1))
        else:
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs)
            phases.append(dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1))
            phases.append(dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval))
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # -------------------------------------------------------------------------
    # Export sample images.
    if rank == 0:
        print('Exporting sample images...', flush=True)
        grid_size, real_images, real_labels = setup_snapshot_image_grid_NEW(training_set, random_seed=random_seed)
        save_image_grid(real_images, os.path.join(run_dir, 'reals.png'), drange=[0, 255], grid_size=grid_size)
        
        grid_z = torch.randn([real_images.shape[0], G.z_dim], device=device).split(batch_gpu)
        if real_labels.size == 0:
            grid_c = torch.zeros([real_images.shape[0], G.c_dim], device=device).split(batch_gpu)
        else:
            grid_c = torch.from_numpy(real_labels).to(device).split(batch_gpu)
        out_images_list = []
        for z, c in zip(grid_z, grid_c):
            out = G_ema(z=z, c=c, noise_mode='const').cpu()
            if out.shape[1] == 4:
                out = out[:, :3]
            out_images_list.append(out)
        fake_images_tensor = torch.cat(out_images_list).detach().cpu()
        fake_images_init = fake_images_tensor.numpy()
        print(f"[DEBUG] fake_images_init shape: {fake_images_init.shape}, dtype: {fake_images_init.dtype}", flush=True)
        save_image_grid(fake_images_init, os.path.join(run_dir, 'fakes_init.png'),
                        drange=[-1, 1], grid_size=grid_size)

    # -------------------------------------------------------------------------
    # Initialize logs.
    if rank == 0:
        print('Initializing logs...', flush=True)
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = {}
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except Exception as err:
            print('Skipping tfevents export:', err, flush=True)
            stats_tfevents = None

    # -------------------------------------------------------------------------
    # Training loop setup.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...\n', flush=True)
    if num_gpus > 1:
        torch.distributed.broadcast(__CUR_NIMG__, 0)
        torch.distributed.broadcast(__CUR_TICK__, 0)
        torch.distributed.broadcast(__BATCH_IDX__, 0)
        torch.distributed.broadcast(__PL_MEAN__, 0)
        torch.distributed.barrier()
    cur_nimg = __CUR_NIMG__.item()
    cur_tick = __CUR_TICK__.item()
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_nimg - start_time
    batch_idx = __BATCH_IDX__.item()
    if progress_fn is not None:
        progress_fn(cur_nimg // 1000, total_kimg)
    if hasattr(loss, 'pl_mean'):
        loss.pl_mean.copy_(__PL_MEAN__)

    # -------------------------------------------------------------------------
    # Main training loop.
    while True:
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_c, phase_real_bb = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).float() / 127.5 - 1).split(batch_gpu)
            phase_real_c   = phase_real_c.to(device).split(batch_gpu)
            phase_real_bb  = phase_real_bb.to(device).float().split(batch_gpu)
            
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set)))
                         for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]
        
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            if phase.name in ['Dmain', 'Dboth', 'Dreg']:
                phase.module.feature_network.requires_grad_(False)
            for real_img, real_c, real_bb, gen_z, gen_c_item in zip(
                phase_real_img, phase_real_c, phase_real_bb, phase_gen_z, phase_gen_c
            ):
                loss.accumulate_gradients(
                    phase=phase.name,
                    real_img=real_img,
                    real_c=real_c,
                    real_bb=real_bb,
                    gen_z=gen_z,
                    gen_c=gen_c_item,
                    gain=phase.interval,
                    cur_nimg=cur_nimg,
                )
            phase.module.requires_grad_(False)
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))
        
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)
        
        cur_nimg += batch_size
        batch_idx += 1
        
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue
        
        tick_end_time = time.time()
        maintenance_time = tick_end_time - tick_start_time

        # --- Debug: print the progress for each tick.
        if rank == 0:
            print(f"[DEBUG] Tick {cur_tick}: cur_nimg = {cur_nimg}, sec/tick = {tick_end_time - tick_start_time:.2f}", flush=True)
        
        stats_collector.update()
        stats_dict = stats_collector.as_dict()
        
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, val in stats_dict.items():
                stats_tfevents.add_scalar(name, val.mean, global_step=global_step, walltime=walltime)
            for name, val in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', val, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)
        
        # Save image snapshots and network checkpoints at the desired tick intervals.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            gw, gh = grid_size
            n_images = gw * gh
            z_batches = torch.randn([n_images, G.z_dim], device=device).split(batch_gpu)
            c_batches = torch.zeros([n_images, G.c_dim], device=device).split(batch_gpu)  # dummy labels for unconditional G
            out_list = []
            for z_b, c_b in zip(z_batches, c_batches):
                fake_b = G_ema(z=z_b, c=c_b, noise_mode='const').cpu()
                if fake_b.shape[1] == 4:
                    fake_b = fake_b[:, :3]
                out_list.append(fake_b)
            fakes = torch.cat(out_list).detach().cpu()
            fakes_np = np.asarray(fakes)
            print(f"[DEBUG] fakes_np shape: {fakes_np.shape}, dtype: {fakes_np.dtype}", flush=True)
            save_image_grid(fakes_np, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'),
                            drange=[-1, 1], grid_size=grid_size)
        
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = {
                'G': G,
                'D': D,
                'G_ema': G_ema,
                'training_set_kwargs': dict(training_set_kwargs),
            }
            for key, value in snapshot_data.items():
                if isinstance(value, torch.nn.Module):
                    snapshot_data[key] = value
                del value
        
        if (rank == 0) and (restart_every > 0) and (network_snapshot_ticks is not None) and (snapshot_data is not None):
            #snapshot_pkl = misc.get_ckpt_path(run_dir)
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            snapshot_data['progress'] = {
                'cur_nimg': torch.LongTensor([cur_nimg]),
                'cur_tick': torch.LongTensor([cur_tick]),
                'batch_idx': torch.LongTensor([batch_idx]),
                'best_fid': best_fid,
            }
            if hasattr(loss, 'pl_mean'):
                snapshot_data['progress']['pl_mean'] = loss.pl_mean.cpu()
            with open(snapshot_pkl, 'wb') as f:
                pickle.dump(snapshot_data, f)
            print(f"[DEBUG] Saved network snapshot to {snapshot_pkl}", flush=True)
        
        if cur_tick and (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...', flush=True)
            for metric in metrics:
                result_dict = metric_main.calc_metric(
                    metric=metric,
                    G=snapshot_data['G_ema'],
                    run_dir=run_dir,
                    cur_nimg=cur_nimg,
                    dataset_kwargs=training_set_kwargs,
                    num_gpus=num_gpus,
                    rank=rank,
                    device=device
                )
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
            snapshot_pkl = os.path.join(run_dir, f'best_model.pkl')
            cur_nimg_txt = os.path.join(run_dir, f'best_nimg.txt')
            if rank == 0:
                if 'fid50k_full' in stats_metrics and stats_metrics['fid50k_full'] < best_fid:
                    best_fid = stats_metrics['fid50k_full']
                    with open(snapshot_pkl, 'wb') as f:
                        dill.dump(snapshot_data, f)
                    with open(cur_nimg_txt, 'w') as f:
                        f.write(str(cur_nimg))
                    print(f"[DEBUG] Saved best model checkpoint to {snapshot_pkl}", flush=True)
        
        del snapshot_data
        
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        
        # --- Add an extra tick print here to see the tick update.
        if rank == 0:
            print(f"[DEBUG] End of Tick {cur_tick}: Total images processed: {cur_nimg//1000:.1f} kimg", flush=True)
        
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break
        
    if rank == 0:
        print('\nExiting...', flush=True)
