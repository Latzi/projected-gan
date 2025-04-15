# train.py
# ------------------------------------------------------------------------------
# Modified so that the dataset is always treated as "no labels" (use_labels=False).
# This avoids any label-grouping code that can cause "too many values to unpack."
# You can still pass bounding-box channels in the dataset. 
# Requires the matching 'training_loop.py' that we updated to strip extra channels
# at snapshot time and forcibly do random subsets for snapshots.
# ------------------------------------------------------------------------------

import os
import click
import re
import json
import tempfile
import torch
import legacy

import dnnlib
from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops
from torch_utils import misc
import numpy  # ensures numpy is imported in the global scope
import numpy as np


def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(
                backend='gloo',
                init_method=init_method,
                rank=rank,
                world_size=c.num_gpus
            )
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(
                backend='nccl',
                init_method=init_method,
                rank=rank,
                world_size=c.num_gpus
            )

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **c)


def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]

    matching_dirs = [
        re.fullmatch(r'\d{5}' + f'-{desc}', x)
        for x in prev_run_dirs
        if re.fullmatch(r'\d{5}' + f'-{desc}', x) is not None
    ]
    if c.restart_every > 0 and len(matching_dirs) > 0:  # expect unique desc, continue in this directory
        assert len(matching_dirs) == 1, f'Multiple directories found for resuming: {matching_dirs}'
        c.run_dir = os.path.join(outdir, matching_dirs[0].group())
    else:
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir, exist_ok=c.restart_every > 0)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt+') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)


def init_dataset_kwargs(data):
    """
    We force use_labels=False so it never tries label-grouping. 
    We'll trust our dataset to return bounding-box channels if needed,
    but not treat them as "class labels."
    """
    try:
        dataset_kwargs = dnnlib.EasyDict(
            class_name='training.dataset.ImageFolderDataset',
            path=data,
            use_labels=False,    # forcibly disabled
            max_size=None,
            xflip=False
        )
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution
        # We do NOT do => dataset_kwargs.use_labels = dataset_obj.has_labels
        dataset_kwargs.max_size = len(dataset_obj)
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')


def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')


@click.command()
@click.option('--outdir', help='Where to save the results', metavar='DIR', required=True)
@click.option('--cfg', type=click.Choice(['fastgan', 'fastgan_lite', 'stylegan2']),
              help='Base configuration', required=True)
@click.option('--data', type=str, help='Training data [ZIP|DIR]', required=True)
@click.option('--gpus', type=click.IntRange(min=1), help='Number of GPUs to use', required=True)
@click.option('--batch', type=click.IntRange(min=1), help='Total batch size', required=True)

# Optional features
@click.option('--cond', type=bool, default=False, help='Train conditional model', show_default=True)
@click.option('--mirror', type=bool, default=False, help='Enable dataset x-flips', show_default=True)
@click.option('--resume', type=str, help='Resume from given network pickle', metavar='[PATH|URL]')

# Misc hyperparameters
@click.option('--batch-gpu', type=click.IntRange(min=1), help='Limit batch size per GPU')
@click.option('--cbase', type=click.IntRange(min=1), default=32768, show_default=True,
              help='Capacity multiplier')
@click.option('--cmax', type=click.IntRange(min=1), default=512, show_default=True,
              help='Max. feature maps')
@click.option('--glr', type=click.FloatRange(min=0), help='G learning rate [default: varies]')
@click.option('--dlr', type=click.FloatRange(min=0), default=0.002, show_default=True,
              help='D learning rate')
@click.option('--map-depth', type=click.IntRange(min=1), help='Mapping network depth [default: varies]')

# Misc settings
@click.option('--desc', type=str, help='String to include in result dir name', metavar='STR')
@click.option('--metrics', type=parse_comma_separated_list, default='fid50k_full', show_default=True,
              help='Quality metrics [NAME|A,B,C|none]')
@click.option('--kimg', type=click.IntRange(min=1), default=25000, show_default=True,
              help='Total training duration in kimg')
@click.option('--tick', type=click.IntRange(min=1), default=4, show_default=True,
              help='How often to print progress (kimg)')
@click.option('--snap', type=click.IntRange(min=1), default=50, show_default=True,
              help='How often to save snapshots (ticks)')
@click.option('--seed', type=click.IntRange(min=0), default=0, show_default=True,
              help='Random seed')
@click.option('--fp32', type=bool, default=False, show_default=True, help='Disable mixed-precision')
@click.option('--nobench', type=bool, default=False, show_default=True, help='Disable cuDNN benchmark')
@click.option('--workers', type=click.IntRange(min=1), default=3, show_default=True,
              help='DataLoader worker processes')
@click.option('-n','--dry-run', is_flag=True, help='Print training options and exit')
@click.option('--restart_every', type=int, default=9999999, show_default=True,
              help='Time interval (secs) to restart code')

def main(**kwargs):
    # Initialize config
    opts = dnnlib.EasyDict(kwargs)
    c = dnnlib.EasyDict()
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=64, w_dim=128, mapping_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)
    # If user sets --cond=True but dataset has no label info => error
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror

    # Hyperparams
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or (opts.batch // opts.gpus)
    c.G_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = opts.cmax
    c.G_kwargs.mapping_kwargs.num_layers = 2
    c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    # Sanity checks
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException(
            '\n'.join(['--metrics can only contain the following values:'] +
                      metric_main.list_valid_metrics())
        )

    # Base configuration
    c.ema_kimg = c.batch_size * 10 / 32
    if opts.cfg == 'stylegan2':
        c.G_kwargs.class_name = 'pg_modules.networks_stylegan2.Generator'
        c.G_kwargs.fused_modconv_default = 'inference_only' 
        use_separable_discs = True
    elif opts.cfg in ['fastgan', 'fastgan_lite']:
        c.G_kwargs = dnnlib.EasyDict(
            class_name='pg_modules.networks_fastgan.Generator',
            cond=opts.cond,
            synthesis_kwargs=dnnlib.EasyDict()
        )
        c.G_kwargs.synthesis_kwargs.lite = (opts.cfg == 'fastgan_lite')
        c.G_opt_kwargs.lr = c.D_opt_kwargs.lr = 0.0002
        use_separable_discs = False

    # Resume
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ema_rampup = None  # disable EMA rampup

    # Restart
    c.restart_every = opts.restart_every

    # Performance toggles
    if opts.fp32:
        c.G_kwargs.num_fp16_res = 0
        c.G_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string
    desc = f'{opts.cfg}-{dataset_name}-gpus{c.num_gpus}-batch{c.batch_size}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Projected and Multi-Scale Discriminators
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.ProjectedGANLoss')
    c.D_kwargs = dnnlib.EasyDict(
        class_name='pg_modules.discriminator.ProjectedDiscriminator',
        diffaug=True,
        interp224=(c.training_set_kwargs.resolution < 224),
        backbone_kwargs=dnnlib.EasyDict(),
    )
    c.D_kwargs.backbone_kwargs.cout = 64
    c.D_kwargs.backbone_kwargs.expand = True
    c.D_kwargs.backbone_kwargs.proj_type = 2
    c.D_kwargs.backbone_kwargs.num_discs = 4
    c.D_kwargs.backbone_kwargs.separable = use_separable_discs
    c.D_kwargs.backbone_kwargs.cond = opts.cond
    # *** Add this line to tell the discriminator how many extra channels are provided ***
    c.D_kwargs.extra_channels = 10   # Change 10 to the correct number based on the dataset

    # Launch
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)

    # Check for restart
    last_snapshot = misc.get_ckpt_path(c.run_dir)
    if os.path.isfile(last_snapshot):
        with dnnlib.util.open_url(last_snapshot) as f:
            cur_nimg = legacy.load_network_pkl(f)['progress']['cur_nimg'].item()
        if (cur_nimg // 1000) < c.total_kimg:
            print('Restart: exit with code 3')
            exit(3)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
