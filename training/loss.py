# loss.py
# ----------------------------------------------------------------------------
# Minimal example showing bounding-box conditioning in ProjectedGANLoss
# ----------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
from torch_utils import training_stats
from torch_utils.ops import upfirdn2d

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        raise NotImplementedError()

# ----------------------------------------------------------------------------

class ProjectedGANLoss(Loss):
    def __init__(self, device, G, D, G_ema,
                 blur_init_sigma=0, blur_fade_kimg=0, **kwargs):
        super().__init__()
        self.device = device
        self.G = G
        self.G_ema = G_ema
        self.D = D
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg

    # -- ADDED: real_bb optional param so we can feed bounding boxes to G if desired.
    def run_G(self, z, c, real_bb=None, update_emas=False):
        """
        Optional: Pass 'real_bb' to the generator so G can learn to incorporate bounding boxes.
        You must also modify the generator code (G.mapping / G.synthesis) to accept this param.
        """
        # Map latent z to w space.
        ws = self.G.mapping(z, c, update_emas=update_emas)

        # Synthesize image. If your G.synthesis supports bounding boxes, pass them here:
        #     img = self.G.synthesis(ws, c, bb=real_bb)
        # For minimal code, let's just ignore real_bb in G:
        img = self.G.synthesis(ws, c, update_emas=False)
        return img

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        """
        Standard ProjectedGAN call to the discriminator. 
        'img' can already have bounding‐box masks concatenated in accumulate_gradients().
        """
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        logits = self.D(img, c)
        return logits

    # -- ADDED: real_bb param in accumulate_gradients
    def accumulate_gradients(self, phase, real_img, real_c, real_bb,
                             gen_z, gen_c, gain, cur_nimg):
        """
        real_img: (N, 3, H, W)
        real_bb:  (N, M, H, W) bounding‐box mask, M could be 1 or more channels
        gen_z:    latent
        gen_c:    label for generator
        """
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        if phase in ['Dreg', 'Greg']:
            # No regularization steps for ProjectedGAN
            return

        # Blur schedule if used (for progressive blur).
        blur_sigma = 0
        if self.blur_fade_kimg > 1:
            blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma

        # --------------------------------------------------------------------
        # Gmain: Maximize logits for generated images
        # --------------------------------------------------------------------
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                # 1) Generate fake image (optionally using bounding boxes)
                gen_img = self.run_G(gen_z, gen_c, real_bb=real_bb)

                # 2) Concatenate bounding‐box mask with fake image => [N, 3+M, H, W]
                #    This is so the discriminator "sees" the requested bounding boxes.
                gen_img_plus_mask = torch.cat([gen_img, real_bb], dim=1)

                # 3) Forward pass through discriminator
                gen_logits = self.run_D(gen_img_plus_mask, gen_c, blur_sigma=blur_sigma)

                # 4) Non‐saturating logistic loss
                loss_Gmain = -gen_logits.mean()

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.backward()

        # --------------------------------------------------------------------
        # Dmain: Minimize logits for generated images and maximize for real
        # --------------------------------------------------------------------
        if do_Dmain:
            # Fake path (generated images)
            with torch.autograd.profiler.record_function('Dgen_forward'):
                # 1) Generate a fake image
                gen_img = self.run_G(gen_z, gen_c, real_bb=real_bb)

                # 2) Concatenate bounding‐box mask for the fake
                gen_img_plus_mask = torch.cat([gen_img, real_bb], dim=1)

                # 3) Evaluate
                gen_logits = self.run_D(gen_img_plus_mask, gen_c, blur_sigma=blur_sigma)

                # 4) Compute loss for fake => we want D to output "fake" => push logits < -1
                #    Hinge or standard logistic; here is typical "softplus" or ReLU:
                #    This code uses the "relativistic" approach: F.relu(1 + gen_logits)
                #    Or a non‐saturating approach. We do (relu(1 + gen_logits)) => for real/fake?
                #    Original code does F.relu(1 + gen_logits). Note that is a variant of hinge.
                loss_Dgen = F.relu(torch.ones_like(gen_logits) + gen_logits).mean()

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.backward()

            # Real path
            with torch.autograd.profiler.record_function('Dreal_forward'):
                # 1) We do not need grad for the real image
                real_img_tmp = real_img.detach()

                # 2) Concatenate bounding‐box mask with real image => [N,3+M,H,W]
                real_img_plus_mask = torch.cat([real_img_tmp, real_bb], dim=1)

                # 3) Evaluate real image
                real_logits = self.run_D(real_img_plus_mask, real_c, blur_sigma=blur_sigma)

                # 4) Compute real loss => we want D to output "real" => push logits > +1
                #    Using same hinge or logistic approach
                loss_Dreal = F.relu(torch.ones_like(real_logits) - real_logits).mean()

                # Logging
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

            with torch.autograd.profiler.record_function('Dreal_backward'):
                loss_Dreal.backward()

