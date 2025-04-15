# discriminator.py
# -----------------------------------------------------------------------------
# Minimal changes to handle extra bounding‐box channels (or channels from other sources).
# This version attempts to merge any channels beyond the first 3 into an RGB image.
# -----------------------------------------------------------------------------

from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pg_modules.blocks import DownBlock, DownBlockPatch, conv2d
from pg_modules.projector import F_RandomProj
from pg_modules.diffaug import DiffAugment


class SingleDisc(nn.Module):
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8,
                 head=None, separable=False, patch=False):
        super().__init__()
        channel_dict = {
            4: 512, 8: 512, 16: 256, 32: 128,
            64: 64, 128: 64, 256: 32, 512: 16, 1024: 8
        }

        # Interpolate for start_sz not in channel_dict
        if start_sz not in channel_dict.keys():
            sizes = np.array(list(channel_dict.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        if ndf is None:
            nfc = channel_dict
        else:
            nfc = {k: ndf for k, v in channel_dict.items()}

        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []
        if head:
            layers += [
                conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        DB = partial(DownBlockPatch, separable=separable) if patch else partial(DownBlock, separable=separable)
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz], nfc[start_sz // 2]))
            start_sz = start_sz // 2

        layers.append(conv2d(nfc[end_sz], 1, 4, 1, 0, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        return self.main(x)


class SingleDiscCond(nn.Module):
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8,
                 head=None, separable=False, patch=False,
                 c_dim=1000, cmap_dim=64, embedding_dim=128):
        super().__init__()
        self.cmap_dim = cmap_dim

        channel_dict = {
            4: 512, 8: 512, 16: 256, 32: 128,
            64: 64, 128: 64, 256: 32, 512: 16, 1024: 8
        }

        if start_sz not in channel_dict.keys():
            sizes = np.array(list(channel_dict.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        if ndf is None:
            nfc = channel_dict
        else:
            nfc = {k: ndf for k, v in channel_dict.items()}

        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []
        if head:
            layers += [
                conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        DB = partial(DownBlockPatch, separable=separable) if patch else partial(DownBlock, separable=separable)
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz], nfc[start_sz // 2]))
            start_sz = start_sz // 2
        self.main = nn.Sequential(*layers)

        # Additional class conditioning
        self.cls = conv2d(nfc[end_sz], self.cmap_dim, 4, 1, 0, bias=False)
        self.embed = nn.Embedding(num_embeddings=c_dim, embedding_dim=embedding_dim)
        self.embed_proj = nn.Sequential(
            nn.Linear(self.embed.embedding_dim, self.cmap_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, c):
        h = self.main(x)
        out = self.cls(h)
        cmap = self.embed_proj(self.embed(c.argmax(1))).unsqueeze(-1).unsqueeze(-1)
        out = (out * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
        return out


class MultiScaleD(nn.Module):
    """
    The multi-scale sub-discriminators, each receiving features from different resolutions.
    """
    def __init__(self, channels, resolutions, num_discs=1, proj_type=2,
                 cond=0, separable=False, patch=False, **kwargs):
        super().__init__()
        assert num_discs in [1, 2, 3, 4]
        self.disc_in_channels = channels[:num_discs]
        self.disc_in_res = resolutions[:num_discs]
        Disc = SingleDiscCond if cond else SingleDisc

        mini_discs = []
        for i, (cin, res) in enumerate(zip(self.disc_in_channels, self.disc_in_res)):
            start_sz = res if not patch else 16
            disc_i = Disc(nc=cin, start_sz=start_sz, end_sz=8,
                          separable=separable, patch=patch)
            mini_discs += [str(i), disc_i],
        self.mini_discs = nn.ModuleDict(mini_discs)

    def forward(self, features, c):
        all_logits = []
        for k, disc in self.mini_discs.items():
            logits_k = disc(features[k], c)  # shape: [N,1,?]
            all_logits.append(logits_k.view(features[k].size(0), -1))
        all_logits = torch.cat(all_logits, dim=1)
        return all_logits


class ProjectedDiscriminator(nn.Module):
    """
    The main Projected Discriminator which:
      1) Optionally applies DiffAugment.
      2) Optionally interpolates inputs to 224.
      3) Extracts features using a pretrained backbone (F_RandomProj).
      4) Merges extra channels (e.g., bounding-box info) into the RGB image.
      5) Computes final logits via MultiScaleD.
    """
    def __init__(self, diffaug=True, interp224=True, backbone_kwargs={}, extra_channels=1, **kwargs):
        super().__init__()
        self.diffaug = diffaug
        self.interp224 = interp224
        self.extra_channels = extra_channels

        # Create a 1x1 conv to map extra channels to 3 channels, if extra channels are provided.
        if self.extra_channels > 0:
            self.bb_conv = nn.Conv2d(self.extra_channels, 3, kernel_size=1)
        else:
            self.bb_conv = None

        # Pretrained feature network (expects 3-channel input)
        self.feature_network = F_RandomProj(**backbone_kwargs)

        # Multi-scale sub-discriminators
        self.discriminator = MultiScaleD(
            channels=self.feature_network.CHANNELS,
            resolutions=self.feature_network.RESOLUTIONS,
            **backbone_kwargs,
        )

    def train(self, mode=True):
        # Force feature network to eval mode (to keep it fixed) if desired.
        self.feature_network = self.feature_network.train(False)
        self.discriminator = self.discriminator.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x, c):
        """
        x: Tensor of shape [N, C, H, W]. If extra channels are provided, then C > 3.
        c: Conditioning label tensor (or dummy tensor if not used).
        """
        # Optionally apply DiffAugment.
        if self.diffaug:
            x = DiffAugment(x, policy='color,translation,cutout')

        # Check if the input has extra channels beyond the first 3.
        if x.shape[1] > 3:
            extra_channel_count = x.shape[1] - 3
            print(f"[DEBUG] Input x has {x.shape[1]} channels; merging extra {extra_channel_count} channel(s).")
            x_img = x[:, :3]
            x_extra = x[:, 3:]
            # If the number of extra channels does not match self.extra_channels,
            # take only the first self.extra_channels channels and warn.
            if x_extra.shape[1] != self.extra_channels:
                print(f"[DEBUG] Warning: Expected {self.extra_channels} extra channels, but got {x_extra.shape[1]}. Using the first {self.extra_channels}.")
                x_extra = x_extra[:, :self.extra_channels]
            # Map extra channels to 3 channels using the 1x1 convolution
            x_mapped = self.bb_conv(x_extra)
            x = x_img + x_mapped  # Merge by addition
            print(f"[DEBUG] After merging, x shape: {x.shape}")

        # Optionally interpolate to 224 if enabled.
        if self.interp224:
            x = F.interpolate(x, 224, mode='bilinear', align_corners=False)
            print(f"[DEBUG] After interpolation, x shape: {x.shape}")

        # Extract features from the pretrained network.
        features = self.feature_network(x)
        # Compute final logits from the multi-scale discriminator.
        logits = self.discriminator(features, c)
        return logits
