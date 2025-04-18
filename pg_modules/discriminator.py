# discriminator.py
# -----------------------------------------------------------------------------
# Updated to concatenate the bounding-box channels, then use a single 1×1 conv
# to return to 3 channels for the pretrained backbone. 
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

        # If given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = channel_dict
        else:
            nfc = {k: ndf for k, v in channel_dict.items()}

        # For feature map discriminators with nfc not in channel_dict (e.g. pretrained backbone)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []
        if head:
            layers += [
                conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        # Down Blocks.
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

        # Additional class conditioning.
        self.cls = conv2d(nfc[end_sz], self.cmap_dim, 4, 1, 0, bias=False)
        self.embed = nn.Embedding(num_embeddings=c_dim, embedding_dim=embedding_dim)
        self.embed_proj = nn.Sequential(
            nn.Linear(self.embed.embedding_dim, self.cmap_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, c):
        h = self.main(x)
        out = self.cls(h)

        # Projection.
        cmap = self.embed_proj(self.embed(c.argmax(1))).unsqueeze(-1).unsqueeze(-1)
        out = (out * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
        return out


class MultiScaleD(nn.Module):
    """
    The multi-scale sub-discriminators, each operating on a different scale.
    """
    def __init__(self, channels, resolutions, num_discs=1,
                 proj_type=2,  # 0 = no projection, 1 = cross-channel mixing, 2 = cross-scale mixing
                 cond=0, separable=False, patch=False, **kwargs):
        super().__init__()
        assert num_discs in [1, 2, 3, 4]

        self.disc_in_channels = channels[:num_discs]
        self.disc_in_res = resolutions[:num_discs]
        Disc = SingleDiscCond if cond else SingleDisc

        mini_discs = []
        for i, (cin, res) in enumerate(zip(self.disc_in_channels, self.disc_in_res)):
            start_sz = res if not patch else 16
            disc_i = Disc(nc=cin, start_sz=start_sz, end_sz=8, separable=separable, patch=patch)
            mini_discs.append((str(i), disc_i))
        self.mini_discs = nn.ModuleDict(mini_discs)

    def forward(self, features, c):
        all_logits = []
        for k, disc in self.mini_discs.items():
            logits_k = disc(features[k], c)
            all_logits.append(logits_k.view(features[k].size(0), -1))
        all_logits = torch.cat(all_logits, dim=1)
        return all_logits


class ProjectedDiscriminator(nn.Module):
    """
    The main Projected Discriminator that:
      1) Optionally applies DiffAugment.
      2) Optionally interpolates the input to 224×224.
      3) Extracts feature maps via a pretrained backbone (which expects 3 channels).
      4) Now concatenates the bounding-box channels to RGB, then uses a single 1×1 conv
         to get back to 3 channels for the backbone.
      5) Feeds the features to multi-scale sub-discriminators.
    """
    def __init__(self, diffaug=True, interp224=True, backbone_kwargs={}, extra_channels=1, **kwargs):
        super().__init__()
        self.diffaug = diffaug
        self.interp224 = interp224
        # extra_channels: number of extra bounding box channels appended to the RGB image.
        self.extra_channels = extra_channels

        # Instead of separate conv for just bb channels, do a single conv that merges
        # (3 + extra_channels) -> 3. This is crucial for a pretrained backbone expecting 3 channels.
        if self.extra_channels > 0:
            # 1×1 conv that takes in 3+extra and outputs 3
            self.merge_conv = nn.Conv2d(3 + self.extra_channels, 3, kernel_size=1)
        else:
            self.merge_conv = None

        # Pretrained feature network for extracting features.
        self.feature_network = F_RandomProj(**backbone_kwargs)

        # Multi-scale discriminators.
        self.discriminator = MultiScaleD(
            channels=self.feature_network.CHANNELS,
            resolutions=self.feature_network.RESOLUTIONS,
            **backbone_kwargs,
        )

    def train(self, mode=True):
        # Force the feature network to evaluation mode.
        self.feature_network = self.feature_network.train(False)
        self.discriminator = self.discriminator.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x, c):
        """
        x: Tensor of shape [N, 3 + extra_channels, H, W]
        c: Conditioning labels (unused or used in conditional discriminators).
        """
        # Optionally apply DiffAugment.
        if self.diffaug:
            x = DiffAugment(x, policy='color,translation,cutout')

        # If extra channels are present, handle them by concatenating & mapping back to 3 ch.
        if self.extra_channels > 0 and x.shape[1] > 3:
            # Separate the first 3 (RGB) and the bounding-box extras
            x_img = x[:, :3]                         # [N, 3,   H, W]
            x_bb  = x[:, 3 : 3+self.extra_channels]  # [N, extra_channels, H, W]

            # Concatenate => shape [N, 3 + extra_channels, H, W]
            x_cat = torch.cat((x_img, x_bb), dim=1)

            # Single 1×1 conv to re-map from (3+extra) to 3
            x = self.merge_conv(x_cat)  # => [N, 3, H, W]

        # If interpolation to 224 is enabled, resize the image.
        if self.interp224:
            x = F.interpolate(x, 224, mode='bilinear', align_corners=False)

        # Extract feature maps via pretrained backbone (which now sees 3 channels).
        features = self.feature_network(x)

        # Compute final logits with the multi-scale discriminator.
        logits = self.discriminator(features, c)
        return logits
