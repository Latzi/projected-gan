# discriminator.py
# -----------------------------------------------------------------------------
# Minimal changes to handle an extra bounding‐box channel (or channels).
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
            4:512, 8:512, 16:256, 32:128,
            64:64, 128:64, 256:32, 512:16, 1024:8
        }

        # interpolate for start_sz not in channel_dict
        if start_sz not in channel_dict.keys():
            sizes = np.array(list(channel_dict.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = channel_dict
        else:
            nfc = {k: ndf for k, v in channel_dict.items()}

        # for feature map discriminators with nfc not in channel_dict
        # (e.g. pretrained backbone)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []
        if head:
            layers += [
                conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        # Down Blocks
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
            4:512, 8:512, 16:256, 32:128,
            64:64, 128:64, 256:32, 512:16, 1024:8
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

        # additional class conditioning
        self.cls = conv2d(nfc[end_sz], self.cmap_dim, 4, 1, 0, bias=False)
        self.embed = nn.Embedding(num_embeddings=c_dim, embedding_dim=embedding_dim)
        self.embed_proj = nn.Sequential(
            nn.Linear(self.embed.embedding_dim, self.cmap_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, c):
        h = self.main(x)
        out = self.cls(h)

        # projection
        cmap = self.embed_proj(self.embed(c.argmax(1))).unsqueeze(-1).unsqueeze(-1)
        out = (out * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
        return out


class MultiScaleD(nn.Module):
    """
    The multi-scale sub-discriminators,
    each seeing different resolution feature maps from the backbone.
    """
    def __init__(
        self,
        channels,
        resolutions,
        num_discs=1,
        proj_type=2,  # 0 = no projection, 1 = cross-channel mixing, 2 = cross-scale mixing
        cond=0,
        separable=False,
        patch=False,
        **kwargs,
    ):
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
        all_logits = torch.cat(all_logits, dim=1)  # gather
        return all_logits


class ProjectedDiscriminator(nn.Module):
    """
    The main Projected Discriminator that:
      1) Optionally applies DiffAugment.
      2) Interpolates to 224 if desired.
      3) Extracts features via F_RandomProj (efficientnet-based).
      4) Passes features into MultiScaleD for final real/fake score.
    """
    def __init__(
        self,
        diffaug=True,
        interp224=True,
        backbone_kwargs={},
        extra_channels=1,  # <--- bounding-box channels
        **kwargs
    ):
        super().__init__()
        self.diffaug = diffaug
        self.interp224 = interp224

        # We define how many bounding-box channels we expect
        # If you have multiple bounding-box channels, set extra_channels accordingly.
        self.extra_channels = extra_channels

        # A small 1x1 conv to map bounding-box mask -> 3 channels, so we can add to the original image
        # If extra_channels=1, it maps [N,1,H,W] -> [N,3,H,W].
        # If you want to cat instead of add, adapt below.
        if self.extra_channels > 0:
            self.bb_conv = nn.Conv2d(self.extra_channels, 3, kernel_size=1)
        else:
            self.bb_conv = None

        # The pretrained feature network:
        self.feature_network = F_RandomProj(**backbone_kwargs)

        # The multi-scale sub-discriminators:
        self.discriminator = MultiScaleD(
            channels=self.feature_network.CHANNELS,
            resolutions=self.feature_network.RESOLUTIONS,
            **backbone_kwargs,
        )

    def train(self, mode=True):
        # Force feature network to "eval" if you want to keep it fixed
        self.feature_network = self.feature_network.train(False)
        self.discriminator = self.discriminator.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x, c):
        """
        x: [N, 3 + extra_channels, H, W]
        c: conditioning label (unused or used in SingleDiscCond)
        """
        # Possibly apply DiffAugment
        if self.diffaug:
            x = DiffAugment(x, policy='color,translation,cutout')

        # If we have bounding-box channels, merge them with the RGB image
        if self.extra_channels > 0 and x.shape[1] == 3 + self.extra_channels:
            # separate the first 3 channels for real image
            x_img = x[:, :3]              # [N,3,H,W]
            x_bb  = x[:, 3:]             # [N,extra_channels,H,W]

            # map bounding-box mask from "extra_channels -> 3" via 1x1 conv
            bb_3 = self.bb_conv(x_bb)    # [N,3,H,W]

            # Add them
            x = x_img + bb_3  # or torch.cat([x_img, bb_3], dim=1) if you prefer

        # Possibly interpolate to 224
        if self.interp224:
            x = F.interpolate(x, 224, mode='bilinear', align_corners=False)

        # Extract features
        features = self.feature_network(x)

        # Multi-scale disc for final logits
        logits = self.discriminator(features, c)
        return logits

