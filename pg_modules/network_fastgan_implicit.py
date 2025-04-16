# networks_fastgan.py
# -----------------------------------------------------------------------------
# Minimal modifications to accept bb_mask in the generator using concatenation.
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
from pg_modules.blocks import (
    InitLayer, UpBlockBig, UpBlockBigCond, UpBlockSmall, UpBlockSmallCond,
    SEBlock, conv2d
)

# ----------------------------------------------------------------------------
# Utility for latents:
# ----------------------------------------------------------------------------
def normalize_second_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

# ----------------------------------------------------------------------------
# A dummy mapping so that the code matches the StyleGAN API.
# ----------------------------------------------------------------------------
class DummyMapping(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, c, **kwargs):
        # Expect z shape [batch_size, z_dim].
        # Return shape [batch_size, 1, z_dim] so it fits the StyleGAN-like usage.
        return z.unsqueeze(1)

# ----------------------------------------------------------------------------
# Main "unconditional" FastGAN Synthesis
# ----------------------------------------------------------------------------
class FastganSynthesis(nn.Module):
    def __init__(self, ngf=128, z_dim=256, nc=3, img_resolution=256, lite=False):
        super().__init__()
        self.img_resolution = img_resolution
        self.z_dim = z_dim

        # channel multiplier
        nfc_multi = {
            2:16, 4:16, 8:8, 16:4, 32:2,
            64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125
        }
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)

        # layers
        self.init = InitLayer(z_dim, channel=nfc[2], sz=4)
        UpBlock = UpBlockSmall if lite else UpBlockBig

        self.feat_8   = UpBlock(nfc[4],  nfc[8])
        self.feat_16  = UpBlock(nfc[8],  nfc[16])
        self.feat_32  = UpBlock(nfc[16], nfc[32])
        self.feat_64  = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlock(nfc[64], nfc[128])
        self.feat_256 = UpBlock(nfc[128], nfc[256])

        self.se_64  = SEBlock(nfc[4],  nfc[64])
        self.se_128 = SEBlock(nfc[8],  nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_big = conv2d(nfc[img_resolution], nc, 3, 1, 1, bias=True)

        if img_resolution > 256:
            self.feat_512 = UpBlock(nfc[256], nfc[512])
            self.se_512   = SEBlock(nfc[32], nfc[512])
        if img_resolution > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])

    def forward(self, input, c, **kwargs):
        """
        input: shape [batch, 1, z_dim] from DummyMapping.
               We do input[:,0] => shape [batch, z_dim].
        c:     if used, just carried (unused in unconditional).
        """
        # map noise to hypersphere as in "Progressive Growing of GANs"
        x = normalize_second_moment(input[:, 0])

        feat_4   = self.init(x)
        feat_8   = self.feat_8(feat_4)
        feat_16  = self.feat_16(feat_8)
        feat_32  = self.feat_32(feat_16)
        feat_64  = self.se_64(feat_4, self.feat_64(feat_32))
        feat_128 = self.se_128(feat_8, self.feat_128(feat_64))

        if self.img_resolution >= 128:
            feat_last = feat_128
        if self.img_resolution >= 256:
            feat_last = self.se_256(feat_16, self.feat_256(feat_last))
        if self.img_resolution >= 512:
            feat_last = self.se_512(feat_32, self.feat_512(feat_last))
        if self.img_resolution >= 1024:
            feat_last = self.feat_1024(feat_last)

        return self.to_big(feat_last)

# ----------------------------------------------------------------------------
# Conditional FastGAN (handles c by embedding).
# ----------------------------------------------------------------------------
class FastganSynthesisCond(nn.Module):
    def __init__(self, ngf=64, z_dim=256, nc=3, img_resolution=256, num_classes=1000, lite=False):
        super().__init__()
        self.z_dim = z_dim
        self.img_resolution = img_resolution

        nfc_multi = {
            2:16, 4:16, 8:8, 16:4, 32:2,
            64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125, 2048:0.125
        }
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)

        self.init = InitLayer(z_dim, channel=nfc[2], sz=4)
        UpBlock = UpBlockSmallCond if lite else UpBlockBigCond

        self.feat_8   = UpBlock(nfc[4],  nfc[8],  z_dim)
        self.feat_16  = UpBlock(nfc[8],  nfc[16], z_dim)
        self.feat_32  = UpBlock(nfc[16], nfc[32], z_dim)
        self.feat_64  = UpBlock(nfc[32], nfc[64], z_dim)
        self.feat_128 = UpBlock(nfc[64], nfc[128], z_dim)
        self.feat_256 = UpBlock(nfc[128], nfc[256], z_dim)

        self.se_64  = SEBlock(nfc[4],  nfc[64])
        self.se_128 = SEBlock(nfc[8],  nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_big = conv2d(nfc[img_resolution], nc, 3, 1, 1, bias=True)

        if img_resolution > 256:
            self.feat_512 = UpBlock(nfc[256], nfc[512])
            self.se_512   = SEBlock(nfc[32], nfc[512])
        if img_resolution > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])

        # Embedding for class c
        self.embed = nn.Embedding(num_classes, z_dim)

    def forward(self, input, c, update_emas=False):
        """
        input: shape [batch, 1, z_dim].
        c: one-hot or single class ID => embed => c in [batch, z_dim].
        """
        # Convert c from one-hot to ID, then embed.
        c_embed = self.embed(c.argmax(1))

        x = normalize_second_moment(input[:, 0])  # [batch, z_dim]

        feat_4   = self.init(x)
        feat_8   = self.feat_8(feat_4, c_embed)
        feat_16  = self.feat_16(feat_8, c_embed)
        feat_32  = self.feat_32(feat_16, c_embed)
        feat_64  = self.se_64(feat_4, self.feat_64(feat_32, c_embed))
        feat_128 = self.se_128(feat_8, self.feat_128(feat_64, c_embed))

        if self.img_resolution >= 128:
            feat_last = feat_128
        if self.img_resolution >= 256:
            feat_last = self.se_256(feat_16, self.feat_256(feat_last, c_embed))
        if self.img_resolution >= 512:
            feat_last = self.se_512(feat_32, self.feat_512(feat_last, c_embed))
        if self.img_resolution >= 1024:
            feat_last = self.feat_1024(feat_last, c_embed)

        return self.to_big(feat_last)

# ----------------------------------------------------------------------------
# Final Generator class that wraps everything
# ----------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(
        self,
        z_dim=256,
        c_dim=0,
        w_dim=0,             # not really used by FastGAN
        img_resolution=256,
        img_channels=3,
        ngf=128,
        cond=0,
        mapping_kwargs={},
        synthesis_kwargs={}
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        # Minimal bounding‐box flatten + embed.
        # If your bounding‐box mask has shape [N, 1, H, W], we set:
        self.bb_in_channels = 1    # <--- Adjust if you have multiple channels.
        self.bb_project = nn.Linear(
            self.bb_in_channels * self.img_resolution * self.img_resolution,
            z_dim
        )
        # NEW: Use concatenation instead of addition.
        # After concatenation, the latent becomes of size 2*z_dim; we project it back to z_dim.
        self.bb_concat_proj = nn.Linear(z_dim * 2, z_dim)

        # Dummy mapping to keep consistent with StyleGAN2 API.
        self.mapping = DummyMapping()

        # Choose conditional or unconditional synthesis.
        Synthesis = FastganSynthesisCond if cond else FastganSynthesis
        self.synthesis = Synthesis(
            ngf=ngf,
            z_dim=z_dim,
            nc=img_channels,
            img_resolution=img_resolution,
            **synthesis_kwargs
        )

    def forward(self, z, c, bb_mask=None, **kwargs):
        """
        z: [batch, z_dim]
        c: [batch, c_dim] or a dummy label.
        bb_mask: [batch, M, H, W], e.g. M = 1 for a single‐channel bounding box.
        """
        # If bounding‐box mask is given, flatten and embed it, then concatenate with z.
        if bb_mask is not None:
            B, M, H, W = bb_mask.shape
            # Flatten to [B, M * H * W]
            mask_flat = bb_mask.view(B, -1).float()
            # Project to [B, z_dim]
            mask_embed = self.bb_project(mask_flat)
            # Concatenate z and the mask embedding along the feature dimension.
            z_cat = torch.cat([z, mask_embed], dim=1)  # shape: [B, 2*z_dim]
            # Project back to the original z_dim.
            z = self.bb_concat_proj(z_cat)             # shape: [B, z_dim]

        # The dummy mapping produces a shape of [B, 1, z_dim].
        w = self.mapping(z, c)
        img = self.synthesis(w, c, **kwargs)
        return img
