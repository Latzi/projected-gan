# generate_images.py
# ----------------------------------------------------------------------------
# Generate images with an optional bounding-box mask.
# - If --bbox-mask points to a single PNG file, the same mask is used for
#   every seed.
# - If --bbox-mask points to a directory, all *.png files inside are sorted
#   alphabetically and cycled through across the seeds (seed index modulo
#   number of masks).
# - No bounding-box is drawn on the saved images.
# ----------------------------------------------------------------------------

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

# ----------------------------------------------------------------------------
def parse_range(s: Union[str, List]) -> List[int]:
    """Parse a comma-separated list or range expression, e.g. '1,2,5-7'."""
    if isinstance(s, list):
        return s
    rng = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for part in s.split(','):
        m = range_re.match(part)
        if m:
            rng.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            rng.append(int(part))
    return rng

# ----------------------------------------------------------------------------
def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    """Parse 'a,b' into (a,b)."""
    if isinstance(s, tuple):
        return s
    a, b = s.split(',')
    return float(a), float(b)

# ----------------------------------------------------------------------------
def make_transform(translate: Tuple[float, float], angle: float) -> np.ndarray:
    """Return 3×3 affine transform for StyleGAN input manipulation."""
    m = np.eye(3)
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m[0, 0], m[0, 1], m[0, 2] = c,  s, translate[0]
    m[1, 0], m[1, 1], m[1, 2] = -s, c, translate[1]
    return m

# ----------------------------------------------------------------------------
@click.command()
@click.option('--network', 'network_pkl', required=True,
              help='Network pickle filename')
@click.option('--seeds', type=parse_range, required=True,
              help='List of random seeds, e.g. "0,1,4-6"')
@click.option('--trunc', 'truncation_psi', type=float, default=1.0,
              show_default=True, help='Truncation psi')
@click.option('--class', 'class_idx', type=int, default=None,
              help='Class label (for conditional networks)')
@click.option('--noise-mode', type=click.Choice(['const', 'random', 'none']),
              default='const', show_default=True, help='Noise mode')
@click.option('--translate', type=parse_vec2, default='0,0',
              show_default=True, metavar='VEC2',
              help='Translate XY, e.g. "0.3,1"')
@click.option('--rotate', type=float, default=0.0, show_default=True,
              metavar='ANGLE', help='Rotation angle in degrees')
@click.option('--bbox-mask', 'bbox_mask_path', type=str, default=None,
              help='Path to a mask PNG **or** a directory of PNGs')
@click.option('--outdir', type=str, required=True, metavar='DIR',
              help='Output directory')
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    translate: Tuple[float, float],
    rotate: float,
    class_idx: Optional[int],
    bbox_mask_path: Optional[str],
):
    """Generate images with optional per-seed bounding-box masks."""
    print(f'Loading network from "{network_pkl}"')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Prepare class label (if conditional)
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('--class must be specified for a conditional network')
        label[0, class_idx] = 1
    elif class_idx is not None:
        print('Note: --class is ignored for an unconditional network.')

    # ------------------------------------------------------------------
    # Build list of mask files (or None)
    mask_files: Optional[List[str]] = None
    if bbox_mask_path is not None:
        if os.path.isdir(bbox_mask_path):
            mask_files = sorted(
                os.path.join(bbox_mask_path, f)
                for f in os.listdir(bbox_mask_path)
                if f.lower().endswith('.png')
            )
            if not mask_files:
                print(f'[WARN] No PNGs found in "{bbox_mask_path}". Masks disabled.')
                mask_files = None
            else:
                print(f'Found {len(mask_files)} mask files in directory.')
        else:
            mask_files = [bbox_mask_path]
            print(f'Using single mask file: {bbox_mask_path}')
    # ------------------------------------------------------------------

    for idx, seed in enumerate(seeds):
        print(f'Generating seed {seed} ({idx+1}/{len(seeds)})')
        z = torch.from_numpy(
            np.random.RandomState(seed).randn(1, G.z_dim).astype(np.float32)
        ).to(device)

        # Apply global transform if supported
        if hasattr(G.synthesis, 'input'):
            m = np.linalg.inv(make_transform(translate, rotate))
            G.synthesis.input.transform.copy_(torch.from_numpy(m).to(device))

        # Select mask for this seed (if any)
        bb_mask = None
        if mask_files is not None:
            mask_path = mask_files[idx % len(mask_files)]
            mask_img = PIL.Image.open(mask_path).convert('L')
            mask = np.asarray(mask_img, dtype=np.float32) / 255.0   # [H,W] in [0,1]
            bb_mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)

        # Forward pass
        img = G(
            z,
            label,
            truncation_psi=truncation_psi,
            noise_mode=noise_mode,
            bb_mask=bb_mask,
        )

        # Save image
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = img[0].permute(1, 2, 0).cpu().numpy()
        out_path = os.path.join(outdir, f'seed{seed:04d}.png')
        PIL.Image.fromarray(img, 'RGB').save(out_path)
        print(f'Saved {out_path}')

# ----------------------------------------------------------------------------
if __name__ == '__main__':
    generate_images()  # pylint: disable=no-value-for-parameter
