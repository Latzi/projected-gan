# generate_images.py
# ----------------------------------------------------------------------------
# Modified to optionally provide a bounding‐box mask to the generator.
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
    '''Parse a comma separated list of numbers or ranges and return a list of ints.
       e.g. '1,2,5-10' => [1, 2, 5, 6, 7, 8, 9, 10]
    '''
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges

# ----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'. e.g. '0,1' => (0,1).'''
    if isinstance(s, tuple):
        return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

# ----------------------------------------------------------------------------

def make_transform(translate: Tuple[float, float], angle: float):
    m = np.eye(3)
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

# ----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g. \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, default=1, show_default=True, help='Truncation psi')
@click.option('--class', 'class_idx', type=int, default=None, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True, help='Noise mode')
@click.option('--translate', type=parse_vec2, default='0,0', show_default=True, metavar='VEC2', help='Translate XY-coordinate (e.g. \'0.3,1\')')
@click.option('--rotate', type=float, default=0, show_default=True, metavar='ANGLE', help='Rotation angle in degrees')
@click.option('--bbox-mask', 'bbox_mask_path', type=str, default=None, help='Path to a bounding box mask PNG (grayscale). Must match the generator resolution.')
@click.option('--outdir', type=str, required=True, metavar='DIR', help='Where to save the output images')
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    translate: Tuple[float, float],
    rotate: float,
    class_idx: Optional[int],
    bbox_mask_path: Optional[str]
):
    """
    Generate images using a pretrained network.

    Example usage:
      python generate_images.py --network=path/to/net.pkl --seeds=0-5 --outdir=out
      python generate_images.py --network=net.pkl --seeds=42 --bbox-mask=path/to/mask.png --outdir=out
    """

    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # If conditional, prepare label.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException(
                '--class=lbl must be specified for a conditional network'
            )
        label[0, class_idx] = 1
    else:
        if class_idx is not None:
            print('Note: --class=lbl is ignored when running on an unconditional network.')

    # Optionally load bounding‐box mask if provided.
    bb_mask = None
    if bbox_mask_path is not None:
        print(f'Loading bounding‐box mask from {bbox_mask_path}...')
        # Load grayscale
        mask_img = PIL.Image.open(bbox_mask_path).convert('L')
        # The mask_img should be the same resolution as G (e.g., 256x256).
        # If you need to resize, do it here:
        # mask_img = mask_img.resize((G.img_resolution, G.img_resolution), PIL.Image.NEAREST)

        mask = np.array(mask_img, dtype=np.float32)
        mask = mask / 255.0  # scale to [0,1]
        # shape => [H,W], we want => [1,1,H,W]
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
        bb_mask = mask  # we'll pass this to G

    # Generate images for each seed.
    for seed_idx, seed in enumerate(seeds):
        print(f'Generating image for seed {seed} ({seed_idx+1}/{len(seeds)})...')
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

        # If using stylegan-like code with transform, we can handle rotation/translation:
        if hasattr(G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        # Forward pass, with optional bounding‐box
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode, bb_mask=bb_mask)

        # Convert to PIL Image
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)  # [N, C, H, W]
        img = img[0].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        PIL.Image.fromarray(img, 'RGB').save(f'{outdir}/seed{seed:04d}.png')

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter
