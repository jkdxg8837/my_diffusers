"""
Merge evaluation images into 2x2 grids.

For each checkpoint-X/output_scale*/PROMPT/ folder containing images,
takes the first 4 images (0.png-3.png) and merges them into a 2x2 grid.

Usage:
    python merge_eval_images.py <input_dir> [--scale SCALE] [--out-res OUT_RES]

Arguments:
    input_dir    Parent directory containing checkpoint-X folders
    --scale      Only process output_scale<SCALE> folders (default: all scales)
    --out-res    Output resolution per tile in pixels (default: 512)
"""

import argparse
import os
import re
from pathlib import Path

from PIL import Image


def merge_4_images(image_paths, tile_size):
    """Merge 4 images into a 2x2 grid, resizing each tile to tile_size x tile_size."""
    grid = Image.new("RGB", (tile_size * 2, tile_size * 2))
    positions = [(0, 0), (tile_size, 0), (0, tile_size), (tile_size, tile_size)]
    for img_path, pos in zip(image_paths, positions):
        img = Image.open(img_path).convert("RGB")
        img = img.resize((tile_size, tile_size), Image.LANCZOS)
        grid.paste(img, pos)
    return grid


def find_image_folders(base_dir, scale_filter=None):
    """
    Yield (checkpoint_step, scale_str, prompt_idx, folder_path) for every
    prompt folder that has at least 4 images named 0.png .. 3.png.
    """
    base = Path(base_dir)
    ckpt_pattern = re.compile(r"^checkpoint-(\d+)$")

    for ckpt_dir in sorted(base.iterdir(), key=lambda p: (
        int(m.group(1)) if (m := ckpt_pattern.match(p.name)) else -1
    )):
        m = ckpt_pattern.match(ckpt_dir.name)
        if not m or not ckpt_dir.is_dir():
            continue
        step = int(m.group(1))

        for scale_dir in sorted(ckpt_dir.iterdir()):
            if not scale_dir.is_dir() or not scale_dir.name.startswith("output_scale"):
                continue
            scale_str = scale_dir.name  # e.g. "output_scale1.0"
            if scale_filter is not None and scale_str != f"output_scale{scale_filter}":
                continue

            for prompt_dir in sorted(scale_dir.iterdir(), key=lambda p: (
                int(p.name) if p.name.isdigit() else p.name
            )):
                if not prompt_dir.is_dir():
                    continue
                imgs = [prompt_dir / f"{i}.png" for i in range(4)]
                if all(img.exists() for img in imgs):
                    yield step, scale_str, prompt_dir.name, prompt_dir, imgs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", help="Parent directory containing checkpoint-X folders")
    parser.add_argument("--scale", default=None, help="Filter to a specific scale value, e.g. '1.0'")
    parser.add_argument("--out-res", type=int, default=512, help="Tile resolution in the output grid (default: 512)")
    args = parser.parse_args()

    base = Path(args.input_dir)
    if not base.is_dir():
        parser.error(f"Not a directory: {base}")

    count = 0
    for step, scale_str, prompt_idx, folder, imgs in find_image_folders(base, args.scale):
        out_path = folder / "grid_2x2.png"
        grid = merge_4_images(imgs, args.out_res)
        grid.save(out_path)
        print(f"checkpoint-{step:>4} / {scale_str} / prompt {prompt_idx:>3}  ->  {out_path}")
        count += 1

    print(f"\nDone. Created {count} grid image(s).")


if __name__ == "__main__":
    main()
