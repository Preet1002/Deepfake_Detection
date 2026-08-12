"""Generate a tiny synthetic dataset to smoke-test the pipeline.

This is NOT training data - it is throwaway noise with an obvious artificial
"artefact" in the fake class, used only to prove that data loading, training,
evaluation and inference all run before you commit to a real dataset.

    python scripts/make_dummy_data.py --dst data/dummy --per-class 120
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def make_image(is_fake: bool, size: int, rng: np.random.Generator) -> Image.Image:
    # Smooth low-frequency blobs standing in for a face.
    coarse = rng.normal(0.5, 0.18, (size // 8, size // 8, 3)).clip(0, 1)
    base = np.array(Image.fromarray(np.uint8(coarse * 255)).resize((size, size), Image.BICUBIC),
                    dtype=np.float32) / 255.0

    if is_fake:
        # Periodic high-frequency grid, the kind of upsampling fingerprint the
        # SRM branch is designed to pick up.
        yy, xx = np.mgrid[0:size, 0:size]
        grid = 0.06 * np.sin(xx * np.pi / 2) * np.sin(yy * np.pi / 2)
        base += grid[..., None]
    else:
        base += rng.normal(0, 0.02, base.shape)      # sensor-like noise

    return Image.fromarray(np.uint8(base.clip(0, 1) * 255))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst", default="data/dummy")
    parser.add_argument("--per-class", type=int, default=120)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    dst = Path(args.dst)
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

    for split, ratio in ratios.items():
        count = max(2, int(args.per_class * ratio))
        for class_name, is_fake in (("real", False), ("fake", True)):
            out_dir = dst / split / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                make_image(is_fake, args.size, rng).save(out_dir / f"{i:05d}.png")
        print(f"{split}: {count} real + {count} fake")

    print(f"\nDummy dataset written to {dst.resolve()}")


if __name__ == "__main__":
    main()
