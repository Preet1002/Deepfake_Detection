"""Dataset and dataloader construction.

Expected layout on disk (label 1 = fake, label 0 = real):

    data/processed/
      train/real/*.jpg   train/fake/*.jpg
      val/real/*.jpg     val/fake/*.jpg
      test/real/*.jpg    test/fake/*.jpg
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .transforms import build_eval_transform, build_train_transform

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_TO_LABEL = {"real": 0, "fake": 1}
LABEL_TO_CLASS = {0: "real", 1: "fake"}

# Datasets disagree on split folder names. Accepting the variants lets us point
# `data.root` straight at a read-only mount (e.g. Kaggle's /kaggle/input, which
# ships train/valid/test) instead of copying gigabytes just to rename a folder.
SPLIT_ALIASES = {
    "train": ["train", "training"],
    "val": ["val", "valid", "validation", "dev"],
    "test": ["test", "testing", "eval"],
}


def list_images(directory: Path) -> List[Path]:
    """List image files under `directory`, recursing into subfolders.

    Measured against an os.scandir version on 20k files: rglob was slightly
    faster, since it already uses scandir internally and reads the file type
    from the directory record rather than calling stat(). Kept simple.
    """
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def resolve_split_dir(root: str | Path, split: str) -> Path:
    """Find the directory for `split` under `root`, accepting name variants."""
    root = Path(root)
    for candidate in SPLIT_ALIASES.get(split, [split]):
        path = root / candidate
        if path.is_dir():
            return path
    tried = SPLIT_ALIASES.get(split, [split])
    present = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    raise FileNotFoundError(
        f"No '{split}' split under {root} (tried {tried}). "
        f"Subdirectories present: {present}"
    )


class FaceImageDataset(Dataset):
    """Loads face images from a split directory containing real/ and fake/."""

    def __init__(self, split_dir: str | Path, transform: Callable | None = None,
                 limit_per_class: Optional[int] = None, seed: int = 42):
        self.split_dir = Path(split_dir)
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        for class_name, label in CLASS_TO_LABEL.items():
            class_dir = self.split_dir / class_name
            if not class_dir.is_dir():
                continue
            print(f"  scanning {class_dir} ...", end="", flush=True)
            paths = list_images(class_dir)
            print(f" {len(paths):,} images", flush=True)
            if limit_per_class is not None and len(paths) > limit_per_class:
                # Shuffle before truncating: filenames are often grouped by
                # source identity, so taking the alphabetical head would sample
                # a biased subset of people.
                random.Random(seed).shuffle(paths)
                paths = paths[:limit_per_class]
            self.samples.extend((path, label) for path in paths)

        if not self.samples:
            raise RuntimeError(
                f"No images under {self.split_dir}. Expected {self.split_dir}/real and "
                f"{self.split_dir}/fake with image files."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        try:
            image = Image.open(path).convert("RGB")
        except OSError as exc:                      # truncated/corrupt file
            raise RuntimeError(f"Could not read image {path}") from exc
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

    def class_counts(self) -> dict[str, int]:
        counts = {"real": 0, "fake": 0}
        for _, label in self.samples:
            counts[LABEL_TO_CLASS[label]] += 1
        return counts


def _balanced_sampler(dataset: FaceImageDataset) -> WeightedRandomSampler:
    """Oversample the minority class so batches stay ~50/50."""
    counts = dataset.class_counts()
    per_class_weight = {
        CLASS_TO_LABEL[name]: 0.0 if counts[name] == 0 else 1.0 / counts[name]
        for name in counts
    }
    weights = [per_class_weight[label] for _, label in dataset.samples]
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


def build_dataloaders(config, splits=("train", "val")) -> dict[str, DataLoader]:
    """Create dataloaders for the requested splits.

    'train' gets augmentation plus class-balanced sampling; everything else gets
    the deterministic eval transform and sequential order.
    """
    root = Path(config.data.root)
    train_tf = build_train_transform(config.data.img_size, config.aug)
    eval_tf = build_eval_transform(config.data.img_size)

    loaders: dict[str, DataLoader] = {}
    for split in splits:
        is_train = split == "train"
        dataset = FaceImageDataset(
            resolve_split_dir(root, split),
            train_tf if is_train else eval_tf,
            limit_per_class=config.data.limit_per_class,
            seed=config.train.seed,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            sampler=_balanced_sampler(dataset) if is_train else None,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            drop_last=is_train,
            persistent_workers=config.data.num_workers > 0,
        )
    return loaders
