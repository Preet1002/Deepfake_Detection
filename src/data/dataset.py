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
from typing import Callable, List, Optional, Sequence, Tuple

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
    """Find the directory for `split` under `root`, accepting name variants.

    Matches by listing the directory rather than probing `root / name`, so it is
    case-insensitive on case-sensitive filesystems too. Datasets differ here:
    the 140k set ships `train/valid/test`, others ship `Train/Validation/Test`,
    and Linux will not find one when you ask for the other.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    present = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    for candidate in SPLIT_ALIASES.get(split, [split]):
        if candidate in present:
            return present[candidate]

    raise FileNotFoundError(
        f"No '{split}' split under {root} (tried {SPLIT_ALIASES.get(split, [split])}). "
        f"Subdirectories present: {sorted(p.name for p in root.iterdir() if p.is_dir())}"
    )


def resolve_class_dir(split_dir: Path, class_name: str) -> Optional[Path]:
    """Locate the real/ or fake/ folder inside a split, ignoring case."""
    for child in split_dir.iterdir():
        if child.is_dir() and child.name.lower() == class_name:
            return child
    return None


class FaceImageDataset(Dataset):
    """Loads face images from a split directory containing real/ and fake/."""

    def __init__(self, split_dir: str | Path | Sequence[str | Path],
                 transform: Callable | None = None,
                 limit_per_class: Optional[int] = None, seed: int = 42):
        # Accept one directory or several; `limit_per_class` then applies per
        # source, so a large dataset cannot drown out a small one.
        if isinstance(split_dir, (str, Path)):
            split_dirs = [Path(split_dir)]
        else:
            split_dirs = [Path(d) for d in split_dir]
        if not split_dirs:
            raise ValueError("At least one split directory is required")

        for directory in split_dirs:
            if not directory.is_dir():
                raise FileNotFoundError(f"Split directory not found: {directory}")

        self.split_dirs = split_dirs
        self.split_dir = split_dirs[0]          # kept for messages/back-compat
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        for directory in split_dirs:
            for class_name, label in CLASS_TO_LABEL.items():
                class_dir = resolve_class_dir(directory, class_name)
                if class_dir is None:
                    continue
                print(f"  scanning {class_dir} ...", end="", flush=True)
                paths = list_images(class_dir)
                print(f" {len(paths):,} images", flush=True)
                if limit_per_class is not None and len(paths) > limit_per_class:
                    # Shuffle before truncating: filenames are often grouped by
                    # source identity, so taking the alphabetical head would
                    # sample a biased subset of people.
                    random.Random(seed).shuffle(paths)
                    paths = paths[:limit_per_class]
                self.samples.extend((path, label) for path in paths)

        if not self.samples:
            listed = ", ".join(str(d) for d in split_dirs)
            raise RuntimeError(
                f"No images under {listed}. Expected real/ and fake/ subfolders "
                f"with image files."
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
    raw_root = config.data.root
    roots = [Path(raw_root)] if isinstance(raw_root, (str, Path)) \
        else [Path(r) for r in raw_root]
    train_tf = build_train_transform(config.data.img_size, config.aug)
    eval_tf = build_eval_transform(config.data.img_size)

    loaders: dict[str, DataLoader] = {}
    for split in splits:
        is_train = split == "train"
        dataset = FaceImageDataset(
            [resolve_split_dir(r, split) for r in roots],
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
