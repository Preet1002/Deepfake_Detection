"""Turn a downloaded dataset into the layout the training code expects.

Two situations are handled:

1. The dataset is already split (e.g. Kaggle "140k Real and Fake Faces" ships
   train/valid/test each containing real/ and fake/). Point --src at the parent
   and the script just normalises the directory names.

2. The dataset is a flat pair of folders (real/ and fake/). Pass --split to
   carve out train/val/test yourself.

Files are hard-linked by default, so a 4 GB dataset does not become 8 GB.

    python -m src.data.prepare --src data/raw/real_vs_fake/real-vs-fake --dst data/processed
    python -m src.data.prepare --src data/raw/my_faces --dst data/processed --split
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Folder names datasets use for each class, normalised to real/ and fake/.
CLASS_ALIASES = {
    "real": {"real", "reals", "original", "originals", "authentic", "live", "0_real"},
    "fake": {"fake", "fakes", "deepfake", "deepfakes", "manipulated", "synthetic",
             "generated", "spoof", "1_fake"},
}
SPLIT_ALIASES = {
    "train": {"train", "training"},
    "val": {"val", "valid", "validation", "dev"},
    "test": {"test", "testing", "eval"},
}


def _canonical(name: str, aliases: Dict[str, set]) -> str | None:
    lowered = name.strip().lower()
    for canonical, options in aliases.items():
        if lowered in options:
            return canonical
    return None


def _list_images(directory: Path) -> List[Path]:
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def _place(src: Path, dst: Path, mode: str) -> None:
    """Materialise `src` at `dst`. Skips files already in place.

    mode='link'    hard-link (default): no extra disk, same filesystem only
    mode='symlink' symbolic link: works across filesystems and read-only mounts
    mode='copy'    real copy: the only option that survives the source moving
    """
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    try:
        dst.hardlink_to(src)
    except (OSError, NotImplementedError):      # cross-device or unsupported FS
        shutil.copy2(src, dst)


def _write_split(images: List[Path], dst_root: Path, split: str, class_name: str,
                 mode: str, face_crop: bool, prefix: str) -> int:
    out_dir = dst_root / split / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cropper = None
    if face_crop:
        from PIL import Image

        from .face import extract_faces
        cropper = (Image, extract_faces)

    written = 0
    for index, path in enumerate(tqdm(images, desc=f"{split}/{class_name}", leave=False)):
        # Prefix keeps names unique when source subfolders repeat filenames.
        target = out_dir / f"{prefix}{index:07d}{path.suffix.lower()}"
        if cropper is not None:
            Image, extract_faces = cropper
            try:
                image = Image.open(path).convert("RGB")
            except OSError:
                continue
            crops = extract_faces(image, max_faces=1)
            crops[0][0].save(target.with_suffix(".jpg"), quality=95)
        else:
            _place(path, target, mode)
        written += 1
    return written


def _detect_layout(src: Path) -> Dict[str, Dict[str, Path]]:
    """Find {split: {class: dir}} or {'_flat': {class: dir}} under src."""
    layout: Dict[str, Dict[str, Path]] = {}

    for child in sorted(p for p in src.iterdir() if p.is_dir()):
        split = _canonical(child.name, SPLIT_ALIASES)
        if split is None:
            continue
        classes = {}
        for grandchild in sorted(p for p in child.iterdir() if p.is_dir()):
            class_name = _canonical(grandchild.name, CLASS_ALIASES)
            if class_name:
                classes[class_name] = grandchild
        if classes:
            layout[split] = classes

    if layout:
        return layout

    flat = {}
    for child in sorted(p for p in src.iterdir() if p.is_dir()):
        class_name = _canonical(child.name, CLASS_ALIASES)
        if class_name:
            flat[class_name] = child
    if flat:
        return {"_flat": flat}

    raise SystemExit(
        f"Could not find real/ and fake/ folders under {src}.\n"
        f"Subdirectories present: {[p.name for p in src.iterdir() if p.is_dir()][:20]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a dataset for training")
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", default="data/processed")
    parser.add_argument("--split", action="store_true",
                        help="create train/val/test yourself (for flat real/ + fake/ sources)")
    parser.add_argument("--ratios", default="0.8,0.1,0.1")
    parser.add_argument("--limit-per-class", type=int, default=None,
                        help="cap images per class per split - useful for a fast first run")
    parser.add_argument("--copy", action="store_true", help="copy instead of hard-linking")
    parser.add_argument("--symlink", action="store_true",
                        help="symlink instead of hard-linking; use when the source is on "
                             "another filesystem or a read-only mount")
    parser.add_argument("--face-crop", action="store_true",
                        help="run face detection and crop (skip if already cropped faces)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    src, dst = Path(args.src).expanduser(), Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"Source directory not found: {src}")

    if args.copy and args.symlink:
        raise SystemExit("Pass at most one of --copy and --symlink.")
    mode = "copy" if args.copy else "symlink" if args.symlink else "link"

    layout = _detect_layout(src)
    totals: Dict[str, Dict[str, int]] = {}

    if "_flat" in layout or args.split:
        ratios = [float(x) for x in args.ratios.split(",")]
        if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
            raise SystemExit("--ratios must be three numbers summing to 1, e.g. 0.8,0.1,0.1")

        # Merge every source split back together, then re-split ourselves.
        merged: Dict[str, List[Path]] = {"real": [], "fake": []}
        for classes in layout.values():
            for class_name, directory in classes.items():
                merged[class_name].extend(_list_images(directory))

        for class_name, images in merged.items():
            random.shuffle(images)
            n_train = int(len(images) * ratios[0])
            n_val = int(len(images) * ratios[1])
            chunks = {
                "train": images[:n_train],
                "val": images[n_train:n_train + n_val],
                "test": images[n_train + n_val:],
            }
            for split, chunk in chunks.items():
                if args.limit_per_class:
                    chunk = chunk[:args.limit_per_class]
                count = _write_split(chunk, dst, split, class_name, mode,
                                     args.face_crop, f"{class_name[0]}_")
                totals.setdefault(split, {})[class_name] = count
    else:
        for split, classes in layout.items():
            for class_name, directory in classes.items():
                images = _list_images(directory)
                random.shuffle(images)
                if args.limit_per_class:
                    images = images[:args.limit_per_class]
                count = _write_split(images, dst, split, class_name, mode,
                                     args.face_crop, f"{class_name[0]}_")
                totals.setdefault(split, {})[class_name] = count

    print(f"\nPrepared dataset at {dst.resolve()}")
    for split in ("train", "val", "test"):
        if split in totals:
            counts = totals[split]
            print(f"  {split:<6}: real={counts.get('real', 0):,}  fake={counts.get('fake', 0):,}")

    missing = [s for s in ("train", "val", "test") if s not in totals]
    if missing:
        print(f"\nWarning: no images for split(s) {missing}. "
              f"Re-run with --split to create them.")


if __name__ == "__main__":
    main()
