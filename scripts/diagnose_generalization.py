"""Does the detector generalise, or has it memorised one dataset's fingerprint?

Scores three groups with the same checkpoint and compares the distributions:

  1. dataset REAL  (held-out test split)
  2. dataset FAKE  (held-out test split)
  3. external images you supply (e.g. StyleGAN faces saved from the web)

Interpretation:

  Groups 1 and 2 separate cleanly AND external fakes score high
      -> the detector learned genuine generator artefacts.
  Groups 1 and 2 separate cleanly BUT external fakes score like reals
      -> it learned something specific to how THIS dataset was built
         (its compression, resampling or export pipeline), not deepfakery.
         That is a dataset-bias result, and it is worth reporting honestly.

Usage (on Kaggle, where the dataset is mounted):

    python -m scripts.diagnose_generalization \
        --checkpoint /kaggle/working/runs/dfnet/best.pt \
        --data-root  /kaggle/input/.../real-vs-fake \
        --external    /kaggle/working/external_faces \
        --n 200
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from src.data.dataset import IMAGE_EXTENSIONS, resolve_class_dir, resolve_split_dir
from src.predict import Detector


def score_paths(detector: Detector, paths: List[Path], detect_faces: bool) -> np.ndarray:
    scores = []
    for path in paths:
        try:
            image = Image.open(path).convert("RGB")
        except OSError:
            continue
        if detect_faces:
            scores.append(detector.predict(image, detect_faces=True).fake_probability)
        else:
            scores.append(detector._score(image))
    return np.array(scores)


def describe(name: str, scores: np.ndarray, threshold: float = 0.5) -> None:
    if scores.size == 0:
        print(f"{name:24s} (no images)")
        return
    called_fake = float((scores >= threshold).mean())
    print(f"{name:24s} n={scores.size:4d}  "
          f"P(fake): mean={scores.mean():.3f} median={np.median(scores):.3f} "
          f"p10={np.percentile(scores,10):.3f} p90={np.percentile(scores,90):.3f}  "
          f"called FAKE: {called_fake:6.1%}")


def sample_images(directory: Path, n: int, seed: int = 0) -> List[Path]:
    paths = sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if len(paths) > n:
        rng = np.random.default_rng(seed)
        paths = [paths[i] for i in rng.choice(len(paths), n, replace=False)]
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--external", default=None,
                        help="folder of images from outside the dataset")
    parser.add_argument("--external-label", default="fake", choices=["real", "fake"])
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--face-crop-external", action="store_true",
                        help="run face detection on the external images first")
    args = parser.parse_args()

    detector = Detector(args.checkpoint)
    print(f"checkpoint : {args.checkpoint}")
    print(f"device     : {detector.device}   img_size: {detector.img_size}\n")

    split_dir = resolve_split_dir(args.data_root, args.split)
    results = {}
    for class_name in ("real", "fake"):
        class_dir = resolve_class_dir(split_dir, class_name)
        if class_dir is None:
            continue
        # Dataset images are already aligned crops - no face detection.
        scores = score_paths(detector, sample_images(class_dir, args.n), detect_faces=False)
        results[class_name] = scores
        describe(f"dataset {class_name}", scores)

    external = None
    if args.external and Path(args.external).is_dir():
        paths = sample_images(Path(args.external), args.n)
        external = score_paths(detector, paths, detect_faces=args.face_crop_external)
        describe(f"external ({args.external_label})", external)

    print()
    if "real" in results and "fake" in results:
        gap = results["fake"].mean() - results["real"].mean()
        print(f"in-dataset separation (mean fake - mean real): {gap:+.3f}")
        if gap > 0.5:
            print("  -> the model separates the two classes on its own data.")

    if external is not None and "real" in results:
        expected_high = args.external_label == "fake"
        hit_rate = float((external >= 0.5).mean()) if expected_high else float((external < 0.5).mean())
        print(f"external agreement with its true label: {hit_rate:.1%}")
        if expected_high and external.mean() < results["real"].mean() + 0.1:
            print("  -> external fakes score like dataset REALS. The detector is keying on\n"
                  "     something specific to this dataset, not on generator artefacts.\n"
                  "     Report this: it is the honest and interesting result.")


if __name__ == "__main__":
    main()
