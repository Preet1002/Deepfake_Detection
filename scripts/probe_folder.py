"""Score a folder of images and summarise, for measuring an unseen generator.

`probe_image.py` sweeps preprocessing variants on one image to work out *why* it
fails. This does the complementary job: one score per image over a whole folder,
reported as a detection rate, so a claim about an unseen generator rests on a
measurement rather than an anecdote.

The checkpoint's own test-set results are loaded automatically for comparison.
That contrast - what it catches on the generators it trained on, against what it
catches here - is the whole point.

    python -m scripts.probe_folder data/unseen_generator --label fake

Add a folder of genuine photographs to get an AUC as well:

    python -m scripts.probe_folder data/unseen_generator --label fake \
        --real-dir demo/stylegan__trained-on/real
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from src.data.dataset import IMAGE_EXTENSIONS
from src.predict import Detector


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")
    paths = sorted(p for p in directory.rglob("*")
                   if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise SystemExit(f"No images under {directory}")
    return paths


def score_all(detector: Detector, paths: list[Path], detect_faces: bool) -> np.ndarray:
    scores = []
    for index, path in enumerate(paths, 1):
        image = Image.open(path).convert("RGB")
        scores.append(detector.predict(image, detect_faces=detect_faces,
                                       explain=False).fake_probability)
        print(f"  scoring {index}/{len(paths)}", end="\r", flush=True)
    print(" " * 40, end="\r")
    return np.array(scores)


def reference_rates(checkpoint: Path, thresholds: list[float]) -> list[tuple]:
    """Detection rates on the checkpoint's own test sets, for contrast.

    Reads the predictions.npz files evaluate.py wrote next to the checkpoint.
    Absent (e.g. a checkpoint downloaded on its own), the comparison is simply
    omitted rather than guessed at.
    """
    rows = []
    for npz in sorted(checkpoint.parent.glob("eval_*/predictions.npz")):
        data = np.load(npz)
        fakes = data["y_score"][data["y_true"] == 1]
        if fakes.size:
            rows.append((npz.parent.name, fakes, thresholds))
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory")
    parser.add_argument("--checkpoint", default="runs/dfnet_multi/best.pt")
    parser.add_argument("--label", choices=["fake", "real"], default="fake",
                        help="ground truth for everything in the folder")
    parser.add_argument("--real-dir", default=None,
                        help="folder of genuine photographs, to compute AUC")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.4, 0.3])
    parser.add_argument("--detect-faces", action="store_true",
                        help="run face detection first (default: treat images "
                             "as already-cropped faces)")
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint)
    detector = Detector(str(checkpoint))

    paths = list_images(Path(args.directory))
    print(f"\n=== {args.directory} — {len(paths)} images, "
          f"ground truth = {args.label} ===")
    scores = score_all(detector, paths, args.detect_faces)

    q = np.percentile(scores, [0, 25, 50, 75, 100])
    print(f"\nP(fake)   min {q[0]:.4f}   q1 {q[1]:.4f}   median {q[2]:.4f}   "
          f"q3 {q[3]:.4f}   max {q[4]:.4f}")

    print(f"\n{'threshold':<12}{'flagged fake':>14}{'':4}{'detection rate' if args.label == 'fake' else 'false alarm rate':>18}")
    for threshold in args.thresholds:
        flagged = int((scores >= threshold).sum())
        print(f"{threshold:<12.2f}{f'{flagged} / {len(scores)}':>14}"
              f"{'':4}{flagged / len(scores) * 100:>17.1f}%")

    rows = reference_rates(checkpoint, args.thresholds)
    if rows and args.label == "fake":
        print(f"\nSame checkpoint, same thresholds, on the generators it "
              f"WAS trained on:")
        for name, fakes, thresholds in rows:
            rates = "   ".join(
                f"@{t:.2f}: {(fakes >= t).mean() * 100:.1f}%" for t in thresholds)
            print(f"  {name:<28} {fakes.size:>6,} fakes   {rates}")

    if args.real_dir:
        real_paths = list_images(Path(args.real_dir))
        print(f"\n=== {args.real_dir} — {len(real_paths)} genuine photographs ===")
        real_scores = score_all(detector, real_paths, args.detect_faces)
        y_true = np.concatenate([np.ones(len(scores)), np.zeros(len(real_scores))])
        y_score = np.concatenate([scores, real_scores])
        if args.label == "real":                     # folder was real, not fake
            y_true = 1 - y_true
        print(f"genuine photographs   median P(fake) = {np.median(real_scores):.4f}")
        print(f"\nAUC on this pairing: {roc_auc_score(y_true, y_score):.4f}  "
              f"(0.5 = chance)")

    print(f"\n{len(paths)} images scored. Report the detection rate together "
          f"with the in-training rates above;\nthe contrast between them is the "
          f"finding, not either number on its own.")


if __name__ == "__main__":
    main()
