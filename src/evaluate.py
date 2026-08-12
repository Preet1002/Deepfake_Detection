"""Evaluate a checkpoint on a split and write report-ready figures.

    python -m src.evaluate --checkpoint runs/dfnet/best.pt --split test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # no display on a headless run
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve
from tqdm import tqdm

from .config import Config, DataConfig, ModelConfig
from .data.dataset import FaceImageDataset, resolve_split_dir
from .data.transforms import build_eval_transform
from .models.dfnet import build_model
from .utils.common import dump_json, load_checkpoint, pick_device
from .utils.metrics import compute_metrics


def plot_roc(y_true, y_score, auc: float, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"DFNet (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC - deepfake detection")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion(metrics: dict, path: Path) -> None:
    matrix = np.array([[metrics["tn"], metrics["fp"]],
                       [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap="Blues")
    labels = ["real", "fake"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    threshold = matrix.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main(argv: list[str] | None = None) -> None:
    """See src.train.main for why argv is injectable."""
    parser = argparse.ArgumentParser(description="Evaluate a DFNet checkpoint")
    parser.add_argument("--checkpoint", default="runs/dfnet/best.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-root", default=None,
                        help="override the data root (use this for cross-dataset tests)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    device = pick_device("auto")
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    saved = checkpoint.get("config", {})
    model_config = ModelConfig(**saved["model"]) if "model" in saved else ModelConfig()
    data_config = DataConfig(**saved["data"]) if "data" in saved else DataConfig()

    root = Path(args.data_root or data_config.root)
    out_dir = Path(args.out_dir or (Path(args.checkpoint).parent / f"eval_{args.split}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(model_config).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])

    split_dir = resolve_split_dir(root, args.split)
    # No limit_per_class here on purpose: the test set is always scored in full,
    # even when training ran on a capped subset.
    dataset = FaceImageDataset(split_dir, build_eval_transform(data_config.img_size))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=data_config.num_workers)
    print(f"Evaluating {len(dataset):,} images from {split_dir} {dataset.class_counts()}")

    scores, labels = [], []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="eval"):
            logits = model(images.to(device))
            scores.append(torch.sigmoid(logits).float().cpu().numpy())
            labels.append(targets.numpy())

    y_score = np.concatenate(scores)
    y_true = np.concatenate(labels)
    metrics = compute_metrics(y_true, y_score)

    print("\n=== Results ===")
    for key in ["accuracy", "auc", "ap", "eer", "precision", "recall", "f1"]:
        if key in metrics:
            print(f"  {key:<10}: {metrics[key]:.4f}")
    print(f"  {'TN/FP/FN/TP':<10}: {metrics['tn']}/{metrics['fp']}/"
          f"{metrics['fn']}/{metrics['tp']}")

    dump_json(out_dir / "metrics.json", metrics)
    np.savez(out_dir / "predictions.npz", y_true=y_true, y_score=y_score)
    if not np.isnan(metrics["auc"]):
        plot_roc(y_true, y_score, metrics["auc"], out_dir / "roc.png")
    plot_confusion(metrics, out_dir / "confusion_matrix.png")
    print(f"\nFigures and metrics written to {out_dir}")


if __name__ == "__main__":
    main()
