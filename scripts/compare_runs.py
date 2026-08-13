"""Compare two evaluated runs on the same test set.

`evaluate.py` writes `predictions.npz` (y_true, y_score) into its output
directory. Point this at two of them to get a *paired* significance test rather
than the unpaired approximation you get from comparing two accuracy figures.

McNemar's test is the right tool here: both models saw exactly the same images,
so what matters is the images where they disagree, not the overall totals.

    python -m scripts.compare_runs \
        --a runs/dfnet/eval_test/predictions.npz        --name-a "RGB+SRM" \
        --b runs/dfnet_no_srm/eval_test/predictions.npz --name-b "RGB only"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.utils.metrics import compute_metrics


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int, float]:
    """Exact binomial McNemar on the discordant pairs. Returns (b, c, p)."""
    from math import comb

    only_a = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    only_b = int(np.sum(~correct_a & correct_b))   # B right, A wrong
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0

    # Two-sided exact test against p=0.5.
    k = min(only_a, only_b)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return only_a, only_b, min(1.0, 2 * tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="predictions.npz for model A")
    parser.add_argument("--b", required=True, help="predictions.npz for model B")
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    a = np.load(args.a)
    b = np.load(args.b)
    ya, yb = a["y_true"], b["y_true"]

    if ya.shape != yb.shape:
        raise SystemExit(f"Different test set sizes: {ya.shape} vs {yb.shape}")
    if not np.array_equal(ya, yb):
        raise SystemExit(
            "The two runs have different label orders - they were not evaluated on "
            "the same split in the same order, so a paired test is invalid.")

    print(f"test set: {ya.size:,} images ({int((ya==0).sum()):,} real, "
          f"{int((ya==1).sum()):,} fake)\n")

    for name, scores in ((args.name_a, a["y_score"]), (args.name_b, b["y_score"])):
        m = compute_metrics(ya, scores, args.threshold)
        print(f"{name:12s} acc={m['accuracy']:.4f}  auc={m['auc']:.4f}  "
              f"eer={m['eer']:.4f}  errors={int(m['fp']+m['fn']):,}")

    correct_a = (a["y_score"] >= args.threshold).astype(int) == ya
    correct_b = (b["y_score"] >= args.threshold).astype(int) == yb
    only_a, only_b, p = mcnemar(correct_a, correct_b)

    print(f"\nMcNemar (paired, threshold {args.threshold}):")
    print(f"  {args.name_a} right / {args.name_b} wrong : {only_a:,}")
    print(f"  {args.name_b} right / {args.name_a} wrong : {only_b:,}")
    print(f"  p = {p:.3g}  ->  {'significant' if p < 0.05 else 'NOT significant'} at 0.05")

    auc_a = roc_auc_score(ya, a["y_score"])
    auc_b = roc_auc_score(ya, b["y_score"])
    print(f"\nAUC: {args.name_a}={auc_a:.4f}  {args.name_b}={auc_b:.4f}  "
          f"(diff {auc_a - auc_b:+.4f})")


if __name__ == "__main__":
    main()
