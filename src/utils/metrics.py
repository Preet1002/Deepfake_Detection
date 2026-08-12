"""Binary-classification metrics reported for the detector.

Convention throughout the project: label 1 = FAKE, label 0 = REAL.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def equal_error_rate(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """EER plus the threshold that achieves it.

    EER is the operating point where false-accept and false-reject rates match;
    it is the standard single-number summary in forensics papers because it does
    not depend on an arbitrary 0.5 cutoff.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = (y_score >= threshold).astype(int)

    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # AUC/EER/AP are undefined when a split contains a single class.
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, y_score))
        out["ap"] = float(average_precision_score(y_true, y_score))
        eer, eer_thr = equal_error_rate(y_true, y_score)
        out["eer"] = eer
        out["eer_threshold"] = eer_thr
    else:
        out.update({"auc": float("nan"), "ap": float("nan"),
                    "eer": float("nan"), "eer_threshold": 0.5})

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out


def format_metrics(metrics: Dict[str, float]) -> str:
    keys = ["accuracy", "auc", "eer", "precision", "recall", "f1"]
    return "  ".join(f"{k}={metrics[k]:.4f}" for k in keys if k in metrics)
