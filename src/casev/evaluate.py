"""Metrics, confidence intervals and significance tests.

All metrics are computed from predictions, never read from plots.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize

from .model import CLASS_NAMES


def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """Accuracy, macro P/R/F1, per-class P/R/F1, one-vs-rest ROC-AUC and PR-AUC."""
    k = proba.shape[1]
    y_pred = proba.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(k), zero_division=0)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f.mean()),
    }
    y_bin = label_binarize(y_true, classes=range(k))
    present = y_bin.sum(0) > 0
    if present.all():
        out["roc_auc_ovr_macro"] = roc_auc_score(y_bin, proba, average="macro")
        out["pr_auc_ovr_macro"] = average_precision_score(y_bin, proba, average="macro")
    for i, name in enumerate(CLASS_NAMES[:k]):
        out[f"{name}_precision"], out[f"{name}_recall"], out[f"{name}_f1"] = float(p[i]), float(r[i]), float(f[i])
        out[f"{name}_support"] = int(s[i])
        if present[i]:
            out[f"{name}_roc_auc"] = roc_auc_score(y_bin[:, i], proba[:, i])
            out[f"{name}_pr_auc"] = average_precision_score(y_bin[:, i], proba[:, i])
            out[f"{name}_prevalence"] = float(y_bin[:, i].mean())  # PR-AUC chance level
    return out


def severe_vs_nonsevere_auc(y_true: np.ndarray, proba: np.ndarray, severe_class: int = 2) -> Dict[str, float]:
    """Binary view: high severity vs (low + moderate)."""
    yb = (y_true == severe_class).astype(int)
    return {"roc_auc": roc_auc_score(yb, proba[:, severe_class]), "pr_auc": average_precision_score(yb, proba[:, severe_class])}


def confusion(y_true: np.ndarray, proba: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, proba.argmax(1), labels=range(proba.shape[1]))


def curves(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    """One-vs-rest ROC and PR curve points per class (for plotting)."""
    out = {}
    for i, name in enumerate(CLASS_NAMES[: proba.shape[1]]):
        yb = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(yb, proba[:, i])
        prec, rec, _ = precision_recall_curve(yb, proba[:, i])
        out[name] = {"fpr": fpr, "tpr": tpr, "precision": prec, "recall": rec}
    return out


def bootstrap_ci(y_true: np.ndarray, proba: np.ndarray, metric: str = "macro_f1", n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 42, groups: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Percentile bootstrap CI. If ``groups`` is given, resample whole groups (videos)."""
    rng = np.random.default_rng(seed)
    vals = []
    if groups is not None:
        ug = np.unique(groups)
        idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, len(y_true), len(y_true))
        else:
            idx = np.concatenate([idx_by_g[g] for g in rng.choice(ug, len(ug), replace=True)])
        if len(np.unique(y_true[idx])) < proba.shape[1]:
            continue
        vals.append(classification_metrics(y_true[idx], proba[idx])[metric])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": classification_metrics(y_true, proba)[metric], "lo": float(lo), "hi": float(hi), "n_boot": len(vals)}


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, float]:
    """Exact McNemar test between two classifiers on the same test set."""
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    b01 = int((a_ok & ~b_ok).sum())  # A right, B wrong
    b10 = int((~a_ok & b_ok).sum())  # A wrong, B right
    n = b01 + b10
    p = 1.0 if n == 0 else binomtest(b01, n, 0.5).pvalue
    return {"a_right_b_wrong": b01, "a_wrong_b_right": b10, "p_value": float(p)}


def summarize_folds(rows) -> Dict[str, str]:
    """Mean ± SD over folds for every numeric key."""
    import pandas as pd

    df = pd.DataFrame(rows)
    return {c: f"{df[c].mean():.3f} ± {df[c].std(ddof=1):.3f}" for c in df.columns if np.issubdtype(df[c].dtype, np.number)}
