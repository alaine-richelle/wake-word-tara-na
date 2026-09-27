"""Evaluation, threshold selection and error analysis for the wake-word model.

Wake-word-specific concerns
---------------------------
A false *positive* (model says "tara na" when it wasn't said) wakes the VCM
for nothing and annoys the user, so we weight precision / false-positive-rate
heavily.  A false *negative* (missed wake word) just means the user says it
again.  The threshold sweep below makes that trade-off explicit instead of
blindly using 0.5.

Metrics
-------
accuracy, precision, recall, F1 (binary, on the "tara_na" class), ROC-AUC
(one-vs-rest on the wake class), confusion matrix, false-positive rate and
false-negative rate.  Per-sample predictions are written to CSV for manual
review, and a human-readable error-analysis report is produced.
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..dataset.dataset import LABELS, WAKE_ID


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict:
    """Compute the full binary metric set for the wake class.

    y_true : 0/1 array (1 = wake)
    y_prob : P(wake) in [0,1]
    """
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    n = y_true.size
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "support": int(y_true.sum()),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],  # rows=[actual unk,wake]
    }


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray,
                    thresholds: List[float]) -> pd.DataFrame:
    """Evaluate many thresholds and return a DataFrame (one row each).

    Useful for choosing an operating point that favours low false positives.
    """
    rows = [binary_metrics(y_true, y_prob, t) for t in thresholds]
    df = pd.DataFrame(rows)
    # FPR per 1000 negative windows (a common wake-word reporting unit)
    neg = (y_true == 0).sum()
    df["fpr_per_1000_neg"] = df["fp"] / neg * 1000.0 if neg else 0.0
    return df


def choose_threshold(df: pd.DataFrame, min_recall: float = 0.95) -> float:
    """Pick the operating threshold for a wake-word detector.

    Strategy (robust-first): among thresholds that meet the recall floor AND
    have zero false positives, choose the one with the widest safety margin.
    The margin is how far the threshold sits from both the highest negative
    score and the lowest positive score (both derived from the sweep's TP/FP/
    FN/TN counts).  A wide margin means the decision is stable to small
    probability shifts (real-world noise), which matters more than picking the
    most extreme threshold.  Falls back to 0.5.
    """
    ok = df[(df["recall"] >= min_recall) & (df["fp"] == 0)].copy()
    if len(ok) == 0:
        ok = df[df["recall"] >= min_recall].copy()
    if len(ok) == 0:
        return 0.5
    # lowest positive score = 1 - (FN/(FN+TP)); highest negative = FP/(FP+TN)
    ok["lowest_pos"] = 1.0 - ok["fn"] / (ok["fn"] + ok["tp"]).replace(0, 1)
    ok["highest_neg"] = ok["fp"] / (ok["fp"] + ok["tn"]).replace(0, 1)
    ok["margin"] = np.minimum(ok["threshold"] - ok["highest_neg"],
                              ok["lowest_pos"] - ok["threshold"])
    return float(ok.loc[ok["margin"].idxmax(), "threshold"])


def per_sample_log(y_true: np.ndarray, y_prob: np.ndarray, threshold: float,
                   paths: np.ndarray, kinds: np.ndarray, speakers: np.ndarray) -> pd.DataFrame:
    """One row per sample: input, gold, prediction, confidence, error type."""
    y_pred = (y_prob >= threshold).astype(int)
    err = np.where(y_pred == y_true, "correct",
                   np.where(y_true == 1, "false_negative", "false_positive"))
    return pd.DataFrame({
        "input": paths,
        "gold": np.where(y_true == 1, "tara_na", "unknown"),
        "prediction": np.where(y_pred == 1, "tara_na", "unknown"),
        "prob_wake": np.round(y_prob, 4),
        "confidence": np.round(np.maximum(y_prob, 1 - y_prob), 4),
        "correct": (y_pred == y_true),
        "error_type": err,
        "kind": kinds,
        "speaker": speakers,
    })


def error_report(df: pd.DataFrame, out_path: str) -> str:
    """Human-readable error analysis: FPs, FNs, hardest speakers/kinds."""
    lines = []
    lines.append("=" * 60)
    lines.append("WAKE-WORD ERROR ANALYSIS")
    lines.append("=" * 60)
    n_fp = int((df.error_type == "false_positive").sum())
    n_fn = int((df.error_type == "false_negative").sum())
    n_ok = int(df.correct.sum())
    lines.append(f"samples: {len(df)}   correct: {n_ok}   "
                 f"false_positives: {n_fp}   false_negatives: {n_fn}")
    lines.append("")

    # worst speakers by error count
    if n_fp + n_fn > 0:
        err_df = df[~df.correct]
        lines.append("Errors by speaker (worst first):")
        by_spk = err_df.groupby("speaker").size().sort_values(ascending=False).head(10)
        for spk, cnt in by_spk.items():
            lines.append(f"  {spk}: {cnt} errors")
        lines.append("")
        lines.append("Errors by kind (worst first):")
        by_kind = err_df.groupby("kind").size().sort_values(ascending=False)
        for kind, cnt in by_kind.items():
            lines.append(f"  {kind}: {cnt} errors")
        lines.append("")
        lines.append("Sample FALSE POSITIVES (model fired on non-wake audio):")
        fps = df[df.error_type == "false_positive"].sort_values("prob_wake", ascending=False).head(10)
        for _, r in fps.iterrows():
            lines.append(f"  {r.input}  [gold=unknown, p_wake={r.prob_wake:.3f}]  ({r.kind}/{r.speaker})")
        lines.append("")
        lines.append("Sample FALSE NEGATIVES (missed wake word):")
        fns = df[df.error_type == "false_negative"].sort_values("prob_wake", ascending=False).head(10)
        for _, r in fns.iterrows():
            lines.append(f"  {r.input}  [gold=tara_na, p_wake={r.prob_wake:.3f}]  ({r.kind}/{r.speaker})")

    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text + "\n")
    return text


def plot_confusion(cm: np.ndarray, out_png: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix (rows=actual, cols=predicted)")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["unknown", "tara_na"]); ax.set_yticklabels(["unknown", "tara_na"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
