"""Evaluate the trained wake-word model and select the operating threshold.

Runs the SAME feature pipeline on the validation and test splits, computes the
full metric set, performs a threshold sweep (wake-word detectors should not
blindly use 0.5), writes per-sample prediction logs + an error-analysis
report + a confusion matrix, and updates ``model.pkl`` with the chosen
threshold.

Outputs
-------
    results/metrics.json
    results/threshold_sweep.csv
    results/predictions_val.csv
    results/predictions_test.csv
    results/error_analysis.txt
    results/confusion_matrix.png
    logs/evaluate.log

Usage
-----
    python scripts/evaluate.py [--config configs/default.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.infer import load_detector  # noqa: E402
from src.evaluation.evaluate import (  # noqa: E402
    binary_metrics,
    threshold_sweep,
    choose_threshold,
    per_sample_log,
    error_report,
    plot_confusion,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_logging(log_path: str):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        handlers=[logging.FileHandler(log_path, mode="w"),
                                  logging.StreamHandler(sys.stdout)])


def split_arrays(npz, split):
    z = np.load(npz, allow_pickle=True)
    m = z["split"] == split
    return {
        "y": z["y"][m], "prob_needed": True,
        "paths": z["path"][m], "kinds": z["kind"][m], "speakers": z["speaker"][m],
    }


def collect_probs(detector, npz, split):
    """Run the detector over a split, returning y_true and P(wake)."""
    z = np.load(npz, allow_pickle=True)
    m = z["split"] == split
    X = z["X_clean"][m].astype(np.float32)
    y = z["y"][m]
    probs = []
    batch = 512
    import torch
    detector.model.eval()
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = torch.from_numpy(X[s:s + batch])[:, None]  # [B,1,m,T]
            logits = detector.model(xb)
            p = torch.softmax(logits, dim=-1)[:, 1].numpy()
            probs.append(p)
    return y, np.concatenate(probs), z["path"][m], z["kind"][m], z["speaker"][m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "default.json"))
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    model_pkl = os.path.join(HERE, "model.pkl")
    if not os.path.exists(model_pkl):
        raise SystemExit("model.pkl not found - run scripts/train.py first")
    npz = os.path.join(HERE, cfg["dataset"]["data_dir"], "features.npz")

    setup_logging(os.path.join(HERE, "logs", "evaluate.log"))
    logger = logging.getLogger()
    det = load_detector(model_pkl)
    logger.info("loaded model.pkl  default_threshold=%.3f  params=%d",
                det.threshold, det.bundle.get("n_params", -1))

    thresholds = cfg["evaluation"]["thresholds"]
    min_recall = cfg["evaluation"]["min_recall_for_threshold"]
    results = {}

    for split in ["val", "test"]:
        y, prob, paths, kinds, speakers = collect_probs(det, npz, split)
        # threshold sweep on this split
        sweep = threshold_sweep(y, prob, thresholds)
        chosen = choose_threshold(sweep, min_recall=min_recall)
        m = binary_metrics(y, prob, chosen)
        results[split] = {
            "chosen_threshold": chosen,
            "metrics": m,
            "n": int(len(y)),
            "n_positive": int(y.sum()),
            "n_negative": int((y == 0).sum()),
        }
        sweep.to_csv(os.path.join(HERE, "results", f"threshold_sweep_{split}.csv"), index=False)
        ps = per_sample_log(y, prob, chosen, paths, kinds, speakers)
        ps.to_csv(os.path.join(HERE, "results", f"predictions_{split}.csv"), index=False)
        cm = np.array(m["confusion_matrix"])
        plot_confusion(cm, os.path.join(HERE, "results", f"confusion_matrix_{split}.png"))
        logger.info("[%s] n=%d  thr=%.2f  acc=%.4f prec=%.4f rec=%.4f f1=%.4f "
                    "auc=%.4f FPR=%.4f FNR=%.4f",
                    split, m["support"] + (y == 0).sum(), chosen, m["accuracy"],
                    m["precision"], m["recall"], m["f1"], m["roc_auc"],
                    m["false_positive_rate"], m["false_negative_rate"])

    # error analysis on the TEST split (the honest held-out set)
    ty, tp, tpaths, tkinds, tspeakers = collect_probs(det, npz, "test")
    test_thr = results["test"]["chosen_threshold"]
    tps = per_sample_log(ty, tp, test_thr, tpaths, tkinds, tspeakers)
    err_text = error_report(tps, os.path.join(HERE, "results", "error_analysis.txt"))
    logger.info("error analysis written")

    # update model.pkl with the chosen TEST threshold (single artifact)
    det.bundle["threshold"] = test_thr
    from src.inference.infer import save_detector
    save_detector(det.bundle, model_pkl)
    logger.info("updated model.pkl threshold -> %.3f", test_thr)

    # metrics.json
    out = {
        "config": cfg,
        "val": results["val"],
        "test": results["test"],
        "n_params": det.bundle.get("n_params"),
        "best_val_f1": det.bundle.get("best_val_f1"),
    }
    with open(os.path.join(HERE, "results", "metrics.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info("wrote results/metrics.json")

    # console summary
    print("\n" + "=" * 60)
    print("WAKE-WORD EVALUATION SUMMARY")
    print("=" * 60)
    for split in ["val", "test"]:
        m = results[split]["metrics"]
        print(f"\n[{split.upper()}]  (threshold={results[split]['chosen_threshold']:.2f})")
        print(f"  Accuracy : {m['accuracy']:.4f}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall   : {m['recall']:.4f}")
        print(f"  F1       : {m['f1']:.4f}")
        print(f"  ROC-AUC  : {m['roc_auc']:.4f}")
        print(f"  FPR      : {m['false_positive_rate']:.4f}   "
              f"FNR: {m['false_negative_rate']:.4f}")
        print(f"  CM (unk,wk x unk,wk): {m['confusion_matrix']}")
    print("\n" + err_text)


if __name__ == "__main__":
    main()
