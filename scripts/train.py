"""Train the tiny "tara na" wake-word classifier.

Reproducible, logged, and produces exactly ONE model artifact (``model.pkl``).
Best model is tracked by validation F1; early stopping prevents overfitting.

Usage
-----
    python scripts/train.py [--config configs/default.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.dataset import (  # noqa: E402
    DatasetConfig,
    WakeWordDataset,
)
from src.model.vcm import ModelConfig, WakeWordCNN  # noqa: E402
from src.preprocessing.audio import FeatureConfig  # noqa: E402
from src.inference.infer import save_detector  # noqa: E402
from src.evaluation.evaluate import binary_metrics  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_logging(log_path: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_probs(model, loader, device) -> tuple:
    """Return (y_true, y_prob_wake) for a split, batching the forward pass."""
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for xb, yb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=-1)
            yp.append(probs[:, 1].cpu().numpy())
            yt.append(yb.numpy())
    return np.concatenate(yt), np.concatenate(yp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "default.json"))
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    tr = cfg["training"]
    feat_cfg = FeatureConfig(**cfg["features"])
    model_cfg = ModelConfig(**{**cfg["model"]})
    ds_cfg = DatasetConfig(
        data_dir=os.path.join(HERE, cfg["dataset"]["data_dir"]),
        feature_cfg=feat_cfg,
        seed=cfg["seed"],
    )

    npz = os.path.join(HERE, cfg["dataset"]["data_dir"], "features.npz")
    if not os.path.exists(npz):
        raise SystemExit("features.npz not found - run scripts/prepare_dataset.py first")

    log_path = os.path.join(HERE, "logs", "train.log")
    setup_logging(log_path)
    logger = logging.getLogger()

    set_seed(cfg["seed"])
    device = torch.device(tr["device"] if tr["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    logger.info("device=%s  seed=%d", device, cfg["seed"])

    train_ds = WakeWordDataset(npz, "train", ds_cfg)
    val_ds = WakeWordDataset(npz, "val", ds_cfg)
    logger.info("train items=%d (clips=%d)  val items=%d",
                len(train_ds), train_ds.meta()["n_clips"], len(val_ds))

    train_dl = DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=True,
                          num_workers=0, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=tr["batch_size"], shuffle=False, num_workers=0)

    model = WakeWordCNN(model_cfg).to(device)
    n_params = model.count_params()
    logger.info("model params=%d (%.2f MB float32)", n_params, n_params * 4 / 1e6)

    # class weighting (positives are the minority class)
    pos_weight = float(tr.get("pos_weight", 1.0))
    if pos_weight == "auto":
        pos = train_ds.meta()["n_positive"]
        neg = train_ds.meta()["n_negative"]
        pos_weight = neg / max(pos, 1)
        logger.info("auto pos_weight=%.3f", pos_weight)
    class_w = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_w)

    optimizer = torch.optim.Adam(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    best_f1 = -1.0
    best_state = None
    patience = tr["early_stopping_patience"]
    bad_epochs = 0
    history = []

    for epoch in range(tr["epochs"]):
        t0 = time.time()
        model.train()
        tot_loss, tot_correct, tot_n = 0.0, 0, 0
        for xb, yb, _ in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * len(yb)
            tot_correct += (torch.argmax(logits, 1) == yb).sum().item()
            tot_n += len(yb)
        train_acc = tot_correct / tot_n
        train_loss = tot_loss / tot_n

        yt, yp = evaluate_probs(model, val_dl, device)
        vm = binary_metrics(yt, yp, 0.5)
        scheduler.step(vm["f1"])
        dt = time.time() - t0
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": vm["accuracy"], "val_f1": vm["f1"], "val_acc": vm["accuracy"],
            "val_precision": vm["precision"], "val_recall": vm["recall"],
            "val_auc": vm["roc_auc"], "sec": dt,
        })
        logger.info("epoch %02d  train_loss=%.4f train_acc=%.4f | "
                    "val_f1=%.4f val_acc=%.4f prec=%.4f rec=%.4f auc=%.4f  (%.1fs)",
                    epoch + 1, train_loss, train_acc, vm["f1"], vm["accuracy"],
                    vm["precision"], vm["recall"], vm["roc_auc"], dt)

        if vm["f1"] > best_f1:
            best_f1 = vm["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info("early stopping at epoch %d (patience=%d)", epoch + 1, patience)
                break

    # restore best
    model.load_state_dict(best_state)
    model.to(device).eval()
    logger.info("best val F1=%.4f", best_f1)

    # ---- save the SINGLE canonical model.pkl -----------------------------
    bundle = {
        "model_state": best_state,
        "model_cfg": model_cfg.as_dict(),
        "feature_cfg": feat_cfg.as_dict(),
        "labels": ["unknown", "tara_na"],
        "wake_id": 1,
        "threshold": cfg["evaluation"]["default_threshold"],  # refined by evaluate.py
        "seed": cfg["seed"],
        "n_params": n_params,
        "best_val_f1": best_f1,
        "history": history,
    }
    model_pkl = os.path.join(HERE, "model.pkl")
    save_detector(bundle, model_pkl)
    logger.info("saved canonical model -> %s (%.2f MB)", model_pkl, os.path.getsize(model_pkl) / 1e6)

    with open(os.path.join(HERE, "logs", "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    logger.info("done")


if __name__ == "__main__":
    main()
