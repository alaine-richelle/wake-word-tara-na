"""Dataset construction for the "tara na" wake-word detector.

Data layout
-----------
``prepare_dataset.py`` synthesizes (via espeak) and organizes audio into::

    data/
        positive/            # clips that CONTAIN "tara na"
            w0000.wav ...    # one file per (speaker, speed, pitch) combo
        negative/            # clips that DO NOT contain "tara na"
            w0000.wav ...
        noise/               # pure background-noise clips (negatives)
        silence/             # pure silence clips (negatives)
        manifest.csv         # path,label,split,speaker,kind,duration_sec,...

Labels are binary: ``tara_na`` (1) vs ``unknown`` (0).

Speaker-disjoint split (leakage prevention)
-------------------------------------------
The espeak *voice* acts as the "speaker".  We partition voices into three
disjoint groups so no voice appears in more than one split:

    train : 80 voices
    val   : 10 voices
    test  : 10 voices

This mirrors the OptionB VCM dataset (80/10/10) and guarantees the model is
scored on voices it never heard in training.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..preprocessing.audio import (
    FEATURE_CONFIG,
    FeatureConfig,
    frame,
    load_wav_16k,
    logmel,
)
from ..preprocessing.augment import AUGMENTATION_CONFIG, AugmentationConfig

LABELS = ["unknown", "tara_na"]
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
WAKE_ID = LABEL2ID["tara_na"]


@dataclass(frozen=True)
class DatasetConfig:
    data_dir: str
    feature_cfg: FeatureConfig = FEATURE_CONFIG
    aug_cfg: AugmentationConfig = AUGMENTATION_CONFIG
    seed: int = 0


class WakeWordDataset(Dataset):
    """Binary wake-word dataset.

    Loads precomputed features from ``features.npz`` (built by
    ``prepare_dataset.py``).  For TRAIN split it can additionally apply the
    cached augmented features (already computed at build time, train-only).
    """

    def __init__(self, npz_path: str, split: str, cfg: DatasetConfig):
        self.cfg = cfg
        z = np.load(npz_path, allow_pickle=True)
        self.X = z["X_clean"].astype(np.float32)          # [N, n_mels, T]
        self.X_aug = z["X_aug"] if "X_aug" in z else None  # [Ntr, n_mels, T]
        self.y = z["y"].astype(np.int64)                  # [N]
        self.split_arr = z["split"]                       # [N]
        self.speaker = z["speaker"]                        # [N]
        self.kind = z["kind"]                              # [N]
        self.path = z["path"]                              # [N]

        mask = self.split_arr == split
        self.idx = np.where(mask)[0]
        self._rng = np.random.default_rng(cfg.seed + hash(split) % 1000)

        # train uses clean + augmented copies (each train clip appears twice)
        self.use_aug = (split == "train") and self.X_aug is not None

    def __len__(self) -> int:
        n = len(self.idx)
        return 2 * n if self.use_aug else n

    def _clean(self, i: int):
        return self.X[i], int(self.y[i]), i

    def __getitem__(self, k: int):
        if self.use_aug:
            if k < len(self.idx):
                i = self.idx[k]
                feat = self.X[i]
            else:
                i = self.idx[k - len(self.idx)]
                feat = self.X_aug[k - len(self.idx)]
        else:
            i = self.idx[k]
            feat = self.X[i]
        return (
            torch.from_numpy(feat.astype(np.float32))[None],  # [1, n_mels, T]
            torch.tensor(int(self.y[i]), dtype=torch.long),
            int(i),
        )

    def meta(self) -> Dict:
        m = self.split_arr == self.idx[0] if len(self.idx) else None
        return {
            "split": self.idx[0] and self.split_arr[self.idx[0]].decode() if len(self.idx) else None,
            "n_clips": len(self.idx),
            "n_items": len(self),
            "n_speakers": int(np.unique(self.speaker[self.idx]).size),
            "n_positive": int((self.y[self.idx] == WAKE_ID).sum()),
            "n_negative": int((self.y[self.idx] == 0).sum()),
        }


def speaker_disjoint_split(
    speakers: List[str], train_frac: float = 0.8, val_frac: float = 0.1, seed: int = 0
) -> Dict[str, List[str]]:
    """Partition a list of unique speakers into disjoint train/val/test sets.

    Deterministic given ``seed``.  Returns {"train": [...], "val": [...], "test": [...]}.
    """
    uniq = sorted(set(speakers))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    uniq = [uniq[i] for i in order]
    n = len(uniq)
    n_tr = int(round(n * train_frac))
    n_va = int(round(n * val_frac))
    return {
        "train": uniq[:n_tr],
        "val": uniq[n_tr : n_tr + n_va],
        "test": uniq[n_tr + n_va :],
    }
