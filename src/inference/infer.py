"""Inference against a saved wake-word model (``model.pkl``).

The pickle bundle produced by ``train.py`` contains everything needed to run
inference in a fresh process:

    {
      "model_state": OrderedDict,     # WakeWordCNN state_dict
      "model_cfg":  dict,             # ModelConfig
      "feature_cfg": dict,            # FeatureConfig
      "labels":     ["unknown","tara_na"],
      "threshold":  float,            # operating point chosen by threshold sweep
      "wake_id":    int,
    }

The SAME feature pipeline used at training time is applied here (16 kHz mono
-> log-mel -> frame), so live/file audio is scored identically to training.
"""
from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from ..model.vcm import WakeWordCNN, ModelConfig
from ..preprocessing.audio import (
    FEATURE_CONFIG,
    FeatureConfig,
    frame,
    load_wav_16k,
    logmel,
)


@dataclass
class Prediction:
    label: str
    wake: bool
    confidence: float          # P(tara_na)
    prob_unknown: float
    prob_wake: float
    latency_ms: float
    threshold: float
    uncertain: bool = False    # True when confidence is close to threshold


class WakeWordDetector:
    def __init__(self, bundle: Dict):
        self.bundle = bundle
        self.feature_cfg = FeatureConfig(**bundle["feature_cfg"])
        self.model_cfg = ModelConfig(**bundle["model_cfg"])
        self.labels = bundle["labels"]
        self.wake_id = bundle["wake_id"]
        self.threshold = float(bundle.get("threshold", 0.5))
        self.device = torch.device("cpu")
        self.model = WakeWordCNN(self.model_cfg)
        self.model.load_state_dict(bundle["model_state"])
        self.model.to(self.device).eval()
        # Warm up: the first forward triggers lazy kernel/graph setup and is
        # much slower than steady-state.  Run one dummy pass so reported
        # latency reflects real per-window cost, not one-time initialization.
        with torch.no_grad():
            self.model(torch.zeros(1, 1, self.model_cfg.n_mels, self.model_cfg.t_frames))

    # ---- feature pipeline (identical to training) -------------------------
    def _features(self, x: np.ndarray) -> torch.Tensor:
        L = logmel(x, self.feature_cfg)
        F = frame(L, self.feature_cfg.t_frames)
        return torch.from_numpy(F.astype(np.float32))[None, None]  # [1,1,m,T]

    @torch.no_grad()
    def predict_waveform(self, x: np.ndarray, threshold: Optional[float] = None) -> Prediction:
        thr = self.threshold if threshold is None else float(threshold)
        feats = self._features(x).to(self.device)
        t0 = time.perf_counter()
        logits = self.model(feats)
        lat = (time.perf_counter() - t0) * 1000.0
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        p_wake = float(probs[self.wake_id])
        p_unk = float(probs[0])
        wake = p_wake >= thr
        margin = abs(p_wake - thr)
        uncertain = margin < 0.10  # within 0.10 of threshold -> flag as uncertain
        label = "tara_na" if wake else "unknown"
        return Prediction(
            label=label,
            wake=wake,
            confidence=p_wake,
            prob_unknown=p_unk,
            prob_wake=p_wake,
            latency_ms=float(lat),
            threshold=thr,
            uncertain=uncertain,
        )

    def predict_file(self, path: str, threshold: Optional[float] = None) -> Prediction:
        x = load_wav_16k(path, self.feature_cfg.sr)
        return self.predict_waveform(x, threshold)


def load_detector(model_pkl: str) -> WakeWordDetector:
    with open(model_pkl, "rb") as f:
        bundle = pickle.load(f)
    return WakeWordDetector(bundle)


def save_detector(bundle: Dict, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


# ---------------------------------------------------------------------------
# Microphone capture (graceful degradation when no audio device is present)
# ---------------------------------------------------------------------------
def record_microphone(duration: float = 2.0, sr: int = 16000) -> Optional[np.ndarray]:
    """Record ``duration`` seconds from the default microphone.

    Returns a float32 mono waveform, or ``None`` if no microphone / sounddevice
    is available (headless environments).  Callers should check for ``None``.
    """
    try:
        import sounddevice as sd
    except Exception:
        return None
    try:
        rec = sd.rec(
            int(duration * sr), samplerate=sr, channels=1, dtype="float32"
        )
        sd.wait()
        return rec[:, 0].astype(np.float32)
    except Exception:
        return None


def microphone_available() -> bool:
    try:
        import sounddevice as sd  # noqa
        return len(sd.query_devices()) > 0
    except Exception:
        return False
