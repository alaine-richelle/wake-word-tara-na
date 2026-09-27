"""Tiny wake-word classifier: depthwise-separable CNN.

Why this architecture?
----------------------
Wake-word detection is a *binary, short-clip* classification problem
("does this ~1 s window contain 'tara na'?").  It does NOT need to transcribe
anything, so a full ASR model (Whisper, wav2vec2, ...) is grossly oversized
and defeats the point of a tiny, on-device gate.

We use a **depthwise-separable CNN** over the 40x128 log-mel spectrogram:

    [1, 40, 128]
      -> (DepthwiseSepConv -> BN -> ReLU -> MaxPool) x 3
      -> Global Average Pooling
      -> Dropout
      -> Linear(hidden) -> ReLU
      -> Linear(2)                 # [unknown, tara_na]

Depthwise-separable convolutions factor a normal convolution into a
per-channel spatial filter followed by a 1x1 channel-mixing filter.  This
cuts the parameter count and FLOPs of each layer by ~8-9x versus a standard
conv at the same width, which is exactly the trade-off we want for a
lightweight, CPU/edge-friendly detector.  Global average pooling removes the
need for a large flatten+FC head and makes the network robust to small timing
shifts of the phrase within the window.

Typical size: ~120k-250k parameters, a few MB in float32, and sub-millisecond
inference per 1.28 s window on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    n_mels: int = 40
    t_frames: int = 128
    in_channels: int = 1
    widths: tuple = (32, 64, 128)   # channel widths of the 3 conv blocks
    kernel: int = 5                 # depthwise kernel size
    stride: int = 1
    pool: int = 2
    hidden: int = 64                # FC hidden width
    dropout: float = 0.3
    num_classes: int = 2

    def as_dict(self) -> Dict:
        d = asdict(self)
        d["widths"] = list(d["widths"])
        return d


class DepthwiseSepConv(nn.Module):
    """Depthwise (spatial) + pointwise (channel) convolution."""

    def __init__(self, cin, cout, kernel, stride=1, pad=None):
        super().__init__()
        if pad is None:
            pad = kernel // 2
        self.dw = nn.Conv2d(cin, cin, kernel, stride=stride, padding=pad, groups=cin)
        self.pw = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class WakeWordCNN(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg
        cin = cfg.in_channels
        blocks = []
        for w in cfg.widths:
            blocks.append(DepthwiseSepConv(cin, w, cfg.kernel, cfg.stride))
            blocks.append(nn.BatchNorm2d(w))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.MaxPool2d(cfg.pool))
            cin = w
        self.features = nn.Sequential(*blocks)
        self.fc = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cin, cfg.hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.num_classes),
        )

    def forward(self, x):
        # x: [B, 1, n_mels, T]
        h = self.features(x)
        h = h.mean(dim=(2, 3))          # global average pooling -> [B, C]
        return self.fc(h)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
