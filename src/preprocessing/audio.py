"""Audio loading and feature extraction for the tiny wake-word model.

Why a log-mel spectrogram?
--------------------------
A raw waveform at 16 kHz is ~16,000 numbers per second.  A tiny CNN cannot
learn robust speech features directly from that, and it is too large for an
on-device model.  We therefore compress each clip into a *log-mel
spectrogram*:

    waveform
        -> STFT (short-time Fourier transform)
        -> power spectrum
        -> mel filterbank (40 triangular filters, 20 Hz - 8 kHz)
        -> log scale (approximates the ear's perceptual loudness)
        -> rescale to [0, 1]

This gives a compact ``[40, T]`` image-like tensor that a small CNN handles
well, at a fraction of the memory/latency of a raw waveform.  It is the same
representation used by the companion Tiny VCM (ME2), so the two models share
one feature pipeline.

The constants below are deliberately small (40 mels, 512-pt FFT, 160-sample
hop) to keep the model lightweight and fast on CPU / edge hardware.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import librosa


@dataclass(frozen=True)
class FeatureConfig:
    """Immutable feature-extraction settings.  Saved into model.pkl so that
    inference always reproduces training-time features exactly."""

    sr: int = 16000          # target sample rate
    n_mels: int = 40         # number of mel bins
    n_fft: int = 512         # STFT window (32 ms)
    hop: int = 160           # hop length (10 ms) -> 100 frames/sec
    fmin: int = 20           # lowest mel frequency
    fmax: int = 8000         # highest mel frequency (half of 16 kHz)
    t_frames: int = 128      # fixed number of time frames (1.28 s)
    db_range: float = 80.0   # dynamic range: (L + db_range) / db_range

    def as_dict(self) -> Dict:
        return asdict(self)


FEATURE_CONFIG = FeatureConfig()


def load_wav_16k(path: str, sr_target: int = FEATURE_CONFIG.sr) -> np.ndarray:
    """Load any audio file, downmix to mono and resample to ``sr_target``.

    Returns a float32 mono waveform in roughly [-1, 1].
    Raises on unreadable / corrupt files (caller decides how to handle).
    """
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != sr_target:
        g = math.gcd(int(sr), sr_target)
        x = resample_poly(x, sr_target // g, int(sr) // g)
    return x.astype(np.float32)


def inspect_audio(path: str) -> Dict:
    """Return a small dictionary describing one audio file.

    Used by the dataset-inspection stage to log sample rate, channels,
    duration, energy and to flag silent / near-silent / corrupt files.
    """
    try:
        info = sf.info(path)
    except Exception as exc:  # unreadable / corrupt
        return {"path": path, "ok": False, "error": repr(exc)}

    x, sr = sf.read(path, dtype="float32", always_2d=False)
    mono = x.mean(axis=1) if x.ndim == 2 else x
    n = int(mono.size)
    rms = float(np.sqrt(np.mean(mono ** 2))) if n else 0.0
    peak = float(np.max(np.abs(mono))) if n else 0.0
    dur = n / sr if sr else 0.0
    # "silent" heuristic: RMS below -50 dBFS
    silent = rms < (10 ** (-50 / 20))
    return {
        "path": path,
        "ok": True,
        "format": info.format,
        "subtype": getattr(info, "subtype", ""),
        "sample_rate": int(sr),
        "channels": int(info.channels),
        "num_samples": n,
        "duration_sec": round(dur, 3),
        "dtype": str(x.dtype),
        "min": float(mono.min()) if n else 0.0,
        "max": float(mono.max()) if n else 0.0,
        "rms": round(rms, 5),
        "peak": round(peak, 5),
        "silent": bool(silent),
    }


def logmel(x: np.ndarray, cfg: FeatureConfig = FEATURE_CONFIG) -> np.ndarray:
    """Waveform -> log-mel spectrogram ``[n_mels, T]`` scaled to ~[0, 1]."""
    mel = librosa.filters.mel(
        sr=cfg.sr, n_fft=cfg.n_fft, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax
    )
    S = np.abs(librosa.stft(x, n_fft=cfg.n_fft, hop_length=cfg.hop)) ** 2
    L = librosa.amplitude_to_db(mel @ S, ref=np.max)
    L = (L + cfg.db_range) / cfg.db_range
    return L.astype(np.float32)


def frame(L: np.ndarray, t_frames: int = FEATURE_CONFIG.t_frames) -> np.ndarray:
    """Pad or truncate a spectrogram to exactly ``t_frames`` columns.

    Short clips are right-padded with zeros (silence); long clips are
    truncated.  This yields a fixed input shape ``[n_mels, t_frames]`` so the
    model can be applied to clips of any length.
    """
    t = L.shape[1]
    if t >= t_frames:
        return L[:, :t_frames]
    pad = np.zeros((L.shape[0], t_frames - t), dtype=L.dtype)
    return np.concatenate([L, pad], axis=1)


def waveform_to_features(x: np.ndarray, cfg: FeatureConfig = FEATURE_CONFIG) -> np.ndarray:
    """Convenience: mono 16 kHz waveform -> fixed ``[n_mels, t_frames]`` feature."""
    return frame(logmel(x, cfg), cfg.t_frames)
