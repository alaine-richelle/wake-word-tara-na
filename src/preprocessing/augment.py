"""Waveform-level augmentation for the wake-word *training* set only.

These augmentations mimic the variations a real device hears from a user:
different speaking speed, pitch, volume, background noise, and the
occasional pause / word break.  They are applied to the raw waveform *before*
feature extraction so the log-mel reflects the perturbation naturally.

Leakage prevention
------------------
Augmentation is only ever called on TRAIN clips.  Validation and test clips
are left untouched (clean features) so that evaluation measures the model on
unseen, unmodified audio.  The train/val/test split is decided *before* any
augmentation, and is speaker-disjoint (see ``src/dataset``).

The config below is deliberately conservative: every knob can be toggled and
its range tuned without touching the training code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.signal import resample_poly


@dataclass(frozen=True)
class AugmentationConfig:
    time_shift: bool = True        # random left/right shift (word position)
    noise: bool = True             # additive background noise
    gain: bool = True              # random volume / gain
    time_stretch: bool = True      # small speed change (speaking rate)
    pitch_shift: bool = True       # small pitch change (voice)
    word_break: bool = True        # insert a short silence mid-clip

    # ranges
    time_shift_frac: float = 0.15  # max shift as fraction of clip length
    noise_db_range: tuple = (-30.0, -12.0)  # SNR range in dB
    gain_db_range: tuple = (-6.0, 4.0)      # volume range in dB
    stretch_range: tuple = (0.85, 1.15)     # speed factor (pitch preserved)
    pitch_semi_range: tuple = (-2.0, 2.0)   # semitone shift
    break_prob: float = 0.35       # probability of inserting a word break
    break_ms_range: tuple = (80, 220)

    def as_dict(self) -> Dict:
        d = {}
        for k, v in self.__dict__.items():
            d[k] = list(v) if isinstance(v, tuple) else v
        return d


AUGMENTATION_CONFIG = AugmentationConfig()


def _resample(x: np.ndarray, factor: float, sr: int = 16000) -> np.ndarray:
    """Resample by ``factor`` (factor>1 => faster/shorter)."""
    g = max(1, int(round(factor * 100)))
    num, den = g, 100
    # reduce
    from math import gcd
    gg = gcd(num, den)
    num, den = num // gg, den // gg
    return resample_poly(x, num, den).astype(np.float32)


def _pitch_shift(x: np.ndarray, semitones: float, sr: int = 16000) -> np.ndarray:
    """Pitch shift by ``semitones`` while preserving duration.

    Standard resample-then-phase-vocoder trick: resample up by 2**(semi/12)
    (raises pitch and shortens), then time-stretch back to the original
    length (restores duration, keeps the new pitch).  Wrapped in a guard so a
    numerical hiccup in the phase vocoder falls back to the unshifted signal
    rather than producing NaNs.
    """
    import librosa

    if abs(semitones) < 1e-3:
        return x
    ratio = 2 ** (semitones / 12.0)
    try:
        x_up = _resample(x, ratio, sr)
        x_out = librosa.effects.time_stretch(x_up, rate=ratio)
        if not np.isfinite(x_out).all():
            return x
        return x_out.astype(np.float32)
    except Exception:
        return x


def _add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white noise at a given SNR (dB)."""
    sig_power = np.mean(x ** 2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = rng.standard_normal(x.shape).astype(np.float32) * np.sqrt(noise_power)
    return (x + noise).astype(np.float32)


def _insert_break(x: np.ndarray, rng: np.random.Generator, ms_range=(80, 220), sr=16000) -> np.ndarray:
    """Insert a short silence gap at a random interior position (word break)."""
    n = x.size
    if n < 400:
        return x
    lo, hi = ms_range
    gap = int(rng.integers(lo, hi) * sr / 1000)
    pos = int(rng.integers(int(n * 0.3), int(n * 0.7)))
    return np.concatenate([x[:pos], np.zeros(gap, dtype=np.float32), x[pos:]]).astype(np.float32)


def augment(
    x: np.ndarray,
    cfg: AugmentationConfig = AUGMENTATION_CONFIG,
    rng: np.random.Generator | None = None,
    sr: int = 16000,
) -> np.ndarray:
    """Apply a random subset of configured augmentations to one waveform.

    Parameters
    ----------
    x : mono float32 waveform at ``sr``.
    cfg : which augmentations to apply and their ranges.
    rng : numpy Generator for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()
    y = x.copy()

    if cfg.pitch_shift:
        semi = float(rng.uniform(*cfg.pitch_semi_range))
        y = _pitch_shift(y, semi, sr)
    if cfg.time_stretch:
        f = float(rng.uniform(*cfg.stretch_range))
        y = _resample(y, f, sr)
    if cfg.gain:
        g = float(rng.uniform(*cfg.gain_db_range))
        y = y * (10 ** (g / 20.0))
    if cfg.word_break and rng.random() < cfg.break_prob:
        y = _insert_break(y, rng, cfg.break_ms_range, sr)
    if cfg.noise:
        snr = float(rng.uniform(*cfg.noise_db_range))
        y = _add_noise(y, snr, rng)
    if cfg.time_shift:
        shift = int(round(float(rng.uniform(-cfg.time_shift_frac, cfg.time_shift_frac)) * y.size))
        if shift > 0:
            y = np.concatenate([np.zeros(shift, dtype=np.float32), y])
        elif shift < 0:
            y = y[-shift:]

    # keep amplitude sane and avoid clipping
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)
