"""Prepare the "tara na" wake-word dataset.

Pipeline
--------
1. Build a pool of *distinct* espeak "speakers" (English accent bases x
   gender/whisper/low modifiers), deduplicated by log-mel spectral distance so
   we get ~100 genuinely different timbres.
2. Partition those speakers into **disjoint** train/val/test groups (80/10/10)
   so no speaker leaks across splits (leakage prevention).
3. Synthesize POSITIVE clips of the exact phrase "tara na" (varying speed and
   pitch per speaker) and NEGATIVE clips (real VCM command phrases + phonetic
   distractors), plus pure NOISE and SILENCE negatives.
4. Extract 40x128 log-mel features for every clip; compute one augmented copy
   of every TRAIN clip (train-only, no leakage).
5. Write ``data/manifest.csv`` and ``data/features.npz``.

Raw WAVs + the feature cache are generated locally and NOT committed.

Usage
-----
    python scripts/prepare_dataset.py [--config configs/default.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
import librosa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.audio import (  # noqa: E402
    FEATURE_CONFIG,
    FeatureConfig,
    frame,
    load_wav_16k,
    logmel,
)
from src.preprocessing.augment import (  # noqa: E402
    AUGMENTATION_CONFIG,
    AugmentationConfig,
    augment,
)
from src.dataset.dataset import LABEL2ID, speaker_disjoint_split  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NEGATIVE_PHRASES = [
    "play music", "what is the weather", "what time is it",
    "turn on the lights", "turn off the lights", "dim the lights to fifty percent",
    "set a timer for ten minutes", "set an alarm for six am",
    "set the temperature to twenty two degrees", "pause", "stop",
    "next track", "volume up", "volume down", "make it louder",
    "remind me to drink water", "what are my reminders", "call mom",
    # phonetic distractors (close sounds, not the wake word)
    "dara na", "tara no", "tara la", "cara na", "tara nah",
    "tara", "na", "take a note", "talk to me", "turn around",
]

# English accent bases x modifiers that actually change the timbre.
ESPEAK_BASES = [
    "en-us", "en-us-n", "en-us-r", "en-gb", "en-gb-n", "en-gb-r",
    "en-gb-sc", "en-gb-w", "en-gb-ct", "en-gb-x", "en-gb-vo",
    "en-au", "en-ca", "en-ie", "en-za", "en-nz", "en-sc",
    "en-029", "en-0880", "en-jb", "en-hk",
]
ESPEAK_MODS = ["", "+f3", "+f4", "+m3", "+m4", "+whisper", "+low"]


def synth(text: str, voice: str, speed: int, pitch: int, out_path: str) -> bool:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        subprocess.run(
            ["espeak", text, "-v", voice, "-s", str(speed), "-p", str(pitch),
             "-w", out_path],
            check=True, capture_output=True, timeout=30,
        )
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except Exception:
        return False


def _mel_mean(x: np.ndarray, cfg) -> np.ndarray:
    mel = librosa.filters.mel(sr=cfg.sr, n_fft=cfg.n_fft, n_mels=cfg.n_mels,
                              fmin=cfg.fmin, fmax=cfg.fmax)
    S = np.abs(librosa.stft(x, n_fft=cfg.n_fft, hop_length=cfg.hop)) ** 2
    return (mel @ S).mean(axis=1)


def build_speaker_pool(cfg, feat_cfg, n_speakers: int, rng) -> list:
    """Return ``n_speakers`` distinct espeak voices (greedy farthest-point)."""
    pool = []
    for base in ESPEAK_BASES:
        for mod in ESPEAK_MODS:
            v = base + mod
            if synth("tara na", v, 150, 55, os.path.join(cfg["data_dir"], "_probe.wav")):
                x = load_wav_16k(os.path.join(cfg["data_dir"], "_probe.wav"), feat_cfg.sr)
                pool.append((v, _mel_mean(x, feat_cfg)))
    if os.path.exists(os.path.join(cfg["data_dir"], "_probe.wav")):
        os.remove(os.path.join(cfg["data_dir"], "_probe.wav"))

    if len(pool) < n_speakers:
        print(f"warning: only {len(pool)} distinct voices available, using all")
        n_speakers = len(pool)

    feats = np.stack([f for _, f in pool])
    names = [v for v, _ in pool]
    # full pairwise distance matrix (vectorized)
    diff = feats[:, None, :] - feats[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    start = int(rng.integers(len(pool)))
    selected = [start]
    for _ in range(n_speakers - 1):
        # for each unselected candidate, its distance to the nearest selected
        cand_mask = np.ones(len(pool), dtype=bool)
        cand_mask[selected] = False
        nearest = dist[:, selected].min(axis=1)
        nearest[~cand_mask] = -1.0
        selected.append(int(np.argmax(nearest)))
    chosen = [names[i] for i in selected]
    cf = dist[np.ix_(selected, selected)]
    iu = np.triu_indices(len(selected), k=1)
    pair = cf[iu]
    print(f"speaker pool: {len(pool)} candidates -> {len(chosen)} distinct "
          f"(min pairwise mel-dist {np.min(pair):.2f}, median {np.median(pair):.2f})")
    return chosen


def make_noise_clip(rng, sr=16000, dur=1.28):
    n = int(dur * sr)
    white = rng.standard_normal(n)
    pink = np.cumsum(rng.standard_normal(n)) * 0.3
    x = 0.6 * white + 0.4 * pink
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.3
    return x.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "default.json"))
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    ds = cfg["dataset"]
    feat_cfg = FeatureConfig(**cfg["features"])
    aug_cfg = AugmentationConfig(**{
        **AUGMENTATION_CONFIG.__dict__,
        **{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg["augmentation"].items()},
    })
    rng = np.random.default_rng(cfg["seed"])

    data_dir = os.path.join(HERE, ds["data_dir"])
    pos_dir = os.path.join(data_dir, "positive")
    neg_dir = os.path.join(data_dir, "negative")
    noise_dir = os.path.join(data_dir, "noise")
    silence_dir = os.path.join(data_dir, "silence")
    cfg["data_dir"] = data_dir

    # ---- distinct speaker pool + disjoint split --------------------------
    voices = build_speaker_pool(cfg, feat_cfg, ds["n_speakers_total"], rng)
    split_map = speaker_disjoint_split(
        voices,
        train_frac=ds["train_speakers"] / ds["n_speakers_total"],
        val_frac=ds["val_speakers"] / ds["n_speakers_total"],
        seed=cfg["seed"],
    )
    voice_to_split = {v: s for s, vs in split_map.items() for v in vs}
    print("split sizes:", {k: len(v) for k, v in split_map.items()})

    rows = []

    def add(path, label, speaker, kind, dur):
        rows.append({
            "path": os.path.relpath(path, data_dir),
            "label": label,
            "split": voice_to_split.get(speaker, "train"),
            "speaker": speaker,
            "kind": kind,
            "duration_sec": round(dur, 3),
        })

    # ---- positives --------------------------------------------------------
    print("synthesizing positives ...")
    for si, voice in enumerate(voices):
        for pi in range(ds["positives_per_speaker"]):
            speed = int(rng.choice(ds["speeds"]))
            pitch = int(ds["base_pitch"] + rng.integers(-ds["pitch_jitter"], ds["pitch_jitter"]))
            out = os.path.join(pos_dir, f"p_{si}_{pi}.wav")
            if synth(cfg["wake_phrase"], voice, speed, pitch, out):
                x = load_wav_16k(out, feat_cfg.sr)
                add(out, "tara_na", voice, "positive", len(x) / feat_cfg.sr)
        if (si + 1) % 20 == 0:
            print(f"  positives {si+1}/{len(voices)}")

    # ---- negatives --------------------------------------------------------
    print("synthesizing negatives ...")
    for si, voice in enumerate(voices):
        for ni in range(ds["negatives_per_speaker"]):
            phrase = NEGATIVE_PHRASES[ni % len(NEGATIVE_PHRASES)]
            speed = int(rng.choice(ds["speeds"]))
            pitch = int(ds["base_pitch"] + rng.integers(-ds["pitch_jitter"], ds["pitch_jitter"]))
            out = os.path.join(neg_dir, f"n_{si}_{ni}.wav")
            if synth(phrase, voice, speed, pitch, out):
                x = load_wav_16k(out, feat_cfg.sr)
                add(out, "unknown", voice, "negative", len(x) / feat_cfg.sr)
        if (si + 1) % 20 == 0:
            print(f"  negatives {si+1}/{len(voices)}")

    # ---- noise + silence negatives ---------------------------------------
    print("generating noise + silence ...")
    for i in range(ds["noise_clips"]):
        x = make_noise_clip(rng, feat_cfg.sr)
        out = os.path.join(noise_dir, f"noise_{i}.wav")
        os.makedirs(noise_dir, exist_ok=True)
        sf.write(out, x, feat_cfg.sr)
        add(out, "unknown", "noise", "noise", len(x) / feat_cfg.sr)
    for i in range(ds["silence_clips"]):
        x = np.zeros(int(1.28 * feat_cfg.sr), dtype=np.float32)
        out = os.path.join(silence_dir, f"silence_{i}.wav")
        os.makedirs(silence_dir, exist_ok=True)
        sf.write(out, x, feat_cfg.sr)
        add(out, "unknown", "silence", "silence", len(x) / feat_cfg.sr)

    man = pd.DataFrame(rows)
    man["y"] = man["label"].map(LABEL2ID)
    man.to_csv(os.path.join(data_dir, "manifest.csv"), index=False)
    print(f"\nmanifest rows: {len(man)}")
    print("label counts:", man["label"].value_counts().to_dict())
    print("split counts:", man["split"].value_counts().to_dict())
    print("kind counts:", man["kind"].value_counts().to_dict())

    # ---- feature extraction ---------------------------------------------
    print("\nextracting features ...")
    X_clean, y, split_a, spk_a, kind_a, path_a = [], [], [], [], [], []
    train_idx = []
    bad = 0
    for i, r in man.iterrows():
        p = os.path.join(data_dir, r["path"])
        try:
            x = load_wav_16k(p, feat_cfg.sr)
        except Exception:
            bad += 1
            continue
        L = frame(logmel(x, feat_cfg), feat_cfg.t_frames)
        X_clean.append(L)
        y.append(int(r["y"]))
        split_a.append(r["split"])
        spk_a.append(r["speaker"])
        kind_a.append(r["kind"])
        path_a.append(r["path"])
        if r["split"] == "train":
            train_idx.append(len(X_clean) - 1)
        if (i + 1) % 2000 == 0:
            print(f"  features {i+1}/{len(man)}")

    X_clean = np.stack(X_clean).astype(np.float32)
    print(f"features: {X_clean.shape}  (bad files skipped: {bad})")

    # ---- train-only augmentation -----------------------------------------
    print("augmenting train clips (train-only) ...")
    X_aug = []
    for gi in train_idx:
        p = os.path.join(data_dir, man.iloc[gi]["path"])
        x = load_wav_16k(p, feat_cfg.sr)
        xa = augment(x, aug_cfg, rng, feat_cfg.sr)
        X_aug.append(frame(logmel(xa, feat_cfg), feat_cfg.t_frames))
        if len(X_aug) % 1000 == 0:
            print(f"  augmented {len(X_aug)}/{len(train_idx)}")
    X_aug = np.stack(X_aug).astype(np.float32)

    np.savez_compressed(
        os.path.join(data_dir, "features.npz"),
        X_clean=X_clean, X_aug=X_aug,
        y=np.array(y, dtype=np.int64),
        split=np.array(split_a), speaker=np.array(spk_a),
        kind=np.array(kind_a), path=np.array(path_a),
        meta=np.array(json.dumps({
            "feature_cfg": feat_cfg.as_dict(),
            "aug_cfg": aug_cfg.as_dict(),
            "n": len(X_clean), "n_train_aug": len(X_aug), "bad": bad,
            "speakers": voices,
        }), dtype=object),
    )
    print(f"\nsaved data/features.npz  X_clean={X_clean.shape} X_aug={X_aug.shape}")

    yarr = np.array(y); sarr = np.array(split_a); spk_arr = np.array(spk_a)
    for s in ["train", "val", "test"]:
        m = (sarr == s)
        print(f"  {s:5s}: n={int(m.sum()):5d}  pos={int((yarr[m]==1).sum()):4d}  "
              f"neg={int((yarr[m]==0).sum()):5d}  spk={np.unique(spk_arr[m]).size}")
    print("done")


if __name__ == "__main__":
    main()
