"""File-based wake-word test.

    python scripts/test_wake_word.py --audio path/to/audio.wav
    python scripts/test_wake_word.py --audio a.wav b.wav c.wav

Loads model.pkl, applies the training-time feature pipeline, and prints the
prediction + confidence + whether the wake word was detected.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference.infer import load_detector  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", nargs="+", required=True, help="one or more audio files")
    ap.add_argument("--model", default=os.path.join(HERE, "model.pkl"))
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the model's stored threshold")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"model not found: {args.model} (run scripts/train.py)")
    det = load_detector(args.model)
    thr = args.threshold
    print(f"Loaded {args.model}  (stored threshold={det.threshold:.3f})\n")

    for path in args.audio:
        if not os.path.exists(path):
            print(f"Input: {path}\n  ERROR: file not found\n")
            continue
        pred = det.predict_file(path, threshold=thr)
        print(f"Input: {os.path.basename(path)}")
        print(f"  Prediction: {pred.label}")
        print(f"  Confidence: {pred.confidence:.4f}  (P(tara_na)={pred.prob_wake:.4f})")
        print(f"  Latency   : {pred.latency_ms:.2f} ms")
        status = "YES" if pred.wake else "NO"
        extra = "  [UNCERTAIN - close to threshold]" if pred.uncertain else ""
        print(f"  Wake word detected: {status}{extra}\n")


if __name__ == "__main__":
    main()
