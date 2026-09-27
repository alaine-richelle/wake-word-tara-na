"""Live microphone wake-word test.

    python scripts/microphone_test.py

Continuously listens through the microphone in short overlapping windows and
prints a banner whenever "tara na" is detected.  Press Ctrl-C to stop.

Microphone requirements
-----------------------
* A default input device (microphone) must be present.
* `sounddevice` + a PortAudio backend.  On Linux: `sudo apt install portaudio19-dev`.
* The OS may prompt for microphone permission the first time.

If no microphone is available (e.g. a headless server), the script prints a
clear message and exits instead of crashing.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference.infer import (  # noqa: E402
    load_detector,
    microphone_available,
    record_microphone,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "model.pkl"))
    ap.add_argument("--window", type=float, default=1.6, help="seconds per window")
    ap.add_argument("--overlap", type=float, default=0.4,
                    help="seconds of overlap between windows (debounce)")
    ap.add_argument("--cooldown", type=float, default=3.0,
                    help="seconds to ignore after a detection")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"model not found: {args.model} (run scripts/train.py)")
    if not microphone_available():
        print("No microphone available (headless environment?).")
        print("On a machine with audio, install PortAudio + sounddevice, then re-run.")
        print("For a headless test, use:  python scripts/test_wake_word.py --audio file.wav")
        return

    det = load_detector(args.model)
    sr = det.feature_cfg.sr
    step = max(0.1, args.window - args.overlap)
    print(f"Listening for '{det.bundle.get('labels', ['?'])[1]}' "
          f"(threshold={det.threshold:.2f}).  Ctrl-C to stop.")
    last_fire = -1e9
    try:
        while True:
            t0 = time.time()
            x = record_microphone(duration=args.window, sr=sr)
            if x is None:
                print("recording failed; retrying...")
                time.sleep(0.2)
                continue
            pred = det.predict_waveform(x)
            elapsed = time.time() - t0
            if pred.wake and (time.time() - last_fire) > args.cooldown:
                last_fire = time.time()
                print(f"\n*** WAKE WORD DETECTED ***  confidence={pred.confidence:.3f} "
                      f"(P={pred.prob_wake:.3f})  [{elapsed*1000:.0f} ms/window]")
            else:
                sys.stdout.write(f"\r  listening... P(tara_na)={pred.prob_wake:.3f}   ")
                sys.stdout.flush()
            # keep cadence steady
            remaining = step - elapsed
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
