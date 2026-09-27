# Wake-Word Detector: "tara na"

A **tiny, on-device-friendly wake-word detector** that listens for the exact
phrase **"tara na"** and only lets audio through to the companion
[Tiny Voice Command Model (VCM)](../ME2) when it hears it.

```
Microphone ─▶ Audio Buffer ─▶ Preprocessing ─▶ Feature Extraction ─▶ Wake-Word Model
                                                                 │
                                                          Confidence vs Threshold
                                                                 │
                                              ┌──────────────────┴──────────────────┐
                                              ▼                                     ▼
                                        "tara na"?                               NO
                                          YES                          ─▶ Continue Listening
                                              │
                                              ▼
                                      Tiny VCM (ME2)
```

This is a **gatekeeper**: it runs continuously and cheaply, and the heavier
command-recognition model only wakes up when the gate opens.

---

## 1. What is wake-word detection?

Wake-word detection (WWD) is a small binary classification problem: given a
short window of audio, decide **did the user say "tara na"?** or **not?**  It
is *not* transcription — the model never needs to spell out what was said,
only to recognise one specific phrase.  That distinction is what makes a tiny
model sufficient where a full speech-recognition system would be enormous.

## 2. Why is it separated from the Tiny VCM?

| Concern | Wake-word detector | Tiny VCM |
|---|---|---|
| Runs | Continuously, always on | Only after wake word |
| Task | Binary: "tara na" present? | Classify one of ~19 commands |
| Must be | Extremely small, low-power, low-latency | Accurate, can be larger |
| Failure mode to minimise | **False positives** (unnecessary wake) | Wrong command |

Separating them means the expensive VCM stays asleep 99% of the time, which is
exactly the battery/CPU budget a real smart device has.  It also lets us tune
the two models independently.

## 3. How audio flows through the system

1. **Capture** – the microphone delivers a continuous stream.
2. **Buffer** – we slice it into ~1.28 s overlapping windows.
3. **Preprocess** – resample to 16 kHz, downmix to mono.
4. **Features** – compute a 40×128 log-mel spectrogram (see §6).
5. **Model** – a tiny CNN outputs `P(tara_na)`.
6. **Decision** – if `P(tara_na) ≥ threshold`, fire the VCM; otherwise keep
   listening.

## 4. Dataset

See [`data/README.md`](data/README.md).  In short: the phrase "tara na" is
synthesized with `espeak` across many voices/speeds/pitches (positives), and a
large set of real command phrases + phonetic distractors + noise + silence form
the negatives.  The espeak voice acts as the "speaker" and is partitioned into
**disjoint** train/val/test groups (80/10/10 voices) so no voice leaks across
splits.  Raw audio is generated locally and **not committed**.

## 5. Audio preprocessing

`src/preprocessing/audio.py`:

* **Resample** any input to **16 kHz** mono.
* **Silence handling** – short clips are right-padded, long clips truncated, to
  a fixed 1.28 s window (128 frames).
* **Inspection** – `inspect_audio()` reports sample rate, channels, duration,
  RMS/peak energy and flags silent or corrupt files (used in dataset QA).

## 6. Feature extraction: log-mel spectrogram

A raw 16 kHz waveform is ~16,000 samples/second — too big and too raw for a
tiny net.  We compress it to a **log-mel spectrogram**:

```
waveform ─▶ STFT (512-pt, 160-hop) ─▶ power ─▶ 40 mel filters (20 Hz–8 kHz)
         ─▶ log ─▶ rescale to [0,1]  ─▶  [40, 128] image
```

This is the same representation the VCM uses, so both models share one feature
pipeline.  40 mels × 128 frames is a deliberate sweet spot: rich enough to
separate "tara na" from other speech, small enough for a ~20 k-parameter model.

**Why not a pretrained audio model (wav2vec2 / Whisper)?**  Those are millions
of parameters and defeat "tiny/on-device".  For a single fixed phrase, a small
hand-built CNN on log-mel features is the right trade-off: far fewer
parameters, millisecond inference, and no giant dependency.  (A pretrained
embedding *could* be explored as a future improvement — see §15.)

## 7. Model architecture

`src/model/vcm.py` — a **depthwise-separable CNN**:

```
[1, 40, 128]
  └─ (DepthwiseSepConv → BatchNorm → ReLU → MaxPool) × 3   # widths 32→64→128
  └─ Global Average Pooling
  └─ Dropout → Linear(64) → ReLU → Dropout → Linear(2)
```

* **Depthwise-separable convs** factor each layer into a per-channel spatial
  filter + a 1×1 channel mixer, cutting parameters/FLOPs ~8–9× vs a standard
  conv — ideal for edge.
* **Global average pooling** removes a large flatten+FC head and adds tolerance
  to small timing shifts of the phrase.
* Output: two logits → `[unknown, tara_na]`.

Total: **~22,000 parameters** (≈ 0.09 MB float32).  Sub-millisecond inference
per window on CPU after warm-up.

## 8. Training

`scripts/train.py` (config in `configs/default.json`):

* Reproducible seed (Python/NumPy/Torch).
* Adam optimizer, LR 1e-3, weight decay, ReduceLROnPlateau scheduler.
* **Class weighting** (positives are the minority class).
* Best-model tracking by **validation F1** + **early stopping**.
* Produces exactly **one** artifact: `model.pkl`.

```bash
python scripts/prepare_dataset.py   # build data + features (once)
python scripts/train.py             # ~minutes on CPU
```

## 9. Evaluation metrics

`scripts/evaluate.py` computes, on both val and test:

* **Accuracy** – overall correctness.
* **Precision** – of the times we *said* "tara na", how often were we right.
* **Recall** – of the times "tara na" *was* said, how often did we catch it.
* **F1** – harmonic mean of precision/recall.
* **ROC-AUC** – threshold-independent separability (one-vs-rest on the wake
  class), computed from **probabilities**, not hard labels.
* **Confusion matrix**, **false-positive rate (FPR)**, **false-negative rate (FNR)**.

For a wake-word detector **FPR is the metric that matters most**: a false
positive wakes the VCM for nothing.  That is why we don't fix the threshold at
0.5.

## 10. Error analysis

`results/error_analysis.txt` + `results/predictions_*.csv` list, per sample:
input, gold, prediction, confidence, and error type.  The report highlights the
worst speakers, the worst clip *kinds* (e.g. noise vs speech), and concrete
false-positive / false-negative examples so you can see *what* the model
struggles with.

## 11. Threshold selection

`scripts/evaluate.py` sweeps thresholds (0.3 … 0.9) and, subject to a recall
floor (default ≥ 0.90), picks the **highest** threshold that still meets it —
i.e. the most selective operating point that doesn't miss too many real wake
words.  The chosen threshold is written back into `model.pkl` and is
configurable via `configs/default.json` and `--threshold` at test time.

## 12. File-based inference

```bash
python scripts/test_wake_word.py --audio path/to/audio.wav
```

```
Input: sample.wav
  Prediction: tara_na
  Confidence: 0.94  (P(tara_na)=0.94)
  Latency   : 0.8 ms
  Wake word detected: YES
```

## 13. Microphone inference

```bash
python scripts/microphone_test.py
```

Continuously listens in overlapping windows and prints a banner on detection.
**Ctrl-C stops it.**  Requires a microphone + `sounddevice`/PortAudio.  On a
headless machine it prints a friendly message and suggests the file-based test
instead.

## 14. Reproducing the experiments

```bash
pip install -r requirements.txt

python scripts/prepare_dataset.py    # 1. synthesize data + features
python scripts/train.py              # 2. train -> model.pkl
python scripts/evaluate.py           # 3. metrics + threshold + error analysis
python scripts/test_wake_word.py --audio data/positive/<some>.wav
python scripts/microphone_test.py    # 4. live (needs a mic)

python -m pytest tests/ -q           # automated tests
```

All randomness is seeded; `configs/default.json` holds every tunable
hyperparameter.

## 15. Limitations

* **Synthetic speech.** `espeak` voices are robotic and acoustically narrow;
  real human voices, accents, rooms and microphones will be harder.  Metrics
  here are therefore an *upper bound* on real-world performance.
* **Single fixed phrase, single language accent.** The model only knows
  "tara na" as pronounced by the espeak English voices.
* **Fixed 1.28 s window.** Very long pauses or very long utterances are
  truncated/padded.
* **CPU training** in this environment (no CUDA), so training is slower than
  on a GPU box — results are unaffected.

## 16. Possible improvements

* Collect **real** "tara na" recordings (the `user_recordings/` flow in the VCM
  repo is a template) and fine-tune.
* Add **distance/room-impulse** simulation (reverb, far-field).
* Try a small **pretrained audio encoder** (e.g. a compact wav2vec2) frozen,
  with a tiny linear head — better robustness at higher cost.
* Streaming **online** decoding with a sliding window + hysteresis to reduce
  false triggers from partial phrases.
* Export to **ONNX / TFLite** for true on-device deployment.

## 17. Integration with the Tiny VCM

The detector and the VCM share the **same feature pipeline** (16 kHz log-mel).
In a real device:

```
mic ─▶ wake-word model (always on)
          │ P(tara_na) ≥ threshold ?
          │  YES
          ▼
     buffer the following ~3 s ─▶ Tiny VCM (ME2) ─▶ command
```

Only when the gate opens does the VCM run, so the heavy model stays asleep.
Both models are tiny and can run side-by-side on the same CPU/edge device.
