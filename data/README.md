# Dataset

The wake-word dataset is **synthesized** with [`espeak`](https://espeak.sourceforge.net/)
so the project is fully reproducible without shipping large raw audio.  Raw
WAVs and the feature cache are generated locally and are **not committed**
(see `.gitignore`).

## What is in the dataset

| Folder        | Label     | Contents                                                        |
|---------------|-----------|-----------------------------------------------------------------|
| `positive/`   | `tara_na` | The exact phrase **"tara na"** spoken by many voices/speeds     |
| `negative/`   | `unknown` | Real VCM command phrases + phonetic distractors ("dara na", ...) |
| `noise/`      | `unknown` | Pure background noise                                           |
| `silence/`    | `unknown` | Pure silence                                                    |

Variation is introduced by varying the espeak **voice** (acts as the
"speaker"), **speaking speed** (110–190 wpm) and **pitch** (±12), plus
waveform augmentation applied to the training split only (speed/pitch/volume/
noise/word-breaks).

## Layout

```
data/
  positive/   *.wav
  negative/   *.wav
  noise/      *.wav
  silence/    *.wav
  manifest.csv        # path,label,split,speaker,kind,duration_sec,y
  features.npz        # X_clean, X_aug, y, split, speaker, kind, path, meta
```

## Split (leakage-free)

The espeak voice is treated as the speaker.  Voices are partitioned into
**disjoint** groups so no voice appears in more than one split:

```
train : 80 voices
val   : 10 voices
test  : 10 voices
```

Noise and silence clips are assigned to `train`.  This mirrors the OptionB
VCM dataset (80/10/10) and guarantees the model is scored on voices it never
heard during training.

## Regenerating

```bash
python scripts/prepare_dataset.py
```

This (re)creates all WAVs, the manifest and the feature cache.  To change the
scale, edit `configs/default.json` (`dataset.*`).
