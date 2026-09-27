"""Automated tests for the wake-word pipeline.

Run with:  python -m pytest tests/ -q     (or: python -m unittest discover tests)

Covers: feature extraction, model loading, inference, threshold behaviour,
malformed audio, and expected wake-word detection on a synthetic fixture.
"""
import os
import sys
import unittest

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.audio import (  # noqa: E402
    FEATURE_CONFIG,
    frame,
    inspect_audio,
    load_wav_16k,
    logmel,
    waveform_to_features,
)
from src.model.vcm import ModelConfig, WakeWordCNN  # noqa: E402
from src.inference.infer import (  # noqa: E402
    WakeWordDetector,
    load_detector,
    save_detector,
)
from src.dataset.dataset import LABEL2ID, speaker_disjoint_split  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(HERE, "tests", "_tmp")


def _mkwav(path, x, sr=16000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, x, sr)


class TestPreprocessing(unittest.TestCase):
    def test_feature_shape(self):
        x = np.random.randn(16000).astype(np.float32) * 0.1
        F = waveform_to_features(x)
        self.assertEqual(F.shape, (FEATURE_CONFIG.n_mels, FEATURE_CONFIG.t_frames))
        self.assertTrue(np.all(F >= 0) and np.all(F <= 1.0001))

    def test_frame_pad_and_trunc(self):
        short = np.zeros((40, 10), dtype=np.float32)
        self.assertEqual(frame(short).shape[1], FEATURE_CONFIG.t_frames)
        long = np.zeros((40, 300), dtype=np.float32)
        self.assertEqual(frame(long).shape[1], FEATURE_CONFIG.t_frames)

    def test_inspect_audio_ok(self):
        p = os.path.join(TMP, "ok.wav")
        _mkwav(p, (np.random.randn(16000) * 0.1).astype(np.float32))
        info = inspect_audio(p)
        self.assertTrue(info["ok"])
        self.assertEqual(info["sample_rate"], 16000)
        self.assertFalse(info["silent"])

    def test_inspect_audio_silent(self):
        p = os.path.join(TMP, "sil.wav")
        _mkwav(p, np.zeros(16000, dtype=np.float32))
        self.assertTrue(inspect_audio(p)["silent"])

    def test_load_malformed(self):
        p = os.path.join(TMP, "bad.wav")
        with open(p, "wb") as f:
            f.write(b"this is not a wav file")
        with self.assertRaises(Exception):
            load_wav_16k(p)

    def test_inspect_missing(self):
        info = inspect_audio(os.path.join(TMP, "does_not_exist.wav"))
        self.assertFalse(info["ok"])


class TestModel(unittest.TestCase):
    def test_forward_shape_and_params(self):
        m = WakeWordCNN(ModelConfig())
        x = torch.randn(4, 1, 40, 128)
        out = m(x)
        self.assertEqual(out.shape, (4, 2))
        self.assertLess(m.count_params(), 500_000)  # must stay tiny

    def test_bundle_roundtrip(self):
        m = WakeWordCNN(ModelConfig())
        bundle = {
            "model_state": m.state_dict(),
            "model_cfg": ModelConfig().as_dict(),
            "feature_cfg": FEATURE_CONFIG.as_dict(),
            "labels": ["unknown", "tara_na"],
            "wake_id": 1,
            "threshold": 0.5,
        }
        p = os.path.join(TMP, "bundle.pkl")
        save_detector(bundle, p)
        det = load_detector(p)
        x = (np.random.randn(16000) * 0.1).astype(np.float32)
        pred = det.predict_waveform(x)
        self.assertIn(pred.label, ["unknown", "tara_na"])
        self.assertAlmostEqual(pred.prob_wake + pred.prob_unknown, 1.0, places=5)


class TestThresholdBehaviour(unittest.TestCase):
    def test_high_threshold_never_fires(self):
        m = WakeWordCNN(ModelConfig())
        bundle = {
            "model_state": m.state_dict(),
            "model_cfg": ModelConfig().as_dict(),
            "feature_cfg": FEATURE_CONFIG.as_dict(),
            "labels": ["unknown", "tara_na"],
            "wake_id": 1,
            "threshold": 0.999,
        }
        det = WakeWordDetector(bundle)
        x = (np.random.randn(16000) * 0.1).astype(np.float32)
        self.assertFalse(det.predict_waveform(x).wake)

    def test_low_threshold_always_fires(self):
        m = WakeWordCNN(ModelConfig())
        bundle = {
            "model_state": m.state_dict(),
            "model_cfg": ModelConfig().as_dict(),
            "feature_cfg": FEATURE_CONFIG.as_dict(),
            "labels": ["unknown", "tara_na"],
            "wake_id": 1,
            "threshold": 0.0,
        }
        det = WakeWordDetector(bundle)
        x = (np.random.randn(16000) * 0.1).astype(np.float32)
        self.assertTrue(det.predict_waveform(x).wake)


class TestSplit(unittest.TestCase):
    def test_speaker_disjoint(self):
        spks = [f"s{i}" for i in range(100)]
        split = speaker_disjoint_split(spks, 0.8, 0.1, seed=0)
        self.assertEqual(len(split["train"]), 80)
        self.assertEqual(len(split["val"]), 10)
        self.assertEqual(len(split["test"]), 10)
        self.assertFalse(set(split["train"]) & set(split["val"]))
        self.assertFalse(set(split["train"]) & set(split["test"]))
        self.assertFalse(set(split["val"]) & set(split["test"]))


class TestEndToEndDetection(unittest.TestCase):
    """Train a throwaway model on a tiny synthetic fixture and confirm it can
    separate two clearly-different signals.  This validates the full
    feature->model->inference path without needing the full dataset."""

    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self.det_dir = os.path.join(TMP, "e2e")
        os.makedirs(self.det_dir, exist_ok=True)

    def _features_for(self, kind):
        # "wake" = tonal burst; "unknown" = broadband noise.  Distinct spectra.
        sr = 16000
        if kind == "wake":
            t = np.arange(sr) / sr
            x = np.sin(2 * np.pi * 440 * t) * 0.3
        else:
            x = np.random.randn(sr) * 0.1
        return waveform_to_features(x.astype(np.float32))[None]

    def test_detects_fixture(self):
        n = 400
        X = np.stack([self._features_for("wake" if i < n // 2 else "unknown")[0]
                      for i in range(n)]).astype(np.float32)
        y = np.array([1] * (n // 2) + [0] * (n // 2))
        X = torch.from_numpy(X)[:, None]
        y = torch.from_numpy(y)

        model = WakeWordCNN(ModelConfig())
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = torch.nn.CrossEntropyLoss()
        for ep in range(25):
            perm = torch.randperm(n)
            for s in range(0, n, 64):
                idx = perm[s:s + 64]
                opt.zero_grad()
                loss = crit(model(X[idx]), y[idx])
                loss.backward()
                opt.step()

        bundle = {
            "model_state": model.state_dict(),
            "model_cfg": ModelConfig().as_dict(),
            "feature_cfg": FEATURE_CONFIG.as_dict(),
            "labels": ["unknown", "tara_na"],
            "wake_id": 1,
            "threshold": 0.5,
        }
        p = os.path.join(self.det_dir, "model.pkl")
        save_detector(bundle, p)
        det = load_detector(p)

        wake_x = (np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * 0.3).astype(np.float32)
        noise_x = (np.random.randn(16000) * 0.1).astype(np.float32)
        self.assertGreater(det.predict_waveform(wake_x).prob_wake,
                           det.predict_waveform(noise_x).prob_wake)


if __name__ == "__main__":
    unittest.main()
