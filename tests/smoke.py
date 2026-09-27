import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src.preprocessing.audio import FEATURE_CONFIG, waveform_to_features, inspect_audio
from src.preprocessing.augment import augment, AUGMENTATION_CONFIG
from src.model.vcm import ModelConfig, WakeWordCNN
from src.dataset.dataset import speaker_disjoint_split, LABEL2ID
from src.inference.infer import WakeWordDetector, save_detector, load_detector
from src.evaluation.evaluate import binary_metrics, threshold_sweep, choose_threshold

# features
x = (np.random.randn(16000)*0.1).astype(np.float32)
F = waveform_to_features(x)
print("feature shape", F.shape, "range", round(float(F.min()),3), round(float(F.max()),3))

# augmentation
xa = augment(x, AUGMENTATION_CONFIG, np.random.default_rng(0))
print("augmented shape", xa.shape, "peak", round(float(np.max(np.abs(xa))),3))

# model
m = WakeWordCNN(ModelConfig())
out = m(torch.randn(2,1,40,128))
print("model out", tuple(out.shape), "params", m.count_params())

# split
sp = speaker_disjoint_split([f"s{i}" for i in range(100)], 0.8, 0.1, 0)
print("split", {k:len(v) for k,v in sp.items()})

# bundle roundtrip + metrics
b = {"model_state": m.state_dict(), "model_cfg": ModelConfig().as_dict(),
     "feature_cfg": FEATURE_CONFIG.as_dict(), "labels":["unknown","tara_na"],
     "wake_id":1, "threshold":0.5}
det = WakeWordDetector(b)
p = det.predict_waveform(x)
print("pred", p.label, round(p.confidence,3), "lat_ms", round(p.latency_ms,2))
yt = np.array([1,1,0,0,1,0]*20); yp = np.array([.9,.7,.1,.2,.8,.05]*20)
mm = binary_metrics(yt, yp, 0.5)
print("metrics acc", round(mm["accuracy"],3), "f1", round(mm["f1"],3), "auc", round(mm["roc_auc"],3))
sw = threshold_sweep(yt, yp, [0.3,0.5,0.7]); print("chosen thr", choose_threshold(sw, 0.9))
print("ALL OK")
