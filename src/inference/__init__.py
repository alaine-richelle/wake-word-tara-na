"""Inference package."""
from .infer import (
    WakeWordDetector,
    Prediction,
    load_detector,
    save_detector,
    record_microphone,
    microphone_available,
)

__all__ = [
    "WakeWordDetector",
    "Prediction",
    "load_detector",
    "save_detector",
    "record_microphone",
    "microphone_available",
]
