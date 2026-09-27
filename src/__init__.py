"""Wake-word detection package for the phrase "tara na".

Modules
-------
preprocessing  : audio loading, log-mel features, augmentation
dataset        : PyTorch Dataset + speaker-disjoint split helpers
model          : tiny depthwise-separable CNN wake-word classifier
inference      : file / microphone inference against a saved model.pkl
evaluation     : metrics, threshold sweep, error analysis, confusion matrix
"""
