"""
Lightweight preprocessing helpers from Wavelet_Analysis/data_preprocessing/preprocess.py.

The Arbin/NASA-specific load_signals_* helpers are intentionally omitted —
Tsinghua loading lives in src/data/tsinghua_loader.py.
"""

import numpy as np


def estimate_fs(time_array):
    """Estimate sampling frequency (Hz) from timestamps."""
    return 1.0 / np.median(np.diff(time_array))


def create_windows(signal, label_array, window_size, stride):
    """Sliding windows. Label = label_array at the end of each window."""
    windows = []
    labels = []
    for i in range(0, len(signal) - window_size + 1, stride):
        windows.append(signal[i:i + window_size])
        labels.append(label_array[i + window_size - 1])
    return np.array(windows), np.array(labels)


def normalize_array(x):
    """Min-max normalize an array to [0, 1]."""
    x = np.asarray(x, dtype=float)
    if np.ptp(x) == 0:
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())
