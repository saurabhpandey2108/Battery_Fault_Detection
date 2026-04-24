"""
CWT scalogram → image utilities. Pure NumPy + cv2 for resize.
(Ported from Wavelet_Analysis/cwt_image/image_utils.py — the only change
is the relative import path for compute_cwt/coi_mask.)

Per-window preprocessing = cubic polynomial detrend + z-score.
Magnitude is expressed in dB; COI pixels are set to NaN and handled in
percentile-based normalization stats.
"""

import numpy as np
import cv2

from .cwt_utils import compute_cwt, coi_mask


_EPS_DB = 1e-10


def preprocess_window(signal):
    """Cubic polynomial detrend + z-score for a 1-D window."""
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    if n < 4:
        return x - x.mean()
    t = np.linspace(0.0, 1.0, n)
    coeffs = np.polyfit(t, x, deg=3)
    trend = np.polyval(coeffs, t)
    x = x - trend
    sd = x.std()
    if sd > 1e-9:
        x = x / sd
    return x


def raw_log_scalogram(signal, fs, scales):
    """dB-magnitude scalogram of a preprocessed window with COI masked to NaN."""
    x = preprocess_window(signal)
    coeffs, freqs = compute_cwt(x, fs, scales)
    mag = np.abs(coeffs)
    db = 20.0 * np.log10(mag + _EPS_DB)

    mask = coi_mask(len(signal), scales)
    db[mask] = np.nan

    return db.astype(np.float32), freqs


def normalize_and_resize(scalogram, vmin, vmax, img_size=(224, 224)):
    """Percentile-range normalize to [0, 1], replace NaN (COI) with 0, resize."""
    span = max(vmax - vmin, 1e-10)
    img = (scalogram - vmin) / span
    img = np.clip(img, 0.0, 1.0)
    img = np.where(np.isnan(img), 0.0, img)
    img = img.astype(np.float32)
    return cv2.resize(img, img_size)


def cwt_to_image(signal, fs, img_size=(224, 224), scales=None,
                 vmin=None, vmax=None):
    if scales is None:
        scales = np.geomspace(2.0, 300.0, 96)
    scalogram, freqs = raw_log_scalogram(signal, fs, scales)

    valid = scalogram[~np.isnan(scalogram)]
    if vmin is None:
        vmin = float(np.percentile(valid, 1)) if valid.size else 0.0
    if vmax is None:
        vmax = float(np.percentile(valid, 99)) if valid.size else 1.0

    img = normalize_and_resize(scalogram, vmin, vmax, img_size=img_size)
    return img, freqs


def stack_channels(*images):
    """Stack 2-D images into HxWxC along the channel axis."""
    flat = [img if img.ndim == 2 else img[..., 0] for img in images]
    return np.stack(flat, axis=-1)
