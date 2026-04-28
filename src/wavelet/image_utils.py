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


def preprocess_window(signal, detrend: str = "cubic"):
    """Detrend + z-score for a 1-D window.

    Parameters
    ----------
    detrend : {"cubic", "mean", "none"}
        - "cubic": fit and subtract a degree-3 polynomial (the original
          Wavelet_Analysis behaviour). Right for V_meas / V_pred where the
          ~1 V OCV envelope across the window would otherwise saturate the
          scalogram at low frequencies.
        - "mean":  subtract only the window mean. The slow drift survives.
          This is what the residual channel needs in this project, because
          the ISC fault signature lives in the slow drift of the residual
          versus SOC (verified by `main.py verify` -- CS_10 drift slope is
          +346 mV per unit-SOC vs -6 on BD, and a cubic fit absorbs that
          almost entirely, deleting the diagnostic).
        - "none":  no centering. Mostly for debugging; CWT has zero DC
          response so this only differs from "mean" in the std used for
          z-score, which can blow up if the window has a huge DC offset.

    Always followed by a divide-by-std (when std > 1e-9) so the dB scale
    is comparable across channels and windows.
    """
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    if n < 4:
        return x - x.mean()

    if detrend == "cubic":
        t = np.linspace(0.0, 1.0, n)
        coeffs = np.polyfit(t, x, deg=3)
        trend = np.polyval(coeffs, t)
        x = x - trend
    elif detrend == "mean":
        x = x - x.mean()
    elif detrend == "none":
        pass
    else:
        raise ValueError(f"Unknown detrend mode: {detrend!r}")

    sd = x.std()
    if sd > 1e-9:
        x = x / sd
    return x


def raw_log_scalogram(signal, fs, scales, detrend: str = "cubic"):
    """dB-magnitude scalogram of a preprocessed window with COI masked to NaN.

    `detrend` is forwarded to `preprocess_window`; see that docstring for
    when each option is appropriate. The default keeps the original
    cubic-detrend behaviour for backward compatibility with existing
    visualisations; the three-scalogram builder picks per-channel.
    """
    x = preprocess_window(signal, detrend=detrend)
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
                 vmin=None, vmax=None, detrend: str = "cubic"):
    if scales is None:
        scales = np.geomspace(2.0, 300.0, 96)
    scalogram, freqs = raw_log_scalogram(signal, fs, scales, detrend=detrend)

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
