"""
Build (224, 224, 3) scalogram stacks from V_meas, V_pred, and the residual
e(t) = V_meas - V_pred.

Channel order is [meas, pred, residual]. Each channel runs through the
same CWT + COI-mask + percentile-normalize + 224x224 resize path, but
with PER-CHANNEL detrend choice:

    V_meas, V_pred -> cubic detrend + z-score
    residual       -> mean-subtract + z-score (no polynomial detrend)

Cubic detrend on the residual would absorb the slow ISC drift (the actual
fault diagnostic). See `build_three_scalogram_image` for the rationale,
and `preprocess_window` in src/wavelet/image_utils.py for the modes.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.wavelet.image_utils import raw_log_scalogram, normalize_and_resize
from src.wavelet.preprocess import create_windows


def _channel_image(signal: np.ndarray,
                   fs: float,
                   scales: np.ndarray,
                   img_size: Tuple[int, int] = (224, 224),
                   pct_lo: float = 1.0,
                   pct_hi: float = 99.0,
                   detrend: str = "cubic") -> np.ndarray:
    """Scalogram → 224×224 float32 image for a single 1-D window."""
    sc, _ = raw_log_scalogram(signal, fs, scales, detrend=detrend)
    valid = sc[~np.isnan(sc)]
    if valid.size:
        vmin = float(np.percentile(valid, pct_lo))
        vmax = float(np.percentile(valid, pct_hi))
    else:
        vmin, vmax = 0.0, 1.0
    return normalize_and_resize(sc, vmin, vmax, img_size=img_size)


def build_three_scalogram_image(v_meas_win: np.ndarray,
                                v_pred_win: np.ndarray,
                                fs: float,
                                scales: np.ndarray,
                                img_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Stack the three scalogram channels into HxWx3 float32.

    Channel order: [V_meas, V_pred, residual = V_meas - V_pred].

    Per-channel preprocessing
    -------------------------
    V_meas, V_pred : cubic detrend + z-score. The ~1 V NCM811 OCV envelope
                     across the window would otherwise saturate the lowest
                     scales of the scalogram and crowd out the fast features.
    residual       : MEAN-subtract + z-score (no polynomial detrend). The
                     ISC fault signature is precisely the slow drift of the
                     residual versus SOC -- a cubic fit absorbs that drift
                     and deletes the diagnostic, which is what was killing
                     BD-vs-CS classification (and the SOC-from-V_pred
                     sanity check, by the same mechanism). See
                     `preprocess_window` in src/wavelet/image_utils.py.
    """
    v_meas_win = np.asarray(v_meas_win, dtype=np.float64)
    v_pred_win = np.asarray(v_pred_win, dtype=np.float64)
    if v_meas_win.shape != v_pred_win.shape:
        raise ValueError("v_meas_win and v_pred_win must have the same shape")

    e_win = v_meas_win - v_pred_win

    img_m = _channel_image(v_meas_win, fs, scales, img_size=img_size,
                           detrend="cubic")
    img_p = _channel_image(v_pred_win, fs, scales, img_size=img_size,
                           detrend="cubic")
    img_e = _channel_image(e_win,      fs, scales, img_size=img_size,
                           detrend="mean")

    stack = np.stack([img_m, img_p, img_e], axis=-1).astype(np.float32)
    return stack


def build_all_windows(v_meas: np.ndarray,
                      v_pred: np.ndarray,
                      fs: float,
                      scales: np.ndarray,
                      window_size: int,
                      stride: int,
                      label: int | float = 0,
                      img_size: Tuple[int, int] = (224, 224)
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Window a paired (V_meas, V_pred) series and produce the full image tensor.

    Both series are windowed with identical indices. Each window yields one
    (H, W, 3) stack. All windows get the same `label` since an ISC fault is a
    file-level property in Phase 1.

    Returns
    -------
    images : (N_windows, H, W, 3) float32
    labels : (N_windows,) ndarray filled with `label`
    """
    v_meas = np.asarray(v_meas, dtype=np.float64)
    v_pred = np.asarray(v_pred, dtype=np.float64)
    if v_meas.shape != v_pred.shape:
        raise ValueError("v_meas and v_pred must have the same length")

    # Window V_meas labeled by `label`; window V_pred with a dummy label array
    # of the same length, aligned by the same indices.
    dummy_labels = np.full(len(v_meas), label, dtype=np.float32)
    meas_windows, _ = create_windows(v_meas, dummy_labels, window_size, stride)
    pred_windows, labels = create_windows(v_pred, dummy_labels, window_size, stride)

    if len(meas_windows) != len(pred_windows):
        raise RuntimeError("Window counts disagree between meas and pred")

    if len(meas_windows) == 0:
        return (np.zeros((0,) + tuple(img_size) + (3,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32))

    images = np.empty((len(meas_windows),) + tuple(img_size) + (3,), dtype=np.float32)
    for k in range(len(meas_windows)):
        images[k] = build_three_scalogram_image(
            meas_windows[k], pred_windows[k], fs, scales, img_size=img_size
        )
    return images, labels.astype(np.float32)
