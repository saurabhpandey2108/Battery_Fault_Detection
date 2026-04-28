"""
Plotting utilities ported from Wavelet_Analysis/visualization/plot_utils.py.

The plotting logic (dB normalization, COI shading, log-scaled freq axis)
is preserved verbatim. Only the imports have been rewritten to match the
new package layout, and the default SCALES comes from src/config.py.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .cwt_utils import compute_cwt, coi_mask
from .image_utils import preprocess_window
from ..config import SCALES


def _scalogram_for_display(v_win, fs, scales=None, detrend: str = "cubic"):
    if scales is None:
        scales = SCALES
    x = preprocess_window(v_win, detrend=detrend)
    coeffs, freqs = compute_cwt(x, fs, scales)
    n = len(v_win)

    mag = np.abs(coeffs)
    db = 20.0 * np.log10(mag + 1e-10)

    mask = coi_mask(n, scales)

    order = np.argsort(freqs)
    f_sorted = freqs[order]
    db_sorted = db[order]
    mask_sorted = mask[order]

    scalogram = np.ma.array(db_sorted, mask=mask_sorted)

    t = np.arange(n)
    dist_to_edge = np.minimum(t, n - 1 - t)
    a_coi = dist_to_edge / np.sqrt(2.0)
    w0 = 6.0
    dt = 1.0 / fs
    coi_freq = w0 / (2 * np.pi * np.where(a_coi < 1e-9, 1e-9, a_coi) * dt)
    coi_freq = np.clip(coi_freq, f_sorted.min(), f_sorted.max())

    return scalogram, f_sorted, coi_freq


def plot_frequency_scalogram(v_win, fs, freqs, save_path, title_suffix="",
                             ax=None, cmap_name='viridis',
                             detrend: str = "cubic"):
    """2D scalogram — dB magnitude, log-scaled freq axis, COI shaded out.

    Can draw onto an existing axis (for multi-panel figures) when `ax` is given;
    otherwise it creates a standalone figure and saves it to `save_path`.

    `detrend` is forwarded to `preprocess_window`. Use "mean" on the residual
    channel; "cubic" on V_meas / V_pred.
    """
    scalogram, f, coi_freq = _scalogram_for_display(v_win, fs, detrend=detrend)

    valid = scalogram.compressed()
    lo, hi = (np.percentile(valid, 1), np.percentile(valid, 99)) if valid.size else (0, 1)
    norm = np.ma.clip((scalogram - lo) / max(hi - lo, 1e-10), 0.0, 1.0)

    time = np.arange(len(v_win)) / fs

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color='lightgray', alpha=0.8)
    mesh = ax.pcolormesh(time, f, norm, cmap=cmap, shading='auto')
    fig.colorbar(mesh, ax=ax, label="Normalized dB Magnitude")

    ax.plot(time, coi_freq, color='white', linewidth=1.2, linestyle='--', alpha=0.9)
    ax.fill_between(time, f.min(), coi_freq, color='white', alpha=0.15, hatch='///',
                    edgecolor='white', linewidth=0)

    ax.set_yscale('log')
    ax.set_ylim(f.min(), f.max())
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Pseudo-frequency (Hz)", fontsize=12)
    ax.set_title(f"CWT Scalogram {title_suffix}", fontsize=14)

    if standalone:
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Saved 2D scalogram -> {save_path}")


def plot_3d_scalogram(v_win, fs, freqs, save_path, title_suffix=""):
    scalogram, f, _ = _scalogram_for_display(v_win, fs)

    valid = scalogram.compressed()
    lo, hi = (np.percentile(valid, 1), np.percentile(valid, 99)) if valid.size else (0, 1)
    norm = np.ma.clip((scalogram - lo) / max(hi - lo, 1e-10), 0.0, 1.0)
    norm_filled = np.ma.filled(norm, np.nan)

    time_axis = np.arange(len(v_win)) / fs
    T, F = np.meshgrid(time_axis, f)

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(T, F, norm_filled, cmap='viridis', edgecolor='none',
                           alpha=0.9, rstride=2, cstride=2)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Normalized dB Magnitude")

    ax.set_xlabel("Time (s)", fontsize=11, labelpad=10)
    ax.set_ylabel("Pseudo-frequency (Hz)", fontsize=11, labelpad=10)
    ax.set_zlabel("Normalized Magnitude", fontsize=11, labelpad=10)
    ax.set_title(f"3D CWT Scalogram {title_suffix}", fontsize=14, pad=20)
    ax.view_init(elev=30, azim=225)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved 3D scalogram -> {save_path}")
