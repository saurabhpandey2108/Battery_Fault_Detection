"""
Render a single side-by-side PNG comparing one BD file and one CS file
at the same resistance index.

Layout (2 rows × 4 cols):
    Row 1 (BD): V_meas/V_pred overlay | V_meas scalogram | V_pred scalogram | residual scalogram
    Row 2 (CS): same

To make brightness directly comparable between BD and CS, each channel's
percentile range (vmin, vmax) is computed on the COMBINED dB distribution
from both files — so a residual that is pure noise in BD and structured
in CS renders at visibly different intensities on the same colormap.
The per-window cubic-detrend + z-score preprocessing inside
`raw_log_scalogram` and the COI mask are untouched (hard constraints).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import SCALES, WINDOW_SIZE, MODELS_DIR, RESULTS_DIR
from .data.tsinghua_loader import load_tsinghua_csv, parse_filename
from .infer_pinn import load_pinn, predict_voltage_series
from .wavelet.image_utils import raw_log_scalogram


def _select_window(pred: dict, window_size: int, start: int) -> dict:
    n = len(pred["v_meas"])
    if n < window_size:
        raise ValueError(f"Series has {n} samples, needs >= {window_size}")
    start = max(0, min(start, n - window_size))
    sl = slice(start, start + window_size)
    return {
        "t": pred["time"][sl] - pred["time"][sl][0],
        "v_meas": pred["v_meas"][sl],
        "v_pred": pred["v_pred"][sl],
        "current": pred["current"][sl],
        "residual": pred["residual"][sl],
    }


def _scalogram(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the same Wavelet_Analysis path as training (cubic-detrend + z-score,
    Morlet CWT, COI mask to NaN). Returns (dB scalogram, freqs, coi-boundary freq)."""
    sc, freqs = raw_log_scalogram(signal, fs, SCALES)
    order = np.argsort(freqs)
    sc_sorted = sc[order]
    f_sorted = freqs[order]

    # COI boundary (highest reliable freq at each time) for shading on the plot
    n = len(signal)
    t = np.arange(n)
    dist = np.minimum(t, n - 1 - t)
    a_coi = dist / np.sqrt(2.0)
    w0 = 6.0
    dt = 1.0 / fs
    coi_freq = w0 / (2 * np.pi * np.where(a_coi < 1e-9, 1e-9, a_coi) * dt)
    coi_freq = np.clip(coi_freq, f_sorted.min(), f_sorted.max())
    return sc_sorted, f_sorted, coi_freq


def _shared_range(arr_bd: np.ndarray, arr_cs: np.ndarray,
                  pct_lo: float = 1.0, pct_hi: float = 99.0) -> tuple[float, float]:
    valid = np.concatenate([
        arr_bd[~np.isnan(arr_bd)].ravel(),
        arr_cs[~np.isnan(arr_cs)].ravel(),
    ])
    if valid.size == 0:
        return 0.0, 1.0
    return float(np.percentile(valid, pct_lo)), float(np.percentile(valid, pct_hi))


def _draw_scalogram(ax, sc, f, coi_freq, t_axis, vmin, vmax, title):
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="lightgray", alpha=0.8)
    masked = np.ma.array(sc, mask=np.isnan(sc))
    mesh = ax.pcolormesh(t_axis, f, masked, cmap=cmap,
                         vmin=vmin, vmax=vmax, shading="auto")
    ax.plot(t_axis, coi_freq, color="white", linewidth=1.0,
            linestyle="--", alpha=0.9)
    ax.fill_between(t_axis, f.min(), coi_freq, color="white",
                    alpha=0.15, hatch="///", edgecolor="white", linewidth=0)
    ax.set_yscale("log")
    ax.set_ylim(f.min(), f.max())
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pseudo-frequency (Hz)")
    ax.set_title(title, fontsize=10)
    return mesh


def compare_bd_cs(bd_file: str,
                  cs_file: str,
                  pinn_weights: str,
                  out_path: str,
                  window_size: int = WINDOW_SIZE,
                  start_sample: int = 0,
                  sampling_time: Optional[float] = None,
                  capacity_as: Optional[float] = None) -> str:
    """Produce a 2×4 figure comparing BD vs CS at identical window indices."""
    nn = load_pinn(pinn_weights)

    bd_pred = predict_voltage_series(nn, load_tsinghua_csv(bd_file)["discharge_df"],
                                     sampling_time=sampling_time, capacity_as=capacity_as)
    cs_pred = predict_voltage_series(nn, load_tsinghua_csv(cs_file)["discharge_df"],
                                     sampling_time=sampling_time, capacity_as=capacity_as)

    bd_w = _select_window(bd_pred, window_size, start_sample)
    cs_w = _select_window(cs_pred, window_size, start_sample)
    fs = 1.0

    # Scalograms on each channel for each file
    bd_sc_m, f, coi = _scalogram(bd_w["v_meas"], fs)
    bd_sc_p, _, _ = _scalogram(bd_w["v_pred"], fs)
    bd_sc_r, _, _ = _scalogram(bd_w["residual"], fs)

    cs_sc_m, _, _ = _scalogram(cs_w["v_meas"], fs)
    cs_sc_p, _, _ = _scalogram(cs_w["v_pred"], fs)
    cs_sc_r, _, _ = _scalogram(cs_w["residual"], fs)

    # Shared per-channel dB ranges so BD and CS are directly comparable
    vmin_m, vmax_m = _shared_range(bd_sc_m, cs_sc_m)
    vmin_p, vmax_p = _shared_range(bd_sc_p, cs_sc_p)
    vmin_r, vmax_r = _shared_range(bd_sc_r, cs_sc_r)

    bd_meta = parse_filename(bd_file)
    cs_meta = parse_filename(cs_file)

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    for row, (label, w, sc_m, sc_p, sc_r, meta) in enumerate([
        ("BD (healthy)", bd_w, bd_sc_m, bd_sc_p, bd_sc_r, bd_meta),
        (f"CS ({cs_meta['ohm']} ohm ISC)", cs_w, cs_sc_m, cs_sc_p, cs_sc_r, cs_meta),
    ]):
        # Column 0: time-domain signals
        ax0 = axes[row, 0]
        ax0.plot(w["t"], w["v_meas"], color="#d95f02", linewidth=1.4, label="V_meas")
        ax0.plot(w["t"], w["v_pred"], color="#1b9e77", linewidth=1.0,
                 linestyle="--", label="V_pred")
        ax0.set_xlabel("Time (s)")
        ax0.set_ylabel("Voltage (V)")
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc="upper right", fontsize=8)
        ax0_r = ax0.twinx()
        ax0_r.plot(w["t"], w["residual"], color="#e7298a",
                   linewidth=0.8, alpha=0.8, label="e(t)")
        ax0_r.set_ylabel("Residual (V)")
        rms = float(np.sqrt(np.mean(w["residual"]**2)))
        ax0.set_title(f"{label}  —  RMS residual = {rms:.4f} V", fontsize=10)

        # Columns 1..3: V_meas / V_pred / residual scalograms
        m1 = _draw_scalogram(axes[row, 1], sc_m, f, coi, w["t"],
                             vmin_m, vmax_m, f"{label} — V_meas scalogram")
        m2 = _draw_scalogram(axes[row, 2], sc_p, f, coi, w["t"],
                             vmin_p, vmax_p, f"{label} — V_pred scalogram")
        m3 = _draw_scalogram(axes[row, 3], sc_r, f, coi, w["t"],
                             vmin_r, vmax_r, f"{label} — residual scalogram")

        if row == 0:
            fig.colorbar(m1, ax=axes[:, 1].tolist(), label="dB magnitude", fraction=0.03, pad=0.02)
            fig.colorbar(m2, ax=axes[:, 2].tolist(), label="dB magnitude", fraction=0.03, pad=0.02)
            fig.colorbar(m3, ax=axes[:, 3].tolist(), label="dB magnitude", fraction=0.03, pad=0.02)

    fig.suptitle(
        f"Three-Scalogram BD vs CS comparison  —  "
        f"{bd_meta['charge_rate']}C {bd_meta['discharge_mode']}, {cs_meta['ohm']} ohm ISC",
        fontsize=13, y=0.995,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison figure -> {out_path}")
    return out_path
