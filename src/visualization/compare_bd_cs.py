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

Per-channel preprocessing (matches `build_three_scalogram_image`):
  V_meas, V_pred  → cubic detrend + z-score (the ~1 V OCV envelope must
                    be removed or it saturates the lowest scales).
  residual        → mean-subtract + z-score (the slow drift IS the ISC
                    diagnostic; cubic detrend would absorb it).
The COI mask is untouched (hard constraint).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import SCALES, WINDOW_SIZE, MODELS_DIR, RESULTS_DIR
from src.data.tsinghua_loader import load_tsinghua_csv, parse_filename
from src.inference.infer_pinn import load_pinn, predict_voltage_series
from src.wavelet.image_utils import raw_log_scalogram
from src.wavelet.scalogram_metrics import (
    pairwise_scalogram_metrics,
    append_metrics_row,
)


SIMILARITY_CSV = os.path.join(RESULTS_DIR, "three_scalograms", "similarity.csv")


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


def _scalogram(signal: np.ndarray, fs: float,
               detrend: str = "cubic") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the same Wavelet_Analysis path as training, with per-channel
    detrend choice. Returns (dB scalogram, freqs, coi-boundary freq).

    Use detrend='cubic' for V_meas / V_pred and detrend='mean' for the
    residual (the ISC fault drift is what we want to keep in the scalogram).
    """
    sc, freqs = raw_log_scalogram(signal, fs, SCALES, detrend=detrend)
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

    # Scalograms on each channel for each file. Per-channel detrend choice
    # matches train_classifier's three_scalogram_builder so figures show
    # what the CNN actually sees.
    bd_sc_m, f, coi = _scalogram(bd_w["v_meas"],   fs, detrend="cubic")
    bd_sc_p, _, _   = _scalogram(bd_w["v_pred"],   fs, detrend="cubic")
    bd_sc_r, _, _   = _scalogram(bd_w["residual"], fs, detrend="mean")

    cs_sc_m, _, _ = _scalogram(cs_w["v_meas"],   fs, detrend="cubic")
    cs_sc_p, _, _ = _scalogram(cs_w["v_pred"],   fs, detrend="cubic")
    cs_sc_r, _, _ = _scalogram(cs_w["residual"], fs, detrend="mean")

    # Pairwise similarity between the three channels of the scalogram stack.
    # Diagnostic expectations:
    #   meas-vs-pred  : HIGH on BD (PINN tracks V_meas), LOWER on CS.
    #   meas-vs-resid : LOW  on BD (residual is noise), HIGHER on CS
    #                   (residual inherits V_meas structure as V_pred drifts).
    bd_metrics = pairwise_scalogram_metrics(bd_sc_m, bd_sc_p, bd_sc_r)
    cs_metrics = pairwise_scalogram_metrics(cs_sc_m, cs_sc_p, cs_sc_r)
    delta_ssim = bd_metrics["ssim_meas_pred"] - cs_metrics["ssim_meas_pred"]
    delta_pcc = bd_metrics["r_meas_pred"] - cs_metrics["r_meas_pred"]
    delta_ssim_resid = cs_metrics["ssim_meas_resid"] - bd_metrics["ssim_meas_resid"]
    delta_pcc_resid = cs_metrics["r_meas_resid"] - bd_metrics["r_meas_resid"]

    # Shared per-channel dB ranges so BD and CS are directly comparable.
    # Residual gets a tighter percentile (10-90) because the BD vs CS gap on
    # the residual is small in dB terms (per-window drift ~18 mV against
    # oscillation std ~23 mV); the wider 1-99 range washes the gap out
    # toward the bottom of the colormap. V_meas / V_pred keep 1-99 since
    # their dynamic range is dominated by DST current pulses.
    vmin_m, vmax_m = _shared_range(bd_sc_m, cs_sc_m, pct_lo=1.0,  pct_hi=99.0)
    vmin_p, vmax_p = _shared_range(bd_sc_p, cs_sc_p, pct_lo=1.0,  pct_hi=99.0)
    vmin_r, vmax_r = _shared_range(bd_sc_r, cs_sc_r, pct_lo=10.0, pct_hi=90.0)

    bd_meta = parse_filename(bd_file)
    cs_meta = parse_filename(cs_file)

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    for row, (label, w, sc_m, sc_p, sc_r, meta, m) in enumerate([
        ("BD (healthy)", bd_w, bd_sc_m, bd_sc_p, bd_sc_r, bd_meta, bd_metrics),
        (f"CS ({cs_meta['ohm']} ohm ISC)", cs_w, cs_sc_m, cs_sc_p, cs_sc_r, cs_meta, cs_metrics),
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

        # Columns 1..3: V_meas / V_pred / residual scalograms.
        # SSIM and Pearson r are reported on each channel where they carry
        # diagnostic meaning:
        #   V_pred panel    -> meas vs pred (drops on CS)
        #   residual panel  -> meas vs resid (rises on CS)
        m1 = _draw_scalogram(axes[row, 1], sc_m, f, coi, w["t"],
                             vmin_m, vmax_m, f"{label} — V_meas scalogram")
        m2 = _draw_scalogram(
            axes[row, 2], sc_p, f, coi, w["t"], vmin_p, vmax_p,
            f"{label} — V_pred scalogram\n"
            f"SSIM(meas,pred)={m['ssim_meas_pred']:.3f}   "
            f"r={m['r_meas_pred']:.3f}")
        m3 = _draw_scalogram(
            axes[row, 3], sc_r, f, coi, w["t"], vmin_r, vmax_r,
            f"{label} — residual scalogram\n"
            f"SSIM(meas,resid)={m['ssim_meas_resid']:.3f}   "
            f"r={m['r_meas_resid']:.3f}")

        if row == 0:
            fig.colorbar(m1, ax=axes[:, 1].tolist(), label="dB magnitude", fraction=0.03, pad=0.02)
            fig.colorbar(m2, ax=axes[:, 2].tolist(), label="dB magnitude", fraction=0.03, pad=0.02)
            fig.colorbar(m3, ax=axes[:, 3].tolist(), label="dB magnitude", fraction=0.03, pad=0.02)

    fig.suptitle(
        f"Three-Scalogram BD vs CS comparison  —  "
        f"{bd_meta['charge_rate']}C {bd_meta['discharge_mode']}, {cs_meta['ohm']} ohm ISC\n"
        f"meas-vs-pred  ΔSSIM = {delta_ssim:+.3f}  Δr = {delta_pcc:+.3f}     "
        f"meas-vs-resid  ΔSSIM = {delta_ssim_resid:+.3f}  Δr = {delta_pcc_resid:+.3f}",
        fontsize=12, y=0.995,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison figure -> {out_path}")

    # Persist both rows to results/scalogram_similarity.csv so the numbers
    # are auditable without re-running the figure render.
    for path, class_, meta, metrics in [
        (bd_file, "BD", bd_meta, bd_metrics),
        (cs_file, "CS", cs_meta, cs_metrics),
    ]:
        append_metrics_row(
            SIMILARITY_CSV,
            source="compare_bd_cs",
            file=os.path.basename(path),
            class_=class_,
            ohm=meta.get("ohm"),
            charge_rate=meta.get("charge_rate"),
            discharge_mode=meta.get("discharge_mode"),
            window_size=window_size,
            start_sample=start_sample,
            metrics=metrics,
        )
    print(f"Appended similarity rows -> {SIMILARITY_CSV}")
    return out_path


# ----------------------------------------------------------------------
#  Hero figure (report / slide layout)
# ----------------------------------------------------------------------
def compare_bd_cs_hero(bd_file: str,
                       cs_file: str,
                       pinn_weights: str,
                       out_path: str,
                       window_size: int = WINDOW_SIZE,
                       start_sample: int = 0,
                       sampling_time: Optional[float] = None,
                       capacity_as: Optional[float] = None) -> str:
    """Three-column "hero" comparison built for a report or slide.

    Layout (2 rows x 3 cols):
        col 1 -- V_meas (orange) and V_pred (green dashed) overlaid in time
        col 2 -- residual e(t) = V_meas - V_pred in time, with smooth mean
        col 3 -- residual scalogram (mean-detrended), shared dB range so
                 BD and CS are directly comparable on the same colormap

    The point of this figure is to make a single visual claim: the residual
    on BD is small and oscillation-dominated, while the residual on CS shows
    the slow ISC drift that re-emerges in the low-frequency band of the
    scalogram. No V_meas / V_pred standalone scalograms (that's what the
    diagnostic 2x4 figure is for); this is the "what does the diagnostic
    actually look like?" version.
    """
    nn = load_pinn(pinn_weights)

    bd_pred = predict_voltage_series(nn, load_tsinghua_csv(bd_file)["discharge_df"],
                                     sampling_time=sampling_time, capacity_as=capacity_as)
    cs_pred = predict_voltage_series(nn, load_tsinghua_csv(cs_file)["discharge_df"],
                                     sampling_time=sampling_time, capacity_as=capacity_as)

    bd_w = _select_window(bd_pred, window_size, start_sample)
    cs_w = _select_window(cs_pred, window_size, start_sample)
    fs = 1.0

    # Residual scalogram only (mean detrend -- preserves the slow ISC drift).
    bd_sc_r, f, coi = _scalogram(bd_w["residual"], fs, detrend="mean")
    cs_sc_r, _, _   = _scalogram(cs_w["residual"], fs, detrend="mean")

    # Tight residual dB range (10-90 percentile) so the BD/CS contrast reads
    # at slide size; the 1-99 default washes the small dB gap to the bottom
    # of the colormap.
    vmin_r, vmax_r = _shared_range(bd_sc_r, cs_sc_r, pct_lo=10.0, pct_hi=90.0)

    bd_meta = parse_filename(bd_file)
    cs_meta = parse_filename(cs_file)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5),
                             gridspec_kw={"width_ratios": [1.1, 1.1, 1.4]})

    last_mesh = None
    for row, (label, color, w, sc_r, meta) in enumerate([
        ("BD healthy",                "#1b9e77", bd_w, bd_sc_r, bd_meta),
        (f"CS {cs_meta['ohm']} Ω ISC", "#d95f02", cs_w, cs_sc_r, cs_meta),
    ]):
        # Col 1 -- voltage overlay
        ax_v = axes[row, 0]
        ax_v.plot(w["t"], w["v_meas"], color="#d95f02", linewidth=1.6, label="V_meas")
        ax_v.plot(w["t"], w["v_pred"], color="#1b9e77", linewidth=1.4,
                  linestyle="--", label="V_pred (PINN)")
        ax_v.set_xlabel("Time (s)", fontsize=11)
        ax_v.set_ylabel("Voltage (V)", fontsize=11)
        ax_v.grid(True, alpha=0.3)
        ax_v.legend(loc="upper right", fontsize=10, framealpha=0.9)
        ax_v.set_title(f"{label}\nVoltage trace ({window_size}-sample window)",
                       fontsize=12, color=color, fontweight="bold")
        ax_v.tick_params(labelsize=10)

        # Col 2 -- residual time series
        ax_r = axes[row, 1]
        residual_mv = w["residual"] * 1000.0
        smooth = np.convolve(residual_mv, np.ones(60) / 60.0, mode="same")
        ax_r.plot(w["t"], residual_mv, color="#e7298a", linewidth=0.8,
                  alpha=0.55, label="raw")
        ax_r.plot(w["t"], smooth, color="black", linewidth=1.6,
                  label="60-s mean")
        ax_r.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
        rms_mv = float(np.sqrt(np.mean(residual_mv ** 2)))
        mean_mv = float(np.mean(residual_mv))
        ax_r.set_xlabel("Time (s)", fontsize=11)
        ax_r.set_ylabel("Residual e(t) (mV)", fontsize=11)
        ax_r.grid(True, alpha=0.3)
        ax_r.legend(loc="upper right", fontsize=10, framealpha=0.9)
        ax_r.set_title(f"Residual e(t) = V_meas − V_pred\n"
                       f"RMS = {rms_mv:.1f} mV    mean = {mean_mv:+.1f} mV",
                       fontsize=12, color=color, fontweight="bold")
        ax_r.tick_params(labelsize=10)

        # Col 3 -- residual scalogram
        ax_sc = axes[row, 2]
        last_mesh = _draw_scalogram(
            ax_sc, sc_r, f, coi, w["t"], vmin_r, vmax_r,
            f"{label}  —  residual scalogram\n(mean-detrend; shared dB range)")
        ax_sc.title.set_color(color)
        ax_sc.title.set_fontweight("bold")
        ax_sc.title.set_fontsize(12)
        ax_sc.tick_params(labelsize=10)

    fig.suptitle(
        f"ISC fault diagnostic — Tsinghua NCM811 {bd_meta['charge_rate']}C "
        f"{bd_meta['discharge_mode']}   |   BD healthy vs CS "
        f"{cs_meta['ohm']} Ω severe ISC",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.012,
        "Reading: BD residual oscillates around zero (oscillations only); "
        "CS residual drifts negative as SOC drains and that slow drift renders "
        "as low-frequency energy in the residual scalogram — the diagnostic feature.",
        ha="center", fontsize=10, color="#444",
    )

    # tight_layout first, leaving room on the right for a manual colorbar.
    # Reserving space via the `rect` parameter avoids the "Axes not compatible
    # with tight_layout" warning that fires when fig.colorbar(ax=axes[:,2])
    # injects a row-spanning colorbar into the layout.
    fig.tight_layout(rect=[0, 0.035, 0.92, 0.965])
    if last_mesh is not None:
        cax = fig.add_axes([0.93, 0.12, 0.012, 0.78])  # [left, bottom, w, h]
        fig.colorbar(last_mesh, cax=cax,
                     label="dB magnitude (residual scalogram)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved hero figure -> {out_path}")
    return out_path
