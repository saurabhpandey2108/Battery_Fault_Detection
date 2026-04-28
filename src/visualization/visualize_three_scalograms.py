"""
CLI visualizer: produce a 2×2 diagnostic PNG for one Tsinghua CSV.

Layout:
    top-left   : V_meas(t) and V_pred(t) overlaid, I(t) on secondary axis
    top-right  : residual e(t) = V_meas - V_pred
    bottom row : V_meas scalogram | V_pred scalogram | residual scalogram
                 (viridis cmap, log-frequency y-axis, COI hatched white)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# File lives at src/visualization/<this>.py -- climb 3 dirnames to reach project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import MODELS_DIR, RESULTS_DIR, SCALES, WINDOW_SIZE
from src.data.tsinghua_loader import load_tsinghua_csv, parse_filename
from src.inference.infer_pinn import predict_voltage_series
from src.wavelet.image_utils import raw_log_scalogram
from src.wavelet.plot_utils import plot_frequency_scalogram
from src.wavelet.scalogram_metrics import (
    pairwise_scalogram_metrics,
    append_metrics_row,
)


SIMILARITY_CSV = os.path.join(RESULTS_DIR, "three_scalograms", "similarity.csv")


def visualize_file(filepath: str,
                   pinn_weights: str,
                   out_dir: str,
                   window_size: int = WINDOW_SIZE,
                   start_sample: int = 0,
                   resistance_label: Optional[str] = None) -> str:
    d = load_tsinghua_csv(filepath)
    pred = predict_voltage_series(pinn_weights, d["discharge_df"])

    n = len(pred["v_meas"])
    if n < window_size:
        raise ValueError(
            f"File {os.path.basename(filepath)} has only {n} engineered samples, "
            f"needs at least window_size={window_size} for scalogram panels"
        )
    start = max(0, min(start_sample, n - window_size))
    sl = slice(start, start + window_size)

    t = pred["time"][sl] - pred["time"][sl][0]
    v_meas = pred["v_meas"][sl]
    v_pred = pred["v_pred"][sl]
    current = pred["current"][sl]
    residual = pred["residual"][sl]
    fs = 1.0

    meta = parse_filename(filepath)
    tag = f"{meta['class_']} {meta['ohm']}ohm {meta['charge_rate']}C {meta['discharge_mode']}"
    if resistance_label:
        tag = f"{tag} ({resistance_label})"

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.3])

    # Top-left — V_meas / V_pred + current
    ax0 = fig.add_subplot(gs[0, :2])
    ax0.plot(t, v_meas, color="#d95f02", linewidth=1.6, label="V_meas")
    ax0.plot(t, v_pred, color="#1b9e77", linewidth=1.2, linestyle="--", label="V_pred")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Voltage (V)")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right")
    ax0_r = ax0.twinx()
    ax0_r.plot(t, current, color="#7570b3", linewidth=0.9, alpha=0.75, label="I")
    ax0_r.set_ylabel("Current (A)")
    ax0.set_title(f"Voltage & current — {tag}")

    # Top-right — residual
    ax1 = fig.add_subplot(gs[0, 2])
    ax1.plot(t, residual, color="#e7298a", linewidth=1.0)
    ax1.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Residual e(t) (V)")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Residual e(t) = V_meas − V_pred")

    # Bottom row — three scalograms
    ax_m = fig.add_subplot(gs[1, 0])
    ax_p = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])

    plot_frequency_scalogram(v_meas, fs, None, save_path=None,
                             title_suffix="— V_meas", ax=ax_m,
                             detrend="cubic")
    plot_frequency_scalogram(v_pred, fs, None, save_path=None,
                             title_suffix="— V_pred", ax=ax_p,
                             detrend="cubic")
    plot_frequency_scalogram(residual, fs, None, save_path=None,
                             title_suffix="— residual e(t)", ax=ax_e,
                             detrend="mean")

    # Pairwise SSIM and Pearson r on the dB-magnitude scalograms with the
    # SAME per-channel detrend the CNN sees (cubic for meas/pred, mean for
    # residual -- preserves the slow drift that carries the ISC signature).
    sc_meas,  _ = raw_log_scalogram(v_meas,   fs, SCALES, detrend="cubic")
    sc_pred,  _ = raw_log_scalogram(v_pred,   fs, SCALES, detrend="cubic")
    sc_resid, _ = raw_log_scalogram(residual, fs, SCALES, detrend="mean")
    metrics = pairwise_scalogram_metrics(sc_meas, sc_pred, sc_resid)

    fig.suptitle(
        f"Three-Scalogram Diagnostic — {os.path.basename(filepath)}\n"
        f"SSIM(meas,pred)={metrics['ssim_meas_pred']:.3f}  "
        f"r={metrics['r_meas_pred']:.3f}     "
        f"SSIM(meas,resid)={metrics['ssim_meas_resid']:.3f}  "
        f"r={metrics['r_meas_resid']:.3f}     "
        f"SSIM(pred,resid)={metrics['ssim_pred_resid']:.3f}  "
        f"r={metrics['r_pred_resid']:.3f}",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir,
                            os.path.splitext(os.path.basename(filepath))[0] + ".png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved -> {out_path}")

    append_metrics_row(
        SIMILARITY_CSV,
        source="visualize_three_scalograms",
        file=os.path.basename(filepath),
        class_=meta.get("class_", ""),
        ohm=meta.get("ohm"),
        charge_rate=meta.get("charge_rate"),
        discharge_mode=meta.get("discharge_mode"),
        window_size=window_size,
        start_sample=start,
        metrics=metrics,
    )
    print(f"Appended similarity row -> {SIMILARITY_CSV}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Render a 2×2 three-scalogram figure for one CSV")
    parser.add_argument("file", help="Path to a Tsinghua CSV")
    parser.add_argument("--pinn-weights", default=os.path.join(MODELS_DIR, "pinn_healthy_no_leak.npz"))
    parser.add_argument("--out-dir", default=os.path.join(RESULTS_DIR, "three_scalograms"))
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--start-sample", type=int, default=0)
    parser.add_argument("--label", default=None, help="Optional free-form annotation")
    args = parser.parse_args()

    visualize_file(
        filepath=args.file,
        pinn_weights=args.pinn_weights,
        out_dir=args.out_dir,
        window_size=args.window_size,
        start_sample=args.start_sample,
        resistance_label=args.label,
    )


if __name__ == "__main__":
    main()
