"""
Pairwise similarity metrics for 2-D dB-magnitude scalograms.

Used to quantify how much spectral structure is shared between the three
scalogram channels in this project:

  V_meas  vs  V_pred   -- HIGH on BD (PINN tracks the measurement closely),
                          LOWER on CS (PINN can only model healthy behaviour
                          while V_meas carries the ISC signature, so the
                          two scalograms drift apart in the time-frequency
                          plane).
  V_meas  vs  residual -- LOW on BD (residual is mostly noise, no shared
                          structure), HIGHER on CS (V_pred drifts off, so
                          the residual inherits the band-limited structure
                          that lives in V_meas).
  V_pred  vs  residual -- mostly informational; complements the above.

Two metrics are reported:

  Pearson r  -- pixel-wise linear correlation on the overlap of finite
                pixels. Cheap, no spatial weighting; sensitive only to
                point-by-point agreement.

  SSIM       -- Structural Similarity Index with a Gaussian window.
                Penalises differences in local mean, local variance, AND
                local correlation, so two scalograms with the same per-
                pixel mean but different "texture" still score below 1.
                NaN pixels (from the COI mask) are filled with the joint
                finite-pixel mean before convolution so windowed mean/
                variance/covariance are well-defined; the fill is the
                same for both inputs and therefore does not bias the
                metric in either direction.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
from scipy.ndimage import gaussian_filter


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation on the overlap of finite pixels."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    valid = ~(np.isnan(a) | np.isnan(b))
    if valid.sum() < 2:
        return float("nan")
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def ssim(a: np.ndarray, b: np.ndarray, sigma: float = 1.5) -> float:
    """Mean SSIM between two 2-D arrays using a Gaussian window."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = ~(np.isnan(a) | np.isnan(b))
    if valid.sum() < 2:
        return float("nan")
    fill = float(np.mean(np.concatenate([a[valid], b[valid]])))
    a = np.where(np.isnan(a), fill, a)
    b = np.where(np.isnan(b), fill, b)
    data_range = float(np.max([a.max(), b.max()]) - np.min([a.min(), b.min()]))
    if data_range <= 0:
        return 1.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_a = gaussian_filter(a, sigma)
    mu_b = gaussian_filter(b, sigma)
    sa2 = gaussian_filter(a * a, sigma) - mu_a * mu_a
    sb2 = gaussian_filter(b * b, sigma) - mu_b * mu_b
    sab = gaussian_filter(a * b, sigma) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * sab + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (sa2 + sb2 + c2)
    return float(np.mean(num / den))


def pairwise_scalogram_metrics(meas: np.ndarray,
                               pred: np.ndarray,
                               resid: np.ndarray) -> Dict[str, float]:
    """All three pairwise SSIM and Pearson-r values for a three-scalogram stack.

    Inputs are 2-D dB-magnitude scalograms (NaN allowed in the COI). Returns
    a dict with six numbers in a stable key order.
    """
    return {
        "ssim_meas_pred":  ssim(meas, pred),
        "r_meas_pred":     pearson_r(meas, pred),
        "ssim_meas_resid": ssim(meas, resid),
        "r_meas_resid":    pearson_r(meas, resid),
        "ssim_pred_resid": ssim(pred, resid),
        "r_pred_resid":    pearson_r(pred, resid),
    }


# ---------------------------------------------------------------------------
#  CSV persistence
# ---------------------------------------------------------------------------
CSV_HEADER = [
    "timestamp", "source", "file", "class", "ohm",
    "charge_rate", "discharge_mode",
    "window_size", "start_sample",
    "ssim_meas_pred", "r_meas_pred",
    "ssim_meas_resid", "r_meas_resid",
    "ssim_pred_resid", "r_pred_resid",
]


def plot_similarity_summary(csv_path: str, png_path: str,
                            source_filter: str = "compare_bd_cs") -> str:
    """Read the similarity CSV and render a 2x3 BD-vs-CS-vs-ohm summary.

    Layout:
        rows : [SSIM, Pearson r]
        cols : [meas-vs-pred, meas-vs-resid, pred-vs-resid]

    Each panel plots BD (green) and CS (orange) against ohm on a log x-axis,
    with the gap shaded. The diagnostic direction is annotated:
        meas-vs-pred  : BD > CS expected (PINN tracks healthy, drifts on ISC)
        meas-vs-resid : CS > BD expected (residual inherits V_meas on faults)
        pred-vs-resid : informational only.

    Returns the figure path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    rows = []
    with open(csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            if r.get("source") != source_filter:
                continue
            try:
                r["ohm"] = int(r["ohm"])
            except (ValueError, KeyError):
                continue
            rows.append(r)

    if not rows:
        raise RuntimeError(
            f"No '{source_filter}' rows in {csv_path}. "
            f"Run `main.py compare --ohm <X>` for each phase-1 ohm first."
        )

    metrics = [
        ("ssim_meas_pred",  "r_meas_pred",  "V_meas vs V_pred",
         "BD > CS  (PINN tracks healthy)"),
        ("ssim_meas_resid", "r_meas_resid", "V_meas vs residual",
         "CS > BD  (residual inherits V_meas on faults)"),
        ("ssim_pred_resid", "r_pred_resid", "V_pred vs residual",
         "informational"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)

    for col, (ssim_key, r_key, pair_label, direction) in enumerate(metrics):
        for row_i, mkey in enumerate([ssim_key, r_key]):
            ax = axes[row_i, col]
            for class_, color in [("BD", "#1b9e77"), ("CS", "#d95f02")]:
                pts = sorted(
                    [(r["ohm"], float(r[mkey])) for r in rows
                     if r["class"] == class_],
                    key=lambda p: p[0],
                )
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, marker="o", color=color, linewidth=1.6,
                        markersize=5, label=class_)

            # Shade the gap between the two lines (if both exist).
            bd_pts = sorted(
                [(r["ohm"], float(r[mkey])) for r in rows
                 if r["class"] == "BD"], key=lambda p: p[0])
            cs_pts = sorted(
                [(r["ohm"], float(r[mkey])) for r in rows
                 if r["class"] == "CS"], key=lambda p: p[0])
            if bd_pts and cs_pts and len(bd_pts) == len(cs_pts):
                xs = [p[0] for p in bd_pts]
                bd_y = np.array([p[1] for p in bd_pts])
                cs_y = np.array([p[1] for p in cs_pts])
                ax.fill_between(xs, bd_y, cs_y,
                                color="gray", alpha=0.15, linewidth=0)

            ax.set_xscale("log")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(loc="best", fontsize=8)
            metric_name = "SSIM" if row_i == 0 else "Pearson r"
            ax.set_ylabel(metric_name)
            if row_i == 1:
                ax.set_xlabel("ISC resistance (ohm, log)")
            if row_i == 0:
                ax.set_title(f"{pair_label}\n[{direction}]", fontsize=10)

    fig.suptitle("Scalogram pairwise similarity vs ISC resistance  "
                 "(one window per file, start_sample=0)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    return png_path


def append_metrics_row(csv_path: str,
                       *,
                       source: str,
                       file: str,
                       class_: str,
                       ohm: Optional[int],
                       charge_rate: Optional[float],
                       discharge_mode: Optional[str],
                       window_size: int,
                       start_sample: int,
                       metrics: Dict[str, float]) -> None:
    """Append a single row to the shared scalogram-similarity CSV.

    Header is written automatically the first time the file is created (or
    if it exists but is empty). All numeric metrics are written as full-
    precision floats; missing categorical fields are written as empty
    strings, which pandas / csv readers parse as NaN / "".
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(CSV_HEADER)
        w.writerow([
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source, file, class_,
            "" if ohm is None else ohm,
            "" if charge_rate is None else charge_rate,
            "" if discharge_mode is None else discharge_mode,
            window_size, start_sample,
            metrics["ssim_meas_pred"], metrics["r_meas_pred"],
            metrics["ssim_meas_resid"], metrics["r_meas_resid"],
            metrics["ssim_pred_resid"], metrics["r_pred_resid"],
        ])
