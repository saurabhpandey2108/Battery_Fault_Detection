"""
Eyeball-check the Tsinghua NCM811 discharge data BEFORE any training.

Produces, under results/dataset_inspection/:
  * one 4-panel PNG per representative file (V, I, dataset SOC, capacity)
  * one overlay PNG comparing V(t) and SOC(t) of all three files

Usage:
    python main.py visualize-dataset
    # or directly:
    python -m src.visualize_dataset

This script is non-negotiable: run it and look at the output before
running train-pinn. The eyeball check catches dataset surprises (units,
sign conventions, NaN runs, sampling-rate drift, monotonicity) that
would silently corrupt training otherwise.
"""

from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# File lives at src/visualization/<this>.py -- climb 3 dirnames to reach project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import DATASET_DIR, RESULTS_DIR, Q_RATED_AH
from src.data.tsinghua_loader import load_tsinghua_csv
from src.data.prepare_pinn_features import coulomb_counted_soc


# Three representative files — paths resolved against DATASET_DIR.
_DEFAULT_FILES = [
    ("BD healthy",    os.path.join("NCM811_NORMAL_TEST", "DST", "ISC_BD_0.5CC_DST_100ohm.csv")),
    ("CS severe ISC", os.path.join("NCM811_ISC_TEST",    "DST", "ISC_CS_0.5CC_DST_10ohm.csv")),
    ("CS mild ISC",   os.path.join("NCM811_ISC_TEST",    "DST", "ISC_CS_0.5CC_DST_1000ohm.csv")),
]
_OVERLAY_COLORS = {"BD healthy": "black", "CS mild ISC": "blue", "CS severe ISC": "red"}


def _per_file_figure(label: str, fname: str, df, out_dir: str) -> dict:
    """Render the 4-panel inspection figure and return summary stats."""
    t = df["time_s"].to_numpy(dtype=np.float64)
    t0 = t[0]
    t_rel = t - t0

    v = df["voltage_V"].to_numpy(dtype=np.float64)
    i = df["current_A"].to_numpy(dtype=np.float64)
    soc = df["soc_pct"].to_numpy(dtype=np.float64)
    cap = df["capacity_Ah"].to_numpy(dtype=np.float64)
    soc_coulomb = coulomb_counted_soc(i, t, q_rated_ah=Q_RATED_AH, soc0=1.0) * 100.0

    dt = np.diff(t)
    dt_med = float(np.median(dt)) if len(dt) else float("nan")

    soc_monotone = bool(np.all(np.diff(soc) <= 1e-9))
    coulomb_monotone = bool(np.all(np.diff(soc_coulomb) <= 1e-3))

    stats = {
        "label": label,
        "file": fname,
        "n_samples": len(t),
        "duration_s": float(t_rel[-1] - t_rel[0]) if len(t) else 0.0,
        "dt_median_s": dt_med,
        "v_min": float(v.min()), "v_max": float(v.max()),
        "i_min": float(i.min()), "i_max": float(i.max()),
        "soc_min": float(soc.min()), "soc_max": float(soc.max()),
        "soc_monotone_decreasing": soc_monotone,
        "soc_has_nan": bool(np.any(np.isnan(soc))),
        "soc_coulomb_start": float(soc_coulomb[0]),
        "soc_coulomb_end": float(soc_coulomb[-1]),
        "coulomb_monotone": coulomb_monotone,
    }

    fig, axes = plt.subplots(5, 1, figsize=(11, 11), sharex=True)

    axes[0].plot(t_rel, v, color="black", linewidth=0.9)
    axes[0].set_ylabel("Voltage (V)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_rel, i, color="red", linewidth=0.9)
    axes[1].axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
    axes[1].set_ylabel("Current (A)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_rel, soc, color="blue", linewidth=0.9, alpha=0.9,
                 label="dataset SOC|DOD/% (per-step)")
    axes[2].set_ylabel("Dataset SOC (%)")
    axes[2].set_ylim(-2, 102)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right", fontsize=8)

    axes[3].plot(t_rel, soc_coulomb, color="purple", linewidth=1.1,
                 label=f"coulomb-counted SOC (Q_rated={Q_RATED_AH:.3f} Ah)")
    axes[3].set_ylabel("Coulomb SOC (%)")
    axes[3].set_ylim(-2, 102)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="upper right", fontsize=8)

    axes[4].plot(t_rel, cap, color="green", linewidth=0.9)
    axes[4].set_ylabel("Capacity (Ah)")
    axes[4].set_xlabel("Time since discharge start (s)")
    axes[4].grid(True, alpha=0.3)

    fig.suptitle(f"{os.path.basename(fname)} -- discharge phase ({label})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(fname))[0]}.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")

    return stats


def _overlay_figure(rows: List[dict], out_path: str):
    """Voltage + dataset-SOC + coulomb-SOC overlay across all three files."""
    fig, (ax_v, ax_s_ds, ax_s_cc) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    for r in rows:
        col = _OVERLAY_COLORS.get(r["label"], "gray")
        t = r["t"]
        ax_v.plot(t, r["v"], color=col, linewidth=1.0,
                  label=f"{r['label']} ({os.path.basename(r['file'])})")
        ax_s_ds.plot(t, r["soc_dataset"], color=col, linewidth=1.0,
                     label=f"{r['label']}")
        ax_s_cc.plot(t, r["soc_coulomb"], color=col, linewidth=1.2,
                     label=f"{r['label']}")

    ax_v.set_ylabel("Voltage (V)")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend(loc="lower left", fontsize=8)
    ax_v.set_title("Discharge voltage overlay  --  BD vs CS at the same DST profile")

    ax_s_ds.set_ylabel("Dataset SOC|DOD (%)")
    ax_s_ds.set_ylim(-2, 102)
    ax_s_ds.grid(True, alpha=0.3)
    ax_s_ds.set_title("Dataset SOC|DOD column (per-step Arbin DOD -- NOT used as input)")

    ax_s_cc.set_ylabel("Coulomb SOC (%)")
    ax_s_cc.set_xlabel("Time since discharge start (s)")
    ax_s_cc.set_ylim(-2, 102)
    ax_s_cc.grid(True, alpha=0.3)
    ax_s_cc.legend(loc="upper right", fontsize=8)
    ax_s_cc.set_title("Coulomb-counted SOC (fault-blind, USED as PINN input)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    out_dir = os.path.join(RESULTS_DIR, "dataset_inspection")
    os.makedirs(out_dir, exist_ok=True)

    summaries: List[dict] = []
    overlay_rows: List[dict] = []

    for label, rel in _DEFAULT_FILES:
        fpath = os.path.join(DATASET_DIR, rel)
        if not os.path.isfile(fpath):
            print(f"  MISSING {fpath} -- skipping")
            continue
        d = load_tsinghua_csv(fpath)
        df = d["discharge_df"]
        stats = _per_file_figure(label, rel, df, out_dir)
        summaries.append(stats)

        t = df["time_s"].to_numpy(dtype=np.float64)
        i_arr = df["current_A"].to_numpy(dtype=np.float64)
        soc_cc = coulomb_counted_soc(i_arr, t, q_rated_ah=Q_RATED_AH, soc0=1.0) * 100.0
        overlay_rows.append({
            "label": label,
            "file": rel,
            "t": t - t[0],
            "v": df["voltage_V"].to_numpy(dtype=np.float64),
            "soc_dataset": df["soc_pct"].to_numpy(dtype=np.float64),
            "soc_coulomb": soc_cc,
        })

    if overlay_rows:
        _overlay_figure(overlay_rows, os.path.join(out_dir, "_comparison_overlay.png"))

    # ----- stdout report -----
    print("\n=== Dataset inspection summary ===")
    for s in summaries:
        flag_dt = "" if abs(s["dt_median_s"] - 1.0) < 1e-3 else "  <-- NOT 1.000s, flag"
        print(f"\n{s['label']:<14s}  {s['file']}")
        print(f"  N samples                : {s['n_samples']}")
        print(f"  duration                 : {s['duration_s']:.1f} s")
        print(f"  median dt                : {s['dt_median_s']:.4f} s{flag_dt}")
        print(f"  voltage range            : [{s['v_min']:.3f}, {s['v_max']:.3f}] V")
        print(f"  current range            : [{s['i_min']:.3f}, {s['i_max']:.3f}] A")
        print(f"  dataset SOC|DOD range    : [{s['soc_min']:.2f}, {s['soc_max']:.2f}] %  (NOT used as input)")
        print(f"  dataset SOC monotone     : {s['soc_monotone_decreasing']}  (per-step DOD, expected non-monotone)")
        print(f"  coulomb SOC start..end   : {s['soc_coulomb_start']:.3f}  ->  {s['soc_coulomb_end']:.3f}  (USED as input)")
        print(f"  coulomb SOC monotone(<=) : {s['coulomb_monotone']}  (False is OK, regen pulses produce small upticks)")

    # Note about what comparison overlay tells us about SOC drainage
    if len(overlay_rows) >= 2:
        print("\n(Read the SOC overlay panel: if CS files reach a lower final SOC than BD")
        print(" at the same elapsed time, the dataset's SOC accounts for the ISC drain.")
        print(" If they end at the same SOC, the dataset's SOC only counts external coulomb")
        print(" flow through the ammeter -- the ISC current is hidden, exactly the")
        print(" condition that lets the residual e(t)=V_meas-V_pred encode the fault.)")


if __name__ == "__main__":
    main()
