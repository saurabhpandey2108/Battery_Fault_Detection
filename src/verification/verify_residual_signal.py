"""
Acceptance gate: does the frozen leakage-free PINN produce a residual
signal that actually distinguishes BD (healthy) from CS (faulty) cells?

Runs the frozen weights on three representative files
  * BD healthy    (NCM811_NORMAL_TEST/DST/ISC_BD_0.5CC_DST_100ohm.csv)
  * CS severe ISC (NCM811_ISC_TEST/DST/ISC_CS_0.5CC_DST_10ohm.csv)
  * CS mild   ISC (NCM811_ISC_TEST/DST/ISC_CS_0.5CC_DST_1000ohm.csv)

For each file: load -> predict -> residual e(t) = V_meas - V_pred.

Why we look at drift, not RMS
------------------------------
The PINN's residual is dominated by counter-phase oscillation against
DST current pulses (a model-fit issue), so RMS is largely noise. The
fault signal lives in the SLOW DRIFT of the residual: as the ISC bleeds
hidden charge, V_meas falls below V_pred, and the residual mean trends
more negative the more discharge has happened. We therefore compute:

  * raw residual stats (RMS, max-abs, mean) -- reported for diagnostics
  * smoothed residual via 60-s moving-average lowpass:
        - smoothed mean
        - smoothed RMS
  * drift slope via linear regression of residual against coulomb SOC:
        residual ~ a + b * SOC_coulomb
        SOC decreases from 1 -> 0 across discharge, and on a fault cell
        residual gets more negative as discharge progresses. So as SOC
        decreases the residual decreases too -> slope b is POSITIVE,
        and a more severe fault has a LARGER positive slope.

Acceptance criteria (drift-aware)
---------------------------------
  [1] CS_10 drift slope - BD drift slope >= 100 mV per unit-SOC
      (the severe ISC produces a clear extra drift in residual vs SOC).
  [2] CS_10 smoothed mean is at least 10 mV more negative than BD's.
  [3] CS_10 coulomb-SOC at end-of-discharge >= BD end + 30 percentage
      points (dataset sanity check that the hidden ISC drain forced
      the cell to hit cutoff voltage with charge unaccounted for in
      the ammeter -- i.e. the fault is real in the data).
  [4] All three runs finite.

Mild ISC (1000 ohm) is intentionally not in the ladder: 4V/1000ohm = 4 mA
drain, ~12 mAh over a 3.5-hour discharge, ~0.5% of cell capacity. That
is below the PINN's residual noise floor and physically imperceptible.
We require severity to be detectable for severe ISC; mild ISC is a Phase 2
problem.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# File lives at src/verification/<this>.py -- climb 3 dirnames to reach project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import DATASET_DIR, MODELS_DIR, RESULTS_DIR
from src.data.tsinghua_loader import load_tsinghua_csv
from src.inference.infer_pinn import predict_voltage_series


_FILES = [
    ("BD 100ohm  (healthy)",   "NCM811_NORMAL_TEST/DST/ISC_BD_0.5CC_DST_100ohm.csv"),
    ("CS 10ohm   (severe ISC)", "NCM811_ISC_TEST/DST/ISC_CS_0.5CC_DST_10ohm.csv"),
    ("CS 1000ohm (mild ISC)",  "NCM811_ISC_TEST/DST/ISC_CS_0.5CC_DST_1000ohm.csv"),
]

# 60-second moving-average lowpass at 1 Hz sampling.
LOWPASS_WINDOW = 60


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or w >= len(x):
        return x.copy()
    kernel = np.ones(w, dtype=np.float64) / w
    # 'same' keeps length; edges are slightly biased but cheap and good enough.
    return np.convolve(x, kernel, mode="same")


def _drift_slope(residual: np.ndarray, soc: np.ndarray) -> tuple[float, float]:
    """Linear regression of residual against SOC. SOC is in [0, 1].
    Returns (slope V per unit-SOC, intercept V)."""
    if residual.size < 2:
        return 0.0, 0.0
    s = soc.astype(np.float64)
    r = residual.astype(np.float64)
    a, b = np.polyfit(s, r, 1)   # numpy returns highest-order first
    return float(a), float(b)    # a = slope, b = intercept


def _evaluate(label: str, rel: str, weights: str) -> dict:
    fp = os.path.join(DATASET_DIR, rel)
    if not os.path.isfile(fp):
        raise FileNotFoundError(fp)

    d = load_tsinghua_csv(fp)
    df = d["discharge_df"]
    pred = predict_voltage_series(weights, df)

    res = pred["residual"]
    soc = pred["soc"]

    finite = (np.all(np.isfinite(res))
              and np.all(np.isfinite(pred["v_pred"]))
              and np.all(np.isfinite(soc)))

    rms = float(np.sqrt(np.mean(res ** 2)))
    max_abs = float(np.max(np.abs(res)))
    mean_res = float(np.mean(res))

    smooth = _moving_average(res, LOWPASS_WINDOW)
    smooth_mean = float(np.mean(smooth))
    smooth_rms = float(np.sqrt(np.mean(smooth ** 2)))

    slope, intercept = _drift_slope(res, soc)

    soc_dataset = df["soc_pct"].to_numpy()
    soc_dataset_drop = float(soc_dataset[0] - soc_dataset[-1])
    soc_cc_drop = float((pred["soc"][0] - pred["soc"][-1]) * 100.0)

    return {
        "label": label,
        "rel": rel,
        "n_samples": len(res),
        "rms_mv": rms * 1000.0,
        "max_abs_mv": max_abs * 1000.0,
        "mean_mv": mean_res * 1000.0,
        "smooth_mean_mv": smooth_mean * 1000.0,
        "smooth_rms_mv": smooth_rms * 1000.0,
        "drift_slope_mv_per_soc": slope * 1000.0,    # mV per unit-SOC (SOC in 0..1)
        "drift_intercept_mv": intercept * 1000.0,
        "soc_dataset_drop_pct": soc_dataset_drop,
        "soc_coulomb_drop_pct": soc_cc_drop,
        "finite": finite,
        "pred": pred,
        "df": df,
        "smooth_residual": smooth,
    }


def _verification_figure(rows, out_path: str):
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(20, 3.6 * n))
    if n == 1:
        axes = axes[None, :]

    for r, axrow in zip(rows, axes):
        pred = r["pred"]
        df = r["df"]
        t = pred["time"] - pred["time"][0]
        v_meas = pred["v_meas"]
        v_pred = pred["v_pred"]
        soc_cc = pred["soc"] * 100.0
        residual = pred["residual"] * 1000.0           # to mV
        smooth = r["smooth_residual"] * 1000.0

        ax_v, ax_s, ax_r, ax_d = axrow

        ax_v.plot(t, v_meas, color="#d95f02", linewidth=1.0, label="V_meas")
        ax_v.plot(t, v_pred, color="#1b9e77", linewidth=1.0,
                  linestyle="--", label="V_pred")
        ax_v.set_ylabel("V (V)")
        ax_v.legend(loc="lower left", fontsize=8)
        ax_v.grid(True, alpha=0.3)
        ax_v.set_title(f"{r['label']}\nV_meas vs V_pred")

        ax_s.plot(t, soc_cc, color="purple", linewidth=1.1)
        ax_s.set_ylabel("Coulomb SOC (%)")
        ax_s.set_xlabel("Time (s)")
        ax_s.set_ylim(-2, 102)
        ax_s.grid(True, alpha=0.3)
        ax_s.set_title(f"Coulomb SOC trajectory  (drop {r['soc_coulomb_drop_pct']:.1f} pp)")

        ax_r.plot(t, residual, color="#e7298a", linewidth=0.7, alpha=0.5,
                  label="raw")
        ax_r.plot(t, smooth, color="black", linewidth=1.5,
                  label=f"{LOWPASS_WINDOW}-s moving avg")
        ax_r.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
        ax_r.set_ylabel("residual (mV)")
        ax_r.set_xlabel("Time (s)")
        ax_r.grid(True, alpha=0.3)
        ax_r.legend(loc="upper right", fontsize=8)
        ax_r.set_title(f"residual e(t) over time\n"
                       f"raw RMS {r['rms_mv']:.0f} mV  smooth mean {r['smooth_mean_mv']:.1f} mV")

        # residual-vs-SOC scatter with regression line
        s_pct = pred["soc"] * 100.0
        ax_d.plot(s_pct, residual, ".", color="#e7298a",
                  alpha=0.25, markersize=1, label="residual")
        x_line = np.array([s_pct.min(), s_pct.max()])
        y_line = (r["drift_slope_mv_per_soc"] * (x_line / 100.0)
                  + r["drift_intercept_mv"])
        ax_d.plot(x_line, y_line, color="black", linewidth=1.6,
                  label=f"fit  slope={r['drift_slope_mv_per_soc']:.1f} mV / unit-SOC")
        ax_d.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
        ax_d.set_xlabel("Coulomb SOC (%)")
        ax_d.set_ylabel("residual (mV)")
        ax_d.invert_xaxis()  # so time flows left -> right (high SOC -> low SOC)
        ax_d.grid(True, alpha=0.3)
        ax_d.legend(loc="lower right", fontsize=8)
        ax_d.set_title("residual vs SOC -- drift signature")

    fig.suptitle("Residual signal verification (drift-aware)  --  frozen leakage-free PINN",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    weights = os.path.join(MODELS_DIR, "pinn_healthy_no_leak.npz")
    if not os.path.isfile(weights):
        raise SystemExit(
            f"frozen weights not found at {weights} -- run train-pinn first"
        )

    rows = [_evaluate(label, rel, weights) for label, rel in _FILES]

    # ---- pretty print
    print()
    print(f"{'File':<24s} | {'RMS (mV)':>10s} {'mean (mV)':>11s} "
          f"{'smooth mean (mV)':>17s} {'smooth RMS (mV)':>17s} "
          f"{'drift mV/SOC':>14s} | {'SOC drop coulomb %':>20s}")
    print("-" * 130)
    for r in rows:
        print(f"{r['label']:<24s} | "
              f"{r['rms_mv']:>10.1f} {r['mean_mv']:>11.2f} "
              f"{r['smooth_mean_mv']:>17.2f} {r['smooth_rms_mv']:>17.2f} "
              f"{r['drift_slope_mv_per_soc']:>14.2f} | "
              f"{r['soc_coulomb_drop_pct']:>20.2f}")

    # ---- acceptance criteria (drift-aware)
    bd = next(r for r in rows if "BD" in r["label"])
    cs10 = next(r for r in rows if "CS 10" in r["label"])
    cs1k = next(r for r in rows if "CS 1000" in r["label"])

    crit_finite = all(r["finite"] for r in rows)

    # CS_10 should have a steeper positive drift slope than BD (faulty cell
    # accumulates more residual drop per unit-SOC drained).
    drift_gap = cs10["drift_slope_mv_per_soc"] - bd["drift_slope_mv_per_soc"]
    crit_drift = drift_gap >= 100.0

    # Smoothed mean of CS_10 should be more negative than BD's by >= 10 mV.
    smooth_gap = bd["smooth_mean_mv"] - cs10["smooth_mean_mv"]
    crit_smooth = smooth_gap >= 10.0

    # Dataset sanity: CS_10 hits cutoff with charge unaccounted-for in the
    # ammeter (= internal ISC drain). End-SOC gap >= 30 percentage points.
    bd_end_pct = 100.0 - bd["soc_coulomb_drop_pct"]
    cs10_end_pct = 100.0 - cs10["soc_coulomb_drop_pct"]
    soc_gap = cs10_end_pct - bd_end_pct
    crit_internal = soc_gap >= 30.0

    print()
    print("Acceptance criteria (drift-aware):")
    print(f"  [1] CS_10 drift slope - BD drift slope >= 100 mV per unit-SOC: "
          f"{drift_gap:>7.1f}  ->  {crit_drift}")
    print(f"  [2] BD smoothed mean - CS_10 smoothed mean >= 10 mV          : "
          f"{smooth_gap:>7.2f}  ->  {crit_smooth}")
    print(f"  [3] CS_10 end SOC - BD end SOC >= 30 percentage points       : "
          f"{soc_gap:>7.2f}  ->  {crit_internal}")
    print(f"  [4] All three runs finite                                    : "
          f"{'        ':>7s}  ->  {crit_finite}")
    # Reference: CS_1000 (mild) is intentionally NOT in the criteria; its drift
    # gap to BD is here for reference only:
    print(f"  reference: CS_1000 drift slope - BD drift slope = "
          f"{cs1k['drift_slope_mv_per_soc'] - bd['drift_slope_mv_per_soc']:.1f} mV/unit-SOC "
          f"(mild ISC, not required to pass)")

    out_png = os.path.join(RESULTS_DIR, "pinn", "residual_verification.png")
    _verification_figure(rows, out_png)
    print(f"\nFigure saved -> {out_png}")

    passed = crit_finite and crit_drift and crit_smooth and crit_internal
    if passed:
        print("\nRESIDUAL SIGNAL VERIFIED -- proceed to scalograms.")
    else:
        print("\nFIX INCOMPLETE -- diagnostic table above")
        sys.exit(1)


if __name__ == "__main__":
    main()
