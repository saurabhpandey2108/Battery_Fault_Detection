"""
Acceptance gate: does the frozen no-leak PINN produce a residual signal
that actually distinguishes BD (healthy) from CS (faulty) cells?

Runs the frozen weights on three representative files
  * BD healthy    (NCM811_NORMAL_TEST/DST/ISC_BD_0.5CC_DST_100ohm.csv)
  * CS severe ISC (NCM811_ISC_TEST/DST/ISC_CS_0.5CC_DST_10ohm.csv)
  * CS mild   ISC (NCM811_ISC_TEST/DST/ISC_CS_0.5CC_DST_1000ohm.csv)

For each file: load -> predict -> residual e(t) = V_meas - V_pred.

Acceptance criteria (per spec):
  1. CS_10ohm residual RMS  >  3 * BD_100ohm residual RMS
  2. CS_10ohm residual visibly drifts negative (V_meas falls below V_pred
     because the ISC drains hidden charge that the PINN can't see).
     Operationalized as: mean(residual_CS_10ohm) < -10 mV.
  3. CS_1000ohm residual RMS lies between BD and CS_10ohm
     (severity ladder preserved).
  4. All three files run without NaN or numerical blow-up.

If all four pass: print "RESIDUAL SIGNAL VERIFIED -- proceed to scalograms."
If any fail:      print "FIX INCOMPLETE -- diagnostic table above" and stop.

Also writes results/residual_verification.png with one row per file:
   row = [V_meas vs V_pred, I(t), SOC_dataset(t) vs SOC_coulomb(t), residual e(t)].
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
    ("BD 100ohm  (healthy)", "NCM811_NORMAL_TEST/DST/ISC_BD_0.5CC_DST_100ohm.csv"),
    ("CS 10ohm   (severe ISC)", "NCM811_ISC_TEST/DST/ISC_CS_0.5CC_DST_10ohm.csv"),
    ("CS 1000ohm (mild ISC)", "NCM811_ISC_TEST/DST/ISC_CS_0.5CC_DST_1000ohm.csv"),
]


def _evaluate(label: str, rel: str, weights: str) -> dict:
    fp = os.path.join(DATASET_DIR, rel)
    if not os.path.isfile(fp):
        raise FileNotFoundError(fp)

    d = load_tsinghua_csv(fp)
    df = d["discharge_df"]
    pred = predict_voltage_series(weights, df)

    res = pred["residual"]
    finite = np.all(np.isfinite(res)) and np.all(np.isfinite(pred["v_pred"]))
    rms = float(np.sqrt(np.mean(res ** 2)))
    max_abs = float(np.max(np.abs(res)))
    mean_res = float(np.mean(res))

    # Final SOC drop from the dataset's SOC|DOD column. The column is per-step
    # and resets often, so "start - end" is a useful summary only because the
    # discharge ends near 0 and starts near 100. We report it as the spec asked
    # while also reporting the cleaner coulomb-counted SOC drop.
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
        "soc_dataset_drop_pct": soc_dataset_drop,
        "soc_coulomb_drop_pct": soc_cc_drop,
        "finite": finite,
        "pred": pred,
        "df": df,
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
        i = pred["current"]
        soc_cc = pred["soc"] * 100.0
        soc_ds = df["soc_pct"].to_numpy()[pred["offset"]:]
        residual = pred["residual"]

        ax_v, ax_i, ax_s, ax_r = axrow

        ax_v.plot(t, v_meas, color="#d95f02", linewidth=1.0, label="V_meas")
        ax_v.plot(t, v_pred, color="#1b9e77", linewidth=1.0,
                  linestyle="--", label="V_pred")
        ax_v.set_ylabel("V (V)")
        ax_v.legend(loc="lower left", fontsize=8)
        ax_v.grid(True, alpha=0.3)
        ax_v.set_title(f"{r['label']} -- V_meas vs V_pred")

        ax_i.plot(t, i, color="red", linewidth=0.8)
        ax_i.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
        ax_i.set_ylabel("I (A)")
        ax_i.grid(True, alpha=0.3)
        ax_i.set_title("Current")

        ax_s.plot(t, soc_ds, color="blue", linewidth=0.7, alpha=0.6,
                  label="dataset SOC|DOD (broken)")
        ax_s.plot(t, soc_cc, color="purple", linewidth=1.1,
                  label="coulomb SOC (input)")
        ax_s.set_ylabel("SOC (%)")
        ax_s.set_ylim(-2, 102)
        ax_s.legend(loc="upper right", fontsize=8)
        ax_s.grid(True, alpha=0.3)
        ax_s.set_title("SOC trajectories")

        ax_r.plot(t, residual * 1000.0, color="#e7298a", linewidth=0.9)
        ax_r.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
        ax_r.set_ylabel("residual (mV)")
        ax_r.set_xlabel("Time since discharge start (s)")
        ax_r.grid(True, alpha=0.3)
        ax_r.set_title(f"residual e(t)  RMS={r['rms_mv']:.1f} mV  mean={r['mean_mv']:.1f} mV")

    fig.suptitle("Residual signal verification  --  frozen leakage-free PINN", fontsize=13)
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

    # --- pretty print the table
    print()
    header = (f"{'File':<24s}  {'Residual RMS (mV)':>18s}  "
              f"{'Max |abs| (mV)':>16s}  {'Mean (mV)':>11s}  "
              f"{'SOC drop dataset (%)':>22s}  {'SOC drop coulomb (%)':>22s}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['label']:<24s}  {r['rms_mv']:>18.2f}  "
              f"{r['max_abs_mv']:>16.2f}  {r['mean_mv']:>11.2f}  "
              f"{r['soc_dataset_drop_pct']:>22.2f}  {r['soc_coulomb_drop_pct']:>22.2f}")

    # --- acceptance criteria
    bd = next(r for r in rows if "BD" in r["label"])
    cs10 = next(r for r in rows if "CS 10" in r["label"])
    cs1k = next(r for r in rows if "CS 1000" in r["label"])

    crit_finite = all(r["finite"] for r in rows)
    crit_ratio = cs10["rms_mv"] > 3.0 * bd["rms_mv"]
    crit_neg = cs10["mean_mv"] < -10.0
    crit_ladder = bd["rms_mv"] < cs1k["rms_mv"] < cs10["rms_mv"]

    print()
    print("Acceptance criteria:")
    print(f"  [1] CS_10ohm RMS > 3 * BD RMS               : "
          f"{cs10['rms_mv']:.2f} > 3 * {bd['rms_mv']:.2f}  ->  {crit_ratio}")
    print(f"  [2] CS_10ohm residual mean < -10 mV         : "
          f"{cs10['mean_mv']:.2f} mV  ->  {crit_neg}")
    print(f"  [3] BD < CS_1000ohm < CS_10ohm  (RMS)       : "
          f"{bd['rms_mv']:.1f} < {cs1k['rms_mv']:.1f} < {cs10['rms_mv']:.1f}  ->  {crit_ladder}")
    print(f"  [4] All three runs finite (no NaN/inf)      : {crit_finite}")

    out_png = os.path.join(RESULTS_DIR, "residual_verification.png")
    _verification_figure(rows, out_png)
    print(f"\nFigure saved -> {out_png}")

    if crit_finite and crit_ratio and crit_neg and crit_ladder:
        print("\nRESIDUAL SIGNAL VERIFIED -- proceed to scalograms.")
    else:
        print("\nFIX INCOMPLETE -- diagnostic table above")
        sys.exit(1)


if __name__ == "__main__":
    main()
