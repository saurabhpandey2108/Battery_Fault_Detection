"""
Train the PINN on healthy (BD) Tsinghua discharge data only.

This script consumes ONLY leakage-free inputs from
`src.data.prepare_pinn_features.build_leakage_free_features` —
i.e. (I, I_window, I_mean_long, I_rms_long, SOC_coulomb, dI/dt). No
voltage or voltage-derived quantity is ever passed to the network as
input. V_meas appears only on the loss side as the supervision signal.

The 8 physics parameters (C1, C2, R0, R1, R2, gamma1, M0, M) remain
free: no bounds, no sigmoid/tanh, no clipping at the network output
layer or anywhere on the forward path. The only safety measures are
gradient-norm clipping inside Adam (clips the gradient, not the
parameters) and a NaN/inf batch skip.

Training contract per epoch:
  * For each BD file: roll out the full discharge with state
    (ir1, ir2, z, h, s) threaded across timesteps, compute RMSE
    against V_meas, take one Adam step using true dV/dp_k Jacobians
    obtained from central finite differences on calculate_voltage.

Outputs
-------
  * models/pinn_healthy_no_leak.npz  — frozen NN weights + arch sentinel
  * results/train_pinn_v2_history.csv — per-epoch metrics
  * results/pinn_inference_dump.csv   — per-sample inference dump on all
                                         BD files using the final weights
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import pandas as pd

# File lives at src/training/<this>.py -- climb 3 dirnames to reach project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import (
    DATASET_DIR, MODELS_DIR, RESULTS_DIR, BATTERY_CONFIG,
    PHASE1_CHARGE_RATE, PHASE1_DISCHARGE_MODE,
    PINN_FEATURE_COUNT, PINN_HIDDEN_LAYERS, PINN_OUTPUT_SIZE,
    PINN_WINDOW_SIZE, PINN_LONG_WINDOW, Q_RATED_AH,
)
from src.data.tsinghua_loader import load_tsinghua_csv, list_phase1_files
from src.data.prepare_pinn_features import build_leakage_free_features
from src.pinn.battery_physics import BatteryModel
from src.pinn.pinn_v2 import PinnV2


# Architecture: 35 -> 64 -> 64 -> 64 -> 8 per spec.
DEFAULT_ARCH = (PINN_FEATURE_COUNT,) + PINN_HIDDEN_LAYERS + (PINN_OUTPUT_SIZE,)
DEFAULT_LR = 3e-4
DEFAULT_EPOCHS = 30
DEFAULT_PATIENCE = 6
DEFAULT_CLIPNORM = 1.0
DEFAULT_TVALID_WINDOW = 8        # moving-average window for val RMSE
FINITE_DIFF_EPS_ABS = 1e-6


# ----------------------------------------------------------------------
#  Data loading — leakage-free feature pack per file
# ----------------------------------------------------------------------
def load_file_pack(filepath: str) -> dict:
    """Return everything the trainer needs for one BD file.

    The returned dict contains:
        X, v_meas, current, time, soc, offset, n_features  (all aligned)
        path, name
    """
    d = load_tsinghua_csv(filepath)
    pack = build_leakage_free_features(d["discharge_df"])
    pack["path"] = filepath
    pack["name"] = os.path.basename(filepath)
    return pack


def load_all_bd_files(root: str, charge_rate: float, discharge_mode: str) -> List[dict]:
    paths = list_phase1_files(root, charge_rate=charge_rate,
                              discharge_mode=discharge_mode, class_="BD")
    if not paths:
        raise RuntimeError(
            f"No BD files found under {root} for {charge_rate}C / {discharge_mode}"
        )
    files = []
    for fp in paths:
        pack = load_file_pack(fp)
        files.append(pack)
        print(f"  loaded {pack['name']:<40s} -> N={len(pack['v_meas'])}  "
              f"SOC[{pack['soc'][0]:.3f} .. {pack['soc'][-1]:.3f}]")
    return files


# ----------------------------------------------------------------------
#  Forward rollout with threaded state + finite-diff Jacobians
# ----------------------------------------------------------------------
def rollout(model: BatteryModel,
            params_seq: np.ndarray,
            current_seq: np.ndarray,
            v_meas: np.ndarray,
            initial_soc: float,
            compute_jacobian: bool = False
            ) -> Tuple[np.ndarray, np.ndarray, int]:
    N = len(current_seq)
    v_pred = np.zeros(N, dtype=np.float64)
    jac = np.zeros((N, 8), dtype=np.float64) if compute_jacobian else None

    ir1, ir2, h, s = 0.0, 0.0, 0.0, 0.0
    z = float(initial_soc)
    n_resets = 0

    for t in range(N):
        p = params_seq[t]
        i = current_seq[t]

        v, ir1_n, ir2_n, z_n, h_n, s_n = model.calculate_voltage(
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], i,
            ir1=ir1, ir2=ir2, z=z, h=h, s=s,
        )

        if (not np.isfinite(v)) or (not np.isfinite(ir1_n)) or \
           (not np.isfinite(ir2_n)) or abs(v) > 10.0:
            v = float(v_meas[t])
            ir1_n, ir2_n = 0.0, 0.0
            n_resets += 1

        v_pred[t] = v

        if compute_jacobian:
            for k in range(8):
                step = FINITE_DIFF_EPS_ABS + 1e-4 * abs(p[k])
                p_plus = p.copy(); p_plus[k] += step
                p_minus = p.copy(); p_minus[k] -= step
                v_plus, *_ = model.calculate_voltage(
                    p_plus[0], p_plus[1], p_plus[2], p_plus[3],
                    p_plus[4], p_plus[5], p_plus[6], p_plus[7], i,
                    ir1=ir1, ir2=ir2, z=z, h=h, s=s,
                )
                v_minus, *_ = model.calculate_voltage(
                    p_minus[0], p_minus[1], p_minus[2], p_minus[3],
                    p_minus[4], p_minus[5], p_minus[6], p_minus[7], i,
                    ir1=ir1, ir2=ir2, z=z, h=h, s=s,
                )
                if np.isfinite(v_plus) and np.isfinite(v_minus):
                    jac[t, k] = (v_plus - v_minus) / (2.0 * step)
                else:
                    jac[t, k] = 0.0

        ir1, ir2, z, h, s = ir1_n, ir2_n, z_n, h_n, s_n

    return v_pred, jac, n_resets


# ----------------------------------------------------------------------
#  Training step (one file -> one rollout -> one Adam update)
# ----------------------------------------------------------------------
def file_level_train(nn: PinnV2,
                     model: BatteryModel,
                     pack: dict,
                     lr: float,
                     clipnorm: float,
                     compute_update: bool = True
                     ) -> Tuple[float, np.ndarray, int]:
    params_seq = nn.forward(pack["X"], training=compute_update)   # (N, 8)
    v_pred, jac, n_resets = rollout(
        model, params_seq, pack["current"], pack["v_meas"],
        initial_soc=BATTERY_CONFIG["constants"]["initial_soc"],
        compute_jacobian=compute_update,
    )

    err = v_pred - pack["v_meas"]
    if not np.all(np.isfinite(err)):
        return float("nan"), params_seq, n_resets
    rmse = float(np.sqrt(np.mean(err ** 2)))

    if compute_update:
        if not np.all(np.isfinite(jac)):
            return rmse, params_seq, n_resets
        N = len(err)
        d_out = (2.0 / N) * err[:, None] * jac
        nn.backward(d_out)
        nn.optimize(lr=lr, clipnorm=clipnorm)

    return rmse, params_seq, n_resets


def evaluate(nn: PinnV2, model: BatteryModel, files: List[dict]) -> float:
    rmses = []
    for f in files:
        r, _, _ = file_level_train(nn, model, f, lr=0.0, clipnorm=0.0,
                                   compute_update=False)
        if np.isfinite(r):
            rmses.append(r)
    if not rmses:
        return float("inf")
    return float(np.mean(rmses))


# ----------------------------------------------------------------------
#  Post-training per-sample dump
# ----------------------------------------------------------------------
def dump_inference_csv(nn: PinnV2, model: BatteryModel,
                       files: List[dict], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cols = ["file", "time_s", "current_A", "soc_coulomb", "V_meas", "V_pred",
            "C1", "C2", "R0", "R1", "R2", "gamma1", "M0", "M"]
    rows = []
    for f in files:
        params_seq = nn.forward(f["X"], training=False)
        v_pred, _, _ = rollout(
            model, params_seq, f["current"], f["v_meas"],
            initial_soc=BATTERY_CONFIG["constants"]["initial_soc"],
            compute_jacobian=False,
        )
        for t in range(len(f["v_meas"])):
            rows.append([
                f["name"], float(f["time"][t]), float(f["current"][t]),
                float(f["soc"][t]), float(f["v_meas"][t]), float(v_pred[t]),
                *[float(p) for p in params_seq[t]],
            ])
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows(rows)
    print(f"[train_pinn_healthy] per-sample dump saved -> {out_path}  ({len(rows)} rows)")


# ----------------------------------------------------------------------
#  CLI entry
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train PINN on healthy BD DST data with leakage-free inputs")
    parser.add_argument("--dataset-root", default=DATASET_DIR)
    parser.add_argument("--charge-rate", type=float, default=PHASE1_CHARGE_RATE)
    parser.add_argument("--discharge-mode", default=PHASE1_DISCHARGE_MODE)
    parser.add_argument("--output",
                        default=os.path.join(MODELS_DIR, "pinn_healthy_no_leak.npz"))
    parser.add_argument("--history-csv",
                        default=os.path.join(RESULTS_DIR, "train_pinn_v2_history.csv"))
    parser.add_argument("--dump-csv",
                        default=os.path.join(RESULTS_DIR, "pinn_inference_dump.csv"))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                        help="Early-stop patience measured on the MOVING AVERAGE "
                             "of val RMSE, ignoring spike epochs.")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--clipnorm", type=float, default=DEFAULT_CLIPNORM)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-dump", action="store_true",
                        help="Skip the final per-sample CSV dump")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="(ignored — file-level training)")
    args = parser.parse_args()
    if args.batch_size is not None:
        print("[train_pinn_healthy] note: --batch-size is ignored (file-level training)")

    np.random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.history_csv), exist_ok=True)

    print(f"[train_pinn_healthy] Phase 1 subset: {args.charge_rate}C / "
          f"{args.discharge_mode}, BD only")
    print(f"[train_pinn_healthy] feature spec: window={PINN_WINDOW_SIZE}, "
          f"long_window={PINN_LONG_WINDOW}, Q_rated={Q_RATED_AH:.4f} Ah")
    print(f"[train_pinn_healthy] inputs: I, I_window({PINN_WINDOW_SIZE}), "
          f"I_mean_long, I_rms_long, SOC_coulomb, dI/dt  -> {PINN_FEATURE_COUNT} features")
    print(f"[train_pinn_healthy] inputs DO NOT include voltage of any kind")

    files_all = load_all_bd_files(args.dataset_root, args.charge_rate,
                                  args.discharge_mode)
    n_val = max(1, int(round(args.val_fraction * len(files_all))))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(files_all))
    val_files = [files_all[i] for i in perm[:n_val]]
    train_files = [files_all[i] for i in perm[n_val:]]
    print(f"[train_pinn_healthy] files -> train {len(train_files)}  val {len(val_files)}")

    sampling_time = BATTERY_CONFIG["constants"]["sampling_time"]
    capacity_as = BATTERY_CONFIG["constants"]["capacity"]
    ocv = BATTERY_CONFIG["ocv_curve"]
    model = BatteryModel(
        sampling_time=sampling_time,
        capacity_as=capacity_as,
        ocv_soc=ocv["soc_points"],
        ocv_voltage=ocv["voltage_points"],
    )

    nn = PinnV2(layer_sizes=DEFAULT_ARCH, dropout_rate=0.0, seed=args.seed)
    tv = np.array(list(BATTERY_CONFIG["target_values"].values()), dtype=np.float64)
    nn.warm_start_output_bias(tv)
    print(f"[train_pinn_healthy] architecture: {DEFAULT_ARCH}, "
          f"warm-start biases = target_values")

    hist_cols = ["epoch", "train_rmse", "val_rmse", "val_rmse_movavg",
                 "n_state_resets",
                 "C1_mean", "C2_mean", "R0_mean", "R1_mean",
                 "R2_mean", "gamma1_mean", "M0_mean", "M_mean",
                 "time_s"]
    with open(args.history_csv, "w", newline="") as fh:
        csv.writer(fh).writerow(hist_cols)

    best_val = float("inf")
    best_val_movavg = float("inf")
    best_params = None
    patience_counter = 0
    val_history: List[float] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        rng.shuffle(train_files)

        train_rmses, resets_epoch = [], 0
        for f in train_files:
            rmse, _, n_resets = file_level_train(
                nn, model, f, lr=args.learning_rate, clipnorm=args.clipnorm,
                compute_update=True,
            )
            resets_epoch += n_resets
            if np.isfinite(rmse):
                train_rmses.append(rmse)

        train_rmse = float(np.mean(train_rmses)) if train_rmses else float("nan")
        val_rmse = evaluate(nn, model, val_files)
        val_history.append(val_rmse)

        window = val_history[-DEFAULT_TVALID_WINDOW:]
        finite_window = [x for x in window if np.isfinite(x)]
        val_movavg = float(np.mean(finite_window)) if finite_window else float("inf")

        last_params = nn.forward(train_files[0]["X"], training=False)
        pm = last_params.mean(axis=0)

        epoch_time = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs} - {epoch_time:.1f}s - "
              f"train RMSE {train_rmse:.4f}  val RMSE {val_rmse:.4f}  "
              f"val-MA {val_movavg:.4f}  resets {resets_epoch}")

        with open(args.history_csv, "a", newline="") as fh:
            csv.writer(fh).writerow([
                epoch, train_rmse, val_rmse, val_movavg, resets_epoch,
                *[float(x) for x in pm], epoch_time,
            ])

        if np.isfinite(val_movavg) and val_movavg < best_val_movavg - 1e-6:
            best_val_movavg = val_movavg
            best_val = val_rmse
            best_params = {k: v.copy() for k, v in nn.params.items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"[train_pinn_healthy] early stopping — moving-avg val RMSE "
                  f"has not improved in {args.patience} epochs")
            break

    if best_params is not None:
        nn.params = best_params
        print(f"[train_pinn_healthy] restored best-MA weights "
              f"(val MA {best_val_movavg:.4f}, val spot {best_val:.4f})")

    nn.save(args.output)

    # Sanity: catch leakage if it ever sneaks back in. With voltage in the
    # feature vector, an MLP can drive RMSE to <5 mV by inverting physics.
    if best_val < 0.005:
        print(f"[train_pinn_healthy] WARNING: val RMSE {best_val:.4f} V "
              f"is suspiciously low. Confirm voltage is NOT in the feature "
              f"vector before trusting these weights.")

    if not args.skip_dump:
        dump_inference_csv(nn, model, files_all, args.dump_csv)


if __name__ == "__main__":
    main()
