"""
Frozen-PINN inference: emit V_pred(t) aligned with V_meas(t).

Inputs to the network are exclusively those produced by
`build_leakage_free_features` (current and current-derived statistics
plus a coulomb-counted SOC). V_meas is NEVER an input — it appears only
as the supervision signal during training and as the comparison target
at inference time, never inside the feature vector.

State threading (ir1, ir2, z, h, s) flows across timesteps within a
single file and is reset only at the start of a new file. A numerical
safety net (NOT a parameter bound) substitutes V_meas and resets ir1,
ir2 when the rollout produces |V|>10 V or non-finite values; this
prevents a single unstable timestep from poisoning the rest of the
file. The 8 physics parameters themselves remain unconstrained.

Output arrays are shorter than the raw CSV by `offset` (= window_size)
samples because the leakage-free feature builder drops the leading rows
whose I_window is incomplete. `time`, `current`, `v_meas`, `v_pred` are
all returned aligned to that trimmed range.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.config import (
    BATTERY_CONFIG, PINN_FEATURE_COUNT, PINN_HIDDEN_LAYERS, PINN_OUTPUT_SIZE,
)
from src.data.prepare_pinn_features import build_leakage_free_features
from src.pinn.battery_physics import BatteryModel
from src.pinn.pinn_v2 import PinnV2


DEFAULT_ARCH = (PINN_FEATURE_COUNT,) + PINN_HIDDEN_LAYERS + (PINN_OUTPUT_SIZE,)


def _build_nn_from_file(path: str) -> PinnV2:
    data = np.load(path, allow_pickle=True)
    if "__arch__" not in data.files:
        raise RuntimeError(
            f"{path} does not carry a v2 '__arch__' sentinel. Retrain the "
            "PINN with the new training script: python main.py train-pinn"
        )
    arch = tuple(int(s) for s in data["__arch__"])
    nn = PinnV2(layer_sizes=arch)
    nn.load(path)
    return nn


def load_pinn(weights_path: str) -> PinnV2:
    return _build_nn_from_file(weights_path)


def predict_voltage_series(pinn_or_path,
                           discharge_df: pd.DataFrame,
                           sampling_time: float | None = None,
                           capacity_as: float | None = None,
                           initial_soc: float | None = None) -> dict:
    """Run the frozen PINN on one Tsinghua discharge DataFrame.

    Parameters
    ----------
    pinn_or_path : str | PinnV2
        Path to .npz weights, or a pre-loaded PinnV2 instance.
    discharge_df : pd.DataFrame
        Discharge-phase frame with columns time_s, current_A, voltage_V
        (from `tsinghua_loader.load_tsinghua_csv`).

    Returns
    -------
    dict with keys time, v_meas, current, v_pred, params, residual,
    soc, offset. All time-aligned arrays have length len(df) - offset.
    """
    if isinstance(pinn_or_path, PinnV2):
        nn = pinn_or_path
    else:
        nn = _build_nn_from_file(pinn_or_path)

    pack = build_leakage_free_features(discharge_df)

    X = pack["X"]
    v_meas = pack["v_meas"]
    current = pack["current"]
    t = pack["time"]

    params_seq = nn.forward(X, training=False)            # (N', 8)

    ts = BATTERY_CONFIG["constants"]["sampling_time"] if sampling_time is None else sampling_time
    cap = BATTERY_CONFIG["constants"]["capacity"] if capacity_as is None else capacity_as
    soc0 = BATTERY_CONFIG["constants"]["initial_soc"] if initial_soc is None else initial_soc

    ocv = BATTERY_CONFIG["ocv_curve"]
    model = BatteryModel(
        sampling_time=ts, capacity_as=cap,
        ocv_soc=ocv["soc_points"], ocv_voltage=ocv["voltage_points"],
    )

    ir1, ir2, h, s = 0.0, 0.0, 0.0, 0.0
    z = float(soc0)
    v_pred = np.zeros(len(v_meas), dtype=np.float64)

    n_state_resets = 0
    for idx in range(len(v_meas)):
        p = params_seq[idx]
        i = current[idx]
        v, ir1_n, ir2_n, z_n, h_n, s_n = model.calculate_voltage(
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], i,
            ir1=ir1, ir2=ir2, z=z, h=h, s=s,
        )
        unstable = (not np.isfinite(v)) or (not np.isfinite(ir1_n)) or \
                   (not np.isfinite(ir2_n)) or (abs(v) > 10.0)
        if unstable:
            ir1_n, ir2_n = 0.0, 0.0
            v = float(v_meas[idx])
            n_state_resets += 1
        v_pred[idx] = v
        ir1, ir2, z, h, s = ir1_n, ir2_n, z_n, h_n, s_n

    if n_state_resets:
        print(f"[infer_pinn] numerical-state resets: {n_state_resets} / {len(v_meas)} steps")

    return {
        "time": t,
        "v_meas": v_meas,
        "current": current,
        "v_pred": v_pred,
        "params": np.asarray(params_seq),
        "residual": v_meas - v_pred,
        "soc": pack["soc"],
        "offset": pack["offset"],
    }
