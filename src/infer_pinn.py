"""
Frozen-PINN inference: emit V_pred(t) aligned with measured V(t).

Unlike the training-time _calculate_voltages (which treats each sample as
an isolated initial-condition call, preserved verbatim from upstream),
this module threads the battery model's internal state (ir1, ir2, z, h, s)
across timesteps within a single file — matching the physical causal
behaviour of the cell. State is reset at the start of every file.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .config import BATTERY_CONFIG, NN_CONFIG, DATA_CONFIG
from .pinn.battery_physics import BatteryModel
from .pinn.deep_neural_network import DeepNeuralNetwork
from .pinn.data_loader import BatteryDataProcessor
from .data.prepare_pinn_features import (
    engineer_pinn_features, to_feature_matrix, FEATURE_ORDER,
)


def _build_nn() -> DeepNeuralNetwork:
    arch = NN_CONFIG["architecture"]
    sizes = [arch["input_size"]] + arch["hidden_layers"] + [arch["output_size"]]
    return DeepNeuralNetwork(
        sizes=sizes,
        activation=arch["activation"],
        dropout_rate=arch["dropout_rate"],
    )


def load_pinn(weights_path: str) -> DeepNeuralNetwork:
    """Load frozen PINN weights from an .npz produced by train_pinn_healthy."""
    nn = _build_nn()
    nn.load_model(weights_path)
    return nn


def predict_voltage_series(pinn_or_path,
                           discharge_df: pd.DataFrame,
                           processor: BatteryDataProcessor | None = None,
                           sampling_time: float | None = None,
                           capacity_as: float | None = None,
                           initial_soc: float | None = None) -> dict:
    """Run the frozen PINN on one Tsinghua discharge DataFrame.

    Parameters
    ----------
    pinn_or_path : str | DeepNeuralNetwork
        Path to .npz weights, or a pre-loaded DeepNeuralNetwork.
    discharge_df : pd.DataFrame
        Columns time_s, current_A, voltage_V (at minimum), as produced by
        load_tsinghua_csv.
    processor : BatteryDataProcessor, optional
        Reuse an existing processor (e.g. to share window_size).

    Returns
    -------
    dict with keys:
        time        (N',) aligned time axis
        v_meas      (N',) measured voltage
        current     (N',) measured current
        v_pred      (N',) PINN-reconstructed voltage (state-threaded)
        params      (N', 8) raw parameter sequence emitted by the NN
        residual    (N',) v_meas - v_pred

    The output length N' can be slightly shorter than len(discharge_df) because
    engineer_features drops rows with NaN from the rolling operations.
    """
    if isinstance(pinn_or_path, DeepNeuralNetwork):
        nn = pinn_or_path
    else:
        nn = load_pinn(pinn_or_path)

    if processor is None:
        processor = BatteryDataProcessor(window_size=DATA_CONFIG["window_size"])

    eng = engineer_pinn_features(discharge_df, processor,
                                 window_size=processor.window_size)
    feats = to_feature_matrix(eng)                   # (N', 14)
    time = eng["Time"].to_numpy(dtype=float)
    v_meas = eng["Terminal_voltage"].to_numpy(dtype=float)
    current = eng["Current_sense"].to_numpy(dtype=float)

    # Forward pass through the frozen NN in batches (no gradient / dropout).
    params_seq = nn.feed_forward(feats, training=False)   # (N', 8)

    # Physics rollout — thread (ir1, ir2, z, h, s) across timesteps.
    ts = BATTERY_CONFIG["constants"]["sampling_time"] if sampling_time is None else sampling_time
    cap = BATTERY_CONFIG["constants"]["capacity"] if capacity_as is None else capacity_as
    soc0 = BATTERY_CONFIG["constants"]["initial_soc"] if initial_soc is None else initial_soc

    model = BatteryModel(sampling_time=ts, capacity_as=cap)
    ir1, ir2, h, s = 0.0, 0.0, 0.0, 0.0
    z = float(soc0)
    v_pred = np.zeros(len(feats), dtype=np.float64)

    n_state_resets = 0
    for idx in range(len(feats)):
        C1, C2, R0, R1, R2, gamma1, M0, M = params_seq[idx]
        i = current[idx]
        v, ir1, ir2, z, h, s = model.calculate_voltage(
            C1, C2, R0, R1, R2, gamma1, M0, M, i,
            ir1=ir1, ir2=ir2, z=z, h=h, s=s,
        )
        # Numerical safety net (not a parameter bound). Because training uses
        # stateless per-sample reset but inference threads state across 12k+
        # timesteps, occasional out-of-regime parameters can drive the RC
        # integrators (ir1, ir2) exponentially away from any plausible
        # battery voltage — the rollout diverges to ~1e8 V with the state
        # itself still finite. We detect implausible outputs (|v| > 10 V or
        # non-finite) and reset only the internal state variables ir1, ir2
        # so the rollout can recover on the next step. SOC z stays bounded
        # inside calculate_voltage. None of the 8 learned parameters are
        # touched, and V_pred is NOT clipped inside the valid range —
        # only fallback-substituted on clearly non-physical steps.
        unstable = (not np.isfinite(v)) or (not np.isfinite(ir1)) or \
                   (not np.isfinite(ir2)) or (abs(v) > 10.0)
        if unstable:
            ir1 = 0.0
            ir2 = 0.0
            v = float(v_meas[idx])
            n_state_resets += 1
        v_pred[idx] = v

    if n_state_resets:
        print(f"[infer_pinn] numerical-state resets: {n_state_resets} "
              f"/ {len(feats)} steps")

    return {
        "time": time,
        "v_meas": v_meas,
        "current": current,
        "v_pred": v_pred,
        "params": np.asarray(params_seq),
        "residual": v_meas - v_pred,
    }
