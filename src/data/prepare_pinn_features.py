"""
Bridge between Tsinghua loader output and the Battery_Passport
BatteryDataProcessor feature contract.

The PINN expects a DataFrame with columns
    Time, Current_sense, Terminal_voltage
and produces 14 engineered features via `engineer_features(window_size=5)`:
    Time, Current_sense, Terminal_voltage,
    Avg_Current, Avg_Voltage, Max_Current, Min_Current, Max_Voltage, Min_Voltage,
    Peak_Current, Peak_Voltage, RMS_Current, RMS_Voltage, dV/dt, dI/dt.

Note: engineer_features leaves 'Time' in the frame and returns 15 columns
(Time + 14 features). The DNN feed_forward consumes the 14 non-Time columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..pinn.data_loader import BatteryDataProcessor


REQUIRED_COLS = ["Time", "Current_sense", "Terminal_voltage"]

# Fixed order of the 14 features the PINN expects.
FEATURE_ORDER = [
    "Current_sense", "Terminal_voltage",
    "Avg_Current", "Avg_Voltage",
    "Max_Current", "Min_Current", "Max_Voltage", "Min_Voltage",
    "Peak_Current", "Peak_Voltage",
    "RMS_Current", "RMS_Voltage",
    "dV/dt", "dI/dt",
]


def tsinghua_to_pinn_frame(discharge_df: pd.DataFrame) -> pd.DataFrame:
    """Rename Tsinghua columns to the Battery_Passport schema."""
    if not {"time_s", "current_A", "voltage_V"}.issubset(discharge_df.columns):
        raise ValueError(
            "discharge_df must contain time_s, current_A, voltage_V; "
            f"got {list(discharge_df.columns)}"
        )
    return pd.DataFrame({
        "Time": discharge_df["time_s"].to_numpy(dtype=float),
        "Current_sense": discharge_df["current_A"].to_numpy(dtype=float),
        "Terminal_voltage": discharge_df["voltage_V"].to_numpy(dtype=float),
    })


def engineer_pinn_features(discharge_df: pd.DataFrame,
                           processor: BatteryDataProcessor | None = None,
                           window_size: int = 5) -> pd.DataFrame:
    """Run BatteryDataProcessor.engineer_features on Tsinghua discharge data."""
    if processor is None:
        processor = BatteryDataProcessor(window_size=window_size)
    pinn_df = tsinghua_to_pinn_frame(discharge_df)
    return processor.engineer_features(pinn_df)


def to_feature_matrix(engineered_df: pd.DataFrame) -> np.ndarray:
    """Return an (N, 14) float64 array in the exact feature order the PINN expects."""
    missing = [c for c in FEATURE_ORDER if c not in engineered_df.columns]
    if missing:
        raise ValueError(f"Engineered df is missing columns: {missing}")
    return engineered_df[FEATURE_ORDER].to_numpy(dtype=np.float64)


def to_training_dataset(engineered_df: pd.DataFrame) -> np.ndarray:
    """Prepare the (N, 15) array the DNN's `train` loop reads.

    The upstream training code slices:
        x                 = row[1:]                 # 14 features (positions 1..14)
        actual_voltages   = row[2]                  # Terminal_voltage
    With column order [Time, Current_sense, Terminal_voltage, <12 engineered>]
    that contract is preserved exactly.
    """
    cols = ["Time"] + FEATURE_ORDER
    missing = [c for c in cols if c not in engineered_df.columns]
    if missing:
        raise ValueError(f"Engineered df is missing columns: {missing}")
    return engineered_df[cols].to_numpy(dtype=np.float64)
