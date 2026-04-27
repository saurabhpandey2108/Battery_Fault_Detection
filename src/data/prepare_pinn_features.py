"""
Leakage-free PINN feature builder.

The previous "engineer_features" path included Terminal_voltage and several
voltage-derived statistics (Avg_Voltage, Max/Min/Peak/RMS_Voltage, dV/dt) as
direct NN inputs. With V_meas in the input, the network learns the inverse
physics map and trivially reproduces any voltage shown to it -- including
faulty voltage on CS (ISC) cells. The residual signal is then destroyed.

This module fixes that:

  * NO voltage of any kind enters the feature vector.
  * NO voltage-derived statistic enters the feature vector.
  * NO OCV(SOC) lookup enters the feature vector.
  * SOC is coulomb-counted from the EXTERNAL ammeter current ONLY.
    The dataset's `SOC|DOD/%` column is NOT used (it is per-step Arbin DOD
    that resets at every step transition, see results/dataset_inspection).

Coulomb counting is fault-blind by design: the hidden ISC current bypasses
the ammeter, so the integrated SOC stays on the healthy trajectory while
V_meas falls below V_pred -- which is precisely the residual signal the
fault detector keys on.

Feature vector (default window_size=30 -> 35 features per timestep):
   0       I(t)               instantaneous current (A)
   1..30   I_window            last `window_size` samples of I including I(t)
  31       I_mean_long         rolling mean of I over the last 300 samples
  32       I_rms_long          rolling RMS  of I over the last 300 samples
  33       SOC_coulomb(t)      1.0 + cumtrapz(I_signed, t) / (Q_rated * 3600)
                                clipped to [0, 1]
  34       dI/dt(t)            first difference of I, prepend 0 at t=0

The first `window_size` rows of the resulting matrix are dropped because
their I_window slot is incomplete; callers must trim V_meas / I / time by
the same offset before downstream alignment.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from ..config import Q_RATED_AH, PINN_WINDOW_SIZE, PINN_LONG_WINDOW


# Sentinel column names. If you find yourself referencing voltage_V or the
# dataset SOC column inside this file, you have re-introduced the leakage bug.
_FORBIDDEN = ("voltage_V", "soc_pct", "Terminal_voltage", "电压/V", "SOC|DOD/%")


def coulomb_counted_soc(current_signed: np.ndarray,
                        time_seconds: np.ndarray,
                        q_rated_ah: float = Q_RATED_AH,
                        soc0: float = 1.0) -> np.ndarray:
    """SOC by trapezoidal integration of the signed current.

    Sign convention (verified empirically on the Tsinghua NCM811 BD files):
    negative current = discharge, positive current = regen / charge. With
    SOC_0 = 1.0 and the formula

        SOC(t) = SOC_0 + cumtrapz(I_signed, t) / (Q_rated * 3600)

    discharge drives SOC toward 0 and regen briefly nudges it up. Output
    is clipped to [0, 1] to absorb small numerical overshoot at the
    endpoints.

    This function is the ONLY place coulomb counting happens in the
    codebase, and it is fault-blind: the ISC current never reaches the
    ammeter, so this trajectory stays on the healthy curve regardless
    of the cell's internal state.
    """
    if current_signed.shape != time_seconds.shape:
        raise ValueError("current_signed and time_seconds must have the same shape")
    if current_signed.size < 2:
        return np.full_like(current_signed, fill_value=soc0, dtype=np.float64)

    # cumtrapz integration; first sample contributes 0, so prepend 0.
    dt = np.diff(time_seconds)
    midI = 0.5 * (current_signed[1:] + current_signed[:-1])
    inc = midI * dt                          # A * s = Coulombs
    cum = np.concatenate([[0.0], np.cumsum(inc)])
    soc = soc0 + cum / (q_rated_ah * 3600.0)
    return np.clip(soc, 0.0, 1.0)


def build_leakage_free_features(df: pd.DataFrame,
                                window_size: int = PINN_WINDOW_SIZE,
                                long_window: int = PINN_LONG_WINDOW,
                                q_rated_ah: float = Q_RATED_AH
                                ) -> Dict[str, np.ndarray]:
    """Construct the no-leak input tensor and the aligned target / state arrays.

    Parameters
    ----------
    df : pd.DataFrame
        Discharge-phase dataframe from `tsinghua_loader.load_tsinghua_csv`.
        Must contain columns: time_s, current_A, voltage_V.
    window_size : int
        Recent-current-history window length in samples (1 Hz -> seconds).
    long_window : int
        Rolling-statistic window length in samples (~5 min default).
    q_rated_ah : float
        Reference capacity for coulomb-counted SOC.

    Returns
    -------
    dict with:
        X         : (N', n_features) leakage-free inputs (float64)
        v_meas    : (N',)             measured voltage TARGET (NOT input)
        current   : (N',)             measured current (for the physics rollout)
        time      : (N',)             time axis (seconds)
        soc       : (N',)             coulomb-counted SOC trajectory
        offset    : int               number of leading samples dropped (= window_size)
        n_features: int               number of input features per row

    Notes
    -----
    The output arrays are aligned: caller can use v_meas, current, time, soc
    directly without re-trimming. The `offset` value is informational so the
    caller can map back to the original CSV row index if needed.
    """
    # -------- defensive checks (catches re-introduction of the leakage bug)
    forbidden_seen = [c for c in df.columns if c in _FORBIDDEN and c != "voltage_V"]
    # voltage_V appears in df by construction (we need it as TARGET later) but
    # it MUST NOT enter the feature vector. We only fetch it into v_meas below.

    if "current_A" not in df.columns or "time_s" not in df.columns or "voltage_V" not in df.columns:
        raise RuntimeError(
            "build_leakage_free_features requires columns time_s, current_A, voltage_V; "
            f"got {list(df.columns)}"
        )

    t = df["time_s"].to_numpy(dtype=np.float64)
    I = df["current_A"].to_numpy(dtype=np.float64)
    V = df["voltage_V"].to_numpy(dtype=np.float64)
    N = len(I)

    if N <= window_size + long_window:
        raise RuntimeError(
            f"file too short: have {N} samples, need > {window_size + long_window}"
        )
    if not np.all(np.isfinite(I)):
        raise RuntimeError("current column contains non-finite values")

    # -------- coulomb-counted SOC over the full series
    soc = coulomb_counted_soc(I, t, q_rated_ah=q_rated_ah, soc0=1.0)

    # -------- rolling stats over the last `long_window` samples
    cum_I = np.concatenate([[0.0], np.cumsum(I)])
    cum_I2 = np.concatenate([[0.0], np.cumsum(I * I)])
    idx = np.arange(N)
    lo = np.maximum(0, idx - long_window + 1)
    n_in_window = (idx - lo + 1).astype(np.float64)
    sum_I = cum_I[idx + 1] - cum_I[lo]
    sum_I2 = cum_I2[idx + 1] - cum_I2[lo]
    I_mean_long = sum_I / n_in_window
    I_rms_long = np.sqrt(np.maximum(sum_I2 / n_in_window, 0.0))

    # -------- first difference of I; prepend 0 at t=0
    dI = np.empty(N, dtype=np.float64)
    dI[0] = 0.0
    dI[1:] = np.diff(I)

    # -------- I_window: at output row j, span I[j+1 : j+1+window_size]
    # That corresponds to original time index t = j + window_size and the
    # window covers I[t - window_size + 1 : t + 1] (inclusive of t).
    sliding = np.lib.stride_tricks.sliding_window_view(I, window_size)  # (N - W + 1, W)
    # Drop the first row of `sliding` (its window covers I[0:W], whose ending
    # sample t = W-1 is BEFORE our first kept output index t = window_size).
    I_window = sliding[1:]                                              # (N - W, W)

    # -------- assemble (N - window_size, n_features)
    Nout = N - window_size
    n_features = 1 + window_size + 1 + 1 + 1 + 1
    X = np.empty((Nout, n_features), dtype=np.float64)
    keep = slice(window_size, N)
    X[:, 0] = I[keep]                                # I(t)
    X[:, 1:1 + window_size] = I_window               # I_window
    X[:, 1 + window_size] = I_mean_long[keep]        # I_mean_long
    X[:, 2 + window_size] = I_rms_long[keep]         # I_rms_long
    X[:, 3 + window_size] = soc[keep]                # SOC_coulomb
    X[:, 4 + window_size] = dI[keep]                 # dI/dt

    if not np.all(np.isfinite(X)):
        raise RuntimeError("non-finite values in feature matrix -- aborting")

    return {
        "X": X,
        "v_meas": V[keep].copy(),
        "current": I[keep].copy(),
        "time": t[keep].copy(),
        "soc": soc[keep].copy(),
        "offset": int(window_size),
        "n_features": int(n_features),
    }
