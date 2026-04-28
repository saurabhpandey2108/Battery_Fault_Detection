"""
Configuration for the merged Battery Fault Detection pipeline.

Physical / PINN settings are inherited from Battery_Passport with the
following edits (per project spec):
    * PSO_CONFIG removed entirely.
    * BATTERY_CONFIG['parameter_bounds'] removed (the 8 parameters are free).
    * BATTERY_CONFIG['constants']['sampling_time'] set to 1.0 (Tsinghua is 1 Hz).

CWT / CNN settings are inherited from Wavelet_Analysis (config/settings.py).
"""

import os
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
#  Project paths
# ──────────────────────────────────────────────────────────────────────────────
# This file lives at src/config/settings.py -- climb three dirnames to reach
# the project root: settings.py -> src/config/ -> src/ -> project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

NORMAL_ROOT = os.path.join(DATASET_DIR, "NCM811_NORMAL_TEST")
ISC_ROOT = os.path.join(DATASET_DIR, "NCM811_ISC_TEST")


# ──────────────────────────────────────────────────────────────────────────────
#  Battery physics (Battery_Passport, with edits)
# ──────────────────────────────────────────────────────────────────────────────
BATTERY_CONFIG = {
    "target_values": {
        "C1": 38000, "C2": 38000,
        "R0": 0.0082, "R1": 0.0158, "R2": 0.0158,
        "gamma1": 100, "M0": 0.002, "M": 0.05,
    },
    "constants": {
        "sampling_time": 1.0,       # Tsinghua NCM811 data is sampled at 1 Hz
        "capacity": 10 * 3600,      # As. NOTE: not tuned to Tsinghua cell; see train script override
        "efficiency": 1,
        "initial_soc": 100,
    },
    # NCM811 open-circuit voltage curve at 25 C, spanning 2.8 V at 0% SOC
    # to 4.2 V at 100% SOC (matches the Tsinghua dataset we actually have).
    # The upstream Battery_Passport curve peaked at 3.5 V, which is LFP-like
    # chemistry and structurally capped V_pred far below V_meas for NCM811.
    "ocv_curve": {
        "soc_points":     [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
                           70.0, 80.0, 90.0, 95.0, 100.0],
        "voltage_points": [2.80, 3.30, 3.45, 3.55, 3.60, 3.63, 3.68, 3.78,
                           3.90, 4.00, 4.10, 4.15, 4.20],
    },
}


# ──────────────────────────────────────────────────────────────────────────────
#  PINN (14 features → 8 params)
# ──────────────────────────────────────────────────────────────────────────────
NN_CONFIG = {
    "architecture": {
        "input_size": 14,
        "hidden_layers": [64, 64, 64],
        "output_size": 8,
        "activation": "relu",
        "dropout_rate": 0.2,
    },
    "training": {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "validation_split": 0.2,
        "early_stopping_patience": 15,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
#  Feature engineering (Battery_Passport data_loader)
# ──────────────────────────────────────────────────────────────────────────────
DATA_CONFIG = {
    "window_size": 5,   # rolling-feature window for engineer_features
    "random_seed": 42,
}


# ──────────────────────────────────────────────────────────────────────────────
#  CWT / CNN (Wavelet_Analysis)
# ──────────────────────────────────────────────────────────────────────────────
WINDOW_SIZE = 1280
STRIDE = 256
SCALES = np.geomspace(2.0, 300.0, 96)

# CNN input shape: 3 channels = [V_meas scalogram, V_pred scalogram, residual scalogram]
IMAGE_SHAPE = (224, 224, 3)

CNN_CONFIG = {
    "epochs": 60,
    "batch_size": 64,
    "early_stop_patience": 12,   # patience on val_auc (max), see train_classifier.py
    "noise_std": 0.10,           # bumped: pushes harder against per-window memorization
    "learning_rate": 1e-4,
    "l2_lambda": 1e-5,           # mild L2 on conv kernels
    "dropout_rate": 0.5,         # main dropout after Flatten; second at half this rate
    "random_seed": 42,
}


# ──────────────────────────────────────────────────────────────────────────────
#  Phase 1 dataset subset
# ──────────────────────────────────────────────────────────────────────────────
PHASE1_CHARGE_RATE = 0.5
PHASE1_DISCHARGE_MODE = "DST"
OHM_VALUES = [10, 20, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


# ──────────────────────────────────────────────────────────────────────────────
#  Coulomb-counted SOC reference capacity
# ──────────────────────────────────────────────────────────────────────────────
# Derived as the median net discharged Ah across the 14 BD DST 0.5C files.
# Each file goes from ~4.18 V down to ~3.05 V (full discharge); the integrated
# discharge current accumulates 2.42-2.56 Ah depending on file. We use the
# median as Q_rated so SOC(t) = 1.0 + cumtrapz(I_signed, t_seconds) / (Q_RATED_AH
# * 3600) lands at ~0 at end-of-discharge for a healthy cell. The dataset's
# own SOC|DOD/% column is not usable as a global SOC because the Arbin tester
# resets it at every step transition (verified empirically).
Q_RATED_AH = 2.5487


# Number of recent current samples used as the I_window feature inside
# build_leakage_free_features. Spec: 30 samples = 30 s of recent current
# history at the dataset's 1 Hz sampling rate.
PINN_WINDOW_SIZE = 30
# Number of samples used for I_mean_long / I_rms_long rolling stats (~5 min).
PINN_LONG_WINDOW = 300
# Total feature count = 1 (I) + window_size (I_window) + 1 (I_mean_long)
#                       + 1 (I_rms_long) + 1 (SOC_coulomb) + 1 (dI/dt)
PINN_FEATURE_COUNT = 1 + PINN_WINDOW_SIZE + 1 + 1 + 1 + 1   # = 35
PINN_HIDDEN_LAYERS = (64, 64, 64)
PINN_OUTPUT_SIZE = 8


# ──────────────────────────────────────────────────────────────────────────────
#  PINN training hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
# Defaults consumed by src/training/train_pinn_healthy.py. CLI flags can
# override them per-run; this is the single source of truth otherwise.
PINN_EPOCHS = 100
PINN_LEARNING_RATE = 1e-4
PINN_PATIENCE = 12             # early-stop patience on val-RMSE moving average
PINN_CLIPNORM = 1.0            # global gradient-norm clip (per Adam step)
PINN_TRAIN_WINDOW = 256        # samples per Adam update (~0.7 DST cycles at 1 Hz)
PINN_TRAIN_STRIDE = 256        # non-overlapping windows
PINN_VAL_FRACTION = 0.2
PINN_VAL_MA_WINDOW = 8         # window for the moving-average val-RMSE
PINN_FD_EPS_ABS = 1e-6         # finite-difference jacobian step floor
PINN_RANDOM_SEED = 42

# Soft physics-realism penalty. Biases the network away from numerically
# pathological parameter combinations (negative R/C, near-zero RC time
# constants) without hard-clipping the 8 free parameters. Set lambda=0.0
# to reproduce the unpenalized loss exactly.
PINN_PHYS_PENALTY_LAMBDA = 0.01
PINN_PHYS_TAU_MIN = 1e-3       # minimum RC time constant (s) treated as stable
