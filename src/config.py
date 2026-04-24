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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    "epochs": 30,
    "batch_size": 64,
    "early_stop_patience": 8,
    "noise_std": 0.05,
    "random_seed": 42,
}


# ──────────────────────────────────────────────────────────────────────────────
#  Phase 1 dataset subset
# ──────────────────────────────────────────────────────────────────────────────
PHASE1_CHARGE_RATE = 0.5
PHASE1_DISCHARGE_MODE = "DST"
OHM_VALUES = [10, 20, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
