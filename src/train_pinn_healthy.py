"""
Train the PINN on healthy (BD) Tsinghua discharge data only.

This mirrors Battery_Passport/train.py with two removals per spec:
    * No PSO (optimization/ package is not ported).
    * No parameter bounds on the 8 electrochemical outputs.

Phase 1: trains on the 14 BD files at charge_rate=0.5, discharge_mode=DST.
The trained NumPy weights are saved to models/pinn_healthy.npz so that
infer_pinn.predict_voltage_series can freeze them for downstream stages.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Make package imports work regardless of cwd when invoked as a script.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import (
    DATASET_DIR, MODELS_DIR, NN_CONFIG, DATA_CONFIG, BATTERY_CONFIG,
    PHASE1_CHARGE_RATE, PHASE1_DISCHARGE_MODE,
)
from src.data.tsinghua_loader import load_tsinghua_csv, list_phase1_files
from src.data.prepare_pinn_features import (
    engineer_pinn_features, to_training_dataset,
)
from src.pinn.data_loader import BatteryDataProcessor
from src.pinn.deep_neural_network import DeepNeuralNetwork


def build_healthy_dataset(root: str,
                          charge_rate: float,
                          discharge_mode: str,
                          window_size: int,
                          verbose: bool = True) -> np.ndarray:
    """Assemble one big (N, 15) array by concatenating engineered features
    from every BD discharge file in the Phase 1 subset."""
    processor = BatteryDataProcessor(window_size=window_size)
    files = list_phase1_files(root, charge_rate=charge_rate,
                              discharge_mode=discharge_mode, class_="BD")
    if not files:
        raise RuntimeError(
            f"No BD files found under {root} for {charge_rate}C / {discharge_mode}"
        )

    parts = []
    for fp in files:
        d = load_tsinghua_csv(fp)
        eng = engineer_pinn_features(d["discharge_df"], processor, window_size=window_size)
        arr = to_training_dataset(eng)
        parts.append(arr)
        if verbose:
            print(f"  {os.path.basename(fp)} -> {arr.shape}")
    dataset = np.vstack(parts)
    if verbose:
        print(f"Total training array: {dataset.shape}")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Train healthy-only PINN on Tsinghua DST data")
    parser.add_argument("--dataset-root", default=DATASET_DIR)
    parser.add_argument("--charge-rate", type=float, default=PHASE1_CHARGE_RATE)
    parser.add_argument("--discharge-mode", default=PHASE1_DISCHARGE_MODE)
    parser.add_argument("--output", default=os.path.join(MODELS_DIR, "pinn_healthy.npz"))
    parser.add_argument("--epochs", type=int, default=NN_CONFIG["training"]["epochs"])
    parser.add_argument("--batch-size", type=int, default=NN_CONFIG["training"]["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=NN_CONFIG["training"]["learning_rate"])
    parser.add_argument("--patience", type=int, default=NN_CONFIG["training"]["early_stopping_patience"])
    parser.add_argument("--seed", type=int, default=DATA_CONFIG["random_seed"])
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"[train_pinn_healthy] Phase 1 subset: {args.charge_rate}C / {args.discharge_mode}, BD only")
    dataset = build_healthy_dataset(
        root=args.dataset_root,
        charge_rate=args.charge_rate,
        discharge_mode=args.discharge_mode,
        window_size=DATA_CONFIG["window_size"],
    )

    arch = NN_CONFIG["architecture"]
    sizes = [arch["input_size"]] + arch["hidden_layers"] + [arch["output_size"]]
    print(f"[train_pinn_healthy] Architecture: {sizes}")

    nn = DeepNeuralNetwork(
        sizes=sizes,
        activation=arch["activation"],
        dropout_rate=arch["dropout_rate"],
    )

    # Warm-start the 8 output units to the config's target physical values.
    # This is an *initial condition* — not a bound. The output layer is still
    # linear and every parameter remains free to drift anywhere during Adam
    # optimization. The purpose is purely numerical: at He-initialized He
    # scale, the random outputs put tau1 = R1*C1 into regions where
    # np.exp(-Ts/tau) overflows and poisons the RMSE with NaN before the
    # optimizer can take any useful step. Starting from physically plausible
    # values keeps the first forward pass finite so training can begin.
    tv = np.array(list(BATTERY_CONFIG["target_values"].values()), dtype=np.float64)
    out_idx = len(sizes) - 1
    nn.params[f"W{out_idx}"] = nn.params[f"W{out_idx}"] * 1e-3
    nn.params[f"b{out_idx}"] = tv.reshape(-1, 1)
    nn.adam_opt["m"][f"W{out_idx}"] = np.zeros_like(nn.params[f"W{out_idx}"])
    nn.adam_opt["v"][f"W{out_idx}"] = np.zeros_like(nn.params[f"W{out_idx}"])
    nn.adam_opt["m"][f"b{out_idx}"] = np.zeros_like(nn.params[f"b{out_idx}"])
    nn.adam_opt["v"][f"b{out_idx}"] = np.zeros_like(nn.params[f"b{out_idx}"])
    print(f"[train_pinn_healthy] warm-started output-layer biases to target_values")

    history = nn.train(
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        l_rate=args.learning_rate,
        validation_split=NN_CONFIG["training"]["validation_split"],
        early_stopping_patience=args.patience,
    )

    # The upstream DNN.train only restores best_params when early stopping
    # fires. If the full epoch budget is used, self.params ends at the final
    # epoch — which can be one of the numerically unstable epochs. Force a
    # restore from best_params before saving so the frozen PINN reflects the
    # best validation checkpoint observed during training.
    if hasattr(nn, "best_params") and np.isfinite(history.get("best_val_loss", np.inf)):
        nn.params = {k: v.copy() for k, v in nn.best_params.items()}
        print("[train_pinn_healthy] restored best-val weights before saving")

    nn.save_model(args.output)
    print(f"[train_pinn_healthy] best val RMSE = {history['best_val_loss']:.6f}")
    print(f"[train_pinn_healthy] weights saved -> {args.output}")


if __name__ == "__main__":
    main()
