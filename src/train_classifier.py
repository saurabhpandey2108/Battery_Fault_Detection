"""
Phase 1 ISC classifier training.

Pipeline per file:
    load Tsinghua CSV
    → discharge_df → engineered PINN features
    → frozen PINN predicts V_pred series
    → window V_meas and V_pred with identical indices
    → build 3-scalogram (224,224,3) stack per window
    → label = 0 (BD healthy) / 1 (CS faulty)

Stratified file-level split (70/15/15 train/val/test) — windows from the
same file never straddle split boundaries.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import (
    DATASET_DIR, MODELS_DIR, CACHE_DIR,
    WINDOW_SIZE, STRIDE, SCALES, IMAGE_SHAPE, CNN_CONFIG,
    PHASE1_CHARGE_RATE, PHASE1_DISCHARGE_MODE,
)
from src.data.tsinghua_loader import load_tsinghua_csv, list_phase1_pairs
from src.infer_pinn import load_pinn, predict_voltage_series
from src.three_scalogram_builder import build_all_windows
from src.wavelet.model import build_cnn_model
from src.pinn.data_loader import BatteryDataProcessor


def _tag(class_: str, ohm: int) -> str:
    return f"{class_}_{ohm}"


def _cache_path(cache_dir: str, class_: str, ohm: int) -> str:
    return os.path.join(cache_dir, f"windows_{class_}_{ohm}ohm.npz")


def build_windows_for_file(filepath: str,
                           label: int,
                           nn,
                           processor: BatteryDataProcessor,
                           fs: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    d = load_tsinghua_csv(filepath)
    pred = predict_voltage_series(nn, d["discharge_df"], processor=processor)
    imgs, lbls = build_all_windows(
        v_meas=pred["v_meas"],
        v_pred=pred["v_pred"],
        fs=fs,
        scales=SCALES,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        label=label,
    )
    return imgs, lbls


def materialize_dataset(pairs: List[Tuple[str, str, int]],
                        pinn_weights: str,
                        cache_dir: str,
                        fs: float = 1.0,
                        use_cache: bool = True,
                        verbose: bool = True) -> List[dict]:
    """Return a list of per-file entries for every BD and CS file.

    Each entry: {"tag", "ohm", "class_", "label", "images", "labels", "file"}.
    """
    os.makedirs(cache_dir, exist_ok=True)
    nn = load_pinn(pinn_weights)
    processor = BatteryDataProcessor(window_size=5)

    entries: List[dict] = []
    for bd_path, cs_path, ohm in pairs:
        for class_, path in (("BD", bd_path), ("CS", cs_path)):
            label = 0 if class_ == "BD" else 1
            cp = _cache_path(cache_dir, class_, ohm)
            if use_cache and os.path.isfile(cp):
                data = np.load(cp)
                imgs, lbls = data["images"], data["labels"]
                if verbose:
                    print(f"  cached {class_} {ohm}ohm -> {imgs.shape}")
            else:
                imgs, lbls = build_windows_for_file(path, label, nn, processor, fs=fs)
                if verbose:
                    print(f"  built  {class_} {ohm}ohm -> {imgs.shape}")
                if use_cache:
                    np.savez_compressed(cp, images=imgs.astype(np.float32),
                                        labels=lbls.astype(np.float32))
            entries.append({
                "tag": _tag(class_, ohm),
                "ohm": ohm,
                "class_": class_,
                "label": label,
                "images": imgs,
                "labels": lbls,
                "file": path,
            })
    return entries


def stratified_file_split(entries: List[dict],
                          train_frac: float = 0.70,
                          val_frac: float = 0.15,
                          seed: int = 42) -> Tuple[List[dict], List[dict], List[dict]]:
    """Split at the FILE level, stratified by class_."""
    rng = np.random.default_rng(seed)

    def _split_group(group: List[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
        n = len(group)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_train = max(int(round(n * train_frac)), 1)
        n_val = max(int(round(n * val_frac)), 1) if n > n_train + 1 else 0
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]
        return ([group[i] for i in train_idx],
                [group[i] for i in val_idx],
                [group[i] for i in test_idx])

    bd = [e for e in entries if e["class_"] == "BD"]
    cs = [e for e in entries if e["class_"] == "CS"]

    bd_tr, bd_va, bd_te = _split_group(bd)
    cs_tr, cs_va, cs_te = _split_group(cs)

    return bd_tr + cs_tr, bd_va + cs_va, bd_te + cs_te


def stack_entries(entries: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    if not entries:
        return (np.zeros((0,) + IMAGE_SHAPE, dtype=np.float32),
                np.zeros((0,), dtype=np.float32))
    X = np.concatenate([e["images"] for e in entries], axis=0).astype(np.float32)
    y = np.concatenate([e["labels"] for e in entries], axis=0).astype(np.float32)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Train fault classifier on three-scalogram stacks")
    parser.add_argument("--dataset-root", default=DATASET_DIR)
    parser.add_argument("--pinn-weights", default=os.path.join(MODELS_DIR, "pinn_healthy.npz"))
    parser.add_argument("--output", default=os.path.join(MODELS_DIR, "fault_classifier.keras"))
    parser.add_argument("--cache-dir", default=os.path.join(CACHE_DIR, "windows_phase1"))
    parser.add_argument("--charge-rate", type=float, default=PHASE1_CHARGE_RATE)
    parser.add_argument("--discharge-mode", default=PHASE1_DISCHARGE_MODE)
    parser.add_argument("--epochs", type=int, default=CNN_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=CNN_CONFIG["batch_size"])
    parser.add_argument("--patience", type=int, default=CNN_CONFIG["early_stop_patience"])
    parser.add_argument("--noise-std", type=float, default=CNN_CONFIG["noise_std"])
    parser.add_argument("--seed", type=int, default=CNN_CONFIG["random_seed"])
    parser.add_argument("--no-cache", action="store_true",
                        help="Rebuild window cache instead of reusing it")
    args = parser.parse_args()

    np.random.seed(args.seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(args.seed)
    except Exception:
        tf = None

    pairs = list_phase1_pairs(args.dataset_root,
                              charge_rate=args.charge_rate,
                              discharge_mode=args.discharge_mode)
    if not pairs:
        raise RuntimeError(
            f"No Phase 1 pairs found under {args.dataset_root} for "
            f"{args.charge_rate}C / {args.discharge_mode}"
        )
    print(f"[train_classifier] {len(pairs)} BD/CS pairs for Phase 1")

    entries = materialize_dataset(
        pairs=pairs,
        pinn_weights=args.pinn_weights,
        cache_dir=args.cache_dir,
        fs=1.0,
        use_cache=not args.no_cache,
    )

    train_entries, val_entries, test_entries = stratified_file_split(
        entries, train_frac=0.70, val_frac=0.15, seed=args.seed
    )
    print(f"[train_classifier] files — train {len(train_entries)}, "
          f"val {len(val_entries)}, test {len(test_entries)}")

    X_train, y_train = stack_entries(train_entries)
    X_val, y_val = stack_entries(val_entries)
    X_test, y_test = stack_entries(test_entries)
    print(f"[train_classifier] windows — train {X_train.shape}, "
          f"val {X_val.shape}, test {X_test.shape}")

    if X_train.size == 0:
        raise RuntimeError("No training windows produced — check window/stride vs. series length")

    model = build_cnn_model(
        image_shape=IMAGE_SHAPE,
        use_temperature_scalar=False,
        noise_std=args.noise_std,
        head="binary",
    )
    model.summary()

    from tensorflow import keras
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if X_val.size else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )

    if X_test.size:
        test_metrics = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
        print(f"[train_classifier] test metrics: {test_metrics}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model.save(args.output)
    print(f"[train_classifier] model saved -> {args.output}")


if __name__ == "__main__":
    main()
