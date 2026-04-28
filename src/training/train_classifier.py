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

# File lives at src/training/<this>.py -- climb 3 dirnames to reach project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import (
    DATASET_DIR, MODELS_DIR, CACHE_DIR, RESULTS_DIR,
    WINDOW_SIZE, STRIDE, SCALES, IMAGE_SHAPE, CNN_CONFIG,
    PHASE1_CHARGE_RATE, PHASE1_DISCHARGE_MODE,
)
from src.data.tsinghua_loader import load_tsinghua_csv, list_phase1_pairs
from src.inference.infer_pinn import load_pinn, predict_voltage_series
from src.scalograms.three_scalogram_builder import build_all_windows
from src.wavelet.model import build_cnn_model


def _tag(class_: str, ohm: int) -> str:
    return f"{class_}_{ohm}"


def _cache_path(cache_dir: str, class_: str, ohm: int) -> str:
    return os.path.join(cache_dir, f"windows_{class_}_{ohm}ohm.npz")


def build_windows_for_file(filepath: str,
                           label: int,
                           nn,
                           fs: float = 1.0,
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (images, fault_labels, soc_per_window)."""
    d = load_tsinghua_csv(filepath)
    pred = predict_voltage_series(nn, d["discharge_df"])
    imgs, lbls = build_all_windows(
        v_meas=pred["v_meas"],
        v_pred=pred["v_pred"],
        fs=fs,
        scales=SCALES,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        label=label,
    )
    soc_window = _per_window_mean(pred["soc"], WINDOW_SIZE, STRIDE)
    return imgs, lbls, soc_window


def _per_window_mean(arr: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Mean of `arr` per window using the same start indices as build_all_windows."""
    arr = np.asarray(arr, dtype=np.float64)
    n_windows = max(0, (len(arr) - window_size) // stride + 1)
    out = np.empty(n_windows, dtype=np.float32)
    for i in range(n_windows):
        s = i * stride
        out[i] = float(np.mean(arr[s:s + window_size]))
    return out


def materialize_dataset(pairs: List[Tuple[str, str, int]],
                        pinn_weights: str,
                        cache_dir: str,
                        fs: float = 1.0,
                        use_cache: bool = True,
                        verbose: bool = True,
                        need_soc: bool = False) -> List[dict]:
    """Return a list of per-file entries for every BD and CS file.

    Each entry contains: tag, ohm, class_, label, file, images, labels, and
    -- when `need_soc` -- soc_window (per-window mean coulomb-counted SOC).

    The cache is backward-compatible: an existing npz that lacks the
    `soc_window` key is upgraded in place by re-running PINN inference for
    that one file (windows are not rebuilt; only the SOC labels).
    """
    os.makedirs(cache_dir, exist_ok=True)
    nn = load_pinn(pinn_weights)

    entries: List[dict] = []
    for bd_path, cs_path, ohm in pairs:
        for class_, path in (("BD", bd_path), ("CS", cs_path)):
            label = 0 if class_ == "BD" else 1
            cp = _cache_path(cache_dir, class_, ohm)
            soc_w = None
            if use_cache and os.path.isfile(cp):
                data = np.load(cp)
                imgs, lbls = data["images"], data["labels"]
                if "soc_window" in data.files:
                    soc_w = data["soc_window"]
                if verbose:
                    print(f"  cached {class_} {ohm}ohm -> {imgs.shape}"
                          + ("  +soc" if soc_w is not None else ""))
            else:
                imgs, lbls, soc_w = build_windows_for_file(path, label, nn, fs=fs)
                if verbose:
                    print(f"  built  {class_} {ohm}ohm -> {imgs.shape}  +soc")

            # Backfill SOC labels if needed (regression mode and cache missed it).
            if need_soc and soc_w is None:
                d = load_tsinghua_csv(path)
                pred = predict_voltage_series(nn, d["discharge_df"])
                soc_w = _per_window_mean(pred["soc"], WINDOW_SIZE, STRIDE)
                if verbose:
                    print(f"    + backfilled soc_window for {class_} {ohm}ohm")

            if use_cache and not os.path.isfile(cp):
                np.savez_compressed(cp,
                                    images=imgs.astype(np.float32),
                                    labels=lbls.astype(np.float32),
                                    soc_window=soc_w.astype(np.float32)
                                        if soc_w is not None
                                        else np.zeros(0, dtype=np.float32))
            elif use_cache and soc_w is not None and "soc_window" not in np.load(cp).files:
                # Upgrade existing cache file with SOC labels in place.
                np.savez_compressed(cp,
                                    images=imgs.astype(np.float32),
                                    labels=lbls.astype(np.float32),
                                    soc_window=soc_w.astype(np.float32))

            entries.append({
                "tag": _tag(class_, ohm),
                "ohm": ohm,
                "class_": class_,
                "label": label,
                "images": imgs,
                "labels": lbls,
                "soc_window": soc_w,
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


def stack_entries(entries: List[dict],
                  channel_slice: int = None,
                  soc_target: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Stack per-file entries into model-ready (X, y) arrays.

    channel_slice : int, optional
        If given, keep only that channel (e.g., 1 for V_pred); X becomes
        (N, H, W, 1). Default keeps all channels.
    soc_target : bool
        If True, y = per-window mean SOC; else y = binary fault label.
    """
    if not entries:
        c = 1 if channel_slice is not None else IMAGE_SHAPE[2]
        return (np.zeros((0,) + IMAGE_SHAPE[:2] + (c,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32))

    X = np.concatenate([e["images"] for e in entries], axis=0).astype(np.float32)
    if channel_slice is not None:
        X = X[..., channel_slice:channel_slice + 1]

    if soc_target:
        y = np.concatenate([e["soc_window"] for e in entries], axis=0).astype(np.float32)
    else:
        y = np.concatenate([e["labels"] for e in entries], axis=0).astype(np.float32)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Train fault classifier on three-scalogram stacks")
    parser.add_argument("--dataset-root", default=DATASET_DIR)
    parser.add_argument("--pinn-weights", default=os.path.join(MODELS_DIR, "pinn_healthy_no_leak.npz"))
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
    parser.add_argument("--mode", choices=["classify", "soc-regress"],
                        default="classify",
                        help="`classify` -> binary BD/CS head; "
                             "`soc-regress` -> regression head, V_pred channel only, "
                             "target = mean coulomb-counted SOC across the window")
    parser.add_argument("--history-csv", default=None,
                        help="Per-epoch metrics dump. Defaults under results/cnn/")
    parser.add_argument("--plot-path", default=None,
                        help="Training-history plot. Defaults under results/cnn/")
    args = parser.parse_args()

    np.random.seed(args.seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(args.seed)
    except Exception:
        tf = None

    # Mode-specific output paths under results/cnn/. Existing args win if set.
    cnn_dir = os.path.join(RESULTS_DIR, "cnn")
    os.makedirs(cnn_dir, exist_ok=True)
    prefix = "soc_" if args.mode == "soc-regress" else ""
    if args.history_csv is None:
        args.history_csv = os.path.join(cnn_dir, f"{prefix}train_history.csv")
    if args.plot_path is None:
        args.plot_path = os.path.join(cnn_dir, f"{prefix}train_history.png")

    pairs = list_phase1_pairs(args.dataset_root,
                              charge_rate=args.charge_rate,
                              discharge_mode=args.discharge_mode)
    if not pairs:
        raise RuntimeError(
            f"No Phase 1 pairs found under {args.dataset_root} for "
            f"{args.charge_rate}C / {args.discharge_mode}"
        )
    print(f"[train_classifier] mode={args.mode}  {len(pairs)} BD/CS pairs for Phase 1")

    entries = materialize_dataset(
        pairs=pairs,
        pinn_weights=args.pinn_weights,
        cache_dir=args.cache_dir,
        fs=1.0,
        use_cache=not args.no_cache,
        need_soc=(args.mode == "soc-regress"),
    )

    train_entries, val_entries, test_entries = stratified_file_split(
        entries, train_frac=0.70, val_frac=0.15, seed=args.seed
    )
    print(f"[train_classifier] files — train {len(train_entries)}, "
          f"val {len(val_entries)}, test {len(test_entries)}")

    if args.mode == "classify":
        X_train, y_train = stack_entries(train_entries)
        X_val, y_val = stack_entries(val_entries)
        X_test, y_test = stack_entries(test_entries)
        head_kind = "binary"
        image_shape = IMAGE_SHAPE
    else:
        # SOC regression: V_pred channel only (channel 1 of the (H,W,3) stack),
        # target = mean coulomb-counted SOC across the window.
        X_train, y_train = stack_entries(train_entries, channel_slice=1, soc_target=True)
        X_val, y_val = stack_entries(val_entries, channel_slice=1, soc_target=True)
        X_test, y_test = stack_entries(test_entries, channel_slice=1, soc_target=True)
        head_kind = "regression"
        image_shape = IMAGE_SHAPE[:2] + (1,)

    print(f"[train_classifier] windows — train {X_train.shape}, "
          f"val {X_val.shape}, test {X_test.shape}")

    if X_train.size == 0:
        raise RuntimeError("No training windows produced — check window/stride vs. series length")

    model = build_cnn_model(
        image_shape=image_shape,
        use_temperature_scalar=False,
        noise_std=args.noise_std,
        head=head_kind,
        learning_rate=CNN_CONFIG.get("learning_rate", 1e-4),
        l2_lambda=CNN_CONFIG.get("l2_lambda", 1e-4),
        dropout_rate=CNN_CONFIG.get("dropout_rate", 0.5),
    )
    model.summary()

    from tensorflow import keras
    if args.mode == "classify":
        # Val_auc is more stable than loss on small validation sets and is
        # the metric we care about for fault detection.
        es_kwargs = dict(monitor="val_auc", mode="max")
    else:
        # No AUC for regression -- monitor val_rmse, lower is better.
        es_kwargs = dict(monitor="val_rmse", mode="min")
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=args.patience, restore_best_weights=True, **es_kwargs,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(args.patience // 2, 2),
            min_lr=1e-6, verbose=1,
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

    # Save model under a mode-specific name so the binary classifier and the
    # SOC regressor don't clobber each other.
    out_path = args.output
    if args.mode == "soc-regress" and out_path.endswith("fault_classifier.keras"):
        out_path = os.path.join(MODELS_DIR, "soc_regressor.keras")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.save(out_path)
    print(f"[train_classifier] model saved -> {out_path}")

    # ---- persist training history (CSV + plot) -----------------------
    save_training_history(history, args.history_csv, args.plot_path,
                          mode=args.mode)

    # ---- evaluate on test set and persist all artifacts --------------
    if X_test.size:
        if args.mode == "classify":
            evaluate_classifier(model, test_entries, X_test, y_test, cnn_dir)
        else:
            evaluate_soc_regressor(model, test_entries, X_test, y_test, cnn_dir)


def save_training_history(history, csv_path: str, plot_path: str,
                          mode: str = "classify") -> None:
    """Write per-epoch metrics to CSV and a 3-panel plot.

    Panels depend on `mode`:
      classify    -> loss / accuracy / AUC
      soc-regress -> loss (MSE) / MAE / RMSE
    """
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = history.history
    if not h:
        print("[train_classifier] empty history, nothing to save")
        return

    keys = list(h.keys())
    n_epochs = len(next(iter(h.values())))

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch"] + keys)
        for i in range(n_epochs):
            w.writerow([i + 1] + [h[k][i] for k in keys])
    print(f"[train_classifier] history -> {csv_path}")

    epochs = np.arange(1, n_epochs + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    def _pair_plot(ax, train_key: str, val_key: str, title: str, ylabel: str,
                   lower_is_better: bool):
        if train_key in h:
            ax.plot(epochs, h[train_key], color="#1b9e77", linewidth=1.6,
                    marker="o", markersize=3, label=f"train {train_key}")
        if val_key in h:
            ax.plot(epochs, h[val_key], color="#d95f02", linewidth=1.6,
                    marker="s", markersize=3, label=f"val {val_key}")
            best_ep = (int(np.argmin(h[val_key]) + 1) if lower_is_better
                       else int(np.argmax(h[val_key]) + 1))
            ax.axvline(best_ep, color="#7570b3", linewidth=1.0,
                       linestyle="--", alpha=0.7, label=f"best epoch {best_ep}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    if mode == "classify":
        _pair_plot(axes[0], "loss", "val_loss", "Loss (BCE)", "loss", True)
        _pair_plot(axes[1], "accuracy", "val_accuracy", "Accuracy", "accuracy", False)
        _pair_plot(axes[2], "auc", "val_auc", "AUC", "AUC", False)
        title = "CNN three-scalogram classifier — training history"
    else:
        _pair_plot(axes[0], "loss", "val_loss", "Loss (MSE)", "MSE", True)
        _pair_plot(axes[1], "mae", "val_mae", "MAE (SOC)", "MAE", True)
        _pair_plot(axes[2], "rmse", "val_rmse", "RMSE (SOC)", "RMSE", True)
        title = "CNN V_pred-scalogram → SOC regressor — training history"

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    print(f"[train_classifier] plot -> {plot_path}")


def evaluate_classifier(model, test_entries, X_test, y_test, cnn_dir: str) -> None:
    """Run the trained binary classifier on the held-out test set and write
    all metrics and predictions to ``results/cnn/`` so the run is auditable.

    Outputs (under cnn_dir)
    -----------------------
    test_metrics.json       loss / acc / auc / precision / recall / f1
                            (window-level AND file-level)
    test_predictions.csv    one row per test window
    test_per_file.csv       one row per test file (mean prob -> verdict)
    confusion_matrix.png    side-by-side window and file confusion matrices
    """
    results_dir = cnn_dir
    import csv
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Per-window predictions
    probs = model.predict(X_test, batch_size=64, verbose=0).ravel()
    preds = (probs >= 0.5).astype(np.int32)
    y_true = y_test.astype(np.int32)

    keras_metrics = model.evaluate(X_test, y_test, verbose=0, return_dict=True)

    def _binary_stats(y, p):
        tp = int(((p == 1) & (y == 1)).sum())
        tn = int(((p == 0) & (y == 0)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc = (tp + tn) / max(len(y), 1)
        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "precision": prec, "recall": rec, "f1": f1, "accuracy": acc}

    win_stats = _binary_stats(y_true, preds)

    # Per-file aggregation: a cell is faulty iff its mean window probability >= 0.5.
    cursor = 0
    file_rows = []
    for e in test_entries:
        n = e["images"].shape[0]
        win_probs = probs[cursor:cursor + n]
        cursor += n
        if n == 0:
            continue
        mean_prob = float(np.mean(win_probs))
        max_prob = float(np.max(win_probs))
        verdict = 1 if mean_prob >= 0.5 else 0
        file_rows.append({
            "file": os.path.basename(e["file"]),
            "class": e["class_"],
            "ohm": e["ohm"],
            "true_label": e["label"],
            "n_windows": n,
            "mean_prob": mean_prob,
            "max_prob": max_prob,
            "verdict": verdict,
        })

    file_y = np.array([r["true_label"] for r in file_rows], dtype=np.int32)
    file_pred = np.array([r["verdict"] for r in file_rows], dtype=np.int32)
    file_stats = _binary_stats(file_y, file_pred) if file_rows else {}

    # JSON metrics
    metrics_path = os.path.join(results_dir, "test_metrics.json")
    out = {
        "keras_metrics": {k: float(v) for k, v in keras_metrics.items()},
        "window_level": win_stats,
        "file_level": file_stats,
        "n_test_windows": int(len(y_true)),
        "n_test_files": int(len(file_rows)),
    }
    with open(metrics_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[train_classifier] test metrics -> {metrics_path}")

    # Per-window predictions CSV
    pred_csv = os.path.join(results_dir, "test_predictions.csv")
    with open(pred_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["window_index", "true_label", "prob", "pred"])
        for i, (yt, pr, pd) in enumerate(zip(y_true, probs, preds)):
            w.writerow([i, int(yt), float(pr), int(pd)])
    print(f"[train_classifier] per-window preds -> {pred_csv}")

    # Per-file predictions CSV
    file_csv = os.path.join(results_dir, "test_per_file.csv")
    with open(file_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "class", "ohm", "true_label",
                    "n_windows", "mean_prob", "max_prob", "verdict", "correct"])
        for r in file_rows:
            w.writerow([r["file"], r["class"], r["ohm"], r["true_label"],
                        r["n_windows"], f"{r['mean_prob']:.6f}",
                        f"{r['max_prob']:.6f}", r["verdict"],
                        int(r["verdict"] == r["true_label"])])
    print(f"[train_classifier] per-file preds -> {file_csv}")

    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, stats, title in [
        (axes[0], win_stats, f"Per-window ({len(y_true)} windows)"),
        (axes[1], file_stats, f"Per-file ({len(file_rows)} files)"),
    ]:
        if not stats:
            ax.axis("off")
            continue
        cm = np.array([[stats["tn"], stats["fp"]],
                       [stats["fn"], stats["tp"]]])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=14,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred BD", "pred CS"])
        ax.set_yticklabels(["true BD", "true CS"])
        ax.set_title(f"{title}\nacc={stats['accuracy']:.3f} "
                     f"prec={stats['precision']:.3f} "
                     f"rec={stats['recall']:.3f} "
                     f"f1={stats['f1']:.3f}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("CNN test-set confusion matrices", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=130)
    plt.close(fig)
    print(f"[train_classifier] confusion matrix -> {cm_path}")


def evaluate_soc_regressor(model, test_entries, X_test, y_test,
                           cnn_dir: str) -> None:
    """Evaluate the SOC-regression CNN on the held-out test set and write:

      results/cnn/soc_test_metrics.json     loss / mae / rmse / R^2
      results/cnn/soc_test_predictions.csv  one row per test window
      results/cnn/soc_predicted_vs_actual.png
                                            scatter (true SOC vs predicted SOC)
                                            with the y=x line; per-class colours.
    """
    import csv
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preds = model.predict(X_test, batch_size=64, verbose=0).ravel()
    y_true = y_test.astype(np.float64)
    y_pred = preds.astype(np.float64)

    keras_metrics = model.evaluate(X_test, y_test, verbose=0, return_dict=True)

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    out = {
        "keras_metrics": {k: float(v) for k, v in keras_metrics.items()},
        "mae_soc": mae,
        "rmse_soc": rmse,
        "r2": r2,
        "n_test_windows": int(len(y_true)),
        "n_test_files": int(len(test_entries)),
    }
    metrics_path = os.path.join(cnn_dir, "soc_test_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[train_classifier] SOC test metrics -> {metrics_path}")

    # Per-window predictions, with per-window class tag for colouring later.
    classes = []
    cursor = 0
    for e in test_entries:
        n = e["images"].shape[0]
        classes.extend([e["class_"]] * n)
        cursor += n

    pred_csv = os.path.join(cnn_dir, "soc_test_predictions.csv")
    with open(pred_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["window_index", "class", "true_soc", "pred_soc", "abs_error"])
        for i, (cl, yt, yp) in enumerate(zip(classes, y_true, y_pred)):
            w.writerow([i, cl, float(yt), float(yp), float(abs(yp - yt))])
    print(f"[train_classifier] SOC per-window preds -> {pred_csv}")

    # Predicted-vs-actual scatter
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    bd_mask = np.array([c == "BD" for c in classes])
    ax.scatter(y_true[bd_mask], y_pred[bd_mask], s=14, alpha=0.6,
               color="#1b9e77", label=f"BD ({bd_mask.sum()})")
    ax.scatter(y_true[~bd_mask], y_pred[~bd_mask], s=14, alpha=0.6,
               color="#d95f02", label=f"CS ({(~bd_mask).sum()})")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.02 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            color="black", linewidth=1.0, linestyle="--", label="y=x")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("True coulomb-counted SOC (mean over window)")
    ax.set_ylabel("Predicted SOC (CNN, V_pred scalogram)")
    ax.set_title(f"SOC regression — MAE={mae:.4f}  RMSE={rmse:.4f}  R^2={r2:.3f}\n"
                 f"({len(y_true)} test windows from {len(test_entries)} files)",
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    sc_path = os.path.join(cnn_dir, "soc_predicted_vs_actual.png")
    fig.savefig(sc_path, dpi=130)
    plt.close(fig)
    print(f"[train_classifier] SOC scatter -> {sc_path}")


if __name__ == "__main__":
    main()
