"""
Top-level CLI for the merged Battery Fault Detection pipeline.

Subcommands
-----------
    list-phase1       List the 14 BD/CS file pairs in the Phase 1 subset.
    train-pinn        Train the healthy-only PINN (saves models/pinn_healthy.npz).
    infer             Render a 2x2 three-scalogram figure for one CSV
                      (uses --file per smoke-test spec).
    compare           Render a 2x4 BD-vs-CS side-by-side figure at one resistance.
    train-classifier  Train the three-scalogram fault classifier
                      (saves models/fault_classifier.keras).
    predict           Run the frozen classifier on one CSV.

Examples
--------
    python main.py list-phase1
    python main.py train-pinn --epochs 50
    python main.py infer --file dataset/NCM811_ISC_TEST/DST/ISC_CS_0.5CC_DST_100ohm.csv
    python main.py compare --ohm 10
    python main.py train-classifier
    python main.py predict --file dataset/NCM811_ISC_TEST/DST/ISC_CS_0.5CC_DST_10ohm.csv
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _cmd_list_phase1(args):
    from src.data.tsinghua_loader import list_phase1_pairs

    pairs = list_phase1_pairs(args.dataset_root,
                              charge_rate=args.charge_rate,
                              discharge_mode=args.discharge_mode)
    print(f"{len(pairs)} pairs for {args.charge_rate}C / {args.discharge_mode}")
    for bd, cs, ohm in pairs:
        print(f"  {ohm:>4} ohm | BD {os.path.basename(bd)} | CS {os.path.basename(cs)}")


def _cmd_visualize_dataset(args):
    from src.visualization import visualize_dataset
    visualize_dataset.main()


def _cmd_verify(args):
    from src.verification import verify_residual_signal
    verify_residual_signal.main()


def _cmd_train_pinn(passthrough):
    from src.training import train_pinn_healthy
    sys.argv = ["train_pinn_healthy"] + list(passthrough)
    train_pinn_healthy.main()


def _cmd_train_classifier(passthrough):
    from src.training import train_classifier
    sys.argv = ["train_classifier"] + list(passthrough)
    train_classifier.main()


def _cmd_infer(args):
    from src.visualization.visualize_three_scalograms import visualize_file

    visualize_file(
        filepath=args.file,
        pinn_weights=args.pinn_weights,
        out_dir=args.out_dir,
        window_size=args.window_size,
        start_sample=args.start_sample,
        resistance_label=args.label,
    )


def _cmd_compare(args):
    from src.visualization.compare_bd_cs import compare_bd_cs
    from src.data.tsinghua_loader import list_phase1_pairs

    pairs = list_phase1_pairs(args.dataset_root,
                              charge_rate=args.charge_rate,
                              discharge_mode=args.discharge_mode,
                              ohms=[args.ohm])
    if not pairs:
        raise SystemExit(
            f"No BD/CS pair at {args.ohm} ohm for {args.charge_rate}C / {args.discharge_mode}"
        )
    bd, cs, ohm = pairs[0]
    out = os.path.join(args.out_dir, f"bd_vs_cs_{ohm}ohm.png")
    compare_bd_cs(
        bd_file=bd,
        cs_file=cs,
        pinn_weights=args.pinn_weights,
        out_path=out,
        window_size=args.window_size,
        start_sample=args.start_sample,
    )


def _cmd_predict(args):
    import numpy as np
    from src.config import SCALES, WINDOW_SIZE, STRIDE
    from src.data.tsinghua_loader import load_tsinghua_csv
    from src.inference.infer_pinn import predict_voltage_series
    from src.scalograms.three_scalogram_builder import build_all_windows
    from tensorflow import keras

    d = load_tsinghua_csv(args.file)
    pred = predict_voltage_series(args.pinn_weights, d["discharge_df"])

    images, _ = build_all_windows(
        v_meas=pred["v_meas"],
        v_pred=pred["v_pred"],
        fs=1.0,
        scales=SCALES,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        label=0,
    )
    if len(images) == 0:
        print("No windows produced - file too short")
        return

    model = keras.models.load_model(args.classifier_weights)
    probs = model.predict(images, verbose=0).ravel()
    print(f"File: {args.file}")
    print(f"Windows: {len(probs)}")
    print(f"Mean fault prob: {float(np.mean(probs)):.4f}")
    print(f"Max  fault prob: {float(np.max(probs)):.4f}")
    print(f"Per-window (first 10): {np.round(probs[:10], 4).tolist()}")
    verdict = "FAULTY (ISC)" if np.mean(probs) >= 0.5 else "HEALTHY"
    print(f"Verdict: {verdict}")


def build_parser():
    from src.config import (
        DATASET_DIR, MODELS_DIR, RESULTS_DIR,
        PHASE1_CHARGE_RATE, PHASE1_DISCHARGE_MODE, WINDOW_SIZE,
    )
    p = argparse.ArgumentParser(prog="battery-fault-detection",
                                description=__doc__.splitlines()[1] if __doc__ else None)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list-phase1", help="List Phase 1 file pairs")
    p_list.add_argument("--dataset-root", default=DATASET_DIR)
    p_list.add_argument("--charge-rate", type=float, default=PHASE1_CHARGE_RATE)
    p_list.add_argument("--discharge-mode", default=PHASE1_DISCHARGE_MODE)
    p_list.set_defaults(func=_cmd_list_phase1)

    p_vd = sub.add_parser("visualize-dataset",
                          help="Eyeball-check the data BEFORE training (Task 0)")
    p_vd.set_defaults(func=_cmd_visualize_dataset)

    p_vr = sub.add_parser("verify",
                          help="Run frozen PINN on BD + CS files, "
                               "print residual-signal acceptance verdict")
    p_vr.set_defaults(func=_cmd_verify)

    sub.add_parser("train-pinn", help="Train PINN on healthy BD data "
                                      "(all extra args forwarded to src.train_pinn_healthy)")
    sub.add_parser("train-classifier", help="Train the three-scalogram classifier "
                                            "(all extra args forwarded to src.train_classifier)")

    # --- infer: render a 2x2 figure for one CSV (smoke-test name) ---
    p_in = sub.add_parser("infer", help="Render a 2x2 three-scalogram figure for one CSV")
    p_in.add_argument("--file", required=True, help="Path to a Tsinghua CSV")
    p_in.add_argument("--pinn-weights", default=os.path.join(MODELS_DIR, "pinn_healthy.npz"))
    p_in.add_argument("--out-dir", default=os.path.join(RESULTS_DIR, "three_scalograms"))
    p_in.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    p_in.add_argument("--start-sample", type=int, default=0)
    p_in.add_argument("--label", default=None)
    p_in.set_defaults(func=_cmd_infer)

    # --- compare: BD-vs-CS side-by-side at one resistance ---
    p_cmp = sub.add_parser("compare", help="BD-vs-CS side-by-side 2x4 figure")
    p_cmp.add_argument("--ohm", type=int, default=10,
                       help="Resistance index (default 10)")
    p_cmp.add_argument("--dataset-root", default=DATASET_DIR)
    p_cmp.add_argument("--charge-rate", type=float, default=PHASE1_CHARGE_RATE)
    p_cmp.add_argument("--discharge-mode", default=PHASE1_DISCHARGE_MODE)
    p_cmp.add_argument("--pinn-weights", default=os.path.join(MODELS_DIR, "pinn_healthy.npz"))
    p_cmp.add_argument("--out-dir", default=os.path.join(RESULTS_DIR, "three_scalograms"))
    p_cmp.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    p_cmp.add_argument("--start-sample", type=int, default=0)
    p_cmp.set_defaults(func=_cmd_compare)

    p_pr = sub.add_parser("predict", help="Run the trained classifier on one CSV")
    p_pr.add_argument("--file", required=True)
    p_pr.add_argument("--pinn-weights", default=os.path.join(MODELS_DIR, "pinn_healthy.npz"))
    p_pr.add_argument("--classifier-weights",
                      default=os.path.join(MODELS_DIR, "fault_classifier.keras"))
    p_pr.set_defaults(func=_cmd_predict)

    return p


def main():
    # train-pinn and train-classifier intentionally bypass argparse so their
    # sub-script's own flags (--epochs, --batch-size, ...) pass through intact.
    argv = sys.argv[1:]
    if argv and argv[0] == "train-pinn":
        _cmd_train_pinn(argv[1:])
        return
    if argv and argv[0] == "train-classifier":
        _cmd_train_classifier(argv[1:])
        return

    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
