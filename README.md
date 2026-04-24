# Battery Fault Detection — Three-Scalogram Residual Analysis

Merged pipeline combining two prior projects into a single ISC (internal
short-circuit) fault detector for the Tsinghua NCM811 dataset:

1. **PINN** (from-scratch NumPy) learns V_pred(t) from V_meas(t) and I(t)
   on healthy cells only.
2. **CWT** produces three Morlet scalograms per window: V_meas, V_pred,
   and the residual e(t) = V_meas − V_pred.
3. **CNN** classifies each window as healthy or ISC-faulty using the
   three-scalogram (224, 224, 3) stack as input.

## Three-scalogram logic & frozen-PINN discipline

The PINN is trained **only on BD (healthy) files**, never on ISC_TEST (CS)
data, and then frozen. This is deliberate: because the network has never
been exposed to an internal short circuit, it can only reconstruct the
voltage response of a healthy cell. When a CS file is fed through the
frozen PINN, `V_pred(t)` is still the voltage the NCM811 would produce
*if it were healthy*, while `V_meas(t)` carries the real ISC signature.
Their difference, `e(t) = V_meas − V_pred`, therefore concentrates the
ISC-induced deviation into an auditable residual channel. Running the
same cubic-detrend + z-score + Morlet-CWT + COI-mask pipeline on all
three of `V_meas`, `V_pred`, and `e(t)` yields a `(224, 224, 3)` image in
which the first two channels encode the raw observable and its healthy
reconstruction, and the third channel — the residual scalogram — is the
diagnostic: dark/diffuse on healthy cells, bright and structured at mid
frequencies on shorted cells. The CNN learns from the **disagreement**
between a physics-grounded healthy prior and the measurement, not from
the measurement alone, so it cannot absorb the fault by memorising
surface-level voltage patterns. If the PINN were ever trained on CS
data, it would learn to fit ISC-perturbed voltage too, the residual
would shrink toward noise on both classes, and the entire experiment
would be invalidated.

## Repo layout

```
src/
  config.py                        # merged config (no PSO, no param bounds, Ts=1.0s)
  pinn/
    battery_physics.py             # BatteryModel.calculate_voltage (8 params free)
    deep_neural_network.py         # 14 -> 64x3 -> 8 NumPy DNN
    data_loader.py                 # BatteryDataProcessor (14 engineered features)
  wavelet/
    cwt_utils.py                   # Morlet CWT (pure NumPy)
    image_utils.py                 # raw_log_scalogram, normalize_and_resize, stack_channels
    preprocess.py                  # estimate_fs, create_windows, normalize_array
    plot_utils.py                  # plot_frequency_scalogram (supports multi-panel axes)
    model.py                       # build_cnn_model (head="binary"|"multiclass"|"regression")
  data/
    tsinghua_loader.py             # two-block UTF-8-BOM CSV parser + Phase 1 helpers
    prepare_pinn_features.py       # Tsinghua -> BatteryDataProcessor feature schema
  train_pinn_healthy.py            # fits PINN on BD files only (no PSO, no bounds)
  infer_pinn.py                    # frozen PINN -> V_pred(t), state threaded per file
  three_scalogram_builder.py       # build_three_scalogram_image, build_all_windows
  train_classifier.py              # file-level stratified 70/15/15 split, binary CNN
  visualize_three_scalograms.py    # 2x2 diagnostic PNG per file
main.py                            # CLI: list-phase1 / train-pinn / train-cls / visualize / predict
dataset/                           # user-supplied Tsinghua data
models/                            # pinn_healthy.npz, fault_classifier.keras
cache/                             # per-file window caches
results/                           # plots, three-scalogram figures
```

## Phase 1 run order

```bash
python main.py list-phase1
python main.py train-pinn --epochs 50
python main.py train-cls
python main.py visualize dataset/NCM811_ISC_TEST/DST/ISC_CS_0.5CC_DST_10ohm.csv
python main.py predict   dataset/NCM811_ISC_TEST/DST/ISC_CS_0.5CC_DST_10ohm.csv
```

## Key design decisions (per spec)

- **Parameter bounds removed.** All `np.clip()` calls on C1, C2, R0, R1, R2,
  gamma1, M0, M inside `BatteryModel.calculate_voltage` are gone. The SOC
  `np.clip(z, 0, 100)` and the epsilon safety factor are kept.
- **`sampling_time = 1.0 s`.** Tsinghua data is 1 Hz; the original 10 kHz
  default from Battery_Passport is replaced in both `src/config.py` and the
  `BatteryModel` constructor.
- **No scaling on NN outputs.** `DeepNeuralNetwork` output layer is linear;
  the 8 parameters enter `calculate_voltage` unmodified.
- **State threading across timesteps.** `infer_pinn.predict_voltage_series`
  explicitly threads `(ir1, ir2, z, h, s)` through the discharge rollout,
  resetting at the start of each file.
- **CNN input = (224, 224, 3).** Channels are [V_meas, V_pred, residual].
  Current is not used as a channel.
- **Binary head by default.** `build_cnn_model(head="binary")` yields one
  sigmoid output with BCE loss + accuracy + AUC. Multiclass severity
  (`head="multiclass"`) is available for Phase 2.
- **File-level split.** 70/15/15 stratified by class at the file level;
  windows from the same file never straddle train/val/test.
- **Window / scales.** `WINDOW_SIZE=1280`, `STRIDE=256`, and
  `SCALES = np.geomspace(2.0, 300.0, 96)` are inherited unchanged from
  Wavelet_Analysis `config/settings.py`.

## Notes

- TensorFlow is imported only when `train-cls` or `predict` runs; the
  non-TF modules (loaders, PINN, CWT, visualizer plumbing) work without it.
- The PINN's `BatteryModel` uses a default capacity of 10 Ah from upstream.
  Tsinghua NCM811 cells are smaller (~2.6 Ah as reported in the CSV); pass
  `capacity_as=` when constructing `BatteryModel` or calling
  `predict_voltage_series` if you want physical SOC tracking to match the
  real cell.
- The `_sources/` folder (Battery_Passport + Wavelet_Analysis clones) is
  kept as a reference and is ignored by git via `.gitignore`.
