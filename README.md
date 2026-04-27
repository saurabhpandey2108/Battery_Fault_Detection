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

Source is grouped by role: configuration, data, models, training, inference,
visualization, verification, and analysis utilities each live in their own
sub-package.

```
src/
  config/
    settings.py                    # all constants (paths, NCM811 OCV, PINN/CWT/CNN config)
    __init__.py                    #   re-exports settings.* so `from src.config import X` works
  data/
    tsinghua_loader.py             # two-block UTF-8-BOM CSV parser + Phase 1 helpers
    prepare_pinn_features.py       # build_leakage_free_features + coulomb_counted_soc
  pinn/
    battery_physics.py             # BatteryModel.calculate_voltage (8 free params)
    deep_neural_network.py         # legacy from-scratch DNN (kept untouched for reference)
    pinn_v2.py                     # active MLP (35 -> 64x3 -> 8) with Adam + grad clipping
    data_loader.py                 # legacy BatteryDataProcessor (kept for reference)
  wavelet/
    cwt_utils.py                   # Morlet CWT (pure NumPy)
    image_utils.py                 # raw_log_scalogram, normalize_and_resize, stack_channels
    preprocess.py                  # estimate_fs, create_windows, normalize_array
    plot_utils.py                  # plot_frequency_scalogram (supports multi-panel axes)
    model.py                       # build_cnn_model (head="binary"|"multiclass"|"regression")
  scalograms/
    three_scalogram_builder.py     # build_three_scalogram_image, build_all_windows
  training/
    train_pinn_healthy.py          # fits PINN on BD files only (leakage-free inputs)
    train_classifier.py            # file-level 70/15/15 split, binary CNN
  inference/
    infer_pinn.py                  # frozen PINN -> V_pred(t), state threaded per file
  visualization/
    visualize_dataset.py           # Task 0 -- 5-panel data inspection + overlay
    visualize_three_scalograms.py  # 2x2 diagnostic PNG per file
    compare_bd_cs.py               # BD-vs-CS side-by-side at one resistance
  verification/
    verify_residual_signal.py      # acceptance gate before training the classifier
main.py                            # CLI: list-phase1 / visualize-dataset / verify /
                                   #      train-pinn / train-classifier / infer /
                                   #      compare / predict
dataset/                           # user-supplied Tsinghua data
models/                            # pinn_healthy_no_leak.npz, fault_classifier.keras
cache/                             # per-file window caches
results/                           # all output artifacts
  dataset_inspection/              #   5-panel PNG per file + comparison overlay
  three_scalograms/                #   diagnostic figures
  train_pinn_v2_history.csv        #   per-epoch metrics
  pinn_inference_dump.csv          #   per-sample (time, V_meas, V_pred, params*8)
  residual_verification.png        #   verify-stage figure (BD/CS comparison)
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

## Why the PINN inputs exclude voltage

If `V_meas` is one of the PINN's inputs, the network learns the inverse
physics map `params = g⁻¹(V_meas, I)` — given any voltage shown to it,
it produces 8 parameters whose physics rollout reproduces that voltage.
This is exactly what training pushes it toward, because the loss is
`||V_pred − V_meas||`. On a CS (faulty) cell, the input `V_meas`
already carries the ISC signature, so the network outputs params that
reproduce the faulty voltage and `V_pred ≈ V_meas` — the residual
collapses to noise on both healthy and faulty cells, and the diagnostic
disappears. Empirically we observed BD vs CS residual ratios of ~1.05×
across all 14 resistance pairs with V in the input.

The fix is to give the network only inputs that **cannot** encode the
fault: instantaneous current, recent and long-window current statistics,
its first difference, and a coulomb-counted SOC derived from the same
external current. None of those quantities can reproduce the ISC-induced
voltage drop, because the ISC current bypasses the ammeter. With this
input set the only way to reduce loss during training is to genuinely
learn the I → V mapping of a healthy cell. At inference time on a CS
cell, the hidden ISC current never reaches the ammeter, the network
keeps predicting the healthy voltage trajectory, and `V_meas` falls
below `V_pred` — producing a structured negative-mean residual that is
the fault detector's input signal.

## SOC handling

The Tsinghua dataset's `SOC|DOD/%` column is **not** a global SOC
trajectory. The Arbin tester resets it at every step transition of the
DST profile, so the column reports per-step depth-of-discharge that
swings between 0 % and 100 % every few seconds. The same applies to the
`capacity_Ah` column. Verified empirically: 277 single-sample SOC jumps
of >50 % in one BD file alone, max delta = 100 % in one sample — see
`results/dataset_inspection/`.

We therefore compute SOC in `src/data/prepare_pinn_features.py` by
trapezoidal coulomb counting on the external current,

```
SOC(t) = 1.0 + cumtrapz(I_signed, t_seconds) / (Q_rated × 3600)
```

with the dataset's sign convention (negative current = discharge) and
`Q_rated = 2.5487 Ah`, taken as the median net discharged Ah across the
14 BD DST 0.5C files. The output is clipped to `[0, 1]` to absorb
numerical overshoot at the endpoints. This is the single coulomb-
counting site in the codebase. It is fault-blind by construction: the
ISC current bypasses the ammeter and is therefore not integrated, so
the SOC estimate stays on the healthy trajectory regardless of the
cell's internal state — exactly the property that lets the residual
encode the fault. This matches the methodology used by Shen et al.
2023 in the original paper that released this dataset.

The dataset's `SOC|DOD/%` and `capacity_Ah` columns are still rendered
in `visualize-dataset` (the per-step DOD pattern is itself a useful
sanity check) but they do not enter the model in any way.
