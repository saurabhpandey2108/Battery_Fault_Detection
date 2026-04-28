# Battery Fault Detection — Three-Scalogram Residual Analysis

End-to-end internal-short-circuit (ISC) fault detector for the **Tsinghua
NCM811** dataset. Three stages, run in order:

1. **PINN** — pure-NumPy MLP, 35 leakage-free current-only features →
   8 ECM parameters, trained on **healthy (BD) cells only** and then
   frozen. Predicts `V_pred(t)`.
2. **CWT** — Morlet continuous wavelet transform on three signals per
   1280-sample window: `V_meas`, `V_pred`, and the residual
   `e(t) = V_meas − V_pred`. Stacked into a `(224, 224, 3)` image.
3. **CNN** — binary classifier (BD vs CS) on the three-scalogram stack,
   stratified file-level 70/15/15 split.

The PINN never sees fault data, so on a CS (faulty) file it still
predicts the healthy voltage trajectory; `V_meas` carries the ISC
signature, and the **residual concentrates the disagreement**. The CNN
learns from that disagreement, not from raw voltage.

---

## Quick start

```bash
# 0. install (Python 3.10+ recommended)
pip install -r requirements.txt

# 1. eyeball-check the data BEFORE training
python main.py visualize-dataset

# 2. enumerate Phase-1 BD/CS file pairs
python main.py list-phase1

# 3. train the leakage-free PINN on healthy BD files
python main.py train-pinn                        # 100 epochs default
python main.py train-pinn --epochs 50 --patience 8

# 4. acceptance gate: residual must distinguish BD from severe ISC
python main.py verify

# 5. (optional) eyeball single-file and BD-vs-CS scalograms
python main.py infer   --file dataset/NCM811_ISC_TEST/DST/ISC_CS_0.5CC_DST_10ohm.csv
python main.py compare --ohm 10
python main.py scalogram-similarity              # summary plot once compare ran for several ohms

# 6. train the binary classifier on the three-scalogram stack
python main.py train-classifier

# 7. predict on one CSV
python main.py predict --file dataset/NCM811_ISC_TEST/DST/ISC_CS_0.5CC_DST_10ohm.csv
```

`train-pinn` and `train-classifier` bypass argparse — every flag after
the subcommand is forwarded to the underlying script (`--epochs`,
`--learning-rate`, `--patience`, `--batch-size`, `--mode`, …).

---

## CLI subcommands

| Subcommand              | What it does                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------- |
| `list-phase1`           | List the 14 BD/CS resistance pairs in the Phase-1 subset (0.5C charge, DST discharge).      |
| `visualize-dataset`     | 4-panel inspection PNG per representative file + comparison overlay (Task 0 sanity check). |
| `train-pinn`            | Train the healthy-only PINN. Saves `models/pinn_healthy_no_leak.npz`.                       |
| `verify`                | Acceptance gate — runs the frozen PINN on BD/CS, checks the drift signature.                |
| `infer`                 | Render a `2×3` three-scalogram diagnostic figure for one CSV.                               |
| `compare`               | Render a `2×4` BD-vs-CS comparison figure at one resistance.                                |
| `scalogram-similarity`  | Plot SSIM / Pearson r vs ohm from the appended similarity CSV.                              |
| `train-classifier`      | Train the binary three-scalogram CNN (or, with `--mode soc-regress`, an SOC-regression sanity head). |
| `predict`               | Run the frozen classifier on one CSV, print mean / max / per-window probabilities and verdict. |

---

## Repo layout

```
src/
  config/
    settings.py                    # SINGLE source of truth for every constant
                                   #   (paths, NCM811 OCV, PINN/CWT/CNN config,
                                   #    training hyperparameters)
    __init__.py                    #   re-exports settings.* via star-import
  data/
    tsinghua_loader.py             # two-block UTF-8-BOM CSV parser, Phase-1 helpers
    prepare_pinn_features.py       # build_leakage_free_features + coulomb_counted_soc
                                   #   (the leakage gate; carries _FORBIDDEN sentinel)
  pinn/
    battery_physics.py             # BatteryModel.calculate_voltage (8 free params, no clipping)
    pinn_v2.py                     # active 35→64×3→8 MLP, Adam + grad-norm clip
    deep_neural_network.py         # legacy from-scratch DNN — kept for reference, not used
    data_loader.py                 # legacy BatteryDataProcessor — kept for reference, not used
  wavelet/
    cwt_utils.py                   # Morlet CWT + COI mask (pure NumPy)
    image_utils.py                 # preprocess_window (cubic / mean / none),
                                   #   raw_log_scalogram, normalize_and_resize, stack_channels
    preprocess.py                  # estimate_fs, create_windows, normalize_array
    plot_utils.py                  # plot_frequency_scalogram (per-channel detrend)
    model.py                       # build_cnn_model (head="binary"|"multiclass"|"regression")
    scalogram_metrics.py           # SSIM + Pearson r utilities + similarity-CSV plotter
  scalograms/
    three_scalogram_builder.py     # build_three_scalogram_image, build_all_windows
                                   #   (per-channel detrend wired in here)
  inference/
    infer_pinn.py                  # frozen PINN → V_pred(t), state threaded per file
                                   #   (numerical safety net substitutes V_meas + resets ir1/ir2
                                   #    on |V|>10 V or non-finite)
  training/
    train_pinn_healthy.py          # window-based training on BD only, file-level state threading
    train_classifier.py            # 70/15/15 file-level split, binary CNN (or SOC regressor)
  verification/
    verify_residual_signal.py      # drift-aware acceptance gate
  visualization/
    visualize_dataset.py           # Task-0 inspection figures
    visualize_three_scalograms.py  # 2×3 diagnostic PNG per file
    compare_bd_cs.py               # 2×4 BD-vs-CS side-by-side at one resistance

main.py                            # CLI dispatcher
dataset/                           # user-supplied Tsinghua data (gitignored)
models/                            # pinn_healthy_no_leak.npz, fault_classifier.keras
cache/windows_phase1/              # per-file (image, label, soc_window) caches
results/
  pinn/                            #   train_history.csv, residual_verification.png,
                                   #   inference_dump.csv, verify.log
  cnn/                             #   train_history.{csv,png}, test_metrics.json,
                                   #   test_predictions.csv, test_per_file.csv,
                                   #   confusion_matrix.png
                                   #   + soc_* counterparts when --mode soc-regress
  three_scalograms/                #   per-file 2×3 diagnostic + bd_vs_cs_*ohm.png +
                                   #   similarity.csv + similarity.png
  dataset_inspection/              #   Task-0 figures
_sources/                          # upstream Battery_Passport + Wavelet_Analysis clones
                                   #   (gitignored, kept for reference)
```

---

## Hard architectural invariants

These are not stylistic preferences — each one is a correctness condition.
Breaking any of them silently destroys the diagnostic.

1. **Leakage-free PINN inputs.** No voltage of any kind (V_meas, OCV(SOC),
   dV/dt, voltage stats) ever enters the PINN feature vector. Inputs are:
   `I, I_window(30), I_mean_long(300), I_rms_long(300), SOC_coulomb, dI/dt`
   = **35 features**. The forbidden columns are listed in
   `prepare_pinn_features._FORBIDDEN` as a sentinel.

2. **PINN is trained on BD files only — never on CS.** Selection lives in
   `train_pinn_healthy.load_all_bd_files` via
   `list_phase1_files(..., class_="BD")`. If CS leaks in, the residual
   collapses to noise on both classes.

3. **No bounds on the 8 physics parameters.** Every `np.clip()` on
   C1, C2, R0, R1, R2, gamma1, M0, M inside
   `BatteryModel.calculate_voltage` has been removed. SOC is still
   clipped to `[0, 100]` and an `eps = 1e-10` safety factor is kept so
   OCV interpolation stays in-domain. Adam grad-norm clip operates on the
   *gradient*, not on the parameters or outputs.

4. **State threading across timesteps.** `(ir1, ir2, z, h, s)` thread
   through every timestep within a single file and reset only at file
   boundaries. The trainer threads state across windows too — Adam fires
   per window, but the physics rollout sees one continuous trajectory.

5. **`sampling_time = 1.0 s`.** Tsinghua data is 1 Hz; never use the
   upstream Battery_Passport default of `1e-4`.

6. **NCM811 OCV curve.** `BATTERY_CONFIG["ocv_curve"]` spans 2.8 → 4.2 V.
   The legacy LFP-like curve in `BatteryModel.setup_ocv_curve` defaults
   caps at ~3.5 V and would structurally prevent `V_pred` from reaching
   `V_meas` on NCM811.

7. **Dataset `SOC|DOD/%` column is unusable.** The Arbin tester resets
   it at every DST step transition (verified empirically — 277 single-
   sample >50% jumps in one BD file alone). We instead coulomb-count
   from the external current with `Q_RATED_AH = 2.5487 Ah` (median net
   discharged Ah across the 14 BD DST 0.5C files).

8. **Window / stride / scales for CWT.** `WINDOW_SIZE = 1280`,
   `STRIDE = 256`, `SCALES = np.geomspace(2.0, 300.0, 96)`,
   `IMAGE_SHAPE = (224, 224, 3)`. Channels are
   `[V_meas, V_pred, residual]`; current is **not** a channel.

9. **File-level train/val/test split.** 70/15/15 stratified by class at
   the file level — windows from one file never straddle splits.

10. **Hyperparameter defaults live in `src/config/settings.py`.** Every
    `PINN_*` and `CNN_*` constant is the single source of truth; the
    training scripts use them as both argparse defaults and function-arg
    defaults so a CLI flag can override per-run without forking the code.

---

## Stage 1 — the leakage-free PINN

### Why voltage cannot be an input

If `V_meas` is one of the PINN inputs, the network learns the inverse
physics map `params = g⁻¹(V_meas, I)` — given any voltage shown to it,
it produces 8 parameters whose physics rollout reproduces that voltage.
The training loss `‖V_pred − V_meas‖` actively pushes it there. On a CS
cell, `V_meas` already carries the ISC signature, so the network outputs
params that reproduce the faulty voltage and `V_pred ≈ V_meas` — the
residual collapses to noise on both classes and the diagnostic
disappears. Empirically we observed BD-vs-CS residual ratios of ~1.05×
across all 14 resistance pairs with V in the input.

The fix is to feed the network only inputs that **cannot** encode the
fault:

* `I(t)` — instantaneous current (1)
* `I_window(30)` — last 30 s of current at 1 Hz (30)
* `I_mean_long(300)`, `I_rms_long(300)` — ~5 min rolling stats (2)
* `SOC_coulomb(t)` — `1.0 + cumtrapz(I_signed, t) / (Q_rated · 3600)`,
  clipped to `[0, 1]` (1)
* `dI/dt(t)` — first difference of `I`, prepended with 0 (1)

Total: **35 features per timestep**. None of these can reproduce the
ISC-induced voltage drop, because the ISC current bypasses the ammeter.
The only way to reduce loss during training is to genuinely learn the
`I → V` mapping of a healthy cell. At inference time on a CS cell, the
hidden ISC current never reaches the ammeter, the network keeps
predicting the healthy voltage trajectory, and `V_meas` falls below
`V_pred` — producing the structured negative-mean residual that the
fault detector keys on.

### Coulomb-counted SOC

The Tsinghua `SOC|DOD/%` column is per-step Arbin DOD, not a global SOC.
A single `coulomb_counted_soc` site in
`src/data/prepare_pinn_features.py` integrates the external current
with the dataset's sign convention (negative = discharge),
`Q_rated = 2.5487 Ah`, and clips the result to `[0, 1]`. This is
**fault-blind by construction**: the ISC current bypasses the ammeter,
so SOC stays on the healthy trajectory regardless of the cell's
internal state — exactly the property that lets the residual encode the
fault. The dataset's `SOC|DOD/%` and `capacity_Ah` columns are still
rendered in `visualize-dataset` (the per-step DOD pattern itself is a
useful sanity check) but they never enter the model.

### PINN architecture and training

* **Network:** `35 → 64 → 64 → 64 → 8`, ReLU hidden, **linear output**
  (the 8 ECM parameters stay free). Pure NumPy implementation in
  `src/pinn/pinn_v2.py`. Output bias is **warm-started** to
  `target_values` from `BATTERY_CONFIG` so the very first forward pass
  is in a numerically stable physics regime.
* **Window-based training (Path B).** For each BD file we walk
  non-overlapping 256-sample windows. Each window: forward NN →
  rollout physics with state threaded from the previous window →
  RMSE vs `V_meas` → one Adam step using true `dV/dp_k` Jacobians from
  central finite differences on `calculate_voltage`. Adam fires ~50 ×
  per file per epoch (vs once per file in the old file-level path),
  while the physics remains continuous across the entire file.
* **Stability measures.**
  * Numerical safety net: if a rollout step produces non-finite values
    or `|V| > 10 V`, that step's `v_pred` is replaced with `v_meas` and
    `ir1, ir2` reset to 0. The corresponding row of the Jacobian is
    masked to zero before the gradient step so a single bad timestep
    cannot drag the rest of the window.
  * **Soft physics-realism penalty.** A small per-step penalty pushes
    `R/C/τ` back toward the stable regime (`relu(-R)`, `relu(-C)`,
    `relu(τ_min − R·C)`) but does **not** clip the parameters. Set
    `--phys-lambda 0` to disable.
  * Adam **gradient-norm clip** (default 1.0) on the global gradient
    flat across all params.
* **Early stopping** on the moving-average val RMSE (default window 8
  epochs, patience 12) — robust against single spike epochs.

Defaults live in `src/config/settings.py`:

| Constant                  | Default | Notes                                    |
| ------------------------- | ------- | ---------------------------------------- |
| `PINN_EPOCHS`             | 100     |                                          |
| `PINN_LEARNING_RATE`      | 1e-4    |                                          |
| `PINN_PATIENCE`           | 12      | on val-RMSE moving average               |
| `PINN_CLIPNORM`           | 1.0     | grad clip, not param clip                |
| `PINN_TRAIN_WINDOW`       | 256     | samples per Adam update                  |
| `PINN_TRAIN_STRIDE`       | 256     | non-overlapping                          |
| `PINN_VAL_FRACTION`       | 0.2     | of BD files, file-level                  |
| `PINN_VAL_MA_WINDOW`      | 8       | epochs of val-RMSE moving average        |
| `PINN_FD_EPS_ABS`         | 1e-6    | finite-difference Jacobian step floor    |
| `PINN_PHYS_PENALTY_LAMBDA`| 0.01    | soft realism penalty (0 to disable)      |
| `PINN_PHYS_TAU_MIN`       | 1e-3 s  | minimum stable RC time constant          |

### Outputs (Stage 1)

* `models/pinn_healthy_no_leak.npz` — frozen weights with an `__arch__`
  sentinel; loaders refuse to read a file without it.
* `results/pinn/train_history.csv` — per-epoch metrics: train RMSE, val
  RMSE, val-MA, n_state_resets, mean of each of the 8 physics params,
  epoch wall time.
* `results/pinn/inference_dump.csv` — per-sample dump on every BD file
  (`time, current, soc_coulomb, V_meas, V_pred, C1..M`).

---

## Stage 2 — verification gate

Before training the classifier, the pipeline asks: **does the residual
actually separate BD from severe ISC?** Run:

```bash
python main.py verify
```

This loads the frozen weights and runs them on three representative
files: `BD_100ohm` (healthy), `CS_10ohm` (severe ISC), `CS_1000ohm`
(mild ISC). For each, it computes:

* raw residual stats (RMS, max-abs, mean) — diagnostics only
* 60-second moving-average lowpass (smoothed mean / smoothed RMS)
* **drift slope** by linear regression of residual vs `SOC_coulomb`

The PINN's residual is dominated by counter-phase oscillation against
DST current pulses (a model-fit artefact), so RMS is mostly noise. The
fault signal lives in the **slow drift** of the residual versus SOC: as
the ISC bleeds hidden charge, `V_meas` falls below `V_pred` and the
residual mean trends more negative the further the cell has discharged.

Acceptance criteria (must all pass):

1. **Drift gap.** `slope(CS_10) − slope(BD) ≥ 100 mV per unit-SOC`
2. **Smoothed-mean gap.** `BD smoothed mean − CS_10 smoothed mean ≥ 10 mV`
3. **Internal-drain gap.** `CS_10 end SOC − BD end SOC ≥ 30 percentage points`
   (sanity check that the dataset itself shows the ISC drain forced the
   cell to hit cutoff with charge unaccounted for in the ammeter)
4. All three runs finite

CS_1000 is intentionally **not** in the ladder — `4 V / 1000 Ω ≈ 4 mA`
× 3.5 h discharge ≈ 12 mAh ≈ 0.5 % of cell capacity, which is below the
PINN's residual noise floor and physically imperceptible. Mild ISC is a
Phase-2 problem.

Outputs:

* `results/pinn/residual_verification.png` — 4-column-per-file figure
  (V overlay, SOC trajectory, residual over time, residual vs SOC with
  fitted drift line)

The script exits non-zero if any criterion fails.

---

## Stage 3 — three-scalogram CNN

### Per-channel detrend (critical, easy to miss)

`preprocess_window` in `src/wavelet/image_utils.py` detrends the window
before every CWT. The original Wavelet_Analysis path always used a
**cubic** detrend + z-score, which is right for `V_meas` and `V_pred` —
without it, the ~1 V NCM811 OCV envelope across a 1280-sample window
saturates the lowest scales of the scalogram and crowds out the fast
features.

It is **wrong for the residual channel.** The ISC fault signature is
precisely the slow drift of the residual versus SOC (verify-gate
measures `+346 mV / unit-SOC` on CS_10 vs `-6` on BD), and a cubic fit
absorbs that drift almost entirely. With cubic detrend on the residual,
binary BD-vs-CS classification collapses to BCE 0.69 (predict 0.5);
the SOC-from-`V_pred` regression sanity-check collapses by the same
mechanism (R² ≈ 0.18, predictions stuck near 0.5). Both failures point
at the same cause: cubic deletes slow-trend information.

`build_three_scalogram_image` therefore wires:

* `V_meas` → `detrend="cubic"`
* `V_pred` → `detrend="cubic"`
* residual → `detrend="mean"` (mean-subtract only; the slow drift survives)

The two visualisation paths (`compare_bd_cs.py`,
`visualize_three_scalograms.py`) make the same per-channel choice so
the figures show what the CNN actually sees.

> **Cache invalidation.** Changing the residual detrend invalidates
> `cache/windows_phase1/`. Delete it before retraining. The PINN
> weights at `models/pinn_healthy_no_leak.npz` are upstream of the
> scalograms and are *not* affected.
>
> **Per-window SNR ceiling — known limitation.** Even with the fix,
> per-window CS_10 drift ≈ 18 mV against oscillation std ≈ 23 mV →
> drift-SNR ≈ 1. The verify gate works because it integrates over the
> whole file (~8000 samples, drift-SNR ≈ 5). The per-window CNN approach
> is therefore SNR-limited under the `WINDOW_SIZE = 1280` invariant.

### CNN architecture (`build_cnn_model`)

```
Input (224, 224, 3)
  ↓ GaussianNoise(σ = noise_std)
  ↓ Conv2D(32) → BN → ReLU → MaxPool(2)
  ↓ Conv2D(64) → BN → ReLU → MaxPool(2)
  ↓ Conv2D(128) → BN → ReLU → MaxPool(2)        # 28×28×128
  ↓ AveragePooling2D(4)                          # 7×7×128
  ↓ Flatten → Dropout(p)
  ↓ Dense(64, ReLU) → Dropout(p/2)
  ↓ head:
       binary      → Dense(1, sigmoid),   BCE,  metrics=[acc, AUC]
       multiclass  → Dense(K, softmax),   sparse CCE
       regression  → Dense(1, linear),    MSE,  metrics=[mae, rmse]
```

* The `AveragePool(4) → Flatten` step shrinks the head bottleneck from
  ~100 k features to 6.3 k (~16× parameter reduction) while preserving
  spatial structure. Plain `GlobalAveragePooling` washed out the
  residual scalogram's localized ISC signature and the model collapsed
  to predicting 0.5.
* L2 weight decay on Conv2D + Dense kernels (`CNN_CONFIG["l2_lambda"]`).
* Optimiser: Adam(lr=1e-4, clipnorm=1.0).
* Early stopping on `val_auc` (max) for classification, `val_rmse`
  (min) for regression. `ReduceLROnPlateau` on `val_loss`.

### Training, evaluation, and outputs

`train_classifier.py`:

* Walks every BD/CS Phase-1 pair, runs the frozen PINN per file, builds
  the per-window stack with `build_all_windows`, caches each file's
  (images, labels, soc_window) under `cache/windows_phase1/`.
* Stratified 70/15/15 file-level split (windows from a file never
  straddle splits).
* `--mode classify` (default) → binary BD/CS head.
* `--mode soc-regress` → V_pred-only channel, per-window mean SOC as
  target. This is a sanity head: a healthy CNN should be able to
  recover SOC from `V_pred` alone — if it cannot, the upstream
  representation is corrupted (this is the failure mode that surfaced
  the cubic-detrend bug).

Test-set artefacts under `results/cnn/`:

* `train_history.{csv,png}` — per-epoch loss / accuracy / AUC,
  best-epoch line marked.
* `test_metrics.json` — Keras metrics + window-level and **file-level**
  precision / recall / F1 / accuracy.
* `test_predictions.csv` — one row per test window.
* `test_per_file.csv` — one row per test file with mean / max prob and
  verdict (mean prob ≥ 0.5 → faulty).
* `confusion_matrix.png` — side-by-side window and file confusion
  matrices.

For `--mode soc-regress`, the same artefacts are emitted with a `soc_`
prefix plus `soc_predicted_vs_actual.png` (true-vs-predicted scatter,
per-class colours, with the y=x line).

CNN defaults (`src/config/settings.py`):

| Constant                              | Default |
| ------------------------------------- | ------- |
| `CNN_CONFIG["epochs"]`                | 60      |
| `CNN_CONFIG["batch_size"]`            | 64      |
| `CNN_CONFIG["early_stop_patience"]`   | 12 (on val_auc max) |
| `CNN_CONFIG["noise_std"]`             | 0.10    |
| `CNN_CONFIG["learning_rate"]`         | 1e-4    |
| `CNN_CONFIG["l2_lambda"]`             | 1e-5    |
| `CNN_CONFIG["dropout_rate"]`          | 0.5     |

---

## Diagnostic visualisations

* **Single file** — `python main.py infer --file <CSV>` renders a 2×3
  PNG (top: V overlay + I, residual; bottom: V_meas / V_pred / residual
  scalograms with COI mask). Scalogram pairwise SSIM and Pearson r are
  printed and appended to `results/three_scalograms/similarity.csv`.
* **BD-vs-CS at one resistance** — `python main.py compare --ohm 10`
  renders a 2×4 figure (V overlay + three scalograms, BD on top,
  CS below). dB ranges are computed jointly across both files so
  brightness is directly comparable. Same per-channel detrend rule as
  the CNN. Both rows are appended to `similarity.csv`.
* **Similarity summary** — `python main.py scalogram-similarity`
  renders a 2×3 summary (SSIM and Pearson r vs ohm, BD vs CS,
  diagnostic direction annotated per pair). Diagnostic expectations:
  * `meas-vs-pred`: HIGH on BD (PINN tracks healthy), LOWER on CS.
  * `meas-vs-resid`: LOW on BD (residual is noise), HIGHER on CS
    (residual inherits V_meas's structure as V_pred drifts).

---

## Dataset

Phase 1 is `0.5C charge / DST discharge`, 14 BD/CS resistance pairs at
`{10, 20, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000}` Ω.
Files live under:

```
dataset/NCM811_NORMAL_TEST/DST/   # BD (healthy)
dataset/NCM811_ISC_TEST/DST/      # CS (faulty)
```

Filenames are `ISC_<class>_<rate>CC_<mode>_<ohm>ohm.csv`, e.g.
`ISC_BD_0.5CC_DST_100ohm.csv`. Each CSV is UTF-8-BOM with two 5-column
blocks (charge | empty separator | discharge), Chinese headers,
sampled at exactly 1 Hz.

The `_sources/` directory contains the upstream **Battery_Passport**
(physics + DNN scaffolding) and **Wavelet_Analysis** (CWT pipeline)
clones this project was merged from. Both are gitignored and kept only
for reference.

---

## Notes

* TensorFlow is imported only when `train-classifier` or `predict`
  runs. The non-TF modules (loaders, PINN, CWT, visualisations,
  verification gate) work without it.
* `BatteryModel`'s default capacity is `10 Ah` from upstream
  Battery_Passport. Tsinghua NCM811 cells are smaller (~2.55 Ah). Pass
  `capacity_as=` when constructing `BatteryModel` or calling
  `predict_voltage_series` if you need physical SOC tracking to match
  the real cell. The model default is fine for residual-as-diagnostic
  (the network learns parameters that compensate).
* When in doubt about whether a feature reintroduces leakage, run
  `python main.py verify`. If the drift gap or smoothed-mean gap
  collapses, voltage has snuck back into the input pipeline.
