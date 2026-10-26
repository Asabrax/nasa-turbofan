# NASA predictive maintenance — all engines

Predict remaining useful life from turbofan sensor histories using XGBoost,
operating-regime normalization and causal temporal features. The final model
achieves **22.63-cycle RMSE** on 707 official test engines, with explicit checks
of engine separation, sensor redundancy, alert uncertainty and long-horizon limits.

![Official test error by dataset](outputs/presentation/figure_03.png)

## Setup after cloning

Read the executed notebooks directly, or download and open `presentation.html`
in a browser for the styled report. No training is required to view the results.

To run the Python workflow, start in the repository root:

```bash
python3 -m venv nasa
./nasa/bin/python -m pip install -r requirements.txt
./nasa/bin/python scripts/verify_robustness.py
```

The recorded runs used Python 3.14 on Linux/WSL2. Package versions from the
baseline run are recorded in `outputs/all_engines/environment.txt`; the dependency
file specifies version ranges, not an exact environment lock. On Windows,
use `nasa\Scripts\python.exe` in place of `./nasa/bin/python`.

The local virtual environment and caches are ignored. Saved data, models,
metrics and executed reports remain included so the linked evidence is available.
NASA dataset provenance and input hashes are retained in `data/source.json`.

## Present this version

- **[explore.ipynb](explore.ipynb)** — the main presentation: decision, evidence,
  validation, sensor selection, improvements, final results and limitations.
- **[presentation.html](presentation.html)** — browser-friendly version with
  section navigation, embedded figures and code collapsed by default.
- **[baseline.ipynb](baseline.ipynb)** — preserved original baseline and EDA.
- **[experiments.ipynb](experiments.ipynb)** — detailed experiment evidence.

The presentation reads existing results; it never retrains models on opening.
It contains 12 executed code cells and seven visually inspected charts. The
complete HTML page has not been browser-inspected because this environment has
no browser installed; local links and section anchors were checked programmatically.

### Completed robustness checks

Both fixed recipes were trained locally on five stratified engine splits in
101.7 seconds using six CPU threads. The all-engine final model and calibrated
threshold were preserved; there was no further selection on the official test.

- The final recipe improves RMSE in all five splits for each tested 100, 150,
  200 and 250-cycle planning window. The 200-cycle gain is 8.31–12.14 cycles.
- Broad lifetime RMSE worsens in four of five splits. The capped model remains
  unsuitable for precise long-horizon RUL estimation.
- Calibrated official-test recall is 93.7% with a 95% Wilson interval of
  88.8–96.5%; precision is 92.5% [87.4–95.7%]. These are conditional sampling
  intervals, not guarantees or estimates of deployment performance.
- Full held-out history replay places the calibrated rule's median first alert
  32–34 cycles before failure across splits, versus 27.5–30 at threshold 30.
  It is passive replay, without modeled inspections or interventions.

Split populations overlap and some engines informed earlier selection, so these
are sensitivity checks, not independent blind replications. First-alert timing
and endpoint recall answer different questions. The report separates alerts
31–60 cycles before failure from those more than 60 cycles early.

[Methods and reproduction](outputs/robustness/METHODS.md),
[training log](outputs/robustness/training.log), and
[split results](outputs/robustness/split_metrics.csv) retain the evidence.
`./nasa/bin/python scripts/verify_robustness.py` independently checks the frozen
model hash, split membership, all 50 endpoint metrics, 2,130 first-alert records,
Wilson calculations, notebook execution and HTML links/table headers/images.
The notebook also includes the original baseline alert row on the same official
test population, paired RUL scatter plots, and a clarified source footer.

### Sensor answer

All 21 sensors are loaded, but **17 sensor channels are retained by the final
model**. After regime normalization, fold-fitted variance filtering removes
**1, 5, 18 and 19**, including all seven features belonging to each sensor.
The final model therefore has **127 inputs**, down from 155 candidate features.

Sensors **6, 10 and 16** have small but nonzero within-regime variation in parts
of the pooled data. A fresh development-only ablation removed them too, leaving
14 sensors. With the same fixed model settings and three development folds:

| Sensor set | Primary CV RMSE | Broad CV RMSE |
| --- | ---: | ---: |
| Current 17 channels | 25.932 | 45.317 |
| Reduced 14 channels | 25.958 | 45.216 |

These tiny differences reverse across protocols. They do not establish a
meaningful winner or prove the three channels are useless. The validated final
model and official test results are unchanged. Full diagnostics are in
`outputs/presentation/sensor_audit.csv` and `sensor_ablation.csv`.

Rebuild and check the presentation:

```bash
./nasa/bin/python scripts/build_notebook.py
./nasa/bin/python scripts/execute_notebook.py
./nasa/bin/python scripts/preview_notebook.py
```

Repeat the sensor diagnostic only when needed:

```bash
./nasa/bin/python scripts/audit_sensors.py
```

## Latest: controlled improvement experiments

Open **[experiments.ipynb](experiments.ipynb)** for the executed follow-up. The
original all-engine baseline remains in `baseline.ipynb` and `outputs/all_engines/`.

**21 configurations were trained locally in 386 seconds.** The selected model
uses a 150-cycle training target cap, fold-fitted regime normalization, richer
causal history, and XGBoost with depth 6 / 800 trees / learning rate 0.03. It keeps
cycle, uses pooled training, and uses ordinary squared error. Engine weighting,
cycle removal, asymmetric loss and separate models were evaluated but did not
win the declared selection criterion. This is a bounded search, not a claim of
global optimality.

| Official test metric | Previous best Random Forest | Selected experiment |
| --- | ---: | ---: |
| RMSE (cycles; true uncapped labels) | 33.14 | **22.63** |
| MAE (cycles) | 23.26 | **16.23** |
| Mean error, predicted minus actual (cycles) | +11.94 | **+0.31** |

Test RMSE decreased **31.7%**. Per-dataset RMSE is FD001 **17.84**, FD002 **24.47**,
FD003 **18.28**, FD004 **23.91** cycles. Selection uses only development-engine
CV; official labels are read after configuration and threshold freeze. The old
test results had already been seen, so this is not a new blind benchmark.

### Model and alert progression

All rows use the same 707 official test endpoints and true uncapped RUL labels.
The maintenance event is actual RUL ≤30 cycles; the alert threshold applies to
predicted RUL. The original Random Forest is included as the previous best test
RMSE comparator; the original XGBoost is the starting model for the experiments.

| Model / decision rule | Alert threshold (cycles) | RMSE (cycles) | Recall | Precision |
| --- | ---: | ---: | ---: | ---: |
| Original Random Forest | 30.00 | 33.14 | 82.4% | 94.2% |
| Original uncapped XGBoost | 30.00 | 33.42 | 81.8% | 94.9% |
| Tuned cap-150 + regime/rich history | 30.00 | 22.63 | 89.9% | 94.7% |
| Same tuned model + calibrated alert | 34.44 | 22.63 | 93.7% | 92.5% |

Calibration changes the alert rule, **not the RUL predictions or RMSE**. Its
34.44-cycle threshold was selected using out-of-fold development predictions.
The per-dataset test table and grouped chart are in Section 7 of
[experiments.ipynb](experiments.ipynb). They show the combined pipeline gain;
component comparisons are reported separately in the validation experiments.

An untouched 142-engine audit supports the improvement within the assumed
200-cycle planning window: RMSE **33.01 -> 25.67**, with a paired engine-bootstrap
95% improvement interval of **4.34–10.59 cycles**. **Tradeoff:** broad validation
including earlier life worsened from **39.80 -> 43.32**. Target capping is not a
uniform improvement over every remaining lifetime; that limitation stays visible
in the notebook and metric tables.

For an actual 30-cycle maintenance event, the OOF-selected prediction threshold
is **34.44 cycles**. Test recall is **93.7%**, precision **92.5%**, with **10 missed
near-failure engines and 12 false alerts**. Audit primary recall was 90%; this is
not an operational guarantee. At the original threshold of 30, test recall is
89.9% and precision 94.7%.

Run or reproduce:

```bash
./nasa/bin/python scripts/check_experiments.py
./nasa/bin/python scripts/run_experiments.py --threads 6
./nasa/bin/python scripts/verify_experiment_results.py
./nasa/bin/python scripts/execute_notebook.py experiments.ipynb
./nasa/bin/python scripts/preview_notebook.py experiments.ipynb
```

`outputs/experiments/selected_model.joblib` contains the fitted normalization,
causal feature builder, selectors and XGBoost model. The directory also holds the
training log, all candidate configurations, CV assignments, OOF predictions,
frozen selection, audit predictions/intervals, test predictions, NASA scores and
threshold results. See `outputs/experiments/METHODS.md` for the protocol and
source references. Old model artifacts are preserved.

Validation passed: causal/future/target isolation, engine separation, custom-loss
derivatives and weighted fitting, score signs, independent endpoint/metric/alert
recomputation, and serialized-model prediction equality. All **10 notebook code
cells** executed; all **seven charts** were visually inspected. The complete
HTML page layout was not browser-inspected.

## Original all-engine baseline

Open **baseline.ipynb** for the executed course-style workflow based on
`../data_aai_sose26`: business understanding, quality checks, EDA, causal feature
engineering, pipelines, grouped cross-validation, model comparison and an
illustrative maintenance flag.

The current experiment uses **all 160,359 training rows from 709 engines** across
FD001–FD004. All **707 test engines** remain held out. The actual FD004 files
contain 249 training and 248 test engines; the bundled data README reverses
these counts. No training rows are sampled away.

Models: mean baseline, Ridge, Random Forest, and **XGBoost**. Compare XGBoost
max depths 3 and 6 using three-fold engine-grouped CV, then refit the selected
candidate from each family on the entire training set. The other model
parameters are fixed starting points. Dataset indicators and operating settings
are included; histories use dataset-plus-engine identifiers to avoid collisions.

## Run training on this machine

From the repository root:

```bash
./nasa/bin/python scripts/train_all.py --threads 6
```

This performs actual CPU training in the project's WSL2 Python environment.
XGBoost uses histogram trees. It logs fit progress and timing to
`outputs/all_engines/training.log`. A run overwrites that experiment's results.

To execute the notebook and refresh its summary/HTML preview:

```bash
./nasa/bin/python scripts/execute_notebook.py baseline.ipynb
./nasa/bin/python scripts/preview_notebook.py baseline.ipynb
```

The baseline notebook loads completed local training results by default. Set
`RETRAIN = True` in its setup cell to train again when running all cells. If
results are missing, it trains automatically. The IPython runner saves executed
cell outputs without requiring nbclient. In a standard Jupyter environment,
install `requirements.txt`, select that interpreter, and run all cells.

## Saved artifacts

All current artifacts are in `outputs/all_engines/`:

- `results.json`: selected model, counts, versions, platform and elapsed times.
- `training.log`: timestamped record of every validation and final fit.
- `model_comparison.csv`, `cv_folds.csv`, `fold_assignments.csv`: model selection
  evidence and engine-level validation membership.
- `test_metrics.csv`: overall and per-dataset metrics for all four model families.
- `test_predictions.csv`: predictions for all 707 test engines, for each family.
- `mean_baseline.joblib`, `ridge.joblib`, `random_forest.joblib`, `xgboost.joblib`:
  actual fitted pipelines, feature order and configuration.
- `xgboost_booster.ubj`: native XGBoost booster; preprocessing is in the pipeline.
- `figure_*.png`, `notebook_preview.html`: notebook figures and readable preview.
- `environment.txt`: installed package versions.

Use `scripts/train_all.py::make_features` on complete available histories before
calling a saved pipeline. The function requires dataset, engine ID, engine key,
cycle, settings and sensors. Select the latest row and the saved feature order.
The notebook demonstrates inference and checks it against the saved predictions.
Only load trusted joblib files.

`explore_fd001.ipynb` and the original files directly under `outputs/` preserve
the previous FD001-only experiment. `scripts/build_fd001_notebook.py` regenerates
that old notebook. `scripts/build_baseline_notebook.py` regenerates `baseline.ipynb`.
`scripts/build_notebook.py` regenerates the main presentation, `explore.ipynb`.
Both clear their target notebook outputs; neither trains models.

## Original baseline interpretation

RUL is uncapped and predictions are clipped at zero. Preprocessing is learned
within each fold. Official test targets are read only after model selection and
final fits. Pooled test metrics give each engine equal weight, whereas training
CV scores all cycles and therefore weights longer histories more. Dataset
indicators are benchmark context; they may not exist for a new real-world fleet.
The original rolling features mix operating conditions; the completed follow-up
experiment above adds regime normalization. The 30-cycle alert is illustrative, not a validated
maintenance policy. This benchmark does not demonstrate deployment readiness.

## Original baseline local run

Training completed in 136 seconds on the project's WSL2 environment, using
six threads for Random Forest and XGBoost. Grouped CV selected XGBoost depth 6
(47.56-cycle CV RMSE), narrowly ahead of Random Forest (47.59).

| Model | Test RMSE (cycles) | Test MAE (cycles) |
| --- | ---: | ---: |
| Random Forest | 33.14 | 23.26 |
| XGBoost | 33.42 | 23.61 |
| Ridge | 40.13 | 32.03 |
| Mean baseline | 65.43 | 54.39 |

All 11 notebook code cells executed successfully. All 20 overall/per-dataset
metric rows were independently recomputed, and raw test-label alignment was
checked for every engine. All four serialized pipelines reproduced their saved
predictions after reloading. The five exported charts were visually inspected;
the full HTML page layout was not browser-inspected.
