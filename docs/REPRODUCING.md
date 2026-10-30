# Reproducing the study

## Script guide

| Area | Scripts | Purpose |
| --- | --- | --- |
| Data setup | [`prepare_data.py`](../scripts/prepare_data.py) | Import the twelve checksum-verified dataset files. |
| Training and experiments | [`train_all.py`](../scripts/train_all.py), [`run_experiments.py`](../scripts/run_experiments.py), [`run_sequences.py`](../scripts/run_sequences.py), [`train_tcn.py`](../scripts/train_tcn.py) | Train the original baselines and compare tree and sequence models. |
| Shared implementation | [`experiment_models.py`](../scripts/experiment_models.py), [`cmapps/`](../scripts/cmapps/) | Normalization, temporal features, estimators, evaluation and artifact helpers. |
| Additional experiments | [`audit_sensors.py`](../scripts/audit_sensors.py), [`run_robustness.py`](../scripts/run_robustness.py), [`run_tcn_followup.py`](../scripts/run_tcn_followup.py), [`run_regularization.py`](../scripts/run_regularization.py) | Compare sensor sets and TCN settings; evaluate split sensitivity, planning windows and alert policies. |
| Verification | [`check_experiments.py`](../scripts/check_experiments.py), [`check_sequences.py`](../scripts/check_sequences.py), [`verify_experiment_results.py`](../scripts/verify_experiment_results.py), [`verify_robustness.py`](../scripts/verify_robustness.py) | Check feature isolation and independently verify saved experiment and robustness results. |
| Notebook rendering | [`execute_notebook.py`](../scripts/execute_notebook.py), [`render_presentation.py`](../scripts/render_presentation.py) | Execute a notebook and render the main presentation as HTML. |

## Read or rebuild the presentation

Edit `explore.ipynb`, `baseline.ipynb` and `experiments.ipynb` directly;
the notebooks are the source documents, with no separate generation scripts.
The main `explore.ipynb` reads small tracked result tables. It does not train.
From the repository root, after installing `requirements.txt`:

```bash
./nasa/bin/python scripts/execute_notebook.py
./nasa/bin/python scripts/render_presentation.py
```

The recorded runs used Python 3.14 on Linux/WSL2, with six training threads.
Recorded package versions are in `outputs/experiments/environment.txt`.
The default requirements now include exact version snapshots for the recorded
Linux x86-64 / Python 3.14 environment. See the dependency notes below before
installing on another platform. On Windows, the Python executable path would be
`nasa\Scripts\python.exe`; the Linux dependency snapshot is not a Windows lock.

## Dependency snapshots and installation limits

`requirements.txt` includes `requirements.lock` (48 base packages).
`requirements-sequences.txt` includes `requirements-sequences.lock`, which adds
24 sequence dependencies to the same base. Versions come from installed package
metadata, following all active transitive dependencies for Linux/Python 3.14.
The sequence snapshot deliberately retains the TensorFlow/Keras prereleases
that produced the saved results; no framework migration was performed.

These are **version snapshots, not artifact-hash locks**. The installed environment
passes `pip check` and an offline installation dry run against these files.
A fresh download probe could not obtain the recorded NumPy 2.5.4 wheel from the
configured package source. Therefore fresh installation availability and wheel
hashes are **not verified**. A lockfile alone cannot preserve unavailable wheels.
Keep the working environment until a separate installation has passed the model
and report checks. A stable-framework migration should be tested separately.

The notebook execution/rendering scripts use IPython directly. Unused nbformat,
nbclient and nbconvert requirements were removed; a full Jupyter frontend is
optional and is not included in these runtime snapshots. Interactive kernel
support remains included through ipykernel.

```bash
./nasa/bin/python -m pip check
./nasa/bin/python -m pip install --dry-run --no-index -r requirements-sequences.txt
```

The dry run checks an **existing** environment; it does not prove that an empty
environment can install the packages. For another platform/Python version,
resolve a separate environment and rerun verification before recording its pins.

## Prepare data

Obtain the C-MAPSS archive from the NASA source listed in
[`data/source.json`](../data/source.json). Import the ZIP containing
`train_FD001.txt` through `train_FD004.txt`, corresponding test files and RUL files:

```bash
./nasa/bin/python scripts/prepare_data.py /path/to/CMAPSSData.zip
```

The importer checks all twelve file hashes against the recorded study inputs
before writing anything. A different dataset version is rejected. Raw files stay
local and ignored by Git. The archived acquisition details in the manifest
record the original run; its original machine path is not needed to reproduce it.

## Train and verify

Run in this order; each training script overwrites its own generated outputs:

```bash
./nasa/bin/python scripts/train_all.py --threads 6
./nasa/bin/python scripts/check_experiments.py
./nasa/bin/python scripts/run_experiments.py --threads 6
./nasa/bin/python scripts/verify_experiment_results.py
./nasa/bin/python scripts/audit_sensors.py
./nasa/bin/python scripts/run_robustness.py
./nasa/bin/python scripts/verify_robustness.py
```

This restores ignored models and prediction exports under `outputs/`. Full
verification requires those artifacts and the raw data. The frozen-model checksum
in the saved robustness protocol applies to the original artifact; rerunning
robustness records and checks the model used for that run.

To rerun the supporting notebooks after training:

```bash
./nasa/bin/python scripts/execute_notebook.py baseline.ipynb
./nasa/bin/python scripts/execute_notebook.py experiments.ipynb
```

Method details: [experiment protocol](../outputs/experiments/METHODS.md) and
[robustness protocol](../outputs/robustness/METHODS.md). The original local runs
remain in the training logs. Repeated splits overlap and are sensitivity checks,
not independent blind replications.

## XGBoost fitting-gap diagnostic

```bash
./nasa/bin/python scripts/run_experiments.py --diagnostics --threads 6
```

This refits only the frozen configuration on the saved folds, checks the original
OOF predictions, and compares fitting/validation endpoints. It does not rerun
selection or modify the final model. See the
[diagnostic interpretation](../outputs/experiments/METHODS.md#xgboost-overfitting-check).

## Optional sequence comparison

[`run_sequences.py`](../scripts/run_sequences.py) compares a compact LSTM and TCN
using the existing development folds, two seeds and an inner engine split for
early stopping. It does not replace the final XGBoost model.

```bash
./nasa/bin/python -m pip install -r requirements-sequences.txt
./nasa/bin/python scripts/run_sequences.py
./nasa/bin/python scripts/run_sequences.py --verify
```

See the [sequence protocol](../outputs/sequences/METHODS.md). Training creates
ignored fold models and prediction exports; full verification needs these files.

## TCN follow-up

[`run_tcn_followup.py`](../scripts/run_tcn_followup.py) compares 30/50-cycle
histories, a larger training budget, three seeds and inner-engine alert
calibration. Completed jobs are resumable; large per-job artifacts stay local.

```bash
./nasa/bin/python scripts/run_tcn_followup.py
./nasa/bin/python scripts/run_tcn_followup.py --verify
./nasa/bin/python scripts/run_tcn_followup.py --diagnostics
```

See the [follow-up protocol](../outputs/tcn_followup/METHODS.md). The existing
final XGBoost model and official test predictions are preserved.

## Bounded regularization comparison

```bash
./nasa/bin/python scripts/run_regularization.py
./nasa/bin/python scripts/run_regularization.py --verify
```

Run the prior TCN follow-up first: this comparison reuses its baseline models
and inner selections. It compares frozen XGBoost regularization options and a
smaller TCN, choosing recipes on inner engines and evaluating separate outer
engines. Large checkpoints stay local; interrupted jobs are resumable. See the
[regularization protocol](../outputs/regularization/METHODS.md).

## What stays out of Git

Raw archives/data, `.joblib`/`.ubj`/`.keras` models, large OOF and trajectory exports,
duplicate notebook previews, and the superseded FD001-only experiment are
excluded. Compact metrics and the inputs needed to rebuild the presentation stay
tracked. Removing files from the current tree does not remove earlier Git history.


## Sensor explanation visuals

`./nasa/bin/python scripts/audit_sensors.py --visuals-only` regenerates the small
sensor matrices using only development-engine histories. It does not retrain
models or rerun the sensor ablation. Then execute and render the main notebook.
The notebook saves `sensor_variation.png`, `sensor_correlation.png`, and a stable
`test_rmse_by_dataset.png` for README links. These named images are included for
presentation use; sequential `figure_*.png` exports remain local.


## Matched alert calibration

After the TCN follow-up and regularization experiments, run
`./nasa/bin/python scripts/run_tcn_followup.py --compare-alerts`.
It fits three inner XGBoost models, reuses/reproduces saved outer predictions,
and calibrates both families with the same rule and engine endpoints. The final
model remains unchanged. See the [protocol](../outputs/alert_comparison/METHODS.md).
Then rebuild the notebook and HTML to export the matched confusion matrices.

## Full-data TCN fit

After the TCN follow-up and final XGBoost fit, run:

```bash
./nasa/bin/python scripts/train_tcn.py
./nasa/bin/python scripts/train_tcn.py --verify
```

This trains three frozen 50-cycle TCNs on all 709 training engines and evaluates
all seeds plus their predeclared equal-weight ensemble on the 707 official test
engines. Epoch counts and final alert thresholds come from development results.
Completed seeds are resumable; incomplete seeds restart deterministically.
`--evaluate` rescores completed fits. Local checkpoints remain ignored by Git;
compact predictions and metrics support report rebuilding. See the
[full-data protocol](../outputs/tcn_full/METHODS.md) for calibration and benchmark limits.

## Feature-generation optimization

Grouped rolling means, population standard deviations and exponential means use
native pandas grouped windows instead of per-group Python transforms. Row labels
are preserved after dropping the group index level, and output is restored to
input order. Windows remain causal and isolated by engine.

On 160,359 training rows, three sequential measurements per implementation gave:

| Feature set | Previous median | Updated median | Speedup |
| --- | ---: | ---: | ---: |
| Basic histories | 0.524 s | 0.398 s | 1.31× |
| Regime-normalized rich histories | 1.732 s | 1.241 s | 1.40× |

The previous implementation is preserved in commit `4428a51`. Measurements used
this machine's recorded environment; they measure feature transformation only,
not total model training. Every float32 feature matched exactly across all train
and test rows, including shuffled-row checks. No models were retrained.
`check_experiments.py` retains an independent per-group reference for all four
window statistics and checks shuffled alignment, engine boundaries and future/
target isolation. Saved XGBoost and TCN predictions were verified after the change.

The experimental asymmetric loss is `0.5 * sample_weight * multiplier * error²`,
where the multiplier is 1.5 for overestimation and 1 otherwise. Weighting both
first and second derivatives is intentional. It changes Hessian totals and thus
interacts with regularization and minimum child weight. Mean-one sample weights
preserve total row weight, not the number of engines. The selected final XGBoost
configuration uses the standard objective, so this documentation changes no result.

## Shared helpers and explicit sequence windows

`scripts/cmapps/` is a lightweight internal package for evaluation, alert
calibration, atomic JSON writes, SHA-256 digests and timestamped logs. Running
entry points as `python scripts/<name>.py` makes this package available without
an installation step. Experiment-specific fitting loops and resumable workers
remain in their scripts. Existing score/helper imports remain available for
compatibility, but the updated consumers import shared helpers directly.

The sequence threshold helper retains the original positive-quantile rule.
XGBoost's `choose_threshold` still returns its full candidate table; its behavior
has not been replaced by the sequence helper.

`SequenceInputs.windows(..., window=30)` and `network(..., window=50)` now require
an explicit window length. No experiment changes `seq.WINDOW`. Preprocessor
objects have no new required state, and serialized model classes stay at their
original module paths. Existing Joblib and Keras files remain usable.

```bash
./nasa/bin/python scripts/check_sequences.py
./nasa/bin/python scripts/run_sequences.py --verify
./nasa/bin/python scripts/run_tcn_followup.py --verify
./nasa/bin/python scripts/run_regularization.py --verify
./nasa/bin/python scripts/train_tcn.py --verify
```

These checks require the optional sequence environment, raw data and saved local
artifacts. They do not retrain models. The window check alternates 30/50/30-cycle
calls within one process and checks an independent prefix/padding oracle,
engine isolation, future/target isolation and legacy preprocessing compatibility.
Environment audit logs and the version snapshots are unchanged by this refactor.
