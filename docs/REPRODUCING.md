# Reproducing the study

## Script guide

| Area | Scripts | Purpose |
| --- | --- | --- |
| Data setup | [`prepare_data.py`](../scripts/prepare_data.py) | Import the twelve checksum-verified dataset files. |
| Training and experiments | [`train_all.py`](../scripts/train_all.py), [`run_experiments.py`](../scripts/run_experiments.py), [`run_sequences.py`](../scripts/run_sequences.py) | Train the original baselines and compare tree and sequence models. |
| Model implementation | [`experiment_models.py`](../scripts/experiment_models.py) | Shared normalization, temporal features and estimators; imported by the experiment scripts. |
| Additional experiments | [`audit_sensors.py`](../scripts/audit_sensors.py), [`run_robustness.py`](../scripts/run_robustness.py), [`run_tcn_followup.py`](../scripts/run_tcn_followup.py), [`run_regularization.py`](../scripts/run_regularization.py) | Compare sensor sets and TCN settings; evaluate split sensitivity, planning windows and alert policies. |
| Verification | [`check_experiments.py`](../scripts/check_experiments.py), [`verify_experiment_results.py`](../scripts/verify_experiment_results.py), [`verify_robustness.py`](../scripts/verify_robustness.py) | Check feature isolation and independently verify saved experiment and robustness results. |
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
Requirements use version ranges rather than an exact lock. On Windows, replace
`./nasa/bin/python` with `nasa\Scripts\python.exe`.

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
