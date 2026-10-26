# Reproducing the study

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

## What stays out of Git

Raw archives/data, `.joblib`/`.ubj` models, large OOF and trajectory exports,
duplicate notebook previews, and the superseded FD001-only experiment are
excluded. Compact metrics and the inputs needed to rebuild the presentation stay
tracked. Removing files from the current tree does not remove earlier Git history.
