"""Build the all-engine course walkthrough; training lives in train_all.py."""
import json
from pathlib import Path
from textwrap import dedent
ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(s):
    cells.append({'cell_type':'markdown','metadata':{},'source':dedent(s).strip().splitlines(True)})
def code(s):
    cells.append({'cell_type':'code','metadata':{},'source':dedent(s).strip().splitlines(True),'outputs':[],'execution_count':None})
md('''
# Predictive maintenance — all NASA C-MAPSS engines

Follow-up: [experiments.ipynb](experiments.ipynb) contains the controlled improvement experiments; this notebook preserves the original baseline.

The tuned follow-up reduced official test RMSE to **22.63 cycles** (true uncapped labels). Its validation-calibrated alert achieved **93.7% recall and 92.5% precision**; threshold calibration does not change RMSE. The broader early-life audit showed a tradeoff, documented in the companion notebook. See the [README progression table](README.md#model-and-alert-progression) for the baseline-to-final comparison.

Course-style workflow adapted from `../data_aai_sose26`: business understanding, data preparation, visual EDA, feature engineering, modeling pipelines, model selection and evaluation.

## 1. Business understanding and scope
Predict **remaining useful life (RUL) in operating cycles** and demonstrate a maintenance flag at predicted RUL ≤ 30 cycles. The 30-cycle horizon is an illustrative assumption, not an approved maintenance policy.

Use **all four subsets, FD001–FD004**, pooling all **709 training engines** and evaluating all **707 test engines**. The actual FD004 files contain 249 training and 248 test engines; their bundled README reverses these counts. The tables below use verified file contents.

FD001/FD003 have one operating condition; FD002/FD004 have six. FD001/FD002 describe one fault mode; FD003/FD004 describe two. Dataset indicators and operating settings help distinguish these domains. Results are reported separately by subset as well as pooled.

### Methods and assumptions
- Training RUL = last cycle of that engine − current cycle. Use uncapped targets.
- Dataset + engine ID identifies a history; train and test IDs are separate namespaces.
- Three-fold validation holds out complete engines. Preprocessing is fitted within each fold.
- Compare a mean baseline, Ridge, Random Forest, and XGBoost. Compare XGBoost depths 3 and 6 using training CV only. The other parameters are fixed classroom starting points, not an exhaustive search.
- Fit each family's selected candidate on **every training row**. Official test engines remain held out.
- Read official test labels only after all model selection and final fits. Evaluate one endpoint per engine.
- Training CV averages errors over all cycles; longer trajectories contribute more. Final test metrics weight engines equally. Fold SD is not a confidence interval.
- Clip negative model predictions to zero consistently during CV and final evaluation.

Local provenance: `data/source.json` and `data/raw/readme.txt`. These are simulated trajectories extracted from a pre-existing archive, not a fresh download. All twelve input hashes are checked. The earlier FD001 experiment is preserved as `explore_fd001.ipynb`.
''')
md('## 2. Setup and local training controls')
code('''
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
ROOT = Path.cwd()
if not (ROOT / 'scripts/train_all.py').exists():
    ROOT = ROOT / 'nasa-rul'
sys.path.insert(0, str(ROOT / 'scripts'))
from train_all import load_inputs, make_features, run_training, DATASETS
OUT = ROOT / 'outputs/all_engines'
OUT.mkdir(parents=True, exist_ok=True)
# The CLI run already trains on this machine. Set True to deliberately train again.
RETRAIN = False
THREADS = 6
plt.rcParams.update({'figure.figsize': (9, 4.5), 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
''')
md('''
## 3. Data quality and target construction
`load_inputs` checks numeric finiteness, 26-column width, input hashes, unique engine/cycle keys and contiguous cycles starting at 1. No missing cells are imputed and no rows are sampled away. Training targets are constructed within dataset/engine groups. Source code is in `scripts/train_all.py` for inspection and reuse.
''')
code('''
train, test, audit = load_inputs()
display(audit[['dataset', 'split', 'rows', 'engines']])
display(train[['dataset', 'engine_id', 'cycle', 'setting_1', 'sensor_2', 'RUL']].head())
print(f'Training: {len(train):,} rows, {train.engine_key.nunique()} engines')
print(f'Test: {len(test):,} rows, {test.engine_key.nunique()} engines')
''')
md('''
## 4. Visual exploration and outliers
Inspect training-engine lifetime distributions by dataset. Sensor extremes can reflect degradation or operating conditions, so they are not automatically deleted. Dataset populations differ; pooling requires per-dataset evaluation.
''')
code('''
lifetimes = train.groupby(['dataset', 'engine_key']).cycle.max().reset_index(name='lifetime')
display(lifetimes.groupby('dataset').lifetime.agg(['count', 'mean', 'median', 'min', 'max']).round(1))
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
bins = np.arange(0, lifetimes.lifetime.max() + 50, 25)
for ax, dataset in zip(axes.flat, DATASETS):
    ax.hist(lifetimes.loc[lifetimes.dataset.eq(dataset), 'lifetime'], bins=bins, edgecolor='white')
    ax.set(title=dataset, xlabel='Lifetime (cycles)', ylabel='Number of training engines')
fig.suptitle('Training lifetime distributions across all four datasets')
fig.tight_layout()
plt.show()
''')
md('''
## 5. Causal feature engineering
Use cycle, three operating settings, 21 sensors, four dataset indicators, five-cycle trailing sensor means and one-cycle sensor changes: **71 features** before fold-specific constant-feature removal. IDs and failure times are excluded.

Rolling means include the current and preceding observations only. At the first cycle, sensor changes equal zero. Grouping by dataset-plus-engine prevents ID collisions across datasets. The checks below ensure truncating an engine history cannot change earlier features.

These simple rolling features can mix operating conditions in FD002/FD004. Raw settings are supplied to the model, but this does not guarantee condition normalization. The per-subset results reveal performance limits; regime-aware sensor normalization is a follow-up experiment.
''')
code('''
X = make_features(train)
assert 'engine_id' not in X and 'RUL' not in X
assert np.isfinite(X.to_numpy()).all()
for dataset in DATASETS:
    prefix = train[(train.dataset == dataset) & (train.engine_id == 1)].iloc[:25]
    pd.testing.assert_frame_equal(make_features(prefix), X.loc[prefix.index])
assert X.loc[train.cycle.eq(1), 'sensor_2_delta'].eq(0).all()
print(f'{X.shape[1]} causal features; checks passed for all four datasets.')
display(X.head(3))
''')
md('''
## 6. Actually train locally
The command below runs the complete experiment in this project's Python environment:

```bash
./nasa/bin/python scripts/train_all.py --threads 6
```

It trains all five candidates on three grouped folds, selects the XGBoost depth by CV, and refits four models (baseline, Ridge, Random Forest and XGBoost) on all 160,359 training rows. The process records each fit in `outputs/all_engines/training.log` and saves actual fitted model files. XGBoost uses CPU histogram training; no remote training service is involved.

This notebook loads the completed local run unless `RETRAIN=True` or results are missing. Its saved outputs therefore refer to that run; they do not imply a new training run every time you open the notebook. CPU threads are bounded to six.
''')
code('''
if RETRAIN or not (OUT / 'results.json').exists():
    run_training(threads=THREADS)
results = json.loads((OUT / 'results.json').read_text())
comparison = pd.read_csv(OUT / 'model_comparison.csv').sort_values('CV_RMSE')
folds = pd.read_csv(OUT / 'cv_folds.csv')
assignments = pd.read_csv(OUT / 'fold_assignments.csv')
assert len(assignments) == 709 and assignments.engine_key.is_unique
assert len(folds) == 15 and folds.groupby('candidate').size().eq(3).all()
display(comparison.round(3))
print('Selected by grouped training CV:', results['selected_candidate'])
print('Training completed (UTC):', results['finished_utc'])
print('Elapsed seconds:', round(results['total_seconds'], 1))
print('Versions:', results['versions'])
''')
md('''
### Training and grouped validation errors
Training error is optimistic; compare it with validation error. Model choice is based on the validation column, never on final test performance. The two XGBoost depths are a small tuning exercise corresponding to the course's modeling and hyperparameter chapters.
''')
code('''
fig, ax = plt.subplots(figsize=(10, 5))
ypos = np.arange(len(comparison))
ax.barh(ypos - .18, comparison.train_RMSE, .36, label='Training')
ax.barh(ypos + .18, comparison.CV_RMSE, .36, xerr=comparison.fold_SD, label='Grouped validation ± fold SD')
ax.set(yticks=ypos, yticklabels=comparison.candidate, xlabel='RMSE (cycles)', title='All-engine model comparison')
ax.invert_yaxis()
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.15), ncol=2)
fig.tight_layout()
plt.show()
''')
md('''
## 7. Final evaluation — every test engine
Each model is evaluated at the last available cycle of each of 707 test engines, using the official RUL files in engine-ID order within each dataset. The table compares every model family, including XGBoost, but the overall selected family remains the one chosen by CV.

Pooled RMSE is calculated from all 707 squared errors, not by averaging subset RMSEs. MAE is in cycles; positive mean error overestimates remaining life. MAPE is omitted because RUL can be zero.
''')
code('''
metrics = pd.read_csv(OUT / 'test_metrics.csv')
predictions = pd.read_csv(OUT / 'test_predictions.csv')
assert predictions.groupby('family').size().eq(707).all()
assert not predictions.duplicated(['family', 'dataset', 'engine_id']).any()
display(metrics[metrics.dataset.eq('ALL')][['family', 'engines_or_rows', 'RMSE_cycles', 'MAE_cycles', 'R2', 'mean_error_cycles']].round(3))
display(metrics.pivot(index='dataset', columns='family', values='RMSE_cycles').round(3))
''')
code('''
subset_metrics = metrics[metrics.dataset.ne('ALL')].pivot(index='dataset', columns='family', values='RMSE_cycles')
ax = subset_metrics.plot.bar(figsize=(11, 5), rot=0)
ax.set(title='Test RMSE by dataset and model', xlabel='Dataset', ylabel='RMSE (cycles)')
ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()
''')
md('### Inspect errors for the CV-selected model')
code('''
selected = predictions[predictions.family.eq(results['selected_family'])]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
markers = ['o', '^', 's', 'x']
for dataset, marker in zip(DATASETS, markers):
    group = selected[selected.dataset.eq(dataset)]
    axes[0].scatter(group.actual_RUL, group.predicted_RUL, label=dataset, marker=marker, s=19, alpha=.6)
limit = max(selected.actual_RUL.max(), selected.predicted_RUL.max()) * 1.05
axes[0].plot([0, limit], [0, limit], '--', color='black')
axes[0].set(xlim=(0, limit), ylim=(0, limit), xlabel='Actual RUL (cycles)', ylabel='Predicted RUL (cycles)', title=results['selected_candidate'])
axes[0].legend()
axes[1].hist(selected.error_cycles, bins=25, edgecolor='white')
axes[1].axvline(0, color='black', linestyle='--')
axes[1].set(title='Positive errors overestimate remaining life', xlabel='Predicted − actual RUL (cycles)', ylabel='Number of engines')
fig.tight_layout()
plt.show()
''')
md('''
## 8. Illustrative maintenance decision
For each family, flag predicted RUL ≤ 30 cycles and compare with actual RUL ≤ 30. Report precision, recall and false negatives, not accuracy alone. These are endpoint-level classification measurements, not complete maintenance-event outcomes. No cost data are supplied, so no financial savings are claimed.
''')
code('''
display(metrics[metrics.dataset.eq('ALL')][['family', 'precision', 'recall', 'F1', 'TP', 'FP', 'FN', 'TN']].round(3))
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(selected.actual_RUL <= 30, selected.predicted_RUL <= 30,
    labels=[False, True], display_labels=['Beyond 30 cycles', 'Within 30 cycles'], colorbar=False)
plt.title('30-cycle rule — CV-selected model, all test engines')
plt.tight_layout()
plt.show()
''')
md('''
## 9. Artifacts and conclusions
All four trained model families are saved under `outputs/all_engines/`, along with per-engine predictions, per-subset metrics, CV fold membership, fit durations, environment details and the training log. The XGBoost booster is also saved in native UBJ format. Use the joblib pipeline for inference because it retains preprocessing; the raw booster does not.

The notebook's summary reports measured results from this run. Per-dataset test metrics and positive errors should guide follow-up work, not further tuning against these test labels. The small search is an educational baseline, not a claim of optimal performance.

Limitations: pooled training weights long trajectories more, operating regimes are not normalized, no prediction intervals are estimated, and test endpoints differ from the all-cycle CV population. Next experiments should use validation endpoints, condition-aware features and engine-level uncertainty. Validate on real operating data and specified maintenance costs before operational use.

Course mapping: missing values/outliers (lectures 4–5), visual EDA (6), feature engineering (7), fitting/model selection (8–9), leakage-free pipelines (9), and evaluation/decision thresholds (10–12).
''')
code('''
# Verify the saved model works on a complete unseen engine history.
import joblib
slug = results['selected_family'].lower().replace(' ', '_')
artifact = joblib.load(OUT / f'{slug}.joblib')
engine = test[(test.dataset == 'FD004') & (test.engine_id == 1)]
latest_features = make_features(engine).iloc[[-1]][artifact['features']]
prediction = float(np.maximum(0, artifact['pipeline'].predict(latest_features))[0])
expected = selected[(selected.dataset == 'FD004') & (selected.engine_id == 1)].predicted_RUL.iloc[0]
assert np.isclose(prediction, expected)
print(f'FD004 engine 1: predicted remaining life {prediction:.2f} cycles.')
print('Saved-model inference matches the test prediction CSV.')
''')
for i,cell in enumerate(cells):
    cell['id']=f'all-{i:03d}'
nb={'cells':cells,'nbformat':4,'nbformat_minor':5,'metadata':{
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.14'},'artifact_dir':'outputs/all_engines'}}
(ROOT/'baseline.ipynb').write_text(json.dumps(nb,indent=1)+'\n')
print('Wrote baseline.ipynb')
