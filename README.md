# NASA turbofan predictive maintenance

Predict remaining useful life from sensor histories across all four NASA C-MAPSS
datasets. XGBoost combines operating-regime normalization, causal history features
and a 150-cycle training target cap. A full-data TCN ensemble provides the
strongest overall RMSE in our follow-up comparison.

**[Read the presentation notebook](explore.ipynb)** ·
[Detailed experiments](experiments.ipynb)

![Test RMSE by dataset](outputs/presentation/test_rmse_by_dataset.png)

## Results

Evaluated on the same 707 official test engines, using true uncapped RUL labels.
The maintenance event is actual RUL ≤30 cycles.

| Model / alert rule | RMSE (cycles) | Recall | Precision |
| --- | ---: | ---: | ---: |
| Original Random Forest, threshold 30 | 33.14 | 82.4% | 94.2% |
| Original XGBoost, threshold 30 | 33.42 | 81.8% | 94.9% |
| Final XGBoost, threshold 30 | 22.63 | 89.9% | 94.7% |
| Final XGBoost, calibrated threshold 34.44 | 22.63 | 93.7% | 92.5% |
| TCN ensemble, calibrated threshold 34.27 | **20.09** | **94.3%** | **96.2%** |

- **Validation:** both families use engine-separated development CV. The XGBoost
  study also includes a held-out audit and five split-sensitivity checks; these
  do not constitute a fresh audit of the full-data TCN. Fold preprocessing uses
  only fitting engines.
- **Sensors:** 17 of 21 survive variance filtering. Removing another three did
  not establish a meaningful improvement. See the [variation map](outputs/presentation/sensor_variation.png)
  and [sensor correlation matrix](outputs/presentation/sensor_correlation.png).
- **Limits:** the XGBoost audit shows worse broad early-life accuracy. Its calibrated test recall has a 95%
  interval of 88.8–96.5%. The official test was previously observed; this is
  simulation-benchmark evidence, not a deployment or savings claim.

## Full-data TCN result

Three 50-cycle TCNs were trained locally on **all 709 engines / 160,359 cycles**,
using development-selected epoch counts. Their **predeclared equal-weight
ensemble** improves test RMSE by **11.2%**, with a paired 95% reduction interval
of **1.54–3.59 cycles**. It catches **150/159** near-failure engines versus
XGBoost's **149/159**, with **6 versus 12 false alerts**.

The ensemble's recall difference is **+0.63 percentage points [−3.68, +4.58]**,
so we cannot claim a conclusive recall improvement. Individual TCN seeds span
20.30–21.32 RMSE and 91.8–98.7% calibrated recall; no favorite seed was selected
from test scores. At threshold 30, ensemble recall is 83.0%, and its near-failure
RMSE is 7.88 versus XGBoost's 6.50. Calibration and the operational objective matter.

The official test was already examined. Intervals are conditional and unadjusted
for multiple comparisons; an independent dataset is still needed. See the
[full-data results and protocol](outputs/tcn_full/METHODS.md) and
[comparison plot](outputs/presentation/full_tcn_comparison.png).

## Sequence model comparison

We also trained small LSTM and TCN models locally on the same development folds.
Primary CV RMSE was **25.93** for XGBoost, **26.43–26.63** for LSTM, and
**26.02–26.19** for TCN across two neural-network seeds. These are development
scores, not the official-test results above. Paired uncertainty intervals include
zero, so the small RMSE gaps are not decisive. The final XGBoost model is unchanged.
See [sequence methods and results](outputs/sequences/METHODS.md) for the bounded
training budget, uncertainty estimates and alert tradeoffs.

The [TCN follow-up](outputs/tcn_followup/METHODS.md) found a stronger development
candidate: **50-cycle histories reached 23.59–24.13 primary CV RMSE** across
three seeds, 7–9% below XGBoost on the same endpoints. Paired intervals favor
TCN for all three seeds. Inner-calibrated alerts achieved **90.8–93.5% recall**
and **90.1–92.2% precision**, with more false alerts. Increasing the budget alone
at 30 cycles did not consistently help. These follow-up development results do
not replace the XGBoost official-test results; TCN still needs independent validation.
The [overfitting check](outputs/tcn_followup/METHODS.md#overfitting-check) found
a 3.91–5.46-cycle training/validation gap: the improvement survives on held-out
engines, but the model is not free of fitting optimism. The
[matched XGBoost check](outputs/experiments/METHODS.md#xgboost-overfitting-check)
found a larger 8.14-cycle gap (17.79 training versus 25.93 validation RMSE).

A [bounded regularization comparison](outputs/regularization/METHODS.md)
reduced fitting gaps but did not reliably improve held-out accuracy. XGBoost's
inner-selected recipe scored 26.06 versus 25.93 RMSE; the smaller TCN was rejected
by inner validation in all nine fold/seed runs. The existing recipes are retained.

A [matched alert comparison](outputs/alert_comparison/METHODS.md) calibrated both
families on identical inner splits to target 90% recall. TCN achieved 90.8–93.5%
recall with 23–30 false alerts, versus XGBoost's 90.4% and 36, on the same 1,701
outer snapshots. This favors TCN's observed tradeoff, but all paired recall
intervals include zero. See the [confusion matrices](outputs/presentation/matched_alert_confusion.png).

## Explore or reproduce

Open [explore.ipynb](explore.ipynb) for the executed report, including plots and
tables. It also runs from the small result tables included here, without raw data
or trained models.

```bash
python3.14 -m venv nasa
./nasa/bin/python -m pip install -r requirements.txt
./nasa/bin/python scripts/execute_notebook.py
```

For an optional browser presentation, run
`./nasa/bin/python scripts/render_presentation.py` and open the generated
`presentation.html` locally. This export is ignored by Git.

Requirements record exact versions from Linux/Python 3.14. Fresh download
availability has not been verified; see the [dependency notes](docs/REPRODUCING.md#dependency-snapshots-and-installation-limits) before rebuilding an environment.

[Reproduction instructions](docs/REPRODUCING.md) cover data setup, actual training,
and verification. [baseline.ipynb](baseline.ipynb) preserves the original analysis.
The supporting notebooks need regenerated data/models to execute fully.

## Repository contents

- `scripts/` — training, feature engineering, checks and report generation.
- `outputs/` — compact results, configurations, methods and split assignments.
- `data/source.json` — NASA source provenance and input checksums.

Raw datasets, model binaries, large prediction exports and duplicate previews are
excluded from Git. They are generated locally by the reproduction workflow.
