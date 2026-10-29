# Full-data TCN comparison

## Frozen design

Train the existing 50-cycle residual TCN on all **709 training engines and
160,359 cycles** across FD001–FD004. Evaluate one final observed endpoint for
each of the **707 official test engines**. This test has already been examined;
this is follow-up benchmark evidence, not a fresh blind validation. The earlier
142-engine audit is included in this full training set and cannot serve as a
new independent audit of these fitted models.

The architecture is unchanged: 32 channels, two causal kernel-3 convolutions per
residual block, dilations 1/2/4/8, dropout 0.1 and 26,241 parameters. Inputs have
26 channels: 17 variance-filtered sensor measurements, cycle, three operating
settings, four dataset indicators and an observation indicator. Windows are
left-padded and causal within each engine. Regime normalization and scaling are
fitted on training histories only. The target is capped at 150 cycles during
training; all reported errors use uncapped labels. Predictions are nonnegative.

Seeds 42, 73 and 101 train for **25, 27 and 30 epochs**, respectively, using
Adam (learning rate 0.001, clipnorm 1), batch size 512 and CPU execution. These
counts are the medians of each seed's three previously selected inner-CV epochs.
Transferring those counts to a larger training set is a fixed heuristic, not a
claim of optimal stopping. No test-based stopping is used.

A fourth predictor is the **predeclared unweighted mean of all three seeds**.
No seed, ensemble weight, epoch count or threshold is chosen from test outcomes.
`protocol.json` records the choices and source hashes before these fits.
Checkpoints and preprocessing objects stay local; compact predictions, metrics
and methods are retained in Git. The original XGBoost artifact is preserved.

## Alert policy and comparison

The event is true RUL ≤30 cycles. Report both a fixed prediction threshold of
30 and a separately calibrated threshold for each predictor. Final calibration
uses the smallest threshold attaining at least 90% recall on **pooled development
out-of-fold primary predictions** (1,701 snapshots from 567 engines). The ensemble
threshold uses the mean of aligned OOF predictions. These development thresholds
transfer unchanged to the all-engine fits. Actual test recall need not reach 90%.

| Predictor | Frozen alert threshold (cycles) |
| --- | ---: |
| XGBoost | 34.43823 |
| TCN seed 42 | 34.56915 |
| TCN seed 73 | 34.82098 |
| TCN seed 101 | 35.44748 |
| TCN equal-weight ensemble | 34.27466 |

This final calibration follows the pooled-OOF approach used for final XGBoost;
it differs from the fold-specific **inner-engine** calibration used in the
[matched development alert comparison](../alert_comparison/METHODS.md).
Do not interpret those two populations or calibration experiments as identical.

`test_metrics.csv` reports all five predictors, both policies and the pooled/four
subset populations. Near-failure RMSE is restricted to true RUL ≤30. Confusion
counts here represent unique test engines, unlike the repeated development
snapshots. Precision and recall alone do not quantify inspection costs or alert
lead time. Test endpoint evaluation is not a replay of an intervention policy.

`paired_comparisons.csv` uses 2,000 paired engine-bootstrap resamples, stratified
by dataset to preserve the official subset mix. Differences are TCN minus
XGBoost; negative RMSE/false-positive-rate differences and positive recall/
precision differences favor TCN. The intervals condition on these fitted models
and thresholds, exclude model-selection/refitting/calibration uncertainty, and
are not adjusted for multiple comparisons. Previously observed benchmark results
do not establish real-fleet performance.

## Observed results

All three seeds completed on 29 October 2026. The table uses frozen calibrated
thresholds; counts are unique engines, with 159 positives and 548 negatives.

| Predictor | RMSE | Near-failure RMSE | Recall | Precision | Missed | False alerts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 22.63 | 6.50 | 93.7% | 92.5% | 10 | 12 |
| TCN seed 42 | 20.30 | 5.63 | 98.7% | 95.2% | 2 | 8 |
| TCN seed 73 | 20.45 | 8.99 | 95.6% | 94.4% | 7 | 9 |
| TCN seed 101 | 21.32 | 10.14 | 91.8% | 97.3% | 13 | 4 |
| Equal-weight ensemble | 20.09 | 7.88 | 94.3% | 96.2% | 9 | 6 |

The ensemble's RMSE reduction is 11.2%, with paired 95% interval 1.54–3.59 cycles.
All individual seeds also have RMSE intervals favoring TCN. The ensemble improves
RMSE in each dataset: FD001 16.92, FD002 22.15, FD003 16.51 and FD004 20.31.

The ensemble catches one additional positive engine with six fewer false alerts.
Its recall difference is +0.63 percentage points [−3.68, +4.58], so a recall
advantage is not established. Seed 42's recall interval excludes zero, but that
is one of multiple comparisons; selecting it now would use the observed test.
Seed 101 has lower recall than XGBoost. Do not claim every seed improves recall.

At the fixed threshold 30, ensemble recall is 83.0%, versus XGBoost's 89.9%.
Near-failure RMSE also favors XGBoost over the ensemble, despite lower overall
TCN error. NASA score sums are 12,112.57 for XGBoost, 13,851.92 for seed 42,
9,266.08 for seed 73, 10,513.87 for seed 101 and 9,025.35 for the ensemble.
These tradeoffs prevent an unconditional claim that one family is always better.

## Reproduction and verification

After regenerating the TCN follow-up OOF predictions and the final XGBoost model:

```bash
./nasa/bin/python scripts/train_tcn.py
./nasa/bin/python scripts/train_tcn.py --verify
```

Completed seed jobs are reused; an interrupted incomplete seed restarts from its
fixed seed. `--evaluate` rescoring requires all three saved fits. Training
checks saved-model prediction reproduction and future/target isolation. Verification
reloads all three models and XGBoost, reproduces aligned predictions, checks
train-only normalization, development-only epoch/threshold rules, the fixed
ensemble, label hashes, confusion counts, errors and the original XGBoost hash.
The main presentation needs only the compact result files, not trained models.
