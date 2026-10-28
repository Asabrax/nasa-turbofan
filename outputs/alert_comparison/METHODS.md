# Matched recall calibration: XGBoost versus TCN

## Result

Both families target at least 90% recall on identical inner calibration engines.
The frozen thresholds are then applied to separate outer evaluation engines.
On the same 1,701 primary endpoints from 567 development engines, 293 endpoints
have true RUL ≤30 and 1,408 do not.

| Model | Recall | Precision | Caught (TP) | Missed (FN) | False alerts (FP) | Correct non-alerts (TN) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost, seed 42 | 90.4% | 88.0% | 265 | 28 | 36 | 1,372 |
| TCN, seed 42 | 92.8% | 92.2% | 272 | 21 | 23 | 1,385 |
| TCN, seed 73 | 93.5% | 90.1% | 274 | 19 | 30 | 1,378 |
| TCN, seed 101 | 90.8% | 91.7% | 266 | 27 | 24 | 1,384 |

**TCN has the better observed alert tradeoff in all three seeds:** fewer missed
positive endpoints and fewer false alerts. These are snapshot counts, not unique
failed engines, first-alert events or operational maintenance actions.

The recall advantage remains uncertain. Paired engine-bootstrap 95% intervals
for TCN minus XGBoost recall, in percentage points, are:

| TCN seed | Recall difference | 95% interval |
| --- | ---: | ---: |
| 42 | +2.39 pp | -1.24 to +6.03 pp |
| 73 | +3.07 pp | -0.35 to +6.65 pp |
| 101 | +0.34 pp | -3.79 to +4.28 pp |

Every recall interval includes zero. Precision differences and false-positive
rate differences favor TCN with intervals excluding zero for seeds 42 and 101,
but not 73. These are conditional, unadjusted intervals on shared development
engines; they do not constitute three independent replications.

**Decision:** the original 50-cycle TCN is a stronger candidate for further alert
validation on this evidence. Do not claim a proven recall advantage or deployment
readiness. Do not choose a favorite seed based on these outer results. Preserve
both existing models and validate the alert policy on genuinely unseen engines.

## Matched protocol

- Same previously saved three outer engine folds and endpoint snapshots.
- Within each fold: 302 fitting engines, 76 inner calibration engines, then a
  refit on all 378 outer fitting engines, evaluated on 189 separate outer engines.
- Event: true RUL ≤30. Alert: predicted RUL ≤the fold-specific threshold.
- Threshold: the smallest prediction among positive inner endpoints that achieves
  at least 90% inner recall. Verify that the next smaller floating-point threshold
  fails the target. The finite positive counts (43, 39 and 36 by fold) yield actual
  inner recalls of 90.7%, 92.3% and 91.7%, respectively, for both families.
- Freeze thresholds before examining outer alert outcomes. Do not adjust them to
  force exactly 90% outer recall. Transfer from the inner model to the larger
  outer refit can change calibration; no achieved-recall guarantee is implied.
- XGBoost uses the frozen depth-six, 800-tree, cap-150 recipe, seed 42. Three inner
  models are trained for this calibration. Outer predictions reuse the original
  OOF records and are reproduced by the saved regularization-baseline refits.
- TCN uses the original 50-cycle, 32-channel, dropout-0.1 recipe and all three seeds
  (42, 73, 101). Saved inner and outer models reproduce their recorded predictions.
  Its 76 inner engines select the stopping epoch and the threshold; outer engines
  remain separate. The comparison aligns calibration data and rule, not every
  aspect of architecture or historical model selection.
- Normalization and feature filtering use the corresponding fitting engines only.
  Training targets are capped, but event labels and evaluation RUL remain uncapped.

This replaces the earlier apples-to-oranges comparison of TCN's inner-calibrated
CV alerts with XGBoost's threshold selected from pooled OOF predictions. The old
XGBoost threshold 34.44 and the existing official-test results remain untouched;
this study uses three new inner-selected XGBoost thresholds (32.29, 36.84, 34.68).
The two calibration approaches should not be conflated.

Primary results above use the pre-existing planning-window endpoint protocol.
Broad-protocol results and fixed-threshold-30 results are retained as sensitivity
checks in the same files; neither changes the primary comparison after seeing
outcomes. All snapshots from a given engine stay together for uncertainty.

## Uncertainty and limitations

Paired percentile bootstrap uses 2,000 whole-engine draws with replacement.
Each draw uses identical sampled engines for XGBoost and a TCN seed. Sum TP,
FP, FN and TN before calculating recall, precision and false-positive rate;
do not average per-engine ratios. Intervals condition on fitted models and fixed
thresholds and exclude calibration/model-selection/refitting uncertainty. No
multiple-comparison adjustment is applied. Seed results share engines.

These development folds have already informed earlier experiments. This is a
fairer comparison of the fixed candidates' calibration procedures, not a fresh
blind test of the full historical search. No audit or official-test labels were
used. Recall and false alarms are snapshot measures; a real policy also needs
first-alert timing, repeated-alarm handling, intervention costs and external
validation. A 90% recall target alone does not establish acceptable risk.

## Files and reproduction

- `calibration.csv`: all 12 fold/model/seed thresholds and inner event counts.
- `metrics.csv` and `fold_metrics.csv`: confusion counts and rates, both protocols
  and fixed/calibrated rules.
- `intervals.csv` and `paired_comparisons.csv`: marginal and paired uncertainty.
- `protocol.json`: settings and source/model hashes frozen before calibration.
- `oof_predictions.csv` and `_runs/`: local detailed predictions and inner
  XGBoost artifacts; ignored by Git.

After the existing TCN follow-up and regularization runs:

```bash
./nasa/bin/python scripts/run_tcn_followup.py --compare-alerts
./nasa/bin/python scripts/execute_notebook.py
./nasa/bin/python scripts/render_presentation.py
```

The comparison checks identical calibration/evaluation rows and labels, separated
engine roles, minimal thresholds, reloaded predictions, and the unchanged final
XGBoost hash. It does not replace a model or select a deployment threshold.
