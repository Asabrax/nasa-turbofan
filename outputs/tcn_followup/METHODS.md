# TCN follow-up: training budget, history length and alert calibration

## Results

All 18 local training jobs completed, producing 27 scored fold configurations
across the same 567 development engines (1,701 endpoints per protocol and seed).
All nine 50-cycle fold/seed runs improved primary RMSE over their matched
30-cycle, 60-epoch-budget runs.

| Configuration | Primary CV RMSE | Broad CV RMSE |
| --- | ---: | ---: |
| Existing XGBoost reference | 25.93 | 45.32 |
| TCN, 30 cycles, maximum 20 epochs | 25.80–26.05 | 45.25–45.73 |
| TCN, 30 cycles, maximum 60 epochs | 25.97–26.00 | 45.47–45.84 |
| TCN, 50 cycles, maximum 60 epochs | **23.59–24.13** | **43.75–43.96** |

Ranges are minimum–maximum across seeds 42, 73 and 101, not confidence intervals.
The 50-cycle TCN reduces primary RMSE by 7.0–9.0% relative to the existing XGBoost
configuration on these development endpoints. The three paired 95% bootstrap
intervals for TCN minus XGBoost primary RMSE are [-3.05, -1.43], [-3.14, -1.54]
and [-2.73, -0.90] cycles respectively. All also favor 50 over 30 cycles at the
same maximum budget; broad-protocol intervals favor 50-cycle TCN too.

Increasing the budget alone did not establish an improvement: only two of nine
30-cycle fold/seed jobs selected a later epoch (19→26 and 20→45). Their effects
on pooled seed-level primary RMSE were -0.05 and +0.19 cycles; the third seed
was unchanged. Both nonzero paired intervals include zero. The 50-cycle runs
selected 22–37 epochs, so their improvement reflects longer histories under the
larger budget; we did not run a 50-cycle/20-epoch arm.

| 50-cycle TCN alert rule | Primary recall | Primary precision | False alerts / 1,701 endpoints |
| --- | ---: | ---: | ---: |
| Fixed threshold 30 | 75.1–81.2% | 97.8–98.3% | 4–5 |
| Inner-calibrated | **90.8–93.5%** | **90.1–92.2%** | 23–30 |

Calibration catches more near-failure endpoints, with more false alerts. These
are snapshot counts, not counts of unique alerted engines or first-alert events.
Each seed has 293 true positive-event endpoints. The nine inner-selected
thresholds range from 34.79 to 39.71 cycles. One outer fold/seed still attained
only 86.7% recall, and two of the three pooled recall intervals extend below
90%; an inner recall target is not a guarantee for future engines.

**Decision:** the 50-cycle TCN is a stronger development candidate in this
comparison. Preserve the published XGBoost test result and treat the TCN recipe
as a candidate for independent validation. No additional audit or official-test
scoring was performed, and the final XGBoost artifact is unchanged.

Saved models were reloaded and their predictions reproduced for both window
lengths. Verification also checked all 30,618 scored rows, engine separation,
inner-only thresholds, epoch selection and the original model checksum.

## Predeclared comparison

The existing 567 development engines, three outer folds and evaluation endpoints
are unchanged. Test and audit endpoints are not evaluated. The final all-engine
XGBoost model is preserved and its checksum is verified.

- Compare 30- and 50-cycle histories with a maximum of 60 inner-training epochs.
- Use seeds 42, 73 and 101 for both windows, not just the best-looking window.
- Keep the prior TCN architecture, target cap 150, Adam learning rate 0.001,
  dropout 0.1, batch size 512, patience four and 0.01-cycle minimum improvement.
- For the 30-cycle window, also retain the epoch selection that would have been
  made with a 20-epoch limit on the SAME training trajectory. Refit this reference
  separately only if its selected epoch differs from the 60-epoch selection.

There are 18 window/fold/seed jobs and 27 scored fold configurations. The
contemporaneous 20-epoch reference isolates the budget change without mixing
runs or CPU thread settings. Increasing the maximum budget cannot affect a run
that already meets the early-stopping condition before epoch 20.

## Engine separation and preprocessing

Within each outer fold, 302 engines fit the inner model, 76 engines determine
early stopping and calibration, and 189 separate engines evaluate the result.
Normalization, sensor selection and metadata scaling use fitting engines only.
Every sequence is a trailing window within one engine, with zero left-padding
and an observation indicator for short prefixes. Target values never enter
features. Prefix-invariance checks run for each dataset and window length.

After inner selection, refit preprocessing and the network on all 378 outer
fitting engines for the selected number of epochs. The outer evaluation engines
never determine the stopping epoch or alert threshold. All available fitting
cycles are used; targets are capped only for training. RMSE uses uncapped labels.

## Alert calibration

Use predictions from the selected inner model on its 76 stopping engines.
Choose the smallest predicted-RUL threshold achieving at least 90% recall for
actual RUL ≤30 on their saved primary endpoints. The selected threshold is then
frozen and transferred to the outer refitted model. Evaluate it on the 189 outer
engines, alongside the fixed threshold of 30. The same inner population selects
the epoch and threshold; only the outer results measure their generalization.

Refitting can change calibration, and the inner event sample is small. Outer
recall is therefore not guaranteed to reach 90%. Report all three fold thresholds
rather than inventing one universal threshold. Pooled calibrated metrics sum
confusion counts from each fold's own frozen rule. Regression predictions and
RMSE do not change with alert calibration.

Do not compare these development metrics with the official-test headline, or
with a threshold evaluated on the same pooled OOF examples used to choose it.
The existing XGBoost fixed-threshold results remain a same-endpoint reference.

`alert_intervals.csv` reports whole-engine percentile bootstrap intervals for
precision and recall, retaining each engine's three endpoints. These are
conditional sampling intervals, not guarantees of achieving the recall target
on a new fleet. The fold models and thresholds are held fixed during resampling.

## Interpretation

Paired whole-engine bootstrap intervals use 2,000 draws, retaining each engine's
three endpoints. Compare the larger budget with the contemporaneous 20-epoch
reference, the 50-cycle window with the 30-cycle window at the same budget, and
both with the existing selected XGBoost configuration. Positive RMSE differences
mean the candidate is worse. These intervals condition on the fitted models and
exclude model-selection/refitting uncertainty. They are follow-up evidence on
already-studied development engines, not independent blind validation. Intervals
are not adjusted for the multiple comparisons; seed runs share the same engines.
XGBoost was selected using these development folds in earlier work, so this is
a comparison of tested configurations rather than a nested comparison of entire
model-selection procedures.

## Reproduction and saved outputs

```bash
./nasa/bin/python -m pip install -r requirements-sequences.txt
./nasa/bin/python scripts/run_tcn_followup.py
./nasa/bin/python scripts/run_tcn_followup.py --verify
```

Two local CPU workers each use three TensorFlow intra-operation threads. A
completed job is saved before scheduling its replacement; rerunning the command
skips completed jobs. An interrupted incomplete job restarts deterministically
from scratch. `_runs/` retains inner and outer models and endpoint predictions
locally, but is ignored by Git. Compact summaries and curves are tracked.

`protocol.json` records the input hashes and settings. A resume rejects a changed
protocol. To intentionally run a different protocol, use a separate output
location rather than overwriting this study. The script supports this specific
predeclared comparison, not arbitrary command-line hyperparameter search.
