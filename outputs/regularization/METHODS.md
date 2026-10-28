# Bounded regularization comparison

## Results and decision

All 12 local jobs completed: three XGBoost folds and nine TCN fold/seed jobs.
The smaller gaps did not translate into a consistent improvement in held-out
accuracy. **Retain the existing XGBoost recipe and the original 50-cycle TCN
candidate; do not adopt these regularization changes.** This is the conclusion
of the bounded comparison, not a claim that no other regularization can help.

### XGBoost

| Candidate or selection procedure | Training primary RMSE | Held-out primary RMSE | Gap | Held-out RUL ≤30 RMSE |
| --- | ---: | ---: | ---: | ---: |
| Existing baseline | 17.79 | **25.93** | 8.14 | **7.51** |
| Early stopping | 21.04 | 26.28 | 5.24 | 7.89 |
| Shallower + regularized | 24.37 | 26.75 | 2.37 | 8.13 |
| Stronger regularization | 20.73 | 26.22 | 5.50 | 8.00 |
| Inner-selected procedure | 19.18 | 26.06 | 6.88 | 7.69 |

Inner validation chose stronger regularization on fold one, early stopping on
fold two, and the existing baseline on fold three. The resulting outer primary
RMSE worsened by 0.12 cycles (conditional paired 95% interval +0.02 to +0.23).
All three individual alternatives also worsened pooled primary RMSE. The
shallow candidate demonstrates the tradeoff clearly: its gap falls from 8.14
to 2.37 while its held-out error rises from 25.93 to 26.75. Shrinking a gap is
not sufficient evidence of a better predictor.

### TCN

| Seed | Existing primary RMSE | Smaller/higher-dropout primary RMSE | Existing gap | Smaller-model gap |
| --- | ---: | ---: | ---: | ---: |
| 42 | 23.69 | 25.79 | 4.56 | 1.00 |
| 73 | 23.59 | 25.63 | 3.91 | 1.71 |
| 101 | 24.13 | 23.62 | 5.46 | 2.04 |

Inner validation retained the existing model in all nine cases, so the
inner-selected procedure exactly reproduces its results. The smaller model
worsens primary RMSE by about 2.1 cycles for seeds 42 and 73; their paired
intervals exclude zero. Seed 101 improves by 0.51 cycles, but its interval
[-1.22, +0.13] includes zero. Selecting seed 101 after seeing these outer results
would violate the frozen selection rule. Near-failure RMSE worsens for all three
seeds: 12.84, 11.49 and 10.13, versus 8.22, 9.31 and 9.51 respectively.

**Implication:** the original gaps warrant transparent reporting, but reducing
capacity or fitting time can sacrifice useful predictive signal. Keep the
tested baselines, freeze further tuning for this comparison, and pursue
independent validation. The existing XGBoost remains better near failure in
these results; the original 50-cycle TCN remains better across the primary
planning window. No alert recalibration or new audit/test evaluation occurred.

Full results: [pooled metrics](metrics.csv), [fold metrics](fold_metrics.csv),
[inner selections](selections.json), and [paired comparisons](paired_comparisons.csv).
The intervals are conditional and unadjusted, as explained below.

## Frozen question and selection rule

Can a small regularization change improve held-out RUL prediction, rather than
merely shrink the training/validation gap? Primary planning-window RMSE remains
the selection metric. Near-failure RMSE (true RUL ≤30) is a separate secondary
outcome; it is not used to select candidates after seeing the outer results.

Use the same 567 development engines, three outer folds and endpoints as the
previous work. Within each fold, 302 engines fit inner models, 76 choose the
recipe/stopping point, and 189 separate engines evaluate it. Refits use all 378
outer fitting engines. Normalization, variance filtering and scaling are fitted
only on the corresponding fitting engines. Targets are capped at 150 for fitting;
all reported RMSE uses true uncapped RUL.

Choices are made separately inside each fold. The `inner_selected` result evaluates
that selection procedure; it is not a single globally chosen hyperparameter set.
The individual candidates are also scored as prespecified diagnostics. They must
not be substituted into the selection procedure based on their outer scores.
The final XGBoost model and all earlier results remain unchanged.

## XGBoost candidates

Keep the previous rich regime-normalized features, seed 42, learning rate 0.03,
row/column sampling 0.85 and training cap. Compare four options:

| Candidate | Depth | Lambda | Minimum child weight | Boosting rounds |
| --- | ---: | ---: | ---: | --- |
| Baseline | 6 | 10 | 1 | Fixed 800 |
| Early stopping | 6 | 10 | 1 | Inner-selected, maximum 800 |
| Shallower + regularized | 4 | 20 | 10 | Inner-selected, maximum 800 |
| Stronger regularization | 6 | 40 | 10 | Inner-selected, maximum 800 |

Fit three inner boosting trajectories through 800 rounds. The baseline and
its early-stopped counterpart share one trajectory. Replay a stopping rule
with patience 40 and a minimum improvement of 0.01 RMSE; later rounds do not
enter that choice. This is equivalent to selecting the earlier prediction
prefix of an additive model; it does not claim to save training compute.
Stopping uses XGBoost's native uncapped evaluation RMSE. Candidate ranking and
reported predictions apply the existing nonnegative prediction clip. Rank the
four candidates by inner primary RMSE, with exact ties favoring the baseline.
Refit each candidate from scratch for its chosen round count on the outer
fitting engines, and score the frozen inner choice on outer engines.

## TCN candidates

Compare the existing 50-cycle model (32 channels, dropout 0.1, 26,241 parameters)
with one prespecified combined change (16 channels, dropout 0.2, 6,993 parameters).
This experiment does not isolate width from dropout. Keep the residual causal
architecture, kernel widths/dilations, dense layer, optimizer, learning rate,
cap, batch size and history length fixed. Both use maximum 60 inner epochs,
patience four and minimum improvement 0.01 RMSE. Test seeds 42, 73 and 101 on all
three folds; no seed is chosen because it looks favorable on outer engines.

Reuse the existing baseline's inner selections and outer models, which were
trained on identical engine roles and preprocessing with three CPU threads.
Train and early-stop the regularized inner model, then compare its best inner
RMSE with the saved baseline's. Freeze this choice before outer refitting or
scoring. Refit the regularized candidate for its selected epoch count on all
378 fitting engines. Evaluate both candidates and the inner-selected procedure.
Recompute fitting/validation predictions for both to measure comparable gaps.
No alert threshold is retuned in this experiment.

## Interpretation and uncertainty

The outer engines are excluded from each model's fitting and inner selection.
However, these development folds have already been studied in prior experiments;
this is a bounded follow-up, not a new blind benchmark or fully independent
estimate of the entire historical search. No audit or official test labels are
used. The current final model is not replaced merely because a gap shrinks.

Pooled RMSE uses summed squared errors and counts, never averaged fold RMSE.
Each engine appears in two fitting folds and one validation fold; pooled target
distributions therefore match. Training predictions are repeated across models,
not independent replications. Report training error, validation error and their
gap together. A smaller gap with worse validation error is not an improvement.

Paired whole-engine bootstrap intervals use 2,000 draws, keeping each engine's
endpoints together. For near-failure scoring, resample engines having at least
one qualifying endpoint and recompute the count-weighted RMSE in each draw.
Differences are candidate minus the matching family/seed baseline; positive is
worse. These intervals condition on the fitted models and inner choices, exclude
selection/refitting uncertainty, and are not adjusted for multiple comparisons.
Seeds share the same engines and are not independent cohort replications.

## Reproduction

```bash
./nasa/bin/python scripts/run_regularization.py
./nasa/bin/python scripts/run_regularization.py --verify
```

The optional sequence environment is required. Baseline TCN checkpoints and
prior XGBoost OOF predictions must be present locally. Two CPU workers use three
threads each. Twelve resumable jobs produce 42 scored fold configurations,
including reused baselines and copies of frozen inner choices; this is not a
claim that 42 distinct models were trained. Interrupted incomplete jobs restart;
completed jobs are retained. Local `_runs/` holds models and detailed predictions
and is excluded from Git, as is the pooled OOF export. Compact metrics, protocol,
selection records and comparisons are retained.


Verification reloaded all saved outer models and reproduced their fitting and
validation predictions. It checked endpoint identities, engine roles, inner
choice records, stopping rules, 42 scored configurations, 47,628 scored endpoint
rows, frozen input/model hashes, 84 pooled metrics and 40 paired point differences.
Report execution and the existing HTML structure/link checks passed; no browser
preview was available for visual inspection of the added prose.
