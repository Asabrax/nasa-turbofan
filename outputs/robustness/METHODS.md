# Fixed-recipe robustness and alert timing

Both recipes were actually retrained locally for seeds 7, 19, 42, 73 and 101,
using six CPU threads. Each split is stratified by dataset: 567 fitting engines
and 142 validation engines. All four datasets and all 709 engines participate in
split assignment. No engine crosses roles within a split. Normalization and
variance filtering are fitted only on fitting histories. Features are causal;
prefix predictions were checked against full-history predictions.

`frozen_protocol.json` preserves the exact recipes and threshold. Each recipe
is trained once per split, then evaluated under all five endpoint scenarios.
Three observation points per engine are sampled without replacement, uniformly
from cycles max(20, lifetime − horizon) through lifetime − 1 for horizons 100,
150, 200 and 250. Broad sampling uses cycles 20 through lifetime − 1. Sampling
uses a deterministic engine/scenario seed, shared by both recipes and across
splits. Evaluation labels remain uncapped. Engines have equal endpoint weight.
These endpoints differ from the initial experiment's audit endpoints.

This is a sensitivity analysis, not new model selection or independent blind
replication. Split validation populations overlap, and engines may have informed
previous configuration selection. Ranges across five splits are descriptive,
not confidence intervals. The saved all-engine final model is unchanged; its
SHA-256 is recorded and checked. No official test labels inform these refits.

## Snapshot uncertainty

The official test has one endpoint per engine. Precision intervals condition on
the number of predicted alerts; recall intervals condition on actual RUL ≤30.
Each is a 95% Wilson binomial interval. Pooled intervals describe the observed
mixture of datasets under an independent-engine sampling interpretation.
They do not account for selection, model refitting or a different fleet mixture.
Per-dataset intervals are also saved in `test_alert_intervals.csv`.

For the original audit-primary population, 2,000 percentile bootstrap draws
resample whole engines and retain all three endpoints per sampled engine.
Undefined precision/recall draws, if any, are excluded. This respects within-engine
endpoint dependence. These intervals describe a separate population from the
official test. They are not evidence of a guaranteed 90% recall target.

## Full-history first alerts

For each held-out engine, the original recipe uses threshold 30. The final recipe
uses 30 and the previously frozen 34.438228607177734. Monitoring starts at cycle
20 and stops one cycle before failure. A flag is predicted RUL ≤ threshold;
the event window remains true RUL ≤30. Failure-cycle detections do not count.
First-alert lead is true RUL at the first flagged cycle. We distinguish 1–30,
31–60, more than 60 cycles, and no alert. A warning at 31 cycles is not treated
as equivalent in practical importance to one at 100 cycles.

The replay is passive: alerts do not change maintenance or engine operation.
Early episodes count transitions into a flag while true RUL exceeds 30, with
monitoring initially unflagged. Any detection within the final 30 cycles is a
repeated-opportunity measure, unlike endpoint recall. Summary plots average
split-level proportions; repeated engines are not counted as independent
replications. Inspection costs, alarm persistence and interventions need a
separate operational policy before deployment conclusions are possible.

## Reproduction

From the repository root:

```bash
./nasa/bin/python scripts/run_robustness.py
./nasa/bin/python scripts/verify_robustness.py
./nasa/bin/python scripts/build_presentation.py
./nasa/bin/python scripts/execute_notebook.py
./nasa/bin/python scripts/render_presentation.py
```

Training overwrites only robustness outputs. Rendering reads saved artifacts.
`training.log` records local fitting; `snapshot_predictions.csv` and
`heldout_history_predictions.csv` retain evidence for recomputation.
