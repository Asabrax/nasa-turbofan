# Experiment protocol and interpretation

The original all-engine models and predictions remain under `../all_engines`.
This run uses all four local C-MAPSS datasets, with 709 training engines and 707
official test engines. The local manifest controls input provenance; all twelve
raw input hashes are checked. No new data downloads are involved.

## Selection protocol

Before training, reserve 142 engines (20%, stratified by dataset, seed 42) for an
untouched audit. Use 567 engines for three-fold engine CV. Every engine has three
fixed synthetic observation endpoints per protocol. Feature transforms include
only the current and earlier measurements. Regime statistics and variance
filtering are fitted on the fitting engines only.

The primary protocol samples true remaining life uniformly from 1 through
min(200, lifetime minus 20), with at least 20 observed cycles. The 200-cycle
planning window is an assumption, not an estimate of official test labels.
The broad sensitivity protocol samples observed cycles uniformly from 20
through lifetime minus 1. It includes longer remaining lifetimes. Each engine
has equal endpoint weight in both protocols. Neither changes true labels.

Compare 18 predefined ablations, then three depth/rounds configurations on the
first-stage winner. Choose the smallest pooled primary OOF RMSE. This bounded
matrix is not exhaustive and does not evaluate all interactions. The primary
selection protocol intentionally emphasizes a finite planning window; a cap
can reduce its error while increasing early-life error on the broad protocol.

Choose the smallest prediction threshold achieving at least 90% OOF recall for
true RUL <=30. This shares OOF data with model selection, so its result is not an
independent recall guarantee. Freeze the model and threshold before audit.
Evaluate reference and selected models trained on the same development engines
on all 142 audit engines. Do not change the selection after audit results.
Bootstrap paired RMSE differences by whole engine (1,000 draws, seed 42).

Refit the frozen configuration on every training row of all 709 engines. Only
then read official test targets and evaluate the 707 endpoints. Because the
previous official test results motivated these experiments, they are not a new
blind external benchmark. No post-test retuning is performed. Refit can change
alert calibration; audit/test recall is reported rather than guaranteed.

## Implemented hypotheses

- Capped training RUL at 120, 130 and 150 versus uncapped; evaluation remains true,
  uncapped RUL. No imposed upper prediction clip.
- Fixed rounded setting regimes, with training-only sensor means/standard
  deviations per dataset/regime; unknown regimes rejected.
- Causal five- and twenty-cycle means, twenty-cycle standard deviations and
  secant slopes, ten-cycle-half-life exponentially weighted means, and deltas.
- Fold-fitted variance filtering, without assuming a universal sensor blacklist.
- Engine-balanced training, removing cycle, separate per-dataset models and a
  1.5x overestimation penalty as matched alternatives.
- Small follow-up XGBoost depth/round-count grid on the chosen feature/target
  setup. No early stopping against audit or test data.
- Fixed versus validation-calibrated maintenance thresholds, NASA score and
  normalized tree-gain feature importance (not causal importance or SHAP).

## References

NASA asymmetric score: Saxena et al., *Damage Propagation Modeling for Aircraft
Engine Prognostics*, equation 9:
https://ntrs.nasa.gov/api/citations/20090029214/downloads/20090029214.pdf

For d = predicted minus actual RUL, sum exp(-d/13)-1 for d<0 and exp(d/10)-1
otherwise. Lower is better. Scores can be dominated by a few large errors.
Retain both sums and per-observation means; do not compare score sums across
different sample counts as though the populations were identical.

XGBoost custom objective contract (including explicit sample weights):
https://xgboost.readthedocs.io/en/latest/python/python_api.html

## Verification

`./nasa/bin/python scripts/check_experiments.py` checks causal truncation,
future/label isolation, engine boundaries, raw baseline feature equivalence,
train-only normalization, unknown-regime rejection, numerical gradients and
Hessians, weighted custom-objective fitting, score signs, and endpoint rules.

`./nasa/bin/python scripts/verify_experiment_results.py` independently recomputes
selection, alert threshold, endpoint membership, official-label alignment,
RMSE/MAE, NASA score and confusion counts. Final model serialization is checked
against predictions on complete official test histories in the training script.
