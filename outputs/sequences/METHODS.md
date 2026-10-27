# Sequence model comparison

This is a bounded development-only comparison of a compact LSTM and a temporal
convolutional network (TCN) against the existing selected XGBoost configuration.
It does not replace the final model or re-evaluate official test labels.

## Results from the local run

Training completed in 1,980 seconds (33 minutes), with 12 scored outer-fold
models plus their inner early-stopping fits. Existing XGBoost predictions were
kept unchanged. All scores below use development CV, not official test endpoints.

| Model | Primary RMSE | Broad RMSE | Recall at 30 | Precision at 30 |
| --- | ---: | ---: | ---: | ---: |
| Existing XGBoost | 25.93 | 45.32 | 81.2% | 93.3% |
| LSTM, two seeds | 26.43–26.63 | 45.69–45.81 | 80.2–81.9% | 90.2–94.4% |
| TCN, two seeds | 26.02–26.19 | 45.75–45.84 | 62.8–70.3% | 98.1–99.5% |

The ranges summarize two seeds, not confidence intervals. All four paired 95%
intervals for primary RMSE difference include zero: these runs do not establish
a clear accuracy advantage for either model family. TCN's much lower recall at
the fixed threshold shows that similar RMSE does not imply similar alert behavior.
Its threshold was not recalibrated. XGBoost remains the existing final model;
this comparison supplies no demonstrated improvement that warrants replacing it.

Matched training-to-validation RMSE gaps range from 1.81 to 5.36 cycles for LSTM
and −0.68 to 3.49 for TCN across folds/seeds. These are descriptive in-sample versus
held-out differences, not proof that either architecture is free of overfitting.
Several fits reached the 20-epoch budget. Different windows, architectures,
training budgets or calibration could change the result.

[Per-seed metrics](metrics.csv), [paired comparisons](paired_comparison.csv),
[training/validation gaps](generalization_gaps.csv), [fit details](fits.csv) and
[training log](training.log) retain the evidence.

## Matched evaluation

Use the saved three-fold assignments for the same 567 development engines and
three primary plus three broad endpoints per engine. Both networks use 30-cycle
trailing windows. Short histories are left-padded with zeros and an observation
indicator distinguishes real measurements from padding. The networks receive
regime-normalized sensors plus cycle, operating settings and dataset indicators.
They do not receive engineered rolling summaries. Normalization, sensor variance
filtering and metadata scaling are fitted on fitting engines only.

Within each outer fold, reserve a stratified 20% of fitting engines for early
stopping. Train on all cycles of the remaining engines, using capped RUL/150 as
the target. Select the epoch count by uncapped RMSE on the saved primary endpoints
of the stopping engines, with patience four, minimum improvement 0.01 cycles,
and a maximum of 20 epochs. Refit preprocessing and the network on ALL outer
fitting engines for that epoch count. Score the separate outer validation engines
only after refitting. The outer validation engines do not determine stopping.

Both architectures use Adam (learning rate 0.001, gradient norm clipping at 1),
batches of 512, dropout 0.1 and random seeds 42 and 73. The LSTM has 32 recurrent
units and a 16-unit dense layer. The TCN has 32 channels, residual blocks with
two causal kernel-3 convolutions at each dilation 1, 2, 4 and 8, then a 16-unit
dense layer. The readout is the last time step. Prediction is clipped only at
zero, not at the training target cap. No window-length or architecture search
is performed after seeing the results.

Report each seed's pooled out-of-fold metrics separately. Seeds are repeated
training runs on the same engines, not additional independent observations.
The existing XGBoost reference uses the same outer folds/endpoints and all fitting
rows, but engineered temporal summaries and a previously selected configuration.
This is a comparison of these specific pipelines and budgets, not an exhaustive
comparison of neural and tree model families. The development data have already
informed earlier experiments. Results are follow-up evidence, not blind validation.

## Overfitting and alerts

`learning_curves.csv` records capped training RMSE and uncapped inner-endpoint
RMSE. Their target definitions and sampled populations differ, so their numerical
gap must not be interpreted as a matched train/validation generalization gap.
Use the inner curve to inspect training progress and the outer endpoints for
reported performance. `generalization_gaps.csv` separately compares uncapped
primary-endpoint errors on fitting and evaluation engines after refitting; these
use the same target and sampling protocol. Training errors remain in-sample. A run reaching the epoch limit is budget-limited; it does
not establish that longer training cannot improve it.

All alert metrics use predicted RUL ≤30 for actual RUL ≤30. The XGBoost-specific
34.44 threshold is not transferred to the neural models. There is no new alert
threshold calibration in this comparison.

When the original XGBoost OOF export is available, `paired_comparison.csv`
reports sequence-minus-XGBoost RMSE differences with 2,000 paired whole-engine
bootstrap draws. Each draw retains the three endpoints together. These intervals
condition on the fitted models and existing development endpoints; they exclude
model-selection and refitting uncertainty. Development engines are not a new
blind sample. Positive differences favor XGBoost; an interval crossing zero does
not establish a meaningful advantage for either model.

## Reproduction

```bash
./nasa/bin/python -m pip install -r requirements-sequences.txt
./nasa/bin/python scripts/run_sequences.py
./nasa/bin/python scripts/run_sequences.py --verify
```

The recorded environment is saved alongside the results. The local run uses
TensorFlow 2.22.0rc0 on CPU with six intra-operation threads, deterministic ops,
and no GPU. Saved `.keras` fold models and detailed OOF exports are ignored by
Git; compact metrics, split roles and learning curves remain tracked. Fold-model
files contain neural weights; preprocessing is reconstructed from their fitting
engines by the verification script and is not a standalone deployment artifact. Raw inputs
must first be prepared as described in `docs/REPRODUCING.md`.

Implementation reference: [TensorFlow causal Conv1D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D).
