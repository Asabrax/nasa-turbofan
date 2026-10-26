"""Meaningful checks for target isolation, causal history and asymmetric loss."""
import numpy as np
import pandas as pd
from experiment_models import HistoryFeatures, asymmetric_squared, ExperimentRegressor
from train_all import load_inputs, make_features, DATASETS, SENSORS
from run_experiments import configs, endpoints, score, choose_threshold

train, _, _ = load_inputs()
small = train[train.engine_id.le(3)]
pd.testing.assert_frame_equal(HistoryFeatures().fit(small).transform(small), make_features(small))
fit = small[small.engine_id.le(2)]
valid = small[small.engine_id.eq(3)]
transformer = HistoryFeatures(normalize=True, rich=True).fit(fit)
full = transformer.transform(valid)
assert transformer.fit_engines_.isdisjoint(valid.engine_key)
before = transformer.means_.copy()
for dataset in DATASETS:
    group = valid[valid.dataset.eq(dataset)]
    prefix = group.iloc[:40]
    pd.testing.assert_frame_equal(transformer.transform(prefix), full.loc[prefix.index])
    changed = group.copy()
    changed.loc[changed.cycle.gt(40), SENSORS] += 1000
    pd.testing.assert_frame_equal(transformer.transform(changed).loc[prefix.index], full.loc[prefix.index])
    # Labels must have no role in prediction feature construction.
    relabeled = group.copy()
    relabeled['RUL'] = -999
    pd.testing.assert_frame_equal(transformer.transform(relabeled), full.loc[group.index])
pd.testing.assert_frame_equal(before, transformer.means_)
assert full.loc[valid.cycle.eq(1), [s + '_delta' for s in SENSORS]].eq(0).all().all()
unknown = valid.iloc[:1].copy()
unknown['setting_1'] = 999
try:
    transformer.transform(unknown)
    raise AssertionError('Unknown regime was not rejected')
except ValueError:
    pass
actual, pred, weights = np.array([10., 10.]), np.array([8., 13.]), np.array([2., 3.])
grad, hess = asymmetric_squared(actual, pred, sample_weight=weights)
def loss(p):
    e = p - actual
    return .5 * np.where(e > 0, 1.5, 1.) * e**2 * weights
eps = 1e-4
np.testing.assert_allclose(grad, (loss(pred + eps) - loss(pred - eps))/(2*eps), rtol=1e-6)
np.testing.assert_allclose(hess, (loss(pred+eps)-2*loss(pred)+loss(pred-eps))/eps**2, rtol=1e-6)
assert score([20,20],[20,20])['NASA_score_sum'] == 0
assert score([20],[30])['NASA_score_sum'] > score([20],[10])['NASA_score_sum']
assert np.isclose(score([20],[7])['NASA_score_sum'], np.e-1)
assert np.isclose(score([20],[30])['NASA_score_sum'], np.e-1)
snap=endpoints(small)
assert snap.groupby(['engine_key','protocol']).size().eq(3).all()
assert snap.actual_RUL.gt(0).all()
assert snap[snap.protocol.eq('primary')].actual_RUL.le(200).all()
threshold,_=choose_threshold([10,20,50],[15,40,45],target=.9)
assert threshold==40
# Fit weighted custom objective with the installed XGBoost API.
config=next(c for c in configs() if c['name']=='rich_cap130_asymmetric')
config=dict(config,trees=3,balanced=True)
model=ExperimentRegressor(config,threads=2).fit(fit)
assert np.isfinite(model.predict(valid)).all()
print('PASS: raw-feature equivalence, future/target isolation, engine boundaries, train-only normalization, unseen-regime rejection, numerical gradient/Hessian, NASA score signs, endpoint alignment and weighted custom-objective fit.')
