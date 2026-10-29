"""Fold-fitted, causal features and reusable RUL predictors for experiments."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBRegressor
from train_all import DATASETS, SENSORS, SETTINGS


def regime_keys(frame):
    # Fixed rounding rules; not inferred from validation/test sensor statistics.
    rounded = frame[SETTINGS].round({'setting_1': 0, 'setting_2': 2, 'setting_3': 0})
    # Integer representation avoids negative-zero string discrepancies.
    return (frame.dataset.astype(str) + ':' + rounded.setting_1.astype(int).astype(str)
            + ':' + (rounded.setting_2 * 100).round().astype(int).astype(str)
            + ':' + rounded.setting_3.astype(int).astype(str))


class HistoryFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, normalize=False, rich=False):
        self.normalize = normalize
        self.rich = rich

    def fit(self, frame, y=None):
        if self.normalize:
            values = frame[SENSORS].copy()
            values['regime'] = regime_keys(frame)
            groups = values.groupby('regime')[SENSORS]
            self.means_ = groups.mean()
            self.scales_ = groups.std(ddof=0).mask(lambda x: x < 1e-6, 1.0).fillna(1.0)
        self.fit_engines_ = set(frame.engine_key)
        return self

    def transform(self, frame):
        assert frame.index.is_unique
        assert not frame.duplicated(['engine_key', 'cycle']).any()
        ordered = frame.sort_values(['engine_key', 'cycle'])
        values = ordered[SENSORS].copy()
        if self.normalize:
            keys = regime_keys(ordered)
            unknown = set(keys) - set(self.means_.index)
            if unknown:
                raise ValueError(f'Operating regimes absent during fit: {sorted(unknown)}')
            means = self.means_.loc[keys].to_numpy()
            scales = self.scales_.loc[keys].to_numpy()
            values = pd.DataFrame((values.to_numpy() - means) / scales,
                                  index=ordered.index, columns=SENSORS)
        features = ordered[['cycle'] + SETTINGS].copy()
        for dataset in DATASETS:
            features[f'dataset_{dataset}'] = ordered.dataset.eq(dataset).astype(int)
        for sensor in SENSORS:
            features[sensor] = values[sensor]
        groups = values.groupby(ordered.engine_key, sort=False)
        # Grouped windows return (engine, row) indices; retain row labels for alignment.
        mean5 = groups.rolling(5, min_periods=1).mean().droplevel(0).add_suffix('_mean5')
        delta = groups.diff().fillna(0).add_suffix('_delta')
        parts = [features, mean5, delta]
        if self.rich:
            mean20 = groups.rolling(20, min_periods=1).mean().droplevel(0).add_suffix('_mean20')
            std20 = groups.rolling(20, min_periods=1).std(ddof=0).droplevel(0).add_suffix('_std20')
            ewm = groups.ewm(halflife=10, adjust=False).mean().droplevel(0).add_suffix('_ewm10')
            # Secant slope over the last 20 observations, with shorter initial windows.
            lag = groups.shift(19)
            first = groups.transform('first')
            denominator = ordered.groupby('engine_key').cumcount().clip(upper=19).replace(0, 1)
            slope = (values - lag.fillna(first)).div(denominator, axis=0).add_suffix('_slope20')
            parts += [mean20, std20, ewm, slope]
        result = pd.concat(parts, axis=1)
        basic_order = (['cycle'] + SETTINGS + SENSORS + [f'dataset_{d}' for d in DATASETS]
                       + [name for s in SENSORS for name in [s + '_mean5', s + '_delta']])
        order = basic_order + [name for name in result if name not in basic_order]
        result = result.loc[frame.index, order].astype(np.float32)
        assert np.isfinite(result.to_numpy()).all()
        return result


def asymmetric_squared(y_true, y_pred, *, sample_weight=None):
    """Derivatives of 0.5 * weight * multiplier * error**2.

    The multiplier is 1.5 for overestimation and 1 otherwise. Both derivatives
    must receive it. This changes Hessian totals and therefore interacts with
    reg_lambda/min_child_weight; it is not a pure change of prediction bias.
    Mean-one sample weights preserve total row weight, not engine count.
    At zero error we use the lower-side curvature convention.
    """
    error = y_pred - y_true
    multiplier = np.where(error > 0, 1.5, 1.0)
    if sample_weight is not None:
        multiplier = multiplier * sample_weight
    return multiplier * error, multiplier * np.ones_like(error)


def build_model(config, threads):
    return XGBRegressor(
        n_estimators=config.get('trees', 300), max_depth=config.get('depth', 6),
        learning_rate=config.get('learning_rate', .05), subsample=.85,
        colsample_bytree=.85, reg_lambda=10, tree_method='hist', device='cpu',
        objective=asymmetric_squared if config.get('asymmetric') else 'reg:squarederror',
        n_jobs=threads, random_state=42)


class ExperimentRegressor:
    """Fitted normalization, feature order, selectors and pooled/subset estimators."""
    def __init__(self, config, threads=6):
        self.config = dict(config)
        self.threads = threads

    def fit_features(self, X, frame):
        self.columns_ = [c for c in X if self.config.get('cycle', True) or c != 'cycle']
        X = X[self.columns_]
        self.models_, self.selectors_ = {}, {}
        groups = DATASETS if self.config.get('separate') else ['ALL']
        for key in groups:
            mask = np.ones(len(frame), dtype=bool) if key == 'ALL' else frame.dataset.eq(key).to_numpy()
            subset = frame.loc[mask]
            selector = VarianceThreshold(threshold=1e-8).fit(X.loc[mask])
            target = subset.RUL.to_numpy()
            cap = self.config.get('cap')
            if cap is not None:
                target = np.minimum(target, cap)
            weights = None
            if self.config.get('balanced'):
                weights = 1 / subset.groupby('engine_key').engine_key.transform('size').to_numpy()
                weights = weights / weights.mean()
            model = build_model(self.config, self.threads)
            model.fit(selector.transform(X.loc[mask]), target, sample_weight=weights)
            self.models_[key], self.selectors_[key] = model, selector
        return self

    def predict_features(self, X, frame):
        X = X[self.columns_]
        pred = np.empty(len(frame), dtype=float)
        for key, model in self.models_.items():
            mask = np.ones(len(frame), dtype=bool) if key == 'ALL' else frame.dataset.eq(key).to_numpy()
            if mask.any():
                pred[mask] = model.predict(self.selectors_[key].transform(X.loc[mask]))
        return np.maximum(0, pred)

    def fit(self, frame):
        self.features_ = HistoryFeatures(self.config.get('normalize', False), self.config.get('rich', False)).fit(frame)
        return self.fit_features(self.features_.transform(frame), frame)

    def predict(self, frame):
        return self.predict_features(self.features_.transform(frame), frame)
