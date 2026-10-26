"""Local, reproducible C-MAPSS training across all four datasets.

Run: ./nasa/bin/python scripts/train_all.py
No test targets are read until grouped CV has selected each model's parameters.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
SENSORS = [f'sensor_{i}' for i in range(1, 22)]
SETTINGS = [f'setting_{i}' for i in range(1, 4)]
COLUMNS = ['engine_id', 'cycle'] + SETTINGS + SENSORS
SEED = 42
WINDOW = 5
HORIZON = 30


def load_inputs():
    manifest = json.loads((ROOT / 'data/source.json').read_text())
    frames = {'train': [], 'test': []}
    audit = []
    for dataset in DATASETS:
        for split in frames:
            relative = f'data/raw/{split}_{dataset}.txt'
            digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            assert digest == manifest['files'][relative]['sha256'], relative
            frame = pd.read_csv(ROOT / relative, sep=r'\s+', header=None)
            assert frame.shape[1] == 26
            frame.columns = COLUMNS
            assert np.isfinite(frame.to_numpy()).all()
            assert not frame.duplicated(['engine_id', 'cycle']).any()
            assert (frame[['engine_id', 'cycle']] > 0).all().all()
            assert (frame[['engine_id', 'cycle']] % 1 == 0).all().all()
            frame = frame.sort_values(['engine_id', 'cycle']).reset_index(drop=True)
            assert frame.groupby('engine_id').cycle.min().eq(1).all()
            assert frame.groupby('engine_id').cycle.diff().dropna().eq(1).all()
            frame['dataset'] = dataset
            frame['engine_key'] = dataset + '_' + frame.engine_id.astype(str)
            if split == 'train':
                frame['RUL'] = frame.groupby('engine_key').cycle.transform('max') - frame.cycle
                assert frame.groupby('engine_key').tail(1).RUL.eq(0).all()
            audit.append({'dataset': dataset, 'split': split, 'rows': len(frame),
                          'engines': frame.engine_key.nunique(), 'sha256': digest})
            frames[split].append(frame)
    train, test = (pd.concat(frames[s], ignore_index=True) for s in ['train', 'test'])
    assert train.engine_key.nunique() == 709
    assert test.engine_key.nunique() == 707
    return train, test, pd.DataFrame(audit)


def make_features(frame):
    """Only present/past data; engine_key scopes history within each dataset."""
    assert not frame.duplicated(['engine_key', 'cycle']).any()
    ordered = frame.sort_values(['engine_key', 'cycle'])
    features = ordered[['cycle'] + SETTINGS + SENSORS].copy()
    for dataset in DATASETS:
        features[f'dataset_{dataset}'] = ordered.dataset.eq(dataset).astype(int)
    groups = ordered.groupby('engine_key', sort=False)
    for sensor in SENSORS:
        features[sensor + '_mean5'] = groups[sensor].transform(
            lambda series: series.rolling(WINDOW, min_periods=1).mean())
        features[sensor + '_delta'] = groups[sensor].diff().fillna(0)
    return features.loc[frame.index].astype(np.float32)


def metrics(actual, predicted):
    positive, flagged = np.asarray(actual) <= HORIZON, np.asarray(predicted) <= HORIZON
    tn, fp, fn, tp = confusion_matrix(positive, flagged, labels=[False, True]).ravel()
    return {'engines_or_rows': len(actual),
            'RMSE_cycles': float(np.sqrt(mean_squared_error(actual, predicted))),
            'MAE_cycles': float(mean_absolute_error(actual, predicted)),
            'R2': float(r2_score(actual, predicted)),
            'mean_error_cycles': float(np.mean(predicted - np.asarray(actual))),
            'precision': float(precision_score(positive, flagged, zero_division=0)),
            'recall': float(recall_score(positive, flagged, zero_division=0)),
            'F1': float(f1_score(positive, flagged, zero_division=0)),
            'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}


def candidates(threads):
    def pipeline(model, scale=False):
        steps = [('variance', VarianceThreshold())]
        if scale:
            steps.append(('scale', StandardScaler()))
        return Pipeline(steps + [('model', model)])
    return [
        ('Mean baseline', 'Mean baseline', pipeline(DummyRegressor())),
        ('Ridge', 'Ridge alpha=100', pipeline(Ridge(alpha=100), scale=True)),
        ('Random forest', 'Random forest depth=14', pipeline(RandomForestRegressor(
            n_estimators=80, max_depth=14, min_samples_leaf=10, max_features=.7,
            n_jobs=threads, random_state=SEED))),
        *[('XGBoost', f'XGBoost depth={depth}', pipeline(XGBRegressor(
            n_estimators=300, max_depth=depth, learning_rate=.05,
            subsample=.85, colsample_bytree=.85, reg_lambda=10,
            objective='reg:squarederror', tree_method='hist', device='cpu',
            n_jobs=threads, random_state=SEED))) for depth in [3, 6]]
    ]


def run_training(threads=6):
    started = time.perf_counter()
    out = ROOT / 'outputs/all_engines'
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / 'training.log'
    log_path.write_text('')
    def log(message):
        line = f'{datetime.now(timezone.utc).isoformat(timespec="seconds")} {message}'
        print(line, flush=True)
        with log_path.open('a') as handle:
            handle.write(line + '\n')

    log(f'Actual local CPU training started: threads={threads}, platform={platform.platform()}')
    train, test, audit = load_inputs()
    audit.to_csv(out / 'data_audit.csv', index=False)
    X, y, groups = make_features(train), train.RUL, train.engine_key
    assert np.isfinite(X.to_numpy()).all()
    assert 'engine_id' not in X and 'RUL' not in X
    # Check causality and engine-boundary reset in every subset.
    for dataset in DATASETS:
        prefix = train[(train.dataset == dataset) & (train.engine_id == 1)].iloc[:25]
        pd.testing.assert_frame_equal(make_features(prefix), X.loc[prefix.index])
    assert X.loc[train.cycle.eq(1), [s + '_delta' for s in SENSORS]].eq(0).all().all()
    log(f'Validated {len(train):,} training rows / 709 engines; {len(test):,} test rows / 707 engines; {X.shape[1]} features.')
    folds = list(GroupKFold(n_splits=3).split(X, y, groups))
    assignments = []
    for fold, (fit_idx, valid_idx) in enumerate(folds, 1):
        assert set(groups.iloc[fit_idx]).isdisjoint(groups.iloc[valid_idx])
        assignments.extend({'engine_key': key, 'validation_fold': fold} for key in sorted(set(groups.iloc[valid_idx])))
        assert set(train.iloc[valid_idx].dataset) == set(DATASETS)
    pd.DataFrame(assignments).to_csv(out / 'fold_assignments.csv', index=False)
    configs = candidates(threads)
    fold_rows, summary_rows = [], []
    for family, name, prototype in configs:
        start = time.perf_counter()
        for fold, (fit_idx, valid_idx) in enumerate(folds, 1):
            model = clone(prototype)
            fit_start = time.perf_counter()
            log(f'Fitting {name}, fold {fold}/3 ({len(fit_idx):,} rows)')
            model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
            pred = np.maximum(0, model.predict(X.iloc[valid_idx]))
            fit_pred = np.maximum(0, model.predict(X.iloc[fit_idx]))
            fold_rows.append({'family': family, 'candidate': name, 'fold': fold,
                              'train_RMSE': float(np.sqrt(mean_squared_error(y.iloc[fit_idx], fit_pred))),
                              'CV_RMSE': float(np.sqrt(mean_squared_error(y.iloc[valid_idx], pred))),
                              'fit_predict_seconds': time.perf_counter() - fit_start})
            pd.DataFrame(fold_rows).to_csv(out / 'cv_folds.csv', index=False)
            log(f'Finished {name}, fold {fold}: RMSE={fold_rows[-1]["CV_RMSE"]:.3f}')
        own = pd.DataFrame([r for r in fold_rows if r['candidate'] == name])
        summary_rows.append({'family': family, 'candidate': name, 'train_RMSE': own.train_RMSE.mean(),
                             'CV_RMSE': own.CV_RMSE.mean(), 'fold_SD': own.CV_RMSE.std(ddof=0),
                             'CV_seconds': time.perf_counter() - start})
        pd.DataFrame(summary_rows).to_csv(out / 'model_comparison.csv', index=False)
    comparison = pd.DataFrame(summary_rows).sort_values('CV_RMSE')
    winner = comparison.iloc[0]['candidate']
    selected = comparison.groupby('family', sort=False).head(1)
    log(f'CV selected overall winner: {winner}; now refitting best candidate in every family on ALL 709 engines.')
    final_models, fit_times = {}, {}
    for _, row in selected.iterrows():
        name = row['candidate']
        prototype = next(model for _, candidate, model in configs if candidate == name)
        start = time.perf_counter()
        log(f'Final fit: {name} on {len(X):,} rows')
        model = clone(prototype).fit(X, y)
        family = row['family']
        final_models[family] = model
        fit_times[family] = time.perf_counter() - start
        slug = family.lower().replace(' ', '_')
        joblib.dump({'pipeline': model, 'features': list(X.columns), 'datasets': DATASETS,
                     'window': WINDOW, 'horizon': HORIZON, 'candidate': name,
                     'training_engines': 709, 'training_rows': len(train)}, out / f'{slug}.joblib')
        if family == 'XGBoost':
            model.named_steps['model'].save_model(out / 'xgboost_booster.ubj')
        log(f'Saved {family}; final fit took {fit_times[family]:.1f}s')

    # Open test labels only after all model/parameter decisions and final fits.
    endpoints = test.groupby('engine_key').tail(1).sort_values(['dataset', 'engine_id']).copy()
    labels = []
    manifest = json.loads((ROOT / 'data/source.json').read_text())
    for dataset in DATASETS:
        relative = f'data/raw/RUL_{dataset}.txt'
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == manifest['files'][relative]['sha256']
        truth = pd.read_csv(ROOT / relative, sep=r'\s+', header=None)
        ids = endpoints.loc[endpoints.dataset.eq(dataset), 'engine_id'].tolist()
        assert truth.shape == (len(ids), 1) and ids == list(range(1, len(ids) + 1))
        assert np.isfinite(truth.to_numpy()).all() and truth[0].ge(0).all()
        labels.extend(truth[0].tolist())
    endpoints['actual_RUL'] = labels
    X_test = make_features(test).loc[endpoints.index]
    metric_rows, prediction_frames = [], []
    for family, model in final_models.items():
        predicted = np.maximum(0, model.predict(X_test))
        prediction = endpoints[['dataset', 'engine_id', 'engine_key', 'cycle', 'actual_RUL']].copy()
        prediction['family'] = family
        prediction['predicted_RUL'] = predicted
        prediction['error_cycles'] = predicted - prediction.actual_RUL
        prediction['maintenance_flag'] = predicted <= HORIZON
        prediction_frames.append(prediction)
        for dataset in ['ALL'] + DATASETS:
            frame = prediction if dataset == 'ALL' else prediction[prediction.dataset.eq(dataset)]
            metric_rows.append({'family': family, 'dataset': dataset,
                                **metrics(frame.actual_RUL.to_numpy(), frame.predicted_RUL.to_numpy())})
        # Verify that the persisted artifact reproduces its predictions.
        slug = family.lower().replace(' ', '_')
        saved = joblib.load(out / f'{slug}.joblib')
        np.testing.assert_allclose(np.maximum(0, saved['pipeline'].predict(X_test[saved['features']])), predicted)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(out / 'test_predictions.csv', index=False)
    test_metrics = pd.DataFrame(metric_rows)
    test_metrics.to_csv(out / 'test_metrics.csv', index=False)
    result = {'datasets': DATASETS, 'training_engines': 709, 'test_engines': 707,
              'training_rows': len(train), 'test_rows': len(test), 'features': list(X.columns),
              'selected_candidate': winner, 'selected_family': comparison.iloc[0]['family'],
              'seed': SEED, 'threads': threads, 'window': WINDOW, 'horizon': HORIZON,
              'final_fit_seconds': fit_times, 'total_seconds': time.perf_counter() - started,
              'finished_utc': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
              'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                           'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__},
              'selected_candidates': selected[['family', 'candidate']].to_dict('records'),
              'test_metrics': metric_rows}
    (out / 'results.json').write_text(json.dumps(result, indent=2))
    log(f'COMPLETE: {result["total_seconds"]:.1f}s elapsed. Saved four fitted model artifacts and verified reload predictions.')
    print(test_metrics[test_metrics.dataset.eq('ALL')].to_string(index=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=min(6, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    run_training(args.threads)
