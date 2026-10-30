"""Controlled local experiments; test labels accessed only after choices freeze."""
from pathlib import Path
import hashlib
import json
import time
from datetime import datetime, timezone
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from experiment_models import HistoryFeatures, ExperimentRegressor
from train_all import ROOT, DATASETS, load_inputs
from cmapps.evaluation import score

OUT = ROOT / 'outputs/experiments'


def endpoints(frame):
    """Three reproducible snapshots per engine per protocol, before failure.

    Primary: uniform true RUL 1..min(200, lifetime-20). This is an explicitly
    assumed 200-cycle planning horizon, not an estimate of test-label density.
    Sensitivity: uniform observation cycle 20..lifetime-1, including early life.
    Sampling uses training lifetimes to create labels, never as model features.
    """
    records = []
    for key, group in frame.groupby('engine_key', sort=True):
        seed = int(hashlib.sha256(('42:' + key).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        lifetime = int(group.cycle.max())
        ranges = {'primary': np.arange(max(20, lifetime - 200), lifetime),
                  'broad': np.arange(20, lifetime)}
        for protocol, possible in ranges.items():
            for snapshot, cycle in enumerate(rng.choice(possible, size=3, replace=False)):
                row = group[group.cycle.eq(cycle)].iloc[0]
                records.append({'row_index': int(row.name), 'engine_key': key, 'dataset': row.dataset,
                                'engine_id': int(row.engine_id), 'cycle': int(cycle),
                                'actual_RUL': float(row.RUL), 'protocol': protocol, 'snapshot': snapshot})
    return pd.DataFrame(records)


def configs():
    base = {'normalize': False, 'rich': False, 'cap': None, 'balanced': False,
            'cycle': True, 'asymmetric': False, 'separate': False,
            'depth': 6, 'trees': 300, 'learning_rate': .05}
    rows = []
    def add(name, **changes):
        rows.append(dict(base, name=name, **changes))
    add('reference_raw_uncapped')
    for cap in [120, 130, 150]:
        add(f'raw_cap{cap}', cap=cap)
    add('regime_uncapped', normalize=True)
    add('regime_cap130', normalize=True, cap=130)
    add('rich_uncapped', normalize=True, rich=True)
    for cap in [120, 130, 150]:
        add(f'rich_cap{cap}', normalize=True, rich=True, cap=cap)
    for cap in [None, 130]:
        label = 'uncapped' if cap is None else 'cap130'
        add(f'rich_{label}_balanced', normalize=True, rich=True, cap=cap, balanced=True)
        add(f'rich_{label}_no_cycle', normalize=True, rich=True, cap=cap, cycle=False)
        add(f'rich_{label}_asymmetric', normalize=True, rich=True, cap=cap, asymmetric=True)
        add(f'separate_{label}', normalize=True, rich=True, cap=cap, separate=True)
    return rows


def choose_threshold(actual, pred, target=.90):
    thresholds = np.unique(np.r_[30., np.asarray(pred), 0.])
    rows = [score(actual, pred, threshold=t) for t in thresholds]
    table = pd.DataFrame(rows)
    feasible = table[table.recall.ge(target)]
    assert len(feasible)
    # Smallest threshold achieving target recall minimizes the alert count.
    threshold = float(feasible.threshold.min())
    return threshold, table


def evaluate_snapshots(model, frame, snapshot_table):
    full_pred = pd.Series(model.predict(frame), index=frame.index)
    result = snapshot_table.copy()
    result['predicted_RUL'] = full_pred.loc[result.row_index].to_numpy()
    return result


def run(threads=6):
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'training.log').write_text('')
    def log(s):
        line = f'{datetime.now(timezone.utc).isoformat(timespec="seconds")} {s}'
        print(line, flush=True)
        with (OUT / 'training.log').open('a') as handle:
            handle.write(line + '\n')
    log('Starting controlled local experiments. Official test labels are not used in selection.')
    train, test, audit = load_inputs()
    audit.to_csv(OUT / 'data_audit.csv', index=False)
    engines = train[['engine_key', 'dataset']].drop_duplicates().sort_values('engine_key')
    dev_keys, audit_keys = train_test_split(engines.engine_key, test_size=.2, random_state=42,
                                           stratify=engines.dataset)
    dev = train[train.engine_key.isin(dev_keys)]
    untouched = train[train.engine_key.isin(audit_keys)]
    assert set(dev.engine_key).isdisjoint(untouched.engine_key)
    split = engines.assign(role=np.where(engines.engine_key.isin(dev_keys), 'development', 'audit'))
    split.to_csv(OUT / 'engine_split.csv', index=False)
    snapshots = endpoints(train)
    snapshots.to_csv(OUT / 'validation_endpoints.csv', index=False)
    dev_engines = engines[engines.engine_key.isin(dev_keys)]
    log(f'{len(dev_keys)} development engines; {len(audit_keys)} untouched audit engines. Three snapshots per engine/protocol.')
    folds = []
    assignments = []
    for fold, (a, b) in enumerate(StratifiedKFold(3, shuffle=True, random_state=42).split(dev_engines, dev_engines.dataset), 1):
        fit_keys, val_keys = dev_engines.iloc[a].engine_key, dev_engines.iloc[b].engine_key
        assert set(fit_keys).isdisjoint(val_keys)
        folds.append((dev[dev.engine_key.isin(fit_keys)], dev[dev.engine_key.isin(val_keys)]))
        assignments.extend({'engine_key': key, 'fold': fold} for key in val_keys)
    pd.DataFrame(assignments).to_csv(OUT / 'cv_engine_folds.csv', index=False)
    all_predictions, fold_metrics = [], []
    all_configs = configs()
    (OUT / 'planned_configs.json').write_text(json.dumps(all_configs, indent=2))

    def run_stage(candidates, stage):
        for fold, (fit, valid) in enumerate(folds, 1):
            cache = {}
            snap = snapshots[snapshots.engine_key.isin(valid.engine_key)]
            for config in candidates:
                t0 = time.perf_counter()
                feature_key = (config['normalize'], config['rich'])
                if feature_key not in cache:
                    transformer = HistoryFeatures(*feature_key).fit(fit)
                    assert transformer.fit_engines_.isdisjoint(valid.engine_key)
                    Xfit, Xvalid = transformer.transform(fit), transformer.transform(valid)
                    # Truncation invariance on each dataset in this validation fold.
                    for dataset in DATASETS:
                        key = valid.loc[valid.dataset.eq(dataset), 'engine_key'].iloc[0]
                        prefix = valid[valid.engine_key.eq(key)].iloc[:35]
                        pd.testing.assert_frame_equal(transformer.transform(prefix), Xvalid.loc[prefix.index])
                    cache[feature_key] = (Xfit, Xvalid)
                Xfit, Xvalid = cache[feature_key]
                model = ExperimentRegressor(config, threads).fit_features(Xfit, fit)
                pred = model.predict_features(Xvalid, valid)
                record = snap.copy()
                record['predicted_RUL'] = pd.Series(pred, index=valid.index).loc[record.row_index].to_numpy()
                record['candidate'], record['fold'], record['stage'] = config['name'], fold, stage
                all_predictions.append(record)
                for protocol in ['primary', 'broad']:
                    p = record[record.protocol.eq(protocol)]
                    fold_metrics.append({'candidate': config['name'], 'stage': stage, 'fold': fold, 'protocol': protocol,
                                         **score(p.actual_RUL, p.predicted_RUL), 'seconds': time.perf_counter() - t0})
                pd.DataFrame(fold_metrics).to_csv(OUT / 'cv_fold_metrics.csv', index=False)
                log(f'{stage} fold {fold}/3 {config["name"]}: primary RMSE={fold_metrics[-2]["RMSE"]:.2f}, broad={fold_metrics[-1]["RMSE"]:.2f}; {time.perf_counter()-t0:.1f}s')
        predictions = pd.concat(all_predictions, ignore_index=True)
        predictions.to_csv(OUT / 'oof_predictions.csv', index=False)
        rows = []
        for (name, protocol), p in predictions.groupby(['candidate', 'protocol']):
            rows.append({'candidate': name, 'protocol': protocol, **score(p.actual_RUL, p.predicted_RUL)})
        table = pd.DataFrame(rows)
        table.to_csv(OUT / 'experiment_comparison.csv', index=False)
        return predictions, table

    oof, comparison = run_stage(all_configs, 'ablation')
    best_name = comparison[comparison.protocol.eq('primary')].sort_values('RMSE').iloc[0].candidate
    best = next(c for c in all_configs if c['name'] == best_name)
    log(f'Feature/target selection: {best_name}. Now run predeclared small depth/rounds grid on that configuration.')
    tuned = [dict(best, name=f'tuned_d{depth}_n{trees}', depth=depth, trees=trees, learning_rate=.03)
             for depth, trees in [(4, 600), (6, 800), (8, 600)]]
    all_configs += tuned
    (OUT / 'all_configs.json').write_text(json.dumps(all_configs, indent=2))
    oof, comparison = run_stage(tuned, 'tuning')
    best_name = comparison[comparison.protocol.eq('primary')].sort_values('RMSE').iloc[0].candidate
    best = next(c for c in all_configs if c['name'] == best_name)
    chosen_oof = oof[oof.candidate.eq(best_name) & oof.protocol.eq('primary')]
    threshold, threshold_table = choose_threshold(chosen_oof.actual_RUL, chosen_oof.predicted_RUL)
    threshold_table.to_csv(OUT / 'threshold_curve.csv', index=False)
    decision = {'selected_config': best, 'alert_threshold': threshold, 'target_recall': .90,
                'selection_metric': 'uncapped pooled primary-endpoint OOF RMSE',
                'frozen_at_utc': datetime.now(timezone.utc).isoformat()}
    (OUT / 'frozen_selection.json').write_text(json.dumps(decision, indent=2))
    log(f'FROZEN: {best_name}; maintenance threshold {threshold:.3f} chosen using OOF predictions only.')

    audit_snapshots = snapshots[snapshots.engine_key.isin(audit_keys)]
    audit_predictions, audit_metrics = [], []
    for label, config in [('reference', all_configs[0]), ('selected', best)]:
        model = ExperimentRegressor(config, threads).fit(dev)
        pred = evaluate_snapshots(model, untouched, audit_snapshots)
        pred['model'] = label
        audit_predictions.append(pred)
        for protocol in ['primary', 'broad']:
            p = pred[pred.protocol.eq(protocol)]
            for alert in sorted(set([30., threshold])):
                audit_metrics.append({'model': label, 'protocol': protocol, **score(p.actual_RUL, p.predicted_RUL, alert)})
        log(f'Untouched audit {label}: primary RMSE={score(pred[pred.protocol.eq("primary")].actual_RUL, pred[pred.protocol.eq("primary")].predicted_RUL)["RMSE"]:.2f}')
    audit_predictions = pd.concat(audit_predictions, ignore_index=True)
    audit_predictions.to_csv(OUT / 'audit_predictions.csv', index=False)
    pd.DataFrame(audit_metrics).to_csv(OUT / 'audit_metrics.csv', index=False)

    # Feature reliance from the development fit; no official test features.
    # This reports model reliance, not causal importance. Cycle removal is tested above.
    importance = []
    for key, estimator in model.models_.items():
        cols = np.array(model.columns_)[model.selectors_[key].get_support()]
        importance.extend({'model_scope': key, 'feature': feature, 'gain_importance': float(value)}
                          for feature, value in zip(cols, estimator.feature_importances_))
    pd.DataFrame(importance).to_csv(OUT / 'feature_importance.csv', index=False)

    # Paired engine bootstrap: preserve correlation among the three snapshots.
    paired = audit_predictions[audit_predictions.protocol.eq('primary')].copy()
    paired['squared_error'] = (paired.predicted_RUL - paired.actual_RUL) ** 2
    paired = paired.groupby(['engine_key', 'model']).squared_error.mean().unstack('model')
    rng = np.random.default_rng(42)
    improvements = []
    for _ in range(1000):
        sample = paired.iloc[rng.integers(0, len(paired), size=len(paired))]
        improvements.append(float(np.sqrt(sample.reference.mean()) - np.sqrt(sample.selected.mean())))
    bootstrap = {'meaning': 'reference RMSE minus selected RMSE; paired audit-engine bootstrap',
                 'replicates': 1000, 'lower_95': float(np.quantile(improvements, .025)),
                 'upper_95': float(np.quantile(improvements, .975))}
    (OUT / 'audit_bootstrap.json').write_text(json.dumps(bootstrap, indent=2))

    log('Refitting frozen selected configuration on every row of all 709 training engines.')
    final_model = ExperimentRegressor(best, threads).fit(train)
    final_model.alert_threshold_ = threshold
    final_model.training_rows_ = len(train)
    final_model.training_engines_ = train.engine_key.nunique()
    joblib.dump(final_model, OUT / 'selected_model.joblib')

    # Only now open official test labels. Previously seen test results motivate
    # this work, so this is not a brand-new blind benchmark. Never tune here.
    test_endpoints = test.groupby('engine_key').tail(1).sort_values(['dataset', 'engine_id'])
    truth_values = []
    manifest = json.loads((ROOT / 'data/source.json').read_text())
    for dataset in DATASETS:
        rel = f'data/raw/RUL_{dataset}.txt'
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == manifest['files'][rel]['sha256']
        truth = pd.read_csv(ROOT / rel, header=None)[0]
        ids = test_endpoints.loc[test_endpoints.dataset.eq(dataset), 'engine_id']
        assert ids.tolist() == list(range(1, len(truth) + 1))
        truth_values.extend(truth.tolist())
    full_pred = pd.Series(final_model.predict(test), index=test.index)
    test_predictions = test_endpoints[['dataset', 'engine_id', 'engine_key', 'cycle']].copy()
    test_predictions['actual_RUL'] = truth_values
    test_predictions['predicted_RUL'] = full_pred.loc[test_endpoints.index].to_numpy()
    test_predictions['flag_fixed30'] = test_predictions.predicted_RUL <= 30
    test_predictions['flag_tuned'] = test_predictions.predicted_RUL <= threshold
    test_predictions.to_csv(OUT / 'test_predictions.csv', index=False)
    restored = joblib.load(OUT / 'selected_model.joblib')
    np.testing.assert_allclose(restored.predict(test), full_pred.to_numpy())
    test_metrics = []
    for dataset in ['ALL'] + DATASETS:
        p = test_predictions if dataset == 'ALL' else test_predictions[test_predictions.dataset.eq(dataset)]
        for alert in sorted(set([30., threshold])):
            test_metrics.append({'model': 'selected', 'dataset': dataset, **score(p.actual_RUL, p.predicted_RUL, alert)})
    # Score prior saved predictions using exactly the same metric definitions.
    old = pd.read_csv(ROOT / 'outputs/all_engines/test_predictions.csv')
    for family, group in old.groupby('family'):
        for dataset in ['ALL'] + DATASETS:
            p = group if dataset == 'ALL' else group[group.dataset.eq(dataset)]
            test_metrics.append({'model': family, 'dataset': dataset, **score(p.actual_RUL, p.predicted_RUL)})
    pd.DataFrame(test_metrics).to_csv(OUT / 'test_metrics.csv', index=False)
    result = dict(decision, training_rows=len(train), training_engines=709, test_engines=707,
                  development_engines=len(dev_keys), audit_engines=len(audit_keys),
                  candidates=len(all_configs), threads=threads, total_seconds=time.perf_counter()-start,
                  completed_utc=datetime.now(timezone.utc).isoformat(), test_metrics=test_metrics,
                  audit_metrics=audit_metrics, audit_bootstrap=bootstrap)
    (OUT / 'results.json').write_text(json.dumps(result, indent=2))
    log(f'COMPLETE in {result["total_seconds"]:.1f}s. Final uncapped test RMSE={score(test_predictions.actual_RUL,test_predictions.predicted_RUL)["RMSE"]:.3f}')
    return result


def diagnostics(threads=6):
    """Refit only the frozen recipe to measure matching training/validation gaps."""
    config=json.loads((OUT/'frozen_selection.json').read_text())['selected_config']
    original=OUT/'selected_model.joblib'
    before=hashlib.sha256(original.read_bytes()).hexdigest()
    train,_,_=load_inputs()
    folds=pd.read_csv(OUT/'cv_engine_folds.csv')
    points=pd.read_csv(OUT/'validation_endpoints.csv')
    saved=pd.read_csv(OUT/'oof_predictions.csv')
    saved=saved[saved.candidate.eq(config['name'])]
    rows=[];checks=[]
    for fold in [1,2,3]:
        started=time.perf_counter()
        fit=train[train.engine_key.isin(folds.loc[folds.fold.ne(fold),'engine_key'])]
        valid=train[train.engine_key.isin(folds.loc[folds.fold.eq(fold),'engine_key'])]
        assert set(fit.engine_key).isdisjoint(valid.engine_key)
        model=ExperimentRegressor(config,threads).fit(fit)
        assert model.features_.fit_engines_.isdisjoint(valid.engine_key)
        predictions=[]
        for frame in [fit,valid]:
            p=points[points.engine_key.isin(frame.engine_key)].copy()
            pred=pd.Series(model.predict(frame),index=frame.index)
            p['prediction']=pred.loc[p.row_index].to_numpy()
            predictions.append(p)
        a,b=predictions
        comparison=b.merge(saved[saved.fold.eq(fold)],on=['engine_key','row_index','protocol','snapshot'],validate='one_to_one')
        assert len(comparison)==1134
        np.testing.assert_allclose(comparison.prediction,comparison.predicted_RUL,rtol=1e-6,atol=1e-4)
        for dataset_name in DATASETS:
            key=valid.loc[valid.dataset.eq(dataset_name),'engine_key'].iloc[0]
            history=valid[valid.engine_key.eq(key)].sort_values('cycle').copy()
            prefix=history.iloc[:35].copy();endpoint=prefix.index[-1]
            expected=model.features_.transform(history).loc[[endpoint]]
            changed=history.copy();changed['RUL']=-999
            sensor_columns=[c for c in changed if c.startswith('sensor_')]
            changed.loc[changed.index[35:],sensor_columns]=1e6
            pd.testing.assert_frame_equal(expected,model.features_.transform(changed).loc[[endpoint]])
            pd.testing.assert_frame_equal(expected,model.features_.transform(prefix).loc[[endpoint]])
        for protocol in ['primary','broad']:
            for subset in ['all']+DATASETS+['RUL_le150','RUL_le30']:
                def select(frame):
                    mask=frame.protocol.eq(protocol)
                    if subset in DATASETS:mask &= frame.dataset.eq(subset)
                    elif subset=='RUL_le150':mask &= frame.actual_RUL.le(150)
                    elif subset=='RUL_le30':mask &= frame.actual_RUL.le(30)
                    return frame[mask]
                aa,bb=select(a),select(b)
                train_sse=float(np.sum((aa.prediction-aa.actual_RUL)**2))
                valid_sse=float(np.sum((bb.prediction-bb.actual_RUL)**2))
                tr=np.sqrt(train_sse/len(aa));va=np.sqrt(valid_sse/len(bb))
                rows.append(dict(candidate=config['name'],fold=fold,protocol=protocol,subset=subset,
                    training_n=len(aa),validation_n=len(bb),training_SSE=train_sse,validation_SSE=valid_sse,
                    training_RMSE=tr,validation_RMSE=va,gap=va-tr))
        checks.append(dict(fold=fold,fit_engines=fit.engine_key.nunique(),validation_engines=valid.engine_key.nunique(),
            maximum_prediction_difference=float(np.max(np.abs(comparison.prediction-comparison.predicted_RUL))),
            seconds=time.perf_counter()-started))
        print(f'Checked XGBoost fold {fold}/3 in {checks[-1]["seconds"]:.1f}s',flush=True)
    gaps=pd.DataFrame(rows);assert len(gaps)==42
    gaps.to_csv(OUT/'generalization_gaps.csv',index=False)
    pooled=[]
    for (protocol,subset),g in gaps.groupby(['protocol','subset']):
        assert g.fold.nunique()==3 and g.training_n.sum()==2*g.validation_n.sum()
        tr=np.sqrt(g.training_SSE.sum()/g.training_n.sum());va=np.sqrt(g.validation_SSE.sum()/g.validation_n.sum())
        pooled.append(dict(candidate=config['name'],protocol=protocol,subset=subset,training_RMSE=tr,
            validation_RMSE=va,gap=va-tr,minimum_fold_gap=g.gap.min(),maximum_fold_gap=g.gap.max()))
    pd.DataFrame(pooled).to_csv(OUT/'generalization_summary.csv',index=False)
    assert hashlib.sha256(original.read_bytes()).hexdigest()==before
    evidence=dict(config=config,fold_checks=checks,original_model_sha256=before,
        interpretation='Descriptive matched-endpoint gaps; repeated development folds, not fresh blind evaluation',
        completed_utc=datetime.now(timezone.utc).isoformat())
    (OUT/'generalization_checks.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('PASS: original OOF predictions reproduced; disjoint engines, causal features, matched uncapped scoring and unchanged final model.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--diagnostics', action='store_true')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.diagnostics:
        diagnostics(args.threads)
    else:
        run(args.threads)
