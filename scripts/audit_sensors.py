"""Inspect fitted sensor retention and validate the conventional 14-sensor subset.

Uses development engines only for the ablation; does not re-open official labels.
"""
import sys,json,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from train_all import ROOT,DATASETS,SENSORS,load_inputs
from experiment_models import HistoryFeatures,ExperimentRegressor,regime_keys
from run_experiments import score

def sensor_visual_inputs():
    """Descriptive sensor matrices from development engines only; no refitting."""
    train,_,_=load_inputs()
    folds=pd.read_csv(ROOT/'outputs/experiments/cv_engine_folds.csv')
    dev=train[train.engine_key.isin(folds.engine_key)].copy()
    assert dev.engine_key.nunique()==567
    normalizer=HistoryFeatures(normalize=True).fit(dev)
    keys=regime_keys(dev)
    values=(dev[SENSORS].to_numpy()-normalizer.means_.loc[keys].to_numpy())/normalizer.scales_.loc[keys].to_numpy()
    normalized=pd.DataFrame(values,columns=SENSORS,index=dev.index)
    # Explicitly mask constant channels: their correlation is undefined, not zero.
    flat=normalized.var(ddof=0).le(1e-8)
    corr=normalized.corr(method='pearson')
    corr.loc[flat,:]=np.nan;corr.loc[:,flat]=np.nan
    records=[]
    for dataset in DATASETS:
        frame=dev[dev.dataset.eq(dataset)]
        grouped=frame.groupby(regime_keys(frame))[SENSORS]
        distinct=grouped.nunique().max();std=grouped.std(ddof=0).max()
        for sensor in SENSORS:
            records.append(dict(dataset=dataset,sensor=sensor,engines=frame.engine_key.nunique(),rows=len(frame),
                max_distinct_within_regime=int(distinct[sensor]),max_within_regime_std=std[sensor],
                varies_within_regime=bool(distinct[sensor]>1)))
    folder=ROOT/'outputs/presentation'
    pd.DataFrame(records).to_csv(folder/'sensor_variation_development.csv',index=False)
    corr.to_csv(folder/'sensor_correlation.csv',index_label='sensor')
    metadata=dict(engines=dev.engine_key.nunique(),rows=len(dev),datasets=DATASETS,
        removed_sensors=flat[flat].index.tolist(),
        correlation='Pearson, dataset/regime-normalized instantaneous sensors; each engine-cycle has equal weight',
        normalization='Descriptive fit on development engines only; not used to alter any model or CV pipeline',
        variation='At least two distinct values within one dataset/operating-regime group',
        limitations='Repeated cycles are not independent samples; correlation is not conditional importance or proof of redundancy')
    (folder/'sensor_visual_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    assert metadata['removed_sensors']==['sensor_1','sensor_5','sensor_18','sensor_19']
    assert corr.loc[~flat,~flat].notna().all().all()
    np.testing.assert_allclose(corr.to_numpy(),corr.to_numpy().T,equal_nan=True)
    np.testing.assert_allclose(np.diag(corr.loc[~flat,~flat]),1)
    assert corr.stack().abs().max()<=1+1e-10
    print(f'Saved sensor matrices from {len(dev):,} cycles / 567 development engines; four constant channels masked.')


if '--visuals-only' in sys.argv:
    sensor_visual_inputs()
    sys.exit(0)

out=ROOT/'outputs/presentation'
out.mkdir(exist_ok=True,parents=True)
train,_,_=load_inputs()
model=joblib.load(ROOT/'outputs/experiments/selected_model.joblib')
selector=model.selectors_['ALL']
retained=np.array(model.columns_)[selector.get_support()]
gain=dict(zip(retained,model.models_['ALL'].feature_importances_))
records=[]
for dataset in DATASETS:
    frame=train[train.dataset.eq(dataset)]
    within=frame[SENSORS].groupby(regime_keys(frame)).std(ddof=0).max()
    for sensor in SENSORS:
        family=[c for c in retained if c==sensor or c.startswith(sensor+'_')]
        records.append({'dataset':dataset,'sensor':sensor,'raw_unique':frame[sensor].nunique(),
                        'raw_std':frame[sensor].std(ddof=0),'max_within_regime_std':within[sensor],
                        'retained_features_pooled':len(family),'pooled_gain':sum(float(gain[c]) for c in family)})
audit=pd.DataFrame(records)
audit.to_csv(out/'sensor_audit.csv',index=False)
summary=audit.groupby('sensor',sort=False).agg(retained_features=('retained_features_pooled','first'),gain=('pooled_gain','first'))
summary.to_csv(out/'sensor_retention.csv')
print(summary.to_string(),flush=True)

# Compare against the saved OOF predictions on exactly the same development folds.
folds=pd.read_csv(ROOT/'outputs/experiments/cv_engine_folds.csv')
endpoints=pd.read_csv(ROOT/'outputs/experiments/validation_endpoints.csv')
config=dict(model.config)
exclude=['sensor_1','sensor_5','sensor_6','sensor_10','sensor_16','sensor_18','sensor_19']
config['name']='14_sensor_ablation'
rows=[]
start=time.perf_counter()
for fold in sorted(folds.fold.unique()):
    fit_keys=folds.loc[folds.fold.ne(fold),'engine_key']
    valid_keys=folds.loc[folds.fold.eq(fold),'engine_key']
    fit=train[train.engine_key.isin(fit_keys)]
    valid=train[train.engine_key.isin(valid_keys)]
    assert set(fit.engine_key).isdisjoint(valid.engine_key)
    features=HistoryFeatures(True,True).fit(fit)
    Xfit,Xvalid=features.transform(fit),features.transform(valid)
    columns=[c for c in Xfit if not any(c==s or c.startswith(s+'_') for s in exclude)]
    candidate=ExperimentRegressor(config,threads=6).fit_features(Xfit[columns],fit)
    pred=pd.Series(candidate.predict_features(Xvalid,valid),index=valid.index)
    snap=endpoints[endpoints.engine_key.isin(valid_keys)].copy()
    snap['predicted_RUL']=pred.loc[snap.row_index].to_numpy()
    snap['fold']=fold
    rows.append(snap)
    p=snap[snap.protocol.eq('primary')]
    print(f'Fold {fold}: reduced-sensor primary RMSE={score(p.actual_RUL,p.predicted_RUL)["RMSE"]:.3f}',flush=True)
reduced=pd.concat(rows,ignore_index=True)
reduced.to_csv(out/'sensor_ablation_oof.csv',index=False)
original=pd.read_csv(ROOT/'outputs/experiments/oof_predictions.csv')
original=original[original.candidate.eq(model.config['name'])]
metrics=[]
for name,frame in [('Current model: 17 retained sensors',original),('Reduced model: 14 sensors',reduced)]:
    for protocol in ['primary','broad']:
        p=frame[frame.protocol.eq(protocol)]
        assert len(p)==567*3
        metrics.append({'model':name,'protocol':protocol,**score(p.actual_RUL,p.predicted_RUL)})
table=pd.DataFrame(metrics)
table.to_csv(out/'sensor_ablation.csv',index=False)
result={'excluded_sensors_candidate':exclude,'current_removed_sensors':[s for s in SENSORS if summary.loc[s,'retained_features']==0],
        'current_sensor_count':int(summary.retained_features.gt(0).sum()),'candidate_sensor_count':14,
        'current_feature_count':len(retained),'candidate_feature_count_before_variance_filter':len(columns),
        'seconds':time.perf_counter()-start,'metrics':metrics,
        'policy':'Diagnostic development-only ablation. Existing final model and official test results are unchanged.'}
(out/'sensor_results.json').write_text(json.dumps(result,indent=2))
print(table[['model','protocol','RMSE']].to_string(index=False))
