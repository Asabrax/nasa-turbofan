"""Frozen-recipe robustness, alert uncertainty and held-out trajectory replay.

No hyperparameter/threshold selection. Official final model remains unchanged.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,time
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.model_selection import train_test_split
from experiment_models import ExperimentRegressor
from train_all import ROOT,load_inputs
from run_experiments import score

OUT=ROOT/'outputs/robustness'
SEEDS=[7,19,42,73,101]
HORIZONS=[100,150,200,250]


def scenario_points(train):
    records=[]
    for key,g in train.groupby('engine_key',sort=True):
        end=int(g.cycle.max())
        for horizon in HORIZONS+[None]:
            label='broad' if horizon is None else f'RUL_1_{horizon}'
            rng=np.random.default_rng(int(hashlib.sha256(f'robustness:42:{key}:{label}'.encode()).hexdigest()[:8],16))
            choices=np.arange(20 if horizon is None else max(20,end-horizon),end)
            for i,cycle in enumerate(rng.choice(choices,size=3,replace=False)):
                row=g[g.cycle.eq(cycle)].iloc[0]
                records.append({'engine_key':key,'dataset':row.dataset,'scenario':label,'snapshot':i,
                                'row_index':int(row.name),'cycle':int(cycle),'actual_RUL':float(row.RUL)})
    return pd.DataFrame(records)


def replay(frame, threshold):
    """Passive flags from cycle 20 to T-1; no intervention/censoring simulation."""
    records=[]
    for key,g in frame.groupby('engine_key',sort=True):
        g=g.sort_values('cycle')
        g=g[(g.cycle>=20)&(g.actual_RUL>0)]
        flag=g.predicted_RUL.le(threshold)
        early=g.actual_RUL.gt(30)
        first=g.loc[flag].iloc[0] if flag.any() else None
        lead=np.nan if first is None else float(first.actual_RUL)
        previous=flag.shift(fill_value=False)
        early_episodes=int((flag & ~previous & early).sum())
        records.append({'engine_key':key,'dataset':g.dataset.iloc[0],
                        'threshold':threshold,'monitored_cycles':len(g),
                        'first_alert_cycle':np.nan if first is None else int(first.cycle),
                        'first_alert_lead':lead,
                        'first_alert_status':'none' if first is None else ('early' if lead>30 else 'within_30'),
                        'any_alert_before_failure':bool(flag.any()),
                        'detected_within_last30':bool((flag & ~early).any()),
                        'early_flag_cycles':int((flag & early).sum()),
                        'early_available_cycles':int(early.sum()),'early_alert_episodes':early_episodes,
                        'first_alert_under10':bool(first is not None and lead<10)})
    return pd.DataFrame(records)


def event_summary(events):
    lead=events.first_alert_lead.dropna()
    return {'engines':len(events),'median_first_alert_lead':float(lead.median()) if len(lead) else None,
            'lead_q25':float(lead.quantile(.25)) if len(lead) else None,
            'lead_q75':float(lead.quantile(.75)) if len(lead) else None,
            'premature_first_alert_fraction':float(events.first_alert_status.eq('early').mean()),
            'first_alert_31_60_fraction':float((events.first_alert_lead.gt(30)&events.first_alert_lead.le(60)).mean()),
            'first_alert_over60_fraction':float(events.first_alert_lead.gt(60).mean()),
            'timely_first_alert_fraction':float(events.first_alert_status.eq('within_30').mean()),
            'no_alert_fraction':float(events.first_alert_status.eq('none').mean()),
            'within_last30_detection':float(events.detected_within_last30.mean()),
            'first_alert_under10_fraction':float(events.first_alert_under10.mean()),
            'mean_early_alert_episodes':float(events.early_alert_episodes.mean()),
            'mean_engine_early_cycle_fraction':float((events.early_flag_cycles/events.early_available_cycles).mean())}


def wilson(k,n):
    if n==0:return (None,None)
    interval=binomtest(int(k),int(n)).proportion_ci(confidence_level=.95,method='wilson')
    return float(interval.low),float(interval.high)


def intervals(threshold):
    # One endpoint per engine: conditional binomial Wilson intervals.
    old=pd.read_csv(ROOT/'outputs/all_engines/test_predictions.csv')
    final=pd.read_csv(ROOT/'outputs/experiments/test_predictions.csv')
    rows=[]
    for name,frame,t in [('Original XGBoost',old[old.family.eq('XGBoost')],30.),
                         ('Final, fixed alert',final,30.),('Final, calibrated alert',final,threshold)]:
        for dataset in ['ALL','FD001','FD002','FD003','FD004']:
            g=frame if dataset=='ALL' else frame[frame.dataset.eq(dataset)]
            m=score(g.actual_RUL,g.predicted_RUL,t)
            for metric,k,n in [('precision',m['TP'],m['TP']+m['FP']),('recall',m['TP'],m['TP']+m['FN'])]:
                lo,hi=wilson(k,n)
                rows.append({'population':'official_test','model':name,'dataset':dataset,'threshold':t,
                             'metric':metric,'estimate':k/n if n else None,'lower95':lo,'upper95':hi,
                             'numerator':k,'denominator':n,'method':'Wilson conditional binomial'})
    pd.DataFrame(rows).to_csv(OUT/'test_alert_intervals.csv',index=False)
    # Three correlated snapshots per audit engine: bootstrap whole engines.
    audit=pd.read_csv(ROOT/'outputs/experiments/audit_predictions.csv')
    records=[]
    for label,g,t in [('Original recipe',audit[audit.model.eq('reference') & audit.protocol.eq('primary')],30.),
                      ('Final, fixed alert',audit[audit.model.eq('selected') & audit.protocol.eq('primary')],30.),
                      ('Final, calibrated alert',audit[audit.model.eq('selected') & audit.protocol.eq('primary')],threshold)]:
        g=g.copy();truth=g.actual_RUL<=30;flag=g.predicted_RUL<=t
        g['TP']=(truth&flag).astype(int);g['FP']=(~truth&flag).astype(int);g['FN']=(truth&~flag).astype(int)
        counts=g.groupby('engine_key')[['TP','FP','FN']].sum().to_numpy()
        rng=np.random.default_rng(42);rep=[]
        for _ in range(2000):
            tp,fp,fn=counts[rng.integers(0,len(counts),len(counts))].sum(axis=0)
            rep.append([tp/(tp+fp) if tp+fp else np.nan,tp/(tp+fn) if tp+fn else np.nan])
        rep=np.array(rep);m=score(g.actual_RUL,g.predicted_RUL,t)
        for i,metric in enumerate(['precision','recall']):
            lo,hi=np.nanquantile(rep[:,i],[.025,.975])
            records.append({'model':label,'metric':metric,'threshold':t,'estimate':m[metric],
                            'lower95':float(lo),'upper95':float(hi),'engines':len(counts),
                            'method':'2000 whole-engine percentile bootstrap draws'})
    pd.DataFrame(records).to_csv(OUT/'audit_alert_intervals.csv',index=False)


def run():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'training.log').write_text('')
    def log(s):
        line=f'{datetime.now(timezone.utc).isoformat(timespec="seconds")} {s}'
        print(line,flush=True)
        with (OUT/'training.log').open('a') as f:f.write(line+'\n')
    original=ROOT/'outputs/experiments/selected_model.joblib'
    digest=hashlib.sha256(original.read_bytes()).hexdigest()
    frozen=json.loads((ROOT/'outputs/experiments/frozen_selection.json').read_text())
    reference=json.loads((ROOT/'outputs/experiments/planned_configs.json').read_text())[0]
    threshold=frozen['alert_threshold']
    plan={'seeds':SEEDS,'planning_windows':HORIZONS,'threshold':threshold,
          'reference_config':reference,'selected_config':frozen['selected_config'],
          'monitor_from_cycle':20,'last_monitored_RUL':1,'model_sha256_before':digest,
          'policy':'Sensitivity analysis of fixed recipes; no reselection or threshold recalibration. Repeated splits overlap and are not independent external replications.'}
    (OUT/'frozen_protocol.json').write_text(json.dumps(plan,indent=2))
    train,_,_=load_inputs()
    engines=train[['engine_key','dataset']].drop_duplicates().sort_values('engine_key')
    points=scenario_points(train)
    points.to_csv(OUT/'scenario_endpoints.csv',index=False)
    metric_rows=[];snapshot_frames=[];event_frames=[];history_frames=[];split_rows=[]
    log('Start fixed-recipe robustness: five engine splits, four planning windows plus broad coverage.')
    for seed in SEEDS:
        fit_keys,valid_keys=train_test_split(engines.engine_key,test_size=.2,random_state=seed,stratify=engines.dataset)
        fit=train[train.engine_key.isin(fit_keys)];valid=train[train.engine_key.isin(valid_keys)]
        assert set(fit.engine_key).isdisjoint(valid.engine_key)
        split_rows.extend({'seed':seed,'engine_key':key,'role':'validation' if key in set(valid_keys) else 'fitting'} for key in engines.engine_key)
        snap=points[points.engine_key.isin(valid_keys)]
        for label,config in [('Original recipe',reference),('Final recipe',frozen['selected_config'])]:
            log(f'Seed {seed}: fitting {label} on {len(fit_keys)} engines; validating {len(valid_keys)} engines.')
            model=ExperimentRegressor(config,threads=6).fit(fit)
            prediction=pd.Series(model.predict(valid),index=valid.index)
            # Reconfirm prefix causality for every validation dataset.
            for dataset in sorted(valid.dataset.unique()):
                key=valid.loc[valid.dataset.eq(dataset),'engine_key'].iloc[0]
                prefix=valid[valid.engine_key.eq(key)].iloc[:40]
                np.testing.assert_allclose(model.predict(prefix),prediction.loc[prefix.index])
            records=snap.copy();records['predicted_RUL']=prediction.loc[records.row_index].to_numpy()
            records['seed']=seed;records['model']=label;snapshot_frames.append(records)
            for scenario,g in records.groupby('scenario'):
                metric_rows.append({'seed':seed,'model':label,'scenario':scenario,**score(g.actual_RUL,g.predicted_RUL,threshold if label=='Final recipe' else 30.)})
            history=valid[['dataset','engine_key','engine_id','cycle','RUL']].rename(columns={'RUL':'actual_RUL'}).copy()
            history['predicted_RUL']=prediction.to_numpy();history['seed']=seed;history['model']=label
            history_frames.append(history)
            for t in ([30.] if label=='Original recipe' else [30.,threshold]):
                events=replay(history,t);events['seed']=seed;events['model']=label;event_frames.append(events)
            log(f'Seed {seed}, {label}: RUL 1–200 RMSE={next(v["RMSE"] for v in metric_rows if v["seed"]==seed and v["model"]==label and v["scenario"]=="RUL_1_200"):.2f}')
        pd.DataFrame(metric_rows).to_csv(OUT/'split_metrics.csv',index=False)
    pd.DataFrame(split_rows).to_csv(OUT/'engine_splits.csv',index=False)
    pd.concat(snapshot_frames,ignore_index=True).to_csv(OUT/'snapshot_predictions.csv',index=False)
    pd.concat(history_frames,ignore_index=True).to_csv(OUT/'heldout_history_predictions.csv',index=False)
    events=pd.concat(event_frames,ignore_index=True);events.to_csv(OUT/'engine_alert_events.csv',index=False)
    event_rows=[]
    for (seed,label,t),g in events.groupby(['seed','model','threshold']):
        event_rows.append({'seed':int(seed),'model':label,'threshold':float(t),**event_summary(g)})
    pd.DataFrame(event_rows).to_csv(OUT/'trajectory_summary.csv',index=False)
    intervals(threshold)
    assert hashlib.sha256(original.read_bytes()).hexdigest()==digest
    result=dict(plan,total_seconds=time.perf_counter()-start,completed_utc=datetime.now(timezone.utc).isoformat(),
                training_engines=709,fit_engines_per_split=567,heldout_engines_per_split=142,
                final_model_unchanged=True,metric_rows=metric_rows,trajectory_rows=event_rows)
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    log(f'COMPLETE in {result["total_seconds"]:.1f}s; original final model SHA256 unchanged.')


if __name__=='__main__':run()
