"""Independently check saved split, endpoint, interval and first-alert evidence."""
import hashlib,json,re
from pathlib import Path
import numpy as np
import pandas as pd
from train_all import load_inputs,ROOT
OUT=ROOT/'outputs/robustness'
protocol=json.loads((OUT/'frozen_protocol.json').read_text())
assert hashlib.sha256((ROOT/'outputs/experiments/selected_model.joblib').read_bytes()).hexdigest()==protocol['model_sha256_before']
train,_,_=load_inputs()
ends=train.groupby('engine_key').cycle.max()
splits=pd.read_csv(OUT/'engine_splits.csv')
history=pd.read_csv(OUT/'heldout_history_predictions.csv')
snap=pd.read_csv(OUT/'snapshot_predictions.csv')
metrics=pd.read_csv(OUT/'split_metrics.csv')
for seed,g in splits.groupby('seed'):
    assert len(g)==709 and g.engine_key.nunique()==709
    valid=set(g.loc[g.role.eq('validation'),'engine_key'])
    assert len(valid)==142
    assert set(history.loc[history.seed.eq(seed),'engine_key'])==valid
    assert set(snap.loc[snap.seed.eq(seed),'engine_key'])==valid
assert np.allclose(history.actual_RUL,history.engine_key.map(ends)-history.cycle)
assert np.allclose(snap.actual_RUL,snap.engine_key.map(ends)-snap.cycle)
assert (snap.cycle>=20).all() and (snap.actual_RUL>0).all()
for row in metrics.itertuples():
    g=snap[(snap.seed==row.seed)&(snap.model==row.model)&(snap.scenario==row.scenario)]
    assert len(g)==426 and g.groupby('engine_key').size().eq(3).all()
    assert np.isclose(np.sqrt(np.mean((g.predicted_RUL-g.actual_RUL)**2)),row.RMSE)
    flag=g.predicted_RUL<=row.threshold; event=g.actual_RUL<=30
    assert (flag&event).sum()==row.TP and (flag&~event).sum()==row.FP
    assert (~flag&event).sum()==row.FN and (~flag&~event).sum()==row.TN
    if row.scenario!='broad':assert g.actual_RUL.le(int(row.scenario.split('_')[-1])).all()
for _,g in snap.groupby(['seed','scenario','engine_key','snapshot']):
    assert len(g)==2 and g.cycle.nunique()==1
by_history={(seed,model,key):g.sort_values('cycle') for (seed,model,key),g in history.groupby(['seed','model','engine_key'])}
events=pd.read_csv(OUT/'engine_alert_events.csv')
assert len(events)==5*3*142
for row in events.itertuples():
    g=by_history[row.seed,row.model,row.engine_key]
    g=g[(g.cycle>=20)&(g.actual_RUL>0)]
    flags=(g.predicted_RUL.to_numpy()<=row.threshold)
    hits=np.flatnonzero(flags)
    assert len(g)==row.monitored_cycles
    if len(hits):
        first=g.iloc[hits[0]]
        assert first.cycle==row.first_alert_cycle and first.actual_RUL==row.first_alert_lead
    else:assert pd.isna(row.first_alert_cycle)
    early=g.actual_RUL.to_numpy()>30
    starts=flags&~np.r_[False,flags[:-1]]
    assert int((starts&early).sum())==row.early_alert_episodes
    assert bool((flags&~early).any())==row.detected_within_last30
for row in pd.read_csv(OUT/'test_alert_intervals.csv').itertuples():
    n=row.denominator; p=row.numerator/n; z=1.959963984540054
    centre=(p+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    assert np.allclose([row.estimate,row.lower95,row.upper95],[p,centre-half,centre+half])
nb=json.loads((ROOT/'explore.ipynb').read_text())
code=[c for c in nb['cells'] if c['cell_type']=='code']
assert all(c['execution_count'] is not None for c in code)
assert not any(o['output_type']=='error' for c in code for o in c['outputs'])
html=(ROOT/'presentation.html').read_text()
ids=set(re.findall(r'id="([^"]+)"',html))
for link in re.findall(r'href="([^"]+)"',html):
    if link.startswith('#'):assert link[1:] in ids,link
    elif not re.match(r'\w+://',link):assert (ROOT/link).exists(),link
for head in re.findall(r'<thead>(.*?)</thead>',html,re.S):assert head.count('<tr')==1
assert html.count('src="data:image/png;base64,')==8
print('PASS: fixed model hash, disjoint engine roles, 50 endpoint metrics, 2,130 first-alert events, Wilson intervals, executed notebook and HTML links/tables/charts.')
