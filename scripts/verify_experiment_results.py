"""Independently verify experiment artifacts without fitting any models."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from train_all import ROOT, DATASETS

out=ROOT/'outputs/experiments'
r=json.loads((out/'results.json').read_text())
split=pd.read_csv(out/'engine_split.csv')
folds=pd.read_csv(out/'cv_engine_folds.csv')
oof=pd.read_csv(out/'oof_predictions.csv')
audit=pd.read_csv(out/'audit_predictions.csv')
comparison=pd.read_csv(out/'experiment_comparison.csv')
assert len(split)==709 and split.engine_key.nunique()==709
assert set(folds.engine_key)==set(split[split.role.eq('development')].engine_key)
assert set(audit.engine_key)==set(split[split.role.eq('audit')].engine_key)
assert set(folds.engine_key).isdisjoint(audit.engine_key)
assert oof.groupby(['candidate','protocol']).size().eq(567*3).all()
assert oof.groupby(['candidate','protocol','engine_key']).size().eq(3).all()
assert audit.groupby(['model','protocol','engine_key']).size().eq(3).all()
for (candidate,protocol),p in oof.groupby(['candidate','protocol']):
    row=comparison[(comparison.candidate==candidate)&(comparison.protocol==protocol)].iloc[0]
    assert np.isclose(np.sqrt(np.mean((p.predicted_RUL-p.actual_RUL)**2)),row.RMSE)
selected=comparison[comparison.protocol.eq('primary')].sort_values('RMSE').iloc[0].candidate
assert selected==r['selected_config']['name']
p=oof[oof.candidate.eq(selected)&oof.protocol.eq('primary')]
positive=p[p.actual_RUL<=30]
expected_threshold=np.sort(positive.predicted_RUL)[math.ceil(.9*len(positive))-1]
assert np.isclose(expected_threshold,r['alert_threshold'])

pred=pd.read_csv(out/'test_predictions.csv')
assert len(pred)==707 and not pred.duplicated(['dataset','engine_id']).any()
for dataset in DATASETS:
    truth=pd.read_csv(ROOT/f'data/raw/RUL_{dataset}.txt',header=None)[0]
    np.testing.assert_array_equal(pred[pred.dataset.eq(dataset)].sort_values('engine_id').actual_RUL,truth)
for csv,rows in [('test_metrics.csv',pred),('audit_metrics.csv',audit)]:
    table=pd.read_csv(out/csv)
    for _,metric in table.iterrows():
        if csv=='test_metrics.csv':
            if metric.model!='selected':continue
            p=rows if metric.dataset=='ALL' else rows[rows.dataset.eq(metric.dataset)]
        else:
            p=rows[rows.model.eq(metric.model)&rows.protocol.eq(metric.protocol)]
        errors=(p.predicted_RUL-p.actual_RUL).to_numpy()
        assert np.isclose(np.sqrt(np.mean(errors**2)),metric.RMSE)
        # Deliberately use scalar math and explicit branches instead of scorer code.
        independent_score=sum(math.exp(-e/13)-1 if e<0 else math.exp(e/10)-1 for e in errors)
        assert np.isclose(independent_score,metric.NASA_score_sum)
        true=p.actual_RUL.le(30);flag=p.predicted_RUL.le(metric.threshold)
        assert (true&flag).sum()==metric.TP
        assert (true&~flag).sum()==metric.FN
        assert (~true&flag).sum()==metric.FP
        assert (~true&~flag).sum()==metric.TN
        assert np.isclose(np.abs(errors).mean(),metric.MAE)
print('PASS: disjoint development/audit engines, every OOF endpoint, selected winner, independent recall threshold, all official labels, RMSE/MAE/NASA score and confusion counts.')
