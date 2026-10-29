"""Frozen full-data TCN training and follow-up official-test evaluation."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
import argparse,gc,hashlib,json,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import run_sequences as seq
from run_experiments import score
from run_tcn_followup import threshold,write_json
from train_all import ROOT,DATASETS,load_inputs

tf=seq.tf
OUT=ROOT/'outputs/tcn_full'
SEEDS=[42,73,101]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def log(message):
    line=f'{datetime.now(timezone.utc).isoformat(timespec="seconds")} {message}'
    print(line,flush=True)
    with (OUT/'training.log').open('a') as f:f.write(line+'\n')

def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    fits=pd.read_csv(ROOT/'outputs/tcn_followup/fits.csv')
    fits=fits[fits.window.eq(50)&fits.budget.eq(60)]
    points=pd.read_csv(ROOT/'outputs/tcn_followup/oof_predictions.csv')
    points=points[points.window.eq(50)&points.budget.eq(60)&points.protocol.eq('primary')]
    keys=['engine_key','row_index','dataset','engine_id','cycle','actual_RUL','protocol','snapshot','fold']
    reference=points[points.seed.eq(42)].sort_values(['engine_key','snapshot']).reset_index(drop=True)
    predictions=[];epochs={};thresholds={};epoch_sources={}
    for seed in SEEDS:
        p=points[points.seed.eq(seed)].sort_values(['engine_key','snapshot']).reset_index(drop=True)
        pd.testing.assert_frame_equal(reference[keys],p[keys])
        assert len(p)==1701 and p.engine_key.nunique()==567
        selected=fits[fits.seed.eq(seed)].sort_values('fold').selected_epoch.astype(int).tolist()
        assert len(selected)==3
        epochs[str(seed)]=int(np.median(selected));epoch_sources[str(seed)]=selected
        thresholds[f'TCN_seed{seed}']=threshold(p.actual_RUL,p.predicted_RUL)
        predictions.append(p.predicted_RUL.to_numpy())
    ensemble=np.mean(predictions,axis=0)
    thresholds['TCN_ensemble']=threshold(reference.actual_RUL,ensemble)
    decision=json.loads((ROOT/'outputs/experiments/frozen_selection.json').read_text())
    thresholds['XGBoost']=decision['alert_threshold']
    source=ROOT/'outputs/experiments/selected_model.joblib'
    prior=json.loads((ROOT/'outputs/tcn_followup/protocol.json').read_text())
    assert digest(source)==prior['original_model_sha256']
    plan=dict(window=50,seeds=SEEDS,epochs=epochs,epoch_sources=epoch_sources,
        epoch_rule='Median of three selected inner-CV epochs for each seed; no test-based stopping',
        thresholds=thresholds,threshold_rule='Smallest threshold achieving >=90% recall on pooled development OOF primary endpoints',
        ensemble_rule='Unweighted arithmetic mean of all three seeds; fixed before test scoring',
        architecture='Existing 32-channel residual TCN; dropout 0.1; 26241 parameters',
        target_cap=150,batch_size=512,learning_rate=.001,workers=2,threads_per_worker=3,
        training_engines=709,test_engines=707,
        original_xgboost_sha256=digest(source),
        source_fits_sha256=digest(ROOT/'outputs/tcn_followup/fits.csv'),
        source_oof_sha256=digest(ROOT/'outputs/tcn_followup/oof_predictions.csv'),
        source_manifest_sha256=digest(ROOT/'data/source.json'),
        benchmark_status='Previously examined official test; follow-up comparison, not fresh blind validation')
    if (OUT/'protocol.json').exists():
        existing=json.loads((OUT/'protocol.json').read_text())
        assert {k:v for k,v in existing.items() if k!='frozen_at_utc'}==plan
    else:
        write_json(OUT/'protocol.json',dict(plan,frozen_at_utc=datetime.now(timezone.utc).isoformat()))
    train,_,audit=load_inputs();audit.to_csv(OUT/'data_audit.csv',index=False)
    processor=seq.SequenceInputs().fit(train)
    assert len(processor.sensors)==17 and len(processor.fit_engines)==709
    if not (OUT/'preprocessor.joblib').exists():joblib.dump(processor,OUT/'preprocessor.joblib')
    else:
        previous=joblib.load(OUT/'preprocessor.joblib')
        np.testing.assert_array_equal(processor.transform(train.iloc[:100]),previous.transform(train.iloc[:100]))
    write_json(OUT/'input_features.json',dict(sensors=processor.sensors,
        channels=len(processor.sensors)+9,training_rows=len(train),training_engines=709,
        source_preprocessor_sha256=digest(OUT/'preprocessor.joblib')))
    return plan

def test_windows(processor,test):
    endpoints=test.groupby('engine_key').tail(1).sort_values(['dataset','engine_id'])
    X,index=processor.windows(test,endpoints.index)
    order=pd.Series(np.arange(len(index)),index=index).loc[endpoints.index].to_numpy()
    return endpoints,X,order

def worker(seed):
    folder=OUT/'_runs'/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'done.json').exists():return
    plan=json.loads((OUT/'protocol.json').read_text());epochs=plan['epochs'][str(seed)]
    seq.WINDOW=50
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    train,test,_=load_inputs();processor=joblib.load(OUT/'preprocessor.joblib')
    assert processor.fit_engines==set(train.engine_key) and len(train)==160359
    start=time.perf_counter()
    X,index=processor.windows(train)
    y=np.minimum(train.loc[index,'RUL'].to_numpy(),150).astype('float32')/150
    model=seq.network('TCN',X.shape[-1],seed);assert model.count_params()==26241
    ds=seq.dataset(X,y,seed);curves=[]
    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs=None):
            curves.append(dict(seed=seed,epoch=epoch+1,training_capped_RMSE=float(np.sqrt(logs['loss'])*150)))
            pd.DataFrame(curves).to_csv(folder/'learning_curves.csv',index=False)
            if (epoch+1)%5==0 or epoch+1==epochs:log(f'TCN seed {seed}: epoch {epoch+1}/{epochs}, training capped RMSE={curves[-1]["training_capped_RMSE"]:.2f}.')
    log(f'TCN seed {seed}: fitting all 160359 cycles / 709 engines for {epochs} frozen epochs.')
    model.fit(ds,epochs=epochs,verbose=0,shuffle=False,callbacks=[Progress()])
    model.save(folder/'model.keras');del X,ds;gc.collect()
    endpoints,xt,order=test_windows(processor,test)
    prediction=seq.predict(model,xt)[order]
    restored=tf.keras.models.load_model(folder/'model.keras')
    np.testing.assert_allclose(seq.predict(restored,xt)[order],prediction,rtol=1e-5,atol=.001)
    p=endpoints[['dataset','engine_id','engine_key','cycle']].copy();p['predicted_RUL']=prediction
    p.to_csv(folder/'test_predictions_unlabeled.csv',index=False)
    for dataset in DATASETS:
        key=test.loc[test.dataset.eq(dataset),'engine_key'].iloc[0]
        history=test[test.engine_key.eq(key)].sort_values('cycle').copy();row=history.index[22]
        expected=processor.windows(history,[row])[0]
        changed=history.copy();changed['RUL']=-999
        changed.loc[changed.index[23:],seq.SENSORS]=1e6
        np.testing.assert_array_equal(expected,processor.windows(changed,[row])[0])
        np.testing.assert_array_equal(expected,processor.windows(history.iloc[:23],[row])[0])
    write_json(folder/'done.json',dict(seed=seed,epochs=epochs,training_rows=len(train),training_engines=709,
        parameters=model.count_params(),model_sha256=digest(folder/'model.keras'),
        wall_seconds=time.perf_counter()-start,completed_utc=datetime.now(timezone.utc).isoformat()))
    log(f'TCN seed {seed}: complete; saved model and unlabeled test predictions.')

def load_truth(endpoints):
    manifest=json.loads((ROOT/'data/source.json').read_text());values=[]
    for dataset in DATASETS:
        rel=f'data/raw/RUL_{dataset}.txt'
        assert digest(ROOT/rel)==manifest['files'][rel]['sha256']
        truth=pd.read_csv(ROOT/rel,sep=r'\s+',header=None)[0]
        ids=endpoints.loc[endpoints.dataset.eq(dataset),'engine_id']
        assert ids.tolist()==list(range(1,len(truth)+1))
        values.extend(truth.tolist())
    return np.asarray(values)

def evaluate():
    plan=json.loads((OUT/'protocol.json').read_text())
    assert all((OUT/'_runs'/f'seed{s}'/'done.json').exists() for s in SEEDS)
    # Only after all frozen fits are saved, open official test labels for scoring.
    train,test,_=load_inputs()
    endpoints=test.groupby('engine_key').tail(1).sort_values(['dataset','engine_id'])
    table=endpoints[['dataset','engine_id','engine_key','cycle']].copy().reset_index(drop=True)
    table['actual_RUL']=load_truth(endpoints)
    original=pd.read_csv(ROOT/'outputs/experiments/test_predictions.csv')
    cols=['dataset','engine_id','engine_key','cycle','actual_RUL']
    pd.testing.assert_frame_equal(table[cols],original[cols],check_dtype=False)
    table['XGBoost']=original.predicted_RUL
    for seed in SEEDS:
        p=pd.read_csv(OUT/'_runs'/f'seed{seed}'/'test_predictions_unlabeled.csv')
        pd.testing.assert_frame_equal(table[cols[:-1]],p[cols[:-1]],check_dtype=False)
        table[f'TCN_seed{seed}']=p.predicted_RUL
    table['TCN_ensemble']=table[[f'TCN_seed{s}' for s in SEEDS]].mean(axis=1)
    table.to_csv(OUT/'test_predictions.csv',index=False)
    rows=[]
    for name in ['XGBoost']+[f'TCN_seed{s}' for s in SEEDS]+['TCN_ensemble']:
        for dataset in ['ALL']+DATASETS:
            p=table if dataset=='ALL' else table[table.dataset.eq(dataset)]
            for rule,t in [('fixed_30',30.),('development_calibrated',plan['thresholds'][name])]:
                result=score(p.actual_RUL,p[name],t)
                near=p[p.actual_RUL.le(30)]
                rows.append(dict(model=name,dataset=dataset,rule=rule,**result,
                    near_failure_RMSE=float(np.sqrt(np.mean((near[name]-near.actual_RUL)**2)))))
    pd.DataFrame(rows).to_csv(OUT/'test_metrics.csv',index=False)
    paired=[];rng=np.random.default_rng(42)
    # Paired engine resampling within dataset preserves the official subset mix.
    ix=np.concatenate([rng.choice(g.index.to_numpy(),size=(2000,len(g)),replace=True) for _,g in table.groupby('dataset')],axis=1)
    truth=table.actual_RUL.to_numpy();positive=truth<=30
    def measures(prediction,t):
        error=(prediction-truth)**2;alert=prediction<=t
        tp=(alert&positive).astype(int);fp=(alert&~positive).astype(int)
        fn=(~alert&positive).astype(int);tn=(~alert&~positive).astype(int)
        point=np.array([np.sqrt(error.mean()),tp.sum()/(tp.sum()+fn.sum()),tp.sum()/max(tp.sum()+fp.sum(),1),fp.sum()/(fp.sum()+tn.sum())])
        a,b,c,d=[v[ix].sum(axis=1) for v in [tp,fp,fn,tn]]
        boot=np.column_stack([np.sqrt(error[ix].mean(axis=1)),a/(a+c),a/np.maximum(a+b,1),b/(b+d)])
        return point,boot
    for name in [f'TCN_seed{s}' for s in SEEDS]+['TCN_ensemble']:
        for rule in ['fixed_30','development_calibrated']:
            t=30. if rule=='fixed_30' else plan['thresholds'][name]
            ref_t=30. if rule=='fixed_30' else plan['thresholds']['XGBoost']
            point,boot=measures(table[name].to_numpy(),t)
            ref,ref_boot=measures(table.XGBoost.to_numpy(),ref_t)
            for j,metric in enumerate(['RMSE','recall','precision','false_positive_rate']):
                low,high=np.quantile(boot[:,j]-ref_boot[:,j],[.025,.975])
                paired.append(dict(model=name,rule=rule,metric=metric,difference=point[j]-ref[j],lower95=low,upper95=high,
                    meaning='TCN minus XGBoost; RMSE cycles, other metrics fractions'))
    pd.DataFrame(paired).to_csv(OUT/'paired_comparisons.csv',index=False)
    fits=[json.loads((OUT/'_runs'/f'seed{s}'/'done.json').read_text()) for s in SEEDS]
    write_json(OUT/'fits.json',fits)
    pd.concat([pd.read_csv(OUT/'_runs'/f'seed{s}'/'learning_curves.csv') for s in SEEDS],ignore_index=True).to_csv(OUT/'learning_curves.csv',index=False)
    assert digest(ROOT/'outputs/experiments/selected_model.joblib')==plan['original_xgboost_sha256']
    write_json(OUT/'results.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        training_engines=709,training_rows=len(train),test_engines=707,seed_models=3,
        ensemble='predeclared equal weights',original_xgboost_unchanged=True,benchmark_status=plan['benchmark_status']))
    log('COMPLETE: all three full-data TCNs and the predeclared ensemble evaluated; original XGBoost unchanged.')

def verify():
    from sklearn.metrics import confusion_matrix
    seq.WINDOW=50
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    plan=json.loads((OUT/'protocol.json').read_text());train,test,_=load_inputs()
    source_fits=pd.read_csv(ROOT/'outputs/tcn_followup/fits.csv')
    source_points=pd.read_csv(ROOT/'outputs/tcn_followup/oof_predictions.csv')
    source_fits=source_fits[source_fits.window.eq(50)&source_fits.budget.eq(60)]
    source_points=source_points[source_points.window.eq(50)&source_points.budget.eq(60)&source_points.protocol.eq('primary')]
    aligned=[]
    for seed in SEEDS:
        selected=source_fits[source_fits.seed.eq(seed)].selected_epoch.to_numpy()
        assert len(selected)==3 and int(np.median(selected))==plan['epochs'][str(seed)]
        points=source_points[source_points.seed.eq(seed)].sort_values(['engine_key','snapshot'])
        assert len(points)==1701 and points.engine_key.nunique()==567
        aligned.append(points.set_index(['engine_key','snapshot'])[['actual_RUL','predicted_RUL']])
    for points in aligned[1:]:
        pd.testing.assert_series_equal(aligned[0].actual_RUL,points.actual_RUL)
    predictions={f'TCN_seed{s}':p.predicted_RUL.to_numpy() for s,p in zip(SEEDS,aligned)}
    predictions['TCN_ensemble']=np.mean(list(predictions.values()),axis=0)
    positive=aligned[0].actual_RUL.to_numpy()<=30
    for name,prediction in predictions.items():
        ordered=np.sort(prediction[positive])
        expected=ordered[int(np.ceil(.9*len(ordered)))-1]
        assert np.isclose(plan['thresholds'][name],expected)
    decision=json.loads((ROOT/'outputs/experiments/frozen_selection.json').read_text())
    assert plan['thresholds']['XGBoost']==decision['alert_threshold']
    processor=joblib.load(OUT/'preprocessor.joblib')
    assert processor.fit_engines==set(train.engine_key)
    fresh=seq.SequenceInputs().fit(train)
    pd.testing.assert_frame_equal(fresh.norm.means_,processor.norm.means_)
    pd.testing.assert_frame_equal(fresh.norm.scales_,processor.norm.scales_)
    endpoints,X,order=test_windows(processor,test)
    p=pd.read_csv(OUT/'test_predictions.csv');np.testing.assert_array_equal(p.actual_RUL,load_truth(endpoints))
    for seed in SEEDS:
        folder=OUT/'_runs'/f'seed{seed}';record=json.loads((folder/'done.json').read_text())
        assert digest(folder/'model.keras')==record['model_sha256']
        curves=pd.read_csv(folder/'learning_curves.csv')
        assert len(curves)==plan['epochs'][str(seed)] and curves.epoch.tolist()==list(range(1,len(curves)+1))
        model=tf.keras.models.load_model(folder/'model.keras')
        assert model.count_params()==26241
        np.testing.assert_allclose(seq.predict(model,X)[order],p[f'TCN_seed{seed}'],rtol=1e-5,atol=.001)
        del model;tf.keras.backend.clear_session()
    np.testing.assert_allclose(p.TCN_ensemble,p[[f'TCN_seed{s}' for s in SEEDS]].mean(axis=1))
    original=joblib.load(ROOT/'outputs/experiments/selected_model.joblib')
    pred=pd.Series(original.predict(test),index=test.index).loc[endpoints.index]
    np.testing.assert_allclose(pred,p.XGBoost,rtol=1e-6,atol=1e-4)
    metrics=pd.read_csv(OUT/'test_metrics.csv');assert len(metrics)==50
    for row in metrics.itertuples():
        g=p if row.dataset=='ALL' else p[p.dataset.eq(row.dataset)]
        expected=30. if row.rule=='fixed_30' else plan['thresholds'][row.model]
        assert np.isclose(row.threshold,expected)
        assert np.isclose(np.sqrt(np.mean((g[row.model]-g.actual_RUL)**2)),row.RMSE)
        tn,fp,fn,tp=confusion_matrix(g.actual_RUL.le(30),g[row.model].le(expected),labels=[False,True]).ravel()
        assert (tn,fp,fn,tp)==(row.TN,row.FP,row.FN,row.TP)
        assert np.isclose(row.recall,tp/(tp+fn))
        assert np.isclose(row.precision,tp/max(tp+fp,1))
        near=g.actual_RUL.le(30)
        assert np.isclose(row.near_failure_RMSE,np.sqrt(np.mean((g.loc[near,row.model]-g.loc[near,'actual_RUL'])**2)))
    pairs=pd.read_csv(OUT/'paired_comparisons.csv');assert len(pairs)==32
    pooled=metrics[metrics.dataset.eq('ALL')].set_index(['model','rule'])
    for row in pairs.itertuples():
        candidate=pooled.loc[(row.model,row.rule)]
        reference=pooled.loc[('XGBoost',row.rule)]
        if row.metric=='false_positive_rate':
            difference=candidate.FP/(candidate.FP+candidate.TN)-reference.FP/(reference.FP+reference.TN)
        else:difference=candidate[row.metric]-reference[row.metric]
        assert np.isclose(row.difference,difference) and row.lower95<=row.upper95
    assert digest(ROOT/'outputs/experiments/selected_model.joblib')==plan['original_xgboost_sha256']
    assert digest(ROOT/'outputs/tcn_followup/fits.csv')==plan['source_fits_sha256']
    assert digest(ROOT/'outputs/tcn_followup/oof_predictions.csv')==plan['source_oof_sha256']
    print('PASS: all 709 training engines, train-only preprocessing, frozen epochs/thresholds, reloaded seed predictions, fixed ensemble, 707 aligned labels, 50 metric rows and unchanged XGBoost.')

def run():
    plan=freeze()
    (OUT/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    pending=[s for s in SEEDS if not (OUT/'_runs'/f'seed{s}'/'done.json').exists()]
    active=[];log(f'Start/resume full-data TCN: {len(pending)} seed fits remain; epochs={plan["epochs"]}.')
    while pending or active:
        while pending and len(active)<2:
            seed=pending.pop(0);folder=OUT/'_runs'/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
            handle=(folder/'console.log').open('w')
            process=subprocess.Popen([sys.executable,__file__,'--worker',str(seed)],stdout=handle,stderr=subprocess.STDOUT)
            active.append((process,handle,seed))
        for item in active[:]:
            process,handle,seed=item
            if process.poll() is not None:
                handle.close();active.remove(item)
                if process.returncode:
                    for p,h,_ in active:p.terminate();h.close()
                    raise RuntimeError(f'Seed {seed} failed; inspect its console.log')
        if active:time.sleep(2)
    evaluate()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int)
    parser.add_argument('--verify',action='store_true');parser.add_argument('--evaluate',action='store_true')
    args=parser.parse_args()
    if args.worker is not None:worker(args.worker)
    elif args.verify:verify()
    elif args.evaluate:evaluate()
    else:run()
