"""Bounded inner-engine regularization selection; outer folds score frozen choices."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
import argparse,gc,hashlib,json,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBRegressor
import run_sequences as seq
import run_tcn_followup as prior
from experiment_models import HistoryFeatures
from train_all import ROOT,DATASETS
from cmapps.utils import append_log, digest, write_json as save
from cmapps.evaluation import score

OUT=ROOT/'outputs/regularization'
SEEDS=[42,73,101]
tf=seq.tf

def log(message):
    append_log(OUT/'training.log', message)

def tcn(features,seed):
    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
    inputs=tf.keras.Input((50,features));x=tf.keras.layers.Conv1D(16,1)(inputs)
    for dilation in [1,2,4,8]:
        skip=x
        for _ in range(2):
            x=tf.keras.layers.Conv1D(16,3,padding='causal',dilation_rate=dilation,activation='relu')(x)
            x=tf.keras.layers.Dropout(.2)(x)
        x=tf.keras.layers.Add()([x,skip])
    x=x[:,-1,:];x=tf.keras.layers.Dense(16,activation='relu')(x);x=tf.keras.layers.Dropout(.2)(x)
    model=tf.keras.Model(inputs,tf.keras.layers.Dense(1)(x))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001,clipnorm=1.),loss='mse')
    return model

def endpoint_predictions(frame,points,prediction):
    p=points.copy();p['predicted_RUL']=pd.Series(prediction,index=frame.index).loc[p.row_index].to_numpy()
    return p

def metrics(training,valid,family,strategy,fold,seed):
    rows=[]
    for protocol in ['primary','broad']:
        for band,limit in [('all',np.inf),('RUL_le150',150),('RUL_le30',30)]:
            a=training[training.protocol.eq(protocol)&training.actual_RUL.le(limit)]
            b=valid[valid.protocol.eq(protocol)&valid.actual_RUL.le(limit)]
            tr=float(np.sum((a.predicted_RUL-a.actual_RUL)**2))
            va=float(np.sum((b.predicted_RUL-b.actual_RUL)**2))
            rows.append(dict(family=family,strategy=strategy,fold=fold,seed=seed,protocol=protocol,band=band,
                training_n=len(a),validation_n=len(b),training_SSE=tr,validation_SSE=va,
                training_RMSE=np.sqrt(tr/len(a)),validation_RMSE=np.sqrt(va/len(b))))
    return rows

def prepare(fold):
    train,outer,valid,fit,stop,stop_points,eval_points=prior.prepare(fold)
    points=pd.read_csv(ROOT/'outputs/experiments/validation_endpoints.csv')
    training_points=points[points.engine_key.isin(outer.engine_key)].copy()
    assert len(training_points)==2268 and len(eval_points)==1134
    return train,outer,valid,fit,stop,stop_points,eval_points,training_points

def tcn_job(fold,seed,folder):
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    train,outer,valid,fit,stop,stop_points,eval_points,training_points=prepare(fold)
    proc=seq.SequenceInputs().fit(fit)
    X,idx,_=prior.windows(proc,fit, window=50);y=np.minimum(train.loc[idx,'RUL'].to_numpy(),150).astype('float32')/150
    Xstop,_,order=prior.windows(proc,stop,stop_points, window=50)
    model=tcn(X.shape[-1],seed);params=model.count_params();ds=seq.dataset(X,y,seed)
    best=np.inf;selected=0;wait=0;curves=[]
    log(f'TCN fold {fold} seed {seed}: inner training, {params} parameters.')
    for epoch in range(1,61):
        h=model.fit(ds,epochs=1,verbose=0,shuffle=False)
        pred=seq.predict(model,Xstop)[order]
        rmse=float(np.sqrt(np.mean((pred-stop_points.actual_RUL)**2)))
        assert np.isfinite(rmse)
        curves.append(dict(epoch=epoch,inner_RMSE=rmse,training_capped_RMSE=np.sqrt(h.history['loss'][0])*150))
        if rmse<best-.01:
            best=rmse;selected=epoch;wait=0
            model.save(folder/'inner.keras')
        else:wait+=1
        if epoch%5==0 or wait>=4:log(f'TCN fold {fold} seed {seed}: epoch {epoch}, best {selected}, inner RMSE {best:.2f}.')
        if wait>=4:break
    pd.DataFrame(curves).to_csv(folder/'curves.csv',index=False)
    base_folder=ROOT/'outputs/tcn_followup/_runs'/f'w50_f{fold}_s{seed}'
    base=json.loads((base_folder/'done.json').read_text())[0]
    winner='regularized' if best<base['inner_RMSE'] else 'baseline'
    save(folder/'selection.json',dict(family='TCN',fold=fold,seed=seed,winner=winner,
        candidates=[dict(name='baseline',inner_RMSE=base['inner_RMSE'],rounds=base['selected_epoch']),
                    dict(name='regularized',inner_RMSE=best,rounds=selected)],parameters=params))
    del model,ds,X,Xstop;gc.collect()
    proc=seq.SequenceInputs().fit(outer)
    X,idx,_=prior.windows(proc,outer, window=50);y=np.minimum(train.loc[idx,'RUL'].to_numpy(),150).astype('float32')/150
    model=tcn(X.shape[-1],seed);ds=seq.dataset(X,y,seed)
    log(f'TCN fold {fold} seed {seed}: refit {selected} epochs; inner choice={winner}.')
    model.fit(ds,epochs=selected,verbose=0,shuffle=False)
    model.save(folder/'outer_regularized.keras')
    del X,ds;gc.collect()
    records=[];predictions=[]
    for name,path in [('regularized',folder/'outer_regularized.keras'),
                      ('baseline',base_folder/f'outer_epoch{base["selected_epoch"]}.keras')]:
        model=tf.keras.models.load_model(path)
        frames=[]
        for frame,points in [(outer,training_points),(valid,eval_points)]:
            xx,_,order=prior.windows(proc,frame,points, window=50)
            pp=points.copy();pp['predicted_RUL']=seq.predict(model,xx)[order];frames.append(pp)
            del xx
        a,b=frames
        if name=='baseline':np.testing.assert_allclose(b.predicted_RUL,pd.read_csv(base_folder/'predictions_60.csv').predicted_RUL,rtol=1e-5,atol=.001)
        a.to_csv(folder/f'training_{name}.csv',index=False);b.to_csv(folder/f'validation_{name}.csv',index=False)
        records+=metrics(a,b,'TCN',name,fold,seed)
        b=b.assign(family='TCN',strategy=name,fold=fold,seed=seed);predictions.append(b)
        if name==winner:
            records+=metrics(a,b,'TCN','inner_selected',fold,seed)
            predictions.append(b.assign(strategy='inner_selected'))
        del model;tf.keras.backend.clear_session()
    return records,pd.concat(predictions,ignore_index=True)

def xgb_model(depth,regularization,child,rounds):
    return XGBRegressor(n_estimators=int(rounds),max_depth=depth,reg_lambda=regularization,
        min_child_weight=child,learning_rate=.03,subsample=.85,colsample_bytree=.85,
        tree_method='hist',device='cpu',objective='reg:squarederror',n_jobs=3,random_state=42)

def stopping_round(errors):
    best=np.inf;selected=0;wait=0
    for i,error in enumerate(errors,1):
        if error<best-.01:best=error;selected=i;wait=0
        else:wait+=1
        if wait>=40:break
    return selected

def xgb_job(fold,seed,folder):
    train,outer,valid,fit,stop,stop_points,eval_points,training_points=prepare(fold)
    transformer=HistoryFeatures(True,True).fit(fit)
    selector=VarianceThreshold(1e-8)
    X=selector.fit_transform(transformer.transform(fit))
    xs=selector.transform(transformer.transform(stop).loc[stop_points.row_index])
    y=np.minimum(fit.RUL.to_numpy(),150)
    candidates=[];curves=[]
    recipes=[('early_stop',6,10,1),('shallow_regularized',4,20,10),('strong_regularized',6,40,10)]
    for name,depth,regularization,child in recipes:
        model=xgb_model(depth,regularization,child,800)
        model.fit(X,y,eval_set=[(xs,stop_points.actual_RUL.to_numpy())],verbose=False)
        errors=model.evals_result()['validation_0']['rmse']
        chosen=stopping_round(errors)
        for label,rounds in ([('baseline',800),(name,chosen)] if name=='early_stop' else [(name,chosen)]):
            pred=model.predict(xs,iteration_range=(0,rounds));pred=np.maximum(0,pred)
            rmse=float(np.sqrt(np.mean((pred-stop_points.actual_RUL)**2)))
            candidates.append(dict(name=label,inner_RMSE=rmse,rounds=rounds,depth=depth,reg_lambda=regularization,min_child_weight=child))
        curves.extend(dict(candidate=name,round=i,inner_RMSE=e) for i,e in enumerate(errors,1))
    # Stable ordering makes exact ties prefer the existing baseline.
    candidates.sort(key=lambda c:c['name']!='baseline')
    winner=min(candidates,key=lambda c:c['inner_RMSE'])['name']
    save(folder/'selection.json',dict(family='XGBoost',fold=fold,seed=42,winner=winner,candidates=candidates))
    pd.DataFrame(curves).to_csv(folder/'curves.csv',index=False)
    transformer=HistoryFeatures(True,True).fit(outer);selector=VarianceThreshold(1e-8)
    X=selector.fit_transform(transformer.transform(outer));xv=selector.transform(transformer.transform(valid))
    xt=selector.transform(transformer.transform(outer).loc[training_points.row_index])
    y=np.minimum(outer.RUL.to_numpy(),150);records=[];predictions=[]
    saved=pd.read_csv(ROOT/'outputs/experiments/oof_predictions.csv')
    saved=saved[saved.candidate.eq('tuned_d6_n800')&saved.fold.eq(fold)]
    for c in candidates:
        name=c['name'];model=xgb_model(c['depth'],c['reg_lambda'],c['min_child_weight'],c['rounds']);model.fit(X,y)
        a=training_points.copy();a['predicted_RUL']=np.maximum(0,model.predict(xt))
        b=endpoint_predictions(valid,eval_points,np.maximum(0,model.predict(xv)))
        if name=='baseline':
            match=b.merge(saved,on=['engine_key','row_index','protocol','snapshot'],validate='one_to_one',suffixes=('_new','_old'))
            np.testing.assert_allclose(match.predicted_RUL_new,match.predicted_RUL_old,rtol=1e-6,atol=1e-4)
        joblib.dump((transformer,selector,model),folder/f'outer_{name}.joblib')
        a.to_csv(folder/f'training_{name}.csv',index=False);b.to_csv(folder/f'validation_{name}.csv',index=False)
        records+=metrics(a,b,'XGBoost',name,fold,42)
        b=b.assign(family='XGBoost',strategy=name,fold=fold,seed=42);predictions.append(b)
        if name==winner:
            records+=metrics(a,b,'XGBoost','inner_selected',fold,42)
            predictions.append(b.assign(strategy='inner_selected'))
    log(f'XGBoost fold {fold}: inner choice={winner}.')
    return records,pd.concat(predictions,ignore_index=True)

def worker(family,fold,seed):
    folder=OUT/'_runs'/f'{family}_f{fold}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'done.json').exists():return
    start=time.perf_counter()
    records,pred=(tcn_job if family=='TCN' else xgb_job)(fold,seed,folder)
    pd.DataFrame(records).to_csv(folder/'metrics.csv',index=False)
    pred.to_csv(folder/'predictions.csv',index=False)
    save(folder/'done.json',dict(family=family,fold=fold,seed=seed,seconds=time.perf_counter()-start))
    log(f'{family} fold {fold} seed {seed}: complete in {time.perf_counter()-start:.1f}s.')

def summarize():
    folders=[p.parent for p in (OUT/'_runs').glob('*/done.json')]
    if not folders:return
    metrics_table=pd.concat([pd.read_csv(p/'metrics.csv') for p in folders],ignore_index=True)
    pred=pd.concat([pd.read_csv(p/'predictions.csv') for p in folders],ignore_index=True)
    choices=[json.loads((p/'selection.json').read_text()) for p in folders]
    metrics_table.to_csv(OUT/'fold_metrics.csv',index=False);pred.to_csv(OUT/'oof_predictions.csv',index=False)
    save(OUT/'selections.json',choices)
    rows=[];paired=[]
    for keys,g in metrics_table.groupby(['family','strategy','seed','protocol','band']):
        if g.fold.nunique()!=3:continue
        assert g.training_n.sum()==2*g.validation_n.sum()
        tr=np.sqrt(g.training_SSE.sum()/g.training_n.sum());va=np.sqrt(g.validation_SSE.sum()/g.validation_n.sum())
        rows.append(dict(zip(['family','strategy','seed','protocol','band'],keys),training_RMSE=tr,validation_RMSE=va,gap=va-tr))
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    for (family,strategy,seed,protocol),g in pred.groupby(['family','strategy','seed','protocol']):
        if strategy=='baseline' or g.fold.nunique()!=3:continue
        baseline=pred[pred.family.eq(family)&pred.strategy.eq('baseline')&pred.seed.eq(seed)&pred.protocol.eq(protocol)]
        keys=['engine_key','row_index','protocol','snapshot']
        p=g.merge(baseline[keys+['predicted_RUL']],on=keys,validate='one_to_one',suffixes=('_candidate','_baseline'))
        assert len(p)==1701
        for band,limit in [('all',np.inf),('RUL_le30',30)]:
            q=p[p.actual_RUL.le(limit)].copy()
            q['a']=(q.predicted_RUL_candidate-q.actual_RUL)**2;q['b']=(q.predicted_RUL_baseline-q.actual_RUL)**2
            grouped=q.groupby('engine_key')[['a','b']].sum();n=q.groupby('engine_key').size().to_numpy()
            sums=grouped.to_numpy();rng=np.random.default_rng(42)
            ix=rng.integers(0,len(n),(2000,len(n)))
            boots=np.sqrt(sums[ix].sum(axis=1)/n[ix].sum(axis=1)[:,None]);diff=boots[:,0]-boots[:,1]
            point=np.sqrt(sums.sum(axis=0)/n.sum());low,high=np.quantile(diff,[.025,.975])
            paired.append(dict(family=family,strategy=strategy,seed=seed,protocol=protocol,band=band,
                difference=point[0]-point[1],lower95=low,upper95=high,engines=len(n)))
    pd.DataFrame(paired).to_csv(OUT/'paired_comparisons.csv',index=False)
    return metrics_table,pred

def verify():
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    folders=[p.parent for p in (OUT/'_runs').glob('*/done.json')]
    assert len(folders)==12
    data={f:prepare(f) for f in [1,2,3]}
    all_metrics=[];all_predictions=[]
    for folder in folders:
        selection=json.loads((folder/'selection.json').read_text())
        family,fold,seed=selection['family'],selection['fold'],selection['seed']
        train,outer,valid,fit,stop,stop_points,eval_points,training_points=data[fold]
        assert selection['winner']==min(selection['candidates'],key=lambda c:c['inner_RMSE'])['name']
        curves=pd.read_csv(folder/'curves.csv')
        if family=='TCN':
            proc=seq.SequenceInputs().fit(fit)
            xx,_,order=prior.windows(proc,stop,stop_points, window=50)
            inner=tf.keras.models.load_model(folder/'inner.keras')
            predicted=seq.predict(inner,xx)[order]
            actual=float(np.sqrt(np.mean((predicted-stop_points.actual_RUL)**2)))
            candidate=next(c for c in selection['candidates'] if c['name']=='regularized')
            assert np.isclose(actual,candidate['inner_RMSE'],rtol=1e-5,atol=.001)
            best=np.inf;chosen=0;wait=0
            for row in curves.itertuples():
                if row.inner_RMSE<best-.01:best=row.inner_RMSE;chosen=row.epoch;wait=0
                else:wait+=1
                if wait>=4:break
            assert chosen==candidate['rounds']
            del inner,xx;tf.keras.backend.clear_session()
        for candidate in selection['candidates']:
            name=candidate['name']
            if family=='XGBoost':
                if name!='baseline':
                    errors=curves[curves.candidate.eq(name)].inner_RMSE.to_numpy()
                    assert stopping_round(errors)==candidate['rounds']
                transformer,selector,model=joblib.load(folder/f'outer_{name}.joblib')
                assert transformer.fit_engines_==set(outer.engine_key)
            else:
                proc=seq.SequenceInputs().fit(outer)
                if name=='regularized':path=folder/'outer_regularized.keras'
                else:path=ROOT/'outputs/tcn_followup/_runs'/f'w50_f{fold}_s{seed}'/f'outer_epoch{candidate["rounds"]}.keras'
                model=tf.keras.models.load_model(path)
            frames=[]
            for label,frame,points in [('training',outer,training_points),('validation',valid,eval_points)]:
                saved=pd.read_csv(folder/f'{label}_{name}.csv')
                cols=['engine_key','row_index','actual_RUL','protocol','snapshot']
                pd.testing.assert_frame_equal(saved[cols].reset_index(drop=True),points[cols].reset_index(drop=True),check_dtype=False)
                if family=='XGBoost':
                    xx=selector.transform(transformer.transform(frame).loc[points.row_index])
                    predicted=np.maximum(0,model.predict(xx))
                else:
                    xx,_,order=prior.windows(proc,frame,points, window=50);predicted=seq.predict(model,xx)[order]
                np.testing.assert_allclose(predicted,saved.predicted_RUL,rtol=1e-5,atol=.001)
                frames.append(saved)
                del xx
            a,b=frames
            all_metrics+=metrics(a,b,family,name,fold,seed)
            all_predictions.append(b.assign(family=family,strategy=name,fold=fold,seed=seed))
            if name==selection['winner']:
                all_metrics+=metrics(a,b,family,'inner_selected',fold,seed)
                all_predictions.append(b.assign(family=family,strategy='inner_selected',fold=fold,seed=seed))
            del model
            if family=='TCN':tf.keras.backend.clear_session()
        print(f'Verified {family} fold {fold} seed {seed}',flush=True)
    expected=pd.DataFrame(all_metrics);actual=pd.read_csv(OUT/'fold_metrics.csv')
    keys=['family','strategy','fold','seed','protocol','band']
    pd.testing.assert_frame_equal(expected.sort_values(keys).reset_index(drop=True),actual.sort_values(keys).reset_index(drop=True),check_exact=False,rtol=1e-6,atol=1e-6)
    pred=pd.concat(all_predictions,ignore_index=True);saved=pd.read_csv(OUT/'oof_predictions.csv')
    keys=['family','strategy','seed','fold','protocol','snapshot','engine_key']
    pd.testing.assert_frame_equal(pred.sort_values(keys).reset_index(drop=True),saved.sort_values(keys).reset_index(drop=True),check_exact=False,rtol=1e-6,atol=1e-6)
    assert len(actual)==252 and len(pred)==47628
    pooled=pd.read_csv(OUT/'metrics.csv');paired=pd.read_csv(OUT/'paired_comparisons.csv')
    assert len(pooled)==84 and len(paired)==40
    for row in pooled.itertuples():
        g=pred[pred.family.eq(row.family)&pred.strategy.eq(row.strategy)&pred.seed.eq(row.seed)&pred.protocol.eq(row.protocol)]
        if row.band!='all':g=g[g.actual_RUL.le(int(row.band.split('le')[-1]))]
        assert np.isclose(np.sqrt(np.mean((g.actual_RUL-g.predicted_RUL)**2)),row.validation_RMSE)
        f=actual[actual.family.eq(row.family)&actual.strategy.eq(row.strategy)&actual.seed.eq(row.seed)&actual.protocol.eq(row.protocol)&actual.band.eq(row.band)]
        assert f.training_n.sum()==2*f.validation_n.sum()
        assert np.isclose(np.sqrt(f.training_SSE.sum()/f.training_n.sum()),row.training_RMSE)
        assert np.isclose(row.validation_RMSE-row.training_RMSE,row.gap)
    for row in paired.itertuples():
        m=pooled[pooled.family.eq(row.family)&pooled.seed.eq(row.seed)&pooled.protocol.eq(row.protocol)&pooled.band.eq(row.band)].set_index('strategy')
        assert np.isclose(m.loc[row.strategy,'validation_RMSE']-m.loc['baseline','validation_RMSE'],row.difference)
    protocol=json.loads((OUT/'protocol.json').read_text())
    for filename,key in [('outputs/experiments/selected_model.joblib','model_sha256'),
                         ('outputs/experiments/cv_engine_folds.csv','folds_sha256'),
                         ('outputs/experiments/validation_endpoints.csv','endpoints_sha256'),
                         ('outputs/tcn_followup/fits.csv','baseline_tcn_fits_sha256')]:
        assert digest(ROOT/filename)==protocol[key]
    print('PASS: all saved outer models reproduce train/validation predictions; inner choices, endpoint roles, stopping rules, metric counts and frozen inputs checked.')


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    config=dict(selection='Minimum uncapped inner primary endpoint RMSE; exact ties prefer baseline',
        folds=3,seeds=SEEDS,tcn=dict(window=50,channels=16,dropout=.2,max_epochs=60,patience=4,min_delta=.01),
        xgboost=dict(max_rounds=800,patience=40,min_delta=.01,recipes=['baseline800','early_stop','depth4_lambda20_child10','depth6_lambda40_child10']),
        workers=2,threads_per_worker=3,model_sha256=digest(ROOT/'outputs/experiments/selected_model.joblib'),
        folds_sha256=digest(ROOT/'outputs/experiments/cv_engine_folds.csv'),endpoints_sha256=digest(ROOT/'outputs/experiments/validation_endpoints.csv'),
        baseline_tcn_fits_sha256=digest(ROOT/'outputs/tcn_followup/fits.csv'))
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==config
    else:save(OUT/'protocol.json',config)
    (OUT/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    jobs=[('XGBoost',f,42) for f in [1,2,3]]+[('TCN',f,s) for f in [1,2,3] for s in SEEDS]
    pending=[j for j in jobs if not (OUT/'_runs'/f'{j[0]}_f{j[1]}_s{j[2]}'/'done.json').exists()]
    active=[];log(f'Start/resume: {len(pending)} of 12 jobs remain.')
    while pending or active:
        while pending and len(active)<2:
            family,fold,seed=pending.pop(0);folder=OUT/'_runs'/f'{family}_f{fold}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
            handle=(folder/'console.log').open('w')
            process=subprocess.Popen([sys.executable,__file__,'--worker',family,str(fold),str(seed)],stdout=handle,stderr=subprocess.STDOUT)
            active.append((process,handle,family,fold,seed))
        for item in active[:]:
            process,handle,*job=item
            if process.poll() is not None:
                handle.close();active.remove(item)
                if process.returncode:
                    for p,h,*_ in active:p.terminate();h.close()
                    raise RuntimeError(f'Failed {job}; inspect console.log')
                summarize()
        if active:time.sleep(2)
    m,p=summarize();assert len(m)==252 and len(p)==47628
    assert digest(ROOT/'outputs/experiments/selected_model.joblib')==config['model_sha256']
    save(OUT/'results.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),jobs=12,scored_configurations=42,final_model_unchanged=True))
    log('COMPLETE: 12 jobs; all choices made on inner engines; original final model unchanged.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',nargs=3);parser.add_argument('--summarize',action='store_true')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.worker:worker(args.worker[0],int(args.worker[1]),int(args.worker[2]))
    elif args.summarize:summarize()
    elif args.verify:verify()
    else:run()
