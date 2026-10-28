"""Resumable TCN window/budget comparison with inner-engine alert calibration."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
import argparse,gc,hashlib,json,math,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import run_sequences as seq
from run_experiments import score
from train_all import ROOT,DATASETS,load_inputs
from sklearn.model_selection import train_test_split

tf=seq.tf
OUT=ROOT/'outputs/tcn_followup'
SEEDS=[42,73,101]
WINDOWS=[30,50]


def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


def log(message):
    line=f'{datetime.now(timezone.utc).isoformat(timespec="seconds")} {message}'
    print(line,flush=True)
    with (OUT/'training.log').open('a') as f:f.write(line+'\n')


def threshold(actual,pred):
    positives=np.asarray(pred)[np.asarray(actual)<=30]
    if not len(positives):raise ValueError('No calibration events')
    return float(np.sort(positives)[math.ceil(.9*len(positives))-1])


def prepare(fold):
    train,_,_=load_inputs()
    folds=pd.read_csv(ROOT/'outputs/experiments/cv_engine_folds.csv')
    endpoints=pd.read_csv(ROOT/'outputs/experiments/validation_endpoints.csv')
    outer=train[train.engine_key.isin(folds.loc[folds.fold.ne(fold),'engine_key'])]
    valid=train[train.engine_key.isin(folds.loc[folds.fold.eq(fold),'engine_key'])]
    engines=outer[['engine_key','dataset']].drop_duplicates().sort_values('engine_key')
    a,b=train_test_split(engines.engine_key,test_size=.2,random_state=100+fold,stratify=engines.dataset)
    fit=outer[outer.engine_key.isin(a)];stop=outer[outer.engine_key.isin(b)]
    assert set(a).isdisjoint(b) and set(outer.engine_key).isdisjoint(valid.engine_key)
    stop_points=endpoints[endpoints.engine_key.isin(b)&endpoints.protocol.eq('primary')].copy()
    eval_points=endpoints[endpoints.engine_key.isin(valid.engine_key)].copy()
    return train,outer,valid,fit,stop,stop_points,eval_points


def windows(processor,frame,points=None):
    X,indices=processor.windows(frame,None if points is None else points.row_index)
    order=None if points is None else pd.Series(np.arange(len(indices)),index=indices).loc[points.row_index].to_numpy()
    return X,indices,order


def worker(window,fold,seed):
    seq.WINDOW=window
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    name=f'w{window}_f{fold}_s{seed}';out=OUT/'_runs'/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'done.json').exists():return
    start=time.perf_counter()
    train,outer,valid,fit,stop,stop_points,eval_points=prepare(fold)
    processor=seq.SequenceInputs().fit(fit)
    Xfit,indices,_=windows(processor,fit)
    yfit=np.minimum(train.loc[indices,'RUL'].to_numpy(),150).astype('float32')/150
    Xstop,_,stop_order=windows(processor,stop,stop_points)
    model=seq.network('TCN',Xfit.shape[-1],seed)
    ds=seq.dataset(Xfit,yfit,seed)
    curves=[];best=float('inf');best_epoch=1;wait=0;chosen={}
    log(f'{name}: inner training (302 fit / 76 stop engines).')
    for epoch in range(1,61):
        h=model.fit(ds,epochs=1,verbose=0,shuffle=False)
        prediction=seq.predict(model,Xstop)[stop_order]
        rmse=float(np.sqrt(np.mean((stop_points.actual_RUL.to_numpy()-prediction)**2)))
        if not np.isfinite(rmse):raise FloatingPointError(f'{name}: non-finite validation error')
        curves.append(dict(window=window,fold=fold,seed=seed,epoch=epoch,training_capped_RMSE=float(np.sqrt(h.history['loss'][0])*150),inner_primary_RMSE=rmse))
        pd.DataFrame(curves).to_csv(out/'curves.csv',index=False)
        if rmse<best-.01:
            best=rmse;best_epoch=epoch;wait=0
            model.save(out/'inner_best.keras')
            p=stop_points.copy();p['predicted_RUL']=prediction;p.to_csv(out/'inner_best.csv',index=False)
        else:wait+=1
        if epoch%5==0 or wait>=4:log(f'{name}: epoch {epoch}, inner RMSE={rmse:.2f}, selected={best_epoch}.')
        if window==30 and (epoch==20 or (epoch<20 and wait>=4)):
            import shutil
            shutil.copy2(out/'inner_best.keras',out/'inner_20.keras')
            shutil.copy2(out/'inner_best.csv',out/'inner_20.csv')
            chosen[20]=dict(selected_epoch=best_epoch,inner_RMSE=best,epochs_observed=epoch)
        if wait>=4:break
    chosen[60]=dict(selected_epoch=best_epoch,inner_RMSE=best,epochs_observed=epoch)
    del model,ds,Xfit,Xstop;gc.collect()
    processor=seq.SequenceInputs().fit(outer)
    Xouter,indices,_=windows(processor,outer)
    youter=np.minimum(train.loc[indices,'RUL'].to_numpy(),150).astype('float32')/150
    Xeval,_,eval_order=windows(processor,valid,eval_points)
    for d in DATASETS:
        key=valid.loc[valid.dataset.eq(d),'engine_key'].iloc[0]
        prefix=valid[valid.engine_key.eq(key)].iloc[:23].copy();row=prefix.index[-1]
        A=processor.windows(valid[valid.engine_key.eq(key)],[row])[0]
        prefix['RUL']=-999
        np.testing.assert_array_equal(A,processor.windows(prefix,[row])[0])
        assert np.all(A[0,:window-len(prefix)]==0)
    results=[];by_epoch={}
    for budget,selection in chosen.items():
        selected=selection['selected_epoch']
        inner=pd.read_csv(out/('inner_20.csv' if budget==20 else 'inner_best.csv'))
        t=threshold(inner.actual_RUL,inner.predicted_RUL)
        if selected not in by_epoch:
            log(f'{name}: refit all 378 fitting engines for {selected} epochs (budget {budget}).')
            model=seq.network('TCN',Xouter.shape[-1],seed)
            ds=seq.dataset(Xouter,youter,seed)
            class Progress(tf.keras.callbacks.Callback):
                def on_epoch_end(self,epoch,logs=None):
                    if (epoch+1)%10==0:log(f'{name}: refit epoch {epoch+1}/{selected}.')
            model.fit(ds,epochs=selected,verbose=0,shuffle=False,callbacks=[Progress()])
            prediction=seq.predict(model,Xeval)[eval_order]
            model.save(out/f'outer_epoch{selected}.keras')
            by_epoch[selected]=prediction
            del model,ds;gc.collect()
        record=eval_points.copy();record['predicted_RUL']=by_epoch[selected]
        record['window']=window;record['budget']=budget;record['fold']=fold;record['seed']=seed
        record['calibrated_threshold']=t
        record.to_csv(out/f'predictions_{budget}.csv',index=False)
        primary=record[record.protocol.eq('primary')]
        fixed=score(primary.actual_RUL,primary.predicted_RUL)
        calibrated=score(primary.actual_RUL,primary.predicted_RUL,t)
        results.append(dict(window=window,budget=budget,fold=fold,seed=seed,**selection,
            threshold=t,calibration_events=int(inner.actual_RUL.le(30).sum()),
            inner_calibrated_recall=score(inner.actual_RUL,inner.predicted_RUL,t)['recall'],
            primary_RMSE=fixed['RMSE'],fixed_recall=fixed['recall'],calibrated_recall=calibrated['recall'],
            fit_engines=outer.engine_key.nunique(),evaluation_engines=valid.engine_key.nunique(),
            parameters=26241,seconds=time.perf_counter()-start))
        log(f'{name}: budget {budget}, outer RMSE={fixed["RMSE"]:.2f}, calibrated recall={calibrated["recall"]:.1%}, threshold={t:.2f}.')
    write_json(out/'done.json',results)
    log(f'{name}: completed in {time.perf_counter()-start:.1f}s.')


def summarize():
    fits=[];predictions=[];curves=[]
    for window in WINDOWS:
        for fold in [1,2,3]:
            for seed in SEEDS:
                run=OUT/'_runs'/f'w{window}_f{fold}_s{seed}'
                if not (run/'done.json').exists():continue
                f=json.loads((run/'done.json').read_text());fits.extend(f)
                curves.append(pd.read_csv(run/'curves.csv'))
                for item in f:predictions.append(pd.read_csv(run/f'predictions_{item["budget"]}.csv'))
    if not fits:return
    pd.DataFrame(fits).to_csv(OUT/'fits.csv',index=False)
    pd.concat(curves,ignore_index=True).to_csv(OUT/'learning_curves.csv',index=False)
    pred=pd.concat(predictions,ignore_index=True);pred.to_csv(OUT/'oof_predictions.csv',index=False)
    metrics=[]
    for (window,budget,seed,protocol),g in pred.groupby(['window','budget','seed','protocol']):
        if g.fold.nunique()!=3:continue
        for rule in ['fixed_30','inner_calibrated']:
            actual=g.actual_RUL.to_numpy();prediction=g.predicted_RUL.to_numpy()
            m=score(actual,prediction)
            if rule=='inner_calibrated':
                flag=prediction<=g.calibrated_threshold.to_numpy();truth=actual<=30
                tp=int((flag&truth).sum());fp=int((flag&~truth).sum());fn=int((~flag&truth).sum());tn=int((~flag&~truth).sum())
                m.update(threshold=None,TP=tp,FP=fp,FN=fn,TN=tn,recall=tp/(tp+fn),precision=tp/max(tp+fp,1))
            metrics.append(dict(window=window,budget=budget,seed=seed,protocol=protocol,rule=rule,**m))
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
    return pred,metrics


def paired_comparisons(pred):
    baseline=pd.read_csv(ROOT/'outputs/experiments/oof_predictions.csv')
    baseline=baseline[baseline.candidate.eq('tuned_d6_n800')]
    rows=[]
    keys=['engine_key','protocol','snapshot','row_index']
    for (window,budget,seed,protocol),g in pred.groupby(['window','budget','seed','protocol']):
        comparators={'XGBoost':baseline[baseline.protocol.eq(protocol)]}
        if budget==60:
            other=pred[(pred.window==30)&(pred.budget==(20 if window==30 else 60))&(pred.seed==seed)&(pred.protocol==protocol)]
            comparators['TCN_30_20' if window==30 else 'TCN_30_60']=other
        for label,ref in comparators.items():
            p=g.merge(ref[keys+['predicted_RUL']],on=keys,suffixes=('_candidate','_reference'),validate='one_to_one')
            assert len(p)==1701
            p['a']=(p.predicted_RUL_candidate-p.actual_RUL)**2;p['b']=(p.predicted_RUL_reference-p.actual_RUL)**2
            per=p.groupby('engine_key')[['a','b']].mean().to_numpy()
            rng=np.random.default_rng(42);indices=rng.integers(0,len(per),(2000,len(per)))
            sample=np.sqrt(per[indices].mean(axis=1));diff=sample[:,0]-sample[:,1]
            point=np.sqrt(per.mean(axis=0));low,high=np.quantile(diff,[.025,.975])
            rows.append(dict(window=window,budget=budget,seed=seed,protocol=protocol,reference=label,difference=point[0]-point[1],lower95=low,upper95=high))
    pd.DataFrame(rows).to_csv(OUT/'paired_comparisons.csv',index=False)


def alert_intervals(pred):
    rows=[]
    for (window,budget,seed,protocol),group in pred.groupby(['window','budget','seed','protocol']):
        for rule in ['fixed_30','inner_calibrated']:
            g=group.copy();truth=g.actual_RUL.le(30)
            flag=g.predicted_RUL.le(30 if rule=='fixed_30' else g.calibrated_threshold)
            g['TP']=(truth&flag).astype(int);g['FP']=(~truth&flag).astype(int);g['FN']=(truth&~flag).astype(int)
            counts=g.groupby('engine_key')[['TP','FP','FN']].sum().to_numpy()
            rng=np.random.default_rng(42)
            samples=counts[rng.integers(0,len(counts),(2000,len(counts)))].sum(axis=1)
            total=counts.sum(axis=0)
            for metric,denominator in [('precision',1),('recall',2)]:
                values=samples[:,0]/np.maximum(samples[:,0]+samples[:,denominator],1)
                low,high=np.quantile(values,[.025,.975])
                rows.append(dict(window=window,budget=budget,seed=seed,protocol=protocol,rule=rule,metric=metric,
                    estimate=total[0]/max(total[0]+total[denominator],1),lower95=low,upper95=high,engines=len(counts),draws=2000))
    pd.DataFrame(rows).to_csv(OUT/'alert_intervals.csv',index=False)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    original=ROOT/'outputs/experiments/selected_model.joblib'
    config=dict(windows=WINDOWS,seeds=SEEDS,max_epochs=60,patience=4,min_delta=.01,batch_size=512,
        threads_per_worker=3,workers=2,calibration='90% recall on inner best-epoch predictions; transfer threshold to outer refit; evaluate outer engines',
        target_cap=150,learning_rate=.001,architecture='Same 32-channel residual TCN',
        original_model_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        split_sha256=hashlib.sha256((ROOT/'outputs/experiments/cv_engine_folds.csv').read_bytes()).hexdigest(),
        endpoints_sha256=hashlib.sha256((ROOT/'outputs/experiments/validation_endpoints.csv').read_bytes()).hexdigest())
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==config
    else:write_json(OUT/'protocol.json',config)
    (OUT/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    jobs=[(w,f,s) for w in WINDOWS for f in [1,2,3] for s in SEEDS]
    pending=[j for j in jobs if not (OUT/'_runs'/f'w{j[0]}_f{j[1]}_s{j[2]}'/'done.json').exists()]
    active=[];start=time.perf_counter();log(f'Start/resume: {len(pending)} of {len(jobs)} jobs remaining.')
    while pending or active:
        while pending and len(active)<2:
            w,f,s=pending.pop(0);folder=OUT/'_runs'/f'w{w}_f{f}_s{s}';folder.mkdir(parents=True,exist_ok=True)
            handle=(folder/'console.log').open('w')
            process=subprocess.Popen([sys.executable,__file__,'--worker',str(w),str(f),str(s)],stdout=handle,stderr=subprocess.STDOUT)
            active.append((process,handle,(w,f,s)))
        for item in active[:]:
            process,handle,job=item
            if process.poll() is not None:
                handle.close();active.remove(item)
                if process.returncode:
                    for p,h,j in active:p.terminate();h.close()
                    raise RuntimeError(f'Worker {job} failed; inspect its console.log')
                summarize()
        if active:time.sleep(2)
    pred,metrics=summarize();assert len(pred)==30618
    paired_comparisons(pred)
    alert_intervals(pred)
    assert hashlib.sha256(original.read_bytes()).hexdigest()==config['original_model_sha256']
    write_json(OUT/'results.json',dict(protocol=config,metrics=metrics,completed_utc=datetime.now(timezone.utc).isoformat(),session_seconds=time.perf_counter()-start))
    log('COMPLETE: 18 jobs, 27 scored fold models; original final XGBoost unchanged.')


def diagnostics():
    """Score saved refits on matching train/validation endpoint protocols."""
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    fits=pd.read_csv(OUT/'fits.csv')
    points=pd.read_csv(ROOT/'outputs/experiments/validation_endpoints.csv')
    original=ROOT/'outputs/experiments/selected_model.joblib'
    before=hashlib.sha256(original.read_bytes()).hexdigest()
    rows=[];curve_rows=[]
    for fold in [1,2,3]:
        train,outer,valid,inner_fit,stop,stop_points,eval_points=prepare(fold)
        fit_points=points[points.engine_key.isin(outer.engine_key)].copy()
        assert set(fit_points.engine_key).isdisjoint(eval_points.engine_key)
        processor=seq.SequenceInputs().fit(outer)
        for window in WINDOWS:
            seq.WINDOW=window
            Xfit,_,fit_order=windows(processor,outer,fit_points)
            Xvalid,_,valid_order=windows(processor,valid,eval_points)
            for dataset_name in DATASETS:
                key=valid.loc[valid.dataset.eq(dataset_name),'engine_key'].iloc[0]
                history=valid[valid.engine_key.eq(key)].sort_values('cycle').copy()
                endpoint=history.index[22]
                expected=processor.windows(history,[endpoint])[0]
                altered=history.copy();altered['RUL']=-999
                altered.loc[altered.index[23:],seq.SENSORS]=1e6
                np.testing.assert_array_equal(expected,processor.windows(altered,[endpoint])[0])
                np.testing.assert_array_equal(expected,processor.windows(history.iloc[:23],[endpoint])[0])
            cache={}
            for item in fits[(fits.fold==fold)&(fits.window==window)].itertuples():
                run=OUT/'_runs'/f'w{window}_f{fold}_s{item.seed}'
                key=(item.seed,item.selected_epoch)
                if key not in cache:
                    model=tf.keras.models.load_model(run/f'outer_epoch{item.selected_epoch}.keras')
                    cache[key]=(seq.predict(model,Xfit)[fit_order],seq.predict(model,Xvalid)[valid_order])
                    del model
                    tf.keras.backend.clear_session()
                fit_pred,valid_pred=cache[key]
                saved=pd.read_csv(run/f'predictions_{item.budget}.csv')
                np.testing.assert_allclose(valid_pred,saved.predicted_RUL,rtol=1e-5,atol=.001)
                a=fit_points.copy();a['prediction']=fit_pred
                b=eval_points.copy();b['prediction']=valid_pred
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
                        train_rmse=np.sqrt(train_sse/len(aa));valid_rmse=np.sqrt(valid_sse/len(bb))
                        rows.append(dict(window=window,budget=item.budget,seed=item.seed,fold=fold,
                            protocol=protocol,subset=subset,training_n=len(aa),validation_n=len(bb),
                            training_SSE=train_sse,validation_SSE=valid_sse,
                            training_RMSE=train_rmse,validation_RMSE=valid_rmse,gap=valid_rmse-train_rmse))
                curve=pd.read_csv(run/'curves.csv');curve=curve[curve.epoch<=item.budget]
                selected=curve[curve.epoch.eq(item.selected_epoch)].iloc[0];last=curve.iloc[-1]
                curve_rows.append(dict(window=window,budget=item.budget,seed=item.seed,fold=fold,
                    selected_epoch=item.selected_epoch,last_epoch=int(last.epoch),
                    selected_inner_RMSE=selected.inner_primary_RMSE,last_inner_RMSE=last.inner_primary_RMSE,
                    inner_change_after_selection=last.inner_primary_RMSE-selected.inner_primary_RMSE,
                    selected_training_loss_RMSE=selected.training_capped_RMSE,
                    last_training_loss_RMSE=last.training_capped_RMSE))
                print(f'Diagnosed window={window}, fold={fold}, seed={item.seed}, budget={item.budget}',flush=True)
            del Xfit,Xvalid;gc.collect()
    gaps=pd.DataFrame(rows)
    assert len(gaps)==27*2*7
    all_rows=gaps[gaps.subset.eq('all')]
    assert all_rows.training_n.eq(1134).all() and all_rows.validation_n.eq(567).all()
    gaps.to_csv(OUT/'generalization_gaps.csv',index=False)
    pd.DataFrame(curve_rows).to_csv(OUT/'stopping_diagnostics.csv',index=False)
    pooled=[]
    for keys,g in gaps.groupby(['window','budget','seed','protocol','subset']):
        assert g.fold.nunique()==3
        tr=np.sqrt(g.training_SSE.sum()/g.training_n.sum())
        va=np.sqrt(g.validation_SSE.sum()/g.validation_n.sum())
        pooled.append(dict(zip(['window','budget','seed','protocol','subset'],keys),
            training_RMSE=tr,validation_RMSE=va,gap=va-tr,
            minimum_fold_gap=g.gap.min(),maximum_fold_gap=g.gap.max()))
    pd.DataFrame(pooled).to_csv(OUT/'generalization_summary.csv',index=False)
    assert hashlib.sha256(original.read_bytes()).hexdigest()==before
    print('PASS: 27 saved fold configurations; matched uncapped scoring, disjoint engines, future/target isolation, reproduced validation predictions and unchanged XGBoost.')


def compare_alerts():
    """Compare frozen model families under identical inner-only recall calibration."""
    import joblib
    from experiment_models import ExperimentRegressor
    target=ROOT/'outputs/alert_comparison';target.mkdir(parents=True,exist_ok=True)
    local=target/'_runs';local.mkdir(exist_ok=True)
    original=ROOT/'outputs/experiments/selected_model.joblib'
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    config=json.loads((ROOT/'outputs/experiments/frozen_selection.json').read_text())['selected_config']
    plan=dict(target_recall=.90,event='actual RUL <=30',primary_protocol='primary',
        rule='Smallest threshold achieving >=90% inner recall; transfer unchanged to outer refit',
        inner_fit_engines=302,inner_calibration_engines=76,outer_fit_engines=378,outer_evaluation_engines=189,
        xgboost_config=config,tcn_window=50,tcn_seeds=SEEDS,bootstrap_draws=2000,
        final_model_sha256=digest(original),
        folds_sha256=digest(ROOT/'outputs/experiments/cv_engine_folds.csv'),
        endpoints_sha256=digest(ROOT/'outputs/experiments/validation_endpoints.csv'),
        xgboost_oof_sha256=digest(ROOT/'outputs/experiments/oof_predictions.csv'),
        tcn_fits_sha256=digest(OUT/'fits.csv'))
    if (target/'protocol.json').exists():assert json.loads((target/'protocol.json').read_text())==plan
    else:write_json(target/'protocol.json',plan)
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    seq.WINDOW=50
    xgb_oof=pd.read_csv(ROOT/'outputs/experiments/oof_predictions.csv')
    xgb_oof=xgb_oof[xgb_oof.candidate.eq(config['name'])]
    predictions=[];calibrations=[]
    cols=['row_index','engine_key','dataset','engine_id','cycle','actual_RUL','protocol','snapshot']
    for fold in [1,2,3]:
        train,outer,valid,fit,stop,stop_points,eval_points=prepare(fold)
        assert fit.engine_key.nunique()==302 and stop.engine_key.nunique()==76
        assert set(fit.engine_key).isdisjoint(stop.engine_key)
        assert set(outer.engine_key).isdisjoint(valid.engine_key)
        inner_model=ExperimentRegressor(config,threads=3).fit(fit)
        assert inner_model.features_.fit_engines_==set(fit.engine_key)
        inner=stop_points.copy()
        inner['predicted_RUL']=pd.Series(inner_model.predict(stop),index=stop.index).loc[inner.row_index].to_numpy()
        joblib.dump(inner_model,local/f'xgb_inner_fold{fold}.joblib')
        inner.to_csv(local/f'xgb_inner_fold{fold}.csv',index=False)
        outer_xgb=xgb_oof[xgb_oof.fold.eq(fold)][cols+['predicted_RUL']].copy()
        # Reuse independently refitted baseline from the regularization comparison.
        transformer,selector,model=joblib.load(ROOT/'outputs/regularization/_runs'/f'XGBoost_f{fold}_s42'/'outer_baseline.joblib')
        assert transformer.fit_engines_==set(outer.engine_key)
        actual=np.maximum(0,model.predict(selector.transform(transformer.transform(valid).loc[eval_points.row_index])))
        np.testing.assert_allclose(actual,outer_xgb.predicted_RUL,rtol=1e-6,atol=1e-4)
        groups=[('XGBoost',42,inner,outer_xgb)]
        proc=seq.SequenceInputs().fit(fit)
        Xstop,_,stop_order=windows(proc,stop,stop_points)
        proc_outer=seq.SequenceInputs().fit(outer)
        Xouter,_,outer_order=windows(proc_outer,valid,eval_points)
        for seed in SEEDS:
            folder=OUT/'_runs'/f'w50_f{fold}_s{seed}'
            inner_tcn=pd.read_csv(folder/'inner_best.csv')
            record=json.loads((folder/'done.json').read_text())[0]
            network=tf.keras.models.load_model(folder/'inner_best.keras')
            np.testing.assert_allclose(seq.predict(network,Xstop)[stop_order],inner_tcn.predicted_RUL,rtol=1e-5,atol=.001)
            network=tf.keras.models.load_model(folder/f'outer_epoch{record["selected_epoch"]}.keras')
            outer_tcn=pd.read_csv(folder/'predictions_60.csv')[cols+['predicted_RUL']]
            np.testing.assert_allclose(seq.predict(network,Xouter)[outer_order],outer_tcn.predicted_RUL,rtol=1e-5,atol=.001)
            groups.append(('TCN',seed,inner_tcn,outer_tcn))
            del network;tf.keras.backend.clear_session()
        for family,seed,inner,heldout in groups:
            pd.testing.assert_frame_equal(inner[cols].reset_index(drop=True),stop_points[cols].reset_index(drop=True),check_dtype=False)
            pd.testing.assert_frame_equal(heldout[cols].reset_index(drop=True),eval_points[cols].reset_index(drop=True),check_dtype=False)
            t=threshold(inner.actual_RUL,inner.predicted_RUL)
            events=inner.actual_RUL.le(30);flags=inner.predicted_RUL.le(t)
            assert (events&flags).sum()/events.sum()>=.90
            below=inner.predicted_RUL.le(np.nextafter(t,-np.inf))
            assert (events&below).sum()/events.sum()<.90
            calibrations.append(dict(model=family,seed=seed,fold=fold,threshold=t,
                calibration_endpoints=len(inner),calibration_events=int(events.sum()),
                calibration_recall=float((events&flags).sum()/events.sum())))
            for rule,cutoff in [('fixed_30',30.),('inner_calibrated',t)]:
                p=heldout.copy();p['model']=family;p['seed']=seed;p['fold']=fold;p['rule']=rule;p['threshold']=cutoff
                p['alert']=p.predicted_RUL.le(cutoff);p['event']=p.actual_RUL.le(30)
                p['TP']=(p.alert&p.event).astype(int);p['FP']=(p.alert&~p.event).astype(int)
                p['FN']=(~p.alert&p.event).astype(int);p['TN']=(~p.alert&~p.event).astype(int)
                predictions.append(p)
        print(f'Calibrated and verified both families on fold {fold}/3',flush=True)
    pred=pd.concat(predictions,ignore_index=True)
    assert len(pred)==27216
    pred.to_csv(target/'oof_predictions.csv',index=False)
    pd.DataFrame(calibrations).to_csv(target/'calibration.csv',index=False)
    def ratios(counts):
        tp,fp,fn,tn=np.moveaxis(np.asarray(counts),-1,0)
        assert np.all(tp+fn>0) and np.all(fp+tn>0) and np.all(tp+fp>0)
        return np.stack([tp/(tp+fn),tp/(tp+fp),fp/(fp+tn)],axis=-1)
    names=['recall','precision','false_positive_rate'];count_cols=['TP','FP','FN','TN']
    totals=[];folds=[];intervals=[];pairs=[]
    keys=['model','seed','protocol','rule']
    for values,g in pred.groupby(keys):
        base=dict(zip(keys,values));counts=g[count_cols].sum().to_numpy()
        assert len(g)==1701 and g.engine_key.nunique()==567
        totals.append(dict(**base,n=len(g),**dict(zip(count_cols,map(int,counts))),**dict(zip(names,ratios(counts)))))
        for fold,f in g.groupby('fold'):
            cc=f[count_cols].sum().to_numpy()
            folds.append(dict(**base,fold=fold,threshold=f.threshold.iloc[0],n=len(f),**dict(zip(count_cols,map(int,cc))),**dict(zip(names,ratios(cc)))))
        by_engine=g.groupby('engine_key')[count_cols].sum()
        rng=np.random.default_rng(42);ix=rng.integers(0,len(by_engine),(2000,len(by_engine)))
        boot=ratios(by_engine.to_numpy()[ix].sum(axis=1))
        for j,name in enumerate(names):
            low,high=np.quantile(boot[:,j],[.025,.975])
            intervals.append(dict(**base,metric=name,estimate=ratios(counts)[j],lower95=low,upper95=high))
        if base['model']=='TCN':
            ref=pred[pred.model.eq('XGBoost')&pred.protocol.eq(base['protocol'])&pred.rule.eq(base['rule'])]
            ref_counts=ref.groupby('engine_key')[count_cols].sum().loc[by_engine.index]
            ref_boot=ratios(ref_counts.to_numpy()[ix].sum(axis=1))
            point=ratios(counts)-ratios(ref_counts.sum().to_numpy())
            for j,name in enumerate(names):
                low,high=np.quantile(boot[:,j]-ref_boot[:,j],[.025,.975])
                pairs.append(dict(seed=base['seed'],protocol=base['protocol'],rule=base['rule'],metric=name,
                    difference=point[j],lower95=low,upper95=high,meaning='TCN minus XGBoost; fractions, not percentage points'))
    pd.DataFrame(totals).to_csv(target/'metrics.csv',index=False)
    pd.DataFrame(folds).to_csv(target/'fold_metrics.csv',index=False)
    pd.DataFrame(intervals).to_csv(target/'intervals.csv',index=False)
    pd.DataFrame(pairs).to_csv(target/'paired_comparisons.csv',index=False)
    assert digest(original)==plan['final_model_sha256']
    write_json(target/'results.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),calibrations=12,
        outer_engines=567,endpoints_per_model_protocol=1701,models_reproduced=True,final_model_unchanged=True))
    print('PASS: identical calibration/evaluation endpoints, minimum inner thresholds, reproduced outer models and unchanged final XGBoost.')


def verify():
    protocol=json.loads((OUT/'protocol.json').read_text())
    predictions=pd.read_csv(OUT/'oof_predictions.csv');metrics=pd.read_csv(OUT/'metrics.csv');fits=pd.read_csv(OUT/'fits.csv')
    assert len(fits)==27 and len(predictions)==30618
    prepared={fold:prepare(fold) for fold in [1,2,3]}
    for row in fits.itertuples():
        run=OUT/'_runs'/f'w{row.window}_f{row.fold}_s{row.seed}'
        inner=pd.read_csv(run/('inner_20.csv' if row.budget==20 else 'inner_best.csv'))
        assert np.isclose(row.threshold,threshold(inner.actual_RUL,inner.predicted_RUL))
        train,outer,valid,fit,stop,stop_points,eval_points=prepared[row.fold]
        assert set(inner.engine_key)==set(stop.engine_key)
        g=predictions[(predictions.window==row.window)&(predictions.budget==row.budget)&(predictions.seed==row.seed)&(predictions.fold==row.fold)]
        cols=['engine_key','row_index','actual_RUL','protocol','snapshot']
        pd.testing.assert_frame_equal(g[cols].reset_index(drop=True),eval_points[cols].reset_index(drop=True),check_dtype=False)
        assert row.fit_engines==378 and row.evaluation_engines==189
        curve=pd.read_csv(run/'curves.csv');curve=curve[curve.epoch<=row.budget]
        best=float('inf');selected=0
        for point in curve.itertuples():
            if point.inner_primary_RMSE<best-.01:best=point.inner_primary_RMSE;selected=point.epoch
        assert selected==row.selected_epoch and np.isclose(best,row.inner_RMSE)
    for row in metrics.itertuples():
        g=predictions[(predictions.window==row.window)&(predictions.budget==row.budget)&(predictions.seed==row.seed)&(predictions.protocol==row.protocol)]
        assert len(g)==1701
        assert np.isclose(np.sqrt(np.mean((g.actual_RUL-g.predicted_RUL)**2)),row.RMSE)
        threshold_values=30 if row.rule=='fixed_30' else g.calibrated_threshold
        flag=g.predicted_RUL<=threshold_values;truth=g.actual_RUL<=30
        assert (flag&truth).sum()==row.TP and (flag&~truth).sum()==row.FP
        assert (~flag&truth).sum()==row.FN and (~flag&~truth).sum()==row.TN
    for window in WINDOWS:
        seq.WINDOW=window
        row=fits[(fits.window==window)&(fits.budget==60)&(fits.fold==1)&(fits.seed==42)].iloc[0]
        run=OUT/'_runs'/f'w{window}_f1_s42'
        train,outer,valid,fit,stop,stop_points,eval_points=prepared[1]
        for which,frame,fit_frame,points,path in [
            ('inner',stop,fit,stop_points,'inner_best.keras'),
            ('outer',valid,outer,eval_points,f'outer_epoch{int(row.selected_epoch)}.keras')]:
            processor=seq.SequenceInputs().fit(fit_frame)
            X,_,order=windows(processor,frame,points)
            model=tf.keras.models.load_model(run/path)
            assert model.count_params()==int(row.parameters)
            actual=seq.predict(model,X)[order]
            expected=pd.read_csv(run/('inner_best.csv' if which=='inner' else 'predictions_60.csv')).predicted_RUL
            np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=.001)
    original=ROOT/'outputs/experiments/selected_model.joblib'
    assert hashlib.sha256(original.read_bytes()).hexdigest()==protocol['original_model_sha256']
    print('PASS: 27 fold results, 30,618 matched endpoints, engine separation, inner-only thresholds, epoch selection, metrics and unchanged final model.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',nargs=3,type=int)
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--diagnostics',action='store_true')
    parser.add_argument('--compare-alerts',action='store_true')
    parser.add_argument('--summarize',action='store_true');args=parser.parse_args()
    if args.worker:worker(*args.worker)
    elif args.verify:verify()
    elif args.diagnostics:diagnostics()
    elif args.compare_alerts:compare_alerts()
    elif args.summarize:
        pred,metrics=summarize()
        assert len(pred)==30618
        paired_comparisons(pred);alert_intervals(pred)
    else:run()
