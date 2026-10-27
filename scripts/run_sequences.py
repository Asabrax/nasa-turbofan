"""Compare compact sequence models on the existing development-engine folds."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/nasa-matplotlib')
import argparse,hashlib,json,time,gc,subprocess
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from experiment_models import HistoryFeatures,regime_keys
from train_all import ROOT,SENSORS,SETTINGS,DATASETS,load_inputs
from run_experiments import score

OUT=ROOT/'outputs/sequences'
WINDOW=30
SEEDS=[42,73]

class SequenceInputs:
    """Training-only regime statistics, channel filtering and metadata scaling."""
    def fit(self,frame):
        self.norm=HistoryFeatures(normalize=True).fit(frame)
        values=self.sensor_values(frame)
        self.sensors=np.array(SENSORS)[values.var(axis=0)>1e-8].tolist()
        self.metadata=StandardScaler().fit(frame[['cycle']+SETTINGS])
        self.fit_engines=set(frame.engine_key)
        return self

    def sensor_values(self,frame):
        keys=regime_keys(frame)
        if set(keys)-set(self.norm.means_.index):raise ValueError('Unseen operating regime')
        return (frame[SENSORS].to_numpy()-self.norm.means_.loc[keys].to_numpy())/self.norm.scales_.loc[keys].to_numpy()

    def transform(self,frame):
        values=self.sensor_values(frame)[:,[SENSORS.index(s) for s in self.sensors]]
        meta=self.metadata.transform(frame[['cycle']+SETTINGS])
        datasets=np.column_stack([frame.dataset.eq(d) for d in DATASETS])
        # Explicit observation indicator distinguishes left padding from real zeros.
        return np.column_stack([values,meta,datasets,np.ones(len(frame))]).astype('float32')

    def windows(self,frame,rows=None):
        """Each window ends at its prediction cycle; no future rows or engine mixing."""
        frame=frame.sort_values(['engine_key','cycle'])
        values=self.transform(frame)
        pieces=[];indices=[]
        for positions in frame.groupby('engine_key',sort=False).indices.values():
            block=values[positions]
            padded=np.pad(block,((WINDOW-1,0),(0,0)))
            window=np.lib.stride_tricks.sliding_window_view(padded,WINDOW,axis=0).transpose(0,2,1)
            idx=frame.index.to_numpy()[positions]
            keep=np.ones(len(idx),bool) if rows is None else np.isin(idx,rows)
            pieces.append(window[keep]);indices.extend(idx[keep])
        X=np.concatenate(pieces).astype('float32')
        index=np.asarray(indices)
        assert np.isfinite(X).all()
        return X,index


def network(kind,features,seed):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    inputs=tf.keras.Input((WINDOW,features))
    if kind=='LSTM':
        x=tf.keras.layers.LSTM(32)(inputs)
    else:
        x=tf.keras.layers.Conv1D(32,1)(inputs)
        for dilation in [1,2,4,8]:
            skip=x
            for _ in range(2):
                x=tf.keras.layers.Conv1D(32,3,padding='causal',dilation_rate=dilation,activation='relu')(x)
                x=tf.keras.layers.Dropout(.1)(x)
            x=tf.keras.layers.Add()([x,skip])
        x=x[:,-1,:]
    x=tf.keras.layers.Dense(16,activation='relu')(x)
    x=tf.keras.layers.Dropout(.1)(x)
    outputs=tf.keras.layers.Dense(1)(x)
    model=tf.keras.Model(inputs,outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001,clipnorm=1.),loss='mse')
    return model


def dataset(X,y,seed):
    options=tf.data.Options()
    options.threading.private_threadpool_size=2
    options.threading.max_intra_op_parallelism=2
    return (tf.data.Dataset.from_tensor_slices((X,y)).shuffle(len(y),seed=seed)
            .batch(512).with_options(options).prefetch(1))


def predict(model,X):
    return np.maximum(0,np.concatenate([model(X[i:i+512],training=False).numpy().ravel() for i in range(0,len(X),512)])*150.)


def run(epochs=20):
    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'training.log').write_text('')
    def log(message):
        line=f'{datetime.now(timezone.utc).isoformat(timespec="seconds")} {message}'
        print(line,flush=True)
        with (OUT/'training.log').open('a') as f:f.write(line+'\n')
    start=time.perf_counter()
    frozen=ROOT/'outputs/experiments/selected_model.joblib'
    original_hash=hashlib.sha256(frozen.read_bytes()).hexdigest() if frozen.exists() else None
    plan=dict(window=WINDOW,seeds=SEEDS,architectures=['LSTM','TCN'],max_epochs=epochs,patience=4,
              batch_size=512,target_cap=150,learning_rate=.001,inner_fraction=.2,
              selection='Inner primary-endpoint uncapped RMSE; refit all outer fitting engines for selected epoch count',
              evaluation='Existing development folds and endpoints only; no audit or official test evaluation',
              original_model_sha256=original_hash,tensorflow=tf.__version__)
    (OUT/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    (OUT/'environment.txt').write_text(subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True))
    train,_,_=load_inputs()
    assignments=pd.read_csv(ROOT/'outputs/experiments/cv_engine_folds.csv')
    endpoints=pd.read_csv(ROOT/'outputs/experiments/validation_endpoints.csv')
    dev=train[train.engine_key.isin(assignments.engine_key)]
    curves=[];predictions=[];fits=[];members=[]
    for fold in [1,2,3]:
        fit_keys=set(assignments.loc[assignments.fold.ne(fold),'engine_key'])
        eval_keys=set(assignments.loc[assignments.fold.eq(fold),'engine_key'])
        outer=dev[dev.engine_key.isin(fit_keys)]
        valid=dev[dev.engine_key.isin(eval_keys)]
        engines=outer[['engine_key','dataset']].drop_duplicates().sort_values('engine_key')
        inner_fit,inner_stop=train_test_split(engines.engine_key,test_size=.2,random_state=100+fold,stratify=engines.dataset)
        assert set(inner_fit).isdisjoint(inner_stop) and fit_keys.isdisjoint(eval_keys)
        for key in sorted(fit_keys|eval_keys):
            role='evaluation' if key in eval_keys else ('early_stopping' if key in set(inner_stop) else 'inner_fit')
            members.append(dict(fold=fold,engine_key=key,role=role))
        pd.DataFrame(members).to_csv(OUT/'engine_roles.csv',index=False)
        fit=outer[outer.engine_key.isin(inner_fit)]
        stop=outer[outer.engine_key.isin(inner_stop)]
        stop_points=endpoints[endpoints.engine_key.isin(inner_stop)&endpoints.protocol.eq('primary')]
        eval_points=endpoints[endpoints.engine_key.isin(eval_keys)]
        processor=SequenceInputs().fit(fit)
        Xfit,ifit=processor.windows(fit)
        yfit=np.minimum(train.loc[ifit,'RUL'].to_numpy(),150).astype('float32')/150
        Xstop,istop=processor.windows(stop,stop_points.row_index)
        stop_lookup=pd.Series(np.arange(len(istop)),index=istop)
        # Use the saved endpoint order, including any overlapping protocol endpoints.
        stop_order=stop_lookup.loc[stop_points.row_index].to_numpy()
        outer_processor=SequenceInputs().fit(outer)
        Xouter,iouter=outer_processor.windows(outer)
        youter=np.minimum(train.loc[iouter,'RUL'].to_numpy(),150).astype('float32')/150
        Xeval,ieval=outer_processor.windows(valid,eval_points.row_index)
        eval_order=pd.Series(np.arange(len(ieval)),index=ieval).loc[eval_points.row_index].to_numpy()
        # Prefix invariance, target isolation and engine isolation on each dataset.
        for d in DATASETS:
            key=valid.loc[valid.dataset.eq(d),'engine_key'].iloc[0]
            prefix=valid[valid.engine_key.eq(key)].iloc[:23].copy()
            row=prefix.index[-1]
            A,_=outer_processor.windows(valid[valid.engine_key.eq(key)],[row])
            prefix['RUL']=-12345
            B,_=outer_processor.windows(prefix,[row])
            np.testing.assert_array_equal(A,B)
            np.testing.assert_allclose(B[0,-1],outer_processor.transform(prefix)[-1])
            assert np.all(B[0,:WINDOW-len(prefix)]==0)
        for kind in ['LSTM','TCN']:
            for seed in SEEDS:
                t0=time.perf_counter();best=np.inf;best_epoch=1;wait=0
                model=network(kind,Xfit.shape[2],seed)
                params=model.count_params()
                log(f'Fold {fold}, {kind}, seed {seed}: early stopping; {len(inner_fit)} fit / {len(inner_stop)} stop / {len(eval_keys)} evaluation engines; {params} parameters.')
                ds=dataset(Xfit,yfit,seed)
                for epoch in range(1,epochs+1):
                    result=model.fit(ds,epochs=1,verbose=0,shuffle=False)
                    pred=predict(model,Xstop)[stop_order]
                    rmse=score(stop_points.actual_RUL,pred)['RMSE']
                    curves.append(dict(fold=fold,model=kind,seed=seed,epoch=epoch,training_capped_RMSE=float(np.sqrt(result.history['loss'][0])*150),inner_primary_RMSE=rmse))
                    pd.DataFrame(curves).to_csv(OUT/'learning_curves.csv',index=False)
                    if rmse<best-.01:best=rmse;best_epoch=epoch;wait=0
                    else:wait+=1
                    if epoch%5==0 or wait>=4:log(f'Fold {fold}, {kind}, seed {seed}: epoch {epoch}, inner RMSE={rmse:.2f}, best epoch={best_epoch}.')
                    if wait>=4:break
                inner_seconds=time.perf_counter()-t0
                del model,ds;gc.collect()
                log(f'Fold {fold}, {kind}, seed {seed}: refitting all {len(fit_keys)} outer fitting engines for {best_epoch} epochs.')
                model=network(kind,Xouter.shape[2],seed)
                ds=dataset(Xouter,youter,seed)
                model.fit(ds,epochs=best_epoch,verbose=0,shuffle=False)
                pred=predict(model,Xeval)[eval_order]
                record=eval_points.copy()
                record['predicted_RUL']=pred
                record['model']=kind;record['seed']=seed;record['fold']=fold
                predictions.append(record)
                fits.append(dict(fold=fold,model=kind,seed=seed,selected_epochs=best_epoch,epochs_observed=epoch,inner_RMSE=best,parameters=model.count_params(),retained_sensors=','.join(outer_processor.sensors),inner_seconds=inner_seconds,total_seconds=time.perf_counter()-t0,fit_engines=len(fit_keys),evaluation_engines=len(eval_keys),training_rows=len(iouter)))
                pd.DataFrame(fits).to_csv(OUT/'fits.csv',index=False)
                allpred=pd.concat(predictions,ignore_index=True)
                allpred.to_csv(OUT/'oof_predictions.csv',index=False)
                model.save(OUT/f'{kind.lower()}_fold{fold}_seed{seed}.keras')
                primary=record[record.protocol.eq('primary')]
                log(f'Fold {fold}, {kind}, seed {seed}: outer primary RMSE={score(primary.actual_RUL,primary.predicted_RUL)["RMSE"]:.2f}; {time.perf_counter()-t0:.1f}s.')
                del model,ds;gc.collect()
        del Xfit,Xstop,Xouter,Xeval;gc.collect()
    allpred=pd.concat(predictions,ignore_index=True)
    rows=[]
    for (kind,seed,protocol),g in allpred.groupby(['model','seed','protocol']):
        rows.append(dict(model=kind,seed=seed,protocol=protocol,**score(g.actual_RUL,g.predicted_RUL)))
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    baseline=pd.read_csv(ROOT/'outputs/experiments/experiment_comparison.csv')
    baseline[baseline.candidate.eq('tuned_d6_n800')].to_csv(OUT/'xgboost_reference.csv',index=False)
    if original_hash:assert hashlib.sha256(frozen.read_bytes()).hexdigest()==original_hash
    (OUT/'results.json').write_text(json.dumps(dict(protocol=plan,elapsed_seconds=time.perf_counter()-start,metrics=rows,completed_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
    log(f'COMPLETE in {time.perf_counter()-start:.1f}s; saved final XGBoost unchanged.')

def diagnostics():
    """Compare uncapped training/validation errors on the same endpoint protocol."""
    train,_,_=load_inputs()
    folds=pd.read_csv(ROOT/'outputs/experiments/cv_engine_folds.csv')
    points=pd.read_csv(ROOT/'outputs/experiments/validation_endpoints.csv')
    oof=pd.read_csv(OUT/'oof_predictions.csv')
    rows=[]
    for fold in [1,2,3]:
        fit_keys=set(folds.loc[folds.fold.ne(fold),'engine_key'])
        fit=train[train.engine_key.isin(fit_keys)]
        endpoints=points[points.engine_key.isin(fit_keys)&points.protocol.eq('primary')]
        processor=SequenceInputs().fit(fit)
        X,index=processor.windows(fit,endpoints.row_index)
        order=pd.Series(np.arange(len(index)),index=index).loc[endpoints.row_index].to_numpy()
        for kind in ['LSTM','TCN']:
            for seed in SEEDS:
                model=tf.keras.models.load_model(OUT/f'{kind.lower()}_fold{fold}_seed{seed}.keras')
                training=score(endpoints.actual_RUL,predict(model,X)[order])['RMSE']
                valid=oof[(oof.fold==fold)&(oof.model==kind)&(oof.seed==seed)&oof.protocol.eq('primary')]
                validation=score(valid.actual_RUL,valid.predicted_RUL)['RMSE']
                rows.append(dict(model=kind,seed=seed,fold=fold,training_primary_RMSE=training,validation_primary_RMSE=validation,gap=validation-training))
                del model
                tf.keras.backend.clear_session()
    pd.DataFrame(rows).to_csv(OUT/'generalization_gaps.csv',index=False)
    baseline_path=ROOT/'outputs/experiments/oof_predictions.csv'
    if baseline_path.exists():
        baseline=pd.read_csv(baseline_path)
        baseline=baseline[baseline.candidate.eq('tuned_d6_n800')]
        comparisons=[]
        keys=['engine_key','protocol','snapshot','row_index']
        for (kind,seed,protocol),group in oof.groupby(['model','seed','protocol']):
            paired=group.merge(baseline[baseline.protocol.eq(protocol)][keys+['predicted_RUL']],on=keys,suffixes=('_sequence','_xgboost'),validate='one_to_one')
            assert len(paired)==len(group)
            paired['sequence_squared']=(paired.predicted_RUL_sequence-paired.actual_RUL)**2
            paired['xgboost_squared']=(paired.predicted_RUL_xgboost-paired.actual_RUL)**2
            per_engine=paired.groupby('engine_key')[['sequence_squared','xgboost_squared']].mean().to_numpy()
            rng=np.random.default_rng(42)
            draws=rng.integers(0,len(per_engine),(2000,len(per_engine)))
            rmse=np.sqrt(per_engine[draws].mean(axis=1))
            differences=rmse[:,0]-rmse[:,1]
            estimate=np.sqrt(per_engine.mean(axis=0))
            low,high=np.quantile(differences,[.025,.975])
            comparisons.append(dict(model=kind,seed=seed,protocol=protocol,RMSE_difference=estimate[0]-estimate[1],lower95=low,upper95=high,engines=len(per_engine),bootstrap_draws=2000))
        pd.DataFrame(comparisons).to_csv(OUT/'paired_comparison.csv',index=False)
    print('Saved matched-protocol errors and available paired comparisons.')


def verify():
    """Recompute metrics and check split roles, endpoints and model reloads."""
    plan=json.loads((OUT/'protocol.json').read_text())
    pred=pd.read_csv(OUT/'oof_predictions.csv')
    roles=pd.read_csv(OUT/'engine_roles.csv')
    fits=pd.read_csv(OUT/'fits.csv')
    curves=pd.read_csv(OUT/'learning_curves.csv')
    metrics=pd.read_csv(OUT/'metrics.csv')
    folds=pd.read_csv(ROOT/'outputs/experiments/cv_engine_folds.csv')
    endpoints=pd.read_csv(ROOT/'outputs/experiments/validation_endpoints.csv')
    train,_,_=load_inputs()
    assert len(fits)==12 and len(pred)==13608
    for fold,g in roles.groupby('fold'):
        assert len(g)==567 and g.engine_key.nunique()==567
        assert g.role.value_counts().to_dict()=={'inner_fit':302,'evaluation':189,'early_stopping':76}
        expected=set(folds.loc[folds.fold.eq(fold),'engine_key'])
        assert set(g.loc[g.role.eq('evaluation'),'engine_key'])==expected
    for (model,seed,fold),g in pred.groupby(['model','seed','fold']):
        keys=set(folds.loc[folds.fold.eq(fold),'engine_key'])
        e=endpoints[endpoints.engine_key.isin(keys)]
        columns=['engine_key','row_index','protocol','snapshot','actual_RUL']
        pd.testing.assert_frame_equal(g[columns].reset_index(drop=True),e[columns].reset_index(drop=True),check_dtype=False)
        assert len(g)==1134 and np.isfinite(g.predicted_RUL).all()
    for row in metrics.itertuples():
        g=pred[(pred.model==row.model)&(pred.seed==row.seed)&(pred.protocol==row.protocol)]
        for key,value in score(g.actual_RUL,g.predicted_RUL).items():
            assert np.isclose(value,getattr(row,key)),(row.model,key)
    for row in fits.itertuples():
        c=curves[(curves.fold==row.fold)&(curves.model==row.model)&(curves.seed==row.seed)]
        best=np.inf;epoch=0
        for point in c.itertuples():
            if point.inner_primary_RMSE<best-.01:best=point.inner_primary_RMSE;epoch=point.epoch
        assert row.selected_epochs==epoch and np.isclose(row.inner_RMSE,best)
        assert row.fit_engines==378 and row.evaluation_engines==189
    fit_keys=set(folds.loc[folds.fold.ne(1),'engine_key'])
    valid_keys=set(folds.loc[folds.fold.eq(1),'engine_key'])
    processor=SequenceInputs().fit(train[train.engine_key.isin(fit_keys)])
    valid=train[train.engine_key.isin(valid_keys)]
    points=endpoints[endpoints.engine_key.isin(valid_keys)]
    X,index=processor.windows(valid,points.row_index)
    order=pd.Series(np.arange(len(index)),index=index).loc[points.row_index].to_numpy()
    for kind in ['LSTM','TCN']:
        model=tf.keras.models.load_model(OUT/f'{kind.lower()}_fold1_seed42.keras')
        expected=pred[(pred.model==kind)&(pred.seed==42)&(pred.fold==1)].predicted_RUL
        np.testing.assert_allclose(predict(model,X)[order],expected,rtol=1e-5,atol=.001)
    original=ROOT/'outputs/experiments/selected_model.joblib'
    if plan['original_model_sha256'] and original.exists():
        assert hashlib.sha256(original.read_bytes()).hexdigest()==plan['original_model_sha256']
    print('PASS: 12 fits, 13,608 matched endpoints, engine roles, epoch selection, metrics, both architecture reloads and unchanged final XGBoost.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--diagnostics',action='store_true')
    args=parser.parse_args()
    if args.epochs<1:parser.error('--epochs must be positive')
    if args.verify:verify()
    elif args.diagnostics:diagnostics()
    else:
        run(args.epochs)
        diagnostics()
