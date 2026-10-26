"""Build the main presentation notebook from verified saved results."""
from pathlib import Path
import json
from textwrap import dedent
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'outputs/experiments/results.json').read_text())
sensors=json.loads((ROOT/'outputs/presentation/sensor_results.json').read_text())
cells=[]
def md(s):
    cells.append(dict(cell_type='markdown',metadata={'slideshow':{'slide_type':'slide'}},source=dedent(s).strip().splitlines(True)))
def code(s):
    cells.append(dict(cell_type='code',metadata={'jupyter':{'source_hidden':True},'slideshow':{'slide_type':'fragment'}},source=dedent(s).strip().splitlines(True),outputs=[],execution_count=None))
md('''
# Predicting when an engine needs maintenance
### NASA C-MAPSS · Applied machine learning

**22.63-cycle test RMSE — 31.7% below our previous best model.**

A controlled study of 709 training engines across four operating/fault scenarios.
The final model uses regime-normalized sensor histories and XGBoost with a
150-cycle training target cap. Evaluation uses the true, uncapped remaining life
of 707 test engines.

**Main finding:** the model improves prediction within the maintenance planning
window. The cap sacrifices accuracy for some engines much earlier in life.

*Presentation guide:* read the sections in order. Code is available for inspection;
training runs and full comparisons are retained in the supporting notebooks.
''')
code('''
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
ROOT=Path.cwd()
if not (ROOT/'outputs/experiments/results.json').exists(): ROOT=ROOT/'nasa-rul'
sys.path.insert(0,str(ROOT/'scripts'))
EXP=ROOT/'outputs/experiments'
ROB=ROOT/'outputs/robustness'
OUT=ROOT/'outputs/presentation'
r=json.loads((EXP/'results.json').read_text())
metrics=pd.read_csv(EXP/'test_metrics.csv')
comparison=pd.read_csv(EXP/'experiment_comparison.csv')
audit=pd.read_csv(EXP/'audit_metrics.csv')
sensor_audit=pd.read_csv(OUT/'sensor_audit.csv')
sensor_results=json.loads((OUT/'sensor_results.json').read_text())
plt.rcParams.update({'figure.figsize':(10,4.8),'font.size':12,'axes.titlesize':15,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'axes.labelcolor':'#25344a','text.color':'#172b4d'})
COLORS={'Original XGBoost':'#637b96','Random Forest':'#9aabc0','Final XGBoost':'#087f8c'}
''')
md('''
## 1. The decision: inspect now, or keep operating?

Estimate **remaining useful life (RUL)** from measurements available at the
current cycle. Then flag engines whose estimated remaining life is short enough
to justify an inspection.

- **Prediction target:** operational cycles remaining until failure.
- **Illustrative maintenance event:** actual RUL ≤30 cycles.
- **Evaluation:** prediction error, missed events and false alerts.

A missed event can delay inspection; a false alert can prompt unnecessary work.
No cost data are provided, so we do not claim financial savings or deployment readiness.
''')
md('''
## 2. The evidence: four datasets, all available engines

Each row contains one engine-cycle observation: three operating settings and
21 sensor measurements. Training histories run to failure; test histories stop
earlier. Engine identifiers are scoped by dataset to prevent histories from mixing.
''')
code('''
data=pd.read_csv(EXP/'data_audit.csv')
table=data.pivot(index='dataset',columns='split',values='engines').reindex(['FD001','FD002','FD003','FD004'])
table=table.rename(columns={'train':'Training engines','test':'Test engines'})
table['Operating conditions']=[1,6,1,6]
table['Fault modes']=[1,1,2,2]
display(table[['Training engines','Test engines','Operating conditions','Fault modes']].rename_axis(index=None,columns=None))
''')
md('''
**160,359 training observations; 104,897 test observations.** Input hashes and engine/cycle keys checked.

**Data checks matter.** The actual FD004 files contain 249 training and 248 test
engines; the bundled README reverses these counts. We use verified file contents.
There are no missing or non-finite measurements in these inputs. Extreme values
are retained because they can reflect operating conditions or degradation.

Source: local NASA C-MAPSS archive, with provenance and hashes in `data/source.json`.
These are simulations, not evidence of performance on a real fleet.
''')
md('''
## 3. A fair comparison: keep engines separate

**567 development engines** → **three-fold engine CV** → **freeze model and alert rule**

Then evaluate **142 untouched audit engines**, refit on all **709 training engines**,
and evaluate the **707 official test engines**.

We compare three synthetic observation points per engine:

- **Primary protocol:** true RUL between 1 and 200 cycles, with at least 20 observed cycles.
- **Broad protocol:** observation points across the engine lifetime, including earlier life.

Every engine contributes equal endpoint weight. Normalization and feature filtering
are learned inside each training fold. Earlier features cannot use future measurements.

**Scope:** the 200-cycle planning window is an explicit assumption. Previous test
results motivated this work, so the official test is not a newly blind benchmark.
''')
md('''
## 4. Are all 21 sensors useful?

**All 21 are read; only 17 sensor channels reach the final fitted model.**

The fitted variance filter removes sensors **1, 5, 18 and 19**, including all their
history features. That leaves **127 inputs**: 119 sensor-derived features plus
cycle, three settings and four dataset indicators.

Sensors **6, 10 and 16** have low variation and low model importance, but they
are not constant across every dataset. Variation is not proof of predictive value;
we tested removal rather than relying on a universal sensor blacklist.
''')
code('''
order=[f'sensor_{i}' for i in range(1,22)]
variation=sensor_audit.pivot(index='dataset',columns='sensor',values='max_within_regime_std')[order]
from matplotlib.colors import ListedColormap
fig,ax=plt.subplots(figsize=(11,3.2))
ax.imshow(variation.gt(1e-6).to_numpy(),cmap=ListedColormap(['#e5eaf0','#087f8c']),vmin=0,vmax=1,aspect='auto')
ax.set(xticks=np.arange(21),xticklabels=np.arange(1,22),yticks=np.arange(4),yticklabels=variation.index,
       xlabel='Sensor number',title='Variation within operating regimes differs by dataset')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#e5eaf0',label='At or below threshold'),Patch(color='#087f8c',label='Above threshold')],
          loc='upper center',bbox_to_anchor=(.5,-.26),ncol=2,frameon=False)
fig.tight_layout()
plt.show()
''')
md('''
The chart uses a within-regime standard deviation threshold of 10⁻⁶ in source
units. It describes variation, **not a ranking of sensor usefulness**.
''')
code('''
sensor_comparison=pd.read_csv(OUT/'sensor_ablation.csv')
display(sensor_comparison.pivot(index='model',columns='protocol',values='RMSE')
        .rename(columns={'primary':'Primary CV RMSE','broad':'Broad CV RMSE'}).rename_axis(index=None,columns=None).round(3))
''')
md('''
**Sensor decision:** dropping 6, 10 and 16 as well gives **25.958** primary CV RMSE,
versus **25.932** with 17 sensors. The difference is tiny and reverses on broad
validation; it does not establish a meaningful advantage for either set.

We retain the existing 17-sensor final model. The 14-sensor comparison used the
same 567 development engines and endpoints; no new official test tuning was done.
A simpler model may be attractive, but this ablation alone does not justify
calling the three sensors useless.
''')
md('''
## 5. What changed in the winning pipeline?

1. **Normalize sensors by operating regime**, using fitting engines only.
2. **Build causal history features:** five- and twenty-cycle means, variability,
   recent slopes, exponentially weighted means and one-cycle changes.
3. **Cap training RUL at 150 cycles**, reducing the need to distinguish very long
   remaining lifetimes. Validation and test labels remain uncapped.
4. **Fit XGBoost:** depth 6, 800 trees, learning rate 0.03.

Cycle remains a feature. Engine weighting, separate dataset models, cycle removal
and asymmetric loss were tested but did not win the declared selection criterion.
The 21-configuration study is a bounded comparison, not an exhaustive search.
''')
code('''
chosen=['reference_raw_uncapped','raw_cap150','rich_cap150','tuned_d6_n800']
labels=['Original recipe','Add 150-cycle target cap','Add normalization + rich history','Tune XGBoost']
primary=comparison[comparison.protocol.eq('primary')].set_index('candidate')
values=primary.loc[chosen,'RMSE'].to_numpy()
fig,ax=plt.subplots(figsize=(10,4))
ax.barh(labels,values,color=['#9aabc0','#637b96','#349aa5','#087f8c'])
ax.invert_yaxis()
ax.set(xlabel='Primary endpoint CV RMSE (cycles)',title='Controlled validation shows where the gains appear',xlim=(0,max(values)*1.2))
for i,v in enumerate(values): ax.text(v+.5,i,f'{v:.2f}',va='center')
fig.tight_layout()
plt.show()
''')
md('''
These bars use the **same development population and endpoint protocol**.
They summarize the combined steps; they do not isolate every interaction between
normalization, history and hyperparameters. The full candidate table is in
[experiments.ipynb](experiments.ipynb).
''')
md('''
## 6. Final test result: lower error across all four datasets

**Test RMSE falls from 33.14 to 22.63 cycles: a 31.7% reduction.**

MAE falls from 23.26 to 16.23 cycles. Mean error drops from +11.94 to +0.31 cycles,
reducing the previous tendency to overestimate remaining life. All results use
the same 707 official endpoints and true uncapped labels.
''')
code('''
fixed=metrics[metrics.threshold.eq(30)]
selected=fixed[fixed.model.eq('selected') & fixed.dataset.eq('ALL')].iloc[0]
prior=fixed[fixed.model.eq('Random forest') & fixed.dataset.eq('ALL')].iloc[0]
assert np.isclose(selected.RMSE,22.6336838386522)
assert np.isclose(prior.RMSE,33.143425)
lookup={'Random forest':'Random Forest','XGBoost':'Original XGBoost','selected':'Final XGBoost'}
chart=fixed[fixed.dataset.ne('ALL') & fixed.model.isin(lookup)].copy()
chart['Model']=chart.model.map(lookup)
chart=chart.pivot(index='dataset',columns='Model',values='RMSE')[['Random Forest','Original XGBoost','Final XGBoost']]
ax=chart.plot.bar(rot=0,color=[COLORS[c] for c in chart],figsize=(10,5))
ax.set(xlabel='Dataset',ylabel='RMSE (cycles)',title='Official test error by dataset')
ax.legend(title=None,loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3,frameon=False)
plt.tight_layout()
plt.show()
display(chart.rename_axis(index=None,columns=None).round(2))
''')
md('''
**Interpretation:** the combined pipeline improves every test subset, with the
largest absolute reduction in FD003. This table cannot attribute the gain to one
sensor, one feature or normalization alone.

The final errors are similar within operating-condition groups: single-condition
FD001/FD003 achieve 17.84/18.28 cycles, versus 24.47/23.91 for six-condition
FD002/FD004. The original errors instead separated more strongly by fault-mode
count. This four-dataset pattern is descriptive; it does not establish causation.
''')
md('''
## 7. Maintenance alerts: make the precision–recall tradeoff explicit

The event remains **actual RUL ≤30 cycles**. The alert threshold applies to the
model's prediction, so changing it does not alter predicted RUL or RMSE.

A threshold of **34.44 predicted cycles** was chosen from out-of-fold development
predictions to target at least 90% recall. On the official test it catches **149 of
159 near-failure engines**, with **12 false alerts**.
''')
code('''
alerts=metrics[metrics.model.isin(['XGBoost','selected']) & metrics.dataset.eq('ALL')].copy()
alerts['Model']=alerts.model.map({'XGBoost':'Original XGBoost','selected':'Final XGBoost'})
alerts=alerts.sort_values(['model','threshold'])
view=alerts[['Model','threshold','precision','recall','TP','FN','FP']].rename(columns={
 'threshold':'Alert threshold','precision':'Precision','recall':'Recall','TP':'Caught','FN':'Missed','FP':'False alerts'})
view['Precision']=view.Precision.map(lambda x:f'{x:.1%}')
view['Recall']=view.Recall.map(lambda x:f'{x:.1%}')
view['Alert threshold']=view['Alert threshold'].round(2)
display(view.set_index('Model').rename_axis(index=None,columns=None))
''')
md('''
At threshold 30, the final model catches 143 engines and raises eight false alerts.
Calibration catches six additional engines at the cost of four additional false
alerts. Audit-primary recall was 90%; test recall is 93.7%. Neither is a guarantee
for another fleet or operating policy.

NASA's asymmetric error score is also reported in the experiment tables. It
penalizes overestimation more heavily, but does not replace inspection costs or
a validated maintenance policy.
''')
md('''
## 8. The result has a boundary

The untouched 142-engine audit confirms a benefit in the planning window:
**33.01 → 25.67 cycles RMSE**. A paired engine-bootstrap 95% interval puts the
improvement at **4.34–10.59 cycles**.

**Earlier-life-inclusive performance worsens: 39.80 → 43.32 cycles RMSE.**
The training cap encourages predictions to level off near 150 cycles, so the
model should not be presented as a precise estimator of much longer remaining life.
''')
code('''
shown=audit[audit.threshold.eq(30)].pivot(index='protocol',columns='model',values='RMSE')
shown=shown.reindex(['primary','broad']).rename(index={'primary':'Planning window','broad':'Earlier life included'},columns={'reference':'Original recipe','selected':'Final recipe'})
ax=shown.plot.bar(rot=0,color=['#637b96','#087f8c'],figsize=(10,4.5))
ax.set(xlabel='',ylabel='Audit RMSE (cycles)',title='Untouched engines reveal both benefit and tradeoff')
ax.legend(title=None,frameon=False)
plt.tight_layout()
plt.show()
''')
code('''
pred=pd.read_csv(EXP/'audit_predictions.csv')
broad=pred[pred.protocol.eq('broad')]
fig,axes=plt.subplots(1,2,figsize=(11,4.7),sharex=True,sharey=True)
limit=np.ceil(max(broad.actual_RUL.max(),broad.predicted_RUL.max())/50)*50
for ax,model,label,color in zip(axes,['reference','selected'],['Original recipe','Final recipe'],['#637b96','#087f8c']):
    points=broad[broad.model.eq(model)]
    ax.scatter(points.actual_RUL,points.predicted_RUL,s=16,alpha=.5,color=color)
    ax.plot([0,limit],[0,limit],color='#52657b',linestyle='--',label='Perfect prediction')
    if model=='selected': ax.axhline(150,color='#b46c29',linestyle=':',label='Training target cap')
    ax.set(xlim=(0,limit),ylim=(0,limit),xlabel='True RUL (cycles)',title=label)
    ax.legend(fontsize=9,frameon=False)
axes[0].set_ylabel('Predicted RUL (cycles)')
fig.suptitle('Same broad-audit endpoints: the long-horizon boundary')
fig.tight_layout()
plt.show()
''')
md('''
The dotted line marks the training target cap, not a hard prediction constraint.
Both panels use the same 426 endpoints from the original 142-engine audit.

## 9. Does the gain survive different splits and windows?

We retrained both frozen recipes locally on **five stratified engine splits**:
567 fitting engines and 142 held-out engines per split. Each scenario uses three
fixed observation points per engine, with at least 20 observed cycles.
Model settings and the calibrated threshold stayed fixed.

**All five splits improve for each 100–250-cycle planning window.** Broad lifetime
coverage worsens in four of five splits. For the 200-cycle window, RMSE improves
by **8.31–12.14 cycles** across these splits.

These are sensitivity checks, not five independent blind replications: held-out
sets overlap, and some engines informed the earlier model selection. New endpoint
sampling also means these numbers need not equal the original audit.
''')
code('''
checks=pd.read_csv(ROB/'split_metrics.csv')
gains=checks.pivot(index=['seed','scenario'],columns='model',values='RMSE')
gains['gain']=gains['Original recipe']-gains['Final recipe']
order=['RUL_1_100','RUL_1_150','RUL_1_200','RUL_1_250','broad']
summary=gains.groupby('scenario')['gain'].agg(['min','mean','max']).reindex(order)
fig,ax=plt.subplots(figsize=(10,4.5))
x=np.arange(len(order))
ax.errorbar(x,summary['mean'],yerr=[summary['mean']-summary['min'],summary['max']-summary['mean']],fmt='o',capsize=6,color='#087f8c',label='Mean and min–max across five splits')
for seed,group in gains.reset_index().groupby('seed'):
    ax.scatter(x,group.set_index('scenario').loc[order,'gain'],s=18,color='#637b96',alpha=.6)
ax.axhline(0,color='#52657b',linewidth=1)
ax.set(xticks=x,xticklabels=['1–100','1–150','1–200','1–250','Broad life'],xlabel='True RUL sampling window (cycles)',ylabel='Original RMSE − final RMSE (cycles)',title='Positive values favor the final recipe')
ax.legend(frameon=False,fontsize=10)
fig.tight_layout()
plt.show()
''')
md('''
Bars show a descriptive range across splits, **not a confidence interval**.
The broad scenario samples observed cycles uniformly across each available life.

## 10. How uncertain are the alert results?

The same 707 official test endpoints support this comparison. Intervals are
95% Wilson intervals for recall among actual events and precision among alerts.
They describe sampling uncertainty conditional on the fixed models and rule;
they exclude selection uncertainty, refitting and changes in fleet conditions.
''')
code('''
intervals=pd.read_csv(ROB/'test_alert_intervals.csv')
shown=intervals[intervals.dataset.eq('ALL')].copy()
shown['Estimate [95% interval]']=shown.apply(lambda row:f'{row.estimate:.1%} [{row.lower95:.1%}, {row.upper95:.1%}]',axis=1)
shown['Policy']=shown.model+' / '+shown.threshold.map(lambda value:f'{value:.2f}')
display(shown.pivot(index='Policy',columns='metric',values='Estimate [95% interval]').rename_axis(index=None,columns=None))
''')
md('''
Calibrated test recall is **93.7% [88.8%, 96.5%]**, and precision is
**92.5% [87.4%, 95.7%]**. The data do not guarantee at least 90% recall.

Separately, on the original audit-primary population, calibrated recall is
**90.0% [83.3%, 95.6%]** and precision **89.0% [82.2%, 95.1%]** using 2,000
whole-engine bootstrap draws. That resamples engines with all three endpoints
together; these audit results are not interchangeable with the official test.

## 11. When does the first alert arrive?

We replayed full held-out engine histories in each of the five splits, monitoring
from **cycle 20 to the cycle before failure**. A first alert at 32 cycles remaining
is slightly ahead of the illustrative 30-cycle window; it should not be equated
with a warning more than 60 cycles early.
''')
code('''
trajectory=pd.read_csv(ROB/'trajectory_summary.csv')
trajectory['Policy']=trajectory.model+' / '+trajectory.threshold.map(lambda value:f'{value:.2f}')
columns=['timely_first_alert_fraction','first_alert_31_60_fraction','first_alert_over60_fraction','no_alert_fraction']
shares=trajectory.groupby('Policy')[columns].mean()
shares.columns=['1–30 cycles','31–60 cycles','More than 60 cycles','No alert before failure']
ax=(shares*100).plot.barh(stacked=True,color=['#087f8c','#9aabc0','#b46c29','#e5eaf0'],figsize=(11,4.2))
ax.set(xlabel='Mean share of held-out engines across five splits (%)',ylabel='',xlim=(0,100),title='First-alert lead time: calibration moves warnings earlier')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.23),ncol=2,frameon=False,fontsize=10)
plt.tight_layout()
plt.show()
timing=trajectory.groupby('Policy').median_first_alert_lead.agg(['min','max'])
timing.columns=['Minimum split median (cycles)','Maximum split median (cycles)']
display(timing.rename_axis(index=None,columns=None))
''')
md('''
The calibrated rule's median first alert arrives **32–34 cycles before failure**
across splits, versus **27.5–30 cycles** for the final model at threshold 30.
All held-out histories have at least one alert within their last 30 cycles before
failure. This repeated opportunity to alert differs from one-endpoint recall;
it is not a reliability guarantee.

This is a **passive replay**: an alert does not trigger a modeled inspection or
change the subsequent trajectory. Repeated warnings and inspection costs still
need an intervention policy. Engines recur across splits, so the chart averages
split proportions rather than claiming 710 independent engines.

## 12. What we can defend—and what comes next

**Supported:** lower planning-window error across the tested splits and horizons,
an explicit sensor ablation, uncertainty estimates for snapshot alerts, and a
measured first-alert tradeoff on simulated held-out histories.

**Not established:** a globally optimal model, real-fleet reliability or savings,
accurate long-horizon RUL, a guaranteed recall target, or a meaningful performance
difference between the 14- and 17-sensor variants.

The next substantive step is external validation and an explicit inspection
policy with costs and intervention outcomes. A new external dataset would
provide stronger evidence than further tuning on this already-observed benchmark.
''')
md('''
## Appendix · Evidence and reproduction

- [Baseline notebook](baseline.ipynb): data exploration and original model comparison.
- [Experiment notebook](experiments.ipynb): all 21 configurations, audit and test results.
- [Methods](outputs/experiments/METHODS.md): endpoint sampling, selection and limitations.
- [Robustness methods](outputs/robustness/METHODS.md), [split metrics](outputs/robustness/split_metrics.csv),
  [alert intervals](outputs/robustness/test_alert_intervals.csv) and
  [first-alert events](outputs/robustness/engine_alert_events.csv).
- [Sensor audit](outputs/presentation/sensor_audit.csv) and
  [sensor ablation](outputs/presentation/sensor_ablation.csv): all channels and matched CV results.
- [Reproduction guide](docs/REPRODUCING.md): regenerate the trained model and
  large prediction exports, which are excluded from Git. Compact report inputs remain included.

This notebook reads saved results; opening or executing it does not train models.
All model fitting was performed locally. Training scripts and timestamped logs
remain in the project. See [README.md](README.md) for reproduction commands.

Method reference: [Saxena et al., NASA scoring function, equation 9](https://ntrs.nasa.gov/api/citations/20090029214/downloads/20090029214.pdf).

Course structure: business understanding → data checks and EDA → feature engineering
→ modeling and validation → evaluation and decision thresholds.
''')
for i,c in enumerate(cells):c['id']=f'presentation-{i:03d}'
nb={'cells':cells,'nbformat':4,'nbformat_minor':5,'metadata':{
 'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
 'language_info':{'name':'python','version':'3.14'},'artifact_dir':'outputs/presentation','presentation':True}}
(ROOT/'explore.ipynb').write_text(json.dumps(nb,indent=1)+'\n')
print('Wrote presentation-ready explore.ipynb')
