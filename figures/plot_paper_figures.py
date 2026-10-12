"""Legible single-column figures from audited saved-score calculations (no inference)."""
from pathlib import Path
import json,sys,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT;sys.path.insert(0,str(ROOT))
from experiments.eval.audited_paper_metrics import canonical
D=json.loads((ROOT/'outputs/recomputed-metrics.json').read_text());OUT=ROOT/'figures/out';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42})
COLORS=['#555555','#1873b8','#16845a']
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight',pad_inches=.05)
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.05);plt.close(fig)
def axes(n=1,height=2.7):return plt.subplots(n,1,figsize=(3.25,height),layout='constrained',squeeze=False)
# Fig 1: references first, optional routes second; no guaranteed calibrated VLM endpoint.
fig,ax=plt.subplots(figsize=(3.25,4.0));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
boxes=[(.05,.82,.9,.13,'Physical-design feature maps\nFrozen CNN and candidate selection'),(.05,.56,.9,.15,'CNN score → fitted isotonic map\nPrimary measured reference'),(.05,.3,.9,.15,'Rendered candidate evidence\nOptional frozen VLM readout'),(.02,.05,.46,.13,'Repeated samples\nMean / median'),(.53,.05,.45,.13,'Multi-agent debate\nMatched controls')]
for x,y,w,h,label in boxes:
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.015',facecolor='#f1f5f7',edgecolor='#506478'))
 ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=8)
for start,end in [((.5,.82),(.5,.71)),((.03,.82),(.03,.375)),((.03,.375),(.05,.375)),((.28,.3),(.25,.18)),((.72,.3),(.75,.18))]:ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#506478'})
save(fig,'fig1_system_overview')
# Fig 3: counts support intervals even where score-level AUC records are absent.
# Qwen2.5 3B/7B screens used 60 localization items; the three later models used 30.
fig,aa=axes(2,4.5);ax,bx=aa[:,0];names=['Qwen\n2.5 3B','Qwen\n2.5 7B','Qwen\n2 7B','Llama\n3.2 11B','Gemma\n3 12B'];ks=np.array([2,16,4,4,6]);n=np.array([60,60,30,30,30]);v=ks/n;z=1.95996398454;den=1+z*z/n;c=(v+z*z/(2*n))/den;h=z*np.sqrt(v*(1-v)/n+z*z/(4*n*n))/den
ax.bar(range(5),v,color='#1873b8');ax.errorbar(range(5),v,yerr=[v-(c-h),c+h-v],fmt='none',c='black',capsize=2);ax.axhline(1/9,color='#555',ls='--');ax.set(xticks=range(5),xticklabels=names,ylabel='Localization accuracy',ylim=(0,.5),title='(a) Wilson 95% intervals (n=60,60,30,30,30)')
bx.bar(range(5),[.49,.785,.811,.55,.589],color='#1873b8');bx.axhline(.5,color='#555',ls='--');bx.set(xticks=range(5),xticklabels=names,ylabel='Strong-vs-clean AUC',ylim=(0,1),title='(b) Exploratory point estimates')
save(fig,'fig3_perception_scale')
# Fig4: v4 actually included; clean is a tier, not specificity.
fig,aa=axes(height=2.9);ax=aa[0,0];x=np.arange(4);tiers=['strong[>=0.20]','mid[0.05,0.20)','weak[0.01,0.05)','clean(<0.01)']
for i,(key,label) in enumerate([('phase7_arms_v3graded','v1 directional'),('phase7_arms_v3graded_v3gatemirror','v3 observation-first'),('phase7_arms_v3graded_v4subtlebias','v4 mild caution')]):
 m=D['prompts'][key];v=[m['tiers'][k]['correct']/m['tiers'][k]['n'] for k in tiers];ax.bar(x+(i-1)*.25,v,.24,label=label,color=COLORS[i])
ax.set(xticks=x,xticklabels=['Strong\nrecall','Mid\nrecall','Weak\nrecall','Clean-tier\naccuracy'],ylim=(0,1.4),ylabel='Rate');ax.legend(frameon=False,loc='upper right');save(fig,'fig4_prompt_sensitivity')
# Fig5 historical performance and policy diagnostics.
fig,aa=axes(2,4.3)
for axis,keys,names in [(aa[0,0],['roc','ece','aurc'],['ROC-AUC','ECE','AURC']),(aa[1,0],['abstain','flag'],['Abstention','Flag rate'])]:
 x=np.arange(len(keys))
 for i,arm in enumerate(['A1','A2','A3']):axis.bar(x+(i-1)*.25,[D['historical_debate'][arm][k] for k in keys],.24,label=arm,color=COLORS[i])
 axis.set(xticks=x,xticklabels=names,ylim=(0,1.16));axis.legend(ncol=3,frameon=False)
aa[0,0].set_title('Historical configurations, n=2000');save(fig,'fig5_debate_failure')
# Fig6 ECE.
fig,aa=axes(height=2.8);ax=aa[0,0]
for i,(key,label) in enumerate([('n1_mean','Single'),('n5_median','Median, N=5'),('n5_mean','Mean, N=5')]):ax.bar(np.arange(2)+(i-1)*.25,[D['ensemble'][n][key]['ece'] for n in ('phase7_selfensemble','phase7_selfensemble_traincross')],.24,label=label,color=COLORS[i])
ax.set(xticks=[0,1],xticklabels=['test_N28','train_N28'],ylabel='ECE (10 bins)',ylim=(0,.16));ax.legend(frameon=False,fontsize=7);save(fig,'fig6_ece_calibration')
# Source arrays always first-five, never aggregate columns overwritten by N=10.
sets=[]
for name,label in [('phase7_selfensemble','test_N28'),('phase7_selfensemble_traincross','train_N28')]:
 rows=json.loads((ROOT/'experiments/results'/name/'per_region.json').read_text());y=np.array([r['is_hotspot'] for r in rows]);single=canonical([r['samples'][0] for r in rows]);mean=canonical([sum(r['samples'][:5])/5 for r in rows]);sets.append((label,y,single,mean))
fig,aa=axes(2,4.4)
for ax,(label,y,single,mean) in zip(aa[:,0],sets):
 for s,color,name in [(single,COLORS[0],'Single'),(mean,COLORS[2],'Mean, N=5')]:
  order=np.lexsort((np.arange(len(y)),-np.round(np.abs(s-.5),12)));risk=np.cumsum((s[order]>=.5)!=y[order])/np.arange(1,len(y)+1);ax.plot(np.arange(1,len(y)+1)/len(y),risk,color=color,label=name)
 ax.set(title=label,xlabel='Coverage',ylabel='Risk',xlim=(0,1),ylim=(0,.4));ax.legend(frameon=False)
save(fig,'fig7_risk_coverage')
# Forest plot, no probability/p-value conversion.
fig,aa=axes(2,4.4);items=[]
for name,label in [('phase7_selfensemble','test'),('phase7_selfensemble_traincross','train')]:
 for key,short in [('n5_mean','mean'),('n5_median','median')]:
  base=D['ensemble'][name]['n1_mean'];estimate=D['ensemble'][name][key]
  items.append((label+' '+short,D['ensemble'][name]['bootstrap_n5'][key],{'roc':estimate['roc']-base['roc'],'ece':estimate['ece']-base['ece']}))
for ax,key,title in [(aa[0,0],'roc_ci','ROC-AUC difference'),(aa[1,0],'ece_ci','ECE difference')]:
 metric=key.removesuffix('_ci')
 for i,(label,m,point) in enumerate(items):lo,hi=m[key];ax.plot([lo,hi],[i,i],lw=2,color=COLORS[1]);ax.scatter([point[metric]],[i],s=12,color=COLORS[1])
 ax.axvline(0,color='#555',ls='--');ax.set(yticks=range(4),yticklabels=[x[0] for x in items],xlabel='Ensemble − single',title=title);ax.invert_yaxis()
save(fig,'fig8_bootstrap_forest')
# Reliability with the exact documented bin assignment.
fig,aa=axes(2,4.7)
for ax,(label,y,single,mean) in zip(aa[:,0],sets):
 for s,color,name in [(single,COLORS[0],'Single'),(mean,COLORS[2],'Mean, N=5')]:
  ids=np.minimum(np.floor(np.round(s*10,10)).astype(int),9);pts=[(s[ids==i].mean(),y[ids==i].mean()) for i in range(10) if np.any(ids==i)]
  ax.plot([v[0] for v in pts],[v[1] for v in pts],'o-',ms=3,color=color,label=name)
 ax.plot([0,1],[0,1],'--',color='#999');ax.set(title=label,xlim=(0,1),ylim=(0,1),xlabel='Predicted P(hotspot)',ylabel='Observed hotspot frequency');ax.legend(frameon=False)
save(fig,'fig10_reliability_diagram')
fig,aa=axes(height=2.8);ax=aa[0,0];ns=[1,3,5,10];ax.plot(ns,[D['ensemble']['phase7_selfensemble'][f'n{n}_mean']['roc'] for n in ns],'o-',color=COLORS[2],label='Mean aggregation')
for arm,cost in [('A2',10.268),('A3',7.832)]:ax.scatter([cost],[D['historical_debate'][arm]['roc']],color=COLORS[1]);ax.annotate(arm,(cost,D['historical_debate'][arm]['roc']),xytext=(0,8),textcoords='offset points',ha='center')
# Corrected matched-control debate (temperature 0.7, own prompt): 12 logical calls per branch.
CE=json.loads((ROOT/'experiments/results/corrected_debate_20260922_v3_n2000/evaluation.json').read_text())['configurations']
for cfg,mk in [('H','s'),('X','D')]:
 for br,fc in [('own','white'),('peer',COLORS[0])]:ax.scatter([12],[CE[cfg]['readouts'][f'{br}_fixed']['roc']],marker=mk,facecolor=fc,edgecolor=COLORS[0],s=22,zorder=3,label=f'Corrected {cfg}, {br}')
ax.set(xlabel='Logical calls per candidate',ylabel='ROC-AUC',ylim=(.45,.9),xlim=(.5,13.2));ax.legend(frameon=False,fontsize=6,loc='lower left');save(fig,'fig11_performance_vs_calls')
fig,aa=axes(height=2.7);ax=aa[0,0];x=[150,450,2000];v=[.067,.033,.025];ax.plot(x,v,'o-',color=COLORS[1]);ax.axhline(0,color='#777',ls='--');ax.errorbar([2000],[.025],yerr=[[.030],[.029]],fmt='none',c=COLORS[1],capsize=3);ax.set(xlabel='Approximate sample size per condition',ylabel='Test − train ROC-AUC',ylim=(-.015,.085));ax.text(.03,.03,'95% interval shown for n=2000 only',transform=ax.transAxes,fontsize=6.8);save(fig,'fig9_cross_design_convergence')
print('Regenerated 10 figures; original rendered feature examples retained.')
