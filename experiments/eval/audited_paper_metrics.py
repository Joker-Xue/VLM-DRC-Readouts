"""Recompute stored evidence with explicit, stable score precision; no inference."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.agents.audited_debate import atomic_json,file_hash
from experiments.eval.metrics import expected_calibration_error as historical_ece
RES=ROOT/'experiments/results'

def canonical(scores):return np.round(np.asarray(scores,dtype=float),12)

def ece(scores,y,bins=10):
    s=canonical(scores);y=np.asarray(y,dtype=float)
    if not len(s):return float('nan')
    if not np.isfinite(s).all() or (s<0).any() or (s>1).any():raise ValueError('Invalid probability')
    ids=np.minimum(np.floor(np.round(s*bins,10)).astype(int),bins-1)
    cnt=np.bincount(ids,minlength=bins);psum=np.bincount(ids,weights=s,minlength=bins);ysum=np.bincount(ids,weights=y,minlength=bins)
    return float(np.abs(psum-ysum).sum()/len(s))

def metrics(scores,y,decisions=None):
    s=canonical(scores);y=np.asarray(y,dtype=int);pred=s>=.5
    order=np.lexsort((np.arange(len(y)),-np.round(np.abs(s-.5),12)))
    risk=np.cumsum(pred[order]!=y[order])/np.arange(1,len(y)+1)
    clipped=np.clip(s,1e-9,1-1e-9)
    out={'n':len(y),'positives':int(y.sum()),'roc':float(roc_auc_score(y,s)),
         'ap':float(average_precision_score(y,s)),'ece':ece(s,y),
         'ece_sensitivity':{str(b):ece(s,y,b) for b in (5,10,15,20)},
         'brier':float(np.mean((s-y)**2)),'nll':float(np.mean(-y*np.log(clipped)-(1-y)*np.log(1-clipped))),
         'aurc':float(risk.mean()),'r80':float(risk[int(np.ceil(.8*len(y)))-1]),
         'score_hit':float(np.mean(pred==y)),'score_flag':float(pred.mean())}
    if decisions is not None:
        d=np.array(decisions);binary=d=='hotspot'
        out.update(hit=float(np.mean(binary==y)),flag=float(binary.mean()),abstain=float(np.mean(d=='abstain')),
                   decision_score_disagreements=int(np.sum(binary!=pred)))
    return out

def bootstrap(a,b,y,reps=3000):
    a,b=canonical(a),canonical(b);y=np.asarray(y,dtype=int);rng=np.random.default_rng(42)
    deltas=[]
    for _ in range(reps):
        ix=rng.integers(0,len(y),len(y));yy=y[ix]
        if len(np.unique(yy))<2:continue
        def area(scores):
            order=np.lexsort((np.arange(len(yy)),-np.round(np.abs(scores-.5),12)))
            return np.mean(np.cumsum((scores[order]>=.5)!=yy[order])/np.arange(1,len(yy)+1))
        deltas.append([roc_auc_score(yy,b[ix])-roc_auc_score(yy,a[ix]),ece(b[ix],yy)-ece(a[ix],yy),area(b[ix])-area(a[ix])])
    values=np.asarray(deltas)
    return {'replicates':len(values),'seed':42,'difference':'B minus A','roc_ci':np.quantile(values[:,0],[.025,.975]).tolist(),
            'ece_ci':np.quantile(values[:,1],[.025,.975]).tolist(),'aurc_ci':np.quantile(values[:,2],[.025,.975]).tolist(),'roc_positive_count':int(np.sum(values[:,0]>0)),
            'ece_negative_count':int(np.sum(values[:,1]<0)),'interpretation':'Nominal region percentile intervals; directional counts are not p-values.'}

def load(name,inputs):
    if 'INVALID' in name:raise ValueError('Invalid run is excluded')
    p=RES/name/'per_region.json';inputs[name]=file_hash(p);return json.loads(p.read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--bootstrap',type=int,default=0);a=ap.parse_args()
    inputs={};out={'definition':{'score_round_decimals':12,'ece':'left-closed equal-width bins; p=1 in final bin','nll_eps':1e-9,'risk':'score >= .5; confidence abs(score-.5); stable input-index ties'},'inputs':inputs}
    base=load('phase7_arms_v3gm_n2000_v2',inputs);y=[r['is_hotspot'] for r in base]
    out['historical_debate']={arm:metrics([r['arms'][arm]['score'] for r in base],y,[r['arms'][arm]['decision'] for r in base]) for arm in ('R0','A1','A2','A3')}
    if a.bootstrap:
        out['historical_debate_bootstrap']={arm:bootstrap([r['arms']['A1']['score'] for r in base],[r['arms'][arm]['score'] for r in base],y,a.bootstrap) for arm in ('A2','A3')}
    out['committed_matched']={}
    for arm in ('A2','A3'):
        keep=[r for r in base if r['arms'][arm]['decision']!='abstain'];yy=[r['is_hotspot'] for r in keep]
        out['committed_matched'][arm]={k:metrics([r['arms'][k]['score'] for r in keep],yy,[r['arms'][k]['decision'] for r in keep]) for k in ('A1',arm)}
    out['ensemble']={}
    for name in ('phase7_selfensemble','phase7_selfensemble_traincross'):
        rows=load(name,inputs);yy=[r['is_hotspot'] for r in rows];scores={};report={}
        for n in (1,3,5,10):
            for kind in ('mean','median'):
                key=f'n{n}_{kind}';ss=[sum(r['samples'][:n])/n if kind=='mean' else float(np.median(r['samples'][:n])) for r in rows]
                scores[key]=ss;report[key]=metrics(ss,yy)
                report[key]['historical_unrounded_ece']=historical_ece(ss,yy)
        if a.bootstrap:
            report['bootstrap_n5']={k:bootstrap(scores['n1_mean'],scores[k],yy,a.bootstrap) for k in ('n5_mean','n5_median')}
        out['ensemble'][name]=report;atomic_json(a.out,out);print('Recomputed '+name,flush=True)
    ab=load('phase7_ablation_no_cnn_prior',inputs)
    assert [(r['patch_id'],r['is_hotspot']) for r in base]==[(r['patch_id'],r['is_hotspot']) for r in ab]
    aa=[r['arms']['A1']['score'] for r in base];bb=[r['arms']['A1']['score'] for r in ab]
    out['no_prior']={'with':metrics(aa,y),'without':metrics(bb,y)}
    if a.bootstrap:out['no_prior']['bootstrap']=bootstrap(aa,bb,y,a.bootstrap)
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import StratifiedKFold
    from experiments.eval.run_arms import load_records,stratified_sample
    out['cnn_calibration']={}
    for selection in ('cnn-argmax','label-argmax'):
        sets={}
        for split in ('test_N28','train_N28'):
            recs=stratified_sample(load_records(split,selection),2000,42)
            sets[split]=(np.array([r['cnn_prior']['peak'] for r in recs]),np.array([r['is_hotspot'] for r in recs]))
            suffix='_cnnargmax' if selection=='cnn-argmax' else ''
            inputs[f'{split}_{selection}_evidence']=file_hash(ROOT/'data/evidence_cache'/f'{split}{suffix}_evidence.jsonl')
        report={}
        for target,(scores,yy) in sets.items():
            source='train_N28' if target=='test_N28' else 'test_N28'
            ss,ys=sets[source]
            calibrated=IsotonicRegression(out_of_bounds='clip').fit(ss,ys).predict(scores)
            report[target]={'raw':metrics(scores,yy),'transferred':metrics(calibrated,yy),'source':source,
                            'fit':'All 2000 source-condition labels; refitted separately for each selection regime.'}
        out['cnn_calibration'][selection]=report
    def scaled(scores,t):
        scores=np.clip(np.asarray(scores),1e-6,1-1e-6)
        return 1/(1+np.exp(-np.log(scores/(1-scores))/t))
    def fit_t(scores,yy):
        options=np.linspace(.05,10,2000)
        losses=[]
        for t in options:
            pr=np.clip(scaled(scores,t),1e-9,1-1e-9)
            losses.append(np.mean(-yy*np.log(pr)-(1-yy)*np.log(1-pr)))
        return float(options[np.argmin(losses)])
    sets={}
    for name in ('phase7_selfensemble','phase7_selfensemble_traincross'):
        rows=load(name,inputs);sets[name]=(np.array([r['samples'][0] for r in rows]),np.array([r['is_hotspot'] for r in rows]))
    out['vlm_temperature']={}
    for target,(scores,yy) in sets.items():
        source=next(k for k in sets if k!=target);ss,ys=sets[source];t=fit_t(ss,ys)
        out['vlm_temperature'][target]={'temperature':t,'source':source,'metrics':metrics(scaled(scores,t),yy)}
    out['prompts']={}
    for name in ('phase7_arms_v3graded','phase7_arms_v3graded_v3gatemirror','phase7_arms_v3graded_v4subtlebias'):
        rows=load(name,inputs);yy=[r['is_hotspot'] for r in rows];ss=[r['arms']['A1']['score'] for r in rows]
        z=metrics(ss,yy,[r['arms']['A1']['decision'] for r in rows])
        z['tiers']={}
        for tier in sorted({r['bucket'] for r in rows}):
            sub=[r for r in rows if r['bucket']==tier]
            z['tiers'][tier]={'n':len(sub),'correct':sum((r['arms']['A1']['decision']=='hotspot')==bool(r['is_hotspot']) for r in sub)}
        if a.bootstrap:
            ci=bootstrap(np.full(len(yy),.5),ss,yy,a.bootstrap)['roc_ci'];z['roc_ci']=[v+.5 for v in ci]
        out['prompts'][name]=z
    out['sample_size_history']={}
    for name in ('phase7_arms_v3graded_v3gatemirror','phase7_arms_v3gm_n450'):
        rows=load(name,inputs);yy=[r['is_hotspot'] for r in rows]
        out['sample_size_history'][name]={arm:metrics([r['arms'][arm]['score'] for r in rows],yy,[r['arms'][arm]['decision'] for r in rows]) for arm in ('A1','A2','A3')}
    atomic_json(a.out,out);print('Saved '+str(a.out),flush=True)

if __name__=='__main__':main()
