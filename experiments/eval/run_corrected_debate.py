"""Corrected fixed-budget communication controls. Never reads historical generation caches."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.agents.audited_debate import (AGENTS,VERSION,AuditedCaller,atomic_json,digest,
    file_hash,readout,system_prompt,visible_history)

FIELDS=('agent','round','vote','p_hotspot','rationale','depends_on_agents')

def clean_message(m): return {k:m[k] for k in FIELDS}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sample',type=int,default=2)
    ap.add_argument('--configs',nargs='+',choices=['H','X'],default=['H','X'])
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--model',default=str(Path('~/.cache/modelscope/hub/models/mlx-community/Qwen2___5-VL-7B-Instruct-4bit').expanduser()))
    ap.add_argument('--temperature',type=float,default=.7)
    ap.add_argument('--max-tokens',type=int,default=384)
    ap.add_argument('--max-attempts',type=int,default=3)
    ap.add_argument('--seed',type=int,default=20260921)
    ap.add_argument('--source',type=Path,default=ROOT/'experiments/results/phase7_arms_v3gm_n2000_v2/per_region.json')
    ap.add_argument('--render-dir',type=Path,default=ROOT/'data/render_cache')
    ap.add_argument('--preflight-only',action='store_true')
    a=ap.parse_args()
    rows=json.loads(a.source.read_text())[:a.sample]
    if len(rows)!=a.sample or a.sample<=0:raise ValueError('Invalid requested sample size')
    missing=[]
    for row in rows:
        for config in a.configs:
            for agent in AGENTS:
                view='drc_pattern_forensics' if config=='H' else agent
                img=a.render_dir/row['patch_id']/f'{view}.png'
                if not img.is_file():missing.append(str(img))
    if missing:raise FileNotFoundError(str(missing[:10]))
    print(f'Preflight passed: {len(rows)} candidates, configurations {a.configs}',flush=True)
    if a.preflight_only:return
    caller=AuditedCaller(a.model,a.out,a.temperature,a.max_tokens,a.seed,a.max_attempts)
    image_manifest={str(a.render_dir/r['patch_id']/f'{v}.png'):file_hash(a.render_dir/r['patch_id']/f'{v}.png')
                    for r in rows for v in (AGENTS if 'X' in a.configs else ['drc_pattern_forensics'])}
    manifest={'runner_sha256':file_hash(Path(__file__)),'images':image_manifest,'version':VERSION,'source_sha256':file_hash(a.source),'candidate_ids':[r['patch_id'] for r in rows],
              'configurations':a.configs,'caller':caller.identity,'fixed_rounds':3,'max_attempts':a.max_attempts,
              'prompts':{c:{agent:system_prompt(agent,'drc_pattern_forensics' if c=='H' else agent) for agent in AGENTS} for c in a.configs}}
    mp=a.out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise RuntimeError('Run manifest mismatch; use a new output directory')
    atomic_json(mp,manifest)
    start=time.monotonic();results=[]
    for index,row in enumerate(rows,1):
        pid=row['patch_id'];result_path=a.out/'regions'/f'{pid}.json'
        if result_path.exists():results.append(json.loads(result_path.read_text()));continue
        result={k:row[k] for k in ('patch_id','is_hotspot','label_strength','bucket') if k in row}
        result['configurations']={}
        for config in a.configs:
            def get(agent,rnd,history):
                view='drc_pattern_forensics' if config=='H' else agent
                return caller.call(pid=pid,config=config,agent=agent,view=view,rnd=rnd,history=history,
                                   image=a.render_dir/pid/f'{view}.png')
            r1=[get(agent,1,[]) for agent in AGENTS]
            branches={}
            for name,peers in [('own',False),('peer',True)]:
                rounds=[r1];history=[clean_message(m) for m in r1]
                for rnd in (2,3):
                    current=[get(agent,rnd,visible_history(history,agent,rnd,peers)) for agent in AGENTS]
                    rounds.append(current);history.extend(clean_message(m) for m in current)
                branches[name]={'rounds':rounds,'fixed':readout(rounds),'early':readout(rounds,early=True)}
            first_broad=next(m for m in r1 if m['agent']=='drc_pattern_forensics')
            result['configurations'][config]={'r1':readout([r1]),'single':{'score':first_broad['p_hotspot'],'decision':first_broad['vote']},'branches':branches}
        atomic_json(result_path,result);results.append(result)
        atomic_json(a.out/'per_region.json',results)
        progress={'completed':index,'total':len(rows),'generated_calls_this_process':caller.generated,
                  'cache_hits_this_process':caller.cache_hits,'generation_elapsed_s_this_process':caller.elapsed,
                  'wall_elapsed_s_this_process':time.monotonic()-start}
        atomic_json(a.out/'progress.json',progress)
        print(json.dumps(progress),flush=True)
    atomic_json(a.out/'per_region.json',results)
    print('Completed '+str(len(results))+' regions.',flush=True)

if __name__=='__main__':main()
