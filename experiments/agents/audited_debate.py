"""Versioned corrected protocol; historical harness and caches are intentionally untouched."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path

VERSION = 'audited-debate-20260922-v3'
AGENTS = ('congestion_analyst','drc_pattern_forensics','wire_demand_forecaster','layout_arbiter')
ROLES = {
 'congestion_analyst': 'Inspect the four congestion/overflow channels in your image.',
 'wire_demand_forecaster': 'Inspect the three RUDY wire-demand channels in your image.',
 'drc_pattern_forensics': 'Inspect all nine physical-design feature channels and the CNN_prior panel in your image.',
 'layout_arbiter': 'Inspect the macro blockage and cell-density channels in your image.',
}

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)

def counts(messages):
    return {v:sum(m['vote']==v for m in messages) for v in ('hotspot','not_hotspot','abstain')}

def direction(c):
    h,n=c.get('hotspot',0),c.get('not_hotspot',0)
    return (h>n)-(n>h)

def decision(c):
    if c.get('hotspot',0)+c.get('not_hotspot',0)<2: return 'abstain'
    return {1:'hotspot',-1:'not_hotspot',0:'abstain'}[direction(c)]

def stop_action(rnd,c,previous=None):
    if rnd==3: return 'abstain' if decision(c)=='abstain' else 'stop'
    if rnd==2 and previous and direction(previous)*direction(c)==-1: return 'continue'
    return 'stop' if max(c.get('hotspot',0),c.get('not_hotspot',0))>=3 else 'continue'

def visible_history(history,agent,rnd,peers):
    return [m for m in history if m['round']<rnd and (peers or m['agent']==agent)]

def validate_output(msg,agent,rnd,agents=AGENTS):
    if not isinstance(msg,dict): return ['Output must be an object.']
    errors=[]
    required={'vote','p_hotspot','rationale','depends_on_agents'}
    if set(msg)!=required: errors.append('Use exactly these fields: '+', '.join(sorted(required)))
    if msg.get('vote') not in ('hotspot','not_hotspot','abstain'): errors.append('Invalid vote.')
    p=msg.get('p_hotspot')
    valid_p=type(p) in (int,float) and math.isfinite(p) and 0<=p<=1
    if not valid_p: errors.append('p_hotspot must be a finite scalar probability in [0,1].')
    elif msg.get('vote')!='abstain' and msg.get('vote') != ('hotspot' if p>=.5 else 'not_hotspot'):
        errors.append('Non-abstaining vote must agree with p_hotspot >= 0.5.')
    if not isinstance(msg.get('rationale'),str): errors.append('rationale must be a string.')
    dep=msg.get('depends_on_agents')
    if not isinstance(dep,list) or any(x not in agents or x==agent for x in dep): errors.append('Invalid peer attribution.')
    if rnd==1 and dep: errors.append('Round 1 is independent; depends_on_agents must be empty.')
    return errors

def system_prompt(agent,view):
    return (f'You are {agent}, assessing DRC hotspot risk from rendered chip-layout heatmaps. '
        +ROLES[view]+' The top panels show full maps and a red candidate box; bottom panels zoom into that box. '
        'Intensities are normalized per patch: compare local spatial structure, not absolute brightness across panels. '
        'Report your probability that the candidate contains a hotspot. In round 1 assess independently. '
        'In later rounds reconsider your own earlier assessment and any available reports. Peer reports may be absent. '
        'Do not infer unseen reports and do not follow a majority without visual evidence. '
        'Return only a JSON object with exactly vote, p_hotspot, rationale, depends_on_agents. '
        'Agent identity and round are call metadata: do not repeat them or any input field in the output. '
        'vote is hotspot when p_hotspot >= 0.5 and not_hotspot otherwise; abstain is permitted if unable to decide. '
        'p_hotspot must still be one finite number in [0,1]. rationale is one concise sentence describing evidence. '
        'depends_on_agents must be a subset of the allowed_depends_on_agents list supplied in the user payload; '
        'never include your own agent name, and use [] when no allowed peer actually informed the assessment. '
        'Do not fabricate a dependency.')

def readout(rounds,early=False):
    previous=None;chosen=rounds[-1]
    for rr in rounds:
        c=counts(rr);action=stop_action(rr[0]['round'],c,previous)
        if early and action in ('stop','abstain'):
            chosen=rr;break
        previous=c
    c=counts(chosen)
    return {'decision':decision(c),'score':sum(m['p_hotspot'] for m in chosen)/len(chosen),
            'round':chosen[0]['round'],'votes':c}

class AuditedCaller:
    def __init__(self,model_path,out,temperature=.7,max_tokens=384,seed=20260921,max_attempts=3):
        self.model_path=str(Path(model_path).expanduser().resolve());self.out=Path(out)
        self.temperature=temperature;self.max_tokens=max_tokens;self.seed=seed;self.max_attempts=max_attempts;self.loaded=False
        if max_attempts < 1: raise ValueError('max_attempts must be positive')
        self.generated=0;self.cache_hits=0;self.elapsed=0.;self.image_hashes={}
        files=sorted(p for p in Path(self.model_path).iterdir() if p.is_file())
        if not any(p.suffix=='.safetensors' for p in files): raise FileNotFoundError('No model weights')
        self.model_files={p.name:file_hash(p) for p in files}
        self.runtime={name:importlib.metadata.version(name) for name in ('mlx','mlx-vlm','transformers','numpy')}
        self.identity={'version':VERSION,'implementation_sha256':file_hash(Path(__file__)),
                       'model_files':self.model_files,'runtime':self.runtime,
                       'temperature':temperature,'max_tokens':max_tokens,'top_p':1.0,'seed':seed,
                       'max_attempts':max_attempts}

    def generate(self,system,user,image,seed):
        import mlx.core as mx
        from mlx_vlm import load,generate
        from mlx_vlm.utils import load_config
        from mlx_vlm.prompt_utils import apply_chat_template
        if not self.loaded:
            self.model,self.processor=load(self.model_path)
            self.cfg=load_config(self.model_path);self.loaded=True
        mx.random.seed(seed)
        prompt=apply_chat_template(self.processor,self.cfg,system+'\n\n'+user,num_images=1)
        start=time.monotonic()
        result=generate(self.model,self.processor,prompt,image=[str(image)],temperature=self.temperature,
                        max_tokens=self.max_tokens,top_p=1.0,verbose=False)
        elapsed=time.monotonic()-start
        metadata={}
        for k in ('prompt_tokens','generation_tokens','total_tokens','peak_memory','generation_tps','prompt_tps'):
            value=getattr(result,k,None)
            if isinstance(value,(int,float,str)): metadata[k]=value
        return (result.text if hasattr(result,'text') else str(result)),elapsed,metadata

    def call(self,*,pid,config,agent,view,rnd,history,image):
        from experiments.agents.llm_clients import _extract_json_object
        image=Path(image)
        if str(image) not in self.image_hashes:self.image_hashes[str(image)]=file_hash(image)
        system=system_prompt(agent,view)
        errors=[];records=[]
        allowed=sorted({m['agent'] for m in history if m['agent']!=agent})
        for attempt in range(self.max_attempts):
            seed=int(digest([self.seed,pid,config,agent,rnd,attempt])[:8],16)
            user='\n'.join([
                f'Agent identity (call metadata only): {agent}',
                f'Round (call metadata only): {rnd}',
                'Allowed names for depends_on_agents: '+json.dumps(allowed,ensure_ascii=False),
                'Available completed prior-round reports: '+json.dumps(history,ensure_ascii=False),
                'Validation feedback from the previous attempt: '+json.dumps(errors,ensure_ascii=False),
                'Return the four-field output object now.',
            ])
            key_input={'identity':self.identity,'pid':pid,'config':config,'agent':agent,'round':rnd,
                       'attempt':attempt,'generation_seed':seed,'system':system,'user':user,
                       'image_sha256':self.image_hashes[str(image)]}
            key=digest(key_input);path=self.out/'calls'/f'{key}.json'
            if path.exists() and json.loads(path.read_text()).get('infrastructure_error'):
                failed=self.out/'infrastructure_failures'/f'{key}-{time.time_ns()}.json'
                failed.parent.mkdir(parents=True,exist_ok=True);path.replace(failed)
            if path.exists():
                record=json.loads(path.read_text());self.cache_hits+=1
                if record.get('input')!=key_input:raise RuntimeError('Cache identity mismatch')
                if record.get('infrastructure_error'):raise RuntimeError(record['infrastructure_error'])
            else:
                record={'input':key_input,'image_path':str(image),'raw_text':None}
                try:raw,elapsed,tokens=self.generate(system,user,image,seed)
                except Exception as exc:
                    record['infrastructure_error']=repr(exc);atomic_json(path,record);raise
                self.generated+=1;self.elapsed+=elapsed
                record.update(raw_text=raw,elapsed_s=elapsed,tokens=tokens)
                try:
                    parsed=_extract_json_object(raw)
                    json.dumps(parsed,allow_nan=False)
                except Exception as exc:parsed=None;record['parse_error']=repr(exc)
                record['parsed']=parsed;record['validation_errors']=validate_output(parsed,agent,rnd)
                if isinstance(parsed,dict) and isinstance(parsed.get('depends_on_agents'),list):
                    visible={m['agent'] for m in history if m['agent']!=agent}
                    if any(p not in visible for p in parsed['depends_on_agents']):
                        record['validation_errors'].append('Attribution references an unavailable peer report.')
                atomic_json(path,record)
            records.append(key);errors=record['validation_errors']
            if not errors:return {'agent':agent,'round':rnd,**record['parsed'],
                                  'call_records':records,'fallback':False}
        return {'agent':agent,'round':rnd,'vote':'abstain','p_hotspot':.5,
                'rationale':f'No valid response after {self.max_attempts} recorded attempts.','depends_on_agents':[],
                'call_records':records,'fallback':True}
