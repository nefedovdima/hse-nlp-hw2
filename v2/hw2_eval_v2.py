"""Same frozen validation rows, same numeric-reference denominator for every model."""
import argparse
import ast
import json
import math
import time
import gc
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from hw2_model_v2 import CodeTransformerV2, ModelConfig, PAD, EOS
from hw2_data_v2 import load_validation
from hw2_common_v2 import ROOT, sha256, TOKENIZER_SHA, write_json, environment
from hw2_safe_v2 import run_numeric, same_answer


def load_model(path, device='cuda', e1=False):
    state=torch.load(path,map_location='cpu',weights_only=False)  # only own trusted checkpoints
    if e1:
        from e1.model import CodeTransformer, ModelConfig as E1Config
        model=CodeTransformer(E1Config(**state['model_config']))
    else:
        if state.get('tokenizer_sha256',state.get('cache_identity',{}).get('tokenizer')) != TOKENIZER_SHA:
            raise ValueError('Checkpoint tokenizer identity missing or incorrect')
        model=CodeTransformerV2(ModelConfig(**state['model_config']))
    model.load_state_dict(state['model'],strict=True)
    metadata={k:state.get(k) for k in ('step','target_tokens','source_tokens','examples','model_config')}
    return model.to(device).eval(),metadata


def amp_context(device, dtype=None):
    if str(device).startswith('cuda'):
        dtype=dtype or (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        return torch.autocast('cuda',dtype=dtype)
    return nullcontext()


@torch.inference_mode()
def predict(model, rows, tokenizer, device='cuda', batch_size=32, max_new_tokens=384, amp=True):
    """Rows are (source, target, row_id); only source participates in generation."""
    model.eval()
    indexed=sorted(enumerate(rows),key=lambda p:len(p[1][0]))
    results=[None]*len(rows)
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    start=time.perf_counter()
    with amp_context(device) if amp else nullcontext():
        for i in range(0,len(rows),batch_size):
            group=indexed[i:i+batch_size]
            src=torch.full((len(group),max(len(r[0]) for _,r in group)),PAD,dtype=torch.long)
            for j,(_,r) in enumerate(group):
                src[j,:len(r[0])]=torch.tensor(r[0])
            out=model.generate(src.to(device),max_new_tokens=max_new_tokens).cpu().tolist()
            for (original,r),ids in zip(group,out):
                eos=EOS in ids
                ids=ids[:ids.index(EOS)] if eos else ids
                results[original]=dict(row_id=r[2],code=tokenizer.decode(ids,skip_special_tokens=True),eos=eos)
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    return results,time.perf_counter()-start


def score(rows,predictions,tokenizer):
    details=[]
    counts=dict(total=len(rows),reference_numeric=0,matches=0,eos=0,syntax_ok=0,
                prediction_numeric=0,reference_unsupported=0,reference_nonnumeric=0,
                reference_runtime_error=0,reference_syntax_error=0)
    for (x,y,rid),p in zip(rows,predictions):
        if p['row_id'] != rid:
            raise ValueError('Prediction ordering mismatch')
        ref=tokenizer.decode(y[:-1],skip_special_tokens=True)
        rs,rv=run_numeric(ref)
        ps,pv=run_numeric(p['code'])
        match=rs=='ok' and ps=='ok' and same_answer(pv,rv)
        try:
            ast.parse(p['code']); syntax=True
        except (SyntaxError,RecursionError):
            syntax=False
        counts['reference_numeric']+=rs=='ok'
        counts['prediction_numeric']+=ps=='ok'
        if rs!='ok':
            counts['reference_'+rs]+=1
        counts['matches']+=int(match)
        counts['eos']+=p['eos']
        counts['syntax_ok']+=syntax
        details.append(dict(row_id=rid,reference_status=rs,prediction_status=ps,
            reference_answer=rv,prediction_answer=pv,match=bool(match),eos=p['eos'],
            syntax_ok=syntax,question=tokenizer.decode(x[:-1]),reference=ref,prediction=p['code']))
    n=counts['reference_numeric']
    counts['agreement_numeric_references']=counts['matches']/n if n else None
    counts['agreement_all_lower_bound']=counts['matches']/len(rows)
    counts['reference_numeric_coverage']=n/len(rows)
    counts['metric_notice']='Restricted AST agreement to original labels; NOT official pass@1 or mathematical truth'
    return counts,details


def evaluate(model,rows,tokenizer,device='cuda',batch_size=32,max_new_tokens=384,amp=True):
    was_training=model.training
    while True:
        try:
            predictions,seconds=predict(model,rows,tokenizer,device,batch_size,max_new_tokens,amp=amp)
            break
        except torch.cuda.OutOfMemoryError:
            if batch_size==1:
                if was_training:model.train()
                raise
            batch_size=max(1,batch_size//2)
            gc.collect();torch.cuda.empty_cache()
            print(f'Validation OOM: retry all SAME rows with batch_size={batch_size}',flush=True)
    metrics,details=score(rows,predictions,tokenizer)
    metrics['generation_seconds']=seconds
    metrics['effective_batch_size']=batch_size
    if was_training:
        model.train()
    return metrics,details


def paired_compare(a,b,seed=42,reps=5000):
    """B-A paired bootstrap on a fixed ref-supported subset; not independent samples."""
    aa={r['row_id']:r for r in a}; bb={r['row_id']:r for r in b}
    if aa.keys()!=bb.keys():
        raise ValueError('Different row IDs')
    for k in aa:
        if (aa[k]['reference_status'],aa[k]['reference_answer']) != (bb[k]['reference_status'],bb[k]['reference_answer']):
            raise ValueError('Reference evaluator/labels differ')
    ids=[k for k in sorted(aa) if aa[k]['reference_status']=='ok']
    delta=np.array([int(bb[k]['match'])-int(aa[k]['match']) for k in ids],dtype=np.float64)
    if not len(delta):
        raise ValueError('No supported references')
    rng=np.random.default_rng(seed)
    means=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(reps)])
    return dict(n=len(delta),delta=float(delta.mean()),ci95=[float(v) for v in np.quantile(means,[.025,.975])],
        gained=int((delta==1).sum()),lost=int((delta==-1).sum()),seed=seed,
        caveat='Row bootstrap; correlated paraphrases may make the interval too narrow')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--validation',default=str(ROOT/'experiments/v2/validation4096.json'))
    p.add_argument('--tokenizer',default=str(ROOT/'data/prepared/bpe8192/tokenizer.json'))
    p.add_argument('--role',choices=['dev','confirm','all'],default='dev')
    p.add_argument('--e1',action='store_true')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--max-new-tokens',type=int,default=384)
    p.add_argument('--device',default='cuda')
    p.add_argument('--precision',choices=['amp','fp16','fp32'],default='amp')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if sha256(a.tokenizer)!=TOKENIZER_SHA:
        raise ValueError('Tokenizer hash mismatch')
    model,meta=load_model(a.checkpoint,a.device,a.e1)
    if a.precision=='fp16':model.half()
    tok=Tokenizer.from_file(a.tokenizer)
    metrics,details=evaluate(model,load_validation(a.validation,a.role),tok,a.device,a.batch_size,a.max_new_tokens,
                            amp=a.precision=='amp')
    metrics.update(meta)
    metrics.update(validation_sha256=sha256(a.validation),checkpoint_sha256=sha256(a.checkpoint),
        role=a.role,max_new_tokens=a.max_new_tokens,precision=a.precision,environment=environment())
    out=Path(a.output)
    write_json(out,metrics)
    write_json(out.with_suffix('.details.json'),details)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    main()
