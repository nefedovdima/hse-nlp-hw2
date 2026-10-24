"""GPU preflight, real-train memorization and training throughput / OOM probes."""
import argparse
import gc
import json
import time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from hw2_model_v2 import ModelConfig,CodeTransformerV2,parameter_count
from hw2_common_v2 import ROOT,check_cache,environment,seed_all,write_json
from hw2_data_v2 import ArrowTrainStream,pack,collate
from hw2_train_v2 import DEFAULT_TRAIN,make_optimizer,token_weights,train_update
from hw2_eval_v2 import amp_context,evaluate


def preflight(root):
    identity=check_cache(Path(root)/'data/prepared/bpe8192')
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    c=ModelConfig(dim=128,heads=2,ff_dim=344,encoder_layers=2,decoder_layers=2,dropout=0.)
    m=CodeTransformerV2(c).cuda().eval()
    src=torch.tensor([[12,13,1,0],[20,21,22,1]],device='cuda')
    dec=torch.tensor([[3,14,15,16],[3,20,21,22]],device='cuda')
    with torch.inference_mode(),amp_context('cuda',torch.bfloat16):
        full=m(src,dec);cache=m.prepare_cache(src,4)
        cached=torch.cat([m.decode_step(dec[:,i:i+1],i,cache)[0] for i in range(4)],1)
    error=float((full.float()-cached.float()).abs().max())
    assert torch.isfinite(cached).all() and error<.2
    return dict(environment=environment(),cache=identity,cached_max_abs_bf16=error,
        note='BF16 implementation smoke check; strict FP32 invariants are in pytest')


def benchmark(config,cache,budgets=(4096,8192,16384),steps=30):
    cfg=json.loads(Path(config).read_text());c={**DEFAULT_TRAIN,**cfg.get('train',{})}
    tok=Tokenizer.from_file(str(Path(cache)/'tokenizer.json'))
    reports=[]
    for budget in budgets:
        m=opt=weights=stream=None
        try:
            seed_all(710)
            m=CodeTransformerV2(ModelConfig(**cfg['model'])).cuda()
            opt=make_optimizer(m,c,'cuda');weights=token_weights(tok,c['numeric_weight'],'cuda')
            stream=ArrowTrainStream(cache,710,c['pool_size'],budget,c['max_examples'])
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            nt=ns=ne=pad=0;secs=0.;loading=0.
            for i in range(steps+4):
                t0=time.perf_counter();g=stream.next_group();t1=time.perf_counter()
                if g is None:raise RuntimeError('Unexpected end of full train')
                torch.cuda.synchronize();t=time.perf_counter()
                # Disable shrinking in this probe: report whether the requested budget fits.
                from hw2_train_v2 import backward_group
                opt.zero_grad(set_to_none=True)
                loss=backward_group(m,g,'cuda',weights,budget,c['max_examples'])
                torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
                torch.cuda.synchronize();dt=time.perf_counter()-t
                if i>=4:
                    nt+=sum(len(y) for x,y,r in g);ns+=sum(len(x) for x,y,r in g);ne+=len(g)
                    pad+=len(g)*(max(len(x) for x,y,r in g)+max(len(y) for x,y,r in g))
                    secs+=dt;loading+=t1-t0
            reports.append(dict(budget=budget,measured_steps=steps,parameters=parameter_count(m.config),
                source_tokens=ns,target_tokens=nt,examples=ne,compute_seconds=secs,data_seconds=loading,
                target_tokens_per_second=nt/(secs+loading),total_tokens_per_second=(nt+ns)/(secs+loading),
                padding_efficiency=(nt+ns)/pad,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                epoch_compute_hours_estimate=821000000/(nt/(secs+loading))/3600,
                notice='Short throughput sample; epoch estimate excludes evaluation/checkpoint overhead and is not a T4 estimate'))
        except torch.cuda.OutOfMemoryError:
            reports.append(dict(budget=budget,oom=True))
        finally:
            if stream:stream.close()
            del m,opt,weights,stream
            gc.collect();torch.cuda.empty_cache()
        print(json.dumps(reports[-1]),flush=True)
    return dict(environment=environment(),config=cfg,measurements=reports)


def overfit(config,cache,steps=1500):
    cfg=json.loads(Path(config).read_text());c={**DEFAULT_TRAIN,**cfg.get('train',{})}
    seed_all(1234)
    stream=ArrowTrainStream(cache,1234,4096,8192,128)
    examples=[]
    while len(examples)<128:examples.extend(stream.next_group())
    examples=examples[:128];stream.close()
    mc=ModelConfig(**{**cfg['model'],'dropout':0.})
    m=CodeTransformerV2(mc).cuda();c.update(peak_lr=8e-4,weight_decay=0.)
    opt=make_optimizer(m,c,'cuda');tok=Tokenizer.from_file(str(Path(cache)/'tokenizer.json'))
    w=token_weights(tok,1.,'cuda');records=[]
    for step in range(1,steps+1):
        loss,grad,_=train_update(m,opt,examples,'cuda',w,4096,32)
        if step==1 or step%100==0 or step==steps:
            metrics,details=evaluate(m,examples,tok,'cuda',16,512)
            exact=sum(r['prediction']==r['reference'] for r in details)/128
            rec=dict(step=step,loss=loss,exact_code=exact,**metrics);records.append(rec)
            print(json.dumps(rec),flush=True)
            if exact>=.95 and metrics['eos']==128:break
    return dict(rows=[r for x,y,r in examples],seed=1234,records=records,
        passed=records[-1]['exact_code']>=.95 and records[-1]['eos']==128,
        note='128 ORIGINAL TRAIN rows only; diagnostic weights discarded and never reused for V2')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','benchmark','overfit'])
    p.add_argument('--root',default=str(ROOT));p.add_argument('--config',default='configs/v2_main.json')
    p.add_argument('--output',required=True);p.add_argument('--steps',type=int)
    a=p.parse_args();cache=Path(a.root)/'data/prepared/bpe8192'
    if a.mode=='preflight':r=preflight(a.root)
    elif a.mode=='benchmark':r=benchmark(a.config,cache,steps=a.steps or 30)
    else:r=overfit(a.config,cache,steps=a.steps or 1500)
    write_json(a.output,r)
    print(json.dumps(r,indent=2))
