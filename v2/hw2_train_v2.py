"""Token-based schedule; full train; atomic resumable checkpoints; dev semantic selection."""
import argparse
import gc
import json
import math
import re
import time
from dataclasses import asdict
from pathlib import Path
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from hw2_model_v2 import CodeTransformerV2, ModelConfig, PAD, parameter_count
from hw2_data_v2 import ArrowTrainStream, collate, pack, load_validation
from hw2_common_v2 import (ROOT, TOKENIZER_SHA, sha256, check_cache, write_json, atomic_save,
    seed_all, rng_state, restore_rng, safe_run_dir, log, environment)
from hw2_eval_v2 import amp_context, evaluate


DEFAULT_TRAIN=dict(seed=42,epochs=1,pool_size=32768,logical_budget=32768,max_examples=256,
    micro_budget=8192,peak_lr=3e-4,warmup_target_tokens=20000000,schedule_target_tokens=821000000,
    min_lr_ratio=.1,weight_decay=.01,numeric_weight=1.,clip=1.,
    checkpoint_every=500,eval_every_tokens=100000000,max_new_tokens=384,eval_batch_size=32)


def lr_at(tokens,c):
    warm=c['warmup_target_tokens']
    if tokens<warm:
        return c['peak_lr']*max(tokens,1)/warm
    x=min(1.,(tokens-warm)/max(1,c['schedule_target_tokens']-warm))
    return c['peak_lr']*(c['min_lr_ratio']+(1-c['min_lr_ratio'])*.5*(1+math.cos(math.pi*x)))


def token_weights(tokenizer,weight,device):
    # Lexical numeric-looking BPE tokens only. Original labels remain unchanged.
    w=torch.ones(tokenizer.get_vocab_size(),device=device)
    if weight!=1.:
        for i in range(4,len(w)):
            if re.fullmatch(r'\s*\d+(?:\.\d+)?\s*',tokenizer.decode([i])):
                w[i]=weight
    w[PAD]=0
    return w


def backward_group(model,group,device,weights,micro_budget,max_examples):
    all_labels=torch.tensor([t for x,y,r in group for t in y],dtype=torch.long)
    denominator=float(weights.detach().cpu()[all_labels].sum())
    total_loss=0.
    for micro in pack(group,micro_budget,max_examples):
        src,dec,y,_=collate(micro)
        if str(device).startswith('cuda'):
            src,dec,y=(z.pin_memory().to(device,non_blocking=True) for z in (src,dec,y))
        else:
            src,dec,y=(z.to(device) for z in (src,dec,y))
        with amp_context(device,torch.bfloat16):
            logits=model(src,dec)
            loss=F.cross_entropy(logits.float().flatten(0,1),y.flatten(),
                weight=weights,ignore_index=PAD,reduction='sum')/denominator
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite loss; stop without overwriting last clean checkpoint')
        loss.backward()
        total_loss+=float(loss.detach())
    return total_loss


def train_update(model,opt,group,device,weights,micro_budget,max_examples,clip=1.):
    # Retry only forward/backward. Same logical rows; no rows are skipped on OOM.
    initial_rng=rng_state()
    while True:
        opt.zero_grad(set_to_none=True)
        try:
            loss=backward_group(model,group,device,weights,micro_budget,max_examples)
            break
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            restore_rng(initial_rng)
            if micro_budget<=1024:
                raise RuntimeError('OOM at minimum micro budget; restart from last.pt with smaller model')
            micro_budget=max(1024,micro_budget//2)
            print(f'OOM: retry SAME logical batch, micro_budget={micro_budget}',flush=True)
    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True)
    opt.step()  # If this fails, restart from last.pt; never save a partially updated optimizer.
    return loss,float(grad),micro_budget


def make_optimizer(model,c,device):
    return torch.optim.AdamW(model.parameters(),lr=c['peak_lr'],betas=(.9,.999),
        weight_decay=c['weight_decay'],fused=str(device).startswith('cuda'))


def checkpoint_state(model,opt,stream,config,identity,val_sha,progress):
    return dict(format_version=2,model=model.state_dict(),optimizer=opt.state_dict(),
        model_config=asdict(model.config),train_config=config,stream=stream.state_dict(),
        cache_identity=identity,validation_sha256=val_sha,tokenizer_sha256=TOKENIZER_SHA,
        rng=rng_state(),**progress)


def run(config_path,root=ROOT,stop_tokens=None,stop_seconds=None,resume=False,device='cuda'):
    root=Path(root)
    cfg=json.loads(Path(config_path).read_text())
    c={**DEFAULT_TRAIN,**cfg.get('train',{})}
    mc=ModelConfig(**cfg['model'])
    cache=root/'data/prepared/bpe8192'
    directory=safe_run_dir(root,cfg['name'])
    checkpoint=directory/'last.pt'
    if checkpoint.exists() and not resume:
        raise FileExistsError('Run exists. Use --resume; a fresh run needs a different config name.')
    if resume and not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if str(device).startswith('cuda') and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('Training requires a BF16-capable CUDA GPU; T4 is for inference only')
    identity=check_cache(cache)
    val_path=root/'experiments/v2/validation4096.json'
    manifest=json.loads(val_path.read_text())
    if manifest['cache_identity']['fingerprint']!=identity['fingerprint']:
        raise ValueError('Validation/cache identity mismatch; do not silently resample')
    val_sha=sha256(val_path)
    val=load_validation(val_path,'dev')
    tok=Tokenizer.from_file(str(cache/'tokenizer.json'))
    seed_all(c['seed'])
    model=CodeTransformerV2(mc).to(device)
    opt=make_optimizer(model,c,device)
    stream=ArrowTrainStream(cache,c['seed'],c['pool_size'],c['logical_budget'],c['max_examples'])
    progress=dict(step=0,target_tokens=0,source_tokens=0,examples=0,train_seconds=0.,
        best_metric=-1.,next_eval_tokens=c['eval_every_tokens'],effective_micro_budget=c['micro_budget'])
    code_sha={p.name:sha256(p) for p in Path(__file__).parent.glob('hw2_*v2.py')}
    if resume:
        s=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if (s['train_config']!=c or s['model_config']!=asdict(mc)
            or s['cache_identity']['fingerprint']!=identity['fingerprint']
            or s['validation_sha256']!=val_sha or s['code_sha256']!=code_sha):
            raise ValueError('Exact resume refused: config, data, validation or source changed')
        model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer'])
        stream.load_state_dict(s['stream'])
        progress={k:s[k] for k in progress}
        restore_rng(s['rng'])
        del s
    else:
        write_json(directory/'config.json',dict(name=cfg['name'],model=asdict(mc),train=c))
    weights=token_weights(tok,c['numeric_weight'],device)
    log_path=directory/'train.jsonl'
    log(log_path,dict(event='start',resume=resume,parameters=parameter_count(mc),environment=environment(),**progress))

    def save(path):
        s=checkpoint_state(model,opt,stream,c,identity,val_sha,progress)
        s['code_sha256']=code_sha
        atomic_save(path,s)

    def validate():
        before=rng_state()
        metrics,details=evaluate(model,val,tok,device,c['eval_batch_size'],c['max_new_tokens'])
        restore_rng(before)
        metrics.update(step=progress['step'],target_tokens=progress['target_tokens'],
            source_tokens=progress['source_tokens'],examples=progress['examples'],role='dev',
            validation_sha256=val_sha,max_new_tokens=c['max_new_tokens'])
        stem=directory/f'dev_{progress["step"]}'
        write_json(stem.with_suffix('.json'),metrics)
        write_json(stem.with_suffix('.details.json'),details)
        log(log_path,dict(event='validation',**metrics))
        metric=metrics['agreement_numeric_references']
        if metric is not None and metric>progress['best_metric']:
            progress['best_metric']=metric
            # Selection checkpoint has weights/metadata only; last.pt is the exact-resume checkpoint.
            atomic_save(directory/'best.pt',dict(model=model.state_dict(),model_config=asdict(mc),
                tokenizer_sha256=TOKENIZER_SHA,validation_sha256=val_sha,
                selected_metric=metrics,code_sha256=code_sha,**progress))

    if ((stop_tokens is not None and progress['target_tokens']>=stop_tokens)
        or (stop_seconds is not None and progress['train_seconds']>=stop_seconds)):
        return directory
    model.train()
    interval_start=time.perf_counter()
    while stream.epoch<c['epochs']:
        group=stream.next_group()
        if group is None:
            log(log_path,dict(event='epoch_finished',epoch=stream.epoch,**progress))
            stream.reset(stream.epoch+1)
            continue
        nt=sum(len(y) for x,y,r in group);ns=sum(len(x) for x,y,r in group)
        lr=lr_at(progress['target_tokens']+nt,c)
        for pg in opt.param_groups:
            pg['lr']=lr
        t=time.perf_counter()
        loss,grad,budget=train_update(model,opt,group,device,weights,
            progress['effective_micro_budget'],c['max_examples'],c['clip'])
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        progress['train_seconds']+=time.perf_counter()-t
        progress['effective_micro_budget']=budget
        progress['step']+=1;progress['target_tokens']+=nt;progress['source_tokens']+=ns
        progress['examples']+=len(group)
        if progress['step']==1 or progress['step']%100==0:
            log(log_path,dict(event='train',loss=loss,grad_norm=grad,lr=lr,epoch=stream.epoch,
                interval_wall_seconds=time.perf_counter()-interval_start,
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30 if str(device).startswith('cuda') else None,
                **progress))
            interval_start=time.perf_counter()
        reached=(stop_tokens is not None and progress['target_tokens']>=stop_tokens) or (
            stop_seconds is not None and progress['train_seconds']>=stop_seconds)
        if progress['target_tokens']>=progress['next_eval_tokens'] or reached:
            while progress['next_eval_tokens']<=progress['target_tokens']:
                progress['next_eval_tokens']+=c['eval_every_tokens']
            save(checkpoint)  # clean resumable state even if generation OOMs
            validate()
        if progress['step']%c['checkpoint_every']==0 or reached:
            save(checkpoint)
        if reached:
            log(log_path,dict(event='paused',**progress))
            stream.close()
            return directory
    validate();save(checkpoint)
    log(log_path,dict(event='finished',**progress))
    stream.close()
    return directory


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/v2_main.json')
    p.add_argument('--root',default=str(ROOT))
    p.add_argument('--stop-tokens',type=int)
    p.add_argument('--stop-seconds',type=float)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    run(a.config,a.root,a.stop_tokens,a.stop_seconds,a.resume,a.device)
