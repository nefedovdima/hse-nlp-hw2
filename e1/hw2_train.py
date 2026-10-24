"""Training helpers. No download, no new targets, full streaming Arrow train."""
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from hw2_batches import ArrowTrainStream, collate, pack_by_length, small_split_sample
from hw2_model import CodeTransformer, ModelConfig, PAD

ROOT = Path(__file__).resolve().parents[1]


def seed_all(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision('high')


def make_model(config=None, device='cuda'):
    m = CodeTransformer(config or ModelConfig())
    return m.to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def loss_and_tokens(model, batch, device='cuda'):
    src, dec, labels, _ = batch
    src = src.to(device, non_blocking=True)
    dec = dec.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
        logits = model(src, dec)
        # Cross entropy is computed with FP32 logits; padding is never scored.
        loss = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), ignore_index=PAD)
    ntokens = labels.ne(PAD).sum().item()
    return loss, ntokens


@torch.no_grad()
def evaluate_loss(model, examples, device='cuda', token_budget=4096):
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0., 0
    groups = pack_by_length(list(examples), token_budget=token_budget, max_examples=64)
    for group in groups:
        loss, tokens = loss_and_tokens(model, collate(group), device)
        total_loss += loss.item() * tokens
        total_tokens += tokens
    if was_training:
        model.train()
    return total_loss / max(1, total_tokens)


def overfit_128(cache, steps=100, device='cuda'):
    """Diagnostic only, does not affect the full-train checkpoint."""
    seed_all(1234)
    examples = small_split_sample(cache, 'train', 128)
    model = make_model(device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=0.)
    groups = pack_by_length(list(examples), token_budget=2048, max_examples=32)
    rng = random.Random(1234)
    started = time.perf_counter()
    print('128 real train rows; full token sequences, batches:', len(groups),
          'parameters:', count_parameters(model), flush=True)
    for step in range(1, steps + 1):
        model.train()
        group = groups[(step - 1) % len(groups)]
        # Rotate each pass, keep deterministic.
        if (step - 1) % len(groups) == 0:
            rng.shuffle(groups)
        opt.zero_grad(set_to_none=True)
        loss, tokens = loss_and_tokens(model, collate(group), device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        if step == 1 or step % 20 == 0 or step == steps:
            diag = evaluate_loss(model, examples, device, token_budget=2048)
            print(f'pilot step={step} batch_loss={loss.item():.4f} '
                  f'loss_on_128={diag:.4f} elapsed={time.perf_counter()-started:.1f}s', flush=True)
    return model


def benchmark_training(model, cache, steps=20, device='cuda'):
    """Timing forward+backward+optimizer on real full-length train batches."""
    stream = ArrowTrainStream(cache, seed=710, token_budget=8192, pool_size=4096)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0003, fused=True)
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times, nsrc, ntgt = [], 0, 0
    # 3 warmup steps not included in time.
    for step in range(steps + 3):
        batch = stream.next_batch()
        if batch is None:
            raise RuntimeError('Unexpected end of Arrow train')
        src, _, labels, _ = batch
        if step >= 3:
            nsrc += src.ne(PAD).sum().item()
            ntgt += labels.ne(PAD).sum().item()
        torch.cuda.synchronize()
        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss, _ = loss_and_tokens(model, batch, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        torch.cuda.synchronize()
        if step >= 3:
            times.append(time.perf_counter() - start)
    elapsed = sum(times)
    info = {'steps': steps, 'avg_step_sec': round(elapsed/steps, 4),
            'source_tokens_per_sec': round(nsrc / elapsed),
            'target_tokens_per_sec': round(ntgt / elapsed),
            'total_tokens_per_sec': round((nsrc + ntgt) / elapsed),
            'gpu_peak_allocated_GiB': round(torch.cuda.max_memory_allocated()/1024**3, 3),
            'gpu_peak_reserved_GiB': round(torch.cuda.max_memory_reserved()/1024**3, 3)}
    print(json.dumps(info, indent=2), flush=True)
    return info


def lr_at(step, peak=5e-4, warmup=1500):
    return peak * min(step / warmup, math.sqrt(warmup / step))


def write_log(path, record):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def save_checkpoint(path, model, opt, stream, step, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        'model': model.state_dict(), 'optimizer': opt.state_dict(),
        'stream': stream.state_dict(), 'step': step, 'model_config': asdict(model.config),
        'train_config': config,
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'python_rng': random.getstate(),
    }
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)
    print('Saved:', path, 'step=', step, flush=True)


def train_full(root=ROOT, stop_at=None, epochs=1, resume=True,
               checkpoint_every=500, validation_every=1000,
               token_budget=8192, max_examples=96, pool_size=4096):
    """Exact resume; stop_at is global step for staged runs, None means all epochs.

    Full 11,244,489-row train stream, dynamically bucketed; no truncation.
    """
    root = Path(root)
    cache = root / 'data/prepared/bpe8192'
    ckpt_path = root / 'checkpoints/e01_last.pt'
    log_path = root / 'reports/e01_train.jsonl'
    settings = {'seed': 42, 'epochs': epochs, 'token_budget': token_budget,
                'max_examples': max_examples, 'pool_size': pool_size,
                'peak_lr': 5e-4, 'warmup': 1500, 'weight_decay': .01}
    seed_all(42)
    stream = ArrowTrainStream(cache, seed=settings['seed'],
                              pool_size=pool_size, token_budget=token_budget,
                              max_examples=max_examples)
    step = 0
    if resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if state['train_config'] != settings:
            raise ValueError('Checkpoint has different training settings; use identical config to resume')
        model = make_model(ModelConfig(**state['model_config']))
        model.load_state_dict(state['model'])
        opt = torch.optim.AdamW(model.parameters(), lr=settings['peak_lr'],
                                weight_decay=settings['weight_decay'], fused=True)
        opt.load_state_dict(state['optimizer'])
        stream.load_state_dict(state['stream'])
        step = state['step']
        torch.set_rng_state(state['torch_rng'])
        if state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng'])
        print('RESUMED from step', step, 'epoch', stream.epoch, flush=True)
    else:
        model = make_model()
        opt = torch.optim.AdamW(model.parameters(), lr=settings['peak_lr'],
                                weight_decay=settings['weight_decay'], fused=True)
        print('NEW FULL-TRAIN RUN (all train shards, no truncation)', flush=True)
    print('parameters=', count_parameters(model), 'epoch=', stream.epoch,
          'start_step=', step, 'stop_at=', stop_at, flush=True)
    if step >= (stop_at if stop_at is not None else float('inf')):
        print('Already reached stop_at; nothing to do')
        return model
    model.train()
    t0 = time.perf_counter()
    interval_loss, interval_tokens, interval_steps = 0., 0, 0
    # We take a fixed, shard-stratified slice of held-out validation only.
    val = small_split_sample(cache, 'validation', count=960, per_shard=16)
    write_log(log_path, {'event': 'run_started', 'step': step, 'resume': resume,
                         'epoch': stream.epoch, 'train_config': settings,
                         'model_config': asdict(model.config)})
    while stream.epoch < epochs:
        batch = stream.next_batch()
        if batch is None:
            print(f'FINISHED epoch {stream.epoch}, full stream consumed', flush=True)
            stream.reset(stream.epoch + 1)
            continue
        step += 1
        lr = lr_at(step, settings['peak_lr'], settings['warmup'])
        for pgroup in opt.param_groups:
            pgroup['lr'] = lr
        opt.zero_grad(set_to_none=True)
        loss, tokens = loss_and_tokens(model, batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        interval_loss += loss.item() * tokens
        interval_tokens += tokens
        interval_steps += 1
        if step == 1 or step % 100 == 0:
            torch.cuda.synchronize()
            rec = {'event': 'train', 'step': step, 'epoch': stream.epoch,
                   'loss': round(interval_loss / max(1, interval_tokens), 5),
                   'target_tokens': interval_tokens, 'lr': lr,
                   'elapsed_sec': round(time.perf_counter() - t0, 1),
                   'gpu_peak_GiB': round(torch.cuda.max_memory_allocated() / 1024**3, 2),
                   'grad_norm': round(float(grad), 4)}
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            write_log(log_path, rec)
            interval_loss, interval_tokens, interval_steps = 0., 0, 0
        if step % validation_every == 0:
            vloss = evaluate_loss(model, val)
            rec = {'event': 'validation', 'step': step, 'loss': round(vloss, 6),
                   'val_examples': len(val), 'note': 'token loss, NOT pass@1'}
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            write_log(log_path, rec)
        if step % checkpoint_every == 0 or (stop_at is not None and step >= stop_at):
            save_checkpoint(ckpt_path, model, opt, stream, step, settings)
        if stop_at is not None and step >= stop_at:
            print('PAUSED at step', step, 'ready to resume full train', flush=True)
            return model
    save_checkpoint(ckpt_path, model, opt, stream, step, settings)
    print('FINISHED requested training epochs, total_steps=', step, flush=True)
    return model
