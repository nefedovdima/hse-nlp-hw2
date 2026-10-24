import hashlib
import json
import os
import random
import platform
from pathlib import Path
import torch

TOKENIZER_SHA = '659f7740d5fc78e8ad7c7b395331fc294a6186f270b6643272dce80e640fc6d2'
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision('high')


def rng_state():
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(s):
    random.setstate(s['python'])
    torch.set_rng_state(s['torch'])
    if s['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s['cuda'])


def environment():
    return dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        capability=torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)


def check_cache(cache, full_hash=False, expected_shards=60):
    cache = Path(cache)
    if sha256(cache/'tokenizer.json') != TOKENIZER_SHA:
        raise ValueError('Tokenizer differs from the supplied E1 cache')
    paths = sorted(cache.glob('part-*.arrow'))
    if len(paths) != expected_shards:
        raise ValueError(f'Expected {expected_shards} shards; found {len(paths)}')
    identity = []
    for p in paths:
        meta_path = p.with_suffix('.meta.json')
        if not meta_path.exists():
            raise FileNotFoundError(f'Required original preparation manifest: {meta_path}')
        meta = json.loads(meta_path.read_text())
        if meta['signature']['tokenizer_sha256'] != TOKENIZER_SHA:
            raise ValueError(f'Shard/tokenizer mismatch: {p}')
        if p.stat().st_size != meta['bytes']:
            raise ValueError(f'Shard size mismatch: {p}')
        if full_hash and sha256(p) != meta['cache_sha256']:
            raise ValueError(f'Shard content mismatch: {p}')
        identity.append((p.name, p.stat().st_size, p.stat().st_mtime_ns, meta['cache_sha256']))
    return dict(tokenizer=TOKENIZER_SHA, shards=identity,
                fingerprint=hashlib.sha256(json.dumps(identity).encode()).hexdigest(),
                content_verified=full_hash)


def log(path, record):
    print(json.dumps(record, ensure_ascii=False), flush=True)
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False)+'\n')


def safe_run_dir(root, name):
    base = (Path(root)/'experiments/v2').resolve()
    path = (base/name).resolve()
    if not path.is_relative_to(base) or path == base:
        raise ValueError('Run directory must be a child of experiments/v2')
    path.mkdir(parents=True, exist_ok=True)
    return path
