"""Streaming Arrow batches; reproducible length bucketing and exact stream resume."""
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import torch

from hw2_model import BOS, EOS, PAD


def collate(items):
    """Each item is (source list, target list, row_id). Padding is dynamic."""
    if not items:
        raise ValueError('Empty batch')
    b = len(items)
    sx = max(len(x) for x, _, _ in items)
    ty = max(len(y) for _, y, _ in items)
    src = torch.full((b, sx), PAD, dtype=torch.long)
    dec = torch.full((b, ty), PAD, dtype=torch.long)
    labels = torch.full((b, ty), PAD, dtype=torch.long)
    ids = []
    for i, (x, y, rid) in enumerate(items):
        if not x or not y or x[-1] != EOS or y[-1] != EOS:
            raise ValueError(f'EOS missing for row_id={rid}')
        src[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        dec[i, 0] = BOS
        if len(y) > 1:
            dec[i, 1:len(y)] = torch.tensor(y[:-1], dtype=torch.long)
        labels[i, :len(y)] = torch.tensor(y, dtype=torch.long)
        ids.append(rid)
    return src, dec, labels, ids


def pack_by_length(items, token_budget=8192, max_examples=96):
    # No truncation: very long rows become batch size 1.
    items.sort(key=lambda r: (max(len(r[0]), len(r[1])), len(r[0]) + len(r[1])))
    groups, group, maxx, maxy = [], [], 0, 0
    for item in items:
        x, y, _ = item
        nx, ny = max(maxx, len(x)), max(maxy, len(y))
        if group and (len(group) >= max_examples or (len(group) + 1) * (nx + ny) > token_budget):
            groups.append(group)
            group, maxx, maxy = [], 0, 0
            nx, ny = len(x), len(y)
        group.append(item)
        maxx, maxy = nx, ny
    if group:
        groups.append(group)
    return groups


class ArrowTrainStream:
    def __init__(self, root, seed=42, pool_size=4096, token_budget=8192, max_examples=96):
        self.paths = sorted(Path(root).glob('part-*.arrow'))
        if len(self.paths) != 60:
            raise ValueError(f'Expected 60 Arrow shards, got {len(self.paths)} in {root}')
        self.seed, self.pool_size = seed, pool_size
        self.token_budget, self.max_examples = token_budget, max_examples
        self.reader, self.mmap = None, None
        self.reset(0)

    def _close(self):
        self.reader = None
        if self.mmap is not None:
            self.mmap.close()
        self.mmap = None

    def reset(self, epoch):
        self._close()
        self.epoch = epoch
        self.rng = random.Random(self.seed + epoch)
        self.order = list(range(len(self.paths)))
        self.rng.shuffle(self.order)
        self.shard_pos, self.record_pos = 0, 0
        self.ready, self.ready_index = [], 0
        self.done = False

    def _read(self):
        while self.shard_pos < len(self.order):
            if self.reader is None:
                self.mmap = pa.memory_map(str(self.paths[self.order[self.shard_pos]]), 'r')
                self.reader = ipc.open_file(self.mmap)
            if self.record_pos == self.reader.num_record_batches:
                self._close()
                self.shard_pos += 1
                self.record_pos = 0
                continue
            batch = self.reader.get_batch(self.record_pos)
            self.record_pos += 1
            return batch
        return None

    def _fill(self):
        pool = []
        while len(pool) < self.pool_size:
            batch = self._read()
            if batch is None:
                break
            # Read only required columns; original row IDs are preserved.
            rows = batch.select(['row_id', 'input_ids', 'labels', 'is_validation']).to_pylist()
            pool.extend((r['input_ids'], r['labels'], r['row_id']) for r in rows if not r['is_validation'])
        if not pool:
            self.done = True
            return
        self.ready = pack_by_length(pool, self.token_budget, self.max_examples)
        self.rng.shuffle(self.ready)
        self.ready_index = 0

    def next_batch(self):
        if self.ready_index >= len(self.ready):
            self._fill()
        if self.done:
            return None
        group = self.ready[self.ready_index]
        self.ready_index += 1
        return collate(group)

    def state_dict(self):
        return dict(epoch=self.epoch, order=self.order, shard_pos=self.shard_pos,
                    record_pos=self.record_pos, ready=self.ready, ready_index=self.ready_index,
                    rng=self.rng.getstate(), done=self.done)

    def load_state_dict(self, state):
        self._close()
        self.epoch = state['epoch']
        self.order, self.shard_pos = state['order'], state['shard_pos']
        self.record_pos = state['record_pos']
        self.ready, self.ready_index = state['ready'], state['ready_index']
        self.done = state['done']
        self.rng = random.Random()
        self.rng.setstate(state['rng'])


def small_split_sample(root, split='train', count=128, per_shard=None):
    """Fixed real rows, no synthetic labels, no changed split."""
    if split not in ('train', 'validation'):
        raise ValueError(split)
    paths = sorted(Path(root).glob('part-*.arrow'))
    if not paths:
        raise FileNotFoundError(root)
    result = []
    for path in paths:
        selected = 0
        with pa.memory_map(str(path), 'r') as source:
            reader = ipc.open_file(source)
            for i in range(reader.num_record_batches):
                rows = reader.get_batch(i).select(['row_id', 'input_ids', 'labels', 'is_validation']).to_pylist()
                for r in rows:
                    if bool(r['is_validation']) != (split == 'validation'):
                        continue
                    result.append((r['input_ids'], r['labels'], r['row_id']))
                    selected += 1
                    if len(result) >= count:
                        return result
                    if per_shard is not None and selected >= per_shard:
                        break
                if per_shard is not None and selected >= per_shard:
                    break
    if len(result) < count:
        raise ValueError(f'Only {len(result)} {split} rows, need {count}')
    return result
