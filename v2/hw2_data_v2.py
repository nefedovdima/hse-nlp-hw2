"""All original training rows, deterministic pools and serializable stream position."""
import copy
import heapq
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import torch
from hw2_model_v2 import PAD, BOS, EOS
from hw2_common_v2 import ROOT, write_json, check_cache, sha256


def collate(items):
    if not items:
        raise ValueError('Empty batch')
    b, sx, sy = len(items), max(len(x) for x,y,r in items), max(len(y) for x,y,r in items)
    src, dec, labels = (torch.full((b,n), PAD, dtype=torch.long) for n in (sx,sy,sy))
    for i,(x,y,r) in enumerate(items):
        if not x or not y or x[-1] != EOS or y[-1] != EOS or PAD in x or PAD in y:
            raise ValueError(f'Invalid prepared sequence for row {r}')
        src[i,:len(x)], labels[i,:len(y)] = torch.tensor(x), torch.tensor(y)
        dec[i,0] = BOS
        dec[i,1:len(y)] = torch.tensor(y[:-1])
    return src, dec, labels, [r for x,y,r in items]


def pack(items, budget=32768, max_examples=256):
    groups, group, mx, my = [], [], 0, 0
    for item in sorted(items, key=lambda r:(max(len(r[0]),len(r[1])),len(r[0])+len(r[1]))):
        nx, ny = max(mx,len(item[0])), max(my,len(item[1]))
        if group and (len(group)>=max_examples or (len(group)+1)*(nx+ny)>budget):
            groups.append(group)
            group, nx, ny = [], len(item[0]),len(item[1])
        group.append(item)
        mx,my = nx,ny
    if group:
        groups.append(group)
    return groups


class ArrowTrainStream:
    def __init__(self, cache, seed=42, pool_size=32768, budget=32768, max_examples=256, expected_shards=60):
        self.paths = sorted(Path(cache).glob('part-*.arrow'))
        if len(self.paths) != expected_shards:
            raise ValueError(f'Expected {expected_shards} shards, found {len(self.paths)}')
        self.seed,self.pool_size,self.budget,self.max_examples = seed,pool_size,budget,max_examples
        self.reader,self.mmap = None,None
        self.reset(0)

    def close(self):
        self.reader = None
        if self.mmap is not None:
            self.mmap.close()
        self.mmap = None

    def reset(self, epoch):
        self.close()
        self.epoch = epoch
        self.rng = random.Random(self.seed+epoch)
        self.order = list(range(len(self.paths)))
        self.rng.shuffle(self.order)
        self.shard_pos,self.record_pos,self.ready_index = 0,0,0
        self.ready,self.done = [],False

    def _read(self):
        while self.shard_pos < len(self.order):
            if self.reader is None:
                self.mmap = pa.memory_map(str(self.paths[self.order[self.shard_pos]]),'r')
                self.reader = ipc.open_file(self.mmap)
            if self.record_pos == self.reader.num_record_batches:
                self.close()
                self.shard_pos += 1
                self.record_pos = 0
                continue
            batch = self.reader.get_batch(self.record_pos)
            self.record_pos += 1
            return batch
        return None

    def next_group(self):
        if self.ready_index >= len(self.ready):
            pool = []
            while len(pool)<self.pool_size:
                b = self._read()
                if b is None:
                    break
                b = b.filter(pc.invert(b.column('is_validation')))
                pool.extend((r['input_ids'],r['labels'],r['row_id']) for r in
                    b.select(['row_id','input_ids','labels']).to_pylist())
            if not pool:
                self.done = True
                return None
            # Shuffle before stable sorting to randomize equal-length examples too.
            self.rng.shuffle(pool)
            self.ready = pack(pool,self.budget,self.max_examples)
            self.rng.shuffle(self.ready)
            self.ready_index = 0
        group = self.ready[self.ready_index]
        self.ready_index += 1
        return group

    def state_dict(self):
        return dict(epoch=self.epoch, order=self.order.copy(), shard_pos=self.shard_pos,
            record_pos=self.record_pos, ready=copy.deepcopy(self.ready), ready_index=self.ready_index,
            rng=self.rng.getstate(), done=self.done,
            settings=(self.seed,self.pool_size,self.budget,self.max_examples))

    def load_state_dict(self, s):
        if tuple(s['settings']) != (self.seed,self.pool_size,self.budget,self.max_examples):
            raise ValueError('Stream settings differ; exact resume refused')
        self.close()
        for k in ('epoch','order','shard_pos','record_pos','ready','ready_index','done'):
            setattr(self,k,copy.deepcopy(s[k]))
        self.rng = random.Random()
        self.rng.setstate(s['rng'])


def select_validation(cache, output, count=4096, seed=20261022):
    """Uniform row-level hash-priority sample over ALL validation rows, not prefixes."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('Frozen validation already exists; use it without resampling')
    identity = check_cache(cache, full_hash=True)
    heap,seen = [],0
    lengths={'source':Counter(),'target':Counter()}
    for p in sorted(Path(cache).glob('part-*.arrow')):
        with pa.memory_map(str(p),'r') as m:
            reader = ipc.open_file(m)
            for i in range(reader.num_record_batches):
                b = reader.get_batch(i)
                b = b.filter(b.column('is_validation'))
                for r in b.select(['row_id','input_ids','labels']).to_pylist():
                    seen += 1
                    lengths['source'][len(r['input_ids'])]+=1
                    lengths['target'][len(r['labels'])]+=1
                    priority = int.from_bytes(hashlib.sha256(f'{seed}:{r["row_id"]}'.encode()).digest(),'big')
                    entry = (-priority, r['row_id'], r)
                    if len(heap)<count:
                        heapq.heappush(heap,entry)
                    elif entry[0] > heap[0][0]:
                        heapq.heapreplace(heap,entry)
    if len(heap) != count:
        raise ValueError('Not enough validation examples')
    rows = [v[2] for v in sorted(heap, key=lambda v:-v[0])]
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(Path(cache)/'tokenizer.json'))
    question_keys=[]
    for r in rows:
        normalized=' '.join(tok.decode(r['input_ids'][:-1]).split()).casefold()
        question_keys.append(normalized)
        h=hashlib.sha256(('role:'+str(seed)+':'+normalized).encode()).digest()
        r['role'] = 'dev' if h[0]<128 else 'confirm'
    def quantiles(hist):
        total=sum(hist.values());result={}
        for q in (0,50,90,95,99,100):
            threshold=max(1,(q*total+99)//100);acc=0
            for length,n in sorted(hist.items()):
                acc+=n
                if acc>=threshold:
                    result[str(q)]=length;break
        return result
    sample_lengths={'source':Counter(len(r['input_ids']) for r in rows),
                    'target':Counter(len(r['labels']) for r in rows)}
    write_json(output,dict(seed=seed, selection='lowest SHA256(seed:row_id), all validation rows',
        population_rows=seen, cache_identity=identity,rows=rows,
        roles=dict(Counter(r['role'] for r in rows)),sample_unique_normalized_questions=len(set(question_keys)),
        population_length_quantiles={k:quantiles(v) for k,v in lengths.items()},
        sample_length_quantiles={k:quantiles(v) for k,v in sample_lengths.items()}))
    return output


def load_validation(path, role='dev'):
    doc = json.loads(Path(path).read_text())
    rows = [r for r in doc['rows'] if role=='all' or r['role']==role]
    if not rows or len({r['row_id'] for r in rows})!=len(rows):
        raise ValueError('Invalid validation manifest')
    return [(r['input_ids'],r['labels'],r['row_id']) for r in rows]


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--cache',default=str(ROOT/'data/prepared/bpe8192'))
    p.add_argument('--output',default=str(ROOT/'experiments/v2/validation4096.json'))
    p.add_argument('--count',type=int,default=4096)
    a=p.parse_args()
    print(select_validation(a.cache,a.output,a.count))
