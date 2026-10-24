"""Read-only full allowed-train audit. No corrected labels, filtering, or test answers."""
import argparse
import ast
import hashlib
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from hw2_common_v2 import ROOT,write_json
from hw2_safe_v2 import run_numeric,same_answer


def is_validation(q):
    norm=' '.join(q.split()).casefold().encode()
    return int.from_bytes(hashlib.sha256(norm).digest()[:8],'big')%100<5


def lengths(cache):
    hist={k:Counter() for k in ('source','target')};n=0
    for p in sorted(Path(cache).glob('part-*.arrow')):
        with pa.memory_map(str(p),'r') as m:
            r=ipc.open_file(m)
            for i in range(r.num_record_batches):
                b=r.get_batch(i);b=b.filter(pc.invert(b.column('is_validation')));n+=len(b)
                hist['source'].update(b.column('input_length').to_pylist())
                hist['target'].update(b.column('target_length').to_pylist())
    return dict(train_rows=n,histograms={k:dict(sorted(v.items())) for k,v in hist.items()},
        source_tokens=sum(k*v for k,v in hist['source'].items()),
        target_tokens=sum(k*v for k,v in hist['target'].items()),
        target_over={str(t):sum(v for k,v in hist['target'].items() if k>t) for t in (64,128,192,256,384,512)})


def duplicates(raw,database,max_rows=None):
    """First pass stores only hashes. Second pass interprets conflicting question groups."""
    db=Path(database)
    if db.exists():raise FileExistsError('Use a new audit database to avoid double counting')
    db.parent.mkdir(parents=True,exist_ok=True)
    conn=sqlite3.connect(db)
    conn.execute('PRAGMA journal_mode=WAL');conn.execute('PRAGMA synchronous=NORMAL')
    conn.execute('CREATE TABLE pairs(q BLOB,c BLOB,n INTEGER,PRIMARY KEY(q,c)) WITHOUT ROWID')
    paths=sorted((Path(raw)/'data/train').glob('part-*.parquet'))
    if len(paths)!=60:raise ValueError('Expected 60 official train parquet files')
    seen=0;complete=True;t=time.perf_counter()
    def iterate():
        for p in paths:
            for b in pq.ParquetFile(p).iter_batches(batch_size=8192,columns=['question','code']):
                yield b.to_pylist()
    for batch in iterate():
        vals=[]
        for r in batch:
            if is_validation(r['question']):continue
            if max_rows is not None and seen>=max_rows:complete=False;break
            seen+=1
            q=hashlib.sha256(' '.join(r['question'].split()).casefold().encode()).digest()
            c=hashlib.sha256(r['code'].encode()).digest()
            vals.append((q,c,1))
        conn.executemany('INSERT INTO pairs VALUES (?,?,?) ON CONFLICT(q,c) DO UPDATE SET n=n+1',vals)
        conn.commit()
        if not complete:break
    pairs=conn.execute('SELECT COUNT(*),SUM(n-1) FROM pairs').fetchone()
    conn.execute('CREATE TABLE conflicts AS SELECT q,COUNT(*) AS variants FROM pairs GROUP BY q HAVING COUNT(*)>1')
    conn.execute('CREATE UNIQUE INDEX conflicts_q ON conflicts(q)');conn.commit()
    conflict_count=conn.execute('SELECT COUNT(*) FROM conflicts').fetchone()[0]
    questions=conn.execute('SELECT COUNT(*) FROM (SELECT q FROM pairs GROUP BY q)').fetchone()[0]
    # Bound review work and disclosure volume; conflict HASH counts above are still exact.
    selected={r[0] for r in conn.execute('SELECT q FROM conflicts ORDER BY q LIMIT 2000')}
    variants={q:{} for q in selected};second_seen=0
    for batch in iterate():
        for r in batch:
            if is_validation(r['question']):continue
            if second_seen>=seen:break
            second_seen+=1
            q=hashlib.sha256(' '.join(r['question'].split()).casefold().encode()).digest()
            if q in selected:
                c=hashlib.sha256(r['code'].encode()).hexdigest()
                if c not in variants[q]:
                    status,value=run_numeric(r['code'])
                    variants[q][c]=dict(status=status,value=value)
        if second_seen>=seen:break
    supported=conflicting=0
    for v in variants.values():
        answers=[r['value'] for r in v.values() if r['status']=='ok']
        if len(answers)>=2:
            supported+=1
            conflicting+=any(not same_answer(a,answers[0]) for a in answers[1:])
    conn.close()
    return dict(train_rows_scanned=seen,full_train=complete,unique_normalized_questions=questions,
        unique_question_code_pairs=pairs[0],duplicate_pair_excess=pairs[1],
        questions_with_multiple_code_strings=conflict_count,semantic_review_groups=len(selected),
        groups_with_two_numeric_programs=supported,groups_with_numeric_disagreement=conflicting,
        seconds=time.perf_counter()-t,
        caveat='Different code strings need not be contradictory. Safe numeric disagreement is not a corrected label. No training data changed.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['lengths','duplicates'])
    p.add_argument('--root',default=str(ROOT));p.add_argument('--output',required=True)
    p.add_argument('--max-rows',type=int);p.add_argument('--database',default='train_duplicates.sqlite')
    a=p.parse_args();root=Path(a.root)
    result=lengths(root/'data/prepared/bpe8192') if a.mode=='lengths' else duplicates(root/'data/raw',a.database,a.max_rows)
    write_json(a.output,result);print(json.dumps(result,indent=2))
