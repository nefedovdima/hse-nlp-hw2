"""Small temporary fixtures verify orchestration; no deliverable model is trained here."""
import json
import shutil
from dataclasses import asdict
from pathlib import Path
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from tokenizers import Tokenizer
from hw2_common_v2 import TOKENIZER_SHA,check_cache,sha256,write_json
from hw2_model_v2 import CodeTransformerV2
from hw2_train_v2 import run
from hw2_data_v2 import select_validation
from hw2_export_v2 import export
from test_model import tiny


def make_fixture(root):
    cache=root/'data/prepared/bpe8192';cache.mkdir(parents=True)
    source=Path(__file__).parents[1]/'evidence/tokenizer.json'
    shutil.copy2(source,cache/'tokenizer.json')
    tok=Tokenizer.from_file(str(source))
    x=tok.encode('Unit fixture.').ids+[1]
    y=tok.encode('def solve():\n    return 2\n').ids+[1]
    validation=[]
    for i in range(60):
        x=tok.encode(f'Unit fixture {i%5}.').ids+[1]
        rows=[dict(row_id=2*i,input_ids=x,labels=y,is_validation=False),
              dict(row_id=2*i+1,input_ids=x,labels=y,is_validation=True)]
        p=cache/f'part-{i:02}.arrow';table=pa.Table.from_pylist(rows)
        with pa.OSFile(str(p),'wb') as f:
            with ipc.new_file(f,table.schema) as w:w.write_table(table)
        write_json(p.with_suffix('.meta.json'),dict(bytes=p.stat().st_size,cache_sha256=sha256(p),
            signature=dict(tokenizer_sha256=TOKENIZER_SHA)))
        if i<2:validation.append(dict(**rows[1],role='dev'))
    identity=check_cache(cache,full_hash=True)
    write_json(root/'experiments/v2/validation4096.json',dict(rows=validation,cache_identity=identity))
    return cache


def test_validation_sampling_full_population_and_group_separation(tmp_path):
    cache=make_fixture(tmp_path)
    one=tmp_path/'sample1.json';two=tmp_path/'sample2.json'
    select_validation(cache,one,count=40,seed=20261022)
    select_validation(cache,two,count=40,seed=20261022)
    a,b=json.loads(one.read_text()),json.loads(two.read_text())
    assert a==b and a['population_rows']==60 and len(a['rows'])==40
    assert len({r['row_id'] for r in a['rows']})==40
    assert all(r['row_id']%2==1 for r in a['rows'])
    roles={}
    for r in a['rows']:
        key=tuple(r['input_ids'])
        if key in roles:assert roles[key]==r['role']
        roles[key]=r['role']
    assert a['sample_unique_normalized_questions']==len(roles)
    assert 'source' in a['population_length_quantiles']


def test_training_entrypoint_pause_resume_and_identity(tmp_path):
    make_fixture(tmp_path)
    config=tmp_path/'config.json'
    c=tiny(vocab_size=8192,encoder_layers=1,decoder_layers=1,dim=16,heads=2,ff_dim=32,max_positions=64)
    write_json(config,dict(name='unit_fixture',model=asdict(c),train=dict(pool_size=4,logical_budget=128,
        max_examples=4,micro_budget=128,eval_every_tokens=100000,max_new_tokens=4,eval_batch_size=2,checkpoint_every=10)))
    out=run(config,root=tmp_path,stop_tokens=1,device='cpu')
    first=torch.load(out/'last.pt',weights_only=False)
    assert first['step']==1 and first['examples']>0
    run(config,root=tmp_path,resume=True,device='cpu')
    final=torch.load(out/'last.pt',weights_only=False)
    assert final['stream']['epoch']==1 and final['examples']==60
    assert final['step']>first['step'] and final['target_tokens']>first['target_tokens']
    # Confirm existing run protection, rather than silently replacing a checkpoint.
    import pytest
    with pytest.raises(FileExistsError):run(config,root=tmp_path,device='cpu')


def test_portable_fp16_export_roundtrip(tmp_path):
    cache=make_fixture(tmp_path)
    table=pa.Table.from_pylist([dict(row_id=i,input_ids=[7,1]) for i in range(1919)])
    with pa.OSFile(str(cache/'test.arrow'),'wb') as f:
        with ipc.new_file(f,table.schema) as w:w.write_table(table)
    c=tiny(vocab_size=8192);m=CodeTransformerV2(c)
    path=tmp_path/'experiments/v2/export_fixture/best.pt';path.parent.mkdir(parents=True)
    torch.save(dict(model=m.state_dict(),model_config=asdict(c),step=12,target_tokens=1234,
        tokenizer_sha256=TOKENIZER_SHA),path)
    before=sha256(path)
    archive=export(path,root=tmp_path)
    assert archive.is_file() and sha256(path)==before
    out=path.parent/'t4_step12';manifest=json.loads((out/'manifest.json').read_text())
    assert all(sha256(out/n)==v for n,v in manifest['files'].items())
    s=torch.load(out/'model_fp16.pt',weights_only=True)
    loaded=CodeTransformerV2(c).half();loaded.load_state_dict(s['model'],strict=True)
    for a,b in zip(m.parameters(),loaded.parameters()):torch.testing.assert_close(a.half(),b,atol=0,rtol=0)
    rows=json.loads((out/'test_inputs.json').read_text())
    assert [r['row_id'] for r in rows]==list(range(1919))
