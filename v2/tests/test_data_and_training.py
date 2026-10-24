import copy
from pathlib import Path
import pytest
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from hw2_data_v2 import ArrowTrainStream,collate,pack
from hw2_model_v2 import CodeTransformerV2
from hw2_train_v2 import (DEFAULT_TRAIN,lr_at,make_optimizer,train_update,backward_group)
from hw2_common_v2 import rng_state,restore_rng,seed_all,atomic_save,safe_run_dir
from test_model import tiny


def fixture_shards(root):
    # Unit-test tensors only. These are never mixed into any training cache.
    for shard in range(3):
        rows=[dict(row_id=shard*6+i,input_ids=[7,8,1],labels=[9,10,1],
                   is_validation=(shard*6+i)%4==0) for i in range(6)]
        table=pa.Table.from_pylist(rows)
        with pa.OSFile(str(root/f'part-{shard:02}.arrow'),'wb') as f:
            with ipc.new_file(f,table.schema) as w:
                w.write_table(table,max_chunksize=2)


def test_shift_and_no_truncation():
    group=[([7,1],[8,9,1],1),([8,9,1],[10,1],2)]
    src,dec,y,ids=collate(group)
    assert dec.tolist()==[[3,8,9],[3,10,0]]
    assert y.tolist()==[[8,9,1],[10,1,0]]
    long=([7]*100+[1],[8]*200+[1],9)
    assert pack([long],budget=8)==[[long]]
    assert collate([long])[2].shape==(1,201)
    with pytest.raises(ValueError):collate([([7],[1],3)])


def test_stream_resume_and_epoch_coverage(tmp_path):
    fixture_shards(tmp_path)
    a=ArrowTrainStream(tmp_path,seed=42,pool_size=4,budget=12,max_examples=2,expected_shards=3)
    first=a.next_group();state=a.state_dict()
    b=ArrowTrainStream(tmp_path,seed=42,pool_size=4,budget=12,max_examples=2,expected_shards=3)
    b.load_state_dict(state)
    seen=[r for x,y,r in first]
    while True:
        aa,bb=a.next_group(),b.next_group();assert aa==bb
        if aa is None:break
        seen.extend(r for x,y,r in aa)
    assert sorted(seen)==[i for i in range(18) if i%4]
    assert len(set(seen))==len(seen)
    a.reset(1);assert a.next_group() is not None
    a.close();b.close()


def test_weighted_accumulation_matches_full_gradient():
    seed_all(41);a=CodeTransformerV2(tiny());b=copy.deepcopy(a)
    group=[([7,1],[8,9,1],1),([8,9,1],[9,1],2),([9,1],[10,11,1],3)]
    w=torch.ones(48);w[0]=0;w[9]=1.25
    la=backward_group(a,group,'cpu',w,1000,8)
    lb=backward_group(b,group,'cpu',w,6,1)
    assert abs(la-lb)<1e-6
    for x,y in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(x.grad,y.grad,atol=3e-6,rtol=3e-5)


def test_optimizer_rng_and_stream_resume(tmp_path):
    fixture_shards(tmp_path)
    c={**DEFAULT_TRAIN,'peak_lr':.001};seed_all(17)
    a=CodeTransformerV2(tiny(dropout=.1));oa=make_optimizer(a,c,'cpu')
    sa=ArrowTrainStream(tmp_path,pool_size=4,budget=12,max_examples=2,expected_shards=3)
    w=torch.ones(48);w[0]=0
    train_update(a,oa,sa.next_group(),'cpu',w,1000,8)
    atomic_save(tmp_path/'state.pt',dict(model=a.state_dict(),optimizer=oa.state_dict(),stream=sa.state_dict(),rng=rng_state()))
    next_group=sa.next_group();expected=train_update(a,oa,next_group,'cpu',w,1000,8)
    b=CodeTransformerV2(tiny(dropout=.1));ob=make_optimizer(b,c,'cpu')
    sb=ArrowTrainStream(tmp_path,pool_size=4,budget=12,max_examples=2,expected_shards=3)
    s=torch.load(tmp_path/'state.pt',weights_only=False)
    b.load_state_dict(s['model']);ob.load_state_dict(s['optimizer']);sb.load_state_dict(s['stream']);restore_rng(s['rng'])
    assert sb.next_group()==next_group
    actual=train_update(b,ob,next_group,'cpu',w,1000,8)
    assert actual==expected
    for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,atol=0,rtol=0)
    sa.close();sb.close()


def test_oom_retries_same_rows(monkeypatch):
    import hw2_train_v2 as train
    seed_all(9);m=CodeTransformerV2(tiny());o=make_optimizer(m,DEFAULT_TRAIN,'cpu')
    w=torch.ones(48);w[0]=0;group=[([7,1],[8,1],1)]
    real=train.backward_group;calls=[]
    def flaky(model,g,device,weights,budget,max_examples):
        calls.append(([r for x,y,r in g],budget))
        if len(calls)==1:raise torch.cuda.OutOfMemoryError('unit fixture')
        return real(model,g,device,weights,budget,max_examples)
    monkeypatch.setattr(train,'backward_group',flaky)
    _,_,budget=train_update(m,o,group,'cpu',w,4096,8)
    assert calls==[([1],4096),([1],2048)] and budget==2048


def test_schedule_and_path_protection(tmp_path):
    c=DEFAULT_TRAIN
    assert lr_at(c['warmup_target_tokens'],c)==pytest.approx(c['peak_lr'])
    assert lr_at(c['schedule_target_tokens'],c)==pytest.approx(c['peak_lr']*.1)
    for name in ('../e1','../../../checkpoints','/tmp/arbitrary','.'):
        with pytest.raises(ValueError):safe_run_dir(tmp_path,name)
