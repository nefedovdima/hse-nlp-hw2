"""Export only a new V2 checkpoint; portable, hashed T4 bundle with unchanged test order."""
import argparse
import json
import shutil
import zipfile
from pathlib import Path
import torch
import pyarrow as pa
import pyarrow.ipc as ipc
from hw2_common_v2 import ROOT,TOKENIZER_SHA,sha256,write_json,atomic_save
from hw2_model_v2 import ModelConfig,CodeTransformerV2,EOS


def export(checkpoint,root=ROOT):
    root=Path(root).resolve();checkpoint=Path(checkpoint).resolve()
    if not checkpoint.is_relative_to(root/'experiments/v2'):
        raise ValueError('Export only from experiments/v2; E1 is never rewritten')
    cache=root/'data/prepared/bpe8192'
    if sha256(cache/'tokenizer.json')!=TOKENIZER_SHA:
        raise ValueError('Tokenizer hash mismatch')
    s=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if s.get('tokenizer_sha256')!=TOKENIZER_SHA:
        raise ValueError('Checkpoint tokenizer identity mismatch')
    out=checkpoint.parent/f't4_step{s["step"]}'
    if out.exists():
        raise FileExistsError(f'Export already exists: {out}; preserve it or choose another checkpoint')
    out.mkdir()
    memo={};weights={}
    for name,value in s['model'].items():
        key=(value.untyped_storage().data_ptr(),value.storage_offset(),tuple(value.shape),tuple(value.stride()))
        if key not in memo:
            memo[key]=value.detach().half().contiguous()
        weights[name]=memo[key]
    model=CodeTransformerV2(ModelConfig(**s['model_config']))
    model.load_state_dict(weights,strict=True)
    atomic_save(out/'model_fp16.pt',dict(model=weights,model_config=s['model_config'],step=s['step'],
        tokenizer_sha256=TOKENIZER_SHA,target_tokens=s['target_tokens']))
    del model,weights,memo
    with pa.memory_map(str(cache/'test.arrow'),'r') as m:
        reader=ipc.open_file(m)
        rows=[r for i in range(reader.num_record_batches)
              for r in reader.get_batch(i).select(['row_id','input_ids']).to_pylist()]
    rows.sort(key=lambda r:r['row_id'])
    if [r['row_id'] for r in rows]!=list(range(1919)):
        raise ValueError('Expected the 1919 original consecutive test row IDs')
    for r in rows:
        assert r['input_ids'] and r['input_ids'][-1]==EOS
        assert all(0<=t<s['model_config']['vocab_size'] for t in r['input_ids'])
    write_json(out/'test_inputs.json',rows)
    shutil.copy2(cache/'tokenizer.json',out/'tokenizer.json')
    base=Path(__file__).parent
    for name in ['hw2_model_v2.py','hw2_infer_t4_v2.py','hw2_common_v2.py','03_inference_T4_v2.ipynb']:
        shutil.copy2(base/name,out/name)
    write_json(out/'manifest.json',dict(checkpoint_step=s['step'],checkpoint_sha256=sha256(checkpoint),
        tokenizer_sha256=TOKENIZER_SHA,rows=1919,
        files={p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file()}))
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(out.iterdir()):z.write(p,p.name)
    print(archive)
    return archive


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--root',default=str(ROOT))
    a=p.parse_args();export(a.checkpoint,a.root)
