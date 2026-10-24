"""One NVIDIA T4, greedy FP16, full 1919-row wall-clock measurement and code-only JSONL."""
import argparse
import ast
import json
import statistics
import time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from hw2_model_v2 import CodeTransformerV2,ModelConfig,PAD,EOS
from hw2_common_v2 import sha256,write_json,TOKENIZER_SHA,environment


@torch.inference_mode()
def generate_all(model,rows,tok,out,batch_size,max_new_tokens):
    torch.cuda.synchronize();start=time.perf_counter()
    indexed=sorted(enumerate(rows),key=lambda p:len(p[1]['input_ids']))
    results=[None]*len(rows)
    for i in range(0,len(rows),batch_size):
        group=indexed[i:i+batch_size]
        src=torch.full((len(group),max(len(r['input_ids']) for _,r in group)),PAD,dtype=torch.long)
        for j,(_,r) in enumerate(group):
            src[j,:len(r['input_ids'])]=torch.tensor(r['input_ids'])
        output=model.generate(src.cuda(),max_new_tokens=max_new_tokens).cpu().tolist()
        for (ix,r),ids in zip(group,output):
            eos=EOS in ids;ids=ids[:ids.index(EOS)] if eos else ids
            results[ix]=dict(row_id=r['row_id'],code=tok.decode(ids,skip_special_tokens=True),eos=eos)
    with Path(out).open('w',encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps({'code':r['code']},ensure_ascii=False)+'\n')
    torch.cuda.synchronize()
    return results,time.perf_counter()-start


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',default='.')
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--max-new-tokens',type=int,default=384)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--output',default='t4_results')
    a=p.parse_args();root=Path(a.bundle).resolve();out=Path(a.output).resolve()
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('This acceptance benchmark must run on an actual NVIDIA T4')
    if a.repeats<1 or a.batch_size<1 or a.max_new_tokens<1:
        raise ValueError('Positive settings required')
    out.mkdir(exist_ok=True,parents=True)
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if Path(name).name!=name or sha256(root/name)!=digest:
            raise ValueError(f'Export hash mismatch: {name}')
    if sha256(root/'tokenizer.json')!=TOKENIZER_SHA:
        raise ValueError('Tokenizer mismatch')
    rows=json.loads((root/'test_inputs.json').read_text())
    if [r['row_id'] for r in rows]!=list(range(1919)):
        raise ValueError('Missing, duplicate or unordered test row IDs')
    tok=Tokenizer.from_file(str(root/'tokenizer.json'))
    load_start=time.perf_counter()
    s=torch.load(root/'model_fp16.pt',map_location='cpu',weights_only=True)
    c=ModelConfig(**s['model_config'])
    if a.max_new_tokens>c.max_positions:
        raise ValueError('Generation exceeds model positions')
    for r in rows:
        if (not r['input_ids'] or r['input_ids'][-1]!=EOS or len(r['input_ids'])>c.max_positions
            or any(t<0 or t>=c.vocab_size for t in r['input_ids'])):
            raise ValueError('Invalid test token IDs')
    model=CodeTransformerV2(c).half().cuda().eval()
    model.load_state_dict(s['model'],strict=True)
    torch.cuda.synchronize();load_seconds=time.perf_counter()-load_start
    del s
    torch.cuda.reset_peak_memory_stats()
    # Separate warm-up; report it explicitly, along with the first full run.
    _,warmup=generate_all(model,rows[:16],tok,out/'warmup.jsonl',16,min(32,a.max_new_tokens))
    times=[];digests=[];details=None
    for i in range(a.repeats):
        file=out/f'predictions_run{i+1}.jsonl'
        details,seconds=generate_all(model,rows,tok,file,a.batch_size,a.max_new_tokens)
        times.append(seconds);digests.append(sha256(file))
        print(json.dumps(dict(run=i+1,seconds=seconds,eos=sum(r['eos'] for r in details))),flush=True)
    syntax_ok=0
    for r in details:
        try:ast.parse(r['code']);syntax_ok+=1
        except (SyntaxError,RecursionError):pass
    report=dict(environment=environment(),rows=len(rows),checkpoint_step=manifest['checkpoint_step'],
        bundle_manifest_sha256=sha256(root/'manifest.json'),batch_size=a.batch_size,max_new_tokens=a.max_new_tokens,
        full_run_seconds=times,median_seconds=statistics.median(times),max_seconds=max(times),
        within_180_seconds=max(times)<=180,within_150_second_margin=max(times)<=150,
        timing_scope='sort, CPU padding, transfers, encoder, cached decoder, detokenization, JSONL write',
        model_load_seconds=load_seconds,warmup_seconds=warmup,
        load_plus_warmup_plus_first_full_seconds=load_seconds+warmup+times[0],
        repeated_outputs_identical=len(set(digests))==1,eos=sum(r['eos'] for r in details),syntax_ok=syntax_ok,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,predictions_sha256=digests)
    write_json(out/'t4_benchmark.json',report)
    write_json(out/'prediction_metadata.json',details)
    (out/'submission.jsonl').write_bytes((out/'predictions_run1.jsonl').read_bytes())
    print(json.dumps(report,indent=2))
    if not report['within_180_seconds']:
        raise SystemExit('FAIL: full generation exceeds 180 seconds. See saved benchmark.')
    if not report['repeated_outputs_identical']:
        raise SystemExit('FAIL: repeated outputs differ. Investigate reproducibility before submitting.')


if __name__=='__main__':main()
