import argparse
import json
from pathlib import Path
from hw2_eval_v2 import paired_compare
from hw2_common_v2 import write_json


def compare(a,b):
    a,b=Path(a),Path(b)
    ma,mb=json.loads(a.read_text()),json.loads(b.read_text())
    for field in ('validation_sha256','role','max_new_tokens'):
        if ma.get(field)!=mb.get(field):raise ValueError(f'Comparison mismatch: {field}')
    da=json.loads(a.with_suffix('.details.json').read_text())
    db=json.loads(b.with_suffix('.details.json').read_text())
    result=paired_compare(da,db)
    result.update(a=str(a),b=str(b),a_tokens=ma.get('target_tokens'),b_tokens=mb.get('target_tokens'),
        same_training_token_budget=ma.get('target_tokens') is not None and ma.get('target_tokens')==mb.get('target_tokens'),
        same_source_tokens=ma.get('source_tokens') is not None and ma.get('source_tokens')==mb.get('source_tokens'))
    result['confirm_quality_gate']=result['delta']>=.02 and result['ci95'][0]>0
    result['note']='Quality gate is necessary, not sufficient: require full train and actual T4 <=150 s margin; AST metric is a proxy'
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('a');p.add_argument('b');p.add_argument('--output',required=True)
    args=p.parse_args();r=compare(args.a,args.b);write_json(args.output,r);print(json.dumps(r,indent=2))
