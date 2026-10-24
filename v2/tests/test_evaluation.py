import json
from pathlib import Path
import pytest
from hw2_safe_v2 import run_numeric,same_answer
from hw2_eval_v2 import paired_compare


@pytest.mark.parametrize('code,answer',[
 ('def solve():\n return 6 * 7',42),
 ('def solve() -> int:\n x=3\n x*=2\n return x',6),
 ('def solve():\n x=0\n for i in range(4):\n  x+=i\n return x',6),
 ('def solve():\n return 2/3',2/3)])
def test_arithmetic(code,answer):assert run_numeric(code)==('ok',answer)


@pytest.mark.parametrize('code',[
 'def solve():\n return [1]*1000000000',
 'def solve():\n x=[1]\n x*=1000000000\n return 1',
 'def solve():\n x=10\n x**=1000000000\n return x',
 'def solve():\n while True:\n  pass\n return 1',
 '@bad\ndef solve():\n return 1',
 'def solve(x=bad()):\n return 1',
 'def solve():\n import os\n return 1',
 'def solve():\n return __import__("os").system("echo bad")',
 'def solve():\n return "x" * 1000000000'])
def test_bounded_rejection(code):assert run_numeric(code)[0]=='unsupported'


@pytest.mark.parametrize('value',['True','"42"','(1, 2)','[1,2]'])
def test_nonnumeric(value):assert run_numeric('def solve():\n return '+value)[0]=='nonnumeric'


def test_tolerance_and_comparison():
    assert same_answer(1+1.9e-6,1)
    a=[dict(row_id=i,reference_status='ok',reference_answer=i,match=False) for i in range(5)]
    b=[dict(**{**r,'match':True}) for r in a]
    r=paired_compare(a,b,reps=50)
    assert r['gained']==5 and r['delta']==1
    with pytest.raises(ValueError):paired_compare(a,b[:-1])


def test_archive_audit_recomputed_counts():
    root=Path(__file__).parents[1]
    rows=json.loads((root/'evidence/final_1024_details.json').read_text())
    n=k=0
    for r in rows:
        rs,rv=run_numeric(r['reference']);ps,pv=run_numeric(r['prediction'])
        n+=rs=='ok';k+=rs=='ok' and ps=='ok' and same_answer(pv,rv)
    assert (n,k)==(819,554)
