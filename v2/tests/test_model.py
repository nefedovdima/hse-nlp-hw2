from dataclasses import replace
import pytest
import torch
from hw2_model_v2 import CodeTransformerV2,ModelConfig,parameter_count


def tiny(**kw):
    d=dict(vocab_size=48,dim=32,heads=2,encoder_layers=2,decoder_layers=2,
        ff_dim=64,dropout=0.,max_positions=32)
    return ModelConfig(**{**d,**kw})


@pytest.mark.parametrize('ffn',['gelu','swiglu'])
@pytest.mark.parametrize('position',['sinusoidal','rope'])
@pytest.mark.parametrize('norm',['layer','rms'])
def test_cache_equivalence(ffn,position,norm):
    torch.manual_seed(7)
    m=CodeTransformerV2(tiny(ffn=ffn,position=position,norm=norm)).eval()
    src=torch.tensor([[8,9,1,0,0],[12,13,14,15,1]])
    dec=torch.tensor([[3,5,6,7,1],[3,7,8,9,1]])
    with torch.inference_mode():
        full=m(src,dec);cache=m.prepare_cache(src,5)
        parts=[m.decode_step(dec[:,i:i+1],i,cache)[0] for i in range(5)]
        torch.testing.assert_close(full,torch.cat(parts,1),atol=2e-6,rtol=2e-5)


def test_causality_and_source_padding():
    torch.manual_seed(9);m=CodeTransformerV2(tiny()).eval()
    src=torch.tensor([[8,9,1]]);dec=torch.tensor([[3,5,6,7]])
    with torch.inference_mode():
        a=m(src,dec)
        future=dec.clone();future[:,2:]=20
        torch.testing.assert_close(a[:,:2],m(src,future)[:,:2])
        padded=torch.nn.functional.pad(src,(0,4),value=0)
        torch.testing.assert_close(a,m(padded,dec),atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('tying',['all','decoder','none'])
@pytest.mark.parametrize('ffn',['gelu','swiglu'])
def test_count_and_tying(tying,ffn):
    m=CodeTransformerV2(tiny(tying=tying,ffn=ffn))
    assert sum(p.numel() for p in m.parameters())==parameter_count(m.config)
    assert (m.head.weight is m.dec_emb.weight)==(tying!='none')
    assert (m.enc_emb is m.dec_emb)==(tying=='all')


def test_cache_guards():
    m=CodeTransformerV2(tiny()).eval();src=torch.tensor([[8,1]])
    with pytest.raises(RuntimeError):m.prepare_cache(src)
    with torch.inference_mode():
        with pytest.raises(ValueError):m.prepare_cache(src,33)
        cache=m.prepare_cache(src,2)
        with pytest.raises(ValueError):m.decode_step(torch.tensor([[3]]),1,cache)
        m.decode_step(torch.tensor([[3]]),0,cache)
        with pytest.raises(ValueError):m.decode_step(torch.tensor([[3]]),0,cache)


def test_eos_for_finished_rows():
    m=CodeTransformerV2(tiny()).eval()
    def step(token,pos,cache):
        logits=torch.zeros(2,1,48)
        logits[0,0,1]=10
        logits[1,0,1 if pos>=2 else 9]=10
        return logits,cache
    m.decode_step=step
    out=m.generate(torch.tensor([[8,1],[9,1]]),8)
    assert out.tolist()==[[1,1,1,1],[9,9,1,1]]


def test_e1_equivalence_and_count():
    from e1.model import CodeTransformer,ModelConfig as C
    m=CodeTransformer(C(vocab_size=48,dim=32,heads=2,encoder_layers=2,decoder_layers=2,
                        ff_dim=64,dropout=0.,max_positions=32)).eval()
    src=torch.tensor([[7,1,0],[8,9,1]]);dec=torch.tensor([[3,4,5],[3,9,8]])
    with torch.inference_mode():
        a=m(src,dec);cache=m.prepare_cache(src);parts=[]
        for i in range(3):
            z,cache=m.decode_step(dec[:,i:i+1],i,cache);parts.append(z)
        torch.testing.assert_close(a,torch.cat(parts,1),atol=2e-6,rtol=2e-5)
    assert parameter_count(ModelConfig(dim=384,heads=6,encoder_layers=4,
        decoder_layers=4,ff_dim=1536,ffn='gelu'))==19693056


def test_all_config_counts_meta():
    import json
    from pathlib import Path
    for path in (Path(__file__).parents[1]/'configs').glob('*.json'):
        c=ModelConfig(**json.loads(path.read_text())['model'])
        with torch.device('meta'):m=CodeTransformerV2(c)
        assert sum(p.numel() for p in m.parameters())==parameter_count(c)
