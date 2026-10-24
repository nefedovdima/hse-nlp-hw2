"""From-scratch encoder-decoder; SDPA, tied embeddings, static inference KV cache."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

PAD, EOS, UNK, BOS = 0, 1, 2, 3


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    dim: int = 768
    heads: int = 12
    encoder_layers: int = 8
    decoder_layers: int = 4
    ff_dim: int = 2048
    ffn: str = 'swiglu'
    norm: str = 'layer'
    position: str = 'sinusoidal'
    tying: str = 'all'
    dropout: float = 0.1
    max_positions: int = 2048

    def __post_init__(self):
        assert self.dim % self.heads == 0 and self.dim % 2 == 0
        assert (self.dim // self.heads) % 2 == 0
        assert self.ffn in ('gelu', 'swiglu')
        assert self.norm in ('layer', 'rms')
        assert self.position in ('sinusoidal', 'rope')
        assert self.tying in ('all', 'decoder', 'none')
        assert self.encoder_layers > 0 and self.decoder_layers > 0


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight.to(x.dtype)


def norm(c):
    return nn.LayerNorm(c.dim) if c.norm == 'layer' else RMSNorm(c.dim)


class Attention(nn.Module):
    def __init__(self, c, rotary=False):
        super().__init__()
        self.heads, self.dh, self.drop = c.heads, c.dim // c.heads, c.dropout
        self.rotary = rotary
        self.q = nn.Linear(c.dim, c.dim, bias=False)
        self.k = nn.Linear(c.dim, c.dim, bias=False)
        self.v = nn.Linear(c.dim, c.dim, bias=False)
        self.out = nn.Linear(c.dim, c.dim, bias=False)
        if rotary:
            a = torch.arange(c.max_positions).float()[:, None]
            b = 10000. ** (-torch.arange(0, self.dh, 2).float() / self.dh)
            self.register_buffer('cos', (a * b).cos(), persistent=False)
            self.register_buffer('sin', (a * b).sin(), persistent=False)

    def project(self, layer, x):
        return layer(x).view(x.size(0), x.size(1), self.heads, self.dh).transpose(1, 2)

    def rotate(self, x, start=0):
        if not self.rotary:
            return x
        cs = self.cos[start:start+x.size(2)].to(x.dtype)[None, None]
        sn = self.sin[start:start+x.size(2)].to(x.dtype)[None, None]
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a * cs - b * sn, a * sn + b * cs), -1).flatten(-2)

    def attend(self, q, k, v, valid=None, causal=False):
        mask = None if valid is None else valid[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                is_causal=causal, dropout_p=self.drop if self.training else 0.)
        y = y.transpose(1, 2).contiguous().flatten(2)
        return self.out(y)

    def forward(self, x, memory=None, valid=None, causal=False):
        m = x if memory is None else memory
        return self.attend(self.rotate(self.project(self.q, x)),
            self.rotate(self.project(self.k, m)), self.project(self.v, m), valid, causal)

    def step(self, x, position, kv=None, cross=None, valid=None):
        q = self.rotate(self.project(self.q, x), position)
        if cross is not None:
            k, v = cross
        else:
            k_new = self.rotate(self.project(self.k, x), position)
            v_new = self.project(self.v, x)
            kv[0][:, :, position:position+1].copy_(k_new)
            kv[1][:, :, position:position+1].copy_(v_new)
            k, v = kv[0][:, :, :position+1], kv[1][:, :, :position+1]
        # A single current query can attend to the entire prefix. is_causal=True
        # would incorrectly mask all but the first key for this non-square case.
        return self.attend(q, k, v, valid)


class FFN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.up = nn.Linear(c.dim, c.ff_dim)
        self.gate = nn.Linear(c.dim, c.ff_dim) if c.ffn == 'swiglu' else None
        self.down = nn.Linear(c.ff_dim, c.dim)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        h = F.gelu(self.up(x)) if self.gate is None else F.silu(self.gate(x)) * self.up(x)
        return self.drop(self.down(h))


class EncoderLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.n2 = norm(c), norm(c)
        self.attn = Attention(c, c.position == 'rope')
        self.ff, self.drop = FFN(c), nn.Dropout(c.dropout)

    def forward(self, x, valid):
        x = x + self.drop(self.attn(self.n1(x), valid=valid))
        return x + self.ff(self.n2(x))


class DecoderLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.n2, self.n3 = norm(c), norm(c), norm(c)
        self.self_attn = Attention(c, c.position == 'rope')
        # Source/target position indices have no shared coordinate system.
        self.cross_attn = Attention(c, rotary=False)
        self.ff, self.drop = FFN(c), nn.Dropout(c.dropout)

    def forward(self, x, memory, valid):
        x = x + self.drop(self.self_attn(self.n1(x), causal=True))
        x = x + self.drop(self.cross_attn(self.n2(x), memory=memory, valid=valid))
        return x + self.ff(self.n3(x))

    def step(self, x, position, kv, cross, valid):
        x = x + self.self_attn.step(self.n1(x), position, kv=kv)
        x = x + self.cross_attn.step(self.n2(x), position, cross=cross, valid=valid)
        return x + self.ff(self.n3(x))


class CodeTransformerV2(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = c = config or ModelConfig()
        self.enc_emb = nn.Embedding(c.vocab_size, c.dim, padding_idx=PAD)
        self.dec_emb = self.enc_emb if c.tying == 'all' else nn.Embedding(c.vocab_size, c.dim, padding_idx=PAD)
        self.head = nn.Linear(c.dim, c.vocab_size, bias=False)
        self.encoder = nn.ModuleList([EncoderLayer(c) for _ in range(c.encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(c) for _ in range(c.decoder_layers)])
        self.enc_norm, self.dec_norm, self.drop = norm(c), norm(c), nn.Dropout(c.dropout)
        a = torch.arange(c.max_positions).float()[:, None]
        b = 10000. ** (-torch.arange(0, c.dim, 2).float() / c.dim)
        pos = torch.stack(((a*b).sin(), (a*b).cos()), -1).flatten(1)
        self.register_buffer('pos', pos, persistent=False)
        self.apply(self._init)
        if c.tying != 'none':
            self.head.weight = self.dec_emb.weight
        with torch.no_grad():
            self.enc_emb.weight[PAD].zero_()
            self.dec_emb.weight[PAD].zero_()

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Embedding, nn.Linear)):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def embed(self, ids, decoder=False, start=0):
        if start < 0 or start + ids.size(1) > self.config.max_positions:
            raise ValueError('Position limit exceeded; no silent truncation')
        emb = self.dec_emb if decoder else self.enc_emb
        x = emb(ids) * math.sqrt(self.config.dim)
        if self.config.position == 'sinusoidal':
            x = x + self.pos[start:start+ids.size(1)].to(x.dtype)
        return self.drop(x)

    def encode(self, src):
        valid = src.ne(PAD)
        x = self.embed(src)
        for layer in self.encoder:
            x = layer(x, valid)
        return self.enc_norm(x), valid

    def forward(self, src, dec):
        memory, valid = self.encode(src)
        x = self.embed(dec, decoder=True)
        for layer in self.decoder:
            x = layer(x, memory, valid)
        return self.head(self.dec_norm(x))

    def prepare_cache(self, src, max_tokens=None):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('Use eval() and inference_mode() for a static KV cache')
        n = max_tokens or self.config.max_positions
        if not 0 < n <= self.config.max_positions:
            raise ValueError('Invalid cache capacity')
        memory, valid = self.encode(src)
        crosses, kvs = [], []
        for layer in self.decoder:
            attn = layer.cross_attn
            k, v = attn.project(attn.k, memory), attn.project(attn.v, memory)
            crosses.append((k, v))
            shape = (src.size(0), self.config.heads, n, self.config.dim // self.config.heads)
            kvs.append((k.new_empty(shape), v.new_empty(shape)))
        return dict(valid=valid, cross=crosses, self_kv=kvs, position=0, capacity=n)

    def decode_step(self, token, position, cache):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('Use eval() and inference_mode()')
        if token.size(1) != 1 or position != cache['position'] or position >= cache['capacity']:
            raise ValueError('Invalid incremental decode position')
        x = self.embed(token, decoder=True, start=position)
        for layer, kv, cross in zip(self.decoder, cache['self_kv'], cache['cross']):
            x = layer.step(x, position, kv, cross, cache['valid'])
        cache['position'] += 1
        return self.head(self.dec_norm(x)), cache

    @torch.inference_mode()
    def generate(self, src, max_new_tokens=384):
        self.eval()
        cache = self.prepare_cache(src, max_new_tokens)
        token = torch.full((len(src), 1), BOS, device=src.device, dtype=torch.long)
        done = torch.zeros(len(src), device=src.device, dtype=torch.bool)
        output = []
        for pos in range(max_new_tokens):
            logits, cache = self.decode_step(token, pos, cache)
            token = logits[:, -1].argmax(-1, keepdim=True)
            token = torch.where(done[:, None], EOS, token)
            done |= token[:, 0].eq(EOS)
            output.append(token)
            if (pos + 1) % 4 == 0 and bool(done.all()):
                break
        return torch.cat(output, 1)


def parameter_count(c):
    """Exact count of distinct trainable parameters, independent of PyTorch allocation."""
    d, f, e, k, v = c.dim, c.ff_dim, c.encoder_layers, c.decoder_layers, c.vocab_size
    embs = {'all': 1, 'decoder': 2, 'none': 3}[c.tying]
    ff = (2*d*f + f+d) if c.ffn == 'gelu' else (3*d*f+2*f+d)
    n = (2 if c.norm == 'layer' else 1)*d
    return embs*v*d + e*(4*d*d+ff+2*n) + k*(8*d*d+ff+3*n) + 2*n
