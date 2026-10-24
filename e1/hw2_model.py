"""Small from-scratch encoder-decoder Transformer with cached greedy decoding."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

PAD, EOS, UNK, BOS = 0, 1, 2, 3


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    dim: int = 384
    heads: int = 6
    encoder_layers: int = 4
    decoder_layers: int = 4
    ff_dim: int = 1536
    dropout: float = 0.1
    max_positions: int = 2048


def sinusoidal_positions(n, d):
    pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.) / d))
    x = pos * frequencies
    result = torch.zeros(n, d, dtype=torch.float32)
    result[:, 0::2] = torch.sin(x)
    result[:, 1::2] = torch.cos(x[:, :result[:, 1::2].shape[1]])
    return result


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.drop = dropout
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def project(self, layer, x):
        b, t, _ = x.shape
        return layer(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, x, memory=None, valid=None, causal=False):
        m = x if memory is None else memory
        q = self.project(self.q, x)
        k = self.project(self.k, m)
        v = self.project(self.v, m)
        mask = None if valid is None else valid[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal,
                                           dropout_p=self.drop if self.training else 0.)
        return self.out(y.transpose(1, 2).contiguous().view(x.shape))

    def step(self, x, old_kv=None, cross_kv=None, valid=None):
        q = self.project(self.q, x)
        if cross_kv is None:
            k = self.project(self.k, x)
            v = self.project(self.v, x)
            if old_kv is not None:
                k = torch.cat((old_kv[0], k), dim=2)
                v = torch.cat((old_kv[1], v), dim=2)
        else:
            k, v = cross_kv
        mask = None if valid is None else valid[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False, dropout_p=0.)
        out = self.out(y.transpose(1, 2).contiguous().view(x.shape))
        return out, (k, v)


class FFN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(c.dim, c.ff_dim), nn.GELU(),
                                 nn.Linear(c.ff_dim, c.dim), nn.Dropout(c.dropout))

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(c.dim), nn.LayerNorm(c.dim)
        self.attn = Attention(c.dim, c.heads, c.dropout)
        self.ff = FFN(c)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, mask):
        x = x + self.drop(self.attn(self.n1(x), valid=mask))
        return x + self.ff(self.n2(x))


class DecoderLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.n2, self.n3 = (nn.LayerNorm(c.dim) for _ in range(3))
        self.self_attn = Attention(c.dim, c.heads, c.dropout)
        self.cross_attn = Attention(c.dim, c.heads, c.dropout)
        self.ff = FFN(c)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, memory, source_valid):
        x = x + self.drop(self.self_attn(self.n1(x), causal=True))
        x = x + self.drop(self.cross_attn(self.n2(x), memory=memory, valid=source_valid))
        return x + self.ff(self.n3(x))

    def step(self, x, self_kv, cross_kv, source_valid):
        z, new_kv = self.self_attn.step(self.n1(x), old_kv=self_kv)
        x = x + z
        z, _ = self.cross_attn.step(self.n2(x), cross_kv=cross_kv, valid=source_valid)
        x = x + z
        x = x + self.ff(self.n3(x))
        return x, new_kv


class CodeTransformer(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.emb = nn.Embedding(c.vocab_size, c.dim, padding_idx=PAD)
        self.register_buffer('pos', sinusoidal_positions(c.max_positions, c.dim), persistent=False)
        self.drop = nn.Dropout(c.dropout)
        self.encoder = nn.ModuleList([EncoderLayer(c) for _ in range(c.encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(c) for _ in range(c.decoder_layers)])
        self.enc_norm = nn.LayerNorm(c.dim)
        self.dec_norm = nn.LayerNorm(c.dim)
        self.head = nn.Linear(c.dim, c.vocab_size, bias=False)
        self.apply(self._init)
        self.head.weight = self.emb.weight  # 3-way tied: encoder/decoder/output
        with torch.no_grad():
            self.emb.weight[PAD].zero_()

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0., std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def _embed(self, x, start=0):
        if start + x.shape[1] > self.config.max_positions:
            raise ValueError(f'Length {start + x.shape[1]} > max_positions={self.config.max_positions}; never truncate silently')
        z = self.emb(x) * math.sqrt(self.config.dim)
        z = z + self.pos[start:start + x.shape[1]].to(dtype=z.dtype)
        return self.drop(z)

    def encode(self, src):
        valid = src.ne(PAD)
        x = self._embed(src)
        for layer in self.encoder:
            x = layer(x, valid)
        return self.enc_norm(x), valid

    def forward(self, src, dec):
        memory, source_valid = self.encode(src)
        x = self._embed(dec)
        for layer in self.decoder:
            x = layer(x, memory, source_valid)
        return self.head(self.dec_norm(x))

    def prepare_cache(self, src):
        memory, valid = self.encode(src)
        crosses = [(layer.cross_attn.project(layer.cross_attn.k, memory),
                    layer.cross_attn.project(layer.cross_attn.v, memory)) for layer in self.decoder]
        return valid, crosses, [None] * len(self.decoder)

    def decode_step(self, last_token, position, cache):
        if self.training:
            raise RuntimeError('decode_step is only for model.eval()')
        valid, crosses, self_kvs = cache
        x = self._embed(last_token, start=position)
        new_kvs = []
        for layer, old_kv, cross_kv in zip(self.decoder, self_kvs, crosses):
            x, new_kv = layer.step(x, old_kv, cross_kv, valid)
            new_kvs.append(new_kv)
        return self.head(self.dec_norm(x)), (valid, crosses, new_kvs)

    @torch.inference_mode()
    def generate(self, src, max_new_tokens=256):
        self.eval()
        cache = self.prepare_cache(src)
        token = torch.full((len(src), 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(len(src), device=src.device, dtype=torch.bool)
        outputs = []
        for step in range(min(max_new_tokens, self.config.max_positions)):
            logits, cache = self.decode_step(token, step, cache)
            token = logits[:, -1, :].argmax(-1, keepdim=True)
            token = torch.where(done[:, None], EOS, token)
            outputs.append(token)
            done |= token.squeeze(1).eq(EOS)
            if bool(done.all()):
                break
        return torch.cat(outputs, dim=1) if outputs else src.new_zeros((len(src), 0))
