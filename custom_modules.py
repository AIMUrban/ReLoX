import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.emb_dim = emb_dim
        pos_encoding = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * -(math.log(10000.0) / emb_dim))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)
        self.dropout = nn.Dropout(0.1)

    def forward(self, out):
        out = out + self.pos_encoding[:, :out.size(1)].detach()
        out = self.dropout(out)

        return out

class TransEncoder(nn.Module):
    def __init__(self, input_dim):
        super(TransEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim,
                                                   activation='gelu',
                                                   batch_first=True,
                                                   dim_feedforward=input_dim*2,
                                                   nhead=2,
                                                   dropout=0)

        self.pe = PositionalEncoding(input_dim)

        encoder_norm = nn.LayerNorm(input_dim)

        # Transformer Encoder
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=2,
                                             norm=encoder_norm)

    def forward(self, x, src_key_padding_mask=None, causal_mask=None):
        x = x * math.sqrt(x.size(-1))
        x = self.pe(x)

        out = self.encoder(x, src_key_padding_mask=src_key_padding_mask, mask=causal_mask)

        return out


class TokenAttn(nn.Module):
    """
    Token-level attention (self or cross).
    q_tokens: [B, Nq, D]
    kv_tokens: [B, Nk, D] or None (self-attn when None)
    key_padding_mask: [B, Nk] bool, True means masked
    """
    def __init__(self, dim, num_heads=2, attn_drop=0.0, proj_drop=0.0, pre_ln=True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pre_ln = pre_ln

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        if self.num_heads > 1:
            self.out_proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        if self.pre_ln:
            self.ln_q = nn.LayerNorm(dim)
            self.ln_kv = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim * 2, dim),
        )
        self.out_ln = nn.LayerNorm(dim)

    def forward(self, q_tokens, kv_tokens=None, key_padding_mask=None):
        is_self = kv_tokens is None
        residual = q_tokens
        if is_self:
            kv_tokens = q_tokens

        if self.pre_ln:
            q_tokens = self.ln_q(q_tokens)
            kv_tokens = self.ln_kv(kv_tokens)

        B, Nq, D = q_tokens.shape
        Nk = kv_tokens.size(1)

        q = self.q_proj(q_tokens).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_tokens).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_tokens).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,Nq,Nk]
        if key_padding_mask is not None:
            mask = key_padding_mask.view(B, 1, 1, Nk)
            attn = attn.masked_fill(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # [B,H,Nq,Dh]
        out = out.transpose(1, 2).contiguous().view(B, Nq, D)
        if self.num_heads > 1:
            out = self.out_proj(out)
        out = self.proj_drop(out)

        # if is_self:
        #     out = out + residual
        #     out = out + self.ffn(out)
        #     out = self.out_ln(out)

        return out
