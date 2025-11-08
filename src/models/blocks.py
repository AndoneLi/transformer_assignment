import torch, torch.nn as nn
from .attention import MultiHeadAttention, build_local_mask, build_alibi_bias

class Positionless(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
    def forward(self, x): return self.fc2(self.dropout(self.act(self.fc1(x))))

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, attn_type='local', window_size=64, rel_pos='alibi'):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.attn_type = attn_type; self.window = window_size; self.rel_pos = rel_pos
        self.n_heads = n_heads

    def forward(self, x):
        B,L,D = x.size()
        if self.attn_type == 'local':
            mask = build_local_mask(L, L, self.window, causal=False, device=x.device)
        else:
            mask = torch.ones(L, L, dtype=torch.bool, device=x.device)
        attn_bias = build_alibi_bias(self.n_heads, L, L, causal=False, device=x.device) if self.rel_pos=='alibi' else None
        sa, _ = self.self_attn(x, x, mask=mask, attn_bias=attn_bias)
        x = self.norm1(x + self.drop(sa))
        ff = self.ffn(x)
        x = self.norm2(x + self.drop(ff))
        return x

class DecoderBlock(nn.Module):
    """作业要求的 Decoder Block：包含自注意力（因果掩码 + 可选局部）、交叉注意力、前馈；全部为 Pre-LN。"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, attn_type='local', window_size=64, rel_pos='alibi'):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model); self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.attn_type = attn_type; self.window = window_size; self.rel_pos = rel_pos
        self.n_heads = n_heads

    def forward(self, x, mem):
        B,L,D = x.size(); M = mem.size(1)
        # 自注意力：因果 + 局部
        if self.attn_type == 'local':
            mask = build_local_mask(L, L, self.window, causal=True, device=x.device)
        else:
            i = torch.arange(L, device=x.device).unsqueeze(1); j = torch.arange(L, device=x.device).unsqueeze(0)
            mask = (i >= j)
        attn_bias = build_alibi_bias(self.n_heads, L, L, causal=True, device=x.device) if self.rel_pos=='alibi' else None
        sa, _ = self.self_attn(x, x, mask=mask, attn_bias=attn_bias)
        x = self.norm1(x + self.drop(sa))
        # 交叉注意力：全连接（解码器每个位置可看见全部编码记忆）
        mem_mask = torch.ones(L, M, dtype=torch.bool, device=x.device)
        ca, _ = self.cross_attn(x, mem, mask=mem_mask, attn_bias=None)
        x = self.norm2(x + self.drop(ca))
        ff = self.ffn(x)
        x = self.norm3(x + self.drop(ff))
        return x
