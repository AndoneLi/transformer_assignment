import math, torch
import torch.nn as nn

def build_local_mask(Lq, Lk, window, causal=False, device=None):
    i = torch.arange(Lq, device=device).unsqueeze(1)
    j = torch.arange(Lk, device=device).unsqueeze(0)
    dist = i - j
    if causal:
        valid = (dist >= 0) & (dist <= window)
    else:
        valid = dist.abs() <= window
    return valid  # (Lq, Lk) bool

def build_alibi_bias(n_heads, Lq, Lk, causal=True, device=None):
    def get_slopes(n):
        import math
        def power_of_two(n): 
            return 2 ** math.floor(math.log2(n))
        m = power_of_two(n)
        slopes = [2**(-8*i/m) for i in range(m)]
        if m < n: slopes += [slopes[-1]] * (n-m)
        return torch.tensor(slopes)
    slopes = get_slopes(n_heads).to(device)
    i = torch.arange(Lq, device=device).unsqueeze(1)
    j = torch.arange(Lk, device=device).unsqueeze(0)
    if causal: rel = (i - j).clamp(min=0)
    else: rel = (i - j).abs()
    bias = -rel.unsqueeze(0) * slopes.view(n_heads, 1, 1)  # (h,Lq,Lk)
    return bias

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None, attn_bias=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B,h,Lq,Lk)
        if attn_bias is not None:
            scores = scores + attn_bias.unsqueeze(0)  # broadcast batch
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(1), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, V), attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention(dropout)

    def _split(self, x):
        B,L,D = x.size()
        return x.view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # (B,h,L,d)

    def _merge(self, x):
        B,h,L,d = x.size()
        return x.transpose(1, 2).contiguous().view(B, L, h*d)

    def forward(self, x_q, x_kv, mask=None, attn_bias=None):
        Q = self._split(self.W_q(x_q))
        K = self._split(self.W_k(x_kv))
        V = self._split(self.W_v(x_kv))
        out, attn = self.attn(Q, K, V, mask=mask, attn_bias=attn_bias)
        out = self._merge(out)
        return self.out_proj(out), attn
