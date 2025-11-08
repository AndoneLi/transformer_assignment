import torch, torch.nn as nn
from .blocks import EncoderBlock, DecoderBlock, Positionless

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len, dropout, attn_type, window_size, rel_pos):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = Positionless()
        self.blocks = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff, dropout, attn_type, window_size, rel_pos) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        x = self.pos(self.embed(x))
        for blk in self.blocks: x = blk(x)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len, dropout, attn_type, window_size, rel_pos):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = Positionless()
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff, dropout, attn_type, window_size, rel_pos) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    def forward(self, y, mem):
        x = self.pos(self.embed(y))
        for blk in self.blocks: x = blk(x, mem)
        x = self.norm(x)
        return self.lm_head(x)

class EncoderOnlyLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len, dropout, attn_type, window_size, rel_pos):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, n_heads, d_ff, n_layers, max_len, dropout, attn_type, window_size, rel_pos)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    def forward(self, x):
        h = self.encoder(x); return self.lm_head(h)

class EncoderDecoder(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model, n_heads, d_ff, n_enc_layers, n_dec_layers, max_len, dropout, attn_type, window_size, rel_pos):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, n_heads, d_ff, n_enc_layers, max_len, dropout, attn_type, window_size, rel_pos)
        self.decoder = Decoder(tgt_vocab, d_model, n_heads, d_ff, n_dec_layers, max_len, dropout, attn_type, window_size, rel_pos)
    def forward(self, src, tgt):
        mem = self.encoder(src); return self.decoder(tgt, mem)
