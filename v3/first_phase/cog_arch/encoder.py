# cog_arch/encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F





class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate, x = self.w1(x).chunk(2, dim=-1)  # split here, not in Sequential
        return self.w2(F.silu(gate) * x)







class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, attention_dropout=0.0):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=attention_dropout,
            batch_first=True
        )

        self.ffn = SwiGLU(hidden_size, intermediate_size)  # ← fixed

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, key_padding_mask=None):
        normed = self.norm1(hidden_states)
        attn_out, _ = self.self_attn(normed, normed, normed,
                                     key_padding_mask=key_padding_mask)
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.ffn(self.norm2(hidden_states))
        return hidden_states






class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_heads=12,
                 intermediate_size=None,   # defaults to 4x hidden_size
                 num_layers=6, max_seq_len=2048):
        super().__init__()

        if intermediate_size is None:
            intermediate_size = hidden_size * 4   # 256*4=1024, not hardcoded 3072

        self.token_embedding    = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        self.layers = nn.ModuleList([
            EncoderLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, token_ids, attention_mask=None):
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        hidden_states = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
        )

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        for layer in self.layers:
            hidden_states = layer(hidden_states, key_padding_mask)

        return self.norm(hidden_states)