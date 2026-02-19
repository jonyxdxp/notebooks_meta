



#  modified "Local Encoder" from BLT paper, to act as a normal encoder for text without the byte logic






import torch
import torch.nn as nn
import torch.nn.functional as F






class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, attention_dropout=0.0):
        super().__init__()

        # Full self-attention (bidirectional)
        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=attention_dropout,
            batch_first=True
        )

        # Feed-forward (SwiGLU style)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size * 2),
            nn.SiLU(),
            nn.Linear(intermediate_size, hidden_size)
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, key_padding_mask=None):
        # Self-attention (no causal mask!)
        normed = self.norm1(hidden_states)
        attn_out, _ = self.self_attn(
            normed,
            normed,
            normed,
            key_padding_mask=key_padding_mask
        )
        hidden_states = hidden_states + attn_out

        # Feed-forward
        hidden_states = hidden_states + self.ffn(self.norm2(hidden_states))
        return hidden_states









class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_heads=12,
                 intermediate_size=3072, num_layers=6, max_seq_len=2048):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        self.layers = nn.ModuleList([
            EncoderLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, token_ids, attention_mask=None):
        """
        token_ids: (batch, seq_len)
        attention_mask: (batch, seq_len) where 1 = keep, 0 = pad
        """

        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        # Token + position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        hidden_states = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
        )

        # Convert attention mask for PyTorch MHA
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        for layer in self.layers:
            hidden_states = layer(hidden_states, key_padding_mask)

        return self.norm(hidden_states)
