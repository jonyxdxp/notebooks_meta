
# cog_arch/decoder.py




import torch
import torch.nn as nn
import torch.nn.functional as F





class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate, x = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * x)







class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, attention_dropout=0.0):
        super().__init__()

        # Masked self-attention — attends only to past tokens
        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

        self.ffn = SwiGLU(hidden_size, intermediate_size)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, causal_mask, key_padding_mask=None):
        normed = self.norm1(hidden_states)

        # causal_mask is an (T, T) additive mask that blocks future positions
        attn_out, _ = self.self_attn(
            normed, normed, normed,
            attn_mask=causal_mask,           # ← the key difference vs encoder
            key_padding_mask=key_padding_mask,
            is_causal=True,
        )

        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.ffn(self.norm2(hidden_states))
        return hidden_states









class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_heads=12,
                 intermediate_size=None,
                 num_layers=6, max_seq_len=2048):
        super().__init__()

        if intermediate_size is None:
            intermediate_size = hidden_size * 4

        self.token_embedding    = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        self.layers = nn.ModuleList([
            DecoderLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])

        self.norm   = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)  # ← produces logits

        # Tie input embeddings to output projection (common practice, saves params)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids, attention_mask=None):
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        hidden_states = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
        )

        # Build causal (lower-triangular) mask once for the whole forward pass.
        # nn.MultiheadAttention expects an additive float mask:
        #   0.0  → allowed to attend
        #  -inf  → blocked (future token)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_mask, key_padding_mask)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)   # (batch, seq_len, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens=100, temperature=1.0, top_k=None):
        """Simple autoregressive greedy / top-k sampler."""
        for _ in range(max_new_tokens):
            logits = self(token_ids)             # (B, T, vocab)
            next_logits = logits[:, -1, :] / temperature  # last position only

            if top_k is not None:
                # Zero out everything outside the top-k
                values, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < values[:, -1:]] = float("-inf")

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            token_ids = torch.cat([token_ids, next_token], dim=1)

        return token_ids