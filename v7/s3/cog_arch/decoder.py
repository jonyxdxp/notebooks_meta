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
    def __init__(self, hidden_size, num_heads, intermediate_size,
                 context_dim=256, attention_dropout=0.0):
        super().__init__()

        # 1. Masked self-attention — attends to past tokens
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

        # 2. Cross-attention — attends to z_fused
        # z_fused lives in context_dim (S1/S2 space, 256)
        # decoder hidden may be different size → project via kdim/vdim
        self.cross_attn = nn.MultiheadAttention(
            embed_dim  = hidden_size,
            num_heads  = num_heads,
            kdim       = context_dim,
            vdim       = context_dim,
            dropout    = attention_dropout,
            batch_first= True,
        )

        # 3. FFN
        self.ffn = SwiGLU(hidden_size, intermediate_size)

        # Norms — one per sublayer
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, z_fused, causal_mask,
                key_padding_mask=None):
        """
        hidden_states : (B, T, hidden_size)  — token representations
        z_fused       : (B, 1, context_dim)  — conditioning from S1/S2/prior
        causal_mask   : (T, T)               — blocks future tokens
        """

        # 1. Self-attention
        normed = self.norm1(hidden_states)
        attn_out, _ = self.self_attn(
            normed, normed, normed,
            attn_mask        = causal_mask,
            key_padding_mask = key_padding_mask,
            is_causal        = True,
        )
        hidden_states = hidden_states + attn_out

        # 2. Cross-attention on z_fused
        # Query: decoder tokens
        # Key/Value: z_fused (single vector expanded to seq of 1)
        normed = self.norm2(hidden_states)
        cross_out, _ = self.cross_attn(
            query = normed,
            key   = z_fused,
            value = z_fused,
        )
        hidden_states = hidden_states + cross_out

        # 3. FFN
        hidden_states = hidden_states + self.ffn(self.norm3(hidden_states))

        return hidden_states


class Decoder(nn.Module):
    """
    Cross-attention decoder conditioned on z_fused.

    z_fused : (B, D) from PoE(dynamics, prior)
              conditions every layer via cross-attention
              D = S1 hidden_size = 256
    """
    def __init__(
        self,
        vocab_size,
        hidden_size       = 256,
        num_heads         = 4,
        intermediate_size = None,
        num_layers        = 4,
        max_seq_len       = 128,
        context_dim       = 256,    # must match S1 hidden_size
    ):
        super().__init__()

        if intermediate_size is None:
            intermediate_size = hidden_size * 4

        self.hidden_size = hidden_size
        self.context_dim = context_dim

        self.token_embedding    = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        self.layers = nn.ModuleList([
            DecoderLayer(
                hidden_size       = hidden_size,
                num_heads         = num_heads,
                intermediate_size = intermediate_size,
                context_dim       = context_dim,
            )
            for _ in range(num_layers)
        ])

        self.norm    = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight   # weight tying

    def forward(self, token_ids, z_fused, attention_mask=None):
        """
        token_ids    : (B, T)    — input token ids (teacher-forced during training)
        z_fused      : (B, D)    — conditioning vector from PoE
        attention_mask: (B, T)   — 1 for real tokens, 0 for padding

        Returns logits : (B, T, vocab_size)
        """
        B, T   = token_ids.shape
        device = token_ids.device

        # Token + position embeddings
        positions     = torch.arange(T, device=device).unsqueeze(0)
        hidden_states = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
        )

        # z_fused: (B, D) → (B, 1, D) for cross-attention
        # Each token attends to z_fused as a single context vector
        z_ctx = z_fused.unsqueeze(1)   # (B, 1, context_dim)

        # Causal mask
        causal_mask = torch.triu(
            torch.full((T, T), float('-inf'), device=device),
            diagonal=1,
        )

        # Padding mask
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        # Forward through layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states    = hidden_states,
                z_fused          = z_ctx,
                causal_mask      = causal_mask,
                key_padding_mask = key_padding_mask,
            )

        hidden_states = self.norm(hidden_states)
        logits        = self.lm_head(hidden_states)   # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids, z_fused, max_new_tokens=50,
                 temperature=1.0, top_k=50):
        """
        Autoregressive generation conditioned on z_fused.

        prompt_ids   : (B, T_prompt) — starting tokens
        z_fused      : (B, D)        — conditioning from PoE
        """
        token_ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            logits      = self(token_ids, z_fused)         # (B, T, vocab)
            next_logits = logits[:, -1, :] / temperature   # last position

            if top_k is not None:
                values, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < values[:, -1:]] = float('-inf')

            probs      = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)   # (B, 1)
            token_ids  = torch.cat([token_ids, next_token], dim=1)

            # Stop if all sequences generated [SEP]
            if (next_token == 102).all():   # 102 = [SEP] in BERT tokenizer
                break

        return token_ids