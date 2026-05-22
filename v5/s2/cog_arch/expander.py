"""
v5/s2/cog_arch/expander.py

TurnExpander: maps a single turn-level vector (B, D) produced by the DM predictor
back into a sequence of token-level embeddings (B, L, D) that can be compared
directly against the target encoder's output.

Architecture:
  1. Linear projection of the context vector z into D.
  2. Add learned positional embeddings for each of the L positions.
  3. Two Transformer encoder layers refine the sequence, using the projected z
     as an additive "conditioning" term injected at every layer.

Why not a cross-attention decoder?
  We only have one conditioning vector (no key/value sequence), so full
  cross-attention degenerates to a simple additive term anyway.  Two
  self-attention layers + an additive condition is lighter and equivalent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RefineLayer(nn.Module):
    """Single pre-norm Transformer encoder layer."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn  = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed,
                                key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class TurnExpander(nn.Module):
    """
    Expands a turn-level summary vector z (B, D) into token-level embeddings
    (B, L, D) suitable for MSE comparison with the target encoder's output.

    Args:
        hidden_dim  : embedding dimension D (must match encoder hidden_size)
        max_seq_len : maximum token sequence length L (must match block_size)
        num_heads   : attention heads in the refinement layers
        num_layers  : number of refinement Transformer layers (default 2)
        dropout     : dropout rate
    """

    def __init__(
        self,
        hidden_dim:  int,
        max_seq_len: int,
        num_heads:   int = 8,
        num_layers:  int = 2,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.max_seq_len = max_seq_len

        # Project conditioning vector to D
        self.z_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Learned positional embeddings for L token positions
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)

        # Refinement layers
        self.layers = nn.ModuleList([
            _RefineLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        # Initialise positional embeddings small so early training is stable
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(
        self,
        z:    torch.Tensor,          # (B, D)  — predicted turn vector from DM
        mask: torch.Tensor | None = None,  # (B, L) bool/int attention mask
    ) -> torch.Tensor:
        """
        Returns:
            expanded : (B, L, D)  — predicted token-level embeddings
        """
        B, D = z.shape
        L    = self.max_seq_len
        device = z.device

        # (B, D) → (B, 1, D) → broadcast to (B, L, D)
        z_cond = self.z_proj(z).unsqueeze(1)                      # (B, 1, D)

        # Positional embeddings: (L, D) → (1, L, D)
        positions = torch.arange(L, device=device)
        pos       = self.pos_embed(positions).unsqueeze(0)         # (1, L, D)

        # Seed sequence: conditioning + positional
        x = z_cond + pos                                           # (B, L, D)

        # Build key_padding_mask for attention: True = ignore
        kpm = None
        if mask is not None:
            kpm = (mask == 0)                                      # (B, L)

        for layer in self.layers:
            # Re-inject conditioning at every layer (additive residual signal)
            x = x + z_cond
            x = layer(x, key_padding_mask=kpm)

        return self.norm(x)                                        # (B, L, D)