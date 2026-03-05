





import torch
import torch.nn as nn


class Actor(nn.Module):
    """
    Predicts the next-turn representation in latent space.

    Architecture
    ------------
    input_proj  : Linear(D → D)         — projects z_t into predictor space
    transformer : N-layer Transformer   — models turn transition dynamics
    output_proj : Linear(D → D)         — projects back to representation space
    norm        : LayerNorm(D)          — stabilizes output scale

    Args
    ----
    hidden_size : int   — must match Stage 1 encoder hidden size (default 256)
    num_heads   : int   — attention heads, must divide hidden_size  (default 4)
    num_layers  : int   — transformer depth (default 2, lightweight by design)
    dropout     : float — dropout on transformer layers (default 0.1)
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int   = 4,
        num_layers: int  = 2,
        dropout: float   = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads   = num_heads
        self.num_layers  = num_layers

        self.input_proj = nn.Linear(hidden_size, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = hidden_size,
            nhead          = num_heads,
            dim_feedforward = hidden_size * 4,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,   # pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers,
        )

        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.norm        = nn.LayerNorm(hidden_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        z : (B, D)  — pooled representation of the current turn (turn_t)
                       produced by the frozen Stage 1 encoder

        Returns
        -------
        z_next : (B, D) — predicted representation of the next turn (turn_{t+1})
        """
        x = self.input_proj(z).unsqueeze(1)   # (B, 1, D)
        x = self.transformer(x)               # (B, 1, D)
        x = self.output_proj(x.squeeze(1))    # (B, D)
        return self.norm(x)                   # (B, D)

    def __repr__(self):
        total = sum(p.numel() for p in self.parameters())
        return (
            f'Actor(hidden_size={self.hidden_size}, '
            f'num_heads={self.num_heads}, '
            f'num_layers={self.num_layers}, '
            f'params={total:,})'
        )