





import math
import torch
import torch.nn as nn
import torch.nn.functional as F







class DM(nn.Module):
    """
    Lightweight transformer that predicts target representations from:
      - z_ctx          : context encoder output  (B, L, D)
      - target_mask    : bool mask               (B, L)  — positions to predict
      - action         : action vector           (B, action_dim) or None

    In Stage 2: action=None → zero vector, contributes nothing.
    In Stage 3: action=a_t  → real conditioning signal.

    Returns z_pred (B, D) — predicted representation at masked positions
    (mean-pooled over target span, matching how z_tgt is computed).
    """

    def __init__(
        self,
        hidden_size:   int,
        num_heads:     int,
        num_layers:    int,
        action_dim:    int,
        max_seq_len:   int = 128,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.action_dim  = action_dim

        # ── Action projection ─────────────────────────────────────────────────
        # Projects action vector into hidden_size and adds it as a bias.
        # When action=zeros this contributes nothing (bias = proj(0) ≈ 0 after
        # zero-init), so Stage 2 trains purely on context.
        self.action_proj = nn.Linear(action_dim, hidden_size, bias=False)
        nn.init.zeros_(self.action_proj.weight)   # zero-init → no-op in Stage 2

        # ── [PRED] token — learnable query for target positions ───────────────
        self.pred_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

        # ── Positional embedding (shared with encoder dim) ────────────────────
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)

        # ── Transformer layers ────────────────────────────────────────────────
        layer = nn.TransformerEncoderLayer(
            d_model         = hidden_size,
            nhead           = num_heads,
            dim_feedforward = hidden_size * 4,
            dropout         = 0.1,
            batch_first     = True,
            norm_first      = True,   # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(hidden_size)

    def forward(
        self,
        z_ctx:       torch.Tensor,                    # (B, L, D)
        target_mask: torch.Tensor,                    # (B, L) bool
        action:      typing.Optional[torch.Tensor],   # (B, action_dim) or None
    ) -> torch.Tensor:                                # (B, D)

        B, L, D = z_ctx.shape

        # ── Action bias (B, 1, D) ─────────────────────────────────────────────
        if action is None:
            action = torch.zeros(B, self.action_dim, device=z_ctx.device)
        action_bias = self.action_proj(action).unsqueeze(1)   # (B, 1, D)

        # ── Replace target positions with [PRED] token + positional emb ──────
        positions = torch.arange(L, device=z_ctx.device)
        pos_emb   = self.pos_emb(positions).unsqueeze(0)      # (1, L, D)

        x = z_ctx + pos_emb                                   # (B, L, D)

        # Overwrite target positions with learnable pred token
        pred_token  = self.pred_token.expand(B, L, D)         # (B, L, D)
        mask_3d     = target_mask.unsqueeze(-1).float()        # (B, L, 1)
        x = x * (1 - mask_3d) + pred_token * mask_3d          # (B, L, D)

        # ── Inject action bias into every position ────────────────────────────
        x = x + action_bias                                    # (B, L, D)

        # ── Transformer ───────────────────────────────────────────────────────
        x = self.transformer(x)
        x = self.norm(x)

        # ── Mean-pool at target positions → z_pred (B, D) ────────────────────
        z_pred = (x * mask_3d).sum(1) / mask_3d.sum(1).clamp(min=1)

        return z_pred