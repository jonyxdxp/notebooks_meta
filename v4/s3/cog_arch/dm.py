





import math
import torch
import torch.nn as nn
import torch.nn.functional as F




class DM(nn.Module):
    """
    Same architecture as Stage 2.
    Only change: action_dim = hidden_size = 256 (was 64 in Stage 2).

    action_proj is zero-init'd → zero-vector input in Stage 2 = no-op.
    In Stage 3: receives real 256-dim encoder mean-pools → activates.
    Weights loaded from Stage 2 EXCEPT action_proj (shape mismatch 64→256).
    action_proj re-initialised to zero → still a no-op at epoch 0,
    then learns from scratch how to use the action signal.
    """

    def __init__(self, hidden_size, num_heads, num_layers,
                 action_dim, max_seq_len=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim  = action_dim

        self.action_proj = nn.Linear(action_dim, hidden_size, bias=False)
        nn.init.zeros_(self.action_proj.weight)

        self.pred_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.pos_emb    = nn.Embedding(max_seq_len, hidden_size)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 4, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(hidden_size)




    def forward(self, z_ctx, target_mask, action=None):
        B, L, D = z_ctx.shape

        if action is None:
            action = torch.zeros(B, self.action_dim, device=z_ctx.device)
        action_bias = self.action_proj(action).unsqueeze(1)   # (B, 1, D)

        positions = torch.arange(L, device=z_ctx.device)
        x = z_ctx + self.pos_emb(positions).unsqueeze(0)

        pred_token = self.pred_token.expand(B, L, D)
        mask_3d    = target_mask.unsqueeze(-1).float()
        x = x * (1 - mask_3d) + pred_token * mask_3d
        x = x + action_bias

        x = self.transformer(x)
        x = self.norm(x)

        return (x * mask_3d).sum(1) / mask_3d.sum(1).clamp(min=1)   # (B, D)
