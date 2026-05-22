# ── Cell 1: Architecture ──────────────────────────────────────────────────────
#
# V-JEPA style dialogue predictor:
#
#  turn_1 → DMI (frozen) → emb_1 ┐
#  turn_2 → DMI (frozen) → emb_2 ├→ Transformer predictor → emb_{N+1}_hat
#  turn_N → DMI (frozen) → emb_N ┘
#                                         ↓
#                               MSE vs true emb_{N+1}
#                               (from DMI encoder, frozen)
#
# Key: everything in embedding space — no text generation, no tokens

import torch, torch.nn as nn, math
import torch.optim as optim
import numpy as np, random
from tqdm.auto import tqdm

class DialogueJEPAPredictor(nn.Module):
    """
    V-JEPA style predictor for dialogue.

    Input:  sequence of DMI turn embeddings [emb_1, ..., emb_N]
    Output: predicted DMI embedding of turn N+1

    Architecture:
      - Linear projection of each turn embedding into predictor dim
      - Learnable [PRED] token appended at position N+1
      - Transformer encoder over all tokens (turn embeddings + PRED token)
      - Output projection from [PRED] token → predicted embedding
    """
    def __init__(self, d_input=768, d_model=512, nhead=8,
                 num_layers=4, dim_feedforward=1024,
                 max_turns=6, dropout=0.1):
        super().__init__()
        self.d_model   = d_model
        self.max_turns = max_turns

        # Project DMI embeddings into predictor space
        self.input_proj = nn.Linear(d_input, d_model)

        # Learnable [PRED] token — the query for the next turn
        self.pred_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional embeddings for each turn position (including PRED)
        self.pos_emb = nn.Embedding(max_turns + 1, d_model)

        # Transformer encoder (bidirectional — all turns see each other)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Project [PRED] output back to DMI embedding space
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_input)   # 768 — same dim as DMI
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, turn_embs, padding_mask=None):
        """
        turn_embs   : (B, N, 768) — N context turn embeddings
        padding_mask: (B, N)      — True where turns are padding
        Returns     : (B, 768)    — predicted next turn embedding
        """
        B, N, _ = turn_embs.shape

        # Project context turns into predictor space
        x = self.input_proj(turn_embs)               # (B, N, d_model)

        # Add positional embeddings to context turns
        positions = torch.arange(N, device=turn_embs.device)
        x = x + self.pos_emb(positions).unsqueeze(0) # (B, N, d_model)

        # Append [PRED] token at position N
        pred_tok = self.pred_token.expand(B, -1, -1)  # (B, 1, d_model)
        pred_pos = self.pos_emb(
            torch.tensor([N], device=turn_embs.device))
        pred_tok = pred_tok + pred_pos.unsqueeze(0)   # (B, 1, d_model)

        # Full sequence: [turn_1, ..., turn_N, PRED]
        seq = torch.cat([x, pred_tok], dim=1)         # (B, N+1, d_model)

        # Extend padding mask to cover [PRED] token (never masked)
        if padding_mask is not None:
            pred_pad = torch.zeros(B, 1, dtype=torch.bool,
                                   device=turn_embs.device)
            full_mask = torch.cat([padding_mask, pred_pad], dim=1)  # (B, N+1)
        else:
            full_mask = None

        # Transformer encoder — all tokens attend to all others
        out = self.encoder(seq, src_key_padding_mask=full_mask)  # (B, N+1, d_model)

        # Read off [PRED] token output (last position)
        pred_out = out[:, -1, :]                      # (B, d_model)

        # Project back to DMI embedding space
        return self.output_proj(pred_out)             # (B, 768)


predictor = DialogueJEPAPredictor().to(device)
n_params  = sum(p.numel() for p in predictor.parameters())
print(f"JEPA predictor: {n_params:,} parameters")
print(f"Input dim:  768 (DMI embedding)")
print(f"Hidden dim: 512")
print(f"Output dim: 768 (predicted DMI embedding)")