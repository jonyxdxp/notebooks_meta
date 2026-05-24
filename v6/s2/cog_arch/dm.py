
import torch
import torch.nn as nn
from config import cfg


class DialogueJEPAPredictor(nn.Module):
    """
    V-JEPA style predictor for dialogue.
    Input:  (B, N, 768) sequence of DMI turn embeddings
    Output: (B, 768)    predicted DMI embedding of turn N+1
    """
    def __init__(self):
        super().__init__()
        d_input = cfg.d_input
        d_model = cfg.d_model

        self.input_proj  = nn.Linear(d_input, d_model)
        self.pred_token  = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_emb     = nn.Embedding(cfg.max_turns + 1, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer,
                                              num_layers=cfg.num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_input))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, turn_embs, padding_mask=None):
        B, N, _ = turn_embs.shape

        x         = self.input_proj(turn_embs)
        positions = torch.arange(N, device=turn_embs.device)
        x         = x + self.pos_emb(positions).unsqueeze(0)

        pred_tok  = self.pred_token.expand(B, -1, -1)
        pred_pos  = self.pos_emb(torch.tensor([N], device=turn_embs.device))
        pred_tok  = pred_tok + pred_pos.unsqueeze(0)

        seq = torch.cat([x, pred_tok], dim=1)

        if padding_mask is not None:
            pred_pad  = torch.zeros(B, 1, dtype=torch.bool,
                                    device=turn_embs.device)
            full_mask = torch.cat([padding_mask, pred_pad], dim=1)
        else:
            full_mask = None

        out      = self.encoder(seq, src_key_padding_mask=full_mask)
        pred_out = out[:, -1, :]
        return self.output_proj(pred_out)
