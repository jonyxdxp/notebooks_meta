"""
dmi_moml_model.py
-----------------
DMI encoder (Transformer, trained fully from scratch - no BERT/RoBERTa init)
adapted from:  Discourse-Mutual-Information-DMI-main/models/core.py

Key change vs. original: removed RoBERTa branch entirely; this file is
exclusively the scratch-trained path (roberta_init=False in the original).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Fix: `higher` cannot differentiate through PyTorch's efficient-attention
# ── kernel (aten::_scaled_dot_product_efficient_attention_backward is not
# ── implemented for autograd).  Force the standard math backend, which has
# ── a full backward pass, so the MOML inner loop remains differentiable.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


# ─────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        if d_model % 2 == 0:
            div = torch.exp(torch.arange(0, d_model, 2).float()
                            * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div)
            pe[:, 1::2] = torch.cos(position * div)
        else:
            div = torch.exp(torch.arange(0, d_model + 1, 2).float()
                            * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div)
            pe[:, 1::2] = torch.cos(position * div[:-1])
        pe = pe.unsqueeze(1)   # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):   # x: (seq, batch, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ScratchTransformerEncoder(nn.Module):
    """
    Vanilla PyTorch TransformerEncoder + sinusoidal positional encoding.
    Input/output convention (same as DMI original):
        forward(x, src_key_padding_mask)
        x: (batch, seq, d_model)  →  returns (batch, seq, d_model)
        mask: True = padding token (ignored)
    """
    def __init__(self, d_model: int, num_layers: int, nhead: int,
                 dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x, src_key_padding_mask):
        # x: (batch, seq, d_model)
        x = x.permute(1, 0, 2)              # → (seq, batch, d_model)
        x = self.pos_encoder(x)
        out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return out.permute(1, 0, 2)         # → (batch, seq, d_model)


# ─────────────────────────────────────────────────────────────
# Full DMI SMI encoder (scratch version)
# ─────────────────────────────────────────────────────────────

class DMIScratchEncoder(nn.Module):
    """
    Mirrors SMI(roberta_init=False) from the DMI paper.

    Architecture:
        embedding  →  ScratchTransformerEncoder  →  CLS-pool
        (context side)  →  c_t
        (response side) →  proj(r_t)  →  z_t

    Loss: InfoNCE  (or any of the other MI estimators from DMI utils)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int         = 256,
        projection_size: int = 256,
        encoder_layers: int  = 4,
        encoder_heads: int   = 4,
        dim_feedforward: int = 1024,
        dropout: float       = 0.1,
        symmetric_loss: bool = False,
    ):
        super().__init__()
        self.d_model         = d_model
        self.symmetric_loss  = symmetric_loss

        # Token embedding (randomly initialised, trained from scratch)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Shared Transformer encoder (context AND response share weights,
        # identical to the original DMI design)
        self.encoder = ScratchTransformerEncoder(
            d_model=d_model,
            num_layers=encoder_layers,
            nhead=encoder_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Linear projection on the response side only (matches DMI's Projection)
        self.proj = nn.Sequential(
            nn.Linear(d_model, projection_size, bias=False),
            nn.BatchNorm1d(projection_size),
        )

        self._reset_parameters()

    # ── init ──────────────────────────────────────────────────
    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── encode one side ───────────────────────────────────────
    def _encode(self, tokens, pad_mask):
        """
        tokens:   (batch, seq)  LongTensor
        pad_mask: (batch, seq)  BoolTensor, True = padding (ignored)
        returns:  (batch, d_model)  — CLS-token representation
        """
        x   = self.embedding(tokens)         # (batch, seq, d_model)
        out = self.encoder(x, pad_mask)      # (batch, seq, d_model)
        return out[:, 0, :]                  # CLS token  (batch, d_model)

    # ── forward ───────────────────────────────────────────────
    def forward(self, context, response, mask_ctx, mask_rsp):
        c_t = self._encode(context, mask_ctx)
        r_t = self._encode(response, mask_rsp)
        z_t = self.proj(r_t)
        # L2 normalize both sides — prevents dimensional collapse
        c_t = F.normalize(c_t, dim=-1)
        z_t = F.normalize(z_t, dim=-1)
        return c_t, z_t

    def encode_context(self, context, mask_ctx):
        """Convenience: encode context only (for downstream tasks)."""
        return self._encode(context, mask_ctx)


# ─────────────────────────────────────────────────────────────
# Loss functions  (verbatim from DMI utils/smile_estimators.py,
#                  inlined here so this file is self-contained)
# ─────────────────────────────────────────────────────────────

def infonce_loss(c_t, z_t, temperature=0.07, symmetric=False):
    # Vectors already normalized — dot product = cosine similarity
    score   = torch.mm(c_t, z_t.t()) / temperature
    log_p   = F.log_softmax(score, dim=1)
    if symmetric:
        log_p0 = F.log_softmax(score, dim=0)
        loss   = -0.5 * torch.mean(torch.diag(log_p)) \
                 -0.5 * torch.mean(torch.diag(log_p0))
    else:
        loss = -torch.mean(torch.diag(log_p))
    mi = math.log(c_t.shape[0]) - loss.item() * temperature
    return score, loss, mi


def jsd_loss(c_t, z_t):
    """Jensen-Shannon divergence estimator."""
    score = torch.mm(c_t, z_t.t())
    f_diag  = score.diag()
    first   = -F.softplus(-f_diag).mean()
    n       = score.size(0)
    second  = (torch.sum(F.softplus(score))
               - torch.sum(F.softplus(f_diag))) / (n * (n - 1.))
    mi      = first - second
    return score, -mi, mi.item()


MI_ESTIMATORS = {
    'infonce': infonce_loss,
    'jsd':     jsd_loss,
}


def compute_loss(c_t, z_t, estimator='infonce', symmetric=False):
    if estimator == 'infonce':
        return infonce_loss(c_t, z_t, symmetric=symmetric)
    elif estimator == 'jsd':
        return jsd_loss(c_t, z_t)
    else:
        raise ValueError(f"Unknown estimator: {estimator}. Use 'infonce' or 'jsd'.")