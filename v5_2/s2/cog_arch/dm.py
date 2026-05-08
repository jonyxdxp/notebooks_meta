"""
model.py — Three-component architecture for dialog next-turn prediction.

Components:
  1. UtteranceEncoder   — frozen (or fine-tunable) DSE-BERT with mean pooling.
                          Maps a single utterance → R^encoder_dim.

  2. CausalContextTransformer — causal self-attention over a sequence of
                          utterance embeddings (history).
                          Input:  (B, T, encoder_dim)
                          Output: (B, encoder_dim)  [last valid position]

  3. ProjectionHead     — 2-layer MLP (optional hidden) that maps the context
                          vector into the same space as the target utterance
                          embedding, enabling InfoNCE comparison.

Full forward pass:
  history (B, T, seq_len) → encode each turn → (B, T, D)
                           → causal transformer → (B, D)
                           → projection → predicted_embedding (B, D)

  target  (B, seq_len)    → encode → target_embedding (B, D)

  InfoNCE(predicted_embedding, target_embedding)
"""

import math
import torch
import torch.nn as nn
from transformers import AutoModel


# ── 1. Utterance Encoder (DSE) ─────────────────────────────────────────────────

class UtteranceEncoder(nn.Module):
    """
    Wraps aws-ai/dse-bert-base (or any BERT-style model) with attention-weighted
    mean pooling — the same pooling strategy used in the DSE paper.
    """

    def __init__(self, model_name: str, cache_dir: str, freeze: bool = True):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.hidden_size = self.bert.config.hidden_size

        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
            print("[model] DSE encoder frozen.")
        else:
            print("[model] DSE encoder will be fine-tuned.")

    @staticmethod
    def mean_pool(token_embeddings: torch.Tensor,
                  attention_mask:   torch.Tensor) -> torch.Tensor:
        """
        Attention-mask-weighted mean over the token dimension.
        token_embeddings: (B, seq_len, D)
        attention_mask:   (B, seq_len)
        returns:          (B, D)
        """
        mask_expanded = attention_mask.unsqueeze(-1).float()          # (B, L, 1)
        sum_emb  = (token_embeddings * mask_expanded).sum(dim=1)       # (B, D)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)            # (B, 1)
        return sum_emb / sum_mask

    def forward(self,
                input_ids:      torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        input_ids:      (B, seq_len)  or  (B, T, seq_len) — handles both.
        attention_mask: same shape as input_ids.
        returns:        (B, D)        or  (B, T, D)
        """
        batched_turns = (input_ids.dim() == 3)

        if batched_turns:
            B, T, L = input_ids.shape
            # flatten turns into batch dimension for a single forward pass
            ids   = input_ids.view(B * T, L)
            masks = attention_mask.view(B * T, L)
        else:
            ids, masks = input_ids, attention_mask

        out = self.bert(input_ids=ids, attention_mask=masks)
        emb = self.mean_pool(out.last_hidden_state, masks)  # (B*T, D) or (B, D)

        if batched_turns:
            emb = emb.view(B, T, -1)                         # (B, T, D)

        return emb


# ── 2. Causal Context Transformer ─────────────────────────────────────────────

class CausalContextTransformer(nn.Module):
    """
    A standard TransformerEncoder with a causal (upper-triangular) attention mask.

    Input:  sequence of utterance embeddings  (B, T, D)
            + history_len (B,) — the number of real (non-padded) turns per sample

    Output: context vector at the last real position  (B, D)

    The causal mask prevents each position from attending to future turns,
    which is essential for next-turn prediction at inference time.
    """

    def __init__(self,
                 embed_dim:  int,
                 n_heads:    int,
                 n_layers:   int,
                 ffn_dim:    int,
                 dropout:    float,
                 max_len:    int):
        super().__init__()

        # positional encoding (sinusoidal, fixed)
        self.pos_enc = SinusoidalPositionalEncoding(embed_dim, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = embed_dim,
            nhead           = n_heads,
            dim_feedforward = ffn_dim,
            dropout         = dropout,
            batch_first     = True,   # (B, T, D) convention
            norm_first      = True,   # pre-norm: more stable at small scale
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = n_layers,
        )

        self.max_len = max_len

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Returns an additive causal mask of shape (T, T).
        Positions in the upper triangle (future tokens) are set to -inf.
        """
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=device),
            diagonal=1,
        )
        return mask                                            # (T, T)

    def _padding_mask(self,
                      history_len: torch.Tensor,
                      T: int,
                      device: torch.device) -> torch.Tensor:
        """
        Returns a boolean key_padding_mask (B, T) — True means IGNORE.
        Padded positions (beyond history_len) are masked out.
        """
        B = history_len.size(0)
        positions = torch.arange(T, device=device).unsqueeze(0)   # (1, T)
        mask = positions >= history_len.unsqueeze(1)               # (B, T)
        return mask

    def forward(self,
                x:           torch.Tensor,
                history_len: torch.Tensor) -> torch.Tensor:
        """
        x:           (B, T, D)  — padded history of utterance embeddings
        history_len: (B,)       — actual number of turns per item
        returns:     (B, D)     — context vector at position history_len-1
        """
        B, T, D = x.shape

        x = self.pos_enc(x)                                        # (B, T, D)

        causal_mask = self._causal_mask(T, x.device)               # (T, T)
        pad_mask    = self._padding_mask(history_len, T, x.device) # (B, T)

        out = self.transformer(
            x,
            mask            = causal_mask,
            src_key_padding_mask = pad_mask,
        )                                                          # (B, T, D)

        # extract the output at the last *real* turn for each item in the batch
        idx = (history_len - 1).clamp(min=0)                      # (B,)
        ctx = out[torch.arange(B, device=x.device), idx]          # (B, D)

        return ctx


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))               # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ── 3. Projection Head ─────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    Maps the context vector into the DSE embedding space for InfoNCE scoring.

    Architecture:
      Linear(D, hidden) → GELU → LayerNorm → Linear(hidden, D)
    If hidden_dim is None, uses a single Linear(D, D).
    """

    def __init__(self, in_dim: int, hidden_dim: int | None, out_dim: int):
        super().__init__()
        if hidden_dim is None:
            self.net = nn.Linear(in_dim, out_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Full Model ─────────────────────────────────────────────────────────────────

class DialogNextTurnPredictor(nn.Module):
    """
    Full model combining the three components above.

    encode_utterances()  — utility to embed a flat batch of utterances (for eval)
    forward()            — training forward: returns (pred_emb, target_emb)
    """

    def __init__(self, cfg):
        super().__init__()

        self.encoder = UtteranceEncoder(
            model_name = cfg.encoder_model,
            cache_dir  = cfg.cache_dir,
            freeze     = cfg.freeze_encoder,
        )
        D = self.encoder.hidden_size

        self.context_transformer = CausalContextTransformer(
            embed_dim = D,
            n_heads   = cfg.ctx_n_heads,
            n_layers  = cfg.ctx_n_layers,
            ffn_dim   = cfg.ctx_ffn_dim,
            dropout   = cfg.ctx_dropout,
            max_len   = cfg.max_history,
        )

        self.projection = ProjectionHead(
            in_dim     = D,
            hidden_dim = cfg.proj_hidden_dim,
            out_dim    = D,
        )

        self.embed_dim = D

    def encode_utterances(self,
                          input_ids:      torch.Tensor,
                          attention_mask: torch.Tensor) -> torch.Tensor:
        """Convenience: embed a flat batch of utterances → (B, D)."""
        return self.encoder(input_ids, attention_mask)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        batch keys (from data.py):
          history_ids:   (B, T, L)
          history_masks: (B, T, L)
          history_len:   (B,)
          target_ids:    (B, L)
          target_mask:   (B, L)

        returns:
          pred_emb   (B, D) — projected context vector
          target_emb (B, D) — DSE embedding of the true next utterance
        """
        # 1. Encode history turns: (B, T, L) → (B, T, D)
        history_emb = self.encoder(
            batch["history_ids"],
            batch["history_masks"],
        )

        # 2. Zero-out padded positions (already handled by padding mask in
        #    the transformer, but zeroing here prevents gradient bleed)
        B, T, D = history_emb.shape
        mask = (torch.arange(T, device=history_emb.device).unsqueeze(0)
                < batch["history_len"].unsqueeze(1))                # (B, T)
        history_emb = history_emb * mask.unsqueeze(-1).float()

        # 3. Context transformer: (B, T, D) → (B, D)
        ctx = self.context_transformer(history_emb, batch["history_len"])

        # 4. Project into DSE space: (B, D) → (B, D)
        pred_emb = self.projection(ctx)

        # 5. Encode target utterance: (B, L) → (B, D)
        target_emb = self.encoder(
            batch["target_ids"],
            batch["target_mask"],
        )

        return pred_emb, target_emb