





import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Primitives
# ==============================================================================

class SinusoidalPositionEmbedding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al. 2017).
    Provides absolute position information to context memory tokens.

    Shape: (1, max_seq_len, d_model) - broadcast over batch.
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe  = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(max_seq_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) - adds positional signal in-place."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnedPositionEmbedding(nn.Module):
    """
    Learnable absolute position embedding for query tokens.

    Each masked position index maps to a trainable vector that tells
    the decoder *where* to predict. Separate from context positional
    encoding - these are the target position queries.

    Args:
        max_seq_len : maximum number of positions
        d_model     : embedding dimension
    """

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(max_seq_len, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """positions: (B, T) int64 -> (B, T, D)"""
        return self.embed(positions)


# ==============================================================================
# 2. FiLM - Feature-wise Linear Modulation
# ==============================================================================

class FiLM(nn.Module):
    """
    Conditions a sequence of hidden states on a single action vector.

        out_i = hidden_i * (1 + scale(a)) + shift(a)

    The (1 + scale) residual formulation means the network starts as
    identity (scale=0, shift=0 after zero-init) and learns residual
    conditioning on top of the null-action path.

    This is critical for Stage 2->3 transfer: the null-action behavior
    is preserved exactly, and Stage 3 fine-tuning learns a delta on top.

    Args:
        d_model  : hidden state dimension
        d_action : action embedding dimension (defaults to d_model)
    """

    def __init__(self, d_model: int, d_action: int = None):
        super().__init__()
        d_action = d_action or d_model

        self.scale_proj = nn.Linear(d_action, d_model)
        self.shift_proj = nn.Linear(d_action, d_model)

        # Zero-init -> identity transform at initialization
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)

    def forward(
        self,
        hidden: torch.Tensor,       # (B, L, D)
        action_emb: torch.Tensor,   # (B, D_action)
    ) -> torch.Tensor:              # (B, L, D)
        scale = self.scale_proj(action_emb).unsqueeze(1)   # (B, 1, D)
        shift = self.shift_proj(action_emb).unsqueeze(1)   # (B, 1, D)
        return hidden * (1.0 + scale) + shift














# ==============================================================================
# 3. Multi-Head Attention
# ==============================================================================

class MultiHeadAttention(nn.Module):
    """
    Standard scaled dot-product multi-head attention.

    Supports both:
      - Self-attention  (query = key = value)
      - Cross-attention (query from decoder, key/value from encoder memory)

    Args:
        d_model : total model dimension
        n_heads : number of attention heads (d_model must be divisible)
        dropout : attention weight dropout probability
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = math.sqrt(self.d_head)

        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(dropout)

        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, S, D) -> (B, H, S, D/H)"""
        B, S, D = x.shape
        return x.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, S, D/H) -> (B, S, D)"""
        B, H, S, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, H * Dh)

    def forward(
        self,
        query: torch.Tensor,                        # (B, T, D)
        key: torch.Tensor,                          # (B, S, D)
        value: torch.Tensor,                        # (B, S, D)
        key_padding_mask: torch.Tensor = None,      # (B, S) bool - True = ignore
        attn_mask: torch.Tensor = None,             # (T, S) additive mask
    ) -> torch.Tensor:                              # (B, T, D)
        Q = self._split_heads(self.q_proj(query))   # (B, H, T, Dh)
        K = self._split_heads(self.k_proj(key))     # (B, H, S, Dh)
        V = self._split_heads(self.v_proj(value))   # (B, H, S, Dh)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T, S)

        if attn_mask is not None:
            scores = scores + attn_mask

        if key_padding_mask is not None:
            # (B, S) -> (B, 1, 1, S) broadcast over heads and query positions
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        weights = self.dropout(F.softmax(scores, dim=-1))
        out     = self._merge_heads(torch.matmul(weights, V))   # (B, T, D)
        return self.out_proj(out)







# ==============================================================================
# 4. Feed-Forward Block
# ==============================================================================

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

        FFN(x) = Linear(GELU(Linear(x)))

    Args:
        d_model : input/output dimension
        d_ff    : intermediate dimension (default: 4 * d_model)
        dropout : dropout between linear layers
    """

    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.fc1     = nn.Linear(d_model, d_ff)
        self.fc2     = nn.Linear(d_ff, d_model)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.dropout(self.act(self.fc1(x)))))
















# ==============================================================================
# 5. Transformer Decoder Layer (Pre-LN)
# ==============================================================================

class DecoderLayer(nn.Module):
    """
    Single Pre-LayerNorm Transformer Decoder layer.

    Pre-LN (normalize before attention) is more stable than Post-LN
    for smaller models and shorter training runs.

    Sub-layers:
      1. Self-attention   - query tokens attend to each other
      2. Cross-attention  - queries attend to FiLM-conditioned context memory
      3. Feed-forward

    Args:
        d_model : model dimension
        n_heads : number of attention heads
        d_ff    : feed-forward intermediate size (default: 4 * d_model)
        dropout : dropout rate
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Sub-layer 1: self-attention
        self.norm1      = nn.LayerNorm(d_model)
        self.self_attn  = MultiHeadAttention(d_model, n_heads, dropout)
        self.drop1      = nn.Dropout(dropout)

        # Sub-layer 2: cross-attention to context memory
        self.norm2      = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.drop2      = nn.Dropout(dropout)

        # Sub-layer 3: feed-forward
        self.norm3      = nn.LayerNorm(d_model)
        self.ff         = FeedForward(d_model, d_ff, dropout)
        self.drop3      = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,                              # (B, T, D)
        memory: torch.Tensor,                           # (B, L, D)
        tgt_key_padding_mask: torch.Tensor = None,      # (B, T) bool
        memory_key_padding_mask: torch.Tensor = None,   # (B, L) bool
    ) -> torch.Tensor:                                  # (B, T, D)

        # 1. Self-attention (Pre-LN)
        tgt = tgt + self.drop1(
            self.self_attn(
                self.norm1(tgt), self.norm1(tgt), self.norm1(tgt),
                key_padding_mask=tgt_key_padding_mask,
            )
        )

        # 2. Cross-attention (Pre-LN)
        tgt = tgt + self.drop2(
            self.cross_attn(
                self.norm2(tgt), memory, memory,
                key_padding_mask=memory_key_padding_mask,
            )
        )

        # 3. Feed-forward (Pre-LN)
        tgt = tgt + self.drop3(self.ff(self.norm3(tgt)))

        return tgt







# ==============================================================================
# 6. Transformer Decoder Stack
# ==============================================================================

class TransformerDecoder(nn.Module):
    """
    Stack of N DecoderLayers followed by a final LayerNorm.

    Args:
        d_model  : model dimension
        n_heads  : attention heads per layer
        n_layers : number of decoder layers (2-4 recommended for DM)
        d_ff     : feed-forward intermediate size (default: 4 * d_model)
        dropout  : dropout rate
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,                              # (B, T, D)
        memory: torch.Tensor,                           # (B, L, D)
        tgt_key_padding_mask: torch.Tensor = None,      # (B, T) bool
        memory_key_padding_mask: torch.Tensor = None,   # (B, L) bool
    ) -> torch.Tensor:                                  # (B, T, D)
        x = tgt
        for layer in self.layers:
            x = layer(
                x, memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.norm(x)


















# ==============================================================================
# 9. DynamicsModel - core predictor module
# ==============================================================================

class DM(nn.Module):
    """
    Core action-conditioned predictor for Text JEPA Stage 2/3.

    Pipeline:
      1. Embed discrete action -> FiLM-condition the context hidden states
      2. Add sinusoidal positional encoding to context (memory)
      3. Build learned-position query tokens for the masked target positions
      4. TransformerDecoder: queries attend to conditioned context
      5. Output projection -> predicted representations (B, T, D)

    Args:
        d_model      : must match Stage-1 encoder hidden size exactly
        n_heads      : attention heads (d_model must be divisible by n_heads)
        n_layers     : number of decoder layers (2 is usually sufficient)
        n_actions    : number of real action classes (excluding null action)
        max_seq_len  : must match Stage-1 max sequence length
        dropout      : dropout throughout the model
    """

    NULL_ACTION_ID = 0   # reserved index - always null / unknown action

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        n_actions: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model     = d_model
        self.max_seq_len = max_seq_len

        # ── Action embedding table ────────────────────────────────────────────
        # Index 0     = null action (Stage 2 / classifier-free guidance null)
        # Index 1..N  = real action classes (Stage 3)
        self.action_embed = nn.Embedding(n_actions + 1, d_model, padding_idx=0)
        nn.init.normal_(self.action_embed.weight, std=0.02)
        with torch.no_grad():
            self.action_embed.weight[self.NULL_ACTION_ID].zero_()

        # ── FiLM conditioning on context hidden states ────────────────────────
        self.film = FiLM(d_model=d_model)

        # ── Sinusoidal positional encoding added to context memory ────────────
        self.ctx_pos_enc = SinusoidalPositionEmbedding(
            d_model, max_seq_len, dropout=0.0)

        # ── Learned position embeddings for query tokens ──────────────────────
        # One embedding per sequence position - indexes masked positions
        self.pos_embed = LearnedPositionEmbedding(max_seq_len, d_model)

        # ── Transformer decoder ───────────────────────────────────────────────
        self.decoder = TransformerDecoder(
            d_model  = d_model,
            n_heads  = n_heads,
            n_layers = n_layers,
            d_ff     = d_model * 4,
            dropout  = dropout,
        )

        # ── Output projection back to encoder representation space ────────────
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        nn.init.xavier_uniform_(self.out_proj[1].weight)
        nn.init.zeros_(self.out_proj[1].bias)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_queries(
        self,
        span_mask: torch.Tensor,   # (B, L) bool
    ):
        """
        Build (B, T, D) query tensor from learned position embeddings
        at the masked positions.

        T = max number of masked positions across the batch.
        Shorter sequences are zero-padded and masked out in attention.

        Returns:
            queries      : (B, T, D)
            positions    : (B, T) int64
            tgt_pad_mask : (B, T) bool - True = padding (ignored in attention)
        """
        B, L   = span_mask.shape
        T      = max(int(span_mask.sum(dim=1).max().item()), 1)
        device = span_mask.device

        positions    = torch.zeros(B, T, dtype=torch.long, device=device)
        tgt_pad_mask = torch.ones(B, T, dtype=torch.bool, device=device)

        for i in range(B):
            pos = span_mask[i].nonzero(as_tuple=False).squeeze(1)
            n   = len(pos)
            if n > 0:
                positions[i, :n]    = pos
                tgt_pad_mask[i, :n] = False   # False = valid token

        queries = self.pos_embed(positions)   # (B, T, D)
        return queries, positions, tgt_pad_mask

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        ctx_hidden: torch.Tensor,                    # (B, L, D)
        span_mask: torch.Tensor,                     # (B, L) bool
        action: torch.Tensor = None,                 # (B,) int  - None = null
        ctx_attention_mask: torch.Tensor = None,     # (B, L) int 1=real/0=pad
    ) -> torch.Tensor:                               # (B, T, D)
        B, L, D = ctx_hidden.shape
        device  = ctx_hidden.device

        # 1. Resolve action -> null if not provided
        if action is None:
            action = torch.full(
                (B,), self.NULL_ACTION_ID, dtype=torch.long, device=device)

        # 2. Embed action and FiLM-condition context
        a_emb    = self.action_embed(action)           # (B, D)
        ctx_cond = self.film(ctx_hidden, a_emb)        # (B, L, D)

        # 3. Add sinusoidal positional encoding to context memory
        ctx_cond = self.ctx_pos_enc(ctx_cond)          # (B, L, D)

        # 4. Build query tokens for masked positions
        queries, _, tgt_pad_mask = self._build_queries(span_mask)

        # 5. Build context padding mask (True = pad token, ignore in attention)
        if ctx_attention_mask is not None:
            mem_pad_mask = (ctx_attention_mask == 0)   # (B, L) bool
        else:
            mem_pad_mask = torch.zeros(B, L, dtype=torch.bool, device=device)

        # 6. Decode: queries attend to conditioned context
        decoded = self.decoder(
            tgt                     = queries,
            memory                  = ctx_cond,
            tgt_key_padding_mask    = tgt_pad_mask,
            memory_key_padding_mask = mem_pad_mask,
        )   # (B, T, D)

        # 7. Project to encoder representation space
        return self.out_proj(decoded)   # (B, T, D)