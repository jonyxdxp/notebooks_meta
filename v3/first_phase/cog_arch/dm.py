

# adapted from https://github.com/alexiglad/EBT/blob/main/model/bi_ebt_adaln.py



# NLP_JEPA



# adapted from https://arxiv.org/pdf/2212.09748 and https://github.com/facebookresearch/DiT :)))
# major change is DiT -> EBT as well as changing output layer to scalar



import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    # x: (B, T, D), shift/scale: (B, D) → unsqueeze para broadcast
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SwiGLU(nn.Module):
    """
    SwiGLU: hidden projects to 2*mlp_hidden, splits, gates with SiLU.
    Matches your Encoder's FFN style.
    """
    def __init__(self, hidden_size, mlp_hidden):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, mlp_hidden * 2, bias=False)
        self.w2 = nn.Linear(mlp_hidden, hidden_size, bias=False)

    def forward(self, x):
        gate, x = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * x)







class Block(nn.Module):
    """
    JEPA predictor block.
    
    adaLN-Zero conditions on c (B, D):
      - 6 chunks for cross-attention + MLP  (self-attn commented out)
      - if you re-enable self-attn, bump to 8 chunks
    
    Attention layout:
      cross_attn : mask tokens (Q) ← context encoder output (K, V)
      [self_attn]: mask tokens attend to each other  ← optional, disabled
      mlp        : SwiGLU feedforward
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()

        # ── Norms ────────────────────────────────────────────────────────────
        # separate norms for x-as-query vs context-as-key/value
        self.norm_q   = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm_kv  = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm_mlp = nn.LayerNorm(hidden_size, elementwise_affine=False)

        # ── Attention ────────────────────────────────────────────────────────
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            batch_first=True,
            bias=False          # common in modern transformers
        )

        # self.norm_self = nn.LayerNorm(hidden_size, elementwise_affine=False)
        # self.self_attn = nn.MultiheadAttention(
        #     hidden_size, num_heads, batch_first=True, bias=False
        # )

        # ── MLP ──────────────────────────────────────────────────────────────
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLU(hidden_size, mlp_hidden)

        # ── adaLN-Zero ───────────────────────────────────────────────────────
        # 6 chunks: (shift_cross, scale_cross, gate_cross, shift_mlp, scale_mlp, gate_mlp)
        # → if self-attn re-enabled: bump to 9 chunks and add (shift_self, scale_self, gate_self)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        # Zero-init: at start of training all gates=0 → block is identity
        # This is critical for stable transfer from stage 1
        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias,   0)

    def forward(
        self,
        x:       torch.Tensor,   # (B, T_mask, D)  mask token queries
        context: torch.Tensor,   # (B, T_ctx,  D)  context encoder output
        c:       torch.Tensor,   # (B, D)           conditioner (zeros or action)
    ) -> torch.Tensor:

        (shift_cross, scale_cross, gate_cross,
         shift_mlp,   scale_mlp,   gate_mlp) = self.adaLN(c).chunk(6, dim=-1)
        # each chunk: (B, D)

        # ── 1. Cross-attention ───────────────────────────────────────────────
        # Query: modulated mask tokens
        # Key/Value: normalized context (no adaLN modulation on context —
        #            it comes from a frozen encoder, we don't want to shift it)
        q   = modulate(self.norm_q(x), shift_cross, scale_cross)  # (B, T_mask, D)
        kv  = self.norm_kv(context)                                # (B, T_ctx,  D)

        cross_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        x = x + gate_cross.unsqueeze(1) * cross_out

        # ── 2. Self-attention (disabled) ─────────────────────────────────────
        # h = modulate(self.norm_self(x), shift_self, scale_self)
        # self_out, _ = self.self_attn(h, h, h)
        # x = x + gate_self.unsqueeze(1) * self_out

        # ── 3. MLP ───────────────────────────────────────────────────────────
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm_mlp(x), shift_mlp, scale_mlp)
        )

        return x







class DM(nn.Module):
    """
    JEPA predictor (stage 1: intra-frame, stage 2: causal with action).

    Stage 1:
        condition = None → zeros → adaLN is identity → pure cross-attention JEPA
    Stage 2:
        condition = action_projector(action) → adaLN modulates with action signal
        encoder is frozen, only adaLN weights + action_projector are trained
    """
    def __init__(
        self,
        hidden_size: int = 768,
        depth:       int = 12,
        num_heads:   int = 12,
        mlp_ratio:   float = 4.0,
        max_seq_len: int = 512,
    ):
        super().__init__()

        # Learnable positional embeddings for target (masked) positions
        self.pos_embed = nn.Embedding(max_seq_len, hidden_size)

        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(hidden_size)

        # Final projection D → D to match target encoder embedding space
        # Linear (no bias) is standard in JEPA to avoid energy trivial solutions
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        # pos_embed: small normal like standard transformers
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        # proj: xavier
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(
        self,
        context_embeds:   torch.Tensor,          # (B, T_ctx,  D)
        target_positions: torch.Tensor,          # (B, T_mask) int indices
        condition:        torch.Tensor = None,   # (B, D) or None
    ) -> torch.Tensor:
        """
        Returns predicted embeddings (B, T_mask, D) in the target encoder space.
        """
        B, T_mask = target_positions.shape

        # Queries are purely positional — no token content leaks into predictions
        x = self.pos_embed(target_positions)   # (B, T_mask, D)

        # Stage 1: no conditioning → zeros keeps adaLN as identity
        if condition is None:
            condition = torch.zeros(B, x.shape[-1], device=x.device, dtype=x.dtype)

        for block in self.blocks:
            x = block(x, context_embeds, condition)

        x = self.norm(x)
        return self.proj(x)   # (B, T_mask, D)