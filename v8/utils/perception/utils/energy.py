




import jax
import jax.numpy as jnp

import flax.linen as nn

from typing import Optional, Any, Callable


def exists(val):
    return val is not None


def default(val, def_val):
    return val if exists(val) else def_val









class EnergyLayerNorm(nn.Module):
    """
    Perform layer norm on the last dimension of input
    While an energy could be defined for this, it is easier to just define the forward operation (activation function) since the
    energy calculation is not needed for the dynamics of the network
    """

    in_dim: int
    dtype: Any = jnp.float32
    use_bias: bool = (
        True  # Whether to use a bias in the layer normalization step or not
    )
    eps: float = 1e-05  # Prevent division by 0

    def setup(self):
        if self.use_bias:
            self.bias = self.param(
                "bias", nn.initializers.zeros, (self.in_dim), self.dtype
            )

        self.gamma = self.param("gamma", nn.initializers.ones, (1), self.dtype)

    def forward(self, x: jnp.ndarray):
        xmeaned = x - x.mean(-1, keepdims=True)
        v = (
            self.gamma
            * xmeaned
            / jnp.sqrt((xmeaned**2.0).mean(-1, keepdims=True) + self.eps)
        )

        if self.use_bias:
            return v + self.bias
        return v

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        return self.forward(x)












class HNN(nn.Module):
    """Hopfield ReLU Layer"""

    in_dim: int
    multiplier: float = 4
    use_bias: bool = True
    dtype: Any = jnp.float32
    fn: Callable = jax.nn.relu

    def setup(self):
        hid_dim = int(self.multiplier * self.in_dim)
        self.proj = nn.Dense(hid_dim, use_bias=self.use_bias, dtype=self.dtype)

    def energy(self, g: jnp.ndarray, adj: jnp.ndarray):
        A = self.fn(self.proj(g))
        return -0.5 * jnp.square(A).sum()

    def energy_and_grad(self, g: jnp.ndarray, adj: jnp.ndarray):
        return jax.value_and_grad(self.energy)(g, adj)

    @nn.compact
    def __call__(self, g: jnp.ndarray, adj: jnp.ndarray):
        return self.energy_and_grad(g, adj)














class HNN_LSE(nn.Module):
    """Hopfield Softmax Layer"""

    in_dim: int
    multiplier: float = 4
    use_bias: bool = True
    beta_init: float = 0.01
    dtype: Any = jnp.float32

    def setup(self):
        hid_dim = int(self.multiplier * self.in_dim)
        self.proj = nn.Dense(hid_dim, use_bias=self.use_bias, dtype=self.dtype)

        self.beta = hid_dim**0.5

    def energy(self, g: jnp.ndarray, adj: jnp.ndarray):
        h = self.proj(g)
        A = jax.nn.logsumexp(self.beta * h, axis=-1)
        return (-1.0 / self.beta) * A.sum()

    def energy_and_grad(self, g: jnp.ndarray, adj: jnp.ndarray):
        return jax.value_and_grad(self.energy)(g, adj)

    @nn.compact
    def __call__(self, g: jnp.ndarray, adj: jnp.ndarray):
        return self.energy_and_grad(g, adj)















class Attention(nn.Module):
    """The energy of attention for a single head"""

    in_dim: int
    nheads: int = 12
    head_dim: int = 64
    use_bias: bool = True
    dtype: Any = jnp.float32
    beta_init: Optional[float] = None

    def setup(self):
        self.Wk = self.param(
            "Wk",
            nn.initializers.normal(0.002),
            (self.nheads, self.head_dim, self.in_dim),
            self.dtype,
        )
        self.Wq = self.param(
            "Wq",
            nn.initializers.normal(0.002),
            (self.nheads, self.head_dim, self.in_dim),
            self.dtype,
        )

        self.Hw = self.param(
            "Hw", nn.initializers.normal(0.002), (self.nheads, self.nheads), self.dtype
        )

        if self.use_bias:
            self.Bk = self.param(
                "Bk", nn.initializers.zeros, (self.nheads, self.head_dim), self.dtype
            )
            self.Bq = self.param(
                "Bq", nn.initializers.zeros, (self.nheads, self.head_dim), self.dtype
            )

        # self.betas = jnp.ones(self.nheads, dtype=self.dtype) * default(self.beta_init, 1.0 / jnp.sqrt(self.head_dim))

        self.betas = self.param(
            "betas",
            lambda key, shape, dtype: nn.initializers.ones(key, shape, dtype)
            * default(self.beta_init, 1.0 / jnp.sqrt(self.head_dim)),
            (self.nheads),
            self.dtype,
        )

    def energy(self, g: jnp.ndarray, adj: jnp.ndarray):
        """Return the energy of the block"""
        K = jnp.einsum("kd, hzd -> khz", g, self.Wk)  # kseq, heads, head_dim
        Q = jnp.einsum("qd, hzd -> qhz", g, self.Wq)  # qseq, heads, head_dim

        if self.use_bias:
            K = K + self.Bk
            Q = Q + self.Bq

        A1 = jnp.einsum("h, qhz, khz -> hqk", self.betas, Q, K)  # NHeads, Nseq, Nseq

        # Attention Matrix times Adjacency Matrix
        if adj is not None:
            A11 = (A1.transpose(1, 2, 0) @ self.Hw) * adj

            A11 = jnp.where(
                A11 == 0, -jnp.inf, A11
            )  # Avoid empty edges s.t. the gradient (softmax) does not account empty edges as part of the distribution

            A21 = jax.nn.logsumexp(A11, 1)  # Nseq, Nheads

            A21 = jnp.where(A21 == -jnp.inf, 0, A21)

            A31 = A21.sum(0)  # Nheads

            A4 = ((-1.0 / self.betas) * A31).sum()
            return A4
 
        A2 = jax.nn.logsumexp(A1.transpose(1, 2, 0) @ self.Hw, 1)  # Nseq, NHeads
        A3 = A2.sum(0)  # Nheads
        A4 = ((-1.0 / self.betas) * A3).sum()
        return A4

    def energy_and_grad(self, g: jnp.ndarray, adj: jnp.ndarray):
        return jax.value_and_grad(self.energy)(g, adj)

    @nn.compact
    def __call__(self, g: jnp.ndarray, adj: jnp.ndarray = None):
        return self.energy_and_grad(g, adj)



















class HopfieldTransformer(nn.Module):
    """Full energy transformer"""

    in_dim: int
    nheads: int = 12
    head_dim: int = 64
    multiplier: float = 4.0
    attn_beta_init: Optional[float] = None
    use_biases_attn: bool = False
    use_biases_chn: bool = False
    dtype: Any = jnp.float32
    atype: str = "relu"

    def setup(self):
        self.attn = Attention(
            in_dim=self.in_dim,
            nheads=self.nheads,
            head_dim=self.head_dim,
            use_bias=self.use_biases_attn,
            beta_init=self.attn_beta_init,
            dtype=self.dtype,
        )

        if self.atype == "relu":
            self.chn = HNN(
                in_dim=self.in_dim,
                multiplier=self.multiplier,
                use_bias=self.use_biases_chn,
                dtype=self.dtype,
            )
        elif self.atype == "gelu":
            self.chn = HNN(
                in_dim=self.in_dim,
                multiplier=self.multiplier,
                use_bias=self.use_biases_chn,
                dtype=self.dtype,
                fn=jax.nn.gelu,
            )
        else:
            self.chn = HNN_LSE(
                in_dim=self.in_dim,
                multiplier=self.multiplier,
                use_bias=self.use_biases_chn,
                dtype=self.dtype,
            )

    def energy(self, g: jnp.ndarray, adj: jnp.ndarray, **kwargs):
        energy = self.attn.energy(g, adj) + self.chn.energy(g, adj)
        return energy

    def energy_and_grad(self, g: jnp.ndarray, adj: jnp.ndarray, **kwargs):
        return jax.value_and_grad(self.energy)(g, adj, **kwargs)

    @nn.compact
    def __call__(self, g: jnp.ndarray, adj: jnp.ndarray, **kwargs):
        return self.energy_and_grad(g, adj, **kwargs)
    























    # ----------------------------------------------------------------------






















# Conceptual architecture
class EnergyTransformer(nn.Module):
    def __init__(self, codebook_dim, n_heads, n_layers):
        self.code_embedding = nn.Embedding(num_codes, codebook_dim)
        self.transformer_encoder = TransformerEncoder(...)
        self.energy_head = nn.Linear(codebook_dim, 1)  # Scalar output


        # self.latent_var = LatentVariable(...)   ---- (doent apply yet in the first training, only in the inference)
        






        
    def forward(self, code_indices):  # [batch, num_codebooks]
        code_embs = self.code_embedding(code_indices)
        hidden_states = self.transformer_encoder(code_embs)
        energies = self.energy_head(hidden_states).squeeze(-1)
        return energies
    




















import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

class EnergyTransformer(nn.Module):
    """Transformer encoder with energy head for scalar energy assignments"""
    
    def __init__(self, codebook_dim, num_heads=8, num_layers=6, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.codebook_dim = codebook_dim
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=codebook_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Energy prediction head
        self.energy_head = nn.Sequential(
            nn.LayerNorm(codebook_dim),
            nn.Linear(codebook_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Learnable positional encoding for codebook entries
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, codebook_dim))
        
    def forward(self, code_embeddings):
        """
        Args:
            code_embeddings: [batch_size, num_codebooks, codebook_dim]
        
        Returns:
            energies: [batch_size, num_codebooks] scalar energies per codebook
            hidden_states: [batch_size, num_codebooks, codebook_dim]
        """
        batch_size, num_codebooks, _ = code_embeddings.shape
        
        # Add positional encoding
        pos_enc = repeat(self.pos_encoding, '1 1 d -> b n d', b=batch_size, n=num_codebooks)
        x = code_embeddings + pos_enc
        
        # Apply transformer
        hidden_states = self.transformer(x)
        
        # Predict energies
        energies = self.energy_head(hidden_states).squeeze(-1)  # [batch, num_codebooks]
        
        return energies, hidden_states





























# v-jepa 2 predictor:





# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial

import torch
import torch.nn as nn

from src.masks.utils import apply_masks
from src.models.utils.modules import Block
from src.models.utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from src.utils.tensors import repeat_interleave_batch, trunc_normal_


class PerceptionPred(nn.Module):
    """Vision Transformer predictor"""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=False,
        use_mask_tokens=False,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
        return_all_tokens=False,
        chop_last_n_tokens=0,
        use_rope=False,
        **kwargs
    ):
        super().__init__()
        self.return_all_tokens = return_all_tokens
        self.chop_last_n_tokens = chop_last_n_tokens

        # Map input to predictor dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # Mask tokens
        self.mask_tokens = None
        self.num_mask_tokens = 0
        if use_mask_tokens:
            self.num_mask_tokens = num_mask_tokens
            self.mask_tokens = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)) for i in range(num_mask_tokens)]
            )

        # Determine positional embedding
        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        # --
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        self.grid_height = img_size[0] // self.patch_size
        self.grid_width = img_size[1] // self.patch_size
        self.grid_depth = num_frames // self.tubelet_size
        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if self.is_video:
            self.num_patches = num_patches = (
                (num_frames // tubelet_size) * (img_size[0] // patch_size) * (img_size[1] // patch_size)
            )
        else:
            self.num_patches = num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        # Position embedding
        self.uniform_power = uniform_power

        self.predictor_pos_embed = None
        if not use_rope:
            self.predictor_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False
            )

        # Attention Blocks
        self.use_rope = use_rope
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    grid_depth=self.grid_depth,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # Normalize & project back to input dimension
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # ------ initialize weights
        if self.predictor_pos_embed is not None:
            self._init_pos_embed(self.predictor_pos_embed.data)  # sincos pos-embed
        self.init_std = init_std
        if not zero_init_mask_tokens:
            for mt in self.mask_tokens:
                trunc_normal_(mt, std=init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        grid_size = self.img_height // self.patch_size  # TODO: update; currently assumes square input
        if self.is_video:
            grid_depth = self.num_frames // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(
                embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=self.uniform_power
            )
        else:
            sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, x, masks_x, masks_y, mask_index=1, has_cls=False):
        """
        :param x: context tokens
        :param masks_x: indices of context tokens in input
        :params masks_y: indices of target tokens in input
        """
        assert (masks_x is not None) and (masks_y is not None), "Cannot run predictor without mask indices"
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks_y, list):
            masks_y = [masks_y]

        # Batch Size
        B = len(x) // len(masks_x)

        # Map context tokens to predictor dimensions
        x = self.predictor_embed(x)
        if has_cls:
            x_cls = x[:, :1, :]
            x = x[:, 1:, :]
        _, N_ctxt, D = x.shape

        # Add positional embedding to ctxt tokens
        if not self.use_rope:
            x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
            x += apply_masks(x_pos_embed, masks_x)

        # Make target tokens
        mask_index = mask_index % self.num_mask_tokens
        pred_tokens = self.mask_tokens[mask_index]
        pred_tokens = pred_tokens.repeat(B, self.num_patches, 1)
        pred_tokens = apply_masks(pred_tokens, masks_y)
        # -- add pos embed
        if not self.use_rope:
            pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
            pos_embs = apply_masks(pos_embs, masks_y)
            pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
            pred_tokens += pos_embs

        # Concatenate context & target tokens
        x = x.repeat(len(masks_x), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # Positions of context & target tokens
        masks_x = torch.cat(masks_x, dim=0)
        masks_y = torch.cat(masks_y, dim=0)
        masks = torch.cat([masks_x, masks_y], dim=1)

        # Put tokens in sorted order
        argsort = torch.argsort(masks, dim=1)  # [B, N]
        masks = torch.stack([masks[i, row] for i, row in enumerate(argsort)], dim=0)
        x = torch.stack([x[i, row, :] for i, row in enumerate(argsort)], dim=0)

        # Remove the last n tokens of sorted sequence before processing
        if self.chop_last_n_tokens > 0:
            x = x[:, : -self.chop_last_n_tokens]
            masks = masks[:, : -self.chop_last_n_tokens]

        if has_cls:
            x = torch.cat([x_cls, x], dim=1)

        # Fwd prop
        for i, blk in enumerate(self.predictor_blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(blk, x, masks, None, use_reentrant=False)
            else:
                x = blk(x, mask=masks, attn_mask=None)
        x = self.predictor_norm(x)

        if has_cls:
            x = x[:, 1:, :]

        # Return output corresponding to target tokens
        if not self.return_all_tokens:
            reverse_argsort = torch.argsort(argsort, dim=1)  # [B, N]
            x = torch.stack([x[i, row, :] for i, row in enumerate(reverse_argsort)], dim=0)
            x = x[:, N_ctxt:]

        x = self.predictor_proj(x)

        return x


def vit_predictor(**kwargs):
    model = VisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model































# --------------------------------------------------------



class ModernHopfieldAttention(nn.Module):
    def __init__(self, d_model, n_heads, beta=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Mismo setup que self-attention
        self.W_q = nn.Linear(d_model, d_model)  # Query = patrón ξ
        self.W_k = nn.Linear(d_model, d_model)  # Keys = memoria X
        self.W_v = nn.Linear(d_model, d_model)  # Values = memoria X
        self.W_o = nn.Linear(d_model, d_model)
        
        # β controla la "temperatura" de la energía
        # Si β = 1/sqrt(d_k), recuperas exactamente self-attention
        self.beta = beta if beta is not None else (1.0 / math.sqrt(self.d_k))
        
    def energy(self, Q, K):
        """
        Función de energía del Modern Hopfield Network
        E(ξ, X) = -lse(β·X^T·ξ) + ½·ξ^T·ξ + const
        
        Donde:
        - ξ (xi) = Query (patrón a recuperar)
        - X = Keys (patrones almacenados en memoria)
        """
        # Similarity matrix: [B, n_heads, T_q, T_k]
        similarity = Q @ K.transpose(-2, -1)  # X^T · ξ
        
        # Log-sum-exp con β scaling
        lse = torch.logsumexp(self.beta * similarity, dim=-1, keepdim=True)
        
        # Energía (sin términos constantes que no afectan gradientes)
        # E = -lse + 0.5 * ||Q||²
        energy = -lse / self.beta
        
        return energy, similarity
    
    def forward(self, x, return_energy=False):
        B, T, C = x.shape
        
        # Proyecciones (igual que self-attention)
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        # Calcula energía y similarity
        energy, similarity = self.energy(Q, K)
        
        # Update rule del Hopfield Network (equivalente a attention)
        # ξ_new = X · softmax(β·X^T·ξ)
        attn_weights = F.softmax(self.beta * similarity, dim=-1)
        out = attn_weights @ V
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.W_o(out)
        
        if return_energy:
            # Energía total: suma sobre heads y secuencia
            total_energy = energy.sum(dim=(1, 2, 3))  # [B]
            return out, total_energy
        
        return out