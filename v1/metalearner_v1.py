


# "Encoder - Decoder" like METALEARNER





    # from https://github.com/kmccleary3301/nested_learning/blob/main/src/nested_learning/hope/block.py








# ENCODER



    
@dataclass
class HOPESelfModBlockConfig:
    dim: int
    cms_levels: Sequence[LevelSpec]
    cms_hidden_multiplier: int = 4
    cms_use_layernorm: bool = True
    activation: str = "gelu"
    qk_l2_norm: bool = True
    cms_flush_partial_at_end: bool = False
    selfmod_adaptive_q: bool = False
    selfmod_local_conv_window: int | None = 4
    eta_scale: float = 1e-3
    selfmod_chunk_size: int = 1
    selfmod_chunk_size_memory: int | None = None
    selfmod_objective: str = "l2"
    selfmod_stopgrad_vhat: bool = True
    selfmod_use_rank1_precond: bool = True
    selfmod_use_alpha: bool = True
    selfmod_use_skip: bool = True
    selfmod_momentum: float = 0.0
    selfmod_online_updates: bool = True
    self_mod_lr: float = 1e-3
    cms_chunk_reduction: str = "sum"
    cms_online_updates: bool = True
    optimizer_configs: Dict[str, dict] = field(default_factory=dict)







class HOPESelfModBlock(nn.Module):
    """
    Paper-defined HOPE block (Eqs. 94–97): self-modifying Titans followed by CMS.

    Fast-state is required for in-context self-mod updates.
    """

    def __init__(self, config: HOPESelfModBlockConfig):
        super().__init__()
        self.config = config
        self.last_update_stats: Dict[str, Dict[str, float]] = {}
        self.surprise_threshold: float | None = None
        self.surprise_metric: str = "l2"
        self.allowed_levels: Set[str] | None = None

        self.selfmod = SelfModifyingTitans(
            SelfModifyingTitansConfig(
                dim=config.dim,
                eta_scale=config.eta_scale,
                chunk_size_other=config.selfmod_chunk_size,
                chunk_size_memory=config.selfmod_chunk_size_memory,
                objective=config.selfmod_objective,
                stopgrad_vhat=config.selfmod_stopgrad_vhat,
                use_rank1_precond=config.selfmod_use_rank1_precond,
                use_alpha=config.selfmod_use_alpha,
                use_skip=config.selfmod_use_skip,
                momentum=config.selfmod_momentum,
                qk_l2_norm=config.qk_l2_norm,
                adaptive_q=config.selfmod_adaptive_q,
                local_conv_window=config.selfmod_local_conv_window,
            )
        )

        self.cms = CMS(
            dim=config.dim,
            levels=config.cms_levels,
            hidden_multiplier=config.cms_hidden_multiplier,
            activation=config.activation,
            use_layernorm=config.cms_use_layernorm,
        )
        
        level_config = LevelConfig(
            specs=config.cms_levels,
            optimizer_configs=config.optimizer_configs,
            default_lr=config.self_mod_lr,
        )
        self.level_manager = LevelOptimizerManager(level_config)



    def forward(
        self,
        x: torch.Tensor,
        *,
        teach_signal: torch.Tensor | None = None,
        surprise_value: float | None = None,
        fast_state: BlockFastState | None = None,
    ) -> torch.Tensor:
        if fast_state is None:
            # Differentiable read path (used for the outer loss).
            o = self.selfmod(x)
            # Explicit update pass (typically called under `torch.no_grad()` after backward).
            if teach_signal is not None and self.config.selfmod_online_updates:
                self.selfmod.apply_updates_inplace(x)
            if teach_signal is not None and self.config.cms_online_updates:
                cms_out = self._cms_forward_online(o, teach_signal, surprise_value)
            else:
                cms_out, cms_inputs, cms_outputs = self.cms(o, return_intermediates=True)
                if teach_signal is not None:
                    self._update_cms(cms_inputs, cms_outputs, teach_signal, surprise_value)
            self.level_manager.tick()
            return cms_out

        if fast_state.selfmod_state is None:
            raise ValueError("fast_state.selfmod_state is required for hope_selfmod variant")
        if self.config.selfmod_online_updates and teach_signal is not None:
            o, updated = self.selfmod.forward_with_updates(x, fast_state.selfmod_state)
            fast_state.selfmod_state = updated
        else:
            o = self.selfmod.forward_with_state(x, fast_state.selfmod_state)
        if teach_signal is not None and self.config.cms_online_updates:
            cms_out = self._cms_forward_online_fast(o, fast_state, teach_signal, surprise_value)
        else:
            cms_out, cms_inputs = self._cms_forward_fast(o, fast_state)
            if teach_signal is not None:
                self._update_cms_fast(fast_state, cms_inputs, teach_signal, surprise_value)
        fast_state.level_manager.tick()
        return cms_out









# --------------------------------------------------------









# the unified Encoder model:



# from https://github.com/kmccleary3301/nested_learning/blob/main/src/nested_learning/model.py






from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol, Sequence, cast

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .fast_state import ModelFastState, build_block_fast_state
from .hope.block import (
    HOPEAttentionBlock,
    HOPEAttentionBlockConfig,
    HOPEBlock,
    HOPEBlockConfig,
    HOPESelfModBlock,
    HOPESelfModBlockConfig,
)
from .levels import LevelSpec
from .transformer import TransformerBlock, TransformerBlockConfig







@dataclass
class ModelConfig:
    vocab_size: int
    dim: int
    num_layers: int
    heads: int
    titan_level: LevelSpec
    cms_levels: Sequence[LevelSpec]
    cms_flush_partial_at_end: bool = False
    cms_use_layernorm: bool = True
    optimizers: Dict[str, dict] | None = None
    teach_scale: float = 1.0
    teach_clip: float = 0.0
    teach_schedule: Dict[str, float] | None = None
    gradient_checkpointing: bool = False
    surprise_threshold: float | None = None
    surprise_metric: str = "l2"
    freeze_backbone: bool = False
    qk_l2_norm: bool = False
    local_conv_window: int | None = None
    self_mod_lr: float = 1e-3
    self_mod_hidden: int = 4
    self_mod_chunk_size: int = 1
    self_mod_chunk_size_memory: int | None = None
    self_mod_objective: str = "l2"
    self_mod_stopgrad_vhat: bool = True
    self_mod_use_rank1_precond: bool = True
    self_mod_use_alpha: bool = True
    self_mod_use_skip: bool = True
    self_mod_momentum: float = 0.0
    self_mod_adaptive_q: bool = False
    self_mod_local_conv_window: int | None = 4
    transformer_mlp_hidden_multiplier: int = 4
    transformer_activation: str = "gelu"
    block_variant: str = "hope_hybrid"







class HOPEModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.base_teach_scale = config.teach_scale
        self.base_teach_clip = config.teach_clip
        self._runtime_teach_scale = config.teach_scale
        self._runtime_teach_clip = config.teach_clip
        self.gradient_checkpointing = config.gradient_checkpointing
        self._surprise_threshold = config.surprise_threshold
        self._surprise_metric = "l2"
        self._allowed_update_levels: set[str] | None = None
        self._allowed_update_layers: set[int] | None = None
        variant = str(config.block_variant).strip().lower()
        
        selfmod_block_config = HOPESelfModBlockConfig(
                dim=config.dim,
                cms_levels=config.cms_levels,
                cms_flush_partial_at_end=config.cms_flush_partial_at_end,
                cms_use_layernorm=config.cms_use_layernorm,
                qk_l2_norm=config.qk_l2_norm,
                selfmod_adaptive_q=config.self_mod_adaptive_q,
                selfmod_local_conv_window=config.self_mod_local_conv_window,
                eta_scale=config.self_mod_lr,
                selfmod_chunk_size=config.self_mod_chunk_size,
                selfmod_chunk_size_memory=config.self_mod_chunk_size_memory,
                selfmod_objective=config.self_mod_objective,
                selfmod_stopgrad_vhat=config.self_mod_stopgrad_vhat,
                selfmod_use_rank1_precond=config.self_mod_use_rank1_precond,
                selfmod_use_alpha=config.self_mod_use_alpha,
                selfmod_use_skip=config.self_mod_use_skip,
                selfmod_momentum=config.self_mod_momentum,
                self_mod_lr=config.self_mod_lr,
                optimizer_configs=config.optimizers or {},
            )
        self.blocks = nn.ModuleList(
                [HOPESelfModBlock(selfmod_block_config) for _ in range(config.num_layers)]
            )
       
        self.norm = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        # Weight tying keeps the LM head gradient aligned with the embedding space.
        self.lm_head.weight = self.embed.weight
        self._latest_update_metrics: Dict[str, float] = {}
        self.set_surprise_metric(config.surprise_metric)
        self.set_surprise_threshold(self._surprise_threshold)
        if config.freeze_backbone:
            self.freeze_backbone()

    def set_teach_runtime(self, *, scale: float | None = None, clip: float | None = None) -> None:
        if scale is not None:
            self._runtime_teach_scale = scale
        if clip is not None:
            self._runtime_teach_clip = clip

    def set_surprise_threshold(self, threshold: float | None) -> None:
        self._surprise_threshold = threshold
        for block in self.blocks:
            cast(_UpdateControlledBlock, block).set_surprise_threshold(threshold)

    def get_surprise_threshold(self) -> float | None:
        return self._surprise_threshold

    def set_surprise_metric(self, metric: str) -> None:
        normalized = str(metric).strip().lower()
        allowed = {"l2", "loss", "logit_entropy"}
        if normalized not in allowed:
            raise ValueError(
                f"Unsupported surprise_metric={metric!r}; expected one of {sorted(allowed)}"
            )
        self._surprise_metric = normalized
        for block in self.blocks:
            cast(_UpdateControlledBlock, block).set_surprise_metric(normalized)

    def get_surprise_metric(self) -> str:
        return self._surprise_metric

    def set_allowed_update_levels(self, levels: set[str] | None) -> None:
        self._allowed_update_levels = levels.copy() if levels is not None else None
        for block in self.blocks:
            cast(_UpdateControlledBlock, block).set_allowed_levels(self._allowed_update_levels)

    def get_allowed_update_levels(self) -> set[str] | None:
        return None if self._allowed_update_levels is None else self._allowed_update_levels.copy()

    def set_allowed_update_layers(self, layers: set[int] | None) -> None:
        if layers is None:
            self._allowed_update_layers = None
            return
        normalized: set[int] = set()
        total = len(self.blocks)
        for idx in layers:
            layer_idx = int(idx)
            if layer_idx < 0:
                layer_idx = total + layer_idx
            if not (0 <= layer_idx < total):
                raise ValueError(f"Invalid layer index {idx} for model with {total} layers")
            normalized.add(layer_idx)
        self._allowed_update_layers = normalized

    def get_allowed_update_layers(self) -> set[int] | None:
        return None if self._allowed_update_layers is None else self._allowed_update_layers.copy()



    def forward(
        self,
        tokens: torch.Tensor,
        *,
        teach_signal: torch.Tensor | None = None,
        teach_signals: list[torch.Tensor] | None = None,
        fast_state: ModelFastState | None = None,
        surprise_value: float | None = None,
    ) -> torch.Tensor:
        logits, _pre_norm = self.forward_with_pre_norm(
            tokens,
            teach_signal=teach_signal,
            teach_signals=teach_signals,
            fast_state=fast_state,
            surprise_value=surprise_value,
        )
        return logits

















    def forward_with_pre_norm(
        self,
        tokens: torch.Tensor,
        *,
        teach_signal: torch.Tensor | None = None,
        teach_signals: list[torch.Tensor] | None = None,
        fast_state: ModelFastState | None = None,
        surprise_value: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._run_blocks(
            tokens,
            teach_signal=teach_signal,
            teach_signals=teach_signals,
            fast_state=fast_state,
            surprise_value=surprise_value,
        )
        pre_norm = cast(torch.Tensor, x)
        x = self.norm(pre_norm)
        logits = self.lm_head(x)
        if teach_signal is not None:
            self._latest_update_metrics = self._gather_block_stats()
        return logits, pre_norm

    def forward_with_block_outputs(
        self,
        tokens: torch.Tensor,
        *,
        teach_signal: torch.Tensor | None = None,
        teach_signals: list[torch.Tensor] | None = None,
        fast_state: ModelFastState | None = None,
        surprise_value: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        x, block_outputs = self._run_blocks(
            tokens,
            teach_signal=teach_signal,
            teach_signals=teach_signals,
            fast_state=fast_state,
            surprise_value=surprise_value,
            collect_outputs=True,
        )
        pre_norm = x
        x = self.norm(x)
        logits = self.lm_head(x)
        if teach_signal is not None or teach_signals is not None:
            self._latest_update_metrics = self._gather_block_stats()
        return logits, pre_norm, block_outputs














    def _run_blocks(
        self,
        tokens: torch.Tensor,
        *,
        teach_signal: torch.Tensor | None,
        fast_state: ModelFastState | None,
        teach_signals: list[torch.Tensor] | None = None,
        surprise_value: float | None = None,
        collect_outputs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.embed(tokens)
        block_outputs: list[torch.Tensor] = []
        runtime_scale = self._runtime_teach_scale
        runtime_clip = self._runtime_teach_clip
        if teach_signals is not None:
            if len(teach_signals) != len(self.blocks):
                raise ValueError(
                    f"teach_signals length {len(teach_signals)} "
                    f"does not match blocks {len(self.blocks)}"
                )
            if teach_signal is not None:
                raise ValueError("Provide either teach_signal or teach_signals, not both.")
        if fast_state is not None and len(fast_state.blocks) != len(self.blocks):
            raise ValueError("fast_state.blocks length does not match model.blocks")

        require_external = self._surprise_metric in {"loss", "logit_entropy"}
        if require_external and self._surprise_threshold is not None:
            if (teach_signal is not None or teach_signals is not None) and surprise_value is None:
                raise ValueError(
                    f"surprise_metric={self._surprise_metric} requires passing surprise_value "
                    "when model.surprise_threshold is set."
                )

        base_surprise = surprise_value
        scaled_global_signal: torch.Tensor | None = None
        if base_surprise is None and teach_signal is not None and self._surprise_metric == "l2":
            scaled_global_signal = teach_signal * runtime_scale
            if runtime_clip > 0:
                norm = scaled_global_signal.norm(dim=-1, keepdim=True)
                scale = torch.clamp(norm / runtime_clip, min=1.0)
                scaled_global_signal = scaled_global_signal / scale
            base_surprise = float(scaled_global_signal.norm(dim=-1).mean().item())

        for idx, block in enumerate(self.blocks):
            block_state = None if fast_state is None else fast_state.blocks[idx]
            scaled_signal = None
            block_surprise = base_surprise
            if teach_signal is not None:
                if scaled_global_signal is None:
                    scaled_signal = teach_signal * runtime_scale
                    if runtime_clip > 0:
                        norm = scaled_signal.norm(dim=-1, keepdim=True)
                        scale = torch.clamp(norm / runtime_clip, min=1.0)
                        scaled_signal = scaled_signal / scale
                else:
                    scaled_signal = scaled_global_signal
                if (
                    self._allowed_update_layers is not None
                    and idx not in self._allowed_update_layers
                ):
                    scaled_signal = None
            if teach_signals is not None:
                scaled_signal = teach_signals[idx] * self._runtime_teach_scale
                if self._surprise_metric == "l2" and base_surprise is None:
                    block_surprise = float(scaled_signal.norm(dim=-1).mean().item())
                if self._runtime_teach_clip > 0:
                    norm = scaled_signal.norm(dim=-1, keepdim=True)
                    scale = torch.clamp(norm / self._runtime_teach_clip, min=1.0)
                    scaled_signal = scaled_signal / scale
                if (
                    self._allowed_update_layers is not None
                    and idx not in self._allowed_update_layers
                ):
                    scaled_signal = None

            def block_call(
                hidden: torch.Tensor,
                *,
                blk=block,
                sig=scaled_signal,
                st=block_state,
                sv=block_surprise,
            ) -> torch.Tensor:
                return blk(
                    hidden,
                    teach_signal=sig,
                    surprise_value=sv,
                    fast_state=st,
                )

            if torch.is_grad_enabled() and self.training and self.gradient_checkpointing:
                x = checkpoint(block_call, x, use_reentrant=False)
            else:
                x = block_call(x)
            if collect_outputs:
                block_outputs.append(x)
        if collect_outputs:
            return x, block_outputs
        return x









    def _gather_block_stats(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for idx, block in enumerate(self.blocks):
            pop_fn = getattr(block, "pop_update_stats", None)
            if callable(pop_fn):
                stats = cast(Dict[str, Dict[str, float]], pop_fn())
                for level_name, payload in stats.items():
                    prefix = f"layer{idx}.{level_name}"
                    for key, value in payload.items():
                        metrics[f"{prefix}.{key}"] = value
        return metrics






    def pop_update_metrics(self) -> Dict[str, float]:
        metrics = self._latest_update_metrics
        self._latest_update_metrics = {}
        return metrics

    def init_fast_state(self) -> ModelFastState:
        states = []
        for block in self.blocks:
            if isinstance(block, HOPEBlock):
                specs = [block.config.titan_level, *block.config.cms_levels]
                state = build_block_fast_state(
                    titan_module=block.titan_memory,
                    cms_blocks=dict(block.cms.blocks.items()),
                    specs=specs,
                    optimizer_configs=block.config.optimizer_configs,
                    default_lr=block.config.self_mod_lr,
                )
                states.append(state)
            elif isinstance(block, HOPEAttentionBlock):
                specs = list(block.config.cms_levels)
                state = build_block_fast_state(
                    titan_module=None,
                    cms_blocks=dict(block.cms.blocks.items()),
                    specs=specs,
                    optimizer_configs=block.config.optimizer_configs,
                    default_lr=block.config.self_mod_lr,
                )
                states.append(state)
            elif isinstance(block, HOPESelfModBlock):
                specs = list(block.config.cms_levels)
                state = build_block_fast_state(
                    titan_module=None,
                    cms_blocks=dict(block.cms.blocks.items()),
                    selfmod_module=block.selfmod,
                    specs=specs,
                    optimizer_configs=block.config.optimizer_configs,
                    default_lr=block.config.self_mod_lr,
                )
                states.append(state)
            elif isinstance(block, TransformerBlock):
                state = build_block_fast_state(
                    titan_module=None,
                    cms_blocks={},
                    specs=(),
                    optimizer_configs={},
                    default_lr=0.0,
                )
                states.append(state)
            else:
                raise TypeError(f"Unsupported block type for fast state: {type(block)}")
        return ModelFastState(blocks=states)

    def freeze_backbone(self) -> None:
        """
        Freeze the shared transformer spine (embeddings, attention blocks, norm, LM head).
        HOPE/TITAN/CMS memories remain trainable for adapter-style finetuning.
        """
        for p in self.embed.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False
        for p in self.lm_head.parameters():
            p.requires_grad = False
        for block in self.blocks:
            attn = getattr(block, "attn", None)
            if isinstance(attn, nn.Module):
                for p in attn.parameters():
                    p.requires_grad = False












class _UpdateControlledBlock(Protocol):
    def set_surprise_threshold(self, threshold: float | None) -> None: ...

    def set_surprise_metric(self, metric: str) -> None: ...

    def set_allowed_levels(self, allowed: set[str] | None) -> None: ...






























# ------------------------------------------------------------







# from https://github.com/test-time-training/mttt/blob/main/model.py



"""
This code is adapted from
https://github.com/google-research/big_vision/blob/main/big_vision/models/vit.py
"""

from typing import Optional, Any, Sequence

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from inner_loop import *


def posemb_sincos_2d(h, w, width, temperature=10_000., dtype=jnp.float32):
  y, x = jnp.mgrid[:h, :w]
  assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
  omega = jnp.arange(width // 4) / (width // 4 - 1)
  omega = 1. / (temperature**omega)
  y = jnp.einsum("m,d->md", y.flatten(), omega)
  x = jnp.einsum("m,d->md", x.flatten(), omega)
  pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
  return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
  if typ == "learn":
    return self.param(name, nn.initializers.normal(stddev=1/np.sqrt(width)),
                      (1, np.prod(seqshape), width), dtype)
  elif typ == "sincos2d":
    return posemb_sincos_2d(*seqshape, width, dtype=dtype)
  else:
    raise ValueError("Unknown posemb type: %s" % typ)


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim

  @nn.compact
  def __call__(self, x):
    """Apply Transformer MlpBlock module."""
    inits = dict(
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
    )
    n, l, d = x.shape  # pylint: disable=unused-variable
    x = nn.Dense(self.mlp_dim or 4 * d, **inits)(x)
    x = nn.gelu(x)
    x = nn.Dense(d, **inits)(x)
    return x


class Encoder1DBlock(nn.Module):
  """Single transformer encoder block (TTT Layer + MLP)."""
  width: int
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 6
  config: Any = None

  @nn.compact
  def __call__(self, x):
    B, N, d = x.shape  # pylint: disable=unused-variable

    y = nn.LayerNorm()(x)
    if self.config.layer_type == "self_attention":
      y = SelfAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform(),
        deterministic=True,
      )(y, y)
      inner_loss_tuple = (jnp.inf, jnp.inf)  # inner loss only applies to TTT Layers
    elif self.config.layer_type == "linear_attention":
      y = LinearAttention(
        width=self.width,
        num_heads=self.num_heads,
        config=self.config.linear_attention,
      )(y)
      inner_loss_tuple = (jnp.inf, jnp.inf)  # inner loss only applies to TTT Layers
    elif self.config.layer_type == "TTT":
      y, inner_loss_tuple = TTTLayer(width=self.width,
                                     num_heads=self.num_heads,
                                     config=self.config.TTT)(y)
    else:
      raise NotImplementedError("Layer Type %s Not Implemented." % (self.config.layer_type))
    x = x + y

    y = nn.LayerNorm()(x)
    y = MlpBlock(mlp_dim=self.mlp_dim)(y)
    x = x + y

    return x, inner_loss_tuple


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""
  width: int
  depth: int
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  config: Any = None

  @nn.compact
  def __call__(self, x):
    inner_loss_tuple_layers = ()
    # Input Encoder
    for lyr in range(self.depth):
      block = Encoder1DBlock(
          name=f"encoderblock_{lyr}",
          width=self.width, mlp_dim=self.mlp_dim, num_heads=self.num_heads,
          config=self.config)
      x, inner_loss_tuple = block(x)
      inner_loss_tuple_layers += (inner_loss_tuple,)

    return nn.LayerNorm(name="encoder_norm")(x), inner_loss_tuple_layers


class Model(nn.Module):
  width: int
  depth: int
  mlp_dim: int
  num_heads: int
  num_classes: int = 1000
  patch_size: Sequence[int] = (16, 16)
  posemb: str = "sincos2d"
  head_zeroinit: bool = True
  config: Any = None

  def setup(self) -> None:
    self.word_embeddings = nn.Conv(
      features=self.width,
      kernel_size=self.patch_size, 
      strides=self.patch_size,
      padding="VALID",
      param_dtype=jnp.float32,
      name="embedding")

    self.pos_emb = get_posemb(
                   self, self.posemb, (224 // self.patch_size[0], 224 // self.patch_size[1]),
                   self.width, "pos_embedding", jnp.float32)

    self.encoder = Encoder(
      width=self.width,
      depth=self.depth,
      mlp_dim=self.mlp_dim,
      num_heads=self.num_heads,
      config=self.config,
      name="Transformer")

    self.pre_logit = nn.Dense(self.width, name="pre_logits")
    kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
    self.head = nn.Dense(self.num_classes, name="head", **kw)

  def __call__(self, image):
    B, H, W, C = image.shape

    tok_emb = self.word_embeddings(image)
    tok_emb = tok_emb.reshape(B, -1, self.width)

    x = tok_emb + self.pos_emb

    x, inner_loss_tuple_layers = self.encoder(x)

    x = jnp.mean(x, axis=1)
    x = nn.tanh(self.pre_logit(x))
    x = self.head(x)

    return x, inner_loss_tuple_layers



























    
###
# TTT Layer
###
class TTTEncoder(nn.Module):
  mlp_dim: int
  config: Any = None

  @nn.compact
  def __call__(self, x):
    if self.config.inner_encoder == "mlp_1":
      y = nn.Dense(self.mlp_dim, use_bias=self.config.inner_encoder_bias,
                   name="inner_Dense_0")(x)
    elif self.config.inner_encoder == "mlp_2":
      y = nn.Dense(int(self.mlp_dim * 4), use_bias=self.config.inner_encoder_bias,
                   name="inner_Dense_0")(x)
      y = nn.gelu(y)
      y = nn.Dense(self.mlp_dim, use_bias=self.config.inner_encoder_bias,
                   name="inner_Dense_1")(y)
    else:
      raise NotImplementedError("Inner Encoder %s Not Implemented." % (self.config.inner_encoder))

    return y


class DummyLinearLayer(nn.Module):
  width: int
  use_bias: bool
  name: str

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(self.width, use_bias=self.use_bias, name=self.name)(x)
    return x


class DummyLayerNorm(nn.Module):
  name: str

  @nn.compact
  def __call__(self, x):
    x = nn.LayerNorm(name=self.name)(x)
    return x


class DummyNoOp(nn.Module):
  @nn.compact
  def __call__(self, x):
    return x








class TTTLayer(nn.Module):
  width: int
  num_heads: int
  config: Any = None
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.psi = DummyLinearLayer(width=self.width // self.num_heads, use_bias=True, name="psi")
    psi_params = self.psi.init(jax.random.PRNGKey(0), jnp.ones([1, self.width]))["params"]

    self.phi = DummyLinearLayer(width=self.width // self.num_heads, use_bias=True, name="phi")
    phi_params = self.phi.init(jax.random.PRNGKey(0), jnp.ones([1, self.width]))["params"]

    self.g = DummyLinearLayer(width=self.width, use_bias=False, name="g")
    g_params = self.g.init(jax.random.PRNGKey(0), jnp.ones([1, self.width // self.num_heads]))["params"]
    self.g_bias = self.param("g_bias", jax.nn.initializers.zeros, (1, self.width), self.dtype)

    self.h = DummyLinearLayer(width=self.width, use_bias=False, name="h")
    h_params = self.h.init(jax.random.PRNGKey(0), jnp.ones([1, self.width // self.num_heads]))["params"]
    self.h_bias = self.param("h_bias", jax.nn.initializers.zeros, (1, self.width), self.dtype)

    self.encoder = TTTEncoder(mlp_dim=self.width // self.num_heads, config=self.config)
    encoder_params = self.encoder.init(jax.random.PRNGKey(0), jnp.ones([1, self.width // self.num_heads]))["params"]

    def get_multi_head_params(params, kernel_init="xavier_uniform"):
      flat_params = traverse_util.flatten_dict(params, sep="/")
      for k in flat_params.keys():
        new_shape = (self.num_heads, *flat_params[k].shape)
        if "kernel" in k:
          if kernel_init == "xavier_uniform":
            initializer = nn.initializers.xavier_uniform(batch_axis=(0,))
          elif kernel_init == "zero":
            initializer = nn.initializers.zeros
          elif kernel_init == "vs_fan_in":
            initializer = nn.initializers.variance_scaling(scale=1., batch_axis=(0,),
                                                           mode="fan_in", distribution="uniform")
          elif kernel_init == "vs_fan_out":
            initializer = nn.initializers.variance_scaling(scale=1., batch_axis=(0,),
                                                           mode="fan_out", distribution="uniform")
          else:
            raise NotImplementedError("Initializer %s Not Implemented." % (kernel_init))
          p = self.param(k, initializer, new_shape, self.dtype)
        elif 'scale' in k:
          # initialize scale to 1
          p = self.param(k, jax.nn.initializers.ones, new_shape, self.dtype)
        else:
          # initialize bias to 0
          p = self.param(k, jax.nn.initializers.zeros, new_shape, self.dtype)
        flat_params[k] = p
      params_init = traverse_util.unflatten_dict(flat_params, sep="/")
      return params_init

    self.encoder_params = get_multi_head_params(encoder_params, self.config.inner_encoder_init)
    self.psi_params = get_multi_head_params(psi_params, "vs_fan_in")
    self.phi_params = get_multi_head_params(phi_params, "vs_fan_in")
    self.g_params = get_multi_head_params(g_params, "vs_fan_out")
    self.h_params = get_multi_head_params(h_params, "vs_fan_out")

    if self.config.decoder_LN:
      self.decoder_LN = DummyLayerNorm()
      decoder_LN_params = self.decoder_LN.init(jax.random.PRNGKey(0), jnp.ones([1, self.width]))["params"]
    else:
      self.decoder_LN = DummyNoOp()
      decoder_LN_params = {}
    self.decoder_LN_params = get_multi_head_params(decoder_LN_params, "layer_norm")

  def __call__(self, batch):
    @partial(vmap)
    def update_embed(sequence):
      """
      vmap over B sequences
      sequence: [N,d]
      """
      def inner_loss_fn(phi_params, encoder_params, g_params, decoder_LN_params, sequence_head):
        inner_input = self.phi.apply({"params": phi_params}, sequence_head)
        inner_input_transformed = self.encoder.apply({"params": encoder_params}, inner_input)
        inner_output = self.g.apply({"params": g_params}, inner_input_transformed)
        inner_output = inner_output + self.g_bias
        inner_output = self.decoder_LN.apply({"params": decoder_LN_params}, inner_output)
        loss = 0.5 * ((inner_output - sequence_head) ** 2).mean() * self.num_heads  # normalizer = N * d / H
        return loss

      @partial(vmap, axis_name="head")
      def parallelize_over_heads(psi_params, phi_params, encoder_params, g_params, decoder_LN_params, h_params,
                                 sequence_head):
        """
        vmap over H heads
        """
        grad_fn = jax.value_and_grad(inner_loss_fn, argnums=1)

        ilr = jnp.asarray(self.config.inner_lr, dtype=jnp.float32)
        inner_loss_tuple = ()
        # TODO: To avoid OOM, manually copy inner iteration for up to 4 times
        if self.config.SGD:
          N = sequence_head.shape[0]
          shuffle_rng = self.make_rng("idx")
          shuffle_rng = jax.random.fold_in(shuffle_rng, jax.lax.axis_index("head"))
          noise = jax.random.uniform(shuffle_rng, (N,), jnp.float32)
          order = jnp.argsort(noise)
          batches = sequence_head[order].reshape(self.config.inner_itr, N // self.config.inner_itr, -1)

          if self.config.inner_itr >= 1:
            inner_loss, grad = grad_fn(phi_params, encoder_params, g_params, decoder_LN_params, batches[0])
            encoder_params = jax.tree_util.tree_map(lambda p, g: p - ilr[0] * g, encoder_params, grad)
            inner_loss_tuple += (inner_loss,)

          if self.config.inner_itr >= 2:
            inner_loss, grad = grad_fn(phi_params, encoder_params, g_params, decoder_LN_params, batches[1])
            encoder_params = jax.tree_util.tree_map(lambda p, g: p - ilr[1] * g, encoder_params, grad)
            inner_loss_tuple += (inner_loss,)

          if self.config.inner_itr >= 3:
            inner_loss, grad = grad_fn(phi_params, encoder_params, g_params, decoder_LN_params, batches[2])
            encoder_params = jax.tree_util.tree_map(lambda p, g: p - ilr[2] * g, encoder_params, grad)
            inner_loss_tuple += (inner_loss,)

          if self.config.inner_itr >= 4:
            inner_loss, grad = grad_fn(phi_params, encoder_params, g_params, decoder_LN_params, batches[3])
            encoder_params = jax.tree_util.tree_map(lambda p, g: p - ilr[3] * g, encoder_params, grad)
            inner_loss_tuple += (inner_loss,)

        else:
          if self.config.inner_itr >= 1:
            inner_loss, grad = grad_fn(phi_params, encoder_params, g_params, decoder_LN_params, sequence_head)
            encoder_params = jax.tree_util.tree_map(lambda p, g: p - ilr[0] * g, encoder_params, grad)
            inner_loss_tuple += (inner_loss,)

          if self.config.inner_itr >= 2:
            inner_loss, grad = grad_fn(phi_params, encoder_params, g_params, decoder_LN_params, sequence_head)
            encoder_params = jax.tree_util.tree_map(lambda p, g: p - ilr[1] * g, encoder_params, grad)
            inner_loss_tuple += (inner_loss,)

          if self.config.inner_itr >= 3:
            inner_loss, grad = grad_fn(phi_params, encoder_params, g_params, decoder_LN_params, sequence_head)
            encoder_params = jax.tree_util.tree_map(lambda p, g: p - ilr[2] * g, encoder_params, grad)
            inner_loss_tuple += (inner_loss,)

          if self.config.inner_itr >= 4:
            inner_loss, grad = grad_fn(phi_params, encoder_params, g_params, decoder_LN_params, sequence_head)
            encoder_params = jax.tree_util.tree_map(lambda p, g: p - ilr[3] * g, encoder_params, grad)
            inner_loss_tuple += (inner_loss,)

        encoder_params_new = encoder_params
        # TODO: For precise profiling, comment out the below 2 lines to avoid unnecessary compute
        inner_loss_final = inner_loss_fn(phi_params, encoder_params_new, g_params, decoder_LN_params, sequence_head)
        inner_loss_tuple += (inner_loss_final,)

        head_embed_new = self.psi.apply({"params": psi_params}, sequence_head)
        head_embed_new = self.encoder.apply({"params": encoder_params_new}, head_embed_new)
        head_embed_new = self.h.apply({"params": h_params}, head_embed_new)

        return head_embed_new, inner_loss_tuple

      sequence = jnp.repeat(jnp.expand_dims(sequence, axis=0), repeats=self.num_heads, axis=0)

      embed_new, inner_loss_tuple = parallelize_over_heads(self.psi_params, self.phi_params,
                                                           self.encoder_params, self.g_params,
                                                           self.decoder_LN_params, self.h_params,
                                                           sequence)
      embed_new = embed_new.sum(axis=0)
      embed_new = embed_new + self.h_bias

      inner_loss_tuple_sum = ()
      for i in range(len(inner_loss_tuple)):
        inner_loss_tuple_sum += (inner_loss_tuple[i].sum(),)

      return embed_new, inner_loss_tuple_sum

    ttt_output, inner_loss_tuple = update_embed(batch)

    inner_loss_tuple_avg = ()
    for i in range(len(inner_loss_tuple)):
      inner_loss_tuple_avg += (inner_loss_tuple[i].mean(),)
    output = ttt_output

    return output, inner_loss_tuple_avg