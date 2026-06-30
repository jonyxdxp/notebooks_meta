
# Optimizers / Update Rules







# from https://github.com/kmccleary3301/nested_learning/blob/main/src/nested_learning/optim/deep.py








from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class DeepMomentumState:
    grad_avg: Optional[torch.Tensor] = None
    sq_avg: Optional[torch.Tensor] = None


class DeepMomentum(nn.Module):
    """Implements momentum variants described in the NL paper."""

    def __init__(
        self,
        *,
        beta: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        variant: str = "preconditioned",
    ) -> None:
        super().__init__()
        self.beta = beta
        self.beta2 = beta2
        self.eps = eps
        self.variant = variant
        self.state: dict[str, DeepMomentumState] = {}
        self.nonlinearity = nn.Tanh() if variant in {"dmgd", "muon"} else nn.Identity()
        self.last_metrics: dict[str, float] = {}

    def reset_state(self) -> None:
        self.state.clear()

    def _precondition(self, grad: torch.Tensor, state: DeepMomentumState) -> torch.Tensor:
        if state.sq_avg is None or state.sq_avg.shape != grad.shape:
            state.sq_avg = torch.zeros_like(grad)
        state.sq_avg.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
        denom = state.sq_avg.sqrt().add_(self.eps)
        return grad / denom

    def _nl_precondition(
        self,
        grad: torch.Tensor,
        context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        metrics: dict[str, float] = {
            "ctx_norm": 0.0,
            "proj_norm": 0.0,
            "proj_skipped": 0.0,
        }
        if context is None:
            return grad, metrics
        ctx = context
        if ctx.ndim > 1:
            ctx = ctx.reshape(-1, ctx.shape[-1]).mean(dim=0)
        ctx_norm = torch.norm(ctx)
        metrics["ctx_norm"] = ctx_norm.item()

        if ctx_norm > 0:
            if grad.ndim == 0 or grad.shape[-1] != ctx.shape[-1]:
                metrics["proj_skipped"] = 1.0
                return grad, metrics
            unit = ctx / (ctx_norm + self.eps)
            # Project grad orthogonal to context (rank-1 projector).
            projection = (grad * unit).sum(dim=-1, keepdim=True) * unit
            update = grad - projection
            metrics["proj_norm"] = torch.norm(update).item()
            return update, metrics
        return grad, metrics

    def forward(  # type: ignore[override]
        self,
        grad: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        param_key: str | None = None,
    ) -> torch.Tensor:
        key = param_key or "__default__"
        state = self.state.get(key)
        if state is None:
            state = DeepMomentumState()
            self.state[key] = state
        if state.grad_avg is None or state.grad_avg.shape != grad.shape:
            state.grad_avg = torch.zeros_like(grad)
        self.last_metrics = {}
        update = grad
        if self.variant in {"preconditioned", "muon"}:
            update = self._precondition(grad, state)
        if self.variant == "l2_objective":
            update = grad + 0.1 * torch.mean(grad, dim=-1, keepdim=True)
        if self.variant == "nl_l2_precond":
            update, metrics = self._nl_precondition(grad, context)
            self.last_metrics.update(metrics)
        if self.variant in {"dmgd", "muon"}:
            update = self.nonlinearity(update)
        state.grad_avg.mul_(self.beta).add_(update, alpha=1 - self.beta)
        return state.grad_avg
















# --------------------------------------------------------




# from https://github.com/test-time-training/e2e/blob/main/ttt/optimizers.py





import re

import jax.numpy as jnp
import optax

from ttt.config import AdamWOptimizerConfig, OptimizerConfig, SGDOptimizerConfig
from ttt.utils.filter_utils import get_mask_fn


def make_adamw_optimizer(config: AdamWOptimizerConfig, weight_decay_mask=None):
    if config.lr == 0.0:
        learning_rate_schedule = optax.constant_schedule(0.0)
    else:
        learning_rate_schedule = optax.warmup_cosine_decay_schedule(
            init_value=config.init_lr,
            peak_value=config.lr,
            warmup_steps=config.lr_warmup_steps,
            decay_steps=config.lr_decay_steps,
            end_value=config.end_lr,
        )

    optimizer_info = dict(learning_rate_schedule=learning_rate_schedule)

    if not config.emb_wd:
        exclude_emb = lambda name: False if re.search("wte", name) else True  # no wd on word embedding
        weight_decay_mask = lambda params: get_mask_fn(exclude_emb, params)
    else:
        weight_decay_mask = None

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.clip_gradient),
        optax.adamw(
            learning_rate=learning_rate_schedule,
            weight_decay=config.weight_decay,
            b1=config.b1,
            b2=config.b2,
            mask=weight_decay_mask,
            mu_dtype=jnp.bfloat16 if config.bf16_momentum else jnp.float32,
        ),
    )

    return optimizer, optimizer_info


def make_sgd_optimizer(config: SGDOptimizerConfig, ilr_multiplier: jnp.ndarray = None):
    learning_rate_schedule = optax.constant_schedule(config.lr * ilr_multiplier)
    optimizer_info = dict(learning_rate_schedule=learning_rate_schedule)
    if config.clip_gradient > 0.0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.clip_gradient),
            optax.sgd(learning_rate=learning_rate_schedule, momentum=None),
        )
    else:
        optimizer = optax.sgd(learning_rate=learning_rate_schedule, momentum=None)
    return optimizer, optimizer_info


def make_optimizer(optimizer_config: OptimizerConfig, ilr_multiplier: jnp.ndarray = None) -> tuple[optax.GradientTransformation, dict]:
    if optimizer_config.optimizer_type == "adamw":
        del ilr_multiplier
        optimizer, optimizer_info = make_adamw_optimizer(optimizer_config)
    elif optimizer_config.optimizer_type == "sgd":
        optimizer, optimizer_info = make_sgd_optimizer(optimizer_config, ilr_multiplier)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_config.optimizer_type}")

    return optimizer, optimizer_info