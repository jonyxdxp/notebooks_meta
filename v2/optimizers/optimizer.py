
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








# ----------------------------------------------------------




# from https://github.com/shjwudp/megabyte/blob/main/optimizers







# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# AnyPrecisionAdamW: a flexible precision AdamW optimizer
# with optional Kahan summation for high precision weight updates.
# Allows direct control over momentum, variance and auxiliary compensation
# buffer dtypes.
# Optional Kahan summation is used to offset precision reduction for
# the weight updates. This allows full training in BFloat16 (equal or
# better than FP32 results in many cases) due to high precision weight upates.

import torch
from torch.optim.optimizer import Optimizer


class AnyPrecisionAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        use_kahan_summation=False,
        momentum_dtype=torch.float32,
        variance_dtype=torch.bfloat16,
        compensation_buffer_dtype=torch.bfloat16,
    ):
        """
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float, optional): learning rate (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square (default: (0.9, 0.999))
            eps (float, optional): term added to the denominator to improve
                numerical stability (default: 1e-8)
            weight_decay (float, optional): weight decay coefficient (default: 1e-2)

            # Any Precision specific
            use_kahan_summation = creates auxiliary buffer to ensure high precision
            model param updates (default: False)
            momentum_dtype = dtype for momentum  (default: BFloat32)
            variance_dtype = dtype for uncentered variance (default: BFloat16)
            compensation_buffer_dtype  = dtype for Kahan summation
                                         buffer (default: BFloat16). Only used if
                                         ``use_kahan_summation=True``.

            # Usage
            This optimizer implements optimizer states, and Kahan summation
            for high precision updates, all in user controlled dtypes.
            Defaults are variance in BF16, Momentum in FP32.
            This can be run in FSDP mixed precision, amp, or full precision,
            depending on what training pipeline you wish to work with.

            Setting to use_kahan_summation = False, and changing momentum and
            variance dtypes to FP32, reverts this to a standard AdamW optimizer.
        """
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            use_kahan_summation=use_kahan_summation,
            momentum_dtype=momentum_dtype,
            variance_dtype=variance_dtype,
            compensation_buffer_dtype=compensation_buffer_dtype,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        if closure is not None:
            with torch.enable_grad():
                # to fix linter, we do not keep the returned loss for use atm.
                closure()

        for group in self.param_groups:

            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            use_kahan_summation = group["use_kahan_summation"]

            momentum_dtype = group["momentum_dtype"]
            variance_dtype = group["variance_dtype"]
            compensation_buffer_dtype = group["compensation_buffer_dtype"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError(
                        "AnyPrecisionAdamW does not support sparse gradients"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:

                    state["step"] = torch.tensor(0.0)

                    # momentum - EMA of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p,
                        dtype=momentum_dtype,
                    )

                    # variance uncentered - EMA of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p,
                        dtype=variance_dtype,
                    )

                    # optional Kahan summation - accumulated error tracker
                    if use_kahan_summation:
                        state["compensation"] = torch.zeros_like(
                            p,
                            dtype=compensation_buffer_dtype,
                        )

                # main processing -------------------------

                # update the steps for each param group update
                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                grad = p.grad

                # weight decay, AdamW style
                if weight_decay:
                    p.data.mul_(1 - lr * weight_decay)

                # update momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # update uncentered variance
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # adjust using bias1
                bias_correction1 = 1 - beta1**step

                step_size = lr / bias_correction1

                # adjust using bias2
                denom_correction = (1 - beta2**step) ** 0.5  # avoids math import

                centered_variance = (exp_avg_sq.sqrt() / denom_correction).add_(
                    eps, alpha=1
                )

                # lr update to compensation
                if use_kahan_summation:
                    compensation = state["compensation"]

                    compensation.addcdiv_(exp_avg, centered_variance, value=-step_size)

                    # update weights with compensation (Kahan summation)
                    # save error back to compensation for next iteration
                    temp_buffer = p.detach().clone()
                    p.data.add_(compensation)
                    compensation.add_(temp_buffer.sub_(p.data))

                else:
                    # usual AdamW updates
                    p.data.addcdiv_(exp_avg, centered_variance, value=-step_size)




















"""The implementation of the addafactor is adapted from
https://github.com/facebookresearch/fairseq/blob/main/fairseq/optim/adafactor.py

these codes follow the license:
"""

# MIT License
# Copyright (c) Facebook, Inc. and its affiliates.

import math

import torch


class Adafactor(torch.optim.Optimizer):
    """Implements Adafactor algorithm.
    This implementation is based on:
    `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost`
    (see https://arxiv.org/abs/1804.04235)
    Note that this optimizer internally adjusts the learning rate
    depending on the *scale_parameter*, *relative_step* and
    *warmup_init* options. To use a manual (external) learning rate
    schedule you should set `scale_parameter=False` and
    `relative_step=False`.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): external learning rate (default: None)
        eps (tuple[float, float]): regularization constans for square gradient
            and parameter scale respectively (default: (1e-30, 1e-3))
        clip_threshold (float): threshold of root mean square of
            final gradient update (default: 1.0)
        decay_rate (float): coefficient used to compute running averages of square
            gradient (default: -0.8)
        beta1 (float): coefficient used for computing running averages of gradient
            (default: None)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        scale_parameter (bool): if True, learning rate is scaled by root mean square of
            parameter (default: True)
        relative_step (bool): if True, time-dependent learning rate is computed
            instead of external learning rate (default: True)
        warmup_init (bool): time-dependent learning rate computation depends on
            whether warm-up initialization is being used (default: False)
    """

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
        dynamic_weight_decay=False,
    ):
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual lr and relative_step options")
        if warmup_init and not relative_step:
            raise ValueError("warmup_init requires relative_step=True")

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
            dynamic_weight_decay=dynamic_weight_decay,
        )
        super(Adafactor, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    def _get_lr(self, param_group, param_state):
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = (
                1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            )
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    def _get_options(self, param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
            .rsqrt_()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(group, grad_shape)
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                        state["exp_avg_sq_col"] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:]
                        ).to(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                group["lr"] = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad**2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2t).add_(
                        update.mean(dim=-1), alpha=1.0 - beta2t
                    )
                    exp_avg_sq_col.mul_(beta2t).add_(
                        update.mean(dim=-2), alpha=1.0 - beta2t
                    )

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_(
                    (self._rms(update) / group["clip_threshold"]).clamp_(min=1.0)
                )
                update.mul_(group["lr"])

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=1 - group["beta1"])
                    update = exp_avg

                # Hard-coded dynamic weight-decay equation, PaLM: Scaling Language Modeling with Pathways
                # dynamic weight-decay (see https://arxiv.org/abs/1804.04235).
                if group["dynamic_weight_decay"]:
                    group["weight_decay"] = group["lr"] ** 2

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(
                        p_data_fp32, alpha=-group["weight_decay"] * group["lr"]
                    )

                p_data_fp32.add_(-update)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss