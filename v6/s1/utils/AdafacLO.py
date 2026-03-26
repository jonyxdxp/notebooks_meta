


# from https://github.com/Belilovsky-Lab/pylo/blob/main/pylo/optim/AdafacLO_naive.py

















"""AdafacLO_Naive: An MLP learned optimizer.

This is a PyTorch implementation of small_fc_lopt from: https://arxiv.org/abs/2203.11860

The following code is adapted from the following Jax implementation: https://github.com/google/learned_optimization/blob/main/learned_optimization/learned_optimizers/adafac_mlp_lopt.py
"""
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

from pylo.models.Meta_MLP import MetaMLP


def init_factors(p):
    shape = p.shape
    f_dims = factored_dims(shape)
    shape = shape + (3,)
    if f_dims is not None:
        d1, d0 = f_dims
        vr_shape = tuple(dim for i, dim in enumerate(shape) if i != d0)
        vc_shape = tuple(dim for i, dim in enumerate(shape) if i != d1)
        v_row = torch.zeros(vr_shape, dtype=torch.float32)
        v_col = torch.zeros(vc_shape, dtype=torch.float32)
        return v_row, v_col, torch.tensor([], dtype=torch.float32)

    else:
        v = torch.zeros(shape, dtype=torch.float32)
        return (
            torch.tensor([], dtype=torch.float32),
            torch.tensor([], dtype=torch.float32),
            v,
        )


def safe_rsqrt(x):
    return torch.rsqrt(
        torch.maximum(x, torch.tensor(1e-9, dtype=x.dtype, device=x.device))
    )


def update_factors(
    v_col, v_row, v_full, g, g_shape, decay_rate: float = 0.9, epsilon: float = 1e-30 #! change
):
    f_dims = factored_dims(g_shape)
    mixing_rate = 1.0 - decay_rate
    rp_shape = [1] * len(g_shape)
    g = g.repeat(rp_shape + [decay_rate.shape[-1]])
    grad_sqr = g * g + epsilon
    if f_dims is not None:
        d1, d0 = f_dims
        decay_rate, mixing_rate = decay_rate.squeeze(0), mixing_rate.squeeze(0)
        # print(f_dims, decay_rate.shape, mixing_rate.shape, grad_sqr.shape, v_row.shape, v_col.shape)
        new_v_row = decay_rate * v_row + mixing_rate * torch.mean(grad_sqr, dim=d0)
        new_v_col = decay_rate * v_col + mixing_rate * torch.mean(grad_sqr, dim=d1)

        reduced_d1 = d1 - 1 if d1 > d0 else d1
        row_col_mean = torch.mean(new_v_row, dim=reduced_d1, keepdim=True)

        row_factor = safe_rsqrt(new_v_row / (row_col_mean + 1e-9))
        col_factor = safe_rsqrt(new_v_col)
        # print(f_dims, mixing_rate.shape, g.shape, row_factor.shape, col_factor.shape)
        y = g * row_factor.unsqueeze(d0) * col_factor.unsqueeze(d1)
        return new_v_col, new_v_row, torch.tensor([], dtype=torch.float32), y

    else:
        new_v = decay_rate * v_full + mixing_rate * grad_sqr
        y = g * safe_rsqrt(new_v + 1e-9)
        return (
            torch.tensor([], dtype=torch.float32),
            torch.tensor([], dtype=torch.float32),
            new_v,
            y,
        )


def tanh_embedding(x):
    x = torch.tensor(x, dtype=torch.float32)
    timescales = torch.tensor(
        [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000], dtype=torch.float32
    )
    embeddings = torch.tanh(x / timescales - 1.0)
    return embeddings


def second_moment_normalizer(x, axis, eps=1e-5):
    mean_squared = torch.mean(x**2, dim=axis, keepdim=True)
    return x * torch.rsqrt(eps + mean_squared)


def factored_dims(shape):
    if len(shape) < 2:
        return None
    sorted_dims = np.argsort(shape)
    return int(sorted_dims[-2]), int(sorted_dims[-1])


def decay_to_param(x):
    return torch.log(1 - x) / 10.0


def param_to_decay(x):
    return 1 - torch.exp(x * 10.0)


class AdafacLO_naive(Optimizer):

    def __init__(
        self,
        params,
        momentum_decays=[0.15216392, 0.14245212, 0.06812963],
        rms_decays=[0.01079706],
        adafactor_decays=[0.18621896, -0.10864615, -0.06185547],
        lr=1.0,
        exp_mult=0.001,
        step_mult=0.01,
        input_size=39,
        hidden_size=32,
        hidden_layers=1,
        initial_momentum_decays=(0.9, 0.99, 0.999),
        initial_rms_decays=(0.999,),
        initial_adafactor_decays=(0.9, 0.99, 0.999),
        max_grad_norm=None,
        concat_weights=True,
        make_separate_weights=False,
        split_weights=False,
        clip_grad=False,
        weight_decay=0.0,
        mup_lrs=None,
        hf_key: Optional[str] = "btherien/mulo",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        momentum_decays = torch.tensor(momentum_decays).to(self.device)
        rms_decays = torch.tensor(rms_decays).to(self.device)
        adafactor_decays = torch.tensor(adafactor_decays).to(self.device)
        mom_decay = param_to_decay(
            decay_to_param(torch.tensor(initial_momentum_decays, device=self.device)) + momentum_decays
        )
        rms_decays = param_to_decay(
            decay_to_param(torch.tensor(initial_rms_decays, device=self.device)) + rms_decays
        )
        adafactor_decays = param_to_decay(
            decay_to_param(torch.tensor(initial_adafactor_decays, device=self.device)) + adafactor_decays
        )
        clip_mom_decays = torch.clip(mom_decay.clone().detach(), 0.0, 1.0).to(
            self.device
        )
        clip_rms_decays = torch.clip(rms_decays.clone().detach(), 0.0, 1.0).to(
            self.device
        )
        clip_adafactor_decays = torch.clip(
            adafactor_decays.clone().detach(), 0.0, 1.0
        ).to(self.device)
        defaults = dict(
            lr=lr,
            exp_mult=exp_mult,
            step_mult=step_mult,
            initial_momentum_decays=clip_mom_decays,
            initial_rms_decays=clip_rms_decays,
            initial_adafactor_decays=clip_adafactor_decays,
            concat_weights=concat_weights,
            make_separate_weights=make_separate_weights,
            split_weights=split_weights,
            clip_grad=clip_grad,
            weight_decay=weight_decay,
            mup_lrs=mup_lrs,
            max_grad_norm=max_grad_norm,
        )
        super(AdafacLO_naive, self).__init__(params, defaults)


        self.network = MetaMLP.from_pretrained(hf_key).to(self.device)



    @torch.no_grad()
    def step(self, loss=None):
        for group in self.param_groups:
            exp_mult = group["exp_mult"]
            step_mult = group["step_mult"]
            max_grad_norm = group["max_grad_norm"]
            weight_decay = group["weight_decay"]
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1
            for p in group["params"]:
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(p, max_grad_norm)
                beta_m = group["initial_momentum_decays"]
                beta_rms = group["initial_rms_decays"]
                beta_adafactor = group["initial_adafactor_decays"]
                p_shape = p.shape

                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["mom"] = torch.zeros(p_shape + (3,)).to(self.device)
                    state["rms"] = torch.zeros(p_shape + (1,)).to(self.device)
                    state["fac_vec_row"], state["fac_vec_col"], state["fac_vec_v"] = (
                        init_factors(p)
                    )
                    state["fac_vec_row"], state["fac_vec_col"], state["fac_vec_v"] = (
                        state["fac_vec_row"].to(self.device),
                        state["fac_vec_col"].to(self.device),
                        state["fac_vec_v"].to(self.device),
                    )

                batch_p = p.unsqueeze(-1)
                batch_g = grad.unsqueeze(-1)

                training_step_feature = tanh_embedding(group["step"] - 1).to(
                    self.device
                )
                axis = list(range(len(p_shape)))
                for _ in axis:
                    beta_m = beta_m[None, ...]
                    beta_rms = beta_rms[None, ...]
                    beta_adafactor = beta_adafactor[None, ...]
                    training_step_feature = training_step_feature[None, ...]
                training_step_feature = training_step_feature.repeat(p_shape + (1,))

                mom = state["mom"]
                rms = state["rms"]
                mom.mul_(beta_m).add_((1 - beta_m) * batch_g)
                rms.mul_(beta_rms).add_((1 - beta_rms) * (batch_g**2))
                (
                    state["fac_vec_col"],
                    state["fac_vec_row"],
                    state["fac_vec_v"],
                    fac_g,
                ) = update_factors(
                    state["fac_vec_col"],
                    state["fac_vec_row"],
                    state["fac_vec_v"],
                    batch_g,
                    p_shape,
                    beta_adafactor,
                )
                fac_vec_col, fac_vec_row, fac_vec_v = (
                    state["fac_vec_col"],
                    state["fac_vec_row"],
                    state["fac_vec_v"],
                )
                rsqrt = torch.rsqrt(rms + 1e-6)
                # inps = [batch_p, batch_g, mom, rms, mom * rsqrt, rsqrt, fac_g]
                inps = [batch_g, batch_p, mom, rms, mom * rsqrt, rsqrt, fac_g]

                f_dims = factored_dims(p_shape)
                if f_dims is not None:
                    d1, d0 = f_dims
                    rp_row = [1] * (1 + len(p_shape))
                    rp_col = [1] * (1 + len(p_shape))
                    rp_row[d0] = p_shape[d0]
                    rp_col[d1] = p_shape[d1]
                    row_feat = fac_vec_row.unsqueeze(d0).repeat(rp_row)
                    col_feat = fac_vec_col.unsqueeze(d1).repeat(rp_col)

                    inps.extend(
                        [
                            row_feat,
                            col_feat,
                            torch.rsqrt(row_feat + 1e-8),
                            torch.rsqrt(col_feat + 1e-8),
                        ]
                    )
                    reduced_d1 = d1 - 1 if d1 > d0 else d1 #!r change
                    row_col_mean = fac_vec_row.mean(dim=reduced_d1, keepdim=True)
                    row_factor = safe_rsqrt(fac_vec_row / (row_col_mean + 1e-9)) #!r change
                    col_factor = safe_rsqrt(fac_vec_col)
                    fac_mom_mult = (
                        mom * row_factor.unsqueeze(d0) * col_factor.unsqueeze(d1)
                    )
                    inps.append(fac_mom_mult)
                else:
                    inps.extend(
                        [
                            fac_vec_v,
                            fac_vec_v,
                            torch.rsqrt(fac_vec_v + 1e-8),
                            torch.rsqrt(fac_vec_v + 1e-8),
                        ]
                    )
                    fac_mom_mult = mom * torch.pow(fac_vec_v + 1e-6, -0.5)
                    inps.append(fac_mom_mult)
                inps = torch.cat(inps, dim=-1)
                inps = second_moment_normalizer(inps, axis=axis)
                inp_stack = torch.cat([inps, training_step_feature], dim=-1)

                direction, magnitude = self.network(inp_stack).split(1, dim=-1)
                step = (
                    direction * torch.exp(magnitude * exp_mult) * step_mult
                ).squeeze(-1)
                p.add_(step, alpha=-group["lr"])
                if weight_decay > 0:
                    p.add_(p, alpha=-weight_decay * group["lr"])
        return
    




















# ------------------------------------------------------------




# from https://github.com/Belilovsky-Lab/pylo/blob/main/pylo/optim/AdafacLO_cuda.py









"""AdafacLO_CUDA: An Cuda-accelerated MLP learned optimizer.

This is a PyTorch implementation of small_fc_lopt from: https://arxiv.org/abs/2203.11860

The following code is adapted from the following Jax implementation: https://github.com/google/learned_optimization/blob/main/learned_optimization/learned_optimizers/adafac_mlp_lopt.py
Inspiration for Adafactor from : https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/adafactor_bv.py
"""

from collections import OrderedDict
from math import isnan
from typing import List, Optional, Tuple, Union

from numpy import dtype
from sympy import beta
import torch
from torch import Tensor, exp_
from torch.optim import Optimizer
from torch import nn
import cuda_lo

from pylo.models.Meta_MLP import MetaMLP


def _get_scalar_dtype():
    """Get the scalar dtype that the optimizer uses for state"""
    return torch.float64


def decay_to_param(x):
    return torch.log(1 - x) / 10.0


def param_to_decay(x):
    return 1 - torch.exp(x * 10.0)


def safe_rsqrt(x):
    return torch.rsqrt(
        torch.maximum(x, torch.tensor(1e-9, dtype=x.dtype, device=x.device))
    )


def tanh_embedding(x):
    x = torch.tensor(x, dtype=torch.float32)
    timescales = torch.tensor(
        [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000], dtype=torch.float32
    )
    embeddings = torch.tanh(x / timescales - 1.0)
    return embeddings


def _factored_dims(
    shape: Tuple[int, ...], factored: bool, min_dim_size_to_factor: int
) -> Optional[tuple[int, int]]:
    """Whether to use a factored second moment estimator.

    This function returns a tuple with the two largest axes to reduce over.
    If no two dimensions have size >= min_dim_size_to_factor, return None.

    Args:
      shape: an input shape
      factored: whether to use factored second-moment estimator for > 2d vars.
      min_dim_size_to_factor: only factor accumulator if two array dimensions have at least this size.

    Returns:
      None or a tuple of ints
    """
    if not factored or len(shape) < 2:
        return None
    sorted_dims = sorted(((x, i) for i, x in enumerate(shape)))
    if shape[sorted_dims[-2][1]] < min_dim_size_to_factor:
        return None
    return int(sorted_dims[-2][1]), int(sorted_dims[-1][1])


class AdafacLO_CUDA(Optimizer):
    """
    PyTorch implementation of AdafacLO, a learned optimizer based on the Adafactor features.
    This optimizer is designed to work with the MetaMLP model for learned optimization.
    """

    def __init__(
        self,
        params,
        lr: float = 1.0,
        min_dim_size_to_factor: int = 32,
        decay_rate: float = 0.8,
        decay_offset: int = 0,
        beta2_cap: float = 0.999,
        initial_momentum_decays=(0.9, 0.99, 0.999),
        initial_rms_decays=(0.999,),
        initial_adafactor_decays=(0.9, 0.99, 0.999),
        momentum_decays=[0.15216392, 0.14245212, 0.06812963],
        rms_decays=[0.01079706],
        adafactor_decays=[0.18621896, -0.10864615, -0.06185547],
        momentum_dtype: Union[str, torch.dtype] = torch.float32,
        eps: Optional[float] = None,
        weight_decay: float = 0.0,
        clipping_threshold: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        unscaled_wd: bool = False,
        step_mult: float = 0.01,
        exp_mult: float = 0.001,
        load_from_file: Optional[str] = None,
        *,
        foreach: Optional[bool] = False,
        hf_key: Optional[str] = "btherien/mulo",
    ):
        if isinstance(momentum_dtype, str):
            if momentum_dtype == "float16":
                momentum_dtype = torch.float16
            elif momentum_dtype == "bfloat16":
                momentum_dtype = torch.bfloat16
            else:
                assert (
                    momentum_dtype == "float32"
                ), f"{momentum_dtype} dtype not supported"
                momentum_dtype = torch.float32
                # FIXME try to check if momentum dtype is appropriate for device? Torch API not great for this.
                # move momentum to device

        defaults = dict(
            lr=lr,
            min_dim_size_to_factor=min_dim_size_to_factor,
            decay_rate=decay_rate,
            decay_offset=decay_offset,
            beta2_cap=beta2_cap,
            momentum_dtype=momentum_dtype,
            eps=eps,
            weight_decay=weight_decay,
            clipping_threshold=clipping_threshold,
            max_grad_norm=max_grad_norm,
            unscaled_wd=unscaled_wd,
            foreach=foreach,
        )
        super().__init__(params, defaults)
        self.device = torch.device("cuda")
        momentum_decays = torch.tensor(momentum_decays, device=self.device)
        rms_decays = torch.tensor(rms_decays, device=self.device)
        adafactor_decays = torch.tensor(adafactor_decays, device=self.device)
        self.metamlp = MetaMLP.from_pretrained(hf_key).to(self.device)

        if load_from_file:
            loaded_params = torch.load(load_from_file)
            self.metamlp.load_state_dict(loaded_params["state_dict"])
            momentum_decays = loaded_params["decays"]["momentum_decays"]
            rms_decays = loaded_params["decays"]["rms_decays"]
            adafactor_decays = loaded_params["decays"]["adafactor_decays"]

        for name, param in self.metamlp.named_parameters():
            param.requires_grad = False

        mom_decay = param_to_decay(
            decay_to_param(torch.tensor(initial_momentum_decays, device=self.device))
            + momentum_decays
        )
        rms_decays = param_to_decay(
            decay_to_param(torch.tensor(initial_rms_decays, device=self.device))
            + rms_decays
        )
        adafactor_decays = param_to_decay(
            decay_to_param(torch.tensor(initial_adafactor_decays, device=self.device))
            + adafactor_decays
        )
        clip_mom_decays = torch.clip(mom_decay.clone().detach(), 0.0, 1.0).to(
            self.device
        )
        clip_rms_decays = torch.clip(rms_decays.clone().detach(), 0.0, 1.0).to(
            self.device
        )
        clip_adafactor_decays = torch.clip(
            adafactor_decays.clone().detach(), 0.0, 1.0
        ).to(self.device)

        self.beta_m = clip_mom_decays
        self.beta_adafactor = clip_adafactor_decays
        self.beta_rms = clip_rms_decays
        self.step_mult = step_mult
        self.exp_mult = exp_mult

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            for p in group["params"]:
                p_state = self.state.get(p, {})
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    p_state["step"] = torch.tensor(
                        float(p_state["step"]), dtype=_get_scalar_dtype()
                    )

                if "exp_avg" in p_state and torch.is_tensor(p_state["exp_avg"]):
                    # FIXME this is a bit of a hack, optimizer.load_state_dict appears to upcast
                    # the momentum to float32 (it's half precision in the state_dict), need to
                    # look into this further. Better to override _process_value_according_to_param_policy?
                    p_state["exp_avg"] = p_state["exp_avg"].to(
                        dtype=self.defaults["momentum_dtype"]
                    )

    @torch.no_grad()
    def step(self, loss=None):

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avg_sq_rs = []
            exp_avg_sq_cs = []
            exp_avg_sqs = []
            state_steps = []
            exp_avgs = []  # For momentum
            max_grad_norm = group["max_grad_norm"]

            for p in group["params"]:
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(p, max_grad_norm)

                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("Sparse gradients not supported")
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]

                if len(state) == 0:
                    # NOTE step on CPU, probably need some more though to make capturable
                    state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())

                    shape = p.grad.shape
                    factored_dims = _factored_dims(
                        shape,
                        factored=True,
                        min_dim_size_to_factor=1,
                    )

                    if factored_dims is not None:
                        dc, dr = factored_dims
                        row_shape = list(p.grad.shape)
                        row_shape[dr] = 1
                        col_shape = list(p.grad.shape)
                        col_shape[dc] = 1
                        state["exp_avg_sq_r"] = p.grad.new_zeros([3] + row_shape)
                        state["exp_avg_sq_c"] = p.grad.new_zeros([3] + col_shape)
                        state["exp_avg_sq"] = torch.zeros_like(
                            p.grad, memory_format=torch.preserve_format
                        )
                    else:
                        state["exp_avg_sq_r"] = p.grad.new_zeros((3,) + p.grad.shape)
                        state["exp_avg_sq_c"] = p.grad.new_zeros((3,) + p.grad.shape)
                        state["exp_avg_sq"] = torch.zeros_like(
                            p.grad, memory_format=torch.preserve_format
                        )

                    state["exp_avg"] = p.grad.new_zeros((3,) + p.grad.shape)

                state_steps.append(state["step"])
                exp_avg_sq_rs.append(state.get("exp_avg_sq_r", None))
                exp_avg_sq_cs.append(state.get("exp_avg_sq_c", None))
                exp_avg_sqs.append(state.get("exp_avg_sq", None))
                exp_avgs.append(state.get("exp_avg", None))

            if group["foreach"]:
                func = _multi_tensor_adafactor
            else:
                func = _single_tensor_adafactor

            func(
                self=self,
                params=params_with_grad,
                grads=grads,
                exp_avg_sq_rs=exp_avg_sq_rs,
                exp_avg_sq_cs=exp_avg_sq_cs,
                exp_avg_sqs=exp_avg_sqs,
                exp_avgs=exp_avgs,
                state_steps=state_steps,
                eps=group["eps"],
                lr=group["lr"],
                step_mult=self.step_mult,
                exp_mult=self.exp_mult,
                weight_decay=group["weight_decay"],
                momentum_dtype=group["momentum_dtype"],
                clipping_threshold=group["clipping_threshold"],
                unscaled_wd=group["unscaled_wd"],
            )

        return loss


def _single_tensor_adafactor(
    self,
    params: List[Tensor],
    grads: List[Tensor],
    exp_avg_sq_rs: List[Optional[Tensor]],
    exp_avg_sq_cs: List[Optional[Tensor]],
    exp_avg_sqs: List[Optional[Tensor]],
    exp_avgs: List[Optional[Tensor]],
    state_steps: List[Tensor],
    *,
    eps: float,
    lr: float,
    step_mult: float,
    exp_mult: float,
    weight_decay: float,
    momentum_dtype: Union[str, torch.dtype],
    clipping_threshold: Optional[float],
    unscaled_wd: bool,
):
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg_sq_r = exp_avg_sq_rs[i]
        exp_avg_sq_c = exp_avg_sq_cs[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg = exp_avgs[i]
        step_t = state_steps[i]
        if eps is None:
            # default eps for avoiding div by zero, diff from float type eps
            eps = 1e-7 if grad.dtype == torch.float16 else 1e-30

        # Update step
        step_t += 1
        one_minus_beta2_t = 1 - self.beta_rms

        grad_sqr = torch.square(grad) + eps
        # NOTE application of eps (epsilon1) mirrors the optax/big vision/t5x approach
        # if exp_avg_sq is None:
        # factorized second moment

        inter_dr_dc = _factored_dims(grad.shape, True, min_dim_size_to_factor=1)
        if inter_dr_dc is not None:
            dc, dr = inter_dr_dc
            exp_avg_sq_r.lerp_(
                grad_sqr.mean(dim=dr, keepdim=True)[None, ...],
                1 - self.beta_adafactor.view(-1, *[1] * grad.dim()).to(grad_sqr.dtype),
            )
            exp_avg_sq_c.lerp_(
                grad_sqr.mean(dim=dc, keepdim=True)[None, ...],
                1 - self.beta_adafactor.view(-1, *[1] * grad.dim()).to(grad_sqr.dtype),
            )
            exp_avg_sq.lerp_(grad_sqr, one_minus_beta2_t.to(grad_sqr.dtype))
            reduce_dc = dc - 1 if dc > dr else dc
            row_col_mean = exp_avg_sq_r.mean(dim=reduce_dc, keepdim=True)
            row_factor = safe_rsqrt(exp_avg_sq_r / (row_col_mean + 1e-9))
            col_factor = safe_rsqrt(exp_avg_sq_c)
            vector_like = 0
        else:
            dc = dr = 0
            exp_avg_sq_r.lerp_(
                grad_sqr[None, ...],
                1 - self.beta_adafactor.view(-1, 1).to(grad_sqr.dtype),
            )
            exp_avg_sq_c.lerp_(
                grad_sqr[None, ...],
                1 - self.beta_adafactor.view(-1, 1).to(grad_sqr.dtype),
            )
            exp_avg_sq.lerp_(grad_sqr, one_minus_beta2_t.to(grad_sqr.dtype))
            row_factor = safe_rsqrt(exp_avg_sq_r + 1e-9)
            col_factor = torch.ones_like(row_factor)
            vector_like = 1

        # Apply momentum (in different dtype)

        if False:
            exp_avg.lerp_(
                grad.to(momentum_dtype)[None, ...],
                1 - self.beta_m.to(momentum_dtype).view([-1] + [1] * grad.dim()),
            )  # ema
        else:
            exp_avg.lerp_(
                grad[None, ...],
                1 - self.beta_m.view([-1] + [1] * grad.dim()).to(grad.dtype),
            )  # ema

        # Scale by learning rate
        # update.mul_(lr)

        second_moment = torch.zeros([28], device="cuda")

        cuda_lo.learned_optimizer_kernel(
            grad,
            param,
            exp_avg,
            exp_avg_sq,
            exp_avg_sq_r,
            exp_avg_sq_c,
            row_factor,
            col_factor,
            second_moment,
            self.metamlp.network.input.weight.to(grad.dtype),
            self.metamlp.network.input.bias.to(grad.dtype),
            self.metamlp.network.linear_0.weight.to(grad.dtype),
            self.metamlp.network.linear_0.bias.to(grad.dtype),
            self.metamlp.network.output.weight.to(grad.dtype),
            self.metamlp.network.output.bias.to(grad.dtype),
            lr,
            step_mult,
            exp_mult,
            1e-6,
            step_t - 1,
            0.0,
            dc,
            dr,
            vector_like,
        )
        if weight_decay > 0:
            param.add_(param, alpha=-weight_decay * lr)


def _multi_tensor_adafactor(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avg_sq_rs: List[Optional[Tensor]],
    exp_avg_sq_cs: List[Optional[Tensor]],
    exp_avg_sqs: List[Optional[Tensor]],
    exp_avgs: List[Optional[Tensor]],
    state_steps: List[Tensor],
    *,
    beta2_decay: float,
    beta2_cap: float,
    min_dim_size_to_factor: int,
    eps: float,
    lr: float,
    weight_decay: float,
    momentum: Optional[float],
    momentum_dtype: Union[str, torch.dtype],
    clipping_threshold: Optional[float],
    unscaled_wd: bool,
):
    # FIXME TODO
    assert False, "multi-tensor fn (foreach=True) not implemented yet"