"""
Memory Efficient Online Meta Learning (MOML)
=============================================
Implementation of Algorithm 1 from:
  "Memory Efficient Online Meta Learning"
  Acar, Zhu, Saligrama — ICML 2021
  https://proceedings.mlr.press/v139/acar21b.html

Key idea
--------
Online meta-learning tasks arrive one by one.  Instead of storing *all* past
task data (like FTML / FTRL), MOML maintains a fixed, constant-size
state-vector (w_t) that summarises prior experience.  The state is used to
build a lightweight quadratic regulariser that debiases the gradient of the
current task loss, steering the meta-model toward the global optimum without
ever re-reading old data.

Algorithm (one round t)
-----------------------
State carried between rounds:  θ (meta-params),  w,  prev_grad
(where prev_grad = ∇[f_{t-1} ∘ U_{t-1}](θ^t))

1. Adapt:   φ_t = U_t(θ_t)          # one MAML inner step
2. Evaluate: suffer f_t(φ_t)

3. Build regulariser  R_t(θ) = -<prev_grad, θ>  +  α/2 ‖θ - w‖²

4. Inner optimisation (K steps):
       θ_{k+1} = θ_k - β * (∇[f_t ∘ U_t](θ_k) + ∇R_t(θ_k))

   where  ∇R_t(θ) = -prev_grad  +  α*(θ - w)

5. Update state:
       grad_new  = ∇[f_t ∘ U_t](θ^{t+1})     # gradient at *new* meta-params
       w_{t+1}   = ½ (w + θ^{t+1} - (1/α)*grad_new)

B-MOML (buffered variant)
-------------------------
Extends MOML by also averaging over the last B task losses / gradients.
Keeps a FIFO buffer of (support_x, support_y, query_x, query_y) tuples,
or optionally uses a random-admission policy (coin flip with prob p).
"""







import copy
from collections import deque
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Helper: MAML inner-loop adaptation (one gradient step)
# ---------------------------------------------------------------------------

def maml_adapt(
    model: nn.Module,
    support_x: Tensor,
    support_y: Tensor,
    loss_fn: Callable,
    inner_lr: float,
    num_inner_steps: int = 1,
) -> nn.Module:
    """
    Return a *fast-adapted* copy of `model` after `num_inner_steps` gradient
    steps on the support set.  The original model is NOT modified.

    This is the adaptation function U_t(θ) from the paper.
    """
    fast_model = copy.deepcopy(model)
    fast_model.train()

    for _ in range(num_inner_steps):
        preds = fast_model(support_x)
        loss = loss_fn(preds, support_y)
        grads = torch.autograd.grad(loss, fast_model.parameters(),
                                    create_graph=True)
        # Manual SGD step (keeps computation graph for meta-gradient)
        with torch.no_grad():
            for p, g in zip(fast_model.parameters(), grads):
                p -= inner_lr * g

    return fast_model


def query_loss_and_grad(
    fast_model: nn.Module,
    meta_params: List[Tensor],
    query_x: Tensor,
    query_y: Tensor,
    loss_fn: Callable,
) -> Tuple[Tensor, List[Tensor]]:
    """
    Evaluate f_t ∘ U_t at the current meta-params.

    We need the gradient w.r.t. the *original* meta-parameters, not the fast
    model's weights (which are a function of the meta-params via the inner
    loop).  Pass `meta_params = list(model.parameters())` and make sure the
    fast model was built with `create_graph=True` in `maml_adapt`.

    Returns
    -------
    loss  : scalar Tensor
    grads : list of Tensors — ∇[f_t ∘ U_t](θ), one per meta-parameter
    """
    preds = fast_model(query_x)
    loss = loss_fn(preds, query_y)
    grads = torch.autograd.grad(loss, meta_params,
                                allow_unused=True, retain_graph=False)
    # Replace None (unused params) with zeros
    grads = [g if g is not None else torch.zeros_like(p)
             for g, p in zip(grads, meta_params)]
    return loss, grads








# ---------------------------------------------------------------------------
# MOML trainer
# ---------------------------------------------------------------------------

class MOML:
    """
    Memory Efficient Online Meta-Learner (Algorithm 1, Acar et al. 2021).

    Parameters
    ----------
    model       : The meta-model (nn.Module).  Its parameters are θ.
    loss_fn     : Task loss function  loss_fn(predictions, labels) -> scalar.
    inner_lr    : η  — learning rate for the MAML adaptation step.
    meta_lr     : β  — learning rate for the outer / meta update.
    alpha       : α  — regulariser coefficient (controls correction strength).
    K           : Number of corrected gradient steps per round.
    num_inner   : Number of inner MAML steps (default = 1).
    buffer_size : B for B-MOML (0 = pure MOML, no task buffer).
    buffer_prob : p for the random-admission buffer policy (0 < p ≤ 1).
    device      : torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        alpha: float = 1.0,
        K: int = 1,
        num_inner: int = 1,
        buffer_size: int = 0,
        buffer_prob: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.alpha = alpha
        self.K = K
        self.num_inner = num_inner
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        # ---- State vectors (fixed memory, O(d)) -------------------------
        # w_t  — direction-correction anchor
        self.w = [torch.zeros_like(p.data)
                  for p in self.model.parameters()]
        # prev_grad  — ∇[f_{t-1} ∘ U_{t-1}](θ^t), initialised to 0
        self.prev_grad = [torch.zeros_like(p.data)
                          for p in self.model.parameters()]

        # ---- Optional task buffer for B-MOML ---------------------------
        self.buffer_size = buffer_size
        self.buffer_prob = buffer_prob
        # Each entry: (support_x, support_y, query_x, query_y)
        self._buffer: deque = deque(maxlen=buffer_size if buffer_size > 0 else 1)

        self.round = 0   # task counter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(
        self,
        support_x: Tensor,
        support_y: Tensor,
        query_x: Tensor,
        query_y: Tensor,
    ) -> float:
        """
        Process one online task (one round of Algorithm 1).

        Returns the query-set loss *before* the meta update (i.e. what the
        algorithm "suffers" at round t).
        """
        sx = support_x.to(self.device)
        sy = support_y.to(self.device)
        qx = query_x.to(self.device)
        qy = query_y.to(self.device)

        self.round += 1

        # ---- Step 1: compute adapted model φ_t = U_t(θ_t) --------------
        meta_params = list(self.model.parameters())
        fast_model = maml_adapt(self.model, sx, sy,
                                self.loss_fn, self.inner_lr, self.num_inner)

        # ---- Step 2: suffer loss f_t(φ_t) -------------------------------
        with torch.no_grad():
            preds = fast_model(qx)
            suffered_loss = self.loss_fn(preds, qy).item()

        # ---- Step 3 + 4: K corrected gradient-descent steps on θ --------
        if self.buffer_size > 0:
            self._maybe_add_to_buffer(sx, sy, qx, qy)
            self._meta_update_buffered(sx, sy, qx, qy, fast_model, meta_params)
        else:
            self._meta_update(sx, sy, qx, qy, fast_model, meta_params)

        return suffered_loss

    def get_adapted_model(
        self,
        support_x: Tensor,
        support_y: Tensor,
    ) -> nn.Module:
        """
        Return a task-specific model for inference (does NOT update state).
        """
        sx = support_x.to(self.device)
        sy = support_y.to(self.device)
        return maml_adapt(self.model, sx, sy,
                          self.loss_fn, self.inner_lr, self.num_inner)

    # ------------------------------------------------------------------
    # Core update  (pure MOML, buffer_size == 0)
    # ------------------------------------------------------------------

    def _meta_update(self, sx, sy, qx, qy, fast_model, meta_params):
        """
        Algorithm 1 inner loop + state update.

        Performs K corrected gradient steps:
            θ_{k+1} = θ_k  -  β * ( ∇[f_t ∘ U_t](θ_k)  +  ∇R_t(θ_k) )

        where  ∇R_t(θ) = -prev_grad + α*(θ - w)
        """
        # We will work with the model's parameters in-place.
        # The computation graph from maml_adapt is needed for the meta-grad.

        # ---- K inner optimisation steps ----------------------------------
        for _ in range(self.K):
            # Re-build the fast model at the *current* meta-params each step
            # (following Eq. 7 — the full f_t ∘ U_t gradient)
            fast_k = maml_adapt(self.model, sx, sy,
                                 self.loss_fn, self.inner_lr, self.num_inner)
            meta_params_k = list(self.model.parameters())

            _, grads_ft = query_loss_and_grad(fast_k, meta_params_k, qx, qy,
                                              self.loss_fn)

            # ∇R_t(θ_k) = -prev_grad + α*(θ_k - w)
            grad_reg = [
                -pg + self.alpha * (p.data - wv)
                for pg, p, wv in zip(self.prev_grad, meta_params_k, self.w)
            ]

            # θ_{k+1} = θ_k - β*(grad_ft + grad_reg)
            with torch.no_grad():
                for p, gf, gr in zip(self.model.parameters(),
                                     grads_ft, grad_reg):
                    p -= self.meta_lr * (gf + gr)

        # ---- State update: w_{t+1}  ----------------------------------------
        # Compute ∇[f_t ∘ U_t](θ^{t+1}) at the NEW meta-params
        fast_new = maml_adapt(self.model, sx, sy,
                               self.loss_fn, self.inner_lr, self.num_inner)
        meta_params_new = list(self.model.parameters())
        _, grad_new = query_loss_and_grad(fast_new, meta_params_new, qx, qy,
                                          self.loss_fn)

        with torch.no_grad():
            # w_{t+1} = ½ * (w_t  +  θ^{t+1}  -  (1/α)*grad_new)   (Eq. 8)
            self.w = [
                0.5 * (wv + p.data - (1.0 / self.alpha) * gn)
                for wv, p, gn in zip(self.w, self.model.parameters(), grad_new)
            ]
            # prev_grad is updated to grad at the new meta-params
            # (this will become ∇f_{t} ∘ U_{t}(θ^{t+1}) used in round t+1)
            self.prev_grad = [g.detach().clone() for g in grad_new]

    # ------------------------------------------------------------------
    # Buffered update  (B-MOML, buffer_size > 0)
    # ------------------------------------------------------------------

    def _maybe_add_to_buffer(self, sx, sy, qx, qy):
        """Random-admission FIFO buffer (coin-flip with prob p)."""
        if torch.rand(1).item() < self.buffer_prob:
            self._buffer.append((
                sx.detach().cpu(), sy.detach().cpu(),
                qx.detach().cpu(), qy.detach().cpu(),
            ))

    def _meta_update_buffered(self, sx, sy, qx, qy, fast_model, meta_params):
        """
        B-MOML: average loss / gradients over last B tasks plus current task.
        """
        for _ in range(self.K):
            # ---- Collect gradients from current + buffered tasks ----------
            all_grads_ft = []

            # Current task
            fast_k = maml_adapt(self.model, sx, sy,
                                 self.loss_fn, self.inner_lr, self.num_inner)
            _, gf = query_loss_and_grad(fast_k, list(self.model.parameters()),
                                        qx, qy, self.loss_fn)
            all_grads_ft.append(gf)

            # Buffered tasks
            for bsx, bsy, bqx, bqy in self._buffer:
                bsx = bsx.to(self.device)
                bsy = bsy.to(self.device)
                bqx = bqx.to(self.device)
                bqy = bqy.to(self.device)
                fast_b = maml_adapt(self.model, bsx, bsy,
                                    self.loss_fn, self.inner_lr, self.num_inner)
                _, gb = query_loss_and_grad(
                    fast_b, list(self.model.parameters()), bqx, bqy, self.loss_fn)
                all_grads_ft.append(gb)

            # Average over tasks  (L^t_B gradient)
            B = len(all_grads_ft)
            avg_grads = [
                sum(g[i] for g in all_grads_ft) / B
                for i in range(len(all_grads_ft[0]))
            ]

            # ∇R^t_B(θ) — same regulariser form but with prev_grad already
            # averaged (we keep prev_grad as the buffered-avg from last round)
            grad_reg = [
                -pg + self.alpha * (p.data - wv)
                for pg, p, wv in zip(self.prev_grad,
                                     self.model.parameters(), self.w)
            ]

            with torch.no_grad():
                for p, gf, gr in zip(self.model.parameters(), avg_grads, grad_reg):
                    p -= self.meta_lr * (gf + gr)

        # ---- State update -----------------------------------------------
        # Compute averaged grad at new meta-params for w / prev_grad update
        all_grads_new = []
        fast_new = maml_adapt(self.model, sx, sy,
                               self.loss_fn, self.inner_lr, self.num_inner)
        _, gn = query_loss_and_grad(fast_new, list(self.model.parameters()),
                                    qx, qy, self.loss_fn)
        all_grads_new.append(gn)
        for bsx, bsy, bqx, bqy in self._buffer:
            bsx = bsx.to(self.device); bsy = bsy.to(self.device)
            bqx = bqx.to(self.device); bqy = bqy.to(self.device)
            fast_b = maml_adapt(self.model, bsx, bsy,
                                 self.loss_fn, self.inner_lr, self.num_inner)
            _, gb = query_loss_and_grad(
                fast_b, list(self.model.parameters()), bqx, bqy, self.loss_fn)
            all_grads_new.append(gb)

        B = len(all_grads_new)
        avg_grad_new = [
            sum(g[i] for g in all_grads_new) / B
            for i in range(len(all_grads_new[0]))
        ]

        with torch.no_grad():
            self.w = [
                0.5 * (wv + p.data - (1.0 / self.alpha) * gn)
                for wv, p, gn in zip(self.w, self.model.parameters(), avg_grad_new)
            ]
            self.prev_grad = [g.detach().clone() for g in avg_grad_new]















# ---------------------------------------------------------------------------
# Quick smoke-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    import numpy as np

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # -----------------------------------------------------------------------
    # Synthetic sinusoid regression (classic MAML benchmark in mini form)
    # Each "task" is a sine wave with random amplitude ∈ [0.1, 5] and
    # phase ∈ [0, π].
    # -----------------------------------------------------------------------

    def sample_sinusoid_task(n_support=5, n_query=10):
        amplitude = torch.FloatTensor(1).uniform_(0.1, 5.0).item()
        phase     = torch.FloatTensor(1).uniform_(0.0, 3.14159).item()
        x  = torch.FloatTensor(n_support + n_query, 1).uniform_(-5, 5)
        y  = amplitude * torch.sin(x + phase)
        return x[:n_support], y[:n_support], x[n_support:], y[n_support:]

    # Small 2-hidden-layer MLP
    meta_model = nn.Sequential(
        nn.Linear(1, 40), nn.ReLU(),
        nn.Linear(40, 40), nn.ReLU(),
        nn.Linear(40, 1),
    )

    mse = nn.MSELoss()

    # ---- Pure MOML -------------------------------------------------------
    moml = MOML(
        model=meta_model,
        loss_fn=mse,
        inner_lr=0.01,
        meta_lr=0.001,
        alpha=1.0,
        K=1,
        num_inner=1,
        buffer_size=0,   # 0 = pure MOML
    )

    print("=== MOML (pure) — 200 online tasks ===")
    losses = []
    for t in range(200):
        sx, sy, qx, qy = sample_sinusoid_task()
        loss = moml.observe(sx, sy, qx, qy)
        losses.append(loss)
        if (t + 1) % 50 == 0:
            print(f"  Round {t+1:3d} | suffered loss = {loss:.4f} "
                  f"| running avg = {sum(losses[-50:])/50:.4f}")

    # ---- B-MOML (buffer of 5) -------------------------------------------
    meta_model2 = nn.Sequential(
        nn.Linear(1, 40), nn.ReLU(),
        nn.Linear(40, 40), nn.ReLU(),
        nn.Linear(40, 1),
    )

    bmoml = MOML(
        model=meta_model2,
        loss_fn=mse,
        inner_lr=0.01,
        meta_lr=0.001,
        alpha=1.0,
        K=1,
        num_inner=1,
        buffer_size=5,      # B = 5
        buffer_prob=0.8,    # coin-flip with p = 0.8
    )

    print("\n=== B-MOML (B=5, p=0.8) — 200 online tasks ===")
    losses2 = []
    for t in range(200):
        sx, sy, qx, qy = sample_sinusoid_task()
        loss = bmoml.observe(sx, sy, qx, qy)
        losses2.append(loss)
        if (t + 1) % 50 == 0:
            print(f"  Round {t+1:3d} | suffered loss = {loss:.4f} "
                  f"| running avg = {sum(losses2[-50:])/50:.4f}")

    print("\nDone.  Both MOML and B-MOML ran without errors.")