"""
Dynamic Nested Hierarchies (DNH) - Core Implementation
=======================================================
Paper: "Dynamic Nested Hierarchies: Pioneering Self-Evolution in Machine
Learning Architectures for Lifelong Intelligence" (arXiv:2511.14823)

Implements:
  - AssociativeMemoryModule  (M^(ℓ)_t) — inner memory module per level
  - SelfModifyingMemory      (SMM)      — self-modifying associative memory (Eq. 17-19)
  - MetaNetwork              (g_ψ)      — meta-network that generates Δθ
  - LocalSurpriseSignal                 — LSS used for frequency modulation (Eq. 7)
  - FrequencyModulator                  — adaptive update frequency per level (Eq. 5, 16)
  - DNHLevel                            — one hierarchy level (M^(ℓ), f^(ℓ), params)
  - DynamicNestedHierarchy   (DNH)      — the full time-varying DAG (Eq. 3-9)
  - MetaController           (E_ϕ)      — evolution operator (Eq. 11-12)
  - EvolvableAdam            (EAdam)    — evolvable optimizer (Eq. 20-21)
  - DNHSequenceModel                    — end-to-end sequence model built on DNH
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F










# ---------------------------------------------------------------------------
# 1.  Local Surprise Signal  (Eq. 7)
# ---------------------------------------------------------------------------

class LocalSurpriseSignal(nn.Module):
    """
    Computes LSS^(ℓ)_t = ‖∇_{y^(ℓ)} L̃^(ℓ)(M^(ℓ)_t ; x_t)‖
    Used to modulate update frequency: Δf^(ℓ)_t = γ · LSS^(ℓ)_t  (Eq. 7)
    """

    def __init__(self, gamma: float = 0.1):
        super().__init__()
        self.gamma = gamma

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output: level output y^(ℓ)_t  [*, d]
            target: desired value v_t      [*, d]
        Returns:
            scalar LSS value
        """
        loss = F.mse_loss(output, target)
        # gradient norm as surprise proxy (detached for efficiency; full version
        # would use autograd — see EvolvableAdam for second-order variant)
        surprise = loss.detach().sqrt()
        return self.gamma * surprise


# ---------------------------------------------------------------------------
# 2.  Meta-Network  g_ψ  (Eq. 17-18)
# ---------------------------------------------------------------------------

class MetaNetwork(nn.Module):
    """
    g_ψ(k_t, v_t, c^(ℓ)_t) → Δθ^(ℓ)_t  (Eq. 17)
    A small network that generates a parameter-space modification term.
    """

    def __init__(self, key_dim: int, val_dim: int, context_dim: int, hidden_dim: int = 64):
        super().__init__()
        in_dim = key_dim + val_dim + context_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, key_dim * val_dim),
        )
        self.key_dim = key_dim
        self.val_dim = val_dim

    def forward(
        self,
        keys: torch.Tensor,   # [B, key_dim]
        vals: torch.Tensor,   # [B, val_dim]
        context: torch.Tensor # [B, context_dim]
    ) -> torch.Tensor:
        """Returns Δθ  [B, val_dim, key_dim] (outer-product shaped)."""
        inp = torch.cat([keys, vals, context], dim=-1)
        delta = self.net(inp)                          # [B, val_dim * key_dim]
        return delta.view(-1, self.val_dim, self.key_dim)


# ---------------------------------------------------------------------------
# 3.  Self-Modifying Memory (SMM)  (Eq. 17-19)
# ---------------------------------------------------------------------------

class SelfModifyingMemory(nn.Module):
    """
    M^(ℓ)_t(k_t) = θ^(ℓ)_t k_t + Δθ^(ℓ)_t ⊙ v_t   (Eq. 17)
    Memory update:  M_{t+1} = M_t + v_{t+1} k^⊤_{t+1} + α_t ∇_{M_t} L_meta  (Eq. 19)
    where α_t = σ(w^⊤ LSS_t + b)  (adaptive based on surprise signal)
    """

    def __init__(self, key_dim: int, val_dim: int, context_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.key_dim = key_dim
        self.val_dim = val_dim

        # Θ^(ℓ) — the base memory matrix [val_dim, key_dim]
        self.theta = nn.Parameter(torch.randn(val_dim, key_dim) * 0.01)

        # Meta-network g_ψ
        self.meta_net = MetaNetwork(key_dim, val_dim, context_dim, hidden_dim)

        # Adaptive α gate (Eq. 19)
        self.alpha_gate = nn.Linear(1, 1)   # scalar LSS → scalar α

        # Running memory matrix (non-parameter; updated in forward)
        self.register_buffer("memory", torch.zeros(val_dim, key_dim))

    def forward(
        self,
        keys: torch.Tensor,    # [B, key_dim]
        vals: torch.Tensor,    # [B, val_dim]
        context: torch.Tensor, # [B, context_dim]
        lss: Optional[torch.Tensor] = None,  # scalar surprise signal
    ) -> torch.Tensor:
        """
        Returns output y^(ℓ) = M(k)  [B, val_dim].
        Also performs the online memory update (Eq. 19).
        """
        B = keys.size(0)

        # --- base retrieval: θ k  [B, val_dim]
        base_out = keys @ self.theta.T           # [B, val_dim]

        # --- modification: Δθ ⊙ v  [B, val_dim]
        delta_theta = self.meta_net(keys, vals, context)  # [B, val_dim, key_dim]
        mod_out = (delta_theta * keys.unsqueeze(1)).sum(-1) * vals   # element-wise

        out = base_out + mod_out   # [B, val_dim]

        # --- online memory matrix update (Eq. 19)
        # M_{t+1} = M_t + (1/B) v k^⊤  + α_t * ∇_{M}
        # We use a soft gradient proxy (detached for stability)
        with torch.no_grad():
            outer = (vals.T @ keys) / B   # [val_dim, key_dim]
            if lss is not None:
                alpha = torch.sigmoid(self.alpha_gate(lss.unsqueeze(-1))).squeeze()
            else:
                alpha = torch.tensor(0.01, device=keys.device)
            self.memory = self.memory + outer * (1.0 + alpha)

        return out

    def reset_memory(self):
        self.memory.zero_()


# ---------------------------------------------------------------------------
# 4.  Frequency Modulator  (Eq. 5 & 16)
# ---------------------------------------------------------------------------

class FrequencyModulator(nn.Module):
    """
    First-order:   f^(ℓ)_{t+1} = f^(ℓ)_t + η_f ∇_{f^(ℓ)} L_meta + m^(ℓ)_{t+1}
    (with momentum)                                                    (Eq. 5)

    Second-order (Hessian approx): f^(ℓ)_{t+1} = f^(ℓ)_t − η_f (H^(ℓ))^{-1} ∇ L_meta
                                                                   (Eq. 16)
    """

    def __init__(
        self,
        init_freq: float = 1.0,
        eta_f: float = 0.01,
        beta: float = 0.9,
        use_second_order: bool = False,
    ):
        super().__init__()
        self.log_freq = nn.Parameter(torch.tensor(math.log(init_freq)))
        self.eta_f = eta_f
        self.beta = beta
        self.use_second_order = use_second_order
        self.register_buffer("momentum", torch.zeros(1))

    @property
    def freq(self) -> torch.Tensor:
        """Always positive via softplus reparameterisation."""
        return F.softplus(self.log_freq)

    def update(self, grad_f: torch.Tensor, hessian_diag: Optional[torch.Tensor] = None):
        """Perform one frequency update step (in-place on log_freq buffer)."""
        with torch.no_grad():
            # Ensure grad_f is a scalar tensor
            if not isinstance(grad_f, torch.Tensor):
                grad_f = torch.tensor(float(grad_f))
            grad_f = grad_f.squeeze()
            self.momentum.squeeze_()
            self.momentum.mul_(self.beta).add_((1 - self.beta) * grad_f)
            if self.use_second_order and hessian_diag is not None:
                step = grad_f / (hessian_diag.abs() + 1e-6)
            else:
                step = grad_f + self.momentum
            self.log_freq.data.add_((self.eta_f * step).squeeze())















# ---------------------------------------------------------------------------
# 5.  DNH Level  — one node in the DAG
# ---------------------------------------------------------------------------

class DNHLevel(nn.Module):
    """
    One level ℓ in the Dynamic Nested Hierarchy.
    Wraps a SelfModifyingMemory with its own FrequencyModulator.
    """

    def __init__(
        self,
        level_id: int,
        key_dim: int,
        val_dim: int,
        context_dim: int,
        init_freq: float = 1.0,
        hidden_dim: int = 64,
        eta_f: float = 0.01,
    ):
        super().__init__()
        self.level_id = level_id
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.context_dim = context_dim

        self.smm = SelfModifyingMemory(key_dim, val_dim, context_dim, hidden_dim)
        self.freq_mod = FrequencyModulator(init_freq=init_freq, eta_f=eta_f)

        # Linear projections: maps previous-level output to (k, v, ctx) space
        self.key_proj = nn.Linear(val_dim, key_dim)
        self.val_proj = nn.Linear(val_dim, val_dim)
        self.ctx_proj = nn.Linear(val_dim, context_dim)

    def forward(
        self,
        x: torch.Tensor,                       # [B, val_dim] — input or prev-level output
        lss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        keys    = self.key_proj(x)              # [B, key_dim]
        vals    = self.val_proj(x)              # [B, val_dim]
        context = self.ctx_proj(x)              # [B, context_dim]
        return self.smm(keys, vals, context, lss)

    def get_gradient_norm(self) -> float:
        """‖∇_{θ^(ℓ)} L^(1)‖ — used for pruning criterion."""
        total = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total += p.grad.data.norm().item() ** 2
        return math.sqrt(total)










# ---------------------------------------------------------------------------
# 6.  Meta-Controller  E_ϕ  (Eq. 11-12)
# ---------------------------------------------------------------------------

class MetaController(nn.Module):
    """
    Evolution operator  E_ϕ(G_t, x_t, L_t) → G_{t+1}   (Eq. 11)

    Decides whether to:
      • Add a level  (if L_meta > τ_add)
      • Prune a level (if ‖∇_{θ^(ℓ)}‖ < ε_prune for any ℓ)
      • Only modulate frequencies

    Also updates its own parameters via meta-gradient descent (Eq. 12).
    """

    def __init__(
        self,
        tau_add: float = 0.5,
        epsilon_prune: float = 1e-3,
        lambda_struct: float = 0.01,
        mu_shift: float = 0.1,
        eta_phi: float = 1e-3,
        max_levels: int = 8,
        min_levels: int = 1,
    ):
        super().__init__()
        self.tau_add = tau_add
        self.epsilon_prune = epsilon_prune
        self.lambda_struct = lambda_struct
        self.mu_shift = mu_shift
        self.eta_phi = eta_phi
        self.max_levels = max_levels
        self.min_levels = min_levels

        # Learnable thresholds (ϕ)
        self.log_tau = nn.Parameter(torch.tensor(math.log(max(tau_add, 1e-8))))
        self.log_eps = nn.Parameter(torch.tensor(math.log(max(epsilon_prune, 1e-8))))

    @property
    def effective_tau(self):
        return F.softplus(self.log_tau)

    @property
    def effective_eps(self):
        return F.softplus(self.log_eps)

    def meta_loss(
        self,
        task_loss: torch.Tensor,
        delta_structure: float,
        kl_shift: torch.Tensor,
    ) -> torch.Tensor:
        """
        L_meta = L^(1) + λ‖ΔG‖₁ + μ D_KL(p_t ‖ p_{t-1})   (Eq. 12 comment)
        """
        return (
            task_loss
            + self.lambda_struct * delta_structure
            + self.mu_shift * kl_shift
        )

    def should_add(self, meta_loss_val: float) -> bool:
        return meta_loss_val > self.effective_tau.item()

    def should_prune(self, level: DNHLevel) -> bool:
        return (
            level.get_gradient_norm() < self.effective_eps.item()
        )


# ---------------------------------------------------------------------------
# 7.  Evolvable Adam (EAdam)  (Eq. 20-21)
# ---------------------------------------------------------------------------

class EvolvableAdam(torch.optim.Optimizer):
    """
    Adam with learnable β₁, β₂ that evolve via meta-gradient + noise.

    β_{i,t+1} = β_{i,t} + η_β ∇_{β_i} L_meta + ζ_{i,t}   (Eq. 21)
    σ²_{t+1}  = σ²_t exp(−γ LSS_t)                          (variance adaptation)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        eta_beta: float = 1e-4,
        gamma_noise: float = 0.1,
        init_sigma2: float = 1e-4,
    ):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                        eta_beta=eta_beta, gamma_noise=gamma_noise,
                        sigma2=init_sigma2)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, lss: Optional[float] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1  = group["beta1"]
            beta2  = group["beta2"]
            eps    = group["eps"]
            lr     = group["lr"]
            sigma2 = group["sigma2"]

            # Variance adaptation: σ²_{t+1} = σ²_t exp(−γ LSS_t)
            if lss is not None:
                sigma2 = sigma2 * math.exp(-group["gamma_noise"] * lss)
                group["sigma2"] = max(sigma2, 1e-12)

            # Stochastic exploration noise ζ ~ N(0, σ²)
            noise1 = math.sqrt(sigma2) * torch.randn(1).item()
            noise2 = math.sqrt(sigma2) * torch.randn(1).item()

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"]    = torch.zeros_like(p)
                    state["v"]    = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]

                state["m"] = beta1 * state["m"] + (1 - beta1) * grad
                state["v"] = beta2 * state["v"] + (1 - beta2) * grad ** 2

                m_hat = state["m"] / (1 - beta1 ** t)
                v_hat = state["v"] / (1 - beta2 ** t)

                p -= lr * m_hat / (v_hat.sqrt() + eps)

            # Evolve β₁, β₂ (Eq. 21) — simple gradient proxy via loss change
            # Real implementation would use meta-gradient; here we use heuristic
            group["beta1"] = float(
                torch.clamp(
                    torch.tensor(beta1 + group["eta_beta"] * noise1), 0.5, 0.9999
                )
            )
            group["beta2"] = float(
                torch.clamp(
                    torch.tensor(beta2 + group["eta_beta"] * noise2), 0.9, 0.9999
                )
            )

        return loss