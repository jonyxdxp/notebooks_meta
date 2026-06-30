# v6/prior_encoder.py
"""
BJEPA Prior components for v6.

Two trainable modules sit on top of the frozen S2 predictor:
  DynamicsHead   : DialogueJEPAPredictor output (B,768) → (μ_dyn, logvar_dyn)
  LearnedGoalPrior : (z_eta: B,768) → (μ_prior, σ_prior)
                     + Product of Experts hard fusion for inference

Dimensions are 768 throughout (BERT/DMI encoder, not v5's 256).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsHead(nn.Module):
    """
    Wraps the frozen S2 predictor output to produce a distribution.

    Input  : z_pred  (B, 768) — output of DialogueJEPAPredictor
    Output : mu      (B, 768)
             logvar  (B, 768)   clamped to [-10, 2] for stability
    """
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.mu_head     = nn.Linear(hidden_dim, hidden_dim)
        self.logvar_head = nn.Linear(hidden_dim, hidden_dim)

        nn.init.xavier_uniform_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.xavier_uniform_(self.logvar_head.weight)
        # init logvar bias to 0  → initial σ ≈ 1
        nn.init.zeros_(self.logvar_head.bias)

    def forward(self, z_pred: torch.Tensor):
        mu     = self.mu_head(z_pred)
        logvar = self.logvar_head(z_pred).clamp(-10, 2)
        return mu, logvar


class LearnedGoalPrior(nn.Module):
    """
    BJEPA Learned Goal Prior — v6 version (768-d).

    At training : z_eta = S1_encoder(turn_{t+1}) = Z_T_real
                  μ_prior = z_eta  (oracle)
                  σ_prior = softplus(log_sigma_goal)  — single learned scalar

    At inference: z_eta = S1_encoder(any reference text the user provides)
                  Same formula, different input.

    The single scalar σ_goal controls how tightly the prior
    constrains the dynamics prediction.  Small σ → prior dominates.
    Large σ → dynamics dominates.
    """
    def __init__(self, z_dim: int = 768, init_log_sigma: float = -1.0):
        super().__init__()
        # Scalar [] — matches checkpoint shape from v5 (not zeros(1))
        self.log_sigma_goal = nn.Parameter(torch.tensor(init_log_sigma))
        self.z_dim = z_dim

    def get_sigma(self) -> torch.Tensor:
        """Returns the current prior sharpness σ (scalar)."""
        return F.softplus(self.log_sigma_goal) + 1e-4

    def forward(self, z_eta: torch.Tensor):
        """
        z_eta : (B, 768)
        Returns μ_prior (B, 768), σ_prior (B, 768)
        """
        sigma   = self.get_sigma()
        mu      = z_eta
        sig     = sigma.expand(z_eta.size(0), self.z_dim)
        return mu, sig

    def product_of_experts(self,
                            mu_dyn:    torch.Tensor,
                            logvar_dyn: torch.Tensor,
                            z_eta:     torch.Tensor) -> torch.Tensor:
        """
        Hard fusion at inference.

        dynamics expert : N(μ_dyn,  exp(logvar_dyn))
        goal prior      : N(z_eta,  σ_goal²)

        posterior mean  = (prec_dyn·μ_dyn + prec_prior·μ_prior)
                          / (prec_dyn + prec_prior)

        Returns posterior mean (B, 768) — same shape as μ_dyn,
        passed directly to the S3 decoder.
        """
        mu_prior, sig_prior = self.forward(z_eta)
        prec_dyn   = logvar_dyn.exp().reciprocal()         # (B, 768)
        prec_prior = sig_prior.pow(2).reciprocal()         # (B, 768)
        prec_post  = prec_dyn + prec_prior
        mu_post    = (prec_dyn * mu_dyn + prec_prior * mu_prior) / prec_post
        return mu_post                                      # (B, 768)


def bjepa_goal_loss(
    mu_dyn:    torch.Tensor,   # (B, 768)
    logvar_dyn: torch.Tensor,  # (B, 768)
    mu_prior:  torch.Tensor,   # (B, 768)  = Z_T_real at training
    sig_prior: torch.Tensor,   # (B, 768)
    Z_T_real:  torch.Tensor,   # (B, 768)
    gamma: float = 0.1,
    beta:  float = 0.01,
):
    """
    BJEPA loss with Goal Prior (soft fusion during training).

    Three terms:
      nll      — dynamics distribution explains Z_T_real
      kl_goal  — KL(dynamics ‖ goal_prior)  soft fusion pull
      kl_normal— KL(goal_prior ‖ N(0,I))    keeps prior from drifting

    Returns: loss (scalar), nll, kl_goal, kl_normal
    """
    # 1. NLL of Z_T_real under the dynamics distribution
    dyn_var = logvar_dyn.exp()
    nll = 0.5 * (
        logvar_dyn + (Z_T_real - mu_dyn).pow(2) / dyn_var
    ).sum(dim=-1).mean()

    # 2. KL(dynamics ‖ goal_prior) — analytical for two Gaussians
    logvar_prior = (sig_prior.pow(2) + 1e-8).log()
    var_rat      = (logvar_dyn - logvar_prior).exp()
    kl_goal = 0.5 * (
        var_rat
        + (mu_prior - mu_dyn).pow(2) / sig_prior.pow(2)
        - 1.0
        - (logvar_dyn - logvar_prior)
    ).sum(dim=-1).mean()

    # 3. KL(goal_prior ‖ N(0,I)) — anchors prior; prevents collapse
    kl_normal = 0.5 * (
        sig_prior.pow(2) + mu_prior.pow(2)
        - 1.0 - logvar_prior
    ).sum(dim=-1).mean()

    loss = nll + gamma * kl_goal + beta * kl_normal
    return loss, nll, kl_goal, kl_normal