# v5/s2/prior_encoder.py
"""
Prior encoder for the low level agent.
Unconditional: learns the marginal distribution p(Z_T)
over S1 target representations.

At inference: sample z_ref ~ N(μ, σ) — no input needed.
Can later be conditioned on high level state Z_C_H.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PriorEncoder(nn.Module):
    """
    Unconditional prior over Z_T space.
    Learns p(Z_T) — the marginal distribution of
    valid next-turn representations.

    Input  : nothing (unconditional) or Z_C_H (future extension)
    Output : (μ, σ) in Z_T space — sample z_ref from this
    """

    def __init__(
        self,
        z_dim      : int = 256,
        hidden_dim : int = 256,
        input_dim  : int = None,   # None = unconditional
    ):
        super().__init__()
        self.z_dim       = z_dim
        self.conditional = input_dim is not None

        if self.conditional:
            # Conditioned on some input (e.g. Z_C_H later)
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
        else:
            # Unconditional: learned parameters only
            # μ and log_σ are free parameters — no input needed
            self.mu       = nn.Parameter(torch.zeros(z_dim))
            self.log_sigma = nn.Parameter(torch.zeros(z_dim))

        if self.conditional:
            self.mu_head    = nn.Linear(hidden_dim, z_dim)
            self.sigma_head = nn.Linear(hidden_dim, z_dim)

    def forward(self, x=None):
        """
        x : (B, input_dim) if conditional, None if unconditional
        returns mu, sigma : (B, z_dim) or (z_dim,)
        """
        if self.conditional:
            assert x is not None
            h     = self.net(x)
            mu    = self.mu_head(h)
            sigma = F.softplus(self.sigma_head(h)) + 1e-4
        else:
            mu    = self.mu
            sigma = F.softplus(self.log_sigma) + 1e-4
        return mu, sigma

    def sample(self, n=1, x=None, device='cuda'):
        """
        Sample z_ref without any target input.
        n    : number of samples
        x    : conditioning input if conditional
        """
        mu, sigma = self.forward(x)

        if not self.conditional:
            # Expand to batch
            mu    = mu.unsqueeze(0).expand(n, -1).to(device)
            sigma = sigma.unsqueeze(0).expand(n, -1).to(device)

        z_ref = mu + sigma * torch.randn_like(mu)
        return z_ref   # (n, z_dim)


def prior_loss(mu, sigma, Z_T_real, beta=1.0):
    """
    Train prior to cover the distribution of real Z_T values.

    NLL  : sampled z_ref should be close to real targets
    KL   : distribution stays close to N(0,I) for clean sampling
    """
    # NLL under Gaussian: how well does N(μ,σ) cover Z_T_real
    nll = 0.5 * (
        ((Z_T_real - mu) / sigma).pow(2)
        + sigma.pow(2).log()
        + torch.log(torch.tensor(2 * 3.14159))
    ).sum(dim=-1).mean()

    # KL[N(μ,σ) || N(0,I)]
    kl = -0.5 * (
        1 + sigma.pow(2).log()
        - mu.pow(2)
        - sigma.pow(2)
    ).sum(dim=-1).mean()

    return nll + beta * kl, nll, kl