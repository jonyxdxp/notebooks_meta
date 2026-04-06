

# Prediction Model (outputs the Value and Policy heads)












# from https://github.com/werner-duvaud/muzero-general/blob/master/models.py







class PredictionNetwork(torch.nn.Module):
    def __init__(
        self,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_value,
        reduced_channels_policy,
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
        block_output_size_value,
        block_output_size_policy,
    ):
        super().__init__()
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        self.conv1x1_value = torch.nn.Conv2d(num_channels, reduced_channels_value, 1)
        self.conv1x1_policy = torch.nn.Conv2d(num_channels, reduced_channels_policy, 1)
        self.block_output_size_value = block_output_size_value
        self.block_output_size_policy = block_output_size_policy
        self.fc_value = mlp(
            self.block_output_size_value, fc_value_layers, full_support_size
        )
        self.fc_policy = mlp(
            self.block_output_size_policy,
            fc_policy_layers,
            action_space_size,
        )

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        value = self.conv1x1_value(x)
        policy = self.conv1x1_policy(x)
        value = value.view(-1, self.block_output_size_value)
        policy = policy.view(-1, self.block_output_size_policy)
        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value












# ----------------------------------------------------------------










"""
EBT-Policy: Energy Unlocks Emergent Physical Reasoning Capabilities
====================================================================
Implementation of the EBT-Policy paper (arXiv:2510.27545).

EBT-Policy is an Energy-Based Transformer policy for visuomotor control.
Instead of predicting actions directly, it learns an *energy function*
E_θ(o, a) that assigns low energy to (observation, action) pairs that are
compatible with expert demonstrations.  Actions are generated at inference
time by minimising this energy through Langevin-MCMC dynamics.

Key components implemented here
--------------------------------
1.  EBTPolicy        – Transformer that outputs a scalar energy given
                       (observation tokens, action trajectory).
2.  EBTPolicyLoss    – Regularised training objective (smooth-L1 on the
                       optimised trajectory vs ground truth, plus gradient-
                       penalty regularisation on the energy landscape).
3.  LangevinSampler  – MCMC sampler with energy-scaled step sizes, Nesterov
                       momentum, pre-sample normalisation, and adaptive
                       early stopping.
4.  EBTPolicyTrainer – Thin wrapper that ties the model, loss and sampler
                       together for a standard train/evaluate loop.
5.  Toy demo         – 2-D "push-T" style task so the whole pipeline can be
                       exercised without a real robot.

Usage
-----
    python ebt_policy.py           # run toy demo
    python ebt_policy.py --help    # see CLI flags
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor






# ─────────────────────────────────────────────────────────────────────────────
# 0.  Configuration dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EBTPolicyConfig:
    """All hyper-parameters for EBT-Policy-S (simulation variant)."""

    # Observation / action dims
    obs_dim: int = 16          # flattened observation size
    action_dim: int = 2        # per-step action size
    action_horizon: int = 8    # number of future steps in a trajectory
    obs_horizon: int = 2       # number of past observations to condition on

    # Transformer backbone
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1

    # ── Training
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    n_epochs: int = 50
    grad_clip: float = 1.0

    # ── Energy landscape regularisation
    grad_penalty_weight: float = 0.01   # λ for ||∇_a E||² regulariser
    energy_reg_weight: float = 0.001    # small L2 on raw energy values

    # ── Langevin MCMC (inference / training unrolling)
    ld_scale_min: float = 0.002         # σ_min  (Table 1 in paper)
    ld_scale_max: float = 0.2           # σ_max
    n_mcmc_train: int = 6               # base MCMC steps during training
    n_mcmc_rand_extra: int = 3          # randomised additional steps (train)
    n_mcmc_infer_max: int = 20          # max MCMC steps at inference
    mcmc_grad_tol: float = 1e-3         # early-stop tolerance on ||∇_a E||

    # Nesterov momentum for Langevin
    nesterov_momentum: float = 0.9

    # ── Sampling
    n_sample_candidates: int = 512      # number of random initialisations
    action_noise_std: float = 1.0       # initial action noise scale








# ─────────────────────────────────────────────────────────────────────────────
# 1.  Model: EBT-Policy Transformer
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal PE, length-agnostic."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d)

    def forward(self, x: Tensor) -> Tensor:           # x: (B, T, d)
        return x + self.pe[:, : x.size(1)]










class EBTPolicy(nn.Module):
    """
    Energy-Based Transformer Policy.

    Takes a sequence of observations (o_{t-H+1}, …, o_t) and a candidate
    action trajectory (a_t, …, a_{t+K-1}) and returns a **scalar energy**
    E ∈ ℝ for each element in the batch.

    Low energy  ⟺  high compatibility between observation and action.

    Architecture
    ────────────
    • Observation tokens  : linear projection → d_model tokens
    • Action tokens       : linear projection → d_model tokens
    • All tokens are concatenated and processed by a standard Transformer
      encoder (bidirectional attention – no causal mask needed here).
    • A learnable [ENERGY] readout token is prepended; its final hidden state
      is projected to a scalar energy value via a 2-layer MLP.
    """

    def __init__(self, cfg: EBTPolicyConfig):
        super().__init__()
        self.cfg = cfg

        # Token projections
        self.obs_proj = nn.Linear(cfg.obs_dim, cfg.d_model)
        self.act_proj = nn.Linear(cfg.action_dim, cfg.d_model)

        # Learnable [ENERGY] CLS-style token
        self.energy_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

        # Energy readout MLP
        self.energy_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
        )

        self._init_weights()

    # ── weight init ──────────────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, obs: Tensor, actions: Tensor) -> Tensor:
        """
        Parameters
        ----------
        obs     : (B, obs_horizon, obs_dim)
        actions : (B, action_horizon, action_dim)

        Returns
        -------
        energy  : (B,) scalar energy per sample
        """
        B = obs.size(0)

        # Project to d_model
        o_tok = self.obs_proj(obs)          # (B, H_o, d)
        a_tok = self.act_proj(actions)       # (B, H_a, d)

        # Prepend learnable energy readout token
        cls = self.energy_token.expand(B, -1, -1)   # (B, 1, d)

        # Concatenate: [E | obs_tokens | action_tokens]
        tokens = torch.cat([cls, o_tok, a_tok], dim=1)  # (B, 1+H_o+H_a, d)

        tokens = self.pos_enc(tokens)

        # Transformer encoder (bidirectional)
        hidden = self.transformer(tokens)   # (B, 1+H_o+H_a, d)

        # Energy from the [ENERGY] token
        energy = self.energy_head(hidden[:, 0, :]).squeeze(-1)  # (B,)
        return energy



















# ─────────────────────────────────────────────────────────────────────────────
# 2.  Langevin / MCMC sampler
# ─────────────────────────────────────────────────────────────────────────────

class LangevinSampler:
    """
    Scaled Langevin Dynamics sampler with:
    • Energy-proportional step sizes  (σ ∝ E)
    • Pre-sample normalisation
    • Nesterov-accelerated gradient updates
    • Adaptive early stopping when ||∇_a E||₂ < tol

    Used both during training (to generate the "optimised" trajectory that
    the loss compares against ground truth) and at inference time.
    """

    def __init__(self, cfg: EBTPolicyConfig):
        self.cfg = cfg

    # ── internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _clamp_actions(a: Tensor, lo: float = -1.0, hi: float = 1.0) -> Tensor:
        return a.clamp(lo, hi)

    def _step_size(self, energy: Tensor) -> Tensor:
        """
        Energy-scaled step size σ(E) = clip(|E|, σ_min, σ_max).
        Shape: (B,) → broadcast to action shape.
        """
        cfg = self.cfg
        sigma = energy.abs().clamp(cfg.ld_scale_min, cfg.ld_scale_max)
        return sigma

    # ── main sampling routine ─────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        model: EBTPolicy,
        obs: Tensor,
        n_steps: int,
        *,
        init_actions: Optional[Tensor] = None,
        add_noise: bool = True,
        early_stop: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Run Langevin MCMC to find action trajectories with low energy.

        Parameters
        ----------
        model       : EBTPolicy
        obs         : (B, obs_horizon, obs_dim)
        n_steps     : number of MCMC steps
        init_actions: (B, action_horizon, action_dim) — if None, sample random
        add_noise   : inject Langevin noise (False for pure gradient descent)
        early_stop  : stop when gradient norm < cfg.mcmc_grad_tol

        Returns
        -------
        actions : (B, action_horizon, action_dim) — low-energy actions
        energies: (B,) — final energies
        """
        cfg = self.cfg
        B = obs.size(0)
        device = obs.device

        # Pre-sample normalisation: normalise obs once
        obs_norm = F.layer_norm(obs, obs.shape[1:])

        # Initialise action candidates
        if init_actions is None:
            a = torch.randn(
                B, cfg.action_horizon, cfg.action_dim, device=device
            ) * cfg.action_noise_std
        else:
            a = init_actions.clone()

        a = self._clamp_actions(a)
        velocity = torch.zeros_like(a)   # Nesterov momentum buffer

        for step in range(n_steps):
            # Nesterov look-ahead
            a_look = a + cfg.nesterov_momentum * velocity

            # Enable grad temporarily for the gradient computation
            a_look = a_look.detach().requires_grad_(True)

            with torch.enable_grad():
                energy = model(obs_norm, a_look)        # (B,)
                grad = torch.autograd.grad(energy.sum(), a_look)[0]  # (B,H_a,d_a)

            # Energy-scaled step size (broadcast over action dims)
            sigma = self._step_size(energy.detach())    # (B,)
            sigma = sigma[:, None, None]                # (B,1,1)

            # Nesterov update
            new_velocity = cfg.nesterov_momentum * velocity - sigma * grad.detach()
            a = a + new_velocity
            velocity = new_velocity

            # Langevin noise injection
            if add_noise:
                noise_scale = (2 * sigma) ** 0.5
                a = a + noise_scale * torch.randn_like(a)

            a = self._clamp_actions(a)

            # Adaptive early stopping
            if early_stop:
                grad_norm = grad.detach().norm(dim=(1, 2))  # (B,)
                if grad_norm.max().item() < cfg.mcmc_grad_tol:
                    break

        # Final energy (no grad)
        with torch.no_grad():
            final_energy = model(obs_norm, a)

        return a.detach(), final_energy.detach()

    # ── best-of-N inference ────────────────────────────────────────────────────

    @torch.no_grad()
    def infer(
        self,
        model: EBTPolicy,
        obs: Tensor,
        n_candidates: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Best-of-N inference:
        1. Draw n_candidates random initial action trajectories.
        2. Run Langevin to find each one's energy minimum.
        3. Return the candidate with lowest energy.

        Parameters
        ----------
        obs          : (1, obs_horizon, obs_dim)  single observation
        n_candidates : number of parallel random starts

        Returns
        -------
        best_action  : (1, action_horizon, action_dim)
        best_energy  : scalar
        """
        cfg = self.cfg
        N = n_candidates or cfg.n_sample_candidates
        device = obs.device

        # Tile the single observation across N candidates
        obs_rep = obs.expand(N, -1, -1)   # (N, H_o, obs_dim)

        actions, energies = self.sample(
            model,
            obs_rep,
            n_steps=cfg.n_mcmc_infer_max,
            add_noise=True,
            early_stop=True,
        )

        # Pick lowest-energy candidate
        best_idx = energies.argmin()
        return actions[best_idx : best_idx + 1], energies[best_idx]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Loss function
# ─────────────────────────────────────────────────────────────────────────────

class EBTPolicyLoss(nn.Module):
    """
    Regularised EBT training objective.

    Unlike contrastive EBM losses, EBT-Policy uses:

    L = SmoothL1(a_opt, a_gt)                    ← trajectory fidelity
      + λ_gp  · E[||∇_a E(o, a)||²]              ← gradient-penalty on landscape
      + λ_reg · E[E(o, a_gt)²]                   ← energy magnitude regulariser

    The model is unrolled for N_train MCMC steps on each training batch;
    the loss is accumulated across the unroll and backpropagated through all
    iterations (i.e., through the sampler).
    """

    def __init__(self, cfg: EBTPolicyConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        model: EBTPolicy,
        obs: Tensor,
        gt_actions: Tensor,
    ) -> Tuple[Tensor, dict]:
        """
        Parameters
        ----------
        obs        : (B, obs_horizon, obs_dim)
        gt_actions : (B, action_horizon, action_dim)

        Returns
        -------
        total_loss : scalar
        info       : dict with individual loss components
        """
        cfg = self.cfg
        B = obs.size(0)
        device = obs.device

        # Pre-sample normalisation
        obs_norm = F.layer_norm(obs, obs.shape[1:])

        # ── Randomised MCMC unroll ─────────────────────────────────────────
        n_steps = cfg.n_mcmc_train + random.randint(0, cfg.n_mcmc_rand_extra)

        # Initialise from noise
        a = torch.randn_like(gt_actions) * cfg.action_noise_std
        a = a.clamp(-1.0, 1.0)
        velocity = torch.zeros_like(a)

        # Unroll gradient-tracked Langevin steps
        traj_loss = torch.tensor(0.0, device=device)

        for step in range(n_steps):
            a_look = a + cfg.nesterov_momentum * velocity
            a_look = a_look.requires_grad_(True)

            energy = model(obs_norm, a_look)            # (B,)
            grad = torch.autograd.grad(
                energy.sum(), a_look, create_graph=True
            )[0]

            # Energy-scaled step
            sigma = energy.abs().clamp(cfg.ld_scale_min, cfg.ld_scale_max)[:, None, None]
            new_velocity = cfg.nesterov_momentum * velocity.detach() - sigma * grad
            a = (a.detach() + new_velocity).clamp(-1.0, 1.0)
            velocity = new_velocity.detach()

            # Accumulate trajectory loss (scaled by predicted energy)
            step_loss = F.smooth_l1_loss(a, gt_actions, reduction="none").mean(dim=(1, 2))
            energy_scale = energy.abs().clamp(1e-4, 1.0).detach()
            traj_loss = traj_loss + (step_loss * energy_scale).mean()

        traj_loss = traj_loss / n_steps

        # ── Gradient-penalty regularisation ──────────────────────────────────
        # Penalise ||∇_a E|| at ground-truth actions
        gt_a = gt_actions.detach().requires_grad_(True)
        energy_gt = model(obs_norm, gt_a)
        gp_grad = torch.autograd.grad(energy_gt.sum(), gt_a, create_graph=True)[0]
        grad_penalty = (gp_grad.norm(dim=(1, 2)) ** 2).mean()

        # ── Energy magnitude regularisation ───────────────────────────────────
        energy_reg = (energy_gt ** 2).mean()

        # ── Total ──────────────────────────────────────────────────────────────
        total = (
            traj_loss
            + cfg.grad_penalty_weight * grad_penalty
            + cfg.energy_reg_weight   * energy_reg
        )

        info = {
            "loss_total":     total.item(),
            "loss_traj":      traj_loss.item(),
            "loss_gp":        grad_penalty.item(),
            "loss_energy_reg": energy_reg.item(),
        }
        return total, info


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Trainer
# ─────────────────────────────────────────────────────────────────────────────

class EBTPolicyTrainer:
    """Minimal training loop for EBT-Policy."""

    def __init__(self, cfg: EBTPolicyConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)

        self.model   = EBTPolicy(cfg).to(self.device)
        self.loss_fn = EBTPolicyLoss(cfg)
        self.sampler = LangevinSampler(cfg)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.n_epochs
        )

    # ── one training step ─────────────────────────────────────────────────────

    def train_step(self, obs: Tensor, actions: Tensor) -> dict:
        self.model.train()
        obs     = obs.to(self.device)
        actions = actions.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        loss, info = self.loss_fn(self.model, obs, actions)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        return info

    # ── evaluation: success rate on a batch ──────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        obs_batch: Tensor,
        gt_actions_batch: Tensor,
        tol: float = 0.1,
    ) -> float:
        """Fraction of samples where best-of-N action is within `tol` of GT."""
        self.model.eval()
        obs_batch = obs_batch.to(self.device)

        successes = 0
        for i in range(obs_batch.size(0)):
            obs_i = obs_batch[i : i + 1]   # (1, H_o, d_obs)
            pred_a, _ = self.sampler.infer(self.model, obs_i, n_candidates=32)
            gt_a = gt_actions_batch[i].to(self.device)
            err = (pred_a.squeeze(0) - gt_a).abs().mean().item()
            if err < tol:
                successes += 1

        return successes / obs_batch.size(0)

    # ── full training loop ────────────────────────────────────────────────────

    def fit(
        self,
        obs_data:    Tensor,   # (N, obs_horizon, obs_dim)
        action_data: Tensor,   # (N, action_horizon, action_dim)
        verbose: bool = True,
    ) -> List[dict]:
        cfg = self.cfg
        N = obs_data.size(0)
        history = []

        for epoch in range(cfg.n_epochs):
            # Shuffle
            idx = torch.randperm(N)
            obs_data    = obs_data[idx]
            action_data = action_data[idx]

            epoch_info: dict = {k: 0.0 for k in
                                ["loss_total", "loss_traj", "loss_gp", "loss_energy_reg"]}
            n_batches = 0

            for start in range(0, N, cfg.batch_size):
                end  = min(start + cfg.batch_size, N)
                obs_b = obs_data[start:end]
                act_b = action_data[start:end]
                info  = self.train_step(obs_b, act_b)
                for k in epoch_info:
                    epoch_info[k] += info[k]
                n_batches += 1

            self.scheduler.step()

            for k in epoch_info:
                epoch_info[k] /= n_batches
            epoch_info["epoch"] = epoch + 1
            history.append(epoch_info)

            if verbose and (epoch + 1) % max(1, cfg.n_epochs // 10) == 0:
                print(
                    f"Epoch {epoch+1:3d}/{cfg.n_epochs}  "
                    f"loss={epoch_info['loss_total']:.4f}  "
                    f"traj={epoch_info['loss_traj']:.4f}  "
                    f"gp={epoch_info['loss_gp']:.4f}"
                )

        return history


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Uncertainty-Aware Inference (energy as confidence)
# ─────────────────────────────────────────────────────────────────────────────

class UncertaintyAwareController:
    """
    Wraps EBTPolicyTrainer to implement the paper's emergent-retry behaviour.

    At each time-step:
    1. Run best-of-N Langevin inference to get action + energy.
    2. If energy > `retry_threshold`, consider the action *uncertain* and
       attempt a retry by re-sampling from a wider distribution.
    3. If energy after retry is still high, signal the environment to pause
       (e.g., the robot should hold position).

    This replicates Figure 2 / Figure 6 of the paper where EBT-Policy
    autonomously detects failed contact and retries without explicit supervision.
    """

    def __init__(
        self,
        trainer: EBTPolicyTrainer,
        retry_threshold: float = 0.3,
        max_retries: int = 3,
    ):
        self.trainer = trainer
        self.retry_threshold = retry_threshold
        self.max_retries = max_retries
        self.sampler = trainer.sampler
        self.model   = trainer.model

    def act(self, obs: Tensor) -> Tuple[Tensor, float, int]:
        """
        Parameters
        ----------
        obs : (1, obs_horizon, obs_dim)

        Returns
        -------
        action      : (1, action_horizon, action_dim)
        energy      : scalar energy value
        n_retries   : how many retries were performed
        """
        obs = obs.to(self.trainer.device)
        n_retries = 0

        for attempt in range(self.max_retries + 1):
            noise_scale = self.trainer.cfg.action_noise_std * (1.5 ** attempt)
            self.sampler.cfg.action_noise_std = noise_scale

            action, energy = self.sampler.infer(self.model, obs)
            energy_val = energy.item()

            if energy_val <= self.retry_threshold or attempt == self.max_retries:
                break

            n_retries += 1

        # Restore noise scale
        self.sampler.cfg.action_noise_std = self.trainer.cfg.action_noise_std

        return action, energy_val, n_retries


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Toy task: 2D point-mass with multi-modal demonstrations
# ─────────────────────────────────────────────────────────────────────────────

class ToyPushTask:
    """
    Minimal 2-D task to verify the pipeline end-to-end.

    The "robot" is a point mass at position (x, y) ∈ [-1,1]².
    Goal: push a "block" from a random start to a fixed target.

    Two qualitatively different pushing strategies are available
    (clockwise vs counter-clockwise approach), creating a *bimodal*
    action distribution — a classic test for implicit policies.
    """

    def __init__(self, obs_horizon: int = 2, action_horizon: int = 8):
        self.obs_horizon    = obs_horizon
        self.action_horizon = action_horizon
        self.obs_dim        = 4   # (robot_x, robot_y, block_x, block_y) × each step
        self.action_dim     = 2

    def _obs_dim_full(self):
        return self.obs_dim  # per-step obs size (stacked outside)

    def generate_demos(
        self, n_demos: int = 2000, seed: int = 42
    ) -> Tuple[Tensor, Tensor]:
        """Generate synthetic bimodal demonstration dataset."""
        rng = np.random.default_rng(seed)
        target = np.array([0.8, 0.8])

        all_obs     = []
        all_actions = []

        for _ in range(n_demos):
            block_start = rng.uniform(-0.5, 0.3, size=2)
            robot_start = rng.uniform(-1.0, 1.0, size=2)

            # Randomly pick one of two approach strategies
            clockwise = rng.random() > 0.5
            sign = 1.0 if clockwise else -1.0

            obs_seq = []
            act_seq = []

            robot = robot_start.copy()
            block = block_start.copy()

            for h in range(self.obs_horizon):
                obs_seq.append(np.concatenate([robot, block]))

            for k in range(self.action_horizon):
                # Simple heuristic: arc-approach to block, then push toward target
                to_block = block - robot
                perp = np.array([-to_block[1], to_block[0]]) * sign * 0.3
                push_dir = (target - block)
                norm = np.linalg.norm(push_dir) + 1e-6
                push_dir /= norm

                delta = to_block * 0.4 + perp * 0.2 + push_dir * 0.3
                delta += rng.normal(0, 0.02, size=2)   # small noise
                delta = np.clip(delta, -1, 1)

                act_seq.append(delta)
                robot = np.clip(robot + delta * 0.1, -1, 1)
                block = np.clip(block + push_dir * 0.05, -1, 1)

            all_obs.append(np.array(obs_seq))      # (H_o, obs_dim)
            all_actions.append(np.array(act_seq))  # (H_a, action_dim)

        obs_tensor = torch.tensor(np.array(all_obs),    dtype=torch.float32)
        act_tensor = torch.tensor(np.array(all_actions), dtype=torch.float32)
        # Normalise actions to [-1, 1]
        act_tensor = act_tensor / (act_tensor.abs().max() + 1e-6)
        return obs_tensor, act_tensor


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Adaptive compute demo
# ─────────────────────────────────────────────────────────────────────────────

def demo_adaptive_compute(trainer: EBTPolicyTrainer, obs_sample: Tensor):
    """
    Show that EBT-Policy can converge with fewer steps on easy inputs
    and more steps on hard ones — the key compute-adaptive property.
    """
    trainer.model.eval()
    obs = obs_sample.to(trainer.device)
    cfg = trainer.cfg

    results = []
    for n_steps in [1, 2, 5, 10, 20]:
        a, energy = trainer.sampler.sample(
            trainer.model, obs, n_steps=n_steps, add_noise=False, early_stop=False
        )
        results.append({"n_steps": n_steps, "mean_energy": energy.mean().item()})

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Main demo
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="EBT-Policy demo")
    p.add_argument("--n_demos",   type=int, default=1000)
    p.add_argument("--n_epochs",  type=int, default=30)
    p.add_argument("--batch_size",type=int, default=64)
    p.add_argument("--d_model",   type=int, default=64)
    p.add_argument("--n_layers",  type=int, default=3)
    p.add_argument("--device",    type=str, default="cpu")
    p.add_argument("--seed",      type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("EBT-Policy: Energy Unlocks Emergent Physical Reasoning")
    print("=" * 60)

    # ── Task ──────────────────────────────────────────────────────────────────
    task = ToyPushTask()

    cfg = EBTPolicyConfig(
        obs_dim        = task.obs_dim,
        action_dim     = task.action_dim,
        obs_horizon    = task.obs_horizon,
        action_horizon = task.action_horizon,
        d_model        = args.d_model,
        n_heads        = 4,
        n_layers       = args.n_layers,
        d_ff           = args.d_model * 4,
        n_epochs       = args.n_epochs,
        batch_size     = args.batch_size,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"\n[1/4] Generating {args.n_demos} bimodal demonstrations …")
    obs_data, act_data = task.generate_demos(n_demos=args.n_demos)
    print(f"      obs shape   : {tuple(obs_data.shape)}")
    print(f"      action shape: {tuple(act_data.shape)}")

    split = int(0.9 * len(obs_data))
    obs_train, obs_val   = obs_data[:split], obs_data[split:]
    act_train, act_val   = act_data[:split], act_data[split:]

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n[2/4] Training EBT-Policy for {args.n_epochs} epochs …")
    trainer = EBTPolicyTrainer(cfg, device=args.device)
    n_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"      Model parameters: {n_params:,}")

    history = trainer.fit(obs_train, act_train, verbose=True)
    final_loss = history[-1]["loss_total"]
    print(f"\n      Final training loss: {final_loss:.4f}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n[3/4] Evaluating action prediction accuracy …")
    success_rate = trainer.evaluate(obs_val[:50], act_val[:50], tol=0.2)
    print(f"      Success rate (tol=0.20): {success_rate*100:.1f}%")

    # ── Adaptive compute demo ─────────────────────────────────────────────────
    print("\n[4/4] Adaptive compute: energy vs. MCMC steps …")
    obs_sample = obs_val[:8]
    compute_results = demo_adaptive_compute(trainer, obs_sample)
    for r in compute_results:
        print(f"      steps={r['n_steps']:2d}  mean_energy={r['mean_energy']:.4f}")

    # ── Uncertainty-aware controller ──────────────────────────────────────────
    print("\n[Bonus] Uncertainty-aware retry controller …")
    controller = UncertaintyAwareController(trainer, retry_threshold=0.15, max_retries=3)
    obs_test = obs_val[:1]
    action, energy, n_retries = controller.act(obs_test)
    print(f"        Final energy : {energy:.4f}")
    print(f"        Retries used : {n_retries}")
    print(f"        Action shape : {tuple(action.shape)}")

    print("\n✓ EBT-Policy demo complete.")
    return trainer, history


if __name__ == "__main__":
    trainer, history = main()