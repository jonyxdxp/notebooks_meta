#!/usr/bin/env python3
"""
Stage 2.5 v2: BJEPA Prior Training — Learned Goal Prior
η = reference turn text → frozen S1 encoder → (μ_prior, σ_prior)
Faithful to BJEPA paper: reuse frozen target encoder as prior encoder.

Trainable components:
  - dynamics_head: S2_output → (μ_dyn, σ_dyn)
  - sigma_goal:    learnable scalar — prior sharpness

S1 and S2 stay completely frozen.
"""

import sys
import os
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = '/content/notebooks_meta'
S2       = f'{ROOT}/v5/s2'
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = Path('/content/drive/MyDrive/metanet/v5/prior_v2')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, ROOT)

from v5.s1.cog_arch.encoder import Encoder
from v5.s2.cog_arch.dm import DM
from v5.s2.config import CFG

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_dataset    = _load_module('data.dataset',    f'{S2}/data/dataset.py')
_dataloader = _load_module('data.dataloader', f'{S2}/data/dataloader.py')

get_stage2_dataloaders = _dataloader.get_stage2_dataloaders
tokenizer              = _dataset.tokenizer

print(f'Device: {DEVICE}')

# ── Helpers ───────────────────────────────────────────────────────────────────

def mean_pool(hidden, mask):
    mask_f = mask.unsqueeze(-1).float()
    return (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)

# ── Architecture ──────────────────────────────────────────────────────────────

class DynamicsHead(nn.Module):
    """
    Maps S2 output → (μ_dyn, σ_dyn) in Z_T space.
    Makes S2 probabilistic without touching S2 weights.
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.mu_head     = nn.Linear(hidden_dim, hidden_dim)
        self.logvar_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z_pred):
        mu     = self.mu_head(z_pred)
        logvar = self.logvar_head(z_pred).clamp(-10, 2)
        return mu, logvar


class LearnedGoalPrior(nn.Module):
    """
    BJEPA Learned Goal Prior — faithful to paper.

    prior_encoder = frozen S1 target encoder (reused as f_θ')
    μ_prior       = S1_encoder(η)   where η = reference turn text
    σ_prior       = exp(log_sigma_goal)  — single learnable scalar

    At training:   η = turn_{t+1}  →  μ_prior = Z_T_real exactly
    At inference:  η = any reference turn text provided by user
    """
    def __init__(self, z_dim=256, init_log_sigma=-1.0):
        super().__init__()
        # Single learnable scalar — how sharp the prior is
        # init_log_sigma=-1.0 → sigma≈0.37 initially (fairly sharp)
        self.log_sigma_goal = nn.Parameter(torch.tensor(init_log_sigma))
        self.z_dim = z_dim

    def get_sigma(self):
        return F.softplus(self.log_sigma_goal) + 1e-4

    def forward(self, z_eta):
        """
        z_eta : (B, D) — S1 encoding of reference turn η
                         at training this is Z_T_real
                         at inference this is S1_encoder(reference_text)

        Returns μ_prior, σ_prior : (B, D)
        """
        sigma = self.get_sigma()
        mu    = z_eta                                         # (B, D)
        sig   = sigma.expand(z_eta.size(0), self.z_dim)     # (B, D)
        return mu, sig

    def product_of_experts(self, mu_dyn, logvar_dyn, z_eta):
        """
        Hard fusion: PoE(dynamics, goal_prior)

        dynamics: (μ_dyn, exp(logvar_dyn))
        prior:    (z_eta, σ_goal)
        """
        mu_prior, sig_prior = self.forward(z_eta)

        prec_dyn   = logvar_dyn.exp().reciprocal()
        prec_prior = sig_prior.pow(2).reciprocal()
        prec_post  = prec_dyn + prec_prior

        mu_post = (prec_dyn * mu_dyn + prec_prior * mu_prior) / prec_post
        return mu_post

# ── Loss ──────────────────────────────────────────────────────────────────────

def bjepa_goal_loss(mu_dyn, logvar_dyn, mu_prior, sig_prior,
                    Z_T_real, gamma=0.1, beta=0.01):
    """
    BJEPA training loss with goal prior.

    1. NLL: dynamics distribution should explain Z_T_real
    2. KL(dynamics || goal_prior): soft fusion — dynamics stays near prior
    3. KL(goal_prior || N(0,I)): keeps prior from drifting too far

    Note: since μ_prior = Z_T_real at training,
    term 1 measures how well dynamics predicts the target
    term 2 measures how close dynamics is to the oracle
    """
    # 1. Dynamics NLL
    dyn_var = logvar_dyn.exp()
    nll = 0.5 * (
        logvar_dyn + (Z_T_real - mu_dyn).pow(2) / dyn_var
    ).sum(dim=-1).mean()

    # 2. KL(dynamics || goal_prior)
    # prior is N(Z_T_real, σ_goal²I) at training
    logvar_prior = (sig_prior.pow(2) + 1e-8).log()
    var_rat      = (logvar_dyn - logvar_prior).exp()
    kl_goal      = 0.5 * (
        var_rat
        + (mu_prior - mu_dyn).pow(2) / sig_prior.pow(2)
        - 1
        - (logvar_dyn - logvar_prior)
    ).sum(dim=-1).mean()

    # 3. KL(goal_prior || N(0,I)) — prior anchoring
    # μ_prior = Z_T_real at training, so this is KL(N(Z_T,σ²) || N(0,I))
    kl_normal = 0.5 * (
        sig_prior.pow(2) + mu_prior.pow(2)
        - 1 - logvar_prior
    ).sum(dim=-1).mean()

    loss = nll + gamma * kl_goal + beta * kl_normal
    return loss, nll, kl_goal, kl_normal

# ── Extract representations ───────────────────────────────────────────────────

def extract_representations(s1_encoder, s2_predictor, loader, device):
    """
    Extract:
      Z_C_pred  = mean_pool(S2(ctx))         — dynamics input
      Z_T_real  = mean_pool(S1(turn_{t+1}))  — oracle target AND η at training
    """
    all_z_C_pred, all_z_T = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Extracting'):
            ctx_ids  = batch['input_ids_a'].to(device)
            ctx_mask = batch['attention_mask_a'].to(device)
            tgt_ids  = batch['input_ids_b'].to(device)
            tgt_mask = batch['attention_mask_b'].to(device)

            # Z_T_real — oracle AND prior input at training
            tgt_h = s1_encoder(tgt_ids, attention_mask=tgt_mask)
            if isinstance(tgt_h, tuple): tgt_h = tgt_h[0]
            z_T   = mean_pool(tgt_h, tgt_mask)

            # Z_C_pred — S2 dynamics prediction
            ctx_h    = s1_encoder(ctx_ids, attention_mask=ctx_mask)
            if isinstance(ctx_h, tuple): ctx_h = ctx_h[0]
            z_pred   = s2_predictor(ctx_h, ctx_h)
            z_C_pred = mean_pool(z_pred, ctx_mask)

            all_z_C_pred.append(z_C_pred.cpu())
            all_z_T.append(z_T.cpu())

    return torch.cat(all_z_C_pred), torch.cat(all_z_T)

# ── Load frozen models ────────────────────────────────────────────────────────

D = CFG.model.hidden_size   # 256

s1_encoder = Encoder(
    vocab_size  = CFG.model.vocab_size,
    hidden_size = D,
    num_heads   = CFG.model.num_heads,
    num_layers  = CFG.model.num_layers,
    max_seq_len = CFG.model.max_seq_len,
).to(DEVICE)
s1_ckpt = torch.load(CFG.training.s1_ckpt, map_location=DEVICE, weights_only=False)
s1_encoder.load_state_dict(s1_ckpt['context_encoder'])
s1_encoder.eval()
for p in s1_encoder.parameters(): p.requires_grad = False
print('S1 encoder loaded and frozen.')

s2_predictor = DM(
    num_frames  = CFG.model.max_seq_len,
    depth       = CFG.model.pred_num_layers,
    heads       = CFG.model.pred_num_heads,
    mlp_dim     = CFG.model.pred_hidden_size * 4,
    input_dim   = CFG.model.dstc,
    hidden_dim  = CFG.model.pred_hidden_size,
    output_dim  = CFG.model.dstc,
    dim_head    = 64, dropout=0.0, emb_dropout=0.0,
).to(DEVICE)
s2_ckpt = torch.load(
    Path(CFG.logging.exp_dir) / 'best.pt',
    map_location=DEVICE, weights_only=False
)
s2_predictor.load_state_dict(s2_ckpt['predictor'])
s2_predictor.eval()
for p in s2_predictor.parameters(): p.requires_grad = False
print(f'S2 loaded and frozen (val_loss={s2_ckpt["val_loss"]:.4f})')

# ── Dataloaders ───────────────────────────────────────────────────────────────

train_loader, val_loader = get_stage2_dataloaders(cfg_obj=CFG, tokenizer=tokenizer)
print(f'Train: {len(train_loader)} | Val: {len(val_loader)}')

# ── Extract and cache ─────────────────────────────────────────────────────────

cache_train = SAVE_DIR / 'reps_train.pt'
cache_val   = SAVE_DIR / 'reps_val.pt'

if cache_train.exists():
    print('Loading cached representations...')
    tr = torch.load(cache_train, weights_only=False)
    vl = torch.load(cache_val,   weights_only=False)
    z_C_pred_train, z_T_train = tr['z_C_pred'], tr['z_T']
    z_C_pred_val,   z_T_val   = vl['z_C_pred'], vl['z_T']
else:
    print('Extracting representations...')
    z_C_pred_train, z_T_train = extract_representations(
        s1_encoder, s2_predictor, train_loader, DEVICE)
    z_C_pred_val, z_T_val = extract_representations(
        s1_encoder, s2_predictor, val_loader, DEVICE)
    torch.save({'z_C_pred': z_C_pred_train, 'z_T': z_T_train}, cache_train)
    torch.save({'z_C_pred': z_C_pred_val,   'z_T': z_T_val},   cache_val)
    print('Saved.')

print(f'Train: {z_T_train.shape} | Val: {z_T_val.shape}')

train_ds = TensorDataset(z_C_pred_train, z_T_train)
val_ds   = TensorDataset(z_C_pred_val,   z_T_val)

enc_train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
enc_val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)

# ── Models ────────────────────────────────────────────────────────────────────

dynamics_head = DynamicsHead(hidden_dim=D).to(DEVICE)
goal_prior    = LearnedGoalPrior(z_dim=D, init_log_sigma=-1.0).to(DEVICE)

print(f'DynamicsHead params : {sum(p.numel() for p in dynamics_head.parameters()):,}')
print(f'GoalPrior params    : {sum(p.numel() for p in goal_prior.parameters()):,}')
print(f'  (just one scalar: log_sigma_goal)')

# ── Optimizer ─────────────────────────────────────────────────────────────────

trainable = list(dynamics_head.parameters()) + list(goal_prior.parameters())
optimizer  = AdamW(trainable, lr=1e-3, weight_decay=0.05)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=30, eta_min=1e-4)

# ── Training loop ─────────────────────────────────────────────────────────────

history = {
    'train_loss': [], 'train_nll': [], 'train_kl_goal': [], 'train_kl_normal': [],
    'val_loss':   [], 'sigma_goal': [],
}
best_val_loss = float('inf')
N_EPOCHS      = 30

print(f'\n{"="*60}')
print(f'  BJEPA Goal Prior — {N_EPOCHS} epochs   device={DEVICE}')
print(f'{"="*60}\n')

for epoch in range(1, N_EPOCHS + 1):

    # Train
    dynamics_head.train(); goal_prior.train()
    t_loss = t_nll = t_kl_g = t_kl_n = 0.0; n = 0

    for (z_C_pred, z_T) in tqdm(enc_train_loader, desc=f'Epoch {epoch:02d}', leave=False):
        z_C_pred = z_C_pred.to(DEVICE)
        z_T      = z_T.to(DEVICE)

        # Dynamics distribution
        mu_dyn, logvar_dyn = dynamics_head(z_C_pred)

        # Goal prior: μ_prior = Z_T_real (η = turn_{t+1} at training)
        mu_prior, sig_prior = goal_prior(z_T)

        loss, nll, kl_g, kl_n = bjepa_goal_loss(
            mu_dyn, logvar_dyn,
            mu_prior, sig_prior,
            z_T,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        t_loss += loss.item(); t_nll  += nll.item()
        t_kl_g += kl_g.item(); t_kl_n += kl_n.item()
        n += 1

    scheduler.step()

    # Val
    dynamics_head.eval(); goal_prior.eval()
    v_loss = 0.0; m = 0

    with torch.no_grad():
        for (z_C_pred, z_T) in enc_val_loader:
            z_C_pred = z_C_pred.to(DEVICE)
            z_T      = z_T.to(DEVICE)
            mu_dyn, logvar_dyn  = dynamics_head(z_C_pred)
            mu_prior, sig_prior = goal_prior(z_T)
            loss, _, _, _       = bjepa_goal_loss(
                mu_dyn, logvar_dyn, mu_prior, sig_prior, z_T)
            v_loss += loss.item(); m += 1

    sigma = goal_prior.get_sigma().item()

    history['train_loss'].append(t_loss/n)
    history['train_nll'].append(t_nll/n)
    history['train_kl_goal'].append(t_kl_g/n)
    history['train_kl_normal'].append(t_kl_n/n)
    history['val_loss'].append(v_loss/m)
    history['sigma_goal'].append(sigma)

    print(
        f'Epoch {epoch:02d}/{N_EPOCHS}  '
        f'train={t_loss/n:.4f} (nll={t_nll/n:.4f} kl_goal={t_kl_g/n:.4f} kl_n={t_kl_n/n:.4f})  '
        f'val={v_loss/m:.4f}  '
        f'σ_goal={sigma:.4f}  '
        f'lr={optimizer.param_groups[0]["lr"]:.2e}'
    )

    if v_loss/m < best_val_loss:
        best_val_loss = v_loss/m
        torch.save({
            'epoch':         epoch,
            'dynamics_head': dynamics_head.state_dict(),
            'goal_prior':    goal_prior.state_dict(),
            'optimizer':     optimizer.state_dict(),
            'scheduler':     scheduler.state_dict(),
            'val_loss':      best_val_loss,
            'sigma_goal':    sigma,
            'hidden_dim':    D,
        }, SAVE_DIR / 'best.pt')
        print(f'  ✓ saved → {SAVE_DIR}/best.pt')

# ── Inference demo ────────────────────────────────────────────────────────────

print('\n── Inference demo ──')
dynamics_head.eval(); goal_prior.eval()

with torch.no_grad():
    # Take 4 val samples
    z_C = z_C_pred_val[:4].to(DEVICE)
    z_T = z_T_val[:4].to(DEVICE)

    mu_dyn, logvar_dyn = dynamics_head(z_C)

    # PoE with oracle η (training mode)
    z_poe_oracle = goal_prior.product_of_experts(mu_dyn, logvar_dyn, z_T)

    # PoE with dynamics only (no prior — baseline)
    z_dynamics_only = mu_dyn

    # Cosine similarities
    z_T_norm = F.normalize(z_T, dim=-1)
    sim_poe  = F.cosine_similarity(
        F.normalize(z_poe_oracle, dim=-1), z_T_norm).mean().item()
    sim_dyn  = F.cosine_similarity(
        F.normalize(z_dynamics_only, dim=-1), z_T_norm).mean().item()

    print(f'  σ_goal learned         : {goal_prior.get_sigma().item():.4f}')
    print(f'  PoE sim to Z_T_real    : {sim_poe:.4f}')
    print(f'  Dynamics-only sim      : {sim_dyn:.4f}')
    print(f'  PoE improvement        : {sim_poe - sim_dyn:+.4f}')

# ── Plot ──────────────────────────────────────────────────────────────────────

epochs_r = range(1, N_EPOCHS + 1)
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

axes[0].plot(epochs_r, history['train_loss'], 'b-', label='train')
axes[0].plot(epochs_r, history['val_loss'],   'r-', label='val')
axes[0].set_title('Total Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_r, history['train_nll'], 'g-')
axes[1].set_title('NLL (dynamics fit)'); axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs_r, history['train_kl_goal'], 'c-')
axes[2].set_title('KL(dynamics || goal_prior)'); axes[2].grid(True, alpha=0.3)

axes[3].plot(epochs_r, history['sigma_goal'], 'm-')
axes[3].set_title('σ_goal (prior sharpness)'); axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(SAVE_DIR / 'training_curves_goal_prior.png', dpi=150)
plt.close(fig)
print(f'\nPlot saved → {SAVE_DIR}/training_curves_goal_prior.png')
print(f'Best val_loss : {best_val_loss:.4f}')
print(f'Final σ_goal  : {goal_prior.get_sigma().item():.4f}')