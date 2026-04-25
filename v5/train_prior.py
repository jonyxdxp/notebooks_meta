#!/usr/bin/env python3
"""
Stage 2.5: BJEPA Prior Training
Learns a static prior (μ_prior, σ_prior) in Z_T space.
Trains two probabilistic heads on top of frozen S2.
S1 and S2 stay completely frozen.
"""

import sys
import os
import copy
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
S1       = f'{ROOT}/v5/s1'
S2       = f'{ROOT}/v5/s2'
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = Path('/content/drive/MyDrive/metanet/v5/prior')
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
    Two linear heads on top of frozen S2 output.
    Makes S2 probabilistic: z_pred → (μ_dyn, logvar_dyn)

    These are the only trainable parameters alongside the prior.
    S2 stays completely frozen.
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.mu_head     = nn.Linear(hidden_dim, hidden_dim)
        self.logvar_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z_pred):
        """
        z_pred : (B, D) — mean-pooled S2 output
        returns μ_dyn, logvar_dyn : (B, D)
        """
        mu     = self.mu_head(z_pred)
        logvar = self.logvar_head(z_pred).clamp(-10, 2)
        return mu, logvar


class StaticPrior(nn.Module):
    """
    Learnable static prior — exactly as in BJEPA paper.
    Just two parameter vectors, no network, no input.

    prior_mu, prior_logvar are optimized during training
    to be a good structural attractor for the dynamics.
    """
    def __init__(self, latent_dim=256):
        super().__init__()
        self.prior_mu     = nn.Parameter(torch.zeros(latent_dim))
        self.prior_logvar = nn.Parameter(torch.zeros(latent_dim))

    def get_prior(self, batch_size, device):
        mu     = self.prior_mu.unsqueeze(0).expand(batch_size, -1)
        logvar = self.prior_logvar.unsqueeze(0).expand(batch_size, -1)
        return mu, logvar

    def product_of_experts(self, mu_dyn, logvar_dyn, batch_size, device):
        """
        Hard fusion at inference: combine dynamics + prior via PoE.
        Returns posterior mean.
        """
        mu_prior, logvar_prior = self.get_prior(batch_size, device)

        prec_dyn   = logvar_dyn.exp().reciprocal()
        prec_prior = logvar_prior.exp().reciprocal()
        prec_post  = prec_dyn + prec_prior

        mu_post = (prec_dyn * mu_dyn + prec_prior * mu_prior) / prec_post
        return mu_post

# ── Loss ──────────────────────────────────────────────────────────────────────

def bjepa_loss(mu_dyn, logvar_dyn, mu_prior, logvar_prior,
               Z_T_real, gamma=0.1, beta=0.01):
    """
    BJEPA training loss — soft fusion.

    1. NLL: dynamics distribution should explain Z_T_real
    2. KL(dynamics || prior): soft regularization toward prior
    3. KL(prior || N(0,I)): keep prior well-behaved
    """
    # 1. Dynamics NLL — how well does (μ_dyn, σ_dyn) explain Z_T_real
    dyn_var = logvar_dyn.exp()
    nll = 0.5 * (
        logvar_dyn + (Z_T_real - mu_dyn).pow(2) / dyn_var
    ).sum(dim=-1).mean()

    # 2. KL(dynamics || prior) — soft fusion regularization
    var_rat  = (logvar_dyn - logvar_prior).exp()
    kl_prior = 0.5 * (
        var_rat
        + (mu_prior - mu_dyn).pow(2) / logvar_prior.exp()
        - 1
        - (logvar_dyn - logvar_prior)
    ).sum(dim=-1).mean()

    # 3. KL(prior || N(0,I)) — keep prior anchored
    kl_normal = -0.5 * (
        1 + logvar_prior - mu_prior.pow(2) - logvar_prior.exp()
    ).sum(dim=-1).mean()

    loss = nll + gamma * kl_prior + beta * kl_normal
    return loss, nll, kl_prior, kl_normal

# ── Extract representations ───────────────────────────────────────────────────

def extract_representations(s1_encoder, s2_predictor, loader, device):
    """
    Extract:
      Z_C_pred = mean_pool(S2(ctx))   — dynamics prediction
      Z_T_real = mean_pool(S1(tgt))   — actual next turn
    """
    all_z_C_pred, all_z_T = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Extracting'):
            ctx_ids  = batch['input_ids_a'].to(device)
            ctx_mask = batch['attention_mask_a'].to(device)
            tgt_ids  = batch['input_ids_b'].to(device)
            tgt_mask = batch['attention_mask_b'].to(device)

            # Z_T_real
            tgt_h = s1_encoder(tgt_ids, attention_mask=tgt_mask)
            if isinstance(tgt_h, tuple): tgt_h = tgt_h[0]
            z_T   = mean_pool(tgt_h, tgt_mask)

            # Z_C_pred — S2 output
            ctx_h    = s1_encoder(ctx_ids, attention_mask=ctx_mask)
            if isinstance(ctx_h, tuple): ctx_h = ctx_h[0]
            z_pred   = s2_predictor(ctx_h, ctx_h)
            z_C_pred = mean_pool(z_pred, ctx_mask)

            all_z_C_pred.append(z_C_pred.cpu())
            all_z_T.append(z_T.cpu())

    return torch.cat(all_z_C_pred), torch.cat(all_z_T)

# ── Load frozen models ────────────────────────────────────────────────────────

s1_encoder = Encoder(
    vocab_size  = CFG.model.vocab_size,
    hidden_size = CFG.model.hidden_size,
    num_heads   = CFG.model.num_heads,
    num_layers  = CFG.model.num_layers,
    max_seq_len = CFG.model.max_seq_len,
).to(DEVICE)

s1_ckpt = torch.load(CFG.training.s1_ckpt, map_location=DEVICE, weights_only=False)
s1_encoder.load_state_dict(s1_ckpt['context_encoder'])
s1_encoder.eval()
for p in s1_encoder.parameters():
    p.requires_grad = False
print('S1 encoder loaded and frozen.')

s2_predictor = DM(
    num_frames  = CFG.model.max_seq_len,
    depth       = CFG.model.pred_num_layers,
    heads       = CFG.model.pred_num_heads,
    mlp_dim     = CFG.model.pred_hidden_size * 4,
    input_dim   = CFG.model.dstc,
    hidden_dim  = CFG.model.pred_hidden_size,
    output_dim  = CFG.model.dstc,
    dim_head    = 64,
    dropout     = 0.0,
    emb_dropout = 0.0,
).to(DEVICE)

s2_ckpt = torch.load(
    Path(CFG.logging.exp_dir) / 'best.pt',
    map_location=DEVICE, weights_only=False
)
s2_predictor.load_state_dict(s2_ckpt['predictor'])
s2_predictor.eval()
for p in s2_predictor.parameters():
    p.requires_grad = False
print(f'S2 loaded and frozen (epoch {s2_ckpt["epoch"]}  val_loss={s2_ckpt["val_loss"]:.4f})')

# ── Dataloaders ───────────────────────────────────────────────────────────────

train_loader, val_loader = get_stage2_dataloaders(cfg_obj=CFG, tokenizer=tokenizer)
print(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')

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

print(f'Train: {z_T_train.shape} | Val: {z_T_val.shape}')

train_ds = TensorDataset(z_C_pred_train, z_T_train)
val_ds   = TensorDataset(z_C_pred_val,   z_T_val)

enc_train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
enc_val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)

# ── Models ────────────────────────────────────────────────────────────────────

D = CFG.model.hidden_size   # 256

dynamics_head = DynamicsHead(hidden_dim=D).to(DEVICE)
prior         = StaticPrior(latent_dim=D).to(DEVICE)

print(f'DynamicsHead params : {sum(p.numel() for p in dynamics_head.parameters()):,}')
print(f'StaticPrior params  : {sum(p.numel() for p in prior.parameters()):,}')

# ── Optimizer ─────────────────────────────────────────────────────────────────

trainable = list(dynamics_head.parameters()) + list(prior.parameters())
optimizer  = AdamW(trainable, lr=1e-3, weight_decay=0.05)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=30, eta_min=1e-4)

# ── Training loop ─────────────────────────────────────────────────────────────

history = {
    'train_loss': [], 'train_nll': [], 'train_kl_prior': [], 'train_kl_normal': [],
    'val_loss':   [], 'val_nll':   [], 'val_kl_prior':   [], 'val_kl_normal':   [],
}
best_val_loss = float('inf')
N_EPOCHS      = 30

print(f'\n{"="*60}')
print(f'  BJEPA Prior — {N_EPOCHS} epochs   device={DEVICE}')
print(f'{"="*60}\n')

for epoch in range(1, N_EPOCHS + 1):

    # Train
    dynamics_head.train(); prior.train()
    t_loss = t_nll = t_kl_p = t_kl_n = 0.0; n = 0

    for (z_C_pred, z_T) in tqdm(enc_train_loader, desc=f'Epoch {epoch:02d}', leave=False):
        z_C_pred = z_C_pred.to(DEVICE)
        z_T      = z_T.to(DEVICE)

        # Dynamics distribution from S2 output
        mu_dyn, logvar_dyn = dynamics_head(z_C_pred)

        # Prior distribution
        mu_prior, logvar_prior = prior.get_prior(z_C_pred.size(0), DEVICE)

        loss, nll, kl_p, kl_n = bjepa_loss(
            mu_dyn, logvar_dyn,
            mu_prior, logvar_prior,
            z_T
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        t_loss += loss.item(); t_nll  += nll.item()
        t_kl_p += kl_p.item(); t_kl_n += kl_n.item()
        n += 1

    scheduler.step()

    # Val
    dynamics_head.eval(); prior.eval()
    v_loss = v_nll = v_kl_p = v_kl_n = 0.0; m = 0

    with torch.no_grad():
        for (z_C_pred, z_T) in enc_val_loader:
            z_C_pred = z_C_pred.to(DEVICE)
            z_T      = z_T.to(DEVICE)

            mu_dyn, logvar_dyn     = dynamics_head(z_C_pred)
            mu_prior, logvar_prior = prior.get_prior(z_C_pred.size(0), DEVICE)

            loss, nll, kl_p, kl_n = bjepa_loss(
                mu_dyn, logvar_dyn,
                mu_prior, logvar_prior,
                z_T
            )
            v_loss += loss.item(); v_nll  += nll.item()
            v_kl_p += kl_p.item(); v_kl_n += kl_n.item()
            m += 1

    history['train_loss'].append(t_loss/n)
    history['train_nll'].append(t_nll/n)
    history['train_kl_prior'].append(t_kl_p/n)
    history['train_kl_normal'].append(t_kl_n/n)
    history['val_loss'].append(v_loss/m)
    history['val_nll'].append(v_nll/m)
    history['val_kl_prior'].append(v_kl_p/m)
    history['val_kl_normal'].append(v_kl_n/m)

    print(
        f'Epoch {epoch:02d}/{N_EPOCHS}  '
        f'train={t_loss/n:.4f} (nll={t_nll/n:.4f} kl_p={t_kl_p/n:.4f} kl_n={t_kl_n/n:.4f})  '
        f'val={v_loss/m:.4f}  '
        f'lr={optimizer.param_groups[0]["lr"]:.2e}'
    )

    if v_loss/m < best_val_loss:
        best_val_loss = v_loss/m
        torch.save({
            'epoch':         epoch,
            'dynamics_head': dynamics_head.state_dict(),
            'prior':         prior.state_dict(),
            'optimizer':     optimizer.state_dict(),
            'scheduler':     scheduler.state_dict(),
            'val_loss':      best_val_loss,
            'hidden_dim':    D,
        }, SAVE_DIR / 'best.pt')
        print(f'  ✓ saved → {SAVE_DIR}/best.pt')

# ── Inference demo ────────────────────────────────────────────────────────────

print('\n── Inference demo (PoE fusion) ──')
dynamics_head.eval(); prior.eval()

with torch.no_grad():
    z_sample  = z_C_pred_val[:8].to(DEVICE)
    mu_dyn, logvar_dyn = dynamics_head(z_sample)

    # Hard fusion: PoE
    z_fused   = prior.product_of_experts(mu_dyn, logvar_dyn, 8, DEVICE)

    print(f'  z_fused shape : {z_fused.shape}')
    print(f'  prior_mu norm : {prior.prior_mu.norm().item():.4f}')
    print(f'  prior_sigma   : {prior.prior_logvar.exp().sqrt().mean().item():.4f}')
    print(f'  dynamics_mu norm (sample): {mu_dyn.norm(dim=-1).mean().item():.4f}')

# ── Plot ──────────────────────────────────────────────────────────────────────

epochs_r = range(1, N_EPOCHS + 1)
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

axes[0].plot(epochs_r, history['train_loss'], 'b-', label='train')
axes[0].plot(epochs_r, history['val_loss'],   'r-', label='val')
axes[0].set_title('Total Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_r, history['train_nll'], 'g-', label='train')
axes[1].plot(epochs_r, history['val_nll'],   color='orange', label='val')
axes[1].set_title('NLL (dynamics fit)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs_r, history['train_kl_prior'], 'c-')
axes[2].set_title('KL(dynamics || prior)')
axes[2].grid(True, alpha=0.3)

axes[3].plot(epochs_r, history['train_kl_normal'], 'm-')
axes[3].set_title('KL(prior || N(0,I))')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(SAVE_DIR / 'training_curves_prior.png', dpi=150)
plt.close(fig)
print(f'Plot saved → {SAVE_DIR}/training_curves_prior.png')
print(f'Best val_loss: {best_val_loss:.4f}')