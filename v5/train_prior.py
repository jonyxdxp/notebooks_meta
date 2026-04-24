#!/usr/bin/env python3
"""
Stage 2.5: Prior Encoder (f_enc) Training
Maps Z_T_real → (μ, σ) in latent z_t space (32-dim)
z_t encodes what S2 couldn't predict (residual)
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
SAVE_DIR = Path('/content/drive/MyDrive/metanet/v5/prior_encoder')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, ROOT)

# ── Imports ───────────────────────────────────────────────────────────────────
from v5.s1.cog_arch.encoder import Encoder
from v5.s2.cog_arch.dm import DM, Projector
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

# ── Architectures ─────────────────────────────────────────────────────────────

class PriorEncoder(nn.Module):
    """
    f_enc: Z_T_real → (μ, logvar) in z_t space (32-dim)

    Input  : Z_T_real (B, D) — S1 mean-pooled representation of turn_{t+1}
             seen ONLY during training, never at inference
    Output : μ, logvar (B, latent_dim)

    At inference: z_t ~ N(0,I) directly — f_enc is discarded
    """
    def __init__(self, input_dim=256, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.mu_head     = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_T):
        h      = self.net(z_T)
        mu     = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-10, 2)
        return mu, logvar

    def sample(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)


class LatentExpander(nn.Module):
    """
    f_exp: z_t (B, latent_dim) → (B, D)
    Projects z_t to S1 representation space.
    Trained to reconstruct the residual Z_T_real - Z_C_pred.
    """
    def __init__(self, latent_dim=32, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, z_t):
        return self.net(z_t)

# ── Loss ──────────────────────────────────────────────────────────────────────

def kl_with_free_bits(mu, logvar, free_bits=0.5):
    """
    KL with minimum per-dimension threshold.
    Prevents complete posterior collapse —
    each dimension must maintain at least free_bits of KL.
    """
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    return kl_per_dim.sum(dim=-1).mean()

def prior_encoder_loss(mu, logvar, Z_T_real, Z_C_pred, z_t, f_exp, beta):
    """
    z_t must encode the RESIDUAL — what S2 couldn't predict.
    This prevents posterior collapse:
      mean(residual) = 0, so f_exp can't cheat with a constant output.

    recon: f_exp(z_t) ≈ Z_T_real - Z_C_pred
    kl:    q(z_t | Z_T_real) → N(0,I)
    """
    residual     = Z_T_real - Z_C_pred.detach()   # (B, D)
    z_t_expanded = f_exp(z_t)                     # (B, D)
    recon_loss   = F.mse_loss(z_t_expanded, residual)
    kl_loss      = kl_with_free_bits(mu, logvar, free_bits=0.5)

    return recon_loss + beta * kl_loss, recon_loss, kl_loss

# ── Extract representations ───────────────────────────────────────────────────

def extract_representations(s1_encoder, s2_predictor, loader, device):
    """
    For each pair (turn_t, turn_{t+1}):
      Z_T_real  = mean_pool(S1_encoder(turn_{t+1}))   actual next turn
      Z_C_pred  = mean_pool(S2_predictor(turn_t))      S2 prediction
    """
    s1_encoder.eval()
    s2_predictor.eval()
    all_z_T, all_z_C_pred = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Extracting representations'):
            ctx_ids  = batch['input_ids_a'].to(device)
            ctx_mask = batch['attention_mask_a'].to(device)
            tgt_ids  = batch['input_ids_b'].to(device)
            tgt_mask = batch['attention_mask_b'].to(device)

            # Z_T_real — actual next turn encoding
            tgt_h = s1_encoder(tgt_ids, attention_mask=tgt_mask)
            if isinstance(tgt_h, tuple): tgt_h = tgt_h[0]
            z_T   = mean_pool(tgt_h, tgt_mask)          # (B, D)

            # Z_C_pred — what S2 predicts from context
            ctx_h    = s1_encoder(ctx_ids, attention_mask=ctx_mask)
            if isinstance(ctx_h, tuple): ctx_h = ctx_h[0]
            z_pred   = s2_predictor(ctx_h, ctx_h)       # (B, L, D)
            z_C_pred = mean_pool(z_pred, ctx_mask)      # (B, D)

            all_z_T.append(z_T.cpu())
            all_z_C_pred.append(z_C_pred.cpu())

    return torch.cat(all_z_T), torch.cat(all_z_C_pred)

# ── Load frozen models ────────────────────────────────────────────────────────

# S1 encoder
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

# S2 predictor
dstc        = CFG.model.dstc
s2_predictor = DM(
    num_frames  = CFG.model.max_seq_len,
    depth       = CFG.model.pred_num_layers,
    heads       = CFG.model.pred_num_heads,
    mlp_dim     = CFG.model.pred_hidden_size * 4,
    input_dim   = dstc,
    hidden_dim  = CFG.model.pred_hidden_size,
    output_dim  = dstc,
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
print(f'S2 predictor loaded and frozen (epoch {s2_ckpt["epoch"]}  val_loss={s2_ckpt["val_loss"]:.4f})')

# ── Dataloaders ───────────────────────────────────────────────────────────────

train_loader, val_loader = get_stage2_dataloaders(
    cfg_obj=CFG, tokenizer=tokenizer
)
print(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')

# ── Extract and cache representations ─────────────────────────────────────────

cache_train = SAVE_DIR / 'reps_train.pt'
cache_val   = SAVE_DIR / 'reps_val.pt'

if cache_train.exists():
    print('Loading cached representations...')
    cache_tr      = torch.load(cache_train)
    cache_vl      = torch.load(cache_val)
    z_T_train     = cache_tr['z_T']
    z_C_pred_train = cache_tr['z_C_pred']
    z_T_val       = cache_vl['z_T']
    z_C_pred_val  = cache_vl['z_C_pred']
else:
    print('Extracting representations (one time)...')
    z_T_train, z_C_pred_train = extract_representations(
        s1_encoder, s2_predictor, train_loader, DEVICE)
    z_T_val, z_C_pred_val = extract_representations(
        s1_encoder, s2_predictor, val_loader, DEVICE)
    torch.save({'z_T': z_T_train, 'z_C_pred': z_C_pred_train}, cache_train)
    torch.save({'z_T': z_T_val,   'z_C_pred': z_C_pred_val},   cache_val)

print(f'Z_T train: {z_T_train.shape} | Z_T val: {z_T_val.shape}')

# Verify residual is zero-mean (sanity check)
residual_mean = (z_T_train - z_C_pred_train).mean().item()
residual_std  = (z_T_train - z_C_pred_train).std().item()
print(f'Residual mean={residual_mean:.4f}  std={residual_std:.4f}')

# ── Build dataloaders from cached tensors ─────────────────────────────────────

train_ds = TensorDataset(z_T_train, z_C_pred_train)
val_ds   = TensorDataset(z_T_val,   z_C_pred_val)

enc_train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
enc_val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)

# ── Models ────────────────────────────────────────────────────────────────────

f_enc = PriorEncoder(
    input_dim  = CFG.model.hidden_size,
    hidden_dim = 256,
    latent_dim = 32,
).to(DEVICE)

f_exp = LatentExpander(
    latent_dim = 32,
    output_dim = CFG.model.hidden_size,
).to(DEVICE)

print(f'f_enc params: {sum(p.numel() for p in f_enc.parameters()):,}')
print(f'f_exp params: {sum(p.numel() for p in f_exp.parameters()):,}')

# ── Optimizer ─────────────────────────────────────────────────────────────────

trainable = list(f_enc.parameters()) + list(f_exp.parameters())
optimizer = AdamW(trainable, lr=1e-3, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=30, eta_min=1e-4
)

def get_beta(epoch, warmup=5, beta_max=4.0):
    """Beta annealing: start at 0.5, reach beta_max after warmup epochs."""
    return min(beta_max, 0.5 + (beta_max - 0.5) * epoch / warmup)

# ── Training loop ─────────────────────────────────────────────────────────────

history = {
    'train_loss': [], 'train_recon': [], 'train_kl': [],
    'val_loss':   [], 'val_recon':   [], 'val_kl':   [],
}
best_val_loss = float('inf')
N_EPOCHS      = 30

print(f'\n{"="*60}')
print(f'  Prior Encoder — {N_EPOCHS} epochs   device={DEVICE}')
print(f'{"="*60}\n')

for epoch in range(1, N_EPOCHS + 1):
    beta = get_beta(epoch)

    # Train
    f_enc.train(); f_exp.train()
    t_loss = t_recon = t_kl = 0.0; n = 0

    for (z_T, z_C_pred) in tqdm(enc_train_loader, desc=f'Epoch {epoch:02d}', leave=False):
        z_T      = z_T.to(DEVICE)
        z_C_pred = z_C_pred.to(DEVICE)

        mu, logvar = f_enc(z_T)
        z_t        = f_enc.sample(mu, logvar)
        loss, recon, kl = prior_encoder_loss(
            mu, logvar, z_T, z_C_pred, z_t, f_exp, beta
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        t_loss += loss.item(); t_recon += recon.item()
        t_kl   += kl.item();   n       += 1

    scheduler.step()

    # Val
    f_enc.eval(); f_exp.eval()
    v_loss = v_recon = v_kl = 0.0; m = 0

    with torch.no_grad():
        for (z_T, z_C_pred) in enc_val_loader:
            z_T      = z_T.to(DEVICE)
            z_C_pred = z_C_pred.to(DEVICE)

            mu, logvar = f_enc(z_T)
            z_t        = f_enc.sample(mu, logvar)
            loss, recon, kl = prior_encoder_loss(
                mu, logvar, z_T, z_C_pred, z_t, f_exp, beta
            )
            v_loss  += loss.item(); v_recon += recon.item()
            v_kl    += kl.item();   m       += 1

    history['train_loss'].append(t_loss/n)
    history['train_recon'].append(t_recon/n)
    history['train_kl'].append(t_kl/n)
    history['val_loss'].append(v_loss/m)
    history['val_recon'].append(v_recon/m)
    history['val_kl'].append(v_kl/m)

    print(
        f'Epoch {epoch:02d}/{N_EPOCHS}  '
        f'train={t_loss/n:.4f} (recon={t_recon/n:.4f} kl={t_kl/n:.4f})  '
        f'val={v_loss/m:.4f} (recon={v_recon/m:.4f} kl={v_kl/m:.4f})  '
        f'beta={beta:.2f}  lr={optimizer.param_groups[0]["lr"]:.2e}'
    )

    if v_loss/m < best_val_loss:
        best_val_loss = v_loss/m
        torch.save({
            'epoch':      epoch,
            'f_enc':      f_enc.state_dict(),
            'f_exp':      f_exp.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'scheduler':  scheduler.state_dict(),
            'val_loss':   best_val_loss,
            'latent_dim': 32,
            'input_dim':  CFG.model.hidden_size,
        }, SAVE_DIR / 'best.pt')
        print(f'  ✓ saved → {SAVE_DIR}/best.pt')

# ── Plot ──────────────────────────────────────────────────────────────────────
epochs_r = range(1, N_EPOCHS + 1)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(epochs_r, history['train_loss'], 'b-', label='train')
axes[0].plot(epochs_r, history['val_loss'],   'r-', label='val')
axes[0].set_title('Total Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_r, history['train_recon'], 'g-', label='train')
axes[1].plot(epochs_r, history['val_recon'],   color='orange', label='val')
axes[1].set_title('Reconstruction Loss (residual)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs_r, history['train_kl'], 'c-', label='train')
axes[2].plot(epochs_r, history['val_kl'],   'm-', label='val')
axes[2].set_title('KL Divergence (free bits=0.5)')
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(SAVE_DIR / 'training_curves_prior_encoder.png', dpi=150)
plt.close(fig)
print(f'\nPlot saved → {SAVE_DIR}/training_curves_prior_encoder.png')
print(f'Best val_loss: {best_val_loss:.4f}')