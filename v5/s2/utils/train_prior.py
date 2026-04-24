#!/usr/bin/env python3
"""
Stage 2.5: Prior Encoder (f_enc) Training
Maps Z_T_real → (μ, σ) in latent z_t space (32-dim)
Trained independently from S1 and S2.
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
ROOT    = '/content/notebooks_meta'
S1      = f'{ROOT}/v5/s1'
S2      = f'{ROOT}/v5/s2'
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = Path('/content/drive/MyDrive/metanet/v5/prior_encoder')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, ROOT)





# ── Imports ───────────────────────────────────────────────────────────────────
from v5.s1.cog_arch.encoder import Encoder
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








# ── f_enc architecture ────────────────────────────────────────────────────────

class PriorEncoder(nn.Module):
    """
    f_enc: Z_T_real → (μ, σ) in z_t space

    Input  : Z_T_real (B, D) — S1 mean-pooled representation of turn_{t+1}
             seen ONLY during training, never at inference
    Output : μ (B, latent_dim), σ (B, latent_dim)
             parameterize q(z_t | Z_T_real)

    At inference: z_t ~ N(0,I) directly — f_enc not used
    Later: high-level policy provides better z_t initialization
    """
    def __init__(
        self,
        input_dim  : int = 256,    # S1 hidden size
        hidden_dim : int = 256,
        latent_dim : int = 32,     # z_t dimension
    ):
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
        """
        z_T : (B, D) — Z_T_real, mean-pooled S1 representation
        returns μ, logvar : (B, latent_dim)
        """
        h      = self.net(z_T)
        mu     = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-10, 2)
        return mu, logvar

    def sample(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)











# ── f_exp: projects z_t back to D dims for injection ─────────────────────────

class LatentExpander(nn.Module):
    """
    f_exp: z_t (B, latent_dim) → (B, D)
    Projects z_t back to S1 representation space
    for injection into S2 output before S3.
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

def prior_encoder_loss(mu, logvar, Z_T_real, z_t, f_exp, beta=1.0):
    """
    Two terms:

    1. Reconstruction: f_exp(z_t) should recover Z_T_real
       — z_t must encode enough about Z_T_real to reconstruct it
       — this is what forces z_t to capture target structure

    2. KL: q(z_t | Z_T_real) → N(0,I)
       — keeps z_t space well-behaved for free sampling at inference
       — beta controls compression pressure
    """
    # Reconstruction: z_t expanded should be close to Z_T_real
    z_t_expanded = f_exp(z_t)                          # (B, D)
    recon_loss   = F.mse_loss(z_t_expanded, Z_T_real)

    # KL: push posterior toward N(0,I)
    kl_loss = -0.5 * (
        1 + logvar - mu.pow(2) - logvar.exp()
    ).sum(dim=-1).mean()

    return recon_loss + beta * kl_loss, recon_loss, kl_loss










# ── Extract Z_T representations ───────────────────────────────────────────────

def extract_target_representations(encoder, loader, device):
    """
    Extract Z_T_real = mean_pool(S1_encoder(turn_{t+1}))
    for all pairs in the dataloader.
    Returns tensor of shape (N, D).
    """
    encoder.eval()
    all_z_T = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Extracting Z_T representations'):
            tgt_ids  = batch['input_ids_b'].to(device)
            tgt_mask = batch['attention_mask_b'].to(device)

            h = encoder(tgt_ids, attention_mask=tgt_mask)
            if isinstance(h, tuple): h = h[0]

            # Mean pool
            mask_f = tgt_mask.unsqueeze(-1).float()
            z_T    = (h * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
            all_z_T.append(z_T.cpu())

    return torch.cat(all_z_T)   # (N, D)











# ── Main ──────────────────────────────────────────────────────────────────────

# Load S1 encoder (frozen)
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

# Dataloaders
train_loader, val_loader = get_stage2_dataloaders(
    cfg_obj=CFG, tokenizer=tokenizer
)
print(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')

# Extract Z_T representations (one time)
z_T_cache = SAVE_DIR / 'z_T_train.pt'
z_T_val_cache = SAVE_DIR / 'z_T_val.pt'

if z_T_cache.exists():
    print('Loading cached Z_T representations...')
    z_T_train = torch.load(z_T_cache)
    z_T_val   = torch.load(z_T_val_cache)
else:
    print('Extracting Z_T representations...')
    z_T_train = extract_target_representations(s1_encoder, train_loader, DEVICE)
    z_T_val   = extract_target_representations(s1_encoder, val_loader,   DEVICE)
    torch.save(z_T_train, z_T_cache)
    torch.save(z_T_val,   z_T_val_cache)

print(f'Z_T train: {z_T_train.shape} | Z_T val: {z_T_val.shape}')

# Build simple dataset from extracted representations
train_ds = TensorDataset(z_T_train)
val_ds   = TensorDataset(z_T_val)

enc_train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
enc_val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)

# Models
f_enc = PriorEncoder(
    input_dim  = CFG.model.hidden_size,   # 256
    hidden_dim = 256,
    latent_dim = 32,
).to(DEVICE)

f_exp = LatentExpander(
    latent_dim = 32,
    output_dim = CFG.model.hidden_size,   # 256
).to(DEVICE)

print(f'f_enc params: {sum(p.numel() for p in f_enc.parameters()):,}')
print(f'f_exp params: {sum(p.numel() for p in f_exp.parameters()):,}')

# Optimizer — trains f_enc and f_exp jointly
trainable = list(f_enc.parameters()) + list(f_exp.parameters())
optimizer = AdamW(trainable, lr=1e-3, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=30, eta_min=1e-4
)

# Beta annealing — start low to avoid posterior collapse
def get_beta(epoch, warmup=10, beta_max=1.0):
    return min(beta_max, beta_max * epoch / warmup)

# Training loop
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

    # ── Train ────────────────────────────────────────────────────────────────
    f_enc.train(); f_exp.train()
    t_loss = t_recon = t_kl = 0.0; n = 0

    for (z_T,) in tqdm(enc_train_loader, desc=f'Epoch {epoch:02d}', leave=False):
        z_T = z_T.to(DEVICE)

        mu, logvar = f_enc(z_T)
        z_t        = f_enc.sample(mu, logvar)
        loss, recon, kl = prior_encoder_loss(mu, logvar, z_T, z_t, f_exp, beta)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        t_loss  += loss.item();  t_recon += recon.item()
        t_kl    += kl.item();    n       += 1

    scheduler.step()

    # ── Val ──────────────────────────────────────────────────────────────────
    f_enc.eval(); f_exp.eval()
    v_loss = v_recon = v_kl = 0.0; m = 0

    with torch.no_grad():
        for (z_T,) in enc_val_loader:
            z_T = z_T.to(DEVICE)
            mu, logvar   = f_enc(z_T)
            z_t          = f_enc.sample(mu, logvar)
            loss, recon, kl = prior_encoder_loss(
                mu, logvar, z_T, z_t, f_exp, beta
            )
            v_loss  += loss.item();  v_recon += recon.item()
            v_kl    += kl.item();    m       += 1

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
axes[1].plot(epochs_r, history['val_recon'],   'orange', label='val')
axes[1].set_title('Reconstruction Loss')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs_r, history['train_kl'], 'c-', label='train')
axes[2].plot(epochs_r, history['val_kl'],   'm-', label='val')
axes[2].set_title('KL Divergence')
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(SAVE_DIR / 'training_curves_prior_encoder.png', dpi=150)
plt.close(fig)
print(f'\nPlot saved → {SAVE_DIR}/training_curves_prior_encoder.png')
print(f'Best val_loss: {best_val_loss:.4f}')