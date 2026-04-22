# v5/s2/train_prior.py
"""
Train the prior encoder independently.
S1 and S2 stay completely frozen.
"""

import sys
import os
import copy
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/content/notebooks_meta'
S1   = f'{ROOT}/v5/s1'
S2   = f'{ROOT}/v5/s2'
sys.path.insert(0, ROOT)

from v5.s2.config import CFG, DEVICE
from v5.s1.cog_arch.encoder import Encoder
from v5.s2.prior_encoder import PriorEncoder, prior_loss

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_dataset    = _load_module('data.dataset',    f'{S2}/data/dataset.py')
_dataloader = _load_module('data.dataloader', f'{S2}/data/dataloader.py')
tokenizer   = _dataset.tokenizer
get_stage2_dataloaders = _dataloader.get_stage2_dataloaders

# ── Load frozen S1 encoder ────────────────────────────────────────────────────

target_encoder = Encoder(
    vocab_size  = CFG.model.vocab_size,
    hidden_size = CFG.model.hidden_size,
    num_heads   = CFG.model.num_heads,
    num_layers  = CFG.model.num_layers,
    max_seq_len = CFG.model.max_seq_len,
).to(DEVICE)

s1_ckpt = torch.load(CFG.training.s1_ckpt, map_location=DEVICE, weights_only=False)
target_encoder.load_state_dict(s1_ckpt['target_encoder'])
target_encoder.eval()
for p in target_encoder.parameters():
    p.requires_grad = False
print('S1 target encoder loaded and frozen.')

# ── Dataloaders ───────────────────────────────────────────────────────────────

train_loader, val_loader = get_stage2_dataloaders(cfg_obj=CFG, tokenizer=tokenizer)
print(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')

# ── Prior encoder ─────────────────────────────────────────────────────────────

prior_encoder = PriorEncoder(
    z_dim      = CFG.model.hidden_size,   # 256
    hidden_dim = 256,
    input_dim  = None,                    # unconditional for now
).to(DEVICE)

print(f'Prior encoder params: {sum(p.numel() for p in prior_encoder.parameters()):,}')

# ── Optimizer ─────────────────────────────────────────────────────────────────

optimizer = AdamW(
    prior_encoder.parameters(),
    lr           = 1e-3,
    weight_decay = 0.05,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=30, eta_min=1e-4)

# ── Beta annealing — avoid posterior collapse ─────────────────────────────────

def get_beta(epoch, warmup=10, beta_max=1.0):
    return min(beta_max, beta_max * epoch / warmup)

# ── Helpers ───────────────────────────────────────────────────────────────────

def mean_pool(hidden, mask):
    mask_f = mask.unsqueeze(-1).float()
    return (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)

def extract_Z_T(batch):
    """Extract real Z_T from target turn via frozen S1 encoder."""
    tgt_ids  = batch['input_ids_b'].to(DEVICE)
    tgt_mask = batch['attention_mask_b'].to(DEVICE)
    with torch.no_grad():
        h = target_encoder(tgt_ids, attention_mask=tgt_mask)
        if isinstance(h, tuple): h = h[0]
        z = mean_pool(h, tgt_mask)   # (B, D)
    return z

# ── Training loop ─────────────────────────────────────────────────────────────

SAVE_DIR = Path(CFG.logging.exp_dir)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

history = {
    'train_loss': [], 'train_nll': [], 'train_kl': [],
    'val_loss':   [], 'val_nll':   [], 'val_kl':   [],
    'mu_norm': [], 'sigma_mean': [],
}

best_val_loss = float('inf')
N_EPOCHS      = 30

print(f'\n{"="*60}')
print(f'  Prior Encoder Training — {N_EPOCHS} epochs')
print(f'{"="*60}\n')

for epoch in range(1, N_EPOCHS + 1):
    beta = get_beta(epoch)

    # ── Train ─────────────────────────────────────────────────────────
    prior_encoder.train()
    t_loss = t_nll = t_kl = 0.0; n = 0

    for batch in tqdm(train_loader, desc=f'Epoch {epoch:02d}', leave=False):
        Z_T_real = extract_Z_T(batch)            # (B, 256)

        mu, sigma = prior_encoder()              # unconditional
        # Expand to batch size
        B         = Z_T_real.size(0)
        mu_b      = mu.unsqueeze(0).expand(B, -1)
        sigma_b   = sigma.unsqueeze(0).expand(B, -1)

        loss, nll, kl = prior_loss(mu_b, sigma_b, Z_T_real, beta)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prior_encoder.parameters(), 1.0)
        optimizer.step()

        t_loss += loss.item(); t_nll += nll.item(); t_kl += kl.item()
        n += 1

    scheduler.step()

    # ── Val ───────────────────────────────────────────────────────────
    prior_encoder.eval()
    v_loss = v_nll = v_kl = 0.0; m = 0

    with torch.no_grad():
        for batch in val_loader:
            Z_T_real  = extract_Z_T(batch)
            B         = Z_T_real.size(0)
            mu, sigma = prior_encoder()
            mu_b      = mu.unsqueeze(0).expand(B, -1)
            sigma_b   = sigma.unsqueeze(0).expand(B, -1)
            loss, nll, kl = prior_loss(mu_b, sigma_b, Z_T_real, beta)
            v_loss += loss.item(); v_nll += nll.item(); v_kl += kl.item()
            m += 1

    # ── Log ───────────────────────────────────────────────────────────
    mu_d, sigma_d = prior_encoder()
    history['train_loss'].append(t_loss / n)
    history['train_nll'].append(t_nll / n)
    history['train_kl'].append(t_kl / n)
    history['val_loss'].append(v_loss / m)
    history['val_nll'].append(v_nll / m)
    history['val_kl'].append(v_kl / m)
    history['mu_norm'].append(mu_d.norm().item())
    history['sigma_mean'].append(sigma_d.mean().item())

    print(
        f'Epoch {epoch:02d}/{N_EPOCHS}  '
        f'train={t_loss/n:.4f} (nll={t_nll/n:.4f} kl={t_kl/n:.4f})  '
        f'val={v_loss/m:.4f}  '
        f'μ_norm={mu_d.norm():.3f}  σ_mean={sigma_d.mean():.3f}  '
        f'β={beta:.2f}'
    )

    if v_loss / m < best_val_loss:
        best_val_loss = v_loss / m
        torch.save({
            'epoch':     epoch,
            'prior':     prior_encoder.state_dict(),
            'val_loss':  best_val_loss,
            'mu':        mu_d.detach().cpu(),
            'sigma':     sigma_d.detach().cpu(),
        }, SAVE_DIR / 'prior_best.pt')
        print(f'  ✓ saved → {SAVE_DIR}/prior_best.pt')

# ── Plot ──────────────────────────────────────────────────────────────────────

epochs_r = range(1, N_EPOCHS + 1)
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0,0].plot(epochs_r, history['train_loss'], label='train')
axes[0,0].plot(epochs_r, history['val_loss'],   label='val')
axes[0,0].set_title('Total Loss'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(epochs_r, history['train_nll'], 'g', label='train nll')
axes[0,1].plot(epochs_r, history['val_nll'],   'r', label='val nll')
axes[0,1].set_title('NLL'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(epochs_r, history['mu_norm'],    'b', label='μ norm')
axes[1,0].plot(epochs_r, history['sigma_mean'], 'orange', label='σ mean')
axes[1,0].set_title('Prior Parameters'); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(epochs_r, history['train_kl'], 'purple')
axes[1,1].set_title('KL Divergence'); axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(SAVE_DIR / 'prior_training_curves.png', dpi=150)
plt.close(fig)
print(f'Plot saved → {SAVE_DIR}/prior_training_curves.png')