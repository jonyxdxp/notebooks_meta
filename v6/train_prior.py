#!/usr/bin/env python3
"""
v6/train_prior.py
Stage 2.5: BJEPA Goal Prior Training for v6.

Fixes over the old file (which pointed at v5 paths):
  - Uses v6/s2/encoder.py  (BERT/DMI 768-d, encode_single)
  - Uses v6/s2/cog_arch/dm.py  (DialogueJEPAPredictor, takes pre-embedded turns)
  - Uses v6/s2/config.py   (cfg SimpleNamespace, d_input=768)
  - No mean_pool needed — dataloader already returns CLS embeddings
  - DynamicsHead / LearnedGoalPrior live in v6/prior_encoder.py

Frozen:  DMI BERT encoder (via encode_single), DialogueJEPAPredictor
Trained: DynamicsHead (768→768 mu/logvar), LearnedGoalPrior (scalar σ_goal)
"""

import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = '/content/notebooks_meta'
V6       = f'{ROOT}/v6'
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = Path('/content/drive/MyDrive/metanet/v6/prior')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, ROOT)
sys.path.insert(0, V6)

# v6-specific imports
from v6.s2.config        import cfg
from v6.s2.cog_arch.dm  import DialogueJEPAPredictor
from v6.prior_encoder    import DynamicsHead, LearnedGoalPrior, bjepa_goal_loss

# v6 data — adjust import to match your actual data.py
# Expected collate output: ctx (B, max_turns, 768), tgt (B, 768),
#                          mask (B, max_turns) bool, lens (B,)
from v6.s2.data.data import make_dataloaders   # adapt name if different

print(f'Device: {DEVICE}')
print(f'Hidden dim (d_input): {cfg.d_input}')   # should print 768

# ── Load frozen S2 predictor ───────────────────────────────────────────────────

S2_CKPT = os.environ.get(
    'S2_CKPT',
    '/content/drive/MyDrive/metanet/v6/s2/best.pt'
)

predictor = DialogueJEPAPredictor().to(DEVICE)
s2_ckpt   = torch.load(S2_CKPT, map_location=DEVICE, weights_only=False)

# Handle checkpoint key variants
state_key = 'predictor' if 'predictor' in s2_ckpt else 'model_state_dict'
predictor.load_state_dict(s2_ckpt[state_key])
predictor.eval()
for p in predictor.parameters():
    p.requires_grad_(False)

val_loss_s2 = s2_ckpt.get('val_loss', s2_ckpt.get('loss', '?'))
print(f'S2 predictor loaded and frozen  (val_loss={val_loss_s2})')

# ── Extract and cache representations ─────────────────────────────────────────

def extract_representations(predictor, loader, device):
    """
    v6 version — no tokenization, no mean_pool.

    Dataloader yields: ctx (B, max_turns, 768), tgt (B, 768), mask, lens
      ctx  = pre-embedded context turns
      tgt  = pre-embedded target turn  =  Z_T_real directly
      mask = True where padding

    Returns:
      z_C_pred : (N, 768)  predictor output
      z_T      : (N, 768)  target turn embedding
    """
    all_z_C, all_z_T = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Extracting'):
            ctx, tgt, mask, lens = batch
            ctx  = ctx.to(device)   # (B, max_turns, 768)
            tgt  = tgt.to(device)   # (B, 768)
            mask = mask.to(device)  # (B, max_turns)

            # S2 predictor: (B, max_turns, 768) → (B, 768)
            z_pred = predictor(ctx, padding_mask=mask)

            all_z_C.append(z_pred.cpu())
            all_z_T.append(tgt.cpu())

    return torch.cat(all_z_C), torch.cat(all_z_T)


train_loader, valid_loader = make_dataloaders()
print(f'Train batches: {len(train_loader)} | Val batches: {len(valid_loader)}')

cache_train = SAVE_DIR / 'reps_train.pt'
cache_val   = SAVE_DIR / 'reps_val.pt'

if cache_train.exists():
    print('Loading cached representations...')
    tr = torch.load(cache_train, weights_only=False)
    vl = torch.load(cache_val,   weights_only=False)
    z_C_train, z_T_train = tr['z_C_pred'], tr['z_T']
    z_C_val,   z_T_val   = vl['z_C_pred'], vl['z_T']
else:
    print('Extracting representations (one-time pass, will be cached)...')
    z_C_train, z_T_train = extract_representations(predictor, train_loader, DEVICE)
    z_C_val,   z_T_val   = extract_representations(predictor, valid_loader, DEVICE)
    torch.save({'z_C_pred': z_C_train, 'z_T': z_T_train}, cache_train)
    torch.save({'z_C_pred': z_C_val,   'z_T': z_T_val},   cache_val)
    print(f'Cached → {SAVE_DIR}')

print(f'Train: {z_T_train.shape} | Val: {z_T_val.shape}')
# Expect: torch.Size([N, 768])

# ── Tensor datasets ────────────────────────────────────────────────────────────

enc_train = DataLoader(TensorDataset(z_C_train, z_T_train),
                       batch_size=256, shuffle=True)
enc_val   = DataLoader(TensorDataset(z_C_val,   z_T_val),
                       batch_size=256, shuffle=False)

# ── Trainable models ───────────────────────────────────────────────────────────

D = cfg.d_input  # 768

dynamics_head = DynamicsHead(hidden_dim=D).to(DEVICE)
goal_prior    = LearnedGoalPrior(z_dim=D, init_log_sigma=-1.0).to(DEVICE)

n_dyn   = sum(p.numel() for p in dynamics_head.parameters())
n_prior = sum(p.numel() for p in goal_prior.parameters())
print(f'DynamicsHead  params: {n_dyn:,}   (two Linear 768→768)')
print(f'GoalPrior     params: {n_prior}   (one scalar: log_sigma_goal)')

# ── Optimiser ──────────────────────────────────────────────────────────────────

trainable = list(dynamics_head.parameters()) + list(goal_prior.parameters())
optimizer  = AdamW(trainable, lr=1e-3, weight_decay=0.05)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.epochs, eta_min=1e-4)

# ── Training loop ──────────────────────────────────────────────────────────────

history = {k: [] for k in
    ['train_loss', 'train_nll', 'train_kl_goal', 'train_kl_normal',
     'val_loss', 'sigma_goal']}
best_val = float('inf')

print(f'\n{"="*60}')
print(f'  BJEPA Goal Prior v6 — {cfg.epochs} epochs   d={D}')
print(f'{"="*60}\n')

for epoch in range(1, cfg.epochs + 1):

    # ── train ──────────────────────────────────────────────────────────────────
    dynamics_head.train()
    goal_prior.train()
    t = {'loss': 0., 'nll': 0., 'kl_g': 0., 'kl_n': 0.}
    n = 0

    for z_C, z_T in tqdm(enc_train, desc=f'Epoch {epoch:02d}', leave=False):
        z_C = z_C.to(DEVICE)
        z_T = z_T.to(DEVICE)

        mu_dyn, logvar_dyn  = dynamics_head(z_C)
        mu_prior, sig_prior = goal_prior(z_T)   # μ_prior = Z_T_real at training

        loss, nll, kl_g, kl_n = bjepa_goal_loss(
            mu_dyn, logvar_dyn,
            mu_prior, sig_prior,
            z_T,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        t['loss'] += loss.item()
        t['nll']  += nll.item()
        t['kl_g'] += kl_g.item()
        t['kl_n'] += kl_n.item()
        n += 1

    scheduler.step()

    # ── val ────────────────────────────────────────────────────────────────────
    dynamics_head.eval()
    goal_prior.eval()
    v_loss = 0.; m = 0

    with torch.no_grad():
        for z_C, z_T in enc_val:
            z_C = z_C.to(DEVICE)
            z_T = z_T.to(DEVICE)
            mu_dyn, logvar_dyn  = dynamics_head(z_C)
            mu_prior, sig_prior = goal_prior(z_T)
            loss, _, _, _       = bjepa_goal_loss(
                mu_dyn, logvar_dyn, mu_prior, sig_prior, z_T)
            v_loss += loss.item()
            m += 1

    sigma = goal_prior.get_sigma().item()

    for k, v in [('train_loss',  t['loss']/n),
                 ('train_nll',   t['nll']/n),
                 ('train_kl_goal', t['kl_g']/n),
                 ('train_kl_normal', t['kl_n']/n),
                 ('val_loss',    v_loss/m),
                 ('sigma_goal',  sigma)]:
        history[k].append(v)

    print(
        f'Epoch {epoch:02d}/{cfg.epochs}  '
        f'train={t["loss"]/n:.4f} '
        f'(nll={t["nll"]/n:.4f} '
        f'kl_goal={t["kl_g"]/n:.4f} '
        f'kl_n={t["kl_n"]/n:.4f})  '
        f'val={v_loss/m:.4f}  '
        f'σ_goal={sigma:.4f}  '
        f'lr={optimizer.param_groups[0]["lr"]:.2e}'
    )

    if v_loss / m < best_val:
        best_val = v_loss / m
        torch.save({
            'epoch':         epoch,
            'dynamics_head': dynamics_head.state_dict(),
            'goal_prior':    goal_prior.state_dict(),
            'optimizer':     optimizer.state_dict(),
            'scheduler':     scheduler.state_dict(),
            'val_loss':      best_val,
            'sigma_goal':    sigma,
            'hidden_dim':    D,
        }, SAVE_DIR / 'best.pt')
        print(f'  ✓ saved → {SAVE_DIR}/best.pt')

# ── Quick sanity check ─────────────────────────────────────────────────────────

print('\n── Inference sanity check ──')
dynamics_head.eval()
goal_prior.eval()

with torch.no_grad():
    z_C = z_C_val[:8].to(DEVICE)
    z_T = z_T_val[:8].to(DEVICE)

    mu_dyn, logvar_dyn  = dynamics_head(z_C)

    # PoE with oracle η (upper bound)
    z_poe  = goal_prior.product_of_experts(mu_dyn, logvar_dyn, z_T)
    # Dynamics only (no prior, lower bound)
    z_dyn  = mu_dyn

    T_norm = F.normalize(z_T,   dim=-1)
    sim_poe = F.cosine_similarity(F.normalize(z_poe, dim=-1), T_norm).mean().item()
    sim_dyn = F.cosine_similarity(F.normalize(z_dyn, dim=-1), T_norm).mean().item()

    print(f'  σ_goal learned      : {goal_prior.get_sigma().item():.4f}')
    print(f'  PoE sim to Z_T_real : {sim_poe:.4f}')
    print(f'  Dynamics-only sim   : {sim_dyn:.4f}')
    print(f'  PoE improvement     : {sim_poe - sim_dyn:+.4f}')
    print(f'  Best val_loss       : {best_val:.4f}')

# ── Plots ──────────────────────────────────────────────────────────────────────

ep = range(1, cfg.epochs + 1)
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

axes[0].plot(ep, history['train_loss'], 'b-', label='train')
axes[0].plot(ep, history['val_loss'],   'r-', label='val')
axes[0].set_title('Total Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(ep, history['train_nll'], 'g-')
axes[1].set_title('NLL (dynamics fit)'); axes[1].grid(True, alpha=0.3)

axes[2].plot(ep, history['train_kl_goal'], 'c-')
axes[2].set_title('KL(dynamics ‖ goal_prior)'); axes[2].grid(True, alpha=0.3)

axes[3].plot(ep, history['sigma_goal'], 'm-')
axes[3].set_title('σ_goal (prior sharpness)'); axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(SAVE_DIR / 'training_curves.png', dpi=150)
plt.close(fig)
print(f'\nPlot → {SAVE_DIR}/training_curves.png')