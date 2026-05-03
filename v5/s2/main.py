#!/usr/bin/env python3
"""
Stage 2: Entrenamiento del Predictor JEPA (Turn Pair Prediction)
"""

import sys
import os
import copy
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = '/content/notebooks_meta'
S1   = f'{ROOT}/v5/s1'
S2   = f'{ROOT}/v5/s2'
sys.path.insert(0, ROOT)

# ── Config de S2 (explícito para no pisar s1/config.py) ──────────────────────
from v5.s2.config import CFG, DEVICE


# ════════════════════════════════════════════════════════════════════════════

# ── Arquitecturas ─────────────────────────────────────────────────────────────
from v5.s1.cog_arch.encoder import Encoder


from v5.s2.data.dataset import VOCAB_SIZE, tokenizer
from v5.s2.data.dataloader import get_stage2_dataloaders


# ========================== Dataloaders ==========================

train_loader, val_loader = get_stage2_dataloaders(cfg_obj=CFG, tokenizer=tokenizer)
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

# ========================== Models ==========================

context_encoder = Encoder(
    vocab_size  = CFG.model.vocab_size,
    hidden_size = CFG.model.hidden_size,
    num_heads   = CFG.model.num_heads,
    num_layers  = CFG.model.num_layers,
    max_seq_len = CFG.model.max_seq_len,
).to(DEVICE)
target_encoder = copy.deepcopy(context_encoder).to(DEVICE)

# Cargar checkpoint Stage 1
s1_ckpt = torch.load(CFG.training.s1_ckpt, map_location=DEVICE, weights_only=False)
context_encoder.load_state_dict(s1_ckpt['context_encoder'])
target_encoder.load_state_dict(s1_ckpt['target_encoder'])
print(f"✓ S1 checkpoint cargado (epoch {s1_ckpt.get('epoch', '?')})")

for enc in (context_encoder, target_encoder):
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
print("Encoders frozen.")

from v5.s2.cog_arch.dm import DM

predictor = DM(
    num_frames = CFG.model.pred_num_frames,   # max_turns-1 = 5, but DM uses this for pos_embedding
    depth      = CFG.model.pred_num_layers,
    heads      = CFG.model.pred_num_heads,
    mlp_dim    = CFG.model.pred_hidden_size * 4,
    input_dim  = CFG.model.hidden_size,
    hidden_dim = CFG.model.pred_hidden_size,
    output_dim = CFG.model.hidden_size,
    dim_head   = 64,
    dropout    = 0.1,
    emb_dropout= 0.1,
).to(DEVICE)


# ========================== Loss / Optimizer ==========================


trainable_params = list(predictor.parameters())
optimizer = AdamW(trainable_params, lr=CFG.optim.lr, weight_decay=CFG.optim.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CFG.optim.epochs, eta_min=CFG.optim.lr * 0.1)
# ========================== Helpers ==========================


def mean_pool(hidden, mask):
    mask_f = mask.unsqueeze(-1).float()
    return (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)






def forward_step(batch):
    hist_ids   = batch['history_ids'].to(DEVICE)
    hist_masks = batch['history_masks'].to(DEVICE)
    tgt_ids    = batch['tgt_ids'].to(DEVICE)
    tgt_mask   = batch['tgt_mask'].to(DEVICE)

    B, T, L = hist_ids.shape

    with torch.no_grad():
        flat_ids   = hist_ids.view(B*T, L)
        flat_masks = hist_masks.view(B*T, L)

        valid  = flat_masks.sum(-1) > 0
        D      = context_encoder.token_embedding.embedding_dim
        z_flat = torch.zeros(B*T, D, device=DEVICE)

        if valid.any():
            h = context_encoder(flat_ids[valid], attention_mask=flat_masks[valid])
            if isinstance(h, tuple): h = h[0]
            z_flat[valid] = mean_pool(h, flat_masks[valid])

        z_seq = z_flat.view(B, T, -1)

        tgt_h = context_encoder(tgt_ids, attention_mask=tgt_mask)
        if isinstance(tgt_h, tuple): tgt_h = tgt_h[0]
        z_tgt = mean_pool(tgt_h, tgt_mask)

    z_pred = predictor(z_seq, torch.zeros_like(z_seq))[:, -1, :]
    loss = F.mse_loss(z_pred, z_tgt)
    return {'loss': loss}




def save_best(epoch, val_loss):
    path = Path(CFG.logging.exp_dir) / 'best.pt'
    torch.save({          # ← indent this whole block one level in
        'epoch':     epoch,
        'predictor': predictor.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'val_loss':  val_loss,
    }, path)
    print(f'  ✓ saved → {path}')

@torch.no_grad()
def validation_loop():
    predictor.eval()
    totals = {}; n = 0
    for batch in val_loader:
        d = forward_step(batch)
        for k, v in d.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n += 1
    predictor.train()   # ← add this
    return {k: v/n for k, v in totals.items()}

# ========================== Training Loop ==========================

history = {'train_loss': [], 'val_loss': []}

best_val_loss = float('inf')
Path(CFG.logging.exp_dir).mkdir(parents=True, exist_ok=True)

# Resume desde best si existe
# Resume desde best si existe
best_ckpt = Path(CFG.logging.exp_dir) / 'best.pt'
start_epoch = 1
print('Training from scratch (epoch 1).')
if best_ckpt.exists():
    ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    predictor.load_state_dict(ckpt['predictor'])
    
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    best_val_loss = ckpt['val_loss']
    start_epoch   = ckpt['epoch'] + 1
    print(f'Resumiendo desde epoch {ckpt["epoch"]}  val_loss={best_val_loss:.4f}')
else:
    print('Sin checkpoint previo, empezando desde cero.')

print(f'\n{"="*60}')
print(f'  Text JEPA S2 — {CFG.optim.epochs} epochs   device={DEVICE}')
print(f'{"="*60}\n')

for epoch in range(start_epoch, CFG.optim.epochs + 1):
    predictor.train()
    totals = {}; n = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch:02d}', leave=False)
    for batch in pbar:
        d = forward_step(batch)
        d['loss'].backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step(); optimizer.zero_grad()
        for k, v in d.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n += 1
        pbar.set_postfix({k: f'{v.item():.4f}' for k, v in d.items()})

    scheduler.step()
    tr = {k: v/n for k, v in totals.items()}
    vl = validation_loop()

    history['train_loss'].append(tr['loss'])
    history['val_loss'].append(vl['loss'])

    print(f'Epoch {epoch:02d}/{CFG.optim.epochs}  train={tr["loss"]:.4f}  val={vl["loss"]:.4f}  lr={optimizer.param_groups[0]["lr"]:.2e}')

    if vl['loss'] < best_val_loss:
        best_val_loss = vl['loss']
        save_best(epoch, best_val_loss)
        print(f'  ★ new best val_loss={best_val_loss:.4f}')

print('\nStage 2 complete.')

# ── Plotting ──────────────────────────────────────────────────────────────────
epochs_range = list(range(1, len(history['train_loss']) + 1))
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(epochs_range, history['train_loss'], 'b-o', label='Train', markersize=4)
ax.plot(epochs_range, history['val_loss'],   'r-s', label='Val',   markersize=4)
ax.set_title('Loss'); ax.legend(); ax.grid(True, alpha=0.3)


plt.tight_layout()
plot_path = Path(CFG.logging.exp_dir) / 'training_curves_stage2.png'
plt.savefig(plot_path, dpi=150); plt.close(fig)
print(f'Plot saved → {plot_path}')
print(f'Best val_loss: {best_val_loss:.4f}')