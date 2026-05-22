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
from torch.optim import AdamW
from tqdm import tqdm
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = '/content/notebooks_meta'
S1   = f'{ROOT}/v5/s1'
S2   = f'{ROOT}/v5/s2'
sys.path.insert(0, ROOT)

# ── Config de S2 (explícito para no pisar s1/config.py) ──────────────────────
from v5.s2.config import CFG, DEVICE

# ── Arquitecturas ─────────────────────────────────────────────────────────────
from v5.s1.cog_arch.encoder import Encoder
from v5.s2.cog_arch.dm import DM, Projector
from v5.s2.losses import VCLoss

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

predictor = DM(
    num_frames = CFG.model.max_seq_len,
    depth      = CFG.model.pred_num_layers,
    heads      = CFG.model.pred_num_heads,
    mlp_dim    = CFG.model.pred_hidden_size * 4,
    input_dim  = CFG.model.dstc,
    hidden_dim = CFG.model.pred_hidden_size,
    output_dim = CFG.model.dstc,
    dim_head   = 64,
    dropout    = 0.1,
    emb_dropout= 0.1,
).to(DEVICE)
print(f"Predictor params: {sum(p.numel() for p in predictor.parameters()):,}")

dstc      = CFG.model.dstc
projector = Projector(f"{dstc}-{dstc*2}-{dstc}").to(DEVICE)
print(f"Projector: {dstc}-{dstc*4}-{dstc*4}")

# ========================== Loss / Optimizer ==========================

ploss       = torch.nn.MSELoss()
regularizer = VCLoss(std_coeff=CFG.loss.std_coeff, cov_coeff=CFG.loss.cov_coeff)

trainable_params = list(predictor.parameters()) + list(projector.parameters())
optimizer = AdamW(trainable_params, lr=CFG.optim.lr, weight_decay=CFG.optim.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CFG.optim.epochs, eta_min=CFG.optim.lr * 0.1)
# ========================== Helpers ==========================

def project_seq(projector, x):
    """
    Aplica el projector a una secuencia (B, L, D).
    BatchNorm1d espera (N, D) — reshapeamos, proyectamos, volvemos.
    """
    B, L, D = x.shape
    x_flat  = x.reshape(B * L, D)          # (B*L, D)
    out     = projector(x_flat)             # (B*L, D')
    return out.reshape(B, L, -1)            # (B, L, D')

def seq_mse_loss(pred, target, mask):
    """
    MSE solo en posiciones no-padding.
    pred, target : (B, L, D)
    mask         : (B, L) — 1 = token real, 0 = padding
    """
    mask_f = mask.unsqueeze(-1).float()     # (B, L, 1)
    diff   = (pred - target) ** 2           # (B, L, D)
    loss   = (diff * mask_f).sum() / (mask_f.sum() * pred.size(-1) + 1e-9)
    return loss

def unpack(batch):
    return (
        batch['input_ids_a'].to(DEVICE),
        batch['attention_mask_a'].to(DEVICE),
        batch['input_ids_b'].to(DEVICE),
        batch['attention_mask_b'].to(DEVICE),
    )

def forward_step(batch):
    ctx_ids, ctx_mask, tgt_ids, tgt_mask = unpack(batch)

    with torch.no_grad():
        ctx_h = context_encoder(ctx_ids, attention_mask=ctx_mask)
        tgt_h = target_encoder(tgt_ids, attention_mask=tgt_mask)
        if isinstance(ctx_h, tuple): ctx_h = ctx_h[0]
        if isinstance(tgt_h, tuple): tgt_h = tgt_h[0]
    # ctx_h, tgt_h : (B, L, D)

    # Predictor: secuencia completa, condicionado por sí mismo
    pred = predictor(ctx_h, ctx_h)          # (B, L, D)

    # Proyectar secuencias completas
    pred_proj = project_seq(projector, pred)            # (B, L, D')
    tgt_proj  = project_seq(projector, tgt_h.detach()) # (B, L, D')

    # Loss solo en tokens reales del target
    pred_loss           = seq_mse_loss(pred_proj, tgt_proj, tgt_mask)

    # VC loss: aplanar a (B*L, D') filtrando padding
    B, L, Dp = pred_proj.shape
    mask_flat = tgt_mask.reshape(B * L).bool()
    pred_flat = pred_proj.reshape(B * L, Dp)[mask_flat]  # solo tokens reales
    vc_loss, _, _ = regularizer(pred_flat)

    loss = pred_loss + vc_loss
    return {'loss': loss, 'pred_loss': pred_loss, 'vc_loss': vc_loss}


def save_best(epoch, val_loss):
    path = Path(CFG.logging.exp_dir) / 'best.pt'
    torch.save({
        'epoch':     epoch,
        'predictor': predictor.state_dict(),
        'projector': projector.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'val_loss':  val_loss,
    }, path)
    print(f'  ✓ saved → {path}')

@torch.no_grad()
def validation_loop():
    predictor.eval(); projector.eval()
    totals = {}; n = 0
    for batch in val_loader:
        d = forward_step(batch)
        for k, v in d.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n += 1
    predictor.train(); projector.train()
    return {k: v/n for k, v in totals.items()}

# ========================== Training Loop ==========================

history = {'train_loss': [], 'train_pred': [], 'train_vc': [],
           'val_loss':   [], 'val_pred':   [], 'val_vc':   []}

best_val_loss = float('inf')
Path(CFG.logging.exp_dir).mkdir(parents=True, exist_ok=True)

# Resume desde best si existe
best_ckpt = Path(CFG.logging.exp_dir) / 'best.pt'
start_epoch = 1
if best_ckpt.exists():
    ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    predictor.load_state_dict(ckpt['predictor'])
    projector.load_state_dict(ckpt['projector'])
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
    predictor.train(); projector.train()
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

    history['train_loss'].append(tr['loss'].item() if torch.is_tensor(tr['loss']) else tr['loss'])
    history['train_pred'].append(tr['pred_loss'].item() if torch.is_tensor(tr['pred_loss']) else tr['pred_loss'])
    history['train_vc'].append(tr['vc_loss'].item() if torch.is_tensor(tr['vc_loss']) else tr['vc_loss'])
    history['val_loss'].append(vl['loss'])
    history['val_pred'].append(vl['pred_loss'])
    history['val_vc'].append(vl['vc_loss'])

    print(
        f'Epoch {epoch:02d}/{CFG.optim.epochs}  '
        f'train={tr["loss"]:.4f} (pred={tr["pred_loss"]:.4f} vc={tr["vc_loss"]:.4f})  '
        f'val={vl["loss"]:.4f}  '
        f'lr={optimizer.param_groups[0]["lr"]:.2e}'
    )

    if vl['loss'] < best_val_loss:
        best_val_loss = vl['loss']
        save_best(epoch, best_val_loss)
        print(f'  ★ new best val_loss={best_val_loss:.4f}')

print('\nStage 2 complete.')

# ── Plotting ──────────────────────────────────────────────────────────────────
epochs_range = list(range(1, len(history['train_loss']) + 1))
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(epochs_range, history['train_loss'], 'b-o', label='Train', markersize=4)
axes[0].plot(epochs_range, history['val_loss'],   'r-s', label='Val',   markersize=4)
axes[0].set_title('Total Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history['train_pred'], 'g-^', label='Train pred', markersize=4)
axes[1].plot(epochs_range, history['val_pred'],   'm-v', label='Val pred',   markersize=4)
axes[1].set_title('Prediction Loss (MSE)'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs_range, history['train_vc'], 'c-o', label='Train VC', markersize=4)
axes[2].plot(epochs_range, history['val_vc'],   'k-s', label='Val VC',   markersize=4)
axes[2].set_title('VC Regularization Loss'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = Path(CFG.logging.exp_dir) / 'training_curves_stage2.png'
plt.savefig(plot_path, dpi=150); plt.close(fig)
print(f'Plot saved → {plot_path}')
print(f'Best val_loss: {best_val_loss:.4f}')