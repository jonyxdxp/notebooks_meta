#!/usr/bin/env python3
"""
Stage 2: Entrenamiento del Predictor JEPA (Turn Pair Prediction)
Predice el siguiente turno (t+1) desde el turno actual (t)
"""

import sys
import os
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar config de Stage 2 (estructura anidada)
sys.path.insert(0, '/content/notebooks_meta/v5/s2')
from config import CFG, DEVICE

# Imports del proyecto
sys.path.insert(0, '/content/notebooks_meta')

from v5.s1.cog_arch.encoder import Encoder
from v5.s2.cog_arch.dm import DM, Projector
from v5.s2.losses import SquareLossSeq, VCLoss
from v5.s1.data.dataloader import get_jepa_dataloaders
from v5.s1.data.dataset import VOCAB_SIZE, tokenizer

import copy

# Variables globales
wandb_run = CFG.logging.log_wandb if hasattr(CFG.logging, 'log_wandb') else False
start_epoch = 0
exp_dir = Path(CFG.logging.exp_dir)
exp_dir.mkdir(parents=True, exist_ok=True)

# ========================== Dataloaders ==========================

train_loader, val_loader = get_jepa_dataloaders(
    cfg_obj=CFG,
    tokenizer=tokenizer,
)

print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

# ========================== Models ==========================

# 1. Encoders congelados desde Stage 1
context_encoder = Encoder(
    vocab_size=CFG.model.vocab_size,
    hidden_size=CFG.model.hidden_size,
    num_heads=CFG.model.num_heads,
    num_layers=CFG.model.num_layers,
    max_seq_len=CFG.model.max_seq_len,
).to(DEVICE)

target_encoder = copy.deepcopy(context_encoder).to(DEVICE)

# Cargar checkpoint Stage 1
if os.path.exists(CFG.training.s1_ckpt):
    s1_ckpt = torch.load(CFG.training.s1_ckpt, map_location=DEVICE)
    context_encoder.load_state_dict(s1_ckpt['context_encoder'])
    target_encoder.load_state_dict(s1_ckpt['target_encoder'])
    print(f"✓ Loaded Stage 1 from {CFG.training.s1_ckpt} (epoch {s1_ckpt.get('epoch', 'unknown')})")
else:
    raise FileNotFoundError(f"Stage 1 checkpoint not found: {CFG.training.s1_ckpt}")

# Freeze encoders
for enc in (context_encoder, target_encoder):
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
print("Encoders frozen.")

# 2. Predictor (DM adaptado para secuencias)
# NOTA: DM espera: num_frames, depth, heads, mlp_dim, input_dim, hidden_dim...
predictor = DM(
    num_frames=CFG.model.max_seq_len,
    depth=CFG.model.pred_num_layers,
    heads=CFG.model.pred_num_heads,
    mlp_dim=CFG.model.pred_hidden_size * 4,  # mlp_dim suele ser 4x hidden
    input_dim=CFG.model.dstc,  # Dimensión de entrada (la representación pooled del encoder)
    hidden_dim=CFG.model.pred_hidden_size,
    output_dim=CFG.model.dstc,  # Predice representación del siguiente turno
    dim_head=64,
    dropout=0.1,
    emb_dropout=0.1,
).to(DEVICE)

print(f"Predictor params: {sum(p.numel() for p in predictor.parameters()):,}")

# 3. Projector (VICReg/VC)
dstc = CFG.model.dstc
projector = Projector(f"{dstc}-{dstc*4}-{dstc*4}").to(DEVICE)
print(f"Projector: {dstc}-{dstc*4}-{dstc*4}")




# ========================== Loss / Optimizer ==========================

ploss     = torch.nn.MSELoss()
regularizer = VCLoss(std_coeff=CFG.loss.std_coeff, cov_coeff=CFG.loss.cov_coeff)

trainable_params = list(predictor.parameters()) + list(projector.parameters())
optimizer = AdamW(trainable_params, lr=CFG.optim.lr, weight_decay=CFG.optim.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CFG.optim.epochs, eta_min=CFG.optim.lr * 0.1)

# ========================== Helpers ==========================

def masked_pool(hidden, mask):
    mask_f = mask.unsqueeze(-1).float()
    return (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

def unpack(batch):
    return (
        batch['context_input_ids'].to(DEVICE),
        batch['context_attention_mask'].to(DEVICE),
        batch['target_input_ids'].to(DEVICE),
        batch['target_attention_mask'].to(DEVICE),
    )

def forward_step(batch):
    ctx_ids, ctx_mask, tgt_ids, tgt_mask = unpack(batch)

    with torch.no_grad():
        ctx_h = context_encoder(ctx_ids, attention_mask=ctx_mask)
        tgt_h = target_encoder(tgt_ids, attention_mask=tgt_mask)
        if isinstance(ctx_h, tuple): ctx_h = ctx_h[0]
        if isinstance(tgt_h, tuple): tgt_h = tgt_h[0]

    ctx_repr = masked_pool(ctx_h, ctx_mask)   # (B, D)
    tgt_repr = masked_pool(tgt_h, tgt_mask)   # (B, D)

    ctx_seq = ctx_repr.unsqueeze(1)            # (B, 1, D)
    c       = torch.zeros_like(ctx_seq)        # conditioning vacío
    pred    = predictor(ctx_seq, c).squeeze(1) # (B, D)

    pred_proj = projector(pred)
    tgt_proj  = projector(tgt_repr.detach())

    pred_loss          = ploss(pred_proj, tgt_proj)
    vc_loss, _, vc_dict = regularizer(pred_proj)
    loss               = pred_loss + vc_loss

    return {'loss': loss, 'pred_loss': pred_loss, 'vc_loss': vc_loss}

def save_best(epoch, val_loss):
    path = Path(CFG.logging.exp_dir) / 'best.pt'
    torch.save({
        'epoch':      epoch,
        'predictor':  predictor.state_dict(),
        'projector':  projector.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'scheduler':  scheduler.state_dict(),
        'val_loss':   val_loss,
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

best_val_loss = float('inf')
Path(CFG.logging.exp_dir).mkdir(parents=True, exist_ok=True)

for epoch in range(1, CFG.optim.epochs + 1):
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
    train_avg = {k: v/n for k, v in totals.items()}
    val_avg   = validation_loop()

    print(
        f'Epoch {epoch:02d}/{CFG.optim.epochs}  '
        f'train={train_avg["loss"]:.4f} (pred={train_avg["pred_loss"]:.4f} vc={train_avg["vc_loss"]:.4f})  '
        f'val={val_avg["loss"]:.4f}  '
        f'lr={optimizer.param_groups[0]["lr"]:.2e}'
    )

    if val_avg['loss'] < best_val_loss:
        best_val_loss = val_avg['loss']
        save_best(epoch, best_val_loss)
        print(f'  ★ new best val_loss={best_val_loss:.4f}')

print('\nStage 2 complete.')