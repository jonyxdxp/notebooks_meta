#!/usr/bin/env python3
"""
Stage 2: Entrenamiento del Predictor JEPA
Usa estructura anidada: CFG.model.hidden_size, CFG.data.batch_size, etc.
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

# Importar config ANTES que todo
sys.path.insert(0, '/content/notebooks_meta/v5/s2')
from config import CFG, DEVICE

# Imports del proyecto
sys.path.insert(0, '/content/notebooks_meta/v5/s1')
from cog_arch.encoder import Encoder
from cog_arch.dm import DM, Projector
from losses import SquareLossSeq, VCLoss
from data.dataloader import get_jepa_dataloaders
from data.dataset import VOCAB_SIZE, tokenizer

import copy







# Variables globales
wandb_run = CFG.logging.log_wandb if hasattr(CFG.logging, 'log_wandb') else False
start_epoch = 0
exp_dir = Path(CFG.logging.exp_dir)












# ========================== Dataloaders ==========================
# Nota: El dataloader espera atributos planos o anidados según tu implementación
# Si falla, ajusta aquí para pasar los valores correctos
train_loader, val_loader = get_jepa_dataloaders(
    cfg_obj=CFG,  # Si el dataloader espera CFG.max_seq_len, necesitarás adaptarlo
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
    print(f"✓ Loaded Stage 1 from {CFG.training.s1_ckpt}")
else:
    raise FileNotFoundError(f"Stage 1 checkpoint not found: {CFG.training.s1_ckpt}")

# Freeze
for enc in (context_encoder, target_encoder):
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
print("Encoders frozen.")






# 2. Predictor
predictor = DM(
    hidden_size=CFG.model.pred_hidden_size,
    num_heads=CFG.model.pred_num_heads,
    num_layers=CFG.model.pred_num_layers,
    action_dim=CFG.model.action_dim,
    max_seq_len=CFG.model.max_seq_len,
).to(DEVICE)

print(f"Predictor params: {sum(p.numel() for p in predictor.parameters()):,}")

# 3. Projector
dstc = CFG.model.dstc
projector = Projector(f"{dstc}-{dstc*4}-{dstc*4}").to(DEVICE)














# ========================== Loss / Optimizer ==========================

ploss = SquareLossSeq()
regularizer = VCLoss(
    std_coeff=CFG.loss.std_coeff,
    cov_coeff=CFG.loss.cov_coeff,
    proj=None
)

# Optimizer: solo predictor + projector
trainable_params = list(predictor.parameters()) + list(projector.parameters())
optimizer = AdamW(
    trainable_params,
    lr=CFG.optim.lr,
    weight_decay=CFG.optim.weight_decay,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=CFG.optim.epochs,
    eta_min=CFG.optim.lr * 0.1,
)
















# ========================== Helpers ==========================

def masked_pool(hidden, mask):
    mask_f = mask.unsqueeze(-1).float()
    summed = (hidden * mask_f).sum(dim=1)
    count = mask_f.sum(dim=1).clamp(min=1)
    return summed / count

def unpack(batch):
    return (
        batch['context_input_ids'].to(DEVICE),
        batch['context_attention_mask'].to(DEVICE),
        batch['target_input_ids'].to(DEVICE),
        batch['target_attention_mask'].to(DEVICE),
        batch['target_mask'].to(DEVICE),
    )

def save_checkpoint(path, model, optimizer, epoch, step, **kwargs):
    path = Path(path) if isinstance(path, str) else path
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **kwargs
    }, path)
    logger.info(f"Saved: {path}")

@torch.no_grad()
def validation_loop():
    predictor.eval()
    projector.eval()
    total_loss = 0.0
    n = 0
    
    for batch in val_loader:
        ctx_ids, ctx_mask, tgt_ids, tgt_mask, tgt_pos_mask = unpack(batch)
        
        ctx_h = context_encoder(ctx_ids, ctx_mask)
        tgt_h = target_encoder(tgt_ids, tgt_mask)
        
        tgt_repr = masked_pool(tgt_h, tgt_pos_mask)
        ctx_repr = masked_pool(ctx_h, ctx_mask)
        
        pred = predictor(ctx_repr)
        pred_proj = projector(pred)
        tgt_proj = projector(tgt_repr)
        
        loss = ploss(pred_proj, tgt_proj)
        total_loss += loss.item()
        n += 1
    
    predictor.train()
    projector.train()
    return total_loss / max(n, 1)













# ========================== Training Loop ==========================

logger.info(f"Starting Stage 2: {CFG.optim.epochs} epochs")
global_step = 0
best_val_loss = float('inf')

for epoch in range(start_epoch, CFG.optim.epochs):
    predictor.train()
    projector.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        ctx_ids, ctx_mask, tgt_ids, tgt_mask, tgt_pos_mask = unpack(batch)
        
        optimizer.zero_grad()

        with torch.no_grad():
            ctx_h = context_encoder(ctx_ids, ctx_mask)
            tgt_h = target_encoder(tgt_ids, tgt_mask)

        tgt_repr = masked_pool(tgt_h, tgt_pos_mask)
        ctx_repr = masked_pool(ctx_h, ctx_mask)

        pred = predictor(ctx_repr)
        pred_proj = projector(pred)
        tgt_proj = projector(tgt_repr)

        pred_loss = ploss(pred_proj, tgt_proj)
        vc_loss = regularizer(pred_proj)
        
        total_loss = pred_loss + vc_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        pbar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "pred": f"{pred_loss.item():.4f}",
            "vc": f"{vc_loss.item():.4f}",
        })
        global_step += 1

    scheduler.step()
    
    # Validation
    if epoch % CFG.logging.log_every == 0:
        val_loss = validation_loop()
        logger.info(f"Epoch {epoch}/{CFG.optim.epochs} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                exp_dir / "best.pth.tar",
                model=predictor,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                val_loss=val_loss
            )

    # Checkpoint periódico
    if epoch % CFG.logging.save_every == 0 and epoch > 0:
        save_checkpoint(
            exp_dir / f"epoch_{epoch:03d}.pth.tar",
            model=predictor,
            optimizer=optimizer,
            epoch=epoch,
            step=global_step,
        )

logger.info("Stage 2 complete!")