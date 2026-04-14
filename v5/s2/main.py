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
    """Mean-pooling usando attention mask (1=real token, 0=padding)"""
    mask_f = mask.unsqueeze(-1).float()  # (B, L, 1)
    summed = (hidden * mask_f).sum(dim=1)  # (B, D)
    count = mask_f.sum(dim=1).clamp(min=1)  # (B, 1)
    return summed / count

def unpack(batch):
    """
    Extrae tensores del batch de pares (Stage 2).
    Estructura: turn_a (context), turn_b (target)
    """
    return (
        batch['input_ids_a'].to(DEVICE),      # Contexto: turno t
        batch['attention_mask_a'].to(DEVICE), # Mask turno t
        batch['input_ids_b'].to(DEVICE),      # Target: turno t+1
        batch['attention_mask_b'].to(DEVICE), # Mask turno t+1
    )

def save_checkpoint(path, model, optimizer, epoch, step, **kwargs):
    """Guarda checkpoint del predictor"""
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
    """Loop de validación"""
    predictor.eval()
    projector.eval()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_vc_loss = 0.0
    n = 0
    
    for batch in val_loader:
        ctx_ids, ctx_mask, tgt_ids, tgt_mask = unpack(batch)
        
        # Encode (frozen)
        ctx_h = context_encoder(ctx_ids, ctx_mask)
        tgt_h = target_encoder(tgt_ids, tgt_mask)
        
        # Pool a representación fija (B, D)
        ctx_repr = masked_pool(ctx_h, ctx_mask.bool() if ctx_mask.dtype == torch.long else ctx_mask)
        tgt_repr = masked_pool(tgt_h, tgt_mask.bool() if tgt_mask.dtype == torch.long else tgt_mask)
        
        # Predictor: predice tgt_repr desde ctx_repr
        # Nota: DM espera (B, T, D), adaptamos dims si es necesario
        if ctx_repr.dim() == 2:
            ctx_repr = ctx_repr.unsqueeze(1)  # (B, 1, D) -> tratar como seq len 1
        
        pred = predictor(ctx_repr)  # (B, T, D) o (B, D)
        if pred.dim() == 3:
            pred = pred.squeeze(1)  # Volver a (B, D)
        
        # Proyectar
        pred_proj = projector(pred)
        tgt_proj = projector(tgt_repr)
        
        # Losses
        pred_loss = ploss(pred_proj, tgt_proj)
        vc_loss = regularizer(pred_proj)
        loss = pred_loss + vc_loss
        
        total_loss += loss.item()
        total_pred_loss += pred_loss.item()
        total_vc_loss += vc_loss.item()
        n += 1
    
    predictor.train()
    projector.train()
    
    return {
        'loss': total_loss / max(n, 1),
        'pred_loss': total_pred_loss / max(n, 1),
        'vc_loss': total_vc_loss / max(n, 1),
    }

# ========================== Training Loop ==========================

logger.info(f"Starting Stage 2: {CFG.optim.epochs} epochs")
global_step = 0
best_val_loss = float('inf')

for epoch in range(start_epoch, CFG.optim.epochs):
    predictor.train()
    projector.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    epoch_loss = 0.0
    epoch_pred = 0.0
    epoch_vc = 0.0
    n_batches = 0
    
    for batch in pbar:
        ctx_ids, ctx_mask, tgt_ids, tgt_mask = unpack(batch)
        
        optimizer.zero_grad()

        # Encode (frozen)
        with torch.no_grad():
            ctx_h = context_encoder(ctx_ids, ctx_mask)
            tgt_h = target_encoder(tgt_ids, tgt_mask)

        # Pool a (B, D)
        ctx_repr = masked_pool(ctx_h, ctx_mask.bool() if ctx_mask.dtype == torch.long else ctx_mask)
        tgt_repr = masked_pool(tgt_h, tgt_mask.bool() if tgt_mask.dtype == torch.long else tgt_mask)

        # Predictor: adaptar dims si es necesario (DM espera 3D)
        if ctx_repr.dim() == 2:
            ctx_seq = ctx_repr.unsqueeze(1)  # (B, 1, D)
        else:
            ctx_seq = ctx_repr
        
        pred = predictor(ctx_seq)
        if pred.dim() == 3:
            pred = pred.squeeze(1)  # (B, D)

        # Proyectar
        pred_proj = projector(pred)
        tgt_proj = projector(tgt_repr)

        # Losses
        pred_loss = ploss(pred_proj, tgt_proj)
        vc_loss = regularizer(pred_proj)
        loss = pred_loss + vc_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        # Logging
        epoch_loss += loss.item()
        epoch_pred += pred_loss.item()
        epoch_vc += vc_loss.item()
        n_batches += 1
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "pred": f"{pred_loss.item():.4f}",
            "vc": f"{vc_loss.item():.4f}",
        })
        global_step += 1

    # Epoch stats
    avg_loss = epoch_loss / n_batches
    avg_pred = epoch_pred / n_batches
    avg_vc = epoch_vc / n_batches
    
    scheduler.step()
    
    # Validation
    if epoch % CFG.logging.log_every == 0:
        val_metrics = validation_loop()
        logger.info(
            f"Epoch {epoch}/{CFG.optim.epochs} | "
            f"Train Loss: {avg_loss:.4f} (P:{avg_pred:.4f} VC:{avg_vc:.4f}) | "
            f"Val Loss: {val_metrics['loss']:.4f}"
        )
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                exp_dir / "best.pth.tar",
                model=predictor,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                val_loss=best_val_loss
            )
            logger.info(f"★ New best model: {best_val_loss:.4f}")

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