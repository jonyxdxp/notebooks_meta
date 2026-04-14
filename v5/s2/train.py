import fire
import copy
import sys
import os
import glob
import typing
from pathlib import Path
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import wandb  # Asumiendo que se usa wandb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path setup
sys.path.insert(0, '/content/notebooks_meta/v5/s2')


from s1.cog_arch import Encoder


from cog_arch.dm import DM, Projector
from losses import SquareLossSeq, VCLoss


from data.dataloader import get_jepa_dataloaders
from data.dataset import VOCAB_SIZE, tokenizer





# Configuración (asumiendo que viene de config.py)
import config
CFG = config.get_cfg()  # o como se obtenga la config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")






# Variables de estado
wandb_run = False  # Setear a True si se inicializa wandb
start_epoch = 0








# ========================== CELL 9: Dataloaders ==========================

train_loader, val_loader = get_jepa_dataloaders(
    cfg_obj=CFG,
    tokenizer=tokenizer,
)

print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")









# ========================== CELL 10: Models ==========================

# 1. Load frozen encoders from Stage 1
context_encoder = Encoder(
    vocab_size=VOCAB_SIZE,
    hidden_size=CFG.hidden_size,
    num_heads=CFG.num_heads,
    num_layers=CFG.num_layers,
    max_seq_len=CFG.max_seq_len,
).to(DEVICE)

target_encoder = copy.deepcopy(context_encoder).to(DEVICE)

# Load Stage 1 checkpoint
if hasattr(CFG, 's1_ckpt') and CFG.s1_ckpt:
    s1_ckpt = torch.load(CFG.s1_ckpt, map_location=DEVICE)
    context_encoder.load_state_dict(s1_ckpt['context_encoder'])
    target_encoder.load_state_dict(s1_ckpt['target_encoder'])
    print(f"Loaded Stage 1 encoders from {CFG.s1_ckpt} (epoch {s1_ckpt.get('epoch', 'unknown')})")
else:
    logger.warning("No Stage 1 checkpoint provided. Starting from scratch.")

# Freeze both encoders — only predictor trains in Stage 2
for enc in (context_encoder, target_encoder):
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False

print("Encoders frozen.")








# 2. Predictor (Dynamic Model)
predictor = DM(
    hidden_size=CFG.pred_hidden_size,
    num_heads=CFG.pred_num_heads,
    num_layers=CFG.pred_num_layers,
    action_dim=CFG.action_dim,
    max_seq_len=CFG.max_seq_len,
).to(DEVICE)

print(f"Predictor params: {sum(p.numel() for p in predictor.parameters()):,}")

# 3. Projector (moved here - consistent placement)
# Asumiendo que CFG tiene cfg.model.dstc o similar
dstc = getattr(CFG.model, 'dstc', 256) if hasattr(CFG, 'model') else 256
projector = Projector(f"{dstc}-{dstc*4}-{dstc*4}").to(DEVICE)









# ========================== CELL 11: Loss / Optimizer ==========================

# Instanciar losses una sola vez (no en cada batch)
ploss = SquareLossSeq()
regularizer = VCLoss(
    std_coeff=CFG.loss.std_coeff, 
    cov_coeff=CFG.loss.cov_coeff,
    proj=None  # Ya aplicamos projector manualmente
)

# CORRECCIÓN CRÍTICA: Optimizar predictor + projector, no los encoders congelados
trainable_params = list(predictor.parameters()) + list(projector.parameters())

optimizer = AdamW(
    trainable_params,  # <-- CAMBIO CRÍTICO
    lr=CFG.lr,
    weight_decay=CFG.weight_decay,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=CFG.optim.epochs * 2,  # decays over 2x the actual run
    eta_min=CFG.lr * 0.3,
)












# ========================== CELL 12: Helpers ==========================

def ema_update(ctx_enc, tgt_enc, decay=0.996):
    """
    Exponential moving average: tgt ← decay*tgt + (1-decay)*ctx
    Solo tiene sentido si ctx_enc está entrenándose (Stage 1).
    En Stage 2, si ctx_enc está congelado, esto es opcional/innecesario.
    """
    with torch.no_grad():
        for p_c, p_t in zip(ctx_enc.parameters(), tgt_enc.parameters()):
            p_t.data.mul_(decay).add_(p_c.data, alpha=1.0 - decay)

def masked_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool hidden states at masked positions per batch item.
    Args:
        hidden: (B, L, D) encoder output
        mask: (B, L) bool — True at target positions (o 1s para posiciones válidas)
    Returns:
        pooled: (B, D)
    """
    mask_f = mask.unsqueeze(-1).float()  # (B, L, 1)
    summed = (hidden * mask_f).sum(dim=1)  # (B, D)
    count = mask_f.sum(dim=1).clamp(min=1)  # (B, 1)
    return summed / count

def unpack(batch):
    """Extrae tensores del batch y los mueve a DEVICE."""
    return (
        batch['context_input_ids'].to(DEVICE),
        batch['context_attention_mask'].to(DEVICE),
        batch['target_input_ids'].to(DEVICE),
        batch['target_attention_mask'].to(DEVICE),
        batch['target_mask'].to(DEVICE),  # Máscara de posiciones target
    )

def save_checkpoint(path, model, optimizer, epoch, step, **kwargs):
    """Guarda checkpoint del modelo."""
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **kwargs
    }, path)
    logger.info(f"Checkpoint saved to {path}")

def validation_loop(val_loader, ctx_enc, tgt_enc, predictor, projector, cfg, device):
    """
    Placeholder para validación. Implementa según tus necesidades.
    """
    predictor.eval()
    projector.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            ctx_ids, ctx_mask, tgt_ids, tgt_mask, tgt_pos_mask = unpack(batch)
            
            # Forward passes
            ctx_h = ctx_enc(ctx_ids, ctx_mask)
            tgt_h = tgt_enc(tgt_ids, tgt_mask)
            
            # Pooling
            tgt_repr = masked_pool(tgt_h, tgt_pos_mask)
            ctx_repr = masked_pool(ctx_h, ctx_mask)
            
            # Predictor
            pred = predictor(ctx_repr)
            
            # Project
            pred_proj = projector(pred)
            tgt_proj = projector(tgt_repr)
            
            # Loss
            loss = ploss(pred_proj, tgt_proj)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    
    predictor.train()
    projector.train()
    
    return {"val/loss": avg_loss}

def log_epoch(epoch, metrics, total_epochs):
    """Log simple de métricas."""
    logger.info(f"Epoch [{epoch}/{total_epochs}] - " + 
                " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))


def encode_turn(input_ids, attention_mask):
    """
    Encode a turn with the frozen encoder → mean-pool over real tokens → (B, D).
    No masking needed here — pool all non-padding positions.
    """
    with torch.no_grad():
        hidden = context_encoder(input_ids, attention_mask=attention_mask)
        if isinstance(hidden, tuple):
            hidden = hidden[0]                        # (B, L, D)
    # mean pool over non-padding tokens
    mask_f = attention_mask.unsqueeze(-1).float()     # (B, L, 1)
    pooled = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
    return pooled










# ========================== Training Loop ==========================

logger.info(f"Starting training for {CFG.optim.epochs} epochs...")
global_step = 0

# Encoders en eval mode permanentemente (frozen)
context_encoder.eval()
target_encoder.eval()

for epoch in range(start_epoch, CFG.optim.epochs):
    predictor.train()
    projector.train()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        (
            ctx_ids,
            ctx_mask,
            tgt_ids,
            tgt_mask,
            tgt_pos_mask,
        ) = unpack(batch)

        optimizer.zero_grad()

        # Encode context (no grad necesario pero lo dejamos por claridad)
        with torch.no_grad():
            ctx_h = context_encoder(ctx_ids, ctx_mask)  # (B, L, D)

        # Encode target (frozen)
        with torch.no_grad():
            tgt_h = target_encoder(tgt_ids, tgt_mask)  # (B, L, D)

        # Pool target representations (solo posiciones enmascaradas/target)
        tgt_repr = masked_pool(tgt_h, tgt_pos_mask)  # (B, D)

        # Pool context representations (usando attention mask para pooling válido)
        # CORRECCIÓN: ctx_mask es attention_mask (1=válido, 0=padding)
        ctx_repr = masked_pool(ctx_h, ctx_mask.bool() if ctx_mask.dtype == torch.long else ctx_mask)

        # Predictor forward
        pred = predictor(ctx_repr)  # (B, D) o (B, L, D) dependiendo de DM

        # Projector
        pred_proj = projector(pred)
        tgt_proj = projector(tgt_repr)

        # Losses (reusando instancias definidas antes)
        pred_loss = ploss(pred_proj, tgt_proj)
        vc_loss = regularizer(pred_proj)  # VCLoss puede aplicarse a pred o al proyectado
        
        total_loss = pred_loss + vc_loss

        total_loss.backward()
        optimizer.step()

        # NOTA: EMA update en Stage 2 es opcional si encoders están congelados
        # Si quieres mantener EMA por consistencia con Stage 1, descomenta:
        # ema_update(context_encoder, target_encoder, decay=CFG.ema_decay)

        # Logging
        pbar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "pred": f"{pred_loss.item():.4f}",
            "vc": f"{vc_loss.item():.4f}",
        })

        global_step += 1

    # Scheduler step (opcional, depende de tu estrategia)
    scheduler.step()

    # Validation
    if epoch % CFG.logging.log_every == 0:
        val_metrics = validation_loop(
            val_loader,
            context_encoder,
            target_encoder,
            predictor,
            projector,
            CFG,
            DEVICE
        )

        train_metrics = {
            "epoch": epoch,
            "train/loss": total_loss.item(),
            "train/pred_loss": pred_loss.item(),
            "train/vc_loss": vc_loss.item(),
        }

        all_metrics = {**train_metrics, **val_metrics}

        if wandb_run:
            wandb.log(all_metrics, step=global_step)

        log_epoch(
            epoch,
            {
                "loss": total_loss.item(),
                "pred": pred_loss.item(),
                "vc": vc_loss.item(),
                "val": val_metrics.get("val/loss", 0),
            },
            total_epochs=CFG.optim.epochs,
        )

    # Checkpoint
    save_checkpoint(
        exp_dir / "latest.pth.tar",
        model=predictor,
        optimizer=optimizer,
        epoch=epoch,
        step=global_step,
    )

    if epoch % CFG.logging.save_every == 0 and epoch > 0:
        save_checkpoint(
            exp_dir / f"epoch_{epoch}.pth.tar",
            model=predictor,
            optimizer=optimizer,
            epoch=epoch,
            step=global_step,
        )

if wandb_run:
    wandb.finish()

logger.info("Training complete!")