
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import cfg
from data.data   import make_dataloaders
from losses   import InfoNCELoss, recall_at_k, mean_reciprocal_rank

from cog_arch.dm  import DialogueJEPAPredictor



# ── Cell 3: Training ──────────────────────────────────────────────────────────

optimizer = optim.AdamW(predictor.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
mse_loss  = nn.MSELoss()
save_path = f'{ckpt_dir}/jepa_predictor_best.pth'

def cosine_sim(a, b):
    """Mean cosine similarity between two (B, D) tensors."""
    a_n = a / (a.norm(dim=1, keepdim=True) + 1e-8)
    b_n = b / (b.norm(dim=1, keepdim=True) + 1e-8)
    return (a_n * b_n).sum(dim=1).mean().item()

best_val_loss = float('inf')

print("Training JEPA predictor...")
for epoch in range(30):
    # ── Train ──
    predictor.train()
    tr_mse, tr_cos, n = 0, 0, 0
    for ctx, tgt, mask, lens in train_loader:
        ctx, tgt, mask = ctx.to(device), tgt.to(device), mask.to(device)

        pred = predictor(ctx, padding_mask=mask)    # (B, 768)
        loss = mse_loss(pred, tgt)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()

        tr_mse += loss.item() * ctx.size(0)
        tr_cos += cosine_sim(pred.detach().cpu(), tgt.cpu()) * ctx.size(0)
        n      += ctx.size(0)

    # ── Validate ──
    predictor.eval()
    vl_mse, vl_cos, vn = 0, 0, 0
    with torch.no_grad():
        for ctx, tgt, mask, lens in valid_loader:
            ctx, tgt, mask = ctx.to(device), tgt.to(device), mask.to(device)
            pred    = predictor(ctx, padding_mask=mask)
            loss    = mse_loss(pred, tgt)
            vl_mse += loss.item() * ctx.size(0)
            vl_cos += cosine_sim(pred.cpu(), tgt.cpu()) * ctx.size(0)
            vn     += ctx.size(0)

    tr_mse /= n;  vl_mse /= vn
    tr_cos /= n;  vl_cos /= vn
    scheduler.step()

    if vl_mse < best_val_loss:
        best_val_loss = vl_mse
        torch.save({'epoch':            epoch + 1,
                    'model_state_dict': predictor.state_dict(),
                    'val_mse':          best_val_loss,
                    'val_cos':          vl_cos}, save_path)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d} | "
              f"Train MSE {tr_mse:.4f} cos {tr_cos:.4f} | "
              f"Valid MSE {vl_mse:.4f} cos {vl_cos:.4f}")

print(f"\nBest valid MSE: {best_val_loss:.4f}")