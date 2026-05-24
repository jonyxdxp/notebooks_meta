
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config        import cfg
from data.data     import make_dataloaders
from cog_arch.dm   import DialogueJEPAPredictor
from losses        import InfoNCELoss, recall_at_k, mean_reciprocal_rank
from evaluate.eval import run_eval


def cosine_sim(a, b):
    a_n = a / (a.norm(dim=1, keepdim=True) + 1e-8)
    b_n = b / (b.norm(dim=1, keepdim=True) + 1e-8)
    return (a_n * b_n).sum(dim=1).mean().item()


def train(predictor, train_loader, valid_loader):
    optimizer = AdamW(predictor.parameters(),
                      lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.MSELoss()
    save_path = os.path.join(cfg.ckpt_dir, "jepa_predictor_best.pth")

    best_val  = float("inf")

    print("Training JEPA predictor...")
    for epoch in range(cfg.epochs):
        # ── Train ──
        predictor.train()
        tr_mse, tr_cos, n = 0, 0, 0
        for ctx, tgt, mask, lens in train_loader:
            ctx, tgt, mask = (ctx.to(cfg.device),
                               tgt.to(cfg.device),
                               mask.to(cfg.device))
            pred = predictor(ctx, padding_mask=mask)
            loss = criterion(pred, tgt)
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
                ctx, tgt, mask = (ctx.to(cfg.device),
                                   tgt.to(cfg.device),
                                   mask.to(cfg.device))
                pred    = predictor(ctx, padding_mask=mask)
                loss    = criterion(pred, tgt)
                vl_mse += loss.item() * ctx.size(0)
                vl_cos += cosine_sim(pred.cpu(), tgt.cpu()) * ctx.size(0)
                vn     += ctx.size(0)

        tr_mse /= n;  vl_mse /= vn
        tr_cos /= n;  vl_cos /= vn
        scheduler.step()

        if vl_mse < best_val:
            best_val = vl_mse
            torch.save({"epoch": epoch + 1,
                        "model_state_dict": predictor.state_dict(),
                        "val_mse": best_val,
                        "val_cos": vl_cos}, save_path)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d} | "
                  f"Train MSE {tr_mse:.4f} cos {tr_cos:.4f} | "
                  f"Valid MSE {vl_mse:.4f} cos {vl_cos:.4f}")

    print(f"\nBest valid MSE: {best_val:.4f}")
    print(f"Saved → {save_path}")
    return save_path


if __name__ == "__main__":
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    train_loader, valid_loader, valid_ds = make_dataloaders()

    predictor = DialogueJEPAPredictor().to(cfg.device)
    n_params  = sum(p.numel() for p in predictor.parameters())
    print(f"JEPA predictor: {n_params:,} parameters")

    save_path = train(predictor, train_loader, valid_loader)

    # Load best and evaluate
    ckpt = torch.load(save_path, map_location=cfg.device, weights_only=False)
    predictor.load_state_dict(ckpt["model_state_dict"])
    run_eval(predictor, valid_ds)
