"""
train.py — Training loop for the dialog next-turn predictor.

Features:
  - Mixed precision (fp16) via torch.cuda.amp
  - Linear warmup + cosine decay LR schedule
  - Separate LR for encoder vs. the rest (when freeze_encoder=False)
  - Full-pool Recall@K and MRR evaluated at the end of every eval epoch
  - Best checkpoint saved by R@1 on the val set
  - Clean per-step + per-epoch logging

Usage (Colab / terminal):
  from config import cfg
  from train  import train
  train(cfg)
"""

import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import cfg
from data   import make_dataloaders
from losses   import InfoNCELoss, recall_at_k, mean_reciprocal_rank
from model  import DialogNextTurnPredictor


# ── Reproducibility ─────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── LR Schedule: linear warmup + cosine decay ──────────────────────────────────

def get_scheduler(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ── Evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    """
    Runs the model over the full loader and returns:
      - avg InfoNCE loss
      - R@1, R@5, R@10 over the full candidate pool
      - MRR
    """
    model.eval()
    total_loss   = 0.0
    all_pred     = []
    all_target   = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast(enabled=(device.type == "cuda")):
            pred_emb, target_emb = model(batch)
            loss = criterion(pred_emb, target_emb)

        total_loss   += loss.item()
        all_pred.append(pred_emb.float().cpu())
        all_target.append(target_emb.float().cpu())

    # pool all embeddings for full-pool retrieval metrics
    all_pred   = torch.cat(all_pred,   dim=0)  # (N, D)
    all_target = torch.cat(all_target, dim=0)  # (N, D)

    metrics = recall_at_k(all_pred, all_target, ks=(1, 5, 10))
    metrics["MRR"]  = mean_reciprocal_rank(all_pred, all_target)
    metrics["loss"] = total_loss / len(loader)

    model.train()
    return metrics


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":              epoch,
        "model_state_dict":   model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics":            metrics,
    }, path)
    print(f"  [ckpt] saved → {path}")


def load_checkpoint(path: str, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"[ckpt] loaded epoch {ckpt['epoch']} from {path}")
    return ckpt["epoch"], ckpt.get("metrics", {})


# ── Main Training Function ─────────────────────────────────────────────────────

def train(cfg=cfg):
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[train] device = {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = make_dataloaders(cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DialogNextTurnPredictor(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable parameters: {n_params:,}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Separate param groups: encoder (lower lr) vs. context transformer + proj
    encoder_params = list(model.encoder.parameters())
    other_params   = (
        list(model.context_transformer.parameters()) +
        list(model.projection.parameters())
    )

    if cfg.freeze_encoder:
        param_groups = [{"params": other_params, "lr": cfg.lr}]
    else:
        param_groups = [
            {"params": encoder_params, "lr": cfg.encoder_lr},
            {"params": other_params,   "lr": cfg.lr},
        ]

    optimizer = AdamW(param_groups, weight_decay=0.01)

    total_steps = len(train_loader) * cfg.num_epochs
    scheduler   = get_scheduler(optimizer, cfg.warmup_steps, total_steps)

    criterion = InfoNCELoss(temperature=cfg.temperature).to(device)
    scaler    = GradScaler(enabled=(cfg.fp16 and device.type == "cuda"))

    # ── Resume from checkpoint if available ───────────────────────────────────
    best_ckpt   = Path(cfg.output_dir) / "best.pt"
    latest_ckpt = Path(cfg.output_dir) / "latest.pt"
    start_epoch = 0
    best_r1     = 0.0

    if latest_ckpt.exists():
        start_epoch, prev_metrics = load_checkpoint(
            str(latest_ckpt), model, optimizer, scheduler
        )
        best_r1     = prev_metrics.get("R@1", 0.0)
        start_epoch += 1

    # ── Training Loop ─────────────────────────────────────────────────────────
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        epoch_loss  = 0.0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(enabled=(cfg.fp16 and device.type == "cuda")):
                pred_emb, target_emb = model(batch)
                loss = criterion(pred_emb, target_emb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            if global_step % cfg.log_steps == 0:
                lr_main = optimizer.param_groups[-1]["lr"]
                print(
                    f"  ep {epoch+1:>3} | step {global_step:>6} "
                    f"| loss {loss.item():.4f} | lr {lr_main:.2e}"
                )

        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - epoch_start
        print(f"\n[epoch {epoch+1}/{cfg.num_epochs}] "
              f"avg_loss={avg_loss:.4f}  time={elapsed:.1f}s")

        # ── Evaluation ────────────────────────────────────────────────────────
        if (epoch + 1) % cfg.eval_every == 0 and val_loader is not None:
            metrics = evaluate(model, val_loader, criterion, device)
            print(
                f"  [val] loss={metrics['loss']:.4f} | "
                f"R@1={metrics['R@1']:.4f} | "
                f"R@5={metrics['R@5']:.4f} | "
                f"R@10={metrics['R@10']:.4f} | "
                f"MRR={metrics['MRR']:.4f}"
            )

            if metrics["R@1"] > best_r1:
                best_r1 = metrics["R@1"]
                save_checkpoint(model, optimizer, scheduler, epoch,
                                metrics, str(best_ckpt))
                print(f"  [val] ★ new best R@1 = {best_r1:.4f}")

        # ── Periodic save ─────────────────────────────────────────────────────
        if (epoch + 1) % cfg.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch,
                            {"R@1": best_r1}, str(latest_ckpt))

    # ── Final test evaluation ──────────────────────────────────────────────────
    if test_loader is not None and best_ckpt.exists():
        print("\n[test] loading best checkpoint for final evaluation...")
        load_checkpoint(str(best_ckpt), model)
        test_metrics = evaluate(model, test_loader, criterion, device)
        print(
            f"[test] loss={test_metrics['loss']:.4f} | "
            f"R@1={test_metrics['R@1']:.4f} | "
            f"R@5={test_metrics['R@5']:.4f} | "
            f"R@10={test_metrics['R@10']:.4f} | "
            f"MRR={test_metrics['MRR']:.4f}"
        )

    print("\n[train] done.")
    return model


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(cfg)