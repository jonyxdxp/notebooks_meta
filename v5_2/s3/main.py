"""
main.py — s3 training: GPT-2 prefix decoder on top of frozen s2.

Pipeline per batch:
  1. Forward frozen s2  →  predicted embedding ẑ  (B, 768)
  2. Forward S3Decoder  →  CE loss over target tokens
  3. Backprop only through S3Decoder (prefix_proj + GPT-2)

Eval every cfg.eval_every epochs:
  - Perplexity on val set
  - BLEU-1/2/4 and Distinct-1/2 on generated samples
  - Print a few example (history → generated) pairs
"""

import os
import sys
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import GPT2Tokenizer

# ── Path setup ────────────────────────────────────────────────────────────────
# Make both s3 (this file's dir) and s2 importable
_s3_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _s3_dir)

from config import cfg

# Import s2 modules
sys.path.insert(0, cfg.s2_module_path)
from s2.cog_arch.dm import DialogNextTurnPredictor
from config import Config as S2Config

# s3 modules
from data.data         import make_dataloaders
from cog_arch.decoder  import S3Decoder
from losses            import perplexity_from_loss, compute_generation_metrics


# ── Seed ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── LR schedule ───────────────────────────────────────────────────────────────

def get_scheduler(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ── Load frozen s2 ────────────────────────────────────────────────────────────

def load_s2(cfg, device) -> DialogNextTurnPredictor:
    s2_cfg = S2Config()
    # propagate shared paths
    s2_cfg.encoder_model  = cfg.encoder_model
    s2_cfg.cache_dir      = cfg.cache_dir
    s2_cfg.encoder_dim    = cfg.encoder_dim
    s2_cfg.ctx_n_heads    = cfg.ctx_n_heads
    s2_cfg.ctx_n_layers   = cfg.ctx_n_layers
    s2_cfg.ctx_ffn_dim    = cfg.ctx_ffn_dim
    s2_cfg.ctx_dropout    = cfg.ctx_dropout
    s2_cfg.max_history    = cfg.max_history
    s2_cfg.proj_hidden_dim = cfg.proj_hidden_dim
    s2_cfg.freeze_encoder  = True

    model = DialogNextTurnPredictor(s2_cfg).to(device)
    ckpt  = torch.load(cfg.s2_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[s3] s2 loaded & frozen from {cfg.s2_checkpoint} "
          f"(epoch {ckpt['epoch']}, R@1={ckpt['metrics'].get('R@1', '?'):.4f})")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    s2_model:  DialogNextTurnPredictor,
    s3_model:  S3Decoder,
    loader,
    gpt2_tok,
    device,
    cfg,
    n_show:    int = 4,
) -> dict:
    s3_model.eval()
    total_loss = 0.0
    hypotheses, references = [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast(enabled=(device.type == "cuda")):
            # get predicted embedding from frozen s2
            z_hat, _ = s2_model(batch)                  # (B, D)
            # decode loss
            _, loss = s3_model(
                z_hat,
                batch["dec_input_ids"],
                batch["dec_labels"],
                batch["dec_attn_mask"],
            )
        total_loss += loss.item()

        # generate a few samples for BLEU / Distinct
        if len(hypotheses) < 512:
            generated = s3_model.generate(
                z_hat[:4].float(),      # generate for first 4 in batch
                gpt2_tok,
                max_new_tokens = cfg.gen_max_new_tokens,
                temperature    = cfg.gen_temperature,
                top_p          = cfg.gen_top_p,
                do_sample      = cfg.gen_do_sample,
            )
            # decode reference targets
            ref_ids = batch["dec_labels"][:4]
            for ref_row in ref_ids:
                valid = ref_row[ref_row != -100]
                references.append(gpt2_tok.decode(valid, skip_special_tokens=True))
            hypotheses.extend(generated)

    avg_loss = total_loss / len(loader)
    metrics  = {"loss": avg_loss, "PPL": perplexity_from_loss(avg_loss)}
    if hypotheses:
        metrics.update(compute_generation_metrics(hypotheses, references))

    # print a few examples
    print(f"\n  {'─'*60}")
    print(f"  Sample generations (val):")
    for h, r in zip(hypotheses[:n_show], references[:n_show]):
        print(f"  REF : {r[:100]}")
        print(f"  GEN : {h[:100]}")
        print()
    print(f"  {'─'*60}")

    s3_model.train()
    return metrics


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(s3_model, optimizer, scheduler, epoch, metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":     epoch,
        "model":     s3_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics":   metrics,
    }, path)
    print(f"  [ckpt] saved → {path}")


def load_checkpoint(path, s3_model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    s3_model.load_state_dict(ckpt["model"])
    if optimizer:  optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler:  scheduler.load_state_dict(ckpt["scheduler"])
    print(f"[ckpt] resumed from epoch {ckpt['epoch']}: {path}")
    return ckpt["epoch"], ckpt.get("metrics", {})


# ── Main ──────────────────────────────────────────────────────────────────────

def train(cfg=cfg):
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[s3] device = {device}")

    # Data
    train_loader, val_loader, test_loader = make_dataloaders(cfg)

    # GPT-2 tokenizer (for generation / decoding references)
    gpt2_tok = GPT2Tokenizer.from_pretrained(cfg.decoder_model, cache_dir=cfg.cache_dir)
    gpt2_tok.pad_token    = gpt2_tok.eos_token
    gpt2_tok.pad_token_id = gpt2_tok.eos_token_id

    # Frozen s2
    s2_model = load_s2(cfg, device)

    # s3 decoder
    s3_model = S3Decoder(cfg).to(device)
    n_params = sum(p.numel() for p in s3_model.parameters() if p.requires_grad)
    print(f"[s3] trainable parameters: {n_params:,}")

    # Optimiser — only s3 parameters
    optimizer = AdamW(s3_model.parameters(), lr=cfg.lr, weight_decay=0.01)
    total_steps = len(train_loader) * cfg.num_epochs
    scheduler   = get_scheduler(optimizer, cfg.warmup_steps, total_steps)
    scaler      = GradScaler(enabled=(cfg.fp16 and device.type == "cuda"))

    # Resume if checkpoint exists
    best_ckpt   = Path(cfg.output_dir) / "best.pt"
    latest_ckpt = Path(cfg.output_dir) / "latest.pt"
    start_epoch = 0
    best_ppl    = float("inf")

    if latest_ckpt.exists():
        start_epoch, prev = load_checkpoint(str(latest_ckpt), s3_model, optimizer, scheduler)
        best_ppl    = prev.get("PPL", float("inf"))
        start_epoch += 1

    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, cfg.num_epochs):
        s3_model.train()
        epoch_loss  = 0.0
        epoch_start = time.time()

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(enabled=(cfg.fp16 and device.type == "cuda")):
                # 1. Get ẑ from frozen s2 (no grad)
                with torch.no_grad():
                    z_hat, _ = s2_model(batch)           # (B, D)

                # 2. Decode and compute CE loss
                _, loss = s3_model(
                    z_hat,
                    batch["dec_input_ids"],
                    batch["dec_labels"],
                    batch["dec_attn_mask"],
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(s3_model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            if global_step % cfg.log_steps == 0:
                print(f"  ep {epoch+1:>3} | step {global_step:>6} "
                      f"| loss {loss.item():.4f} "
                      f"| ppl {perplexity_from_loss(loss.item()):.2f} "
                      f"| lr {optimizer.param_groups[0]['lr']:.2e}")

        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - epoch_start
        print(f"\n[epoch {epoch+1}/{cfg.num_epochs}] "
              f"avg_loss={avg_loss:.4f}  "
              f"ppl={perplexity_from_loss(avg_loss):.2f}  "
              f"time={elapsed:.1f}s")

        # Eval
        if (epoch + 1) % cfg.eval_every == 0 and val_loader is not None:
            metrics = evaluate(s2_model, s3_model, val_loader, gpt2_tok, device, cfg)
            print(f"  [val] loss={metrics['loss']:.4f} | "
                  f"PPL={metrics['PPL']:.2f} | "
                  f"BLEU-1={metrics.get('BLEU-1', 0):.4f} | "
                  f"BLEU-4={metrics.get('BLEU-4 (BP)', 0):.4f} | "
                  f"D-1={metrics.get('Distinct-1', 0):.4f} | "
                  f"D-2={metrics.get('Distinct-2', 0):.4f}")

            if metrics["PPL"] < best_ppl:
                best_ppl = metrics["PPL"]
                save_checkpoint(s3_model, optimizer, scheduler, epoch,
                                metrics, str(best_ckpt))
                print(f"  [val] ★ new best PPL = {best_ppl:.2f}")

        if (epoch + 1) % cfg.save_every == 0:
            save_checkpoint(s3_model, optimizer, scheduler, epoch,
                            {"PPL": best_ppl}, str(latest_ckpt))

    # Final test eval
    if test_loader is not None and best_ckpt.exists():
        print("\n[test] loading best checkpoint...")
        load_checkpoint(str(best_ckpt), s3_model)
        test_metrics = evaluate(s2_model, s3_model, test_loader, gpt2_tok, device, cfg)
        print(f"[test] PPL={test_metrics['PPL']:.2f} | "
              f"BLEU-1={test_metrics.get('BLEU-1', 0):.4f} | "
              f"BLEU-4={test_metrics.get('BLEU-4 (BP)', 0):.4f} | "
              f"D-1={test_metrics.get('Distinct-1', 0):.4f} | "
              f"D-2={test_metrics.get('Distinct-2', 0):.4f}")

    print("\n[s3] done.")
    return s3_model


if __name__ == "__main__":
    train(cfg)