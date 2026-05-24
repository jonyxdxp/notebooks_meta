import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
import numpy as np
from tqdm.auto import tqdm
from config import cfg
from losses import mean_reciprocal_rank


def run_eval(predictor, valid_ds, n_samples=1000, seed=42):
    predictor.eval()
    random.seed(seed)

    indices = random.sample(range(len(valid_ds)),
                            min(n_samples, len(valid_ds)))
    pred_all, tgt_all, lens_all = [], [], []

    with torch.no_grad():
        for idx in tqdm(indices, desc="Encoding eval set"):
            ctx, tgt, n = valid_ds[idx]
            mask = (torch.arange(cfg.max_turns) >= n)
            pred = predictor(
                ctx.unsqueeze(0).to(cfg.device),
                padding_mask=mask.unsqueeze(0).to(cfg.device))
            pred_all.append(pred.squeeze(0).cpu())
            tgt_all.append(tgt)
            lens_all.append(n)

    pred_t = torch.stack(pred_all)   # (N, 768)
    tgt_t  = torch.stack(tgt_all)    # (N, 768)

    # Normalize
    pred_n = pred_t / (pred_t.norm(dim=1, keepdim=True) + 1e-8)
    tgt_n  = tgt_t  / (tgt_t.norm(dim=1,  keepdim=True) + 1e-8)

    # ── Recall@1 with K random negatives (1-of-K+1) ──────────────────────────
    print("\n=== JEPA Predictor Evaluation ===")
    for K in [9, 49, 99]:
        hits = 0
        for i in range(len(pred_n)):
            neg_idx    = random.sample(
                [j for j in range(len(tgt_n)) if j != i], K)
            candidates = [i] + neg_idx
            scores     = pred_n[i] @ tgt_n[candidates].T
            if scores.argmax().item() == 0:
                hits += 1
        r = hits / len(pred_n)
        print(f"  Recall@1 (1-of-{K+1:>3}): {r:.4f}  "
              f"(random={1/(K+1):.4f})  "
              f"gain={r - 1/(K+1):+.4f}")

    # ── MRR over full pool ────────────────────────────────────────────────────
    mrr = mean_reciprocal_rank(pred_t, tgt_t)
    print(f"  MRR (full pool):   {mrr:.4f}  (random≈{1/len(pred_t):.4f})")

    # ── Mean cosine similarity pred vs target ─────────────────────────────────
    cos = (pred_n * tgt_n).sum(dim=1).mean().item()
    print(f"  Mean cos sim:      {cos:.4f}")

    # ── Breakdown by context length (1-of-10) ─────────────────────────────────
    print("\nRecall@1 (1-of-10) by context length:")
    print(f"  {'N turns':<12} {'R@1':>8} {'vs random':>10} {'N':>6}")
    print("  " + "─" * 40)
    for n_ctx in range(1, cfg.max_turns + 1):
        idx_n = [i for i, l in enumerate(lens_all) if l == n_ctx]
        if len(idx_n) < 10: continue
        hits = 0
        for i in idx_n:
            neg_idx    = random.sample(
                [j for j in range(len(tgt_n)) if j != i], 9)
            candidates = [i] + neg_idx
            scores     = pred_n[i] @ tgt_n[candidates].T
            if scores.argmax().item() == 0:
                hits += 1
        r1 = hits / len(idx_n)
        print(f"  {n_ctx:<12} {r1:>8.4f} {r1-0.1:>+10.4f} {len(idx_n):>6}")

    # ── Baseline: last turn embedding as predictor ────────────────────────────
    last_embs = torch.stack([
        valid_ds[idx][0][valid_ds[idx][2] - 1]
        for idx in indices])
    last_n = last_embs / (last_embs.norm(dim=1, keepdim=True) + 1e-8)

    hits_last = 0
    for i in range(len(last_n)):
        neg_idx    = random.sample(
            [j for j in range(len(tgt_n)) if j != i], 9)
        candidates = [i] + neg_idx
        scores     = last_n[i] @ tgt_n[candidates].T
        if scores.argmax().item() == 0:
            hits_last += 1

    print(f"\n  Baseline (last turn emb): {hits_last/len(last_n):.4f}")
    print(f"  JEPA predictor:           "
          f"{sum(1 for i in range(len(pred_n)) if (pred_n[i] @ tgt_n[[i]+random.sample([j for j in range(len(tgt_n)) if j!=i],9)].T).argmax()==0)/len(pred_n):.4f}")
    print(f"  Random baseline:          0.1000")

    return {"mrr": mrr, "cos": cos}
