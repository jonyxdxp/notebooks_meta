

# from https://github.com/facebookresearch/eb_jepa/blob/main/examples/image_jepa/eval.py







"""
Evaluation utilities for self-supervised learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


class LinearProbe(nn.Module):
    """Linear probe classifier for evaluating representations."""

    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


def evaluate_linear_probe(model, linear_probe, val_loader, device, use_amp=True):
    """Evaluate linear probe on validation set."""
    model.eval()
    linear_probe.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                features, _ = model(data)

            outputs = linear_probe(features.float())
            loss = F.cross_entropy(outputs, target)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(val_loader)

    return accuracy, avg_loss















    # --------------------------------------------













    # ── Cell: Masked Span Recovery Evaluation ────────────────────────────────────
#
# For each utterance in the test set:
#   1. Create a masked version (same collator as training)
#   2. Encode masked → z_ctx  (context encoder, trained)
#   3. Encode original → z_tgt (context encoder, trained)
#   4. Measure cosine similarity sim(z_ctx, z_tgt)
#   5. Compare against sim(z_ctx, z_random) for N random distractors
#
# Metric: Recall@K — is the correct original in the top-K most similar?
# If the encoder learned to predict masked spans, sim(ctx, correct) >> sim(ctx, random)

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# ── Encode helper (mean pool, no grad) ───────────────────────────────────────

@torch.no_grad()
def encode_batch(encoder, input_ids, attention_mask):
    hidden = encoder(input_ids, attention_mask=attention_mask)
    if isinstance(hidden, tuple):
        hidden = hidden[0]                              # (B, L, D)
    mask_f = attention_mask.unsqueeze(-1).float()
    return F.normalize(
        (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1),
        dim=-1
    )                                                   # (B, D) L2-normalized

# ── Build a pool of test embeddings (originals) ───────────────────────────────
# We'll use these as the distractor bank

print('Building test embedding pool …')

# Use the plain (non-JEPA) dataloader — we want clean input_ids + attention_mask
plain_test_loader = torch.utils.data.DataLoader(
    dataset['test'],
    batch_size  = 256,
    shuffle     = False,
    num_workers = 0,
    collate_fn  = lambda batch: {
        'input_ids':      torch.stack([b['input_ids']      for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
    }
)

all_embeddings_trained = []
all_embeddings_random  = []

for batch in tqdm(plain_test_loader, desc='Encoding test pool'):
    ids  = batch['input_ids'].to(DEVICE)
    mask = batch['attention_mask'].to(DEVICE)
    all_embeddings_trained.append(encode_batch(context_encoder, ids, mask).cpu())
    all_embeddings_random.append(encode_batch(random_encoder,   ids, mask).cpu())

pool_trained = torch.cat(all_embeddings_trained, dim=0)  # (N_test, D)
pool_random  = torch.cat(all_embeddings_random,  dim=0)

print(f'  pool size: {pool_trained.shape[0]:,} utterances  dim={pool_trained.shape[1]}')

# ── JEPA collator to produce masked views ─────────────────────────────────────
# Reuse the same collator from training — identical masking distribution

jepa_test_loader = torch.utils.data.DataLoader(
    dataset['test'],
    batch_size  = 256,
    shuffle     = False,
    num_workers = 0,
    collate_fn  = collator,   # JEPAMaskCollator from training
)

# ── Recall@K evaluation ───────────────────────────────────────────────────────

def recall_at_k(ctx_embs, pool_embs, ks=(1, 5, 10)):
    """
    For each query i, rank all pool entries by cosine similarity.
    Recall@K = fraction of queries where correct entry (index i) is in top-K.
    ctx_embs and pool_embs are L2-normalized → dot product = cosine sim.
    """
    N = ctx_embs.shape[0]
    # Full similarity matrix (N, N)
    sim = ctx_embs @ pool_embs.T    # (N, N)
    results = {}
    for k in ks:
        topk_indices = sim.topk(k, dim=1).indices   # (N, K)
        correct = torch.arange(N).unsqueeze(1)       # (N, 1)
        hits = (topk_indices == correct).any(dim=1).float()
        results[f'R@{k}'] = hits.mean().item() * 100
    return results


print('\nEncoding masked (context) views …')

ctx_embs_trained = []
ctx_embs_random  = []

for batch in tqdm(jepa_test_loader, desc='Encoding masked views'):
    ctx_ids  = batch['context_input_ids'].to(DEVICE)
    ctx_mask = batch['context_attention_mask'].to(DEVICE)
    ctx_embs_trained.append(encode_batch(context_encoder, ctx_ids, ctx_mask).cpu())
    ctx_embs_random.append(encode_batch(random_encoder,   ctx_ids, ctx_mask).cpu())

ctx_trained = torch.cat(ctx_embs_trained, dim=0)   # (N_test, D)
ctx_random  = torch.cat(ctx_embs_random,  dim=0)

# ── Run Recall@K ──────────────────────────────────────────────────────────────

print('\nComputing Recall@K …')

r_trained = recall_at_k(ctx_trained, pool_trained)
r_random  = recall_at_k(ctx_random,  pool_random)

# Random chance baseline (uniform retrieval over N items)
N = pool_trained.shape[0]
r_chance = {f'R@{k}': k / N * 100 for k in (1, 5, 10)}

print(f'\n{"="*52}')
print(f'  Masked Span Recovery — Recall@K  (N={N:,})')
print(f'{"="*52}')
print(f'  {"":20s}  R@1      R@5      R@10')
print(f'  {"Random chance":20s}  '
      f'{r_chance["R@1"]:.3f}%  {r_chance["R@5"]:.3f}%  {r_chance["R@10"]:.3f}%')
print(f'  {"Random encoder":20s}  '
      f'{r_random["R@1"]:.2f}%   {r_random["R@5"]:.2f}%   {r_random["R@10"]:.2f}%')
print(f'  {"Trained encoder":20s}  '
      f'{r_trained["R@1"]:.2f}%   {r_trained["R@5"]:.2f}%   {r_trained["R@10"]:.2f}%')
print(f'  {"Gap (trained-random)":20s}  '
      f'{r_trained["R@1"]-r_random["R@1"]:+.2f}%   '
      f'{r_trained["R@5"]-r_random["R@5"]:+.2f}%   '
      f'{r_trained["R@10"]-r_random["R@10"]:+.2f}%')
print(f'{"="*52}')

# ── Also report mean reciprocal rank (MRR) ────────────────────────────────────

@torch.no_grad()
def mean_reciprocal_rank(ctx_embs, pool_embs):
    sim  = ctx_embs @ pool_embs.T          # (N, N)
    N    = sim.shape[0]
    correct = torch.arange(N)
    ranks = (sim.argsort(dim=1, descending=True) == correct.unsqueeze(1)).float().argmax(dim=1) + 1
    return (1.0 / ranks.float()).mean().item()

mrr_trained = mean_reciprocal_rank(ctx_trained, pool_trained)
mrr_random  = mean_reciprocal_rank(ctx_random,  pool_random)

print(f'\n  MRR  random={mrr_random:.4f}  trained={mrr_trained:.4f}  '
      f'gap={mrr_trained - mrr_random:+.4f}')
print(f'  (MRR=1.0 means always ranks correct first; random chance ≈ {1/N:.4f})')