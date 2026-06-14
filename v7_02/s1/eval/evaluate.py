"""
evaluate.py
───────────
Evaluation pipeline for the DMI + MOML encoder.

Protocol (standard for dialog response retrieval)
──────────────────────────────────────────────────
For each query (context utterance):
  1. Pool of candidates = 1 correct response + 99 random negatives
  2. Rank candidates by cosine similarity to context encoding
  3. Measure R@1, R@2, R@5, MRR

Three conditions compared
──────────────────────────
  A   Random init          (no adapt)
  B   Pretrain static      (no adapt)
  B+  Pretrain + adapt
  C   MOML static          (no adapt)
  C*  MOML + adapt         ← target

Usage
─────
    from v7_02.s1.eval.evaluate import evaluate_all_conditions, print_results

    results = evaluate_all_conditions(
        moml_ckpt_path = '.../dmi_moml_best.pt',
        pretrain_ckpt  = '.../dmi_pretrain_best.pt',
        test_ds        = em_test_ds,
        tokenizer      = tokenizer,
        vocab_size     = vocab_size,
        device         = device,
    )
    print_results(results)
"""

from __future__ import annotations

import copy
import random
import sys
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

# ── robust import of test_time_adapt regardless of working directory ──────────
try:
    from v7_02.s1.adapt.test_time_adapt import encode_utterances
except ModuleNotFoundError:
    # Fallback: add repo root to path and try again
    _here = os.path.dirname(os.path.abspath(__file__))          # .../eval/
    _root = os.path.dirname(os.path.dirname(os.path.dirname(_here)))  # repo root
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from v7_02.s1.adapt.test_time_adapt import encode_utterances


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval metrics
# ─────────────────────────────────────────────────────────────────────────────

def retrieval_metrics(ctx_vecs, rsp_vecs, all_rsp, n_cands=100):
    """
    For each context i:
      candidates = [rsp_vecs[i]] + (n_cands-1) random negatives from all_rsp
      rank by cosine similarity, record rank of correct response.
    Returns dict: R@1, R@2, R@5, MRR, median_rank.
    """
    N   = len(ctx_vecs)
    rng = np.random.RandomState(42)
    ranks = []

    for i in range(N):
        neg_idx = rng.choice(len(all_rsp), size=min(n_cands - 1, len(all_rsp) - 1),
                             replace=False)
        negs    = all_rsp[neg_idx]
        cands   = np.concatenate([rsp_vecs[i:i+1], negs], axis=0)
        scores  = cands @ ctx_vecs[i]
        order   = np.argsort(-scores)
        rank    = int(np.where(order == 0)[0][0]) + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    return {
        'R@1':         float(np.mean(ranks == 1)),
        'R@2':         float(np.mean(ranks <= 2)),
        'R@5':         float(np.mean(ranks <= 5)),
        'MRR':         float(np.mean(1.0 / ranks)),
        'median_rank': float(np.median(ranks)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Inner adaptation on CR pairs (used inside evaluate, no external dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _adapt_on_pairs(model, pairs, pad_id, device, n_steps, lr):
    """Adapt a copy of model on (ctx_tensor, rsp_tensor) pairs via InfoNCE."""
    from torch.nn.utils.rnn import pad_sequence

    adapted = copy.deepcopy(model).to(device)
    opt     = torch.optim.SGD(adapted.parameters(), lr=lr)
    adapted.train()

    def _collate(ps):
        ctx_list, rsp_list = zip(*ps)
        ctx = pad_sequence(ctx_list, batch_first=True, padding_value=pad_id)
        rsp = pad_sequence(rsp_list, batch_first=True, padding_value=pad_id)
        return (ctx.to(device), rsp.to(device),
                (ctx == pad_id).to(device),
                (rsp == pad_id).to(device))

    for _ in range(n_steps):
        batch       = pairs if len(pairs) <= 32 else random.sample(pairs, 32)
        ctx, rsp, mc, mr = _collate(batch)
        c_t, z_t    = adapted(ctx, rsp, mc, mr)
        c_t         = F.normalize(c_t, dim=-1)
        z_t         = F.normalize(z_t, dim=-1)
        score       = torch.mm(c_t, z_t.t())
        loss        = -torch.mean(torch.diag(F.log_softmax(score, dim=1)))
        opt.zero_grad(); loss.backward(); opt.step()

    adapted.eval()
    return adapted


# ─────────────────────────────────────────────────────────────────────────────
# Per-emotion evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_one_condition(model, test_ds, tokenizer, device,
                            n_candidates=100, n_support=50,
                            n_inner_steps=5, lr_inner=1e-2,
                            adapt=False, verbose=True):
    """
    Evaluate one encoder condition on all emotions in test_ds.
    adapt=True  → adapt on n_support pairs before measuring.
    adapt=False → use encoder as-is.
    Returns {per_emotion: {...}, macro: {...}}.
    """
    if not hasattr(test_ds, 'pairs_by_emotion'):
        raise ValueError("test_ds must be EmotionAwareDataset")

    buckets  = test_ds.pairs_by_emotion()
    pad_id   = test_ds.pad_id
    macro    = defaultdict(list)
    per_em   = {}

    # Build a global negative pool (all responses from all emotions)
    all_rsp_texts = []
    for pairs in buckets.values():
        for _, rsp_ids in pairs[:30]:
            all_rsp_texts.append(
                tokenizer.decode(rsp_ids.tolist(), skip_special_tokens=True))

    # Encode negative pool once with the base model
    # (re-encoding per condition happens inside the loop when adapt=True)
    neg_vecs_base = encode_utterances(model, all_rsp_texts, tokenizer, device)

    for emotion, pairs in buckets.items():
        if len(pairs) < n_support + 10:
            continue

        random.shuffle(pairs)
        support = pairs[:n_support]
        query   = pairs[n_support:n_support + 200]   # cap for speed

        # ── Optional adaptation ───────────────────────────────────────────────
        if adapt and len(support) >= 4:
            eval_model = _adapt_on_pairs(
                model, support, pad_id, device, n_inner_steps, lr_inner)
        else:
            eval_model = model

        # ── Encode query pairs ────────────────────────────────────────────────
        ctx_texts = [tokenizer.decode(c.tolist(), skip_special_tokens=True)
                     for c, _ in query]
        rsp_texts = [tokenizer.decode(r.tolist(), skip_special_tokens=True)
                     for _, r in query]

        ctx_vecs = encode_utterances(eval_model, ctx_texts, tokenizer, device)
        rsp_vecs = encode_utterances(eval_model, rsp_texts, tokenizer, device)

        # Use base neg pool if not adapted, re-encode if adapted
        if adapt and eval_model is not model:
            neg_vecs = encode_utterances(eval_model, all_rsp_texts,
                                         tokenizer, device)
        else:
            neg_vecs = neg_vecs_base

        all_rsp  = np.concatenate([rsp_vecs, neg_vecs], axis=0)
        metrics  = retrieval_metrics(ctx_vecs, rsp_vecs, all_rsp,
                                     n_cands=min(n_candidates, len(all_rsp)))
        per_em[emotion] = metrics
        for k, v in metrics.items():
            macro[k].append(v)

        if verbose:
            print(f"  {emotion:20s}  R@1={metrics['R@1']:.3f}  "
                  f"MRR={metrics['MRR']:.3f}")

        if adapt and eval_model is not model:
            del eval_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {'per_emotion': per_em,
            'macro': {k: float(np.mean(v)) for k, v in macro.items()}}


# ─────────────────────────────────────────────────────────────────────────────
# Full five-condition comparison
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_conditions(moml_ckpt_path, pretrain_ckpt, test_ds,
                             tokenizer, vocab_size, device,
                             n_candidates=100, n_support=50,
                             n_inner_steps=5, lr_inner=1e-2):
    from v7_02.s1.cog_arch.encoder import DMIScratchEncoder

    def _load(path, vocab_size, device):
        ckpt  = torch.load(path, map_location=device)
        cfg   = ckpt.get('cfg', {})
        model = DMIScratchEncoder(
            vocab_size      = ckpt.get('vocab_size', vocab_size),
            d_model         = cfg.get('d_model', 256),
            projection_size = cfg.get('projection_size', 256),
            encoder_layers  = cfg.get('encoder_layers', 4),
            encoder_heads   = cfg.get('encoder_heads', 4),
            dim_feedforward = cfg.get('dim_feedforward', 1024),
            dropout         = cfg.get('dropout', 0.1),
        ).to(device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        return model

    kw = dict(test_ds=test_ds, tokenizer=tokenizer, device=device,
              n_candidates=n_candidates, n_support=n_support,
              n_inner_steps=n_inner_steps, lr_inner=lr_inner)
    results = {}

    print("\n── A: Random init (no adapt) ──")
    rand_model = DMIScratchEncoder(vocab_size=vocab_size, d_model=256,
        projection_size=256, encoder_layers=4, encoder_heads=4,
        dim_feedforward=1024).to(device)
    results['A_random'] = evaluate_one_condition(rand_model, adapt=False, **kw)
    del rand_model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n── B: Pretrain static (no adapt) ──")
    pt = _load(pretrain_ckpt, vocab_size, device)
    results['B_pretrain_static']  = evaluate_one_condition(pt, adapt=False, **kw)

    print("\n── B+: Pretrain + adapt ──")
    results['B_pretrain_adapted'] = evaluate_one_condition(pt, adapt=True,  **kw)
    del pt; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n── C: MOML static (no adapt) ──")
    mm = _load(moml_ckpt_path, vocab_size, device)
    results['C_moml_static']  = evaluate_one_condition(mm, adapt=False, **kw)

    print("\n── C*: MOML + adapt ──")
    results['C_moml_adapted'] = evaluate_one_condition(mm, adapt=True,  **kw)
    del mm; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results






# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def print_results(results):
    labels = {
        'A_random':           'A   Random init          (no adapt)',
        'B_pretrain_static':  'B   Pretrain static      (no adapt)',
        'B_pretrain_adapted': 'B+  Pretrain + adapt',
        'C_moml_static':      'C   MOML static          (no adapt)',
        'C_moml_adapted':     'C*  MOML + adapt         ← target',
    }
    hdr = f"{'Condition':<42}  {'R@1':>6}  {'R@2':>6}  {'R@5':>6}  {'MRR':>6}"
    print("\n" + "="*len(hdr))
    print(hdr)
    print("="*len(hdr))
    for key, label in labels.items():
        if key not in results:
            continue
        m = results[key]['macro']
        if not m or 'R@1' not in m:          # ← guard
            print(f"{label:<42}  (no emotions had enough pairs — "
                  f"reduce n_support)")
            continue
        print(f"{label:<42}  {m['R@1']:>6.3f}  {m['R@2']:>6.3f}  "
              f"{m['R@5']:>6.3f}  {m['MRR']:>6.3f}")
    print("="*len(hdr))
    # ... rest unchanged

    if 'C_moml_adapted' in results and 'B_pretrain_adapted' in results:
        delta = (results['C_moml_adapted']['macro']['R@1'] -
                 results['B_pretrain_adapted']['macro']['R@1'])
        sign  = '+' if delta >= 0 else ''
        print(f"\n  MOML gain over pretrain+adapt  ΔR@1 = {sign}{delta:.3f}")
        if   delta >  0.01: print("  ✓ MOML meta-initialization is beneficial.")
        elif delta > -0.01: print("  ~ MOML and pretrain+adapt are equivalent.")
        else:               print("  ✗ MOML init hurts — check meta-training.")