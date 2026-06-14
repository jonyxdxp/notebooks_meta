"""
evaluate.py
───────────
Evaluation pipeline for the DMI + MOML encoder.

Protocol (standard for dialog response retrieval)
──────────────────────────────────────────────────
For each query (context utterance):
  1. Pool of candidates = 1 correct response + 99 random negatives
  2. Rank candidates by cosine similarity to context encoding
  3. Measure R@1 (is the correct response ranked 1st?), R@2, MRR

Three conditions compared
──────────────────────────
  A. Random init    — freshly initialised encoder, no adaptation
  B. Pretrain only  — InfoNCE pretrained, NO test-time adaptation
  C. MOML + adapt   — MOML meta-init + K inner steps on support pairs

If C > B > A:  MOML meta-learning is working as intended.
If C ≈ B > A:  Pretrain is doing the work; MOML doesn't help.
If C ≈ B ≈ A:  Neither pretrain nor MOML helped — data/arch problem.

Usage
─────
    from evaluate import evaluate_all_conditions, print_results

    results = evaluate_all_conditions(
        moml_ckpt_path = '/content/drive/MyDrive/dmi_moml_ckpts/dmi_moml_best.pt',
        pretrain_ckpt  = '/content/drive/MyDrive/dmi_moml_ckpts/dmi_pretrain_best.pt',
        test_ds        = em_test_ds,          # EmotionAwareDataset, test split
        tokenizer      = tokenizer,
        vocab_size     = vocab_size,
        device         = device,
        n_candidates   = 100,
        n_support      = 50,
        n_inner_steps  = 5,
        lr_inner       = 1e-2,
    )
    print_results(results)
"""

from __future__ import annotations

import copy
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from test_time_adapt import test_time_adapt, encode_utterances


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval metrics
# ─────────────────────────────────────────────────────────────────────────────

def retrieval_metrics(
    ctx_vecs:  np.ndarray,   # (N, d)  context representations
    rsp_vecs:  np.ndarray,   # (N, d)  correct response representations
    all_rsp:   np.ndarray,   # (M, d)  all response candidates (M >> N)
    n_cands:   int = 100,
) -> dict:
    """
    For each context i:
      - candidates = [rsp_vecs[i]] + (n_cands-1) random negatives from all_rsp
      - rank candidates by cosine similarity to ctx_vecs[i]
      - record rank of the correct response

    Returns dict with R@1, R@2, R@5, MRR.
    """
    N        = len(ctx_vecs)
    ranks    = []
    rng      = np.random.RandomState(42)

    # Pre-build negative pool (exclude correct responses)
    neg_pool = all_rsp   # use all_rsp as pool; collision probability is tiny

    for i in range(N):
        # Sample n_cands-1 negatives (may accidentally include true response
        # with tiny probability — acceptable for large pools)
        neg_idx  = rng.choice(len(neg_pool), size=n_cands - 1, replace=False)
        negs     = neg_pool[neg_idx]                          # (n_cands-1, d)
        cands    = np.concatenate([rsp_vecs[i:i+1], negs], axis=0)  # (n_cands, d)

        # Cosine similarity (vectors are L2-normalised so dot = cosine)
        scores   = cands @ ctx_vecs[i]                        # (n_cands,)
        order    = np.argsort(-scores)                        # descending
        rank     = int(np.where(order == 0)[0][0]) + 1       # 1-indexed
        ranks.append(rank)

    ranks = np.array(ranks)
    return {
        'R@1':  float(np.mean(ranks == 1)),
        'R@2':  float(np.mean(ranks <= 2)),
        'R@5':  float(np.mean(ranks <= 5)),
        'MRR':  float(np.mean(1.0 / ranks)),
        'median_rank': float(np.median(ranks)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-emotion evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_one_condition(
    model,
    test_ds,
    tokenizer,
    device:       str,
    n_candidates: int   = 100,
    n_support:    int   = 50,
    n_inner_steps:int   = 5,
    lr_inner:     float = 1e-2,
    adapt:        bool  = False,
    verbose:      bool  = True,
) -> dict:
    """
    Evaluate one encoder condition on the test set.

    If adapt=True: for each emotion, adapt the encoder on n_support pairs
    first, then evaluate on the remaining pairs.
    If adapt=False: use the encoder as-is.

    Returns per-emotion metrics and macro-averaged metrics.
    """
    if not hasattr(test_ds, 'pairs_by_emotion'):
        raise ValueError("test_ds must be EmotionAwareDataset "
                         "(needs pairs_by_emotion() method)")

    buckets = test_ds.pairs_by_emotion()
    per_emotion = {}

    # Build global response pool for negative sampling
    all_responses = []
    for pairs in buckets.values():
        for _, rsp in pairs:
            all_responses.append(rsp)

    # Encode all responses once with the base model for the negative pool
    # (we'll re-encode for each condition as needed)

    macro = defaultdict(list)

    for emotion, pairs in buckets.items():
        if len(pairs) < n_support + 10:
            continue   # not enough pairs for support + query

        random.shuffle(pairs)
        support = pairs[:n_support]
        query   = pairs[n_support:]

        # ── Optional: adapt encoder on support ───────────────────────────────
        if adapt and len(support) >= 4:
            # Convert support pairs to conversation-like turns
            # We adapt using the (ctx, rsp) pairs directly via InfoNCE
            eval_model = _adapt_on_pairs(
                model, support, tokenizer, device,
                n_inner_steps, lr_inner, test_ds.pad_id)
        else:
            eval_model = model

        # ── Encode query contexts and responses ───────────────────────────────
        ctx_texts = []
        rsp_texts = []
        for ctx_ids, rsp_ids in query[:200]:   # cap at 200 for speed
            ctx_texts.append(tokenizer.decode(
                ctx_ids.tolist(), skip_special_tokens=True))
            rsp_texts.append(tokenizer.decode(
                rsp_ids.tolist(), skip_special_tokens=True))

        ctx_vecs = encode_utterances(eval_model, ctx_texts, tokenizer, device)
        rsp_vecs = encode_utterances(eval_model, rsp_texts, tokenizer, device)

        # Build response pool from all other emotions (strong negatives)
        neg_texts = []
        for other_em, other_pairs in buckets.items():
            if other_em == emotion:
                continue
            for ctx_ids, rsp_ids in other_pairs[:50]:
                neg_texts.append(tokenizer.decode(
                    rsp_ids.tolist(), skip_special_tokens=True))
        neg_vecs = encode_utterances(eval_model, neg_texts, tokenizer, device)
        all_rsp  = np.concatenate([rsp_vecs, neg_vecs], axis=0)

        metrics = retrieval_metrics(
            ctx_vecs, rsp_vecs, all_rsp,
            n_cands=min(n_candidates, len(all_rsp)))

        per_emotion[emotion] = metrics
        for k, v in metrics.items():
            macro[k].append(v)

        if verbose:
            print(f"  {emotion:20s}  R@1={metrics['R@1']:.3f}  "
                  f"R@2={metrics['R@2']:.3f}  MRR={metrics['MRR']:.3f}")

        # Free adapted model if we made a copy
        if adapt and eval_model is not model:
            del eval_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    macro_avg = {k: float(np.mean(v)) for k, v in macro.items()}
    return {'per_emotion': per_emotion, 'macro': macro_avg}


def _adapt_on_pairs(model, pairs, tokenizer, device,
                    n_steps, lr, pad_id):
    """Adapt encoder directly on (ctx_tensor, rsp_tensor) pairs."""
    from torch.nn.utils.rnn import pad_sequence

    adapted = copy.deepcopy(model).to(device)
    opt     = torch.optim.SGD(adapted.parameters(), lr=lr)
    adapted.train()

    def collate(ps):
        ctx_list, rsp_list = zip(*ps)
        ctx = pad_sequence(ctx_list, batch_first=True, padding_value=pad_id)
        rsp = pad_sequence(rsp_list, batch_first=True, padding_value=pad_id)
        return (ctx.to(device), rsp.to(device),
                (ctx==pad_id).to(device), (rsp==pad_id).to(device))

    for _ in range(n_steps):
        batch = pairs if len(pairs) <= 32 else random.sample(pairs, 32)
        ctx, rsp, mc, mr = collate(batch)
        c_t, z_t = adapted(ctx, rsp, mc, mr)
        c_t = F.normalize(c_t, dim=-1)
        z_t = F.normalize(z_t, dim=-1)
        score = torch.mm(c_t, z_t.t())
        loss  = -torch.mean(torch.diag(F.log_softmax(score, dim=1)))
        opt.zero_grad(); loss.backward(); opt.step()

    adapted.eval()
    return adapted


# ─────────────────────────────────────────────────────────────────────────────
# Full three-condition comparison
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_conditions(
    moml_ckpt_path:  str,
    pretrain_ckpt:   str,
    test_ds,
    tokenizer,
    vocab_size:      int,
    device:          str,
    n_candidates:    int   = 100,
    n_support:       int   = 50,
    n_inner_steps:   int   = 5,
    lr_inner:        float = 1e-2,
) -> dict:
    """
    Run all three conditions and return results dict.
    """
    from v7_02.s1.cog_arch.encoder import DMIScratchEncoder

    def _load_encoder(ckpt_path, vocab_size, device):
        ckpt  = torch.load(ckpt_path, map_location=device)
        cfg   = ckpt.get('cfg', {})
        model = DMIScratchEncoder(
            vocab_size       = ckpt.get('vocab_size', vocab_size),
            d_model          = cfg.get('d_model', 256),
            projection_size  = cfg.get('projection_size', 256),
            encoder_layers   = cfg.get('encoder_layers', 4),
            encoder_heads    = cfg.get('encoder_heads', 4),
            dim_feedforward  = cfg.get('dim_feedforward', 1024),
            dropout          = cfg.get('dropout', 0.1),
        ).to(device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        return model

    results = {}
    kw = dict(
        test_ds      = test_ds,
        tokenizer    = tokenizer,
        device       = device,
        n_candidates = n_candidates,
        n_support    = n_support,
        n_inner_steps= n_inner_steps,
        lr_inner     = lr_inner,
    )

    # ── A: Random init (no training, no adaptation) ───────────────────────────
    print("\n── Condition A: Random init (no adaptation) ──")
    rand_model = DMIScratchEncoder(
        vocab_size=vocab_size, d_model=256, projection_size=256,
        encoder_layers=4, encoder_heads=4, dim_feedforward=1024,
    ).to(device)
    results['A_random'] = evaluate_one_condition(
        rand_model, adapt=False, **kw)
    del rand_model; torch.cuda.empty_cache()

    # ── B: Pretrain only (static, no adaptation) ──────────────────────────────
    print("\n── Condition B: Pretrain only (no adaptation) ──")
    pretrain_model = _load_encoder(pretrain_ckpt, vocab_size, device)
    results['B_pretrain_static'] = evaluate_one_condition(
        pretrain_model, adapt=False, **kw)

    # ── B+: Pretrain + adaptation (is adaptation itself helpful?) ─────────────
    print("\n── Condition B+: Pretrain + adaptation ──")
    results['B_pretrain_adapted'] = evaluate_one_condition(
        pretrain_model, adapt=True, **kw)
    del pretrain_model; torch.cuda.empty_cache()

    # ── C: MOML init + adaptation ─────────────────────────────────────────────
    print("\n── Condition C: MOML init + adaptation ──")
    moml_model = _load_encoder(moml_ckpt_path, vocab_size, device)
    results['C_moml_adapted'] = evaluate_one_condition(
        moml_model, adapt=True, **kw)

    # ── C_static: MOML init without adaptation ────────────────────────────────
    print("\n── Condition C_static: MOML init (no adaptation) ──")
    results['C_moml_static'] = evaluate_one_condition(
        moml_model, adapt=False, **kw)
    del moml_model; torch.cuda.empty_cache()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def print_results(results: dict):
    """Print a clean comparison table."""
    labels = {
        'A_random':           'A  Random init          (no adapt)',
        'B_pretrain_static':  'B  Pretrain static      (no adapt)',
        'B_pretrain_adapted': 'B+ Pretrain + adapt',
        'C_moml_static':      'C  MOML static          (no adapt)',
        'C_moml_adapted':     'C* MOML + adapt         ← target',
    }

    header = f"{'Condition':<42}  {'R@1':>6}  {'R@2':>6}  {'R@5':>6}  {'MRR':>6}"
    print("\n" + "="*len(header))
    print(header)
    print("="*len(header))

    for key, label in labels.items():
        if key not in results:
            continue
        m = results[key]['macro']
        print(f"{label:<42}  "
              f"{m['R@1']:>6.3f}  "
              f"{m['R@2']:>6.3f}  "
              f"{m['R@5']:>6.3f}  "
              f"{m['MRR']:>6.3f}")

    print("="*len(header))

    # Highlight MOML gain over pretrain+adapt
    if 'C_moml_adapted' in results and 'B_pretrain_adapted' in results:
        delta = (results['C_moml_adapted']['macro']['R@1'] -
                 results['B_pretrain_adapted']['macro']['R@1'])
        sign  = '+' if delta >= 0 else ''
        print(f"\n  MOML gain over pretrain+adapt  ΔR@1 = {sign}{delta:.3f}")
        if delta > 0.01:
            print("  ✓ MOML meta-initialization is beneficial.")
        elif delta > -0.01:
            print("  ~ MOML and pretrain+adapt are equivalent.")
        else:
            print("  ✗ MOML init hurts — check meta-training quality.")