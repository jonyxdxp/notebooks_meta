"""
test_time_adapt.py
──────────────────
Test-time adaptation of the DMI encoder to a new conversation.

Two adaptation modes, selected automatically based on conversation length:

  LONG (≥ 4 turns): InfoNCE on (context, response) pairs extracted
    from the conversation itself. Same objective as meta-training —
    the MOML gradient landscape was shaped for exactly this.

  SHORT (< 4 turns): SimCSE — encode each utterance twice with
    different dropout masks, treat the two views as a positive pair,
    all other utterances as negatives.
    NOTE: SimCSE adaptation is only theoretically well-grounded if
    the meta-init was also trained with SimCSE tasks. The current
    MOML checkpoint was trained with InfoNCE only, so SimCSE adaptation
    is provided as a best-effort fallback, not a guaranteed improvement.

Usage
─────
    from test_time_adapt import test_time_adapt, encode_utterances

    # Adapt encoder to a specific conversation
    adapted = test_time_adapt(
        meta_model        = moml_model,   # your MOML checkpoint
        conversation_turns= ["Hi!", "Hello, how are you?", "Fine thanks"],
        tokenizer         = tokenizer,
        device            = device,
        n_steps           = 5,
        lr                = 1e-2,
    )

    # Encode utterances with the adapted encoder
    vecs = encode_utterances(adapted, ["Some utterance"], tokenizer, device)
    # vecs: (N, d_model) numpy array, L2-normalised
"""

from __future__ import annotations

import copy
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


# ─────────────────────────────────────────────────────────────────────────────
# Collation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _collate_pairs(pairs, pad_id, device):
    """(ctx_tensor, rsp_tensor) list → batched tensors + masks."""
    ctx_list, rsp_list = zip(*pairs)
    ctx = pad_sequence(ctx_list, batch_first=True, padding_value=pad_id)
    rsp = pad_sequence(rsp_list, batch_first=True, padding_value=pad_id)
    return (ctx.to(device), rsp.to(device),
            (ctx == pad_id).to(device),
            (rsp == pad_id).to(device))


def _tokenize_turns(turns, tokenizer, max_ctx_len=64, max_rsp_len=32):
    """
    Build (ctx_ids, rsp_ids) pairs from a list of conversation turns.
    Returns list of (LongTensor, LongTensor).
    """
    EOU   = '__eou__'
    pairs = []
    toks  = [f"{t.strip()} {EOU}" for t in turns]
    for j in range(1, len(toks)):
        ctx_text = ' '.join(toks[:j])
        rsp_text = toks[j]
        ctx_ids  = tokenizer.encode(
            ctx_text, add_special_tokens=True,
            max_length=max_ctx_len, truncation=True)
        rsp_ids  = tokenizer.encode(
            rsp_text, add_special_tokens=True,
            max_length=max_rsp_len, truncation=True)
        pairs.append((
            torch.tensor(ctx_ids, dtype=torch.long),
            torch.tensor(rsp_ids, dtype=torch.long),
        ))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# InfoNCE loss (inline, no external dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _infonce(c_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    score   = torch.mm(c_t, z_t.t())
    log_p   = F.log_softmax(score, dim=1)
    return  -torch.mean(torch.diag(log_p))


# ─────────────────────────────────────────────────────────────────────────────
# SimCSE loss  (dropout augmentation, no labels needed)
# ─────────────────────────────────────────────────────────────────────────────

def _simcse_loss(model: nn.Module, utt_tensors: list,
                 pad_id: int, device: str,
                 temperature: float = 0.05) -> torch.Tensor:
    """
    SimCSE on a list of utterance token tensors.
    Two forward passes with different dropout → two views per utterance.
    Positive pair = (view1_i, view2_i).  All cross-utterance pairs are negatives.
    """
    if len(utt_tensors) < 2:
        raise ValueError("Need ≥ 2 utterances for SimCSE.")

    utts = pad_sequence(utt_tensors, batch_first=True, padding_value=pad_id)
    mask = (utts == pad_id)
    utts = utts.to(device)
    mask = mask.to(device)

    # Two stochastic forward passes (dropout differs each time)
    model.train()

    def _encode_once(tokens, pad_mask):
        x   = model.embedding(tokens)
        out = model.encoder(x, pad_mask)
        return out[:, 0, :]   # CLS representation

    z1 = _encode_once(utts, mask)   # (N, d)
    z2 = _encode_once(utts, mask)   # (N, d) — different dropout

    # Normalise
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    # NT-Xent
    N     = z1.size(0)
    sims  = torch.mm(z1, z2.t()) / temperature          # (N, N)
    labels= torch.arange(N, device=device)
    loss  = F.cross_entropy(sims, labels)
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Main adaptation function
# ─────────────────────────────────────────────────────────────────────────────

def test_time_adapt(
    meta_model,
    conversation_turns: List[str],
    tokenizer,
    device:       str,
    n_steps:      int   = 5,
    lr:           float = 1e-2,
    max_ctx_len:  int   = 64,
    max_rsp_len:  int   = 32,
    min_turns_for_infonce: int = 4,
    verbose:      bool  = False,
) -> nn.Module:
    """
    Adapt a copy of meta_model to a specific conversation.
    meta_model is NOT modified — adaptation happens on a deep copy.

    Parameters
    ──────────
    meta_model          : DMIScratchEncoder (MOML checkpoint)
    conversation_turns  : list of utterance strings, chronological order
    tokenizer           : BertTokenizerFast (same as training)
    device              : 'cuda' | 'cpu'
    n_steps             : gradient steps of inner adaptation
    lr                  : inner learning rate (default 1e-2, same as MOML)
    min_turns_for_infonce: use InfoNCE if len(turns) >= this, else SimCSE

    Returns
    ───────
    adapted : nn.Module in eval mode, on device
              Calling meta_model again returns the original (unmodified).
    """
    pad_id  = tokenizer.pad_token_id
    adapted = copy.deepcopy(meta_model).to(device)
    opt     = torch.optim.SGD(adapted.parameters(), lr=lr)

    n_turns = len(conversation_turns)

    # ── Mode selection ────────────────────────────────────────────────────────
    if n_turns >= min_turns_for_infonce:
        mode = 'infonce'
        pairs = _tokenize_turns(
            conversation_turns, tokenizer, max_ctx_len, max_rsp_len)
        if verbose:
            print(f"[adapt] InfoNCE mode  |  {len(pairs)} CR pairs  |  "
                  f"{n_steps} steps  |  lr={lr}")
    else:
        mode = 'simcse'
        # Tokenise individual utterances (not pairs)
        utt_tensors = []
        for turn in conversation_turns:
            ids = tokenizer.encode(
                turn + ' __eou__', add_special_tokens=True,
                max_length=max_ctx_len, truncation=True)
            utt_tensors.append(torch.tensor(ids, dtype=torch.long))
        if verbose:
            print(f"[adapt] SimCSE mode   |  {n_turns} utterances  |  "
                  f"{n_steps} steps  |  lr={lr}")

    if n_turns < 2:
        if verbose:
            print("[adapt] Conversation too short — returning meta-init unchanged.")
        adapted.eval()
        return adapted

    # ── Adaptation loop ───────────────────────────────────────────────────────
    adapted.train()

    for step in range(n_steps):
        opt.zero_grad()

        if mode == 'infonce':
            # Sample a batch (use all if short, else random subset)
            batch = pairs if len(pairs) <= 32 else random.sample(pairs, 32)
            ctx, rsp, mc, mr = _collate_pairs(batch, pad_id, device)
            c_t, z_t = adapted(ctx, rsp, mc, mr)
            # L2-normalise before InfoNCE (improves stability)
            c_t = F.normalize(c_t, dim=-1)
            z_t = F.normalize(z_t, dim=-1)
            loss = _infonce(c_t, z_t)

        else:   # simcse
            loss = _simcse_loss(adapted, utt_tensors, pad_id, device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapted.parameters(), 1.0)
        opt.step()

        if verbose:
            print(f"  step {step+1}/{n_steps}  loss={loss.item():.4f}", end='\r')

    if verbose:
        print()

    adapted.eval()
    return adapted


# ─────────────────────────────────────────────────────────────────────────────
# Encoding utility
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_utterances(
    model,
    utterances:  List[str],
    tokenizer,
    device:      str,
    max_len:     int   = 64,
    batch_size:  int   = 64,
    normalize:   bool  = True,
) -> np.ndarray:
    """
    Encode a list of utterance strings → L2-normalised numpy array (N, d_model).

    Uses the context-side encoder (CLS pooling, no projection head).
    These are the representations you'd use for retrieval/clustering/etc.
    """
    pad_id = tokenizer.pad_token_id
    model.eval()
    all_vecs = []

    for i in range(0, len(utterances), batch_size):
        batch_text = utterances[i:i+batch_size]
        ids_list   = [
            torch.tensor(
                tokenizer.encode(
                    f"{t} __eou__", add_special_tokens=True,
                    max_length=max_len, truncation=True),
                dtype=torch.long)
            for t in batch_text
        ]
        tokens = pad_sequence(ids_list, batch_first=True, padding_value=pad_id)
        mask   = (tokens == pad_id)
        tokens = tokens.to(device)
        mask   = mask.to(device)

        vecs = model.encode_context(tokens, mask)   # (B, d_model)
        if normalize:
            vecs = F.normalize(vecs, dim=-1)
        all_vecs.append(vecs.cpu().numpy())

    return np.concatenate(all_vecs, axis=0)