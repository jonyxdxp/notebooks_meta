"""
test_time_adapt.py
──────────────────
Test-time adaptation of the DMI encoder to a new conversation.
Place at: v7_02/s1/adapt/test_time_adapt.py
"""

from __future__ import annotations

import copy
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def _collate_pairs(pairs, pad_id, device):
    ctx_list, rsp_list = zip(*pairs)
    ctx = pad_sequence(ctx_list, batch_first=True, padding_value=pad_id)
    rsp = pad_sequence(rsp_list, batch_first=True, padding_value=pad_id)
    return (ctx.to(device), rsp.to(device),
            (ctx == pad_id).to(device),
            (rsp == pad_id).to(device))


def _tokenize_turns(turns, tokenizer, max_ctx_len=64, max_rsp_len=32):
    EOU   = '__eou__'
    pairs = []
    toks  = [f"{t.strip()} {EOU}" for t in turns]
    for j in range(1, len(toks)):
        ctx_ids = tokenizer.encode(' '.join(toks[:j]),
            add_special_tokens=True, max_length=max_ctx_len, truncation=True)
        rsp_ids = tokenizer.encode(toks[j],
            add_special_tokens=True, max_length=max_rsp_len, truncation=True)
        pairs.append((torch.tensor(ctx_ids, dtype=torch.long),
                      torch.tensor(rsp_ids, dtype=torch.long)))
    return pairs


def _infonce(c_t, z_t):
    score = torch.mm(c_t, z_t.t())
    return -torch.mean(torch.diag(F.log_softmax(score, dim=1)))


def _simcse_loss(model, utt_tensors, pad_id, device, temperature=0.05):
    utts = pad_sequence(utt_tensors, batch_first=True,
                        padding_value=pad_id).to(device)
    mask = (utts == pad_id).to(device)
    model.train()

    def _enc(t, m):
        x   = model.embedding(t)
        out = model.encoder(x, m)
        return out[:, 0, :]

    z1 = F.normalize(_enc(utts, mask), dim=-1)
    z2 = F.normalize(_enc(utts, mask), dim=-1)
    N  = z1.size(0)
    return F.cross_entropy(torch.mm(z1, z2.t()) / temperature,
                           torch.arange(N, device=device))


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
    meta_model is NOT modified. Returns adapted model in eval mode.

    LONG conversation (>= min_turns_for_infonce): InfoNCE on (ctx, rsp) pairs.
    SHORT conversation: SimCSE dropout augmentation.
    """
    pad_id  = tokenizer.pad_token_id
    adapted = copy.deepcopy(meta_model).to(device)
    opt     = torch.optim.SGD(adapted.parameters(), lr=lr)
    n_turns = len(conversation_turns)

    if n_turns < 2:
        if verbose: print("[adapt] Too short — returning meta-init unchanged.")
        adapted.eval(); return adapted

    if n_turns >= min_turns_for_infonce:
        mode  = 'infonce'
        pairs = _tokenize_turns(conversation_turns, tokenizer,
                                max_ctx_len, max_rsp_len)
        if verbose:
            print(f"[adapt] InfoNCE | {len(pairs)} CR pairs | "
                  f"{n_steps} steps | lr={lr}")
    else:
        mode = 'simcse'
        utt_tensors = [
            torch.tensor(tokenizer.encode(
                t + ' __eou__', add_special_tokens=True,
                max_length=max_ctx_len, truncation=True), dtype=torch.long)
            for t in conversation_turns
        ]
        if verbose:
            print(f"[adapt] SimCSE | {n_turns} utterances | "
                  f"{n_steps} steps | lr={lr}")

    adapted.train()
    for step in range(n_steps):
        opt.zero_grad()
        if mode == 'infonce':
            batch       = pairs if len(pairs) <= 32 else random.sample(pairs, 32)
            ctx, rsp, mc, mr = _collate_pairs(batch, pad_id, device)
            c_t, z_t    = adapted(ctx, rsp, mc, mr)
            loss        = _infonce(F.normalize(c_t, dim=-1),
                                   F.normalize(z_t, dim=-1))
        else:
            loss = _simcse_loss(adapted, utt_tensors, pad_id, device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapted.parameters(), 1.0)
        opt.step()
        if verbose:
            print(f"  step {step+1}/{n_steps}  loss={loss.item():.4f}", end='\r')

    if verbose: print()
    adapted.eval()
    return adapted


@torch.no_grad()
def encode_utterances(model, utterances: List[str], tokenizer,
                       device: str, max_len: int = 64,
                       batch_size: int = 64,
                       normalize: bool = True) -> np.ndarray:
    """
    Encode utterance strings → L2-normalised numpy array (N, d_model).
    Uses context-side CLS representation (no projection head).
    """
    pad_id   = tokenizer.pad_token_id
    model.eval()
    all_vecs = []

    for i in range(0, len(utterances), batch_size):
        batch = utterances[i:i+batch_size]
        ids_list = [
            torch.tensor(tokenizer.encode(
                f"{t} __eou__", add_special_tokens=True,
                max_length=max_len, truncation=True), dtype=torch.long)
            for t in batch
        ]
        tokens = pad_sequence(ids_list, batch_first=True, padding_value=pad_id)
        mask   = (tokens == pad_id)
        vecs   = model.encode_context(tokens.to(device), mask.to(device))
        if normalize:
            vecs = F.normalize(vecs, dim=-1)
        all_vecs.append(vecs.cpu().numpy())

    return np.concatenate(all_vecs, axis=0)