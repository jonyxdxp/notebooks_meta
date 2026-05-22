#!/usr/bin/env python3
"""
S3 COLD Inference — Option A
Constrained decoding with Langevin dynamics on the S3 decoder output space.

Pipeline at inference:
  conversation history
      → S1 encoder  (frozen)  → z_seq
      → S2 predictor (frozen) → z_pred   (= z_fused from PoE at real inference)
      → COLD loop on decoder  → constrained discrete text

z_T is FIXED throughout the entire Langevin loop.
Only the soft output sequence ỹ (= y_logits + epsilon) is optimized.

Constraint tokens are hard-wired externally (e.g. required keywords / entity names).
The decoder itself provides the fluency signal — no separate LM needed.
"""

import sys, os
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── Paths (mirror your s3_train.py) ──────────────────────────────────────────
ROOT   = '/content/notebooks_meta'
S1     = f'{ROOT}/v5/s1'
S2     = f'{ROOT}/v5/s2'
S3     = f'{ROOT}/v5/s3'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.insert(0, ROOT)

from v5.s1.cog_arch.encoder import Encoder
from v5.s3.cog_arch.decoder import Decoder
from v5.s2.config import CFG
from v5.s2.cog_arch.dm import DM

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_dataset  = _load_module('data.dataset', f'{S2}/data/dataset.py')
tokenizer = _dataset.tokenizer   # BERT WordPiece, vocab_size=30522







# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Soft forward pass through your decoder
# ═══════════════════════════════════════════════════════════════════════════════

def soft_forward_decoder(decoder, y_soft_logits, z_T, temp_in=0.001):
    """
    Differentiable forward pass through the S3 decoder using soft (continuous)
    token representations instead of discrete token IDs.

    Instead of:
        embeds = decoder.token_embedding(tgt_ids)          # lookup discrete tokens

    We do:
        probs  = softmax(y_soft_logits / temp_in)          # (B, L, V) — soft distribution
        embeds = probs @ decoder.token_embedding.weight    # (B, L, D) — weighted sum of embeddings

    This makes the entire forward pass differentiable w.r.t. y_soft_logits,
    so gradients can flow back to epsilon during the Langevin loop.

    Args:
        decoder        : your S3 Decoder (frozen weights, but we compute grads through it)
        y_soft_logits  : (B, L, V)  — current soft token logits (y_logits + epsilon)
        z_T            : (B, D)     — fixed JEPA latent context
        temp_in        : temperature for converting logits → soft probs (lower = harder)

    Returns:
        logits_out     : (B, L, V)  — next-token logit predictions (fluency signal)

    NOTE: Your Decoder must expose:
        - decoder.token_embedding   : nn.Embedding(vocab_size, hidden_size)
        - decoder.pos_embedding     : positional encoding module
        - decoder.layers            : transformer decoder layers with cross-attn to z_T
        - decoder.norm / decoder.head : final layernorm + lm head projection

    If your Decoder has a different internal structure, adjust the body of this
    function to match — the key operation is the soft embedding line marked below.
    """
    B, L, V = y_soft_logits.shape

    # ── Soft embedding lookup (the key differentiable step) ──────────────────
    soft_probs = F.softmax(y_soft_logits / temp_in, dim=-1)          # (B, L, V)
    embeds = soft_probs @ decoder.token_embedding.weight             # (B, L, D)
    # ─────────────────────────────────────────────────────────────────────────

    # Add positional encoding (assume decoder exposes this)
    embeds = decoder.pos_embedding(embeds)                           # (B, L, D)

    # z_T is (B, D) — expand to (B, 1, D) so cross-attention can attend to it
    # as a single "memory" token.  If your decoder already handles this shape,
    # remove the unsqueeze.
    z_ctx = z_T.unsqueeze(1)                                         # (B, 1, D)

    # Build causal mask for self-attention
    causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=embeds.device)

    # Pass through transformer decoder layers
    # Adjust this block if your Decoder uses a different API
    hidden = embeds
    for layer in decoder.layers:
        hidden = layer(
            tgt      = hidden,
            memory   = z_ctx,
            tgt_mask = causal_mask,
        )

    # Final norm + projection to vocab
    if hasattr(decoder, 'norm'):
        hidden = decoder.norm(hidden)
    logits_out = decoder.head(hidden)                                 # (B, L, V)

    return logits_out










# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Constraint losses
# ═══════════════════════════════════════════════════════════════════════════════

def keyword_loss_maxprob(y_soft_logits, keyword_token_ids):
    """
    Hard lexical constraint loss.

    For each required keyword (given as a list of token IDs after BPE),
    find the position in the output that gives the highest probability to
    that token, and penalize when that probability is low.

    Loss = -sum_k  log( max_t  softmax(y_logits)[t, token_id_k] )

    Gradient pushes the soft vector at the best-matching position to
    concentrate probability mass on the required token.

    Args:
        y_soft_logits    : (B, L, V)
        keyword_token_ids: list of int  — one token ID per required keyword
                           (use the first subword piece if a word splits into multiple)

    Returns:
        loss : scalar
    """
    probs = F.softmax(y_soft_logits, dim=-1)     # (B, L, V)
    total = 0.0
    for tok_id in keyword_token_ids:
        # probability assigned to this keyword token at each output position
        tok_probs = probs[:, :, tok_id]          # (B, L)
        # best position for this token across the sequence
        max_prob, _ = tok_probs.max(dim=-1)      # (B,)
        total = total + (-torch.log(max_prob + 1e-9)).mean()
    return total / max(len(keyword_token_ids), 1)


def keyword_loss_bleu1(y_soft_logits, keyword_token_ids, temp=1.0):
    """
    Soft BLEU-1 keyword coverage loss (mirrors c_loss_2 in original COLD).

    Encourages at least one position per keyword to have high probability.
    Slightly softer than maxprob — better for multi-subword keywords.

    Args:
        y_soft_logits    : (B, L, V)
        keyword_token_ids: list of int
        temp             : temperature for soft coverage

    Returns:
        loss : scalar
    """
    probs = F.softmax(y_soft_logits / temp, dim=-1)   # (B, L, V)
    total = 0.0
    for tok_id in keyword_token_ids:
        coverage = probs[:, :, tok_id].sum(dim=-1)    # (B,)  sum over positions
        coverage = coverage.clamp(max=1.0)            # cap at 1 (one inclusion is enough)
        total = total + (-torch.log(coverage + 1e-9)).mean()
    return total / max(len(keyword_token_ids), 1)
















# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Initialization
# ═══════════════════════════════════════════════════════════════════════════════

def initialize_logits(decoder, z_T, length, init_temp, device):
    """
    Get initial soft logits ỹ⁽⁰⁾ by running a greedy forward pass.

    Starts from [CLS]=101, autoregressively generates `length` tokens,
    collects the raw logits at each step.

    Returns:
        init_logits : (1, length, V)  — initial soft sequence (batch=1)
    """
    decoder.eval()
    prompt = torch.tensor([[101]], dtype=torch.long, device=device)  # [CLS]
    all_logits = []

    with torch.no_grad():
        cur = prompt
        for _ in range(length):
            # Use teacher-forcing style: pass what we have so far
            # z_T is (1, D)
            logits = decoder(cur, z_T, attention_mask=None)  # (1, t, V)
            next_logit = logits[:, -1:, :]                   # (1, 1, V)
            all_logits.append(next_logit)
            next_tok = next_logit.argmax(-1)                 # greedy
            cur = torch.cat([cur, next_tok], dim=1)

    init_logits = torch.cat(all_logits, dim=1)               # (1, length, V)
    return init_logits / init_temp











# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Discretization
# ═══════════════════════════════════════════════════════════════════════════════

def discretize(y_soft_logits, tokenizer, keyword_token_ids=None, topk=10):
    """
    Convert soft logits → discrete token IDs.

    For positions that have a keyword token in their top-k, force that token.
    For all other positions, take the argmax.

    Args:
        y_soft_logits    : (B, L, V)
        tokenizer        : BERT tokenizer
        keyword_token_ids: list of int  — tokens to force-include
        topk             : int          — candidate window size

    Returns:
        token_ids : (B, L)  — discrete output
        texts     : list[str]
    """
    B, L, V = y_soft_logits.shape
    logits = y_soft_logits.detach().clone()

    if keyword_token_ids and topk > 0:
        # Build keyword vocab mask  (1, 1, V) broadcast over all positions
        z_mask = torch.zeros(V, device=logits.device)
        z_mask[keyword_token_ids] = 1e4          # boost keyword token scores
        logits = logits + z_mask.unsqueeze(0).unsqueeze(0)

    token_ids = logits.argmax(-1)                # (B, L)

    texts = []
    for b in range(B):
        toks = token_ids[b].tolist()
        text = tokenizer.decode(toks, skip_special_tokens=True)
        texts.append(text)

    return token_ids, texts












# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — Main COLD loop
# ═══════════════════════════════════════════════════════════════════════════════

def cold_decode(
    decoder,
    z_T,
    constraint_words,           # list[str]  e.g. ["Paris", "hotel", "reservation"]
    tokenizer,
    length          = 30,       # output token length
    num_iters       = 500,      # Langevin iterations
    stepsize        = 0.1,      # Adam lr for epsilon
    constraint_w    = 0.2,      # λ : weight of constraint vs fluency
    input_temp      = 1.0,      # temperature for soft input to fluency model
    output_temp     = 1.0,      # temperature for fluency model output
    init_temp       = 0.1,      # temperature for initialization
    gs_mean         = 0.0,      # Langevin noise mean
    gs_std          = 0.01,     # Langevin noise std
    topk            = 10,       # top-k for discretization
    print_every     = 100,
    verbose         = True,
    device          = DEVICE,
):
    """
    Run COLD constrained decoding using the S3 decoder as the fluency model.

    z_T is fixed throughout — we only optimize epsilon (the perturbation on
    the soft output sequence).

    Energy function:
        E(ỹ) = (1 - λ) * fluency_loss(ỹ)  +  λ * keyword_loss(ỹ)

    where fluency_loss is the NLL of ỹ under the decoder (conditioned on z_T),
    and keyword_loss forces the required tokens to appear somewhere in ỹ.

    Args:
        decoder          : loaded S3 Decoder (weights frozen — grads flow through,
                           but we only update epsilon)
        z_T              : (1, D) or (B, D) — fixed JEPA latent
        constraint_words : list of strings — words that MUST appear in output
        tokenizer        : BERT WordPiece tokenizer
        ... (see defaults above)

    Returns:
        final_text  : list[str]  — decoded output for each batch item
        token_ids   : (B, L)    — discrete token sequence
    """

    # ── 1. Tokenize constraint words ──────────────────────────────────────────
    #
    # BERT tokenizer adds [CLS]=101 and [SEP]=102 around each word.
    # We strip those and take only the actual subword pieces.
    # For multi-subword words (e.g. "reservation" → ["reservation"]),
    # we take only the FIRST piece as the anchor for the keyword loss.
    # This is conservative but avoids fighting for every subword simultaneously.
    #
    keyword_token_ids = []
    for word in constraint_words:
        ids = tokenizer.encode(word, add_special_tokens=True)
        # ids = [101, tok1, tok2, ..., 102]  — strip CLS and SEP
        word_toks = ids[1:-1]
        if word_toks:
            keyword_token_ids.append(word_toks[0])   # anchor on first subword
            if verbose:
                pieces = tokenizer.convert_ids_to_tokens(word_toks)
                print(f"  Keyword '{word}' → tokens {word_toks} → {pieces} (anchor: {word_toks[0]})")

    if not keyword_token_ids:
        raise ValueError("No valid keyword tokens found. Check constraint_words and tokenizer.")

    # ── 2. Initialize soft sequence ỹ⁽⁰⁾ ────────────────────────────────────
    B = z_T.shape[0]
    assert B == 1, "COLD inference currently supports batch_size=1. Expand z_T if needed."

    y_logits = initialize_logits(decoder, z_T, length, init_temp, device)
    # y_logits: (1, length, V)

    if verbose:
        _, init_texts = discretize(y_logits, tokenizer, keyword_token_ids, topk)
        print(f"\n[init] {init_texts[0]}")

    # ── 3. Learnable perturbation epsilon ────────────────────────────────────
    #
    # We do NOT optimize decoder weights — they stay frozen.
    # We only optimize epsilon, a small additive perturbation on y_logits.
    # The effective soft sequence at each step is:  ỹ = y_logits + epsilon
    #
    epsilon = nn.Parameter(torch.zeros_like(y_logits))
    optim   = torch.optim.Adam([epsilon], lr=stepsize)

    # Freeze all decoder parameters (just to be safe — grads flow through, not into)
    for p in decoder.parameters():
        p.requires_grad = False

    # ── 4. Langevin dynamics loop ─────────────────────────────────────────────
    for it in range(num_iters):
        optim.zero_grad()

        y_logits_ = y_logits + epsilon    # (1, L, V)  — current soft sequence

        # ── Fluency loss ─────────────────────────────────────────────────────
        #
        # Run a soft forward pass: the decoder attends to z_T and predicts
        # next-token distributions given the soft input sequence.
        # The fluency loss is the NLL between predicted distribution and input.
        #
        # This is the "decoder knows what it wants to say given z_T" signal.
        # If ỹ is incoherent with z_T, this loss is high.
        #
        fluency_logits = soft_forward_decoder(
            decoder, y_logits_ / 0.001, z_T, temp_in=0.001)   # (1, L, V)

        # Shift: predict token[t+1] from token[t]
        pred  = fluency_logits[:, :-1, :] / output_temp        # (1, L-1, V)
        target = F.softmax(y_logits_[:, 1:, :] / input_temp, dim=-1)  # (1, L-1, V)

        # Soft NLL: KL-like cross-entropy between predicted and soft target
        fluency_loss = -(target * F.log_softmax(pred, dim=-1)).sum(-1).mean()

        # ── Keyword constraint loss ───────────────────────────────────────────
        #
        # Forces probability mass at some position to concentrate on each
        # required keyword token. Gradient pushes y_logits toward including them.
        #
        kw_loss = keyword_loss_maxprob(y_logits_, keyword_token_ids)

        # ── Combined energy ───────────────────────────────────────────────────
        loss = (1.0 - constraint_w) * fluency_loss + constraint_w * kw_loss

        # ── Backward + update epsilon ─────────────────────────────────────────
        if it < num_iters - 1:
            loss.backward()
            optim.step()

        # ── Logging ───────────────────────────────────────────────────────────
        if verbose and (it == 0 or (it + 1) % print_every == 0 or it == num_iters - 1):
            with torch.no_grad():
                _, cur_texts = discretize(y_logits_, tokenizer, keyword_token_ids, topk)
            print(f"  iter {it+1:4d} | loss={loss.item():.4f} "
                  f"fluency={fluency_loss.item():.4f} kw={kw_loss.item():.4f} "
                  f"| {cur_texts[0]}")

        # ── Langevin noise injection ───────────────────────────────────────────
        #
        # This is what makes it Langevin dynamics (not just gradient descent).
        # Adding Gaussian noise prevents getting stuck in local optima and
        # allows exploring the constraint-satisfying region of the space.
        #
        if it < num_iters - 1:
            noise   = torch.normal(mean=gs_mean, std=gs_std,
                                   size=epsilon.shape, device=device)
            y_logits = (y_logits + noise).detach()
            epsilon  = nn.Parameter(torch.zeros_like(y_logits))
            optim    = torch.optim.Adam([epsilon], lr=stepsize)

    # ── 5. Discretization ─────────────────────────────────────────────────────
    with torch.no_grad():
        final_logits = (y_logits + epsilon).detach()
        token_ids, final_texts = discretize(
            final_logits, tokenizer, keyword_token_ids, topk)

    if verbose:
        print(f"\n[final] {final_texts[0]}")

    return final_texts, token_ids












# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — Model loading helpers
# ═══════════════════════════════════════════════════════════════════════════════

def mean_pool(hidden, mask):
    mask_f = mask.unsqueeze(-1).float()
    return (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)


def load_models(
    s1_ckpt_path = None,   # defaults to CFG.training.s1_ckpt
    s2_ckpt_path = '/content/drive/MyDrive/metanet/v5/s2/checkpoints/best.pt',
    s3_ckpt_path = '/content/drive/MyDrive/metanet/v5/s3/checkpoints/best.pt',
    device       = DEVICE,
):
    """Load and freeze S1, S2, S3. Return (s1_encoder, predictor, decoder)."""

    # ── S1 Encoder ────────────────────────────────────────────────────────────
    s1_encoder = Encoder(
        vocab_size  = CFG.model.vocab_size,
        hidden_size = CFG.model.hidden_size,
        num_heads   = CFG.model.num_heads,
        num_layers  = CFG.model.num_layers,
        max_seq_len = CFG.model.max_seq_len,
    ).to(device)
    ckpt = torch.load(s1_ckpt_path or CFG.training.s1_ckpt,
                      map_location=device, weights_only=False)
    s1_encoder.load_state_dict(ckpt['context_encoder'])
    s1_encoder.eval()
    for p in s1_encoder.parameters(): p.requires_grad = False

    # ── S2 Predictor ──────────────────────────────────────────────────────────
    predictor = DM(
        num_frames = CFG.model.pred_num_frames,
        depth      = CFG.model.pred_num_layers,
        heads      = CFG.model.pred_num_heads,
        mlp_dim    = CFG.model.pred_hidden_size * 4,
        input_dim  = CFG.model.hidden_size,
        hidden_dim = CFG.model.pred_hidden_size,
        output_dim = CFG.model.hidden_size,
        dim_head   = 64, dropout=0.1, emb_dropout=0.1,
    ).to(device)
    s2_ckpt = torch.load(s2_ckpt_path, map_location=device, weights_only=False)
    predictor.load_state_dict(s2_ckpt['predictor'])
    predictor.eval()
    for p in predictor.parameters(): p.requires_grad = False

    # ── S3 Decoder ────────────────────────────────────────────────────────────
    decoder = Decoder(
        vocab_size   = CFG.model.vocab_size,
        hidden_size  = 256,
        num_heads    = 4,
        num_layers   = 4,
        max_seq_len  = CFG.model.max_seq_len,
        context_dim  = CFG.model.hidden_size,
    ).to(device)
    s3_ckpt = torch.load(s3_ckpt_path, map_location=device, weights_only=False)
    decoder.load_state_dict(s3_ckpt['decoder'])
    decoder.eval()
    # NOTE: we do NOT call requires_grad=False on decoder params here,
    # because soft_forward_decoder needs to propagate gradients THROUGH
    # the decoder's computation graph (to reach epsilon).
    # We just never call optimizer.step() on decoder params.

    print(f"S1, S2, S3 loaded on {device}.")
    return s1_encoder, predictor, decoder


def encode_history(s1_encoder, predictor, history_ids, history_masks, device):
    """
    Given a batch of conversation history turns, produce z_pred (the predicted
    next-turn latent) using S1 + S2.

    Args:
        history_ids   : (B, T, L)  — token IDs for each of T history turns
        history_masks : (B, T, L)  — attention masks

    Returns:
        z_pred : (B, D)
    """
    with torch.no_grad():
        B, T, L = history_ids.shape
        flat_ids   = history_ids.view(B*T, L).to(device)
        flat_masks = history_masks.view(B*T, L).to(device)
        valid      = flat_masks.sum(-1) > 0
        D          = s1_encoder.token_embedding.embedding_dim
        z_flat     = torch.zeros(B*T, D, device=device)

        h = s1_encoder(flat_ids[valid], attention_mask=flat_masks[valid])
        if isinstance(h, tuple): h = h[0]
        z_flat[valid] = mean_pool(h, flat_masks[valid])
        z_seq = z_flat.view(B, T, -1)

        z_pred = predictor(z_seq, torch.zeros_like(z_seq))[:, -1, :]
    return z_pred   # (B, D)











# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — End-to-end inference entry point
# ═══════════════════════════════════════════════════════════════════════════════

def generate_constrained(
    history_turns,       # list[str]  — previous dialog turns
    constraint_words,    # list[str]  — keywords that must appear in the response
    s1_encoder, predictor, decoder,
    tokenizer,
    max_history_turns = 5,
    output_length     = 30,
    cold_iters        = 500,
    constraint_weight = 0.2,
    device            = DEVICE,
    verbose           = True,
):
    """
    Full pipeline: history → z_pred → COLD constrained decoding.

    Example:
        texts, ids = generate_constrained(
            history_turns    = ["Hi!", "What can I help you with?", "I need a hotel."],
            constraint_words = ["Paris", "hotel", "reservation"],
            ...
        )
    """
    # ── Tokenize history ──────────────────────────────────────────────────────
    max_len = CFG.model.max_seq_len
    turns   = history_turns[-max_history_turns:]  # keep last N turns
    T       = len(turns)

    ids_list, masks_list = [], []
    for turn in turns:
        enc = tokenizer(turn, max_length=max_len, padding='max_length',
                        truncation=True, return_tensors='pt')
        ids_list.append(enc['input_ids'])
        masks_list.append(enc['attention_mask'])

    # (1, T, L)
    hist_ids   = torch.stack(ids_list,   dim=1)   # (1, T, L)
    hist_masks = torch.stack(masks_list, dim=1)

    # ── Get z_pred from S1 + S2 ───────────────────────────────────────────────
    z_pred = encode_history(s1_encoder, predictor, hist_ids, hist_masks, device)
    # z_pred: (1, D)

    if verbose:
        print(f"\nz_pred shape: {z_pred.shape}  norm: {z_pred.norm().item():.3f}")
        print(f"Constraints: {constraint_words}")

    # ── COLD decoding ─────────────────────────────────────────────────────────
    texts, token_ids = cold_decode(
        decoder          = decoder,
        z_T              = z_pred,
        constraint_words = constraint_words,
        tokenizer        = tokenizer,
        length           = output_length,
        num_iters        = cold_iters,
        constraint_w     = constraint_weight,
        device           = device,
        verbose          = verbose,
    )

    return texts, token_ids












# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — Quick test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Loading models...")
    s1_enc, pred, dec = load_models()

    history = [
        "Hello, I need some help planning a trip.",
        "Sure, where would you like to go?",
        "I am thinking about Europe.",
    ]
    keywords = ["Paris", "hotel"]

    texts, ids = generate_constrained(
        history_turns    = history,
        constraint_words = keywords,
        s1_encoder       = s1_enc,
        predictor        = pred,
        decoder          = dec,
        tokenizer        = tokenizer,
        output_length    = 30,
        cold_iters       = 500,
        constraint_weight= 0.2,
        verbose          = True,
    )

    print("\n=== Final output ===")
    print(texts[0])