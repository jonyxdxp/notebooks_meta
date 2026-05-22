"""
Test-Time Training (TTT) for the MOML-trained Text-JEPA Encoder
================================================================

Conceptual flow
---------------
                     ┌──────────────────────────────────────┐
  User input         │         TTT inner loop               │
  (raw string)  ───► │  1. tokenize + mask (JEPA style)     │
                     │  2. N gradient steps on BCS loss      │
                     │     using θ* as starting point        │
                     │  3. φ_task = adapted params           │
                     └──────────────────────────────────────┘
                                      │
                                      ▼
                     encode full input with φ_task
                     pool at masked span positions
                                      │
                                      ▼
                              z  (B, D)  ← utterance representation

Why this works
--------------
MOML trained θ* so that the inner loop is maximally informative — a few
gradient steps on any new input's self-supervised JEPA signal quickly
specialises the encoder to that input's distributional properties.
The masked-span pooling is used both at train time and at test time, so
there is no train/test mismatch.

The auxiliary task (JEPA span prediction via BCS) requires NO labels —
only the raw input tokens. This makes the TTT loop fully unsupervised.

Usage
-----
    from ttt_encoder import TTTEncoder

    enc = TTTEncoder.from_checkpoint(
        ckpt_path   = 'best.pt',
        encoder_cls = Encoder,
        encoder_cfg = dict(vocab_size=30522, hidden_size=256,
                           num_heads=4, num_layers=4, max_seq_len=128),
        loss_fn     = BCS(lmbd=10.0),
        tokenizer   = tokenizer,
        device      = torch.device('cuda'),
    )

    z = enc.encode("Hey, I'm feeling really overwhelmed lately.")
    # z : (1, 256)  — ready for retrieval / downstream tasks
"""

import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.func import functional_call







# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sample_spans(
    valid_len: int,
    num_spans: int,
    span_len:  int,
    maskable_start: int = 1,   # skip [CLS]
) -> List[Tuple[int, int]]:
    """
    Sample non-overlapping random spans inside [maskable_start, valid_len-1).
    Mirrors JEPAMaskCollator._sample_spans exactly.
    """
    maskable_end = valid_len - 1   # skip [SEP]
    region_len   = maskable_end - maskable_start
    if region_len <= 0:
        return []

    span_len  = min(span_len, region_len)
    available = list(range(maskable_start, maskable_end - span_len + 1))
    spans     = []

    for _ in range(num_spans):
        if not available:
            break
        idx  = random.randint(0, len(available) - 1)
        s, e = available[idx], available[idx] + span_len
        spans.append((s, e))
        available = [x for x in available if (x + span_len <= s) or (x >= e)]

    return spans


def _make_jepa_batch(
    text:          str,
    tokenizer,
    mask_token_id: int,
    max_seq_len:   int,
    num_spans:     int,
    span_len:      int,
    device:        torch.device,
    seed:          Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tokenize one string and produce a JEPA context/target batch of size 1.

    Returns
    -------
    ctx_ids   : (1, L)  — masked tokens (context)
    ctx_mask  : (1, L)  — attention mask
    tgt_ids   : (1, L)  — original tokens (target)
    tgt_mask  : (1, L)  — attention mask (same as ctx_mask)
    span_mask : (1, L)  — bool, True at masked positions
    """
    if seed is not None:
        random.seed(seed)

    enc = tokenizer(
        text,
        max_length        = max_seq_len,
        padding           = 'max_length',
        truncation        = True,
        add_special_tokens= True,
        return_attention_mask = True,
        return_tensors    = 'pt',
    )
    tgt_ids  = enc['input_ids']        # (1, L)
    attn     = enc['attention_mask']   # (1, L)
    ctx_ids  = tgt_ids.clone()

    valid_len  = int(attn[0].sum().item())
    span_mask  = torch.zeros(1, max_seq_len, dtype=torch.bool)

    for s, e in _sample_spans(valid_len, num_spans, span_len):
        span_mask[0, s:e] = True

    ctx_ids[span_mask] = mask_token_id

    return (
        ctx_ids.to(device),
        attn.to(device),
        tgt_ids.to(device),
        attn.to(device),
        span_mask.to(device),
    )


def _masked_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool hidden states at masked (span) positions.

    hidden : (B, L, D)
    mask   : (B, L)   bool
    returns: (B, D)
    """
    mask_f = mask.unsqueeze(-1).float()           # (B, L, 1)
    summed = (hidden * mask_f).sum(dim=1)         # (B, D)
    count  = mask_f.sum(dim=1).clamp(min=1)       # (B, 1)
    return summed / count











# ─────────────────────────────────────────────────────────────────────────────
# TTTEncoder
# ─────────────────────────────────────────────────────────────────────────────

class TTTEncoder:
    """
    Test-Time Training wrapper around a MOML-pretrained encoder.

    For each call to .encode():
      1. The raw input is tokenized and span-masked (same as training).
      2. N inner-loop gradient steps adapt θ* → φ_task on the JEPA
         self-supervised objective (no labels required).
      3. The full input is encoded with φ_task and pooled at the masked
         positions to produce the utterance representation z.

    The original θ* is NEVER modified — adaptation is purely functional,
    using torch.func.functional_call, so .encode() is stateless and safe
    to call in any order or from multiple threads.

    Parameters
    ----------
    encoder       : the MOML-trained context encoder (nn.Module)
    loss_fn       : BCS instance (same as used during training)
    tokenizer     : BERT tokenizer
    inner_lr      : η — same value used in MOML training
    num_steps     : N inner-loop gradient steps at test time (1–5 typical)
    num_spans     : number of masked spans (matches JEPAMaskCollator)
    span_len      : length of each span   (matches JEPAMaskCollator)
    max_seq_len   : tokenizer max length  (matches CFG.max_seq_len)
    device        : torch.device
    return_mode   : 'span'  — pool at masked positions (default, train-aligned)
                    'mean'  — pool over all non-pad tokens
                    'cls'   — return the [CLS] token vector
    """

    def __init__(
        self,
        encoder:      nn.Module,
        loss_fn,
        tokenizer,
        inner_lr:     float = 1e-4,
        num_steps:    int   = 3,
        num_spans:    int   = 4,
        span_len:     int   = 8,
        max_seq_len:  int   = 128,
        device:       Optional[torch.device] = None,
        return_mode:  str   = 'span',
    ):
        self.encoder     = encoder
        self.loss_fn     = loss_fn
        self.tokenizer   = tokenizer
        self.inner_lr    = inner_lr
        self.num_steps   = num_steps
        self.num_spans   = num_spans
        self.span_len    = span_len
        self.max_seq_len = max_seq_len
        self.device      = device or torch.device('cpu')
        self.return_mode = return_mode

        assert return_mode in ('span', 'mean', 'cls'), \
            f"return_mode must be 'span', 'mean', or 'cls', got '{return_mode}'"

        # Cache buffers — never change, shared across all adapt() calls
        self._buffers = dict(self.encoder.named_buffers())

        # Freeze base params — θ* is read-only
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    # ── main API ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(
        self,
        text:       str,
        num_steps:  Optional[int] = None,
        seed:       Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encode a single string with TTT inner-loop adaptation.

        Parameters
        ----------
        text      : raw input string (the user's utterance)
        num_steps : override self.num_steps for this call
        seed      : fix random span sampling for reproducibility

        Returns
        -------
        z : (1, D) tensor — the adapted utterance representation
        """
        steps = num_steps if num_steps is not None else self.num_steps

        ctx_ids, ctx_mask, tgt_ids, tgt_mask, span_mask = _make_jepa_batch(
            text          = text,
            tokenizer     = self.tokenizer,
            mask_token_id = self.tokenizer.mask_token_id,
            max_seq_len   = self.max_seq_len,
            num_spans     = self.num_spans,
            span_len      = self.span_len,
            device        = self.device,
            seed          = seed,
        )

        # θ* as the starting parameter dict
        phi = {n: p.clone() for n, p in self.encoder.named_parameters()}

        # ── TTT inner loop ────────────────────────────────────────────────────
        phi = self._adapt(phi, ctx_ids, ctx_mask, span_mask, steps)

        # ── encode + pool with adapted params ─────────────────────────────────
        z = self._encode_with(phi, tgt_ids, tgt_mask, span_mask, ctx_ids, ctx_mask)
        return z

    def encode_batch(
        self,
        texts:      List[str],
        num_steps:  Optional[int] = None,
        seed:       Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encode a list of strings independently (each gets its own adaptation).

        Returns
        -------
        Z : (N, D) tensor
        """
        return torch.cat([
            self.encode(t, num_steps=num_steps, seed=seed)
            for t in texts
        ], dim=0)

    # ── inner loop ────────────────────────────────────────────────────────────

    def _adapt(
        self,
        phi:       dict,
        ctx_ids:   torch.Tensor,
        ctx_mask:  torch.Tensor,
        span_mask: torch.Tensor,
        steps:     int,
    ) -> dict:
        """
        Run `steps` gradient steps of the JEPA self-supervised objective,
        returning updated param dict φ_task.

        Each step:
          1. Forward: functional_call(encoder, φ, context_tokens)
          2. Pool at span positions → z_ctx
          3. Forward: functional_call(encoder, φ, full_tokens) — the target
             representation is produced by the *same* (adapted) encoder
             on the unmasked tokens (since we have no separate target encoder
             at test time).
          4. BCS(z_ctx, z_tgt) → loss
          5. φ ← φ - η * ∇_φ loss
        """
        for step_i in range(steps):
            # Enable grad just for this step (encoder params are normally frozen)
            phi = {n: p.requires_grad_(True) for n, p in phi.items()}

            # ── context forward (masked) ──────────────────────────────────────
            h_ctx = functional_call(
                self.encoder, (phi, self._buffers),
                args=(ctx_ids,), kwargs={'attention_mask': ctx_mask},
            )
            if isinstance(h_ctx, tuple):
                h_ctx = h_ctx[0]
            z_ctx = _masked_pool(h_ctx, span_mask)   # (1, D)

            # ── target forward (unmasked, same adapted params) ────────────────
            # At test time we don't have a separate EMA target encoder.
            # We use a stop-gradient copy of φ as the target — this matches
            # the Bootstrap-style SSL used in several TTT-SSL works and avoids
            # the degenerate z_ctx == z_tgt collapse.
            with torch.no_grad():
                phi_sg = {n: p.detach() for n, p in phi.items()}
                h_tgt  = functional_call(
                    self.encoder, (phi_sg, self._buffers),
                    args=(ctx_ids,),   # same masked context — predict from self
                    kwargs={'attention_mask': ctx_mask},
                )
                if isinstance(h_tgt, tuple):
                    h_tgt = h_tgt[0]
                # Use the UNMASKED positions as the target signal
                unmask = ~span_mask
                z_tgt  = _masked_pool(h_tgt, unmask)   # (1, D)

            # ── BCS loss + gradient step ──────────────────────────────────────
            loss = self.loss_fn(z_ctx, z_tgt)['loss']

            grads = torch.autograd.grad(loss, phi.values())

            phi = {
                n: (p - self.inner_lr * g).detach().requires_grad_(False)
                for (n, p), g in zip(phi.items(), grads)
            }

        return phi

    # ── encoding with adapted params ──────────────────────────────────────────

    @torch.no_grad()
    def _encode_with(
        self,
        phi:       dict,
        tgt_ids:   torch.Tensor,
        tgt_mask:  torch.Tensor,
        span_mask: torch.Tensor,
        ctx_ids:   torch.Tensor,
        ctx_mask:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Final forward pass with φ_task on the FULL (unmasked) input.
        Pooling strategy is controlled by self.return_mode.
        """
        h = functional_call(
            self.encoder, (phi, self._buffers),
            args=(tgt_ids,), kwargs={'attention_mask': tgt_mask},
        )
        if isinstance(h, tuple):
            h = h[0]   # (1, L, D)

        if self.return_mode == 'span':
            # Pool at the same span positions used for adaptation
            # Train/test aligned: encoder learned to produce good reps here
            return _masked_pool(h, span_mask)           # (1, D)

        elif self.return_mode == 'mean':
            # Pool over all non-padding positions
            non_pad = tgt_mask.bool()
            return _masked_pool(h, non_pad)             # (1, D)

        else:  # 'cls'
            return h[:, 0, :]                           # (1, D)

    # ── convenience: load from checkpoint ────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path:   str,
        encoder_cls,
        encoder_cfg: dict,
        loss_fn,
        tokenizer,
        inner_lr:    float = 1e-4,
        num_steps:   int   = 3,
        num_spans:   int   = 4,
        span_len:    int   = 8,
        max_seq_len: int   = 128,
        device:      Optional[torch.device] = None,
        return_mode: str   = 'span',
    ) -> 'TTTEncoder':
        """
        Build a TTTEncoder directly from a MOML checkpoint.

        Example
        -------
        enc = TTTEncoder.from_checkpoint(
            ckpt_path   = '/content/drive/.../best.pt',
            encoder_cls = Encoder,
            encoder_cfg = dict(
                vocab_size  = 30522,
                hidden_size = 256,
                num_heads   = 4,
                num_layers  = 4,
                max_seq_len = 128,
            ),
            loss_fn     = BCS(lmbd=10.0),
            tokenizer   = tokenizer,
            device      = torch.device('cuda'),
        )
        """
        device = device or torch.device('cpu')

        encoder = encoder_cls(**encoder_cfg).to(device)
        ckpt    = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(ckpt['context_encoder'])
        encoder.eval()

        print(f'  ✓ loaded encoder from {ckpt_path}  '
              f'(epoch {ckpt.get("epoch", "?")})')

        return cls(
            encoder     = encoder,
            loss_fn     = loss_fn,
            tokenizer   = tokenizer,
            inner_lr    = inner_lr,
            num_steps   = num_steps,
            num_spans   = num_spans,
            span_len    = span_len,
            max_seq_len = max_seq_len,
            device      = device,
            return_mode = return_mode,
        )

    # ── diagnostics ───────────────────────────────────────────────────────────

    def probe(self, text: str, max_steps: int = 10) -> None:
        """
        Print loss at each inner-loop step for a given input.
        Useful for choosing num_steps and inner_lr.
        """
        ctx_ids, ctx_mask, tgt_ids, tgt_mask, span_mask = _make_jepa_batch(
            text          = text,
            tokenizer     = self.tokenizer,
            mask_token_id = self.tokenizer.mask_token_id,
            max_seq_len   = self.max_seq_len,
            num_spans     = self.num_spans,
            span_len      = self.span_len,
            device        = self.device,
        )

        phi = {n: p.clone() for n, p in self.encoder.named_parameters()}

        print(f'\nTTT probe — "{text[:60]}…"')
        print(f'  {"step":>4}  {"loss":>10}  {"Δ":>10}')
        print(f'  {"-"*30}')

        prev_loss = None
        for i in range(max_steps):
            phi = {n: p.requires_grad_(True) for n, p in phi.items()}

            h_ctx = functional_call(
                self.encoder, (phi, self._buffers),
                args=(ctx_ids,), kwargs={'attention_mask': ctx_mask},
            )
            if isinstance(h_ctx, tuple): h_ctx = h_ctx[0]
            z_ctx = _masked_pool(h_ctx, span_mask)

            with torch.no_grad():
                phi_sg = {n: p.detach() for n, p in phi.items()}
                h_tgt  = functional_call(
                    self.encoder, (phi_sg, self._buffers),
                    args=(ctx_ids,), kwargs={'attention_mask': ctx_mask},
                )
                if isinstance(h_tgt, tuple): h_tgt = h_tgt[0]
                z_tgt = _masked_pool(h_tgt, ~span_mask)

            loss  = self.loss_fn(z_ctx, z_tgt)['loss']
            delta = f'{loss.item() - prev_loss:+.6f}' if prev_loss else '       —'
            print(f'  {i:>4}  {loss.item():>10.6f}  {delta:>10}')
            prev_loss = loss.item()

            grads = torch.autograd.grad(loss, phi.values())
            phi   = {
                n: (p - self.inner_lr * g).detach().requires_grad_(False)
                for (n, p), g in zip(phi.items(), grads)
            }
        print()














# ─────────────────────────────────────────────────────────────────────────────
# Quick usage example (run as script to verify)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/content/notebooks_meta/v6/s1')

    from cog_arch.encoder import Encoder
    from losses import BCS
    from data.dataset import tokenizer
    from config import CFG, DEVICE

    CKPT = '/content/drive/MyDrive/metanet/v6/s1/checkpoints/best.pt'

    enc = TTTEncoder.from_checkpoint(
        ckpt_path   = CKPT,
        encoder_cls = Encoder,
        encoder_cfg = dict(
            vocab_size  = 30522,
            hidden_size = CFG.hidden_size,
            num_heads   = CFG.num_heads,
            num_layers  = CFG.num_layers,
            max_seq_len = CFG.max_seq_len,
        ),
        loss_fn     = BCS(lmbd=10.0),
        tokenizer   = tokenizer,
        inner_lr    = CFG.lr,
        num_steps   = 3,
        num_spans   = CFG.num_target_spans,
        span_len    = CFG.target_span_length,
        max_seq_len = CFG.max_seq_len,
        device      = DEVICE,
        return_mode = 'span',
    )

    # ── probe: watch loss decrease over inner-loop steps ─────────────────────
    test_inputs = [
        "Hey, I'm feeling really overwhelmed lately.",
        "Can you help me book a flight to Paris next Tuesday?",
        "I haven't spoken to my sister in three years.",
    ]

    for text in test_inputs:
        enc.probe(text, max_steps=7)

    # ── encode ────────────────────────────────────────────────────────────────
    Z = enc.encode_batch(test_inputs, num_steps=3)
    print(f'Output shape : {Z.shape}')           # (3, 256)
    print(f'Norm per row : {Z.norm(dim=-1)}')

    # ── similarity sanity check ───────────────────────────────────────────────
    Z_norm = Z / Z.norm(dim=-1, keepdim=True)
    sim    = Z_norm @ Z_norm.T
    print(f'\nCosine similarity matrix:')
    print(sim.cpu().numpy().round(3))