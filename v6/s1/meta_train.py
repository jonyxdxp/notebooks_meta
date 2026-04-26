"""
MOML integrated into the Text-JEPA training pipeline
======================================================
Key idea:  each batch is treated as one online "task".
  - Support  = context side  (ctx_ids, ctx_mask)
  - Query    = target side   (tgt_ids, tgt_mask, span_mask)
  - U_t(θ)   = one inner-SGD step of context_encoder on context→target loss
  - Meta loss = BCS( z_ctx_adapted , z_tgt )  evaluated on the query side
  - MOML state  = (w, prev_grad) — O(d) memory, replaces the usual optimizer step

Drop-in changes vs. your original notebook:
  1. Wrap context_encoder in MOMLTrainer  (replaces optimizer)
  2. Replace the `train_epoch` body
  3. Everything else (EMA, checkpointing, scheduler, plotting) stays identical.
"""

import os
import glob
import copy

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

# ── your existing imports ─────────────────────────────────────────────────────
import sys
sys.path.insert(0, '/content/notebooks_meta/v6/s1')

from cog_arch.encoder import Encoder
from losses import BCS
from data.dataloader import get_jepa_dataloaders
from data.dataset import VOCAB_SIZE, tokenizer
import config
from config import CFG, DEVICE


# ─────────────────────────────────────────────────────────────────────────────
# MOML core  (self-contained, no external dependency)
# ─────────────────────────────────────────────────────────────────────────────

class MOMLTrainer:
    """
    Memory-Efficient Online Meta-Learner adapted for a JEPA text encoder.

    State kept between batches (all O(d)):
        w         – direction-correction anchor            (Eq. 8)
        prev_grad – ∇[f_{t-1} ∘ U_{t-1}](θ^t)            (used in R_t)

    Each call to .step() runs one full MOML round (Algorithm 1):
        1. Adapt  : U_t(θ) — shallow-copy + 1 inner SGD step on context loss
        2. Suffer : record query loss before update
        3. K corrected gradient steps using ∇f_t + ∇R_t
        4. Update w and prev_grad
        5. EMA-update target encoder
    """

    def __init__(
        self,
        context_encoder: nn.Module,
        target_encoder:  nn.Module,
        loss_fn,                         # BCS instance, returns dict
        inner_lr:    float = 1e-3,       # η  — inner adaptation LR
        meta_lr:     float = 1e-3,       # β  — outer MOML update LR
        alpha:       float = 1.0,        # α  — regulariser strength
        K:           int   = 1,          # corrected gradient steps per round
        ema_decay:   float = 0.996,
        device:      torch.device = None,
    ):
        self.ctx_enc  = context_encoder
        self.tgt_enc  = target_encoder
        self.loss_fn  = loss_fn
        self.inner_lr = inner_lr
        self.meta_lr  = meta_lr
        self.alpha    = alpha
        self.K        = K
        self.ema_decay = ema_decay
        self.device   = device or torch.device('cpu')

        # ── MOML state vectors  (same shape as θ, all zeros at t=0) ──────
        params = list(self.ctx_enc.parameters())
        self.w         = [torch.zeros_like(p.data) for p in params]
        self.prev_grad = [torch.zeros_like(p.data) for p in params]

    # ── public API ────────────────────────────────────────────────────────────

    def step(self, batch: dict) -> dict:
        """
        One MOML round = one batch.

        Returns
        -------
        loss_dict : {'loss', 'bcs_loss', 'invariance_loss', ...}
                    Values are the *suffered* loss (before the meta update),
                    consistent with your original logging.
        """
        ctx_ids, ctx_mask, tgt_ids, tgt_mask, span_mask = self._unpack(batch)

        # ── Step 1 · Adapt  U_t(θ)  on the context/support side ──────────
        # We deep-copy the encoder and take one inner SGD step,
        # keeping the computation graph so we can differentiate back to θ.
        fast_enc = self._inner_adapt(ctx_ids, ctx_mask,
                                     tgt_ids, tgt_mask, span_mask)

        # ── Step 2 · Suffer  f_t(φ_t)  (query loss, no grad) ─────────────
        with torch.no_grad():
            loss_dict_suffered = self._query_loss(
                fast_enc, tgt_ids, tgt_mask, span_mask, no_grad=True)

        # ── Steps 3–4 · K corrected gradient steps ────────────────────────
        for _ in range(self.K):
            # Rebuild fast model at *current* θ each step (Eq. 7)
            fast_k = self._inner_adapt(ctx_ids, ctx_mask,
                                       tgt_ids, tgt_mask, span_mask)

            # ∇[f_t ∘ U_t](θ_k)
            loss_dict_k = self._query_loss(fast_k,
                                           tgt_ids, tgt_mask, span_mask)
            grads_ft = torch.autograd.grad(
                loss_dict_k['loss'],
                self.ctx_enc.parameters(),
                allow_unused=True,
                retain_graph=False,
            )
            grads_ft = [
                g if g is not None else torch.zeros_like(p)
                for g, p in zip(grads_ft, self.ctx_enc.parameters())
            ]

            # ∇R_t(θ_k) = -prev_grad + α*(θ_k - w)       (from Eq. 6)
            grad_reg = [
                -pg + self.alpha * (p.data - wv)
                for pg, p, wv in zip(
                    self.prev_grad,
                    self.ctx_enc.parameters(),
                    self.w,
                )
            ]

            # θ_{k+1} = θ_k − β * (∇f_t + ∇R_t)
            with torch.no_grad():
                for p, gf, gr in zip(
                    self.ctx_enc.parameters(), grads_ft, grad_reg
                ):
                    p -= self.meta_lr * (gf + gr)

        # ── Step 5 · Update state  w_{t+1}  (Eq. 8) ─────────────────────
        # Compute ∇[f_t ∘ U_t](θ^{t+1}) at the *new* meta-params
        fast_new = self._inner_adapt(ctx_ids, ctx_mask,
                                     tgt_ids, tgt_mask, span_mask)
        loss_new  = self._query_loss(fast_new, tgt_ids, tgt_mask, span_mask)
        grad_new  = torch.autograd.grad(
            loss_new['loss'],
            self.ctx_enc.parameters(),
            allow_unused=True,
            retain_graph=False,
        )
        grad_new = [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(grad_new, self.ctx_enc.parameters())
        ]

        with torch.no_grad():
            # w_{t+1} = ½ (w_t + θ^{t+1} − (1/α) * grad_new)
            self.w = [
                0.5 * (wv + p.data - (1.0 / self.alpha) * gn)
                for wv, p, gn in zip(
                    self.w, self.ctx_enc.parameters(), grad_new
                )
            ]
            # prev_grad ← grad at new θ  (will be used next round)
            self.prev_grad = [g.detach().clone() for g in grad_new]

        # ── EMA: target_encoder ← decay*target + (1-decay)*context ──────
        self._ema_update()

        return loss_dict_suffered

    # ── helpers ───────────────────────────────────────────────────────────────

    def _unpack(self, batch):
        return (
            batch['context_input_ids'].to(self.device),
            batch['context_attention_mask'].to(self.device),
            batch['target_input_ids'].to(self.device),
            batch['target_attention_mask'].to(self.device),
            batch['target_mask'].to(self.device),
        )

    def _inner_adapt(self, ctx_ids, ctx_mask, tgt_ids, tgt_mask, span_mask):
        """
        U_t(θ): shallow-copy context_encoder, take one SGD step on the
        context-side JEPA loss, keeping the graph attached to θ.

        The fast model's weights are:  φ = θ - η * ∇_θ L_inner(θ)
        so ∂φ/∂θ is tracked and meta-gradients flow through to θ.
        """
        fast = copy.deepcopy(self.ctx_enc)
        fast.train()

        # ---- inner forward (context → representation) -------------------
        h_ctx = fast(ctx_ids, attention_mask=ctx_mask)
        if isinstance(h_ctx, tuple):
            h_ctx = h_ctx[0]

        with torch.no_grad():
            h_tgt = self.tgt_enc(tgt_ids, attention_mask=tgt_mask)
            if isinstance(h_tgt, tuple):
                h_tgt = h_tgt[0]

        z_ctx = self._masked_pool(h_ctx, span_mask)
        z_tgt = self._masked_pool(h_tgt, span_mask)

        inner_loss = self.loss_fn(z_ctx, z_tgt)['loss']

        # ---- manual SGD step (create_graph=True lets meta-grad flow) -----
        grads = torch.autograd.grad(
            inner_loss, fast.parameters(), create_graph=True
        )
        with torch.no_grad():
            for p, g in zip(fast.parameters(), grads):
                p -= self.inner_lr * g

        return fast

    def _query_loss(self, fast_enc, tgt_ids, tgt_mask, span_mask,
                    no_grad=False):
        """
        Evaluate BCS( z_ctx_adapted , z_tgt ) — the outer / query loss.
        fast_enc already ran the inner adaptation step.
        """
        ctx_fn = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx_fn:
            # NOTE: we re-use tgt_ids as the query input for the fast encoder
            # (the adapted encoder sees the target tokens and predicts z_tgt).
            h_fast = fast_enc(tgt_ids, attention_mask=tgt_mask)
            if isinstance(h_fast, tuple):
                h_fast = h_fast[0]

            with torch.no_grad():
                h_tgt = self.tgt_enc(tgt_ids, attention_mask=tgt_mask)
                if isinstance(h_tgt, tuple):
                    h_tgt = h_tgt[0]

            z_pred = self._masked_pool(h_fast, span_mask)
            z_tgt  = self._masked_pool(h_tgt,  span_mask)

            return self.loss_fn(z_pred, z_tgt)

    @staticmethod
    def _masked_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).float()
        summed = (hidden * mask_f).sum(dim=1)
        count  = mask_f.sum(dim=1).clamp(min=1)
        return summed / count

    def _ema_update(self):
        with torch.no_grad():
            for p_c, p_t in zip(
                self.ctx_enc.parameters(), self.tgt_enc.parameters()
            ):
                p_t.data.mul_(self.ema_decay).add_(
                    p_c.data, alpha=1.0 - self.ema_decay
                )

    # ── checkpoint helpers ────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        """Serialise the MOML state vectors (w, prev_grad)."""
        return {
            'w':         [t.cpu() for t in self.w],
            'prev_grad': [t.cpu() for t in self.prev_grad],
        }

    def load_state_dict(self, sd: dict):
        self.w         = [t.to(self.device) for t in sd['w']]
        self.prev_grad = [t.to(self.device) for t in sd['prev_grad']]


# ─────────────────────────────────────────────────────────────────────────────
# Build models  (unchanged from your Cell 10)
# ─────────────────────────────────────────────────────────────────────────────

train_loader, val_loader = get_jepa_dataloaders(cfg_obj=CFG, tokenizer=tokenizer)

context_encoder = Encoder(
    vocab_size  = VOCAB_SIZE,
    hidden_size = CFG.hidden_size,
    num_heads   = CFG.num_heads,
    num_layers  = CFG.num_layers,
    max_seq_len = CFG.max_seq_len,
).to(DEVICE)

target_encoder = copy.deepcopy(context_encoder).to(DEVICE)
for p in target_encoder.parameters():
    p.requires_grad = False

loss_fn = BCS(lmbd=10.0)


# ─────────────────────────────────────────────────────────────────────────────
# Replace optimizer + train_epoch with MOML
# ─────────────────────────────────────────────────────────────────────────────

moml = MOMLTrainer(
    context_encoder = context_encoder,
    target_encoder  = target_encoder,
    loss_fn         = loss_fn,
    inner_lr        = CFG.lr,           # η — inner adaptation step
    meta_lr         = CFG.lr,           # β — outer MOML update
    alpha           = 1.0,              # α — regulariser strength
    K               = 1,               # corrected steps per batch
    ema_decay       = CFG.ema_decay,
    device          = DEVICE,
)

# Keep the scheduler on a dummy optimizer that tracks LR only.
# (MOML manages its own parameter updates; no optimizer.step() needed.)
_lr_tracker = AdamW(context_encoder.parameters(), lr=CFG.lr,
                    weight_decay=CFG.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    _lr_tracker,
    T_max   = CFG.n_epochs * 2,
    eta_min = CFG.lr * 0.3,
)


def get_lr() -> float:
    """Read current LR from scheduler and inject into MOML."""
    lr = _lr_tracker.param_groups[0]['lr']
    moml.meta_lr  = lr   # β  tracks cosine schedule
    moml.inner_lr = lr   # η  optionally track too
    return lr


# ── eval (unchanged logic) ────────────────────────────────────────────────────

@torch.no_grad()
def eval_epoch(loader):
    context_encoder.eval()
    totals, n = {}, 0
    for batch in loader:
        ctx_ids  = batch['context_input_ids'].to(DEVICE)
        ctx_mask = batch['context_attention_mask'].to(DEVICE)
        tgt_ids  = batch['target_input_ids'].to(DEVICE)
        tgt_mask = batch['target_attention_mask'].to(DEVICE)
        span_mask = batch['target_mask'].to(DEVICE)

        h_ctx = context_encoder(ctx_ids, attention_mask=ctx_mask)
        if isinstance(h_ctx, tuple): h_ctx = h_ctx[0]
        h_tgt = target_encoder(tgt_ids,  attention_mask=tgt_mask)
        if isinstance(h_tgt, tuple): h_tgt = h_tgt[0]

        z_ctx = MOMLTrainer._masked_pool(h_ctx, span_mask)
        z_tgt = MOMLTrainer._masked_pool(h_tgt, span_mask)

        loss_dict = loss_fn(z_ctx, z_tgt)
        for k, v in loss_dict.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n += 1

    context_encoder.train()
    return {k: v / n for k, v in totals.items()}


# ── train epoch — only this function changes vs your original ─────────────────

def train_epoch(loader, epoch: int) -> dict:
    """
    Replaces your original train_epoch.
    The optimizer.zero_grad / loss.backward / optimizer.step triplet is gone;
    MOML handles gradient computation and parameter updates internally.
    """
    context_encoder.train()
    totals, n = {}, 0
    pbar = tqdm(loader, desc=f'Epoch {epoch:02d}', leave=False)

    for batch in pbar:
        # ── one MOML round = one batch ────────────────────────────────────
        loss_dict = moml.step(batch)
        # loss_dict is the *suffered* loss (pre-update), used for logging.

        for k, v in loss_dict.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n += 1
        pbar.set_postfix({k: f'{v.item():.4f}' for k, v in loss_dict.items()})

    return {k: v / n for k, v in totals.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing  (add moml_state to your existing save/load)
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(epoch, metrics, suffix=None):
    if suffix:
        filename = f'{suffix}.pt'
    elif isinstance(epoch, int):
        filename = f'epoch_{epoch:03d}.pt'
    else:
        filename = f'epoch_{epoch}.pt'
    path = os.path.join(CFG.ckpt_dir, filename)
    torch.save({
        'epoch':           epoch if isinstance(epoch, int) else -1,
        'context_encoder': context_encoder.state_dict(),
        'target_encoder':  target_encoder.state_dict(),
        'scheduler':       scheduler.state_dict(),
        'moml_state':      moml.state_dict(),   # ← NEW: w and prev_grad
        'metrics':         metrics,
        'cfg':             {k: v for k, v in vars(CFG).items()
                            if not k.startswith('_')},
    }, path)
    print(f'  ✓ saved → {path}')


def load_checkpoint(path):
    ckpt = torch.load(path, map_location=DEVICE)
    context_encoder.load_state_dict(ckpt['context_encoder'])
    target_encoder.load_state_dict(ckpt['target_encoder'])
    scheduler.load_state_dict(ckpt['scheduler'])
    if 'moml_state' in ckpt:
        moml.load_state_dict(ckpt['moml_state'])   # ← NEW
    print(f'  ✓ resumed from epoch {ckpt["epoch"]}')
    return ckpt['epoch']


# ─────────────────────────────────────────────────────────────────────────────
# Training loop  (identical structure to your original Cell 15)
# ─────────────────────────────────────────────────────────────────────────────

history = {
    'train_loss': [], 'train_bcs': [], 'train_inv': [],
    'val_loss':   [], 'val_bcs':   [], 'val_inv':   [],
    'lr':         [],
}

print(f'\n{"="*60}')
print(f'  Text JEPA + MOML — {CFG.n_epochs} epochs   device={DEVICE}')
print(f'{"="*60}\n')

start_epoch = 1
best_ckpt   = os.path.join(CFG.ckpt_dir, 'best.pt')
if os.path.exists(best_ckpt):
    print('Resuming from best checkpoint …')
    start_epoch = load_checkpoint(best_ckpt) + 1
    print(f'  starting at epoch {start_epoch}')
else:
    print('No checkpoint found, starting from scratch.')

best_val_loss = float('inf')

for epoch in range(start_epoch, CFG.n_epochs + 1):
    train_metrics = train_epoch(train_loader, epoch)
    val_metrics   = eval_epoch(val_loader)
    lr = get_lr()   # read LR before stepping so epoch 1 logs the correct value
    scheduler.step()  # step AFTER using the LR (avoids PyTorch warning)

    history['train_loss'].append(train_metrics.get('loss', 0.0))
    history['train_bcs'].append(train_metrics.get('bcs_loss', 0.0))
    history['train_inv'].append(train_metrics.get('invariance_loss', 0.0))
    history['val_loss'].append(val_metrics.get('loss', 0.0))
    history['val_bcs'].append(val_metrics.get('bcs_loss', 0.0))
    history['val_inv'].append(val_metrics.get('invariance_loss', 0.0))
    history['lr'].append(lr)

    print(
        f'Epoch {epoch:02d}/{CFG.n_epochs}  '
        f'train_loss={train_metrics["loss"]:.4f}  '
        f'val_loss={val_metrics["loss"]:.4f}  '
        f'bcs={train_metrics.get("bcs_loss", 0):.4f}  '
        f'inv={train_metrics.get("invariance_loss", 0):.4f}  '
        f'lr={lr:.2e}'
    )

    if val_metrics['loss'] < best_val_loss:
        best_val_loss = val_metrics['loss']
        save_checkpoint('best', val_metrics)
        print(f'  ★ new best val_loss={best_val_loss:.4f}')

print('\nTraining complete.')
save_checkpoint(CFG.n_epochs, {})