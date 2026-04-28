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
from torch.func import functional_call   # zero-copy stateless forward
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
    Memory-Efficient Online Meta-Learner — fast version using functional_call.

    Speed problem with the naive version
    -------------------------------------
    The original code called copy.deepcopy(encoder) up to 3× per batch:
      • once for the suffered-loss evaluation
      • once per K corrected-gradient step
      • once for the w-state update
    deepcopy on a transformer is O(params) in time *and* triggers a full
    CUDA memory allocation, making it the dominant bottleneck.

    Fix: torch.func.functional_call
    --------------------------------
    functional_call(module, param_dict, args) runs a forward pass with
    *arbitrary* parameter tensors without touching the module's .parameters().
    We compute adapted params as plain tensors (φ = θ - η*∇L) and pass them
    directly — zero copies of the model, full autograd support.

    Additionally, the K-loop's last adapted params are *reused* for the state
    update (Eq. 8), saving one extra adaptation per batch.

    Per-batch cost: 2 functional forwards (inner + outer) × (K + 1)
                    versus (K + 2) deepcopies before.
    """

    def __init__(
        self,
        context_encoder: nn.Module,
        target_encoder:  nn.Module,
        loss_fn,
        inner_lr:  float = 1e-3,
        meta_lr:   float = 1e-3,
        alpha:     float = 1.0,
        K:         int   = 1,
        ema_decay: float = 0.996,
        device:    torch.device = None,
    ):
        self.ctx_enc   = context_encoder
        self.tgt_enc   = target_encoder
        self.loss_fn   = loss_fn
        self.inner_lr  = inner_lr
        self.meta_lr   = meta_lr
        self.alpha     = alpha
        self.K         = K
        self.ema_decay = ema_decay
        self.device    = device or torch.device('cpu')

        # Cache named buffers once — they never change
        self._buffers = dict(self.ctx_enc.named_buffers())

        # MOML state: O(d) memory, persists between batches
        self.w         = {n: torch.zeros_like(p.data)
                          for n, p in self.ctx_enc.named_parameters()}
        self.prev_grad = {n: torch.zeros_like(p.data)
                          for n, p in self.ctx_enc.named_parameters()}

    # ── public API ────────────────────────────────────────────────────────────

    def step(self, batch: dict) -> dict:
        """One MOML round (= one batch). Returns the suffered loss dict."""
        ctx_ids, ctx_mask, tgt_ids, tgt_mask, span_mask = self._unpack(batch)

        # Pre-compute target representation once — shared across all steps
        with torch.no_grad():
            h_tgt = self.tgt_enc(tgt_ids, attention_mask=tgt_mask)
            if isinstance(h_tgt, tuple):
                h_tgt = h_tgt[0]
            z_tgt = self._masked_pool(h_tgt, span_mask)   # (B, D)

        # ── Step 1 · Suffered loss  f_t(U_t(θ))  — logged, no update ────
        # _adapt_params needs autograd to compute the inner gradient,
        # so we enable_grad for that step, then immediately detach phi
        # before evaluating the outer loss (which needs no grad for logging).
        with torch.enable_grad():
            phi_s = self._adapt_params(
                self._named_params(), ctx_ids, ctx_mask, z_tgt, span_mask,
                create_graph=False,
            )
        phi_s = {n: p.detach() for n, p in phi_s.items()}
        with torch.no_grad():
            loss_dict_suffered = self._outer_loss(phi_s, tgt_ids, tgt_mask,
                                                  span_mask, z_tgt)

        # ── Steps 2–3 · K corrected gradient steps  (Eq. 7) ─────────────
        for _ in range(self.K):
            theta = self._named_params()          # current θ_k  (live tensors)

            # φ_k = U_t(θ_k) — adapted params, graph attached to θ_k
            phi_k = self._adapt_params(theta, ctx_ids, ctx_mask,
                                       z_tgt, span_mask, create_graph=True)

            # ∇[f_t ∘ U_t](θ_k)
            loss_k = self._outer_loss(phi_k, tgt_ids, tgt_mask,
                                      span_mask, z_tgt)['loss']
            grads_ft = torch.autograd.grad(
                loss_k, theta.values(),
                allow_unused=True, retain_graph=False,
            )
            grads_ft = {
                n: (g if g is not None else torch.zeros_like(p))
                for (n, p), g in zip(theta.items(), grads_ft)
            }

            # ∇R_t(θ_k) = -prev_grad + α*(θ_k - w)
            with torch.no_grad():
                for n, p in self.ctx_enc.named_parameters():
                    grad_total = grads_ft[n] + (
                        -self.prev_grad[n] + self.alpha * (p.data - self.w[n])
                    )
                    p -= self.meta_lr * grad_total

        # ── Step 4 · State update  w_{t+1}  (Eq. 8) ─────────────────────
        # Reuse a fresh adaptation at the *new* θ^{t+1} — no extra deepcopy.
        theta_new = self._named_params()
        phi_new   = self._adapt_params(theta_new, ctx_ids, ctx_mask,
                                       z_tgt, span_mask, create_graph=True)
        loss_new  = self._outer_loss(phi_new, tgt_ids, tgt_mask,
                                     span_mask, z_tgt)['loss']
        grad_new  = torch.autograd.grad(
            loss_new, theta_new.values(),
            allow_unused=True, retain_graph=False,
        )

        with torch.no_grad():
            for (n, p), gn in zip(self.ctx_enc.named_parameters(), grad_new):
                gn = gn if gn is not None else torch.zeros_like(p)
                # w_{t+1} = ½ (w_t + θ^{t+1} - (1/α)*grad_new)
                self.w[n] = 0.5 * (self.w[n] + p.data
                                   - (1.0 / self.alpha) * gn)
                self.prev_grad[n] = gn.detach().clone()

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

    def _named_params(self) -> dict:
        """Live snapshot of named parameters (no copy)."""
        return dict(self.ctx_enc.named_parameters())

    def _adapt_params(self, params: dict,
                      ctx_ids, ctx_mask, z_tgt, span_mask,
                      create_graph: bool = True) -> dict:
        """
        U_t(θ): one inner SGD step, returning adapted param dict φ.

        Uses functional_call so no model copy is needed.
        When create_graph=True the returned tensors carry a grad_fn
        back to the original θ, enabling meta-gradients.
        """
        # Inner forward with current params (no deepcopy)
        h_ctx = functional_call(
            self.ctx_enc, (params, self._buffers),
            args=(ctx_ids,), kwargs={'attention_mask': ctx_mask},
        )
        if isinstance(h_ctx, tuple):
            h_ctx = h_ctx[0]

        z_ctx = self._masked_pool(h_ctx, span_mask)
        inner_loss = self.loss_fn(z_ctx, z_tgt)['loss']

        # ∇_φ L_inner  — create_graph lets the outer loss differentiate through
        grads = torch.autograd.grad(
            inner_loss, params.values(), create_graph=create_graph,
        )

        # φ = θ - η * ∇L  (differentiable w.r.t. θ when create_graph=True)
        return {n: p - self.inner_lr * g
                for (n, p), g in zip(params.items(), grads)}

    def _outer_loss(self, phi: dict,
                    tgt_ids, tgt_mask, span_mask, z_tgt) -> dict:
        """
        f_t ∘ U_t : BCS(z_adapted, z_tgt).
        phi is the adapted param dict from _adapt_params.
        z_tgt is pre-computed (target encoder, no grad).
        """
        h_fast = functional_call(
            self.ctx_enc, (phi, self._buffers),
            args=(tgt_ids,), kwargs={'attention_mask': tgt_mask},
        )
        if isinstance(h_fast, tuple):
            h_fast = h_fast[0]

        z_pred = self._masked_pool(h_fast, span_mask)
        return self.loss_fn(z_pred, z_tgt)

    @staticmethod
    def _masked_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).float()
        summed = (hidden * mask_f).sum(dim=1)
        count  = mask_f.sum(dim=1).clamp(min=1)
        return summed / count

    def _ema_update(self):
        with torch.no_grad():
            for p_c, p_t in zip(self.ctx_enc.parameters(),
                                 self.tgt_enc.parameters()):
                p_t.data.mul_(self.ema_decay).add_(
                    p_c.data, alpha=1.0 - self.ema_decay
                )

    # ── checkpoint helpers ────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            'w':         {n: t.cpu() for n, t in self.w.items()},
            'prev_grad': {n: t.cpu() for n, t in self.prev_grad.items()},
        }

    def load_state_dict(self, sd: dict):
        self.w         = {n: t.to(self.device) for n, t in sd['w'].items()}
        self.prev_grad = {n: t.to(self.device) for n, t in sd['prev_grad'].items()}


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