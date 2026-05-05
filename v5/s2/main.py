#!/usr/bin/env python3
"""
Stage 2: Predictor training (Turn Pair Prediction)

Changes vs. original:
  1. BUG FIX  — target turn is now encoded with `target_encoder`, not
                `context_encoder`.
  2. OPTION B — target representation is token-level embeddings (B, L, D),
                not a mean-pooled vector.  The DM still operates on the
                mean-pooled history sequence (B, T, D); a new TurnExpander
                maps its (B, D) output to (B, L, D) for comparison.
  3. CLEAN UP — zeroed conditioning removed; DM is called with a proper
                zeros tensor that acts as a no-op (no code change needed
                there, but noted).
"""

import sys
import os
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = '/content/notebooks_meta'
S1   = f'{ROOT}/v5/s1'
S2   = f'{ROOT}/v5/s2'
sys.path.insert(0, ROOT)

# ── Config ────────────────────────────────────────────────────────────────────
from v5.s2.config import CFG, DEVICE

# ── Architectures ─────────────────────────────────────────────────────────────
from v5.s1.cog_arch.encoder   import Encoder
from v5.s2.cog_arch.dm        import DM
from v5.s2.cog_arch.expander  import TurnExpander   # NEW

# ── Data ──────────────────────────────────────────────────────────────────────
from v5.s2.data.dataset     import VOCAB_SIZE, tokenizer
from v5.s2.data.dataloader  import get_stage2_dataloaders

# ══════════════════════════════════════════════════════════════════════════════
# Dataloaders
# ══════════════════════════════════════════════════════════════════════════════

train_loader, val_loader = get_stage2_dataloaders(cfg_obj=CFG, tokenizer=tokenizer)
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

# ══════════════════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════════════════

context_encoder = Encoder(
    vocab_size  = CFG.model.vocab_size,
    hidden_size = CFG.model.hidden_size,
    num_heads   = CFG.model.num_heads,
    num_layers  = CFG.model.num_layers,
    max_seq_len = CFG.model.max_seq_len,
).to(DEVICE)

target_encoder = copy.deepcopy(context_encoder).to(DEVICE)

# ── Load Stage 1 checkpoint ───────────────────────────────────────────────────
s1_ckpt = torch.load(CFG.training.s1_ckpt, map_location=DEVICE, weights_only=False)
context_encoder.load_state_dict(s1_ckpt['context_encoder'])
target_encoder.load_state_dict(s1_ckpt['target_encoder'])
print(f"✓ S1 checkpoint loaded (epoch {s1_ckpt.get('epoch', '?')})")

# Freeze both encoders — S2 only trains the predictor + expander
for enc in (context_encoder, target_encoder):
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
print("Encoders frozen.")

# ── DM predictor (unchanged architecture) ─────────────────────────────────────
predictor = DM(
    num_frames = CFG.model.pred_num_frames,
    depth      = CFG.model.pred_num_layers,
    heads      = CFG.model.pred_num_heads,
    mlp_dim    = CFG.model.pred_hidden_size * 4,
    input_dim  = CFG.model.hidden_size,
    hidden_dim = CFG.model.pred_hidden_size,
    output_dim = CFG.model.hidden_size,   # outputs D, not L*D
    dim_head   = 64,
    dropout    = 0.1,
    emb_dropout= 0.1,
).to(DEVICE)

# ── TurnExpander (NEW) ─────────────────────────────────────────────────────────
# Maps the (B, D) DM output back to (B, L, D) token-level embeddings so we can
# compare directly against the target encoder's output sequence.
expander = TurnExpander(
    hidden_dim  = CFG.model.hidden_size,
    max_seq_len = CFG.model.max_seq_len,     # must match block_size / L
    num_heads   = CFG.model.num_heads,       # reuse encoder head count
    num_layers  = 2,
    dropout     = 0.1,
).to(DEVICE)

print(f"Predictor params  : {sum(p.numel() for p in predictor.parameters()):,}")
print(f"Expander params   : {sum(p.numel() for p in expander.parameters()):,}")

# ══════════════════════════════════════════════════════════════════════════════
# Loss / Optimizer / Scheduler
# ══════════════════════════════════════════════════════════════════════════════

trainable_params = list(predictor.parameters()) + list(expander.parameters())
optimizer = AdamW(trainable_params, lr=CFG.optim.lr, weight_decay=CFG.optim.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CFG.optim.epochs, eta_min=CFG.optim.lr * 0.1)

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token dim using attention mask. Returns (B, D)."""
    mask_f = mask.unsqueeze(-1).float()                           # (B, L, 1)
    return (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)


def masked_token_mse(
    pred: torch.Tensor,   # (B, L, D)
    tgt:  torch.Tensor,   # (B, L, D)
    mask: torch.Tensor,   # (B, L)  — 1 = real token, 0 = padding
) -> torch.Tensor:
    """
    MSE over real (non-padding) token positions only.
    Averaging over tokens then over batch gives equal weight to every example
    regardless of sequence length.
    """
    mask_f  = mask.unsqueeze(-1).float()                          # (B, L, 1)
    diff_sq = (pred - tgt).pow(2)                                 # (B, L, D)
    # per-token mean over D, then mean over real tokens, then mean over batch
    per_tok = diff_sq.mean(dim=-1)                                # (B, L)
    per_ex  = (per_tok * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)  # (B,)
    return per_ex.mean()


# ══════════════════════════════════════════════════════════════════════════════
# Forward step
# ══════════════════════════════════════════════════════════════════════════════

def forward_step(batch: dict) -> dict:
    """
    Single JEPA S2 forward pass.

    History path  (context_encoder, mean-pool)  → (B, T, D)
    Predictor     (DM)                           → (B, D)
    Expander      (TurnExpander)                 → (B, L, D)  z_pred

    Target path   (target_encoder, NO pool)      → (B, L, D)  z_tgt
                   ^^^^^^^^^^^^^^^^^^^
                   FIX: was context_encoder; must be target_encoder.

    Loss: masked MSE between z_pred and z_tgt over non-padding positions.
    """
    hist_ids   = batch['history_ids'].to(DEVICE)    # (B, T, L)
    hist_masks = batch['history_masks'].to(DEVICE)  # (B, T, L)
    tgt_ids    = batch['tgt_ids'].to(DEVICE)        # (B, L)
    tgt_mask   = batch['tgt_mask'].to(DEVICE)       # (B, L)

    B, T, L = hist_ids.shape
    D = context_encoder.token_embedding.embedding_dim

    # ── History: encode and mean-pool each turn ───────────────────────────────
    with torch.no_grad():
        flat_ids   = hist_ids.view(B * T, L)
        flat_masks = hist_masks.view(B * T, L)
        valid      = flat_masks.sum(-1) > 0              # skip all-padding turns

        z_flat = torch.zeros(B * T, D, device=DEVICE)
        if valid.any():
            h = context_encoder(flat_ids[valid], attention_mask=flat_masks[valid])
            if isinstance(h, tuple):
                h = h[0]
            z_flat[valid] = mean_pool(h, flat_masks[valid])

        z_seq = z_flat.view(B, T, D)                    # (B, T, D)

        # ── Target: token-level embeddings from TARGET encoder ────────────────
        # FIX: was context_encoder — must be target_encoder for stable targets.
        tgt_h = target_encoder(tgt_ids, attention_mask=tgt_mask)  # (B, L, D)
        if isinstance(tgt_h, tuple):
            tgt_h = tgt_h[0]
        z_tgt = tgt_h                                   # (B, L, D)  — NO mean_pool

    # ── Predictor: (B, T, D) → (B, D) ────────────────────────────────────────
    # DM returns (B, T, D); we take the last position as the next-turn prediction.
    # The zero conditioning tensor is a no-op (AdaLN weights initialised to zero).
    z_pred_vec = predictor(z_seq, torch.zeros_like(z_seq))[:, -1, :]  # (B, D)

    # ── Expander: (B, D) → (B, L, D) ─────────────────────────────────────────
    z_pred = expander(z_pred_vec, tgt_mask)             # (B, L, D)

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss = masked_token_mse(z_pred, z_tgt, tgt_mask)

    return {'loss': loss}


# ══════════════════════════════════════════════════════════════════════════════
# Checkpointing
# ══════════════════════════════════════════════════════════════════════════════

def save_best(epoch: int, val_loss: float) -> None:
    path = Path(CFG.logging.exp_dir) / 'best.pt'
    torch.save({
        'epoch':     epoch,
        'predictor': predictor.state_dict(),
        'expander':  expander.state_dict(),      # NEW — save expander too
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'val_loss':  val_loss,
    }, path)
    print(f'  ✓ saved → {path}')


def load_checkpoint(path: Path) -> tuple[int, float]:
    """Returns (start_epoch, best_val_loss)."""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    predictor.load_state_dict(ckpt['predictor'])
    if 'expander' in ckpt:
        expander.load_state_dict(ckpt['expander'])
    else:
        print('  ⚠ checkpoint has no expander weights — starting expander from scratch')
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    print(f'  ✓ resumed from epoch {ckpt["epoch"]}  val_loss={ckpt["val_loss"]:.4f}')
    return ckpt['epoch'] + 1, ckpt['val_loss']


# ══════════════════════════════════════════════════════════════════════════════
# Eval / Train loops
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validation_loop() -> dict:
    predictor.eval()
    expander.eval()
    totals: dict = {}
    n = 0
    for batch in val_loader:
        d = forward_step(batch)
        for k, v in d.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n += 1
    predictor.train()
    expander.train()
    return {k: v / n for k, v in totals.items()}


def train_one_epoch(epoch: int) -> dict:
    predictor.train()
    expander.train()
    totals: dict = {}
    n = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch:02d}', leave=False)
    for batch in pbar:
        d = forward_step(batch)
        d['loss'].backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        optimizer.zero_grad()
        for k, v in d.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n += 1
        pbar.set_postfix({k: f'{v.item():.4f}' for k, v in d.items()})
    return {k: v / n for k, v in totals.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ══════════════════════════════════════════════════════════════════════════════

history = {'train_loss': [], 'val_loss': []}
best_val_loss = float('inf')
Path(CFG.logging.exp_dir).mkdir(parents=True, exist_ok=True)

best_ckpt = Path(CFG.logging.exp_dir) / 'best.pt'
start_epoch = 1
if best_ckpt.exists():
    start_epoch, best_val_loss = load_checkpoint(best_ckpt)
    print(f'Resuming from epoch {start_epoch}')
else:
    print('No checkpoint found, starting from scratch.')

print(f'\n{"="*60}')
print(f'  Text JEPA S2 — {CFG.optim.epochs} epochs   device={DEVICE}')
print(f'{"="*60}\n')

for epoch in range(start_epoch, CFG.optim.epochs + 1):
    tr = train_one_epoch(epoch)
    vl = validation_loop()
    scheduler.step()

    history['train_loss'].append(tr['loss'])
    history['val_loss'].append(vl['loss'])

    print(
        f'Epoch {epoch:02d}/{CFG.optim.epochs}  '
        f'train={tr["loss"]:.4f}  val={vl["loss"]:.4f}  '
        f'lr={optimizer.param_groups[0]["lr"]:.2e}'
    )

    if vl['loss'] < best_val_loss:
        best_val_loss = vl['loss']
        save_best(epoch, best_val_loss)
        print(f'  ★ new best val_loss={best_val_loss:.4f}')

print('\nStage 2 complete.')

# ── Plotting ──────────────────────────────────────────────────────────────────
epochs_range = list(range(1, len(history['train_loss']) + 1))
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(epochs_range, history['train_loss'], 'b-o', label='Train', markersize=4)
ax.plot(epochs_range, history['val_loss'],   'r-s', label='Val',   markersize=4)
ax.set_title('Stage 2 — MSE loss (token embeddings)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Masked token MSE')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

plot_path = Path(CFG.logging.exp_dir) / 'training_curves_stage2.png'
plt.savefig(plot_path, dpi=150)
plt.close(fig)
print(f'Plot saved → {plot_path}')
print(f'Best val_loss: {best_val_loss:.4f}')