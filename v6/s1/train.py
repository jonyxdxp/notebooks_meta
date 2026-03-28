
# ── Cell 3: Imports ───────────────────────────────────────────────────────────


import copy
import sys
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


sys.path.insert(0, '/content/notebooks_meta/v6/s1')

from cog_arch.encoder import Encoder
from losses import BCS   # BCS kept as optional alternative



from data.dataloader import get_jepa_dataloaders
from data.dataset import VOCAB_SIZE, tokenizer


import config






# ── Cell 9: Build dataloaders ─────────────────────────────────────────────────

train_loader, val_loader = get_jepa_dataloaders(
    cfg_obj    = CFG,
    tokenizer  = tokenizer,
)

print(f"Train batches : {len(train_loader)}  |  Val batches : {len(val_loader)}")







# ── Cell 10: Models ───────────────────────────────────────────────────────────

context_encoder = Encoder(
    vocab_size   = VOCAB_SIZE,
    hidden_size  = CFG.hidden_size,
    num_heads    = CFG.num_heads,
    num_layers   = CFG.num_layers,
    max_seq_len  = CFG.max_seq_len,
).to(DEVICE)

target_encoder = copy.deepcopy(context_encoder).to(DEVICE)
for p in target_encoder.parameters():
    p.requires_grad = False   # updated only via EMA

print(f"Params (context encoder) : {sum(p.numel() for p in context_encoder.parameters()):,}")








# ── Cell 11: Loss / optimizer / scheduler ────────────────────────────────────

# loss_fn = VICRegLoss(std_coeff=CFG.std_coeff, cov_coeff=CFG.cov_coeff)

loss_fn = BCS(lmbd=10.0)   # lmbd controls Gaussianity regularization vs invariance



optimizer = AdamW(
    context_encoder.parameters(),
    lr           = CFG.lr,
    weight_decay = CFG.weight_decay,
)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max  = CFG.n_epochs * 2,   # was n_epochs — decays over 2x the actual run
    eta_min = CFG.lr * 0.3,      # was 0.1 — don't let it drop as low
)












# ── Cell 12: Helpers ──────────────────────────────────────────────────────────

def ema_update(ctx_enc, tgt_enc, decay=CFG.ema_decay):
    """Exponential moving average: tgt ← decay*tgt + (1-decay)*ctx"""
    with torch.no_grad():
        for p_c, p_t in zip(ctx_enc.parameters(), tgt_enc.parameters()):
            p_t.data.mul_(decay).add_(p_c.data, alpha=1.0 - decay)


def masked_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool hidden states at masked positions per batch item.

    Args:
        hidden : (B, L, D)  encoder output
        mask   : (B, L)     bool — True at target positions

    Returns:
        pooled : (B, D)
    """
    mask_f = mask.unsqueeze(-1).float()            # (B, L, 1)
    summed = (hidden * mask_f).sum(dim=1)          # (B, D)
    count  = mask_f.sum(dim=1).clamp(min=1)        # (B, 1)
    return summed / count


def unpack(batch):
    return (
        batch['context_input_ids'].to(DEVICE),
        batch['context_attention_mask'].to(DEVICE),
        batch['target_input_ids'].to(DEVICE),
        batch['target_attention_mask'].to(DEVICE),
        batch['target_mask'].to(DEVICE),
    )























# ── Cell 13: Train / eval steps ───────────────────────────────────────────────

def forward_step(batch):
    """
    Single JEPA forward pass (no predictor).

    Returns a dict of scalar losses.
    """
    ctx_ids, ctx_mask, tgt_ids, tgt_mask, span_mask = unpack(batch)

    # ── context encoder (grad flows here) ────────────────────────────────────
    # Encoder is expected to return (sequence_hidden, pooled) or just hidden.
    # Adjust the indexing below to match your Encoder's actual return signature.
    ctx_hidden = context_encoder(ctx_ids, attention_mask=ctx_mask)   # (B, L, D)
    if isinstance(ctx_hidden, tuple):
        ctx_hidden = ctx_hidden[0]

    # ── target encoder (no grad, EMA) ────────────────────────────────────────
    with torch.no_grad():
        tgt_hidden = target_encoder(tgt_ids, attention_mask=tgt_mask)
        if isinstance(tgt_hidden, tuple):
            tgt_hidden = tgt_hidden[0]

    # ── pool only at masked positions ────────────────────────────────────────
    z_ctx = masked_pool(ctx_hidden, span_mask)   # (B, D)
    z_tgt = masked_pool(tgt_hidden, span_mask)   # (B, D)

    # ── loss ─────────────────────────────────────────────────────────────────
    return loss_fn(z_ctx, z_tgt)   # dict with 'loss', 'std_loss', 'cov_loss', etc.


@torch.no_grad()
def eval_epoch(loader):
    context_encoder.eval()
    totals = {}
    n = 0
    for batch in loader:
        loss_dict = forward_step(batch)
        for k, v in loss_dict.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n += 1
    return {k: v / n for k, v in totals.items()}


def train_epoch(loader, epoch):
    context_encoder.train()
    totals = {}
    n = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch:02d}', leave=False)
    for batch in pbar:
        loss_dict = forward_step(batch)
        loss = loss_dict['loss']

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(context_encoder.parameters(), 1.0)
        optimizer.step()

        # EMA update after every gradient step
        ema_update(context_encoder, target_encoder)

        for k, v in loss_dict.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n += 1

        pbar.set_postfix({k: f'{v.item():.4f}' for k, v in loss_dict.items()})

    return {k: v / n for k, v in totals.items()}

# ── Cell 14: Checkpointing ────────────────────────────────────────────────────

def save_checkpoint(epoch, metrics):
    path = os.path.join(CFG.ckpt_dir, f'epoch_{epoch:03d}.pt')
    torch.save({
        'epoch':           epoch,
        'context_encoder': context_encoder.state_dict(),
        'target_encoder':  target_encoder.state_dict(),
        'optimizer':       optimizer.state_dict(),
        'scheduler':       scheduler.state_dict(),
        'metrics':         metrics,
        'cfg':             CFG,
    }, path)
    print(f'  ✓ saved → {path}')


def load_checkpoint(path):
    ckpt = torch.load(path, map_location=DEVICE)
    context_encoder.load_state_dict(ckpt['context_encoder'])
    target_encoder.load_state_dict(ckpt['target_encoder'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    print(f'  ✓ resumed from epoch {ckpt["epoch"]}')
    return ckpt['epoch']

























# ── Cell 15: Training loop ────────────────────────────────────────────────────

history = {
    'train_loss': [], 'train_bcs': [], 'train_inv': [],
    'val_loss':   [], 'val_bcs':   [], 'val_inv':   [],
    'lr':         [],
}

print(f'\n{"="*60}')
print(f'  Text JEPA — {CFG.n_epochs} epochs   device={DEVICE}')
print(f'{"="*60}\n')

# Resume from checkpoint if one exists
import glob
start_epoch = 1
ckpts = sorted(glob.glob(os.path.join(CFG.ckpt_dir, 'epoch_*.pt')))
if ckpts:
    latest = ckpts[-1]
    print(f'Resuming from {latest} …')
    start_epoch = load_checkpoint(latest) + 1
    print(f'  starting at epoch {start_epoch}')
else:
    print('No checkpoint found, starting from scratch.')

best_val_loss = float('inf')

for epoch in range(start_epoch, CFG.n_epochs + 1):
    train_metrics = train_epoch(train_loader, epoch)
    val_metrics   = eval_epoch(val_loader)
    scheduler.step()

    history['train_loss'].append(train_metrics.get('loss', 0.0))
    history['train_bcs'].append(train_metrics.get('bcs_loss', 0.0))
    history['train_inv'].append(train_metrics.get('invariance_loss', 0.0))
    history['val_loss'].append(val_metrics.get('loss', 0.0))
    history['val_bcs'].append(val_metrics.get('bcs_loss', 0.0))
    history['val_inv'].append(val_metrics.get('invariance_loss', 0.0))
    history['lr'].append(optimizer.param_groups[0]['lr'])

    print(
        f'Epoch {epoch:02d}/{CFG.n_epochs}  '
        f'train_loss={train_metrics["loss"]:.4f}  '
        f'val_loss={val_metrics["loss"]:.4f}  '
        f'bcs={train_metrics.get("bcs_loss", 0):.4f}  '
        f'inv={train_metrics.get("invariance_loss", 0):.4f}  '
        f'lr={optimizer.param_groups[0]["lr"]:.2e}'
    )

    if val_metrics['loss'] < best_val_loss:
        best_val_loss = val_metrics['loss']
        save_checkpoint('best', val_metrics)
        print(f'  ★ new best val_loss={best_val_loss:.4f}')

    if epoch % 5 == 0:
        save_checkpoint(epoch, {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}})

print('\nTraining complete.')
save_checkpoint(CFG.n_epochs, {})





if __name__ == "__main__":
    fire.Fire(run)