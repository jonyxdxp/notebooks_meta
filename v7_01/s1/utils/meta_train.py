
import fire
import copy
import sys
import os
import glob
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, '/content/notebooks_meta/v6/s1')

from cog_arch.encoder import Encoder
from losses import BCS   # BCS kept as optional alternative



from data.dataloader import get_jepa_dataloaders
from data.dataset import VOCAB_SIZE, tokenizer











# Helper class to allow dot notation access to dictionary keys
class _C:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, _C(v) if isinstance(v, dict) else v)

CFG_dict = dict(
    # Model
    hidden_size  = 256,
    num_heads    = 4,       # 256 / 4 = 64 per head
    num_layers   = 4,
    max_seq_len  = 128,
    mlp_ratio    = 4.0,

    # JEPA masking
    num_target_spans   = 4,
    target_span_length = 8,

    # Training
    lr           = 1e-4,
    weight_decay = 0.05,
    n_epochs     = 20,
    ema_decay     = 0.996,
    batch_size   = 64,
    eval_batch   = 128,
    num_workers  = 0,

    # VICReg
    std_coeff = 50.0,   # was 25.0
    cov_coeff = 1.0,

    # Paths
    cache_dir    = '/content/drive/MyDrive/data/cache',
    ckpt_dir     = '/content/notebooks_meta/v4/s1/checkpoints',
    raw_data_dir = '/content/data/dailydialog_processed', # Updated to new, consistent path for extracted data
    tokenizer_name = 'bert-base-uncased',
)





# ── Functional forward ────────────────────────────────────────────────────────
#
# Standard forward_step calls encoder(x) which uses encoder.parameters()
# internally. For MAML we need to run the encoder with a *different* set of
# parameters (the fast weights φ') while still being able to differentiate
# back to the original φ.
#
# torch.func.functional_call does exactly this: it temporarily substitutes
# the module's parameter dict with the one you pass in, runs the forward
# pass, and keeps the computation graph attached to the original tensors.

def forward_step_functional(
    context_encoder: nn.Module,
    target_encoder:  nn.Module,
    params:          dict,           # fast weights (may differ from encoder.parameters())
    batch:           dict,
    loss_fn,
    device:          str,
) -> dict:
    """
    JEPA forward pass using an explicit parameter dict for the context encoder.
    Target encoder always uses its own (EMA) parameters — no fast weights there.
    """
    ctx_ids, ctx_mask, tgt_ids, tgt_mask, span_mask = unpack(batch, device)

    # context encoder: use the provided params (may be adapted fast weights)
    ctx_hidden = torch.func.functional_call(
        context_encoder, params, (ctx_ids,), {'attention_mask': ctx_mask}
    )
    if isinstance(ctx_hidden, tuple):
        ctx_hidden = ctx_hidden[0]

    # target encoder: always EMA weights, no grad
    with torch.no_grad():
        tgt_hidden = target_encoder(tgt_ids, attention_mask=tgt_mask)
        if isinstance(tgt_hidden, tuple):
            tgt_hidden = tgt_hidden[0]

    z_ctx = masked_pool(ctx_hidden, span_mask)   # (B, D)
    z_tgt = masked_pool(tgt_hidden, span_mask)   # (B, D)

    return loss_fn(z_ctx, z_tgt)










# ── Inner loop ────────────────────────────────────────────────────────────────

def inner_adapt(
    context_encoder: nn.Module,
    target_encoder:  nn.Module,
    support_batch:   dict,
    loss_fn,
    device:          str,
    alpha:           float,
    n_steps:         int = 1,
    grad_clip:       float = 1.0,
) -> dict:
    """
    Starting from the current φ (context_encoder.parameters()), take n_steps
    gradient steps on the support batch to produce task-adapted params φ'.

    create_graph=True keeps the second-order graph so the outer meta-gradient
    can differentiate through this step back to φ.

    Returns a dict {name: tensor} of fast weights — same keys as
    context_encoder.named_parameters().
    """
    # clone current params as the starting point for adaptation
    params = {k: v.clone().requires_grad_(True)
              for k, v in context_encoder.named_parameters()}

    for _ in range(n_steps):
        loss_dict = forward_step_functional(
            context_encoder, target_encoder, params, support_batch, loss_fn, device
        )
        loss = loss_dict['loss']

        grads = torch.autograd.grad(
            loss, params.values(),
            create_graph=True,   # second-order: keep graph for meta-gradient
            retain_graph=True,
        )

        # clip gradients inside the inner loop to stabilise second-order grads
        grads = [g.clamp(-grad_clip, grad_clip) for g in grads]

        params = {
            k: v - alpha * g
            for (k, v), g in zip(params.items(), grads)
        }

    return params











# ── Batch splitting ───────────────────────────────────────────────────────────

def split_batch(batch: dict, support_ratio: float = 0.5) -> tuple:
    """
    Split a batch dict into support and query halves along dim 0.
    All tensors in the batch are split at the same index.
    """
    B = next(iter(batch.values())).shape[0]
    n_support = max(1, int(B * support_ratio))

    support, query = {}, {}
    for k, v in batch.items():
        support[k] = v[:n_support]
        query[k]   = v[n_support:]

    return support, query









# ── Helpers (unchanged from original) ────────────────────────────────────────

def ema_update(ctx_enc, tgt_enc, decay):
    with torch.no_grad():
        for p_c, p_t in zip(ctx_enc.parameters(), tgt_enc.parameters()):
            p_t.data.mul_(decay).add_(p_c.data, alpha=1.0 - decay)


def masked_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).float()
    summed = (hidden * mask_f).sum(dim=1)
    count  = mask_f.sum(dim=1).clamp(min=1)
    return summed / count


def unpack(batch: dict, device: str) -> tuple:
    return (
        batch['context_input_ids'].to(device),
        batch['context_attention_mask'].to(device),
        batch['target_input_ids'].to(device),
        batch['target_attention_mask'].to(device),
        batch['target_mask'].to(device),
    )









# ── Online meta train/eval steps ─────────────────────────────────────────────

def online_meta_step(
    batch:           dict,
    context_encoder: nn.Module,
    target_encoder:  nn.Module,
    loss_fn,
    device:          str,
    alpha:           float,          # inner-loop lr
    n_inner_steps:   int   = 1,
    support_ratio:   float = 0.5,
    grad_clip:       float = 1.0,
) -> dict:
    """
    One round of online meta-learning (FTML).

    1. Split batch into support / query
    2. Inner adapt φ → φ' on support
    3. Compute meta-loss using φ' on query
    4. Return loss dict (caller does backward + outer step)

    Gradients flow through the inner adapt back to φ via the second-order graph,
    so optimizer.step() on context_encoder.parameters() updates the
    meta-initialization, not just a task-specific solution.
    """
    support, query = split_batch(batch, support_ratio)

    # φ → φ'  (differentiable inner adaptation)
    adapted_params = inner_adapt(
        context_encoder, target_encoder,
        support, loss_fn, device,
        alpha=alpha, n_steps=n_inner_steps, grad_clip=grad_clip,
    )

    # meta-loss: how well does φ' do on the query portion?
    loss_dict = forward_step_functional(
        context_encoder, target_encoder,
        adapted_params, query, loss_fn, device,
    )

    return loss_dict


def train_epoch_online(
    loader,
    context_encoder, target_encoder,
    optimizer, loss_fn,
    device, cfg, epoch,
) -> dict:
    context_encoder.train()
    totals = {}
    n = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch:02d}', leave=False)
    for batch in pbar:
        loss_dict = online_meta_step(
            batch,
            context_encoder, target_encoder,
            loss_fn, device,
            alpha         = cfg.inner_lr,
            n_inner_steps = cfg.n_inner_steps,
            support_ratio = cfg.support_ratio,
            grad_clip     = cfg.grad_clip,
        )
        loss = loss_dict['loss']

        optimizer.zero_grad()
        loss.backward()      # meta-gradient flows through inner_adapt back to φ
        torch.nn.utils.clip_grad_norm_(context_encoder.parameters(), cfg.grad_clip)
        optimizer.step()

        # EMA: target encoder tracks context encoder slowly
        ema_update(context_encoder, target_encoder, cfg.ema_decay)

        for k, v in loss_dict.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n += 1

        pbar.set_postfix({k: f'{v.item():.4f}' for k, v in loss_dict.items()})

    return {k: v / n for k, v in totals.items()}




@torch.no_grad()
def eval_epoch(
    loader,
    context_encoder, target_encoder,
    loss_fn, device, cfg,
) -> dict:
    """
    Eval: adapt on each batch's support half, measure loss on query half.
    Mirrors the online meta logic but without gradients or optimizer steps.
    """
    context_encoder.eval()
    totals = {}
    n = 0

    for batch in loader:
        support, query = split_batch(batch, cfg.support_ratio)

        # adapt without second-order graph (eval only)
        with torch.enable_grad():
            adapted_params = inner_adapt(
                context_encoder, target_encoder,
                support, loss_fn, device,
                alpha=cfg.inner_lr, n_steps=cfg.n_inner_steps,
                grad_clip=cfg.grad_clip,
            )

        loss_dict = forward_step_functional(
            context_encoder, target_encoder,
            adapted_params, query, loss_fn, device,
        )

        for k, v in loss_dict.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n += 1

    return {k: v / n for k, v in totals.items()}












# ── Checkpointing (unchanged) ─────────────────────────────────────────────────

def save_checkpoint(epoch, metrics, context_encoder, target_encoder,
                    optimizer, scheduler, cfg):
    path = os.path.join(cfg.ckpt_dir, f'epoch_{epoch}.pt')
    torch.save({
        'epoch':           epoch,
        'context_encoder': context_encoder.state_dict(),
        'target_encoder':  target_encoder.state_dict(),
        'optimizer':       optimizer.state_dict(),
        'scheduler':       scheduler.state_dict(),
        'metrics':         metrics,
        'cfg':             cfg,
    }, path)
    print(f'  saved → {path}')


def load_checkpoint(path, context_encoder, target_encoder,
                    optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    context_encoder.load_state_dict(ckpt['context_encoder'])
    target_encoder.load_state_dict(ckpt['target_encoder'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    print(f'  resumed from epoch {ckpt["epoch"]}')
    return ckpt['epoch']






















# ── Main training loop ────────────────────────────────────────────────────────

def run(cfg=None):
    if cfg is None:
        from config import CFG as cfg    # your existing config object

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── models ───────────────────────────────────────────────────────────────
    context_encoder = Encoder(
        vocab_size  = cfg.vocab_size,
        hidden_size = cfg.hidden_size,
        num_heads   = cfg.num_heads,
        num_layers  = cfg.num_layers,
        max_seq_len = cfg.max_seq_len,
    ).to(DEVICE)

    target_encoder = copy.deepcopy(context_encoder).to(DEVICE)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # ── loss / optimizer / scheduler ─────────────────────────────────────────
    loss_fn = BCS(lmbd=10.0)

    optimizer = AdamW(
        context_encoder.parameters(),
        lr           = cfg.lr,
        weight_decay = cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = cfg.n_epochs * 2,
        eta_min = cfg.lr * 0.3,
    )

    # ── dataloaders ───────────────────────────────────────────────────────────
    train_loader, val_loader = get_jepa_dataloaders(cfg_obj=cfg, tokenizer=tokenizer)

    # ── resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    ckpts = sorted(glob.glob(os.path.join(cfg.ckpt_dir, 'epoch_*.pt')))
    if ckpts:
        start_epoch = load_checkpoint(
            ckpts[-1], context_encoder, target_encoder,
            optimizer, scheduler, DEVICE
        ) + 1

    # ── training loop ─────────────────────────────────────────────────────────
    history = {k: [] for k in [
        'train_loss', 'train_bcs', 'train_inv',
        'val_loss',   'val_bcs',   'val_inv', 'lr',
    ]}
    best_val_loss = float('inf')

    print(f'\nText JEPA + Online Meta-Learning — {cfg.n_epochs} epochs   device={DEVICE}')
    print(f'inner_lr={cfg.inner_lr}  n_inner_steps={cfg.n_inner_steps}  '
          f'support_ratio={cfg.support_ratio}\n')

    for epoch in range(start_epoch, cfg.n_epochs + 1):

        train_metrics = train_epoch_online(
            train_loader,
            context_encoder, target_encoder,
            optimizer, loss_fn,
            DEVICE, cfg, epoch,
        )

        val_metrics = eval_epoch(
            val_loader,
            context_encoder, target_encoder,
            loss_fn, DEVICE, cfg,
        )

        scheduler.step()

        history['train_loss'].append(train_metrics.get('loss', 0.0))
        history['train_bcs'].append(train_metrics.get('bcs_loss', 0.0))
        history['train_inv'].append(train_metrics.get('invariance_loss', 0.0))
        history['val_loss'].append(val_metrics.get('loss', 0.0))
        history['val_bcs'].append(val_metrics.get('bcs_loss', 0.0))
        history['val_inv'].append(val_metrics.get('invariance_loss', 0.0))
        history['lr'].append(optimizer.param_groups[0]['lr'])

        print(
            f'Epoch {epoch:02d}/{cfg.n_epochs}  '
            f'train={train_metrics["loss"]:.4f}  '
            f'val={val_metrics["loss"]:.4f}  '
            f'bcs={train_metrics.get("bcs_loss", 0):.4f}  '
            f'lr={optimizer.param_groups[0]["lr"]:.2e}'
        )

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint('best', val_metrics, context_encoder, target_encoder,
                            optimizer, scheduler, cfg)
            print(f'  new best val_loss={best_val_loss:.4f}')

        if epoch % 5 == 0:
            save_checkpoint(epoch, {**train_metrics, **{f'val_{k}': v
                            for k, v in val_metrics.items()}},
                            context_encoder, target_encoder,
                            optimizer, scheduler, cfg)

    print('\nTraining complete.')
    save_checkpoint(cfg.n_epochs, {}, context_encoder, target_encoder,
                    optimizer, scheduler, cfg)
    return history


if __name__ == '__main__':
    fire.Fire(run)