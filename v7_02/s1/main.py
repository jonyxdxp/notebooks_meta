"""
dmi_moml_trainer.py
-------------------
MOML (Memory Efficient Online Meta-Learning) outer loop wired to the
DMI InfoNCE objective.

Algorithm recap (from MOML paper, Algorithm 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Maintain: meta-model θ,  omega ω (running center),  lambda λ (dual var)
Init:     ω = θ₀,  λ = 0

For each task t:
  1. Inner loop (MAML-style, via `higher`):
       θ_adapted = θ  (copy)
       for k = 1…K:
           L_support = InfoNCE(θ_adapted, support_t)
           θ_adapted ← θ_adapted − α_inner ∇L_support

  2. Outer loss (evaluated at θ_adapted for the query loss,
                  proximal terms evaluated at θ):
       L_outer(θ) = L_query(θ_adapted)
                  + [−λᵀθ]                 ← dual coupling
                  + [−α·ωᵀθ + α/2·‖θ‖²]   ← proximal to ω

  3. Outer gradient step:
       θ ← θ − lr_outer · ∇_θ L_outer(θ)

  4. Update MOML state:
       λ ← λ − α·(θ − ω)
       ω ← ½·(θ + ω) − 1/(2α)·λ

Note: step 2's proximal terms differentiate through θ (outer params),
NOT through θ_adapted.  `higher` handles differentiating through the
inner loop for the query loss part.
"""

import os
import math
import random
import copy
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import higher
from tqdm.auto import tqdm

from dmi_moml_model import DMIScratchEncoder, compute_loss


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

@dataclass
class MOMLConfig:
    # ── paths ──────────────────────────────────────────────────
    data_root:      str  = '/content/Discourse-Mutual-Information-DMI-main/data/dailydialog'
    output_path:    str  = '/content/drive/MyDrive/dmi_moml_ckpts'

    # ── tokeniser ──────────────────────────────────────────────
    max_ctx_len:    int  = 150   # max tokens for context  (orig. DMI = 300)
    max_resp_len:   int  = 60    # max tokens for response

    # ── model (scratch) ────────────────────────────────────────
    d_model:        int  = 256   # hidden dim  (orig. DMI = 512)
    dim_feedforward:int  = 1024  # FFN width   (orig. DMI = 2048)
    encoder_layers: int  = 4
    encoder_heads:  int  = 4
    projection_size:int  = 256
    dropout:        float= 0.1
    symmetric_loss: bool = False
    estimator:      str  = 'infonce'   # 'infonce' | 'jsd'

    # ── MOML hyperparameters ───────────────────────────────────
    alpha:          float= 5.0   # proximal weight  (paper: 1, 5, 10 all tested)
    n_tasks:        int  = 300   # total tasks to process
    pairs_per_task: int  = 240   # ≈ 30 DailyDialog dialogs × 8 pairs

    # ── inner loop ─────────────────────────────────────────────
    num_inner_steps:int  = 3     # gradient steps on support set
    lr_inner:       float= 5e-4  # inner SGD lr

    # ── outer loop ─────────────────────────────────────────────
    lr_outer:       float= 1e-4  # outer Adam lr
    batch_size:     int  = 32    # InfoNCE batch — needs ≥16 for good negatives
    grad_clip:      float= 1.0

    # ── logging / ckpt ────────────────────────────────────────
    seed:           int  = 42
    log_every:      int  = 10    # tasks
    val_every:      int  = 50    # tasks
    val_batches:    int  = 40    # batches used for validation
    save_best:      bool = True


# ─────────────────────────────────────────────────────────────
# MOML utilities
# ─────────────────────────────────────────────────────────────

def flat_params(model: nn.Module, device) -> torch.Tensor:
    """All trainable parameters as a single flat vector (detached)."""
    return torch.cat(
        [p.detach().reshape(-1) for p in model.parameters()]
    ).to(device)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int):
    import random, numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────
# Core: one MOML task step
# ─────────────────────────────────────────────────────────────

def moml_task_step(
    meta_model:       DMIScratchEncoder,
    support_pairs:    list,
    query_pairs:      list,
    collate_fn,
    omega:            torch.Tensor,   # flat, on device
    lam:              torch.Tensor,   # flat, on device
    cfg:              MOMLConfig,
    device:           str,
    outer_opt:        torch.optim.Optimizer,
):
    """
    Executes one full MOML step for a single task.

    Returns dict with scalar metrics, or None if the task is too small.
    """
    meta_model.train()

    def make_batches(pairs):
        random.shuffle(pairs)
        bs = cfg.batch_size
        return [pairs[i:i+bs] for i in range(0, len(pairs), bs)
                if len(pairs[i:i+bs]) >= 8]   # min 8 pairs for InfoNCE

    spt_batches = make_batches(support_pairs)
    qry_batches = make_batches(query_pairs)

    if not spt_batches or not qry_batches:
        return None   # task too small — skip

    # Inner optimiser (not stateful between tasks — fresh each time)
    inner_opt = torch.optim.SGD(meta_model.parameters(), lr=cfg.lr_inner)

    outer_opt.zero_grad()

    total_qry_loss = torch.tensor(0.0, device=device)
    total_qry_mi   = 0.0
    n_qry          = 0

    with higher.innerloop_ctx(
            meta_model, inner_opt,
            copy_initial_weights=False,   # share initial weights with outer model
    ) as (fnet, diffopt):

        # ── Inner loop: adapt fnet on support batches ──────────
        for _ in range(cfg.num_inner_steps):
            for spt_b in spt_batches:
                ctx, rsp, m_ctx, m_rsp = collate_fn(spt_b)
                ctx, rsp   = ctx.to(device), rsp.to(device)
                m_ctx, m_rsp = m_ctx.to(device), m_rsp.to(device)

                c_t, z_t = fnet(ctx, rsp, m_ctx, m_rsp)
                _, spt_loss, _ = compute_loss(
                    c_t, z_t, cfg.estimator, cfg.symmetric_loss)
                diffopt.step(spt_loss)

        # ── Outer loop: query InfoNCE + MOML proximal ──────────
        for qry_b in qry_batches:
            ctx, rsp, m_ctx, m_rsp = collate_fn(qry_b)
            ctx, rsp   = ctx.to(device), rsp.to(device)
            m_ctx, m_rsp = m_ctx.to(device), m_rsp.to(device)

            # Query loss evaluated at ADAPTED parameters (through inner loop)
            c_t, z_t = fnet(ctx, rsp, m_ctx, m_rsp)
            _, qry_loss, qry_mi = compute_loss(
                c_t, z_t, cfg.estimator, cfg.symmetric_loss)

            total_qry_loss = total_qry_loss + qry_loss
            total_qry_mi  += qry_mi
            n_qry         += 1

        if n_qry == 0:
            return None

        avg_qry_loss = total_qry_loss / n_qry

        # ── MOML proximal terms on the OUTER (meta) parameters ─
        # These differentiate through θ (not θ_adapted), so we use
        # meta_model.parameters(), NOT fnet.parameters().
        meta_flat = torch.cat([p.reshape(-1) for p in meta_model.parameters()])

        loss_lambda = -torch.sum(meta_flat * lam)
        loss_omega  = (-cfg.alpha * torch.sum(meta_flat * omega)
                       + (cfg.alpha / 2.0) * torch.sum(meta_flat * meta_flat))

        total_loss = avg_qry_loss + loss_lambda + loss_omega
        total_loss.backward()

    # ── Gradient step on outer / meta model ────────────────────
    if cfg.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(meta_model.parameters(), cfg.grad_clip)
    outer_opt.step()

    return {
        'total_loss': total_loss.item(),
        'qry_loss':   avg_qry_loss.item(),
        'qry_mi':     total_qry_mi / n_qry,
        'n_spt_b':    len(spt_batches),
        'n_qry_b':    n_qry,
    }


# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(meta_model, valid_ds, cfg: MOMLConfig, device: str) -> dict:
    meta_model.eval()
    pairs = list(valid_ds.cr_pairs)
    random.shuffle(pairs)
    batches = [pairs[i:i + cfg.batch_size]
               for i in range(0, len(pairs), cfg.batch_size)
               if len(pairs[i:i + cfg.batch_size]) >= 8]
    batches = batches[:cfg.val_batches]

    total_mi   = 0.0
    total_loss = 0.0
    n          = 0
    for b in batches:
        ctx, rsp, m_ctx, m_rsp = valid_ds.collate(b)
        ctx, rsp   = ctx.to(device), rsp.to(device)
        m_ctx, m_rsp = m_ctx.to(device), m_rsp.to(device)
        c_t, z_t   = meta_model(ctx, rsp, m_ctx, m_rsp)
        _, loss, mi = compute_loss(c_t, z_t, cfg.estimator, cfg.symmetric_loss)
        total_loss += loss.item()
        total_mi   += mi
        n          += 1

    meta_model.train()
    return {
        'val_loss': total_loss / n if n else 0,
        'val_mi':   total_mi   / n if n else 0,
    }


# ─────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────

def train_moml_dmi(cfg: MOMLConfig, train_ds, valid_ds,
                   vocab_size: int, device: str):
    """
    Full MOML training loop.

    Parameters
    ----------
    cfg        : MOMLConfig
    train_ds   : DialogCRDataset  (has .cr_pairs and .collate())
    valid_ds   : DialogCRDataset
    vocab_size : int
    device     : 'cuda' | 'cpu'

    Returns
    -------
    meta_model : trained DMIScratchEncoder
    history    : dict of logged scalars
    """
    from dmi_moml_data import OnlineTaskStream

    set_seed(cfg.seed)
    os.makedirs(cfg.output_path, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────
    print("\n── Building DMI encoder (from scratch) ──")
    meta_model = DMIScratchEncoder(
        vocab_size       = vocab_size,
        d_model          = cfg.d_model,
        projection_size  = cfg.projection_size,
        encoder_layers   = cfg.encoder_layers,
        encoder_heads    = cfg.encoder_heads,
        dim_feedforward  = cfg.dim_feedforward,
        dropout          = cfg.dropout,
        symmetric_loss   = cfg.symmetric_loss,
    ).to(device)
    n_par = count_params(meta_model)
    print(f"   Parameters: {n_par/1e6:.2f}M")

    # ── MOML state ────────────────────────────────────────────
    omega = flat_params(meta_model, device)     # running center  ω
    lam   = torch.zeros_like(omega)             # dual variable   λ

    # ── Optimiser ─────────────────────────────────────────────
    outer_opt = torch.optim.Adam(meta_model.parameters(), lr=cfg.lr_outer)

    # ── Task stream ───────────────────────────────────────────
    task_stream = OnlineTaskStream(
        cr_pairs       = train_ds.cr_pairs,
        pairs_per_task = cfg.pairs_per_task,
        n_tasks        = cfg.n_tasks,
        seed           = cfg.seed,
    )

    # ── History ───────────────────────────────────────────────
    history = {
        'tasks': [], 'total_loss': [], 'qry_loss': [], 'qry_mi': [],
        'val_tasks': [], 'val_loss': [], 'val_mi': [],
    }
    best_val_mi = -float('inf')

    # ── Header ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  MOML + DMI  |  encoder from scratch  |  DailyDialog")
    print(f"{'='*65}")
    print(f"  n_tasks={cfg.n_tasks}, pairs_per_task={cfg.pairs_per_task}")
    print(f"  alpha={cfg.alpha}, inner_steps={cfg.num_inner_steps}")
    print(f"  lr_inner={cfg.lr_inner}, lr_outer={cfg.lr_outer}")
    print(f"  batch_size={cfg.batch_size}, estimator={cfg.estimator}")
    print(f"  d_model={cfg.d_model}, layers={cfg.encoder_layers}, "
          f"heads={cfg.encoder_heads}")
    print(f"{'='*65}\n")

    pbar = tqdm(enumerate(task_stream), total=cfg.n_tasks, desc='Tasks')
    skipped = 0

    for task_idx, (spt_pairs, qry_pairs) in pbar:

        # ── MOML task step ────────────────────────────────────
        metrics = moml_task_step(
            meta_model    = meta_model,
            support_pairs = spt_pairs,
            query_pairs   = qry_pairs,
            collate_fn    = train_ds.collate,
            omega         = omega,
            lam           = lam,
            cfg           = cfg,
            device        = device,
            outer_opt     = outer_opt,
        )

        if metrics is None:
            skipped += 1
            continue

        # ── MOML dual-variable update ─────────────────────────
        #   λ ← λ − α·(θ − ω)
        #   ω ← ½(θ + ω) − 1/(2α)·λ
        curr_theta = flat_params(meta_model, device)
        lam   = lam   - cfg.alpha * (curr_theta - omega)
        omega = 0.5 * (curr_theta + omega) - (0.5 / cfg.alpha) * lam

        # ── Logging ───────────────────────────────────────────
        pbar.set_postfix({
            'L':  f"{metrics['qry_loss']:.3f}",
            'MI': f"{metrics['qry_mi']:.3f}",
        })

        if (task_idx + 1) % cfg.log_every == 0:
            history['tasks'].append(task_idx + 1)
            history['total_loss'].append(metrics['total_loss'])
            history['qry_loss'].append(metrics['qry_loss'])
            history['qry_mi'].append(metrics['qry_mi'])
            tqdm.write(
                f"[Task {task_idx+1:4d}] "
                f"total_loss={metrics['total_loss']:.4f}  "
                f"qry_InfoNCE={metrics['qry_loss']:.4f}  "
                f"qry_MI≈{metrics['qry_mi']:.3f}  "
                f"spt_batches={metrics['n_spt_b']}"
            )

        # ── Validation ────────────────────────────────────────
        if (task_idx + 1) % cfg.val_every == 0:
            val_metrics = validate(meta_model, valid_ds, cfg, device)
            val_mi   = val_metrics['val_mi']
            val_loss = val_metrics['val_loss']
            history['val_tasks'].append(task_idx + 1)
            history['val_loss'].append(val_loss)
            history['val_mi'].append(val_mi)
            tqdm.write(
                f"\n  ▶ [Val @ task {task_idx+1}]  "
                f"val_loss={val_loss:.4f}  val_MI≈{val_mi:.4f}"
            )

            if cfg.save_best and val_mi > best_val_mi:
                best_val_mi = val_mi
                _save_checkpoint(meta_model, outer_opt, omega, lam,
                                 task_idx + 1, val_mi, cfg, vocab_size,
                                 name='dmi_moml_best.pt')
                tqdm.write(f"  ✓ Checkpoint saved  (best val_MI={best_val_mi:.4f})\n")

    # ── Save final ────────────────────────────────────────────
    _save_checkpoint(meta_model, outer_opt, omega, lam,
                     cfg.n_tasks, best_val_mi, cfg, vocab_size,
                     name='dmi_moml_final.pt')

    print(f"\nTraining complete.  Best val MI: {best_val_mi:.4f}  "
          f"(skipped {skipped} under-sized tasks)")
    return meta_model, history


# ─────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────

def _save_checkpoint(model, opt, omega, lam, task, val_mi, cfg, vocab_size, name):
    path = os.path.join(cfg.output_path, name)
    torch.save({
        'task':             task,
        'model_state_dict': model.state_dict(),
        'opt_state_dict':   opt.state_dict(),
        'omega':            omega.cpu(),
        'lambda':           lam.cpu(),
        'val_mi':           val_mi,
        'cfg':              asdict(cfg),
        'vocab_size':       vocab_size,
    }, path)


def load_checkpoint(path: str, device: str):
    """
    Loads a saved checkpoint and reconstructs the model.
    Returns (model, omega, lam, cfg_dict, task_idx).
    """
    ckpt = torch.load(path, map_location=device)
    cfg_dict = ckpt['cfg']

    model = DMIScratchEncoder(
        vocab_size       = ckpt['vocab_size'],
        d_model          = cfg_dict['d_model'],
        projection_size  = cfg_dict['projection_size'],
        encoder_layers   = cfg_dict['encoder_layers'],
        encoder_heads    = cfg_dict['encoder_heads'],
        dim_feedforward  = cfg_dict['dim_feedforward'],
        dropout          = cfg_dict['dropout'],
        symmetric_loss   = cfg_dict['symmetric_loss'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    omega = ckpt['omega'].to(device)
    lam   = ckpt['lambda'].to(device)
    return model, omega, lam, cfg_dict, ckpt['task']