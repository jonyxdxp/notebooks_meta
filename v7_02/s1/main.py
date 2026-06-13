"""
main.py
-------
MOML training loop wired to the DMI InfoNCE objective.
Imports match the repo layout:
    cog_arch/encoder.py  →  DMIScratchEncoder, compute_loss
    data/dataset.py      →  DialogCRDataset, OnlineTaskStream
"""




import warnings
warnings.filterwarnings('ignore', category=UserWarning)


import sys, os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


import math
import random
import copy
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import higher
from tqdm.auto import tqdm

# ── local imports (repo layout) ───────────────────────────────────────
from v7_02.s1.cog_arch.encoder import DMIScratchEncoder, compute_loss
from v7_02.s1.data.dataset     import (
    DialogCRDataset, OnlineTaskStream,
    EmotionAwareDataset, EmotionTaskStream,
)




# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

@dataclass
class MOMLConfig:
    # ── paths ─────────────────────────────────────────────────────────
    data_root:       str   = '/content/Discourse-Mutual-Information-DMI-main/data/dailydialog'
    output_path:     str   = '/content/drive/MyDrive/dmi_moml_ckpts'

    # ── tokeniser ─────────────────────────────────────────────────────
    max_ctx_len:     int   = 150
    max_resp_len:    int   = 60

    # ── model (scratch) ───────────────────────────────────────────────
    d_model:         int   = 256
    dim_feedforward: int   = 1024
    encoder_layers:  int   = 4
    encoder_heads:   int   = 4
    projection_size: int   = 256
    dropout:         float = 0.1
    symmetric_loss:  bool  = False
    estimator:       str   = 'infonce'   # 'infonce' | 'jsd'

    # ── MOML hyperparameters ──────────────────────────────────────────
    alpha:           float = 5.0
    n_tasks:         int   = 300
    pairs_per_task:  int   = 240

    # ── inner loop ────────────────────────────────────────────────────
    num_inner_steps: int   = 3
    lr_inner:        float = 5e-4

    # ── outer loop ────────────────────────────────────────────────────
    lr_outer:        float = 1e-4
    batch_size:      int   = 32
    grad_clip:       float = 1.0

    # ── memory / graph-size controls ──────────────────────────────────
    # max_inner_batches: how many support batches to use PER inner step.
    #   1  = stochastic MAML (one random batch per step) — lowest memory.
    #   None = use all batches (original behaviour, OOMs on T4 with bs≥32).
    max_inner_batches: Optional[int] = 1
    # max_outer_batches: how many query batches to backward through.
    #   1  = single query batch per task — lowest memory, unbiased gradient.
    #   None = use all query batches (requires retain_graph; high memory).
    max_outer_batches: Optional[int] = 1

    # ── logging / ckpt ────────────────────────────────────────────────
    seed:            int   = 42
    log_every:       int   = 10
    val_every:       int   = 50
    val_batches:     int   = 40
    save_best:       bool  = True


# ─────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────

def flat_params(model: nn.Module, device) -> torch.Tensor:
    return torch.cat(
        [p.detach().reshape(-1) for p in model.parameters()]
    ).to(device)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────
# Core: one MOML task step
# ─────────────────────────────────────────────────────────────────────

def moml_task_step(
    meta_model,
    support_pairs,
    query_pairs,
    collate_fn,
    omega:        torch.Tensor,
    lam:          torch.Tensor,
    cfg:          MOMLConfig,
    device:       str,
    outer_opt:    torch.optim.Optimizer,
):
    """
    One full MOML step for a single task.

    Inner loop  : adapt a functional copy of meta_model on support pairs
                  using InfoNCE loss (via `higher` — fully differentiable).
                  Each inner step uses ONE randomly sampled support batch
                  (stochastic MAML) to keep the `higher` graph small.
    Outer loss  : InfoNCE on ONE query batch (evaluated at adapted params)
                  + MOML proximal terms on the OUTER (meta) params θ:
                      −λᵀθ  +  (−α·ωᵀθ + α/2·‖θ‖²)
    Returns dict of scalar metrics, or None if task is too small.

    Memory budget
    ─────────────
    The `higher` library keeps ALL forward-pass activations alive for every
    inner step so it can differentiate through the inner loop.  With the
    original code that iterated through ALL spt_batches × inner_steps, the
    live graph grew to ~14 GB on a T4 (OOM).

    The fix keeps only:
      cfg.max_inner_batches (default 1) support batches per step
      cfg.max_outer_batches (default 1) query batches for the outer backward
    Typical peak VRAM with the recommended config: ~0.8–1.2 GB.
    """
    meta_model.train()

    def make_batches(pairs):
        random.shuffle(pairs)
        bs = cfg.batch_size
        return [pairs[i:i+bs] for i in range(0, len(pairs), bs)
                if len(pairs[i:i+bs]) >= 8]

    spt_batches = make_batches(support_pairs)
    qry_batches = make_batches(query_pairs)
    if not spt_batches or not qry_batches:
        return None

    # ── Limit batch counts so the higher graph stays small ─────────────
    n_inner = (cfg.max_inner_batches
               if cfg.max_inner_batches is not None
               else len(spt_batches))
    n_outer = (cfg.max_outer_batches
               if cfg.max_outer_batches is not None
               else len(qry_batches))
    n_inner = max(1, min(n_inner, len(spt_batches)))
    n_outer = max(1, min(n_outer, len(qry_batches)))

    inner_opt = torch.optim.SGD(meta_model.parameters(), lr=cfg.lr_inner)
    outer_opt.zero_grad()

    total_qry_loss_val = 0.0
    total_qry_mi       = 0.0

    with higher.innerloop_ctx(
        meta_model, inner_opt, copy_initial_weights=False
    ) as (fnet, diffopt):

        # ── Inner loop: ONE random spt batch per step ──────────────────
        # Stochastic MAML: sampling one batch per step is unbiased and
        # keeps the higher computation graph linear in num_inner_steps.
        for _ in range(cfg.num_inner_steps):
            spt_b = random.choice(spt_batches[:n_inner * 2]  # slight diversity
                                  if len(spt_batches) > 1 else spt_batches)
            ctx, rsp, m_ctx, m_rsp = collate_fn(spt_b)
            ctx, rsp     = ctx.to(device),   rsp.to(device)
            m_ctx, m_rsp = m_ctx.to(device), m_rsp.to(device)
            c_t, z_t     = fnet(ctx, rsp, m_ctx, m_rsp)
            _, spt_loss, _ = compute_loss(
                c_t, z_t, cfg.estimator, cfg.symmetric_loss)
            diffopt.step(spt_loss)
            # Free input tensors immediately — they are no longer needed
            del ctx, rsp, m_ctx, m_rsp, c_t, z_t, spt_loss

        # ── Outer: query InfoNCE (through inner loop) ──────────────────
        # We accumulate query losses and backward through the SAME higher
        # graph, so we must retain_graph=True for all but the last batch.
        qry_pool = qry_batches.copy()
        random.shuffle(qry_pool)
        outer_batches_used = qry_pool[:n_outer]

        # Pre-compute MOML proximal terms (these don't need the inner graph)
        meta_flat   = torch.cat([p.reshape(-1) for p in meta_model.parameters()])
        loss_lambda = -torch.sum(meta_flat * lam)
        loss_omega  = (-cfg.alpha * torch.sum(meta_flat * omega)
                       + (cfg.alpha / 2.0) * torch.sum(meta_flat * meta_flat))
        proximal    = loss_lambda + loss_omega

        for i, qry_b in enumerate(outer_batches_used):
            is_last = (i == len(outer_batches_used) - 1)
            ctx, rsp, m_ctx, m_rsp = collate_fn(qry_b)
            ctx, rsp     = ctx.to(device),   rsp.to(device)
            m_ctx, m_rsp = m_ctx.to(device), m_rsp.to(device)
            c_t, z_t     = fnet(ctx, rsp, m_ctx, m_rsp)
            _, qry_loss, qry_mi = compute_loss(
                c_t, z_t, cfg.estimator, cfg.symmetric_loss)

            # Scale by 1/n_outer so the effective LR is consistent
            scaled_qry = qry_loss / n_outer
            if is_last:
                # Add proximal on last batch so it only backprop-s once
                batch_loss = scaled_qry + proximal / n_outer
                batch_loss.backward(retain_graph=False)
            else:
                batch_loss = scaled_qry
                batch_loss.backward(retain_graph=True)

            total_qry_loss_val += qry_loss.item()
            total_qry_mi       += qry_mi
            del ctx, rsp, m_ctx, m_rsp, c_t, z_t, qry_loss, scaled_qry, batch_loss

    # ── Free higher graph and release GPU cache ─────────────────────────
    del meta_flat, loss_lambda, loss_omega, proximal
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if cfg.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(meta_model.parameters(), cfg.grad_clip)
    outer_opt.step()

    avg_qry_loss = total_qry_loss_val / n_outer
    avg_qry_mi   = total_qry_mi       / n_outer

    return {
        'total_loss': avg_qry_loss,     # proximal is tiny vs InfoNCE
        'qry_loss':   avg_qry_loss,
        'qry_mi':     avg_qry_mi,
        'n_spt_b':    len(spt_batches),
        'n_qry_b':    n_outer,
    }


# ─────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────

# In main.py — replace the validate function (remove @torch.no_grad() decorator)

def validate(meta_model, valid_ds, cfg, device):
    """
    MAML-style validation: adapt K steps on support, measure MI on query.
    No @torch.no_grad() decorator — the inner loop needs gradients.
    """
    import copy
    meta_model.eval()

    if hasattr(valid_ds, 'pairs_by_emotion'):
        buckets = valid_ds.pairs_by_emotion()
    else:
        buckets = {'all': list(valid_ds.cr_pairs)}

    total_mi, total_loss, n_tasks = 0.0, 0.0, 0

    for emotion, pairs in buckets.items():
        if len(pairs) < cfg.batch_size * 2:
            continue
        random.shuffle(pairs)
        half = len(pairs) // 2
        spt  = pairs[:half]
        qry  = pairs[half:half + cfg.batch_size]
        if len(qry) < 8:
            continue

        # Inner adaptation — needs gradients, so NO torch.no_grad() here
        adapted   = copy.deepcopy(meta_model)
        adapted.train()
        inner_opt = torch.optim.SGD(adapted.parameters(), lr=cfg.lr_inner)

        for _ in range(cfg.num_inner_steps):
            b = random.sample(spt, min(cfg.batch_size, len(spt)))
            ctx, rsp, mc, mr = valid_ds.collate(b)
            ctx, rsp, mc, mr = (ctx.to(device), rsp.to(device),
                                 mc.to(device),  mr.to(device))
            c_t, z_t = adapted(ctx, rsp, mc, mr)
            _, loss, _ = compute_loss(c_t, z_t, cfg.estimator, cfg.symmetric_loss)
            inner_opt.zero_grad()
            loss.backward()
            inner_opt.step()

        # Query measurement — no gradients needed here
        adapted.eval()
        with torch.no_grad():
            ctx, rsp, mc, mr = valid_ds.collate(qry)
            ctx, rsp, mc, mr = (ctx.to(device), rsp.to(device),
                                 mc.to(device),  mr.to(device))
            c_t, z_t = adapted(ctx, rsp, mc, mr)
            _, loss, mi = compute_loss(c_t, z_t, cfg.estimator, cfg.symmetric_loss)

        total_mi   += mi
        total_loss += loss.item()
        n_tasks    += 1
        del adapted
        torch.cuda.empty_cache()

    meta_model.train()
    return {
        'val_loss': total_loss / n_tasks if n_tasks else 0.0,
        'val_mi':   total_mi   / n_tasks if n_tasks else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────

def train_moml_dmi(cfg: MOMLConfig, train_ds, valid_ds,
                   vocab_size: int, device: str,
                   resume_from=None,
                   task_stream=None):
    """
    Full MOML training loop.

    Parameters
    ----------
    cfg          : MOMLConfig
    train_ds     : DialogCRDataset (or EmotionAwareDataset)
                   Used for collation and validation source.
    valid_ds     : DialogCRDataset — used for periodic validation.
    vocab_size   : int
    device       : 'cuda' | 'cpu'
    resume_from  : optional tuple (model, omega, lam) loaded via
                   load_checkpoint().  When given the existing weights
                   and MOML state are preserved; set cfg.n_tasks to the
                   number of *additional* tasks you want to run.
    task_stream  : optional iterable of (spt_pairs, qry_pairs).
                   Pass an EmotionTaskStream here to get emotion-aware
                   tasks.  If None, builds OnlineTaskStream from train_ds.

    Returns
    -------
    meta_model : trained DMIScratchEncoder
    history    : dict of logged scalars
    """
    set_seed(cfg.seed)
    os.makedirs(cfg.output_path, exist_ok=True)

    # ── Model + MOML state ─────────────────────────────────────────────
    if resume_from is not None:
        meta_model, omega, lam = resume_from
        meta_model = meta_model.to(device)
        omega      = omega.to(device)
        lam        = lam.to(device)
        print(f"\n── Resuming DMI encoder (from checkpoint) ──")
        print(f"   Parameters: {count_params(meta_model)/1e6:.2f}M")
    else:
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
        print(f"   Parameters: {count_params(meta_model)/1e6:.2f}M")
        omega = flat_params(meta_model, device)
        lam   = torch.zeros_like(omega)

    # ── Optimiser ──────────────────────────────────────────────────────
    outer_opt = torch.optim.Adam(meta_model.parameters(), lr=cfg.lr_outer)

    # ── Task stream ────────────────────────────────────────────────────
    if task_stream is None:
        task_stream = OnlineTaskStream(
            cr_pairs       = train_ds.cr_pairs,
            pairs_per_task = cfg.pairs_per_task,
            n_tasks        = cfg.n_tasks,
            seed           = cfg.seed,
        )

    # ── History ────────────────────────────────────────────────────────
    history = {
        'tasks': [], 'total_loss': [], 'qry_loss': [], 'qry_mi': [],
        'val_tasks': [], 'val_loss': [], 'val_mi': [],
    }
    best_val_mi = -float('inf')

    dataset_name = (type(task_stream).__name__
                    if task_stream is not None
                    else 'OnlineTaskStream')
    print(f"\n{'='*65}")
    print(f"  MOML + DMI  |  {'resumed' if resume_from else 'scratch'}  "
          f"|  {dataset_name}")
    print(f"{'='*65}")
    print(f"  n_tasks={cfg.n_tasks}  pairs_per_task={cfg.pairs_per_task}  "
          f"alpha={cfg.alpha}")
    print(f"  inner_steps={cfg.num_inner_steps}  lr_inner={cfg.lr_inner}  "
          f"lr_outer={cfg.lr_outer}")
    print(f"  batch_size={cfg.batch_size}  estimator={cfg.estimator}")
    print(f"  d_model={cfg.d_model}  layers={cfg.encoder_layers}  "
          f"heads={cfg.encoder_heads}")
    print(f"{'='*65}\n")

    pbar    = tqdm(enumerate(task_stream), total=cfg.n_tasks, desc='Tasks')
    skipped = 0

    for task_idx, (spt_pairs, qry_pairs) in pbar:

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

        # ── MOML dual-variable update ───────────────────────────────────
        #   λ ← λ − α·(θ − ω)
        #   ω ← ½(θ + ω) − 1/(2α)·λ
        curr_theta = flat_params(meta_model, device)
        lam   = lam   - cfg.alpha * (curr_theta - omega)
        omega = 0.5 * (curr_theta + omega) - (0.5 / cfg.alpha) * lam

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
                f"[Task {task_idx+1:4d}]  "
                f"total_loss={metrics['total_loss']:.4f}  "
                f"qry_InfoNCE={metrics['qry_loss']:.4f}  "
                f"qry_MI≈{metrics['qry_mi']:.3f}  "
                f"spt_batches={metrics['n_spt_b']}"
            )

        if (task_idx + 1) % cfg.val_every == 0:
            val_m   = validate(meta_model, valid_ds, cfg, device)
            val_mi  = val_m['val_mi']
            val_loss= val_m['val_loss']
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
                tqdm.write(
                    f"  ✓ Checkpoint saved  (best val_MI={best_val_mi:.4f})\n"
                )

    _save_checkpoint(meta_model, outer_opt, omega, lam,
                     cfg.n_tasks, best_val_mi, cfg, vocab_size,
                     name='dmi_moml_final.pt')
    print(f"\nDone.  Best val MI: {best_val_mi:.4f}  "
          f"(skipped {skipped} under-sized tasks)")
    return meta_model, history


# ─────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────

def _save_checkpoint(model, opt, omega, lam, task, val_mi, cfg,
                     vocab_size, name):
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
    Loads a checkpoint and reconstructs the model.
    Returns (model, omega, lam, cfg_dict, task_idx).
    """
    ckpt     = torch.load(path, map_location=device)
    cfg_dict = ckpt['cfg']
    model    = DMIScratchEncoder(
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