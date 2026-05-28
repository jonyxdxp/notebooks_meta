#!/usr/bin/env python3
"""
v6/inference.py
Full pipeline inference — BJEPA Goal Prior v6.

Four conditioning modes (same as v5, now with 768-d):
  oracle    : PoE(dynamics, Z_T_real)        upper bound
  reference : PoE(dynamics, encode_single(text))  user-provided
  dynamics  : μ_dyn only                     no prior
  random    : N(0, I)                         lower bound

Changes from v5 inference.py:
  - All paths point to v6
  - No mean_pool (BERT CLS token used directly)
  - Predictor takes (B, max_turns, 768) not tokenised sequences
  - DynamicsHead / LearnedGoalPrior live in v6/prior_encoder.py
  - encode_single() imported from v6/s2/encoder.py
"""

import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT   = '/content/notebooks_meta'
V6     = f'{ROOT}/v6'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.insert(0, ROOT)
sys.path.insert(0, V6)

from v6.s2.config       import cfg
from v6.s2.cog_arch.dm  import DialogueJEPAPredictor
from v6.s2.encoder      import encode_single          # BERT/DMI 768-d
from v6.prior_encoder   import DynamicsHead, LearnedGoalPrior
from v6.s3.cog_arch.decoder import SingleTurnDecoderNew as Decoder

# v6 data loader (for batch-level eval)
from v6.s2.data.data import make_dataloaders

print(f'Device: {DEVICE}')

# ── Load S2 predictor ──────────────────────────────────────────────────────────

S2_CKPT    = os.environ.get('S2_CKPT',    '/content/drive/MyDrive/data/dmi_checkpoints/jepa_predictor_best.pth')
PRIOR_CKPT = os.environ.get('PRIOR_CKPT', '/content/drive/MyDrive/metanet/v6/prior/best.pt')
S3_CKPT    = os.environ.get('S3_CKPT',    '/content/drive/MyDrive/data/dmi_checkpoints/decoder_SINGLE_best.pth')

predictor = DialogueJEPAPredictor().to(DEVICE)
s2_ckpt   = torch.load(S2_CKPT, map_location=DEVICE, weights_only=False)
state_key = 'predictor' if 'predictor' in s2_ckpt else 'model_state_dict'
predictor.load_state_dict(s2_ckpt[state_key])
predictor.eval()
for p in predictor.parameters(): p.requires_grad_(False)
print('S2 predictor loaded.')

# ── Load prior components ──────────────────────────────────────────────────────

D = cfg.d_input   # 768

dynamics_head = DynamicsHead(hidden_dim=D).to(DEVICE)
goal_prior    = LearnedGoalPrior(z_dim=D).to(DEVICE)

prior_ckpt = torch.load(PRIOR_CKPT, map_location=DEVICE, weights_only=False)
dynamics_head.load_state_dict(prior_ckpt['dynamics_head'])
goal_prior.load_state_dict(prior_ckpt['goal_prior'])
dynamics_head.eval(); goal_prior.eval()
for p in dynamics_head.parameters(): p.requires_grad_(False)
for p in goal_prior.parameters():    p.requires_grad_(False)
print(f'Prior loaded  (σ_goal={goal_prior.get_sigma().item():.4f})')

# ── Load S3 decoder ────────────────────────────────────────────────────────────



# ── Encode a reference text ────────────────────────────────────────────────────

@torch.no_grad()
def encode_reference(text: str, batch_size: int) -> torch.Tensor:
    """
    Encode free-form reference text with the frozen BERT/DMI encoder.
    Returns (B, 768) — same vector repeated across the batch.
    """
    z = encode_single(text)          # (768,) CPU — uses v6/s2/encoder.py
    return z.unsqueeze(0).expand(batch_size, -1).to(DEVICE)  # (B, 768)

# ── Build context tensor from a list of strings ────────────────────────────────

@torch.no_grad()
def build_context(utterances: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Embed a dialogue history and return (ctx, mask) ready for the predictor.

    ctx  : (1, max_turns, 768)
    mask : (1, max_turns)  True = padding
    """
    embs = [encode_single(u) for u in utterances]
    n    = min(len(embs), cfg.max_turns)
    ctx  = torch.zeros(1, cfg.max_turns, 768)
    ctx[0, :n] = torch.stack(embs[:n])
    mask = torch.arange(cfg.max_turns).unsqueeze(0) >= n   # (1, max_turns)
    return ctx.to(DEVICE), mask.to(DEVICE)

# ── Core inference step ────────────────────────────────────────────────────────

@torch.no_grad()
def get_goal_embedding(
    ctx:            torch.Tensor,       # (B, max_turns, 768)
    mask:           torch.Tensor,       # (B, max_turns) bool
    mode:           str = 'dynamics',   # oracle | reference | dynamics | random
    z_T_real:       torch.Tensor = None,  # (B, 768) required for oracle
    reference_text: str = None,           # required for reference mode
) -> torch.Tensor:
    """
    Returns the goal embedding (B, 768) for the chosen conditioning mode.
    This is what gets passed to the S3 decoder.
    """
    # S2 → dynamics distribution
    z_pred             = predictor(ctx, padding_mask=mask)       # (B, 768)
    mu_dyn, logvar_dyn = dynamics_head(z_pred)

    B = ctx.size(0)

    if mode == 'oracle':
        assert z_T_real is not None, "oracle mode needs z_T_real"
        return goal_prior.product_of_experts(mu_dyn, logvar_dyn, z_T_real)

    elif mode == 'reference':
        assert reference_text is not None, "reference mode needs reference_text"
        z_ref = encode_reference(reference_text, B)
        return goal_prior.product_of_experts(mu_dyn, logvar_dyn, z_ref)

    elif mode == 'dynamics':
        return mu_dyn

    elif mode == 'random':
        return torch.randn_like(mu_dyn)

    else:
        raise ValueError(f"Unknown mode '{mode}'. "
                         "Choose: oracle | reference | dynamics | random")

# ── Batch-level qualitative eval ───────────────────────────────────────────────

@torch.no_grad()
def run_batch_inference(batch, n_samples=3, reference_text=None):
    """
    Run all four modes on one dataloader batch.
    batch: (ctx, tgt, mask, lens)  from make_dataloaders()
    """
    ctx, tgt, mask, lens = batch
    ctx  = ctx[:n_samples].to(DEVICE)
    tgt  = tgt[:n_samples].to(DEVICE)
    mask = mask[:n_samples].to(DEVICE)
    B    = ctx.size(0)

    z_pred             = predictor(ctx, padding_mask=mask)
    mu_dyn, logvar_dyn = dynamics_head(z_pred)

    modes = {
        'oracle':   goal_prior.product_of_experts(mu_dyn, logvar_dyn, tgt),
        'dynamics': mu_dyn,
        'random':   torch.randn_like(mu_dyn),
    }
    if reference_text is not None:
        z_ref = encode_reference(reference_text, B)
        modes['reference'] = goal_prior.product_of_experts(mu_dyn, logvar_dyn, z_ref)

    results = {}
    for mode_name, z_goal in modes.items():
        generated = decoder.generate(
            prompt_ids     = torch.full((B, 1), 101, dtype=torch.long, device=DEVICE),
            z_fused        = z_goal,
            max_new_tokens = 30,
            temperature    = 0.8,
            top_k          = 50,
        )
        results[mode_name] = generated

    results['target'] = tgt
    return results

# ── Quantitative eval ──────────────────────────────────────────────────────────

@torch.no_grad()
def eval_similarity(loader, n_batches=10, reference_text=None) -> dict:
    """
    Cosine similarity between each conditioning mode's output and Z_T_real.
    Higher = closer to the true next turn in embedding space.
    """
    sims = {k: [] for k in ['oracle', 'reference', 'dynamics', 'random']}

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        ctx, tgt, mask, lens = batch
        ctx  = ctx.to(DEVICE)
        tgt  = tgt.to(DEVICE)
        mask = mask.to(DEVICE)
        B    = ctx.size(0)

        z_pred             = predictor(ctx, padding_mask=mask)
        mu_dyn, logvar_dyn = dynamics_head(z_pred)

        T_norm = F.normalize(tgt, dim=-1)

        candidates = {
            'oracle':   goal_prior.product_of_experts(mu_dyn, logvar_dyn, tgt),
            'dynamics': mu_dyn,
            'random':   torch.randn_like(mu_dyn),
        }
        if reference_text is not None:
            z_ref = encode_reference(reference_text, B)
            candidates['reference'] = goal_prior.product_of_experts(
                mu_dyn, logvar_dyn, z_ref)

        for k, z in candidates.items():
            sims[k].append(
                F.cosine_similarity(F.normalize(z, dim=-1), T_norm).mean().item()
            )

    return {k: sum(v) / max(len(v), 1) for k, v in sims.items()}

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    REFERENCE = "i understand what you mean, that makes sense."

    _, val_loader = make_dataloaders()

    # ── qualitative ──
    print(f'\n{"="*70}')
    print(f'  v6 Inference — BJEPA Goal Prior  (σ_goal={goal_prior.get_sigma().item():.4f})')
    print(f'  Reference: "{REFERENCE}"')
    print(f'{"="*70}\n')

    for bi, batch in enumerate(val_loader):
        if bi >= 3:
            break
        results = run_batch_inference(batch, n_samples=3,
                                       reference_text=REFERENCE)
        print(f'─── Batch {bi+1} ───')
        # (tokenizer decode left to the caller since we don't import it here)
        for mode in ['oracle', 'reference', 'dynamics', 'random']:
            if mode in results:
                print(f'  {mode:10s}: shape={results[mode].shape}')

    # ── quantitative ──
    sims = eval_similarity(val_loader, n_batches=10, reference_text=REFERENCE)

    print(f'\n{"="*70}')
    print('  Cosine similarity to Z_T_real:')
    print(f'{"="*70}')
    for k, v in sorted(sims.items(), key=lambda x: -x[1]):
        print(f'  {k:12s} : {v:.4f}')

    gap_oracle   = sims['oracle']   - sims['random']
    gap_dynamics = sims['dynamics'] - sims['random']
    print(f'\n  Dynamics recovery : {gap_dynamics/gap_oracle*100:.1f}% of oracle gap')
    if 'reference' in sims:
        gap_ref = sims['reference'] - sims['random']
        print(f'  Reference recovery: {gap_ref/gap_oracle*100:.1f}% of oracle gap')
    print(f'  σ_goal : {goal_prior.get_sigma().item():.4f}')
    print(f'{"="*70}')