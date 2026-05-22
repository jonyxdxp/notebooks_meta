#!/usr/bin/env python3
"""
Inference: Full pipeline proof
Compares three conditioning modes:
  1. Oracle    — Z_T_real (upper bound)
  2. PoE       — dynamics + prior fusion (actual system)
  3. Random    — N(0,I) sample (lower bound baseline)
"""

import sys
import os
import copy
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT     = '/content/notebooks_meta'
S1       = f'{ROOT}/v5/s1'
S2       = f'{ROOT}/v5/s2'
S3       = f'{ROOT}/v5/s3'
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.insert(0, ROOT)

from v5.s1.cog_arch.encoder import Encoder
from v5.s2.cog_arch.dm import DM
from v5.s3.cog_arch.decoder import Decoder
from v5.s2.config import CFG

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_dataset    = _load_module('data.dataset',    f'{S2}/data/dataset.py')
_dataloader = _load_module('data.dataloader', f'{S2}/data/dataloader.py')

get_stage2_dataloaders = _dataloader.get_stage2_dataloaders
tokenizer              = _dataset.tokenizer

print(f'Device: {DEVICE}')

# ── Helpers ───────────────────────────────────────────────────────────────────

def mean_pool(hidden, mask):
    mask_f = mask.unsqueeze(-1).float()
    return (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)

# ── Prior components (from train_prior.py) ────────────────────────────────────

class DynamicsHead(torch.nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.mu_head     = torch.nn.Linear(hidden_dim, hidden_dim)
        self.logvar_head = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z_pred):
        mu     = self.mu_head(z_pred)
        logvar = self.logvar_head(z_pred).clamp(-10, 2)
        return mu, logvar

class StaticPrior(torch.nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.prior_mu     = torch.nn.Parameter(torch.zeros(latent_dim))
        self.prior_logvar = torch.nn.Parameter(torch.zeros(latent_dim))

    def get_prior(self, batch_size, device):
        mu     = self.prior_mu.unsqueeze(0).expand(batch_size, -1)
        logvar = self.prior_logvar.unsqueeze(0).expand(batch_size, -1)
        return mu, logvar

    def product_of_experts(self, mu_dyn, logvar_dyn, batch_size, device):
        mu_prior, logvar_prior = self.get_prior(batch_size, device)
        prec_dyn   = logvar_dyn.exp().reciprocal()
        prec_prior = logvar_prior.exp().reciprocal()
        prec_post  = prec_dyn + prec_prior
        mu_post    = (prec_dyn * mu_dyn + prec_prior * mu_prior) / prec_post
        return mu_post

    def sample_prior(self, batch_size, device):
        """Sample from learned prior — NOT N(0,I)."""
        mu, logvar = self.get_prior(batch_size, device)
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

# ── Load all models ───────────────────────────────────────────────────────────

D = CFG.model.hidden_size   # 256

# S1 encoder
s1_encoder = Encoder(
    vocab_size  = CFG.model.vocab_size,
    hidden_size = D,
    num_heads   = CFG.model.num_heads,
    num_layers  = CFG.model.num_layers,
    max_seq_len = CFG.model.max_seq_len,
).to(DEVICE)
s1_ckpt = torch.load(CFG.training.s1_ckpt, map_location=DEVICE, weights_only=False)
s1_encoder.load_state_dict(s1_ckpt['context_encoder'])
s1_encoder.eval()
for p in s1_encoder.parameters(): p.requires_grad = False
print('S1 loaded.')

# S2 predictor
s2_predictor = DM(
    num_frames  = CFG.model.max_seq_len,
    depth       = CFG.model.pred_num_layers,
    heads       = CFG.model.pred_num_heads,
    mlp_dim     = CFG.model.pred_hidden_size * 4,
    input_dim   = CFG.model.dstc,
    hidden_dim  = CFG.model.pred_hidden_size,
    output_dim  = CFG.model.dstc,
    dim_head    = 64,
    dropout     = 0.0,
    emb_dropout = 0.0,
).to(DEVICE)
s2_ckpt = torch.load(
    Path(CFG.logging.exp_dir) / 'best.pt',
    map_location=DEVICE, weights_only=False
)
s2_predictor.load_state_dict(s2_ckpt['predictor'])
s2_predictor.eval()
for p in s2_predictor.parameters(): p.requires_grad = False
print('S2 loaded.')

# Prior (dynamics head + static prior)
dynamics_head = DynamicsHead(hidden_dim=D).to(DEVICE)
prior         = StaticPrior(latent_dim=D).to(DEVICE)

prior_ckpt = torch.load(
    '/content/drive/MyDrive/metanet/v5/prior/best.pt',
    map_location=DEVICE, weights_only=False
)
dynamics_head.load_state_dict(prior_ckpt['dynamics_head'])
prior.load_state_dict(prior_ckpt['prior'])
dynamics_head.eval()
prior.eval()
for p in dynamics_head.parameters(): p.requires_grad = False
for p in prior.parameters():         p.requires_grad = False
print('Prior loaded.')

# S3 decoder
decoder = Decoder(
    vocab_size   = CFG.model.vocab_size,
    hidden_size  = 256,
    num_heads    = 4,
    num_layers   = 4,
    max_seq_len  = CFG.model.max_seq_len,
    context_dim  = D,
).to(DEVICE)
s3_ckpt = torch.load(
    '/content/drive/MyDrive/metanet/v5/s3/checkpoints/best.pt',
    map_location=DEVICE, weights_only=False
)
decoder.load_state_dict(s3_ckpt['decoder'])
decoder.eval()
for p in decoder.parameters(): p.requires_grad = False
print(f'S3 loaded (val_ppl={s3_ckpt["val_ppl"]:.1f}).')

# ── Dataloader ────────────────────────────────────────────────────────────────

_, val_loader = get_stage2_dataloaders(cfg_obj=CFG, tokenizer=tokenizer)

# ── Inference function ────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(batch, n_samples=5, max_new_tokens=30,
                  temperature=0.8, top_k=50):
    """
    Run all three conditioning modes on the same batch.
    Returns dict of generated text per mode.
    """
    ctx_ids  = batch['input_ids_a'][:n_samples].to(DEVICE)
    ctx_mask = batch['attention_mask_a'][:n_samples].to(DEVICE)
    tgt_ids  = batch['input_ids_b'][:n_samples].to(DEVICE)
    tgt_mask = batch['attention_mask_b'][:n_samples].to(DEVICE)
    B        = ctx_ids.size(0)

    # ── Encode context and target ─────────────────────────────────────────────
    ctx_h = s1_encoder(ctx_ids, attention_mask=ctx_mask)
    if isinstance(ctx_h, tuple): ctx_h = ctx_h[0]

    tgt_h = s1_encoder(tgt_ids, attention_mask=tgt_mask)
    if isinstance(tgt_h, tuple): tgt_h = tgt_h[0]

    # ── S2 dynamics prediction ────────────────────────────────────────────────
    z_pred   = s2_predictor(ctx_h, ctx_h)
    z_C_pred = mean_pool(z_pred, ctx_mask)    # (B, D)

    # ── Z_T_real — oracle conditioning ───────────────────────────────────────
    Z_T_real = mean_pool(tgt_h, tgt_mask)     # (B, D)

    # ── PoE conditioning ──────────────────────────────────────────────────────
    mu_dyn, logvar_dyn = dynamics_head(z_C_pred)
    z_poe = prior.product_of_experts(mu_dyn, logvar_dyn, B, DEVICE)

    # ── Random baseline ───────────────────────────────────────────────────────
    z_random = torch.randn_like(Z_T_real)

    # ── Prompt: [CLS] token ───────────────────────────────────────────────────
    prompt = torch.full((B, 1), 101, dtype=torch.long, device=DEVICE)

    # ── Generate for each mode ────────────────────────────────────────────────
    results = {}
    for mode, z_cond in [
        ('oracle', Z_T_real),
        ('poe',    z_poe),
        ('random', z_random),
    ]:
        generated = decoder.generate(
            prompt_ids     = prompt,
            z_fused        = z_cond,
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
            top_k          = top_k,
        )
        results[mode] = generated

    # ── Decode context turns too ──────────────────────────────────────────────
    results['context'] = ctx_ids
    results['target']  = tgt_ids

    return results

# ── Decode tokens to text ─────────────────────────────────────────────────────

def decode(token_ids):
    return tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)

# ── Run on a few val batches ──────────────────────────────────────────────────

print(f'\n{"="*70}')
print('  Full Pipeline Inference — Oracle vs PoE vs Random')
print(f'{"="*70}\n')

N_BATCHES = 3
N_SAMPLES = 3

for batch_idx, batch in enumerate(val_loader):
    if batch_idx >= N_BATCHES:
        break

    results = run_inference(batch, n_samples=N_SAMPLES)

    print(f'─── Batch {batch_idx + 1} ───────────────────────────────────────────────')

    for i in range(N_SAMPLES):
        ctx_text    = decode(results['context'][i])
        tgt_text    = decode(results['target'][i])
        oracle_text = decode(results['oracle'][i])
        poe_text    = decode(results['poe'][i])
        random_text = decode(results['random'][i])

        print(f'\n  Sample {i+1}:')
        print(f'  Context  : {ctx_text[:80]}')
        print(f'  Target   : {tgt_text[:80]}')
        print(f'  Oracle   : {oracle_text[:80]}')
        print(f'  PoE      : {poe_text[:80]}')
        print(f'  Random   : {random_text[:80]}')

    print()

# ── Quantitative eval: cosine similarity to target ───────────────────────────

print(f'\n{"="*70}')
print('  Quantitative: cosine similarity to Z_T_real')
print(f'{"="*70}\n')

@torch.no_grad()
def eval_similarity(loader, n_batches=10):
    """
    For each mode, measure cosine similarity between
    the conditioning vector and Z_T_real.
    This tells us how well each mode approximates the oracle.
    """
    sims = {'oracle': [], 'poe': [], 'random': []}

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= n_batches:
            break

        ctx_ids  = batch['input_ids_a'].to(DEVICE)
        ctx_mask = batch['attention_mask_a'].to(DEVICE)
        tgt_ids  = batch['input_ids_b'].to(DEVICE)
        tgt_mask = batch['attention_mask_b'].to(DEVICE)
        B        = ctx_ids.size(0)

        ctx_h = s1_encoder(ctx_ids, attention_mask=ctx_mask)
        if isinstance(ctx_h, tuple): ctx_h = ctx_h[0]
        tgt_h = s1_encoder(tgt_ids, attention_mask=tgt_mask)
        if isinstance(tgt_h, tuple): tgt_h = tgt_h[0]

        z_pred   = s2_predictor(ctx_h, ctx_h)
        z_C_pred = mean_pool(z_pred, ctx_mask)
        Z_T_real = mean_pool(tgt_h,  tgt_mask)

        mu_dyn, logvar_dyn = dynamics_head(z_C_pred)
        z_poe    = prior.product_of_experts(mu_dyn, logvar_dyn, B, DEVICE)
        z_random = torch.randn_like(Z_T_real)

        Z_norm = F.normalize(Z_T_real, dim=-1)

        for mode, z in [('oracle', Z_T_real), ('poe', z_poe), ('random', z_random)]:
            sim = F.cosine_similarity(F.normalize(z, dim=-1), Z_norm).mean().item()
            sims[mode].append(sim)

    return {k: sum(v)/len(v) for k, v in sims.items()}

sims = eval_similarity(val_loader)

print(f'  Avg cosine similarity to Z_T_real:\n')
print(f'  Oracle  : {sims["oracle"]:.4f}  (should be 1.0 — self-similarity)')
print(f'  PoE     : {sims["poe"]:.4f}  (higher = better conditioning)')
print(f'  Random  : {sims["random"]:.4f}  (baseline — near 0)')

gap_poe    = sims['poe']    - sims['random']
gap_oracle = sims['oracle'] - sims['random']
print(f'\n  PoE improvement over random : {gap_poe:.4f}')
print(f'  Oracle improvement (ceiling): {gap_oracle:.4f}')
print(f'  PoE recovery of oracle gap  : {gap_poe/gap_oracle*100:.1f}%')
print(f'\n{"="*70}')