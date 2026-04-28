#!/usr/bin/env python3
"""
Inference: Full pipeline proof — BJEPA Goal Prior v2
Conditioning modes:
  1. Oracle    — PoE(dynamics, Z_T_real)      upper bound
  2. Reference — PoE(dynamics, S1(ref_text))  user-provided reference
  3. Dynamics  — μ_dyn only                   no prior
  4. Random    — N(0,I)                        lower bound
"""

import sys
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT   = '/content/notebooks_meta'
S2     = f'{ROOT}/v5/s2'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# ── Prior components — must match train_prior_v2.py exactly ──────────────────

class DynamicsHead(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.mu_head     = nn.Linear(hidden_dim, hidden_dim)
        self.logvar_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z_pred):
        mu     = self.mu_head(z_pred)
        logvar = self.logvar_head(z_pred).clamp(-10, 2)
        return mu, logvar


class LearnedGoalPrior(nn.Module):
    def __init__(self, z_dim=256):
        super().__init__()
        # ✅ scalar tensor [] to match checkpoint — was torch.zeros(1)
        self.log_sigma_goal = nn.Parameter(torch.zeros([]))
        self.z_dim = z_dim
        

    def get_sigma(self):
        return F.softplus(self.log_sigma_goal) + 1e-4

    def forward(self, z_eta):
        sigma = self.get_sigma()
        sig   = sigma.expand(z_eta.size(0), self.z_dim)
        return z_eta, sig

    def product_of_experts(self, mu_dyn, logvar_dyn, z_eta):
        """
        Hard PoE fusion:
          dynamics: N(μ_dyn, exp(logvar_dyn))
          prior:    N(z_eta, σ_goal²)
        Returns posterior mean.
        """
        mu_prior, sig_prior = self.forward(z_eta)
        prec_dyn   = logvar_dyn.exp().reciprocal()
        prec_prior = sig_prior.pow(2).reciprocal()
        prec_post  = prec_dyn + prec_prior
        mu_post    = (prec_dyn * mu_dyn + prec_prior * mu_prior) / prec_post
        return mu_post

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
    dim_head    = 64, dropout=0.0, emb_dropout=0.0,
).to(DEVICE)
s2_ckpt = torch.load(
    Path(CFG.logging.exp_dir) / 'best.pt',
    map_location=DEVICE, weights_only=False
)
s2_predictor.load_state_dict(s2_ckpt['predictor'])
s2_predictor.eval()
for p in s2_predictor.parameters(): p.requires_grad = False
print('S2 loaded.')

# Prior v2 — dynamics head + learned goal prior
dynamics_head = DynamicsHead(hidden_dim=D).to(DEVICE)
goal_prior    = LearnedGoalPrior(z_dim=D).to(DEVICE)

prior_ckpt = torch.load(
    '/content/drive/MyDrive/metanet/v5/prior_v2/best.pt',
    map_location=DEVICE, weights_only=False
)
dynamics_head.load_state_dict(prior_ckpt['dynamics_head'])
goal_prior.load_state_dict(prior_ckpt['goal_prior'])
dynamics_head.eval(); goal_prior.eval()
for p in dynamics_head.parameters(): p.requires_grad = False
for p in goal_prior.parameters():    p.requires_grad = False
print(f'Prior v2 loaded (σ_goal={goal_prior.get_sigma().item():.4f})')

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
print(f'S3 loaded (val_ppl={s3_ckpt["val_ppl"]:.1f})')

# ── Dataloader ────────────────────────────────────────────────────────────────

_, val_loader = get_stage2_dataloaders(cfg_obj=CFG, tokenizer=tokenizer)

# ── Encode a reference text ───────────────────────────────────────────────────

@torch.no_grad()
def encode_reference(text, batch_size):
    """Encode a reference turn text with frozen S1 encoder."""
    enc   = tokenizer(
        text, return_tensors='pt',
        max_length=128, truncation=True, padding='max_length'
    ).to(DEVICE)
    # ✅ pass input_ids positionally, attention_mask as kwarg
    h     = s1_encoder(enc['input_ids'], attention_mask=enc['attention_mask'])
    if isinstance(h, tuple): h = h[0]
    z_ref = mean_pool(h, enc['attention_mask'])   # (1, D)
    return z_ref.expand(batch_size, -1)           # (B, D)

# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(batch, n_samples=3, max_new_tokens=30,
                  temperature=0.8, top_k=50,
                  reference_text=None):
    """
    Runs four conditioning modes on the same batch:
      oracle    : PoE(dynamics, Z_T_real)
      reference : PoE(dynamics, S1(reference_text))  if provided
      dynamics  : μ_dyn only — no prior
      random    : N(0,I)
    """
    ctx_ids  = batch['input_ids_a'][:n_samples].to(DEVICE)
    ctx_mask = batch['attention_mask_a'][:n_samples].to(DEVICE)
    tgt_ids  = batch['input_ids_b'][:n_samples].to(DEVICE)
    tgt_mask = batch['attention_mask_b'][:n_samples].to(DEVICE)
    B        = ctx_ids.size(0)

    # Encode
    ctx_h = s1_encoder(ctx_ids, attention_mask=ctx_mask)
    if isinstance(ctx_h, tuple): ctx_h = ctx_h[0]
    tgt_h = s1_encoder(tgt_ids, attention_mask=tgt_mask)
    if isinstance(tgt_h, tuple): tgt_h = tgt_h[0]

    # Representations
    z_pred   = s2_predictor(ctx_h, ctx_h)
    z_C_pred = mean_pool(z_pred, ctx_mask)    # (B, D)
    Z_T_real = mean_pool(tgt_h,  tgt_mask)   # (B, D)

    # Dynamics distribution
    mu_dyn, logvar_dyn = dynamics_head(z_C_pred)

    # Conditioning vectors
    z_oracle   = goal_prior.product_of_experts(mu_dyn, logvar_dyn, Z_T_real)
    z_dynamics = mu_dyn
    z_random   = torch.randn_like(Z_T_real)

    modes = [
        ('oracle',   z_oracle),
        ('dynamics', z_dynamics),
        ('random',   z_random),
    ]

    if reference_text is not None:
        z_ref = encode_reference(reference_text, B)
        z_poe_ref = goal_prior.product_of_experts(mu_dyn, logvar_dyn, z_ref)
        modes.insert(1, ('reference', z_poe_ref))

    # Generate
    prompt  = torch.full((B, 1), 101, dtype=torch.long, device=DEVICE)
    results = {'context': ctx_ids, 'target': tgt_ids}

    for mode, z_cond in modes:
        generated = decoder.generate(
            prompt_ids     = prompt,
            z_fused        = z_cond,
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
            top_k          = top_k,
        )
        results[mode] = generated

    return results

# ── Decode ────────────────────────────────────────────────────────────────────

def decode(token_ids):
    return tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)

# ── Qualitative eval ──────────────────────────────────────────────────────────

REFERENCE_TEXT = "i understand what you mean, that makes sense."
N_BATCHES      = 3
N_SAMPLES      = 3

print(f'\n{"="*70}')
print(f'  Full Pipeline Inference — Goal Prior v2')
print(f'  Reference: "{REFERENCE_TEXT}"')
print(f'{"="*70}\n')

for batch_idx, batch in enumerate(val_loader):
    if batch_idx >= N_BATCHES: break

    results = run_inference(
        batch,
        n_samples      = N_SAMPLES,
        reference_text = REFERENCE_TEXT,
    )

    print(f'─── Batch {batch_idx+1} ──────────────────────────────────────────')

    for i in range(N_SAMPLES):
        print(f'\n  Sample {i+1}:')
        print(f'  Context   : {decode(results["context"][i])[:80]}')
        print(f'  Target    : {decode(results["target"][i])[:80]}')
        print(f'  Oracle    : {decode(results["oracle"][i])[:80]}')
        print(f'  Reference : {decode(results["reference"][i])[:80]}')
        print(f'  Dynamics  : {decode(results["dynamics"][i])[:80]}')
        print(f'  Random    : {decode(results["random"][i])[:80]}')
    print()

# ── Quantitative eval ─────────────────────────────────────────────────────────

print(f'\n{"="*70}')
print('  Quantitative: cosine similarity to Z_T_real')
print(f'{"="*70}\n')

@torch.no_grad()
def eval_similarity(loader, n_batches=10):
    sims = {'oracle': [], 'dynamics': [], 'random': []}

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= n_batches: break

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

        z_oracle   = goal_prior.product_of_experts(mu_dyn, logvar_dyn, Z_T_real)
        z_dynamics = mu_dyn
        z_random   = torch.randn_like(Z_T_real)

        Z_norm = F.normalize(Z_T_real, dim=-1)
        for mode, z in [
            ('oracle',   z_oracle),
            ('dynamics', z_dynamics),
            ('random',   z_random),
        ]:
            sim = F.cosine_similarity(F.normalize(z, dim=-1), Z_norm).mean().item()
            sims[mode].append(sim)

    return {k: sum(v)/len(v) for k, v in sims.items()}

sims = eval_similarity(val_loader)

print(f'  Avg cosine similarity to Z_T_real:\n')
print(f'  Oracle   : {sims["oracle"]:.4f}   upper bound')
print(f'  Dynamics : {sims["dynamics"]:.4f}   S2 only, no prior')
print(f'  Random   : {sims["random"]:.4f}   lower bound')

gap_oracle   = sims['oracle']   - sims['random']
gap_dynamics = sims['dynamics'] - sims['random']

print(f'\n  Dynamics recovery : {gap_dynamics/gap_oracle*100:.1f}%  of oracle gap')
print(f'  Oracle recovery   : {gap_oracle/gap_oracle*100:.1f}%  (ceiling)')
print(f'  σ_goal            : {goal_prior.get_sigma().item():.4f}')
print(f'\n{"="*70}')