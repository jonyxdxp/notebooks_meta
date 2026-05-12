"""
eval_s2.py — Retrieval accuracy para Stage 2
Dado el turno t, el predictor genera ẑ_{t+1} y buscamos
si z_{t+1} real está entre los top-k vecinos más cercanos.
"""

import sys
import os
import copy
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = '/content/notebooks_meta'
S1   = f'{ROOT}/v5/s1'
S2   = f'{ROOT}/v5/s2'
sys.path.insert(0, ROOT)

from v5.s2.config import CFG, DEVICE
from v5.s1.cog_arch.encoder import Encoder
from v5.s2.cog_arch.dm import DM, Projector
from v5.s2.data.dataset import VOCAB_SIZE, tokenizer

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_dataloader          = _load_module('data.dataloader', f'{S2}/data/dataloader.py')
get_stage2_dataloaders = _dataloader.get_stage2_dataloaders

# ── Cargar modelos ────────────────────────────────────────────────────────────

context_encoder = Encoder(
    vocab_size  = CFG.model.vocab_size,
    hidden_size = CFG.model.hidden_size,
    num_heads   = CFG.model.num_heads,
    num_layers  = CFG.model.num_layers,
    max_seq_len = CFG.model.max_seq_len,
).to(DEVICE)
target_encoder = copy.deepcopy(context_encoder).to(DEVICE)

s1_ckpt = torch.load(CFG.training.s1_ckpt, map_location=DEVICE, weights_only=False)
context_encoder.load_state_dict(s1_ckpt['context_encoder'])
target_encoder.load_state_dict(s1_ckpt['target_encoder'])

for enc in (context_encoder, target_encoder):
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False

dstc      = CFG.model.dstc
predictor = DM(
    num_frames  = CFG.model.max_seq_len,
    depth       = CFG.model.pred_num_layers,
    heads       = CFG.model.pred_num_heads,
    mlp_dim     = CFG.model.pred_hidden_size * 4,
    input_dim   = dstc,
    hidden_dim  = CFG.model.pred_hidden_size,
    output_dim  = dstc,
    dim_head    = 64,
    dropout     = 0.0,
    emb_dropout = 0.0,
).to(DEVICE)

s2_ckpt = torch.load(
    Path(CFG.logging.exp_dir) / 'best.pt',
    map_location=DEVICE, weights_only=False
)
predictor.load_state_dict(s2_ckpt['predictor'])
predictor.eval()
print(f'S2 checkpoint cargado (epoch {s2_ckpt["epoch"]}  val_loss={s2_ckpt["val_loss"]:.4f})')

# ── Dataloader ────────────────────────────────────────────────────────────────

_, val_loader = get_stage2_dataloaders(cfg_obj=CFG, tokenizer=tokenizer)

# ── Helpers ───────────────────────────────────────────────────────────────────

def mean_pool(hidden, mask):
    """(B, L, D) → (B, D) — pool solo tokens reales."""
    mask_f = mask.unsqueeze(-1).float()
    return (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)

# ── Extraer representaciones ──────────────────────────────────────────────────

@torch.no_grad()
def extract(loader):
    """
    Para cada par (turno_t, turno_{t+1}):
      - z_pred  : predicción del predictor sobre turno_t  → pool → (N, D)
      - z_real  : encoding real de turno_{t+1}            → pool → (N, D)
    """
    all_pred, all_real = [], []

    for batch in tqdm(loader, desc='Extrayendo representaciones'):
        ctx_ids  = batch['input_ids_a'].to(DEVICE)
        ctx_mask = batch['attention_mask_a'].to(DEVICE)
        tgt_ids  = batch['input_ids_b'].to(DEVICE)
        tgt_mask = batch['attention_mask_b'].to(DEVICE)

        # Encoder frozen
        ctx_h = context_encoder(ctx_ids, attention_mask=ctx_mask)
        tgt_h = target_encoder(tgt_ids,  attention_mask=tgt_mask)
        if isinstance(ctx_h, tuple): ctx_h = ctx_h[0]
        if isinstance(tgt_h, tuple): tgt_h = tgt_h[0]

        # Predictor: secuencia completa → pool
        pred_seq = predictor(ctx_h, ctx_h)              # (B, L, D)
        z_pred   = mean_pool(pred_seq, ctx_mask)        # (B, D)
        z_real   = mean_pool(tgt_h,   tgt_mask)        # (B, D)

        all_pred.append(z_pred.cpu())
        all_real.append(z_real.cpu())

    return torch.cat(all_pred), torch.cat(all_real)

# ── Retrieval ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def retrieval_accuracy(z_pred, z_real, ks=(1, 5, 10)):
    """
    Para cada z_pred[i], busca los k vecinos más cercanos en z_real.
    Hit@k = 1 si z_real[i] está entre los k más cercanos.

    Args:
        z_pred : (N, D) — representaciones predichas
        z_real : (N, D) — representaciones reales correspondientes
        ks     : top-k a evaluar

    Returns:
        dict con hit@k para cada k, más métricas de similaridad
    """
    # Normalizar para cosine similarity
    z_pred_n = F.normalize(z_pred, dim=-1)   # (N, D)
    z_real_n = F.normalize(z_real, dim=-1)   # (N, D)

    # Matriz de similaridad completa (N, N)
    # sim[i, j] = cosine_sim(z_pred[i], z_real[j])
    print(f'Computando matriz de similaridad ({z_pred_n.size(0)}x{z_real_n.size(0)}) …')
    sim = z_pred_n @ z_real_n.T              # (N, N)

    N = sim.size(0)
    results = {}

    for k in ks:
        # Top-k índices para cada query
        topk_idx = sim.topk(k, dim=-1).indices   # (N, k)
        # Ground truth: el índice correcto para query i es i
        targets  = torch.arange(N).unsqueeze(1)  # (N, 1)
        hits     = (topk_idx == targets).any(dim=-1).float()
        results[f'hit@{k}'] = hits.mean().item()

    # Métricas adicionales de similaridad
    # Diagonal = similaridad pred[i] con real[i] (el par correcto)
    diag_sim  = sim.diagonal()
    # Similaridad con un random negativo para comparar
    rand_idx  = (torch.arange(N) + torch.randint(1, N, (N,))) % N
    rand_sim  = sim[torch.arange(N), rand_idx]

    results['mean_correct_sim']  = diag_sim.mean().item()
    results['mean_random_sim']   = rand_sim.mean().item()
    results['sim_gap']           = (diag_sim - rand_sim).mean().item()
    results['N']                 = N

    return results

# ── Main ──────────────────────────────────────────────────────────────────────

print('\n── Extrayendo representaciones del val set …')
z_pred, z_real = extract(val_loader)
print(f'   z_pred: {z_pred.shape}  |  z_real: {z_real.shape}')

print('\n── Calculando retrieval accuracy …')
results = retrieval_accuracy(z_pred, z_real, ks=(1, 5, 10, 20))

print('\n' + '='*50)
print('  Retrieval Accuracy — Stage 2')
print('='*50)
for k in (1, 5, 10, 20):
    bar = '█' * int(results[f'hit@{k}'] * 40)
    print(f'  Hit@{k:<3} {results[f"hit@{k}"]:.3f}  {bar}')
print()
print(f'  Cosine sim (correcto) : {results["mean_correct_sim"]:+.4f}')
print(f'  Cosine sim (random)   : {results["mean_random_sim"]:+.4f}')
print(f'  Gap                   : {results["sim_gap"]:+.4f}')
print(f'  N val samples         : {results["N"]:,}')
print('='*50)

# ── Baseline aleatorio ────────────────────────────────────────────────────────
print('\n── Baseline aleatorio (para comparar):')
N = z_pred.size(0)
for k in (1, 5, 10, 20):
    expected = min(k / N * 100, 100)
    print(f'  Hit@{k:<3} esperado por azar: {k/N:.4f} ({expected:.2f}%)')


















# -----------------------------------------------------------------








# v5/s2/validate_prior.py
"""
Validate that the prior encoder actually conditions S2's prediction.

Three tests:
  1. Distribution test   — sampled z_ref lives on the Z_T manifold
  2. Conditioning test   — different z_ref → different S2 outputs
  3. Direction test      — z_ref closer to target → S2 output closer to target
"""

import sys, copy, torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

ROOT = '/content/notebooks_meta'
sys.path.insert(0, ROOT)

from v5.s2.config import CFG, DEVICE
from v5.s1.cog_arch.encoder import Encoder
from v5.s2.cog_arch.dm import DM, Projector
from v5.s2.prior_encoder import PriorEncoder

# ── Load everything ───────────────────────────────────────────────────────────

def mean_pool(hidden, mask):
    mask_f = mask.unsqueeze(-1).float()
    return (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)

# S1 encoders
context_encoder = Encoder(
    vocab_size=CFG.model.vocab_size, hidden_size=CFG.model.hidden_size,
    num_heads=CFG.model.num_heads,   num_layers=CFG.model.num_layers,
    max_seq_len=CFG.model.max_seq_len,
).to(DEVICE)
target_encoder = copy.deepcopy(context_encoder).to(DEVICE)

s1_ckpt = torch.load(CFG.training.s1_ckpt, map_location=DEVICE, weights_only=False)
context_encoder.load_state_dict(s1_ckpt['context_encoder'])
target_encoder.load_state_dict(s1_ckpt['target_encoder'])
context_encoder.eval(); target_encoder.eval()

# S2 predictor (retrained with meaningful c)
dstc      = CFG.model.dstc
predictor = DM(
    num_frames=CFG.model.max_seq_len, depth=CFG.model.pred_num_layers,
    heads=CFG.model.pred_num_heads,   mlp_dim=CFG.model.pred_hidden_size * 4,
    input_dim=dstc,                   hidden_dim=CFG.model.pred_hidden_size,
    output_dim=dstc,                  dim_head=64, dropout=0.0, emb_dropout=0.0,
).to(DEVICE)

s2_ckpt = torch.load(Path(CFG.logging.exp_dir) / 'best.pt',
                     map_location=DEVICE, weights_only=False)
predictor.load_state_dict(s2_ckpt['predictor'])
predictor.eval()
print(f'S2 loaded (epoch {s2_ckpt["epoch"]} val_loss={s2_ckpt["val_loss"]:.4f})')

# Prior encoder
prior_encoder = PriorEncoder(
    z_dim=CFG.model.hidden_size, hidden_dim=256, input_dim=None
).to(DEVICE)
prior_ckpt = torch.load(Path(CFG.logging.exp_dir) / 'prior_best.pt',
                        map_location=DEVICE, weights_only=False)
prior_encoder.load_state_dict(prior_ckpt['prior'])
prior_encoder.eval()
print(f'Prior loaded (epoch {prior_ckpt["epoch"]} val_loss={prior_ckpt["val_loss"]:.4f})')

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Distribution test
# Sampled z_ref should be statistically similar to real Z_T
# ─────────────────────────────────────────────────────────────────────────────

print('\n' + '='*60)
print('TEST 1 — Distribution of sampled z_ref vs real Z_T')
print('='*60)

import importlib.util
def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

S2_PATH = f'{ROOT}/v5/s2'
_dataset    = _load_module('data.dataset',    f'{S2_PATH}/data/dataset.py')
_dataloader = _load_module('data.dataloader', f'{S2_PATH}/data/dataloader.py')
tokenizer   = _dataset.tokenizer
_, val_loader = _dataloader.get_stage2_dataloaders(cfg_obj=CFG, tokenizer=tokenizer)

# Collect real Z_T
real_Z_T = []
with torch.no_grad():
    for batch in tqdm(val_loader, desc='Collecting real Z_T'):
        tgt_ids  = batch['input_ids_b'].to(DEVICE)
        tgt_mask = batch['attention_mask_b'].to(DEVICE)
        h        = target_encoder(tgt_ids, attention_mask=tgt_mask)
        if isinstance(h, tuple): h = h[0]
        real_Z_T.append(mean_pool(h, tgt_mask).cpu())

real_Z_T = torch.cat(real_Z_T)   # (N, 256)

# Sample z_ref from prior
N = real_Z_T.size(0)
with torch.no_grad():
    sampled_z_ref = prior_encoder.sample(n=N, device=DEVICE).cpu()

# Compare statistics
print(f'\n  Real Z_T:')
print(f'    mean norm  : {real_Z_T.norm(dim=-1).mean():.4f}')
print(f'    std  norm  : {real_Z_T.norm(dim=-1).std():.4f}')
print(f'    mean val   : {real_Z_T.mean():.4f}')
print(f'    std  val   : {real_Z_T.std():.4f}')

print(f'\n  Sampled z_ref:')
print(f'    mean norm  : {sampled_z_ref.norm(dim=-1).mean():.4f}')
print(f'    std  norm  : {sampled_z_ref.norm(dim=-1).std():.4f}')
print(f'    mean val   : {sampled_z_ref.mean():.4f}')
print(f'    std  val   : {sampled_z_ref.std():.4f}')

# Cosine similarity between sampled z_ref and nearest real Z_T
real_norm    = F.normalize(real_Z_T, dim=-1)
sampled_norm = F.normalize(sampled_z_ref, dim=-1)
sim_matrix   = sampled_norm[:100] @ real_norm[:100].T   # subsample for speed
nn_sim       = sim_matrix.max(dim=-1).values
print(f'\n  Nearest neighbor cosine sim (sampled→real): {nn_sim.mean():.4f} ± {nn_sim.std():.4f}')

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Conditioning test
# Different z_ref → different S2 outputs?
# ─────────────────────────────────────────────────────────────────────────────

print('\n' + '='*60)
print('TEST 2 — Does different z_ref produce different S2 output?')
print('='*60)

batch = next(iter(val_loader))
ctx_ids  = batch['input_ids_a'][:8].to(DEVICE)
ctx_mask = batch['attention_mask_a'][:8].to(DEVICE)

with torch.no_grad():
    ctx_h = context_encoder(ctx_ids, attention_mask=ctx_mask)
    if isinstance(ctx_h, tuple): ctx_h = ctx_h[0]

    # Prediction with zeros (old way)
    c_zeros = torch.zeros_like(ctx_h)
    pred_zeros = predictor(ctx_h, c_zeros)
    z_zeros    = mean_pool(pred_zeros, ctx_mask)

    # Prediction with z_ref from prior (sample 1)
    z_ref_1 = prior_encoder.sample(n=1, device=DEVICE).expand(8, -1)
    c_1     = z_ref_1.unsqueeze(1).expand_as(ctx_h)
    pred_1  = predictor(ctx_h, c_1)
    z_1     = mean_pool(pred_1, ctx_mask)

    # Prediction with z_ref from prior (sample 2 — different sample)
    z_ref_2 = prior_encoder.sample(n=1, device=DEVICE).expand(8, -1)
    c_2     = z_ref_2.unsqueeze(1).expand_as(ctx_h)
    pred_2  = predictor(ctx_h, c_2)
    z_2     = mean_pool(pred_2, ctx_mask)

# How different are the outputs?
sim_zeros_1 = F.cosine_similarity(z_zeros, z_1, dim=-1).mean()
sim_zeros_2 = F.cosine_similarity(z_zeros, z_2, dim=-1).mean()
sim_1_2     = F.cosine_similarity(z_1,     z_2, dim=-1).mean()

print(f'\n  Cosine sim (zeros vs sample_1) : {sim_zeros_1:.4f}')
print(f'  Cosine sim (zeros vs sample_2) : {sim_zeros_2:.4f}')
print(f'  Cosine sim (sample_1 vs sample_2): {sim_1_2:.4f}')
print(f'\n  → If sim(sample_1 vs sample_2) < sim(zeros vs sample_X):')
print(f'    different z_ref produces meaningfully different outputs ✓')

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Direction test
# z_ref closer to real target → S2 output closer to real target?
# ─────────────────────────────────────────────────────────────────────────────

print('\n' + '='*60)
print('TEST 3 — Does z_ref closer to target produce better prediction?')
print('='*60)

batch    = next(iter(val_loader))
ctx_ids  = batch['input_ids_a'][:32].to(DEVICE)
ctx_mask = batch['attention_mask_a'][:32].to(DEVICE)
tgt_ids  = batch['input_ids_b'][:32].to(DEVICE)
tgt_mask = batch['attention_mask_b'][:32].to(DEVICE)

with torch.no_grad():
    ctx_h = context_encoder(ctx_ids, attention_mask=ctx_mask)
    tgt_h = target_encoder(tgt_ids,  attention_mask=tgt_mask)
    if isinstance(ctx_h, tuple): ctx_h = ctx_h[0]
    if isinstance(tgt_h, tuple): tgt_h = tgt_h[0]

    z_tgt_real = mean_pool(tgt_h, tgt_mask)   # real target (B, D)

    sims_to_target = []

    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # Interpolate between random noise and real target
        # alpha=0.0 → pure noise, alpha=1.0 → real target
        noise  = prior_encoder.sample(n=32, device=DEVICE)
        z_ref  = (1 - alpha) * noise + alpha * z_tgt_real

        c      = z_ref.unsqueeze(1).expand_as(ctx_h)
        pred   = predictor(ctx_h, c)
        z_pred = mean_pool(pred, ctx_mask)

        sim = F.cosine_similarity(z_pred, z_tgt_real, dim=-1).mean()
        sims_to_target.append(sim.item())
        print(f'  alpha={alpha:.2f} (noise→target) | '
              f'pred similarity to real target: {sim:.4f}')

print(f'\n  → If similarity increases with alpha:')
print(f'    z_ref closer to target → prediction closer to target ✓')
print(f'\n  Δ sim (alpha=1.0 - alpha=0.0): {sims_to_target[-1]-sims_to_target[0]:+.4f}')

print('\n' + '='*60)
print('Validation complete.')
print('='*60)