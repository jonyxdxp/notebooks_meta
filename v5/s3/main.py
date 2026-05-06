#!/usr/bin/env python3
"""
Stage 3: Cross-attention Decoder Training
Trains S3 decoder to generate turn_{t+1} text from Z_T_real.
At inference, Z_T_real is swapped for z_fused from PoE.
S1 and S2 stay completely frozen.
"""

import sys
import os
import copy
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = '/content/notebooks_meta'
S1       = f'{ROOT}/v5/s1'
S2       = f'{ROOT}/v5/s2'
S3       = f'{ROOT}/v5/s3'
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = Path('/content/drive/MyDrive/metanet/v5/s3/checkpoints')
SAVE_DIR.mkdir(parents=True, exist_ok=True)



sys.path.insert(0, ROOT)

from v5.s1.cog_arch.encoder import Encoder
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
    return (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)

# ── Dataset ───────────────────────────────────────────────────────────────────
class S3Dataset(Dataset):
    def __init__(self, chunk_files):
        self.chunks = []
        offset = 0
        self.index = []   # maps global idx → (chunk_idx, local_idx)
        for ci, cf in enumerate(chunk_files):
            c = torch.load(cf, weights_only=False)
            self.chunks.append(c)
            for li in range(c['z_T'].shape[0]):
                self.index.append((ci, li))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ci, li = self.index[idx]
        c = self.chunks[ci]
        return {
            'z_T':     c['z_T'][li].float(),
            'tgt_ids': c['tgt_ids'][li],
            'tgt_mask':c['tgt_mask'][li],
        }

# ── Extract Z_T_real and collect target token ids ─────────────────────────────
def extract_s3_data(s1_encoder, predictor, loader, device, save_path):
    save_path = Path(save_path)
    z_chunks, id_chunks, mask_chunks = [], [], []
    chunk_idx = 0

    def flush(idx):
        p = save_path.parent / f'{save_path.stem}_chunk{idx}.pt'
        torch.save({
            'z_T':      torch.cat(z_chunks).float(),
            'tgt_ids':  torch.cat(id_chunks),
            'tgt_mask': torch.cat(mask_chunks),
        }, p)
        z_chunks.clear(); id_chunks.clear(); mask_chunks.clear()
        torch.cuda.empty_cache()
        print(f'  flushed → {p}')
        return idx + 1

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc='Extracting')):
            hist_ids   = batch['history_ids'].to(device)
            hist_masks = batch['history_masks'].to(device)
            tgt_ids    = batch['tgt_ids'].to(device)
            tgt_mask   = batch['tgt_mask'].to(device)
            B, T, L    = hist_ids.shape
            flat_ids   = hist_ids.view(B*T, L)
            flat_masks = hist_masks.view(B*T, L)
            valid      = flat_masks.sum(-1) > 0
            h_flat = s1_encoder(flat_ids, attention_mask=flat_masks)
            if isinstance(h_flat, tuple): h_flat = h_flat[0]
            h_flat[~valid] = 0.0
            z_pred = predictor(h_flat.view(B, T, L, -1))
            z_chunks.append(z_pred.cpu().half())
            id_chunks.append(tgt_ids.cpu())
            mask_chunks.append(tgt_mask.cpu())
            if (i + 1) % 100 == 0:
                chunk_idx = flush(chunk_idx)

    if z_chunks:
        chunk_idx = flush(chunk_idx)

    # return chunk file list — Dataset will load lazily
    chunk_files = sorted(save_path.parent.glob(f'{save_path.stem}_chunk*.pt'))
    print(f'Extraction done: {chunk_idx} chunks → {save_path.parent}')
    return chunk_files

# ── Loss ──────────────────────────────────────────────────────────────────────

def lm_loss(logits, tgt_ids, tgt_mask):
    """
    Standard language modeling loss with teacher forcing.

    logits   : (B, T, vocab_size)
    tgt_ids  : (B, T)
    tgt_mask : (B, T) — ignore padding positions

    Shift: predict token[i+1] from token[i]
    Input  tokens: [CLS] t1 t2 t3 [SEP] [PAD]
    Target tokens: t1    t2 t3 [SEP] [PAD] [PAD]
    """
    # Shift
    shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
    shift_labels = tgt_ids[:, 1:].contiguous()       # (B, T-1)
    shift_mask   = tgt_mask[:, 1:].contiguous()      # (B, T-1)

    # Flatten
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='none',
    )

    # Mask padding
    loss = (loss * shift_mask.view(-1).float()).sum()
    loss = loss / shift_mask.sum().clamp(min=1)

    return loss

    

# ── Load S1 encoder (frozen) ──────────────────────────────────────────────────

s1_encoder = Encoder(
    vocab_size  = CFG.model.vocab_size,
    hidden_size = CFG.model.hidden_size,
    num_heads   = CFG.model.num_heads,
    num_layers  = CFG.model.num_layers,
    max_seq_len = CFG.model.max_seq_len,
).to(DEVICE)

s1_ckpt = torch.load(CFG.training.s1_ckpt, map_location=DEVICE, weights_only=False)
s1_encoder.load_state_dict(s1_ckpt['context_encoder'])
s1_encoder.eval()
for p in s1_encoder.parameters():
    p.requires_grad = False
print('S1 encoder loaded and frozen.')


# DELETE TurnPredictor class and replace with:
from v5.s2.cog_arch.dm import DM

predictor = DM(
    num_turns  = CFG.model.max_turns - 1,   # 5
    seq_len    = CFG.model.max_seq_len,      # 128
    depth      = CFG.model.pred_num_layers,
    heads      = CFG.model.pred_num_heads,
    mlp_dim    = CFG.model.pred_hidden_size * 4,
    input_dim  = CFG.model.hidden_size,
    hidden_dim = CFG.model.pred_hidden_size,
    output_dim = CFG.model.hidden_size,
    dim_head=64, dropout=0.1, emb_dropout=0.1,
).to(DEVICE)


s2_ckpt = torch.load('/content/drive/MyDrive/metanet/v5/s2/checkpoints/best.pt', map_location=DEVICE, weights_only=False)
predictor.load_state_dict(s2_ckpt['predictor'])
predictor.eval()
for p in predictor.parameters(): p.requires_grad = False

print('S2 predictor loaded and frozen.')



# ── Dataloaders ───────────────────────────────────────────────────────────────

train_loader, val_loader = get_stage2_dataloaders(
    cfg_obj=CFG, tokenizer=tokenizer,
    # override batch size for extraction
)

print(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')

# ── Extract and cache S3 data ─────────────────────────────────────────────────


# REPLACE the entire extraction section with just this:

cache_train    = Path('/content/s3_data_train.pt')   # local, fast
cache_val      = Path('/content/s3_data_val.pt')
best_ckpt_path = SAVE_DIR / 'best.pt'                # decoder checkpoint still on Drive

manifest_train = Path(str(cache_train) + '.manifest')
manifest_val   = Path(str(cache_val)   + '.manifest')

if not manifest_train.exists():
    print('Extracting train...')
    train_chunks = extract_s3_data(s1_encoder, predictor, train_loader, DEVICE, cache_train)
    torch.save([str(f) for f in train_chunks], manifest_train)
else:
    train_chunks = [Path(f) for f in torch.load(manifest_train, weights_only=False)]

if not manifest_val.exists():
    print('Extracting val...')
    val_chunks = extract_s3_data(s1_encoder, predictor, val_loader, DEVICE, cache_val)
    torch.save([str(f) for f in val_chunks], manifest_val)
else:
    val_chunks = [Path(f) for f in torch.load(manifest_val, weights_only=False)]

train_ds = S3Dataset(train_chunks)
val_ds   = S3Dataset(val_chunks)
print(f'Train: {len(train_ds)} | Val: {len(val_ds)}')


# ── Build datasets ────────────────────────────────────────────────────────────



s3_train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0)
s3_val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0)

# ── Decoder ───────────────────────────────────────────────────────────────────

decoder = Decoder(
    vocab_size        = CFG.model.vocab_size,   # 30522
    hidden_size       = 256,
    num_heads         = 4,
    num_layers        = 4,
    max_seq_len       = CFG.model.max_seq_len,  # 128
    context_dim       = CFG.model.hidden_size,  # 256 — must match S1
).to(DEVICE)

total_params = sum(p.numel() for p in decoder.parameters())
print(f'Decoder params: {total_params:,}')




# ── Optimizer ─────────────────────────────────────────────────────────────────

N_EPOCHS = 20
optimizer = AdamW(decoder.parameters(), lr=1e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=N_EPOCHS, eta_min=1e-5)


# THEN resume block:
best_val_loss = float('inf')
start_epoch = 1
best_ckpt_path = SAVE_DIR / 'best.pt'
if best_ckpt_path.exists():
    ckpt = torch.load(best_ckpt_path, map_location=DEVICE, weights_only=False)
    decoder.load_state_dict(ckpt['decoder'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    best_val_loss = ckpt['val_loss']
    start_epoch = ckpt['epoch'] + 1
    print(f'Resumed from epoch {ckpt["epoch"]}')

# ── Training loop ─────────────────────────────────────────────────────────────

history = {
    'train_loss': [], 'train_ppl': [],
    'val_loss':   [], 'val_ppl':   [],
}

N_EPOCHS      = 20

print(f'\n{"="*60}')
print(f'  S3 Decoder — {N_EPOCHS} epochs   device={DEVICE}')
print(f'{"="*60}\n')

for epoch in range(start_epoch, N_EPOCHS + 1):

    # ── Train ────────────────────────────────────────────────────────────────
    decoder.train()
    t_loss = 0.0; n = 0

    for batch in tqdm(s3_train_loader, desc=f'Epoch {epoch:02d}', leave=False):
        z_T      = batch['z_T'].to(DEVICE)        # (B, D) — oracle conditioning
        tgt_ids  = batch['tgt_ids'].to(DEVICE)    # (B, L)
        tgt_mask = batch['tgt_mask'].to(DEVICE)   # (B, L)

        # Forward: decoder generates turn_{t+1} conditioned on Z_T_real
        logits = decoder(tgt_ids, z_T, tgt_mask)  # (B, L, vocab)

        loss = lm_loss(logits, tgt_ids, tgt_mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()

        t_loss += loss.item(); n += 1

    scheduler.step()

    # ── Val ──────────────────────────────────────────────────────────────────
    decoder.eval()
    v_loss = 0.0; m = 0

    with torch.no_grad():
        for batch in s3_val_loader:
            z_T      = batch['z_T'].to(DEVICE)
            tgt_ids  = batch['tgt_ids'].to(DEVICE)
            tgt_mask = batch['tgt_mask'].to(DEVICE)

            logits = decoder(tgt_ids, z_T, tgt_mask)
            loss   = lm_loss(logits, tgt_ids, tgt_mask)
            v_loss += loss.item(); m += 1

    train_loss = t_loss / n
    val_loss   = v_loss / m
    train_ppl  = torch.exp(torch.tensor(train_loss)).item()
    val_ppl    = torch.exp(torch.tensor(val_loss)).item()

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_ppl'].append(train_ppl)
    history['val_ppl'].append(val_ppl)

    print(
        f'Epoch {epoch:02d}/{N_EPOCHS}  '
        f'train_loss={train_loss:.4f}  train_ppl={train_ppl:.1f}  '
        f'val_loss={val_loss:.4f}  val_ppl={val_ppl:.1f}  '
        f'lr={optimizer.param_groups[0]["lr"]:.2e}'
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch':      epoch,
            'decoder':    decoder.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'scheduler':  scheduler.state_dict(),
            'val_loss':   best_val_loss,
            'val_ppl':    val_ppl,
            'config': {
                'vocab_size':   CFG.model.vocab_size,
                'hidden_size':  256,
                'num_heads':    4,
                'num_layers':   4,
                'max_seq_len':  CFG.model.max_seq_len,
                'context_dim':  CFG.model.hidden_size,
            }
        }, SAVE_DIR / 'best.pt')
        print(f'  ✓ saved → {SAVE_DIR}/best.pt')

# ── Quick generation sample ───────────────────────────────────────────────────

print('\n── Generation samples (oracle conditioning) ──')
decoder.eval()

with torch.no_grad():
    # Take first 3 val examples
    # Take first 3 val examples from dataset directly
    sample = [val_ds[i] for i in range(3)]
    sample_z_T  = torch.stack([s['z_T']     for s in sample]).to(DEVICE)  # (3, L, D)
    sample_tgt  = torch.stack([s['tgt_ids'] for s in sample])              # (3, L)

    # Prompt: just [CLS] token (101 in BERT)
    prompt = torch.full((3, 1), 101, dtype=torch.long, device=DEVICE)

    generated = decoder.generate(
        prompt_ids     = prompt,
        z_fused        = sample_z_T,
        max_new_tokens = 30,
        temperature    = 0.8,
        top_k          = 50,
    )

    print('\nTarget turns:')
    for i in range(3):
        tokens = sample_tgt[i].tolist()
        text   = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f'  [{i}] {text}')

    print('\nGenerated turns (oracle z_T):')
    for i in range(3):
        tokens = generated[i].tolist()
        text   = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f'  [{i}] {text}')

# ── Plot ──────────────────────────────────────────────────────────────────────

epochs_r = range(1, len(history['train_loss']) + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs_r, history['train_loss'], 'b-', label='train')
axes[0].plot(epochs_r, history['val_loss'],   'r-', label='val')
axes[0].set_title('Cross Entropy Loss')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_r, history['train_ppl'], 'b-', label='train')
axes[1].plot(epochs_r, history['val_ppl'],   'r-', label='val')
axes[1].set_title('Perplexity')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(SAVE_DIR / 'training_curves_s3.png', dpi=150)
plt.close(fig)
print(f'\nPlot saved → {SAVE_DIR}/training_curves_s3.png')
print(f'Best val_loss: {best_val_loss:.4f}  val_ppl: {torch.exp(torch.tensor(best_val_loss)).item():.1f}')