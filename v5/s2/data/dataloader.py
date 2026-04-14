# dataloader





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
import datasets
import transformers

from data.dataset import get_dailydialog_dataset, JEPAMaskCollator, tokenizer






# ── Cell S2-3: Build dataloaders ──────────────────────────────────────────────

pair_dataset = get_turn_pair_dataset(
    cache_dir = CFG.cache_dir,
    tokenizer = tokenizer,
    block_size = CFG.max_seq_len,
)

collator = TurnPairCollator()

train_loader = torch.utils.data.DataLoader(
    pair_dataset['train'],
    batch_size  = CFG.batch_size,
    shuffle     = True,
    num_workers = 0,
    pin_memory  = True,
    collate_fn  = collator,
)
val_loader = torch.utils.data.DataLoader(
    pair_dataset['validation'],
    batch_size  = CFG.eval_batch,
    shuffle     = False,
    num_workers = 0,
    pin_memory  = True,
    collate_fn  = collator,
)

print(f"Train pairs : {len(pair_dataset['train']):,}")
print(f"Train batches : {len(train_loader)}  |  Val batches : {len(val_loader)}")
