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
import tokenizers






# ---------------------------------------------------------------------------
# JEPA dataloaders
# ---------------------------------------------------------------------------

def get_jepa_dataloaders(
    cfg_obj, # Changed parameter name to avoid conflict with global CFG
    tokenizer,
    skip_train: bool = False,
    skip_valid: bool = False,
    valid_seed: typing.Optional[int] = None,
):
  """Build train / validation DataLoaders for JEPA phase-1 training.

  Expects cfg_obj.num_target_spans and cfg_obj.target_span_length
  (defaults: 4 and 8).  Batch items have keys produced by JEPAMaskCollator.
  """
  num_gpus  = max(torch.cuda.device_count(), 1)
  block_size = cfg_obj.max_seq_len # Access via dot notation

  if cfg_obj.eval_batch % num_gpus != 0:
    raise ValueError(
      f'Eval batch size {cfg_obj.eval_batch} '
      f'not divisible by {num_gpus} GPUs.')

  dataset_dict = get_dailydialog_dataset(
    cache_dir=cfg_obj.cache_dir,
    tokenizer=tokenizer,
    block_size=block_size,
    raw_data_dir=cfg_obj.raw_data_dir, # Pass raw_data_dir explicitly
  )

  if tokenizer.mask_token_id is None:
    raise ValueError(
      f'Tokenizer must have a mask_token for JEPA masking: {tokenizer}')

  collator = JEPAMaskCollator(
    mask_token_id=tokenizer.mask_token_id,
    pad_token_id=tokenizer.pad_token_id,
    num_target_spans=cfg_obj.num_target_spans,
    target_span_length=cfg_obj.target_span_length,
  )

  train_loader = valid_loader = None

  if not skip_train:
    train_loader = torch.utils.data.DataLoader(
    dataset_dict['train'],
    batch_size=cfg_obj.batch_size,
    num_workers=cfg_obj.num_workers,
    pin_memory=True,
    shuffle=True,
    persistent_workers=cfg_obj.num_workers > 0,
    collate_fn=collator,
    )
    train_loader.tokenizer = tokenizer

  if not skip_valid:
    generator    = torch.Generator().manual_seed(valid_seed) if valid_seed else None
    valid_loader = torch.utils.data.DataLoader(
    dataset_dict['validation'],
    batch_size=cfg_obj.eval_batch,
    num_workers=cfg_obj.num_workers,
    pin_memory=True,
    shuffle=valid_seed is not None,
    generator=generator,
    persistent_workers=cfg_obj.num_workers > 0,
    collate_fn=collator,
    )
    valid_loader.tokenizer = tokenizer

  return train_loader, valid_loader