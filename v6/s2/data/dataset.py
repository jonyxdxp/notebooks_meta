
# ── Cell 3: Imports ───────────────────────────────────────────────────────────


import os  # <-- Falta esto

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

CFG = _C(CFG_dict)
print("Config:")






tokenizer = transformers.AutoTokenizer.from_pretrained(CFG.tokenizer_name)
# BERT already has [MASK], [PAD], [CLS], [SEP] — nothing to patch.
# For GPT-style tokenizers that lack these tokens, add them here.

VOCAB_SIZE = tokenizer.vocab_size
print(f"Vocab : {VOCAB_SIZE}  |  mask_token_id : {tokenizer.mask_token_id}")
















# ── Cell S2-1: Dataset — consecutive turn pairs ───────────────────────────────

def get_turn_pair_dataset(
    cache_dir: str,
    tokenizer,
    block_size: int = 128,
    raw_data_dir: str = CFG.raw_data_dir,
) -> datasets.DatasetDict:
    """
    Builds a dataset of (turn_t, turn_{t+1}) pairs from dialogs.
    Each example: two consecutive utterances from alternating speakers.
    """
    _cache_path = os.path.join(cache_dir, f'turn_pairs_bs{block_size}')
    if os.path.exists(_cache_path):
        print(f'Loading from cache: {_cache_path}')
        return datasets.load_from_disk(_cache_path).with_format('torch')

    split_txts = {
        'train':      os.path.join(raw_data_dir, 'train',      'dialogues_train.txt'),
        'validation': os.path.join(raw_data_dir, 'validation', 'dialogues_validation.txt'),
        'test':       os.path.join(raw_data_dir, 'test',       'dialogues_test.txt'),
    }

     # ── Step 1: download raw data if txt files not already on disk ────────────
    if not all(os.path.exists(p) for p in split_txts.values()):
        print('Downloading DailyDialog …')

        raw = None

        for repo in ['benjaminbeilharz/better_daily_dialog']:
            try:
                print(f'  trying {repo} …')
                raw = datasets.load_dataset(repo)

                import pandas as pd
                split_dicts = {}
                for split in raw:
                    df = raw[split].to_pandas()
                    dialogs = (
                        df.sort_values(['dialog_id', 'turn_type'])
                          .groupby('dialog_id')['utterance']
                          .apply(list)
                          .tolist()
                    )
                    split_dicts[split] = datasets.Dataset.from_dict({'dialog': dialogs})
                raw = datasets.DatasetDict(split_dicts)

                print(f'  ✓ loaded {sum(len(raw[s]) for s in raw):,} dialogues from {repo}')
                break
            except Exception as e:
                print(f'  ✗ {e}')
                raw = None

        if raw is None:
            raise RuntimeError("All dataset sources failed. Check your internet connection.")

        for split, txt_path in split_txts.items():
            os.makedirs(os.path.dirname(txt_path), exist_ok=True)
            with open(txt_path, 'w', encoding='utf-8') as f:
                for example in raw[split]:
                    turns = [t.strip() for t in example['dialog'] if t.strip()]
                    if turns:
                        f.write(' __eou__ '.join(turns) + ' __eou__\n')
            with open(txt_path) as f:
                n = sum(1 for l in f if l.strip())
            print(f'  wrote {split} ({n:,} dialogues) → {txt_path}')

    def _load_pairs(filepath):
        """Read dialogs and yield (turn_t, turn_{t+1}) consecutive pairs."""
        pairs_a, pairs_b = [], []   # turn_t, turn_{t+1}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                turns = [t.strip() for t in line.split('__eou__') if t.strip()]
                # Every consecutive pair: (turn_0→turn_1), (turn_1→turn_2), ...
                for i in range(len(turns) - 1):
                    pairs_a.append(turns[i])
                    pairs_b.append(turns[i + 1])
        return pairs_a, pairs_b

    def _tokenize_pair(examples):
        enc_a = tokenizer(
            examples['turn_a'],
            max_length=block_size, padding='max_length',
            truncation=True, return_attention_mask=True,
            return_token_type_ids=False,
        )
        enc_b = tokenizer(
            examples['turn_b'],
            max_length=block_size, padding='max_length',
            truncation=True, return_attention_mask=True,
            return_token_type_ids=False,
        )
        return {
            'input_ids_a':      enc_a['input_ids'],
            'attention_mask_a': enc_a['attention_mask'],
            'input_ids_b':      enc_b['input_ids'],
            'attention_mask_b': enc_b['attention_mask'],
        }

    print('Building turn-pair dataset …')
    tokenized_splits = {}
    for split, txt_path in split_txts.items():
        a, b = _load_pairs(txt_path)
        print(f'  {split:12s}: {len(a):>6,} turn pairs')
        raw_ds = datasets.Dataset.from_dict({'turn_a': a, 'turn_b': b})
        tokenized_splits[split] = raw_ds.map(
            _tokenize_pair, batched=True, num_proc=1,
            remove_columns=['turn_a', 'turn_b'],
            desc=f'Tokenizing {split}',
        )

    dataset_dict = datasets.DatasetDict(tokenized_splits)
    os.makedirs(cache_dir, exist_ok=True)
    dataset_dict.save_to_disk(_cache_path)
    print(f'Saved to cache: {_cache_path}')
    return dataset_dict.with_format('torch')



















# ── Cell S2-2: Collator ───────────────────────────────────────────────────────

class TurnPairCollator:
    """Simple collator — no masking needed, just stack pairs."""
    def __call__(self, batch):
        return {
            'input_ids_a':      torch.stack([b['input_ids_a']      for b in batch]),
            'attention_mask_a': torch.stack([b['attention_mask_a'] for b in batch]),
            'input_ids_b':      torch.stack([b['input_ids_b']      for b in batch]),
            'attention_mask_b': torch.stack([b['attention_mask_b'] for b in batch]),
        }