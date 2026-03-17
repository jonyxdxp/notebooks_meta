
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











tokenizer = transformers.AutoTokenizer.from_pretrained(CFG.tokenizer_name)
# BERT already has [MASK], [PAD], [CLS], [SEP] — nothing to patch.
# For GPT-style tokenizers that lack these tokens, add them here.

VOCAB_SIZE = tokenizer.vocab_size
print(f"Vocab : {VOCAB_SIZE}  |  mask_token_id : {tokenizer.mask_token_id}")

















# ── Cell 7: Dataset ───────────────────────────────────────────────────────────

def get_dailydialog_dataset(
    cache_dir: str,
    tokenizer,
    block_size: int = 128,
    num_proc: int = 1,
    raw_data_dir: str = '/content/drive/MyDrive/data/dailydialog_raw',
) -> datasets.DatasetDict:

    _cache_path = os.path.join(cache_dir, f'dailydialog_jepa_bs{block_size}')
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

    # ── Step 2: read utterances ───────────────────────────────────────────────
    def _load_utterances(filepath):
        utterances = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                turns = [t.strip() for t in line.split('__eou__') if t.strip()]
                utterances.extend(turns)
        return utterances

    def _tokenize(examples):
        return tokenizer(
            examples['text'],
            max_length=block_size,
            padding='max_length',
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

    print('Building DailyDialog JEPA dataset …')
    tokenized_splits = {}
    for split, txt_path in split_txts.items():
        utterances = _load_utterances(txt_path)
        print(f'  {split:12s}: {len(utterances):>6,} utterances')
        raw_ds = datasets.Dataset.from_dict({'text': utterances})
        tokenized_splits[split] = raw_ds.map(
            _tokenize, batched=True, num_proc=num_proc,
            remove_columns=['text'], desc=f'Tokenizing {split}')

    dataset_dict = datasets.DatasetDict(tokenized_splits)
    os.makedirs(cache_dir, exist_ok=True)
    dataset_dict.save_to_disk(_cache_path)
    print(f'Saved to cache: {_cache_path}')
    return dataset_dict.with_format('torch')






















# ── Cell 8: Collator ──────────────────────────────────────────────────────────

class JEPAMaskCollator:
    """
    Returns context (masked) and target (clean) views of the same sequence.

    Keys returned:
      context_input_ids      (B, L) — masked tokens replaced with [MASK]
      context_attention_mask (B, L)
      target_input_ids       (B, L) — original, unmasked
      target_attention_mask  (B, L)
      target_mask            (B, L) bool — True at positions that were masked
    """

    def __init__(
        self,
        mask_token_id: int,
        pad_token_id: int,
        num_target_spans: int = 4,
        target_span_length: int = 8,
    ):
        self.mask_token_id    = mask_token_id
        self.pad_token_id     = pad_token_id
        self.num_target_spans = num_target_spans
        self.target_span_len  = target_span_length

    def __call__(self, batch):
        input_ids      = torch.stack([b['input_ids']      for b in batch])
        attention_mask = torch.stack([b['attention_mask'] for b in batch])
        B, L = input_ids.shape

        context_input_ids = input_ids.clone()
        target_mask       = torch.zeros(B, L, dtype=torch.bool)

        for i in range(B):
            valid_len = int(attention_mask[i].sum().item())
            for s, e in self._sample_spans(1, valid_len - 1):
                target_mask[i, s:e] = True
            context_input_ids[i, target_mask[i]] = self.mask_token_id

        return {
            'context_input_ids':      context_input_ids,
            'context_attention_mask': attention_mask,
            'target_input_ids':       input_ids,
            'target_attention_mask':  attention_mask,
            'target_mask':            target_mask,
        }

    def _sample_spans(self, maskable_start, maskable_end):
        region_len = maskable_end - maskable_start
        if region_len <= 0:
            return []
        span_len  = min(self.target_span_len, region_len)
        available = list(range(maskable_start, maskable_end - span_len + 1))
        spans = []
        for _ in range(self.num_target_spans):
            if not available:
                break
            idx = torch.randint(len(available), (1,)).item()
            s, e = available[idx], available[idx] + span_len
            spans.append((s, e))
            available = [x for x in available if (x + span_len <= s) or (x >= e)]
        return spans