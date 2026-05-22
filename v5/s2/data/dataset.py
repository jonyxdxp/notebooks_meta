import os
import torch
import datasets
import transformers

# ── reuse tokenizer/config from s1 ───────────────────────────────────────────
from v5.s1.data.dataset import tokenizer, CFG, VOCAB_SIZE

MAX_TURNS = 6   # history window: up to 5 context turns + 1 target

def get_multiturn_dataset(
    cache_dir: str,
    tokenizer,
    block_size: int = 128,
    max_turns: int = MAX_TURNS,
    raw_data_dir: str = CFG.raw_data_dir,
) -> datasets.DatasetDict:
    """
    Each example: a conversation window of up to max_turns turns.
      history_ids   : (max_turns-1, block_size)  — context turns (padded)
      history_masks : (max_turns-1, block_size)
      history_len   : int  — how many context turns are real (vs padded)
      tgt_ids       : (block_size,)  — turn to predict
      tgt_mask      : (block_size,)
    """
    _cache_path = os.path.join(cache_dir, f'multiturn_mt{max_turns}_bs{block_size}')
    if os.path.exists(_cache_path):
        print(f'Loading from cache: {_cache_path}')
        return datasets.load_from_disk(_cache_path).with_format('torch')

    split_txts = {
        'train':      os.path.join(raw_data_dir, 'train',      'dialogues_train.txt'),
        'validation': os.path.join(raw_data_dir, 'validation', 'dialogues_validation.txt'),
        'test':       os.path.join(raw_data_dir, 'test',       'dialogues_test.txt'),
    }

    # ── download raw data if txt files missing ────────────────────────────────
    if not all(os.path.exists(p) for p in split_txts.values()):
        print('Downloading DailyDialog …')
        raw = None
        for repo in ['benjaminbeilharz/better_daily_dialog']:
            try:
                raw = datasets.load_dataset(repo)
                import pandas as pd
                split_dicts = {}
                for split in raw:
                    df = raw[split].to_pandas()
                    dialogs = (
                        df.sort_values(['dialog_id', 'turn_type'])
                          .groupby('dialog_id')['utterance']
                          .apply(list).tolist()
                    )
                    split_dicts[split] = datasets.Dataset.from_dict({'dialog': dialogs})
                raw = datasets.DatasetDict(split_dicts)
                break
            except Exception as e:
                print(f'  ✗ {e}'); raw = None

        if raw is None:
            raise RuntimeError("Download failed.")

        for split, txt_path in split_txts.items():
            os.makedirs(os.path.dirname(txt_path), exist_ok=True)
            with open(txt_path, 'w', encoding='utf-8') as f:
                for example in raw[split]:
                    turns = [t.strip() for t in example['dialog'] if t.strip()]
                    if turns:
                        f.write(' __eou__ '.join(turns) + ' __eou__\n')
            print(f'  wrote {split} → {txt_path}')


    

    # ── load dialogs ──────────────────────────────────────────────────────────
    def _load_windows(filepath):
        """
        For each conversation, slide a window and yield:
          history = turns[max(0, i-ctx_len) : i]
          target  = turns[i]
        where ctx_len = max_turns - 1
        """
        ctx_len = max_turns - 1
        all_histories, all_history_lens, all_targets = [], [], []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                turns = [t.strip() for t in line.split('__eou__') if t.strip()]
                if len(turns) < 2:
                    continue

                for i in range(1, len(turns)):
                    history = turns[max(0, i - ctx_len): i]  # 1..ctx_len real turns
                    target  = turns[i]

                    # pad history on the LEFT with empty strings
                    pad_len = ctx_len - len(history)
                    padded  = [''] * pad_len + history

                    all_histories.append(padded)          # list of ctx_len strings
                    all_history_lens.append(len(history)) # how many are real
                    all_targets.append(target)

        return all_histories, all_history_lens, all_targets

    PAD_ID = tokenizer.pad_token_id
    ctx_len = max_turns - 1

    def _tokenize(examples):
        # histories: list[list[str]]  shape (batch, ctx_len)
        # flatten → tokenize → reshape
        batch_size = len(examples['target'])

        # flatten all history turns into one list
        flat_hist = [turn for hist in examples['history'] for turn in hist]
        enc_hist  = tokenizer(
            flat_hist,
            max_length=block_size, padding='max_length',
            truncation=True, return_attention_mask=True,
            return_token_type_ids=False,
        )
        # reshape back to (batch, ctx_len, block_size)
        hist_ids   = [enc_hist['input_ids'][i*ctx_len:(i+1)*ctx_len]   for i in range(batch_size)]
        hist_masks = [enc_hist['attention_mask'][i*ctx_len:(i+1)*ctx_len] for i in range(batch_size)]

        # zero out padded turns (empty string tokenizes to [CLS][SEP] — mask those out)
        for b in range(batch_size):
            real = examples['history_len'][b]
            for t in range(ctx_len - real):   # left-padded positions
                hist_ids[b][t]   = [PAD_ID] * block_size
                hist_masks[b][t] = [0]       * block_size

        enc_tgt = tokenizer(
            examples['target'],
            max_length=block_size, padding='max_length',
            truncation=True, return_attention_mask=True,
            return_token_type_ids=False,
        )

        return {
            'history_ids':   hist_ids,
            'history_masks': hist_masks,
            'history_len':   examples['history_len'],
            'tgt_ids':       enc_tgt['input_ids'],
            'tgt_mask':      enc_tgt['attention_mask'],
        }

    print('Building multi-turn dataset …')
    tokenized_splits = {}
    for split, txt_path in split_txts.items():
        histories, hist_lens, targets = _load_windows(txt_path)
        print(f'  {split:12s}: {len(targets):>6,} windows')
        raw_ds = datasets.Dataset.from_dict({
            'history':     histories,
            'history_len': hist_lens,
            'target':      targets,
        })
        tokenized_splits[split] = raw_ds.map(
            _tokenize, batched=True, num_proc=1,
            remove_columns=['history', 'target'],
            desc=f'Tokenizing {split}',
        )

    dataset_dict = datasets.DatasetDict(tokenized_splits)
    os.makedirs(cache_dir, exist_ok=True)
    dataset_dict.save_to_disk(_cache_path)
    print(f'Saved to cache: {_cache_path}')
    return dataset_dict.with_format('torch')


class MultiTurnCollator:
    def __call__(self, batch):
        return {
            'history_ids':   torch.stack([b['history_ids'].detach().clone()   if torch.is_tensor(b['history_ids'])   else torch.tensor(b['history_ids'])   for b in batch]),
            'history_masks': torch.stack([b['history_masks'].detach().clone() if torch.is_tensor(b['history_masks']) else torch.tensor(b['history_masks']) for b in batch]),
            'history_len':   torch.tensor([b['history_len'] for b in batch]),
            'tgt_ids':       torch.stack([b['tgt_ids']   for b in batch]),
            'tgt_mask':      torch.stack([b['tgt_mask']  for b in batch]),
        }