"""
data/data.py — Dataset for s3 decoder training.

Each item contains:
  - history_ids / history_masks / history_len  →  fed to frozen s2 to get ẑ
  - target_ids / target_mask                   →  fed to s2 DSE encoder (kept
                                                   for consistency but unused
                                                   in s3 loss)
  - dec_input_ids / dec_labels                 →  GPT-2 teacher-forcing inputs

GPT-2 sequence layout (for one sample):
  [BOS] t1 t2 ... tN [EOS]   ← dec_input_ids  (length: 1 + N + 1, padded to max)
       t1 t2 ... tN [EOS] -100 ...             ← dec_labels (shift by 1, pad=-100)

The prefix token (ẑ projected into GPT-2 space) is prepended inside the model,
not here — so the data only needs to hold the raw target token ids.
"""

import os
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, GPT2Tokenizer


# ── File loader (same as s2) ───────────────────────────────────────────────────

def load_dialogues(data_dir: str, split: str) -> List[List[str]]:
    data_dir = Path(data_dir)
    SPLIT_FILES = {
        "train": data_dir / "train"      / "train"      / "dialogues_train.txt",
        "val":   data_dir / "validation" / "validation" / "dialogues_validation.txt",
        "test":  data_dir / "test"       / "test"       / "dialogues_test.txt",
    }
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split '{split}'")
    path = SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    dialogues = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("__eou__") if p.strip()]
            if parts:
                dialogues.append(parts)
    return dialogues


def build_windows(
    dialogues: List[List[str]],
    min_turns: int,
    max_history: int,
) -> List[Tuple[List[str], str]]:
    windows = []
    for dialogue in dialogues:
        if len(dialogue) < min_turns:
            continue
        for t in range(1, len(dialogue)):
            history = dialogue[max(0, t - max_history): t]
            target  = dialogue[t]
            windows.append((history, target))
    return windows


# ── Dataset ───────────────────────────────────────────────────────────────────

class S3Dataset(Dataset):
    """
    Returns both the s2-format history tensors (to get ẑ from the frozen s2)
    and the GPT-2-format decoder targets (for cross-entropy loss).
    """

    def __init__(
        self,
        dialogues:      List[List[str]],
        dse_tokenizer,          # BERT-style tokenizer for history/target
        gpt2_tokenizer,         # GPT-2 tokenizer for decoder targets
        max_seq_len:    int,
        max_target_len: int,
        min_turns:      int,
        max_history:    int,
    ):
        self.dse_tok    = dse_tokenizer
        self.gpt2_tok   = gpt2_tokenizer
        self.max_seq    = max_seq_len
        self.max_tgt    = max_target_len
        self.max_hist   = max_history
        self.windows    = build_windows(dialogues, min_turns, max_history)

    def __len__(self):
        return len(self.windows)

    def _dse_encode(self, text: str) -> dict:
        return self.dse_tok(
            text,
            max_length=self.max_seq,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def _gpt2_encode(self, text: str):
        """
        Encode target utterance for teacher-forcing.
        Returns dec_input_ids and dec_labels tensors of fixed length.

        Layout:
          input:  [BOS] w1 w2 ... wN [EOS] [PAD] ...
          labels: w1 w2 ... wN [EOS] -100  -100  ...
          (labels are shifted left by 1 relative to input; pads are -100)
        """
        bos = self.gpt2_tok.bos_token_id
        eos = self.gpt2_tok.eos_token_id
        pad = self.gpt2_tok.pad_token_id   # set to eos_token_id in __init__

        token_ids = self.gpt2_tok.encode(text, add_special_tokens=False)
        # truncate to leave room for BOS + EOS
        token_ids = token_ids[: self.max_tgt - 2]

        full   = [bos] + token_ids + [eos]   # length: 1 + N + 1
        labels = token_ids + [eos]            # length: N + 1  (no BOS in labels)

        # pad to max_tgt
        pad_len_full   = self.max_tgt - len(full)
        pad_len_labels = self.max_tgt - len(labels)

        dec_input_ids = full   + [pad] * pad_len_full
        dec_labels    = labels + [-100] * pad_len_labels   # -100 = ignore index

        # also need attention mask for the decoder inputs
        dec_attn_mask = [1] * len(full) + [0] * pad_len_full

        return (
            torch.tensor(dec_input_ids, dtype=torch.long),
            torch.tensor(dec_labels,    dtype=torch.long),
            torch.tensor(dec_attn_mask, dtype=torch.long),
        )

    def __getitem__(self, idx: int) -> dict:
        history_texts, target_text = self.windows[idx]

        # ── s2-format history tensors ──────────────────────────────────────────
        ids_list, mask_list = [], []
        for utt in history_texts:
            enc = self._dse_encode(utt)
            ids_list.append(enc["input_ids"].squeeze(0))
            mask_list.append(enc["attention_mask"].squeeze(0))

        pad_ids  = torch.zeros(self.max_seq, dtype=torch.long)
        pad_mask = torch.zeros(self.max_seq, dtype=torch.long)
        history_len = len(ids_list)
        while len(ids_list) < self.max_hist:
            ids_list.append(pad_ids)
            mask_list.append(pad_mask)

        # ── s2-format target (DSE) ─────────────────────────────────────────────
        tgt_enc = self._dse_encode(target_text)

        # ── GPT-2 decoder targets ──────────────────────────────────────────────
        dec_input_ids, dec_labels, dec_attn_mask = self._gpt2_encode(target_text)

        return {
            # s2 inputs
            "history_ids":    torch.stack(ids_list),
            "history_masks":  torch.stack(mask_list),
            "history_len":    torch.tensor(history_len, dtype=torch.long),
            "target_ids":     tgt_enc["input_ids"].squeeze(0),
            "target_mask":    tgt_enc["attention_mask"].squeeze(0),
            # s3 decoder targets
            "dec_input_ids":  dec_input_ids,
            "dec_labels":     dec_labels,
            "dec_attn_mask":  dec_attn_mask,
        }


# ── Collate + DataLoader factory ──────────────────────────────────────────────

def collate_fn(batch: list) -> dict:
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0].keys()}


def make_dataloaders(cfg) -> tuple:
    # DSE tokenizer (BERT-style) — for history encoding via s2
    dse_tokenizer = AutoTokenizer.from_pretrained(
        cfg.encoder_model,
        cache_dir=cfg.cache_dir,
    )

    # GPT-2 tokenizer — for decoder targets
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(
        cfg.decoder_model,
        cache_dir=cfg.cache_dir,
    )
    # GPT-2 has no pad token by default — use EOS as pad
    gpt2_tokenizer.pad_token    = gpt2_tokenizer.eos_token
    gpt2_tokenizer.pad_token_id = gpt2_tokenizer.eos_token_id

    loaders = {}
    for split in ("train", "val", "test"):
        try:
            dialogues = load_dialogues(cfg.raw_data_dir, split)
        except FileNotFoundError:
            if split == "test":
                loaders[split] = None
                continue
            raise

        dataset = S3Dataset(
            dialogues      = dialogues,
            dse_tokenizer  = dse_tokenizer,
            gpt2_tokenizer = gpt2_tokenizer,
            max_seq_len    = cfg.max_seq_len,
            max_target_len = cfg.max_target_len,
            min_turns      = cfg.min_turns,
            max_history    = cfg.max_history,
        )
        print(f"[data] {split:>5} — {len(dialogues):>6} dialogues → "
              f"{len(dataset):>7} windows")

        loaders[split] = DataLoader(
            dataset,
            batch_size  = cfg.batch_size,
            shuffle     = (split == "train"),
            num_workers = cfg.num_workers,
            collate_fn  = collate_fn,
            pin_memory  = True,
        )

    return loaders["train"], loaders["val"], loaders.get("test")