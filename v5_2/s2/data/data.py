"""
data.py — Dataset and DataLoader for the dialog next-turn prediction task.

Expected data format (flexible):
  - JSON:  list of dialogues, each dialogue is a list of utterance strings.
           e.g. [ ["Hi", "Hello, how can I help?", "I need a booking."], ... ]
  - JSONL: one dialogue per line, same inner format.
  - TXT:   utterances separated by "\t" or newline-per-utterance,
           dialogues separated by blank line (DailyDialog raw style).

The loader creates all valid (history, target) windows from each dialogue:
  For a dialogue [u1, u2, u3, u4], it produces:
    ([u1],         u2)
    ([u1, u2],     u3)
    ([u1, u2, u3], u4)
  capped at cfg.max_history turns of history.
"""

import json
import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_dialogues(data_dir: str, split: str) -> List[List[str]]:
    """
    Try several common file formats for the given split (train/val/test).
    Returns a list of dialogues; each dialogue is a list of utterance strings.
    """
    data_dir = Path(data_dir)
    candidates = [
        data_dir / f"{split}.json",
        data_dir / f"{split}.jsonl",
        data_dir / f"{split}.txt",
        data_dir / split / "dialogues.json",
    ]

    for path in candidates:
        if not path.exists():
            continue

        suffix = path.suffix.lower()

        if suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            # handle both list-of-lists and list-of-dicts {"utterances": [...]}
            dialogues = []
            for item in data:
                if isinstance(item, list):
                    dialogues.append([str(u).strip() for u in item if str(u).strip()])
                elif isinstance(item, dict):
                    key = next((k for k in ("utterances", "turns", "dialog", "dialogue") if k in item), None)
                    if key:
                        dialogues.append([str(u).strip() for u in item[key] if str(u).strip()])
            return dialogues

        if suffix == ".jsonl":
            dialogues = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if isinstance(item, list):
                        dialogues.append([str(u).strip() for u in item if str(u).strip()])
                    elif isinstance(item, dict):
                        key = next((k for k in ("utterances", "turns", "dialog", "dialogue") if k in item), None)
                        if key:
                            dialogues.append([str(u).strip() for u in item[key] if str(u).strip()])
            return dialogues

        if suffix == ".txt":
            # DailyDialog style: utterances separated by " __eou__ " or newlines,
            # dialogues separated by blank lines or one dialogue per line.
            dialogues = []
            current: List[str] = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        if current:
                            dialogues.append(current)
                            current = []
                    elif " __eou__" in line:
                        # DailyDialog raw: each line is a full dialogue
                        parts = [p.strip() for p in line.split("__eou__") if p.strip()]
                        if parts:
                            dialogues.append(parts)
                    else:
                        current.append(line)
            if current:
                dialogues.append(current)
            return dialogues

    raise FileNotFoundError(
        f"Could not find a recognised data file for split='{split}' in {data_dir}.\n"
        f"Tried: {[str(c) for c in candidates]}"
    )


def build_windows(
    dialogues: List[List[str]],
    min_turns: int,
    max_history: int,
) -> List[Tuple[List[str], str]]:
    """
    Expand each dialogue into (history_list, target_utterance) pairs.
    Skips dialogues shorter than min_turns.
    """
    windows = []
    for dialogue in dialogues:
        if len(dialogue) < min_turns:
            continue
        for t in range(1, len(dialogue)):
            history = dialogue[max(0, t - max_history): t]
            target  = dialogue[t]
            windows.append((history, target))
    return windows


# ── Dataset ────────────────────────────────────────────────────────────────────

class DialogNextTurnDataset(Dataset):
    """
    Each item is a dict with pre-tokenized history turns and target utterance.
    Tokenisation is done at construction time (fast, cached-friendly).
    """

    def __init__(
        self,
        dialogues: List[List[str]],
        tokenizer: AutoTokenizer,
        max_seq_len: int,
        min_turns: int,
        max_history: int,
    ):
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.max_history = max_history
        self.windows     = build_windows(dialogues, min_turns, max_history)

    def __len__(self) -> int:
        return len(self.windows)

    def _encode(self, text: str) -> dict:
        return self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __getitem__(self, idx: int) -> dict:
        history_texts, target_text = self.windows[idx]

        # tokenise every history turn individually
        history_ids      = []
        history_masks    = []
        for utt in history_texts:
            enc = self._encode(utt)
            history_ids.append(enc["input_ids"].squeeze(0))       # (seq_len,)
            history_masks.append(enc["attention_mask"].squeeze(0))

        # pad history to max_history with zero tensors
        pad_ids  = torch.zeros(self.max_seq_len, dtype=torch.long)
        pad_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
        history_len = len(history_ids)
        while len(history_ids) < self.max_history:
            history_ids.append(pad_ids)
            history_masks.append(pad_mask)

        target_enc = self._encode(target_text)

        return {
            # (max_history, seq_len)
            "history_ids":    torch.stack(history_ids),
            "history_masks":  torch.stack(history_masks),
            "history_len":    torch.tensor(history_len, dtype=torch.long),
            # (seq_len,)
            "target_ids":     target_enc["input_ids"].squeeze(0),
            "target_mask":    target_enc["attention_mask"].squeeze(0),
        }


# ── Collate + DataLoader factory ───────────────────────────────────────────────

def collate_fn(batch: list) -> dict:
    """Default collate works fine since we pre-pad; this just stacks tensors."""
    keys = batch[0].keys()
    return {k: torch.stack([item[k] for item in batch]) for k in keys}


def make_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from cfg paths.
    The tokenizer is loaded once and shared.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.encoder_model,
        cache_dir=cfg.cache_dir,
    )

    loaders = {}
    for split in ("train", "val", "test"):
        try:
            dialogues = load_dialogues(cfg.raw_data_dir, split)
        except FileNotFoundError as e:
            # allow missing test split without crashing
            if split == "test":
                print(f"[data] test split not found, skipping. ({e})")
                loaders[split] = None
                continue
            raise

        dataset = DialogNextTurnDataset(
            dialogues   = dialogues,
            tokenizer   = tokenizer,
            max_seq_len = cfg.max_seq_len,
            min_turns   = cfg.min_turns,
            max_history = cfg.max_history,
        )
        print(f"[data] {split:>5} — {len(dialogues):>6} dialogues → "
              f"{len(dataset):>7} (history, target) windows")

        loaders[split] = DataLoader(
            dataset,
            batch_size  = cfg.batch_size,
            shuffle     = (split == "train"),
            num_workers = cfg.num_workers,
            collate_fn  = collate_fn,
            pin_memory  = True,
        )

    return loaders["train"], loaders["val"], loaders.get("test")