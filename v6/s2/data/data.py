<<<<<<< HEAD
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
    Load DailyDialog splits from the extracted ijcnlp_dailydialog structure.

    Expected layout (after unzipping train.zip / validation.zip / test.zip):
      <data_dir>/
        train/train/dialogues_train.txt
        validation/validation/dialogues_validation.txt
        test/test/dialogues_test.txt

    Each .txt file has one dialogue per line; utterances are separated by
    the token ' __eou__ ' (with trailing space after last utterance too).
    """
    data_dir = Path(data_dir)

    # Exact relative paths for the ijcnlp_dailydialog double-nested structure
    SPLIT_FILES = {
        "train": data_dir / "train"      / "train"      / "dialogues_train.txt",
        "val":   data_dir / "validation" / "validation" / "dialogues_validation.txt",
        "test":  data_dir / "test"       / "test"       / "dialogues_test.txt",
    }

    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split '{split}'. Expected one of: {list(SPLIT_FILES)}")

    path = SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find data file for split='{split}'.\n"
            f"Expected: {path}\n"
            f"Run the extraction cell first to unzip train.zip / validation.zip / test.zip."
        )

    dialogues = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split on __eou__ and drop empty strings (trailing token produces one)
            parts = [p.strip() for p in line.split("__eou__") if p.strip()]
            if parts:
                dialogues.append(parts)

    return dialogues


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
=======
# ── Cell 2: Dataset ───────────────────────────────────────────────────────────

from torch.utils.data import Dataset, DataLoader

class JEPADialogueDataset(Dataset):
    """
    Each sample: (context_embs, target_emb, n_context_turns)

    context_embs : (MAX_TURNS, 768) — padded sequence of turn embeddings
    target_emb   : (768,)           — DMI embedding of next turn
    n_turns      : int              — actual number of context turns
    """
    def __init__(self, dialog_file, max_turns=6, max_dialogs=None):
        self.samples  = []
        self.max_turns = max_turns
        dialogs = open(dialog_file).readlines()
        if max_dialogs:
            dialogs = dialogs[:max_dialogs]

        bert_med.eval()
        with torch.no_grad():
            for dialog in tqdm(dialogs,
                               desc=f"Building {dialog_file.split('/')[-1]}"):
                utts = [u.strip() for u in dialog.strip().split('__eou__')
                        if u.strip()]
                if len(utts) < 2: continue
                utts = utts[:max_turns + 1]

                # Encode all turns individually
                embs = [encode_single(u) for u in utts]  # list of (768,)

                # Build samples: use turns 0..i as context, turn i+1 as target
                for i in range(1, len(embs)):
                    context_embs = embs[:i]              # list of i tensors
                    target_emb   = embs[i]               # (768,)

                    # Pad context to max_turns
                    n = len(context_embs)
                    padded = torch.zeros(max_turns, 768)
                    padded[:n] = torch.stack(context_embs)

                    self.samples.append((padded, target_emb, n))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_jepa(batch):
    ctx_embs, tgt_embs, lengths = zip(*batch)
    ctx    = torch.stack(ctx_embs)                        # (B, MAX_TURNS, 768)
    tgt    = torch.stack(tgt_embs)                        # (B, 768)
    lens   = torch.tensor(lengths)                        # (B,)

    # Padding mask: True where turns are padding
    B, T, _ = ctx.shape
    mask = torch.arange(T).unsqueeze(0) >= lens.unsqueeze(1)  # (B, T)
    return ctx, tgt, mask, lens

print("Building JEPA datasets...")
train_ds     = JEPADialogueDataset(f'{base}/dialogues_train.txt')
valid_ds     = JEPADialogueDataset(f'{base}/dialogues_valid.txt')
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                          collate_fn=collate_jepa, num_workers=2)
valid_loader = DataLoader(valid_ds, batch_size=128, shuffle=False,
                          collate_fn=collate_jepa, num_workers=2)

print(f"Train: {len(train_ds):,} context-target pairs")
print(f"Valid: {len(valid_ds):,} context-target pairs")
>>>>>>> master
