"""
dmi_moml_data.py
----------------
Data loading and online task-stream construction for MOML + DMI training.

Task definition
~~~~~~~~~~~~~~~
MOML processes tasks *sequentially* (online). We define each task as a
contiguous window of `pairs_per_task` context-response pairs drawn from the
training corpus.  Within each task the pairs are randomly split 50/50 into
a *support* set (inner-loop adaptation) and a *query* set (outer-loop
meta-update).

Why this works with DailyDialog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DailyDialog has ~11 k training dialogs × ~7 turns each ≈ 76 k CR pairs.
With pairs_per_task=240 (≈30 dialogs × 8 pairs) we get ~317 sequential
tasks per pass through the data — enough for MOML's omega/lambda variables
to converge.  Run multiple epochs to get the n_tasks you need.

Better datasets for meta-learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DailyDialog is fine for prototyping, but for stronger results consider:
  • MultiWOZ 2.1  — 7 domains (hotel/restaurant/taxi/train/…)  →  natural
                    task boundaries; domain ID can be used to define tasks.
  • PersonaChat   — different personas = different "tasks".
  • EmpatheticDialogues — 32 emotion categories.
Replace `data_path` in Config and swap in the appropriate loader below.
"""

import random
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm


# ─────────────────────────────────────────────────────────────
# Raw dialog → tokenised CR pairs
# ─────────────────────────────────────────────────────────────

class DialogCRDataset:
    """
    Reads a DailyDialog-format file (one dialog per line, turns separated by
    ' __eou__ ') and tokenises every context-response pair.

    Each item: (ctx_ids: LongTensor, rsp_ids: LongTensor)
    """
    EOU = '__eou__'

    def __init__(self, data_path: str, tokenizer,
                 max_ctx_len: int = 150, max_resp_len: int = 60):
        self.tokenizer   = tokenizer
        self.max_ctx_len = max_ctx_len
        self.max_rsp_len = max_resp_len
        self.pad_id      = tokenizer.pad_token_id
        self.cr_pairs    = []          # List[(ctx_tensor, rsp_tensor)]
        self._load(data_path)

    # ── loading ───────────────────────────────────────────────
    def _load(self, path: str):
        print(f"Loading {path}")
        raw_dialogs = []
        with open(path) as f:
            for line in f:
                turns = (line.strip()
                             .strip(self.EOU)
                             .split(f' {self.EOU} '))
                turns = [t.strip() for t in turns if t.strip()]
                if len(turns) >= 2:
                    raw_dialogs.append(turns)

        print(f"  {len(raw_dialogs)} dialogs.  Tokenising…")
        for dialog in tqdm(raw_dialogs, desc='Tokenising', leave=False):
            # mark each turn with __eou__
            turns = [f"{t} {self.EOU}" for t in dialog]
            for j in range(1, len(turns)):
                ctx_text = ' '.join(turns[:j])
                rsp_text = turns[j]
                ctx_ids  = self.tokenizer.encode(
                    ctx_text, add_special_tokens=True,
                    max_length=self.max_ctx_len, truncation=True)
                rsp_ids  = self.tokenizer.encode(
                    rsp_text, add_special_tokens=True,
                    max_length=self.max_rsp_len, truncation=True)
                self.cr_pairs.append((
                    torch.tensor(ctx_ids, dtype=torch.long),
                    torch.tensor(rsp_ids, dtype=torch.long),
                ))
        print(f"  → {len(self.cr_pairs):,} CR pairs.")

    # ── collation ─────────────────────────────────────────────
    def collate(self, pairs):
        """
        Pads a list of (ctx, rsp) tuples → batched tensors + masks.
        Returns: ctx, rsp, mask_ctx, mask_rsp
        mask: BoolTensor  True = padding position (ignored by Transformer)
        """
        ctx_list, rsp_list = zip(*pairs)
        ctx = pad_sequence(ctx_list, batch_first=True,
                           padding_value=self.pad_id)
        rsp = pad_sequence(rsp_list, batch_first=True,
                           padding_value=self.pad_id)
        mask_ctx = (ctx == self.pad_id)
        mask_rsp = (rsp == self.pad_id)
        return ctx, rsp, mask_ctx, mask_rsp


# ─────────────────────────────────────────────────────────────
# Online task stream
# ─────────────────────────────────────────────────────────────

class OnlineTaskStream:
    """
    Yields (support_pairs, query_pairs) sequentially, simulating an
    online stream of tasks as described in the MOML paper.

    Each task is a window of `pairs_per_task` CR pairs sampled from a
    random position in the corpus (with replacement across epochs, so
    n_tasks can exceed corpus size / pairs_per_task).

    The window is shuffled and split 50/50 into support and query.
    """

    def __init__(
        self,
        cr_pairs:       list,
        pairs_per_task: int,
        n_tasks:        int,
        seed:           int  = 42,
    ):
        self.cr_pairs       = cr_pairs
        self.pairs_per_task = pairs_per_task
        self.n_tasks        = n_tasks
        self.rng            = random.Random(seed)

    def __len__(self):
        return self.n_tasks

    def __iter__(self):
        n = len(self.cr_pairs)
        for _ in range(self.n_tasks):
            # Random contiguous window — "online" ordering
            start = self.rng.randint(0, max(0, n - self.pairs_per_task))
            chunk = list(self.cr_pairs[start: start + self.pairs_per_task])
            self.rng.shuffle(chunk)
            half     = max(4, len(chunk) // 2)  # ensure at least 4 pairs per split
            support  = chunk[:half]
            query    = chunk[half:]
            yield support, query

# ─────────────────────────────────────────────────────────────
# EmpatheticDialogues  —  emotion-aware dataset + task stream
# ─────────────────────────────────────────────────────────────

import os
import glob as _glob


class EmpatheticDialogDataset:
    """
    Loads EmpatheticDialogues from the per-emotion files produced by
    preprocess_empathetic_dialogues() and exposes:

      self.cr_pairs          — flat list of all (ctx, rsp) tensors
                               (used for validation, same API as DialogCRDataset)
      self.emotion_to_pairs  — {emotion_str: [(ctx, rsp), …]}
                               (fed into EmpatheticTaskStream for training)

    Parameters
    ----------
    data_dir    : directory that contains the by_emotion/ sub-folder
    split       : 'train' or 'valid'
    tokenizer   : HuggingFace tokenizer
    max_ctx_len : truncation length for context sequences
    max_resp_len: truncation length for response sequences
    """

    EOU = '__eou__'

    def __init__(self, data_dir: str, split: str, tokenizer,
                 max_ctx_len: int = 150, max_resp_len: int = 60):
        self.tokenizer   = tokenizer
        self.max_ctx_len = max_ctx_len
        self.max_rsp_len = max_resp_len
        self.pad_id      = tokenizer.pad_token_id

        self.cr_pairs         = []
        self.emotion_to_pairs = {}

        by_emo_dir = os.path.join(data_dir, 'by_emotion')
        if not os.path.isdir(by_emo_dir):
            raise FileNotFoundError(
                f"by_emotion/ not found under {data_dir}.\n"
                "Run preprocess_empathetic_dialogues() first."
            )

        emo_files = sorted(_glob.glob(os.path.join(by_emo_dir, f'*_{split}.txt')))
        if not emo_files:
            raise FileNotFoundError(
                f"No *_{split}.txt files in {by_emo_dir}.\n"
                "Run preprocess_empathetic_dialogues() first."
            )

        print(f"Loading EmpatheticDialogues ({split}): {len(emo_files)} emotions")
        for emo_file in tqdm(emo_files, desc='Emotions', leave=False):
            emotion = os.path.basename(emo_file).replace(f'_{split}.txt', '')
            pairs   = self._load_file(emo_file)
            if pairs:
                self.emotion_to_pairs[emotion] = pairs
                self.cr_pairs.extend(pairs)

        print(f"  {len(self.emotion_to_pairs)} emotions  "
              f"| {len(self.cr_pairs):,} total CR pairs")

    # ── internal ──────────────────────────────────────────────
    def _load_file(self, path: str):
        pairs = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                turns = line.strip().split(f' {self.EOU} ')
                turns = [t.strip() for t in turns if t.strip()]
                if len(turns) < 2:
                    continue
                turns = [f"{t} {self.EOU}" for t in turns]
                for j in range(1, len(turns)):
                    ctx_text = ' '.join(turns[:j])
                    rsp_text = turns[j]
                    ctx_ids  = self.tokenizer.encode(
                        ctx_text, add_special_tokens=True,
                        max_length=self.max_ctx_len, truncation=True)
                    rsp_ids  = self.tokenizer.encode(
                        rsp_text, add_special_tokens=True,
                        max_length=self.max_rsp_len, truncation=True)
                    pairs.append((
                        torch.tensor(ctx_ids, dtype=torch.long),
                        torch.tensor(rsp_ids, dtype=torch.long),
                    ))
        return pairs

    # ── collation (same API as DialogCRDataset) ───────────────
    def collate(self, pairs):
        ctx_list, rsp_list = zip(*pairs)
        ctx     = pad_sequence(ctx_list, batch_first=True,
                               padding_value=self.pad_id)
        rsp     = pad_sequence(rsp_list, batch_first=True,
                               padding_value=self.pad_id)
        mask_ctx = (ctx == self.pad_id)
        mask_rsp = (rsp == self.pad_id)
        return ctx, rsp, mask_ctx, mask_rsp


class EmpatheticTaskStream:
    """
    Domain-aware MOML task stream for EmpatheticDialogues.

    Each task = one randomly sampled emotion type.  The inner loop adapts
    the encoder to that emotion's conversational style; the outer loop
    evaluates on held-out pairs from the same emotion.  This gives MOML
    32 distinct task distributions instead of DailyDialog's 1.

    Parameters
    ----------
    emotion_to_pairs : dict  {emotion_str: [(ctx_tensor, rsp_tensor), …]}
                             — from EmpatheticDialogDataset.emotion_to_pairs
    n_tasks          : int   total number of tasks to yield
    pairs_per_task   : int   max CR pairs sampled per task (≤ emotion size)
    seed             : int
    min_pairs        : int   emotions with fewer pairs are skipped
    """

    def __init__(
        self,
        emotion_to_pairs: dict,
        n_tasks:          int,
        pairs_per_task:   int,
        seed:             int = 42,
        min_pairs:        int = 32,
    ):
        self.emotion_to_pairs = {
            e: p for e, p in emotion_to_pairs.items() if len(p) >= min_pairs
        }
        self.emotions       = sorted(self.emotion_to_pairs.keys())
        self.n_tasks        = n_tasks
        self.pairs_per_task = pairs_per_task
        self.rng            = random.Random(seed)

        skipped = len(emotion_to_pairs) - len(self.emotions)
        print(f"EmpatheticTaskStream ready: {len(self.emotions)} emotions "
              f"({skipped} skipped, < {min_pairs} pairs)  "
              f"| {sum(len(v) for v in self.emotion_to_pairs.values()):,} pairs total")

    def __len__(self):
        return self.n_tasks

    def __iter__(self):
        for _ in range(self.n_tasks):
            emotion = self.rng.choice(self.emotions)
            pairs   = self.emotion_to_pairs[emotion]

            # Sample up to pairs_per_task without replacement
            n     = min(len(pairs), self.pairs_per_task)
            chunk = self.rng.sample(pairs, n)

            half    = max(4, len(chunk) // 2)
            support = chunk[:half]
            query   = chunk[half:]
            yield support, query