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

import json
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from collections import defaultdict


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
# EmpatheticDialogues support
# ─────────────────────────────────────────────────────────────

class EmotionAwareDataset(DialogCRDataset):
    """
    Extends DialogCRDataset to tag each CR pair with an emotion label.

    Requires a companion JSON file written by preprocess_empathetic.py:
        {dialog_index (int): emotion_label (str), …}

    Extra attribute
    ───────────────
    pair_emotions : list[str] — parallel to self.cr_pairs

    Convenience method
    ──────────────────
    pairs_by_emotion() → dict[str, list[(ctx, rsp)]]
        Used by EmotionTaskStream to build per-emotion task buckets.
    """
    EOU = '__eou__'

    def __init__(self, data_path: str, emotion_json: str, tokenizer,
                 max_ctx_len: int = 150, max_resp_len: int = 60):
        # Must be set before super().__init__ calls _load
        self._emotion_json = emotion_json
        self.pair_emotions: list = []
        super().__init__(data_path, tokenizer, max_ctx_len, max_resp_len)

    def _load(self, path: str):
        # Load emotion map: {dialog_idx_str → emotion_label}
        with open(self._emotion_json) as f:
            raw = json.load(f)
        emotion_map = {int(k): v for k, v in raw.items()}

        print(f"Loading {path}")
        raw_dialogs, dialog_emotions = [], []
        with open(path) as f:
            for line in f:
                turns = (line.strip()
                             .strip(self.EOU)
                             .split(f' {self.EOU} '))
                turns = [t.strip() for t in turns if t.strip()]
                if len(turns) >= 2:
                    idx = len(raw_dialogs)
                    raw_dialogs.append(turns)
                    dialog_emotions.append(emotion_map.get(idx, 'unknown'))

        n_emotions = len(set(dialog_emotions))
        print(f"  {len(raw_dialogs)} dialogs across {n_emotions} emotions.  "
              f"Tokenising…")

        for dialog, emotion in tqdm(
                zip(raw_dialogs, dialog_emotions),
                total=len(raw_dialogs), desc='Tokenising', leave=False):
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
                self.pair_emotions.append(emotion)

        print(f"  → {len(self.cr_pairs):,} CR pairs  "
              f"({n_emotions} emotion categories).")

    def pairs_by_emotion(self) -> dict:
        """Returns {emotion: [(ctx, rsp), …]} — used by EmotionTaskStream."""
        buckets = defaultdict(list)
        for pair, em in zip(self.cr_pairs, self.pair_emotions):
            buckets[em].append(pair)
        return dict(buckets)


class EmotionTaskStream:
    """
    Yields (support_pairs, query_pairs) where every task = one emotion.

    Why this is better than OnlineTaskStream on homogeneous data
    ─────────────────────────────────────────────────────────────
    MOML's value comes from fast adaptation across *diverse* tasks.
    With DailyDialog all tasks look identical (chitchat), so ω barely moves.
    EmpatheticDialogues has 32 emotion categories with meaningfully different
    conversational patterns — the inner loop now has a real distribution shift
    to adapt to, and ω learns a genuinely useful meta-initialization.

    Usage
    ─────
    ds    = EmotionAwareDataset(train_path, emotion_json, tokenizer)
    stream = EmotionTaskStream(ds.pairs_by_emotion(), n_tasks=2000)
    # then pass stream= to train_moml_dmi
    """

    def __init__(
        self,
        pairs_by_emotion:  dict,          # {emotion: [(ctx, rsp), …]}
        n_tasks:           int,
        pairs_per_task:    int  = 200,    # cap per task for memory
        min_pairs:         int  = 16,     # skip emotions with too few pairs
        seed:              int  = 42,
    ):
        self.by_emotion    = {e: p for e, p in pairs_by_emotion.items()
                              if len(p) >= min_pairs}
        self.emotions      = sorted(self.by_emotion.keys())
        self.n_tasks       = n_tasks
        self.pairs_per_task = pairs_per_task
        self.rng           = random.Random(seed)

        if not self.emotions:
            raise ValueError("No emotion bucket has enough pairs. "
                             "Lower min_pairs or check your data.")
        counts = [len(self.by_emotion[e]) for e in self.emotions]
        print(f"EmotionTaskStream ready: {len(self.emotions)} emotions | "
              f"pairs/emotion: {min(counts)}–{max(counts)}")

    def __len__(self):
        return self.n_tasks

    def __iter__(self):
        for _ in range(self.n_tasks):
            emotion = self.rng.choice(self.emotions)
            pool    = list(self.by_emotion[emotion])
            self.rng.shuffle(pool)
            chunk   = pool[:self.pairs_per_task]
            if len(chunk) < 8:
                continue
            half    = max(4, len(chunk) // 2)
            yield chunk[:half], chunk[half:]