"""
preprocess_empathetic.py
------------------------
Downloads EmpatheticDialogues from HuggingFace and converts it to
DailyDialog format (one dialog per line, turns joined by ' __eou__ '),
grouped by emotion type so EmpatheticTaskStream can build domain-aware tasks.

Output structure
~~~~~~~~~~~~~~~~
{out_dir}/
  dialogues_train.txt       — all training dialogs (combined)
  dialogues_valid.txt       — all validation dialogs (combined)
  by_emotion/
    {emotion}_train.txt     — one file per emotion, 32 files × 2 splits
    {emotion}_valid.txt

Run once:
  from v7_02.s1.data.preprocess_empathetic import preprocess_empathetic_dialogues
  emotions = preprocess_empathetic_dialogues('/path/to/empathetic')
"""

import os
import collections


def preprocess_empathetic_dialogues(out_dir: str) -> list:
    """
    Download, reconstruct, and save EmpatheticDialogues.

    Returns
    -------
    emotions : list[str]   the 32 emotion labels found in the training set
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run:  pip install -q datasets")

    os.makedirs(out_dir, exist_ok=True)
    by_emo_dir = os.path.join(out_dir, 'by_emotion')
    os.makedirs(by_emo_dir, exist_ok=True)

    print("Downloading EmpatheticDialogues from HuggingFace…")
    dataset = load_dataset("empathetic_dialogues")

    all_emotions = []

    for hf_split, suffix in [('train', 'train'), ('validation', 'valid')]:
        print(f"\nProcessing '{hf_split}' split…")

        # ── group utterances by conversation id ───────────────
        convs      = collections.defaultdict(list)   # conv_id → [(idx, text)]
        emo_of     = {}                               # conv_id → emotion label

        for row in dataset[hf_split]:
            cid = row['conv_id']
            convs[cid].append((int(row['utterance_idx']), row['utterance']))
            emo_of[cid] = row['context']             # same emotion for all rows in conv

        # ── reconstruct ordered dialog turns ──────────────────
        by_emotion  = collections.defaultdict(list)
        all_lines   = []

        for cid, turns in convs.items():
            turns.sort(key=lambda x: x[0])
            utterances = [text.strip().replace('\n', ' ') for _, text in turns]
            if len(utterances) < 2:
                continue                              # need ≥ 2 turns for CR pairs
            emotion = emo_of[cid]
            line    = ' __eou__ '.join(utterances)
            all_lines.append(line)
            by_emotion[emotion].append(line)

        # ── write combined file ────────────────────────────────
        combined = os.path.join(out_dir, f'dialogues_{suffix}.txt')
        with open(combined, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_lines))
        print(f"  {len(all_lines):,} dialogs  →  {combined}")

        # ── write per-emotion files ────────────────────────────
        emotions = sorted(by_emotion.keys())
        for emotion in emotions:
            emo_path = os.path.join(by_emo_dir, f'{emotion}_{suffix}.txt')
            with open(emo_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(by_emotion[emotion]))

        if suffix == 'train':
            all_emotions = emotions
            print(f"  {len(emotions)} emotions:")
            for e in emotions:
                print(f"    {e:30s}: {len(by_emotion[e]):4d} dialogs")

    print(f"\nAll done.  Data saved to: {out_dir}")
    return all_emotions
