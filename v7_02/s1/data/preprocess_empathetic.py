"""
preprocess_empathetic.py
------------------------
Downloads EmpatheticDialogues from HuggingFace using direct CSV download
(no `datasets` library needed — avoids the "Dataset scripts are no longer
supported" error in datasets >= 3.0).

Dataset CSV columns
~~~~~~~~~~~~~~~~~~~
conv_id, utterance_idx, context (= emotion), prompt, utterance,
speaker_idx, tags, selfeval, distractors

Output structure
~~~~~~~~~~~~~~~~
{out_dir}/
  dialogues_train.txt       — all training dialogs (__eou__ format)
  dialogues_valid.txt       — all validation dialogs
  by_emotion/
    {emotion}_train.txt     — one file per emotion  (32 × 2 files)
    {emotion}_valid.txt

Run once:
  from v7_02.s1.data.preprocess_empathetic import preprocess_empathetic_dialogues
  emotions = preprocess_empathetic_dialogues('/content/drive/MyDrive/empathetic')
"""

import os
import collections

# ── HuggingFace CSV URLs ───────────────────────────────────────────────
_HF_BASE = (
    "https://huggingface.co/datasets/empathetic_dialogues"
    "/resolve/main/{split}.csv"
)


def preprocess_empathetic_dialogues(out_dir: str) -> list:
    """
    Download, reconstruct, and save EmpatheticDialogues.

    Parameters
    ----------
    out_dir : str   directory where output files are written

    Returns
    -------
    emotions : list[str]   the 32 emotion labels found in the training set
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Run:  pip install -q pandas")

    os.makedirs(out_dir, exist_ok=True)
    by_emo_dir = os.path.join(out_dir, 'by_emotion')
    os.makedirs(by_emo_dir, exist_ok=True)

    all_emotions = []

    for hf_split, suffix in [('train', 'train'), ('valid', 'valid')]:
        url = _HF_BASE.format(split=hf_split)
        print(f"Downloading {hf_split} split from HuggingFace…")
        print(f"  {url}")

        df = pd.read_csv(url, on_bad_lines='skip')
        print(f"  {len(df):,} rows  |  columns: {list(df.columns)}")

        # ── reconstruct ordered dialog turns ──────────────────
        # Group by conv_id, sort by utterance_idx, collect utterances.
        # Each row's `context` column holds the emotion label for the
        # whole conversation.
        convs      = collections.defaultdict(list)   # conv_id → [(idx, text)]
        emo_of     = {}                               # conv_id → emotion label

        for _, row in df.iterrows():
            cid  = str(row['conv_id'])
            idx  = int(row['utterance_idx'])
            text = str(row['utterance']).strip().replace('\n', ' ')
            emo  = str(row['context']).strip()
            convs[cid].append((idx, text))
            emo_of[cid] = emo

        # ── build lines + group by emotion ────────────────────
        by_emotion = collections.defaultdict(list)
        all_lines  = []

        for cid, turns in convs.items():
            turns.sort(key=lambda x: x[0])
            utterances = [t for _, t in turns if t]
            if len(utterances) < 2:
                continue
            emotion = emo_of[cid]
            line    = ' __eou__ '.join(utterances)
            all_lines.append(line)
            by_emotion[emotion].append(line)

        # ── write combined file ────────────────────────────────
        combined = os.path.join(out_dir, f'dialogues_{suffix}.txt')
        with open(combined, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_lines))
        print(f"  → {len(all_lines):,} dialogs saved to {combined}")

        # ── write per-emotion files ────────────────────────────
        emotions = sorted(by_emotion.keys())
        for emotion in emotions:
            emo_path = os.path.join(by_emo_dir, f'{emotion}_{suffix}.txt')
            with open(emo_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(by_emotion[emotion]))

        if suffix == 'train':
            all_emotions = emotions
            print(f"\n  {len(emotions)} emotions found:")
            for e in emotions:
                print(f"    {e:30s}: {len(by_emotion[e]):4d} dialogs")

    print(f"\nAll done.  Data saved to: {out_dir}")
    return all_emotions