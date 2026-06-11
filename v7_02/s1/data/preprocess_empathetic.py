"""
preprocess_empathetic.py
─────────────────────────
Downloads EmpatheticDialogues from HuggingFace and converts it to the
DailyDialog format expected by EmotionAwareDataset.

Output files (written to `output_dir`)
───────────────────────────────────────
  dialogues_train.txt         one dialog per line, turns joined by __eou__
  dialogues_valid.txt
  dialogues_test.txt
  dialogues_train_emotions.json   {dialog_index: emotion_label}
  dialogues_valid_emotions.json
  dialogues_test_emotions.json

Usage (in Colab)
────────────────
  from v7_02.s1.data.preprocess_empathetic import preprocess_empathetic
  preprocess_empathetic('/content/empathetic_dialogues')

Dataset facts
─────────────
  • ~24 850 conversations, ~108 k utterances
  • 32 emotion categories: admiration, amusement, anger, annoyance,
    anticipation, approval, caring, confusion, curiosity, desire,
    disappointment, disapproval, disgust, embarrassment, excitement,
    fear, gratitude, grief, joy, love, nervousness, optimism, pride,
    realisation, relief, remorse, sadness, surprise, trust …
  • Average ~4.4 turns/dialog  →  ~3.4 CR pairs/dialog
  • Total CR pairs: ~84 k train / ~6 k valid / ~6 k test

Why it beats DailyDialog for MOML
──────────────────────────────────
  32 clearly distinct task distributions (emotion types) give MOML's
  inner loop a genuine signal to adapt to.  The encoder must learn
  representations that transfer quickly from, say, "admiration" dialogs
  to "grief" dialogs — exactly the meta-learning problem.
"""

from __future__ import annotations
import json
import os
from collections import defaultdict
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────

def preprocess_empathetic(output_dir: str, verbose: bool = True) -> dict:
    """
    Download EmpatheticDialogues and write to output_dir in DailyDialog
    format + companion emotion JSON files.

    Returns a summary dict: {split: {'n_dialogs', 'n_pairs', 'emotions'}}.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Run:  !pip install -q datasets\n"
            "then retry preprocess_empathetic()."
        )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary = {}

    split_map = {
        'train':      'dialogues_train',
        'validation': 'dialogues_valid',
        'test':       'dialogues_test',
    }

    for hf_split, file_stem in split_map.items():
        if verbose:
            print(f"\nProcessing split: {hf_split} …")

        ds = load_dataset('empathetic_dialogues', split=hf_split,
                          trust_remote_code=True)

        # ── Group rows by conversation ─────────────────────────────────
        # Each row: conv_id, utterance_idx, context (emotion), utterance
        convs    = defaultdict(list)   # conv_id → [(idx, text)]
        emotions = {}                  # conv_id → emotion label

        for row in ds:
            cid = row['conv_id']
            # EmpatheticDialogues uses '_comma_' as a comma escape
            text = row['utterance'].replace('_comma_', ',').strip()
            convs[cid].append((row['utterance_idx'], text))
            # 'context' is the emotion label; same for all rows of a conv
            if cid not in emotions:
                emotions[cid] = row['context'].strip()

        # ── Sort turns and build flat dialog list ──────────────────────
        dialog_lines   = []
        emotion_labels = {}           # dialog_index → emotion

        for cid, turns in convs.items():
            turns.sort(key=lambda x: x[0])
            utterances = [t[1] for t in turns if t[1]]
            if len(utterances) < 2:
                continue
            line  = ' __eou__ '.join(utterances) + ' __eou__'
            idx   = len(dialog_lines)
            dialog_lines.append(line)
            emotion_labels[idx] = emotions[cid]

        # ── Write dialog file ──────────────────────────────────────────
        txt_path = os.path.join(output_dir, f'{file_stem}.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(dialog_lines))

        # ── Write emotion JSON ─────────────────────────────────────────
        json_path = os.path.join(output_dir, f'{file_stem}_emotions.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(emotion_labels, f)

        unique_emotions = set(emotion_labels.values())
        # Rough CR pair count: sum(turns-1) per dialog
        n_cr = sum(
            len(line.split(' __eou__ ')) - 2   # turns minus trailing eou
            for line in dialog_lines
        )

        summary[hf_split] = {
            'n_dialogs': len(dialog_lines),
            'n_pairs':   n_cr,
            'emotions':  sorted(unique_emotions),
        }

        if verbose:
            print(f"  {len(dialog_lines):,} dialogs  |  "
                  f"~{n_cr:,} CR pairs  |  "
                  f"{len(unique_emotions)} emotion categories")
            print(f"  → {txt_path}")
            print(f"  → {json_path}")

    if verbose:
        print(f"\nDone. Files written to: {output_dir}")
        print("Emotion categories:", sorted(summary['train']['emotions']))

    return summary


# ─────────────────────────────────────────────────────────────────────
# Quick smoke-test (run directly: python preprocess_empathetic.py)
# ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/empathetic_test'
    summary = preprocess_empathetic(out)
    for split, info in summary.items():
        print(f"{split}: {info['n_dialogs']} dialogs, "
              f"{info['n_pairs']} pairs")
