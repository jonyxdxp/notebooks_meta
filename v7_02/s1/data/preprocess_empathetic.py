"""
preprocess_empathetic.py
------------------------
Downloads EmpatheticDialogues from Facebook's public CDN and converts it
to DailyDialog format (one dialog per line, turns joined by ' __eou__ '),
grouped by emotion type so EmpatheticTaskStream can build domain-aware tasks.

Source
~~~~~~
https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz
  └── empatheticdialogues/
        train.csv   valid.csv   test.csv

CSV columns: conv_id, utterance_idx, context (= emotion label), prompt,
             utterance, speaker_idx, tags, selfeval, distractors

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
import io
import collections
import tarfile
import urllib.request

_CDN_URL = (
    "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues"
    "/empatheticdialogues.tar.gz"
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

    # ── Download tar.gz into memory ───────────────────────────────────
    print(f"Downloading EmpatheticDialogues from Facebook CDN…")
    print(f"  {_CDN_URL}")
    req = urllib.request.Request(
        _CDN_URL,
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req) as response:
        raw = response.read()
    print(f"  Downloaded {len(raw) / 1e6:.1f} MB")

    # ── Open tar in memory, read CSVs ─────────────────────────────────
    splits_map = {'train': 'train', 'valid': 'valid'}
    all_emotions = []

    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as tar:
        for hf_split, suffix in splits_map.items():
            # File inside the archive
            inner_path = f'empatheticdialogues/{hf_split}.csv'
            print(f"\nProcessing {inner_path}…")

            member = tar.getmember(inner_path)
            f = tar.extractfile(member)

            import pandas as pd
            df = pd.read_csv(f, on_bad_lines='skip')
            print(f"  {len(df):,} rows  |  columns: {list(df.columns)}")

            # ── reconstruct dialogs ────────────────────────────────────
            convs  = collections.defaultdict(list)
            emo_of = {}

            for _, row in df.iterrows():
                cid  = str(row['conv_id'])
                idx  = int(row['utterance_idx'])
                text = str(row['utterance']).strip().replace('\n', ' ')
                emo  = str(row['context']).strip()
                convs[cid].append((idx, text))
                emo_of[cid] = emo

            by_emotion = collections.defaultdict(list)
            all_lines  = []

            for cid, turns in convs.items():
                turns.sort(key=lambda x: x[0])
                utterances = [t for _, t in turns if t and t != 'nan']
                if len(utterances) < 2:
                    continue
                emotion = emo_of[cid]
                line    = ' __eou__ '.join(utterances)
                all_lines.append(line)
                by_emotion[emotion].append(line)

            # ── write combined file ────────────────────────────────────
            combined = os.path.join(out_dir, f'dialogues_{suffix}.txt')
            with open(combined, 'w', encoding='utf-8') as out:
                out.write('\n'.join(all_lines))
            print(f"  → {len(all_lines):,} dialogs saved to {combined}")

            # ── write per-emotion files ────────────────────────────────
            emotions = sorted(by_emotion.keys())
            for emotion in emotions:
                emo_path = os.path.join(by_emo_dir, f'{emotion}_{suffix}.txt')
                with open(emo_path, 'w', encoding='utf-8') as out:
                    out.write('\n'.join(by_emotion[emotion]))

            if suffix == 'train':
                all_emotions = emotions
                print(f"\n  {len(emotions)} emotions:")
                for e in emotions:
                    print(f"    {e:30s}: {len(by_emotion[e]):4d} dialogs")

    print(f"\nAll done.  Data saved to: {out_dir}")
    return all_emotions