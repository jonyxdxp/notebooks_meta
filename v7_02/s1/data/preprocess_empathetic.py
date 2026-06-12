"""
preprocess_empathetic.py
------------------------
Converts the EmpatheticDialogues tar.gz to DailyDialog format
(one dialog per line, turns joined by ' __eou__ '), grouped by emotion.

Usage
~~~~~
  from v7_02.s1.data.preprocess_empathetic import preprocess_empathetic_dialogues

  # Option A — local file (already downloaded)
  emotions = preprocess_empathetic_dialogues(
      out_dir   = '/content/drive/MyDrive/empathetic_dialogues',
      local_tar = '/content/drive/MyDrive/data/empatheticdialogues.tar.gz',
  )

  # Option B — auto-download via wget
  emotions = preprocess_empathetic_dialogues(
      out_dir = '/content/drive/MyDrive/empathetic_dialogues',
  )

Output structure
~~~~~~~~~~~~~~~~
{out_dir}/
  dialogues_train.txt       — all training dialogs (__eou__ format)
  dialogues_valid.txt       — all validation dialogs
  by_emotion/
    {emotion}_train.txt     — one file per emotion  (32 × 2 files)
    {emotion}_valid.txt
"""

import os, io, collections, tarfile, subprocess

_CDN_URL  = (
    "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues"
    "/empatheticdialogues.tar.gz"
)
_TMP_PATH = "/tmp/empatheticdialogues.tar.gz"


def preprocess_empathetic_dialogues(
    out_dir:   str,
    local_tar: str = None,   # pass your downloaded file path here
) -> list:
    """
    Parameters
    ----------
    out_dir   : directory where processed files are written
    local_tar : path to an already-downloaded .tar.gz  (optional).
                If None, the file is downloaded automatically via wget.

    Returns
    -------
    emotions : list[str]  the 32 emotion labels in the training split
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Run:  pip install -q pandas")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'by_emotion'), exist_ok=True)

    # ── Resolve tar path ──────────────────────────────────────────────
    if local_tar:
        if not os.path.exists(local_tar):
            raise FileNotFoundError(f"local_tar not found: {local_tar}")
        tar_path = local_tar
        print(f"Using local file: {tar_path}  "
              f"({os.path.getsize(tar_path)/1e6:.1f} MB)")
    else:
        tar_path = _TMP_PATH
        print(f"Downloading from Facebook CDN…\n  {_CDN_URL}")
        result = subprocess.run(
            ["wget", "-q", "--show-progress", "-O", tar_path, _CDN_URL]
        )
        if result.returncode != 0 or not os.path.exists(tar_path):
            raise RuntimeError(
                "wget failed.  Download manually:\n"
                f"  {_CDN_URL}\n"
                f"Then pass: local_tar='<your path>'"
            )
        print(f"  Downloaded {os.path.getsize(tar_path)/1e6:.1f} MB")

    # ── Parse tar ─────────────────────────────────────────────────────
    all_emotions = []

    with tarfile.open(tar_path, mode='r:gz') as tar:
        for hf_split, suffix in [('train', 'train'), ('valid', 'valid')]:
            inner = f'empatheticdialogues/{hf_split}.csv'
            print(f"\nProcessing {inner}…")

            csv_bytes = tar.extractfile(tar.getmember(inner)).read()
            df = pd.read_csv(io.BytesIO(csv_bytes), on_bad_lines='skip')
            print(f"  {len(df):,} rows")

            convs, emo_of = collections.defaultdict(list), {}
            for _, row in df.iterrows():
                cid = str(row['conv_id'])
                convs[cid].append((int(row['utterance_idx']),
                                   str(row['utterance']).strip().replace('\n', ' ')))
                emo_of[cid] = str(row['context']).strip()

            by_emotion, all_lines = collections.defaultdict(list), []
            for cid, turns in convs.items():
                turns.sort(key=lambda x: x[0])
                utts = [t for _, t in turns if t and t != 'nan']
                if len(utts) < 2:
                    continue
                line = ' __eou__ '.join(utts)
                all_lines.append(line)
                by_emotion[emo_of[cid]].append(line)

            # write combined
            with open(os.path.join(out_dir, f'dialogues_{suffix}.txt'),
                      'w', encoding='utf-8') as f:
                f.write('\n'.join(all_lines))
            print(f"  → {len(all_lines):,} dialogs")

            # write per-emotion
            emotions = sorted(by_emotion)
            for emo in emotions:
                with open(os.path.join(out_dir, 'by_emotion',
                                       f'{emo}_{suffix}.txt'),
                          'w', encoding='utf-8') as f:
                    f.write('\n'.join(by_emotion[emo]))

            if suffix == 'train':
                all_emotions = emotions
                print(f"  {len(emotions)} emotions:")
                for e in emotions:
                    print(f"    {e:30s}: {len(by_emotion[e]):4d} dialogs")

    print(f"\nDone → {out_dir}")
    return all_emotions