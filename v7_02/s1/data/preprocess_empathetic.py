"""
preprocess_empathetic.py
─────────────────────────
Downloads EmpatheticDialogues and converts to DailyDialog format.

WHY THE OLD VERSION BROKE
──────────────────────────
  load_dataset('facebook/empathetic_dialogues') fails with:
  "RuntimeError: Dataset scripts are no longer supported"
  because the HF repo still ships a legacy empathetic_dialogues.py loader.
  load_dataset tries to execute it → blocked.

THIS VERSION'S FIX
──────────────────
  We bypass load_dataset entirely.  The actual data files (train.csv,
  valid.csv, test.csv) are plain files in the HF repo — we download them
  with hf_hub_download() which never executes any Python scripts.
  Three fallback strategies in order:
    1. hf_hub_download  (works in Colab with no HF token needed for public data)
    2. Direct HTTP from HuggingFace CDN
    3. trust_remote_code=True load_dataset (datasets >= 2.20 sometimes allows this)

Output files (written to `output_dir`)
───────────────────────────────────────
  dialogues_train.txt / dialogues_valid.txt / dialogues_test.txt
      one dialog per line, turns joined by ' __eou__ '
  dialogues_train_emotions.json  {dialog_index: emotion_label}
  dialogues_valid_emotions.json
  dialogues_test_emotions.json

EmpatheticDialogues CSV column layout
──────────────────────────────────────
  conv_id, utterance_idx, context, prompt, speaker_idx, utterance, selfeval, tags
  • context  = emotion label (32 categories)
  • utterance = text, with '_comma_' replacing literal commas
  • utterance_idx = 1-based turn number within conversation
"""

from __future__ import annotations

import csv
import json
import os
import urllib.request
from collections import defaultdict
from io import StringIO
from pathlib import Path


# ── Download helpers ──────────────────────────────────────────────────────────

def _download_csv(output_dir: str, csv_filename: str) -> str:
    """
    Try three strategies to get one CSV file.  Returns the local path.
    Raises RuntimeError if all fail.
    """
    local = os.path.join(output_dir, csv_filename)
    if os.path.exists(local) and os.path.getsize(local) > 1000:
        return local  # already cached

    errors = []

    # ── Strategy 1: huggingface_hub (no loading script, just file download) ──
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id   = 'facebook/empathetic_dialogues',
            filename  = csv_filename,
            repo_type = 'dataset',
            local_dir = output_dir,
        )
        # hf_hub_download may put it in a sub-folder; copy to expected place
        if path != local:
            import shutil
            shutil.copy(path, local)
        return local
    except Exception as e:
        errors.append(f"hf_hub_download: {e}")

    # ── Strategy 2: direct HTTPS from HuggingFace CDN ────────────────────────
    cdn_urls = [
        f'https://huggingface.co/datasets/facebook/empathetic_dialogues/resolve/main/{csv_filename}',
        f'https://huggingface.co/datasets/facebook/empathetic_dialogues/resolve/refs%2Fconvert%2Fparquet/{csv_filename}',
    ]
    for url in cdn_urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'python'})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            with open(local, 'wb') as f:
                f.write(data)
            if os.path.getsize(local) > 1000:
                return local
        except Exception as e:
            errors.append(f"HTTP {url}: {e}")

    # ── Strategy 3: load_dataset with trust_remote_code (datasets >= 2.20) ───
    try:
        from datasets import load_dataset
        split = csv_filename.replace('.csv', '')   # 'train', 'valid', 'test'
        # 'valid' → 'validation' for HF splits
        hf_split = 'validation' if split == 'valid' else split
        ds = load_dataset(
            'facebook/empathetic_dialogues',
            split=hf_split,
            trust_remote_code=True,
        )
        # Convert to CSV and save
        import io as _io
        buf = _io.StringIO()
        writer = csv.writer(buf)
        # Write header
        writer.writerow(['conv_id', 'utterance_idx', 'context', 'prompt',
                         'speaker_idx', 'utterance', 'selfeval', 'tags'])
        for row in ds:
            writer.writerow([
                row.get('conv_id', ''),
                row.get('utterance_idx', ''),
                row.get('context', ''),
                row.get('prompt', ''),
                row.get('speaker_idx', ''),
                row.get('utterance', '').replace(',', '_comma_'),
                row.get('selfeval', ''),
                row.get('tags', ''),
            ])
        with open(local, 'w', encoding='utf-8') as f:
            f.write(buf.getvalue())
        return local
    except Exception as e:
        errors.append(f"load_dataset(trust_remote_code): {e}")

    raise RuntimeError(
        f"All download strategies failed for {csv_filename}.\n"
        + "\n".join(f"  • {e}" for e in errors)
        + "\n\nManual fix: download the file from\n"
        + "  https://huggingface.co/datasets/facebook/empathetic_dialogues/tree/main\n"
        + f"and place it at:  {local}"
    )


# ── CSV parsing ───────────────────────────────────────────────────────────────

def _parse_empathetic_csv(csv_path: str):
    """
    Parse one EmpatheticDialogues CSV file.

    Returns
    -------
    convs : dict[conv_id → list[(utterance_idx, text)]]
    emotions : dict[conv_id → emotion_label]
    """
    convs    = defaultdict(list)
    emotions = {}

    with open(csv_path, encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            # First non-empty row is the header
            if header is None:
                header = [h.strip() for h in row]
                continue
            if len(row) < 6:
                continue   # skip malformed rows
            try:
                # Handle both named and positional columns
                if 'conv_id' in header:
                    def col(name, fallback_idx):
                        try:
                            return row[header.index(name)].strip()
                        except (ValueError, IndexError):
                            return row[fallback_idx].strip() if fallback_idx < len(row) else ''
                else:
                    def col(name, fallback_idx):
                        return row[fallback_idx].strip() if fallback_idx < len(row) else ''

                conv_id  = col('conv_id',       0)
                utt_idx  = col('utterance_idx', 1)
                context  = col('context',       2)
                utterance= col('utterance',     5)

                if not conv_id or not utterance or utterance == 'utterance':
                    continue

                text = utterance.replace('_comma_', ',')
                convs[conv_id].append((int(utt_idx), text))
                if conv_id not in emotions:
                    emotions[conv_id] = context

            except Exception:
                continue   # skip unparseable rows

    return convs, emotions


# ── Main entry point ─────────────────────────────────────────────────────────

def preprocess_empathetic(output_dir: str, verbose: bool = True) -> dict:
    """
    Download + preprocess EmpatheticDialogues.
    Returns summary dict {split: {'n_dialogs', 'n_pairs', 'emotions'}}.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    split_map = [
        ('train',      'train.csv',  'dialogues_train'),
        ('validation', 'valid.csv',  'dialogues_valid'),
        ('test',       'test.csv',   'dialogues_test'),
    ]

    summary = {}

    for split_name, csv_file, stem in split_map:
        if verbose:
            print(f"\nProcessing split: {split_name} …")

        # 1. Acquire CSV
        csv_path = _download_csv(output_dir, csv_file)
        if verbose:
            size_kb = os.path.getsize(csv_path) // 1024
            print(f"  CSV: {csv_path}  ({size_kb} KB)")

        # 2. Parse
        convs, emotions = _parse_empathetic_csv(csv_path)

        # 3. Build flat dialog list
        dialog_lines   = []
        emotion_labels = {}

        for cid, turns in convs.items():
            turns.sort(key=lambda x: x[0])
            utterances = [t[1] for t in turns
                          if t[1] and t[1].lower() != 'nan']
            if len(utterances) < 2:
                continue
            line = ' __eou__ '.join(utterances) + ' __eou__'
            idx  = len(dialog_lines)
            dialog_lines.append(line)
            emotion_labels[idx] = emotions.get(cid, 'unknown')

        # 4. Write output files
        txt_path  = os.path.join(output_dir, f'{stem}.txt')
        json_path = os.path.join(output_dir, f'{stem}_emotions.json')

        with open(txt_path,  'w', encoding='utf-8') as f:
            f.write('\n'.join(dialog_lines))
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(emotion_labels, f)

        unique_emotions = set(emotion_labels.values())
        # Count CR pairs: each dialog with N turns has N-1 pairs
        n_cr = sum(
            len(line.split(' __eou__ ')) - 2   # trailing __eou__ → -2
            for line in dialog_lines
        )

        summary[split_name] = {
            'n_dialogs' : len(dialog_lines),
            'n_pairs'   : n_cr,
            'emotions'  : sorted(unique_emotions),
        }

        if verbose:
            print(f"  {len(dialog_lines):,} dialogs  |  "
                  f"~{n_cr:,} CR pairs  |  "
                  f"{len(unique_emotions)} emotion categories")
            print(f"  → {txt_path}")
            print(f"  → {json_path}")

    if verbose:
        print(f"\nDone. Files written to: {output_dir}")
        all_em = summary.get('train', {}).get('emotions', [])
        if all_em:
            print("Emotions:", all_em)

    return summary


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/empathetic_test'
    summary = preprocess_empathetic(out)
    for sp, info in summary.items():
        print(f"{sp}: {info['n_dialogs']} dialogs, {info['n_pairs']} pairs, "
              f"{len(info['emotions'])} emotions")