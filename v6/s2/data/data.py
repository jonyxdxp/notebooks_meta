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