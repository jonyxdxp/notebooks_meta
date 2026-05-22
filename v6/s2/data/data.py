
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from v6.s2.config import (BASE, MAX_TURNS, DEVICE, BATCH_SIZE)
from v6.s2.encoder import encode_single   # encoder must also be a module

class JEPADialogueDataset(Dataset):
    """
    Each sample: (context_embs, target_emb, n_context_turns)
    context_embs : (MAX_TURNS, 768) padded
    target_emb   : (768,)
    n_turns      : int
    """
    def __init__(self, dialog_file, max_dialogs=None):
        self.samples   = []
        self.max_turns = MAX_TURNS
        dialogs = open(dialog_file).readlines()
        if max_dialogs:
            dialogs = dialogs[:max_dialogs]

        with torch.no_grad():
            for dialog in tqdm(dialogs,
                               desc=f"Building {dialog_file.split('/')[-1]}"):
                utts = [u.strip() for u in
                        dialog.strip().split('__eou__') if u.strip()]
                if len(utts) < 2: continue
                utts = utts[:MAX_TURNS + 1]
                embs = [encode_single(u) for u in utts]

                for i in range(1, len(embs)):
                    context_embs = embs[:i]
                    target_emb   = embs[i]
                    n            = len(context_embs)
                    padded       = torch.zeros(MAX_TURNS, 768)
                    padded[:n]   = torch.stack(context_embs)
                    self.samples.append((padded, target_emb, n))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


def collate_jepa(batch):
    ctx_embs, tgt_embs, lengths = zip(*batch)
    ctx  = torch.stack(ctx_embs)
    tgt  = torch.stack(tgt_embs)
    lens = torch.tensor(lengths)
    B, T, _ = ctx.shape
    mask = torch.arange(T).unsqueeze(0) >= lens.unsqueeze(1)
    return ctx, tgt, mask, lens


def make_dataloaders():
    train_ds = JEPADialogueDataset(f'{BASE}/dialogues_train.txt')
    valid_ds = JEPADialogueDataset(f'{BASE}/dialogues_valid.txt')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  collate_fn=collate_jepa,
                              num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_jepa,
                              num_workers=2)
    print(f"Train: {len(train_ds):,} | Valid: {len(valid_ds):,}")
    return train_loader, valid_loader
