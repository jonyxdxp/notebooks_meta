
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from config  import cfg
from encoder import encode_single


class JEPADialogueDataset(Dataset):
    def __init__(self, dialog_file, max_dialogs=None):
        self.samples = []
        dialogs = open(dialog_file).readlines()
        if max_dialogs:
            dialogs = dialogs[:max_dialogs]
        with torch.no_grad():
            for dialog in tqdm(dialogs,
                               desc=f"Building {dialog_file.split('/')[-1]}"):
                utts = [u.strip() for u in
                        dialog.strip().split("__eou__") if u.strip()]
                if len(utts) < 2: continue
                utts = utts[:cfg.max_turns + 1]
                embs = [encode_single(u) for u in utts]
                for i in range(1, len(embs)):
                    n      = i
                    padded = torch.zeros(cfg.max_turns, 768)
                    padded[:n] = torch.stack(embs[:i])
                    self.samples.append((padded, embs[i], n))

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
    train_ds = JEPADialogueDataset(f"{cfg.base}/dialogues_train.txt")
    valid_ds = JEPADialogueDataset(f"{cfg.base}/dialogues_valid.txt")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True,  collate_fn=collate_jepa,
                              num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size,
                              shuffle=False, collate_fn=collate_jepa,
                              num_workers=2)
    print(f"Train: {len(train_ds):,} | Valid: {len(valid_ds):,}")
    return train_loader, valid_loader, valid_ds
