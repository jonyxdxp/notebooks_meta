# ── Fast dataset using pre-cached embeddings ──────────────────────────────────

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class CachedUTTDataset(Dataset):
    def __init__(self, utterances, embeddings, max_tok=MAX_TOK_LEN):
        self.utterances  = utterances
        self.embeddings  = embeddings
        self.max_tok     = max_tok

    def __len__(self): return len(self.utterances)

    def __getitem__(self, idx):
        utt    = self.utterances[idx]
        emb    = self.embeddings[idx]
        tokens = tokenizer.encode(
            utt, add_special_tokens=False,
            max_length=self.max_tok - 2, truncation=True)
        tokens = torch.tensor(
            [tokenizer.cls_token_id] + tokens + [tokenizer.sep_token_id],
            dtype=torch.long)
        return emb, tokens

def collate_fn(batch):
    embs, toks = zip(*batch)
    return (torch.stack(embs),
            pad_sequence(toks, batch_first=True,
                         padding_value=tokenizer.pad_token_id))

train_ds     = CachedUTTDataset(train_utts, train_embs_dmi)
valid_ds     = CachedUTTDataset(valid_utts, valid_embs_dmi)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                          collate_fn=collate_fn, num_workers=2)
valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False,
                          collate_fn=collate_fn, num_workers=2)

print(f"Train batches: {len(train_loader)}")
print(f"Valid batches: {len(valid_loader)}")







# :::::::::::::::::::::::::::::






# ── Dataset + training (same as before, new encoder + tokenizer) ──────────────

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class SingleTurnDatasetNew(Dataset):
    def __init__(self, dialog_file, max_dialogs=None):
        self.samples = []
        dialogs = open(dialog_file).readlines()
        if max_dialogs:
            dialogs = dialogs[:max_dialogs]
        model_new_enc.eval()
        with torch.no_grad():
            for dialog in tqdm(dialogs,
                               desc=f"Building {dialog_file.split('/')[-1]}"):
                utts = [u.strip() for u in dialog.strip().split('__eou__')
                        if u.strip()]
                for utt in utts[:MAX_TURNS]:
                    emb    = encode_single_new(utt)
                    tokens = tokenizer_new.encode(
                        utt, add_special_tokens=True,
                        max_length=MAX_TOK_LEN, truncation=True)
                    self.samples.append((
                        emb,
                        torch.tensor(tokens, dtype=torch.long)
                    ))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_fn_new(batch):
    embs, token_seqs = zip(*batch)
    embs   = torch.stack(embs)
    padded = pad_sequence(token_seqs, batch_first=True,
                          padding_value=tokenizer_new.pad_token_id)
    return embs, padded

print("Building datasets with new encoder...")
train_ds_new = SingleTurnDatasetNew(f'{base}/dialogues_train.txt')
valid_ds_new = SingleTurnDatasetNew(f'{base}/dialogues_valid.txt')

train_loader_new = DataLoader(train_ds_new, batch_size=64, shuffle=True,
                               collate_fn=collate_fn_new, num_workers=2)
valid_loader_new = DataLoader(valid_ds_new, batch_size=64, shuffle=False,
                               collate_fn=collate_fn_new, num_workers=2)
print(f"Train: {len(train_ds_new):,} | Valid: {len(valid_ds_new):,}")