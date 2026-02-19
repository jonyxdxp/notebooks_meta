

# example data-loader:






import torch
from torch.utils.data import Dataset, DataLoader

class BPEDataset(Dataset):
    def __init__(self, tokenized_file):
        self.data = torch.load(tokenized_file)  # List of token ID tensors

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, pad_id=0):
    max_len = max(len(seq) for seq in batch)
    padded = torch.tensor([seq + [pad_id]*(max_len-len(seq)) for seq in batch])
    attention_mask = (padded != pad_id).long()
    return padded, attention_mask

def get_dataloader(tokenized_file, batch_size=8):
    dataset = BPEDataset(tokenized_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
