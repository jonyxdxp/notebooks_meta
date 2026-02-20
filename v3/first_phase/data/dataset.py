

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












# ----------------------------------------------------













from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize (you'll need to implement or use a tokenizer)
        tokens = self.tokenizer.encode(text)
        
        # Pad/truncate
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'labels': torch.tensor(tokens, dtype=torch.long)  # For LM
        }

# Usage
def prepare_data():
    # Example texts
    texts = ["Hello world", "Mamba models are efficient", ...]
    
    # Simple tokenizer (replace with your actual tokenizer)
    class SimpleTokenizer:
        def __init__(self, vocab):
            self.vocab = vocab
            self.vocab_size = len(vocab)
        
        def encode(self, text):
            return [self.vocab.get(word, 0) for word in text.split()]
    
    tokenizer = SimpleTokenizer(your_vocabulary)
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return dataloader








# ---------------------------------------------







import torch
from torch.utils.data import Dataset
import numpy as np



class JEPAEmbeddingDataset(Dataset):
    """JEPA-style dataset with context and target embeddings."""
    def __init__(self, embedding_files, context_len=4, target_len=1):
        self.context_len = context_len
        self.target_len = target_len
        self.data = []
        for file in embedding_files:
            embeddings = np.load(file)
            for i in range(len(embeddings) - context_len - target_len):
                self.data.append((embeddings[i:i+context_len], embeddings[i+context_len:i+context_len+target_len]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)