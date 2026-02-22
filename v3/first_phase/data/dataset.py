

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













        # ----------------------------------------------------------










    """Dialogue dataset."""

import json
import datasets

logger = datasets.logging.get_logger(__name__)


class Dataset(datasets.GeneratorBasedBuilder):
    """Dialogue dataset."""
    def _info(self):
        features = datasets.Features({"src": datasets.Value("string"), "tgt": datasets.Value("string")})
        return datasets.DatasetInfo(description="Dialogue dataset.", features=features)

    def _split_generators(self, dl_manager):
        data_files = self.config.data_files

        splits = []
        if 'train' in data_files:
            splits.append(datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_files["train"]}))
        if 'validation' in data_files:
            splits.append(datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": data_files["validation"]}))
        if 'test' in data_files:
            splits.append(datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": data_files["test"]}))

        return splits

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                yield id_, {"src": data["src"], "tgt": data["tgt"]}




















# --------------------------------------------------








# from https://github.com/alexiglad/EBT/blob/main/data/nlp/fineweb_dataloader.py




from transformers import AutoTokenizer
from datasets import load_dataset, get_dataset_config_names, load_from_disk
import torch
from torch.utils.data import Dataset
from functools import partial
from datasets import Dataset as hf_Dataset
import os
import json

class FineWebDataset(Dataset):
    def __init__(self, hparams): # dont use tokenizer is in collator
        if hparams.execution_mode != "pretrain":
            raise ValueError("FineWeb is a pretrain dataset, no other execution modes supported.")
            
        #NOTE there is only 1 split (train) so every other split does the same here
        self.max_length = hparams.context_length+1
        hf_home = os.getenv('HF_HOME')
        dataset_dir = hparams.dataset_dir if hparams.dataset_dir != "" else hf_home
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer, clean_up_tokenization_spaces = False)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id # just for reference the tokenizer is fast

        if hparams.pretokenize_dataset:
            save_path = os.path.join(dataset_dir, hparams.dataset_name + '_preprocessed', hparams.tokenizer.replace('/', '_'), "max_length_" + str(self.max_length))
            print("pretokenized dataset save_path", save_path)

            if os.path.exists(save_path): # load dataset it exists
                print(f"loading {hparams.dataset_name} dataset")
                self.dataset = load_from_disk(save_path)
            else: # need to create dataset
                print(f"no pre-tokenized {hparams.dataset_name} dataset with correct settings, loading and saving")
                self.dataset = load_dataset("HuggingFaceFW/fineweb", "sample-100BT", split = "train", cache_dir=dataset_dir, trust_remote_code=True, keep_in_memory = False)

                num_proc = hparams.num_workers * hparams.num_gpus
                print("num_proc using for dataset map", num_proc) # found that if have 192 cpus then cannot use 96 (it freezes), so 48 was good. make sure to test this with your own hardware and adjust num workers accordingly
                # NOTE this code may freeze and takes a very long time to run, make sure to test what values for num_proc and num_workers are best
                self.dataset = self.dataset.map(self.tokenization, num_proc = num_proc) # batched=True, batch_size=hparams.batch_size_per_device,
                print("done preprocessing dataset")
                self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
                print("done formatting dataset")
                self.dataset.save_to_disk(save_path)
        else:
            self.dataset = load_dataset("HuggingFaceFW/fineweb", "sample-100BT", split = "train", cache_dir=dataset_dir, trust_remote_code=True, keep_in_memory = False)

        self.hparams = hparams

    def tokenization(self, example):
        return self.tokenizer(example['text'], padding=True, truncation=True, max_length=self.max_length)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.hparams.pretokenize_dataset:
            return self.dataset[idx]
        else:
            return self.dataset[idx]['text']














# -----------------------------------------------------




# modified from https://github.com/jerber/lang-jepa/blob/main/src/common/datasets/fineweb_edu.py







from dataclasses import dataclass

from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from src.common.datasets.utils.sentence_splitting import (
    SentenceSplitter,
    SentenceSplitterConfig,
)


@dataclass
class Sentence:
    text: str
    start_idx: int
    end_idx: int


@dataclass
class DatasetOutput:
    context: str
    target: str


class InteractionDataset(Dataset):
    def __init__(
        self,
        *,
        train_file: str,
        limit: int | None,
        min_length: int,
        window_size: int = 25,
        min_sentences: int = 2,
        tokenizer: PreTrainedTokenizer | None = None,
        max_tokens: int | None = None,
        cache_dir: str = "~/.cache/huggingface/datasets",
    ):
        """Enhanced dataset wrapper with precise sentence boundary handling.

        Args:
            train_file: Which dataset file to load
            limit: Number of documents to process
            min_length: Minimum text length to consider
            window_size: Number of sentences to use as context (default: 25)
            min_sentences: Minimum sentences required (default: 2)
            tokenizer: Optional tokenizer for length checking
            max_tokens: Optional maximum tokens per context window
            cache_dir: HuggingFace cache directory
        """
        self.samples: list[DatasetOutput] = []
        self.stats = {
            "total_docs": 0,
            "docs_processed": 0,
            "docs_rejected_length": 0,
            "docs_rejected_sentences": 0,
            "context_target_pairs": 0,
            "pairs_rejected_length": 0,
        }

        # Load dataset
        print(f"Loading dataset with {window_size}-sentence sliding window...")
        ds = load_dataset(
            path="HuggingFaceFW/fineweb-edu",
            name=train_file,
            split="train",
            streaming=True,
            cache_dir=cache_dir,
        )

        # Initialize sentence splitter
        splitter = SentenceSplitter(SentenceSplitterConfig())

        # Process documents
        pbar = tqdm(total=limit, desc="Processing documents", unit="docs")

        for doc in ds:
            self.stats["total_docs"] += 1
            text = doc.get("text", "").strip()

            # Check minimum length
            if len(text) < min_length:
                self.stats["docs_rejected_length"] += 1
                continue

            try:
                # Split into sentences
                sentences = splitter([text])[0]
                if len(sentences) < min_sentences:
                    self.stats["docs_rejected_sentences"] += 1
                    continue

                # Find sentence boundaries in original text
                sentence_objs: list[Sentence] = []
                search_start = 0

                for sent in sentences:
                    # Find the sentence in the original text
                    start_idx = text.index(sent, search_start)
                    end_idx = start_idx + len(sent)

                    sentence_objs.append(
                        Sentence(text=sent, start_idx=start_idx, end_idx=end_idx)
                    )
                    search_start = end_idx

                # Create context-target pairs with sliding window
                for i in range(1, len(sentence_objs)):
                    # Get previous sentences as context (up to window_size)
                    start_sent_idx = max(0, i - window_size)

                    # Get exact text slice from original document
                    context_start = sentence_objs[start_sent_idx].start_idx
                    context_end = sentence_objs[i - 1].end_idx
                    context = text[context_start:context_end]

                    # Get target sentence with exact boundaries
                    target = text[sentence_objs[i].start_idx : sentence_objs[i].end_idx]

                    # Check token length if tokenizer provided
                    if tokenizer and max_tokens:
                        context_tokens = len(tokenizer.encode(context))
                        if context_tokens > max_tokens:
                            self.stats["pairs_rejected_length"] += 1
                            continue

                    self.samples.append(
                        DatasetOutput(
                            context=context,
                            target=target,
                        )
                    )
                    self.stats["context_target_pairs"] += 1

                self.stats["docs_processed"] += 1
                pbar.update(1)

                if limit and self.stats["docs_processed"] >= limit:
                    break

            except Exception as e:
                print(f"Error processing document: {e}")
                continue

        pbar.close()

        # Print statistics
        print("\nDataset Processing Statistics:")
        print(f"Total documents seen: {self.stats['total_docs']:,}")
        print(f"Documents processed: {self.stats['docs_processed']:,}")
        print(f"Documents rejected (length): {self.stats['docs_rejected_length']:,}")
        print(
            f"Documents rejected (sentences): {self.stats['docs_rejected_sentences']:,}"
        )
        print(f"Context-target pairs generated: {self.stats['context_target_pairs']:,}")
        print(f"Pairs rejected (length): {self.stats['pairs_rejected_length']:,}")

        if not self.samples:
            raise RuntimeError(
                f"No valid samples found in dataset ({train_file}). "
                f"Try adjusting the minimum length ({min_length}) or "
                f"minimum sentences ({min_sentences}) requirements."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DatasetOutput:
        return self.samples[idx]


def worker_init_fn(worker_id: int) -> None:
    """Initialize any worker-specific resources."""
    # No need for worker-specific initialization anymore since we process
    # everything in __init__
    pass























# --------------------------------------------------------






# from https://github.com/sdan/nanoEBM/blob/master/nanoebm/data.py




# From https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/projects/chargpt/chargpt.py#L42
import torch
from pathlib import Path


class CharDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, block_size=256, split="train", split_ratio=0.9):
        text = Path(path).read_text(encoding="utf-8")
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(split_ratio * len(data))
        self.data = data[:n] if split == "train" else data[n:]
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]  # (T)
        y = chunk[1:]  # (T)
        return x, y


def get_loader(path, block_size, batch_size, split):
    ds = CharDataset(path, block_size, split)
    return (
        torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True),
        ds,
    )