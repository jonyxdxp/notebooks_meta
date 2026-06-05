import re
import numpy as np
import torch
from transformers import RobertaTokenizer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


def encode_context(model, utterances, max_len=128):   # ← add model here
    text = ' </s> '.join(utterances)
    tokens = tokenizer(text, return_tensors='pt',
                       max_length=max_len, truncation=True,
                       padding='max_length')
    input_ids = tokens['input_ids'].to(device)
    with torch.no_grad():
        emb = model.embedding(input_ids)
        out = model.encoder(emb, None)
        c_t = out[:, 0, :]
    return c_t.squeeze().cpu().numpy()


def load_data(dialog_file, act_file):
    dialogs = open(dialog_file).readlines()
    acts    = open(act_file).readlines()
    contexts, labels = [], []
    for dialog, act_line in zip(dialogs, acts):
        utterances = [u.strip() for u in dialog.strip().split('__eou__') if u.strip()]
        act_labels = [int(a) - 1 for a in act_line.strip().split()]
        for i in range(1, len(utterances)):
            contexts.append(utterances[:i])
            labels.append(act_labels[i])
    return contexts, labels


def tag_act(utterance):
    u = utterance.strip()
    if u.endswith('?'):
        return 1  # Question
    if re.match(r'^(please|could you|can you|would you|let|make|stop|try|go|come|tell|give|take|bring|put|help)', u.lower()):
        return 2  # Directive
    if re.match(r"^(i will|i'll|i can|i could|i would|i'd|we will|we'll|sure|of course|definitely)", u.lower()):
        return 3  # Commissive
    return 0  # Inform


def load_data_with_tags(dialog_file):
    dialogs = open(dialog_file).readlines()
    contexts, labels = [], []
    for dialog in dialogs:
        utterances = [u.strip() for u in dialog.strip().split('__eou__') if u.strip()]
        for i in range(1, len(utterances)):
            contexts.append(utterances[:i])
            labels.append(tag_act(utterances[i]))
    return contexts, labels














class DatasetObject:
    """
    Wraps dailydialog into n_tasks sequential chunks for MOML continual learning.

    Attributes:
        trn_x / trn_y  : list[list]  – one entry per task, training contexts / labels
        tst_x / tst_y  : list[list]  – one entry per task, test contexts / labels
        name           : str         – used for save-path construction in train_MOML
        dataset        : str         – dataset identifier string
    """

    def __init__(self, dataset, train_file, valid_file, test_file, n_tasks=10):
        self.dataset = dataset
        self.name    = dataset                                  # e.g. 'dailydialog'

        trn_ctx, trn_lbl = load_data_with_tags(train_file)
        tst_ctx, tst_lbl = load_data_with_tags(test_file)

        self.trn_x, self.trn_y = self._chunk(trn_ctx, trn_lbl, n_tasks)
        self.tst_x, self.tst_y = self._chunk(tst_ctx, tst_lbl, n_tasks)

        print(f'[DatasetObject] {n_tasks} tasks | '
              f'train: {[len(x) for x in self.trn_x]} samples | '
              f'test:  {[len(x) for x in self.tst_x]} samples')

    @staticmethod
    def _chunk(contexts, labels, n_tasks):
        n          = len(contexts)
        chunk_size = n // n_tasks
        task_x, task_y = [], []
        for i in range(n_tasks):
            start = i * chunk_size
            end   = (start + chunk_size) if (i < n_tasks - 1) else n
            task_x.append(contexts[start:end])
            task_y.append(labels[start:end])
        return task_x, task_y









class Dataset(torch.utils.data.Dataset):
    def __init__(self, contexts, labels, train=True, dataset_name='dailydialog', max_len=128):
        self.contexts = contexts
        self.labels   = labels
        self.max_len  = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        utterances = self.contexts[idx]
        text       = ' </s> '.join(utterances)
        tokens     = tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
        )
        input_ids = tokens['input_ids'].squeeze(0)          # (max_len,)
        label     = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, label