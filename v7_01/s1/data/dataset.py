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