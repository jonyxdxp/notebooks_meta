



tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def encode_context(utterances, max_len=128):
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

base = '/content/Discourse-Mutual-Information-DMI-main/data/dailydialog'
train_ctx, train_lbl = load_data(f'{base}/dialogues_train.txt',      f'{base}/dialogues_act.txt')
valid_ctx, valid_lbl = load_data(f'{base}/dialogues_validation.txt',  f'{base}/dialogues_act_validation.txt')

print(f"Train: {len(train_ctx)} samples, Valid: {len(valid_ctx)} samples")
print(f"Label distribution train: {np.bincount(train_lbl)}")








# :::::::::::::







# ── Re-load data (run this if train_ctx is not defined) ──────────────────
import numpy as np
import sys
sys.path.insert(0, '/content/Discourse-Mutual-Information-DMI-main')

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

base = '/content/Discourse-Mutual-Information-DMI-main/data/dailydialog'
train_ctx, train_lbl = load_data(f'{base}/dialogues_train.txt',
                                  f'{base}/dialogues_act.txt')
valid_ctx, valid_lbl = load_data(f'{base}/dialogues_validation.txt',
                                  f'{base}/dialogues_act_validation.txt')

# aliases expected by the TF-IDF cell
y_train = train_lbl
y_valid  = valid_lbl

print(f"Train: {len(train_ctx)} | Valid: {len(valid_ctx)}")
print(f"Label distribution train: {np.bincount(y_train)}")







# :::::::::::::::::::::::::::::







import re

def tag_act(utterance):
    u = utterance.strip()
    # Question → ends with ? or starts with question word
    if u.endswith('?'):
        return 1  # Question
    # Directive → imperative-like (starts with verb in base form)
    if re.match(r'^(please|could you|can you|would you|let|make|stop|try|go|come|tell|give|take|bring|put|help)', u.lower()):
        return 2  # Directive
    # Commissive → commitment/promise language
    if re.match(r"^(i will|i'll|i can|i could|i would|i'd|we will|we'll|sure|of course|definitely)", u.lower()):
        return 3  # Commissive
    # Default → Inform
    return 0

def load_data_with_tags(dialog_file):
    dialogs = open(dialog_file).readlines()
    contexts, labels = [], []
    for dialog in dialogs:
        utterances = [u.strip() for u in dialog.strip().split('__eou__') if u.strip()]
        for i in range(1, len(utterances)):
            contexts.append(utterances[:i])
            labels.append(tag_act(utterances[i]))
    return contexts, labels

base = '/content/Discourse-Mutual-Information-DMI-main/data/dailydialog'
train_ctx, train_lbl = load_data_with_tags(f'{base}/dialogues_train.txt')
valid_ctx, valid_lbl = load_data_with_tags(f'{base}/dialogues_valid.txt')

import numpy as np
print(f"Train: {len(train_ctx)} samples")
print(f"Label distribution: {np.bincount(train_lbl)}")
print(f"  0=Inform: {np.bincount(train_lbl)[0]}")
print(f"  1=Question: {np.bincount(train_lbl)[1]}")
print(f"  2=Directive: {np.bincount(train_lbl)[2]}")
print(f"  3=Commissive: {np.bincount(train_lbl)[3]}")




