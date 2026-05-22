
import torch
from transformers import BertModel, BertTokenizer
from v6.s2.config import DMI_CKPT, BERT_NAME, CKPT_DIR, DEVICE

print("Loading DMI medium encoder...")
ckpt_med   = torch.load(DMI_CKPT, map_location=DEVICE, weights_only=False)
state_dict = {k.replace("module.", ""): v
              for k, v in ckpt_med["model_state_dict"].items()}

bert_med = BertModel.from_pretrained(
    BERT_NAME, add_pooling_layer=False).to(DEVICE)
encoder_state = {k.replace("encoder.", "", 1): v
                 for k, v in state_dict.items()
                 if k.startswith("encoder.")}
bert_med.load_state_dict(encoder_state, strict=False)
bert_med.eval()

tokenizer_med = BertTokenizer.from_pretrained(BERT_NAME)
print(f"DMI encoder ready — AUC: {ckpt_med['auc']:.4f}")


def encode_single(text, max_len=64):
    tokens = tokenizer_med(text, return_tensors="pt",
                           max_length=max_len, truncation=True,
                           padding="max_length")
    with torch.no_grad():
        out = bert_med(
            input_ids=tokens["input_ids"].to(DEVICE),
            attention_mask=tokens["attention_mask"].to(DEVICE))
    return out.last_hidden_state[:, 0, :].squeeze().cpu()


def encode_context(utterances, max_len=128):
    text   = " [SEP] ".join(utterances)
    tokens = tokenizer_med(text, return_tensors="pt",
                           max_length=max_len, truncation=True,
                           padding="max_length")
    with torch.no_grad():
        out = bert_med(
            input_ids=tokens["input_ids"].to(DEVICE),
            attention_mask=tokens["attention_mask"].to(DEVICE))
    return out.last_hidden_state[:, 0, :].squeeze().cpu()
