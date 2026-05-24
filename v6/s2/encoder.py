
import torch
from transformers import BertModel, BertTokenizer
from config import cfg

print("Loading DMI medium encoder...")
_ckpt      = torch.load(cfg.dmi_ckpt, map_location=cfg.device, weights_only=False)
_state     = {k.replace("module.", ""): v
              for k, v in _ckpt["model_state_dict"].items()}

_bert = BertModel.from_pretrained(
    cfg.bert_name, add_pooling_layer=False).to(cfg.device)
_enc_state = {k.replace("encoder.", "", 1): v
              for k, v in _state.items() if k.startswith("encoder.")}
_bert.load_state_dict(_enc_state, strict=False)
_bert.eval()

_tokenizer = BertTokenizer.from_pretrained(cfg.bert_name)
print(f"DMI encoder ready — AUC: {_ckpt['auc']:.4f}")


def encode_single(text, max_len=64):
    tok = _tokenizer(text, return_tensors="pt",
                     max_length=max_len, truncation=True, padding="max_length")
    with torch.no_grad():
        out = _bert(input_ids=tok["input_ids"].to(cfg.device),
                    attention_mask=tok["attention_mask"].to(cfg.device))
    return out.last_hidden_state[:, 0, :].squeeze().cpu()


def encode_context(utterances, max_len=128):
    text = " [SEP] ".join(utterances)
    tok  = _tokenizer(text, return_tensors="pt",
                      max_length=max_len, truncation=True, padding="max_length")
    with torch.no_grad():
        out = _bert(input_ids=tok["input_ids"].to(cfg.device),
                    attention_mask=tok["attention_mask"].to(cfg.device))
    return out.last_hidden_state[:, 0, :].squeeze().cpu()
