import math
import torch
import torch.nn as nn

# ── Constants (set once here, imported everywhere else) ────────────────────────
from transformers import AutoTokenizer

TOKENIZER_NAME = "bert-base-uncased"   # or whichever you were using
tokenizer_new  = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
D_DEC          = 256                   # set to whatever your notebook had


# ── PositionalEncoding ─────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class SingleTurnDecoderNew(nn.Module):
    def __init__(self,
                 vocab_size=tokenizer_new.vocab_size,
                 pad_token_id=tokenizer_new.pad_token_id,
                 hidden_size=D_DEC,      # was d_model
                 nhead=8,
                 num_layers=4,
                 dim_feedforward=1024,
                 dropout=0.1,
                 context_dim=512):       # was enc_dim
        super().__init__()
        self.d_model      = hidden_size
        self.pad_token_id = pad_token_id
        self.tok_emb      = nn.Embedding(vocab_size, hidden_size,
                                         padding_idx=pad_token_id)
        self.pos_enc      = PositionalEncoding(hidden_size, dropout=dropout)
        self.mem_proj     = nn.Linear(context_dim, hidden_size)  # enc_dim → context_dim
        dec_layer         = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True)
        self.decoder      = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out          = nn.Linear(hidden_size, vocab_size)
        

    def forward(self, tgt_tokens, emb):
        B, T         = tgt_tokens.shape
        causal_mask  = nn.Transformer.generate_square_subsequent_mask(
            T, device=tgt_tokens.device)
        pad_mask     = (tgt_tokens == self.pad_token_id)
        x            = self.pos_enc(
            self.tok_emb(tgt_tokens) * math.sqrt(self.d_model))
        memory       = self.mem_proj(emb).unsqueeze(1)
        out          = self.decoder(tgt=x, memory=memory,
                                    tgt_mask=causal_mask,
                                    tgt_key_padding_mask=pad_mask)
        return self.out(out)
    

    @torch.no_grad()
    def generate(self, prompt_ids, z_fused, max_new_tokens=30,
                 temperature=1.0, top_k=50):
        generated = prompt_ids
        for _ in range(max_new_tokens):
            logits = self(generated, z_fused)[:, -1, :]
            if top_k:
                v, _ = logits.topk(top_k)
                logits[logits < v[:, -1:]] = -float('inf')
            probs    = torch.softmax(logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_tok], dim=1)
        return generated

Decoder = SingleTurnDecoderNew