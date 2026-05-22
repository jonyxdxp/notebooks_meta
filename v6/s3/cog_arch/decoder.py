
# ── PositionalEncoding (needed by decoders) ───────────────────────────────────
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
    




# ── Custom DMI single-turn decoder ────────────────────────────────────────────
class SingleTurnDecoderNew(nn.Module):
    def __init__(self, vocab_size=tokenizer_new.vocab_size,
                 d_model=D_DEC, nhead=8, num_layers=4,
                 dim_feedforward=1024, dropout=0.1, enc_dim=512):
        super().__init__()
        self.d_model  = d_model
        self.tok_emb  = nn.Embedding(vocab_size, d_model,
                                     padding_idx=tokenizer_new.pad_token_id)
        self.pos_enc  = PositionalEncoding(d_model, dropout=dropout)
        self.mem_proj = nn.Linear(enc_dim, d_model)
        dec_layer     = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True)
        self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out      = nn.Linear(d_model, vocab_size)
    def forward(self, tgt_tokens, emb):
        B, T        = tgt_tokens.shape
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=tgt_tokens.device)
        pad_mask    = (tgt_tokens == tokenizer_new.pad_token_id)
        x           = self.pos_enc(
            self.tok_emb(tgt_tokens) * math.sqrt(self.d_model))
        memory      = self.mem_proj(emb).unsqueeze(1)
        out         = self.decoder(tgt=x, memory=memory,
                                   tgt_mask=causal_mask,
                                   tgt_key_padding_mask=pad_mask)
        return self.out(out)