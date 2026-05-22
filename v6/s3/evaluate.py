
# ── Evaluate directly on valid dialogs ───────────────────────────────────────

from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer as rs_module
import nltk; nltk.download('punkt', quiet=True)

def decode_single(emb, max_len=40):
    model_single.eval()
    bos, eos  = tokenizer_med.cls_token_id, tokenizer_med.sep_token_id
    generated = [bos]
    emb_b     = emb.unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(max_len):
            inp      = torch.tensor([generated], device=device)
            logits   = model_single(inp, emb_b)
            next_tok = logits[0, -1].argmax().item()
            if next_tok == eos: break
            generated.append(next_tok)
    return tokenizer_med.decode(generated[1:], skip_special_tokens=True)

rouge      = rs_module.RougeScorer(['rougeL'], use_stemmer=True)
refs, hyps, rL = [], [], []
dialogs    = open(f'{base}/dialogues_valid.txt').readlines()

random.seed(42)
for dialog in tqdm(random.sample(dialogs, 300), desc="Evaluating"):
    utts = [u.strip() for u in dialog.strip().split('__eou__') if u.strip()]
    if len(utts) < 1: continue
    for utt in utts[:MAX_TURNS]:
        emb   = encode_single(utt)
        recon = decode_single(emb)
        if not recon.strip(): recon = "i see"
        refs.append([utt.lower().split()])
        hyps.append(recon.lower().split())
        rL.append(rouge.score(utt, recon)['rougeL'].fmeasure)

print("\n=== Single-Turn Reconstruction (this checkpoint) ===")
print(f"BLEU-1:  {corpus_bleu(refs, hyps, weights=(1,0,0,0)):.4f}")
print(f"BLEU-2:  {corpus_bleu(refs, hyps, weights=(.5,.5,0,0)):.4f}")
print(f"BLEU-4:  {corpus_bleu(refs, hyps, weights=(.25,.25,.25,.25)):.4f}")
print(f"ROUGE-L: {np.mean(rL):.4f}")
print(f"Valid PPL: {ckpt['val_ppl']:.2f}")

print("\n=== vs previous best run ===")
print(f"{'Model':<35} {'BLEU-1':>8} {'ROUGE-L':>8} {'PPL':>8}")
print("─" * 55)
print(f"{'DMI medium (previous)':<35} {'0.3277':>8} {'0.5690':>8} {'15.36':>8}")
print(f"{'This checkpoint':<35} "
      f"{corpus_bleu(refs,hyps,weights=(1,0,0,0)):>8.4f} "
      f"{np.mean(rL):>8.4f} {ckpt['val_ppl']:>8.2f}")















# ::::::::::::::::::::::::::::::









# ── Complete evaluation + qualitative comparison ──────────────────────────────

ckpt_dir = '/content/drive/MyDrive/data/dmi_checkpoints'
ckpt     = torch.load(f'{ckpt_dir}/decoder_SINGLE_best.pth',
                       map_location=device, weights_only=False)

print("\n=== Results comparison ===")
print(f"{'Model':<35} {'BLEU-1':>8} {'BLEU-2':>8} {'BLEU-4':>8} {'ROUGE-L':>8} {'PPL':>8}")
print("─" * 75)
print(f"{'DMI medium (best run)':<35} {'0.3277':>8} {'0.2277':>8} {'0.1269':>8} {'0.5690':>8} {'15.36':>8}")
print(f"{'This checkpoint':<35} {'0.2685':>8} {'0.1602':>8} {'0.0616':>8} {'0.4409':>8} {ckpt['val_ppl']:>8.2f}")
print(f"{'Full dialogue → each turn':<35} {'0.1053':>8} {'0.0465':>8} {'  N/A':>8} {'0.1429':>8} {'52.41':>8}")
print(f"{'DMI cond → next utterance':<35} {'0.0422':>8} {'0.0175':>8} {'  N/A':>8} {'0.1428':>8} {'66.93':>8}")

# ── Qualitative: DMI single decoder vs full dialogue decoder ──────────────────
# Reload full dialogue decoder for comparison
class TurnReconstructionDecoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=D_DEC,
                 nhead=8, num_layers=4, dim_feedforward=1024,
                 max_turns=MAX_TURNS, dropout=0.1):
        super().__init__()
        self.d_model  = d_model
        self.tok_emb  = nn.Embedding(vocab_size, d_model,
                                     padding_idx=tokenizer_med.pad_token_id)
        self.pos_enc  = PositionalEncoding(d_model, dropout=dropout)
        self.turn_emb = nn.Embedding(max_turns, 768)
        self.mem_proj = nn.Linear(768, d_model)
        dec_layer     = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True)
        self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out      = nn.Linear(d_model, vocab_size)
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def forward(self, tgt_tokens, c_t, turn_pos):
        B, T        = tgt_tokens.shape
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=tgt_tokens.device)
        pad_mask    = (tgt_tokens == tokenizer_med.pad_token_id)
        x           = self.pos_enc(
            self.tok_emb(tgt_tokens) * math.sqrt(self.d_model))
        pos_emb     = self.turn_emb(turn_pos)
        memory      = self.mem_proj(c_t + pos_emb).unsqueeze(1)
        out         = self.decoder(tgt=x, memory=memory,
                                   tgt_mask=causal_mask,
                                   tgt_key_padding_mask=pad_mask)
        return self.out(out)

model_recon = TurnReconstructionDecoder().to(device)
ckpt_recon  = torch.load(f'{ckpt_dir}/decoder_RECON_best.pth',
                          map_location=device, weights_only=False)
model_recon.load_state_dict(ckpt_recon['model_state_dict'])
model_recon.eval()
print(f"\nLoaded RECON decoder — PPL {ckpt_recon['val_ppl']:.2f}")

def decode_single(emb, max_len=40):
    model_single.eval()
    bos, eos  = tokenizer_med.cls_token_id, tokenizer_med.sep_token_id
    generated = [bos]
    emb_b     = emb.unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(max_len):
            inp      = torch.tensor([generated], device=device)
            logits   = model_single(inp, emb_b)
            next_tok = logits[0, -1].argmax().item()
            if next_tok == eos: break
            generated.append(next_tok)
    return tokenizer_med.decode(generated[1:], skip_special_tokens=True)

def decode_recon(c_t, turn_pos, max_len=40):
    model_recon.eval()
    bos, eos  = tokenizer_med.cls_token_id, tokenizer_med.sep_token_id
    generated = [bos]
    c_t_b     = c_t.unsqueeze(0).to(device)
    pos_b     = torch.tensor([turn_pos], device=device)
    with torch.no_grad():
        for _ in range(max_len):
            inp      = torch.tensor([generated], device=device)
            logits   = model_recon(inp, c_t_b, pos_b)
            next_tok = logits[0, -1].argmax().item()
            if next_tok == eos: break
            generated.append(next_tok)
    return tokenizer_med.decode(generated[1:], skip_special_tokens=True)

# ── Side-by-side qualitative examples ────────────────────────────────────────
print("\n=== Qualitative Comparison ===")
print("SINGLE: encode_single(utt) → decode that one utt")
print("RECON:  encode_context(all utts) + position → decode that utt")
print("─" * 80)

dialogs = open(f'{base}/dialogues_valid.txt').readlines()
random.seed(42)

for dialog in random.sample(dialogs, 6):
    utts = [u.strip() for u in dialog.strip().split('__eou__') if u.strip()]
    if len(utts) < 3: continue
    utts  = utts[:MAX_TURNS]
    c_t   = encode_context(utts)

    print(f"\nDIALOGUE ({len(utts)} turns):")
    print(f"{'Turn':<6} {'ORIGINAL':<35} {'SINGLE decoder':<35} {'RECON decoder':<35}")
    print("─" * 111)
    for pos, utt in enumerate(utts):
        emb          = encode_single(utt)
        out_single   = decode_single(emb)
        out_recon    = decode_recon(c_t, pos)

        orig_words   = set(utt.lower().split())
        ov_single    = len(orig_words & set(out_single.lower().split())) / (len(orig_words)+1e-8)
        ov_recon     = len(orig_words & set(out_recon.lower().split()))  / (len(orig_words)+1e-8)

        print(f"[{pos}]    {utt[:33]:<35} "
              f"{out_single[:33]:<35} ({ov_single:.2f}) "
              f"{out_recon[:33]:<35} ({ov_recon:.2f})")
    print()