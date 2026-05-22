# ── Train amkskam─────────────────────────────────────────────────────────────────────

optimizer_new = torch.optim.AdamW(
    model_single_new.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler_new = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_new, T_max=30)
criterion_new = nn.CrossEntropyLoss(
    ignore_index=tokenizer_new.pad_token_id, label_smoothing=0.1)

best_val_loss_new = float('inf')
save_path_new     = f'{ckpt_dir}/decoder_SINGLE_NEW_best.pth'

print("Training single-turn decoder on NEW encoder...")
for epoch in range(30):
    model_single_new.train()
    tr_loss, n = 0, 0
    for embs, tokens in train_loader_new:
        embs, tokens = embs.to(device), tokens.to(device)
        inp    = tokens[:, :-1]
        tgt    = tokens[:, 1:]
        logits = model_single_new(inp, embs)
        loss   = criterion_new(
            logits.reshape(-1, VOCAB_SIZE_NEW), tgt.reshape(-1))
        optimizer_new.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model_single_new.parameters(), 1.0)
        optimizer_new.step()
        tr_loss += loss.item() * embs.size(0)
        n       += embs.size(0)

    model_single_new.eval()
    vl_loss, vn = 0, 0
    with torch.no_grad():
        for embs, tokens in valid_loader_new:
            embs, tokens = embs.to(device), tokens.to(device)
            logits = model_single_new(tokens[:, :-1], embs)
            loss   = criterion_new(
                logits.reshape(-1, VOCAB_SIZE_NEW), tokens[:, 1:].reshape(-1))
            vl_loss += loss.item() * embs.size(0)
            vn      += embs.size(0)

    tr_ppl = np.exp(tr_loss / n)
    vl_ppl = np.exp(vl_loss / vn)
    scheduler_new.step()

    if vl_loss / vn < best_val_loss_new:
        best_val_loss_new = vl_loss / vn
        torch.save({'epoch':            epoch + 1,
                    'model_state_dict': model_single_new.state_dict(),
                    'val_ppl':          np.exp(best_val_loss_new)},
                   save_path_new)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d} | Train PPL {tr_ppl:.2f} | "
              f"Valid PPL {vl_ppl:.2f}")

print(f"\nBest valid PPL: {np.exp(best_val_loss_new):.2f}")
print(f"\n=== Encoder comparison ===")
print(f"DMI medium (BERT init, Reddit, AUC 0.945): PPL 15.36")
print(f"DMI custom (no BERT init, DD,  AUC 0.790): PPL {np.exp(best_val_loss_new):.2f}")