# ── Cell 4: Evaluation — Recall@1 in embedding space ─────────────────────────
# Does the predicted embedding rank the TRUE next turn
# above N random distractors?
# This is the natural evaluation for a JEPA predictor —
# no text generation needed.

predictor.eval()
random.seed(42)

# Sample evaluation pairs
eval_samples = random.sample(list(range(len(valid_ds))), 1000)
ctx_all, tgt_all, pred_all = [], [], []

with torch.no_grad():
    for idx in tqdm(eval_samples, desc="Encoding eval set"):
        ctx, tgt, n = valid_ds[idx]
        mask = (torch.arange(MAX_TURNS) >= n)
        pred = predictor(
            ctx.unsqueeze(0).to(device),
            padding_mask=mask.unsqueeze(0).to(device))
        ctx_all.append(ctx)
        tgt_all.append(tgt)
        pred_all.append(pred.squeeze(0).cpu())

pred_all = torch.stack(pred_all)   # (1000, 768)
tgt_all  = torch.stack(tgt_all)    # (1000, 768)

# Normalize for cosine similarity
pred_n = pred_all / (pred_all.norm(dim=1, keepdim=True) + 1e-8)
tgt_n  = tgt_all  / (tgt_all.norm(dim=1,  keepdim=True) + 1e-8)

# Recall@1 with K random negatives
for K in [9, 49, 99]:
    hits = 0
    for i in range(len(pred_n)):
        neg_idx    = random.sample(
            [j for j in range(len(tgt_n)) if j != i], K)
        candidates = [i] + neg_idx
        scores     = (pred_n[i] @ tgt_n[candidates].T)
        if scores.argmax().item() == 0:
            hits += 1
    print(f"Recall@1 (1-of-{K+1}): {hits/len(pred_n):.4f}  "
          f"(random={1/(K+1):.4f})")

# ── Breakdown by context length ───────────────────────────────────────────────
print("\nRecall@1 (1-of-10) by number of context turns:")
print(f"{'N context turns':<20} {'Recall@1':>10} {'N samples':>10}")
print("─" * 42)
for n_ctx in range(1, MAX_TURNS + 1):
    indices = [i for i, idx in enumerate(eval_samples)
               if valid_ds[idx][2] == n_ctx]
    if len(indices) < 10: continue

    hits = 0
    for i in indices:
        neg_idx    = random.sample(
            [j for j in range(len(tgt_n)) if j != i], 9)
        candidates = [i] + neg_idx
        scores     = pred_n[i] @ tgt_n[candidates].T
        if scores.argmax().item() == 0:
            hits += 1
    print(f"{n_ctx:<20} {hits/len(indices):>10.4f} {len(indices):>10}")