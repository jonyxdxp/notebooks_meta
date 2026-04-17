"""
eval.py — Evaluation utilities for Text JEPA (Stage 1)
Provides:
  - mean_pool            : (B, L, D) hidden → (B, D) sentence vector
  - LinearProbe          : thin classifier head for downstream tasks
  - train_linear_probe   : fits probe on frozen encoder representations
  - evaluate_linear_probe: accuracy + loss on val set
  - representation_quality: alignment & uniformity (label-free SSL metrics)
  - run_full_eval        : convenience wrapper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ── Pooling ───────────────────────────────────────────────────────────────────

def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool encoder hidden states, ignoring padding.

    Args:
        hidden          : (B, L, D)
        attention_mask  : (B, L)  — 1 for real tokens, 0 for padding

    Returns:
        pooled : (B, D)
    """
    mask = attention_mask.unsqueeze(-1).float()        # (B, L, 1)
    summed = (hidden * mask).sum(dim=1)                # (B, D)
    count  = mask.sum(dim=1).clamp(min=1e-9)           # (B, 1)
    return summed / count


# ── Linear probe ─────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# ── Extract representations ───────────────────────────────────────────────────

@torch.no_grad()
def extract_representations(
    encoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run encoder over the full loader, return (representations, labels).

    Expects each batch to be a dict with keys:
        input_ids, attention_mask, labels   (labels can be -1 if unavailable)

    Returns:
        reps   : (N, D)
        labels : (N,)
    """
    encoder.eval()
    all_reps, all_labels = [], []

    for batch in tqdm(loader, desc='Extracting reps', leave=False):
        # DESPUÉS
        ids  = batch.get('input_ids', batch.get('context_input_ids')).to(device)
        mask = batch.get('attention_mask', batch.get('context_attention_mask')).to(device)

        hidden = encoder(ids, attention_mask=mask)
        if isinstance(hidden, tuple):
            hidden = hidden[0]                         # (B, L, D)

        reps = mean_pool(hidden, mask)                 # (B, D)
        all_reps.append(reps.cpu())

        labels = batch.get('labels', torch.full((ids.size(0),), -1))
        all_labels.append(labels.cpu())

    return torch.cat(all_reps), torch.cat(all_labels)


# ── Train linear probe ────────────────────────────────────────────────────────

def train_linear_probe(
    train_reps: torch.Tensor,
    train_labels: torch.Tensor,
    feature_dim: int,
    num_classes: int,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> LinearProbe:
    """
    Fit a linear probe on pre-extracted representations.
    Keeps encoder frozen — only the probe is trained.
    """
    probe = LinearProbe(feature_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    dataset = TensorDataset(train_reps, train_labels)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    probe.train()
    for epoch in range(epochs):
        total_loss, correct, n = 0.0, 0, 0
        for reps, labels in loader:
            reps, labels = reps.to(device), labels.to(device)
            logits = probe(reps)
            loss   = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            correct    += logits.argmax(1).eq(labels).sum().item()
            n          += len(labels)

        if (epoch + 1) % 5 == 0:
            print(f'  probe epoch {epoch+1:02d}/{epochs}  '
                  f'loss={total_loss/n:.4f}  acc={100*correct/n:.1f}%')

    return probe


# ── Evaluate linear probe ─────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_linear_probe(
    probe: LinearProbe,
    val_reps: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> dict:
    """
    Evaluate a trained linear probe on pre-extracted val representations.

    Returns dict with 'accuracy' and 'loss'.
    """
    probe.eval()
    dataset = TensorDataset(val_reps, val_labels)
    loader  = DataLoader(dataset, batch_size=batch_size)

    total_loss, correct, n = 0.0, 0, 0
    for reps, labels in loader:
        reps, labels = reps.to(device), labels.to(device)
        logits = probe(reps)
        loss   = F.cross_entropy(logits, labels)
        total_loss += loss.item() * len(labels)
        correct    += logits.argmax(1).eq(labels).sum().item()
        n          += len(labels)

    return {
        'accuracy': 100.0 * correct / n,
        'loss':     total_loss / n,
    }


# ── Representation quality (label-free) ──────────────────────────────────────

@torch.no_grad()
def representation_quality(reps: torch.Tensor) -> dict:
    """
    Alignment & uniformity — standard SSL representation quality metrics.
    Wang & Isola (2020): https://arxiv.org/abs/2005.10242

    alignment  : how similar positive pairs are (lower = better)
                 here approximated as avg cosine similarity between
                 adjacent pairs (assumes loader returns augmented pairs)
    uniformity : how uniformly spread reps are on the unit hypersphere
                 (lower = better, ideally ~ -2.0)

    Args:
        reps : (N, D) — raw representations, will be L2-normalised internally

    Returns:
        dict with 'uniformity', 'mean_norm', 'std_norm'
    """
    reps_norm = F.normalize(reps, dim=-1)              # (N, D)

    # Uniformity: avg pairwise Gaussian kernel on unit sphere
    # Use a random subsample if N is large to keep it fast
    max_samples = 2048
    if reps_norm.size(0) > max_samples:
        idx = torch.randperm(reps_norm.size(0))[:max_samples]
        sub = reps_norm[idx]
    else:
        sub = reps_norm

    sq_pdist = torch.pdist(sub, p=2).pow(2)
    uniformity = sq_pdist.mul(-2).exp().mean().log().item()

    norms = reps.norm(dim=-1)
    return {
        'uniformity': uniformity,        # lower → more uniform (target ≈ -2)
        'mean_norm':  norms.mean().item(),
        'std_norm':   norms.std().item(),
    }


# ── Convenience wrapper ───────────────────────────────────────────────────────

def run_full_eval(
    encoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    feature_dim: int,
    probe_epochs: int = 20,
) -> dict:
    """
    Full evaluation pipeline:
      1. Extract representations for train + val
      2. Report representation quality (label-free)
      3. Train linear probe on train reps
      4. Evaluate probe on val reps

    Returns a flat dict of all metrics.
    """
    print('── Extracting train representations …')
    train_reps, train_labels = extract_representations(encoder, train_loader, device)

    print('── Extracting val representations …')
    val_reps, val_labels = extract_representations(encoder, val_loader, device)

    print('── Representation quality (val) …')
    quality = representation_quality(val_reps)
    print(f'   uniformity={quality["uniformity"]:.3f}  '
          f'mean_norm={quality["mean_norm"]:.3f}  '
          f'std_norm={quality["std_norm"]:.3f}')

    results = {**quality}

    if train_labels.eq(-1).all():
        print('── No labels found — skipping linear probe.')
    else:
        print(f'── Training linear probe ({num_classes} classes, {probe_epochs} epochs) …')
        probe = train_linear_probe(
            train_reps, train_labels,
            feature_dim=feature_dim,
            num_classes=num_classes,
            device=device,
            epochs=probe_epochs,
        )
        probe_metrics = evaluate_linear_probe(probe, val_reps, val_labels, device)
        print(f'   probe accuracy={probe_metrics["accuracy"]:.1f}%  '
              f'loss={probe_metrics["loss"]:.4f}')
        results.update(probe_metrics)

    return results