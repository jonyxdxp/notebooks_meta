
# (Losses and Regularization ("retention" in Cognitive terms))




# from https://github.com/facebookresearch/eb_jepa/blob/main/eb_jepa/losses.py






import torch
import torch.nn as nn
import torch.nn.functional as F


def sq_loss(x, y, reduction="mean"):
    """Simple square loss (MSE)."""
    return nn.functional.mse_loss(x, y, reduction=reduction)


def square_cost_seq(state, predi):
    """Square loss between two [B, C, T, H, W] sequences."""
    return sq_loss(state, predi)







class SquareLossSeq(nn.Module):
    """Square loss over a sequence [B, C, T, H, W] (feature dim at dim 1)."""

    def __init__(self, proj=None):
        super().__init__()
        self.proj = nn.Identity() if proj is None else proj

    def forward(self, state, predi):
        state = self.proj(state.transpose(0, 1).flatten(1).transpose(0, 1))
        predi = self.proj(predi.transpose(0, 1).flatten(1).transpose(0, 1))
        return square_cost_seq(state, predi)






class VCLoss(nn.Module):
    """Variance-Covariance loss attracting means to zero and covariance to identity."""

    def __init__(self, std_coeff, cov_coeff, proj=None):
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.proj = nn.Identity() if proj is None else proj
        self.std_loss_fn = HingeStdLoss(std_margin=1.0)
        self.cov_loss_fn = CovarianceLoss()

    def forward(self, x, actions=None):
        x = x.transpose(0, 1).flatten(1).transpose(0, 1)  # [B*T*H*W, C]
        fx = self.proj(x)  # [B*T*H*W, C']

        std_loss = self.std_loss_fn(fx)
        cov_loss = self.cov_loss_fn(fx)

        loss = self.std_coeff * std_loss + self.cov_coeff * cov_loss
        total_unweighted_loss = std_loss + cov_loss
        loss_dict = {
            "std_loss": std_loss.item(),
            "cov_loss": cov_loss.item(),
        }
        return loss, total_unweighted_loss, loss_dict







class HingeStdLoss(torch.nn.Module):
    def __init__(
        self,
        std_margin: float = 1.0,
    ):
        """
        Encourages each feature to maintain at least a minimum standard deviation.
        Features with std below the margin incur a penalty of (std_margin - std).
        Args:
            std_margin (float, default=1.0):
                Minimum desired standard deviation per feature.
        """
        super().__init__()
        self.std_margin = std_margin

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, D] where N is number of samples, D is feature dimension
        Returns:
            std_loss: Scalar tensor with the hinge loss on standard deviations
        """
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(self.std_margin - std))
        return std_loss







class CovarianceLoss(torch.nn.Module):
    def __init__(self):
        """
        Penalizes off-diagonal elements of the covariance matrix to encourage
        feature decorrelation.

        Normalizes by D * (D - 1) where D is feature dimensionality.
        """
        super().__init__()

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, D] where N is number of samples, D is feature dimension
        """
        batch_size = x.shape[0]
        num_features = x.shape[-1]
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / (batch_size - 1)  # [D, D]
        # Calculate off-diagonal loss
        cov_loss = self.off_diagonal(cov).pow(2).mean()

        return cov_loss








# v6/s2/losses.py — add combined loss

class MSEInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07, alpha=0.5):
        super().__init__()
        self.infonce = InfoNCELoss(temperature)
        self.mse     = nn.MSELoss()
        self.alpha   = alpha

    def forward(self, pred, target):
        return (self.alpha * self.infonce(pred, target) +
                (1 - self.alpha) * self.mse(pred, target))







# --------------------------------------------------------







"""
loss.py — InfoNCE (NT-Xent) contrastive loss and Recall@K evaluation.

InfoNCE intuition for our task:
  - Each sample in a batch has a predicted context embedding (pred) and a
    true next-utterance embedding (target).
  - The diagonal of the similarity matrix = correct pairs (positives).
  - All off-diagonal entries = in-batch negatives.
  - The model is trained to make each pred closest to its own target.

Recall@K measures: for each context in a candidate pool, is the true
  next utterance in the model's top-K retrieved candidates?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── InfoNCE Loss ───────────────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE (NT-Xent) loss over in-batch negatives.

    Given:
      pred   (B, D) — projected context vectors        (anchors)
      target (B, D) — DSE embeddings of true next turn (positives)

    Computes bidirectional cross-entropy:
      L = 0.5 * [ CE(sim(pred, target), diag) + CE(sim(target, pred), diag) ]

    Using both directions stabilises training and is standard in SimCSE / CLIP.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self,
                pred:   torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        pred:   (B, D)
        target: (B, D)
        returns: scalar loss
        """
        B = pred.size(0)

        # L2-normalise so cosine similarity = dot product
        pred_n   = F.normalize(pred,   dim=-1)   # (B, D)
        target_n = F.normalize(target, dim=-1)   # (B, D)

        # similarity matrix (B, B)
        sim = torch.matmul(pred_n, target_n.T) / self.temperature

        # labels: diagonal indices are the positives
        labels = torch.arange(B, device=pred.device)

        # bidirectional cross-entropy
        loss_p2t = F.cross_entropy(sim,   labels)   # pred   → target direction
        loss_t2p = F.cross_entropy(sim.T, labels)   # target → pred   direction

        return 0.5 * (loss_p2t + loss_t2p)


# ── Recall@K Metric ────────────────────────────────────────────────────────────

@torch.no_grad()
def recall_at_k(
    pred_embs:   torch.Tensor,
    target_embs: torch.Tensor,
    ks: tuple = (1, 5, 10),
) -> dict:
    """
    Given a pool of predictions and a pool of candidate targets, compute
    Recall@K: the fraction of queries for which the true target is in
    the top-K retrieved items.

    pred_embs:   (N, D) — context predictions (one per conversation window)
    target_embs: (N, D) — corresponding true next utterances
    ks:          tuple of K values to evaluate

    This evaluates retrieval in the full pool (N candidates), not just a batch,
    so it should be called after accumulating embeddings over the whole eval set.
    """
    pred_n   = F.normalize(pred_embs,   dim=-1)
    target_n = F.normalize(target_embs, dim=-1)

    # (N, N) similarity; entry [i, j] = similarity of pred_i to target_j
    sim = torch.matmul(pred_n, target_n.T)          # (N, N)

    results = {}
    for k in ks:
        # for each query i, check if target i is in the top-k columns
        top_k_indices = sim.topk(k, dim=-1).indices  # (N, k)
        ground_truth  = torch.arange(sim.size(0), device=sim.device).unsqueeze(1)
        correct = (top_k_indices == ground_truth).any(dim=-1).float()
        results[f"R@{k}"] = correct.mean().item()

    return results


# ── Mean Reciprocal Rank ───────────────────────────────────────────────────────

@torch.no_grad()
def mean_reciprocal_rank(
    pred_embs:   torch.Tensor,
    target_embs: torch.Tensor,
) -> float:
    """
    MRR over the full pool. Useful as a single summary metric.
    """
    pred_n   = F.normalize(pred_embs,   dim=-1)
    target_n = F.normalize(target_embs, dim=-1)

    sim  = torch.matmul(pred_n, target_n.T)          # (N, N)
    N    = sim.size(0)

    # rank of the correct item for each query (1-indexed)
    sorted_idx = sim.argsort(dim=-1, descending=True)                   # (N, N)
    gt         = torch.arange(N, device=sim.device).unsqueeze(1)        # (N, 1)
    ranks      = (sorted_idx == gt).nonzero(as_tuple=False)[:, 1] + 1  # (N,)

    mrr = (1.0 / ranks.float()).mean().item()
    return mrr