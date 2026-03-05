
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







