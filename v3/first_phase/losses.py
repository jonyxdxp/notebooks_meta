

# (Losses and Regularization ("retention" in Cognitive terms))


# will also be the "Energy Function" for our EBM, since we will be training the EBM to minimize this loss function, so the loss function is effectively the energy function that the EBM is modeling. 
# The EBM will learn to assign low energy (loss) to good reconstructions and high energy (loss) to bad reconstructions, thus learning a landscape of the loss function.




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







class TemporalSimilarityLoss(torch.nn.Module):
    def __init__(self):
        """
        Temporal Similarity Loss.
        Encourages consecutive frames to have similar representations by penalizing
        the squared difference between consecutive time steps.
        """
        super().__init__()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [T, N, D] where T is time steps, N is batch size, D is feature dimension
        """
        if x.shape[0] <= 1:
            return torch.tensor(0.0, device=x.device)
        sim_loss_t = (x[1:] - x[:-1]).pow(2).mean()
        return sim_loss_t







class InverseDynamicsLoss(torch.nn.Module):
    def __init__(self, idm: nn.Module):
        """
        Predicts actions from consecutive states and compares with ground truth actions.
        Args:
            idm (nn.Module): Inverse dynamics model that takes (state_t, state_t+1) and predicts action
        """
        super().__init__()
        self.idm = idm

    def forward(self, x: torch.Tensor, actions: torch.Tensor):
        """
        Args:
            x: [T, B, D] - States across time steps
            actions: [B, A, T] - Ground truth actions between consecutive states
        """
        if x.shape[0] <= 1 or actions is None:
            return torch.tensor(0.0, device=x.device)

        t, b, d = x.shape

        states_t = x[:-1].transpose(0, 1)  # [B, T-1, D]
        states_t_plus_1 = x[1:].transpose(0, 1)  # [B, T-1, D]

        states_t_flat = states_t.reshape(-1, d)  # [B*(T-1), D]
        states_t_plus_1_flat = states_t_plus_1.reshape(-1, d)  # [B*(T-1), D]

        pred_actions = self.idm(states_t_flat, states_t_plus_1_flat)  # [B*(T-1), A]
        target_actions = actions.transpose(1, 2)[:, :-1].reshape(
            -1, actions.size(1)
        )  # [B*(T-1), A]
        idm_loss = F.mse_loss(pred_actions, target_actions)

        return idm_loss







class VC_IDM_Sim_Regularizer(torch.nn.Module):
    def __init__(
        self,
        cov_coeff: float,
        std_coeff: float,
        sim_coeff_t: float,
        idm_coeff: float = 0.0,
        idm: nn.Module = None,
        std_margin: float = 1,
        first_t_only: bool = True,
        projector: nn.Module = None,
        spatial_as_samples: bool = False,
        sim_t_after_proj: bool = False,
        idm_after_proj: bool = False,
    ):
        """
        Composite Regularizer combining multiple losses

        This is a composite loss that combines:
        - Hinge Standard Deviation Loss
        - Covariance Decorrelation Loss
        - Temporal Similarity Loss
        - Inverse Dynamics Model Loss

        Args:
            cov_coeff (float): Weight for covariance loss
            std_coeff (float): Weight for std hinge loss
            sim_coeff_t (float): Weight for temporal similarity loss
            idm_coeff (float): Weight for inverse dynamics loss
            idm (nn.Module): Inverse dynamics model
            std_margin (float): Minimum desired std per feature
            first_t_only (bool): Use only first time slice for std/cov loss
            projector (nn.Module): Optional projection layer
            spatial_as_samples (bool): Treat spatial locations as samples
            sim_t_after_proj (bool): Apply temporal loss after projection
            idm_after_proj (bool): Apply IDM loss after projection
        """
        super().__init__()
        self.cov_coeff = cov_coeff
        self.std_coeff = std_coeff
        self.sim_coeff_t = sim_coeff_t
        self.idm_coeff = idm_coeff

        self.first_t_only = first_t_only
        self.projector = nn.Identity() if projector is None else projector
        self.spatial_as_samples = spatial_as_samples
        self.sim_t_after_proj = sim_t_after_proj
        self.idm_after_proj = idm_after_proj

        # Initialize individual loss components
        self.std_loss_fn = HingeStdLoss(std_margin=std_margin)
        self.cov_loss_fn = CovarianceLoss()
        self.sim_loss_fn = TemporalSimilarityLoss()
        self.idm_loss_fn = InverseDynamicsLoss(idm) if idm is not None else None

    def forward(self, x, actions=None):
        """
        Args:
            x: [B, C, T, H, W] - Input activations. Internally reshaped to either
                [1, B, D] when first_t_only=True or [T*B, D] otherwise, with D=C*H*W.
            actions: [B, A, T] - Optional actions for IDM loss
        """
        b, c, t, h, w = x.shape

        # divergent gradient paths for x_unprojected and x_projected
        x_unprojected = x.permute(2, 0, 1, 3, 4).reshape(t, b, -1)  # [T, B, C*H*W]

        x_flat = x.permute(0, 2, 3, 4, 1).reshape(-1, c)  # [B*T*H*W, C]
        x_proj = self.projector(x_flat)  # [B*T*H*W, C_out]
        c_out = x_proj.shape[-1]
        x_projected = x_proj.view(b, t, h, w, c_out)  # [B, T, H, W, C_out]
        x_projected_reshaped = x_projected.permute(2, 0, 1, 3, 4).reshape(
            t, b, -1
        )  # [T, B, C_out*H*W]

        # SIM_T LOSS
        if self.sim_t_after_proj:
            sim_loss_t = self.sim_loss_fn(x_projected_reshaped)
        else:
            sim_loss_t = self.sim_loss_fn(x_unprojected)

        # IDM LOSS
        idm_loss = torch.tensor(0.0, device=x.device)
        if self.idm_coeff > 0 and self.idm_loss_fn is not None and actions is not None:
            if self.idm_after_proj:
                idm_loss = self.idm_loss_fn(x_projected_reshaped, actions)
            else:
                idm_loss = self.idm_loss_fn(x_unprojected, actions)

        # STD and COV LOSS
        if self.spatial_as_samples:
            if self.first_t_only:
                # Use only first time: [B*H*W, C_out]
                x_for_vc = x_projected[:, 0].reshape(b * h * w, c_out)
                assert x_for_vc.shape == (b * h * w, c_out)
            else:
                # Use all times: [B*T*H*W, C_out]
                x_for_vc = x_projected.reshape(-1, c_out)
                assert x_for_vc.shape == (b * t * h * w, c_out)
        else:
            x_for_vc = x_projected.permute(0, 1, 4, 2, 3).reshape(
                b, t, -1
            )  # [B, T, C_out*H*W]
            if self.first_t_only:
                # Use only first time: [B, C_out*H*W]
                x_for_vc = x_for_vc[:, 0]
                assert x_for_vc.shape == (b, c_out * h * w)
            else:
                # Use all times: [B*T, C_out*H*W]
                x_for_vc = x_for_vc.reshape(-1, x_for_vc.size(-1))
                assert x_for_vc.shape == (b * t, c_out * h * w)
        # [B*T, C_out*H*W] if first_t_only=False and spatial_as_samples=False
        # or [B, C_out*H*W] if first_t_only=True and spatial_as_samples=False
        # or [B*H*W, C_out] if first_t_only=True spatial_as_samples=True
        # or [B*T*H*W, C_out] if first_t_only=False spatial_as_samples=True
        std_loss = self.std_loss_fn(x_for_vc)
        cov_loss = self.cov_loss_fn(x_for_vc)

        total_weighted_loss = (
            self.cov_coeff * cov_loss
            + self.std_coeff * std_loss
            + self.sim_coeff_t * sim_loss_t
            + self.idm_coeff * idm_loss
        )
        total_unweighted_loss = cov_loss + std_loss + sim_loss_t + idm_loss

        loss_dict = {
            "cov_loss": cov_loss.item(),
            "std_loss": std_loss.item(),
            "sim_loss_t": sim_loss_t.item(),
            "idm_loss": idm_loss if isinstance(idm_loss, float) else idm_loss.item(),
        }

        return total_weighted_loss, total_unweighted_loss, loss_dict









class VICRegLoss(nn.Module):
    """VICReg loss combining invariance, variance (std), and covariance terms."""

    def __init__(self, std_coeff=1.0, cov_coeff=1.0):
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.std_loss_fn = HingeStdLoss(std_margin=1.0)
        self.cov_loss_fn = CovarianceLoss()

    def forward(self, z1, z2):
        """Compute VICReg loss.

        Args:
            z1: [B, D] - First projection tensor
            z2: [B, D] - Second projection tensor

        Returns:
            dict with keys: loss, invariance_loss, var_loss, cov_loss
        """
        # Invariance loss (similarity)
        sim_loss = F.mse_loss(z1, z2)

        # Variance loss (applied to both views and summed)
        var_loss = self.std_loss_fn(z1) + self.std_loss_fn(z2)

        # Covariance loss (applied to both views and summed)
        cov_loss = self.cov_loss_fn(z1) + self.cov_loss_fn(z2)

        total_loss = sim_loss + self.std_coeff * var_loss + self.cov_coeff * cov_loss

        return {
            "loss": total_loss,
            "invariance_loss": sim_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
        }










######################################################
# BCS (Batched Characteristic Slicing) loss for SIGReg


def all_reduce(x, op):
    """All-reduce operation for distributed training."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        op = dist.ReduceOp.__dict__[op]
        dist.all_reduce(x, op=op)
        return x
    else:
        return x


def epps_pulley(x, t_min=-3, t_max=3, n_points=10):
    """Epps-Pulley test statistic for Gaussianity."""
    # integration points
    t = torch.linspace(t_min, t_max, n_points, device=x.device)
    # theoretical CF for N(0, 1)
    exp_f = torch.exp(-0.5 * t**2)
    # ECF
    x_t = x.unsqueeze(2) * t  # (N, M, T)
    ecf = (1j * x_t).exp().mean(0)
    ecf = all_reduce(ecf, op="AVG")
    # weighted L2 distance
    err = exp_f * (ecf - exp_f).abs() ** 2
    T = torch.trapz(err, t, dim=1)
    return T


class BCS(nn.Module):
    """BCS (Batched Characteristic Slicing) loss for SIGReg."""

    def __init__(self, num_slices=256, lmbd=10.0):
        super().__init__()
        self.num_slices = num_slices
        self.step = 0
        self.lmbd = lmbd

    def forward(self, z1, z2):
        with torch.no_grad():
            dev = z1.device
            g = torch.Generator(device=dev)
            g.manual_seed(self.step)
            proj_shape = (z1.size(1), self.num_slices)
            A = torch.randn(proj_shape, device=dev, generator=g)
            A /= A.norm(p=2, dim=0)
        view1 = z1 @ A
        view2 = z2 @ A

        self.step += 1
        bcs = (epps_pulley(view1).mean() + epps_pulley(view2).mean()) / 2
        invariance_loss = F.mse_loss(z1, z2).mean()
        total_loss = invariance_loss + self.lmbd * bcs
        return {"loss": total_loss, "bcs_loss": bcs, "invariance_loss": invariance_loss}


















# ------------------------------------------------------------


















# from https://github.com/galilai-group/llm-jepa/blob/main/finetune.py




# LLM-JEPA LOSS








# class RepresentationTrainer(Trainer):
#     """
#     Trainer to regularize representations.
#     """
    
#     def __init__(self, *args, **kwargs):
#         # Extract custom loss parameters
#         self.lbd = kwargs.pop('lbd', 1.0)
#         self.gamma = kwargs.pop('gamma', 1.0)
#         self.last_token = kwargs.pop('last_token', -2)
#         self.debug = kwargs.pop('debug', 0)
#         self.additive_mask = kwargs.pop('additive_mask', False)
#         self.jepa_l2 = kwargs.pop('jepa_l2', False)
#         self.jepa_mse = kwargs.pop('jepa_mse', False)
#         self.infonce = kwargs.pop('infonce', False)
#         self.jepa_ratio = kwargs.pop('jepa_ratio', -1.0)
#         assert self.jepa_l2 + self.jepa_mse <= 1, "Only one of jepa_l2 and jepa_mse can be True."
#         super().__init__(*args, **kwargs)
    
#     def _last_token_index(self, input_ids, labels, attention_mask):
#         index = []
#         def unpad(input_ids, attention_mask):
#             result = []
#             can_break = False
#             for id, mask in zip(input_ids, attention_mask):
#                 if mask != 0:
#                     can_break = True
#                 if mask == 0 and can_break:
#                     break
#                 result.append(id)
#             return result

#         for i in range(input_ids.shape[0]):
#             uii = unpad(input_ids[i], attention_mask[i])
#             if self.debug == 1 and torch.cuda.current_device() == 0:
#                 print(f"====={len(uii)}=====")
#                 print(input_ids[i][len(uii) - 4], input_ids[i][len(uii) - 3], input_ids[i][len(uii) - 2], input_ids[i][len(uii) - 1], -100 if len(uii) >= len(input_ids[i]) else input_ids[i][len(uii)])
#                 print(labels[i][len(uii) - 4], labels[i][len(uii) - 3], labels[i][len(uii) - 2], labels[i][len(uii) - 1], -100 if len(uii) >= len(labels[i]) else labels[i][len(uii)])
#                 print(attention_mask[i][len(uii) - 4], attention_mask[i][len(uii) - 3], attention_mask[i][len(uii) - 2], attention_mask[i][len(uii) - 1], -100 if len(uii) >= len(attention_mask[i]) else attention_mask[i][len(uii)])
#             index.append(len(uii) + self.last_token)
        
#         index_tensor = torch.tensor(index).to(input_ids.device)
#         if self.debug == 1 and torch.cuda.current_device() == 0:
#             print(index_tensor)

#         return index_tensor
    
#     def _build_additive_mask(self, k: int):
#         mask = torch.zeros((k, k), dtype=torch.float32)
#         mask[torch.triu(torch.ones(k, k), diagonal=1) == 1] = -torch.inf
#         return mask

#     def build_with_additive_mask(self, inputs):
#         if self.jepa_ratio > 0.0:
#             if torch.rand(1).item() > self.jepa_ratio:
#                 return {
#                     "input_ids": inputs["input_ids"],
#                     "labels": inputs["labels"],
#                     "attention_mask": inputs["attention_mask"],
#                 }, True
#         batch_size = inputs["input_ids"].shape[0]
#         seq_length = inputs["input_ids"].shape[-1]
#         device = inputs["input_ids"].device
#         mask = torch.full((batch_size * 2, 1, seq_length, seq_length), -torch.inf).to(device)
#         last_token = self._last_token_index(inputs["input_ids"], inputs["labels"], inputs["attention_mask"])        
#         last_token_user = self._last_token_index(inputs["input_ids_user"], inputs["labels_user"], inputs["attention_mask_user"])
#         last_token_assistant = self._last_token_index(inputs["input_ids_assistant"], inputs["labels_assistant"], inputs["attention_mask_assistant"])
#         for i in range(inputs["input_ids_user"].shape[0]):
#             length, length_user, length_assistant = last_token[i] + 1, last_token_user[i] + 1, last_token_assistant[i] + 1
#             inputs["input_ids_user"][i, length_user:length_user + length_assistant] = inputs["input_ids_assistant"][i, :length_assistant]
#             inputs["labels_user"][i, length_user:length_user + length_assistant] = inputs["labels_assistant"][i, :length_assistant]
#             mask[i, :, 0:length, 0:length] = self._build_additive_mask(length)
#             mask[i + batch_size, :, 0:length_user, 0:length_user] = self._build_additive_mask(length_user)
#             mask[i + batch_size, :, length_user:length_user + length_assistant, length_user:length_user + length_assistant] = self._build_additive_mask(length_assistant)
#         self._last_token_user = last_token_user
#         self._last_token_assistant = last_token_assistant + last_token_user + 1
#         return {
#                 "input_ids": torch.cat([inputs["input_ids"],
#                                         inputs["input_ids_user"]], dim=0),
#                 "labels": torch.cat([inputs["labels"],
#                                     inputs["labels_user"]], dim=0),
#                 "attention_mask": mask,
#             }, False

#     def forward(self, model, inputs):
#         """
#         Custom forward pass that handles all model calls.
#         """
#         # Main forward pass for language modeling
#         if self.additive_mask:
#             llm_inputs, skip_jepa = self.build_with_additive_mask(inputs)
#         else:
#             llm_inputs = {
#                 "input_ids": torch.cat([inputs["input_ids"],
#                                         inputs["input_ids_user"],
#                                         inputs["input_ids_assistant"]], dim=0),
#                 "labels": torch.cat([inputs["labels"],
#                                     inputs["labels_user"],
#                                     inputs["labels_assistant"]], dim=0),
#                 "attention_mask": torch.cat([inputs["attention_mask"],
#                                             inputs["attention_mask_user"],
#                                             inputs["attention_mask_assistant"]], dim=0),
#             }
#         if self.debug == 7 and torch.cuda.current_device() == 0:
#             torch.set_printoptions(threshold=float("inf"))
#             torch.set_printoptions(linewidth=360)
#             print(">>>input_ids<<<")
#             print(llm_inputs["input_ids"])
#             print(">>>labels<<<")
#             print(llm_inputs["labels"])
#             print(">>>attention_mask<<<")
#             print(llm_inputs["attention_mask"])
#             if self.additive_mask:
#                 print(">>>last_token_user<<<")
#                 print(self._last_token_user)
#                 print(">>>last_token_assistant<<<")
#                 print(self._last_token_assistant)
#         if self.debug == 7:
#             exit(0)
#         if self.debug == 2 and torch.cuda.current_device() == 0:
#             print("=====before:outputs=====")
#             print("input_ids shapes:")
#             print(llm_inputs["input_ids"].shape)
#             print("labels shapes::")
#             print(llm_inputs["labels"].shape)
#             print("attention_mask shapes:")
#             print(llm_inputs["attention_mask"].shape)

#         with torch.set_grad_enabled(True):
#             outputs = model(**llm_inputs, output_hidden_states=True)

#         if self.debug == 2 and torch.cuda.current_device() == 0:
#             print(f"=====outputs.loss.shape:{outputs.loss.shape}=====")
#             print(f"=====outputs.hidden_states[-1].shape:{outputs.hidden_states[-1].shape}=====")
        
#         if self.additive_mask:
#             if skip_jepa:
#                 user_hidden_states = None
#                 assistant_hidden_states = None
#             else:    
#                 batch_size = llm_inputs["input_ids"].shape[0] // 2
#                 user_hidden_states = outputs.hidden_states[-1][batch_size: batch_size * 2]
#                 assistant_hidden_states = user_hidden_states
#         else:
#             batch_size = llm_inputs["input_ids"].shape[0] // 3
#             user_hidden_states = outputs.hidden_states[-1][batch_size: batch_size * 2]
#             assistant_hidden_states = outputs.hidden_states[-1][batch_size * 2:]

#         if self.debug == 2 and torch.cuda.current_device() == 0:
#             print(f"====={user_hidden_states.shape}=====")
#             print(f"====={assistant_hidden_states.shape}=====")
       
#         # Return all outputs needed for loss computation
#         return {
#             'main_outputs': outputs,
#             'user_hidden_states': user_hidden_states,
#             'assistant_hidden_states': assistant_hidden_states,
#         }

#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         """
#         Compute loss with additional regularization terms.
#         """
#         # Get indeices
#         if not self.additive_mask:
#             index_user = self._last_token_index(inputs["input_ids_user"], inputs["labels_user"], inputs["attention_mask_user"])
#             index_assistant = self._last_token_index(inputs["input_ids_assistant"], inputs["labels_assistant"], inputs["attention_mask_assistant"])
#         first_dim = inputs["input_ids_user"].shape[0]
#         if self.debug == 1 and torch.cuda.current_device() == 0:
#             print("=====last tokens=====")
#             print(inputs["input_ids_user"][range(first_dim), index_user])
#             print(inputs["input_ids_user"][range(first_dim), index_user - 1])
#             print(inputs["input_ids_assistant"][range(first_dim), index_assistant])
#             print(inputs["input_ids_assistant"][range(first_dim), index_assistant - 1])

#         # Get all forward pass results
#         forward_results = self.forward(model, inputs)
        
#         # Extract main language modeling loss
#         main_outputs = forward_results['main_outputs']
#         lm_loss = main_outputs.loss

#         # Compute representation similarity loss
#         user_hidden_states = forward_results['user_hidden_states']
#         assistant_hidden_states = forward_results['assistant_hidden_states']
        
#         # Get embeddings (using last token of each sequence)
#         if user_hidden_states is not None:
#             if self.additive_mask:
#                 index_user = self._last_token_user
#                 index_assistant = self._last_token_assistant
#             user_embedding = user_hidden_states[range(first_dim), index_user, :]
#             assistant_embedding = assistant_hidden_states[range(first_dim), index_assistant, :]
            
#             # Compute cosine similarity
#             cosine_similarity = F.cosine_similarity(user_embedding, assistant_embedding, dim=-1)
#             if self.debug == 1 and torch.cuda.current_device() == 0:
#                 print(user_embedding.shape, assistant_embedding.shape)
#                 print(cosine_similarity.shape)
    
#             # Compute total loss
#             if self.jepa_l2:
#                 jepa_loss = torch.linalg.norm(user_embedding - assistant_embedding, ord=2, dim=-1).mean()
#             elif self.jepa_mse:
#                 jepa_loss = torch.mean((user_embedding - assistant_embedding) ** 2)
#             elif self.infonce:
#                 ue_norm = F.normalize(user_embedding, p=2, dim=1)
#                 ae_norm = F.normalize(assistant_embedding, p=2, dim=1)
#                 cosine_sim = torch.mm(ue_norm, ae_norm.T)
#                 infonce_logit = cosine_sim / 0.07  # temperature
#                 infonce_label = torch.arange(cosine_sim.size(0), device=cosine_sim.device)
#                 jepa_loss = F.cross_entropy(infonce_logit, infonce_label)
#                 if self.debug == 8:
#                     print(cosine_sim.shape, infonce_logit.shape, infonce_label.shape, jepa_loss.shape)
#                     exit(0)
#             else:
#                 jepa_loss = 1.0 - torch.mean(cosine_similarity)
#         else:
#             jepa_loss = 0.0

#         total_loss = self.gamma * lm_loss + self.lbd * jepa_loss

#         if self.debug == 2 and torch.cuda.current_device() == 0:
#             print(lm_loss, self.lbd, torch.mean(cosine_similarity))

#         if self.debug == 1 or self.debug == 2:
#             exit(0)

#         if self.debug == 5 and torch.cuda.current_device() == 0:
#             print(f"llm_loss: {lm_loss.float()}, jepa_loss: {jepa_loss.float()}")

#         return (total_loss, main_outputs) if return_outputs else total_loss
