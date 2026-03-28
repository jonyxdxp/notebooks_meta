

# (Losses and Regularization ("retention" in Cognitive terms))


# will also be the "Energy Function" for our EBM, since we will be training the EBM to minimize this loss function, so the loss function is effectively the energy function that the EBM is modeling. 
# The EBM will learn to assign low energy (loss) to good reconstructions and high energy (loss) to bad reconstructions, thus learning a landscape of the loss function.




# from https://github.com/facebookresearch/eb_jepa/blob/main/eb_jepa/losses.py










import torch
import torch.nn as nn
import torch.nn.functional as F












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












# ------------------------------------------------------------








# meta-learning loss





# ── Cell S2-2: Complementarity loss ───────────────────────────────────────────

class ComplementarityLoss(nn.Module):
    """
    Two-term loss for Model 2:

      L = L_repr(z2)  +  λ * L_ortho(z1, z2)

    L_repr  : same BCS/VICReg self-supervised loss as Stage 1
              (Model 2 should still form a good representation)

    L_ortho : penalizes alignment between Model 1 and Model 2 embeddings.
              We want z2 to explain variance NOT already in z1's subspace.
              Implemented as: λ * ||z1_norm · z2_norm^T||_F^2  (Frobenius)
              i.e. punish any batch-level cosine similarity between the two.
    """
    def __init__(self, repr_loss_fn, ortho_coeff: float = 1.0):
        super().__init__()
        self.repr_loss_fn = repr_loss_fn   # reuse BCS from Stage 1
        self.ortho_coeff  = ortho_coeff

    def forward(
        self,
        z2_ctx: torch.Tensor,   # (B, D) Model 2 context embedding
        z2_tgt: torch.Tensor,   # (B, D) Model 2 target embedding  (EMA)
        z1_ctx: torch.Tensor,   # (B, D) Model 1 context embedding (frozen, no grad)
    ):
        # ── representation loss (same objective as Stage 1) ──────────────────
        repr_dict = self.repr_loss_fn(z2_ctx, z2_tgt)

        # ── orthogonality penalty ─────────────────────────────────────────────
        z2_n = F.normalize(z2_ctx, dim=-1)          # (B, D)
        z1_n = F.normalize(z1_ctx, dim=-1)          # (B, D)
        # Cross-covariance between Model1 and Model2 embeddings in this batch
        cross_cov = (z1_n.T @ z2_n) / z2_n.shape[0]   # (D, D)
        ortho_loss = cross_cov.pow(2).sum()

        total = repr_dict['loss'] + self.ortho_coeff * ortho_loss

        return {
            'loss':       total,
            'repr_loss':  repr_dict['loss'].detach(),
            'ortho_loss': ortho_loss.detach(),
            **{k: v for k, v in repr_dict.items() if k != 'loss'},
        }















# ----------------------------------------------------------



# from https://github.com/galilai-group/llm-jepa/blob/main/finetune.py











class RepresentationTrainer(Trainer):
    """
    Trainer to regularize representations.
    """
    
    def __init__(self, *args, **kwargs):
        # Extract custom loss parameters
        self.lbd = kwargs.pop('lbd', 1.0)
        self.gamma = kwargs.pop('gamma', 1.0)
        self.last_token = kwargs.pop('last_token', -2)
        self.debug = kwargs.pop('debug', 0)
        self.additive_mask = kwargs.pop('additive_mask', False)
        self.jepa_l2 = kwargs.pop('jepa_l2', False)
        self.jepa_mse = kwargs.pop('jepa_mse', False)
        self.infonce = kwargs.pop('infonce', False)
        self.jepa_ratio = kwargs.pop('jepa_ratio', -1.0)
        assert self.jepa_l2 + self.jepa_mse <= 1, "Only one of jepa_l2 and jepa_mse can be True."
        super().__init__(*args, **kwargs)
    
    def _last_token_index(self, input_ids, labels, attention_mask):
        index = []
        def unpad(input_ids, attention_mask):
            result = []
            can_break = False
            for id, mask in zip(input_ids, attention_mask):
                if mask != 0:
                    can_break = True
                if mask == 0 and can_break:
                    break
                result.append(id)
            return result

        for i in range(input_ids.shape[0]):
            uii = unpad(input_ids[i], attention_mask[i])
            if self.debug == 1 and torch.cuda.current_device() == 0:
                print(f"====={len(uii)}=====")
                print(input_ids[i][len(uii) - 4], input_ids[i][len(uii) - 3], input_ids[i][len(uii) - 2], input_ids[i][len(uii) - 1], -100 if len(uii) >= len(input_ids[i]) else input_ids[i][len(uii)])
                print(labels[i][len(uii) - 4], labels[i][len(uii) - 3], labels[i][len(uii) - 2], labels[i][len(uii) - 1], -100 if len(uii) >= len(labels[i]) else labels[i][len(uii)])
                print(attention_mask[i][len(uii) - 4], attention_mask[i][len(uii) - 3], attention_mask[i][len(uii) - 2], attention_mask[i][len(uii) - 1], -100 if len(uii) >= len(attention_mask[i]) else attention_mask[i][len(uii)])
            index.append(len(uii) + self.last_token)
        
        index_tensor = torch.tensor(index).to(input_ids.device)
        if self.debug == 1 and torch.cuda.current_device() == 0:
            print(index_tensor)

        return index_tensor
    
    def _build_additive_mask(self, k: int):
        mask = torch.zeros((k, k), dtype=torch.float32)
        mask[torch.triu(torch.ones(k, k), diagonal=1) == 1] = -torch.inf
        return mask

    def build_with_additive_mask(self, inputs):
        if self.jepa_ratio > 0.0:
            if torch.rand(1).item() > self.jepa_ratio:
                return {
                    "input_ids": inputs["input_ids"],
                    "labels": inputs["labels"],
                    "attention_mask": inputs["attention_mask"],
                }, True
        batch_size = inputs["input_ids"].shape[0]
        seq_length = inputs["input_ids"].shape[-1]
        device = inputs["input_ids"].device
        mask = torch.full((batch_size * 2, 1, seq_length, seq_length), -torch.inf).to(device)
        last_token = self._last_token_index(inputs["input_ids"], inputs["labels"], inputs["attention_mask"])        
        last_token_user = self._last_token_index(inputs["input_ids_user"], inputs["labels_user"], inputs["attention_mask_user"])
        last_token_assistant = self._last_token_index(inputs["input_ids_assistant"], inputs["labels_assistant"], inputs["attention_mask_assistant"])
        for i in range(inputs["input_ids_user"].shape[0]):
            length, length_user, length_assistant = last_token[i] + 1, last_token_user[i] + 1, last_token_assistant[i] + 1
            inputs["input_ids_user"][i, length_user:length_user + length_assistant] = inputs["input_ids_assistant"][i, :length_assistant]
            inputs["labels_user"][i, length_user:length_user + length_assistant] = inputs["labels_assistant"][i, :length_assistant]
            mask[i, :, 0:length, 0:length] = self._build_additive_mask(length)
            mask[i + batch_size, :, 0:length_user, 0:length_user] = self._build_additive_mask(length_user)
            mask[i + batch_size, :, length_user:length_user + length_assistant, length_user:length_user + length_assistant] = self._build_additive_mask(length_assistant)
        self._last_token_user = last_token_user
        self._last_token_assistant = last_token_assistant + last_token_user + 1
        return {
                "input_ids": torch.cat([inputs["input_ids"],
                                        inputs["input_ids_user"]], dim=0),
                "labels": torch.cat([inputs["labels"],
                                    inputs["labels_user"]], dim=0),
                "attention_mask": mask,
            }, False

    def forward(self, model, inputs):
        """
        Custom forward pass that handles all model calls.
        """
        # Main forward pass for language modeling
        if self.additive_mask:
            llm_inputs, skip_jepa = self.build_with_additive_mask(inputs)
        else:
            llm_inputs = {
                "input_ids": torch.cat([inputs["input_ids"],
                                        inputs["input_ids_user"],
                                        inputs["input_ids_assistant"]], dim=0),
                "labels": torch.cat([inputs["labels"],
                                    inputs["labels_user"],
                                    inputs["labels_assistant"]], dim=0),
                "attention_mask": torch.cat([inputs["attention_mask"],
                                            inputs["attention_mask_user"],
                                            inputs["attention_mask_assistant"]], dim=0),
            }
        if self.debug == 7 and torch.cuda.current_device() == 0:
            torch.set_printoptions(threshold=float("inf"))
            torch.set_printoptions(linewidth=360)
            print(">>>input_ids<<<")
            print(llm_inputs["input_ids"])
            print(">>>labels<<<")
            print(llm_inputs["labels"])
            print(">>>attention_mask<<<")
            print(llm_inputs["attention_mask"])
            if self.additive_mask:
                print(">>>last_token_user<<<")
                print(self._last_token_user)
                print(">>>last_token_assistant<<<")
                print(self._last_token_assistant)
        if self.debug == 7:
            exit(0)
        if self.debug == 2 and torch.cuda.current_device() == 0:
            print("=====before:outputs=====")
            print("input_ids shapes:")
            print(llm_inputs["input_ids"].shape)
            print("labels shapes::")
            print(llm_inputs["labels"].shape)
            print("attention_mask shapes:")
            print(llm_inputs["attention_mask"].shape)

        with torch.set_grad_enabled(True):
            outputs = model(**llm_inputs, output_hidden_states=True)

        if self.debug == 2 and torch.cuda.current_device() == 0:
            print(f"=====outputs.loss.shape:{outputs.loss.shape}=====")
            print(f"=====outputs.hidden_states[-1].shape:{outputs.hidden_states[-1].shape}=====")
        
        if self.additive_mask:
            if skip_jepa:
                user_hidden_states = None
                assistant_hidden_states = None
            else:    
                batch_size = llm_inputs["input_ids"].shape[0] // 2
                user_hidden_states = outputs.hidden_states[-1][batch_size: batch_size * 2]
                assistant_hidden_states = user_hidden_states
        else:
            batch_size = llm_inputs["input_ids"].shape[0] // 3
            user_hidden_states = outputs.hidden_states[-1][batch_size: batch_size * 2]
            assistant_hidden_states = outputs.hidden_states[-1][batch_size * 2:]

        if self.debug == 2 and torch.cuda.current_device() == 0:
            print(f"====={user_hidden_states.shape}=====")
            print(f"====={assistant_hidden_states.shape}=====")
       
        # Return all outputs needed for loss computation
        return {
            'main_outputs': outputs,
            'user_hidden_states': user_hidden_states,
            'assistant_hidden_states': assistant_hidden_states,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss with additional regularization terms.
        """
        # Get indeices
        if not self.additive_mask:
            index_user = self._last_token_index(inputs["input_ids_user"], inputs["labels_user"], inputs["attention_mask_user"])
            index_assistant = self._last_token_index(inputs["input_ids_assistant"], inputs["labels_assistant"], inputs["attention_mask_assistant"])
        first_dim = inputs["input_ids_user"].shape[0]
        if self.debug == 1 and torch.cuda.current_device() == 0:
            print("=====last tokens=====")
            print(inputs["input_ids_user"][range(first_dim), index_user])
            print(inputs["input_ids_user"][range(first_dim), index_user - 1])
            print(inputs["input_ids_assistant"][range(first_dim), index_assistant])
            print(inputs["input_ids_assistant"][range(first_dim), index_assistant - 1])

        # Get all forward pass results
        forward_results = self.forward(model, inputs)
        
        # Extract main language modeling loss
        main_outputs = forward_results['main_outputs']
        lm_loss = main_outputs.loss

        # Compute representation similarity loss
        user_hidden_states = forward_results['user_hidden_states']
        assistant_hidden_states = forward_results['assistant_hidden_states']
        
        # Get embeddings (using last token of each sequence)
        if user_hidden_states is not None:
            if self.additive_mask:
                index_user = self._last_token_user
                index_assistant = self._last_token_assistant
            user_embedding = user_hidden_states[range(first_dim), index_user, :]
            assistant_embedding = assistant_hidden_states[range(first_dim), index_assistant, :]
            
            # Compute cosine similarity
            cosine_similarity = F.cosine_similarity(user_embedding, assistant_embedding, dim=-1)
            if self.debug == 1 and torch.cuda.current_device() == 0:
                print(user_embedding.shape, assistant_embedding.shape)
                print(cosine_similarity.shape)
    
            # Compute total loss
            if self.jepa_l2:
                jepa_loss = torch.linalg.norm(user_embedding - assistant_embedding, ord=2, dim=-1).mean()
            elif self.jepa_mse:
                jepa_loss = torch.mean((user_embedding - assistant_embedding) ** 2)
            elif self.infonce:
                ue_norm = F.normalize(user_embedding, p=2, dim=1)
                ae_norm = F.normalize(assistant_embedding, p=2, dim=1)
                cosine_sim = torch.mm(ue_norm, ae_norm.T)
                infonce_logit = cosine_sim / 0.07  # temperature
                infonce_label = torch.arange(cosine_sim.size(0), device=cosine_sim.device)
                jepa_loss = F.cross_entropy(infonce_logit, infonce_label)
                if self.debug == 8:
                    print(cosine_sim.shape, infonce_logit.shape, infonce_label.shape, jepa_loss.shape)
                    exit(0)
            else:
                jepa_loss = 1.0 - torch.mean(cosine_similarity)
        else:
            jepa_loss = 0.0

        total_loss = self.gamma * lm_loss + self.lbd * jepa_loss

        if self.debug == 2 and torch.cuda.current_device() == 0:
            print(lm_loss, self.lbd, torch.mean(cosine_similarity))

        if self.debug == 1 or self.debug == 2:
            exit(0)

        if self.debug == 5 and torch.cuda.current_device() == 0:
            print(f"llm_loss: {lm_loss.float()}, jepa_loss: {jepa_loss.float()}")

        return (total_loss, main_outputs) if return_outputs else total_loss






