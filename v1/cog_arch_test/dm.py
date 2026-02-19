





# from https://github.com/alexiglad/EBT/blob/main/model/ar_ebt_default.py







# (setup_ebt)










class EBTDefault(nn.Module):
    def __init__(self, params: EBTModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (EBTModelArgs): Model configuration parameters.

        Attributes:
            params (EBTModelArgs): Model configuration parameters.
            n_layers (int): Number of layers in the model.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len
        )

        self.final_layer = nn.Linear(params.dim, 1, bias = False)
        init_whole_model_weights(self.final_layer, self.params.weight_initialization)


    def forward(self, embeddings: torch.Tensor, start_pos: int, mcmc_step = None):  # NOTE mcmc_step not used here
        """
        Perform a forward pass through the Transformer model.

        Args:
            embeds (torch.Tensor): Embeddings (instead of tokens since is for vision).
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output energies after applying the Transformer model.

        """
        _bsz, seqlen = embeddings.shape[:2]
        seqlen = (seqlen+2) // 2 # do this since passed in seqlen is 2(S-1) so add 2 div 2 = S
        self.freqs_cis = self.freqs_cis.to(embeddings.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=embeddings.device
            )

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=embeddings.device),
                mask
            ]).type_as(embeddings)
            # causal mask is like this by default 0, -inf, -inf
            #                         0, 0,    -inf
            #                         0, 0,    0
                


            for i, layer in enumerate(self.layers):
                embeddings = layer(embeddings, start_pos, freqs_cis, mask)
            embeddings = self.norm(embeddings)

            energies = self.final_layer(embeddings)

            energies = energies[:, embeddings.shape[1] // 2:]
            return energies
        



































# from https://github.com/alexiglad/EBT/blob/main/model/nlp/ebt.py










import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as L
import torch.optim as optim
from torchmetrics import Accuracy

from transformers import AutoTokenizer

import math
import random
import os
from model.model_utils import *
from model.replay_buffer import CausalReplayBuffer





class EBT_NLP(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):#passed in from model ckpt
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))
        
        tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer, clean_up_tokenization_spaces = False)
        self.tokenizer_pad_token_id = tokenizer.eos_token_id # is token 0, was right padding things
        
        self.vocab_size = len(tokenizer) # self.vocab_size = self.tokenizer.vocab_size caused errors since is smaller than len(self.tokenizer), is 50254 for neox-20b, len tokenizer is 50277 so decided to use that
        
        self.alpha = nn.Parameter(torch.tensor(float(self.hparams.mcmc_step_size)), requires_grad=self.hparams.mcmc_step_size_learnable)
        self.langevin_dynamics_noise_std = nn.Parameter(torch.tensor(float(self.hparams.langevin_dynamics_noise)), requires_grad=False) # if using self.hparams.langevin_dynamics_noise_learnable this will be turned on in warm_up_finished func

        self.embeddings = nn.Embedding(self.vocab_size, self.hparams.embedding_dim)
        init_whole_model_weights(self.embeddings, self.hparams.weight_initialization_method, weight_initialization_gain=self.hparams.weight_initialization_gain)
        
        self.log_softmax = nn.LogSoftmax(dim = -1)
        self.softmax = nn.Softmax(dim = -1)
        
        if not self.hparams.vocab_to_embed_uses_prob_dist: # if are not using the prob dist * embed as vocab to embed
            if 'learnable_process_memory' in self.hparams and self.hparams.learnable_process_memory and self.hparams.process_memory_type != None:
                self.vocab_to_embed = Memory_Gating_MLP(self.vocab_size, self.hparams.embedding_dim, self.hparams.process_memory_type, self.hparams.process_memory_linear_layer)
            elif 'learnable_process_memory' in self.hparams and self.hparams.learnable_process_memory:
                assert self.hparams.num_modality_processing_mlp_layers > 1, "must set self.hparams.num_modality_processing_mlp_layers > 1 if not using self.hparams.process_memory_type"
                self.vocab_to_embed = Memory_Augmented_MLP(self.vocab_size, self.hparams.embedding_dim, self.hparams.embedding_dim, self.hparams.embedding_dim, dropout_rate=0, layer_norm=True, num_hidden_layers = self.hparams.num_modality_processing_mlp_layers)
            elif self.hparams.num_modality_processing_mlp_layers != 1:
                self.vocab_to_embed = MLP(self.vocab_size, self.hparams.embedding_dim, self.hparams.embedding_dim, dropout_rate=0, layer_norm=True, num_hidden_layers = self.hparams.num_modality_processing_mlp_layers - 2)
            else:
                self.vocab_to_embed = nn.Linear(self.vocab_size, self.hparams.embedding_dim, bias = False, device = self.device) #NOTE this is ebt special, since we want to input a prob dist and pred this prob dist but the transformer needs an embedding as input
            init_whole_model_weights(self.vocab_to_embed, self.hparams.weight_initialization_method, weight_initialization_gain=self.hparams.weight_initialization_gain)



        self.transformer = setup_ebt(self.hparams)
        
        self.finished_warming_up = False

        self.mcmc_replay_buffer = 'mcmc_replay_buffer' in self.hparams and self.hparams.mcmc_replay_buffer and self.hparams.execution_mode != "inference"
        if self.mcmc_replay_buffer:
            replay_buffer_max_size = self.hparams.mcmc_replay_buffer_size
            self.replay_buffer_samples = self.hparams.batch_size_per_device * self.hparams.mcmc_replay_buffer_sample_bs_percent
            self.replay_buffer = CausalReplayBuffer(max_size=replay_buffer_max_size, sample_size=self.replay_buffer_samples)

        # DEBUGGING CODE ################################################################################################################################################
        if self.hparams.debug_unused_parameters:
            self.used_parameters = set()
            self.parameters_not_to_check = set() # dont check these since may be frozen or dont want them to update
        
    def forward(self, x, start_pos = 0, learning = True, return_raw_logits = False, replay_buffer_logits = None, no_randomness = True): # accepts input_ids as input; a lot of the logic here is just for S2 params, see pseudocode in paper for a more concise view of how this works. it can be < 10 LOC
        predicted_distributions = []
        predicted_energies = []

        real_embeddings_input = self.embeddings(x)
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        
        alpha = torch.clamp(self.alpha, min=0.0001)
        if not no_randomness and self.hparams.randomize_mcmc_step_size_scale != 1:
            expanded_alpha = alpha.expand(batch_size, seq_length, 1)

            scale = self.hparams.randomize_mcmc_step_size_scale
            low = alpha / scale
            high = alpha * scale
            alpha = low + torch.rand_like(expanded_alpha) * (high - low)

        langevin_dynamics_noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)

        predicted_tokens = self.corrupt_embeddings(real_embeddings_input) # B, S, V
        if replay_buffer_logits is not None: # using replay buffer, use the logits instead of corruption
            predicted_tokens[batch_size - replay_buffer_logits.shape[0]:] = replay_buffer_logits # NOTE this assumes the fresh data is concatted first
                
        
        mcmc_steps = [] # in the general case of no randomize_mcmc_num_steps then this has len == self.hparams.randomize_mcmc_num_steps
        for step in range(self.hparams.mcmc_num_steps):
            if not no_randomness and hasattr(self.hparams, 'randomize_mcmc_num_steps') and self.hparams.randomize_mcmc_num_steps > 0:
                if self.hparams.randomize_mcmc_num_steps_final_landscape: # makes so only applies rand steps to final landscape
                    if step == (self.hparams.mcmc_num_steps - 1):
                        min_steps = 1 if self.hparams.randomize_mcmc_num_steps_min == 0 else self.hparams.randomize_mcmc_num_steps_min
                        repeats = torch.randint(min_steps, self.hparams.randomize_mcmc_num_steps + 2, (1,)).item()
                        mcmc_steps.extend([step] * repeats)
                    else:
                        mcmc_steps.append(step)
                else:
                    min_steps = 1 if self.hparams.randomize_mcmc_num_steps_min == 0 else self.hparams.randomize_mcmc_num_steps_min
                    repeats = torch.randint(min_steps, self.hparams.randomize_mcmc_num_steps + 2, (1,)).item()
                    mcmc_steps.extend([step] * repeats)
            elif no_randomness and hasattr(self.hparams, 'randomize_mcmc_num_steps') and self.hparams.randomize_mcmc_num_steps > 0: # use max steps
                if step == (self.hparams.mcmc_num_steps - 1): # i found this was a better pretraining metric and was more stable, only do several steps on final energy landscape instead of over all energy landscapes
                    mcmc_steps.extend([step] * (self.hparams.randomize_mcmc_num_steps + 1))
                else:
                    mcmc_steps.append(step)
            else:
                mcmc_steps.append(step)

        with torch.set_grad_enabled(True):
            for i, mcmc_step in enumerate(mcmc_steps):
                
                if self.hparams.no_mcmc_detach:
                    predicted_tokens.requires_grad_().reshape(batch_size, seq_length, self.vocab_size) # B, S, V
                else: # default, do detach
                    predicted_tokens = predicted_tokens.detach().requires_grad_().reshape(batch_size, seq_length, self.vocab_size) # B, S, V

                if self.hparams.langevin_dynamics_noise != 0:
                    ld_noise = torch.randn_like(predicted_tokens.detach()) * langevin_dynamics_noise_std # langevin dynamics
                    predicted_tokens = predicted_tokens + ld_noise

                if self.hparams.normalize_initial_condition:
                    if self.hparams.normalize_initial_condition_only_first_step:
                        if mcmc_step == 0:
                            predicted_tokens = self.softmax(predicted_tokens)
                    else:
                        predicted_tokens = self.softmax(predicted_tokens)
                        
                    if self.hparams.vocab_to_embed_uses_prob_dist: # predicted_embeds is B, S, V; embed is V, D
                        predicted_embeddings = torch.matmul(predicted_tokens, self.embeddings.weight) #BS, S, D
                    else:
                        predicted_embeddings = self.vocab_to_embed(predicted_tokens) #BS, S, D
                else:
                    predicted_embeddings = self.vocab_to_embed(predicted_tokens) #BS, S, D
                
                all_embeddings = torch.cat((real_embeddings_input, predicted_embeddings), dim = 1) # B, 2*S, D
                
                energy_preds = self.transformer(all_embeddings, start_pos = start_pos, mcmc_step=mcmc_step) # is B, 2*S, D; checked and there are no in place ops; mcmc_step only applies to when using certain types of ebt
                energy_preds = energy_preds.reshape(-1, 1)
                predicted_energies.append(energy_preds)
                
                if self.hparams.truncate_mcmc:  #retain_graph defaults to create_graph value here; if learning is true then create_graph else dont (inference)
                    if i == (len(mcmc_steps) - 1):
                        predicted_tokens_grad = torch.autograd.grad([energy_preds.sum()], [predicted_tokens], create_graph=learning)[0]
                    else:
                        predicted_tokens_grad = torch.autograd.grad([energy_preds.sum()], [predicted_tokens], create_graph=False)[0]
                else:
                    predicted_tokens_grad = torch.autograd.grad([energy_preds.sum()], [predicted_tokens], create_graph=learning)[0]
                # predicted_tokens_grad has shape B, S, V
                
                if self.hparams.clamp_futures_grad:
                    min_and_max = self.hparams.clamp_futures_grad_max_change / (self.alpha) # use self.alpha and not random alpha to clamp
                    # predicted_tokens_grad = scale_clamp(predicted_tokens_grad, -min_and_max, min_and_max)
                    predicted_tokens_grad = torch.clamp(predicted_tokens_grad, min = -min_and_max, max = min_and_max)
                    
                if torch.isnan(predicted_tokens_grad).any() or torch.isinf(predicted_tokens_grad).any():
                    raise ValueError("NaN or Inf gradients detected during MCMC.")
                
                predicted_tokens = predicted_tokens - alpha * predicted_tokens_grad # do this to tokens will be unnormalize prob dist convert to prob dist after  
                
                if self.hparams.absolute_clamp != 0.0:
                    predicted_tokens = torch.clamp(predicted_tokens, min = -self.hparams.absolute_clamp, max = self.hparams.absolute_clamp)
                
                if self.hparams.sharpen_predicted_distribution != 0.0:
                    predicted_tokens = predicted_tokens / self.hparams.sharpen_predicted_distribution

                if return_raw_logits:
                    predicted_tokens_for_loss = predicted_tokens # BS, S, V
                else:
                    predicted_tokens_for_loss = self.log_softmax(predicted_tokens).reshape(-1, self.vocab_size) # BS*S, V
                predicted_distributions.append(predicted_tokens_for_loss)        

        return predicted_distributions, predicted_energies
    












# --------------------------------------








# see nanoEBM



# from https://github.com/sdan/nanoEBM/blob/master/nanoebm/model.py










"""
EBM - Minimal Energy-Based Model

Energy-Based Model for language, its based on:
1. Yann LeCun's lecture on Energy-Based Models: https://atcold.github.io/NYU-DLSP20/en/week07/07-1/
2. Stefano Ermon's CS236 Lecture 11: https://deepgenerativemodels.github.io/assets/slides/cs236_lecture11.pdf

The model learns an energy function E(x, y) where low energy = good text.

- System 1: Regular forward pass
- System 2: Gradient descent on logits to minimize energy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import random
from contextlib import nullcontext

from .transformer import Transformer
from .config import ModelConfig









class EBM(nn.Module):
    """Energy-Based Model from https://atcold.github.io/NYU-DLSP20/en/week07/07-1/"""    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        self.transformer = Transformer(config)
        
        # This linear layer defines our energy function
        # E(hidden_state, token) = -hidden_state @ W[token]
        # NOTE: No weight tying - energy head needs its own parameters for proper energy landscape
        self.energy_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # https://alexiglad.github.io/blog/2025/ebt/ uses fixed step size for gradient descent as one of several stability tricks
        self.register_buffer('alpha', torch.tensor(config.alpha_value))
        
        # Initialize weights (energy head gets special initialization)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for energy head with larger scale for meaningful gradients."""
        if isinstance(module, nn.Linear) and module == self.energy_head:
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.1)
    
    def get_hidden_states(self, idx_or_soft: torch.Tensor) -> torch.Tensor:
        """
        Extract hidden states from transformer.

        Args:
            idx_or_soft: Either hard tokens (B, T) or soft embeddings (B, T, n_embd)

        Returns:
            Hidden states (B, T, n_embd)
        """
        device = idx_or_soft.device

        # Check if hard tokens or soft embeddings based on dimensionality
        if idx_or_soft.dim() == 2:
            # Hard tokens (B, T)
            t = idx_or_soft.size(1)
            tok_emb = self.transformer.transformer.wte(idx_or_soft)
        else:
            # Soft embeddings (B, T, n_embd)
            t = idx_or_soft.size(1)
            tok_emb = idx_or_soft

        assert t <= self.config.block_size

        # Add positional embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        pos_emb = self.transformer.transformer.wpe(pos)
        x = self.transformer.transformer.drop(tok_emb + pos_emb)

        # Pass through transformer blocks
        for block in self.transformer.transformer.h:
            x = block(x)
        x = self.transformer.transformer.ln_f(x)

        return x  # (B, T, n_embd)
    
    def system1_direct_energy(self, idx: torch.Tensor) -> torch.Tensor:
        """
        System 1: Regular forward pass.
        
        Returns: logits where logit = -energy (high logit = low energy = good)
        """
        h = self.get_hidden_states(idx)  # (B, T, n_embd)
        energy = self.energy_head(h)  # (B, T, V)
        return -energy  # Flip sign: low energy should have high logit
    
    def system2_refine(
        self,
        idx: torch.Tensor,
        steps: int = None,
        return_trajectory: bool = False,
        detach_hidden: bool = False,
        use_soft_tokens: bool = False
    ) -> torch.Tensor:
        """
        System 2: Gradient descent on the logits to minimize expected energy.

        The objective is to minimize E_p[E(x,y)] where p = softmax(logits)
        and E(x,y) comes from the learned energy head.

        Args:
            idx: Input tokens
            steps: Number of refinement steps (None = use config/random)
            return_trajectory: Whether to return intermediate logits
            detach_hidden: Whether to detach hidden states (for stable early training)
            use_soft_tokens: If True, recompute energies each step from soft embeddings (context shifts).
                           If False, compute energies once from hard tokens (frozen context, default).
        """
        # Ensure gradients are enabled for refinement even if called under no_grad
        grad_ctx = torch.enable_grad() if not torch.is_grad_enabled() else nullcontext()
        with grad_ctx:
            B, T = idx.shape
            V = self.config.vocab_size
            device = idx.device

            # Determine number of steps
            if steps is None:
                if self.training:
                    steps = random.randint(2, 3)
                else:
                    steps = self.config.refine_steps

            # Get initial hidden states and energy values
            h = self.get_hidden_states(idx)  # (B, T, n_embd)
            if detach_hidden:
                h = h.detach()

            energies = self.energy_head(h)  # (B, T, V)

            # Initialize logits from System 1
            logits = -energies.clone()  # S1 initialization
            logits = logits.requires_grad_(True)

            trajectory = [logits.clone()] if return_trajectory else []

            # Get embedding matrix for soft tokens if needed
            embedding_matrix = self.transformer.transformer.wte.weight if use_soft_tokens else None

            # Track energy for early stopping
            prev_energy = None
            early_stop_patience = 0

            # Gradient descent loop
            for step in range(steps):
                # Current probability distribution
                probs = F.softmax(logits, dim=-1)  # (B, T, V)

                # Recompute energies from soft embeddings if using soft tokens
                if use_soft_tokens:
                    # Compute soft embeddings: weighted average of token embeddings
                    # soft_emb[b, t, :] = sum_v probs[b, t, v] * embedding_matrix[v, :]
                    soft_embeddings = torch.einsum('btv,ve->bte', probs, embedding_matrix)  # (B, T, n_embd)

                    # Recompute hidden states from soft embeddings
                    h_soft = self.get_hidden_states(soft_embeddings)

                    # Recompute energies (energy landscape shifts with changing context)
                    energies = self.energy_head(h_soft)  # (B, T, V)

                # Expected energy under current distribution: E_p[E(x,y)]
                # This is what we minimize in EBM
                expected_energy = (probs * energies).sum(dim=-1).mean()

                # Compute gradient of expected energy w.r.t. logits
                # Using autograd for second-order gradients during training
                grad = torch.autograd.grad(
                    expected_energy,  # Minimize expected energy
                    logits,
                    create_graph=self.training
                )[0]

                # Step size
                step_size = self.alpha
                if self.training:
                    # Small jitter during training
                    jitter = 1.0 + 0.1 * (torch.rand(1, device=device) - 0.5)  # [0.95, 1.05]
                    step_size = step_size * jitter

                # Gradient descent step
                logits = logits - step_size * grad.clamp(-5, 5)

                # Add Langevin noise for exploration (only during training)
                if self.config.langevin_noise > 0 and self.training and step < steps - 1:
                    noise_scale = self.config.langevin_noise * (1.0 - step / steps)  # Decay noise
                    logits = logits + noise_scale * torch.randn_like(logits)

                # Center logits for numerical stability
                logits = logits - logits.mean(dim=-1, keepdim=True)

                # Prepare for next iteration
                logits = logits.requires_grad_(True)

                if return_trajectory:
                    trajectory.append(logits.clone())

                # Early stopping based on energy convergence
                current_energy = expected_energy.detach().item()
                if prev_energy is not None:
                    energy_change = abs(prev_energy - current_energy)
                    if energy_change < self.config.energy_convergence_threshold:
                        early_stop_patience += 1
                        if early_stop_patience >= 2:
                            break
                    else:
                        early_stop_patience = 0
                prev_energy = current_energy

        # Detach outputs in eval mode to avoid holding graphs
        if not self.training:
            logits = logits.detach()
            if return_trajectory:
                trajectory = [t.detach() for t in trajectory]

        if return_trajectory:
            return logits, trajectory
        return logits
    




    
    def forward(
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor] = None,
        use_refine: bool = True,
        refine_steps: int = 2
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """
        Forward pass with optional refinement.
        
        Properly computes energy using the energy head, not from logits.
        
        Args:
            idx: Input tokens (B, T)
            targets: Target tokens for loss computation (B, T)
            use_refine: Whether to use System 2 refinement
            refine_steps: Number of refinement steps
        
        Returns: (loss, logits, metrics)
        """
        metrics = {}
        
        # Get hidden states and energies (same for both S1 and S2)
        h = self.get_hidden_states(idx)
        energies = self.energy_head(h)  # (B, T, V)
        
        # System 1: Direct readout
        logits_s1 = -energies
        probs_s1 = F.softmax(logits_s1, dim=-1)
        EE_s1 = (probs_s1 * energies).sum(dim=-1).mean()  # Expected energy S1
        
        if use_refine:
            # System 2: Refined prediction
            logits_s2 = self.system2_refine(idx, steps=refine_steps, use_soft_tokens=self.config.use_soft_tokens)
            probs_s2 = F.softmax(logits_s2, dim=-1)
            EE_s2 = (probs_s2 * energies).sum(dim=-1).mean()  # Expected energy S2
            logits = logits_s2
            
            # Track energy metrics (using names expected by train.py)
            metrics['initial_energy'] = EE_s1.item()  # E0: System 1 energy
            metrics['final_energy'] = EE_s2.item()    # EK: System 2 energy after refinement
            metrics['energy_gap'] = (EE_s1 - EE_s2).item()  # Should be positive (improvement from thinking)
        else:
            logits = logits_s1
            metrics['EE_s1'] = EE_s1.item()
            metrics['EE_s2'] = EE_s1.item()
            metrics['delta_EE'] = 0.0
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Compute NLL for both S1 and S2
            nll_s1 = F.cross_entropy(
                logits_s1.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1
            )
            
            if use_refine:
                nll_s2 = F.cross_entropy(
                    logits_s2.view(-1, self.config.vocab_size),
                    targets.view(-1),
                    ignore_index=-1
                )
                loss = nll_s2
                
                # Track NLL metrics
                ppl_s2 = torch.exp(nll_s2).item()
                metrics.update({
                    'nll_s1': nll_s1.item(),
                    'nll_s2': nll_s2.item(),
                    'ppl_s1': torch.exp(nll_s1).item(),
                    'ppl_s2': ppl_s2,
                    'perplexity': ppl_s2,  # Main perplexity metric for logging
                    'delta_nll': (nll_s1 - nll_s2).item(),  # Should be positive
                })
                
            else:
                loss = nll_s1
                metrics.update({
                    'nll_s1': nll_s1.item(),
                    'ppl_s1': torch.exp(nll_s1).item(),
                    'perplexity': torch.exp(nll_s1).item(),
                })

            # Optional: in-batch InfoNCE over sequence energies (conditional EBM contrast)
            if getattr(self.config, 'info_nce_weight', 0.0) > 0.0:
                # energies: (B, T, V) from current batch contexts (idx)
                # Build pairwise energy matrix E_ij = sum_t E(x_i, y_j[t]) using gather without extra forwards
                B, T = idx.shape
                device = idx.device
                # Prepare for broadcasting: (B, 1, T, V) and (1, B, T, 1)
                energies_b = energies.unsqueeze(1).expand(B, B, T, energies.size(-1))
                targets_b = targets.unsqueeze(0).expand(B, B, T)
                # Gather energies for each pair (i,j) across vocab axis
                pairwise_token_E = torch.gather(energies_b, dim=-1, index=targets_b.unsqueeze(-1)).squeeze(-1)  # (B,B,T)
                # Sum over time to get sequence energy; normalize by T for scale stability
                E_mat = pairwise_token_E.sum(dim=-1) / float(T)  # (B,B)
                # Convert to scores (higher is better) and apply temperature
                tau = max(1e-6, float(getattr(self.config, 'info_nce_temperature', 1.0)))
                S = (-E_mat) / tau  # (B,B)
                targets_inbatch = torch.arange(B, device=device)
                info_nce_loss = F.cross_entropy(S, targets_inbatch)
                # Combine
                weight = float(self.config.info_nce_weight)
                loss = loss + weight * info_nce_loss if loss is not None else weight * info_nce_loss
                # Metrics
                with torch.no_grad():
                    acc = (S.argmax(dim=1) == targets_inbatch).float().mean().item()
                metrics.update({
                    'info_nce_loss': info_nce_loss.item(),
                    'info_nce_acc': acc,
                })

        return loss, logits, metrics
    





    
    @torch.no_grad()
    def compute_energy(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Compute raw energy for contrastive training.
        This returns the actual energy (not negative) for use in contrastive divergence.
        
        Args:
            idx: Input tokens (B, T)
            
        Returns:
            Energy values (B,) - one energy per sequence
        """
        # Get hidden states
        h = self.get_hidden_states(idx)  # (B, T, n_embd)
        
        # Get energy values for all tokens
        energies = self.energy_head(h)  # (B, T, V)
        
        # For contrastive training, we need the energy of the actual tokens
        # Gather the energy of the target tokens
        B, T = idx.shape
        
        # Get energy of actual tokens in the sequence
        token_energies = energies.gather(2, idx.unsqueeze(-1)).squeeze(-1)  # (B, T)
        
        # Sum over sequence length to get total energy per sample
        total_energy = token_energies.sum(dim=1)  # (B,)
        
        return total_energy
    
    def compute_energy_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute expected energy from logits for contrastive training.
        
        Args:
            logits: Logit values (B, T, V)
            
        Returns:
            Expected energy (B,) - one energy per sequence
        """
        # This is a simplified version for contrastive training
        # We use negative entropy as a proxy for energy
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # (B, T)
        
        # Return negative entropy (high entropy = high energy)
        return -entropy.sum(dim=1)  # (B,)
    
    def forward_with_contrastive(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_refine: bool = True,
        refine_steps: int = 2,
        contrastive_loss_fn: Optional[object] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """
        Forward pass with optional contrastive divergence loss.
        
        Args:
            idx: Input tokens (B, T)
            targets: Target tokens for loss computation (B, T)
            use_refine: Whether to use System 2 refinement
            refine_steps: Number of refinement steps
            contrastive_loss_fn: Optional contrastive loss function
            
        Returns: (loss, logits, metrics)
        """
        # Get standard forward pass results
        nll_loss, logits, metrics = self.forward(idx, targets, use_refine, refine_steps)
        
        # Add contrastive loss if enabled
        if contrastive_loss_fn is not None and targets is not None:
            # Compute contrastive divergence loss
            cd_loss = contrastive_loss_fn(targets)
            
            # Combine losses
            total_loss = nll_loss + self.config.contrastive_weight * cd_loss
            
            # Update metrics
            metrics['nll_loss'] = nll_loss.item() if nll_loss is not None else 0
            metrics['cd_loss'] = cd_loss.item()
            metrics['total_loss'] = total_loss.item()
            
            return total_loss, logits, metrics
        
        return nll_loss, logits, metrics
    
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_thinking: bool = True,
        think_steps: int = 4
    ) -> torch.Tensor:
        """
        Generate tokens with optional System 2 thinking.
        
        Args:
            idx: Initial context tokens (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Optional top-k filtering
            use_thinking: Whether to use System 2 refinement
            think_steps: Number of refinement steps when thinking
        
        Returns: Generated token sequence
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Get logits for next token
            if use_thinking:
                logits = self.system2_refine(idx_cond, steps=think_steps, use_soft_tokens=self.config.use_soft_tokens)
            else:
                logits = self.system1_direct_energy(idx_cond)
            
            # Focus on last position
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx






