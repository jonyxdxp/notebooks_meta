



def hopfield_energy(x, Z, beta):
    # x: (D,), Z: (D, N)
    dots = beta * (Z.T @ x)           # (N,)
    return -torch.logsumexp(dots, dim=0)

def refine(x_base, Z_abstract, beta=1.0, lam=0.5, 
           alpha=0.01, steps=10):
    """
    x_base:     (D,)   — output of DM_1 (base dynamics)
    Z_abstract: (D, N) — outputs of DM_2...DM_k (abstract dynamics)
    """
    x = x_base.clone().requires_grad_(True)
    
    for _ in range(steps):
        E = hopfield_energy(x, Z_abstract, beta) \
          + lam * ((x - x_base.detach()) ** 2).sum()
        
        grad = torch.autograd.grad(E, x)[0]
        x = (x - alpha * grad).detach().requires_grad_(True)
    
    return x.detach()



















# ------------------------------------------------------
















# from https://github.com/alexiglad/EBT/blob/main/model/img/ebt_denoise.py









    
def ebt_advanced_inference(self, noised_x, learning = True, no_randomness = True): #NOTE should eventually add more features from NLP and VID, for now is same as forward but with more steps
        predicted_x_list = []
        predicted_energies_list = []
        batch_size = noised_x.shape[0]

        predicted_x = noised_x.clone().detach()

        alpha = torch.clamp(self.alpha, min=0.0001)
        if not no_randomness and self.hparams.randomize_mcmc_step_size_scale != 1:
            expanded_alpha = alpha.expand(batch_size, 1, 1, 1)

            scale = self.hparams.randomize_mcmc_step_size_scale
            low = alpha / scale
            high = alpha * scale
            alpha = low + torch.rand_like(expanded_alpha) * (high - low)

        langevin_dynamics_noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)

        mcmc_steps = [] # in the general case of no randomize_mcmc_num_steps then this has len == self.hparams.randomize_mcmc_num_steps
        for step in range(self.hparams.infer_ebt_num_steps):
            if not no_randomness and hasattr(self.hparams, 'randomize_mcmc_num_steps') and self.hparams.randomize_mcmc_num_steps > 0:
                if self.hparams.randomize_mcmc_num_steps_final_landscape: # makes so only applies rand steps to final landscape
                    if step == (self.hparams.infer_ebt_num_steps - 1):
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
                if step == (self.hparams.infer_ebt_num_steps - 1): # i found this was a better pretraining metric and was more stable, only do several steps on final energy landscape instead of over all energy landscapes
                    mcmc_steps.extend([step] * (self.hparams.randomize_mcmc_num_steps + 1))
                else:
                    mcmc_steps.append(step)
            else:
                mcmc_steps.append(step)

        with torch.set_grad_enabled(True): # set to true for validation since grad would be off
            for i, mcmc_step in enumerate(mcmc_steps):
                if self.hparams.no_mcmc_detach:
                    predicted_x = predicted_x.requires_grad_() # B, C, W, H
                else: # default, do detach
                    predicted_x = predicted_x.detach().requires_grad_() # B, C, W, H
                
                if self.hparams.langevin_dynamics_noise != 0 and not (no_randomness and self.hparams.no_langevin_during_eval):
                    ld_noise = torch.randn_like(predicted_x.detach(), device=predicted_x.device) * langevin_dynamics_noise_std # langevin dynamics
                    predicted_x = predicted_x + ld_noise
                                
                condition = self.learned_adaln_condition(torch.tensor([0], device=predicted_x.device))
                condition = condition.expand(batch_size, -1)  # Expand to match batch size
                energy_preds = self.transformer(predicted_x, condition).squeeze()
                energy_preds = energy_preds.mean(dim=[1]).reshape(-1) # B
                predicted_energies_list.append(energy_preds)

                if self.hparams.truncate_mcmc:  #retain_graph defaults to create_graph value here; if learning is true then create_graph else dont (inference)
                    if i == (len(mcmc_steps) - 1):
                        predicted_embeds_grad = torch.autograd.grad([energy_preds.sum()], [predicted_x], create_graph=learning)[0]
                    else:
                        predicted_embeds_grad = torch.autograd.grad([energy_preds.sum()], [predicted_x], create_graph=False)[0]
                else:
                    predicted_embeds_grad = torch.autograd.grad([energy_preds.sum()], [predicted_x], create_graph=learning)[0]
                
                if self.hparams.clamp_futures_grad:
                    min_and_max = self.hparams.clamp_futures_grad_max_change / (self.alpha)
                    # predicted_embeds_grad = scale_clamp(predicted_embeds_grad, -min_and_max, min_and_max)
                    predicted_embeds_grad = torch.clamp(predicted_embeds_grad, min = -min_and_max, max = min_and_max)

                if torch.isnan(predicted_embeds_grad).any() or torch.isinf(predicted_embeds_grad).any():
                    raise ValueError("NaN or Inf gradients detected during MCMC.")
                    
                predicted_x = predicted_x - alpha * predicted_embeds_grad
                
                predicted_x_list.append(predicted_x)

        return predicted_x_list, predicted_energies_list