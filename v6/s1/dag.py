

# from Dynamic Nested Hierarchies paper












# ---------------------------------------------------------------------------
# 5.  DNH Level  — one node in the DAG
# ---------------------------------------------------------------------------

class DNHLevel(nn.Module):
    """
    One level ℓ in the Dynamic Nested Hierarchy.
    Wraps a SelfModifyingMemory with its own FrequencyModulator.
    """

    def __init__(
        self,
        level_id: int,
        key_dim: int,
        val_dim: int,
        context_dim: int,
        init_freq: float = 1.0,
        hidden_dim: int = 64,
        eta_f: float = 0.01,
    ):
        super().__init__()
        self.level_id = level_id
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.context_dim = context_dim

        self.smm = SelfModifyingMemory(key_dim, val_dim, context_dim, hidden_dim)
        self.freq_mod = FrequencyModulator(init_freq=init_freq, eta_f=eta_f)

        # Linear projections: maps previous-level output to (k, v, ctx) space
        self.key_proj = nn.Linear(val_dim, key_dim)
        self.val_proj = nn.Linear(val_dim, val_dim)
        self.ctx_proj = nn.Linear(val_dim, context_dim)

    def forward(
        self,
        x: torch.Tensor,                       # [B, val_dim] — input or prev-level output
        lss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        keys    = self.key_proj(x)              # [B, key_dim]
        vals    = self.val_proj(x)              # [B, val_dim]
        context = self.ctx_proj(x)              # [B, context_dim]
        return self.smm(keys, vals, context, lss)

    def get_gradient_norm(self) -> float:
        """‖∇_{θ^(ℓ)} L^(1)‖ — used for pruning criterion."""
        total = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total += p.grad.data.norm().item() ** 2
        return math.sqrt(total)










# ---------------------------------------------------------------------------
# 6.  Meta-Controller  E_ϕ  (Eq. 11-12)
# ---------------------------------------------------------------------------

class MetaController(nn.Module):
    """
    Evolution operator  E_ϕ(G_t, x_t, L_t) → G_{t+1}   (Eq. 11)

    Decides whether to:
      • Add a level  (if L_meta > τ_add)
      • Prune a level (if ‖∇_{θ^(ℓ)}‖ < ε_prune for any ℓ)
      • Only modulate frequencies

    Also updates its own parameters via meta-gradient descent (Eq. 12).
    """

    def __init__(
        self,
        tau_add: float = 0.5,
        epsilon_prune: float = 1e-3,
        lambda_struct: float = 0.01,
        mu_shift: float = 0.1,
        eta_phi: float = 1e-3,
        max_levels: int = 8,
        min_levels: int = 1,
    ):
        super().__init__()
        self.tau_add = tau_add
        self.epsilon_prune = epsilon_prune
        self.lambda_struct = lambda_struct
        self.mu_shift = mu_shift
        self.eta_phi = eta_phi
        self.max_levels = max_levels
        self.min_levels = min_levels

        # Learnable thresholds (ϕ)
        self.log_tau = nn.Parameter(torch.tensor(math.log(max(tau_add, 1e-8))))
        self.log_eps = nn.Parameter(torch.tensor(math.log(max(epsilon_prune, 1e-8))))

    @property
    def effective_tau(self):
        return F.softplus(self.log_tau)

    @property
    def effective_eps(self):
        return F.softplus(self.log_eps)

    def meta_loss(
        self,
        task_loss: torch.Tensor,
        delta_structure: float,
        kl_shift: torch.Tensor,
    ) -> torch.Tensor:
        """
        L_meta = L^(1) + λ‖ΔG‖₁ + μ D_KL(p_t ‖ p_{t-1})   (Eq. 12 comment)
        """
        return (
            task_loss
            + self.lambda_struct * delta_structure
            + self.mu_shift * kl_shift
        )

    def should_add(self, meta_loss_val: float) -> bool:
        return meta_loss_val > self.effective_tau.item()

    def should_prune(self, level: DNHLevel) -> bool:
        return (
            level.get_gradient_norm() < self.effective_eps.item()
        )










