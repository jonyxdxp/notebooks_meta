


# from https://github.com/aryateja2106/nested-learning/blob/main/src/core/optimizers.py









class DeepMomentum(Optimizer):
    """
    Deep Momentum Gradient Descent (DMGD).

    Replaces linear momentum with an MLP to increase memory capacity
    for compressing past gradients.

    Update rule (Equation 50):
        W_{i+1} = W_i + m_{i+1}(u_i)
        m_{i+1} = α_{i+1} m_i - η_t ∇L^(2)(m_i; u_i, 1)

    where:
        - u_i = ∇L(W_i; x_i) (the gradient)
        - m(·) is an MLP that maps gradients to updates
        - L^(2) is the internal objective of momentum

    Args:
        params: Parameters to optimize
        lr: Learning rate
        momentum: Momentum decay factor
        hidden_dim: Hidden dimension of momentum MLP
    """

    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.9, hidden_dim: int = 64):
        defaults = dict(lr=lr, momentum=momentum, hidden_dim=hidden_dim)
        super().__init__(params, defaults)

        # Create momentum networks for each parameter
        self._momentum_networks = {}

    def _get_momentum_network(self, p: torch.Tensor, hidden_dim: int) -> nn.Module:
        """Get or create momentum network for a parameter."""
        key = id(p)
        if key not in self._momentum_networks:
            in_dim = p.numel()
            self._momentum_networks[key] = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, in_dim)
            ).to(p.device)
        return self._momentum_networks[key]

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            hidden_dim = group["hidden_dim"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.flatten()

                # Get momentum network
                m_net = self._get_momentum_network(p, hidden_dim)

                # Compute momentum update
                with torch.enable_grad():
                    m_output = m_net(grad)

                # Update parameters
                p.data.add_(m_output.view_as(p.data), alpha=-lr)

        return loss












# -----------------------------------------------------------













# ---------------------------------------------------------------------------
# 2.  Meta-Network  g_ψ  (Eq. 17-18)
# ---------------------------------------------------------------------------

class MetaNetwork(nn.Module):
    """
    g_ψ(k_t, v_t, c^(ℓ)_t) → Δθ^(ℓ)_t  (Eq. 17)
    A small network that generates a parameter-space modification term.
    """

    def __init__(self, key_dim: int, val_dim: int, context_dim: int, hidden_dim: int = 64):
        super().__init__()
        in_dim = key_dim + val_dim + context_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, key_dim * val_dim),
        )
        self.key_dim = key_dim
        self.val_dim = val_dim

    def forward(
        self,
        keys: torch.Tensor,   # [B, key_dim]
        vals: torch.Tensor,   # [B, val_dim]
        context: torch.Tensor # [B, context_dim]
    ) -> torch.Tensor:
        """Returns Δθ  [B, val_dim, key_dim] (outer-product shaped)."""
        inp = torch.cat([keys, vals, context], dim=-1)
        delta = self.net(inp)                          # [B, val_dim * key_dim]
        return delta.view(-1, self.val_dim, self.key_dim)













# ---------------------------------------------------------------------------
# 3.  Self-Modifying Memory (SMM)  (Eq. 17-19)
# ---------------------------------------------------------------------------

class SelfModifyingMemory(nn.Module):
    """
    M^(ℓ)_t(k_t) = θ^(ℓ)_t k_t + Δθ^(ℓ)_t ⊙ v_t   (Eq. 17)
    Memory update:  M_{t+1} = M_t + v_{t+1} k^⊤_{t+1} + α_t ∇_{M_t} L_meta  (Eq. 19)
    where α_t = σ(w^⊤ LSS_t + b)  (adaptive based on surprise signal)
    """

    def __init__(self, key_dim: int, val_dim: int, context_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.key_dim = key_dim
        self.val_dim = val_dim

        # Θ^(ℓ) — the base memory matrix [val_dim, key_dim]
        self.theta = nn.Parameter(torch.randn(val_dim, key_dim) * 0.01)

        # Meta-network g_ψ
        self.meta_net = MetaNetwork(key_dim, val_dim, context_dim, hidden_dim)

        # Adaptive α gate (Eq. 19)
        self.alpha_gate = nn.Linear(1, 1)   # scalar LSS → scalar α

        # Running memory matrix (non-parameter; updated in forward)
        self.register_buffer("memory", torch.zeros(val_dim, key_dim))

    def forward(
        self,
        keys: torch.Tensor,    # [B, key_dim]
        vals: torch.Tensor,    # [B, val_dim]
        context: torch.Tensor, # [B, context_dim]
        lss: Optional[torch.Tensor] = None,  # scalar surprise signal
    ) -> torch.Tensor:
        """
        Returns output y^(ℓ) = M(k)  [B, val_dim].
        Also performs the online memory update (Eq. 19).
        """
        B = keys.size(0)

        # --- base retrieval: θ k  [B, val_dim]
        base_out = keys @ self.theta.T           # [B, val_dim]

        # --- modification: Δθ ⊙ v  [B, val_dim]
        delta_theta = self.meta_net(keys, vals, context)  # [B, val_dim, key_dim]
        mod_out = (delta_theta * keys.unsqueeze(1)).sum(-1) * vals   # element-wise

        out = base_out + mod_out   # [B, val_dim]

        # --- online memory matrix update (Eq. 19)
        # M_{t+1} = M_t + (1/B) v k^⊤  + α_t * ∇_{M}
        # We use a soft gradient proxy (detached for stability)
        with torch.no_grad():
            outer = (vals.T @ keys) / B   # [val_dim, key_dim]
            if lss is not None:
                alpha = torch.sigmoid(self.alpha_gate(lss.unsqueeze(-1))).squeeze()
            else:
                alpha = torch.tensor(0.01, device=keys.device)
            self.memory = self.memory + outer * (1.0 + alpha)

        return out

    def reset_memory(self):
        self.memory.zero_()









# ---------------------------------------------------------------------------
# 4.  Frequency Modulator  (Eq. 5 & 16)
# ---------------------------------------------------------------------------

class FrequencyModulator(nn.Module):
    """
    First-order:   f^(ℓ)_{t+1} = f^(ℓ)_t + η_f ∇_{f^(ℓ)} L_meta + m^(ℓ)_{t+1}
    (with momentum)                                                    (Eq. 5)

    Second-order (Hessian approx): f^(ℓ)_{t+1} = f^(ℓ)_t − η_f (H^(ℓ))^{-1} ∇ L_meta
                                                                   (Eq. 16)
    """

    def __init__(
        self,
        init_freq: float = 1.0,
        eta_f: float = 0.01,
        beta: float = 0.9,
        use_second_order: bool = False,
    ):
        super().__init__()
        self.log_freq = nn.Parameter(torch.tensor(math.log(init_freq)))
        self.eta_f = eta_f
        self.beta = beta
        self.use_second_order = use_second_order
        self.register_buffer("momentum", torch.zeros(1))

    @property
    def freq(self) -> torch.Tensor:
        """Always positive via softplus reparameterisation."""
        return F.softplus(self.log_freq)

    def update(self, grad_f: torch.Tensor, hessian_diag: Optional[torch.Tensor] = None):
        """Perform one frequency update step (in-place on log_freq buffer)."""
        with torch.no_grad():
            # Ensure grad_f is a scalar tensor
            if not isinstance(grad_f, torch.Tensor):
                grad_f = torch.tensor(float(grad_f))
            grad_f = grad_f.squeeze()
            self.momentum.squeeze_()
            self.momentum.mul_(self.beta).add_((1 - self.beta) * grad_f)
            if self.use_second_order and hessian_diag is not None:
                step = grad_f / (hessian_diag.abs() + 1e-6)
            else:
                step = grad_f + self.momentum
            self.log_freq.data.add_((self.eta_f * step).squeeze())




