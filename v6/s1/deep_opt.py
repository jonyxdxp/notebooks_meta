


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















# ----------------------------------------------










def demo_deep_momentum_gd():
    """
    Demo 1: DeepMomentumGD with true nested optimization.

    Shows that memory modules are actually being trained via internal loss.
    """
    print("\n" + "=" * 60)
    print("Demo 1: DeepMomentumGD with Nested Optimization")
    print("=" * 60)

    # Simple model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    # Create optimizer with nested learning
    optimizer = DeepMomentumGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.9,
        memory_lr=1e-4,
        use_shared_memory=True,
    )

    # Simple regression task
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)

    print(f"\nTraining with DeepMomentumGD (memory modules learn via internal loss)...")
    print(f"Memory modules: {len(optimizer.get_memory_modules())}")

    losses = []
    for epoch in range(50):
        optimizer.zero_grad()
        pred = model(X)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            stats = optimizer.get_internal_loss_stats()
            print(f"  Epoch {epoch + 1}: Loss = {loss.item():.4f}, "
                  f"Steps = {stats['step_count']}")

    print(f"\nFinal loss: {losses[-1]:.4f}")
    print(f"Loss reduction: {losses[0]:.4f} -> {losses[-1]:.4f} "
          f"({100 * (1 - losses[-1] / losses[0]):.1f}% improvement)")

    # Compare with simple momentum
    print("\nComparing with SimpleMomentumGD (no nested learning)...")
    model2 = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    optimizer2 = SimpleMomentumGD(model2.parameters(), lr=1e-3, momentum=0.9)

    losses2 = []
    for epoch in range(50):
        optimizer2.zero_grad()
        pred = model2(X)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer2.step()
        losses2.append(loss.item())

    print(f"SimpleMomentumGD final loss: {losses2[-1]:.4f}")
    print(f"DeepMomentumGD final loss:   {losses[-1]:.4f}")

    return losses[-1] < losses2[-1] * 1.5  # Should be competitive












    # Model
    # model = nn.Linear(10, 5, bias=False)

    # Deep Momentum optimizer
    # optimizer = DeepMomentumGD(
    #     model.parameters(),
    #     lr=0.01,
    #     momentum=0.9,
    #     memory_depth=2,  # Use 2-layer MLP for momentum memory
    # )















# ----------------------------------------------------





class DMGD(Optimizer):
    """Faithful DMGD per Eq. 23 of the NL paper."""

    def __init__(self, params, lr=1e-3, momentum=0.9,
                 hidden_dim=64, memory_lr=1e-4):
        defaults = dict(lr=lr, momentum=momentum,
                        hidden_dim=hidden_dim, memory_lr=memory_lr)
        super().__init__(params, defaults)
        self._memory_nets = {}
        self._memory_optims = {}

    def _get_memory(self, p, hidden_dim, memory_lr):
        key = id(p)
        if key not in self._memory_nets:
            d = p.numel()
            net = nn.Sequential(
                nn.Linear(d, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, d)
            ).to(p.device)
            self._memory_nets[key] = net
            # Each memory has its own optimizer — this IS the inner loop
            self._memory_optims[key] = torch.optim.SGD(net.parameters(), lr=memory_lr)
        return self._memory_nets[key], self._memory_optims[key]

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p.data)

                u = p.grad.detach().flatten()   # u_i = ∇L(W_i; x_i)
                m_net, m_optim = self._get_memory(
                    p, group['hidden_dim'], group['memory_lr'])

                # ── inner loop: train m via L^(2) ─────────────────────────
                # Paper Eq. 18: internal obj = -<m(u), u>  (dot-product similarity)
                # One gradient step on this objective IS the m_{i+1} update
                m_optim.zero_grad()
                m_out = m_net(u)
                internal_loss = -torch.dot(m_out, u.detach())   # Eq. 18
                internal_loss.backward()
                m_optim.step()

                # ── outer loop: apply m to update W ───────────────────────
                with torch.no_grad():
                    m_out_detached = m_net(u).flatten()
                    # Eq. 23: m_{i+1} = α*m_i + m_net(u_i)
                    state['momentum'].mul_(group['momentum'])
                    state['momentum'].add_(m_out_detached.view_as(p))
                    # W_{i+1} = W_i + m_{i+1}(u_i)
                    p.data.add_(state['momentum'], alpha=-group['lr'])

        return loss








        





# ---------------------------------------------




# from https://github.com/erikl2/nested-learning/blob/main/src/nested_learning/optimizers/nested_dmgd.py














"""
Nested Deep Momentum GD with Meta-Learning

An attempt at implementing the nested learning framework from the paper:
- Memory modules that can be trained via meta-learning
- Internal loss via `meta_step()` method
- Proper gradient flow through memory modules

STATUS: NEEDS VALIDATION
- Code structure attempts to follow paper's concepts
- Meta-learning loop is implemented but not thoroughly tested
- No experimental validation against paper's results
- May have bugs or deviations from paper's formulation

This is an improvement over deep_momentum.py (which uses static MLPs),
but should still be considered experimental/research-quality code.
"""

import torch
from torch.optim import Optimizer
import torch.nn as nn
from typing import Optional, List, Callable


class LearnedMemoryModule(nn.Module):
    """
    MLP that learns to compress gradients.

    This is the core "memory" that replaces linear momentum.
    It will be trained via meta-learning.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int = 2,
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for i in range(depth):
            if i == depth - 1:
                # Last layer
                layers.append(nn.Linear(current_dim, output_dim))
            else:
                # Hidden layers
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))  # Add stability
                layers.append(nn.ReLU())
                current_dim = hidden_dim

        self.network = nn.Sequential(*layers)

        # Initialize with small weights for stability
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, grad: torch.Tensor, momentum: torch.Tensor) -> torch.Tensor:
        """
        Process gradient and momentum to produce new momentum.

        Args:
            grad: Current gradient (flattened)
            momentum: Previous momentum (flattened)

        Returns:
            New momentum
        """
        # Concatenate gradient and momentum as input context
        x = torch.cat([grad, momentum], dim=-1)
        return self.network(x)


class NestedDeepMomentumGD(Optimizer):
    """
    Deep Momentum GD with proper nested optimization.

    This implements the full nested learning framework:
    - Outer loop: Optimize model parameters using learned memory
    - Inner loop: Optimize memory module parameters via meta-learning

    Args:
        params: Model parameters to optimize
        memory_optimizer: Optimizer for the memory modules (e.g., Adam)
        lr: Learning rate for model parameters
        memory_lr: Learning rate for memory module parameters
        momentum: Momentum coefficient
        memory_depth: Depth of memory MLP
        memory_hidden_dim: Hidden dimension of memory MLP
        meta_learning: Whether to enable meta-learning for memory
    """

    def __init__(
        self,
        params,
        memory_optimizer: Optional[Optimizer] = None,
        lr: float = 1e-3,
        memory_lr: float = 1e-4,
        momentum: float = 0.9,
        memory_depth: int = 2,
        memory_hidden_dim: int = 64,
        meta_learning: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            memory_depth=memory_depth,
            memory_hidden_dim=memory_hidden_dim,
            memory_lr=memory_lr,
        )
        super().__init__(params, defaults)

        self.meta_learning = meta_learning
        self.memory_modules = {}
        self.memory_optimizer = memory_optimizer

        # Initialize memory modules for each parameter
        self._init_memory_modules()

    def _init_memory_modules(self):
        """Initialize learned memory modules for each parameter group."""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                # Initialize momentum buffer
                state['momentum_buffer'] = torch.zeros_like(p.data)

                # Get dimensions
                param_numel = p.numel()
                hidden_dim = group['memory_hidden_dim']

                # Create learned memory module
                # Input: [gradient, momentum] (concatenated)
                # Output: new momentum
                memory = LearnedMemoryModule(
                    input_dim=param_numel * 2,  # grad + momentum
                    hidden_dim=hidden_dim,
                    output_dim=param_numel,
                    depth=group['memory_depth'],
                )

                # Move to same device as parameter
                memory = memory.to(p.device)

                # Store in state and module dict
                state['memory_module'] = memory
                self.memory_modules[id(p)] = memory

        # Create optimizer for memory modules if meta-learning is enabled
        if self.meta_learning and self.memory_optimizer is None:
            memory_params = []
            for memory in self.memory_modules.values():
                memory_params.extend(memory.parameters())

            self.memory_optimizer = torch.optim.Adam(
                memory_params,
                lr=self.param_groups[0]['memory_lr'],
            )

    def get_memory_modules(self) -> List[nn.Module]:
        """Return list of all memory modules."""
        return list(self.memory_modules.values())

    def step(self, closure: Optional[Callable] = None, create_graph: bool = False):
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
                    Required for meta-learning.
            create_graph: If True, preserve computation graph for meta-learning.
                         This allows backprop through the optimization step.

        Note: For meta-learning to work, set create_graph=True. This keeps the
        computation graph through the memory modules so meta_step can compute
        gradients w.r.t. memory module parameters.

        The key insight: we store the memory outputs and use them in
        compute_meta_gradients() to create a gradient path back to memory modules.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Store outputs for meta-learning backward pass
        self._last_memory_outputs = []

        # Outer loop: Update model parameters using learned memory
        for group in self.param_groups:
            lr = group['lr']
            momentum_coef = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                memory_module = state['memory_module']

                # Detach gradient - we don't need higher order gradients through p.grad
                # We only need gradients through the memory module
                grad = p.grad.detach()

                # Flatten tensors
                grad_flat = grad.flatten()
                momentum_flat = state['momentum_buffer'].detach().flatten()

                # Process through learned memory module
                memory_module.train()
                new_momentum_flat = memory_module(
                    grad_flat.unsqueeze(0),
                    momentum_flat.unsqueeze(0),
                ).squeeze(0)

                # Store for meta-learning - this is what allows gradient flow
                if create_graph:
                    self._last_memory_outputs.append(new_momentum_flat)

                # Reshape back
                new_momentum = new_momentum_flat.view_as(p)

                # Compute full update with momentum (keep graph through new_momentum)
                combined_momentum = momentum_coef * state['momentum_buffer'].detach() + new_momentum

                # Update momentum buffer (detached copy - no in-place ops that affect graph)
                state['momentum_buffer'] = combined_momentum.detach().clone()

                # Update parameters (detached - actual parameter update)
                with torch.no_grad():
                    p.add_(combined_momentum.detach(), alpha=-lr)

        return loss

    def compute_meta_gradients(self, meta_loss: torch.Tensor):
        """
        Compute and accumulate gradients to memory modules from meta_loss.

        Because parameter updates happen in no_grad, the meta_loss doesn't
        directly connect to memory modules. This method creates that connection
        using a surrogate loss based on the stored memory outputs.

        The approach:
        1. We stored the memory module outputs during step()
        2. We compute a "surrogate" gradient target from meta_loss
        3. We backprop through the stored outputs to update memory modules

        Args:
            meta_loss: The meta loss computed after optimization steps
        """
        if not hasattr(self, '_last_memory_outputs') or not self._last_memory_outputs:
            return

        # Create a surrogate loss that connects meta_loss to memory outputs
        # The idea: we want memory outputs that would have led to lower meta_loss
        # We approximate this by backpropping meta_loss through model params,
        # then using those gradients as targets for the memory outputs

        # Get model parameters that were updated
        params = []
        for group in self.param_groups:
            for p in group['params']:
                params.append(p)

        # Get gradients of meta_loss w.r.t. current model parameters
        try:
            param_grads = torch.autograd.grad(
                meta_loss,
                params,
                create_graph=False,
                retain_graph=True,
                allow_unused=True,
            )
        except RuntimeError:
            # If meta_loss doesn't depend on params (shouldn't happen), skip
            return

        # Create surrogate loss: we want memory outputs to produce updates
        # that move parameters in the direction that reduces meta_loss
        # L_surrogate = sum_i (memory_output_i * grad_meta_i)
        # Minimizing this encourages memory to produce updates aligned with -grad_meta
        # Use meta_loss to get correct device/dtype
        surrogate_loss = meta_loss.new_zeros(())
        for i, (memory_out, grad) in enumerate(zip(self._last_memory_outputs, param_grads)):
            if grad is not None and memory_out.grad_fn is not None:
                # Reshape grad to match memory output
                grad_flat = grad.flatten()
                if grad_flat.shape == memory_out.shape:
                    # Want memory_out to point opposite to grad (for descent)
                    surrogate_loss = surrogate_loss + (memory_out * grad_flat).sum()

        # Backprop surrogate loss to memory modules
        if surrogate_loss.grad_fn is not None:
            surrogate_loss.backward()

    def meta_step(self, meta_loss: torch.Tensor):
        """
        Perform meta-learning step to improve memory modules.

        This implements the inner optimization loop L̃^(2) from the paper.

        The key challenge: standard backprop through meta_loss won't reach
        memory modules because parameter updates happen in no_grad().

        Solution: We use compute_meta_gradients() to create a surrogate loss
        that connects the meta_loss signal to memory module outputs.

        Args:
            meta_loss: Loss that depends on how well the optimizer performs
                      (e.g., validation loss after several optimization steps)

        Note: For this to work, you must call optimizer.step(create_graph=True)
        during the inner optimization loop to preserve the computation graph.
        """
        if not self.meta_learning:
            return

        if self.memory_optimizer is None:
            raise ValueError("Meta-learning enabled but no memory optimizer provided")

        # Zero gradients first
        self.memory_optimizer.zero_grad()

        # Check if we have stored memory outputs from create_graph=True steps
        if hasattr(self, '_last_memory_outputs') and self._last_memory_outputs:
            # Use surrogate loss approach to connect meta_loss to memory modules
            self.compute_meta_gradients(meta_loss)
        else:
            # Fallback: try direct backprop (may not work if graph not retained)
            try:
                meta_loss.backward(retain_graph=True)
            except RuntimeError:
                # Graph was not retained, memory modules won't be updated
                import warnings
                warnings.warn(
                    "meta_step called but computation graph not available. "
                    "Make sure to call step(create_graph=True) during inner loop."
                )
                return

        # Clip gradients for stability
        for memory in self.memory_modules.values():
            torch.nn.utils.clip_grad_norm_(memory.parameters(), max_norm=1.0)

        # Update memory module parameters
        self.memory_optimizer.step()

        # Clear stored outputs
        if hasattr(self, '_last_memory_outputs'):
            self._last_memory_outputs = []

    def state_dict(self):
        """
        Return state dict including memory modules.
        """
        state = super().state_dict()

        # Add memory modules
        memory_state = {}
        for param_id, memory in self.memory_modules.items():
            memory_state[param_id] = memory.state_dict()

        state['memory_modules'] = memory_state

        if self.memory_optimizer is not None:
            state['memory_optimizer'] = self.memory_optimizer.state_dict()

        return state

    def load_state_dict(self, state_dict):
        """
        Load state dict including memory modules.
        """
        # Load memory modules
        if 'memory_modules' in state_dict:
            memory_state = state_dict.pop('memory_modules')
            for param_id, memory_dict in memory_state.items():
                if param_id in self.memory_modules:
                    self.memory_modules[param_id].load_state_dict(memory_dict)

        # Load memory optimizer
        if 'memory_optimizer' in state_dict and self.memory_optimizer is not None:
            mem_opt_state = state_dict.pop('memory_optimizer')
            self.memory_optimizer.load_state_dict(mem_opt_state)

        # Load base optimizer state
        super().load_state_dict(state_dict)


def create_meta_learning_task_distribution(
    num_tasks: int = 100,
    input_dim: int = 10,
    output_dim: int = 1,
    task_type: str = 'regression',
) -> List[Callable]:
    """
    Create a distribution of tasks for meta-learning the optimizer.

    This is used to train the memory modules on a variety of optimization
    problems so they learn general optimization strategies.

    Args:
        num_tasks: Number of tasks in distribution
        input_dim: Input dimension for tasks
        output_dim: Output dimension for tasks
        task_type: Type of tasks ('regression', 'classification', 'mixed')

    Returns:
        List of task generator functions
    """
    tasks = []

    for _ in range(num_tasks):
        if task_type == 'regression' or (task_type == 'mixed' and torch.rand(1).item() < 0.5):
            # Random regression task
            true_weights = torch.randn(input_dim, output_dim)
            noise_level = torch.rand(1).item() * 0.5

            def task_fn():
                X = torch.randn(32, input_dim)
                y = X @ true_weights + noise_level * torch.randn(32, output_dim)
                return X, y

            tasks.append(task_fn)
        else:
            # Random classification task (binary)
            true_weights = torch.randn(input_dim, output_dim)

            def task_fn():
                X = torch.randn(32, input_dim)
                logits = X @ true_weights
                y = (logits > 0).float()
                return X, y

            tasks.append(task_fn)

    return tasks






























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




