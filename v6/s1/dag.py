

# from https://github.com/kmccleary3301/nested_learning/blob/main/src/nested_learning/optim/manager.py












from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
from torch import nn

from ..levels import LevelClock, LevelSpec
from .factory import build_optimizer

from .deep import DeepMomentum














def build_optimizer(config: Dict[str, Any]) -> DeepMomentum:
    opt_type = config.get("type", "deep_momentum").lower()
    if opt_type != "deep_momentum":
        raise ValueError(f"Unsupported optimizer type {opt_type}")
    params = config.get("params", {})
    return DeepMomentum(**params)











@dataclass
class LevelConfig:
    specs: Sequence[LevelSpec]
    optimizer_configs: Dict[str, dict]
    default_lr: float


class LevelOptimizerManager:
    def __init__(self, config: LevelConfig):
        self.clock = LevelClock(config.specs)
        self.learning_rates: Dict[str, float] = {}
        self.optimizers = {}
        self._last_metrics: Dict[str, Dict[str, float]] = {}
        for spec in config.specs:
            key = spec.optimizer_key or "default"
            optim_cfg = config.optimizer_configs.get(key, {"type": "deep_momentum", "params": {}})
            lr = optim_cfg.get("lr", config.default_lr)
            params_cfg = optim_cfg.get("params", {})
            optimizer = build_optimizer(
                {"type": optim_cfg.get("type", "deep_momentum"), "params": params_cfg}
            )
            self.optimizers[spec.name] = optimizer
            self.learning_rates[spec.name] = lr

    def should_update(self, level: str) -> bool:
        return self.clock.should_update(level)

    def optimize(
        self,
        level: str,
        module: nn.Module,
        loss: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        force: bool = False,
    ) -> float:
        if (not force) and (not self.should_update(level)):
            return 0.0
        named_params: Tuple[Tuple[str, torch.nn.Parameter], ...] = tuple(
            (name, param) for name, param in module.named_parameters() if param.requires_grad
        )
        if not named_params:
            return 0.0
        params = tuple(param for _, param in named_params)
        grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
        grads_dict: Dict[str, torch.Tensor] = {}
        for (name, _), grad in zip(named_params, grads, strict=True):
            if grad is None:
                continue
            grads_dict[name] = grad
        return self.apply_module_grads(
            level,
            module,
            grads_dict,
            context=context,
            force=True,
        )

    def apply_module_grads(
        self,
        level: str,
        module: nn.Module,
        grads: Dict[str, torch.Tensor],
        *,
        context: torch.Tensor | None = None,
        force: bool = False,
    ) -> float:
        if (not force) and (not self.should_update(level)):
            return 0.0
        optimizer = self.optimizers[level]
        lr = self.learning_rates[level]
        total_norm = 0.0
        with torch.no_grad():
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                grad = grads.get(name)
                if grad is None:
                    continue
                update = optimizer(grad, context=context, param_key=name)
                param.add_(update, alpha=-lr)
                total_norm += grad.norm().item()
        self.clock.record_update(level)
        metrics = getattr(optimizer, "last_metrics", None)
        if metrics:
            self._last_metrics[level] = dict(metrics)
        else:
            self._last_metrics[level] = {}
        return total_norm

    def tick(self) -> None:
        self.clock.tick()

    def pop_last_metrics(self, level: str) -> Dict[str, float]:
        return self._last_metrics.pop(level, {})

    def apply_grads(
        self,
        level: str,
        params: Dict[str, torch.Tensor],
        grads: Dict[str, torch.Tensor],
        *,
        context: torch.Tensor | None = None,
        force: bool = False,
        differentiable: bool = False,
    ) -> tuple[Dict[str, torch.Tensor], float]:
        if (not force) and (not self.should_update(level)):
            return params, 0.0
        optimizer = self.optimizers[level]
        lr = self.learning_rates[level]
        updated: Dict[str, torch.Tensor] = {}
        total_norm = 0.0
        if differentiable:
            for name, param in params.items():
                grad = grads.get(name)
                if grad is None:
                    updated[name] = param
                    continue
                updated[name] = param - lr * grad
                total_norm += float(grad.detach().norm().item())
            self.clock.record_update(level)
            self._last_metrics[level] = {"differentiable_updates": 1.0}
            return updated, total_norm
        with torch.no_grad():
            for name, param in params.items():
                grad = grads.get(name)
                if grad is None:
                    updated[name] = param
                    continue
                update = optimizer(grad, context=context, param_key=name)
                updated[name] = (param - lr * update).detach()
                total_norm += grad.norm().item()
        self.clock.record_update(level)
        metrics = getattr(optimizer, "last_metrics", None)
        if metrics:
            self._last_metrics[level] = dict(metrics)
        else:
            self._last_metrics[level] = {}
        return updated, total_norm












# from https://github.com/kmccleary3301/nested_learning/blob/main/src/nested_learning/levels.py




from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping, Sequence


@dataclass(frozen=True)
class LevelSpec:
    """Configuration for a nested-learning level."""

    name: str
    update_period: int
    warmup_steps: int = 0
    jitter: int = 0
    optimizer_key: str | None = None

    def __post_init__(self) -> None:
        if self.update_period <= 0:
            msg = f"update_period for level {self.name} must be positive"
            raise ValueError(msg)
        if self.warmup_steps < 0:
            msg = f"warmup_steps for level {self.name} must be non-negative"
            raise ValueError(msg)
        if self.jitter < 0:
            msg = f"jitter for level {self.name} must be non-negative"
            raise ValueError(msg)


@dataclass
class LevelState:
    last_step: int = -1
    updates: int = 0


class LevelClock:
    """Deterministic scheduler for Nested Learning level updates."""

    def __init__(self, specs: Sequence[LevelSpec]):
        self._specs: Dict[str, LevelSpec] = {spec.name: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("Duplicate level names provided to LevelClock")
        self._state: MutableMapping[str, LevelState] = {name: LevelState() for name in self._specs}
        self._step: int = 0
        self._timeline: List[dict] = []

    @property
    def step(self) -> int:
        return self._step

    def tick(self) -> None:
        self._step += 1

    def should_update(self, name: str) -> bool:
        spec = self._specs[name]
        state = self._state[name]
        if self._step < spec.warmup_steps:
            return False
        delta = self._step - state.last_step
        period = spec.update_period
        if spec.jitter:
            period = period + (self._step % (spec.jitter + 1))
        return state.last_step < 0 or delta >= period

    def record_update(self, name: str) -> None:
        state = self._state[name]
        state.last_step = self._step
        state.updates += 1
        self._timeline.append({"step": self._step, "level": name})

    def levels_in_frequency_order(self) -> List[LevelSpec]:
        return sorted(self._specs.values(), key=lambda spec: spec.update_period)

    def stats(self) -> Dict[str, LevelState]:
        return {
            name: LevelState(state.last_step, state.updates) for name, state in self._state.items()
        }

    def timeline(self) -> List[dict]:
        return list(self._timeline)


def ensure_level_specs(entries: Iterable[LevelSpec]) -> List[LevelSpec]:
    """Ensure deterministic ordering and validate duplicates."""

    specs = list(entries)
    seen = set()
    ordered: List[LevelSpec] = []
    for spec in specs:
        if spec.name in seen:
            msg = f"Duplicate level spec {spec.name}"
            raise ValueError(msg)
        seen.add(spec.name)
        ordered.append(spec)
    return ordered



















# --------------------------------------------------------------------------


















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










