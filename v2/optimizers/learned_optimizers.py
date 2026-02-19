

# learned optimizer / momentum / etc






# from https://github.com/kolejnyy/titans-lmm/blob/master/neural_memory.py








import torch
from torch import nn, optim
from torch.nn.functional import normalize
from torch.func import functional_call

class NeuralMemory(nn.Module):

    def __init__(self, emb_dim = 16, n_layers = 2, hidden_dim = 32, alpha = 0.999, eta = 0.60, theta = 0.05):
        super().__init__()

        # Define the layers of the network
        self.layers = None
        if n_layers == 1:
            self.layers = nn.ModuleList([nn.Linear(emb_dim, emb_dim)])
        else:
            self.layers = nn.ModuleList([])
            self.layers.append(nn.Sequential(
                nn.Linear(emb_dim, hidden_dim),
                nn.SiLU()
            ))
            for k in range(n_layers - 2):
                self.layers.append(nn.Sequential(
                    nn.Linear(emb_dim, hidden_dim),
                    nn.SiLU()
                ))
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, emb_dim)
            ))

        # Mapping to keys
        self.K = nn.Linear(emb_dim, emb_dim, bias = False)

        # Mapping to values
        self.V = nn.Linear(emb_dim, emb_dim, bias = False)

        torch.nn.init.xavier_uniform_(self.K.weight)
        torch.nn.init.xavier_uniform_(self.V.weight)

        self.alpha = alpha
        self.eta = eta
        self.theta = theta

        self.silu = nn.SiLU()
        self.surprise = {}

    def retrieve(self, x):

        return functional_call(self, dict(self.named_parameters()), x)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def update(self, x):

        z = x.detach()

        # Evaluate the corresponding keys and values
        keys = normalize(self.silu(self.K(z)))
        vals = self.silu(self.V(z))

        # Propagate the keys through the model
        for layer in self.layers:
            keys = layer(keys)

        # Calculate the loss || M(keys) - vals ||_2 ^2
        loss = ((keys - vals) ** 2).mean(axis=0).sum()

        # Compute gradients of aux loss w.r.t. NMM's parameters
        grads = torch.autograd.grad(loss, self.parameters())

        # Update the surprise dictionary and the parameters of the network
        updated_params = {}

        for (name, param), grad in zip(self.named_parameters(), grads):
            if self.surprise.get(name, None) is None:
                self.surprise[name] = torch.zeros_like(grad)
            self.surprise[name] = self.surprise[name] * self.eta - self.theta * grad
            updated_params[name] = self.alpha * param.data + self.surprise[name] if not name[0] in ['K', 'V'] else param.data
            param.data = updated_params[name]

        return loss.item(), updated_params








# ------------------------------------------------------




# from https://github.com/lucidrains/titans-pytorch/blob/main/titans_pytorch/memory_models.py







import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Parameter, ParameterList

from einops import rearrange

# functions

def l2norm(t):
    return F.normalize(t, dim = -1)

# norms

class LayerNorm(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        self.gamma = Parameter(torch.zeros(dim))

    def forward(self, x):
        gamma = self.gamma

        if gamma.ndim == 2:
            gamma = rearrange(gamma, 'b d -> b 1 d')

        return self.ln(x) * (gamma + 1.)

# norm + residual wrapper, as used in original TTT paper
# but could be removed

class ResidualNorm(Module):
    def __init__(
        self,
        dim,
        model: Module
    ):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.model = model

    def forward(self, x):

        out = self.model(x)

        return self.norm(out) + x

# memory mlp proposed in TTT

class MemoryMLP(Module):
    def __init__(
        self,
        dim,
        depth,
        expansion_factor = 2.
    ):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)
        dims = (dim, *((dim_hidden,) * (depth - 1)), dim)

        self.weights = ParameterList([Parameter(torch.randn(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

    def forward(
        self,
        x
    ):
        for ind, weight in enumerate(self.weights):
            is_first = ind == 0

            if not is_first:
                x = F.gelu(x)

            x = x @ weight

        return x

# memory mlp, but with gated residual + final projection

class GatedResidualMemoryMLP(Module):
    def __init__(
        self,
        dim,
        depth,
        expansion_factor = 4.
    ):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)

        self.weights = ParameterList([
            ParameterList([
                Parameter(torch.randn(dim, dim_hidden)),
                Parameter(torch.randn(dim_hidden, dim)),
                Parameter(torch.randn(dim * 2, dim)),
            ]) for _ in range(depth)
        ])

        self.final_proj = Parameter(torch.randn(dim, dim))

        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(
        self,
        x
    ):

        for weight1, weight2, to_gates in self.weights:
            res = x

            hidden = x @ weight1
            hidden = F.gelu(hidden)
            branch_out = hidden @ weight2

            # gated residual

            gates = cat((branch_out, res), dim = -1) @ to_gates
            x = res.lerp(branch_out, gates.sigmoid())

        return x @ self.final_proj

# memory mlp with factorized weights
# so can tradeoff capacity for smaller chunk sizes

class FactorizedMemoryMLP(Module):
    def __init__(
        self,
        dim,
        depth,
        k = 32
    ):
        super().__init__()
        self.weights = ParameterList([
            ParameterList([
                Parameter(torch.randn(dim, k)),
                Parameter(torch.randn(k, dim)),
            ]) for _ in range(depth)
        ])

        for weight1, weight2 in self.weights:
            nn.init.xavier_uniform_(weight1)
            nn.init.xavier_uniform_(weight2)

    def forward(
        self,
        x
    ):

        for ind, (weight1, weight2) in enumerate(self.weights):
            is_first = ind == 0

            if not is_first:
                x = F.gelu(x)

            x = x @ weight1 @ weight2

        return x

# an MLP modelled after the popular swiglu ff in modern transformers

class MemorySwiGluMLP(Module):
    def __init__(
        self,
        dim,
        depth = 1, # default to 2 layer MLP from TTT, depth of 2 would be 4 layer MLP, but done as 2 feedforwards with residual
        expansion_factor = 4.
    ):
        super().__init__()

        dim_inner = int(dim * expansion_factor * 2 / 3)

        weights = []

        for _ in range(depth):
            weights.append(ParameterList([
                Parameter(torch.randn(dim, dim_inner * 2)),
                Parameter(torch.randn(dim_inner, dim)),
            ]))

        self.weights = ParameterList(weights)
        self.norm = LayerNorm(dim)

    def forward(self, x):

        for w1, w2 in self.weights:
            residual = x

            x, gates = (x @ w1).chunk(2, dim = -1)

            x = x * F.gelu(gates)

            x = x @ w2

            x = x + residual

        return self.norm(x)

# improvised attention as memory module

class MemoryAttention(Module):
    def __init__(
        self,
        dim,
        scale = 8.,
        expansion_factor = 2.
    ):
        super().__init__()
        self.scale = scale
        dim_ff_hidden = int(dim * expansion_factor)

        self.weights = ParameterList([
            Parameter(torch.randn(dim, dim)), # queries
            Parameter(torch.randn(dim, dim)), # keys
            Parameter(torch.randn(dim, dim)), # values
            Parameter(torch.randn(dim, dim_ff_hidden)), # ff w1
            Parameter(torch.randn(dim_ff_hidden, dim)), # ff w2
        ])

        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

    def forward(self, x):

        wq, wk, wv, ffw1, ffw2 = self.weights

        q = l2norm(x @ wq)
        k = l2norm(x @ wk)
        v = x @ wv

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            scale = self.scale,
            is_causal = True
        )

        # parallel attention + feedforward block
        # as in PaLM + Gpt-J

        h = F.gelu(x @ ffw1)
        ff_out = h @ ffw2

        return attn_out + ff_out
    









    # ------------------------------------------------------------------







    # from https://github.com/kmccleary3301/nested_learning/blob/main/src/nested_learning/titan/memory.py






    from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from ..assoc_memory import AssocMemory


@dataclass
class TitanMemoryConfig:
    dim: int
    hidden_multiplier: int = 4
    layers: int = 2
    activation: str = "gelu"


def _activation(name: str) -> nn.Module:
    if name.lower() == "relu":
        return nn.ReLU()
    if name.lower() == "gelu":
        return nn.GELU()
    if name.lower() == "silu":
        return nn.SiLU()
    msg = f"Unsupported activation {name}"
    raise ValueError(msg)


class TitanMemory(AssocMemory):
    """Simplified TITAN-style associative memory."""

    def __init__(self, config: TitanMemoryConfig):
        super().__init__()
        self.config = config
        hidden = config.dim * config.hidden_multiplier
        blocks = []
        activation = _activation(config.activation)
        for layer_idx in range(config.layers - 1):
            blocks.extend([nn.Linear(config.dim if layer_idx == 0 else hidden, hidden), activation])
        blocks.append(nn.Linear(hidden if config.layers > 1 else config.dim, config.dim))
        self.net = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(config.dim)
        self.grad_clip = 1.0

    def forward(self, query: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn = self.net(query)
        if self.training and self.grad_clip > 0:
            with torch.no_grad():
                norm = attn.norm(dim=-1, keepdim=True)
                scale = torch.clamp(norm / self.grad_clip, min=1.0)
            attn = attn / scale
        return self.norm(attn)

    def surprise(self, residual: torch.Tensor) -> torch.Tensor:
        return residual.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def update(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        error_signal: torch.Tensor | None = None,
        lr: float = 1e-3,
    ) -> None:
        with torch.enable_grad():
            key_detached = key.detach().requires_grad_(True)
            prediction = self.forward(key_detached)
            target = value.detach()
            if error_signal is None:
                loss = torch.mean((prediction - target) ** 2)
            else:
                loss = torch.mean(error_signal * prediction)
        grads = torch.autograd.grad(loss, list(self.net.parameters()), retain_graph=False)
        for param, grad in zip(self.net.parameters(), grads, strict=False):
            if grad is None:
                continue
            param.add_(grad, alpha=-lr)

    @torch.no_grad()
    def apply_deltas(self, deltas: Dict[str, torch.Tensor], scale: float = 1.0) -> None:
        for name, tensor in deltas.items():
            target = dict(self.named_parameters()).get(name)
            if target is None:
                continue
            target.add_(tensor, alpha=scale)