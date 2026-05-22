# Transformer arquitecture for Perception EBM


# basicall, defines an Energy function over the latent space of an encoder, 
# to shape the latent space to discard irrelevant/unwanted features






# la Variable Latente la incluiremos recien en la Inferencia, no en el Entrenamiento






# Energy Transformer (ET) Block 





import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from dataclasses import dataclass
from typing import *



@dataclass
class ETConfig():
  D: int = 768 # Token dimension of ET
  Y: int = 64 # Token dimension of each query and key
  n_heads: int = 12 # Number of heads
  scale_mems: float = 4. # Scale the number of memories by this factor relative to token dimension D






class EnergyLayerNorm(eqx.Module):
  """Define our primary activation function (modified LayerNorm) as a lagrangian with energy"""
  gamma: jax.Array  # Scaling scalar
  delta: jax.Array  # Bias
  use_bias: bool
  eps: float
  
  def __init__(self, dim: int, use_bias:bool = True, eps:float = 1e-5):
    self.use_bias = use_bias
    self.gamma = jnp.ones(())
    self.delta = jnp.zeros(dim)
    self.eps = eps
    
  def lagrangian(self, x):
    """The integral of the standard LayerNorm, with the following twist: `gamma` is a scalar, not a vector of shape `dim` as in the original layernorm """
    D = x.shape[-1]
    xmeaned = x - x.mean(-1, keepdims=True)
    t1 = D * self.gamma * jnp.sqrt((1 / D * xmeaned**2).sum() + self.eps)
    if not self.use_bias: 
      return t1
    t2 = (self.delta * x).sum()
    return t1 + t2

  def g(self, x):
    """The manual derivative of the lagrangian. 
    
    You could compute this with autograd, but it is more efficient and clear to implement it directly
    """
    xmeaned = x - x.mean(-1, keepdims=True)
    v = self.gamma * (xmeaned) / jnp.sqrt((xmeaned**2).mean(-1, keepdims=True)+ self.eps)
    if self.use_bias:
        return v + self.delta
    return v

  def __call__(self, x):
    """An alias for the activation function `g`"""
    return self.g(x)
    
  def energy(self, x):
    """Compute the energy of this Lagrangian through the Legendre Transform"""
    return (self.g(x) * x).sum() - self.lagrangian(x)








    
class EnergyAttention(eqx.Module):
  """Our novel attention with energy

  Has only two learnable parameters, Wk and Wq
  """
  Wq: jax.Array
  Wk: jax.Array
  config: ETConfig = eqx.field(static=True)

  def __init__(self, key:jr.PRNGKey, config:ETConfig):
    kkey, qkey = jr.split(key)
    self.Wk = jr.normal(kkey, (config.n_heads, config.Y, config.D))
    self.Wq = jr.normal(qkey, (config.n_heads, config.Y, config.D))
    self.config = config

  def energy(self, g:jnp.ndarray):
    """Return the energy of the block. The update rule is autograd through this function"""
    beta = 1/jnp.sqrt(self.config.Y)
    K = jnp.einsum("kd,hzd->khz", g, self.Wk) # nKeys,nHeads,Y
    Q = jnp.einsum("qd,hzd->qhz", g, self.Wq) # nQueries,nHeads,Y
    A = jax.nn.logsumexp(beta * jnp.einsum("qhz,khz->hqk", Q, K), -1) # nHeads,nQueries,nKeys
    return -1/beta * A.sum()




#  pero lo que necesitamos ahora, es una forma mucho mas eficiente de implementar el transformer.....escrito sin ver el teclado




class HopfieldNetwork(eqx.Module):
  """ A simple Hopfield Network (we use ReLU as the activation function) replaces the MLP in traditional Transformers """
  Xi: jax.Array

  def __init__(self, key:jr.PRNGKey, config:ETConfig):
    nmems = int(config.scale_mems * config.D)
    self.Xi = jr.normal(key, (config.D, nmems))

  def energy(self, g:jnp.ndarray):
    """Return the Hopfield Network's energy"""
    hid = jnp.einsum("nd,dm->nm", g, self.Xi) # nTokens, nMems
    E = -0.5 * (jax.nn.relu(hid) ** 2).sum()
    return E











class EnergyTransformer(eqx.Module):
  """A simple wrapper class that sums the energies of the Hopfield Network and the Attention"""
  attn: EnergyAttention
  hn: HopfieldNetwork
  config: ETConfig = eqx.field(static=True)
  
  def __init__(self, key:jr.PRNGKey, config:ETConfig):
    attn_key, hn_key = jr.split(key)
    self.attn = EnergyAttention(attn_key, config)
    self.hn = HopfieldNetwork(hn_key, config)
    self.config = config

  def energy(self, g:jnp.ndarray):
    """Return the energy of the whole Transformer"""
    return self.attn.energy(g) + self.hn.energy(g)
  


























  # -----------------------------------------------------------------------------





# from https://mcbal.github.io/post/attention-as-energy-minimization-visualizing-energy-landscapes/


# what if this Perception module is an Optimization process???









def minimize_energy(
    energy_func,
    queries,
    keys,
    mask=None,
    step_size=1.0,
    num_steps=1,
    return_trajs=False,
):
    """Minimize energy function with respect to queries.
    
    Keeps track of energies and trajectories for logging and plotting.
    """
    out = defaultdict(list)
    out["queries"].append(queries)
    for _ in range(num_steps):
        energies = energy_func(queries, keys, mask=mask)
        grad_queries = torch.autograd.grad(
            energies, queries, grad_outputs=torch.ones_like(energies),
            create_graph=True,  # enables double backprop for optimization
        )[0]
        queries = queries - step_size * grad_queries
        out["queries"].append(queries)
        out["energies"].append(energies)
    out["energies"].append(energy_func(queries, keys, mask=mask))
    if return_trajs:
        return out
    return out["queries"][-1]







def hopfield_energy(state_patterns, stored_patterns, scale, mask=None):
    kinetic = 0.5 * einsum("b i d, b i d -> b i", state_patterns, state_patterns)
    scaled_dot_product = scale * einsum(
        "b i d, b j d -> b i j", state_patterns, stored_patterns
    )
    if mask is not None:
        max_neg_value = -torch.finfo(scaled_dot_product.dtype).max
        scaled_dot_product.masked_fill_(~mask, max_neg_value)
    potential = -(1.0 / scale) * torch.logsumexp(scaled_dot_product, dim=2)
    return kinetic + potential








class EnergyBasedAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=1,
        dim_head=2,
        scale=None,
        energy_func=None,
    ):
        super().__init__()

        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        # Linear transformations (queries, keys, output).
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.energy_func = energy_func if energy_func else hopfield_energy
        self.heads = heads
        self.scale = scale if scale is not None else dim_head ** -0.5

    def forward(
        self,
        x,
        context=None,
        mask=None,
        scale=None,
        bare_attn=False,
        step_size=1.0,
        num_steps=1,
        return_trajs=False,
    ):
        # Bare checks.
        if bare_attn:
            assert self.heads == 1, "only a single head when bare attention"
            if context is not None:
                assert (
                    x.shape[-1] == context.shape[-1]
                ), "query_dim/context_dim must match"

        scale = scale if scale is not None else self.scale
        context = context if context is not None else x

        q = x if bare_attn else self.to_q(x)
        k = context if bare_attn else self.to_k(context)

        h = self.heads
        q, k = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k))

        if mask is not None:
            mask = repeat(mask, "b j -> (b h) () j", h=h)

        # Minimize energy with respect to queries.
        outputs = minimize_energy(
            partial(self.energy_func, scale=scale),
            q,
            k,
            mask=mask,
            step_size=step_size,
            num_steps=num_steps,
            return_trajs=return_trajs,
        )
        if return_trajs:
            return outputs

        out = rearrange(outputs, "(b h) n d -> b n (h d)", h=h)
        return out if bare_attn or h == 1 else self.to_out(out)
    

