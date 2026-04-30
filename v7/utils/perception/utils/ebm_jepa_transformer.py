import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import Optional, Callable
from functools import partial

# ============================================================================
# Configuration
# ============================================================================

class JEPAConfig:
    """Configuration for JEPA-EBM architecture"""
    # Data dimensions
    input_dim: int = 128        # Raw input dimension
    embed_dim: int = 256        # Embedding space dimension
    
    # Transformer predictor
    n_heads: int = 8
    head_dim: int = 32
    n_layers: int = 6
    
    # Energy-based settings
    energy_steps: int = 5       # Gradient steps for energy minimization
    step_size: float = 0.1
    beta: float = 1.0           # Temperature
    
    # Training
    mask_ratio: float = 0.3     # For masked prediction
    ema_decay: float = 0.996    # For target encoder (like I-JEPA)


# ============================================================================
# Energy-Based Attention (Hopfield-style)
# ============================================================================

class EnergyAttention(eqx.Module):
    """Energy-based attention as perception module"""
    Wq: jax.Array
    Wk: jax.Array
    Wo: jax.Array  # Output projection
    config: JEPAConfig = eqx.field(static=True)
    
    def __init__(self, key: jr.PRNGKey, config: JEPAConfig):
        keys = jr.split(key, 3)
        d = config.n_heads * config.head_dim
        
        self.Wq = jr.normal(keys[0], (config.n_heads, config.head_dim, config.embed_dim)) * 0.02
        self.Wk = jr.normal(keys[1], (config.n_heads, config.head_dim, config.embed_dim)) * 0.02
        self.Wo = jr.normal(keys[2], (config.embed_dim, d)) * 0.02
        self.config = config
    
    def energy(self, q: jnp.ndarray, k: jnp.ndarray) -> float:
        """
        Compute energy for query-key configuration
        Args:
            q: queries [n_queries, n_heads, head_dim]
            k: keys [n_keys, n_heads, head_dim]
        Returns:
            energy: scalar
        """
        beta = self.config.beta / jnp.sqrt(self.config.head_dim)
        # Hopfield energy: -1/β * sum(logsumexp(β * Q @ K^T))
        scores = jnp.einsum("qhd,khd->hqk", q, k)  # [n_heads, n_queries, n_keys]
        log_partition = jax.nn.logsumexp(beta * scores, axis=-1)  # [n_heads, n_queries]
        energy = -1.0 / beta * jnp.sum(log_partition)
        return energy
    
    def minimize_energy(self, q_init: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
        """
        Minimize energy w.r.t queries using gradient descent
        Args:
            q_init: initial queries [n_queries, n_heads, head_dim]
            k: keys (fixed) [n_keys, n_heads, head_dim]
        Returns:
            q_final: optimized queries [n_queries, n_heads, head_dim]
        """
        q = q_init
        energy_fn = lambda q: self.energy(q, k)
        
        for _ in range(self.config.energy_steps):
            grad = jax.grad(energy_fn)(q)
            q = q - self.config.step_size * grad
        
        return q
    
    def __call__(self, context: jnp.ndarray, target_init: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            context: [n_context, embed_dim] - context tokens
            target_init: [n_target, embed_dim] - initial target queries
        Returns:
            output: [n_target, embed_dim] - predicted targets
        """
        # Project to multi-head space
        K = jnp.einsum("kd,hzd->khz", context, self.Wk)      # [n_context, n_heads, head_dim]
        Q_init = jnp.einsum("qd,hzd->qhz", target_init, self.Wq)  # [n_target, n_heads, head_dim]
        
        # Minimize energy to find optimal queries
        Q_opt = self.minimize_energy(Q_init, K)
        
        # Project back to embedding dimension
        Q_flat = Q_opt.reshape(Q_opt.shape[0], -1)  # [n_target, n_heads * head_dim]
        output = Q_flat @ self.Wo.T  # [n_target, embed_dim]
        
        return output


# ============================================================================
# Transformer Predictor (Stack of Energy Attention + FFN)
# ============================================================================

class FFN(eqx.Module):
    """Feed-forward network"""
    W1: jax.Array
    W2: jax.Array
    
    def __init__(self, key: jr.PRNGKey, dim: int, hidden_dim: int = None):
        keys = jr.split(key, 2)
        hidden_dim = hidden_dim or 4 * dim
        self.W1 = jr.normal(keys[0], (hidden_dim, dim)) * 0.02
        self.W2 = jr.normal(keys[1], (dim, hidden_dim)) * 0.02
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return (jax.nn.gelu(x @ self.W1.T)) @ self.W2.T


class DynamicTanh(eqx.Module):
    """Dynamic Tanh normalization (replacement for LayerNorm)"""
    alpha: jax.Array
    
    def __init__(self, key: jr.PRNGKey = None):
        self.alpha = jnp.ones((1,))
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.tanh(self.alpha * x)


class EnergyTransformerLayer(eqx.Module):
    """Single transformer layer with energy-based attention"""
    attention: EnergyAttention
    ffn: FFN
    norm1: DynamicTanh
    norm2: DynamicTanh
    
    def __init__(self, key: jr.PRNGKey, config: JEPAConfig):
        keys = jr.split(key, 4)
        self.attention = EnergyAttention(keys[0], config)
        self.ffn = FFN(keys[1], config.embed_dim)
        self.norm1 = DynamicTanh(keys[2])
        self.norm2 = DynamicTanh(keys[3])
    
    def __call__(self, context: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            context: [n_context, embed_dim]
            target: [n_target, embed_dim]
        Returns:
            target_out: [n_target, embed_dim]
        """
        # Energy-based cross-attention
        target = target + self.attention(context, self.norm1(target))
        
        # Feed-forward
        target = target + self.ffn(self.norm2(target))
        
        return target


class EnergyTransformerPredictor(eqx.Module):
    """Stack of energy transformer layers - the Perception Module"""
    layers: list
    config: JEPAConfig = eqx.field(static=True)
    
    def __init__(self, key: jr.PRNGKey, config: JEPAConfig):
        keys = jr.split(key, config.n_layers)
        self.layers = [
            EnergyTransformerLayer(keys[i], config)
            for i in range(config.n_layers)
        ]
        self.config = config
    
    def __call__(self, context_embed: jnp.ndarray, target_embed: jnp.ndarray) -> jnp.ndarray:
        """
        Predict target embeddings from context embeddings
        Args:
            context_embed: [n_context, embed_dim] - encoded context
            target_embed: [n_target, embed_dim] - initial target (e.g., learnable queries)
        Returns:
            predicted_target: [n_target, embed_dim]
        """
        x = target_embed
        for layer in self.layers:
            x = layer(context_embed, x)
        return x


# ============================================================================
# Encoder (shared for context and target)
# ============================================================================

class Encoder(eqx.Module):
    """Simple encoder: input -> embeddings"""
    embed: jax.Array
    proj: jax.Array
    
    def __init__(self, key: jr.PRNGKey, config: JEPAConfig):
        keys = jr.split(key, 2)
        # Positional embeddings (assume max 512 positions)
        self.embed = jr.normal(keys[0], (512, config.embed_dim)) * 0.02
        self.proj = jr.normal(keys[1], (config.embed_dim, config.input_dim)) * 0.02
    
    def __call__(self, x: jnp.ndarray, positions: jnp.ndarray = None) -> jnp.ndarray:
        """
        Args:
            x: [n_tokens, input_dim]
            positions: [n_tokens] - position indices
        Returns:
            embeddings: [n_tokens, embed_dim]
        """
        # Project input
        h = x @ self.proj.T  # [n_tokens, embed_dim]
        
        # Add positional embeddings
        if positions is not None:
            h = h + self.embed[positions]
        
        return h


# ============================================================================
# Complete JEPA-EBM Architecture
# ============================================================================

class JEPA_EBM(eqx.Module):
    """
    Joint Embedding Predictive Architecture with Energy-Based Perception
    Following Yann LeCun's cognitive architecture vision
    """
    context_encoder: Encoder
    target_encoder: Encoder  # EMA updated
    predictor: EnergyTransformerPredictor
    learnable_queries: jax.Array  # Queries for masked positions
    config: JEPAConfig = eqx.field(static=True)
    
    def __init__(self, key: jr.PRNGKey, config: JEPAConfig):
        keys = jr.split(key, 4)
        
        # Encoders (target encoder will be EMA of context encoder)
        self.context_encoder = Encoder(keys[0], config)
        self.target_encoder = Encoder(keys[1], config)
        
        # Energy-based predictor (perception module)
        self.predictor = EnergyTransformerPredictor(keys[2], config)
        
        # Learnable queries for masked positions
        max_masked = int(512 * config.mask_ratio) + 1
        self.learnable_queries = jr.normal(keys[3], (max_masked, config.embed_dim)) * 0.02
        
        self.config = config
    
    def energy(self, 
               context: jnp.ndarray, 
               target: jnp.ndarray,
               context_mask: jnp.ndarray,
               target_mask: jnp.ndarray) -> float:
        """
        Compute JEPA energy: prediction error in embedding space
        
        Args:
            context: [n_tokens, input_dim] - visible context
            target: [n_tokens, input_dim] - target to predict
            context_mask: [n_tokens] - which positions are context (1) or masked (0)
            target_mask: [n_tokens] - which positions to predict
        
        Returns:
            energy: scalar - ||predicted_embeddings - target_embeddings||²
        """
        # Get context positions
        context_positions = jnp.where(context_mask)[0]
        target_positions = jnp.where(target_mask)[0]
        
        # Encode context with context encoder
        context_embed = self.context_encoder(
            context[context_positions], 
            context_positions
        )
        
        # Encode target with target encoder (no gradients through this)
        target_embed = jax.lax.stop_gradient(
            self.target_encoder(target[target_positions], target_positions)
        )
        
        # Initialize target queries (learnable)
        n_targets = len(target_positions)
        target_queries = self.learnable_queries[:n_targets]
        
        # Predict target embeddings from context using energy-based predictor
        predicted_embed = self.predictor(context_embed, target_queries)
        
        # Energy is prediction error (L2 distance in embedding space)
        energy = jnp.sum((predicted_embed - target_embed) ** 2)
        
        return energy
    
    def __call__(self,
                 x: jnp.ndarray,
                 context_mask: jnp.ndarray,
                 target_mask: jnp.ndarray) -> tuple:
        """
        Forward pass for training
        
        Args:
            x: [n_tokens, input_dim] - full input sequence
            context_mask: [n_tokens] - context positions
            target_mask: [n_tokens] - target positions to predict
        
        Returns:
            energy: scalar
            predicted_embed: [n_targets, embed_dim]
            target_embed: [n_targets, embed_dim]
        """
        energy = self.energy(x, x, context_mask, target_mask)
        
        # Also return embeddings for analysis
        context_positions = jnp.where(context_mask)[0]
        target_positions = jnp.where(target_mask)[0]
        
        context_embed = self.context_encoder(x[context_positions], context_positions)
        target_embed = self.target_encoder(x[target_positions], target_positions)
        
        n_targets = len(target_positions)
        target_queries = self.learnable_queries[:n_targets]
        predicted_embed = self.predictor(context_embed, target_queries)
        
        return energy, predicted_embed, target_embed


# ============================================================================
# Training utilities
# ============================================================================

def update_ema(source: eqx.Module, target: eqx.Module, decay: float) -> eqx.Module:
    """Update target encoder with EMA of source encoder"""
    def update_array(src, tgt):
        if isinstance(src, jnp.ndarray) and isinstance(tgt, jnp.ndarray):
            return decay * tgt + (1 - decay) * src
        return tgt
    
    return jax.tree_map(update_array, source, target)


def create_masks(key: jr.PRNGKey, seq_len: int, mask_ratio: float):
    """Create random context and target masks for JEPA training"""
    n_mask = int(seq_len * mask_ratio)
    indices = jr.permutation(key, seq_len)
    
    target_indices = indices[:n_mask]
    context_indices = indices[n_mask:]
    
    context_mask = jnp.zeros(seq_len, dtype=bool).at[context_indices].set(True)
    target_mask = jnp.zeros(seq_len, dtype=bool).at[target_indices].set(True)
    
    return context_mask, target_mask


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    # Configuration
    config = JEPAConfig()
    key = jr.PRNGKey(0)
    
    # Create model
    model = JEPA_EBM(key, config)
    
    # Dummy input
    key, subkey = jr.split(key)
    x = jr.normal(subkey, (64, config.input_dim))  # 64 tokens
    
    # Create masks
    key, subkey = jr.split(key)
    context_mask, target_mask = create_masks(subkey, 64, config.mask_ratio)
    
    # Forward pass
    energy, pred_embed, target_embed = model(x, context_mask, target_mask)
    
    print(f"Energy: {energy:.4f}")
    print(f"Predicted shape: {pred_embed.shape}")
    print(f"Target shape: {target_embed.shape}")
    print(f"Context tokens: {context_mask.sum()}")
    print(f"Target tokens: {target_mask.sum()}")