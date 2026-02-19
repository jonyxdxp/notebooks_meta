

# byte encoder as in "BLT" (or "megabyte/hnet/etc")









"""
Byte Latent Transformer (BLT) - Proof of Concept Implementation
Based on: "Byte Latent Transformer: Patches Scale Better Than Tokens" (Meta, 2024)

Key Components:
- Entropy-based dynamic patching (no fixed tokenizer)
- Local Encoder: bytes → patches via cross-attention aggregation
- Global Transformer: processes patch-level representations  
- Local Decoder: patches → bytes via cross-attention disaggregation
- Hash n-gram byte embeddings for local models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass





@dataclass
class BLTConfig:
    """Configuration for Byte Latent Transformer"""
    # Model dimensions
    hidden_size: int = 768
    num_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 8192
    
    # Architecture depths
    encoder_layers: int = 2        # Local encoder depth (lightweight)
    global_layers: int = 12        # Global transformer depth (heavy)
    decoder_layers: int = 2        # Local decoder depth (lightweight)
    entropy_model_layers: int = 4  # Entropy prediction model
    
    # Patching configuration
    max_patch_size: int = 8        # Maximum bytes per patch
    min_patch_size: int = 1        # Minimum bytes per patch
    entropy_threshold: float = 2.0 # Threshold for patch boundary
    
    # Hash n-gram config
    num_hashes: int = 50000        # Hash embedding vocabulary
    ngram_sizes: Tuple[int, ...] = (3, 4, 5)  # N-gram window sizes
    
    # Attention config
    attention_dropout: float = 0.0
    rope_theta: float = 10000.0






class RoPE(nn.Module):
    """Rotary Position Embeddings (used in Global Transformer)"""
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()






def apply_rope(q, k, cos, sin):
    """Apply rotary embeddings to queries and keys"""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed







class HashNGramEmbedding(nn.Module):
    """
    Hash-based n-gram embeddings for byte sequences.
    Augments byte embeddings with n-gram context (Section 3.2.1 of paper)
    """
    def __init__(self, config: BLTConfig):
        super().__init__()
        self.ngram_sizes = config.ngram_sizes
        self.num_hashes = config.num_hashes
        
        # Separate embedding table for each n-gram size
        self.hash_embeddings = nn.ModuleDict({
            str(n): nn.Embedding(config.num_hashes, config.hidden_size)
            for n in config.ngram_sizes
        })
        
    def hash_ngrams(self, bytes_seq: torch.Tensor, n: int) -> torch.Tensor:
        """
        Simple rolling hash for byte n-grams
        bytes_seq: (batch, seq_len)
        returns: (batch, seq_len) hashed values
        """
        batch_size, seq_len = bytes_seq.shape
        
        if seq_len < n:
            return torch.zeros_like(bytes_seq)
        
        # Simple polynomial rolling hash: h = sum(byte * prime^i) % num_hashes
        hashes = torch.zeros_like(bytes_seq)
        prime = 31
        
        for i in range(n):
            if i == 0:
                hashes = bytes_seq[:, i:seq_len-n+i+1].long()
            else:
                hashes = (hashes * prime + bytes_seq[:, i:seq_len-n+i+1].long()) % self.num_hashes
        
        # Pad beginning with zeros for positions where n-gram doesn't fit
        padding = torch.zeros(batch_size, n-1, dtype=torch.long, device=bytes_seq.device)
        hashes = torch.cat([padding, hashes], dim=1)
        
        return hashes
    
    def forward(self, bytes_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bytes_seq: (batch, seq_len) byte values (0-255)
        Returns:
            ngram_embeds: (batch, seq_len, hidden_size) aggregated n-gram features
        """
        batch_size, seq_len = bytes_seq.shape
        ngram_embeds = torch.zeros(
            batch_size, seq_len, next(iter(self.hash_embeddings.values())).embedding_dim,
            device=bytes_seq.device
        )
        
        # Aggregate embeddings from different n-gram sizes
        for n in self.ngram_sizes:
            hashes = self.hash_ngrams(bytes_seq, n)  # (batch, seq_len)
            embeds = self.hash_embeddings[str(n)](hashes)  # (batch, seq_len, hidden)
            ngram_embeds = ngram_embeds + embeds  # Sum aggregation
        
        return ngram_embeds








class CrossAttention(nn.Module):
    """Cross-attention layer for byte↔patch interaction (Section 3.2.2)"""
    def __init__(self, config: BLTConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,           # Query source (batch, q_len, hidden)
        encoder_hidden_states: torch.Tensor,   # Key/Value source (batch, kv_len, hidden)
        attention_mask: Optional[torch.Tensor] = None
    ):
        batch_size, q_len, _ = hidden_states.shape
        kv_len = encoder_hidden_states.shape[1]
        
        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(encoder_hidden_states)
        value = self.v_proj(encoder_hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, -1)
        return self.o_proj(attn_output)














class LocalEncoderLayer(nn.Module):
    """
    Local Encoder Layer: Self-attention on bytes + Cross-attention to patches
    Uses block-causal attention mask with local window
    """
    def __init__(self, config: BLTConfig, is_final_layer: bool = False):
        super().__init__()
        self.is_final_layer = is_final_layer
        
        # Self-attention on bytes (local window, block-causal)
        self.self_attn = nn.MultiheadAttention(
            config.hidden_size, 
            config.num_heads, 
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # Cross-attention: bytes attend to patch representations
        # Only used in final layer to produce patch embeddings
        if is_final_layer:
            self.cross_attn = CrossAttention(config)
            self.cross_attn_norm = nn.LayerNorm(config.hidden_size)
        
        # Feed-forward network (SwiGLU)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size * 2),
            nn.SiLU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,           # (batch, byte_len, hidden)
        patch_queries: Optional[torch.Tensor] = None,  # (batch, num_patches, hidden)
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None
    ):
        # Self-attention on bytes
        normed = self.norm1(hidden_states)
        attn_out, _ = self.self_attn(normed, normed, normed, attn_mask=self_attn_mask)
        hidden_states = hidden_states + attn_out
        
        # Cross-attention to patches (only in final layer)
        if self.is_final_layer and patch_queries is not None:
            normed = self.norm2(hidden_states)
            cross_out = self.cross_attn(patch_queries, normed, cross_attn_mask)
            patch_embeddings = self.cross_attn_norm(cross_out)
            return patch_embeddings
        
        # FFN for byte representations
        hidden_states = hidden_states + self.ffn(self.norm2(hidden_states))
        return hidden_states




class LocalEncoder(nn.Module):
    """
    Local Encoder (Section 3.2):
    - Lightweight transformer processing raw bytes
    - Hash n-gram embeddings for byte context
    - Cross-attention aggregation into patch representations
    """
    def __init__(self, config: BLTConfig):
        super().__init__()
        self.config = config
        
        # Byte embedding (256 possible byte values)
        self.byte_embedding = nn.Embedding(256, config.hidden_size)
        
        # Hash n-gram embeddings (Section 3.2.1)
        self.ngram_embeddings = HashNGramEmbedding(config)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            LocalEncoderLayer(config, is_final_layer=(i == config.encoder_layers - 1))
            for i in range(config.encoder_layers)
        ])
        
        # Initialize patch queries (learned or pooled from bytes)
        self.patch_query_init = "pool"  # 'pool' or 'learned'
        
    def create_local_attention_mask(self, seq_len: int, window_size: int = 512, device: torch.device = None):
        """Create block-causal attention mask for local attention"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1) * float('-inf')
        # Apply windowing: bytes can only attend to previous `window_size` bytes
        window_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=-window_size)
        mask = mask.masked_fill(window_mask == 0, float('-inf'))
        return mask
    
    def forward(
        self, 
        byte_seq: torch.Tensor,           # (batch, byte_len)
        patch_boundaries: torch.Tensor,   # (batch, num_patches) indices where patches start
        num_patches: int
    ):
        """
        Encode bytes into patch representations
        
        Args:
            byte_seq: Raw byte sequences (0-255)
            patch_boundaries: Indices [0, p1, p2, ...] where each patch starts
            num_patches: Total number of patches
        """
        batch_size, byte_len = byte_seq.shape
        
        # Byte embeddings + hash n-gram features
        byte_embeds = self.byte_embedding(byte_seq.long())  # (batch, byte_len, hidden)
        ngram_embeds = self.ngram_embeddings(byte_seq)      # (batch, byte_len, hidden)
        hidden_states = byte_embeds + ngram_embeds
        
        # Create attention mask (local window + causal)
        attn_mask = self.create_local_attention_mask(byte_len, device=byte_seq.device)
        
        # Pass through encoder layers (all except last process bytes)
        for layer in self.layers[:-1]:
            hidden_states = layer(hidden_states, self_attn_mask=attn_mask)
        
        # Final layer: prepare patch queries by max-pooling bytes in each patch
        # (Paper mentions max-pooling or learned queries for cross-attention init)
        patch_queries = []
        for b in range(batch_size):
            batch_queries = []
            for i in range(num_patches):
                start_idx = patch_boundaries[b, i].item()
                end_idx = patch_boundaries[b, i+1].item() if i+1 < num_patches else byte_len
                
                # Max pool over bytes in this patch
                patch_bytes = hidden_states[b, start_idx:end_idx]  # (patch_len, hidden)
                pooled = patch_bytes.max(dim=0)[0]  # (hidden,)
                batch_queries.append(pooled)
            patch_queries.append(torch.stack(batch_queries))
        
        patch_queries = torch.stack(patch_queries)  # (batch, num_patches, hidden)
        
        # Final layer with cross-attention to produce patch embeddings
        patch_embeddings = self.layers[-1](
            hidden_states, 
            patch_queries=patch_queries,
            self_attn_mask=attn_mask
        )
        
        return patch_embeddings