

# "SELECTIVE MEMORY" Mechanism



# Descentralized Dense Associative Memory Networks



# will be trained jointly with the Dynamics Model


# training objective: store only relevant State-Action pairs, this is, those that lead to "surprises" in the AGENT




import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List
import math

class ModernHopfield(nn.Module):
    """
    Modern Hopfield Network with continuous states and high memory capacity.
    
    Based on: "Hopfield Networks is All You Need" (Ramsauer et al., 2020)
    """
    
    def __init__(self, 
                 input_dim: int, 
                 memory_size: int,
                 beta: float = 1.0,
                 update_steps: int = 1,
                 dropout: float = 0.0):
        """
        Args:
            input_dim: Dimension of input patterns
            memory_size: Maximum number of stored patterns
            beta: Inverse temperature parameter
            update_steps: Number of update steps during retrieval
            dropout: Dropout rate for memory patterns
        """
        super().__init__()
        self.input_dim = input_dim
        self.memory_size = memory_size
        self.beta = beta
        self.update_steps = update_steps
        self.dropout = dropout
        
        # Memory matrix (stored patterns)
        self.memory = nn.Parameter(torch.zeros(memory_size, input_dim))
        
        # Initialize with orthogonal patterns for better memory capacity
        self._init_memory()
        
    def _init_memory(self):
        """Initialize memory with orthogonal patterns."""
        with torch.no_grad():
            # Create orthogonal matrix
            orthogonal = torch.empty(self.memory_size, self.input_dim)
            nn.init.orthogonal_(orthogonal)
            self.memory.data = orthogonal[:self.memory_size]
    
    def store(self, patterns: torch.Tensor):
        """
        Store patterns in memory.
        
        Args:
            patterns: Tensor of shape (batch_size, input_dim) or (input_dim,)
        """
        if patterns.dim() == 1:
            patterns = patterns.unsqueeze(0)
            
        batch_size = patterns.size(0)
        
        with torch.no_grad():
            # Replace oldest memories with new patterns
            if batch_size <= self.memory_size:
                self.memory.data[:batch_size] = patterns
            else:
                self.memory.data[:] = patterns[:self.memory_size]
    
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the energy of state x.
        
        Args:
            x: State tensor of shape (batch_size, input_dim)
            
        Returns:
            Energy values of shape (batch_size,)
        """
        # Apply dropout to memory during training
        if self.training and self.dropout > 0:
            memory = F.dropout(self.memory, p=self.dropout, training=True)
        else:
            memory = self.memory
            
        # Compute energy: -log(sum(exp(beta * memory @ x^T)))
        similarities = self.beta * torch.einsum('md,bd->bm', memory, x)
        max_sim = similarities.max(dim=-1, keepdim=True)[0]
        exp_sim = torch.exp(similarities - max_sim)
        energy = -max_sim.squeeze() - torch.log(exp_sim.sum(dim=-1))
        
        return energy
    
    def retrieve(self, 
                 query: torch.Tensor, 
                 update_steps: Optional[int] = None) -> torch.Tensor:
        """
        Retrieve pattern from memory using query.
        
        Args:
            query: Query tensor of shape (batch_size, input_dim)
            update_steps: Number of update steps (overrides default)
            
        Returns:
            Retrieved patterns of shape (batch_size, input_dim)
        """
        if update_steps is None:
            update_steps = self.update_steps
            
        x = query.clone()
        
        for step in range(update_steps):
            # Apply dropout to memory during training
            if self.training and self.dropout > 0:
                memory = F.dropout(self.memory, p=self.dropout, training=True)
            else:
                memory = self.memory
                
            # Compute attention weights
            similarities = self.beta * torch.einsum('md,bd->bm', memory, x)
            attention_weights = F.softmax(similarities, dim=-1)
            
            # Update state
            x = torch.einsum('bm,md->bd', attention_weights, memory)
            
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (same as retrieve)."""
        return self.retrieve(x)









class MultiHeadModernHopfield(nn.Module):
    """
    Multi-head Modern Hopfield Network for increased capacity.
    """
    
    def __init__(self,
                 input_dim: int,
                 memory_size: int,
                 num_heads: int = 8,
                 beta: float = 1.0,
                 update_steps: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.memory_size = memory_size
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        assert self.head_dim * num_heads == input_dim, "input_dim must be divisible by num_heads"
        
        self.heads = nn.ModuleList([
            ModernHopfield(
                input_dim=self.head_dim,
                memory_size=memory_size,
                beta=beta,
                update_steps=update_steps,
                dropout=dropout
            ) for _ in range(num_heads)
        ])
        
    def store(self, patterns: torch.Tensor):
        """Store patterns across all heads."""
        if patterns.dim() == 1:
            patterns = patterns.unsqueeze(0)
            
        # Split patterns into heads
        pattern_heads = patterns.chunk(self.num_heads, dim=-1)
        
        for head, pattern_head in zip(self.heads, pattern_heads):
            head.store(pattern_head)
    
    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve patterns using multi-head attention."""
        if query.dim() == 1:
            query = query.unsqueeze(0)
            
        # Split query into heads
        query_heads = query.chunk(self.num_heads, dim=-1)
        
        # Retrieve from each head
        output_heads = []
        for head, query_head in zip(self.heads, query_heads):
            output_heads.append(head.retrieve(query_head))
            
        # Concatenate results
        return torch.cat(output_heads, dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.retrieve(x)












# Example usage and demonstration
def demonstrate_hopfield():
    """Demonstrate the modern Hopfield network with a simple example."""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Parameters
    input_dim = 128
    memory_size = 100
    batch_size = 10
    
    # Create Hopfield network
    hopfield = ModernHopfield(
        input_dim=input_dim,
        memory_size=memory_size,
        beta=1.0,
        update_steps=3
    )
    
    # Generate some random patterns to store
    patterns = torch.randn(memory_size, input_dim)
    hopfield.store(patterns)
    
    print(f"Stored {memory_size} patterns of dimension {input_dim}")
    
    # Create queries (slightly noisy versions of stored patterns)
    query_indices = torch.randint(0, memory_size, (batch_size,))
    queries = patterns[query_indices] + 0.5 * torch.randn(batch_size, input_dim)
    
    print(f"Created {batch_size} noisy queries")
    
    # Retrieve patterns
    with torch.no_grad():
        retrieved = hopfield.retrieve(queries)
        
        # Compute similarity between retrieved and original patterns
        similarities = F.cosine_similarity(retrieved, patterns[query_indices], dim=-1)
        print(f"Average cosine similarity between retrieved and original: {similarities.mean():.4f}")
        
        # Compute energy of queries and retrieved patterns
        query_energy = hopfield.energy(queries)
        retrieved_energy = hopfield.energy(retrieved)
        print(f"Average energy - queries: {query_energy.mean():.4f}, retrieved: {retrieved_energy.mean():.4f}")
        
        # Test multi-head version
        print("\nTesting Multi-Head Modern Hopfield:")
        multi_hopfield = MultiHeadModernHopfield(
            input_dim=input_dim,
            memory_size=memory_size,
            num_heads=8
        )
        multi_hopfield.store(patterns)
        
        retrieved_multi = multi_hopfield.retrieve(queries)
        similarities_multi = F.cosine_similarity(retrieved_multi, patterns[query_indices], dim=-1)
        print(f"Multi-head average cosine similarity: {similarities_multi.mean():.4f}")






# Additional utility functions
def create_orthogonal_patterns(num_patterns: int, pattern_dim: int) -> torch.Tensor:
    """Create orthogonal patterns for optimal memory storage."""
    patterns = torch.empty(num_patterns, pattern_dim)
    nn.init.orthogonal_(patterns)
    return patterns

def add_noise(patterns: torch.Tensor, noise_level: float = 0.3) -> torch.Tensor:
    """Add Gaussian noise to patterns."""
    return patterns + noise_level * torch.randn_like(patterns)

if __name__ == "__main__":
    demonstrate_hopfield()























# ------------------------------------------------


# usage example:


# Create network
hopfield = ModernHopfield(input_dim=256, memory_size=1000)

# Store patterns
patterns = torch.randn(100, 256)  # 100 patterns of dim 256
hopfield.store(patterns)

# Retrieve with noisy query
noisy_query = patterns[0] + 0.4 * torch.randn(256)
retrieved = hopfield.retrieve(noisy_query)

# Check similarity
similarity = F.cosine_similarity(retrieved, patterns[0].unsqueeze(0))
print(f"Retrieval similarity: {similarity.item():.4f}")
















# ------------------------------------------



import numpy as np

class ModernHopfieldNetwork:
    """
    Implementación de una Red de Hopfield Moderna con memoria asociativa persistente.
    Usa una formulación key-value donde los keys definen el paisaje de energía
    y los values son lo que se recupera.
    """
    
    def __init__(self, key_dim, value_dim, beta=1.0):
        """
        Args:
            key_dim: Dimensión de los keys (patrones de búsqueda)
            value_dim: Dimensión de los values (contenido a recuperar)
            beta: Parámetro de temperatura inversa (mayor = más selectivo)
        """
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.beta = beta
        
        # Memoria persistente (inicialmente vacía)
        self.keys = []  # Lista de keys almacenados
        self.values = []  # Lista de values correspondientes
        
    def store(self, key, value):
        """
        Almacena un par key-value en la memoria persistente.
        
        Args:
            key: Vector de dimensión key_dim
            value: Vector de dimensión value_dim
        """
        key = np.array(key).reshape(-1)
        value = np.array(value).reshape(-1)
        
        assert key.shape[0] == self.key_dim, f"Key debe tener dimensión {self.key_dim}"
        assert value.shape[0] == self.value_dim, f"Value debe tener dimensión {self.value_dim}"
        
        # Normalizar el key para mejor estabilidad
        key = key / (np.linalg.norm(key) + 1e-8)
        
        self.keys.append(key)
        self.values.append(value)
        
    def energy(self, query):
        """
        Calcula la energía para un query dado.
        Energía baja = buen match con patrones almacenados.
        
        Args:
            query: Vector de dimensión key_dim
            
        Returns:
            Energía (escalar)
        """
        if len(self.keys) == 0:
            return 0.0
            
        query = np.array(query).reshape(-1)
        query = query / (np.linalg.norm(query) + 1e-8)
        
        K = np.array(self.keys)  # Shape: (num_memories, key_dim)
        
        # Producto punto entre query y todos los keys
        similarities = self.beta * K @ query  # Shape: (num_memories,)
        
        # Energía = -log-sum-exp de las similitudes
        max_sim = np.max(similarities)
        energy = -max_sim - np.log(np.sum(np.exp(similarities - max_sim)))
        
        return energy
    
    def retrieve(self, query):
        """
        Recupera un value basado en el query usando atención softmax.
        
        Args:
            query: Vector de dimensión key_dim
            
        Returns:
            retrieved_value: Vector de dimensión value_dim
            attention_weights: Pesos de atención sobre cada memoria
        """
        if len(self.keys) == 0:
            return np.zeros(self.value_dim), np.array([])
            
        query = np.array(query).reshape(-1)
        query = query / (np.linalg.norm(query) + 1e-8)
        
        K = np.array(self.keys)  # Shape: (num_memories, key_dim)
        V = np.array(self.values)  # Shape: (num_memories, value_dim)
        
        # Calcula similitudes
        similarities = self.beta * K @ query  # Shape: (num_memories,)
        
        # Softmax para obtener pesos de atención
        max_sim = np.max(similarities)
        exp_sim = np.exp(similarities - max_sim)
        attention_weights = exp_sim / np.sum(exp_sim)
        
        # Recupera value como combinación ponderada
        retrieved_value = V.T @ attention_weights  # Shape: (value_dim,)
        
        return retrieved_value, attention_weights
    
    def clear_memory(self):
        """Limpia toda la memoria persistente."""
        self.keys = []
        self.values = []
        
    def get_memory_size(self):
        """Retorna el número de patrones almacenados."""
        return len(self.keys)


# ============= EJEMPLO DE USO =============

if __name__ == "__main__":
    print("=== Memoria Asociativa Persistente con Modern Hopfield Network ===\n")
    
    # Crear red con keys de dimensión 5 y values de dimensión 3
    hopfield = ModernHopfieldNetwork(key_dim=5, value_dim=3, beta=2.0)
    
    # Almacenar algunos patrones asociativos
    print("Almacenando memorias...\n")
    
    # Memoria 1: "perro" -> información sobre perros
    key_perro = [1.0, 0.5, 0.2, 0.1, 0.0]
    value_perro = [1.0, 0.0, 0.0]  # Codificación: "animal doméstico"
    hopfield.store(key_perro, value_perro)
    print("Memoria 1: perro -> [1, 0, 0]")
    
    # Memoria 2: "gato" -> información sobre gatos  
    key_gato = [0.9, 0.6, 0.1, 0.0, 0.1]
    value_gato = [1.0, 0.0, 0.5]  # Similar pero ligeramente diferente
    hopfield.store(key_gato, value_gato)
    print("Memoria 2: gato -> [1, 0, 0.5]")
    
    # Memoria 3: "robot" -> información sobre robots
    key_robot = [0.0, 0.1, 0.2, 0.9, 1.0]
    value_robot = [0.0, 1.0, 0.0]  # Codificación: "máquina"
    hopfield.store(key_robot, value_robot)
    print("Memoria 3: robot -> [0, 1, 0]")
    
    print(f"\nMemoria total: {hopfield.get_memory_size()} patrones\n")
    
    # Recuperar memorias con queries
    print("=== Recuperación de Memorias ===\n")
    
    # Query exacto
    query1 = [1.0, 0.5, 0.2, 0.1, 0.0]
    retrieved1, weights1 = hopfield.retrieve(query1)
    energy1 = hopfield.energy(query1)
    print(f"Query 1 (perro exacto):")
    print(f"  Energía: {energy1:.4f}")
    print(f"  Pesos atención: {weights1}")
    print(f"  Value recuperado: {retrieved1}\n")
    
    # Query ruidoso (perro con ruido)
    query2 = [0.95, 0.45, 0.25, 0.15, 0.05]
    retrieved2, weights2 = hopfield.retrieve(query2)
    energy2 = hopfield.energy(query2)
    print(f"Query 2 (perro con ruido):")
    print(f"  Energía: {energy2:.4f}")
    print(f"  Pesos atención: {weights2}")
    print(f"  Value recuperado: {retrieved2}\n")
    
    # Query intermedio (mezcla perro-robot)
    query3 = [0.5, 0.3, 0.2, 0.5, 0.5]
    retrieved3, weights3 = hopfield.retrieve(query3)
    energy3 = hopfield.energy(query3)
    print(f"Query 3 (mezcla perro-robot):")
    print(f"  Energía: {energy3:.4f}")
    print(f"  Pesos atención: {weights3}")
    print(f"  Value recuperado: {retrieved3}\n")
    
    # Query exacto robot
    query4 = [0.0, 0.1, 0.2, 0.9, 1.0]
    retrieved4, weights4 = hopfield.retrieve(query4)
    energy4 = hopfield.energy(query4)
    print(f"Query 4 (robot exacto):")
    print(f"  Energía: {energy4:.4f}")
    print(f"  Pesos atención: {weights4}")
    print(f"  Value recuperado: {retrieved4}\n")


























# -----------------------------------------------------------





# Cell
class HopfieldTransformer(nn.Module):
    """Full energy transformer"""
    tokdim: int = 768
    nheads: int = 12
    kspace_dim:int = 64
    hidden_ratio:float = 4.
    attn_beta_init: Optional[float] = None
    use_biases_attn:bool = False
    use_biases_chn:bool = False
    param_dtype:Any = jnp.float32

    def setup(self):
        self.attn = MultiheadAttention(
            tokdim=self.tokdim,
            nheads=self.nheads,
            kspace_dim=self.kspace_dim,
            use_bias=self.use_biases_attn,
            beta_init=self.attn_beta_init,
            param_dtype=self.param_dtype
        )
        self.chn = CHNReLU(tokdim=self.tokdim, hidden_ratio=self.hidden_ratio, param_dtype=self.param_dtype)

    def energy(self, g:jnp.ndarray):
        attn_energy = self.attn.energy(g)
        chn_energy = self.chn.energy(g)
        return attn_energy + chn_energy

    def manual_grad(self, g:jnp.ndarray):
        return self.attn.manual_grad(g) + self.chn.manual_grad(g)

    def energy_and_grad(self, g:jnp.ndarray):
        return jax.value_and_grad(self.energy)(g)