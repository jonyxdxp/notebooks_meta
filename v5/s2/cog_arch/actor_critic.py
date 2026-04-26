

# value function and policy network for actor and critic modules in the Cognitive Architecture







# (from td-mpc2 architecture)




# ---------------------------------------------------------------------------
# Value Network (Two-headed Critic):  (z, a)  →  Q
# ---------------------------------------------------------------------------
 
class ValueNetwork(nn.Module):
    """
    Two-headed Q-function (TD3 / SAC style) to reduce overestimation bias.
 
    Q1, Q2 = V(z, a)
 
    During target computation: Q_target = min(Q1, Q2)
 
    Args:
        latent_dim: Size of z.
        action_dim: Dimensionality of the action space.
        hidden_dim: Width of the hidden layers.
        num_layers: Number of hidden layers.
        task_dim:   Task embedding dimension (0 = single-task).
    """
 
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        task_dim: int = 0,
    ):
        super().__init__()
        in_dim = latent_dim + action_dim + task_dim
 
        # Two independent Q-heads
        self.Q1 = mlp(in_dim, hidden_dim, 1, num_layers=num_layers, act=SimNorm(8))
        self.Q2 = mlp(in_dim, hidden_dim, 1, num_layers=num_layers, act=SimNorm(8))
 
    def forward(
        self,
        z: Tensor,
        a: Tensor,
        task_emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            z:        (B, latent_dim)
            a:        (B, action_dim)
            task_emb: (B, task_dim) or None
        Returns:
            q1, q2: each (B, 1)
        """
        x = torch.cat([z, a] if task_emb is None else [z, a, task_emb], dim=-1)
        return self.Q1(x), self.Q2(x)
 
    def min_q(
        self,
        z: Tensor,
        a: Tensor,
        task_emb: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns min(Q1, Q2) — used for TD targets."""
        q1, q2 = self.forward(z, a, task_emb)
        return torch.min(q1, q2)
 
 















# ---------------------------------------------------------------------------
# Policy Prior (Actor):  z  →  a
# ---------------------------------------------------------------------------
 
class PolicyPrior(nn.Module):
    """
    Amortized policy that maps latent state z to a deterministic action mean.
 
    Used as:
      1. Warm-start for trajectory optimization (planning).
      2. Standalone reactive policy (zero-shot, no planning).
 
    Trained via MSE regression onto the planner's selected action a*.
 
    Output is tanh-squashed to [-1, 1].
 
    Args:
        latent_dim: Size of z.
        action_dim: Dimensionality of the action space.
        hidden_dim: Width of the hidden layers.
        num_layers: Number of hidden layers.
        task_dim:   Task embedding dimension (0 = single-task).
    """
 
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        task_dim: int = 0,
    ):
        super().__init__()
        in_dim = latent_dim + task_dim
 
        self.net = mlp(
            in_dim, hidden_dim, action_dim,
            num_layers=num_layers,
            act=SimNorm(8),
            out_act=None,           # tanh applied separately for clarity
        )
 
    def forward(
        self,
        z: Tensor,
        task_emb: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            z:        (B, latent_dim)
            task_emb: (B, task_dim) or None
        Returns:
            a: (B, action_dim)  in [-1, 1]
        """
        x = z if task_emb is None else torch.cat([z, task_emb], dim=-1)
        return torch.tanh(self.net(x))
 
    def bc_loss(self, z: Tensor, a_target: Tensor, task_emb: Optional[Tensor] = None) -> Tensor:
        """
        Behavioral cloning loss: MSE between predicted action and planner target.
 
        Args:
            z:        (B, latent_dim)
            a_target: (B, action_dim)  — first action from trajectory optimizer
            task_emb: (B, task_dim) or None
        Returns:
            scalar loss
        """
        a_pred = self.forward(z, task_emb)
        return F.mse_loss(a_pred, a_target)
 
















    def q_values(
        self, z: Tensor, a: Tensor, task_ids: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        return self.value(z, a, self.get_task_emb(task_ids))
 




    def act(self, obs: Tensor, task_ids: Optional[Tensor] = None) -> Tensor:
        """Single-step reactive action (no planning)."""
        z = self.encode(obs, task_ids)
        return self.policy(z, self.get_task_emb(task_ids))
