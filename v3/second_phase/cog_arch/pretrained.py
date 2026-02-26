
def setup_stage2(pretrained_ebt: EBTAdaLN, action_dim: int, lr_body=1e-5, lr_action=1e-3):
    """
    Configura el fine-tuning causal con tasas de aprendizaje diferenciadas.
    """
    model = EBTAdaLN_Causal.from_pretrained(pretrained_ebt, action_dim)

    # El action_projector es nuevo → puede aprender más rápido
    # El resto heredó conocimiento → learning rate pequeño para no olvidar
    optimizer = torch.optim.AdamW([
        {"params": model.action_projector.parameters(), "lr": lr_action},
        {"params": model.layers.parameters(),           "lr": lr_body},
        {"params": model.norm.parameters(),             "lr": lr_body},
        {"params": model.final_layer.parameters(),      "lr": lr_body},
    ])

    return model, optimizer


# Loop de entrenamiento etapa 2 (encoder congelado)
def train_step_stage2(context_encoder, target_encoder, predictor, optimizer, x_prev, x_curr, action):
    """
    x_prev: (B, seq_len)  — observación t-1
    x_curr: (B, seq_len)  — observación t
    action: (B, action_dim)
    """
    with torch.no_grad():
        z_context = context_encoder(x_prev)   # (B, S, D) — encoder congelado
        z_target  = target_encoder(x_curr)    # (B, S, D) — encoder congelado (EMA o mismo)

    # El dual-stream del EBTAdaLN espera [z_context || z_pred_init]
    # z_pred_init puede ser z_context desplazado, ruido, o ceros
    z_pred_init = torch.zeros_like(z_context)
    embeddings = torch.cat([z_context, z_pred_init], dim=1)  # (B, 2S, D)

    # Predicción condicionada en acción
    pred_energies = predictor(embeddings, action)  # (B, S, 1) o lo que uses como loss

    # Loss en espacio de embeddings (L2 sobre representaciones normalizadas)
    z_pred = pred_energies  # adaptar según tu output head
    loss = F.mse_loss(
        F.normalize(z_pred, dim=-1),
        F.normalize(z_target, dim=-1).detach()
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss















# --------------------------------------------------------












class Stage2Trainer:
    def __init__(self, context_encoder, target_encoder, predictor, action_dim, hidden_size):
        
        # Encoders completamente congelados
        self.context_encoder = context_encoder
        self.target_encoder  = target_encoder
        for p in self.context_encoder.parameters():
            p.requires_grad = False
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Proyector de acción: único módulo nuevo
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # Predictor: cargado desde etapa 1
        # Solo adaLN + cross-attn entrenan (o todo con lr diferencial)
        self.predictor = predictor

    def training_step(self, x_prev, x_curr, action):
        """
        x_prev:  (B, T) — estado t-1
        x_curr:  (B, T) — estado t  (ground truth)
        action:  (B, action_dim)
        """

        # 1. Encoders congelados — no_grad
        with torch.no_grad():
            ctx_embeds    = self.context_encoder(x_prev)  # (B, T, D)
            target_embeds = self.target_encoder(x_curr)   # (B, T, D)

        # 2. Proyectar acción
        action_cond = self.action_proj(action)  # (B, D)

        # 3. Predictor intenta predecir target_embeds dado contexto + acción
        B, T = x_curr.shape
        target_positions = torch.arange(T, device=x_curr.device)
        target_positions = target_positions.unsqueeze(0).expand(B, -1)  # (B, T)

        pred_embeds = self.predictor(
            context_embeds  = ctx_embeds,
            target_positions = target_positions,
            condition        = action_cond
        )  # (B, T, D)

        # 4. Loss JEPA: similaridad coseno en espacio latente
        # target_encoder usa EMA — no hay colapso
        loss = self.jepa_loss(pred_embeds, target_embeds)
        return loss

    def jepa_loss(self, pred, target):
        """
        Loss estándar JEPA: L2 en espacio normalizado
        equivalente a maximizar cosine similarity
        """
        pred   = F.normalize(pred,   dim=-1)
        target = F.normalize(target, dim=-1).detach()  # stop-gradient en target
        return 2 - 2 * (pred * target).sum(dim=-1).mean()