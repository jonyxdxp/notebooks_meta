
# LATENT-VARIABLE REGULARIZED (VARIATIONAL) AUTOENCODER (LV-RAE)


# Unconditional "Latent-Variable Generative Energy-Based Model" (LVGEBM) 
# with Amortized Inference (training an Encoder to give an approximate solution (the target, that will serve as the "reference" conditioning input) to the inference optimization problem)







# v5/vae/train_vae.py
import sys, torch
from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/content/notebooks_meta')
from v5.vae.model import LatentVAE, vae_loss

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = Path('/content/drive/MyDrive/metanet/v5/vae')
SAVE_DIR.mkdir(parents=True, exist_ok=True)







# v5/vae/model.py

class LatentVAE(nn.Module):
    """
    VAE trained on S1 representations.
    Learns a smooth, navigable b-space over S1's latent space.
    
    Input  : z ∈ ℝ^256  (S1 mean-pooled representation)
    Latent : b ∈ ℝ^64   (navigable coordinate in b-space)
    Output : z'∈ ℝ^256  (reconstructed S1 representation)
    """

    def __init__(
        self,
        input_dim  : int = 256,
        hidden_dim : int = 512,
        latent_dim : int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # ── Encoder z → b ─────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.mu_head     = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # ── Decoder b → z' ────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, z):
        h      = self.encoder(z)
        mu     = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-10, 2)   # numerical stability
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def decode(self, b):
        return self.decoder(b)

    def forward(self, z):
        mu, logvar = self.encode(z)
        b          = self.reparametrize(mu, logvar)
        z_recon    = self.decode(b)
        return z_recon, mu, logvar, b


def vae_loss(z, z_recon, mu, logvar, beta=1.0):
    recon = F.mse_loss(z_recon, z, reduction='mean')
    kl    = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
    return recon + beta * kl, recon, kl














# ── Load S1 representations ───────────────────────────────────────────────────
data    = torch.load('/content/drive/MyDrive/metanet/v5/s1_reps_val.pt')
all_z   = data['z']                    # (N, 256)
print(f'Loaded {all_z.shape} S1 representations')

# Train / val split
N       = all_z.size(0)
idx     = torch.randperm(N)
split   = int(0.9 * N)
z_train = all_z[idx[:split]].to(DEVICE)
z_val   = all_z[idx[split:]].to(DEVICE)

train_loader = DataLoader(
    TensorDataset(z_train),
    batch_size = 256,
    shuffle    = True,
)
val_loader = DataLoader(
    TensorDataset(z_val),
    batch_size = 256,
)

# ── Model ─────────────────────────────────────────────────────────────────────
vae = LatentVAE(input_dim=256, hidden_dim=512, latent_dim=64).to(DEVICE)
print(f'VAE params: {sum(p.numel() for p in vae.parameters()):,}')

optimizer = AdamW(vae.parameters(), lr=1e-3, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=1e-4)

# ── Beta annealing — start low, increase gradually ───────────────────────────
# Prevents posterior collapse during early training
def get_beta(epoch, max_epoch=20, beta_max=4.0):
    return min(beta_max, beta_max * epoch / max_epoch)

# ── Training loop ─────────────────────────────────────────────────────────────
history = {'train_loss': [], 'val_loss': [], 'recon': [], 'kl': []}
best_val_loss = float('inf')
N_EPOCHS = 50

for epoch in range(1, N_EPOCHS + 1):
    beta = get_beta(epoch)
    
    # Train
    vae.train()
    t_loss = 0.0; n = 0
    for (z,) in train_loader:
        z_recon, mu, logvar, b = vae(z)
        loss, recon, kl = vae_loss(z, z_recon, mu, logvar, beta)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        optimizer.step()
        t_loss += loss.item(); n += 1
    
    scheduler.step()

    # Val
    vae.eval()
    v_loss = 0.0; v_recon = 0.0; v_kl = 0.0; m = 0
    with torch.no_grad():
        for (z,) in val_loader:
            z_recon, mu, logvar, b = vae(z)
            loss, recon, kl = vae_loss(z, z_recon, mu, logvar, beta)
            v_loss  += loss.item()
            v_recon += recon.item()
            v_kl    += kl.item()
            m += 1

    history['train_loss'].append(t_loss / n)
    history['val_loss'].append(v_loss / m)
    history['recon'].append(v_recon / m)
    history['kl'].append(v_kl / m)

    print(
        f'Epoch {epoch:02d}/{N_EPOCHS}  '
        f'train={t_loss/n:.4f}  val={v_loss/m:.4f}  '
        f'recon={v_recon/m:.4f}  kl={v_kl/m:.4f}  '
        f'beta={beta:.2f}  lr={optimizer.param_groups[0]["lr"]:.2e}'
    )

    if v_loss / m < best_val_loss:
        best_val_loss = v_loss / m
        torch.save({
            'epoch':   epoch,
            'vae':     vae.state_dict(),
            'val_loss': best_val_loss,
        }, SAVE_DIR / 'best.pt')
        print(f'  ✓ saved → {SAVE_DIR}/best.pt')

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
epochs_r  = range(1, N_EPOCHS + 1)

axes[0].plot(epochs_r, history['train_loss'], label='train')
axes[0].plot(epochs_r, history['val_loss'],   label='val')
axes[0].set_title('Total Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_r, history['recon'], 'g')
axes[1].set_title('Reconstruction Loss'); axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs_r, history['kl'], 'r')
axes[2].set_title('KL Divergence'); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(SAVE_DIR / 'training_curves_vae.png', dpi=150)
plt.close(fig)
print(f'Plot saved → {SAVE_DIR}/training_curves_vae.png')
print(f'Best val_loss: {best_val_loss:.4f}')
















# v5/vae/inference.py

@torch.no_grad()
def sample_reference(vae, n_samples=1, label_centroid=None, sigma=0.5):
    """
    Sample a reference z_ref from b-space.
    
    Options:
      - Pure sample:     b ~ N(0, I)
      - Near centroid:   b ~ N(b_centroid, sigma²I)
      - Interpolation:   b = alpha*b_A + (1-alpha)*b_B
    """
    if label_centroid is not None:
        # Sample near a known region
        b = label_centroid + sigma * torch.randn(
            n_samples, vae.latent_dim,
            device=label_centroid.device
        )
    else:
        # Pure sample from prior
        b = torch.randn(n_samples, vae.latent_dim)
    
    z_ref = vae.decode(b)   # (n_samples, 256) — back in S1 space
    return z_ref, b


def build_reference_bank(vae, s1_encoder, turns_by_label, device):
    """
    Build centroids in b-space for each label.
    turns_by_label: dict {label: [text, text, ...]}
    """
    bank = {}
    for label, turns in turns_by_label.items():
        # Encode turns with S1
        z_list = []
        for turn in turns:
            ids    = tokenizer(turn, return_tensors='pt',
                               max_length=128, truncation=True,
                               padding='max_length').to(device)
            h      = s1_encoder(**ids)
            if isinstance(h, tuple): h = h[0]
            z      = mean_pool(h, ids['attention_mask'])
            z_list.append(z)
        
        z_all   = torch.cat(z_list)           # (N_label, 256)
        mu, _   = vae.encode(z_all)           # (N_label, 64)
        centroid = mu.mean(0, keepdim=True)   # (1, 64)
        bank[label] = centroid
        print(f'  {label}: {len(turns)} turns → centroid computed')
    
    return bank


def langevin_condition(z_start, z_ref, predictor, 
                       n_steps=50, eta=0.01, 
                       lambda_semantic=1.0, noise_scale=0.01):
    """
    COLD Langevin: steer z_start toward z_ref
    while staying coherent with S2 dynamics.
    
    z_start : (B, L, D) — initial context sequence
    z_ref   : (B, D)    — reference from VAE decoder
    """
    z = z_start.clone().detach().requires_grad_(True)

    for step in range(n_steps):
        # E_dynamics: consistency with S2 predictor
        pred         = predictor(z, z)
        E_dynamics   = F.mse_loss(pred, z.detach())

        # E_semantic: closeness to z_ref in pooled space
        z_pooled     = z.mean(dim=1)              # (B, D)
        E_semantic   = 1 - F.cosine_similarity(
            z_pooled, z_ref, dim=-1
        ).mean()

        E = E_dynamics + lambda_semantic * E_semantic

        grad = torch.autograd.grad(E, z)[0]
        
        with torch.no_grad():
            noise  = noise_scale * torch.randn_like(z)
            z      = z - eta * grad + noise
            z.requires_grad_(True)

    return z.detach()