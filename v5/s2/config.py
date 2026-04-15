"""
Config loader para YAML anidado (nested structure)
Soporta: CFG.model.hidden_size, CFG.data.batch_size, etc.
"""
import os
import yaml
import torch
from types import SimpleNamespace

def dict_to_namespace(d):
    """Convierte diccionario anidado a SimpleNamespace recursivamente"""
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = dict_to_namespace(value)
        return SimpleNamespace(**d)
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d

# Cargar YAML
config_path = os.path.join(os.path.dirname(__file__), 'config/default.yaml')
with open(config_path, 'r') as f:
    cfg_dict = yaml.safe_load(f)

CFG = dict_to_namespace(cfg_dict)

# Determinar dispositivo
if CFG.meta.device == "auto":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device(CFG.meta.device)

# Crear directorios
os.makedirs(CFG.logging.exp_dir, exist_ok=True)
os.makedirs(os.path.join(CFG.logging.exp_dir, "/content/drive/MyDrive/metanet/v5/s2/checkpoints"), exist_ok=True)

# Actualizar s1_ckpt si es null
if CFG.training.s1_ckpt is None or CFG.training.s1_ckpt == "null":
    # Default path a Stage 1
    CFG.training.s1_ckpt = "/content/drive/MyDrive/metanet/v5/s1/checkpoints/epoch_best.pt"

print(f"Config loaded: seed={CFG.meta.seed}, device={DEVICE}")
print(f"Stage 1 checkpoint: {CFG.training.s1_ckpt}")