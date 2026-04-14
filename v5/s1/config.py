"""
Config loader para JEPA Texto - Versión plana
"""
import os
import yaml
import torch
from types import SimpleNamespace

# Path al YAML (misma carpeta que este archivo)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'sigreg.yaml')

def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# Cargar configuración
cfg_dict = load_yaml_config(CONFIG_PATH)
CFG = SimpleNamespace(**cfg_dict)

# Determinar dispositivo
if CFG.device == "auto":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device(CFG.device)

# Crear directorios necesarios
os.makedirs(CFG.ckpt_dir, exist_ok=True)
os.makedirs(CFG.cache_dir, exist_ok=True)
os.makedirs(CFG.raw_data_dir, exist_ok=True)

print(f"Config loaded: seed={CFG.seed}, device={DEVICE}")
print(f"Model: hidden={CFG.hidden_size}, seq_len={CFG.max_seq_len}")
print(f"Checkpoints: {CFG.ckpt_dir}")