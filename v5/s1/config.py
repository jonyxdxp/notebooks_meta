"""
Sistema de configuración para JEPA Texto
Carga el YAML y expone CFG, DEVICE, y otras constantes globales
"""
import os
import yaml
import torch
from types import SimpleNamespace
from pathlib import Path

def load_yaml_config(path="sigreg.yaml"):
    """Carga YAML y retorna dict"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def dict_to_namespace(d):
    """Convierte dict anidado a SimpleNamespace (acceso con punto)"""
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = dict_to_namespace(value)
        return SimpleNamespace(**d)
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d

def get_device(config):
    """Determina el dispositivo según config"""
    device_str = config.meta.device
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)

# Cargar configuración global
# Busca el YAML en el directorio actual o en /content/notebooks_meta/v5/s1/
config_paths = [
    "sigreg.yaml",
    "/content/notebooks_meta/v5/s1/sigreg.yaml",
    "/content/notebooks_meta/v5/config/sigreg.yaml",
]

CFG = None
for path in config_paths:
    if os.path.exists(path):
        print(f"Loading config from: {path}")
        cfg_dict = load_yaml_config(path)
        CFG = dict_to_namespace(cfg_dict)
        break

if CFG is None:
    raise FileNotFoundError(f"No se encontró sigreg.yaml en: {config_paths}")

# Variables globales accesibles desde cualquier módulo
DEVICE = get_device(CFG)

# Crear directorios necesarios
if hasattr(CFG.logging, 'exp_dir'):
    os.makedirs(CFG.logging.exp_dir, exist_ok=True)
else:
    CFG.logging.exp_dir = "./experiments"
    os.makedirs(CFG.logging.exp_dir, exist_ok=True)

if hasattr(CFG, 'ckpt_dir'):
    os.makedirs(CFG.ckpt_dir, exist_ok=True)
else:
    CFG.ckpt_dir = os.path.join(CFG.logging.exp_dir, "checkpoints")
    os.makedirs(CFG.ckpt_dir, exist_ok=True)

# Alias para compatibilidad con código anterior
CFG.n_epochs = CFG.optim.epochs if hasattr(CFG.optim, 'epochs') else 50

print(f"Config loaded: {CFG.meta.seed=}, {DEVICE=}")
print(f"Checkpoints dir: {CFG.ckpt_dir}")