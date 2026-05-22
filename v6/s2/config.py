
import torch
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = os.environ.get("DAILYDIALOG_PATH",
              "/content/drive/MyDrive/data/dailydialog")
CKPT_DIR    = os.environ.get("CKPT_DIR",
              "/content/drive/MyDrive/data/dmi_checkpoints")

# ── Model ─────────────────────────────────────────────────────────────────────
MAX_TURNS   = 6
D_INPUT     = 768
D_MODEL     = 512
NHEAD       = 8
NUM_LAYERS  = 4
DIM_FF      = 1024
DROPOUT     = 0.1

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE  = 128
EPOCHS      = 30
LR          = 1e-4
WEIGHT_DECAY= 1e-4

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── DMI encoder ───────────────────────────────────────────────────────────────
DMI_CKPT = os.environ.get("DMI_CKPT",
           "/content/drive/MyDrive/data/dmi_checkpoints/DMI_medium_model.pth")
BERT_NAME = "google/bert_uncased_L-8_H-768_A-12"
