
import torch, os
from types import SimpleNamespace

BASE         = os.environ.get("DAILYDIALOG_PATH",
               "/content/Discourse-Mutual-Information-DMI-main/data/dailydialog")
CKPT_DIR     = os.environ.get("CKPT_DIR",
               "/content/drive/MyDrive/data/dmi_checkpoints")
DMI_CKPT     = os.environ.get("DMI_CKPT",
               "/content/drive/MyDrive/data/dmi_checkpoints/DMI_medium_model.pth")
BERT_NAME    = "google/bert_uncased_L-8_H-768_A-12"
MAX_TURNS    = 6
D_INPUT      = 768
D_MODEL      = 512
NHEAD        = 8
NUM_LAYERS   = 4
DIM_FF       = 1024
DROPOUT      = 0.1
BATCH_SIZE   = 128
EPOCHS       = 30
LR           = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = SimpleNamespace(
    base=BASE, ckpt_dir=CKPT_DIR, dmi_ckpt=DMI_CKPT, bert_name=BERT_NAME,
    max_turns=MAX_TURNS, d_input=D_INPUT, d_model=D_MODEL, nhead=NHEAD,
    num_layers=NUM_LAYERS, dim_ff=DIM_FF, dropout=DROPOUT,
    batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
    device=DEVICE,
)
