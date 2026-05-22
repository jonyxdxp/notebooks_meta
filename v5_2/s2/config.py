"""
config.py — central configuration for the dialog next-turn prediction experiment.
All paths, model hyper-params, and training settings live here.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────────
    raw_data_dir:   str = "/content/data/dailydialog_raw/ijcnlp_dailydialog"
    cache_dir:      str = "/content/drive/MyDrive/data/cache"
    output_dir:     str = "/content/drive/MyDrive/checkpoints/dialog_next_turn"

    # ── Utterance encoder (DSE) ────────────────────────────────────────────────
    encoder_model:  str = "aws-ai/dse-bert-base"   # HF model id
    encoder_dim:    int = 768                        # DSE hidden size
    freeze_encoder: bool = True                      # freeze DSE weights during training
    #   set to False to fine-tune end-to-end (much slower, needs lower lr)

    # ── Causal context transformer ─────────────────────────────────────────────
    # Operates on sequences of utterance embeddings (not token sequences).
    ctx_n_heads:    int = 8
    ctx_n_layers:   int = 4
    ctx_ffn_dim:    int = 2048
    ctx_dropout:    float = 0.1
    max_history:    int = 10          # max number of past turns to condition on

    # ── Projection head ────────────────────────────────────────────────────────
    # Maps context vector → DSE embedding space for InfoNCE comparison.
    proj_hidden_dim: int = 512        # set to None to use a single linear layer

    # ── InfoNCE loss ───────────────────────────────────────────────────────────
    temperature:    float = 0.07      # standard SimCSE / MoCo value

    # ── Training ───────────────────────────────────────────────────────────────
    batch_size:     int = 64
    num_epochs:     int = 20
    lr:             float = 1e-4      # for the context transformer + proj head
    encoder_lr:     float = 2e-5     # used only when freeze_encoder=False
    warmup_steps:   int = 200
    grad_clip:      float = 1.0
    eval_every:     int = 1           # evaluate on val set every N epochs
    save_every:     int = 2
    seed:           int = 42

    # ── Data ───────────────────────────────────────────────────────────────────
    min_turns:      int = 3           # skip dialogs shorter than this
    num_workers:    int = 2
    max_seq_len:    int = 64          # max token length per utterance

    # ── Misc ───────────────────────────────────────────────────────────────────
    device:         str = "cuda"      # "cpu" if no GPU
    fp16:           bool = True       # mixed precision (requires CUDA)
    log_steps:      int = 50


cfg = Config()