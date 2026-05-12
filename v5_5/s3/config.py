"""
config.py — s3: embedding-to-text decoder stage.

s3 sits on top of s2. It:
  1. Loads the frozen s2 model (DSE encoder + causal context transformer)
  2. Projects the predicted embedding into GPT-2 space as a prefix token
  3. Fine-tunes GPT-2 to generate the next utterance conditioned on that prefix
"""
from dataclasses import dataclass


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────────
    raw_data_dir:   str = "/content/data/dailydialog_raw/ijcnlp_dailydialog"
    cache_dir:      str = "/content/drive/MyDrive/data/cache"
    output_dir:     str = "/content/drive/MyDrive/checkpoints/dialog_s3"

    # ── s2 checkpoint (frozen) ─────────────────────────────────────────────────
    s2_checkpoint:  str = "/content/drive/MyDrive/checkpoints/dialog_next_turn/best.pt"
    s2_module_path: str = "/content/notebooks_meta/v5_2/s2"

    # ── s2 architecture dims (must match s2/config.py exactly) ────────────────
    encoder_model:   str   = "aws-ai/dse-bert-base"
    encoder_dim:     int   = 768
    ctx_n_heads:     int   = 8
    ctx_n_layers:    int   = 4
    ctx_ffn_dim:     int   = 2048
    ctx_dropout:     float = 0.1
    max_history:     int   = 10
    max_seq_len:     int   = 64
    proj_hidden_dim: int   = 512
    freeze_encoder:  bool  = True

    # ── GPT-2 decoder ──────────────────────────────────────────────────────────
    decoder_model:   str = "gpt2"     # 117M, hidden_size=768 — matches DSE dim
    max_target_len:  int = 64         # max tokens for target utterance

    # ── Training ───────────────────────────────────────────────────────────────
    batch_size:     int   = 32
    num_epochs:     int   = 15
    lr:             float = 5e-5
    warmup_steps:   int   = 200
    grad_clip:      float = 1.0
    eval_every:     int   = 3
    save_every:     int   = 3
    seed:           int   = 42

    # ── Generation ─────────────────────────────────────────────────────────────
    gen_max_new_tokens: int   = 60
    gen_temperature:    float = 0.8
    gen_top_p:          float = 0.9
    gen_do_sample:      bool  = True

    # ── Misc ───────────────────────────────────────────────────────────────────
    device:      str  = "cuda"
    fp16:        bool = True
    log_steps:   int  = 50
    num_workers: int  = 2
    min_turns:   int  = 3


cfg = Config()