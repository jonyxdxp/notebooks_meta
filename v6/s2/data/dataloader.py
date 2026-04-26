# dataloader.py - Stage 2 (Turn pairs, no masking)
import sys
import torch
from typing import Optional

# Importar desde el dataset de Stage 2 (turn pairs)
from v5.s2.data.dataset import (
    get_turn_pair_dataset,
    TurnPairCollator,
    tokenizer,
    CFG
)

def get_stage2_dataloaders(
    cfg_obj,
    tokenizer,
    skip_train: bool = False,
    skip_valid: bool = False,
):
    """
    Crea dataloaders para Stage 2 (pares de turnos consecutivos).
    No usa masking, solo pares (contexto, target) = (turno_t, turno_{t+1})
    """
    
    # Cargar dataset de pares
    pair_dataset = get_turn_pair_dataset(
        cache_dir=cfg_obj.data.cache_dir if hasattr(cfg_obj, 'data') else cfg_obj.cache_dir,
        tokenizer=tokenizer,
        block_size=cfg_obj.model.max_seq_len if hasattr(cfg_obj, 'model') else cfg_obj.max_seq_len,
    )
    
    collator = TurnPairCollator()
    
    train_loader = None
    val_loader = None
    
    batch_size = cfg_obj.data.batch_size if hasattr(cfg_obj, 'data') else cfg_obj.batch_size
    eval_batch = cfg_obj.data.eval_batch if hasattr(cfg_obj, 'data') else getattr(cfg_obj, 'eval_batch', batch_size)
    num_workers = cfg_obj.data.num_workers if hasattr(cfg_obj, 'data') else cfg_obj.num_workers
    
    if not skip_train:
        train_loader = torch.utils.data.DataLoader(
            pair_dataset['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collator,
        )
        train_loader.tokenizer = tokenizer

    if not skip_valid:
        val_loader = torch.utils.data.DataLoader(
            pair_dataset['validation'],
            batch_size=eval_batch,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collator,
        )
        val_loader.tokenizer = tokenizer

    return train_loader, val_loader


# Mantener nombre alternativo para compatibilidad si se necesita
get_jepa_dataloaders = get_stage2_dataloaders