import torch
from v5.s2.data.dataset import (
    get_multiturn_dataset, MultiTurnCollator, tokenizer, CFG
)

def get_stage2_dataloaders(cfg_obj, tokenizer, skip_train=False, skip_valid=False):
    ds = get_multiturn_dataset(
        cache_dir  = cfg_obj.data.cache_dir,
        tokenizer  = tokenizer,
        block_size = cfg_obj.model.max_seq_len,
    )
    collator    = MultiTurnCollator()
    batch_size  = cfg_obj.data.batch_size
    eval_batch  = cfg_obj.data.eval_batch
    num_workers = cfg_obj.data.num_workers

    train_loader = torch.utils.data.DataLoader(
        ds['train'], batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=collator,
    ) if not skip_train else None

    val_loader = torch.utils.data.DataLoader(
        ds['validation'], batch_size=eval_batch, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collator,
    ) if not skip_valid else None

    return train_loader, val_loader