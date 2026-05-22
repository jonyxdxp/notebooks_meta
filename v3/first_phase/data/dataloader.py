

import os
import typing

import datasets
import torch
import transformers
import tokenizers

import utils

LOGGER = utils.get_logger(__name__)







# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def get_tokenizer(config):
  tokenizer = transformers.AutoTokenizer.from_pretrained(
    config.data.tokenizer_name_or_path)

  if (isinstance(tokenizer, transformers.GPT2TokenizerFast)
      or isinstance(tokenizer, transformers.GPT2Tokenizer)):
    tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
      (tokenizer.bos_token, tokenizer.bos_token_id),
      (tokenizer.eos_token, tokenizer.eos_token_id))

  if tokenizer.bos_token is None:
    if tokenizer.cls_token is None:
      raise AttributeError(
        'Tokenizer must have a bos_token or '
        f'cls_token: {tokenizer}')
    tokenizer.bos_token = tokenizer.cls_token
  if tokenizer.eos_token is None:
    if tokenizer.sep_token is None:
      raise AttributeError(
        'Tokenizer must have a eos_token or '
        f'sep_token: {tokenizer}')
    tokenizer.eos_token = tokenizer.sep_token
  if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

  return tokenizer








# ---------------------------------------------------------------------------
# DailyDialog dataset loader
# ---------------------------------------------------------------------------

def get_dailydialog_dataset(
    cache_dir: str,
    tokenizer,
    block_size: int = 128,
    num_proc: int = len(os.sched_getaffinity(0)),
) -> datasets.DatasetDict:
  """Load DailyDialog and flatten every dialogue turn into its own sample.

  Each individual utterance becomes one independent sequence so the model
  learns the common / invariant features within a single conversational turn.

  Returns a DatasetDict with keys 'train', 'validation', 'test', each
  containing columns 'input_ids' and 'attention_mask' (torch tensors of
  shape [block_size]).
  """
  _cache_path = os.path.join(cache_dir, f'dailydialog_jepa_bs{block_size}')

  if utils.fsspec_exists(_cache_path):
    LOGGER.info(f'Loading DailyDialog JEPA dataset from cache: {_cache_path}')
    return datasets.load_from_disk(_cache_path).with_format('torch')

  LOGGER.info('Building DailyDialog JEPA dataset ...')
  raw = datasets.load_dataset(
    'daily_dialog', cache_dir=cache_dir, trust_remote_code=True)

  def _flatten_turns(examples):
    utterances = []
    for dialog in examples['dialog']:
      utterances.extend(utt.strip() for utt in dialog if utt.strip())
    return {'text': utterances}

  def _tokenize(examples):
    return tokenizer(
      examples['text'],
      max_length=block_size,
      padding='max_length',
      truncation=True,
      add_special_tokens=True,
      return_attention_mask=True,
      return_token_type_ids=False,
    )

  tokenized_splits = {}
  for split in ('train', 'validation', 'test'):
    flat = raw[split].map(
      _flatten_turns,
      batched=True,
      remove_columns=raw[split].column_names,
      num_proc=num_proc,
      desc=f'Flattening {split}',
    )
    tokenized_splits[split] = flat.map(
      _tokenize,
      batched=True,
      num_proc=num_proc,
      remove_columns=['text'],
      desc=f'Tokenizing {split}',
    )

  dataset_dict = datasets.DatasetDict(tokenized_splits)
  dataset_dict.save_to_disk(_cache_path)
  LOGGER.info(f'DailyDialog JEPA dataset saved to: {_cache_path}')
  return dataset_dict.with_format('torch')








# ---------------------------------------------------------------------------
# JEPA mask collator
# ---------------------------------------------------------------------------

class JEPAMaskCollator:
  """Produces context / target view pairs for JEPA phase-1 training.

  Both views come from the same tokenized sequence:
    - target_input_ids:   clean, unmasked sequence  → target encoder (stop-grad)
    - context_input_ids:  target spans replaced with [MASK] → context encoder
    - target_mask:        bool (B, L), True at masked positions → predictor loss

  Span sampling: ``num_target_spans`` non-overlapping spans of
  ``target_span_length`` tokens, sampled uniformly from the maskable region
  (excluding BOS, EOS, and padding).
  """

  def __init__(
      self,
      mask_token_id: int,
      pad_token_id: int,
      num_target_spans: int = 4,
      target_span_length: int = 8,
  ):
    self.mask_token_id = mask_token_id
    self.pad_token_id = pad_token_id
    self.num_target_spans = num_target_spans
    self.target_span_length = target_span_length

  def __call__(
      self,
      batch: typing.List[typing.Dict[str, torch.Tensor]],
  ) -> typing.Dict[str, torch.Tensor]:
    input_ids      = torch.stack([b['input_ids']      for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    B, L = input_ids.shape

    target_input_ids  = input_ids.clone()
    context_input_ids = input_ids.clone()
    target_mask = torch.zeros(B, L, dtype=torch.bool)

    for i in range(B):
      valid_len = int(attention_mask[i].sum().item())
      # maskable region: skip BOS (idx 0) and EOS (last real token)
      spans = self._sample_spans(maskable_start=1, maskable_end=valid_len - 1)
      for s, e in spans:
        target_mask[i, s:e] = True
      context_input_ids[i, target_mask[i]] = self.mask_token_id

    return {
      'context_input_ids':      context_input_ids,   # (B, L)
      'context_attention_mask': attention_mask,       # (B, L)
      'target_input_ids':       target_input_ids,     # (B, L)
      'target_attention_mask':  attention_mask,       # (B, L)
      'target_mask':            target_mask,          # (B, L) bool
    }

  def _sample_spans(
      self,
      maskable_start: int,
      maskable_end: int,
  ) -> typing.List[typing.Tuple[int, int]]:
    region_len = maskable_end - maskable_start
    if region_len <= 0:
      return []

    span_len  = min(self.target_span_length, region_len)
    available = list(range(maskable_start, maskable_end - span_len + 1))

    spans: typing.List[typing.Tuple[int, int]] = []
    for _ in range(self.num_target_spans):
      if not available:
        break
      idx = torch.randint(len(available), (1,)).item()
      s, e = available[idx], available[idx] + span_len
      spans.append((s, e))
      available = [x for x in available if (x + span_len <= s) or (x >= e)]

    return spans








# ---------------------------------------------------------------------------
# JEPA dataloaders
# ---------------------------------------------------------------------------

def get_jepa_dataloaders(
    config,
    tokenizer,
    skip_train: bool = False,
    skip_valid: bool = False,
    valid_seed: typing.Optional[int] = None,
):
  """Build train / validation DataLoaders for JEPA phase-1 training.

  Expects config.jepa.num_target_spans and config.jepa.target_span_length
  (defaults: 4 and 8).  Batch items have keys produced by JEPAMaskCollator.
  """
  num_gpus  = max(torch.cuda.device_count(), 1)
  block_size = config.model.length

  if config.loader.eval_global_batch_size % num_gpus != 0:
    raise ValueError(
      f'Eval batch size {config.loader.eval_global_batch_size} '
      f'not divisible by {num_gpus} GPUs.')

  dataset_dict = get_dailydialog_dataset(
    cache_dir=config.data.cache_dir,
    tokenizer=tokenizer,
    block_size=block_size,
  )

  if tokenizer.mask_token_id is None:
    raise ValueError(
      f'Tokenizer must have a mask_token for JEPA masking: {tokenizer}')

  jepa_cfg = getattr(config, 'jepa', None)
  collator = JEPAMaskCollator(
    mask_token_id=tokenizer.mask_token_id,
    pad_token_id=tokenizer.pad_token_id,
    num_target_spans=getattr(jepa_cfg, 'num_target_spans', 4),
    target_span_length=getattr(jepa_cfg, 'target_span_length', 8),
  )

  train_loader = valid_loader = None

  if not skip_train:
    train_loader = torch.utils.data.DataLoader(
      dataset_dict['train'],
      batch_size=config.loader.batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=True,
      persistent_workers=True,
      collate_fn=collator,
    )
    train_loader.tokenizer = tokenizer

  if not skip_valid:
    generator    = torch.Generator().manual_seed(valid_seed) if valid_seed else None
    valid_loader = torch.utils.data.DataLoader(
      dataset_dict['validation'],
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=valid_seed is not None,
      generator=generator,
      collate_fn=collator,
    )
    valid_loader.tokenizer = tokenizer

  return train_loader, valid_loader