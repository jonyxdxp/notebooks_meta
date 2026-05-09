"""
cog_arch/decoder.py — S3: GPT-2 prefix decoder conditioned on s2 embedding.

Architecture:
  1. PrefixProjection  — linear layer mapping ẑ (768) → GPT-2 hidden dim (768).
                         Produces a single "virtual prefix token" embedding.

  2. S3Decoder         — wraps GPT-2 and prepends the projected ẑ as the first
                         token in the embedding sequence before the target tokens.

Forward pass (training):
  ẑ (B, 768)              →  prefix_proj  →  prefix_emb (B, 1, 768)
  dec_input_ids (B, L)    →  GPT2 embedding table  →  token_embs (B, L, 768)
  cat([prefix_emb, token_embs], dim=1)  →  GPT-2 transformer  →  logits (B, 1+L, vocab)
  loss = CE(logits[:, 1:], dec_labels)  ← skip the prefix position in the loss

Why prepend as an embedding rather than as a token id:
  ẑ is a continuous vector — it doesn't correspond to any real token.
  Injecting it directly into the embedding stream (before the transformer layers)
  lets every attention head in GPT-2 attend to it as if it were a real token,
  without needing to discretise it first.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config


class PrefixProjection(nn.Module):
    """
    Projects the s2 predicted embedding into GPT-2's residual stream dimension.
    Two-layer MLP with GELU + LayerNorm for stable gradients.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, D) → (B, 1, D)"""
        return self.net(z).unsqueeze(1)


class S3Decoder(nn.Module):
    """
    GPT-2 decoder conditioned on a continuous prefix embedding (ẑ from s2).

    Training:
        logits, loss = model(z_hat, dec_input_ids, dec_labels, dec_attn_mask)

    Inference:
        generated_ids = model.generate(z_hat, tokenizer, max_new_tokens, ...)
    """

    def __init__(self, cfg):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(
            cfg.decoder_model,
            cache_dir=cfg.cache_dir,
        )
        gpt2_dim = self.gpt2.config.hidden_size   # 768 for gpt2-small

        self.prefix_proj = PrefixProjection(
            in_dim  = cfg.encoder_dim,   # 768 from DSE
            out_dim = gpt2_dim,
        )

        self.gpt2_dim = gpt2_dim
        print(f"[s3] GPT-2 hidden dim: {gpt2_dim} | vocab: {self.gpt2.config.vocab_size}")

    def forward(
        self,
        z_hat:         torch.Tensor,   # (B, D) predicted embedding from s2
        dec_input_ids: torch.Tensor,   # (B, L) GPT-2 input token ids
        dec_labels:    torch.Tensor,   # (B, L) CE targets (-100 = ignore)
        dec_attn_mask: torch.Tensor,   # (B, L) attention mask for token ids
    ):
        """
        Returns (logits, loss).

        The prefix token is prepended to the token embedding sequence.
        A corresponding prefix attention mask (all 1s) is prepended too.
        Loss is computed only over the token positions (not the prefix).
        """
        B, L = dec_input_ids.shape

        # 1. Project ẑ into GPT-2 embedding space → (B, 1, D)
        prefix_emb = self.prefix_proj(z_hat)    # (B, 1, D)

        # 2. Embed the decoder input tokens → (B, L, D)
        token_embs = self.gpt2.transformer.wte(dec_input_ids)   # (B, L, D)

        # 3. Concatenate prefix + token embeddings → (B, 1+L, D)
        inputs_embeds = torch.cat([prefix_emb, token_embs], dim=1)

        # 4. Build attention mask: prefix always attended to
        prefix_mask   = torch.ones(B, 1, device=dec_attn_mask.device, dtype=dec_attn_mask.dtype)
        full_attn_mask = torch.cat([prefix_mask, dec_attn_mask], dim=1)   # (B, 1+L)

        # 5. Build labels: -100 for prefix position, then dec_labels
        prefix_label_ignore = torch.full(
            (B, 1), fill_value=-100,
            device=dec_labels.device, dtype=dec_labels.dtype
        )
        full_labels = torch.cat([prefix_label_ignore, dec_labels], dim=1)  # (B, 1+L)

        # 6. GPT-2 forward pass
        outputs = self.gpt2(
            inputs_embeds = inputs_embeds,
            attention_mask = full_attn_mask,
            labels         = full_labels,
        )

        return outputs.logits, outputs.loss

    @torch.no_grad()
    def generate(
        self,
        z_hat:      torch.Tensor,   # (1, D) or (B, D)
        tokenizer,
        max_new_tokens: int  = 60,
        temperature:    float = 0.8,
        top_p:          float = 0.9,
        do_sample:      bool  = True,
    ) -> list[str]:
        """
        Autoregressively generate utterances conditioned on ẑ.
        Returns a list of decoded strings (one per item in the batch).
        """
        B = z_hat.size(0)
        device = z_hat.device

        # Prefix embedding: (B, 1, D)
        prefix_emb = self.prefix_proj(z_hat)

        # Start with BOS token
        bos_ids  = torch.full((B, 1), tokenizer.bos_token_id,
                              device=device, dtype=torch.long)
        bos_embs = self.gpt2.transformer.wte(bos_ids)   # (B, 1, D)

        # initial inputs_embeds = [prefix, BOS]
        inputs_embeds = torch.cat([prefix_emb, bos_embs], dim=1)  # (B, 2, D)

        # We use past_key_values to efficiently extend the sequence
        # by feeding embeddings for the first step, then token ids after
        generated = [[] for _ in range(B)]
        finished  = [False] * B

        # First forward step: feed the prefix + BOS as embeddings
        out = self.gpt2(inputs_embeds=inputs_embeds, use_cache=True)
        past   = out.past_key_values
        logits = out.logits[:, -1, :]   # (B, vocab) — logits after BOS

        for _ in range(max_new_tokens):
            # Sample or greedy from logits
            if do_sample:
                # temperature scaling
                logits_scaled = logits / max(temperature, 1e-8)
                # top-p (nucleus) sampling
                sorted_logits, sorted_idx = torch.sort(logits_scaled, descending=True)
                cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                # remove tokens with cumulative prob above top_p
                sorted_logits[cumprobs - torch.softmax(sorted_logits, dim=-1) > top_p] = float("-inf")
                # scatter back
                logits_filtered = torch.full_like(logits_scaled, float("-inf"))
                logits_filtered.scatter_(1, sorted_idx, sorted_logits)
                probs  = torch.softmax(logits_filtered, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)   # (B, 1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)       # (B, 1)

            # Record generated tokens
            for i in range(B):
                tok_id = next_token[i, 0].item()
                if tok_id == tokenizer.eos_token_id:
                    finished[i] = True
                if not finished[i]:
                    generated[i].append(tok_id)

            if all(finished):
                break

            # Next step: use token ids (much cheaper than re-embedding everything)
            out    = self.gpt2(input_ids=next_token, past_key_values=past, use_cache=True)
            past   = out.past_key_values
            logits = out.logits[:, -1, :]

        return [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]