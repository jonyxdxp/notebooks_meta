"""
losses.py — s3 loss and evaluation metrics.

Training loss: cross-entropy (teacher forcing) over the target utterance tokens.
This is just standard GPT-2 LM loss — already computed inside S3Decoder.forward().

Eval metrics:
  - Perplexity  : exp(mean CE loss) — standard LM metric; lower is better
  - BLEU-1/2/4  : n-gram overlap between generated and reference utterances
  - Distinct-1/2: ratio of unique unigrams/bigrams in generated output
                  (measures diversity; low = repetitive/degenerate output)
"""

import math
import collections
from typing import List


# ── Perplexity ─────────────────────────────────────────────────────────────────

def perplexity_from_loss(avg_ce_loss: float) -> float:
    """PPL = exp(mean CE loss). Clipped to avoid overflow on bad early checkpoints."""
    return math.exp(min(avg_ce_loss, 20.0))


# ── BLEU ───────────────────────────────────────────────────────────────────────

def _ngrams(tokens: List[str], n: int) -> collections.Counter:
    return collections.Counter(
        tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)
    )


def corpus_bleu(
    hypotheses: List[str],
    references: List[str],
    max_n: int = 4,
) -> dict:
    """
    Compute corpus-level BLEU-1 through BLEU-{max_n}.
    Uses add-1 smoothing to avoid zero scores on short outputs.

    hypotheses: list of generated utterances (plain strings)
    references: list of ground-truth utterances (plain strings)
    """
    # Simple whitespace tokenisation — good enough for English dialog eval
    hyp_tokens = [h.lower().split() for h in hypotheses]
    ref_tokens = [r.lower().split() for r in references]

    scores = {}
    for n in range(1, max_n + 1):
        clip_count = 0
        total_count = 0

        for hyp, ref in zip(hyp_tokens, ref_tokens):
            hyp_ng = _ngrams(hyp, n)
            ref_ng = _ngrams(ref, n)
            # clipped count: min(hyp_count, ref_count) per n-gram
            for ng, cnt in hyp_ng.items():
                clip_count  += min(cnt, ref_ng.get(ng, 0))
            total_count += max(len(hyp) - n + 1, 0)

        # add-1 smoothing
        precision = (clip_count + 1) / (total_count + 1)
        scores[f"BLEU-{n}"] = precision

    # Brevity penalty
    hyp_len = sum(len(h) for h in hyp_tokens)
    ref_len = sum(len(r) for r in ref_tokens)
    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / max(hyp_len, 1))

    # Final BLEU-4 with BP
    bleu4_raw = scores["BLEU-4"]
    scores["BLEU-4 (BP)"] = bp * bleu4_raw

    return scores


# ── Distinct ──────────────────────────────────────────────────────────────────

def distinct_n(hypotheses: List[str], n: int) -> float:
    """
    Distinct-N: ratio of unique n-grams to total n-grams across all hypotheses.
    Higher = more diverse outputs.
    """
    all_ngrams = []
    for hyp in hypotheses:
        tokens = hyp.lower().split()
        all_ngrams.extend(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def compute_generation_metrics(
    hypotheses: List[str],
    references: List[str],
) -> dict:
    """
    Convenience: run all eval metrics and return a flat dict.
    """
    metrics = {}
    metrics.update(corpus_bleu(hypotheses, references))
    metrics["Distinct-1"] = distinct_n(hypotheses, 1)
    metrics["Distinct-2"] = distinct_n(hypotheses, 2)
    return metrics