"""Generation-quality metrics: BLEU-4, chrF++, ROUGE-L against reference
continuations, plus reference-free fluency/diversity diagnostics
(repetition rate, Distinct-1/2).
"""
import re
from typing import List

import sacrebleu
from rouge_score import rouge_scorer


class _UnicodeWordTokenizer:
    """rouge_score's built-in default tokenizer strips every character
    outside `[a-z0-9]` (ASCII-only) when normalizing tokens, which silently
    reduces every Devanagari word to an empty string -- so ROUGE-L against
    Hindi/Nepali text always scores at or near zero regardless of actual
    overlap (verified: even scoring identical Devanagari text against
    itself returns 0.0 with the default tokenizer). `\\w` in Python's `re`
    is Unicode-aware by default for `str` patterns, so this simple
    whitespace/punctuation-agnostic word tokenizer works correctly for
    Devanagari (and any other Unicode script) instead.
    """

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text, flags=re.UNICODE)


def corpus_bleu_chrf(hypotheses: List[str], references: List[str]) -> dict:
    """Corpus-level BLEU-4 and chrF++ (sacrebleu). `references` holds one
    reference string per hypothesis (single-reference set)."""
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)  # chrF++
    return {"bleu": bleu.score, "chrf++": chrf.score}


def avg_rouge_l(hypotheses: List[str], references: List[str]) -> dict:
    """Average sentence-level ROUGE-L F1 over all (hypothesis, reference) pairs."""
    scorer = rouge_scorer.RougeScorer(
        ["rougeL"], use_stemmer=False, tokenizer=_UnicodeWordTokenizer()
    )
    scores = [scorer.score(ref, hyp)["rougeL"].fmeasure for hyp, ref in zip(hypotheses, references)]
    return {"rouge_l_f1": sum(scores) / len(scores) if scores else 0.0}


def _ngrams(tokens: list, n: int) -> list:
    return list(zip(*[tokens[i:] for i in range(n)]))


def repetition_rate(texts: List[str], n: int = 4) -> float:
    """Fraction of n-grams that repeat an earlier n-gram within the same
    generated text (whitespace-tokenized), averaged over all texts."""
    rates = []
    for text in texts:
        grams = _ngrams(text.split(), n)
        if not grams:
            rates.append(0.0)
            continue
        seen = set()
        repeats = 0
        for g in grams:
            if g in seen:
                repeats += 1
            seen.add(g)
        rates.append(repeats / len(grams))
    return sum(rates) / len(rates) if rates else 0.0


def distinct_n(texts: List[str], n: int) -> float:
    """Distinct-N: unique n-grams / total n-grams, pooled over all texts."""
    all_grams = []
    for text in texts:
        all_grams.extend(_ngrams(text.split(), n))
    if not all_grams:
        return 0.0
    return len(set(all_grams)) / len(all_grams)


def diversity_stats(texts: List[str]) -> dict:
    return {
        "repetition_rate_4gram": repetition_rate(texts, 4),
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
    }
