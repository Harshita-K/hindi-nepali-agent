"""Corpus statistics computation, shared across languages.

Kept separate from visualization (dataviz code lives in each language's
report generation step) per the assignment's "keep visualization and
computation logic in separate functions" guideline.
"""
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def whitespace_token_count(text: str) -> int:
    """Cheap proxy for token count before a tokenizer exists (for planning
    corpus size). Final token counts must use the trained tokenizer.
    """
    return len(text.split())


def compute_corpus_stats(docs: list[dict]) -> dict:
    """docs: list of {"text": ..., "source": ..., "source_type": "manual"|"downloaded", ...}"""
    total_docs = len(docs)
    total_chars = sum(len(d["text"]) for d in docs)
    total_ws_tokens = sum(whitespace_token_count(d["text"]) for d in docs)

    by_type = Counter(d.get("source_type", "unknown") for d in docs)
    tokens_by_type = Counter()
    for d in docs:
        tokens_by_type[d.get("source_type", "unknown")] += whitespace_token_count(d["text"])

    by_source = Counter(d.get("source", "unknown") for d in docs)

    manual_tokens = tokens_by_type.get("manual", 0)
    manual_fraction = manual_tokens / total_ws_tokens if total_ws_tokens else 0.0

    return {
        "total_docs": total_docs,
        "total_chars": total_chars,
        "total_whitespace_tokens": total_ws_tokens,
        "docs_by_source_type": dict(by_type),
        "whitespace_tokens_by_source_type": dict(tokens_by_type),
        "docs_by_source": dict(by_source),
        "manual_token_fraction": round(manual_fraction, 4),
    }


def save_stats(stats: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def sample_corpus_to_target(
    docs: list[dict],
    target_total_tokens: int,
    min_manual_fraction: float,
    seed: int = 42,
    enforce_ratio: bool = True,
    downloaded_target_tokens: int | None = None,
) -> list[dict]:
    """Subsample cleaned+deduped docs toward target_total_tokens.

    enforce_ratio=True (default): GUARANTEE manual_tokens / total_tokens >=
    min_manual_fraction as a hard floor. All available manual docs are kept,
    and downloaded is capped by whichever is tighter -- the ratio floor or
    the token target. If manual is too scarce to reach target_total_tokens
    at that ratio, the corpus comes out smaller than target rather than
    diluting below the floor.

    enforce_ratio=False: fill downloaded up to exactly target_total_tokens
    regardless of ratio (still uses all available manual first). Total size
    is prioritized over the ratio -- use only when you're intentionally
    trading the floor for hitting the token target while manual is still
    growing toward it separately (report the resulting fraction honestly).

    downloaded_target_tokens: when set (only meaningful with
    enforce_ratio=False), caps downloaded directly at this token count
    instead of deriving it from target_total_tokens - manual. Use when the
    goal is a specific downloaded size (e.g. ~400M) rather than a specific
    total.
    """
    import random

    rng = random.Random(seed)
    manual_docs = [d for d in docs if d.get("source_type") == "manual"]
    downloaded_docs = [d for d in docs if d.get("source_type") != "manual"]
    rng.shuffle(manual_docs)
    rng.shuffle(downloaded_docs)

    manual_used_tokens = sum(whitespace_token_count(d["text"]) for d in manual_docs)

    if enforce_ratio and min_manual_fraction > 0:
        max_downloaded_by_ratio = manual_used_tokens * (1 - min_manual_fraction) / min_manual_fraction
    else:
        max_downloaded_by_ratio = float("inf")
    if downloaded_target_tokens is not None:
        max_downloaded_by_target = downloaded_target_tokens
    else:
        max_downloaded_by_target = max(target_total_tokens - manual_used_tokens, 0)
    downloaded_budget = min(max_downloaded_by_ratio, max_downloaded_by_target)

    downloaded_used, running = [], 0
    for d in downloaded_docs:
        t = whitespace_token_count(d["text"])
        if running + t > downloaded_budget:
            continue  # skip this one, keep trying smaller docs to fill the budget
        downloaded_used.append(d)
        running += t

    return manual_docs + downloaded_used
