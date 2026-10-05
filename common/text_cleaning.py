"""Language-agnostic text cleaning utilities shared by hindi/ and nepali/ pipelines.

Both target languages are written in Devanagari, so script filtering,
normalization, and dedup logic can be shared; only source lists and
language-id codes differ per language (see each language's data_config.yaml).
"""
import hashlib
import re
import unicodedata

import ftfy

DEVANAGARI_RANGE = r"ऀ-ॿ"
# Devanagari block + common punctuation/digits/whitespace we want to keep.
_KEEP_CHARS_RE = re.compile(
    rf"[^{DEVANAGARI_RANGE}\s।॥.,!?;:()\"'\-0-9]"
)
_MULTI_WHITESPACE_RE = re.compile(r"\s+")
_MULTI_DANDA_RE = re.compile(r"([।॥])\1+")


def normalize_unicode(text: str) -> str:
    """Fix mojibake/encoding issues and normalize to NFC form."""
    text = ftfy.fix_text(text)
    return unicodedata.normalize("NFC", text)


def devanagari_fraction(text: str) -> float:
    """Fraction of non-space characters that fall in the Devanagari block."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    dev = sum(1 for c in chars if "ऀ" <= c <= "ॿ")
    return dev / len(chars)


def filter_script(text: str, min_devanagari_fraction: float = 0.6) -> str | None:
    """Keep only lines that are predominantly Devanagari script.

    Returns None if the whole document falls below the threshold (i.e. it's
    likely the wrong language / mostly boilerplate / mostly English).
    """
    if devanagari_fraction(text) < min_devanagari_fraction:
        return None
    lines = [
        line for line in text.splitlines()
        if devanagari_fraction(line) >= min_devanagari_fraction or line.strip() == ""
    ]
    return "\n".join(lines)


def strip_disallowed_chars(text: str) -> str:
    """Remove characters outside Devanagari + basic punctuation/digits."""
    return _KEEP_CHARS_RE.sub("", text)


def clean_whitespace(text: str) -> str:
    text = _MULTI_DANDA_RE.sub(r"\1", text)
    text = _MULTI_WHITESPACE_RE.sub(" ", text)
    return text.strip()


def clean_document(text: str, min_devanagari_fraction: float = 0.6) -> str | None:
    """Full pipeline: normalize -> script-filter -> strip -> whitespace clean."""
    text = normalize_unicode(text)
    text = filter_script(text, min_devanagari_fraction)
    if text is None:
        return None
    text = strip_disallowed_chars(text)
    text = clean_whitespace(text)
    return text if len(text) >= 20 else None  # drop near-empty scraps


def content_hash(text: str) -> str:
    """Hash used for exact-duplicate detection across sources."""
    normalized = _MULTI_WHITESPACE_RE.sub(" ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def dedup_documents(docs: list[dict]) -> list[dict]:
    """Exact-duplicate removal by content hash. `docs` items need a 'text' key."""
    seen = set()
    out = []
    for d in docs:
        h = content_hash(d["text"])
        if h in seen:
            continue
        seen.add(h)
        d["content_hash"] = h
        out.append(d)
    return out
