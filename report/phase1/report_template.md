# Phase 1 Report — Data Collection and Tokenizer Construction

## 1. Language selection

- **Model H (higher-resource):** Hindi. Justification: <fill in — e.g. largest
  Indian-language public text volume: Wikipedia, IndicCorp, OSCAR/CC100, large
  news ecosystem>.
- **Model L (lower-resource):** Nepali (from the allowed list).

## 2. Dataset statistics

Fill from `report/phase1/hindi_corpus_stats.json` and `nepali_corpus_stats.json`.

| | Hindi (Model H) | Nepali (Model L) |
|---|---|---|
| Total documents | | |
| Total tokens (whitespace proxy) | | |
| Target tokens | 500,000,000 | 500,000,000 (or justified shortfall) |
| Manual token fraction | | |
| Manual sources | Hindi news scraping, OCR'd books | Nepali news scraping, OCR'd books |
| Downloaded sources | Hindi Wikipedia, OSCAR/IndicCorp | Nepali Wikipedia, OSCAR/IndicCorp |
| Train / Val / Test doc counts | | |

**Shortfall justification (Nepali, if applicable):** <fill in>

## 3. Cleaning steps applied

1. Unicode normalization (NFC) + mojibake repair (`ftfy`).
2. Script filtering — keep only lines with >=60% Devanagari characters.
3. Whitespace/punctuation normalization.
4. Exact-duplicate removal via content hashing.
5. Minimum length filter (drop docs < 20 chars post-cleaning).

## 4. Tokenizer

Fill from `report/phase1/hindi_tokenizer_report.json` / `nepali_tokenizer_report.json`.

| | Hindi | Nepali |
|---|---|---|
| Algorithm | SentencePiece BPE | SentencePiece BPE |
| Vocabulary size | 32,000 | 24,000 |
| Avg chars/token | | |
| Unknown-token rate | | |

**Tokenization examples:** <paste a few example sentences + their pieces from
the tokenizer report>

## 5. Manual collection description

Describe concretely what you scraped/OCR'd by hand:
- Sites scraped (list URLs/domains used in `urls_manual.txt`).
- Books/PDFs OCR'd (list titles/sources, e.g. archive.org, Wikisource).
- Any text typed/transcribed manually.
