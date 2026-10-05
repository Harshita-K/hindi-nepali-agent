"""Sweep candidate SentencePiece vocab sizes for one language and report
fertility / unknown-token rate / embedding-parameter-cost trade-offs on
held-out validation text -- gives the Phase 1 vocab-size choice (8,000) an
actual data-driven comparison, instead of only the upfront parameter-budget
argument in report/phase1/report.md.

Trains temporary tokenizers into a throwaway scratch directory -- never
touches the real, already-graded {lang}/tokenizer/vocab/{lang}_spm.model
that Phase 2 pretraining actually uses.

    python -m common.tokenizer_vocab_sweep --lang hindi
    python -m common.tokenizer_vocab_sweep --lang nepali --vocab_sizes 2000 4000 8000 16000 24000 32000
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import sentencepiece as spm
import yaml

LANGS = {
    "hindi": REPO_ROOT / "hindi",
    "nepali": REPO_ROOT / "nepali",
}


def jsonl_to_plain_text(jsonl_path: Path, out_path: Path, max_docs: int = None) -> None:
    """Writes every doc's text (max_docs=None) or, when max_docs is set, a
    systematic (every-Nth-line) sample of ~max_docs docs -- avoids the
    ordering bias a naive head -n N would carry, since build_corpus.py
    sorts documents manual-first before deduping (see report/phase1/report.md
    section 8), so the first N lines of a full split skew manual. Also
    avoids materializing a full-corpus-sized plain-text copy on disk, which
    for a multi-GB train.jsonl can exceed available disk space."""
    if max_docs is None:
        with jsonl_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
            for line in fin:
                text = json.loads(line)["text"]
                fout.write(text.replace("\n", " ") + "\n")
        return

    with jsonl_path.open(encoding="utf-8") as fin:
        total_lines = sum(1 for _ in fin)
    stride = max(1, total_lines // max_docs)

    with jsonl_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        written = 0
        for i, line in enumerate(fin):
            if i % stride == 0 and written < max_docs:
                text = json.loads(line)["text"]
                fout.write(text.replace("\n", " ") + "\n")
                written += 1


def train_and_eval(
    train_txt: Path, val_jsonl: Path, vocab_size: int, model_type: str,
    d_model: int, scratch_dir: Path, n_val_docs: int,
) -> dict:
    """Train one candidate tokenizer, then evaluate it on the SAME val-doc
    sample used for every other vocab size in the sweep, so fertility/UNK
    numbers are directly comparable across the sweep (not an artifact of
    encoding a different sample each time)."""
    model_prefix = scratch_dir / f"sweep_{vocab_size}"
    spm.SentencePieceTrainer.Train(
        input=str(train_txt),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=0.9995,  # matches train_tokenizer.py exactly
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    )
    sp = spm.SentencePieceProcessor(model_file=str(model_prefix) + ".model")

    docs = [json.loads(l) for l in val_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()][:n_val_docs]
    total_chars = total_tokens = total_unk = total_words = 0
    used_ids = set()
    for d in docs:
        ids = sp.encode(d["text"], out_type=int)
        total_chars += len(d["text"])
        total_tokens += len(ids)
        total_words += len(d["text"].split())  # whitespace-word count -- fertility = tokens/word
        total_unk += sum(1 for i in ids if i == sp.unk_id())
        used_ids.update(ids)

    embedding_params = vocab_size * d_model
    return {
        "vocab_size": vocab_size,
        # Standard tokenizer-fertility definition: subword tokens per
        # whitespace word (not chars/token, which is a related but
        # different compression-ratio metric -- kept alongside for reference).
        "fertility_tokens_per_word": round(total_tokens / total_words, 4),
        "avg_chars_per_token": round(total_chars / total_tokens, 3),
        "unknown_token_rate_pct": round(100 * total_unk / total_tokens, 4),
        "vocab_pieces_used_fraction": round(len(used_ids) / vocab_size, 4),
        # Lower vocab -> higher fertility -> more tokens needed to encode the
        # SAME val-doc sample -> longer sequences -> more attention compute
        # per training step at a fixed block_size/batch_size. This is the
        # other side of the trade-off from embedding-table size below.
        "tokens_to_encode_fixed_val_sample": total_tokens,
        "embedding_table_params": embedding_params,
        "embedding_pct_of_25M_budget": round(100 * embedding_params / 25_000_000, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, choices=list(LANGS))
    parser.add_argument("--vocab_sizes", type=int, nargs="+", default=[2000, 4000, 8000, 16000, 24000, 32000])
    parser.add_argument("--model_type", default=None, help="Default: read from configs/data_config.yaml (bpe)")
    parser.add_argument("--n_val_docs", type=int, default=2000, help="Matches train_tokenizer.py's own eval sample size")
    parser.add_argument("--d_model", type=int, default=None, help="Default: read from configs/model_config.yaml (512)")
    parser.add_argument(
        "--max_train_docs", type=int, default=None,
        help="Systematically sample this many train docs instead of using the full split "
        "(recommended for a multi-GB train.jsonl -- the full corpus is temporarily duplicated "
        "as plain text, which can exceed available disk space). None = full corpus.",
    )
    args = parser.parse_args()

    lang_dir = LANGS[args.lang]
    train_jsonl = lang_dir / "data" / "splits" / "train.jsonl"
    val_jsonl = lang_dir / "data" / "splits" / "val.jsonl"
    report_path = REPO_ROOT / "report" / "phase1" / f"{args.lang}_vocab_size_sweep.json"

    data_cfg = yaml.safe_load((lang_dir / "configs" / "data_config.yaml").read_text())
    model_cfg = yaml.safe_load((lang_dir / "configs" / "model_config.yaml").read_text())
    model_type = args.model_type or data_cfg["tokenizer"]["model_type"]
    d_model = args.d_model or model_cfg["d_model"]
    actual_vocab_size = data_cfg["tokenizer"]["vocab_size"]

    with tempfile.TemporaryDirectory(prefix=f"vocab_sweep_{args.lang}_") as tmp:
        scratch_dir = Path(tmp)
        train_txt = scratch_dir / "train_plain.txt"
        print(f"Converting {train_jsonl} to plain text (one-time, shared across the sweep)"
              + (f", systematic sample of ~{args.max_train_docs} docs..." if args.max_train_docs else "..."))
        jsonl_to_plain_text(train_jsonl, train_txt, max_docs=args.max_train_docs)

        results = []
        for vs in sorted(set(args.vocab_sizes)):
            print(f"Training {model_type} vocab_size={vs} ...")
            results.append(train_and_eval(train_txt, val_jsonl, vs, model_type, d_model, scratch_dir, args.n_val_docs))

    report = {
        "language": args.lang,
        "model_type": model_type,
        "d_model": d_model,
        "n_val_docs": args.n_val_docs,
        "max_train_docs": args.max_train_docs,
        "sampling": "systematic (every-Nth-line)" if args.max_train_docs else "full corpus, no sampling",
        "chosen_vocab_size": actual_vocab_size,
        "sweep": results,
    }
    print("\n" + json.dumps(report, indent=2))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved to {report_path}")


if __name__ == "__main__":
    main()
