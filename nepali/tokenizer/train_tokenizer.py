"""Train a Nepali-only SentencePiece tokenizer from scratch on the train
split. No pretrained tokenizers are used, per the assignment's hard
constraint. Also reports fertility / unknown-token rate / token-frequency
statistics on the val split.

Run from repo root:
    python nepali/tokenizer/train_tokenizer.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import sentencepiece as spm
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

LANG_DIR = Path(__file__).resolve().parents[1]  # nepali/
CONFIG_PATH = LANG_DIR / "configs" / "data_config.yaml"
TRAIN_JSONL = LANG_DIR / "data" / "splits" / "train.jsonl"
VAL_JSONL = LANG_DIR / "data" / "splits" / "val.jsonl"
VOCAB_DIR = LANG_DIR / "tokenizer" / "vocab"
MODEL_PREFIX = VOCAB_DIR / "nepali_spm"
PLAIN_TEXT_TMP = LANG_DIR / "data" / "splits" / "_train_plain.txt"
REPORT_PATH = REPO_ROOT / "report" / "phase1" / "nepali_tokenizer_report.json"


def jsonl_to_plain_text(jsonl_path: Path, out_path: Path) -> None:
    with jsonl_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            text = json.loads(line)["text"]
            fout.write(text.replace("\n", " ") + "\n")


def train(vocab_size: int, model_type: str) -> None:
    VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_to_plain_text(TRAIN_JSONL, PLAIN_TEXT_TMP)
    spm.SentencePieceTrainer.Train(
        input=str(PLAIN_TEXT_TMP),
        model_prefix=str(MODEL_PREFIX),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=0.9995,  # high coverage needed for Devanagari + rare glyphs
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    )


def compute_token_frequency_stats(sp: spm.SentencePieceProcessor, id_counts: Counter, total_tokens: int) -> dict:
    """Distribution of how often each vocab piece was actually used, over
    the same val-split sample evaluate_on_val() encodes. Reports the top-N
    most frequent pieces (id, piece string, count, share of all tokens),
    how much of the vocabulary went entirely unused, and summary stats
    (min/max/mean/median) over pieces that did appear at least once.
    """
    vocab_size = sp.get_piece_size()
    used_ids = set(id_counts)
    unused = vocab_size - len(used_ids)

    top_n = 20
    top = [
        {
            "id": tid,
            "piece": sp.id_to_piece(tid),
            "count": count,
            "share_of_tokens": round(count / total_tokens, 6) if total_tokens else None,
        }
        for tid, count in id_counts.most_common(top_n)
    ]

    counts = sorted(id_counts.values())
    n = len(counts)
    median = counts[n // 2] if n % 2 else (counts[n // 2 - 1] + counts[n // 2]) / 2

    return {
        "vocab_pieces_used": len(used_ids),
        "vocab_pieces_unused": unused,
        "vocab_pieces_unused_fraction": round(unused / vocab_size, 4) if vocab_size else None,
        "used_piece_frequency": {
            "min": counts[0] if counts else None,
            "max": counts[-1] if counts else None,
            "mean": round(sum(counts) / n, 2) if n else None,
            "median": median if counts else None,
        },
        "top_pieces": top,
    }


def evaluate_on_val(sp: spm.SentencePieceProcessor) -> dict:
    docs = [json.loads(line) for line in VAL_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    total_chars, total_tokens, total_unk = 0, 0, 0
    id_counts: Counter = Counter()
    examples = []
    for d in docs[:2000]:
        text = d["text"]
        ids = sp.encode(text, out_type=int)
        total_chars += len(text)
        total_tokens += len(ids)
        total_unk += sum(1 for i in ids if i == sp.unk_id())
        id_counts.update(ids)
        if len(examples) < 5:
            examples.append({"text": text[:80], "pieces": sp.encode(text[:80], out_type=str)})
    return {
        "vocab_size": sp.get_piece_size(),
        "avg_chars_per_token": round(total_chars / total_tokens, 3) if total_tokens else None,
        "unknown_token_rate": round(total_unk / total_tokens, 6) if total_tokens else None,
        "token_frequency_stats": compute_token_frequency_stats(sp, id_counts, total_tokens),
        "examples": examples,
    }


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    tok_cfg = config["tokenizer"]
    train(vocab_size=tok_cfg["vocab_size"], model_type=tok_cfg["model_type"])

    sp = spm.SentencePieceProcessor(model_file=str(MODEL_PREFIX) + ".model")
    report = evaluate_on_val(sp)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    PLAIN_TEXT_TMP.unlink(missing_ok=True)
    print(f"Tokenizer saved to {MODEL_PREFIX}.model / .vocab")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
