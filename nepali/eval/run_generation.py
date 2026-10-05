"""Generation-quality eval for Model L (Nepali): greedy decoding plus
temperature sampling (0.5, 1.0, 1.5) from held-out test prefixes, scored
against reference continuations with BLEU-4 / chrF++ / ROUGE-L, plus
repetition-rate / Distinct-1 / Distinct-2 diversity diagnostics.

    python nepali/eval/run_generation.py --ckpt /content/drive/MyDrive/LMA/nepali/output/best.pt
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import sentencepiece as spm
import torch

from common.eval.generate import generate
from common.eval.metrics import avg_rouge_l, corpus_bleu_chrf, diversity_stats
from common.model.config import GPTConfig
from common.model.transformer import GPT
from common.train.checkpoint import load_checkpoint

LANG_DIR = Path(__file__).resolve().parents[1]
MODEL_CFG_PATH = LANG_DIR / "configs" / "model_config.yaml"
TEST_JSONL = LANG_DIR / "data" / "splits" / "test.jsonl"
SPM_MODEL = LANG_DIR / "tokenizer" / "vocab" / "nepali_spm.model"
REPORT_PATH = REPO_ROOT / "report" / "phase2" / "nepali_generation.json"
SAMPLES_PATH = REPO_ROOT / "report" / "phase2" / "nepali_generation_samples.txt"

CONDITIONS = [("greedy", None), ("temp_0.5", 0.5), ("temp_1.0", 1.0), ("temp_1.5", 1.5)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n_examples", type=int, default=100)
    parser.add_argument("--prefix_len", type=int, default=32)
    parser.add_argument("--gen_len", type=int, default=64)
    parser.add_argument("--n_qualitative", type=int, default=5, help="How many examples to save as readable text samples")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = spm.SentencePieceProcessor(model_file=str(SPM_MODEL))

    model_cfg = GPTConfig.from_yaml(MODEL_CFG_PATH)
    model = GPT(model_cfg).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)

    docs = [json.loads(l) for l in TEST_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    docs = docs[: args.n_examples]

    results = {}
    qualitative_lines = []

    for cond_name, temp in CONDITIONS:
        hyps, refs = [], []
        for i, doc in enumerate(docs):
            ids = sp.encode(doc["text"], out_type=int)
            if len(ids) < args.prefix_len + args.gen_len:
                continue
            prefix_ids = ids[: args.prefix_len]
            ref_ids = ids[args.prefix_len : args.prefix_len + args.gen_len]

            idx = torch.tensor([prefix_ids], device=device)
            out = generate(model, idx, args.gen_len, temperature=temp, eos_id=sp.eos_id())
            gen_ids = out[0, args.prefix_len :].tolist()

            hyp_text = sp.decode(gen_ids)
            ref_text = sp.decode(ref_ids)
            hyps.append(hyp_text)
            refs.append(ref_text)

            if i < args.n_qualitative:
                qualitative_lines.append(
                    f"=== {cond_name} | example {i} ===\n"
                    f"PREFIX: {sp.decode(prefix_ids)}\n"
                    f"REFERENCE: {ref_text}\n"
                    f"GENERATED: {hyp_text}\n"
                )

        bleu_chrf = corpus_bleu_chrf(hyps, refs)
        rouge = avg_rouge_l(hyps, refs)
        div = diversity_stats(hyps)
        results[cond_name] = {**bleu_chrf, **rouge, **div, "n_examples": len(hyps)}
        print(cond_name, json.dumps(results[cond_name], indent=2))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    SAMPLES_PATH.write_text("\n".join(qualitative_lines), encoding="utf-8")
    print(f"Saved metrics to {REPORT_PATH}, samples to {SAMPLES_PATH}")


if __name__ == "__main__":
    main()
