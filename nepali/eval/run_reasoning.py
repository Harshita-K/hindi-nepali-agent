"""Reasoning-task eval for Model L (Nepali) -- identical protocol to
hindi/eval/run_reasoning.py; see that file's docstring for the rationale.

Default (greedy decoding -- the project's reported numbers):
    python nepali/eval/run_reasoning.py \
        --pretrained_ckpt /content/drive/MyDrive/LMA/nepali/output/best.pt \
        --finetuned_ckpt /content/drive/MyDrive/LMA/nepali/finetune_output/best.pt

Temperature-1 diagnostic (writes to a separate report path):
    python nepali/eval/run_reasoning.py \
        --pretrained_ckpt /content/drive/MyDrive/LMA/nepali/output/best.pt \
        --finetuned_ckpt /content/drive/MyDrive/LMA/nepali/finetune_output/best.pt \
        --temperature 1.0 --report_path report/phase3/nepali_reasoning_eval_temp1.json

Answer-restricted diagnostic (score only the entities actually named in
each prompt, via teacher-forced log-probability, instead of free
generation -- isolates comparison ability from output-format compliance;
writes to a separate report path):
    python nepali/eval/run_reasoning.py \
        --pretrained_ckpt /content/drive/MyDrive/LMA/nepali/output/best.pt \
        --finetuned_ckpt /content/drive/MyDrive/LMA/nepali/finetune_output/best.pt \
        --answer_restricted --report_path report/phase3/nepali_reasoning_eval_restricted.json
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import sentencepiece as spm
import torch

from common.finetune.data import load_jsonl
from common.finetune.reasoning_eval import evaluate_answer_restricted, evaluate_exact_match, pick_qualitative_examples
from common.model.config import GPTConfig
from common.model.transformer import GPT
from common.train.checkpoint import load_checkpoint

LANG_DIR = Path(__file__).resolve().parents[1]
MODEL_CFG_PATH = LANG_DIR / "configs" / "model_config.yaml"
TEST_JSONL = LANG_DIR / "data" / "reasoning" / "test.jsonl"
SPM_MODEL = LANG_DIR / "tokenizer" / "vocab" / "nepali_spm.model"
REPORT_PATH = REPO_ROOT / "report" / "phase3" / "nepali_reasoning_eval.json"


def load_model(ckpt_path: str, model_cfg: GPTConfig, device: str) -> GPT:
    model = GPT(model_cfg).to(device)
    load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()
    return model


def _load_name_pool_and_equal_phrase():
    """Loads PERSON_NAMES/OBJECT_NAMES/both_equal_phrase directly from
    nepali/finetune/generate_reasoning_data.py by file path (not as a
    package import -- nepali/ has no __init__.py, and this avoids any
    naming collision with common.finetune)."""
    gen_path = LANG_DIR / "finetune" / "generate_reasoning_data.py"
    spec = importlib.util.spec_from_file_location("nepali_generate_reasoning_data", gen_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    name_pool = list(mod.PERSON_NAMES) + list(mod.OBJECT_NAMES)
    return name_pool, mod.NEPALI_CONFIG.both_equal_phrase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", required=True)
    parser.add_argument("--finetuned_ckpt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=None, help="None (default) = greedy decoding. >0 = sample at this temperature (seed-dependent -- diagnostic only). Ignored if --answer_restricted is set.")
    parser.add_argument("--seed", type=int, default=42, help="Only matters when --temperature is set (greedy decoding is already deterministic).")
    parser.add_argument("--answer_restricted", action="store_true", help="Score only the entities actually named in each prompt (teacher-forced log-probability) instead of free generation.")
    parser.add_argument("--report_path", default=str(REPORT_PATH), help="Where to save the result JSON (default: report/phase3/nepali_reasoning_eval.json)")
    args = parser.parse_args()

    report_path = Path(args.report_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.temperature and not args.answer_restricted:
        torch.manual_seed(args.seed)
    sp = spm.SentencePieceProcessor(model_file=str(SPM_MODEL))
    model_cfg = GPTConfig.from_yaml(MODEL_CFG_PATH)
    test_examples = load_jsonl(TEST_JSONL)

    mode = "answer-restricted" if args.answer_restricted else f"free generation (temperature={args.temperature})"

    if args.answer_restricted:
        name_pool, equal_phrase = _load_name_pool_and_equal_phrase()

        def run_eval(model):
            return evaluate_answer_restricted(model, sp, test_examples, name_pool, equal_phrase, device)
    else:
        def run_eval(model):
            return evaluate_exact_match(model, sp, test_examples, device, args.max_new_tokens, temperature=args.temperature)

    print(f"Evaluating pretrained checkpoint on {len(test_examples)} test examples ({mode})...")
    pretrained_model = load_model(args.pretrained_ckpt, model_cfg, device)
    pretrained_results = run_eval(pretrained_model)
    del pretrained_model
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"Evaluating finetuned checkpoint on {len(test_examples)} test examples ({mode})...")
    finetuned_model = load_model(args.finetuned_ckpt, model_cfg, device)
    finetuned_results = run_eval(finetuned_model)

    fixed, still_wrong = pick_qualitative_examples(pretrained_results["records"], finetuned_results["records"])

    summary = {
        "mode": "answer_restricted" if args.answer_restricted else "free_generation",
        "temperature": None if args.answer_restricted else args.temperature,
        "seed": args.seed if (args.temperature and not args.answer_restricted) else None,
        "pretrained": {k: v for k, v in pretrained_results.items() if k != "records"},
        "finetuned": {k: v for k, v in finetuned_results.items() if k != "records"},
        "examples_finetuning_fixed": fixed,
        "examples_still_wrong_after_finetuning": still_wrong,
    }

    print(json.dumps({k: v for k, v in summary.items() if "examples" not in k}, ensure_ascii=False, indent=2))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to {report_path}")


if __name__ == "__main__":
    main()
