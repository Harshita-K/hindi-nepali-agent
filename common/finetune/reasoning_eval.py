"""Exact-match evaluation for the reasoning finetune task.

Generates the model's answer from each test prompt and string-compares it
against the ground-truth answer. Defaults to greedy decoding (stopping at
EOS -- deterministic, appropriate for exact-match scoring, and the setting
used for the project's reported numbers); an optional `temperature` runs
multinomial sampling instead, for checking whether accuracy changes under
non-greedy decoding (results become seed-dependent at temperature > 0, so
this is a diagnostic, not a replacement for the greedy numbers). Meant to
be run twice per language on the same held-out test set -- once with the
*pretrained* checkpoint, once with the *finetuned* one -- so the reported
number is a real before/after comparison, not finetuned-accuracy in
isolation.
"""
from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn.functional as F

from common.eval.generate import generate


def generate_answer(model, sp, prompt: str, device: str, max_new_tokens: int = 8, temperature: Optional[float] = None) -> str:
    """Generate from `prompt` (which already ends in the language's answer
    cue, e.g. "उत्तर:") and return just the generated continuation, stripped
    and with any trailing EOS token removed. `temperature=None` (default) is
    greedy/argmax decoding; `temperature>0` samples instead."""
    prompt_ids = sp.encode(prompt, out_type=int)
    idx = torch.tensor([prompt_ids], device=device)
    out = generate(model, idx, max_new_tokens, temperature=temperature, eos_id=sp.eos_id())
    gen_ids = out[0, len(prompt_ids):].tolist()
    if gen_ids and gen_ids[-1] == sp.eos_id():
        gen_ids = gen_ids[:-1]
    return sp.decode(gen_ids).strip()


def evaluate_exact_match(model, sp, test_examples: List[dict], device: str, max_new_tokens: int = 8, temperature: Optional[float] = None) -> dict:
    """Runs generation over every test example and reports overall accuracy
    plus a breakdown by task type (direct/transitive/multihop) and
    attribute (age/height/price/quantity), since a flat accuracy number
    can hide that a model does well on simple direct comparisons but fails
    at the multi-hop chaining the spec cares most about."""
    model.eval()
    records = []
    for ex in test_examples:
        pred = generate_answer(model, sp, ex["prompt"], device, max_new_tokens, temperature=temperature)
        is_correct = pred.strip() == ex["answer"].strip()
        records.append({
            "prompt": ex["prompt"],
            "answer": ex["answer"],
            "predicted": pred,
            "correct": is_correct,
            "task_type": ex["task_type"],
            "attribute": ex["attribute"],
        })

    by_task = defaultdict(lambda: [0, 0])
    by_attr = defaultdict(lambda: [0, 0])
    for r in records:
        by_task[r["task_type"]][0] += int(r["correct"])
        by_task[r["task_type"]][1] += 1
        by_attr[r["attribute"]][0] += int(r["correct"])
        by_attr[r["attribute"]][1] += 1

    n = len(records)
    correct = sum(r["correct"] for r in records)
    return {
        "overall_accuracy": correct / n if n else 0.0,
        "n_examples": n,
        "accuracy_by_task_type": {k: v[0] / v[1] for k, v in by_task.items()},
        "accuracy_by_attribute": {k: v[0] / v[1] for k, v in by_attr.items()},
        "records": records,
    }


def pick_qualitative_examples(pretrained_records, finetuned_records, n_fixed=10, n_still_wrong=5):
    """Pairs up the two runs' records by prompt (both ran on the same test
    set in the same order, so zip is safe) and picks the most informative
    cases: examples finetuning *fixed* (pretrained wrong, finetuned right --
    direct evidence finetuning worked) and examples still wrong after
    finetuning (for honest error analysis)."""
    fixed, still_wrong = [], []
    for pre, fin in zip(pretrained_records, finetuned_records):
        assert pre["prompt"] == fin["prompt"], "pretrained/finetuned records must be aligned to the same test set"
        pair = {
            "prompt": pre["prompt"],
            "answer": pre["answer"],
            "pretrained_predicted": pre["predicted"],
            "finetuned_predicted": fin["predicted"],
            "task_type": pre["task_type"],
        }
        if not pre["correct"] and fin["correct"]:
            fixed.append(pair)
        elif not fin["correct"]:
            still_wrong.append(pair)
    return fixed[:n_fixed], still_wrong[:n_still_wrong]


@torch.no_grad()
def score_candidate(model, sp, prompt: str, candidate: str, device: str) -> float:
    """Length-normalized log-probability the model assigns to `candidate`
    (+ EOS) as a continuation of `prompt`, via one teacher-forced forward
    pass -- no generation/sampling involved. This is the scoring primitive
    behind answer-restricted (multiple-choice-style) evaluation: instead of
    asking the model to freely generate and hoping it happens to emit
    exactly the right bare string (evaluate_exact_match's free-generation
    approach), this asks "which of a small set of known-valid candidates
    does the model find most probable," which isolates comparison/reasoning
    ability from output-formatting/instruction-following ability. Averaging
    (not summing) log-probs over the candidate span avoids biasing toward
    shorter candidate strings."""
    prompt_ids = sp.encode(prompt, out_type=int)
    cand_ids = sp.encode(" " + candidate, out_type=int) + [sp.eos_id()]
    full_ids = prompt_ids + cand_ids
    idx = torch.tensor([full_ids[:-1]], device=device)
    target = torch.tensor(full_ids[1:], device=device)

    logits, _, _ = model(idx, targets=None)
    logits = logits[0]  # (T, vocab)

    # Same span-boundary convention as common/finetune/data.py::encode_example:
    # target[j] is part of the candidate+EOS span iff j >= len(prompt_ids) - 1.
    start = len(prompt_ids) - 1
    cand_logits = logits[start:]
    cand_target = target[start:]

    log_probs = F.log_softmax(cand_logits, dim=-1)
    token_logprobs = log_probs.gather(1, cand_target.unsqueeze(1)).squeeze(1)
    return token_logprobs.mean().item()


def extract_candidates(prompt: str, name_pool: List[str], equal_phrase: Optional[str] = None) -> List[str]:
    """Recovers the small set of valid answer choices for one prompt by
    scanning it for occurrences of known entity names -- checked
    longest-first so a name that happens to be a substring of a longer one
    isn't matched instead of it. `equal_phrase` (e.g. "दोनों बराबर हैं") is
    always appended if given, since "both equal" is a valid answer for any
    `direct` example regardless of which two entities are actually named."""
    found = []
    for name in sorted(set(name_pool), key=len, reverse=True):
        if name in prompt and name not in found:
            found.append(name)
    if equal_phrase:
        found.append(equal_phrase)
    return found


def evaluate_answer_restricted(
    model, sp, test_examples: List[dict], name_pool: List[str], equal_phrase: str, device: str,
) -> dict:
    """Answer-restricted counterpart to evaluate_exact_match: for each
    example, restricts the model's answer to the small set of entities
    actually named in that prompt (plus the fixed "both equal" phrase),
    scores each candidate by teacher-forced log-probability, and picks the
    argmax. A model can score well here even if it would never
    spontaneously emit a bare entity name under free generation -- this is
    what makes pretrained (non-finetuned) accuracy typically nonzero here,
    unlike evaluate_exact_match's free-generation eval."""
    model.eval()
    records = []
    skipped = 0
    for ex in test_examples:
        candidates = extract_candidates(ex["prompt"], name_pool, equal_phrase)
        if len(candidates) < 2:
            skipped += 1
            continue
        scores = [score_candidate(model, sp, ex["prompt"], c, device) for c in candidates]
        pred = candidates[max(range(len(scores)), key=lambda i: scores[i])]
        is_correct = pred.strip() == ex["answer"].strip()
        records.append({
            "prompt": ex["prompt"],
            "answer": ex["answer"],
            "predicted": pred,
            "candidates": candidates,
            "correct": is_correct,
            "task_type": ex["task_type"],
            "attribute": ex["attribute"],
        })

    by_task = defaultdict(lambda: [0, 0])
    by_attr = defaultdict(lambda: [0, 0])
    for r in records:
        by_task[r["task_type"]][0] += int(r["correct"])
        by_task[r["task_type"]][1] += 1
        by_attr[r["attribute"]][0] += int(r["correct"])
        by_attr[r["attribute"]][1] += 1

    n = len(records)
    correct = sum(r["correct"] for r in records)
    return {
        "overall_accuracy": correct / n if n else 0.0,
        "n_examples": n,
        "n_skipped_fewer_than_2_candidates": skipped,
        "accuracy_by_task_type": {k: v[0] / v[1] for k, v in by_task.items()},
        "accuracy_by_attribute": {k: v[0] / v[1] for k, v in by_attr.items()},
        "records": records,
    }
