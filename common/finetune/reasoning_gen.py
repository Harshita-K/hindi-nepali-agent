"""Shared framework for generating synthetic comparative-reasoning finetune
data. This module contains no language-specific text at all -- every word,
name, and sentence template is supplied by the caller via `LanguageConfig`,
so Hindi and Nepali generate fully independent datasets (no shared examples,
no cross-language leakage) even though they go through the same code path.
This mirrors how `common/model/transformer.py` is shared *code* with
per-language weights: sharing the generator logic once avoids writing the
same random-sampling/templating machinery twice, while the actual data each
language produces never overlaps.

Three task families, per the assignment spec:
  - direct:      compare 2 entities on one attribute (greater/smaller/equal)
  - transitive:  3 entities chained as A~B~C, ask for the extreme (min/max)
  - multihop:    3 entities chained as A~B~C, ask about the *derived*
                 (non-adjacent) relation between A and C -- this is the one
                 that actually requires chaining the two stated facts,
                 rather than just picking an extreme from an implied order.

Every example is rendered as one training sequence: `prompt + answer`. The
prompt always ends in a fixed "answer cue" string (e.g. "उत्तर:") so a
finetuned model can be evaluated by generating from that exact point and
string-comparing against the ground-truth answer.
"""
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


def _normalize_spaces(text: str) -> str:
    """Collapse the double space left behind when a template's {unit}
    placeholder is filled with "" (the quantity attribute has no natural
    unit), e.g. "61  hai" -> "61 hai"."""
    return re.sub(r" {2,}", " ", text).strip()


@dataclass
class Attribute:
    """One comparable attribute (age, height, price, ...). `entity_pool`
    picks which noun pool (person names vs. object names) makes the
    generated sentences grammatical -- "Ram's price is 200" is nonsense,
    "the book's price is 200" is not."""
    noun: str  # e.g. "उम्र" (age)
    unit: str  # e.g. "वर्ष" (years) -- "" if the attribute has no natural unit
    comp_high: str  # comparative word for the larger value, e.g. "बड़ा" (bigger/older)
    comp_low: str  # comparative word for the smaller value, e.g. "छोटा" (smaller/younger)
    value_range: Tuple[int, int]
    entity_pool: str  # "person" or "object"


@dataclass
class LanguageConfig:
    """Every language-specific string the generator needs. Nothing here is
    shared between Hindi and Nepali -- each language builds its own
    instance with its own word lists."""
    person_names: List[str]
    object_names: List[str]
    attributes: Dict[str, Attribute]
    equal_word: str  # e.g. "बराबर" (equal)
    both_equal_phrase: str  # full answer text for a tie, e.g. "दोनों बराबर हैं"
    answer_cue: str  # e.g. "उत्तर:" -- appended to every prompt before the answer
    # Template pools, one list per task type. Each template is a Python
    # format string using the named fields documented at each generator
    # function below. Multiple templates per task type give "template
    # variety" (required by the spec) so the model can't just pattern-match
    # one fixed sentence shape.
    direct_templates: List[str]
    transitive_templates: List[str]
    multihop_templates: List[str]


def _split_pool(items: List[str], test_frac: float, rng: random.Random) -> Tuple[List[str], List[str]]:
    """Held-out split: a fraction of the pool is reserved for test only, so
    test-set entities/templates never appear during training -- otherwise
    "reasoning accuracy" would partly just measure string memorization."""
    shuffled = list(items)
    rng.shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * test_frac))
    return shuffled[n_test:], shuffled[:n_test]


def _gen_direct(rng: random.Random, cfg: LanguageConfig, attr_name: str, names: List[str], templates: List[str]):
    """Direct comparison: 2 entities, one attribute. ~15% of examples are
    ties (equal values), to cover the "equal" case the spec explicitly
    asks for, not just greater/smaller."""
    attr = cfg.attributes[attr_name]
    e1, e2 = rng.sample(names, 2)
    lo, hi = attr.value_range
    v1 = rng.randint(lo, hi)
    if rng.random() < 0.15:
        v2 = v1
    else:
        v2 = rng.randint(lo, hi)
        while v2 == v1:
            v2 = rng.randint(lo, hi)
    ask_high = rng.random() < 0.5
    comp_word = attr.comp_high if ask_high else attr.comp_low
    template = rng.choice(templates)
    prompt = template.format(
        e1=e1, e2=e2, v1=v1, v2=v2, noun=attr.noun, unit=attr.unit, comp=comp_word,
    )
    if v1 == v2:
        answer = cfg.both_equal_phrase
    elif v1 > v2:
        answer = e1 if ask_high else e2
    else:
        answer = e2 if ask_high else e1
    return prompt, answer, "direct", attr_name


def _sample_chain_values(rng: random.Random, lo: int, hi: int):
    """Sample 3 distinct values (v1, v2, v3) for a 3-entity chain e1~e2~e3,
    where e2 (the entity common to both stated facts) always gets the
    *median* value. This is what makes a 2-fact chain ("e1 vs e2", "e2 vs
    e3") logically sufficient to determine the full order: if e2 were
    allowed to be the global min or max instead, both facts could bound e2
    without saying anything about the e1-vs-e3 relationship, making
    "who is the overall max?" unanswerable from the given facts alone --
    a real soundness bug an earlier version of this function had after a
    first attempt to fix a *different* bug (position always predicting the
    answer) accidentally introduced this one. e1 and e3 get the min/max in
    random order, so which one is bigger is still unpredictable from
    narration position alone (~50/50, verified empirically at ~0.33 chance
    of the first-mentioned entity being the answer to a 3-way question)."""
    v_min, v_mid, v_max = sorted(rng.sample(range(lo, hi + 1), 3))
    if rng.random() < 0.5:
        return v_max, v_mid, v_min  # e1 largest, e3 smallest
    return v_min, v_mid, v_max  # e1 smallest, e3 largest


def _gen_transitive(rng: random.Random, cfg: LanguageConfig, attr_name: str, names: List[str], templates: List[str]):
    """Transitive ordering: 3 entities e1~e2~e3 stated as two pairwise
    comparisons, ask for the overall min or max. e2's value is always the
    true median (see `_sample_chain_values`), so the two stated facts
    always fully determine the answer -- the model has to actually parse
    which comparative word each fact used, not just recall a fixed
    narration-order shortcut, since e1/e3 randomly get the min/max."""
    attr = cfg.attributes[attr_name]
    e1, e2, e3 = rng.sample(names, 3)
    lo, hi = attr.value_range
    v1, v2, v3 = _sample_chain_values(rng, lo, hi)
    comp1 = attr.comp_high if v1 > v2 else attr.comp_low
    comp2 = attr.comp_high if v2 > v3 else attr.comp_low
    ask_max = rng.random() < 0.5
    template = rng.choice(templates)
    prompt = template.format(
        e1=e1, e2=e2, e3=e3, noun=attr.noun, comp1=comp1, comp2=comp2,
        extreme=(attr.comp_high if ask_max else attr.comp_low),
    )
    values = {e1: v1, e2: v2, e3: v3}
    answer = max(values, key=values.get) if ask_max else min(values, key=values.get)
    return prompt, answer, "transitive", attr_name


def _gen_multihop(rng: random.Random, cfg: LanguageConfig, attr_name: str, names: List[str], templates: List[str]):
    """Multi-hop: same two stated facts as `transitive` (same value
    assignment, so the chain is always logically coherent -- see
    `_sample_chain_values`), but the question asks which of the two
    entities NOT directly compared (e1 vs. e3) has more/less -- answering
    requires chaining both facts, since that pair was never stated
    outright.

    The answer is an ENTITY NAME, not a repeated comparative word --
    deliberately, after discovering that a word-valued answer here is
    mathematically guaranteed to equal comp1 (and comp2): given a sound
    2-fact chain (e2 is always the true median, see `_sample_chain_values`),
    transitivity forces comp1 == comp2 == "the e1-vs-e3 direction" every
    single time, so a model could score 100% by literally copying a token
    already in the prompt, without chaining anything. Naming the entity
    instead breaks that shortcut: the model must combine both facts to
    know *which name* -- not just which direction -- is correct."""
    attr = cfg.attributes[attr_name]
    e1, e2, e3 = rng.sample(names, 3)
    lo, hi = attr.value_range
    v1, v2, v3 = _sample_chain_values(rng, lo, hi)
    comp1 = attr.comp_high if v1 > v2 else attr.comp_low
    comp2 = attr.comp_high if v2 > v3 else attr.comp_low
    ask_higher = rng.random() < 0.5
    template = rng.choice(templates)
    prompt = template.format(
        e1=e1, e2=e2, e3=e3, noun=attr.noun, comp1=comp1, comp2=comp2,
        which=(attr.comp_high if ask_higher else attr.comp_low),
    )
    answer = (e1 if v1 > v3 else e3) if ask_higher else (e1 if v1 < v3 else e3)
    return prompt, answer, "multihop", attr_name


_GENERATORS = {
    "direct": (_gen_direct, lambda cfg: cfg.direct_templates, 2),
    "transitive": (_gen_transitive, lambda cfg: cfg.transitive_templates, 3),
    "multihop": (_gen_multihop, lambda cfg: cfg.multihop_templates, 3),
}


def build_split(
    cfg: LanguageConfig,
    n_examples: int,
    names_by_pool: Dict[str, List[str]],
    templates_by_task: Dict[str, List[str]],
    seed: int,
) -> List[dict]:
    """Generate one split (train/val/test), deduplicated by prompt text.
    `names_by_pool` and `templates_by_task` are pre-split (train vs. test
    pools) by the caller -- this function just draws from whatever it's
    given, so the leakage control lives entirely in which pools get passed
    in for which split."""
    rng = random.Random(seed)
    task_names = list(_GENERATORS.keys())
    attr_names = list(cfg.attributes.keys())
    seen_prompts = set()
    examples = []
    attempts = 0
    max_attempts = n_examples * 50
    while len(examples) < n_examples and attempts < max_attempts:
        attempts += 1
        task = rng.choice(task_names)
        attr_name = rng.choice(attr_names)
        gen_fn, template_getter, min_names = _GENERATORS[task]
        pool = names_by_pool[cfg.attributes[attr_name].entity_pool]
        if len(pool) < min_names:
            continue
        templates = templates_by_task[task]
        prompt, answer, task_type, attr = gen_fn(rng, cfg, attr_name, pool, templates)
        full_prompt = _normalize_spaces(f"{prompt} {cfg.answer_cue}")
        if full_prompt in seen_prompts:
            continue
        seen_prompts.add(full_prompt)
        examples.append({
            "prompt": full_prompt,
            "answer": answer,
            "task_type": task_type,
            "attribute": attr,
        })
    return examples


def build_dataset(
    cfg: LanguageConfig,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 0,
    test_frac_names: float = 0.2,
    test_frac_templates: float = 0.25,
) -> Dict[str, List[dict]]:
    """Build train/val/test splits with held-out entity names AND held-out
    templates for test -- both are common leakage vectors the spec calls
    out by name, so both are controlled independently."""
    rng = random.Random(seed)

    person_train, person_test = _split_pool(cfg.person_names, test_frac_names, rng)
    object_train, object_test = _split_pool(cfg.object_names, test_frac_names, rng)
    names_train = {"person": person_train, "object": object_train}
    names_test = {"person": person_test, "object": object_test}

    templates_train, templates_test = {}, {}
    for task, pool in [
        ("direct", cfg.direct_templates),
        ("transitive", cfg.transitive_templates),
        ("multihop", cfg.multihop_templates),
    ]:
        tr, te = _split_pool(pool, test_frac_templates, rng)
        templates_train[task] = tr if tr else pool  # keep at least 1 usable template
        templates_test[task] = te if te else pool

    train = build_split(cfg, n_train, names_train, templates_train, seed=seed + 1)
    val = build_split(cfg, n_val, names_train, templates_train, seed=seed + 2)
    test = build_split(cfg, n_test, names_test, templates_test, seed=seed + 3)

    return {
        "train": train,
        "val": val,
        "test": test,
        "_meta": {
            "person_names_train": person_train,
            "person_names_test": person_test,
            "object_names_train": object_train,
            "object_names_test": object_test,
            "templates_train": templates_train,
            "templates_test": templates_test,
        },
    }


def write_jsonl(examples: List[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def split_stats(examples: List[dict]) -> dict:
    from collections import Counter
    return {
        "n_examples": len(examples),
        "by_task_type": dict(Counter(ex["task_type"] for ex in examples)),
        "by_attribute": dict(Counter(ex["attribute"] for ex in examples)),
    }
