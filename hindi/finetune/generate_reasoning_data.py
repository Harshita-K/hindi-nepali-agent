"""Generate Hindi's synthetic comparative-reasoning finetune dataset.

Grammar note: every template phrases comparisons as "X ki {noun} ... hai"
(X's {attribute} is ...) rather than "X bada hai" (X is bigger) -- the
attribute noun (umra/oonchai/keemat/matra, all feminine) is always the
grammatical subject, and the comparative words (zyaada/kam) don't inflect
for gender. This means the generator never needs to track each entity
name's grammatical gender to produce a correct sentence, while the
resulting Hindi is still natural ("kiski umra zyaada hai?" is exactly how
this question is actually asked).

Run directly to write hindi/data/reasoning/{train,val,test}.jsonl and print
a summary. `--n_train/--n_val/--n_test` override the default sizes.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.finetune.reasoning_gen import Attribute, LanguageConfig, build_dataset, split_stats, write_jsonl

PERSON_NAMES = [
    "राम", "श्याम", "गीता", "सीता", "मोहन", "सुनीता", "विजय", "कविता",
    "अनिल", "रीता", "सुरेश", "प्रिया", "राजेश", "नेहा", "अमित", "पूजा",
    "संजय", "दीपा", "विकास", "माया", "रोहन", "स्वाति", "मनोज", "आरती",
    "करण", "इशा", "गौरव", "सपना", "तरुण", "मीना",
]

OBJECT_NAMES = [
    "किताब", "पेन", "बैग", "घड़ी", "साइकिल", "मोबाइल", "कुर्सी", "मेज़",
    "टोकरी", "जूता", "छाता", "थैला", "चश्मा", "पंखा", "लैंप",
]

ATTRIBUTES = {
    "age": Attribute(
        noun="उम्र", unit="वर्ष", comp_high="ज़्यादा", comp_low="कम",
        value_range=(8, 80), entity_pool="person",
    ),
    "height": Attribute(
        noun="ऊँचाई", unit="सेंटीमीटर", comp_high="ज़्यादा", comp_low="कम",
        value_range=(100, 195), entity_pool="person",
    ),
    "price": Attribute(
        noun="कीमत", unit="रुपये", comp_high="ज़्यादा", comp_low="कम",
        value_range=(10, 2000), entity_pool="object",
    ),
    "quantity": Attribute(
        noun="मात्रा", unit="", comp_high="ज़्यादा", comp_low="कम",
        value_range=(1, 100), entity_pool="object",
    ),
}

DIRECT_TEMPLATES = [
    "{e1} की {noun} {v1} {unit} है। {e2} की {noun} {v2} {unit} है। किसकी {noun} {comp} है?",
    "{e1} की {noun} {v1} {unit} है, जबकि {e2} की {noun} {v2} {unit} है। बताइए किसकी {noun} {comp} है।",
    "मान लीजिए {e1} की {noun} {v1} {unit} है और {e2} की {noun} {v2} {unit} है। इन दोनों में से किसकी {noun} {comp} है?",
    "अगर {e1} की {noun} {v1} {unit} हो और {e2} की {noun} {v2} {unit} हो, तो किसकी {noun} {comp} होगी?",
]

TRANSITIVE_TEMPLATES = [
    "{e1} की {noun}, {e2} की {noun} से {comp1} है। {e2} की {noun}, {e3} की {noun} से {comp2} है। इन तीनों में सबसे {extreme} {noun} किसकी है?",
    "पता है कि {e1} की {noun}, {e2} की {noun} से {comp1} है, और {e2} की {noun}, {e3} की {noun} से {comp2} है। इनमें सबसे {extreme} {noun} किसकी है, बताइए।",
    "तीन लोगों/वस्तुओं {e1}, {e2}, {e3} में से: {e1} की {noun}, {e2} से {comp1} है; {e2} की {noun}, {e3} से {comp2} है। सबसे {extreme} {noun} किसकी है?",
]

MULTIHOP_TEMPLATES = [
    "{e1} की {noun}, {e2} की {noun} से {comp1} है। {e2} की {noun}, {e3} की {noun} से {comp2} है। {e1} और {e3} में से किसकी {noun} {which} है?",
    "यह ज्ञात है कि {e1} की {noun}, {e2} की {noun} से {comp1} है, और {e2} की {noun}, {e3} की {noun} से {comp2} है। इससे बताइए, {e1} और {e3} में से किसकी {noun} {which} है?",
    "मान लीजिए {e1} की {noun}, {e2} की {noun} से {comp1} है, और {e2} की {noun}, {e3} की {noun} से {comp2} है। इन दोनों तथ्यों से, {e1} और {e3} में से किसकी {noun} {which} है?",
]

HINDI_CONFIG = LanguageConfig(
    person_names=PERSON_NAMES,
    object_names=OBJECT_NAMES,
    attributes=ATTRIBUTES,
    equal_word="बराबर",
    both_equal_phrase="दोनों बराबर हैं",
    answer_cue="उत्तर:",
    direct_templates=DIRECT_TEMPLATES,
    transitive_templates=TRANSITIVE_TEMPLATES,
    multihop_templates=MULTIHOP_TEMPLATES,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_val", type=int, default=800)
    ap.add_argument("--n_test", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "reasoning"))
    args = ap.parse_args()

    dataset = build_dataset(HINDI_CONFIG, args.n_train, args.n_val, args.n_test, seed=args.seed)
    out_dir = Path(args.out_dir)

    for split in ["train", "val", "test"]:
        write_jsonl(dataset[split], out_dir / f"{split}.jsonl")

    stats = {split: split_stats(dataset[split]) for split in ["train", "val", "test"]}
    stats["held_out_person_names_test_only"] = dataset["_meta"]["person_names_test"]
    stats["held_out_object_names_test_only"] = dataset["_meta"]["object_names_test"]
    stats["template_counts"] = {
        task: {"train": len(dataset["_meta"]["templates_train"][task]), "test": len(dataset["_meta"]["templates_test"][task])}
        for task in ["direct", "transitive", "multihop"]
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nWrote train/val/test to {out_dir}")


if __name__ == "__main__":
    main()
