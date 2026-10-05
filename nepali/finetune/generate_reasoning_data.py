"""Generate Nepali's synthetic comparative-reasoning finetune dataset.

Same grammar strategy as the Hindi generator (see hindi/finetune/
generate_reasoning_data.py for the full rationale): every comparison is
phrased as "X-ko {attribute} bhandaa ... cha" (X's {attribute} is ... than
Y's), so the comparative word never needs to agree with an entity's gender
-- it agrees with the attribute noun instead, which is fixed per attribute.
Nepali's "-ko" postposition attaches directly to the entity name (no space,
unlike Hindi's separate "ki"), so templates below write "{e1}को" not
"{e1} को".

This is a fully independent dataset from Hindi's -- separate word lists,
separate templates, separate generated examples -- even though it's built
by calling the same shared generator code in common/finetune/.

Run directly to write nepali/data/reasoning/{train,val,test}.jsonl and
print a summary.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.finetune.reasoning_gen import Attribute, LanguageConfig, build_dataset, split_stats, write_jsonl

PERSON_NAMES = [
    "राम", "श्याम", "गीता", "सीता", "हरि", "सुनिता", "विजय", "कमला",
    "सूरज", "माया", "विकास", "पार्वती", "दिनेश", "सरिता", "प्रकाश", "इन्दिरा",
    "गोपाल", "कृष्णा", "मनोज", "सुनिल", "अनिता", "राजु", "शोभा", "विमल",
    "बिनोद", "सरस्वती", "निर्मल", "कान्छी", "धन", "फूलमाया",
]

OBJECT_NAMES = [
    "किताब", "कलम", "झोला", "घडी", "साइकल", "मोबाइल", "कुर्सी", "टेबल",
    "डालो", "जुत्ता", "छाता", "थैलो", "चश्मा", "पङ्खा", "लामटिन",
]

ATTRIBUTES = {
    "age": Attribute(
        noun="उमेर", unit="वर्ष", comp_high="बढी", comp_low="कम",
        value_range=(8, 80), entity_pool="person",
    ),
    "height": Attribute(
        noun="उचाइ", unit="सेन्टिमिटर", comp_high="बढी", comp_low="कम",
        value_range=(100, 195), entity_pool="person",
    ),
    "price": Attribute(
        noun="मूल्य", unit="रुपैयाँ", comp_high="बढी", comp_low="कम",
        value_range=(10, 2000), entity_pool="object",
    ),
    "quantity": Attribute(
        noun="मात्रा", unit="", comp_high="बढी", comp_low="कम",
        value_range=(1, 100), entity_pool="object",
    ),
}

DIRECT_TEMPLATES = [
    "{e1}को {noun} {v1} {unit} छ। {e2}को {noun} {v2} {unit} छ। कसको {noun} {comp} छ?",
    "{e1}को {noun} {v1} {unit} छ, जबकि {e2}को {noun} {v2} {unit} छ। कसको {noun} {comp} छ, भन्नुहोस्।",
    "मानौं {e1}को {noun} {v1} {unit} छ र {e2}को {noun} {v2} {unit} छ। यी दुई मध्ये कसको {noun} {comp} छ?",
    "यदि {e1}को {noun} {v1} {unit} छ र {e2}को {noun} {v2} {unit} छ भने, कसको {noun} {comp} होला?",
]

TRANSITIVE_TEMPLATES = [
    "{e1}को {noun}, {e2}को {noun} भन्दा {comp1} छ। {e2}को {noun}, {e3}को {noun} भन्दा {comp2} छ। यी तीन मध्ये सबैभन्दा {extreme} {noun} कसको हो?",
    "थाहा छ कि {e1}को {noun}, {e2}को {noun} भन्दा {comp1} छ, र {e2}को {noun}, {e3}को {noun} भन्दा {comp2} छ। यीमध्ये सबैभन्दा {extreme} {noun} कसको हो, भन्नुहोस्।",
    "{e1}, {e2}, {e3} मध्ये: {e1}को {noun}, {e2}को भन्दा {comp1} छ; {e2}को {noun}, {e3}को भन्दा {comp2} छ। सबैभन्दा {extreme} {noun} कसको हो?",
]

MULTIHOP_TEMPLATES = [
    "{e1}को {noun}, {e2}को {noun} भन्दा {comp1} छ। {e2}को {noun}, {e3}को {noun} भन्दा {comp2} छ। {e1} र {e3} मध्ये कसको {noun} {which} छ?",
    "यो थाहा छ कि {e1}को {noun}, {e2}को {noun} भन्दा {comp1} छ, र {e2}को {noun}, {e3}को {noun} भन्दा {comp2} छ। यसबाट भन्नुहोस्, {e1} र {e3} मध्ये कसको {noun} {which} छ?",
    "मानौं {e1}को {noun}, {e2}को {noun} भन्दा {comp1} छ, र {e2}को {noun}, {e3}को {noun} भन्दा {comp2} छ। यी दुई तथ्यबाट, {e1} र {e3} मध्ये कसको {noun} {which} छ?",
]

NEPALI_CONFIG = LanguageConfig(
    person_names=PERSON_NAMES,
    object_names=OBJECT_NAMES,
    attributes=ATTRIBUTES,
    equal_word="बराबर",
    both_equal_phrase="दुवै बराबर छन्",
    answer_cue="जवाफ:",
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

    dataset = build_dataset(NEPALI_CONFIG, args.n_train, args.n_val, args.n_test, seed=args.seed)
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
