"""Builds a compact prompt/pretrained/finetuned comparison table for the
report, directly from the qualitative examples already captured by
run_reasoning.py (report/phase3/{lang}_reasoning_eval.json ->
"examples_finetuning_fixed" / "examples_still_wrong_after_finetuning").

Needs no checkpoints or Colab -- the real pretrained/finetuned generations
were already saved by the eval run; this just selects a representative,
task-type-diverse sample and formats it as JSON + a ready-to-paste LaTeX
table.

    python -m common.finetune.build_example_table
    python -m common.finetune.build_example_table --n_fixed 2 --n_still_wrong 1
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

LANGS = {
    "hindi": "Model H (Hindi)",
    "nepali": "Model L (Nepali)",
}


def select_examples(data: dict, n_fixed: int, n_still_wrong: int) -> list:
    """Picks up to n_fixed "fixed" examples (pretrained wrong, finetuned
    right -- the clearest evidence finetuning worked) spread across
    different task types where possible, plus up to n_still_wrong "still
    wrong" examples (honest failure cases)."""
    rows = []

    def pick_diverse(pool, n):
        chosen, seen_types = [], set()
        for ex in pool:
            if len(chosen) >= n:
                break
            if ex["task_type"] not in seen_types or len(pool) <= n:
                chosen.append(ex)
                seen_types.add(ex["task_type"])
        # fill remaining slots from whatever's left if task-type diversity
        # didn't use up the full quota
        for ex in pool:
            if len(chosen) >= n:
                break
            if ex not in chosen:
                chosen.append(ex)
        return chosen[:n]

    for ex in pick_diverse(data["examples_finetuning_fixed"], n_fixed):
        rows.append({**ex, "outcome": "fixed_by_finetuning"})
    for ex in pick_diverse(data["examples_still_wrong_after_finetuning"], n_still_wrong):
        rows.append({**ex, "outcome": "still_wrong_after_finetuning"})
    return rows


def escape_latex(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def build_latex_table(all_rows: list) -> str:
    lines = [
        r"\begin{table}[H]\centering",
        r"\caption{Example prompts: pretrained vs.\ finetuned predictions (real model outputs)}",
        r"\small",
        r"\begin{tabular}{p{1.3cm}p{1cm}p{5.2cm}p{1.6cm}p{2.6cm}p{1.6cm}}\toprule",
        r"Lang.\ & Task & Prompt & Answer & Pretrained pred. & Finetuned pred.\\\midrule",
    ]
    for r in all_rows:
        pred_pre = escape_latex(r["pretrained_predicted"]) or r"\textit{(empty)}"
        pred_fin = escape_latex(r["finetuned_predicted"]) or r"\textit{(empty)}"
        fin_fmt = f"\\textbf{{{pred_fin}}}" if r["outcome"] == "fixed_by_finetuning" else pred_fin
        lines.append(
            f"{r['lang_label']} & {r['task_type'].capitalize()} & {escape_latex(r['prompt'])} & "
            f"{escape_latex(r['answer'])} & {pred_pre} & {fin_fmt}\\\\"
        )
    lines += [r"\bottomrule\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_fixed", type=int, default=2, help="Examples per language where finetuning fixed a wrong pretrained answer")
    parser.add_argument("--n_still_wrong", type=int, default=1, help="Examples per language still wrong after finetuning")
    parser.add_argument("--out_json", default=str(REPO_ROOT / "report" / "phase3" / "example_table.json"))
    parser.add_argument("--out_tex", default=str(REPO_ROOT / "report" / "phase3" / "example_table.tex"))
    args = parser.parse_args()

    all_rows = []
    for lang, label in LANGS.items():
        eval_path = REPO_ROOT / "report" / "phase3" / f"{lang}_reasoning_eval.json"
        data = json.loads(eval_path.read_text(encoding="utf-8"))
        rows = select_examples(data, args.n_fixed, args.n_still_wrong)
        for r in rows:
            r["language"] = lang
            r["lang_label"] = label
        all_rows.extend(rows)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(all_rows)} example rows to {args.out_json}")

    latex = build_latex_table(all_rows)
    Path(args.out_tex).write_text(latex, encoding="utf-8")
    print(f"Saved LaTeX table to {args.out_tex}")
    print("\nNote: contains native Devanagari text -- requires XeLaTeX/LuaLaTeX + "
          "fontspec (see earlier guidance), not plain pdflatex.")
    print("\n" + latex)


if __name__ == "__main__":
    main()
