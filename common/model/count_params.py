"""Report the exact trainable parameter count for a model config, and
(optionally) record it into report/phase2/param_counts.json keyed by the
config's parent language directory name.

    python -m common.model.count_params --config hindi/configs/model_config.yaml
    python -m common.model.count_params --config nepali/configs/model_config.yaml
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.model.config import GPTConfig
from common.model.transformer import GPT

REPORT_PATH = REPO_ROOT / "report" / "phase2" / "param_counts.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a model_config.yaml")
    parser.add_argument("--no-report", action="store_true", help="Print only, don't write to report/phase2/param_counts.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = GPTConfig.from_yaml(config_path)
    model = GPT(cfg)

    total = model.num_parameters()
    non_embedding = model.num_parameters(non_embedding=True)

    entry = {
        "config_path": str(config_path),
        "config": cfg.to_dict(),
        "total_parameters": total,
        "non_embedding_parameters": non_embedding,
    }
    print(json.dumps(entry, indent=2))

    if not args.no_report:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        report = json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else {}
        lang = config_path.resolve().parents[1].name  # e.g. "hindi" from hindi/configs/model_config.yaml
        report[lang] = entry
        REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nRecorded under '{lang}' in {REPORT_PATH}")


if __name__ == "__main__":
    main()
