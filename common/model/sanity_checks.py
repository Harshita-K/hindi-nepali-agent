"""Empirical sanity checks for the from-scratch GPT implementation.

Run standalone to verify, for a given model config, that the causal mask
actually blocks future information (spec requirement 2.1.2: "verify
empirically that the model cannot see the future -- show that changing
token t+1 does not change the logits at position t").

    python -m common.model.sanity_checks --config hindi/configs/model_config.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.model.config import GPTConfig
from common.model.transformer import GPT


@torch.no_grad()
def verify_causal_masking(model: GPT, vocab_size: int, seq_len: int = 16, trials: int = 5) -> dict:
    """For several random sequences, change only the last token and confirm
    logits at every earlier position are unchanged (within float tolerance).
    """
    model.eval()
    results = []
    for _ in range(trials):
        idx = torch.randint(0, vocab_size, (1, seq_len))
        logits1, _, _ = model(idx)

        idx2 = idx.clone()
        idx2[0, -1] = (idx2[0, -1] + 1) % vocab_size  # perturb only the last token
        logits2, _, _ = model(idx2)

        max_diff = (logits1[:, :-1] - logits2[:, :-1]).abs().max().item()
        results.append(max_diff)

    passed = all(d < 1e-4 for d in results)
    return {"passed": passed, "max_logit_diff_per_trial": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a model_config.yaml")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    cfg = GPTConfig.from_yaml(args.config)
    model = GPT(cfg)

    result = verify_causal_masking(model, cfg.vocab_size, args.seq_len, args.trials)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("FAILED: causal mask leaked future information")
    print("PASSED: causal mask verified -- future tokens do not affect earlier logits.")


if __name__ == "__main__":
    main()
