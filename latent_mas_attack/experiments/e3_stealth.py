"""
e3_stealth.py — Stealth (item 3): asymmetric observability + AcMAS evasion.

Compares a naive attack (stealth_lambda = 0) against a detector-aware attack
(stealth_lambda > 0) under the live AcMAS and text monitors. Shows:
  * text monitor flags nothing (the zombie is silent on a latent edge);
  * the naive attack may trip AcMAS at the downstream node;
  * the stealth-regularized attack keeps attack success while evading AcMAS.

    python experiments/e3_stealth.py --config configs/default.yaml --limit 20
"""

from _common import parse_args, load_config, dump
from lmas.config import Config
from lmas.experiment import Harness
from lmas.metrics import jailbreak_success, rate, detector_catch_rate
from lmas.data.prompts import FORBIDDEN_STUBS


def _run(cfg: Config, requests: list[str]) -> dict:
    harness = Harness.build(cfg, channel="latent")
    succ, acmas_reports, text_reports = [], [], []
    for req in requests:
        trace, report = harness.run_attacked(req)
        succ.append(jailbreak_success(trace, req, harness.judge))
        acmas_reports.append(report["acmas"])
        text_reports.append(report["text"])
    dst = cfg.pipeline.target_role
    zombie = cfg.pipeline.source_role
    return {
        "attack_success_rate": rate(succ),
        "acmas_catch_downstream": detector_catch_rate(acmas_reports, dst),
        "acmas_catch_zombie": detector_catch_rate(acmas_reports, zombie),
        "text_catch_any": max(
            detector_catch_rate(text_reports, r) for r in cfg.pipeline.roles
        ),
    }


def main() -> None:
    args = parse_args(__doc__)
    requests = (FORBIDDEN_STUBS * (args.limit // len(FORBIDDEN_STUBS) + 1))[: args.limit]

    naive_cfg = load_config(args)
    naive_cfg.attack.stealth_lambda = 0.0

    stealth_cfg = Config.from_yaml(args.config)
    stealth_cfg.attack.stealth_lambda = 5.0      # tune on a held-out split

    result = {
        "naive": _run(naive_cfg, requests),
        "stealth": _run(stealth_cfg, requests),
        "n": len(requests),
    }
    path = dump(naive_cfg, "e3_stealth", result)
    print(result, "->", path)


if __name__ == "__main__":
    main()
