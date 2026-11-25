"""
e1_jailbreak.py — Objective (item 1): jailbreak / exfil via the handoff.

Runs the security objective through the latent handoff and reports attack
success plus what the detectors saw. Set attack.objective to "jailbreak" or
"exfil" in the config.

    python experiments/e1_jailbreak.py --config configs/default.yaml --limit 20
"""

from _common import parse_args, load_config, dump
from lmas.experiment import Harness
from lmas.metrics import jailbreak_success, exfil_success, rate, detector_catch_rate
from lmas.data.prompts import FORBIDDEN_STUBS, HarmBenchLoader


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args)
    harness = Harness.build(cfg, channel="latent")

    # Use the user's real benchmark if the config points at one; else safe stubs.
    if cfg.judge.harmbench_path:
        requests = HarmBenchLoader(cfg.judge.harmbench_path).load(limit=args.limit)
    else:
        requests = (FORBIDDEN_STUBS * (args.limit // len(FORBIDDEN_STUBS) + 1))[: args.limit]

    successes, acmas_reports, text_reports = [], [], []
    for req in requests:
        trace, report = harness.run_attacked(req)
        if cfg.attack.objective == "exfil":
            ok = exfil_success(trace, cfg.attack.target_secret)
        else:
            ok = jailbreak_success(trace, req, harness.judge)
        successes.append(ok)
        acmas_reports.append(report["acmas"])
        text_reports.append(report["text"])

    zombie_role = cfg.pipeline.source_role
    dst_role = cfg.pipeline.target_role
    result = {
        "objective": cfg.attack.objective,
        "attack_success_rate": rate(successes),
        "acmas_catch_zombie": detector_catch_rate(acmas_reports, zombie_role),
        "acmas_catch_downstream": detector_catch_rate(acmas_reports, dst_role),
        "text_catch_zombie": detector_catch_rate(text_reports, zombie_role),
        "n": len(requests),
    }
    path = dump(cfg, "e1_jailbreak", result)
    print(result, "->", path)


if __name__ == "__main__":
    main()
