"""
e0_reproduce.py — sanity check: clean pipeline + accuracy attack.

Reproduces the target paper's setting: run benign tasks clean (baseline
utility), then run the accuracy objective and confirm the latent handoff drops
downstream accuracy. No security objective here — this validates the testbed.

    python experiments/e0_reproduce.py --config configs/default.yaml --limit 10
"""

from _common import parse_args, load_config, dump
from lmas.experiment import Harness
from lmas.metrics import accuracy, rate
from lmas.data.prompts import BENIGN_TASKS


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args)
    cfg.attack.objective = "accuracy"

    harness = Harness.build(cfg, channel="latent")
    tasks = (BENIGN_TASKS * (args.limit // len(BENIGN_TASKS) + 1))[: args.limit]

    clean_hits, attacked_hits = [], []
    for q in tasks:
        gold = ""  # supply gold answers for real accuracy; '' => treats as miss
        clean = harness.pipeline.run(q, attacker=None)
        attacked, _ = harness.run_attacked(q)
        clean_hits.append(accuracy(clean, gold))
        attacked_hits.append(accuracy(attacked, gold))

    result = {
        "clean_accuracy": rate(clean_hits),
        "attacked_accuracy": rate(attacked_hits),
        "n": len(tasks),
    }
    path = dump(cfg, "e0_reproduce", result)
    print(result, "->", path)


if __name__ == "__main__":
    main()
