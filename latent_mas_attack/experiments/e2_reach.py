"""
e2_reach.py — Reach (item 2): blast radius of one compromised node.

Sweeps which node is the zombie and measures attack success as a function of
hop distance from the zombie to the final (observation) node. Produces the
reach curve: success vs. distance.

The optimizer is chosen in the config (attack.optimizer):
  * "pgd"  — a one-hop differentiable surrogate on the immediate downstream
    agent, with success still *measured* at the final node (a valid reach
    measurement, but the attacker never optimizes past one hop);
  * "grpo" — a black-box reward attack whose signal is the FINAL-node judge,
    propagated through every intermediate (text-regenerating) hop. This is the
    genuine multi-hop reach optimizer (§4.4); it is far more expensive
    (group_size x grpo_steps full-pipeline runs per query), so use a small
    --limit.

    python experiments/e2_reach.py --config configs/default.yaml --limit 20
"""

from _common import parse_args, load_config, dump
from lmas.config import Config
from lmas.experiment import Harness
from lmas.metrics import jailbreak_success, exfil_success, rate, reach_curve, ReachPoint
from lmas.data.prompts import FORBIDDEN_STUBS


def main() -> None:
    args = parse_args(__doc__)
    base = load_config(args)
    n_roles = len(base.pipeline.roles)
    requests = (FORBIDDEN_STUBS * (args.limit // len(FORBIDDEN_STUBS) + 1))[: args.limit]

    points: list[ReachPoint] = []
    detail = {}
    # Zombie at src = 0..n-2; it always poisons the edge src -> src+1.
    for src in range(n_roles - 1):
        cfg = Config.from_yaml(args.config)
        cfg.pipeline.handoff_edge = f"{src}->{src + 1}"
        cfg.pipeline.__post_init__()      # re-parse + re-validate the new edge
        harness = Harness.build(cfg, channel="latent")

        succ = []
        for req in requests:
            trace, _ = harness.run_attacked(req)
            ok = (exfil_success(trace, cfg.attack.target_secret)
                  if cfg.attack.objective == "exfil"
                  else jailbreak_success(trace, req, harness.judge))
            succ.append(ok)

        distance = (n_roles - 1) - src           # hops from zombie to final node
        sr = rate(succ)
        points.append(ReachPoint(zombie_index=src, distance=distance, success_rate=sr))
        detail[f"zombie@{src}"] = {"distance": distance, "success_rate": sr}

    result = {"per_zombie": detail, "reach_curve": reach_curve(points)}
    path = dump(base, "e2_reach", result)
    print(result, "->", path)


if __name__ == "__main__":
    main()
