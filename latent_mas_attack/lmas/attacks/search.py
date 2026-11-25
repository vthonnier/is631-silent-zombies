"""
search.py — pick the attack site (carrier, layer, strength) cheaply.

Prior work grid-searches layer x strength blindly and reports the best cell.
We frame site selection as black-box optimization: a function evaluates one
(carrier, layer, norm_ratio) configuration end to end (attack + score) and
Bayesian optimization proposes the next configuration from past results.

Falls back to a grid if scikit-optimize is unavailable, so the module always
runs. The objective passed in returns a scalar to MAXIMIZE (e.g. ASR).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import itertools

try:
    from skopt import gp_minimize
    from skopt.space import Categorical, Integer, Real
    _HAVE_SKOPT = True
except Exception:                       # pragma: no cover - optional dep
    _HAVE_SKOPT = False


@dataclass
class SiteSpace:
    """The discrete/continuous choices for where and how hard to hit."""
    carriers: list[str]
    layers: list[int]
    norm_ratios: list[float]


# A site is (carrier, layer, norm_ratio); the evaluator returns a score to max.
Site = tuple[str, int, float]
Evaluator = Callable[[Site], float]


def search_site(space: SiteSpace, evaluate: Evaluator, n_calls: int = 25) -> tuple[Site, float]:
    """Return (best_site, best_score). Uses BO if available, else full grid."""
    if _HAVE_SKOPT and n_calls < len(space.carriers) * len(space.layers) * len(space.norm_ratios):
        dims = [
            Categorical(space.carriers, name="carrier"),
            Integer(min(space.layers), max(space.layers), name="layer"),
            Real(min(space.norm_ratios), max(space.norm_ratios), name="norm_ratio"),
        ]
        res = gp_minimize(
            lambda x: -evaluate((x[0], int(x[1]), float(x[2]))),
            dims, n_calls=n_calls, random_state=0,
        )
        best = (res.x[0], int(res.x[1]), float(res.x[2]))
        return best, -res.fun

    # Grid fallback: exhaustive over the listed values.
    best_site, best_score = None, float("-inf")
    for site in itertools.product(space.carriers, space.layers, space.norm_ratios):
        score = evaluate(site)
        if score > best_score:
            best_site, best_score = site, score
    return best_site, best_score
