"""
grpo.py — multi-hop reach via a reward signal (§4.4).

PGD (pgd.py) optimizes a differentiable surrogate on the *immediate* downstream
agent: one hop, one backprop. But a perturbation at edge (c -> c+1) reaches the
final node through several agents that each *regenerate text* — a discrete,
non-differentiable chain. To optimize for success at the FINAL node we therefore
use a black-box reinforcement objective instead of a gradient path.

We use a GRPO-style update: at each step sample a GROUP of candidate
perturbations, run each through the *entire* pipeline, score the final answer
with the judge, and move the perturbation mean along the group-relative
advantage (reward standardized within the group). This is the continuous-action
/ evolution-strategies form of group-relative policy optimization — no value
network, no differentiability assumption on the chain.

The perturbation is a position-independent direction (one (1, heads, 1, head_dim)
vector per targeted layer, broadcast over the sequence), so the learned vector is
reusable across queries of different length — the same parameterization as PGD's
"fixed" mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import torch

from ..config import Config
from ..models import Backbone
from ..handoff import LatentHandoff
from ..pipeline import MultiAgentPipeline, PipelineTrace, Attacker
from .stealth import project_to_norm_ball

# A reward maps a finished pipeline trace to a scalar to MAXIMIZE (e.g. the
# judge's harmfulness score of the final answer).
Reward = Callable[[PipelineTrace], float]


@dataclass
class GRPOReachAttack:
    """Optimize a reusable handoff direction against the final-node reward."""
    backbone: Backbone
    cfg: Config
    pipeline: MultiAgentPipeline
    reward_fn: Reward
    _mu: list[torch.Tensor] | None = field(default=None, repr=False)
    _clean0: LatentHandoff | None = field(default=None, repr=False)

    # --------------------------------------------------------- candidate hook
    def _inject_attacker(self, direction: list[torch.Tensor]) -> Attacker:
        """An Attacker that adds a fixed `direction` (broadcast) to the clean
        handoff — no optimization, just injection of a given candidate."""
        dtype = self.backbone.dtype

        def attacker(clean: LatentHandoff, pkv: object) -> LatentHandoff:
            adv = [(c + d).detach().to(dtype)
                   for c, d in zip(clean.tensors, direction)]
            return LatentHandoff(clean.carrier, clean.layers, adv)

        return attacker

    def _capture_clean(self, question: str) -> LatentHandoff:
        """Run once with an identity attacker to read the clean handoff shapes."""
        grabbed: dict[str, LatentHandoff] = {}

        def grab(clean: LatentHandoff, pkv: object) -> LatentHandoff:
            grabbed["h"] = clean.clone()
            return clean                     # identity: inject clean back (no-op)

        self.pipeline.run(question, attacker=grab)
        return grabbed["h"]

    @staticmethod
    def _zeros_like_direction(clean: LatentHandoff) -> list[torch.Tensor]:
        """One position-independent (1, heads, 1, head_dim) zero vector per layer."""
        return [t.new_zeros((1, t.shape[1], 1, t.shape[3])) for t in clean.tensors]

    # ---------------------------------------------------------------- fit
    @torch.no_grad()
    def fit(self, question: str) -> list[torch.Tensor]:
        """Optimize the perturbation mean for `question`; return and store it."""
        ac = self.cfg.attack
        clean = self._capture_clean(question)
        self._clean0 = clean
        mu = self._zeros_like_direction(clean)

        for _ in range(ac.grpo_steps):
            eps_group: list[list[torch.Tensor]] = []
            rewards: list[float] = []
            for _g in range(ac.grpo_group_size):
                eps = [torch.randn_like(m) for m in mu]
                cand = [m + ac.grpo_sigma * e for m, e in zip(mu, eps)]
                trace = self.pipeline.run(question, attacker=self._inject_attacker(cand))
                rewards.append(float(self.reward_fn(trace)))
                eps_group.append(eps)

            r = torch.tensor(rewards, dtype=torch.float32)
            adv = (r - r.mean()) / (r.std() + 1e-6)          # group-relative advantage
            adv_f = [float(a) for a in adv]                   # plain floats (device-safe)

            # ES / score-function estimate of grad_mu E[reward], then ascend.
            g = ac.grpo_group_size
            for li in range(len(mu)):
                grad = sum(adv_f[j] * eps_group[j][li] for j in range(g)) / g
                mu[li] = mu[li] + ac.grpo_lr * ac.grpo_sigma * grad

            if ac.norm_ratio is not None:
                project_to_norm_ball(mu, clean, ac.norm_ratio)

        self._mu = [m.detach() for m in mu]
        return self._mu

    # ------------------------------------------------------- attacker handle
    def as_attacker(self) -> Attacker:
        """Inject the fitted direction. Call fit() first (else injects zeros)."""
        direction = self._mu if self._mu is not None else None

        def attacker(clean: LatentHandoff, pkv: object) -> LatentHandoff:
            d = direction if direction is not None else self._zeros_like_direction(clean)
            return self._inject_attacker(d)(clean, pkv)

        return attacker
