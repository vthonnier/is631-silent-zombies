"""
stealth.py — keep the attack below the defender's radar.

Two mechanisms, both from the paper's threat model:

* norm-matching : scale the perturbation so the poisoned handoff has (at most)
  the clean handoff's L2 norm. This defeats a magnitude detector on the edge
  (norm-matched attacks evaded one in *When Latent Agents Lie*).
* AcMAS penalty : a differentiable term that keeps the DOWNSTREAM agent's probe
  hidden near the benign centroid, so an activation detector on that node does
  not fire even though the handoff is poisoned. This is the adaptive attacker
  AcMAS lists as an open problem.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F

from ..handoff import LatentHandoff


def project_to_norm_ball(
    delta_tensors: list[torch.Tensor],
    clean: LatentHandoff,
    ratio: float,
) -> None:
    """Scale delta in place so ||clean + delta|| <= ratio * ||clean||.

    Operates on the whole handoff jointly (one global scale) so the geometry of
    the perturbation is preserved; only its magnitude is clipped.

    Two delta shapes are supported:
      * ``delta`` matches ``clean`` elementwise (per_input mode): we reproject
        exactly onto the ball via (clean+delta) *= scale.
      * ``delta`` is a position-independent direction that broadcasts over
        ``clean`` (fixed mode: shape (1, heads, 1, head_dim) added to every
        position). The exact reprojection identity does not hold under
        broadcasting, so we shrink the direction's magnitude by ``scale``,
        which reduces ||clean+delta|| monotonically toward the ball.
    """
    clean_norm = clean.norm()
    # norm of the resulting handoff clean + delta (broadcasts if delta is reduced)
    sq = sum(((c + d).float() ** 2).sum()
             for c, d in zip(clean.tensors, delta_tensors))
    result_norm = torch.sqrt(sq)
    limit = ratio * clean_norm
    if result_norm > limit:
        scale = (limit / (result_norm + 1e-8)).item()
        for c, d in zip(clean.tensors, delta_tensors):
            if d.shape == c.shape:
                # exact: (clean + delta) *= scale  =>  delta <- scale*(c+d) - c
                d.mul_(scale).add_(c, alpha=(scale - 1.0))
            else:
                # broadcast direction: magnitude-only shrink
                d.mul_(scale)


@dataclass
class StealthRegularizer:
    """AcMAS-distance penalty on the downstream agent's probe hidden."""
    centroid: torch.Tensor          # (hidden,) benign mean, unit-normalized
    lam: float                      # penalty weight (0 disables)

    def penalty(self, probe_hidden: torch.Tensor) -> torch.Tensor:
        """lam * (1 - cos(probe, centroid)); differentiable in probe."""
        if self.lam == 0.0:
            return probe_hidden.new_zeros(())
        cos = F.cosine_similarity(probe_hidden.float().reshape(1, -1),
                                  self.centroid.reshape(1, -1), dim=1)
        return self.lam * (1.0 - cos.squeeze())


def fit_centroid(probe_hiddens: list[torch.Tensor]) -> torch.Tensor:
    """Benign centroid from a list of (1, hidden) probe states -> (hidden,)."""
    stacked = torch.cat([h.float().reshape(1, -1) for h in probe_hiddens], dim=0)
    mu = stacked.mean(dim=0)
    return mu / (mu.norm() + 1e-8)
