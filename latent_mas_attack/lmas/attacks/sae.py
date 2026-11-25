"""
sae.py — SAE feature steering (§4.6), the interpretable alternative to raw-vector
PGD.

Instead of optimizing an unconstrained perturbation, steer the downstream agent
along the decoder direction(s) of chosen sparse-autoencoder (SAE) features — a
refusal feature, a harmful-compliance feature, etc. Fewer dimensions, a
mechanistic account, and a plausibly more transferable/stealthier perturbation.
This is the carrier="hidden" attack: the produced steering vectors are added to
the residual stream of the target layer during the downstream agent's forward
(pipeline installs the hooks; see handoff.make_hidden_steering_hooks).

Dependency handling: a real SAE is loaded via sae-lens from a release the user
provides. If sae-lens (or the released SAE) is unavailable we fall back to a
deterministic pseudo-feature direction so the mechanism stays exercisable — this
is clearly flagged and must NOT be used for reported numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import torch

from ..config import Config
from ..models import Backbone
from ..handoff import LatentHandoff
from ..pipeline import Attacker


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-8)


@dataclass
class SAESteerer:
    """Build residual-steering vectors from SAE feature directions.

    as_attacker() returns an Attacker for the hidden carrier: it ignores the
    clean hidden handoff's values (using only its layer list) and returns a
    handoff whose tensors are the additive steering vectors, one per targeted
    layer (all the same SAE direction; SAE steering is normally single-layer).
    """
    backbone: Backbone
    cfg: Config
    _directions: list[torch.Tensor] | None = field(default=None, repr=False)
    _is_stub: bool = False

    # ------------------------------------------------------------ feature dir
    def _feature_direction(self) -> torch.Tensor:
        """A (d_model,) steering direction = strength * sum of chosen feature
        decoder directions (each unit-normalized). Real SAE if available."""
        sae_cfg = self.cfg.sae
        d_model = self.backbone.hidden_size
        device, dtype = self.backbone.device, torch.float32

        feats = sae_cfg.feature_ids or [0]
        try:
            from sae_lens import SAE  # optional dependency
            sae = SAE.from_pretrained(sae_cfg.release, sae_cfg.sae_id, device=str(device))
            if isinstance(sae, tuple):           # some versions return (sae, cfg, ...)
                sae = sae[0]
            w_dec = sae.W_dec.to(device=device, dtype=dtype)   # (d_sae, d_model)
            direction = torch.zeros(d_model, device=device, dtype=dtype)
            for f in feats:
                direction = direction + _unit(w_dec[f])
            self._is_stub = False
        except Exception as exc:  # pragma: no cover - env-dependent
            print(f"[sae] sae-lens/SAE unavailable ({exc!r}); using STUB "
                  "pseudo-feature directions (not for reported numbers).")
            direction = torch.zeros(d_model, device=device, dtype=dtype)
            for f in feats:
                g = torch.Generator(device="cpu").manual_seed(int(f) + 1)
                direction = direction + _unit(torch.randn(d_model, generator=g).to(device))
            self._is_stub = True

        return sae_cfg.strength * _unit(direction)

    def build(self) -> "SAESteerer":
        base = self._feature_direction()
        n_layers = len(self.cfg.pipeline.layers)
        self._directions = [base.clone() for _ in range(n_layers)]
        return self

    # ------------------------------------------------------- attacker handle
    def as_attacker(self) -> Attacker:
        if self._directions is None:
            self.build()
        dirs = self._directions

        def attacker(clean: LatentHandoff, pkv: object) -> LatentHandoff:
            # carrier="hidden": tensors are the additive steering vectors.
            return LatentHandoff("hidden", list(clean.layers),
                                 [d.clone() for d in dirs])

        return attacker
