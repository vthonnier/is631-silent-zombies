"""
pgd.py — the optimizer: a soft prompt in KV space.

Given the zombie's clean handoff and full cache, PGD finds a perturbation delta
(added to the targeted carrier/layers) that minimizes an objective evaluated on
the downstream agent, subject to a norm-ball constraint and an optional AcMAS
stealth penalty. It exposes an `Attacker` callback for the pipeline.

Flow per optimization:
    1. delta <- 0, requires_grad
    2. inject clean + delta into the targeted cache slots
    3. forward the downstream agent over eval_ids (with that cache)
    4. loss = objective.score(logits, eval_ids) + stealth.penalty(probe)
    5. Adam step on delta; project delta onto the norm ball
    6. repeat for `steps`
"""

from __future__ import annotations

from dataclasses import dataclass
import torch

from ..config import Config
from ..models import Backbone
from ..agent import Agent
from ..handoff import LatentHandoff, snapshot_cache, build_cache
from ..pipeline import LATENT_CUE, Attacker
from .stealth import StealthRegularizer, project_to_norm_ball


@dataclass
class LatentPGDAttack:
    """Optimizes a latent-handoff perturbation and yields an Attacker callback."""
    backbone: Backbone
    cfg: Config
    downstream_agent: Agent
    objective: object                       # Jailbreak / TargetSequence objective
    stealth: StealthRegularizer | None = None

    def __post_init__(self) -> None:
        self._fixed_delta: list[torch.Tensor] | None = None

    # ------------------------------------------------------------ delta shape
    @staticmethod
    def _new_delta(t: torch.Tensor, mode: str) -> torch.Tensor:
        """A leaf perturbation for one targeted (B, heads, T, head_dim) tensor.

        per_input : full shape — a free perturbation at every position (ceiling).
        fixed     : a position-independent direction (1, heads, 1, head_dim)
                    broadcast over batch and the T axis, so the SAME vector is
                    reusable across queries whose zombie sequence length differs
                    (the review's critical #3 — a full-shape 'fixed' delta
                    captured on one query fails c+d on the next).
        """
        if mode == "fixed":
            shape = (1, t.shape[1], 1, t.shape[3])
            return t.new_zeros(shape).requires_grad_(True)
        return torch.zeros_like(t, requires_grad=True)

    # ------------------------------------------------------------ core step
    def optimize(self, clean: LatentHandoff, pkv: object) -> LatentHandoff:
        """Run PGD for one (clean handoff, cache) pair; return the adv handoff.

        We snapshot the whole zombie cache once (detached). Each step we rebuild
        a fresh cache from that snapshot with the targeted (carrier, layers)
        slots replaced by clean+delta, so the only grad path is through delta
        and the cache never grows across steps.
        """
        ac = self.cfg.attack
        dtype = self.backbone.dtype
        idx = 0 if clean.carrier == "key" else 1            # K or V slot
        keys, values = snapshot_cache(pkv, self.backbone.num_layers)
        target = keys if idx == 0 else values               # the list we edit
        past_len = keys[0].shape[2]                         # zombie cache length

        # Downstream probe input: same cue the pipeline feeds dst in latent mode.
        base_prompt_ids = self.downstream_agent._prompt_ids(LATENT_CUE)
        base_len = base_prompt_ids.shape[1]
        eval_ids = self.objective.prepare(self.backbone, base_prompt_ids)

        # Attention mask over prepended cache + eval tokens, so positions are
        # correct when the poisoned cache is prepended (matches agent._probe).
        attn = torch.ones((1, past_len + eval_ids.shape[1]), dtype=torch.long,
                          device=eval_ids.device)

        # One leaf delta per targeted layer tensor (shape depends on mode).
        delta = [self._new_delta(t, ac.mode) for t in clean.tensors]
        opt = torch.optim.Adam(delta, lr=ac.lr)

        for _ in range(ac.steps):
            opt.zero_grad()
            # Patch targeted layers with clean+delta (delta broadcasts in fixed
            # mode); other layers stay detached.
            for layer, c, d in zip(clean.layers, clean.tensors, delta):
                target[layer] = (c + d).to(dtype)
            cache = build_cache(keys, values)

            out = self.backbone.model(
                eval_ids, past_key_values=cache, attention_mask=attn,
                use_cache=True, output_hidden_states=self.stealth is not None,
            )
            loss, _ = self.objective.score(out.logits, eval_ids)
            if self.stealth is not None:
                # Read the SAME activation the detector measures: top layer, the
                # last *prompt* position (base_len - 1), under the poisoned
                # handoff. eval_ids may append target tokens after the prompt
                # (TargetSequenceObjective), so -1 would be the wrong position.
                probe = out.hidden_states[-1][:, base_len - 1, :]
                loss = loss + self.stealth.penalty(probe)

            loss.backward()
            opt.step()

            if ac.norm_ratio is not None:
                with torch.no_grad():
                    project_to_norm_ball(delta, clean, ac.norm_ratio)

        adv_tensors = [(c + d).detach().to(dtype) for c, d in zip(clean.tensors, delta)]
        if ac.mode == "fixed":
            # Store the (possibly reduced) direction; as_attacker() broadcasts it
            # onto future clean handoffs of any sequence length.
            self._fixed_delta = [d.detach() for d in delta]
        return LatentHandoff(clean.carrier, clean.layers, adv_tensors)

    # ------------------------------------------------------- attacker handle
    def as_attacker(self) -> Attacker:
        """Return the (clean, pkv) -> adv callback the pipeline calls.

        per_input : optimize afresh each call (ceiling).
        fixed     : optimize once, then reuse the stored delta (practicality).
        """
        def attacker(clean: LatentHandoff, pkv: object) -> LatentHandoff:
            if self.cfg.attack.mode == "fixed" and self._fixed_delta is not None:
                adv = [(c + d).detach().to(self.backbone.dtype)
                       for c, d in zip(clean.tensors, self._fixed_delta)]
                return LatentHandoff(clean.carrier, clean.layers, adv)
            return self.optimize(clean, pkv)
        return attacker
