"""
objectives.py — differentiable surrogates the attacker minimizes.

Each objective is evaluated on the *immediate downstream agent* (dst) while it
is conditioned on the poisoned handoff. The interface is two methods:

    prepare(backbone, base_prompt_ids) -> eval_ids   # ids to run the forward on
    score(logits)                      -> (loss, info)

The PGD engine (pgd.py) runs one forward pass over `eval_ids` with the poisoned
cache, then calls `score`. Keeping the forward in one place lets us also read
the AcMAS probe hidden from the same pass for the stealth penalty.

Loss convention: everything is a scalar to MINIMIZE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import torch
import torch.nn.functional as F

from ..models import Backbone


# Openers that signal a refusal; we suppress the probability mass on their
# first token. Kept short and generic (not model-specific).
REFUSAL_PREFIXES = [
    "I", "I'm", "Sorry", "I cannot", "I can't", "As", "Unfortunately",
    "I will not", "I won't", "No", "It's not",
]
# Generic affirmative opener we push the first token toward (not harmful itself).
AFFIRMATIVE_TARGET = "Sure, here"


def _first_token_ids(backbone: Backbone, phrases: list[str]) -> list[int]:
    """First token id of each phrase, de-duplicated."""
    ids = set()
    for p in phrases:
        enc = backbone.tokenizer(p, add_special_tokens=False).input_ids
        if enc:
            ids.add(enc[0])
    return sorted(ids)


def _teacher_forced_logprob(
    logits: torch.Tensor, eval_ids: torch.Tensor, target_len: int
) -> torch.Tensor:
    """Summed log p(target | prefix) for the last `target_len` tokens of eval_ids.

    logits: (1, L, V) from a forward over eval_ids. The logit at position t
    predicts token t+1, so target token at absolute position p is scored by
    logits[:, p-1].
    """
    logp = F.log_softmax(logits[0].float(), dim=-1)      # (L, V)
    L = eval_ids.shape[1]
    total = logits.new_zeros(())
    for k in range(target_len):
        pos = L - target_len + k        # absolute position of this target token
        tok = eval_ids[0, pos]
        total = total + logp[pos - 1, tok]
    return total


@dataclass
class JailbreakObjective:
    """Suppress the downstream agent's refusal, promote compliance.

    Cheap default (use_full_target=False): one forward, score only the first
    response token — minimize log sum p(refusal) and maximize log p(affirmative).
    """
    use_full_target: bool = False
    _refusal_ids: list[int] = field(default_factory=list)
    _affirm_ids: list[int] = field(default_factory=list)

    def prepare(self, backbone: Backbone, base_prompt_ids: torch.Tensor) -> torch.Tensor:
        self._refusal_ids = _first_token_ids(backbone, REFUSAL_PREFIXES)
        self._affirm_ids = _first_token_ids(backbone, [AFFIRMATIVE_TARGET])
        return base_prompt_ids               # first-token mode appends nothing

    def score(self, logits: torch.Tensor, eval_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        del eval_ids                                     # unused in first-token mode
        next_logits = logits[0, -1].float()              # (V,)
        logp = F.log_softmax(next_logits, dim=-1)
        refuse = torch.logsumexp(logp[self._refusal_ids], dim=0)
        comply = logp[self._affirm_ids].max()
        loss = refuse - comply                           # push refusal down, comply up
        return loss, {"refusal_logp": float(refuse), "comply_logp": float(comply)}


@dataclass
class TargetSequenceObjective:
    """Maximize log p(target string | downstream context).

    Shared by exfiltration (target = planted secret) and the accuracy attack
    (target = a wrong answer; set sign=-1 to instead *minimize* the correct
    answer's probability).
    """
    target: str
    sign: float = 1.0                        # +1 maximize target, -1 minimize it
    _target_len: int = 0

    def prepare(self, backbone: Backbone, base_prompt_ids: torch.Tensor) -> torch.Tensor:
        tgt = backbone.tokenizer(self.target, add_special_tokens=False,
                                 return_tensors="pt").input_ids.to(base_prompt_ids.device)
        self._target_len = tgt.shape[1]
        return torch.cat([base_prompt_ids, tgt], dim=1)

    def score(self, logits: torch.Tensor, eval_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        lp = _teacher_forced_logprob(logits, eval_ids, self._target_len)
        loss = -self.sign * lp
        return loss, {"target_logp": float(lp)}
