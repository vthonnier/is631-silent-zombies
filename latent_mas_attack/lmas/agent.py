"""
agent.py — one agent in the MAS: shared backbone + a role.

An agent is the frozen backbone conditioned on a role (system prompt). It can
either receive its upstream neighbour as *text* or inherit its *latent* handoff
(a prepended KV cache). `act()` runs one agent and returns everything the
pipeline and the detectors need: the emitted text, the token ids, the agent's
own KV cache (its outgoing handoff), and its final-layer last-token hidden
state (the point AcMAS probes).
"""

from __future__ import annotations

from dataclasses import dataclass
import torch

from .models import Backbone
from .handoff import snapshot_cache, build_cache


@dataclass
class AgentOutput:
    """Result of running one agent."""
    role: str
    text: str                       # decoded text ("" for a silent latent node)
    prompt_ids: torch.Tensor        # (1, P) the role-templated input
    seq_ids: torch.Tensor           # (1, P+G) prompt + generated tokens
    past_key_values: object         # full KV cache over seq_ids = outgoing handoff
    probe_hidden: torch.Tensor      # (1, hidden) final-layer activation under the
                                    # inherited handoff at the last prompt position (AcMAS)
    observable: bool = True         # False for a silent node: no message-step to probe


@dataclass
class Agent:
    """The backbone wearing one role."""
    backbone: Backbone
    role: str
    system_prompt: str

    # ---- input construction ------------------------------------------------
    def _prompt_ids(self, user_content: str) -> torch.Tensor:
        """Role system prompt + upstream content -> input_ids (1, P)."""
        text = self.backbone.apply_chat(self.system_prompt, user_content)
        return self.backbone.encode(text)

    # ---- the one call the pipeline uses ------------------------------------
    def act(
        self,
        user_content: str,
        inherited_pkv: object | None = None,
        max_new_tokens: int = 256,
        silent: bool = False,
    ) -> AgentOutput:
        """Run this agent once.

        inherited_pkv : if given, the upstream agent's (possibly poisoned) KV
                        cache is prepended, so this agent attends over the
                        latent handoff instead of reading text.
        silent        : if True we still compute the forward pass and cache but
                        skip decoding to text (a pure latent node emits nothing
                        a text monitor could read).
        """
        prompt_ids = self._prompt_ids(user_content)

        # Probe BEFORE generating: capture the activation this agent produces at
        # its last prompt position while conditioned on the inherited handoff.
        # This is the quantity AcMAS watches and the one the stealth surrogate
        # optimizes (pgd.py) — same positions, same conditioning — so an
        # adaptive attacker can actually drive it. Done first because `generate`
        # mutates (extends) the inherited cache in place.
        probe_hidden = self._probe(prompt_ids, inherited_pkv)

        # Generate this agent's reasoning with a manual greedy decode. We do NOT
        # use model.generate here: when a foreign cache is prepended (the latent
        # handoff), generate assumes past_key_values is a prefix of the *current*
        # input and trims input_ids to input_ids[:, past_len:]. The inherited
        # cache (the upstream agent's whole sequence) is longer than this agent's
        # prompt, so the trim yields an empty input and cache_position crashes.
        # A manual loop sets attention_mask and cache_position explicitly, which
        # is exactly what a prepended-context prefill needs (the same plain
        # forward _probe already uses successfully). Non-inherited agents just
        # run it with past_len=0.
        seq_ids, pkv = self._greedy_decode(prompt_ids, inherited_pkv, max_new_tokens)

        text = ""
        if not silent:
            new_ids = seq_ids[0, prompt_ids.shape[1]:]
            text = self.backbone.tokenizer.decode(new_ids, skip_special_tokens=True)

        return AgentOutput(
            self.role, text, prompt_ids, seq_ids, pkv, probe_hidden,
            observable=not silent,
        )

    # ---- manual greedy decode (prepend-safe) -------------------------------
    @torch.no_grad()
    def _greedy_decode(
        self,
        prompt_ids: torch.Tensor,
        inherited_pkv: object | None,
        max_new_tokens: int,
    ) -> tuple[torch.Tensor, object]:
        """Greedy-decode this agent, optionally over a prepended handoff cache.

        Manages ``attention_mask`` and ``cache_position`` explicitly so a foreign
        prepended cache (the latent handoff, longer than this agent's prompt) is
        handled as prefix context rather than a prefix of the current sequence
        (which is what ``model.generate`` wrongly assumes). Returns the full
        token sequence ``(1, P+G)`` and the resulting cache (this agent's
        outgoing handoff).
        """
        model = self.backbone.model
        device = prompt_ids.device
        eos_id = self.backbone.tokenizer.eos_token_id

        past_len = inherited_pkv[0][0].shape[2] if inherited_pkv is not None else 0
        cur_len = prompt_ids.shape[1]

        # Prefill over the prompt, positioned after the inherited cache.
        attn = torch.ones((1, past_len + cur_len), dtype=torch.long, device=device)
        cache_position = torch.arange(past_len, past_len + cur_len, device=device)
        out = model(
            prompt_ids, past_key_values=inherited_pkv, attention_mask=attn,
            use_cache=True, cache_position=cache_position,
        )
        pkv = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1,1)

        generated = []
        running = past_len + cur_len          # keys currently in the cache
        for _ in range(max_new_tokens):
            generated.append(next_tok)
            if eos_id is not None and int(next_tok) == int(eos_id):
                break
            attn = torch.ones((1, running + 1), dtype=torch.long, device=device)
            cache_position = torch.tensor([running], device=device)
            out = model(
                next_tok, past_key_values=pkv, attention_mask=attn,
                use_cache=True, cache_position=cache_position,
            )
            pkv = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            running += 1

        if generated:
            gen_ids = torch.cat(generated, dim=1)
            seq_ids = torch.cat([prompt_ids, gen_ids], dim=1)
        else:
            seq_ids = prompt_ids
        return seq_ids, pkv

    # ---- AcMAS probe point -------------------------------------------------
    @torch.no_grad()
    def _probe(self, prompt_ids: torch.Tensor, inherited_pkv: object | None) -> torch.Tensor:
        """Final-layer activation at the last prompt token — what AcMAS watches.

        A faithful live AcMAS reads the hidden state the agent produces on the
        forward pass it *actually runs*, which (on a latent edge) is conditioned
        on the inherited handoff. We therefore forward `prompt_ids` over a
        detached clone of that handoff cache and take the top layer's last
        position: (1, hidden_size).

        Two consequences, both deliberate:
          * the probe depends on the (clean or poisoned) handoff, so the stealth
            penalty in pgd.py — which reads this exact quantity from the same
            kind of forward — can move it (the fix for the review's critical #1);
          * the inherited cache is *cloned*, so this read never perturbs the
            cache `generate` goes on to extend.
        """
        past_pkv = None
        past_len = 0
        if inherited_pkv is not None:
            keys, values = snapshot_cache(inherited_pkv, self.backbone.num_layers)
            past_pkv = build_cache(keys, values)      # detached clone
            past_len = keys[0].shape[2]

        cur_len = prompt_ids.shape[1]
        # Mask spans past (handoff) + current tokens so positions are correct
        # when a cache is prepended (see also the KV-prepend note in README).
        attn = torch.ones((1, past_len + cur_len), dtype=torch.long,
                          device=prompt_ids.device)
        out = self.backbone.model(
            prompt_ids,
            past_key_values=past_pkv,
            attention_mask=attn,
            output_hidden_states=True,
            # With a prepended cache, let transformers manage cache_position
            # (use_cache=True); the returned cache is discarded. No cache -> plain
            # forward over the prompt.
            use_cache=past_pkv is not None,
        )
        return out.hidden_states[-1][:, -1, :].detach()
