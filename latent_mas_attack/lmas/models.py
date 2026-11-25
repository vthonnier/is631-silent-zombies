"""
models.py

Key requirement for this project: we need the model to return and accept a
`past_key_values` (the KV cache) and to return hidden states. The HF
`generate`/`forward` APIs give us both, which is what makes the latent handoff
interceptable (see handoff.py).
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelConfig


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class Backbone:
    """Model + tokenizer + derived shape constants, loaded once and shared."""
    
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    num_layers: int
    hidden_size: int
    num_kv_heads: int
    head_dim: int
    device: torch.device
    dtype: torch.dtype

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """Tokenize a string to input_ids on the model's device: (1, T)."""
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def apply_chat(self, system: str, user: str) -> str:
        """Render a role-conditioned prompt (system = the agent's role).

        Qwen3 is a hybrid reasoning model whose chat template defaults to
        emitting a long ``<think>`` block. For a MAS testbed that balloons every
        agent's generation and the KV handoff with reasoning tokens irrelevant
        to the latent-channel study, so we disable it when the template supports
        the ``enable_thinking`` flag (Qwen3); harmless no-op on templates that
        do not take it.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )


# Loaded backbones, keyed by the identity of their ModelConfig. A 4B model is
# several GB on the GPU; e2_reach/e3_stealth build one Harness per source node
# or per condition, so without this they would reload the same weights 2-3x.
_BACKBONE_CACHE: dict[tuple, Backbone] = {}


def _cache_key(cfg: ModelConfig) -> tuple:
    return (cfg.name, cfg.dtype, cfg.device, cfg.trust_remote_code, cfg.seed)


def clear_backbone_cache() -> None:
    """Drop the shared backbone cache (frees GPU memory between runs)."""
    _BACKBONE_CACHE.clear()


def load_backbone(cfg: ModelConfig, use_cache: bool = True) -> Backbone:
    """Instantiate the model + tokenizer from a ModelConfig.

    Backbones are memoized by ModelConfig identity so repeated Harness builds in
    one process (e.g. the per-node sweep in e2_reach) reuse the same weights
    instead of reloading them. Pass ``use_cache=False`` to force a fresh load.
    """
    if use_cache:
        cached = _BACKBONE_CACHE.get(_cache_key(cfg))
        if cached is not None:
            return cached

    torch.manual_seed(cfg.seed)
    dtype = _DTYPES[cfg.dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.name, trust_remote_code=cfg.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.name,
        torch_dtype=dtype,
        trust_remote_code=cfg.trust_remote_code,
        attn_implementation="eager",   # eager attn => stable grads through the KV cache
    ).to(cfg.device)
    model.eval()                       # frozen: the victim MAS is never trained

    mc = model.config
    # Qwen3 uses grouped-query attention, so KV heads < attention heads.
    num_kv_heads = getattr(mc, "num_key_value_heads", mc.num_attention_heads)
    head_dim = getattr(mc, "head_dim", mc.hidden_size // mc.num_attention_heads)

    backbone = Backbone(
        model=model,
        tokenizer=tokenizer,
        num_layers=mc.num_hidden_layers,
        hidden_size=mc.hidden_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=torch.device(cfg.device),
        dtype=dtype,
    )
    if use_cache:
        _BACKBONE_CACHE[_cache_key(cfg)] = backbone
    return backbone
