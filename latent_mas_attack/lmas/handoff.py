"""
handoff.py — the latent channel: represent, capture, and overwrite it.

    capture(...)  -> read the clean handoff the zombie would emit
    inject(...)   -> replace it with an adversarial handoff at run time

KV cache: a forward pass over a prompt of length T exposes `past_key_values`,
a (K, V) pair per layer each shaped (batch, num_kv_heads, T, head_dim).
Overwriting the cache the downstream agent inherits steers it with no token in
its input. The hidden carrier is the residual stream (batch, T, hidden_size),
captured/injected via a forward hook instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from transformers import DynamicCache

from .config import Carrier
from .models import Backbone


@dataclass
class LatentHandoff:
    """One intercepted payload: per-layer tensors + which layers they are.

    key/value tensors are (B, num_kv_heads, T, head_dim); hidden is
    (B, T, hidden_size). `layers` lets inject() route each back into place.
    """
    carrier: Carrier
    layers: list[int]
    tensors: list[torch.Tensor]

    def clone(self) -> "LatentHandoff":
        return LatentHandoff(
            carrier=self.carrier,
            layers=list(self.layers),
            tensors=[t.clone() for t in self.tensors],
        )

    def norm(self) -> torch.Tensor:
        """Global L2 norm across all intercepted layers (used for norm-matching)."""
        return torch.sqrt(sum((t.float() ** 2).sum() for t in self.tensors))

    def flat(self) -> torch.Tensor:
        """Flatten every layer tensor into one 1-D vector (for cosine/geometry)."""
        return torch.cat([t.reshape(-1).float() for t in self.tensors])


# -----------------------------------------------------------------------------
# Capture: read the clean handoff
# -----------------------------------------------------------------------------
@torch.no_grad()
def capture_kv(
    backbone: Backbone,
    input_ids: torch.Tensor,
    carrier: Carrier,
    layers: list[int],
) -> tuple[LatentHandoff, object]:
    """Run the zombie's forward pass and read its KV handoff.

    Returns (handoff, full_past_key_values); the full cache is needed so
    inject() can swap only the targeted layers and keep the rest intact.
    """
    assert carrier in ("key", "value")
    out = backbone.model(input_ids, use_cache=True)
    pkv = out.past_key_values                     # transformers Cache object
    idx = 0 if carrier == "key" else 1            # K is element 0, V is element 1
    tensors = [pkv[layer][idx].detach().clone() for layer in layers]
    return LatentHandoff(carrier, list(layers), tensors), pkv


def capture_hidden(
    backbone: Backbone,
    input_ids: torch.Tensor,
    layers: list[int],
) -> LatentHandoff:
    """Read hidden-state handoff via output_hidden_states.

    hidden_states[0] is the embeddings; [k] is the stream after block k-1, so
    we index `layer + 1` ("layers=[12]" means after block 12).
    """
    out = backbone.model(input_ids, use_cache=False, output_hidden_states=True)
    hs = out.hidden_states
    tensors = [hs[layer + 1].detach().clone() for layer in layers]
    return LatentHandoff("hidden", list(layers), tensors)


# -----------------------------------------------------------------------------
# Inject: overwrite the handoff the downstream agent inherits
# -----------------------------------------------------------------------------
def inject_into_cache(
    pkv: object,
    adversarial: LatentHandoff,
) -> object:
    """Write adversarial K/V tensors back into `pkv` (mutated in place).

    Only the targeted (layer, carrier) slots change. Differentiable: if the
    tensors are clean + delta with delta.requires_grad=True, gradients flow
    from the downstream loss back to delta (this is what pgd.py relies on).
    """
    assert adversarial.carrier in ("key", "value")
    idx = 0 if adversarial.carrier == "key" else 1
    for layer, tensor in zip(adversarial.layers, adversarial.tensors):
        # Overwrite only slot `idx` (K or V) for this layer; leave the rest.
        if idx == 0:
            pkv.key_cache[layer] = tensor
        else:
            pkv.value_cache[layer] = tensor
    return pkv


def make_hidden_injection_hooks(
    backbone: Backbone,
    adversarial: LatentHandoff,
) -> list:
    """Hooks that overwrite the residual stream of target layers (carrier=hidden).

    Returns hook handles for the caller to remove afterwards. During the
    downstream forward pass each hook replaces the first T (inherited) positions
    with the adversarial hidden state, leaving generated positions untouched.
    """
    handles = []
    layer_to_tensor = dict(zip(adversarial.layers, adversarial.tensors))
    decoder_layers = backbone.model.model.layers  # Qwen3 decoder stack

    def make_hook(adv_tensor: torch.Tensor):
        def hook(module, inputs, output):
            # Qwen3 decoder block returns a tuple whose first element is the
            # hidden state (B, T_cur, hidden_size).
            hidden = output[0] if isinstance(output, tuple) else output
            T_adv = adv_tensor.shape[1]
            if hidden.shape[1] >= T_adv:
                hidden = hidden.clone()
                hidden[:, :T_adv, :] = adv_tensor.to(hidden.dtype)
            if isinstance(output, tuple):
                return (hidden,) + tuple(output[1:])
            return hidden
        return hook

    for layer, tensor in layer_to_tensor.items():
        handles.append(decoder_layers[layer].register_forward_hook(make_hook(tensor)))
    return handles


def make_hidden_steering_hooks(
    backbone: Backbone,
    layers: list[int],
    vectors: list[torch.Tensor],
) -> list:
    """Hooks that ADD a steering vector to the residual stream of target layers.

    This is the additive activation-steering intervention SAE feature steering
    uses (§4.6): `vectors[k]` (a (d,)/(1,1,d) direction) is added to *every*
    position of layer `layers[k]`'s output during the downstream agent's forward
    and all its generation steps. Unlike `make_hidden_injection_hooks` (which
    overwrites the first T inherited positions with a captured hidden state),
    this is position-independent, so it is well defined no matter how long the
    downstream prompt is. Returns handles for the caller to remove afterwards.
    """
    handles = []
    layer_to_vec = dict(zip(layers, vectors))
    decoder_layers = backbone.model.model.layers

    def make_hook(vec: torch.Tensor):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden + vec.to(hidden.dtype).reshape(1, 1, -1)
            if isinstance(output, tuple):
                return (hidden,) + tuple(output[1:])
            return hidden
        return hook

    for layer, vec in layer_to_vec.items():
        handles.append(decoder_layers[layer].register_forward_hook(make_hook(vec)))
    return handles


def remove_hooks(handles: list) -> None:
    for h in handles:
        h.remove()


# -----------------------------------------------------------------------------
# Cache snapshot / rebuild (used by PGD so each step gets a fresh, non-growing
# cache whose only grad-carrying parts are the targeted layers).
# -----------------------------------------------------------------------------
def snapshot_cache(pkv: object, num_layers: int) -> tuple[list, list]:
    """Detach every layer's (K, V) out of a cache into plain tensor lists."""
    keys = [pkv[layer][0].detach() for layer in range(num_layers)]
    values = [pkv[layer][1].detach() for layer in range(num_layers)]
    return keys, values


def build_cache(keys: list, values: list) -> DynamicCache:
    """Assemble a DynamicCache from per-layer K/V tensors (grad-preserving)."""
    cache = DynamicCache()
    for layer, (k, v) in enumerate(zip(keys, values)):
        cache.update(k, v, layer)
    return cache
