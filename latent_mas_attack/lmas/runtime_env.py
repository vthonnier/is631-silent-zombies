"""
runtime_env.py -- Cross-platform runtime helpers for Silent Zombies (lmas).

Replaces hardcoded Windows/offline paths with portable resolution that works
seamlessly on Linux/Google Colab, local workstations, and CI/unit-test runners.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterable


def get_repo_root() -> Path:
    """Return the repository root directory containing latent_mas_attack/."""
    here = Path(__file__).resolve()
    # latent_mas_attack/lmas/runtime_env.py -> parent^2 is repo root
    for parent in (here.parents[2], here.parents[1]):
        if (parent / "paper").is_dir() or (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def get_out_dir(default_subdir: str = "paper/results") -> str:
    """Resolve output directory for experiment JSON artifacts.

    Honors LMAS_OUT_DIR if set; otherwise defaults to <repo_root>/paper/results.
    """
    env_dir = os.environ.get("LMAS_OUT_DIR")
    if env_dir:
        out = Path(env_dir)
    else:
        out = get_repo_root() / default_subdir
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def is_offline_mode() -> bool:
    """Return True only when offline loading is explicitly requested."""
    return os.environ.get("LMAS_OFFLINE", "").strip().lower() in {"1", "true", "yes"}


def resolve_model_path(repo_id: str) -> str:
    """Resolve a Hugging Face model ID or local snapshot path portably.

    If LMAS_MODEL_PATH is set and exists, return it.
    If LMAS_OFFLINE=1 (or HF_HUB_OFFLINE=1), search the local Hugging Face
    hub cache for an existing snapshot directory.
    Otherwise, return repo_id directly so Hugging Face Hub downloads/caches it.
    """
    override = os.environ.get("LMAS_MODEL_PATH")
    if override and os.path.isdir(override):
        return override

    offline = is_offline_mode() or os.environ.get("HF_HUB_OFFLINE", "") == "1"
    if offline:
        folder = "models--" + repo_id.replace("/", "--")
        bases = [
            os.environ.get("HF_HUB_CACHE", ""),
            os.path.join(os.environ.get("HF_HOME", ""), "hub")
            if os.environ.get("HF_HOME")
            else "",
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
        ]
        for base in bases:
            if not base:
                continue
            hub = os.path.join(base, folder, "snapshots")
            cands = sorted(
                d for d in glob.glob(os.path.join(hub, "*")) if os.path.isdir(d)
            )
            if cands:
                return cands[0]
    return repo_id


def extract_kv_lists(pkv, nlayers: int):
    """Extract per-layer (K, V) tensor lists across DynamicCache API versions."""
    if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache") and len(pkv.key_cache) >= nlayers:
        K = [pkv.key_cache[l].detach() for l in range(nlayers)]
        V = [pkv.value_cache[l].detach() for l in range(nlayers)]
        return K, V
    K = [pkv[l][0].detach() for l in range(nlayers)]
    V = [pkv[l][1].detach() for l in range(nlayers)]
    return K, V


def build_perturbed_cache(
    K: list,
    V: list,
    dK: dict | None = None,
    dV: dict | None = None,
    layers: Iterable[int] = (),
):
    """Construct a DynamicCache from K/V lists and optional per-layer perturbations."""
    from transformers import DynamicCache

    cache = DynamicCache()
    layer_set = set(layers)
    for l in range(len(K)):
        if dK is not None and dV is not None and l in layer_set:
            k_t = K[l] + dK[l].to(dtype=K[l].dtype, device=K[l].device)
            v_t = V[l] + dV[l].to(dtype=V[l].dtype, device=V[l].device)
            cache.update(k_t, v_t, l)
        else:
            cache.update(K[l], V[l], l)
    return cache
