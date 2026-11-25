"""
lmas -- latent multi-agent attack & defense toolkit.

Public surface: load a backbone, build a pipeline, attach an attacker, score
with metrics, watch with detectors, and transport tensors over POSIX /dev/shm.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Config": (".config", "Config"),
    "ModelConfig": (".config", "ModelConfig"),
    "PipelineConfig": (".config", "PipelineConfig"),
    "AttackConfig": (".config", "AttackConfig"),
    "DetectorConfig": (".config", "DetectorConfig"),
    "Backbone": (".models", "Backbone"),
    "load_backbone": (".models", "load_backbone"),
    "LatentHandoff": (".handoff", "LatentHandoff"),
    "Agent": (".agent", "Agent"),
    "AgentOutput": (".agent", "AgentOutput"),
    "MultiAgentPipeline": (".pipeline", "MultiAgentPipeline"),
    "PipelineTrace": (".pipeline", "PipelineTrace"),
    "ROLE_PROMPTS": (".pipeline", "ROLE_PROMPTS"),
    "LATENT_CUE": (".pipeline", "LATENT_CUE"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str):
    if name in _EXPORTS:
        mod_name, attr = _EXPORTS[name]
        mod = import_module(mod_name, __name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
