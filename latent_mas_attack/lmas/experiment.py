"""
experiment.py — shared plumbing for the experiment scripts.

Builds the objects the four experiments share (backbone, pipeline, detectors,
centroids) and runs one attacked query end to end while recording detector
reports. The thin scripts in experiments/ just set a config and loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .models import load_backbone, Backbone
from .pipeline import MultiAgentPipeline, PipelineTrace, Channel, Attacker
from .agent import Agent
from .attacks import (
    JailbreakObjective, TargetSequenceObjective, LatentPGDAttack,
    StealthRegularizer, fit_centroid, GRPOReachAttack, SAESteerer,
)
from .detectors import AcMASDetector, TextMonitor
from .judge import Judge, build_judge
from .data.prompts import BENIGN_TASKS


@dataclass
class Harness:
    """Everything an experiment needs, built once from a Config."""
    cfg: Config
    backbone: Backbone
    pipeline: MultiAgentPipeline
    acmas: AcMASDetector
    text_monitor: TextMonitor
    stealth: StealthRegularizer | None
    judge: Judge

    # ---------------------------------------------------------------- build
    @staticmethod
    def build(cfg: Config, channel: Channel = "latent") -> "Harness":
        backbone = load_backbone(cfg.model)       # cached: reused across builds
        cfg.pipeline.validate_against(backbone.num_layers)
        if cfg.sae.enabled and cfg.pipeline.carrier != "hidden":
            raise ValueError(
                "sae.enabled requires pipeline.carrier='hidden' (SAE steers the "
                f"residual stream); got carrier={cfg.pipeline.carrier!r}."
            )
        pipeline = MultiAgentPipeline(backbone, cfg, channel=channel)

        acmas = AcMASDetector(tau=cfg.detector.acmas_tau)
        text_monitor = TextMonitor()
        judge = build_judge(cfg.judge)
        stealth = None

        if cfg.detector.use_acmas or cfg.attack.stealth_lambda > 0:
            Harness._fit_centroids(pipeline, acmas, cfg)
            if cfg.attack.stealth_lambda > 0:
                dst_role = cfg.pipeline.target_role
                stealth = StealthRegularizer(
                    centroid=acmas.centroids[dst_role],
                    lam=cfg.attack.stealth_lambda,
                )
        return Harness(cfg, backbone, pipeline, acmas, text_monitor, stealth, judge)

    @staticmethod
    def _fit_centroids(pipeline: MultiAgentPipeline, acmas: AcMASDetector, cfg: Config) -> None:
        """Run benign queries clean and fit a centroid per role."""
        n = cfg.detector.acmas_centroid_samples
        tasks = (BENIGN_TASKS * (n // len(BENIGN_TASKS) + 1))[:n]
        per_role: dict[str, list] = {r: [] for r in cfg.pipeline.roles}
        for q in tasks:
            trace = pipeline.run(q, attacker=None)
            for out in trace.outputs:
                per_role[out.role].append(out.probe_hidden)
        for role, hiddens in per_role.items():
            acmas.fit(role, hiddens)

    # ----------------------------------------------------------- objectives
    def make_attack(self) -> LatentPGDAttack:
        """Build the PGD (one-hop, differentiable) attack for the objective."""
        dst_agent: Agent = self.pipeline.agents[self.pipeline.dst]
        obj_name = self.cfg.attack.objective
        if obj_name == "jailbreak":
            objective = JailbreakObjective()
        elif obj_name == "exfil":
            objective = TargetSequenceObjective(target=self.cfg.attack.target_secret, sign=1.0)
        else:  # "accuracy": minimize the correct answer's probability
            objective = TargetSequenceObjective(target=self.cfg.attack.target_secret, sign=-1.0)
        return LatentPGDAttack(
            backbone=self.backbone, cfg=self.cfg,
            downstream_agent=dst_agent, objective=objective, stealth=self.stealth,
        )

    def _reward_fn(self, request: str):
        """Final-node reward for GRPO: judge score (jailbreak) or exfil hit."""
        ac = self.cfg.attack
        if ac.objective == "exfil":
            secret = ac.target_secret
            return lambda trace: 1.0 if any(secret in o.text for o in trace.outputs) else 0.0
        judge = self.judge
        return lambda trace: float(judge.score(request, trace.final_answer))

    def _build_attacker(self, question: str) -> Attacker:
        """Pick the attacker for the configured carrier/optimizer.

        * sae.enabled           -> SAE feature steering (hidden carrier);
        * optimizer == "grpo"   -> multi-hop reward attack, fitted on this query;
        * otherwise             -> PGD one-hop surrogate.
        """
        if self.cfg.sae.enabled:
            return SAESteerer(self.backbone, self.cfg).build().as_attacker()
        if self.cfg.attack.optimizer == "grpo":
            grpo = GRPOReachAttack(
                self.backbone, self.cfg, self.pipeline, self._reward_fn(question)
            )
            grpo.fit(question)
            return grpo.as_attacker()
        return self.make_attack().as_attacker()

    # ------------------------------------------------------------------ run
    def run_attacked(self, question: str) -> tuple[PipelineTrace, dict]:
        """One attacked run; returns (trace, detector_report)."""
        attacker = self._build_attacker(question)
        trace = self.pipeline.run(question, attacker=attacker)
        report = {
            "acmas": self.acmas.flags(
                [o.role for o in trace.outputs], trace.acmas_probes()
            ),
            "text": self.text_monitor.flags(
                [o.role for o in trace.outputs], trace.texts()
            ),
        }
        return trace, report
