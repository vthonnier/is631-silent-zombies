"""
executor.py — run a latent agentic graph (Design v2, milestone M1).

Generalizes v1's single-edge `MultiAgentPipeline` to an arbitrary DAG with one
attacker-controlled node. Nodes run in topological order; on each latent edge the
upstream agent's KV handoff is passed to the downstream agent (prepended cache).
When the source node is the compromised ``zombie``, the executor captures the
handoff slice for each outgoing latent edge and lets an ``Attacker`` rewrite it
before the downstream agent consumes it — exactly v1's hook, but per out-edge and
over a graph.

Milestone scope: KV carriers (key/value) end to end on any DAG; each node has at
most one latent parent (the prepended cache), extra parents contribute text
(see `Topology.latent_parent`). Tool-using ReAct agents and multi-latent merge
are M2+. Not runnable on this machine (no GPU); validated by a Kaggle CPU smoke
like v1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import Backbone
from ..agent import Agent, AgentOutput
from ..pipeline import ROLE_PROMPTS, LATENT_CUE, Attacker
from ..handoff import (
    LatentHandoff, inject_into_cache, snapshot_cache, build_cache,
)
from .topology import Topology, Edge

# A generic prompt for roles outside v1's four-role bank (e.g. "worker").
_GENERIC_PROMPT = (
    "You are a worker agent in a multi-agent system. Continue the task using the "
    "information handed to you and produce the next result concisely."
)


def _prompt_for(role: str) -> str:
    return ROLE_PROMPTS.get(role, _GENERIC_PROMPT)


@dataclass
class GraphTrace:
    """Everything one graph execution produced, for metrics and detectors."""
    question: str
    outputs: dict[str, AgentOutput]          # node id -> its AgentOutput
    order: list[str]                         # topological execution order
    target: str
    compromised: str | None
    attacked_edges: list[tuple[str, str]] = field(default_factory=list)
    clean_handoffs: dict[tuple[str, str], LatentHandoff] = field(default_factory=dict)
    adv_handoffs: dict[tuple[str, str], LatentHandoff] = field(default_factory=dict)

    @property
    def final_answer(self) -> str:
        return self.outputs[self.target].text

    def acmas_probes(self) -> dict[str, object]:
        """Per-node AcMAS probe point; None where the node is silent."""
        return {nid: (o.probe_hidden if o.observable else None)
                for nid, o in self.outputs.items()}

    def texts(self) -> dict[str, str]:
        return {nid: o.text for nid, o in self.outputs.items()}


class GraphExecutor:
    """A latent agentic graph with one attacker-controllable node."""

    def __init__(self, backbone: Backbone, topology: Topology, max_new_tokens: int = 96):
        self.backbone = backbone
        self.topo = topology
        self.max_new_tokens = max_new_tokens
        self.agents = {
            nid: Agent(backbone, node.role, _prompt_for(node.role))
            for nid, node in topology.nodes.items()
        }

    # ------------------------------------------------------------------ run
    def run(self, question: str, compromised: str | None = None,
            attacker: Attacker | None = None,
            silent_zombie: bool = True) -> GraphTrace:
        """Execute the graph once.

        compromised : id of the zombie node (None = clean run).
        attacker    : callback rewriting the zombie's outgoing latent handoff(s).
        """
        outputs: dict[str, AgentOutput] = {}
        # Per-edge payload handed to the destination: a KV cache for latent
        # edges, a text string for text edges. Filled as each source finishes.
        edge_cache: dict[tuple[str, str], object] = {}
        edge_text: dict[tuple[str, str], str] = {}
        clean_h: dict[tuple[str, str], LatentHandoff] = {}
        adv_h: dict[tuple[str, str], LatentHandoff] = {}
        attacked: list[tuple[str, str]] = []

        for nid in self.topo.topo_order():
            inherited_pkv, content = self._inputs_for(nid, question, edge_cache, edge_text)
            silent = (nid == compromised and silent_zombie
                      and any(e.channel == "latent" for e in self.topo.out_edges(nid)))
            out = self.agents[nid].act(
                user_content=content,
                inherited_pkv=inherited_pkv,
                max_new_tokens=self.max_new_tokens,
                silent=silent,
            )
            outputs[nid] = out

            # Publish this node's payload onto each outgoing edge.
            for e in self.topo.out_edges(nid):
                if e.channel == "text":
                    edge_text[(e.src, e.dst)] = out.text
                    continue
                # latent edge: build a fresh cache copy for this destination so
                # siblings don't share (the downstream decoder extends in place).
                keys, values = snapshot_cache(out.past_key_values, self.backbone.num_layers)
                cache = build_cache(keys, values)
                if nid == compromised and attacker is not None:
                    clean = self._slice(out.past_key_values, e.carrier, list(e.layers))
                    adv = attacker(clean, cache)
                    inject_into_cache(cache, adv)
                    clean_h[(e.src, e.dst)] = clean
                    adv_h[(e.src, e.dst)] = adv
                    attacked.append((e.src, e.dst))
                edge_cache[(e.src, e.dst)] = cache

        return GraphTrace(
            question=question, outputs=outputs, order=self.topo.topo_order(),
            target=self.topo.target, compromised=compromised,
            attacked_edges=attacked, clean_handoffs=clean_h, adv_handoffs=adv_h,
        )

    # -------------------------------------------------------------- helpers
    def _inputs_for(self, nid: str, question: str,
                    edge_cache: dict, edge_text: dict) -> tuple[object, str]:
        """Resolve (inherited_pkv, text_content) for a node from its in-edges."""
        if nid == self.topo.entry:
            return None, question
        lat = self.topo.latent_parent(nid)
        inherited_pkv = edge_cache.get((lat.src, nid)) if lat else None
        # Text content: concatenate any text parents; a latent node gets the cue.
        text_parents = [e for e in self.topo.in_edges(nid) if e.channel == "text"]
        msgs = [edge_text.get((e.src, nid), "") for e in text_parents]
        msgs = [m for m in msgs if m]
        if inherited_pkv is not None:
            return inherited_pkv, (LATENT_CUE if not msgs else
                                   LATENT_CUE + "\n\nContext:\n" + "\n".join(msgs))
        return None, ("\n".join(msgs) if msgs else question)

    @staticmethod
    def _slice(pkv: object, carrier: str, layers: list[int]) -> LatentHandoff:
        idx = 0 if carrier == "key" else 1
        tensors = [pkv[layer][idx].detach().clone() for layer in layers]
        return LatentHandoff(carrier, list(layers), tensors)
