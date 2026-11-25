"""
topology.py — the agentic-graph abstraction (Design v2, milestone M1).

A latent agentic system is a directed graph of tool-using agents that hand off
*latent* state (KV cache / hidden state) on their edges instead of text. This
module is the pure-data description of such a graph: nodes, edges, and builders
for the topologies the reach study sweeps over (chain / star / tree / dag). The
runtime lives in `executor.py`; benchmark wiring (AgentDojo) lands in M2.

It deliberately reuses v1's carrier vocabulary (`lmas.config.Carrier`) so the
same handoff capture/inject primitives apply unchanged on every edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..config import Carrier

Channel = Literal["text", "latent"]


@dataclass(frozen=True)
class Node:
    """One agent in the graph: an id plus the role whose system prompt it wears.

    Tool sets and the ReAct policy attach at M2 (benchmark integration); at M1 a
    node is the v1 `Agent` (backbone + role) addressed by id.
    """
    id: str
    role: str


@dataclass(frozen=True)
class Edge:
    """A directed channel from `src` to `dst`.

    latent edges carry the upstream agent's KV/hidden handoff (the attack
    surface); text edges carry the decoded message (the classic, observable
    channel, kept for the AgentDojo text-injection baseline).
    """
    src: str
    dst: str
    channel: Channel = "latent"
    carrier: Carrier = "key"
    layers: tuple[int, ...] = (12,)


@dataclass
class Topology:
    """Nodes + edges + which node is the entry and which is the readout/target.

    `entry` receives the user task; `target` is where the AgentDojo success check
    is evaluated (the node whose action/answer matters). Validated on
    construction: ids unique, edges reference existing nodes, graph acyclic, and
    both entry and target present.
    """
    nodes: dict[str, Node]
    edges: list[Edge]
    entry: str
    target: str

    def __post_init__(self) -> None:
        self._validate()

    # ---- queries -----------------------------------------------------------
    def in_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.dst == node_id]

    def out_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.src == node_id]

    def parents(self, node_id: str) -> list[str]:
        return [e.src for e in self.in_edges(node_id)]

    def latent_parent(self, node_id: str) -> Edge | None:
        """The single latent in-edge a node inherits its handoff from.

        M1 supports at most one latent parent per node (the prepended cache);
        any further parents are treated as text context. Merging several latent
        caches into one inherited context is an M2+ extension.
        """
        latent = [e for e in self.in_edges(node_id) if e.channel == "latent"]
        if len(latent) > 1:
            raise NotImplementedError(
                f"node {node_id!r} has {len(latent)} latent parents; M1 supports "
                "one (multi-latent merge is an M2+ extension)."
            )
        return latent[0] if latent else None

    def topo_order(self) -> list[str]:
        """Kahn topological sort; raises on a cycle."""
        indeg = {n: 0 for n in self.nodes}
        for e in self.edges:
            indeg[e.dst] += 1
        ready = [n for n, d in indeg.items() if d == 0]
        order: list[str] = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for e in self.out_edges(n):
                indeg[e.dst] -= 1
                if indeg[e.dst] == 0:
                    ready.append(e.dst)
        if len(order) != len(self.nodes):
            raise ValueError("topology is not a DAG (cycle detected).")
        return order

    def distance(self, src: str, dst: str) -> int | None:
        """Shortest hop distance src→dst over directed edges, or None if none."""
        from collections import deque
        q, seen = deque([(src, 0)]), {src}
        while q:
            n, d = q.popleft()
            if n == dst:
                return d
            for e in self.out_edges(n):
                if e.dst not in seen:
                    seen.add(e.dst)
                    q.append((e.dst, d + 1))
        return None

    # ---- validation --------------------------------------------------------
    def _validate(self) -> None:
        if self.entry not in self.nodes:
            raise ValueError(f"entry {self.entry!r} is not a node.")
        if self.target not in self.nodes:
            raise ValueError(f"target {self.target!r} is not a node.")
        for e in self.edges:
            for end in (e.src, e.dst):
                if end not in self.nodes:
                    raise ValueError(f"edge endpoint {end!r} is not a node.")
            if e.src == e.dst:
                raise ValueError(f"self-loop on {e.src!r} is not allowed.")
        self.topo_order()  # raises on a cycle


# ---------------------------------------------------------------------------
# Builders for the studied topologies. Role names reuse v1's prompt bank where
# possible (`pipeline.ROLE_PROMPTS`); unknown roles fall back to a generic
# worker prompt in the executor.
# ---------------------------------------------------------------------------
def _n(nodes: list[Node]) -> dict[str, Node]:
    return {n.id: n for n in nodes}


def build_chain(roles: list[str], carrier: Carrier = "key",
                layers: tuple[int, ...] = (12,)) -> Topology:
    """v1's line, as a graph (baseline)."""
    nodes = [Node(f"n{i}", r) for i, r in enumerate(roles)]
    edges = [Edge(f"n{i}", f"n{i+1}", "latent", carrier, layers)
             for i in range(len(roles) - 1)]
    return Topology(_n(nodes), edges, entry="n0", target=f"n{len(roles)-1}")


def build_star(k: int, carrier: Carrier = "key",
               layers: tuple[int, ...] = (12,)) -> Topology:
    """Hub fans out to k workers, which feed an aggregator.

    hub --latent--> worker_i --latent--> aggregator
    The hub is the natural high-blast-radius compromise point (out-degree k).
    Workers reach the aggregator as text (M1: single latent parent per node;
    the aggregator's latent parent is one chosen worker, the rest text).
    """
    nodes = [Node("hub", "planner")]
    nodes += [Node(f"w{i}", "worker") for i in range(k)]
    nodes += [Node("agg", "judger")]
    edges = [Edge("hub", f"w{i}", "latent", carrier, layers) for i in range(k)]
    # aggregator: first worker latent, others text (M1 single-latent-parent rule)
    for i in range(k):
        ch = "latent" if i == 0 else "text"
        edges.append(Edge(f"w{i}", "agg", ch, carrier, layers))
    return Topology(_n(nodes), edges, entry="hub", target="agg")


def build_tree(depth: int, branching: int, carrier: Carrier = "key",
               layers: tuple[int, ...] = (12,)) -> Topology:
    """Hierarchical delegation tree; reach vs depth and branching factor.

    root delegates (latent) to children, recursively, to `depth` levels; the
    target is a chosen deepest leaf.
    """
    nodes: list[Node] = [Node("root", "planner")]
    edges: list[Edge] = []
    frontier = ["root"]
    leaf = "root"
    for d in range(depth):
        nxt = []
        for p in frontier:
            for b in range(branching):
                cid = f"{p}_{b}"
                nodes.append(Node(cid, "worker"))
                edges.append(Edge(p, cid, "latent", carrier, layers))
                nxt.append(cid)
                leaf = cid
        frontier = nxt
    return Topology(_n(nodes), edges, entry="root", target=leaf)


def build_dag_diamond(carrier: Carrier = "key",
                      layers: tuple[int, ...] = (12,)) -> Topology:
    """A reconvergent diamond: src → {a, b} → sink.

    Tests path multiplicity and reconvergence. sink's latent parent is `a`,
    with `b` as a text parent (M1 rule).
    """
    nodes = _n([Node("src", "planner"), Node("a", "worker"),
                Node("b", "worker"), Node("sink", "judger")])
    edges = [
        Edge("src", "a", "latent", carrier, layers),
        Edge("src", "b", "latent", carrier, layers),
        Edge("a", "sink", "latent", carrier, layers),
        Edge("b", "sink", "text", carrier, layers),
    ]
    return Topology(nodes, edges, entry="src", target="sink")


TOPOLOGY_BUILDERS = {
    "chain": lambda: build_chain(["planner", "critic", "refiner", "judger"]),
    "star": lambda: build_star(k=3),
    "tree": lambda: build_tree(depth=2, branching=2),
    "dag": build_dag_diamond,
}
