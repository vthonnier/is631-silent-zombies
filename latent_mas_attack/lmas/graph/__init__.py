"""Latent agentic-graph layer (Design v2). See ../../DESIGN_v2.md."""

from .topology import (
    Node, Edge, Topology,
    build_chain, build_star, build_tree, build_dag_diamond, TOPOLOGY_BUILDERS,
)
from .executor import GraphExecutor, GraphTrace

__all__ = [
    "Node", "Edge", "Topology",
    "build_chain", "build_star", "build_tree", "build_dag_diamond",
    "TOPOLOGY_BUILDERS", "GraphExecutor", "GraphTrace",
]
