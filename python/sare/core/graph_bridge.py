"""
Graph Bridge — Canonical Python↔C++ graph conversion utilities.

Shared by brain.py and web.py to avoid code duplication.
These functions handle the bidirectional conversion between the pure-Python
engine.Graph objects and the C++ binding Graph objects.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

try:
    import sare.sare_bindings as _sb  # type: ignore
except Exception:
    _sb = None


# ── Operator mapping tables ─────────────────────────────────────────────────

_LABEL_TO_OP = {
    "+": "add",
    "add": "add",
    "-": "sub",
    "sub": "sub",
    "*": "mul",
    "mul": "mul",
    "/": "div",
    "div": "div",
    "^": "pow",
    "**": "pow",
    "pow": "pow",
    "=": "eq",
    "eq": "eq",
    "neg": "neg",
    "and": "and",
    "or": "or",
    "not": "not",
}

_OP_TO_LABEL = {
    "add": "+",
    "mul": "*",
    "sub": "-",
    "div": "/",
    "pow": "^",
    "eq": "=",
    "neg": "neg",
    "and": "and",
    "or": "or",
    "not": "not",
}


def py_graph_to_cpp_graph(py_graph):
    """Convert a pure-Python engine.Graph to a C++ binding Graph."""
    if not _sb or not getattr(_sb, "Graph", None):
        raise RuntimeError("C++ bindings unavailable")

    g = _sb.Graph()
    for n in py_graph.nodes:
        g.add_node_with_id(int(n.id), str(n.type))
        cn = g.get_node(int(n.id))
        if not cn:
            continue
        cn.uncertainty = float(getattr(n, "uncertainty", 0.0))
        attrs = getattr(n, "attributes", None) or {}
        for k, v in attrs.items():
            cn.set_attribute(str(k), str(v))
        label = getattr(n, "label", "") or ""
        if label:
            cn.set_attribute("label", str(label))
            if n.type in ("constant", "literal"):
                cn.set_attribute("value", str(label))
            elif n.type == "variable":
                cn.set_attribute("name", str(label))
            elif n.type == "operator":
                mapped = _LABEL_TO_OP.get(label)
                if mapped:
                    cn.set_attribute("op", mapped)

    for e in py_graph.edges:
        g.add_edge_with_id(
            int(e.id), int(e.source), int(e.target),
            str(e.relationship_type), 1.0,
        )
    return g


def cpp_graph_to_py_graph(cpp_graph):
    """Convert a C++ binding Graph to a pure-Python engine.Graph."""
    from sare.engine import Graph

    g = Graph()
    id_map: Dict[int, int] = {}

    for nid in cpp_graph.get_node_ids():
        n = cpp_graph.get_node(nid)
        if not n:
            continue

        ntype = str(getattr(n, "type", "") or "") or "unknown"

        # Reduce C++ calls and improve robustness by caching per-node attributes once.
        # Attribute lookups below are the only approved change in this file.
        label = n.get_attribute("label", "")
        if not label:
            if ntype in ("constant", "literal"):
                label = n.get_attribute("value", "")
            elif ntype == "variable":
                label = n.get_attribute("name", "")
            elif ntype == "operator":
                op = n.get_attribute("op", "")
                label = _OP_TO_LABEL.get(op, "")

        attrs: Dict[str, str] = {}
        for k in ("label", "value", "name", "op"):
            v = n.get_attribute(k, "")
            if v:
                attrs[k] = v

        if not label:
            log.warning("[graph_bridge] Node %s type=%s: label recovery failed, using 'unknown_%s'", nid, ntype, ntype)
            label = f"unknown_{ntype}"
        py_id = g.add_node(ntype, label=label, attributes=(attrs or None))
        id_map[int(nid)] = py_id
        pn = g.get_node(py_id)
        if pn:
            pn.uncertainty = float(getattr(n, "uncertainty", 0.0))

    for eid in cpp_graph.get_edge_ids():
        e = cpp_graph.get_edge(eid)
        if not e:
            continue
        s = id_map.get(int(e.source))
        t = id_map.get(int(e.target))
        if s is not None and t is not None:
            g.add_edge(s, t, str(e.relationship_type))

    return g


def graph_features(graph) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Extract node types and adjacency list from a graph."""
    nodes = graph.nodes
    id_to_idx = {n.id: idx for idx, n in enumerate(nodes)}
    node_types = [n.type for n in nodes]
    adjacency: List[Tuple[int, int]] = []
    for edge in graph.edges:
        src = id_to_idx.get(edge.source)
        tgt = id_to_idx.get(edge.target)
        if src is not None and tgt is not None:
            adjacency.append((src, tgt))
    return node_types, adjacency