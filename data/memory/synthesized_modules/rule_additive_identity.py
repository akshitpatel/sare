"""Auto-generated from Oracle-validated rule: additive_identity — +(x, 0) → x"""
from sare.engine import Transform, Graph
from typing import List, Tuple

class Discovered_AdditiveIdentity(Transform):
    """Discovered rule: +(x, 0) → x"""

    def name(self) -> str:
        return 'additive_identity'

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for outer in graph.nodes:
            if outer.type == "operator" and outer.label in ('+', '+'):
                out_children = graph.outgoing(outer.id)
                for e_out in out_children:
                    inner = graph.get_node(e_out.target)
                    if inner and inner.type == "operator" and inner.label == outer.label:
                        in_children = graph.outgoing(inner.id)
                        for e_in in in_children:
                            target = graph.get_node(e_in.target)
                            if target and target.type in ("variable", "symbol", "constant"):
                                matches.append({"outer": outer.id, "inner": inner.id, "target": target.id})
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        outer_id = context["outer"]
        inner_id = context["inner"]
        target_id = context["target"]
        for e in list(g.edges):
            if e.target == outer_id:
                g.remove_edge(e.id)
                g.add_edge(e.source, target_id, e.relationship_type)
        for e in list(g.edges):
            if e.source == inner_id or e.target == inner_id:
                g.remove_edge(e.id)
        for e in list(g.edges):
            if e.source == outer_id or e.target == outer_id:
                g.remove_edge(e.id)
        g.remove_node(outer_id)
        g.remove_node(inner_id)
        return g, -3.0
