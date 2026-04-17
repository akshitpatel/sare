"""Auto-generated from Oracle-validated rule: discovered_-_20_identity — -(x, 20) → x"""
from sare.engine import Transform, Graph
from typing import List, Tuple

class Discovered_Discovered20Identity(Transform):
    """Discovered rule: -(x, 20) → x"""

    CONST_VAL = '20'
    OP_LABELS = ['-']

    def name(self) -> str:
        return 'discovered_-_20_identity'

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in self.OP_LABELS:
                children = graph.outgoing(n.id)
                for e in children:
                    child = graph.get_node(e.target)
                    if child and child.type == "constant" and child.label == self.CONST_VAL:
                        other = None
                        for e2 in children:
                            if e2.id != e.id:
                                other = graph.get_node(e2.target)
                        if other:
                            matches.append({"op": n.id, "const": child.id, "other": other.id})
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        op_id = context["op"]
        const_id = context["const"]
        other_id = context["other"]
        for e in list(g.edges):
            if e.target == op_id:
                g.remove_edge(e.id)
                g.add_edge(e.source, other_id, e.relationship_type)
        for e in list(g.edges):
            if e.source == op_id or e.target == op_id:
                g.remove_edge(e.id)
        g.remove_node(op_id)
        g.remove_node(const_id)
        return g, -3.0
