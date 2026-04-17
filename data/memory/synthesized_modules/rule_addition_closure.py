"""Auto-generated from Oracle-validated rule: addition_closure — +(c1, c2) → c3"""
from sare.engine import Transform, Graph
from typing import List, Tuple

class Discovered_AdditionClosure(Transform):
    """Discovered rule: +(c1, c2) → c3"""

    CONST_VAL = '0'
    OP_LABELS = ['+']

    def name(self) -> str:
        return 'addition_closure'

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in self.OP_LABELS:
                children = graph.outgoing(n.id)
                const_node = None
                var_node = None
                for e in children:
                    child = graph.get_node(e.target)
                    if child and child.type == "constant" and child.label == self.CONST_VAL:
                        const_node = child
                    elif child and child.type in ("variable", "symbol"):
                        var_node = child
                if const_node and var_node:
                    matches.append({"op": n.id, "const": const_node.id, "var": var_node.id})
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        op_id = context["op"]
        const_id = context["const"]
        var_id = context["var"]
        for e in list(g.edges):
            if e.target == op_id:
                g.remove_edge(e.id)
                g.add_edge(e.source, const_id, e.relationship_type)
        for e in list(g.edges):
            if e.source == op_id or e.target == op_id:
                g.remove_edge(e.id)
        g.remove_node(op_id)
        g.remove_node(var_id)
        return g, -3.0
