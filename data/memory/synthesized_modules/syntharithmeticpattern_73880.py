# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-17 02:42:37
# Name: syntharithmeticpattern_73880
from typing import Tuple, List, Dict, Any, Optional
from sare.engine import Transform, Graph

from typing import Tuple, List, Dict, Any, Optional
import math
import re
from sare.engine import Transform, Graph

class SynthArithmeticPattern_73880(Transform):
    def name(self):
        return "syntharithmeticpattern_73880"

    def match(self, graph: Graph) -> List[Dict[str, Any]]:
        matches = []
        for node in graph.nodes:
            if node.type == "operator" and node.label == "/":
                children = graph.outgoing(node.id)
                if len(children) != 2:
                    continue
                num_edge, den_edge = children[0], children[1]
                num_node = graph.get_node(num_edge.target)
                den_node = graph.get_node(den_edge.target)
                if num_node.type == "constant" and den_node.type == "constant":
                    try:
                        num_val = float(num_node.label)
                        den_val = float(den_node.label)
                        if den_val != 0:
                            matches.append({
                                "div_node": node.id,
                                "num_node": num_node.id,
                                "den_node": den_node.id,
                                "num_val": num_val,
                                "den_val": den_val
                            })
                    except ValueError:
                        pass
        return matches

    def apply(self, graph: Graph, context: Dict[str, Any]) -> Tuple[Graph, float]:
        g = graph.clone()
        div_id = context["div_node"]
        num_id = context["num_node"]
        den_id = context["den_node"]
        num_val = context["num_val"]
        den_val = context["den_val"]

        result_val = num_val / den_val
        if result_val.is_integer():
            result_label = str(int(result_val))
        else:
            frac = self._simplify_fraction(num_val, den_val)
            if frac[1] == 1:
                result_label = str(frac[0])
            else:
                result_label = f"{frac[0]}/{frac[1]}"

        new_const_id = g.add_node("constant", result_label, {})

        incoming_edges = g.incoming(div_id)
        for edge in incoming_edges:
            g.add_edge(edge.source, new_const_id, edge.relationship_type)

        g.remove_node(div_id)
        g.remove_node(num_id)
        g.remove_node(den_id)

        return g, -5.0

    def _simplify_fraction(self, num: float, den: float) -> Tuple[int, int]:
        num_int = int(num)
        den_int = int(den)
        gcd_val = math.gcd(num_int, den_int)
        return num_int // gcd_val, den_int // gcd_val