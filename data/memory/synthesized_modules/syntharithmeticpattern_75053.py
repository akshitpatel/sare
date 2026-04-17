# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-17 03:03:00
# Name: syntharithmeticpattern_75053
from typing import Tuple, List, Dict, Any, Optional
from sare.engine import Transform, Graph

from typing import Tuple, List, Dict, Any, Optional
import math
import re
from sare.engine import Transform, Graph


class SynthArithmeticPattern_75053(Transform):
    def name(self) -> str:
        return "syntharithmeticpattern_75053"

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
                if num_node.type != "constant" or den_node.type != "constant":
                    continue
                try:
                    num_val = float(num_node.label)
                    den_val = float(den_node.label)
                except ValueError:
                    continue
                if den_val == 0:
                    continue
                if num_val.is_integer() and den_val.is_integer():
                    gcd_val = math.gcd(int(num_val), int(den_val))
                    if gcd_val > 1:
                        matches.append({
                            "div_node": node.id,
                            "num_node": num_node.id,
                            "den_node": den_node.id,
                            "num_val": num_val,
                            "den_val": den_val,
                            "gcd": gcd_val
                        })
        return matches

    def apply(self, graph: Graph, context: Dict[str, Any]) -> Tuple[Graph, float]:
        g = graph.clone()
        div_id = context["div_node"]
        num_id = context["num_node"]
        den_id = context["den_node"]
        num_val = context["num_val"]
        den_val = context["den_val"]
        gcd_val = context["gcd"]

        new_num_val = num_val / gcd_val
        new_den_val = den_val / gcd_val

        new_num_id = g.add_node("constant", str(new_num_val), {})
        new_den_id = g.add_node("constant", str(new_den_val), {})

        incoming_edges = g.incoming(div_id)
        for edge in incoming_edges:
            g.add_edge(edge.source, new_num_id, edge.relationship_type)

        if new_den_val != 1:
            new_div_id = g.add_node("operator", "/", {})
            g.add_edge(new_div_id, new_num_id, "operand")
            g.add_edge(new_div_id, new_den_id, "operand")
            for edge in incoming_edges:
                g.remove_edge(edge.id)
                g.add_edge(edge.source, new_div_id, edge.relationship_type)
            g.remove_node(div_id)
            g.remove_node(num_id)
            g.remove_node(den_id)
        else:
            for edge in incoming_edges:
                g.remove_edge(edge.id)
                g.add_edge(edge.source, new_num_id, edge.relationship_type)
            g.remove_node(div_id)
            g.remove_node(num_id)
            g.remove_node(den_id)

        return g, -2.0