# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-16 19:02:05
# Name: syntharithmeticpattern_46262
from sare.engine import Transform, Graph

from typing import List, Tuple, Dict, Optional
import math
import re
from sare.engine import Transform, Graph

class SynthArithmeticPattern_46262(Transform):
    def name(self):
        return "syntharithmeticpattern_46262"

    def match(self, graph: Graph) -> List[Dict]:
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
                    matches.append({
                        "div_node": node.id,
                        "num_node": num_node.id,
                        "den_node": den_node.id,
                        "num_val": num_val,
                        "den_val": den_val
                    })
        return matches

    def apply(self, graph: Graph, context: Dict):
        g = graph.clone()
        div_node_id = context["div_node"]
        num_val = context["num_val"]
        den_val = context["den_val"]
        gcd_val = math.gcd(int(num_val), int(den_val))
        new_num = num_val / gcd_val
        new_den = den_val / gcd_val
        if new_den == 1:
            new_node_id = g.add_node("constant", str(int(new_num)), {})
        else:
            new_node_id = g.add_node("constant", f"{int(new_num)}/{int(new_den)}", {})
        incoming_edges = g.incoming(div_node_id)
        for edge in incoming_edges:
            g.add_edge(edge.source, new_node_id, edge.relationship_type)
        g.remove_node(div_node_id)
        g.remove_node(context["num_node"])
        g.remove_node(context["den_node"])
        return g, -2.0