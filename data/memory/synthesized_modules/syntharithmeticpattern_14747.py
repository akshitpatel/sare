# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-16 10:16:35
# Name: syntharithmeticpattern_14747
from sare.engine import Transform, Graph

from sare.engine import Transform, Graph
from typing import List, Tuple, Dict, Optional
import math
import re

class SynthArithmeticPattern_14747(Transform):
    def name(self):
        return "syntharithmeticpattern_14747"

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
                    gcd_val = math.gcd(int(num_val), int(den_val))
                    if gcd_val > 1:
                        matches.append({
                            "frac": node.id,
                            "num": num_node.id,
                            "den": den_node.id,
                            "gcd": gcd_val
                        })
        return matches

    def apply(self, graph: Graph, context: Dict) -> Tuple[Graph, float]:
        g = graph.clone()
        frac_id = context["frac"]
        num_id = context["num"]
        den_id = context["den"]
        gcd_val = context["gcd"]

        num_node = g.get_node(num_id)
        den_node = g.get_node(den_id)
        num_val = float(num_node.label)
        den_val = float(den_node.label)

        new_num_val = num_val / gcd_val
        new_den_val = den_val / gcd_val

        new_num_id = g.add_node("constant", str(int(new_num_val)), {})
        new_den_id = g.add_node("constant", str(int(new_den_val)), {})

        new_frac_id = g.add_node("operator", "/", {})
        g.add_edge(new_frac_id, new_num_id, "operand")
        g.add_edge(new_frac_id, new_den_id, "operand")

        for edge in list(g.edges):
            if edge.target == frac_id:
                g.remove_edge(edge.id)
                g.add_edge(edge.source, new_frac_id, edge.relationship_type)

        g.remove_node(frac_id)
        g.remove_node(num_id)
        g.remove_node(den_id)

        return g, -2.0