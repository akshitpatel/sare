# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-16 15:30:39
# Name: syntharithmeticpattern_33581
from sare.engine import Transform, Graph

from sare.engine import Transform, Graph
from typing import List, Tuple, Dict, Optional
import math
import re

class SynthArithmeticPattern_33581(Transform):
    def name(self) -> str:
        return "syntharithmeticpattern_33581"

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
                            "gcd": gcd_val,
                            "num_val": num_val,
                            "den_val": den_val
                        })
        return matches

    def apply(self, graph: Graph, context: Dict) -> Tuple[Graph, float]:
        g = graph.clone()
        frac_id = context["frac"]
        num_id = context["num"]
        den_id = context["den"]
        gcd_val = context["gcd"]
        num_val = context["num_val"]
        den_val = context["den_val"]

        new_num_val = num_val / gcd_val
        new_den_val = den_val / gcd_val

        new_num_id = g.add_node("constant", str(int(new_num_val) if new_num_val.is_integer() else new_num_val), {})
        new_den_id = g.add_node("constant", str(int(new_den_val) if new_den_val.is_integer() else new_den_val), {})

        for edge in list(g.edges):
            if edge.source == frac_id:
                if edge.target == num_id:
                    g.remove_edge(edge.id)
                    g.add_edge(frac_id, new_num_id, edge.relationship_type)
                elif edge.target == den_id:
                    g.remove_edge(edge.id)
                    g.add_edge(frac_id, new_den_id, edge.relationship_type)

        g.remove_node(num_id)
        g.remove_node(den_id)

        if new_den_val == 1:
            for edge in list(g.edges):
                if edge.target == frac_id:
                    g.remove_edge(edge.id)
                    g.add_edge(edge.source, new_num_id, edge.relationship_type)
            g.remove_node(frac_id)
            g.remove_node(new_den_id)
            return g, -5.0

        return g, -2.0