# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-16 19:03:19
# Name: syntharithmeticpattern_46354
from sare.engine import Transform, Graph

from typing import List, Tuple, Dict, Optional
import math
import re
from sare.engine import Transform, Graph

class SynthArithmeticPattern_46354(Transform):
    def name(self) -> str:
        return "syntharithmeticpattern_46354"

    def match(self, graph: Graph) -> List[Dict]:
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label == "/":
                children = graph.outgoing(n.id)
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
                            "div": n.id,
                            "num": num_node.id,
                            "den": den_node.id,
                            "num_val": num_val,
                            "den_val": den_val,
                            "gcd": gcd_val
                        })
        return matches

    def apply(self, graph: Graph, context: Dict) -> Tuple[Graph, float]:
        g = graph.clone()
        div_id = context["div"]
        num_id = context["num"]
        den_id = context["den"]
        num_val = context["num_val"]
        den_val = context["den_val"]
        gcd_val = context["gcd"]

        new_num_val = num_val / gcd_val
        new_den_val = den_val / gcd_val

        new_num_id = g.add_node("constant", str(int(new_num_val)) if new_num_val.is_integer() else str(new_num_val), {})
        new_den_id = g.add_node("constant", str(int(new_den_val)) if new_den_val.is_integer() else str(new_den_val), {})

        for e in list(g.edges):
            if e.target == div_id:
                if e.source is not None:
                    new_div_id = g.add_node("operator", "/", {})
                    g.add_edge(e.source, new_div_id, e.relationship_type)
                    g.add_edge(new_div_id, new_num_id, "operand")
                    g.add_edge(new_div_id, new_den_id, "operand")
                    g.remove_node(div_id)
                    g.remove_node(num_id)
                    g.remove_node(den_id)
                    break

        return g, -2.0