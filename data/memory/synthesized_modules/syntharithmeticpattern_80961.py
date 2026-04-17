# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-17 04:39:51
# Name: syntharithmeticpattern_80961
from typing import Tuple, List, Dict, Any, Optional
from sare.engine import Transform, Graph

from typing import Tuple, List, Dict, Any, Optional
from sare.engine import Transform, Graph
import math
import re

class SynthArithmeticPattern_80961(Transform):
    def name(self): return "syntharithmeticpattern_80961"

    def match(self, graph) -> List[Dict[str, Any]]:
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
                            "div_node": n.id,
                            "num_node": num_node.id,
                            "den_node": den_node.id,
                            "num_val": num_val,
                            "den_val": den_val,
                            "gcd": gcd_val
                        })
        return matches

    def apply(self, graph: Graph, context: Dict[str, Any]) -> Tuple[Graph, float]:
        g = graph.clone()
        div_node = context["div_node"]
        num_node = context["num_node"]
        den_node = context["den_node"]
        num_val = context["num_val"]
        den_val = context["den_val"]
        gcd_val = context["gcd"]

        new_num_val = num_val / gcd_val
        new_den_val = den_val / gcd_val

        new_num_id = g.add_node("constant", str(new_num_val), {})
        new_den_id = g.add_node("constant", str(new_den_val), {})

        for e in list(g.edges):
            if e.source == div_node:
                if e.target == num_node:
                    g.remove_edge(e.id)
                    g.add_edge(div_node, new_num_id, e.relationship_type)
                elif e.target == den_node:
                    g.remove_edge(e.id)
                    g.add_edge(div_node, new_den_id, e.relationship_type)

        g.remove_node(num_node)
        g.remove_node(den_node)

        return g, -2.0