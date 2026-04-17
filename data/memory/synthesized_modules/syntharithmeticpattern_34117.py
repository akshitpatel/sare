# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-16 15:39:34
# Name: syntharithmeticpattern_34117
from sare.engine import Transform, Graph

from typing import List, Tuple, Dict, Optional
import math
import re
from sare.engine import Transform, Graph

class SynthArithmeticPattern_34117(Transform):
    def name(self) -> str:
        return "syntharithmeticpattern_34117"

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
                            "frac": n.id,
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

        if new_den_val == 1:
            new_const_id = g.add_node("constant", str(int(new_num_val)), {})
            for e in list(g.edges):
                if e.target == frac_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, new_const_id, e.relationship_type)
            g.remove_node(frac_id)
            g.remove_node(num_id)
            g.remove_node(den_id)
        else:
            new_num_id = g.add_node("constant", str(int(new_num_val)), {})
            new_den_id = g.add_node("constant", str(int(new_den_val)), {})
            for e in list(g.edges):
                if e.target == frac_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, frac_id, e.relationship_type)
            for e in list(g.outgoing(frac_id)):
                g.remove_edge(e.id)
            g.add_edge(frac_id, new_num_id, "numerator")
            g.add_edge(frac_id, new_den_id, "denominator")
            g.remove_node(num_id)
            g.remove_node(den_id)

        return g, -2.0