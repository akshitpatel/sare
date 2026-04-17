# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-16 17:36:07
# Name: syntharithmeticpattern_41066
from sare.engine import Transform, Graph

from typing import List, Tuple, Dict, Optional
import math
import re
from sare.engine import Transform, Graph

class SynthArithmeticPattern_41066(Transform):
    def name(self) -> str:
        return "syntharithmeticpattern_41066"

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
                            "div_node": node.id,
                            "num_node": num_node.id,
                            "den_node": den_node.id,
                            "num_val": num_val,
                            "den_val": den_val,
                            "gcd": gcd_val
                        })
        return matches

    def apply(self, graph: Graph, context: Dict) -> Tuple[Graph, float]:
        g = graph.clone()
        div_node = context["div_node"]
        num_node = context["num_node"]
        den_node = context["den_node"]
        num_val = context["num_val"]
        den_val = context["den_val"]
        gcd_val = context["gcd"]

        new_num_val = num_val / gcd_val
        new_den_val = den_val / gcd_val

        new_num_id = g.add_node("constant", str(int(new_num_val)), {})
        new_den_id = g.add_node("constant", str(int(new_den_val)), {})

        for edge in list(g.edges):
            if edge.source == div_node:
                if edge.target == num_node:
                    g.remove_edge(edge.id)
                    g.add_edge(div_node, new_num_id, edge.relationship_type)
                elif edge.target == den_node:
                    g.remove_edge(edge.id)
                    g.add_edge(div_node, new_den_id, edge.relationship_type)

        g.remove_node(num_node)
        g.remove_node(den_node)

        if new_den_val == 1:
            parent_edges = g.incoming(div_node)
            if parent_edges:
                parent_edge = parent_edges[0]
                g.remove_edge(parent_edge.id)
                g.remove_node(div_node)
                g.add_edge(parent_edge.source, new_num_id, parent_edge.relationship_type)
            else:
                g.remove_node(div_node)
                return g, -2.0

        return g, -1.0