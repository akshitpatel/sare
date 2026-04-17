# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-17 06:19:27
# Name: syntharithmeticpattern_86900
from typing import Tuple, List, Dict, Any, Optional
from sare.engine import Transform, Graph

from typing import Tuple, List, Dict, Any, Optional
import math
import re
from sare.engine import Transform, Graph

class SynthArithmeticPattern_86900(Transform):
    def name(self):
        return "syntharithmeticpattern_86900"

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
                    matches.append({
                        "frac": node.id,
                        "num": num_node.id,
                        "den": den_node.id,
                        "num_val": num_val,
                        "den_val": den_val
                    })
        return matches

    def apply(self, graph: Graph, context: Dict[str, Any]) -> Tuple[Graph, float]:
        g = graph.clone()
        frac_id = context["frac"]
        num_val = context["num_val"]
        den_val = context["den_val"]
        gcd_val = math.gcd(int(num_val), int(den_val))
        if gcd_val == 1:
            return g, 0.0
        new_num = num_val / gcd_val
        new_den = den_val / gcd_val
        if new_den == 1:
            const_id = g.add_node("constant", str(int(new_num)), {})
            for e in list(g.edges):
                if e.target == frac_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, const_id, e.relationship_type)
            g.remove_node(frac_id)
            g.remove_node(context["num"])
            g.remove_node(context["den"])
            return g, -5.0
        else:
            new_num_id = g.add_node("constant", str(int(new_num)), {})
            new_den_id = g.add_node("constant", str(int(new_den)), {})
            new_frac_id = g.add_node("operator", "/", {})
            g.add_edge(new_frac_id, new_num_id, "operand")
            g.add_edge(new_frac_id, new_den_id, "operand")
            for e in list(g.edges):
                if e.target == frac_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, new_frac_id, e.relationship_type)
            g.remove_node(frac_id)
            g.remove_node(context["num"])
            g.remove_node(context["den"])
            return g, -3.0