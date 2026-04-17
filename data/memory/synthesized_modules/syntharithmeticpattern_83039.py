# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-17 05:15:14
# Name: syntharithmeticpattern_83039
from typing import Tuple, List, Dict, Any, Optional
from sare.engine import Transform, Graph

from typing import Tuple, List, Dict, Any, Optional
import math
import re
from sare.engine import Transform, Graph

class SynthArithmeticPattern_83039(Transform):
    def name(self):
        return "syntharithmeticpattern_83039"

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
                if num_val == den_val:
                    matches.append({"frac": node.id, "num": num_node.id, "den": den_node.id})
                elif num_val % den_val == 0:
                    matches.append({"frac": node.id, "num": num_node.id, "den": den_node.id})
                elif den_val % num_val == 0:
                    matches.append({"frac": node.id, "num": num_node.id, "den": den_node.id})
        return matches

    def apply(self, graph: Graph, context: Dict[str, Any]) -> Tuple[Graph, float]:
        g = graph.clone()
        frac_id = context["frac"]
        num_id = context["num"]
        den_id = context["den"]
        num_node = g.get_node(num_id)
        den_node = g.get_node(den_id)
        num_val = float(num_node.label)
        den_val = float(den_node.label)
        if num_val == den_val:
            new_node_id = g.add_node("constant", "1", {})
        elif num_val % den_val == 0:
            new_val = num_val / den_val
            if new_val.is_integer():
                new_label = str(int(new_val))
            else:
                new_label = str(new_val)
            new_node_id = g.add_node("constant", new_label, {})
        elif den_val % num_val == 0:
            new_val = den_val / num_val
            if new_val.is_integer():
                new_label = str(int(new_val))
            else:
                new_label = str(new_val)
            new_node_id = g.add_node("constant", f"1/{new_label}", {})
        else:
            return graph, 0.0
        for edge in list(g.edges):
            if edge.target == frac_id:
                g.remove_edge(edge.id)
                g.add_edge(edge.source, new_node_id, edge.relationship_type)
        g.remove_node(frac_id)
        g.remove_node(num_id)
        g.remove_node(den_id)
        return g, -2.0