# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-17 06:06:39
# Name: syntharithmeticpattern_86094
from typing import Tuple, List, Dict, Any, Optional
from sare.engine import Transform, Graph

from typing import Tuple, List, Dict, Any, Optional
import math
import re
from sare.engine import Transform, Graph


class SynthArithmeticPattern_86094(Transform):
    def name(self):
        return "syntharithmeticpattern_86094"

    def match(self, graph):
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
                if num_val == den_val:
                    matches.append({"div": n.id, "num": num_node.id, "den": den_node.id})
                elif num_val % den_val == 0:
                    matches.append({"div": n.id, "num": num_node.id, "den": den_node.id})
                elif den_val % num_val == 0:
                    matches.append({"div": n.id, "num": num_node.id, "den": den_node.id})
        return matches

    def apply(self, graph, context):
        g = graph.clone()
        div_id = context["div"]
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
            new_node_id = g.add_node("constant", new_label, {})
        else:
            return graph, 0.0
        for e in list(g.edges):
            if e.target == div_id:
                g.remove_edge(e.id)
                g.add_edge(e.source, new_node_id, e.relationship_type)
        g.remove_node(div_id)
        g.remove_node(num_id)
        g.remove_node(den_id)
        return g, -5.0