# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-17 02:36:29
# Name: syntharithmeticpattern_73572
from typing import Tuple, List, Dict, Any, Optional
from sare.engine import Transform, Graph

from typing import Tuple, List, Dict, Any, Optional
import math
import re
from sare.engine import Transform, Graph

class SynthArithmeticPattern_73572(Transform):
    def name(self):
        return "syntharithmeticpattern_73572"

    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("+", "-", "*", "/"):
                children = graph.outgoing(n.id)
                if len(children) != 2:
                    continue
                left = graph.get_node(children[0].target)
                right = graph.get_node(children[1].target)
                if not left or not right:
                    continue
                if left.type == "constant" and right.type == "constant":
                    try:
                        left_val = float(left.label)
                        right_val = float(right.label)
                        if n.label == "+":
                            result = left_val + right_val
                        elif n.label == "-":
                            result = left_val - right_val
                        elif n.label == "*":
                            result = left_val * right_val
                        elif n.label == "/":
                            if right_val == 0:
                                continue
                            result = left_val / right_val
                        else:
                            continue
                        if result.is_integer():
                            result = int(result)
                        else:
                            result = self._simplify_fraction(result)
                        matches.append({
                            "op": n.id,
                            "left": left.id,
                            "right": right.id,
                            "result": result
                        })
                    except ValueError:
                        continue
        return matches

    def _simplify_fraction(self, val):
        if isinstance(val, float):
            if val.is_integer():
                return int(val)
            tolerance = 1e-10
            for denom in range(1, 101):
                numer = round(val * denom)
                if abs(val - numer / denom) < tolerance:
                    return f"{numer}/{denom}"
        return val

    def apply(self, graph, context):
        g = graph.clone()
        op_id = context["op"]
        left_id = context["left"]
        right_id = context["right"]
        result = context["result"]

        incoming_edges = g.incoming(op_id)
        parent_edge = incoming_edges[0] if incoming_edges else None

        new_node_id = None
        if isinstance(result, (int, float)):
            new_node_id = g.add_node("constant", str(result), {})
        elif isinstance(result, str) and '/' in result:
            new_node_id = g.add_node("constant", result, {})

        if new_node_id and parent_edge:
            g.add_edge(parent_edge.source, new_node_id, parent_edge.relationship_type)
            g.remove_node(op_id)
            g.remove_node(left_id)
            g.remove_node(right_id)
        elif new_node_id:
            g.remove_node(op_id)
            g.remove_node(left_id)
            g.remove_node(right_id)

        return g, -5.0