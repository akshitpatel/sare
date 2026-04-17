# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-16 20:21:53
# Name: syntharithmeticpattern_51048
from sare.engine import Transform, Graph

from typing import List, Tuple, Dict, Optional
import math
import re
from sare.engine import Transform, Graph

class SynthArithmeticPattern_51048(Transform):
    def name(self) -> str:
        return "syntharithmeticpattern_51048"

    def match(self, graph: Graph) -> List[Dict]:
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("+", "-", "*", "/"):
                children = graph.outgoing(n.id)
                if len(children) != 2:
                    continue
                left_id = children[0].target
                right_id = children[1].target
                left = graph.get_node(left_id)
                right = graph.get_node(right_id)
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
                        matches.append({
                            "op": n.id,
                            "left": left_id,
                            "right": right_id,
                            "result": result
                        })
                    except ValueError:
                        continue
        return matches

    def apply(self, graph: Graph, context: Dict) -> Tuple[Graph, float]:
        g = graph.clone()
        op_id = context["op"]
        left_id = context["left"]
        right_id = context["right"]
        result = context["result"]
        result_str = str(result) if not result.is_integer() else str(int(result))
        new_const_id = g.add_node("constant", result_str, {})
        incoming_edges = g.incoming(op_id)
        for e in incoming_edges:
            g.add_edge(e.source, new_const_id, e.relationship_type)
        g.remove_node(op_id)
        g.remove_node(left_id)
        g.remove_node(right_id)
        return g, -5.0