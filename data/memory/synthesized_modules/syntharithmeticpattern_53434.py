# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-16 21:00:58
# Name: syntharithmeticpattern_53434
from sare.engine import Transform, Graph

from typing import List, Tuple, Dict, Optional
import math
import re
from sare.engine import Transform, Graph

class SynthArithmeticPattern_53434(Transform):
    def name(self):
        return "syntharithmeticpattern_53434"

    def match(self, graph):
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
                elif n.label == "*" and left_id == right_id:
                    matches.append({
                        "op": n.id,
                        "left": left_id,
                        "right": right_id,
                        "square": True
                    })
        return matches

    def apply(self, graph, context):
        g = graph.clone()
        op_id = context["op"]
        if "square" in context:
            left_id = context["left"]
            left = g.get_node(left_id)
            if left.type == "variable":
                exp_node = g.add_node("operator", "^", {})
                g.add_edge(exp_node, left_id, "base")
                const_node = g.add_node("constant", "2", {})
                g.add_edge(exp_node, const_node, "exponent")
                for e in g.incoming(op_id):
                    g.remove_edge(e.id)
                    g.add_edge(e.source, exp_node, e.relationship_type)
                g.remove_node(op_id)
                return g, -2.0
        else:
            left_id = context["left"]
            right_id = context["right"]
            result_val = context["result"]
            result_str = str(result_val)
            if result_val.is_integer():
                result_str = str(int(result_val))
            result_node = g.add_node("constant", result_str, {})
            for e in g.incoming(op_id):
                g.remove_edge(e.id)
                g.add_edge(e.source, result_node, e.relationship_type)
            g.remove_node(op_id)
            g.remove_node(left_id)
            g.remove_node(right_id)
            return g, -5.0