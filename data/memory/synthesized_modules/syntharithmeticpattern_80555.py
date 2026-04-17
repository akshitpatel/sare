# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-17 04:33:20
# Name: syntharithmeticpattern_80555
from typing import Tuple, List, Dict, Any, Optional
from sare.engine import Transform, Graph

from typing import Tuple, List, Dict, Any, Optional
import math
import re
from sare.engine import Transform, Graph

class SynthArithmeticPattern_80555(Transform):
    def name(self):
        return "syntharithmeticpattern_80555"

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
                            matches.append({"op": n.id, "left": left.id, "right": right.id, "type": "add"})
                        elif n.label == "-":
                            matches.append({"op": n.id, "left": left.id, "right": right.id, "type": "sub"})
                        elif n.label == "*":
                            matches.append({"op": n.id, "left": left.id, "right": right.id, "type": "mul"})
                        elif n.label == "/":
                            matches.append({"op": n.id, "left": left.id, "right": right.id, "type": "div"})
                    except ValueError:
                        pass
                elif n.label == "*" and left.type == "variable" and right.type == "variable":
                    if left.label == right.label:
                        matches.append({"op": n.id, "var": left.id, "type": "square"})
        return matches

    def apply(self, graph, context):
        g = graph.clone()
        if context["type"] in ("add", "sub", "mul", "div"):
            op_id = context["op"]
            left_id = context["left"]
            right_id = context["right"]
            left_node = g.get_node(left_id)
            right_node = g.get_node(right_id)
            left_val = float(left_node.label)
            right_val = float(right_node.label)
            result = None
            if context["type"] == "add":
                result = left_val + right_val
            elif context["type"] == "sub":
                result = left_val - right_val
            elif context["type"] == "mul":
                result = left_val * right_val
            elif context["type"] == "div":
                if right_val == 0:
                    return graph, 0.0
                result = left_val / right_val
            new_const_id = g.add_node("constant", str(result), {})
            for e in list(g.edges):
                if e.target == op_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, new_const_id, e.relationship_type)
            g.remove_node(op_id)
            g.remove_node(left_id)
            g.remove_node(right_id)
            return g, -5.0
        elif context["type"] == "square":
            op_id = context["op"]
            var_id = context["var"]
            var_node = g.get_node(var_id)
            new_pow_id = g.add_node("operator", "^", {})
            new_const_id = g.add_node("constant", "2", {})
            for e in list(g.edges):
                if e.target == op_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, new_pow_id, e.relationship_type)
            g.add_edge(new_pow_id, var_id, "operand")
            g.add_edge