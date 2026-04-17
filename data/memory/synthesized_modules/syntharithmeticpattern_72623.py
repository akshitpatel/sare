# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-17 02:20:43
# Name: syntharithmeticpattern_72623
from typing import Tuple, List, Dict, Any, Optional
from sare.engine import Transform, Graph

from typing import Tuple, List, Dict, Any, Optional
import math
import re
from sare.engine import Transform, Graph

class SynthArithmeticPattern_72623(Transform):
    def name(self):
        return "syntharithmeticpattern_72623"

    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("+", "-", "*", "/"):
                children = graph.outgoing(n.id)
                if len(children) != 2:
                    continue
                left_edge, right_edge = children[0], children[1]
                left = graph.get_node(left_edge.target)
                right = graph.get_node(right_edge.target)
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
                            "left": left.id,
                            "right": right.id,
                            "result": result
                        })
                    except ValueError:
                        continue
                elif n.label == "*" and left.type == "variable" and right.type == "variable":
                    if left.label == right.label:
                        matches.append({
                            "op": n.id,
                            "var": left.id,
                            "same_var": True
                        })
        return matches

    def apply(self, graph, context):
        g = graph.clone()
        op_id = context["op"]
        if "same_var" in context:
            var_id = context["var"]
            var_node = g.get_node(var_id)
            exp_node_id = g.add_node("operator", "^", {})
            two_node_id = g.add_node("constant", "2", {})
            for e in list(g.edges):
                if e.target == op_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, exp_node_id, e.relationship_type)
            g.add_edge(exp_node_id, var_id, "arg")
            g.add_edge(exp_node_id, two_node_id, "arg")
            g.remove_node(op_id)
            return g, -2.0
        else:
            left_id = context["left"]
            right_id = context["right"]
            result = context["result"]
            result_str = str(result)
            if result.is_integer():
                result_str = str(int(result))
            result_node_id = g.add_node("constant", result_str, {})
            for e in list(g.edges):
                if e.target == op_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, result_node_id, e.relationship_type)
            g.remove_node(op_id)
            g.remove_node(left_id)
            g.remove_node(right_id)
            return g, -5.0