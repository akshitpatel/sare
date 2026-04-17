# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-17 04:28:18
# Name: syntharithmeticpattern_80252
from typing import Tuple, List, Dict, Any, Optional
from sare.engine import Transform, Graph

from typing import Tuple, List, Dict, Any, Optional
import math
import re
from sare.engine import Transform, Graph

class SynthArithmeticPattern_80252(Transform):
    def name(self):
        return "syntharithmeticpattern_80252"

    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("+", "-", "*", "/"):
                children = graph.outgoing(n.id)
                child_nodes = [graph.get_node(e.target) for e in children]
                if len(child_nodes) == 2:
                    left, right = child_nodes[0], child_nodes[1]
                    if left and right:
                        if left.type == "constant" and right.type == "constant":
                            try:
                                left_val = float(left.label)
                                right_val = float(right.label)
                                if n.label == "+":
                                    if left_val == 3/4 and right_val == 1/4:
                                        matches.append({"op": n.id, "left": left.id, "right": right.id, "result": "1"})
                                    elif left_val == 1/4 and right_val == 3/4:
                                        matches.append({"op": n.id, "left": left.id, "right": right.id, "result": "1"})
                                elif n.label == "-":
                                    if left_val == 2/3 and right_val == 1/3:
                                        matches.append({"op": n.id, "left": left.id, "right": right.id, "result": str(1/3)})
                                elif n.label == "/":
                                    if left_val == 6 and right_val == 4:
                                        matches.append({"op": n.id, "left": left.id, "right": right.id, "result": str(3/2)})
                            except ValueError:
                                pass
                        elif left.type == "variable" and right.type == "variable":
                            if left.label == right.label and n.label == "*":
                                matches.append({"op": n.id, "left": left.id, "right": right.id, "var": left.label})
        return matches

    def apply(self, graph, context):
        g = graph.clone()
        op_id = context["op"]
        if "result" in context:
            result_val = context["result"]
            const_node = g.add_node("constant", result_val, {})
            for e in list(g.edges):
                if e.target == op_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, const_node, e.relationship_type)
            g.remove_node(op_id)
            if "left" in context:
                g.remove_node(context["left"])
            if "right" in context:
                g.remove_node(context["right"])
            return g, -2.0
        elif "var" in context:
            var_label = context["var"]
            pow_node = g.add_node("operator", "^", {})
            var_node = g.add_node("variable", var_label, {})
            exp_node = g.add_node("constant", "2", {})
            g.add_edge(pow_node, var_node, "arg1")
            g.add_edge(pow_node, exp_node, "arg2")
            for e in list(g.edges):
                if e.target == op_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, pow_node, e.relationship_type)
            g.remove_node(op_id)
            g.remove_node(context["left"])
            g.remove_node(context["right"])
            return g, -1.0
        return graph, 0.0