# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-17 05:58:15
# Name: syntharithmeticpattern_85653
from typing import Tuple, List, Dict, Any, Optional
from sare.engine import Transform, Graph

from typing import Tuple, List, Dict, Any, Optional
import math
import re
from sare.engine import Transform, Graph

class SynthArithmeticPattern_85653(Transform):
    def name(self):
        return "syntharithmeticpattern_85653"

    def match(self, graph):
        matches = []
        for node in graph.nodes:
            if node.type == "operator" and node.label == "/":
                children = graph.outgoing(node.id)
                if len(children) == 2:
                    num_edge, den_edge = children[0], children[1]
                    num_node = graph.get_node(num_edge.target)
                    den_node = graph.get_node(den_edge.target)
                    if num_node and den_node and num_node.type == "constant" and den_node.type == "constant":
                        try:
                            num_val = float(num_node.label)
                            den_val = float(den_node.label)
                            if den_val != 0 and num_val % den_val == 0:
                                matches.append({"div_node": node.id, "num": num_node.id, "den": den_node.id})
                        except ValueError:
                            pass
            elif node.type == "operator" and node.label == "*":
                children = graph.outgoing(node.id)
                if len(children) == 2:
                    left_edge, right_edge = children[0], children[1]
                    left_node = graph.get_node(left_edge.target)
                    right_node = graph.get_node(right_edge.target)
                    if left_node and right_node and left_node.type == "variable" and right_node.type == "variable":
                        if left_node.label == right_node.label:
                            matches.append({"mul_node": node.id, "var_left": left_node.id, "var_right": right_node.id})
        return matches

    def apply(self, graph, context):
        g = graph.clone()
        if "div_node" in context:
            div_id = context["div_node"]
            num_id = context["num"]
            den_id = context["den"]
            num_node = g.get_node(num_id)
            den_node = g.get_node(den_id)
            num_val = float(num_node.label)
            den_val = float(den_node.label)
            result_val = num_val / den_val
            result_id = g.add_node("constant", str(int(result_val)), {})
            for edge in list(g.edges):
                if edge.target == div_id:
                    g.remove_edge(edge.id)
                    g.add_edge(edge.source, result_id, edge.relationship_type)
            g.remove_node(div_id)
            g.remove_node(num_id)
            g.remove_node(den_id)
            return g, -5.0
        elif "mul_node" in context:
            mul_id = context["mul_node"]
            var_left_id = context["var_left"]
            var_right_id = context["var_right"]
            var_node = g.get_node(var_left_id)
            exp_id = g.add_node("operator", "^", {})
            two_id = g.add_node("constant", "2", {})
            g.add_edge(exp_id, var_left_id, "arg")
            g.add_edge(exp_id, two_id, "arg")
            for edge in list(g.edges):
                if edge.target == mul_id:
                    g.remove_edge(edge.id)
                    g.add_edge(edge.source, exp_id, edge.relationship_type)
            g.remove_node(mul_id)
            g.remove_node(var_right_id)
            return g, -2.0
        return graph, 0.0