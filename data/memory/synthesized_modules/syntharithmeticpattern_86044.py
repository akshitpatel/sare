# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-17 06:05:02
# Name: syntharithmeticpattern_86044
from typing import Tuple, List, Dict, Any, Optional
from sare.engine import Transform, Graph

from typing import Tuple, List, Dict, Any, Optional
import math
import re
from sare.engine import Transform, Graph

class SynthArithmeticPattern_86044(Transform):
    def name(self):
        return "syntharithmeticpattern_86044"

    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label == "/":
                children = graph.outgoing(n.id)
                if len(children) == 2:
                    num_edge, den_edge = children[0], children[1]
                    num_node = graph.get_node(num_edge.target)
                    den_node = graph.get_node(den_edge.target)
                    if num_node and den_node and num_node.type == "constant" and den_node.type == "constant":
                        try:
                            num_val = float(num_node.label)
                            den_val = float(den_node.label)
                            if den_val != 0 and num_val % den_val == 0:
                                matches.append({"div": n.id, "num": num_node.id, "den": den_node.id})
                        except ValueError:
                            pass
            elif n.type == "operator" and n.label == "*":
                children = graph.outgoing(n.id)
                if len(children) == 2:
                    left_edge, right_edge = children[0], children[1]
                    left_node = graph.get_node(left_edge.target)
                    right_node = graph.get_node(right_edge.target)
                    if left_node and right_node and left_node.type == "variable" and right_node.type == "variable":
                        if left_node.label == right_node.label:
                            matches.append({"mul": n.id, "var": left_node.id})
        return matches

    def apply(self, graph, context):
        g = graph.clone()
        if "div" in context:
            div_id = context["div"]
            num_id = context["num"]
            den_id = context["den"]
            num_node = g.get_node(num_id)
            den_node = g.get_node(den_id)
            num_val = float(num_node.label)
            den_val = float(den_node.label)
            result_val = int(num_val / den_val)
            new_const_id = g.add_node("constant", str(result_val), {})
            for e in list(g.edges):
                if e.target == div_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, new_const_id, e.relationship_type)
            g.remove_node(div_id)
            g.remove_node(num_id)
            g.remove_node(den_id)
            return g, -5.0
        elif "mul" in context:
            mul_id = context["mul"]
            var_id = context["var"]
            var_node = g.get_node(var_id)
            new_pow_id = g.add_node("operator", "^", {})
            g.add_edge(new_pow_id, var_id, "operand")
            exp_const_id = g.add_node("constant", "2", {})
            g.add_edge(new_pow_id, exp_const_id, "operand")
            for e in list(g.edges):
                if e.target == mul_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, new_pow_id, e.relationship_type)
            g.remove_node(mul_id)
            return g, -2.0
        return g, 0.0