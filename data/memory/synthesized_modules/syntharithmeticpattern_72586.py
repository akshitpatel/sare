# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-17 02:20:35
# Name: syntharithmeticpattern_72586
from typing import Tuple, List, Dict, Any, Optional
from sare.engine import Transform, Graph

from typing import Tuple, List, Dict, Any, Optional
import math
import re
from sare.engine import Transform, Graph

class SynthArithmeticPattern_72586(Transform):
    def name(self):
        return "syntharithmeticpattern_72586"

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
                            if den_val != 0:
                                matches.append({
                                    "div_node": n.id,
                                    "num": num_node.id,
                                    "den": den_node.id,
                                    "num_val": num_val,
                                    "den_val": den_val
                                })
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
                            matches.append({
                                "mul_node": n.id,
                                "var_left": left_node.id,
                                "var_right": right_node.id,
                                "var_label": left_node.label
                            })
        return matches

    def apply(self, graph, context):
        g = graph.clone()
        if "div_node" in context:
            div_id = context["div_node"]
            num_val = context["num_val"]
            den_val = context["den_val"]
            result_val = num_val / den_val
            if result_val.is_integer():
                result_val = int(result_val)
            else:
                result_val = round(result_val, 6)
            result_node_id = g.add_node("constant", str(result_val), {})
            for e in list(g.edges):
                if e.target == div_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, result_node_id, e.relationship_type)
            g.remove_node(div_id)
            g.remove_node(context["num"])
            g.remove_node(context["den"])
            return g, -5.0
        elif "mul_node" in context:
            mul_id = context["mul_node"]
            var_label = context["var_label"]
            pow_node_id = g.add_node("operator", "^", {})
            var_node_id = g.add_node("variable", var_label, {})
            exp_node_id = g.add_node("constant", "2", {})
            g.add_edge(pow_node_id, var_node_id, "operand")
            g.add_edge(pow_node_id, exp_node_id, "operand")
            for e in list(g.edges):
                if e.target == mul_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, pow_node_id, e.relationship_type)
            g.remove_node(mul_id)
            g.remove_node(context["var_left"])
            g.remove_node(context["var_right"])
            return g, -2.0
        return graph, 0.0