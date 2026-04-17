# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-16 12:41:28
# Name: syntharithmeticpattern_23403
from sare.engine import Transform, Graph

from typing import List, Tuple, Dict, Optional
import math
import re
from sare.engine import Transform, Graph

class SynthArithmeticPattern_23403(Transform):
    def name(self):
        return "syntharithmeticpattern_23403"

    def match(self, graph: Graph) -> List[Dict]:
        matches = []
        for node in graph.nodes:
            if node.type == "operator" and node.label == "/":
                outgoing = graph.outgoing(node.id)
                if len(outgoing) != 2:
                    continue
                num_edge, den_edge = outgoing[0], outgoing[1]
                num_node = graph.get_node(num_edge.target)
                den_node = graph.get_node(den_edge.target)
                if num_node.type == "constant" and den_node.type == "constant":
                    try:
                        num_val = float(num_node.label)
                        den_val = float(den_node.label)
                        if den_val != 0:
                            matches.append({
                                "div_node": node.id,
                                "num_node": num_node.id,
                                "den_node": den_node.id,
                                "num_val": num_val,
                                "den_val": den_val
                            })
                    except ValueError:
                        pass
            elif node.type == "operator" and node.label == "*":
                outgoing = graph.outgoing(node.id)
                if len(outgoing) == 2:
                    left_edge, right_edge = outgoing[0], outgoing[1]
                    left_node = graph.get_node(left_edge.target)
                    right_node = graph.get_node(right_edge.target)
                    if left_node.type == "variable" and right_node.type == "variable":
                        if left_node.label == right_node.label:
                            matches.append({
                                "mul_node": node.id,
                                "var_node_left": left_node.id,
                                "var_node_right": right_node.id,
                                "var_label": left_node.label
                            })
        return matches

    def apply(self, graph: Graph, context: Dict):
        g = graph.clone()
        if "div_node" in context:
            div_id = context["div_node"]
            num_val = context["num_val"]
            den_val = context["den_val"]
            result_val = num_val / den_val
            result_id = g.add_node("constant", str(result_val), {})
            for edge in list(g.edges):
                if edge.target == div_id:
                    g.remove_edge(edge.id)
                    g.add_edge(edge.source, result_id, edge.relationship_type)
            g.remove_node(div_id)
            g.remove_node(context["num_node"])
            g.remove_node(context["den_node"])
            return g, -5.0
        elif "mul_node" in context:
            mul_id = context["mul_node"]
            var_label = context["var_label"]
            pow_id = g.add_node("operator", "^", {})
            var_id = g.add_node("variable", var_label, {})
            const_id = g.add_node("constant", "2", {})
            g.add_edge(pow_id, var_id, "left")
            g.add_edge(pow_id, const_id, "right")
            for edge in list(g.edges):
                if edge.target == mul_id:
                    g.remove_edge(edge.id)
                    g.add_edge(edge.source, pow_id, edge.relationship_type)
            g.remove_node(mul_id)
            g.remove_node(context["var_node_left"])
            g.remove_node(context["var_node_right"])
            return g, -2.0
        return g, 0.0