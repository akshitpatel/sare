# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-16 23:27:25
# Name: syntharithmeticpattern_62194
from sare.engine import Transform, Graph

from sare.engine import Transform, Graph
from typing import List, Tuple, Dict, Optional
import math
import re

class SynthArithmeticPattern_62194(Transform):
    def name(self) -> str:
        return "syntharithmeticpattern_62194"

    def match(self, graph: Graph) -> List[Dict]:
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
                                matches.append({
                                    "div_node": node.id,
                                    "num_node": num_node.id,
                                    "den_node": den_node.id,
                                    "value": num_val / den_val
                                })
                        except ValueError:
                            continue
            elif node.type == "operator" and node.label == "*":
                children = graph.outgoing(node.id)
                if len(children) == 2:
                    left_edge, right_edge = children[0], children[1]
                    left_node = graph.get_node(left_edge.target)
                    right_node = graph.get_node(right_edge.target)
                    if left_node and right_node and left_node.type == "variable" and right_node.type == "variable":
                        if left_node.label == right_node.label:
                            matches.append({
                                "mul_node": node.id,
                                "var_node": left_node.id,
                                "var_label": left_node.label
                            })
        return matches

    def apply(self, graph: Graph, context: Dict) -> Tuple[Graph, float]:
        g = graph.clone()
        if "div_node" in context:
            div_node = context["div_node"]
            value = context["value"]
            new_node_id = g.add_node("constant", str(int(value) if value.is_integer() else str(value)), {})
            for edge in list(g.edges):
                if edge.target == div_node:
                    g.remove_edge(edge.id)
                    g.add_edge(edge.source, new_node_id, edge.relationship_type)
            g.remove_node(div_node)
            g.remove_node(context["num_node"])
            g.remove_node(context["den_node"])
            return g, -5.0
        elif "mul_node" in context:
            mul_node = context["mul_node"]
            var_label = context["var_label"]
            new_node_id = g.add_node("operator", "^", {})
            var_node_id = g.add_node("variable", var_label, {})
            exp_node_id = g.add_node("constant", "2", {})
            g.add_edge(new_node_id, var_node_id, "operand")
            g.add_edge(new_node_id, exp_node_id, "operand")
            for edge in list(g.edges):
                if edge.target == mul_node:
                    g.remove_edge(edge.id)
                    g.add_edge(edge.source, new_node_id, edge.relationship_type)
            g.remove_node(mul_node)
            g.remove_node(context["var_node"])
            other_var_id = None
            for edge in g.outgoing(mul_node):
                node = g.get_node(edge.target)
                if node and node.id != context["var_node"]:
                    other_var_id = node.id
            if other_var_id:
                g.remove_node(other_var_id)
            return g, -4.0
        return g,