# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-16 14:12:00
# Name: syntharithmeticpattern_28891
from sare.engine import Transform, Graph

from sare.engine import Transform, Graph
from typing import List, Tuple, Dict, Optional
import math
import re

class SynthArithmeticPattern_28891(Transform):
    def name(self): return "syntharithmeticpattern_28891"

    def match(self, graph) -> List[Dict]:
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("+", "-", "/"):
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
                        if n.label == "+" and left_val + right_val == 1.0:
                            matches.append({"op": n.id, "left": left.id, "right": right.id, "result": "1"})
                        elif n.label == "-" and left_val - right_val == 0.0:
                            matches.append({"op": n.id, "left": left.id, "right": right.id, "result": "0"})
                        elif n.label == "/" and right_val != 0.0 and left_val % right_val == 0:
                            matches.append({"op": n.id, "left": left.id, "right": right.id, "result": str(int(left_val / right_val))})
                    except ValueError:
                        continue
            elif n.type == "operator" and n.label == "*":
                children = graph.outgoing(n.id)
                if len(children) != 2:
                    continue
                left = graph.get_node(children[0].target)
                right = graph.get_node(children[1].target)
                if not left or not right:
                    continue
                if left.type == "variable" and right.type == "variable" and left.label == right.label:
                    matches.append({"op": n.id, "var": left.id, "result": "square"})
        return matches

    def apply(self, graph, context) -> Tuple[Graph, float]:
        g = graph.clone()
        op_id = context["op"]
        if "result" in context and context["result"] == "square":
            var_id = context["var"]
            var_node = g.get_node(var_id)
            new_node_id = g.add_node("operator", "^", {})
            g.add_edge(new_node_id, var_id, "operand")
            const_node_id = g.add_node("constant", "2", {})
            g.add_edge(new_node_id, const_node_id, "operand")
            for e in list(g.edges):
                if e.target == op_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, new_node_id, e.relationship_type)
            g.remove_node(op_id)
            return g, -2.0
        else:
            result_val = context["result"]
            new_node_id = g.add_node("constant", result_val, {})
            for e in list(g.edges):
                if e.target == op_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, new_node_id, e.relationship_type)
            g.remove_node(op_id)
            g.remove_node(context["left"])
            g.remove_node(context["right"])
            return g, -3.0