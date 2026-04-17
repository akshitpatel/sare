# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-17 01:59:36
# Name: syntharithmeticpattern_71359
from typing import Tuple, List, Dict, Any, Optional
from sare.engine import Transform, Graph

from typing import Tuple, List, Dict, Any, Optional
import math
import re
from sare.engine import Transform, Graph

class SynthArithmeticPattern_71359(Transform):
    def name(self):
        return "syntharithmeticpattern_71359"

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
                new_node_id = g.add_node("function", "square", {"base": left.label})
                for e in g.incoming(op_id):
                    g.add_edge(e.source, new_node_id, e.relationship_type)
                g.remove_node(op_id)
                return g, -2.0
            elif left.type == "constant":
                try:
                    val = float(left.label)
                    result = val * val
                    new_node_id = g.add_node("constant", str(result), {})
                    for e in g.incoming(op_id):
                        g.add_edge(e.source, new_node_id, e.relationship_type)
                    g.remove_node(op_id)
                    g.remove_node(left_id)
                    return g, -3.0
                except ValueError:
                    pass
        else:
            left_id = context["left"]
            right_id = context["right"]
            result = context["result"]
            new_node_id = g.add_node("constant", str(result), {})
            for e in g.incoming(op_id):
                g.add_edge(e.source, new_node_id, e.relationship_type)
            g.remove_node(op_id)
            g.remove_node(left_id)
            g.remove_node(right_id)
            return g, -4.0
        return graph, 0.0