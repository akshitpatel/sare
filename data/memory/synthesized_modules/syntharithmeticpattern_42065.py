# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-16 17:51:23
# Name: syntharithmeticpattern_42065
from sare.engine import Transform, Graph

from typing import List, Dict, Optional
import math
import re
from sare.engine import Transform, Graph

class SynthArithmeticPattern_42065(Transform):
    def name(self):
        return "syntharithmeticpattern_42065"
    
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
                
                if left and right and left.type == "constant" and right.type == "constant":
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
                            if right_val != 0:
                                result = left_val / right_val
                            else:
                                continue
                        else:
                            continue
                        
                        if result.is_integer():
                            result_label = str(int(result))
                        else:
                            result_label = str(result)
                        
                        matches.append({
                            "op": n.id,
                            "left": left.id,
                            "right": right.id,
                            "result": result_label
                        })
                    except ValueError:
                        continue
        return matches
    
    def apply(self, graph, context):
        g = graph.clone()
        op_id = context["op"]
        left_id = context["left"]
        right_id = context["right"]
        result_label = context["result"]
        
        result_node_id = g.add_node("constant", result_label, {})
        
        incoming_edges = g.incoming(op_id)
        for e in incoming_edges:
            g.add_edge(e.source, result_node_id, e.relationship_type)
        
        g.remove_node(op_id)
        g.remove_node(left_id)
        g.remove_node(right_id)
        
        return g, -5.0