# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-16 13:33:04
# Name: syntharithmeticpattern_26554
from sare.engine import Transform, Graph

from typing import List, Dict, Optional
import math
import re
from sare.engine import Transform, Graph

class SynthArithmeticPattern_26554(Transform):
    def name(self):
        return "syntharithmeticpattern_26554"
    
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
                                matches.append({
                                    "div_node": n.id,
                                    "num": num_node.id,
                                    "den": den_node.id,
                                    "value": num_val / den_val
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
                                "var_name": left_node.label
                            })
        
        return matches
    
    def apply(self, graph, context):
        g = graph.clone()
        
        if "div_node" in context:
            div_node = context["div_node"]
            num_id = context["num"]
            den_id = context["den"]
            value = context["value"]
            
            new_const_id = g.add_node("constant", str(int(value) if value.is_integer() else value), {})
            
            for e in list(g.edges):
                if e.target == div_node:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, new_const_id, e.relationship_type)
            
            g.remove_node(div_node)
            g.remove_node(num_id)
            g.remove_node(den_id)
            return g, -5.0
        
        elif "mul_node" in context:
            mul_node = context["mul_node"]
            var_left = context["var_left"]
            var_right = context["var_right"]
            var_name = context["var_name"]
            
            exp_node = g.add_node("operator", "^", {})
            var_node = g.add_node("variable", var_name, {})
            const_node = g.add_node("constant", "2", {})
            
            g.add_edge(exp_node, var_node, "operand")
            g.add_edge(exp_node, const_node, "operand")
            
            for e in list(g.edges):
                if e.target == mul_node:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, exp_node, e.relationship_type)
            
            g.remove_node(mul_node)
            g.remove_node(var_left)
            g.remove_node(var_right)
            return g, -4.0
        
        return graph, 0.0