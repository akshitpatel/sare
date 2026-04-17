# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-16 14:58:12
# Name: syntharithmeticpattern_31626
from sare.engine import Transform, Graph

from typing import List, Tuple, Dict, Optional
import math
import re
from sare.engine import Transform, Graph

class SynthArithmeticPattern_31626(Transform):
    def name(self) -> str:
        return "syntharithmeticpattern_31626"
    
    def match(self, graph: Graph) -> List[Dict]:
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
                                    "frac": n.id,
                                    "num": num_node.id,
                                    "den": den_node.id,
                                    "value": num_val / den_val
                                })
                        except ValueError:
                            continue
                            
            elif n.type == "operator" and n.label in ("+", "-"):
                children = graph.outgoing(n.id)
                if len(children) == 2:
                    left_edge, right_edge = children[0], children[1]
                    left_node = graph.get_node(left_edge.target)
                    right_node = graph.get_node(right_edge.target)
                    
                    if left_node and right_node and left_node.type == "operator" and right_node.type == "operator":
                        if left_node.label == "/" and right_node.label == "/":
                            left_children = graph.outgoing(left_node.id)
                            right_children = graph.outgoing(right_node.id)
                            
                            if len(left_children) == 2 and len(right_children) == 2:
                                left_num = graph.get_node(left_children[0].target)
                                left_den = graph.get_node(left_children[1].target)
                                right_num = graph.get_node(right_children[0].target)
                                right_den = graph.get_node(right_children[1].target)
                                
                                if (left_num and left_den and right_num and right_den and
                                    left_num.type == "constant" and left_den.type == "constant" and
                                    right_num.type == "constant" and right_den.type == "constant"):
                                    
                                    try:
                                        left_den_val = float(left_den.label)
                                        right_den_val = float(right_den.label)
                                        if left_den_val == right_den_val:
                                            matches.append({
                                                "op": n.id,
                                                "left_frac": left_node.id,
                                                "right_frac": right_node.id,
                                                "left_num": left_num.id,
                                                "left_den": left_den.id,
                                                "right_num": right_num.id,
                                                "right_den": right_den.id,
                                                "operator": n.label
                                            })
                                    except ValueError:
                                        continue
            
            elif n.type == "operator" and n.label == "*":
                children = graph.outgoing(n.id)
                if len(children) == 2:
                    left_edge, right_edge = children[0], children[1]
                    left_node = graph.get_node(left_edge.target)
                    right_node = graph.get_node(right_edge.target)
                    
                    if left_node and right_node and left_node.id == right_node.id:
                        matches.append({
                            "square": n.id,
                            "base": left_node.id
                        })
        
        return matches
    
    def apply(self, graph: Graph, context: Dict) -> Tuple[Graph, float]:
        g = graph.clone()
        
        if "frac" in context:
            frac_id = context["frac"]
            value = context["value"]
            
            new_node_id = g.add_node("constant", str(int(value) if value.is_integer() else str(value)), {})
            
            for e in list(g.edges):
                if e.target == frac_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, new_node_id, e.relationship_type)
            
            g.remove_node(frac_id)
            g.remove_node(context["num"])
            g.remove_node(context["den"])
            
            return g, -5.0
            
        elif "op" in context:
            op_id = context["op"]
            left_num_id = context["left_num"]
            right_num_id = context["right_num"]
            den_id = context["left_den"]
            operator = context["operator"]
            
            left_num_node = g.get_node(left_num_id)
            right_num_node = g.get_node(right_num_id)
            den_node = g.get_node(den_id)
            
            try:
                left_val = float(left_num_node.label)
                right_val = float(right_num_node.label)
                den_val = float(den_node.label)
                
                if operator == "+":
                    result_num = left_val + right_val
                else:
                    result_num = left_val - right_val
                
                if result_num % den_val == 0:
                    result_val = result_num / den_val
                    new_label = str(int(result_val) if result_val.is_integer() else str(result_val))
                else:
                    new_label = f"{int(result_num) if result_num.is_integer() else result_num}/{int(den_val) if den_val.is_integer() else den_val}"
                
                new_node_id = g.add_node("constant" if "/" not in new_label else "operator", 
                                        new_label, {})
                
                for e in list(g.edges):
                    if e.target == op_id:
                        g.remove_edge(e.id)
                        g.add_edge(e.source, new_node_id, e.relationship_type)
                
                g.remove_node(op_id)
                g.remove_node(context["left_frac"])
                g.remove_node(context["right_frac"])
                g.remove_node(left_num_id)
                g.remove_node(right_num_id)
                g.remove_node(den_id)
                
                return g, -7.0
                
            except (ValueError, AttributeError):
                return graph, 0.0
                
        elif "square" in context:
            square_id = context["square"]
            base_id = context["base"]
            
            base_node = g.get_node(base_id)
            
            if base_node.type == "variable":
                new_node_id = g.add_node("operator", "^", {})
                g.add_edge(new_node_id, base_id, "base")
                exp_node_id = g.add_node("constant", "2", {})
                g.add_edge(new_node_id, exp_node_id, "exponent")
            elif base_node.type == "constant":
                try:
                    val = float(base_node.label)
                    result = val * val
                    new_label = str(int(result) if result.is_integer() else str(result))
                    new_node_id = g.add_node("constant", new_label, {})
                except ValueError:
                    new_node_id = g.add_node("operator", "^", {})
                    g.add_edge(new_node_id, base_id, "base")
                    exp_node_id = g.add_node("constant", "2", {})
                    g.add_edge(new_node_id, exp_node_id, "exponent")
            else:
                new_node_id = g.add_node("operator", "^", {})
                g.add_edge(new_node_id, base_id, "base")
                exp_node_id = g.add_node("constant", "2", {})
                g.add_edge(new_node_id, exp_node_id, "exponent")
            
            for e in list(g.edges):
                if e.target == square_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, new_node_id, e.relationship_type)
            
            g.remove_node(square_id)
            
            return g, -4.0
        
        return graph, 0.0