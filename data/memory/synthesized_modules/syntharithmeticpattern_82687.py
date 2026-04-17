# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-17 05:09:11
# Name: syntharithmeticpattern_82687
from typing import Tuple, List, Dict, Any, Optional
from sare.engine import Transform, Graph

from typing import Tuple, List, Dict, Any, Optional
import math
import re
from sare.engine import Transform, Graph

class SynthArithmeticPattern_82687(Transform):
    def name(self):
        return "syntharithmeticpattern_82687"

    def match(self, graph: Graph) -> List[Dict[str, Any]]:
        matches = []
        for node in graph.nodes:
            # Look for division operator
            if node.type == "operator" and node.label == "/":
                children = graph.outgoing(node.id)
                if len(children) != 2:
                    continue
                num_edge, den_edge = children[0], children[1]
                num_node = graph.get_node(num_edge.target)
                den_node = graph.get_node(den_edge.target)
                
                if num_node.type != "constant" or den_node.type != "constant":
                    continue
                
                try:
                    num_val = float(num_node.label)
                    den_val = float(den_node.label)
                except ValueError:
                    continue
                
                if den_val == 0:
                    continue
                
                # Check if numerator and denominator share a common factor
                if num_val.is_integer() and den_val.is_integer():
                    gcd_val = math.gcd(int(num_val), int(den_val))
                    if gcd_val > 1:
                        matches.append({
                            "div_node": node.id,
                            "num_node": num_node.id,
                            "den_node": den_node.id,
                            "num_val": num_val,
                            "den_val": den_val,
                            "gcd": gcd_val
                        })
                # Check if division yields an integer
                elif num_val % den_val == 0:
                    matches.append({
                        "div_node": node.id,
                        "num_node": num_node.id,
                        "den_node": den_node.id,
                        "num_val": num_val,
                        "den_val": den_val,
                        "gcd": None
                    })
        return matches

    def apply(self, graph: Graph, context: Dict[str, Any]) -> Tuple[Graph, float]:
        g = graph.clone()
        div_node_id = context["div_node"]
        num_node_id = context["num_node"]
        den_node_id = context["den_node"]
        num_val = context["num_val"]
        den_val = context["den_val"]
        gcd_val = context["gcd"]
        
        # Find incoming edges to the division node
        incoming_edges = g.incoming(div_node_id)
        
        if gcd_val is not None and gcd_val > 1:
            # Simplify fraction by dividing numerator and denominator by gcd
            new_num = num_val / gcd_val
            new_den = den_val / gcd_val
            
            # Create new simplified constant nodes
            if new_den == 1:
                # Result is integer
                new_const_id = g.add_node("constant", str(int(new_num)), {})
            else:
                # Result is simplified fraction
                new_const_id = g.add_node("constant", f"{int(new_num)}/{int(new_den)}", {})
        else:
            # Division yields integer
            new_const_id = g.add_node("constant", str(int(num_val / den_val)), {})
        
        # Replace division node with new constant
        for edge in incoming_edges:
            g.add_edge(edge.source, new_const_id, edge.relationship_type)
        
        # Remove old nodes
        g.remove_node(div_node_id)
        g.remove_node(num_node_id)
        g.remove_node(den_node_id)
        
        return g, -5.0