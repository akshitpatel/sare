import json
from typing import Dict, Any, Optional
from sare.engine import Graph, Node

class ActiveQuestioner:
    """
    Identifies high-uncertainty regions of an expression graph and generates
    conversational English questions asking for the rule to simplify them.
    """
    def __init__(self):
        self.templates = {
            "arithmetic": "I'm stuck on {expr}. Can you teach me the rule to simplify this?",
            "logic": "I don't know how to evaluate {expr}. What does it reduce to?",
            "algebra": "I hit a wall trying to expand {expr}. How do I handle this?",
            "sets": "I'm unsure about the set operation {expr}. What is its minimal form?",
            "general": "I'm not sure how to simplify {expr}. Can you provide the rule in English?"
        }

    def identify_locus(self, graph: Graph) -> Optional[Node]:
        """Finds the node with the highest uncertainty score (must be > 0.0)"""
        highest_node = None
        max_u = 0.0
        
        for node in graph.nodes():
            if hasattr(node, "uncertainty") and node.uncertainty > max_u:
                max_u = node.uncertainty
                highest_node = node
                
        # Fallback if uncertainty isn't formally populated yet: just pick the root
        if not highest_node and graph.nodes():
            return graph.root()
            
        return highest_node

    def _extract_subgraph_expr(self, graph: Graph, node: Node, depth: int = 1) -> str:
        """Extracts a readable string representation of the subgraph at radius `depth`"""
        # A simple stringifier for now. In a full implementation, this uses a robust AST-to-text stringifier.
        if not node:
            return "?"
        
        # Primitive formatting
        kids = graph.children(node)
        if not kids:
            return str(node.value)
            
        if len(kids) == 1:
            return f"{node.value}({self._extract_subgraph_expr(graph, kids[0], depth-1)})"
        elif len(kids) == 2:
            left = self._extract_subgraph_expr(graph, kids[0], depth-1)
            right = self._extract_subgraph_expr(graph, kids[1], depth-1)
            return f"({left} {node.value} {right})"
            
        # N-ary
        k_strs = [self._extract_subgraph_expr(graph, k, depth-1) for k in kids]
        return f"{node.value}({', '.join(k_strs)})"

    def formulate_question(self, graph: Graph, domain: str = "general") -> Dict[str, Any]:
        """Returns a structured question object or None if no locus found."""
        locus = self.identify_locus(graph)
        if not locus:
            # Fallback to the whole graph string if no locus
            expr_str = str(graph)
        else:
            expr_str = self._extract_subgraph_expr(graph, locus, depth=2)
            
        # Clean up common ugly AST strings
        expr_str = expr_str.replace("+-", "-").replace(" * ", "*")
            
        template = self.templates.get(domain, self.templates["general"])
        question_text = template.format(expr=f"'{expr_str}'")
        
        return {
            "question": question_text,
            "target_expr": expr_str,
            "domain": domain,
            "locus_node_id": locus.id if locus else None
        }
