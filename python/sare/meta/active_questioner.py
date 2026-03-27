import json
from typing import Dict, Any, Optional, List
from sare.engine import Graph, Node
from sare.memory.world_model import get_world_model
from sare.meta.homeostasis import get_homeostatic_system
from sare.memory.autobiographical import get_autobiographical_memory

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
        self._world_model = get_world_model()
        self._homeostasis = get_homeostatic_system()
        self._autobio = get_autobiographical_memory()
        self._recent_questions = []  # track last 5 questions to avoid repeats
        self._last_ask_cycle = 0

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
        if not node:
            return "?"
        
        kids = graph.children(node)
        if not kids:
            return str(node.value)
            
        if len(kids) == 1:
            return f"{node.value}({self._extract_subgraph_expr(graph, kids[0], depth-1)})"
        elif len(kids) == 2:
            left = self._extract_subgraph_expr(graph, kids[0], depth-1)
            right = self._extract_subgraph_expr(graph, kids[1], depth-1)
            return f"({left} {node.value} {right})"
            
        k_strs = [self._extract_subgraph_expr(graph, k, depth-1) for k in kids]
        return f"{node.value}({', '.join(k_strs)})"

    def formulate_question(self, graph: Graph, domain: str = "general",
                           recent_failures: int = 0) -> Dict[str, Any]:
        """Returns a structured question object or None if no locus found."""
        locus = self.identify_locus(graph)
        if not locus:
            expr_str = str(graph)
        else:
            expr_str = self._extract_subgraph_expr(graph, locus, depth=2)

        expr_str = expr_str.replace("+-", "-").replace(" * ", "*")

        # Try LLM-generated Socratic question first
        question_text = self._llm_question(expr_str, domain, recent_failures)
        if not question_text:
            template = self.templates.get(domain, self.templates["general"])
            question_text = template.format(expr=f"'{expr_str}'")

        return {
            "question": question_text,
            "target_expr": expr_str,
            "domain": domain,
            "locus_node_id": locus.id if locus else None
        }

    def _llm_question(self, expr: str, domain: str, recent_failures: int) -> str:
        """Generate a Socratic question via LLM. Returns empty string on failure."""
        try:
            from sare.interface.llm_bridge import _call_llm
            context = (
                f"I am an AI learning {domain} math. I've failed {recent_failures} times "
                f"trying to simplify the expression: {expr}\n"
                f"Generate one short, specific Socratic question (≤20 words) I should ask "
                f"a human teacher to learn the rule needed. Return ONLY the question."
            )
            q = _call_llm(context).strip()
            if q and len(q) > 10:
                return q
        except Exception:
            pass
        return ""

    def should_ask_question(self, domain: str, recent_failures: int, current_cycle: int) -> bool:
        """
        Decides whether to generate a question now.
        Returns True if confusion level exceeds threshold.
        """
        # Avoid asking too frequently: minimum 10 cycles between questions
        if current_cycle - self._last_ask_cycle < 10:
            return False
        
        # Check homeostasis: high social drive encourages asking
        social_drive = self._homeostasis.get_drive_level("social")
        if social_drive > 0.8:
            return True
        
        # Check world model surprise for this domain
        domain_surprise = self._world_model.get_domain_surprise(domain)
        if domain_surprise is not None and domain_surprise > 2.5:
            return True
        
        # Check recent failure streak
        if recent_failures >= 5:
            return True
        
        # Check autobiographical memory: if we've been stuck on this domain for a while
        episodes = self._autobio.retrieve_similar([], top_k=20)
        domain_fail_episodes = [e for e in episodes if e.domain == domain and e.event_type == "stuck_period"]
        if len(domain_fail_episodes) >= 3:
            return True
        
        # Default: don't ask
        return False

    def record_question_asked(self, question: Dict[str, Any], current_cycle: int):
        """Record that a question was asked to avoid repeats."""
        self._recent_questions.append(question["target_expr"])
        if len(self._recent_questions) > 5:
            self._recent_questions.pop(0)
        self._last_ask_cycle = current_cycle
        # Satisfy social drive a bit because we're seeking interaction
        self._homeostasis.satisfy("social", 0.15)