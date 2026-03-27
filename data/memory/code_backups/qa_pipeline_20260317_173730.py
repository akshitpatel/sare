"""
QAPipeline — solves question-answering problems from context + commonsense KB.
Represents questions as graphs with unknown (?) nodes that inference transforms fill in.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Dict, List, Optional, Tuple, Union

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine / Graph import — graceful fallback
# ---------------------------------------------------------------------------
try:
    from sare.engine import Graph
except ImportError:
    try:
        from sare.sare_bindings import Graph
    except ImportError:
        log.warning("QAPipeline: no Graph implementation found; using None placeholder.")
        Graph = None  # type: ignore

# ---------------------------------------------------------------------------
# GeneratedProblem import
# ---------------------------------------------------------------------------
try:
    from sare.curiosity.curriculum_generator import GeneratedProblem
except ImportError:
    log.warning("QAPipeline: could not import GeneratedProblem.")
    GeneratedProblem = None  # type: ignore


# ---------------------------------------------------------------------------
# Helper: build a fresh Graph safely
# ---------------------------------------------------------------------------
def _new_graph() -> Optional[object]:
    if Graph is None:
        return None
    try:
        return Graph()
    except Exception as exc:
        log.error("QAPipeline: failed to instantiate Graph: %s", exc)
        return None


# ---------------------------------------------------------------------------
# QAPipeline
# ---------------------------------------------------------------------------

class QAPipeline:
    """
    Converts natural-language questions into SARE graph problems and
    extracts answers from solved graphs.

    Graph structure for a question "What does X Y?":
        concept_node(X) -[has_relation]-> relation_node(Y) -[object]-> unknown_node(?)

    SARE inference transforms are expected to replace the unknown node
    with a concrete concept drawn from the commonsense KB.
    """

    def __init__(self, commonsense_kb=None):
        self._kb = commonsense_kb

    # ------------------------------------------------------------------
    # build_question_graph
    # ------------------------------------------------------------------
    def build_question_graph(self, question: str, context: dict = None) -> Optional[object]:
        """
        Parse a simple 'What does X Y?' or 'What is X?' question into a graph.

        Produced graph:
            concept_node(subject) -[has_relation]-> relation_node(relation)
                                                         -[object]-> unknown_node(?)

        Falls back to a generic single-unknown graph if parsing fails.
        """
        g = _new_graph()
        if g is None:
            return None

        subject, relation = self._parse_question(question, context)

        try:
            subject_id = g.add_node("concept", subject)
            relation_id = g.add_node("relation", relation)
            unknown_id = g.add_node("unknown", "?")

            g.add_edge(subject_id, relation_id, "has_relation")
            g.add_edge(relation_id, unknown_id, "object")
        except Exception as exc:
            log.error("QAPipeline.build_question_graph: graph construction failed: %s", exc)
            return g

        return g

    def _parse_question(self, question: str, context: dict = None) -> Tuple[Union[str, List[str]], str]:
        """
        Return (subject, relation) extracted from the question string.
        Recognises patterns:
          - "What does X Y?"   → subject=X, relation=Y
          - "What is X?"       → subject=X, relation=IsA
          - "What has X?"      → subject=X, relation=HasA
          - "What causes X?"   → subject=X, relation=Causes
          - "Where is X located?" → subject=X, relation=Location
          - "When did event Y happen?" → subject=Y, relation=Time
          - "Which is larger, X or Y?" → subject=[X, Y], relation=larger
          - "What is the most common Y?" → subject=Y, relation=most_common
          - "Why does X Y?"    → subject=X, relation=Cause
          - "Why is X?"        → subject=X, relation=Cause
          - "Why did X Y?"     → subject=X, relation=Cause
        Falls back to ("unknown_subject", "unknown_relation").
        """
        q = question.strip().rstrip("?").strip()

        # Helper function to extract quoted phrases
        def extract_quoted(text: str) -> Optional[str]:
            """Extract first quoted phrase from text."""
            match = re.search(r'"([^"]+)"', text)
            if match:
                return match.group(1).lower()
            match = re.search(r"'([^']+)'", text)
            if match:
                return match.group(1).lower()
            return None

        # Helper function to extract noun chunks (simple heuristic)
        def extract_noun_chunk(text: str) -> str:
            """Extract a noun chunk from text using simple heuristics."""
            # Remove common determiners and prepositions at start
            text = re.sub(r'^(the|a|an|some|this|that|these|those)\s+', '', text, flags=re.IGNORECASE)
            
            # Take first noun phrase (simplified)
            words = text.split()
            if not words:
                return text
            
            # Find first noun-like word (heuristic: not a common verb/adverb)
            common_verbs = {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                           'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                           'can', 'shall', 'must', 'need', 'ought'}
            common_adverbs = {'not', 'very', 'really', 'quickly', 'slowly', 'always', 'never',
                             'often', 'sometimes', 'usually', 'already', 'just'}
            
            for i, word in enumerate(words):
                if word.lower() not in common_verbs and word.lower() not in common_adverbs:
                    # Take from this word onward as the noun chunk
                    return ' '.join(words[i:])
            
            return text

        # Helper to extract comparative entities (X or Y)
        def extract_comparative_entities(text: str) -> List[str]:
            """Extract entities from comparative patterns like 'X or Y'."""
            # Pattern: "X or Y" or "X and Y"
            match = re.search(r'(\w+(?:\s+\w+)*)\s+(?:or|and)\s+(\w+(?:\s+\w+)*)', text, re.IGNORECASE)
            if match:
                return [match.group(1).strip().lower(), match.group(2).strip().lower()]
            
            # Pattern: "between X and Y"
            match = re.search(r'between\s+(\w+(?:\s+\w+)*)\s+and\s+(\w+(?:\s+\w+)*)', text, re.IGNORECASE)
            if match:
                return [match.group(1).strip().lower(), match.group(2).strip().lower()]
            
            return []

        # Try quoted phrase first
        quoted = extract_quoted(q)
        if quoted:
            return quoted, "IsA"

        # Comparative patterns (must come before simple patterns)
        comparative_patterns = [
            # "Which is larger, X or Y?" or "Which is more expensive, X or Y?"
            (r'which\s+is\s+(?:more\s+)?(\w+),?\s+(\w+(?:\s+\w+)*)\s+or\s+(\w+(?:\s+\w+)*)',
             lambda m: ([m.group(2).lower(), m.group(3).lower()], m.group(1).lower())),
            
            # "Which X is larger, A or B?" (e.g., "Which planet is larger, Earth or Mars?")
            (r'which\s+(\w+)\s+is\s+(?:more\s+)?(\w+),?\s+(\w+(?:\s+\w+)*)\s+or\s+(\w+(?:\s+\w+)*)',
             lambda m: ([m.group(3).lower(), m.group(4).lower()], f"{m.group(2).lower()}_of_{m.group(1).lower()}")),
            
            # "Is X larger than Y?" or "Is X more expensive than Y?"
            (r'is\s+(\w+(?:\s+\w+)*)\s+(?:more\s+)?(\w+)(?:er)?\s+than\s+(\w+(?:\s+\w+)*)',
             lambda m: ([m.group(1).lower(), m.group(3).lower()], m.group(2).lower())),
            
            # "What is larger, X or Y?" or "What is more common, X or Y?"
            (r'what\s+is\s+(?:more\s+)?(\w+),?\s+(\w+(?:\s+\w+)*)\s+or\s+(\w+(?:\s+\w+)*)',
             lambda m: ([m.group(2).lower(), m.group(3).lower()], m.group(1).lower())),
        ]
        
        for pattern, extractor in comparative_patterns:
            match = re.search(pattern, q, re.IGNORECASE)
            if match:
                subjects, relation = extractor(match)
                # Convert comparative adjectives to relation names
                if relation in ['larger', 'bigger', 'greater']:
                    relation = 'LargerThan'
                elif relation in ['smaller', 'lesser']:
                    relation = 'SmallerThan'
                elif relation in ['more', 'common']:
                    relation = 'MoreCommon'
                elif relation in ['less', 'rare']:
                    relation = 'LessCommon'
                elif relation in ['better']:
                    relation = 'BetterThan'
                elif relation in ['worse']:
                    relation = 'WorseThan'
                elif relation in ['faster']:
                    relation = 'FasterThan'
                elif relation in ['slower']:
                    relation = 'SlowerThan'
                elif relation in ['older']:
                    relation = 'OlderThan'
                elif relation in ['newer']:
                    relation = 'NewerThan'
                elif relation in ['higher']:
                    relation = 'HigherThan'
                elif relation in ['lower']:
                    relation = 'LowerThan'
                # Keep other comparative relations as-is
                return subjects, relation

        # Superlative patterns
        superlative_patterns = [
            # "What is the most common Y?" or "What is the largest X?"
            (r'what\s+is\s+the\s+(?:most\s+)?(\w+)(?:est)?\s+(\w+(?:\s+\w+)*)',
             lambda m: (m.group(2).lower(), f"most_{m.group(1).lower()}" if 'most' in q.lower() else f"{m.group(1).lower()}est")),
            
            # "Which X is the most common?" or "Which planet is the largest?"
            (r'which\s+(\w+)\s+is\s+the\s+(?:most\s+)?(\w+)(?:est)?',
             lambda m: (m.group(1).lower(), f"most_{m.group(2).lower()}" if 'most' in q.lower() else f"{m.group(2).lower()}est")),
            
            # "What Y is the most common?" (e.g., "What bird is the most common?")
            (r'what\s+(\w+)\s+is\s+the\s+(?:most\s+)?(\w+)(?:est)?',
             lambda m: (m.group(1).lower(), f"most_{m.group(2).lower()}" if 'most' in q.lower() else f"{m.group(2).lower()}est")),
        ]
        
        for pattern, extractor in superlative_patterns:
            match = re.search(pattern, q, re.IGNORECASE)
            if match:
                subject, relation = extractor(match)
                # Clean up relation names
                if relation.startswith('most_'):
                    # Keep as-is for most_common, most_expensive, etc.
                    pass
                elif relation.endswith('est'):
                    # Convert to MostX pattern for consistency
                    base_adj = relation[:-3]  # Remove 'est'
                    relation = f"Most{base_adj.capitalize()}"
                return subject, relation

        # Why patterns — causal question forms mapped to "Cause" relation
        # These must be checked before the generic simple patterns to avoid
        # "Why does X Y?" being partially matched by "what does" style patterns.
        why_patterns = [
            # "Why does X Y?" (e.g., "Why does it rain?", "Why does the sky appear blue?")
            (r'why\s+does\s+(\w+(?:\s+\w+)*)\s+(\w+(?:\s+\w+)*)',
             lambda m: (m.group(1).lower(), 'Cause')),

            # "Why did X Y?" (e.g., "Why did the chicken cross the road?")
            (r'why\s+did\s+(\w+(?:\s+\w+)*)\s+(\w+(?:\s+\w+)*)',
             lambda m: (m.group(1).lower(), 'Cause')),

            # "Why is X?" (e.g., "Why is the sky blue?")
            (r'why\s+is\s+(\w+(?:\s+\w+)*)',
             lambda m: (m.group(1).lower(), 'Cause')),

            # "Why are X?" (e.g., "Why are leaves green?")
            (r'why\s+are\s+(\w+(?:\s+\w+)*)',
             lambda m: (m.group(1).lower(), 'Cause')),

            # "Why was X?" (e.g., "Why was the war started?")
            (r'why\s+was\s+(\w+(?:\s+\w+)*)',
             lambda m: (m.group(1).lower(), 'Cause')),

            # "Why were X?" (e.g., "Why were the pyramids built?")
            (r'why\s+were\s+(\w+(?:\s+\w+)*)',
             lambda m: (m.group(1).lower(), 'Cause')),

            # Generic "Why X?" fallback (e.g., "Why photosynthesis?")
            (r'why\s+(\w+(?:\s+\w+)*)',
             lambda m: (m.group(1).lower(), 'Cause')),
        ]

        for pattern, extractor in why_patterns:
            match = re.search(pattern, q, re.IGNORECASE)
            if match:
                return extractor(match)

        # Simple fact-based patterns (existing)
        patterns = [
            # "What does X Y?" (e.g., "What does photosynthesis produce?")
            (r'what\s+does\s+(\w+(?:\s+\w+)*)\s+(\w+(?:\s+\w+)*)',
             lambda m: (m.group(1).lower(), m.group(2).lower())),
            
            # "What is X?" (e.g., "What is photosynthesis?")
            (r'what\s+is\s+(\w+(?:\s+\w+)*)',
             lambda m: (m.group(1).lower(), 'IsA')),
            
            # "What has X?" (e.g., "What has feathers?")
            (r'what\s+has\s+(\w+(?:\s+\w+)*)',
             lambda m: (m.group(1).lower(), 'HasA')),
            
            # "What causes X?" (e.g., "What causes rain?")
            (r'what\s+causes\s+(\w+(?:\s+\w+)*)',
             lambda m: (m.group(1).lower(), 'Causes')),
            
            # "Where is X located?" (e.g., "Where is Paris located?")
            (r'where\s+is\s+(\w+(?:\s+\w+)*)\s+located',
             lambda m: (m.group(1).lower(), 'Location')),
            
            # "When did event Y happen?" (e.g., "When did World War II happen?")
            (r'when\s+did\s+(?:event\s+)?(\w+(?:\s+\w+)*)\s+happen',
             lambda m: (m.group(1).lower(), 'Time')),
            
            # "Who is X?" (e.g., "Who is Albert Einstein?")
            (r'who\s+is\s+(\w+(?:\s+\w+)*)',
             lambda m: (m.group(1).lower(), 'IsA')),
            
            # "How does X Y?" (e.g., "How does photosynthesis work?")
            (r'how\s+does\s+(\w+(?:\s+\w+)*)\s+(\w+(?:\s+\w+)*)',
             lambda m: (m.group(1).lower(), m.group(2).lower())),
            
            # "Why does X Y?" (e.g., "Why does it rain?")
            (r'why\s+does\s+(\w+(?:\s+\w+)*)\s+(\w+(?:\s+\w+)*)',
             lambda m: (m.group(1).lower(), f"reason_for_{m.group(2).lower()}")),
        ]
        
        for pattern, extractor in patterns:
            match = re.search(pattern, q, re.IGNORECASE)
            if match:
                return extractor(match)
        
        # Fallback: try to extract noun chunk as subject
        subject = extract_noun_chunk(q)
        return subject, "unknown_relation"

    # ------------------------------------------------------------------
    # extract_answer
    # ------------------------------------------------------------------
    def extract_answer(self, graph, question: str = None) -> Optional[str]:
        """
        Extract the answer from a solved graph.

        Looks for a node that was originally unknown (?) and now has a value.
        Falls back to scanning for any node with a 'value' attribute.
        """
        if graph is None:
            return None

        try:
            # Strategy 1: Find node that was originally unknown
            for node_id in graph.nodes():
                node = graph.get_node(node_id)
                if node is None:
                    continue
                    
                # Check if this was the unknown node
                if hasattr(node, "attributes") and node.attributes.get("original_label") == "?":
                    return str(node.label)
                
                # Check node type if available
                if hasattr(node, "type") and node.type == "unknown":
                    return str(node.label)
            
            # Strategy 2: Find node with 'answer' or 'value' attribute
            for node_id in graph.nodes():
                node = graph.get_node(node_id)
                if node is None:
                    continue
                    
                if hasattr(node, "attributes"):
                    if "answer" in node.attributes:
                        return str(node.attributes["answer"])
                    if "value" in node.attributes:
                        return str(node.attributes["value"])
            
            # Strategy 3: Look for the node connected to the relation node
            # This assumes the graph structure: subject -[has_relation]-> relation -[object]-> answer
            for node_id in graph.nodes():
                node = graph.get_node(node_id)
                if node is None:
                    continue
                    
                # Check if this is a relation node
                if hasattr(node, "type") and node.type == "relation":
                    # Find outgoing 'object' edges
                    for edge in graph.edges_from(node_id):
                        if hasattr(edge, "label") and edge.label == "object":
                            target_node = graph.get_node(edge.target)
                            if target_node:
                                return str(target_node.label)
            
            # Strategy 4: Return the label of the first node that isn't the subject or relation
            # This is a last resort
            nodes = list(graph.nodes())
            if len(nodes) >= 3:
                # Assuming order: subject, relation, answer
                answer_node = graph.get_node(nodes[2])
                if answer_node:
                    return str(answer_node.label)
            
        except Exception as exc:
            log.error("QAPipeline.extract_answer: %s", exc)
        
        return None

    # ------------------------------------------------------------------
    # build_from_fact
    # ------------------------------------------------------------------
    def build_from_fact(self, subject: str, relation: str, obj: str) -> Tuple[Optional[object], str]:
        """
        Build a QA graph from a known fact (subject, relation, object).
        The object is hidden behind an unknown node so the solver can verify what the graph
        should be.
        """
        g = _new_graph()
        if g is None:
            return None, obj

        try:
            subject_id = g.add_node("concept", subject.lower())
            relation_id = g.add_node("relation", relation)
            unknown_id = g.add_node("unknown", "?")

            g.add_edge(subject_id, relation_id, "has_relation")
            g.add_edge(relation_id, unknown_id, "object")

            # Tag the unknown node with metadata so transforms can find it
            # Stored as graph-level attribute via node attribute dict if supported
            try:
                node = g.get_node(unknown_id)
                if node is not None and hasattr(node, "attributes"):
                    node.attributes["expected_answer"] = obj.lower()
                    node.attributes["relation"] = relation
                    node.attributes["subject"] = subject.lower()
            except Exception:
                pass

        except Exception as exc:
            log.error("QAPipeline.build_from_fact: %s", exc)

        return g, obj

    # ------------------------------------------------------------------
    # generate_qa_problems
    # ------------------------------------------------------------------
    def generate_qa_problems(self, kb=None, n: int = 10) -> List[object]:
        """
        Generate up to `n` QA GeneratedProblem objects from KB facts.
        Each problem hides one triple's object behind an unknown node.
        """
        if GeneratedProblem is None:
            log.error("QAPipeline.generate_qa_problems: GeneratedProblem not available.")
            return []

        kb = kb or self._kb
        if kb is None:
            log.warning("QAPipeline.generate_qa_problems: no KB provided; returning empty list.")
            return []

        triples: List[Tuple[str, str, str]] = []

        # Support kb.to_triples() if available
        if hasattr(kb, "to_triples"):
            try:
                triples = list(kb.to_triples())
            except Exception as exc:
                log.warning("QAPipeline: kb.to_triples() failed: %s", exc)

        # Fallback: iterate _forward dict directly
        if not triples and hasattr(kb, "_forward"):
            try:
                for subj, facts in kb._forward.items():
                    for rel, obj in facts:
                        triples.append((subj, rel, obj))
            except Exception as exc:
                log.warning("QAPipeline: _forward iteration failed: %s", exc)

        if not triples:
            log.warning("QAPipeline.generate_qa_problems: KB has no iterable triples.")
            return []

        # Limit to n triples
        triples = triples[:n]

        problems = []
        for subject, relation, obj in triples:
            try:
                graph, answer = self.build_from_fact(subject, relation, obj)
                if graph is None:
                    continue
                prob = GeneratedProblem(
                    id=f"qa_{uuid.uuid4().hex[:8]}",
                    graph=graph,
                    origin=f"kb_fact:{subject}:{relation}:{obj}",
                    status="pending",
                    domain="qa",
                )
                problems.append(prob)
            except Exception as exc:
                log.warning("QAPipeline: failed to build problem for (%s,%s,%s): %s",
                            subject, relation, obj, exc)

        log.info("QAPipeline: generated %d QA problems", len(problems))
        return problems


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_SINGLETON: Optional[QAPipeline] = None


def get_qa_pipeline() -> QAPipeline:
    """Return the module-level singleton QAPipeline (no KB attached by default)."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = QAPipeline()
    return _SINGLETON