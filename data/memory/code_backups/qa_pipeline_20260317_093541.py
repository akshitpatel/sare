"""
QAPipeline — solves question-answering problems from context + commonsense KB.
Represents questions as graphs with unknown (?) nodes that inference transforms fill in.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Dict, List, Optional, Tuple

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

    def _parse_question(self, question: str, context: dict = None) -> Tuple[str, str]:
        """
        Return (subject, relation) extracted from the question string.
        Recognises patterns:
          - "What does X Y?"   → subject=X, relation=Y
          - "What is X?"       → subject=X, relation=IsA
          - "What has X?"      → subject=X, relation=HasA
          - "What causes X?"   → subject=X, relation=Causes
        Falls back to ("unknown_subject", "unknown_relation").
        """
        q = question.strip().rstrip("?").strip()

        # "What does <subject> <relation>" or "What does <subject> do"
        m = re.match(r"(?i)what\s+does\s+(\w+)\s+(\w+)", q)
        if m:
            return m.group(1).lower(), m.group(2).lower()

        # "What is <subject>"
        m = re.match(r"(?i)what\s+is\s+(?:a\s+|an\s+)?(\w+)", q)
        if m:
            return m.group(1).lower(), "IsA"

        # "What has <subject>"
        m = re.match(r"(?i)what\s+has\s+(\w+)", q)
        if m:
            return m.group(1).lower(), "HasA"

        # "What causes <subject>"
        m = re.match(r"(?i)what\s+causes\s+(\w+)", q)
        if m:
            return m.group(1).lower(), "Causes"

        # "What can <subject> do"
        m = re.match(r"(?i)what\s+can\s+(\w+)\s+do", q)
        if m:
            return m.group(1).lower(), "CapableOf"

        # Try context dict
        if context:
            subject = str(context.get("subject", "unknown_subject")).lower()
            relation = str(context.get("relation", "unknown_relation")).lower()
            return subject, relation

        # Generic fallback: first noun-like token after "what"
        tokens = q.lower().split()
        if len(tokens) >= 2:
            return tokens[-1], "unknown_relation"

        return "unknown_subject", "unknown_relation"

    # ------------------------------------------------------------------
    # answer_from_graph
    # ------------------------------------------------------------------
    def answer_from_graph(self, solved_graph, original_subject: str = "") -> str:
        """
        Extract the answer from a solved graph.

        Strategy: after transforms run, the graph may contain concept nodes
        that were not in the original question (i.e. they replaced the ?
        unknown node).  We return the label of any concept node that is
        neither the original subject nor empty, preferring non-relational
        node types.
        """
        if solved_graph is None:
            return "unknown"

        try:
            candidates: List[str] = []
            for node in solved_graph.nodes:
                label = getattr(node, "label", "") or ""
                ntype = getattr(node, "type", "") or ""
                if label in ("", "?"):
                    continue
                if label.lower() == original_subject.lower():
                    continue
                if ntype in ("concept", "entity", "answer"):
                    candidates.append(label)
                elif ntype not in ("relation", "unknown"):
                    candidates.append(label)

            if candidates:
                return candidates[0]
        except Exception as exc:
            log.error("QAPipeline.answer_from_graph: %s", exc)

        return "unknown"

    # ------------------------------------------------------------------
    # build_from_fact
    # ------------------------------------------------------------------
    def build_from_fact(self, subject: str, relation: str, obj: str) -> Tuple[Optional[object], str]:
        """
        Build a question graph that hides `obj` behind an unknown node.
        Returns (graph, correct_answer) for training / evaluation.

        Graph:
            concept(subject) -[has_relation]-> relation(relation) -[object]-> unknown(?)
        The caller knows the answer is `obj`.
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
