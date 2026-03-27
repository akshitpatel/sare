"""
FactIngester — Parses Q&A pairs into triples and distributes across KB layers.

Writes verified facts into:
  - WorldModel.add_fact() (domain-indexed fact strings)
  - WorldModel.update_belief() (structured subject/predicate/object — queryable)
  - KnowledgeGraph.add_belief() (belief nodes for later graph reasoning)
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

# Lazy singleton for ConceptMemory — reuse across ingest() calls
_concept_memory = None


def _get_concept_memory():
    global _concept_memory
    if _concept_memory is None:
        try:
            from sare.memory.concept_formation import ConceptMemory
            _concept_memory = ConceptMemory()
            _concept_memory.load()
        except Exception:
            pass
    return _concept_memory

# Triple extraction patterns (question + answer → subject/predicate/object)
_WHAT_OF_RE  = re.compile(r"what is (?:the )?(\w[\w\s]*?) of (.+?)[\?\.]*$", re.IGNORECASE)
_DEF_RE      = re.compile(r"(?:define|what is|what are) (.+?)[\?\.]*$", re.IGNORECASE)
_CAPITAL_RE  = re.compile(r"(?:capital|capitol) of (.+?)[\?\.]*$", re.IGNORECASE)
_SPEED_RE    = re.compile(r"speed of (.+?)[\?\.]*$", re.IGNORECASE)
_FORMULA_RE  = re.compile(r"formula for (.+?)[\?\.]*$", re.IGNORECASE)
_WHO_RE      = re.compile(r"who (?:is|was|invented|discovered|wrote) (.+?)[\?\.]*$", re.IGNORECASE)
_WHEN_RE     = re.compile(r"when (?:was|did|is) (.+?)[\?\.]*$", re.IGNORECASE)
_HOW_MANY_RE = re.compile(r"how many (.+?)[\?\.]*$", re.IGNORECASE)


def _extract_triples(question: str, answer: str) -> List[Tuple[str, str, str]]:
    """Extract (subject, predicate, object) triples from a Q&A pair.

    Returns structured triples that enable direct KB lookup without LLM.
    """
    triples: List[Tuple[str, str, str]] = []
    q = question.strip()

    # Pattern: "what is the X of Y?" → (Y, X, answer)
    m = _WHAT_OF_RE.search(q)
    if m:
        predicate = m.group(1).strip().lower().replace(" ", "_")
        subject   = m.group(2).strip().lower()
        triples.append((subject, predicate, answer.strip()))

    # Pattern: "capital of X?" → (X, capital, answer)
    if not triples:
        m = _CAPITAL_RE.search(q)
        if m:
            subject = m.group(1).strip().lower()
            triples.append((subject, "capital", answer.strip()))

    # Pattern: "speed of X?" → (X, speed, answer)
    if not triples:
        m = _SPEED_RE.search(q)
        if m:
            subject = m.group(1).strip().lower()
            triples.append((subject, "speed", answer.strip()))

    # Pattern: "formula for X?" → (X, formula, answer)
    if not triples:
        m = _FORMULA_RE.search(q)
        if m:
            subject = m.group(1).strip().lower()
            triples.append((subject, "formula", answer.strip()))

    # Pattern: "who invented/is X?" → (X, inventor/is, answer)
    if not triples:
        m = _WHO_RE.search(q)
        if m:
            subject = m.group(1).strip().lower()
            triples.append((subject, "who", answer.strip()))

    # Pattern: "when was/did X?" → (X, when, answer)
    if not triples:
        m = _WHEN_RE.search(q)
        if m:
            subject = m.group(1).strip().lower()
            triples.append((subject, "when", answer.strip()))

    # Pattern: "how many X?" → (X, count, answer)
    if not triples:
        m = _HOW_MANY_RE.search(q)
        if m:
            subject = m.group(1).strip().lower()
            triples.append((subject, "count", answer.strip()))

    # Pattern: "define/what is X?" → (X, definition, answer)
    if not triples:
        m = _DEF_RE.search(q)
        if m:
            subject = m.group(1).strip().lower()
            triples.append((subject, "definition", answer.strip()))

    # Fallback: use truncated question as subject
    if not triples:
        subject = q[:40].strip().lower()
        triples.append((subject, "answer", answer.strip()))

    return triples


# ── Spatial perception ────────────────────────────────────────────────────────

_SPATIAL_PATTERNS = [
    (re.compile(r'\b(.+?)\s+is\s+above\s+(.+)',     re.I), "above"),
    (re.compile(r'\b(.+?)\s+is\s+below\s+(.+)',     re.I), "below"),
    (re.compile(r'\b(.+?)\s+is\s+inside\s+(.+)',    re.I), "inside"),
    (re.compile(r'\b(.+?)\s+is\s+next\s+to\s+(.+)', re.I), "next_to"),
    (re.compile(r'\b(.+?)\s+is\s+near\s+(.+)',      re.I), "near"),
    (re.compile(r'\b(.+?)\s+contains\s+(.+)',       re.I), "contains"),
    (re.compile(r'\b(.+?)\s+surrounds\s+(.+)',      re.I), "surrounds"),
]


def _extract_spatial_triples(text: str) -> List[Tuple[str, str, str]]:
    """Extract spatial relation triples from free text.
    Returns (subject, 'spatial_relation', 'above:object') triples."""
    triples: List[Tuple[str, str, str]] = []
    for pat, rel_type in _SPATIAL_PATTERNS:
        m = pat.search(text)
        if m:
            subj = m.group(1).strip().lower()[:40]
            obj  = m.group(2).strip().lower()[:40]
            if subj and obj:
                triples.append((subj, "spatial_relation", f"{rel_type}:{obj}"))
    return triples


class FactIngester:
    """
    Ingest a Q&A pair into the persistent KB layers.
    Thread-safe (each call creates fresh references to singletons).
    """

    def ingest(self, question: str, answer: str, domain: str,
               confidence: float = 0.75) -> int:
        """
        Parse Q&A into triples and store across KB layers.
        Returns the number of new facts stored.
        """
        if not question or not answer or answer == "[no solver available]":
            return 0
        # Skip very short/useless answers
        if len(answer.strip()) < 2:
            return 0

        triples = _extract_triples(question, answer)
        # Gap-5: add spatial perception triples from free-text input
        triples.extend(_extract_spatial_triples(question + " " + answer))
        stored  = 0

        for subject, predicate, obj in triples:
            # Human-readable fact string for unstructured search
            fact_str = f"{subject} {predicate}: {obj}"

            # 1. Store in WorldModel as a domain fact (unstructured, for existing search)
            try:
                from sare.memory.world_model import get_world_model
                wm = get_world_model()
                wm.add_fact(domain=domain, fact=fact_str, confidence=confidence)
                # Also store as a structured belief (subject/predicate/value) for direct lookup
                belief_key = f"{subject}::{predicate}"
                if hasattr(wm, 'update_belief'):
                    wm.update_belief(
                        subject=subject,
                        predicate=predicate,
                        value=obj,
                        confidence=confidence,
                        domain=domain,
                    )
                stored += 1
                # Build a graph episode for concept clustering (language → graph)
                try:
                    from sare.perception.graph_builders import SentenceGraphBuilder
                    _g = SentenceGraphBuilder().build_from_triple(subject, predicate, obj, hide="object")
                    _cm = _get_concept_memory()
                    if _cm is not None:
                        _cm.record(_g, f"fi_{subject}_{predicate}", [predicate])
                except Exception:
                    pass
            except Exception as e:
                log.debug("[FactIngester] WorldModel store failed: %s", e)

            # 2. Store in KnowledgeGraph as a belief node
            try:
                from sare.memory.knowledge_graph import KnowledgeGraph
                kg = KnowledgeGraph()
                key = f"{subject}:{predicate}"
                kg.add_belief(key=key, confidence=confidence, domain=domain,
                              description=fact_str)
                kg.save()
            except Exception as e:
                log.debug("[FactIngester] KnowledgeGraph store failed: %s", e)

        return stored
