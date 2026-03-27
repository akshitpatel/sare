"""
FactIngester — Parses Q&A pairs into triples and distributes across KB layers.

Writes verified facts into:
  - WorldModel.add_fact() (domain-indexed fact strings)
  - KnowledgeGraph.add_belief() (belief nodes for later graph reasoning)
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

log = logging.getLogger(__name__)

# Triple extraction patterns (question + answer → subject/predicate/object)
_WHAT_OF_RE = re.compile(r"what is (?:the )?(\w[\w\s]*?) of (.+?)[\?\.]*$", re.IGNORECASE)
_DEF_RE     = re.compile(r"(?:define|what is|what are) (.+?)[\?\.]*$", re.IGNORECASE)
_IS_RE      = re.compile(r"(.+?) is (?:defined as |known as )?(.+)", re.IGNORECASE)


def _extract_triples(question: str, answer: str) -> List[Tuple[str, str, str]]:
    """Extract (subject, predicate, object) triples from a Q&A pair."""
    triples: List[Tuple[str, str, str]] = []

    m = _WHAT_OF_RE.search(question)
    if m:
        predicate = m.group(1).strip().lower()
        subject   = m.group(2).strip().lower()
        triples.append((subject, predicate, answer.strip()))

    if not triples:
        m = _DEF_RE.search(question)
        if m:
            subject = m.group(1).strip().lower()
            triples.append((subject, "answer", answer.strip()))

    if not triples:
        # Fallback: use truncated question as subject
        subject = question[:40].strip().lower()
        triples.append((subject, "answer", answer.strip()))

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

        triples = _extract_triples(question, answer)
        stored  = 0

        for subject, predicate, obj in triples:
            # Build a human-readable fact string for WorldModel
            fact_str = f"{subject} {predicate}: {obj}"

            # 1. Store in WorldModel as a domain fact
            try:
                from sare.memory.world_model import get_world_model
                wm = get_world_model()
                wm.add_fact(domain=domain, fact=fact_str, confidence=confidence)
                stored += 1
            except Exception as e:
                log.debug("[FactIngester] WorldModel store failed: %s", e)

            # 2. Store in KnowledgeGraph as a belief node
            try:
                from sare.memory.knowledge_graph import KnowledgeGraph
                kg = KnowledgeGraph()
                key = f"{subject}:{predicate}"
                kg.add_belief(key=key, confidence=confidence, domain=domain,
                              description=f"{fact_str}")
                kg.save()
            except Exception as e:
                log.debug("[FactIngester] KnowledgeGraph store failed: %s", e)

        return stored
