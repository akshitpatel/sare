"""
SelfVerifier — checks LLM answers for consistency before KB storage.

Verification strategies:
  - Math/arithmetic: answer must begin with a valid numeric expression
  - Factual/science: check for contradiction against existing WorldModel beliefs
  - Default: accept with moderate confidence multiplier
"""

from __future__ import annotations

import re
import logging
from typing import Optional, Tuple

log = logging.getLogger(__name__)

# Reuse the same patterns as FactIngester for subject/predicate extraction
_WHAT_OF_RE  = re.compile(r"what is (?:the )?(\w[\w\s]*?) of (.+?)[\?\.]*$", re.IGNORECASE)
_CAPITAL_RE  = re.compile(r"(?:capital|capitol) of (.+?)[\?\.]*$", re.IGNORECASE)
_SPEED_RE    = re.compile(r"speed of (.+?)[\?\.]*$", re.IGNORECASE)
_FORMULA_RE  = re.compile(r"formula for (.+?)[\?\.]*$", re.IGNORECASE)
_DEF_RE      = re.compile(r"(?:define|what is|what are) (.+?)[\?\.]*$", re.IGNORECASE)


def _extract_subject_predicate(question: str) -> Tuple[Optional[str], Optional[str]]:
    q = question.strip()
    m = _WHAT_OF_RE.search(q)
    if m:
        return m.group(2).strip().lower(), m.group(1).strip().lower().replace(" ", "_")
    m = _CAPITAL_RE.search(q)
    if m:
        return m.group(1).strip().lower(), "capital"
    m = _SPEED_RE.search(q)
    if m:
        return m.group(1).strip().lower(), "speed"
    m = _FORMULA_RE.search(q)
    if m:
        return m.group(1).strip().lower(), "formula"
    m = _DEF_RE.search(q)
    if m:
        return m.group(1).strip().lower(), "definition"
    return None, None


def _semantic_distance(a: str, b: str) -> float:
    """Token overlap distance: 0.0 = identical, 1.0 = no overlap."""
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    if not a_words and not b_words:
        return 0.0
    overlap = len(a_words & b_words)
    union = len(a_words | b_words)
    return 1.0 - (overlap / max(union, 1))


class SelfVerifier:
    """Verify LLM answers for consistency before storing in KB."""

    def verify(self, question: str, answer: str, domain: str) -> Tuple[bool, float]:
        """Returns (is_consistent, confidence_multiplier).

        confidence_multiplier is applied to the base ingest confidence.
        """
        if not answer or not answer.strip():
            return False, 0.0

        if domain in ("math", "arithmetic", "algebra"):
            return self._verify_symbolic(answer)
        if domain in ("factual", "science"):
            return self._verify_factual(question, answer)
        return True, 0.9

    def _verify_symbolic(self, answer: str) -> Tuple[bool, float]:
        """Accept if answer leads with a valid number or symbolic expression."""
        clean = answer.strip().split()[0] if answer.strip() else ""
        try:
            float(clean.rstrip(',.'))
            return True, 1.0
        except ValueError:
            pass
        # Accept symbolic fractions, units, variable forms
        if re.match(r'^[\d\.\-\+\/\^xXyYzZ\s]+$', clean):
            return True, 0.9
        return False, 0.3

    def _verify_factual(self, question: str, answer: str) -> Tuple[bool, float]:
        """Check for contradiction against existing WorldModel beliefs."""
        subject, predicate = _extract_subject_predicate(question)
        if not subject or not predicate:
            return True, 0.85
        try:
            from sare.memory.world_model import get_world_model
            existing = get_world_model().get_belief(subject, predicate)
            if existing is None or not existing.value:
                return True, 0.9
            # If high-confidence existing belief differs significantly, reject
            if (existing.confidence >= 0.8
                    and _semantic_distance(existing.value, answer) > 0.7):
                log.debug(
                    "[SelfVerifier] Contradiction: q=%s, existing=%s, new=%s",
                    question[:60], existing.value[:40], answer[:40],
                )
                return False, 0.2
        except Exception:
            pass
        return True, 1.0


_VERIFIER_SINGLETON: Optional[SelfVerifier] = None


def get_self_verifier() -> SelfVerifier:
    global _VERIFIER_SINGLETON
    if _VERIFIER_SINGLETON is None:
        _VERIFIER_SINGLETON = SelfVerifier()
    return _VERIFIER_SINGLETON
