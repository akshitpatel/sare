"""
DialogueContext — Multi-turn Conversation Tracker

Real language understanding requires tracking context across turns:
  - Pronoun resolution: "it", "that", "this" → most recent entity
  - Entity tracking: which concepts have been mentioned
  - Topic drift: is the user changing subject?
  - Intent history: what kind of requests have been made

This module is wired into Brain.parse() to provide context-aware
interpretation of natural language inputs.

Wiring:
    ctx = DialogueContext()
    ctx.add_turn("user", "What is x + 0?")
    resolved = ctx.resolve("Simplify it")   → "Simplify x + 0"
    ctx.add_turn("brain", "x + 0 = x")
    ctx.get_context_entities()              → ["x", "0", "x + 0"]
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

_PRONOUNS = {"it", "that", "this", "those", "these", "them", "they",
             "the result", "the answer", "the expression"}

_MATH_PATTERN = re.compile(
    r'[a-zA-Z_]\w*\s*[\+\-\*/\^=]\s*[\w\d\(\)]|'
    r'\b(?:x|y|z|a|b|n)\b\s*[\+\-\*/\^]|'
    r'\d+\s*[\+\-\*/\^]\s*\d+'
)

_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "solve":     ["solve", "simplify", "compute", "evaluate", "calculate", "find"],
    "explain":   ["explain", "why", "how", "what is", "describe", "tell me"],
    "compare":   ["compare", "difference", "versus", "vs", "better"],
    "generate":  ["generate", "create", "make", "produce", "give me"],
    "recall":    ["remember", "recall", "what was", "earlier", "previous"],
}


def _detect_intent(text: str) -> str:
    t = text.lower()
    for intent, kws in _INTENT_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return intent
    return "general"


def _extract_entities(text: str) -> List[str]:
    """Extract math-like entities and quoted terms from text."""
    entities: List[str] = []
    # Math tokens
    for m in _MATH_PATTERN.finditer(text):
        tok = m.group().strip()
        if len(tok) >= 2:
            entities.append(tok)
    # Variables (single letters used as expressions)
    for v in re.findall(r'\b([a-z])\b', text):
        if v not in entities:
            entities.append(v)
    # Quoted terms
    for q in re.findall(r'"([^"]+)"', text):
        entities.append(q)
    # Bare domain keywords
    for kw in ["logic", "arithmetic", "calculus", "physics", "algebra", "geometry"]:
        if kw in text.lower() and kw not in entities:
            entities.append(kw)
    return list(dict.fromkeys(entities))[:8]  # dedupe, max 8


def _compute_topic_similarity(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = len(tokens_a & tokens_b)
    union   = len(tokens_a | tokens_b)
    return overlap / max(union, 1)


@dataclass
class Turn:
    """One dialogue turn."""
    speaker:   str                  # "user" | "brain"
    text:      str
    entities:  List[str]            = field(default_factory=list)
    intent:    str                  = "general"
    domain:    Optional[str]        = None
    timestamp: float                = field(default_factory=time.time)

    def token_set(self) -> Set[str]:
        return set(re.findall(r'\b\w+\b', self.text.lower()))

    def to_dict(self) -> dict:
        return {
            "speaker":   self.speaker,
            "text":      self.text[:80],
            "entities":  self.entities[:4],
            "intent":    self.intent,
            "domain":    self.domain,
        }


class DialogueContext:
    """
    Maintains a sliding window of recent turns for context-aware NLU.

    Usage::
        ctx = DialogueContext(window=8)
        ctx.add_turn("user", "Solve x + 0")
        ctx.add_turn("brain", "x + 0 = x (add_zero_elim)")
        resolved = ctx.resolve("Now simplify it * 1")
        # → "Now simplify x * 1"
    """

    DRIFT_THRESHOLD = 0.15   # similarity below this → topic drift

    def __init__(self, window: int = 8):
        self._window  = window
        self._turns: deque = deque(maxlen=window)
        self._entity_history: deque = deque(maxlen=40)   # all mentioned entities
        self._topic_tokens: Set[str] = set()             # current topic tokens
        self._drift_events: List[dict] = []
        self._total_turns   = 0
        self._total_drifts  = 0
        self._session_start = time.time()

    # ── Add turn ──────────────────────────────────────────────────────────────

    def add_turn(self, speaker: str, text: str,
                 domain: Optional[str] = None) -> Turn:
        """Record a new dialogue turn."""
        entities = _extract_entities(text)
        intent   = _detect_intent(text) if speaker == "user" else "response"
        # Auto-detect domain from entities
        if not domain:
            for kw in ["logic", "arithmetic", "calculus", "physics", "algebra"]:
                if kw in text.lower():
                    domain = kw
                    break
        turn = Turn(speaker=speaker, text=text, entities=entities,
                    intent=intent, domain=domain)
        # Check topic drift
        new_tokens = turn.token_set()
        if self._topic_tokens:
            sim = _compute_topic_similarity(self._topic_tokens, new_tokens)
            if sim < self.DRIFT_THRESHOLD and speaker == "user":
                self._total_drifts += 1
                self._drift_events.append({
                    "from_topic": list(self._topic_tokens)[:3],
                    "to_topic":   list(new_tokens)[:3],
                    "similarity": round(sim, 3),
                    "turn": self._total_turns,
                })
                self._topic_tokens = new_tokens   # reset topic
            else:
                self._topic_tokens |= new_tokens
        else:
            self._topic_tokens = new_tokens
        for e in entities:
            if e not in self._entity_history:
                self._entity_history.appendleft(e)
        self._turns.append(turn)
        self._total_turns += 1
        return turn

    # ── Resolve pronouns ──────────────────────────────────────────────────────

    def resolve(self, text: str) -> str:
        """
        Replace pronouns in text with the most recently mentioned entity.
        E.g. "Simplify it" → "Simplify x + 0"
        """
        resolved = text
        candidate = self._most_recent_entity()
        if not candidate:
            return resolved
        for pronoun in sorted(_PRONOUNS, key=len, reverse=True):
            if pronoun in resolved.lower():
                pattern = re.compile(re.escape(pronoun), re.IGNORECASE)
                resolved = pattern.sub(candidate, resolved, count=1)
                break
        return resolved

    def _most_recent_entity(self) -> Optional[str]:
        """Return the most recently mentioned mathematical entity."""
        # Prefer entities from the most recent brain response (= last answer)
        for turn in reversed(list(self._turns)):
            if turn.speaker == "brain" and turn.entities:
                return turn.entities[0]
        # Fall back to most recent user entity
        for turn in reversed(list(self._turns)):
            if turn.entities:
                return turn.entities[0]
        return list(self._entity_history)[0] if self._entity_history else None

    # ── Context queries ───────────────────────────────────────────────────────

    def get_context_entities(self, n: int = 5) -> List[str]:
        """All unique entities mentioned in the recent context window."""
        seen: List[str] = []
        for turn in reversed(list(self._turns)):
            for e in turn.entities:
                if e not in seen:
                    seen.append(e)
                if len(seen) >= n:
                    return seen
        return seen

    def current_domain(self) -> Optional[str]:
        """Most recently mentioned domain."""
        for turn in reversed(list(self._turns)):
            if turn.domain:
                return turn.domain
        return None

    def last_intent(self) -> str:
        """Most recent user intent."""
        for turn in reversed(list(self._turns)):
            if turn.speaker == "user":
                return turn.intent
        return "general"

    def is_continuation(self) -> bool:
        """True if the current turn continues the same topic."""
        return self._total_drifts == 0 or (
            len(self._drift_events) > 0 and
            self._drift_events[-1]["turn"] < self._total_turns - 2
        )

    def get_context_for_parse(self) -> dict:
        """Return a context dict suitable for passing to NL parser."""
        return {
            "recent_entities":  self.get_context_entities(4),
            "current_domain":   self.current_domain(),
            "last_intent":      self.last_intent(),
            "is_continuation":  self.is_continuation(),
            "turn_count":       self._total_turns,
        }

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        recent = [t.to_dict() for t in list(self._turns)[-5:]]
        return {
            "total_turns":      self._total_turns,
            "window_size":      self._window,
            "active_turns":     len(self._turns),
            "total_drifts":     self._total_drifts,
            "current_domain":   self.current_domain(),
            "last_intent":      self.last_intent(),
            "context_entities": self.get_context_entities(6),
            "recent_turns":     recent,
            "drift_events":     self._drift_events[-3:],
            "session_age_s":    round(time.time() - self._session_start, 1),
        }
