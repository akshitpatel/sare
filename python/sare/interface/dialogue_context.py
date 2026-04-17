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
from typing import Dict, List, Optional, Set


_PRONOUNS = {
    "it",
    "that",
    "this",
    "those",
    "these",
    "them",
    "they",
    "the result",
    "the answer",
    "the expression",
}

_MATH_PATTERN = re.compile(
    r'[a-zA-Z_]\w*\s*[\+\-\*/\^=]\s*[\w\d\(\)]|'
    r'\b(?:x|y|z|a|b|n)\b\s*[\+\-\*/\^]|'
    r'\d+\s*[\+\-\*/\^]\s*\d+'
)

_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "solve": ["solve", "simplify", "compute", "evaluate", "calculate", "find"],
    "explain": ["explain", "why", "how", "what is", "describe", "tell me"],
    "compare": ["compare", "difference", "versus", "vs", "better"],
    "generate": ["generate", "create", "make", "produce", "give me"],
    "recall": ["remember", "recall", "what was", "earlier", "previous"],
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
    for m in _MATH_PATTERN.finditer(text):
        tok = m.group().strip()
        if len(tok) >= 2:
            entities.append(tok)
    for v in re.findall(r'\b([a-z])\b', text):
        if v not in entities:
            entities.append(v)
    for q in re.findall(r'"([^"]+)"', text):
        entities.append(q)
    for kw in ["logic", "arithmetic", "calculus", "physics", "algebra", "geometry"]:
        if kw in text.lower() and kw not in entities:
            entities.append(kw)
    return list(dict.fromkeys(entities))[:8]


def _compute_topic_similarity(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return overlap / max(union, 1)


@dataclass
class Turn:
    """One dialogue turn."""
    speaker: str
    text: str
    entities: List[str] = field(default_factory=list)
    intent: str = "general"
    domain: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def token_set(self) -> Set[str]:
        return set(re.findall(r'\b\w+\b', self.text.lower()))

    def to_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "text": self.text[:80],
            "entities": self.entities[:4],
            "intent": self.intent,
            "domain": self.domain,
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

    DRIFT_THRESHOLD = 0.15

    def __init__(self, window: int = 8):
        self._window = window
        self._turns: deque = deque(maxlen=window)
        self._entity_history: deque = deque(maxlen=40)
        self._topic_tokens: Set[str] = set()
        self._drift_events: List[dict] = []
        self._total_turns = 0
        self._total_drifts = 0
        self._session_start = time.time()
        self._most_recent_brain_entity: Optional[str] = None

    def add_turn(self, speaker: str, text: str, domain: Optional[str] = None) -> Turn:
        """Record a new dialogue turn."""
        entities = _extract_entities(text)
        intent = _detect_intent(text) if speaker == "user" else "response"
        if not domain:
            for kw in ["logic", "arithmetic", "calculus", "physics", "algebra"]:
                if kw in text.lower():
                    domain = kw
                    break
        turn = Turn(speaker=speaker, text=text, entities=entities, intent=intent, domain=domain)

        new_tokens = turn.token_set()
        if self._topic_tokens:
            sim = _compute_topic_similarity(self._topic_tokens, new_tokens)
            if sim < self.DRIFT_THRESHOLD and speaker == "user":
                self._total_drifts += 1
                self._drift_events.append(
                    {
                        "from_topic": list(self._topic_tokens)[:3],
                        "to_topic": list(new_tokens)[:3],
                        "similarity": round(sim, 3),
                        "turn": self._total_turns,
                    }
                )
                self._topic_tokens = new_tokens
            else:
                self._topic_tokens |= new_tokens
        else:
            self._topic_tokens = new_tokens

        for e in entities:
            if e not in self._entity_history:
                self._entity_history.appendleft(e)

        self._turns.append(turn)
        self._total_turns += 1

        if speaker == "brain" and entities:
            self._most_recent_brain_entity = entities[0]

        return turn

    def resolve(self, text: str) -> str:
        """Resolve simple pronouns against recent dialogue entities."""
        lowered = text.lower()

        has_pronoun = any(
            pron in lowered if " " in pron else re.search(r"\b" + re.escape(pron) + r"\b", lowered)
            for pron in _PRONOUNS
        )
        if not has_pronoun:
            return text

        replacement: Optional[str] = self._most_recent_brain_entity
        if replacement is None:
            for turn in reversed(self._turns):
                if turn.speaker == "brain" and turn.entities:
                    replacement = turn.entities[0]
                    break
            if replacement is None and self._entity_history:
                replacement = self._entity_history[0]

        if not replacement:
            return text

        resolved = text
        for pron in sorted(_PRONOUNS, key=len, reverse=True):
            if " " in pron:
                resolved = re.sub(re.escape(pron), replacement, resolved, flags=re.IGNORECASE)
            else:
                resolved = re.sub(r"\b" + re.escape(pron) + r"\b", replacement, resolved, flags=re.IGNORECASE)
        return resolved

    def get_context_entities(self) -> List[str]:
        """Return recently mentioned entities, most recent first."""
        return list(self._entity_history)

    def get_recent_turns(self) -> List[Turn]:
        """Return the current dialogue window."""
        return list(self._turns)

    def get_intent_history(self) -> List[str]:
        """Return intents for recent user turns."""
        return [t.intent for t in self._turns if t.speaker == "user"]

    def topic_drifted(self) -> bool:
        """Whether the latest user turn appears to have drifted topic."""
        if not self._turns:
            return False
        last = self._turns[-1]
        return last.speaker == "user" and any(
            event.get("turn") == self._total_turns - 1 for event in self._drift_events
        )

    def stats(self) -> dict:
        """Basic dialogue-context statistics."""
        return {
            "turns": self._total_turns,
            "drifts": self._total_drifts,
            "entities_tracked": len(self._entity_history),
            "window": self._window,
            "session_seconds": round(time.time() - self._session_start, 3),
        }

    def summary(self) -> dict:
        """Compact state summary for debugging."""
        return {
            "recent_turns": [t.to_dict() for t in self._turns],
            "entities": self.get_context_entities()[:8],
            "intent_history": self.get_intent_history()[-8:],
            "topic_tokens": list(self._topic_tokens)[:12],
            "drift_events": self._drift_events[-5:],
            "stats": self.stats(),
        }