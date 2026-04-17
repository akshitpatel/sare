"""
InnerMonologue — SARE-HX's stream of consciousness.

A rolling buffer of the system's reasoning thoughts, with timestamp,
context tag, and emotional tone. Exposed via /api/mind/stream endpoint.
Gives the system the ability to observe its own reasoning.
"""

from __future__ import annotations

import logging
import time
import threading
from collections import deque
from typing import List, Optional

log = logging.getLogger(__name__)


class InnerMonologue:
    """Rolling buffer of the system's reasoning stream."""

    MAX_ENTRIES = 500  # last 500 thoughts

    def __init__(self):
        self._thoughts: deque = deque(maxlen=self.MAX_ENTRIES)
        self._lock = threading.Lock()

    # Minimum seconds before the same (context, thought-prefix) can be re-logged
    DEDUP_WINDOW = 60.0

    def think(self, thought: str, context: str = "", emotion: str = "neutral"):
        """Record a thought with timestamp, context tag, and emotional tone."""
        now = time.time()
        key = (context, thought[:80])
        with self._lock:
            # Suppress exact-duplicate messages within the dedup window
            if self._thoughts:
                last = self._thoughts[-1]
                if (last.get("context") == context
                        and last.get("thought", "")[:80] == thought[:80]
                        and now - last["timestamp"] < self.DEDUP_WINDOW):
                    return
            entry = {
                "timestamp": now,
                "thought": thought,
                "context": context,
                "emotion": emotion,
            }
            self._thoughts.append(entry)
        log.debug("[InnerMonologue] %s: %s", context or "thought", thought[:80])

    def reflect(self, topic: str) -> str:
        """Retrieve recent thoughts about a topic (for metacognition)."""
        topic_lower = topic.lower()
        with self._lock:
            relevant = [
                t for t in self._thoughts
                if topic_lower in t["thought"].lower()
                or topic_lower in t.get("context", "").lower()
            ]
        if not relevant:
            return f"No recent thoughts about '{topic}'."
        recent = relevant[-5:]
        lines = [f"[{t['context']}] {t['thought']}" for t in recent]
        return "\n".join(lines)

    def narrate(self, last_n: int = 20) -> str:
        """Return human-readable narrative of recent thinking."""
        with self._lock:
            recent = list(self._thoughts)[-last_n:]
        if not recent:
            return "Mind is quiet — no recent thoughts."
        lines = []
        for t in recent:
            ts = time.strftime("%H:%M:%S", time.localtime(t["timestamp"]))
            ctx = f"[{t['context']}] " if t.get("context") else ""
            emo = f" ({t['emotion']})" if t.get("emotion", "neutral") != "neutral" else ""
            lines.append(f"{ts} {ctx}{t['thought']}{emo}")
        return "\n".join(lines)

    def get_stream(self, last_n: int = 50) -> List[dict]:
        """For /api/mind/stream endpoint — returns last N thoughts."""
        with self._lock:
            return list(self._thoughts)[-last_n:]

    def get_stats(self) -> dict:
        with self._lock:
            thoughts = list(self._thoughts)
        emotions = {}
        contexts = {}
        for t in thoughts:
            em = t.get("emotion", "neutral")
            emotions[em] = emotions.get(em, 0) + 1
            ctx = t.get("context", "general")
            contexts[ctx] = contexts.get(ctx, 0) + 1
        return {
            "total_thoughts": len(thoughts),
            "emotion_distribution": emotions,
            "top_contexts": sorted(contexts.items(), key=lambda x: -x[1])[:5],
            "latest_thought": thoughts[-1]["thought"] if thoughts else "",
        }


# ── Singleton ──────────────────────────────────────────────────────────────────

_INNER_MONOLOGUE: Optional[InnerMonologue] = None


def get_inner_monologue() -> InnerMonologue:
    global _INNER_MONOLOGUE
    if _INNER_MONOLOGUE is None:
        _INNER_MONOLOGUE = InnerMonologue()
    return _INNER_MONOLOGUE
