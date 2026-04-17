"""
GlobalBuffer — Cross-session Working Memory for Global Workspace

Miller (1956) established the capacity of working memory at 7±2 items.
This module implements a broadcast-aware persistent buffer that:

  - Receives all GlobalWorkspace broadcasts above threshold
  - Maintains exactly CAPACITY items (default 7)
  - Applies temporal decay: old items fade in salience
  - Promotes the most salient item to "attention spotlight"
  - Exposes context to Brain.solve() for context-aware reasoning

The key difference from WorkingMemory (solve-time) is that GlobalBuffer
persists *between* solves, accumulating cross-session contextual state.

Wiring:
  gw.subscribe('global_buffer', buffer.receive)
  buffer.tick()                          # called every learn_cycle
  context = buffer.get_active_context()  # available to solve()
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

CAPACITY      = 7      # Miller's magic number: 7±2 slots
DECAY_RATE    = 0.08   # salience lost per cognitive cycle (≈ 12 cycles to fade)
MIN_SALIENCE  = 0.05   # items below this are evicted
BOOST_RECENCY = 0.15   # salience boost when an item is accessed again


@dataclass
class BufferItem:
    """One item held in the global buffer."""
    content:    Any
    source:     str
    msg_type:   str
    salience:   float
    timestamp:  float = field(default_factory=time.time)
    access_count: int = 0

    def decay(self, rate: float = DECAY_RATE):
        self.salience = max(0.0, self.salience - rate)

    def boost(self, amount: float = BOOST_RECENCY):
        self.salience = min(1.0, self.salience + amount)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def to_dict(self) -> dict:
        payload = self.content
        if hasattr(payload, "to_dict"):
            payload = payload.to_dict()
        elif not isinstance(payload, (dict, str, int, float, list)):
            payload = str(payload)
        return {
            "content":      payload,
            "source":       self.source,
            "msg_type":     self.msg_type,
            "salience":     round(self.salience, 3),
            "access_count": self.access_count,
            "age_s":        round(self.age_seconds, 1),
        }


class GlobalBuffer:
    """
    Cross-session working memory buffer wired to GlobalWorkspace.

    Usage (Brain._boot_knowledge)::

        self.global_buffer = GlobalBuffer()
        if self.workspace:
            self.workspace.subscribe('global_buffer',
                                     self.global_buffer.receive)

    Usage (Brain.learn_cycle)::

        self.global_buffer.tick()       # decay + evict

    Usage (Brain.solve)::

        ctx = self.global_buffer.get_active_context()
        # pass ctx to engine for context-aware transform selection
    """

    def __init__(self, capacity: int = CAPACITY):
        self._capacity = capacity
        self._slots: List[BufferItem] = []           # active items
        self._evicted: deque = deque(maxlen=50)      # recently evicted (for recall)
        self._attention: Optional[BufferItem] = None  # current spotlight
        self._total_received  = 0
        self._total_evicted   = 0
        self._tick_count      = 0
        self._cycle_broadcasts: deque = deque(maxlen=20)  # last 20 spotlight items

    # ── Receive from GlobalWorkspace ───────────────────────────────────────────

    def receive(self, msg) -> None:
        """Handler registered with GlobalWorkspace.subscribe()."""
        self._total_received += 1
        content = msg.payload if hasattr(msg, "payload") else msg
        item = BufferItem(
            content   = content,
            source    = msg.source_module if hasattr(msg, "source_module") else "?",
            msg_type  = msg.msg_type if hasattr(msg, "msg_type") else "event",
            salience  = msg.salience if hasattr(msg, "salience") else 0.5,
        )
        # Check if a similar item is already in buffer → boost instead
        for existing in self._slots:
            if (existing.msg_type == item.msg_type and
                    existing.source == item.source):
                existing.boost()
                existing.access_count += 1
                return
        self._add(item)

    def _add(self, item: BufferItem) -> None:
        """Add item; evict lowest-salience if over capacity."""
        self._slots.append(item)
        if len(self._slots) > self._capacity:
            # Evict the lowest-salience item
            self._slots.sort(key=lambda x: x.salience, reverse=True)
            evicted = self._slots.pop()
            self._evicted.append(evicted)
            self._total_evicted += 1

    # ── Cognitive cycle tick ───────────────────────────────────────────────────

    def tick(self) -> None:
        """
        Called every learn_cycle. Apply decay, evict faded items,
        update attention spotlight.
        """
        self._tick_count += 1
        # Decay all items
        for item in self._slots:
            item.decay()
        # Evict items below minimum salience
        before = len(self._slots)
        live = [it for it in self._slots if it.salience >= MIN_SALIENCE]
        dead = [it for it in self._slots if it.salience < MIN_SALIENCE]
        for d in dead:
            self._evicted.append(d)
            self._total_evicted += 1
        self._slots = live
        # Sort by salience
        self._slots.sort(key=lambda x: x.salience, reverse=True)
        # Set attention spotlight = highest salience item
        if self._slots:
            self._attention = self._slots[0]
            self._cycle_broadcasts.append({
                "tick":     self._tick_count,
                "focus":    self._attention.msg_type,
                "source":   self._attention.source,
                "salience": round(self._attention.salience, 3),
            })
        else:
            self._attention = None

    # ── Context API ───────────────────────────────────────────────────────────

    def get_active_context(self) -> Dict[str, Any]:
        """
        Return a context dict for use in Brain.solve() / engine.
        Includes: attention focus, top active items, relevant domains.
        """
        items = self._slots[:5]
        domains = list({
            it.content.get("domain", "")
            for it in items
            if isinstance(it.content, dict) and it.content.get("domain")
        })
        expressions = [
            it.content.get("expression") or it.content.get("result", "")
            for it in items
            if isinstance(it.content, dict)
        ]
        return {
            "attention_type":    self._attention.msg_type if self._attention else None,
            "attention_source":  self._attention.source  if self._attention else None,
            "active_domains":    domains,
            "recent_expressions": [e for e in expressions if e][:3],
            "buffer_size":       len(self._slots),
            "spotlight_salience": round(self._attention.salience, 3) if self._attention else 0.0,
        }

    def get_domain_focus(self) -> Optional[str]:
        """Return the most recently active domain from the buffer."""
        for item in self._slots:
            if isinstance(item.content, dict):
                d = item.content.get("domain")
                if d:
                    return d
        return None

    def recall(self, msg_type: str) -> List[BufferItem]:
        """Search active + evicted buffer for items of a given msg_type."""
        active  = [it for it in self._slots   if it.msg_type == msg_type]
        evicted = [it for it in self._evicted if it.msg_type == msg_type]
        return (active + evicted)[:5]

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "capacity":         self._capacity,
            "active_items":     len(self._slots),
            "total_received":   self._total_received,
            "total_evicted":    self._total_evicted,
            "tick_count":       self._tick_count,
            "attention_focus":  self._attention.to_dict() if self._attention else None,
            "slots":            [it.to_dict() for it in self._slots],
            "recent_spotlights": list(self._cycle_broadcasts)[-8:],
        }
