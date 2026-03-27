"""
GlobalWorkspace — The Single Key Architectural Change

Global Workspace Theory (Baars, 1988) is the most empirically supported
model of consciousness and high-level cognition in neuroscience.

The core idea:
  - Each cognitive module (memory, reasoning, perception, planning, emotion)
    runs in parallel in its own "specialist" bubble
  - A global workspace is a shared broadcast medium
  - When one module broadcasts to the workspace, ALL other modules receive it
  - Only the most "salient" / high-priority item gets broadcast at a time
  - This creates the feeling of a unified, coherent cognitive state

Why this is the "single key change" for SARE-HX:

  Without it:  modules are silos; SelfModel doesn't know what ConceptGraph
               discovered; MetaLearner doesn't know about new conjectures

  With it:     every breakthrough immediately propagates to all modules;
               the brain operates as a unified cognitive system

Implementation:
  - WorkspaceMessage: typed message with salience score
  - GlobalWorkspace: priority queue + broadcast log
  - Modules register handlers; high-salience broadcasts wake them
  - Brain posts to workspace; all modules read it on their next tick

This is what separates "a bag of AI modules" from "a unified mind."
"""

from __future__ import annotations

import heapq
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)


# ── Message types ──────────────────────────────────────────────────────────────

MESSAGE_TYPES = {
    # Perception events
    "new_expression":      "A new expression has been perceived",
    "nl_parsed":           "Natural language successfully parsed to expression",
    "parse_failed":        "Parser could not understand input",
    # Reasoning events
    "solve_success":       "Problem solved — energy reduced significantly",
    "solve_failed":        "Problem could not be simplified further",
    "new_transform":       "A new transform was applied successfully",
    # Memory events
    "concept_grounded":    "A concept has a new grounded example",
    "concept_abstracted":  "A concept promoted to symbolic rule",
    "cross_domain_merge":  "Two concepts from different domains merged",
    # Learning events
    "conjecture_born":     "A new conjecture was generated",
    "conjecture_verified": "A conjecture was confirmed by evidence",
    "conjecture_falsified":"A conjecture was falsified",
    "transfer_promoted":   "A cross-domain transfer was verified and promoted",
    # Metacognition events
    "weakness_detected":   "SelfModel detected a weak domain",
    "goal_generated":      "A new learning goal was auto-generated",
    "goal_achieved":       "A learning goal was achieved",
    "strategy_updated":    "Best search strategy was updated",
    # World events
    "physics_event":       "Physics simulation generated an event",
    "knowledge_ingested":  "New knowledge was ingested from external text",
    # Meta-learning events
    "meta_tune_complete":  "MetaLearner promoted a new best config",
    "agent_race_winner":   "Multi-agent race completed; winner identified",
    "conjecture_debated":  "Multi-agent debate reached a verdict",
}


@dataclass
class WorkspaceMessage:
    """
    A message broadcast to the global workspace.

    salience: 0.0–1.0; only messages above the broadcast threshold
              get propagated to all listeners.
    """
    msg_type: str
    payload: Dict[str, Any]
    source_module: str
    salience: float          # 0.0 (background noise) → 1.0 (urgent)
    timestamp: float = field(default_factory=time.time)
    broadcast_count: int = 0

    def __lt__(self, other: "WorkspaceMessage") -> bool:
        return self.salience > other.salience   # max-heap via negation

    def to_dict(self) -> dict:
        return {
            "type": self.msg_type,
            "source": self.source_module,
            "salience": round(self.salience, 3),
            "summary": MESSAGE_TYPES.get(self.msg_type, self.msg_type),
            "payload_keys": list(self.payload.keys()),
            "ts": round(self.timestamp, 2),
        }


class GlobalWorkspace:
    """
    Shared broadcast medium connecting all cognitive modules.

    Usage:
        gw = GlobalWorkspace()
        gw.subscribe("memory", lambda msg: handle_memory(msg))

        gw.post(WorkspaceMessage(
            msg_type="solve_success",
            payload={"expression": "x+0", "result": "x", "delta": 1.5},
            source_module="engine",
            salience=0.8,
        ))

        # On next tick, all subscribers receive the high-salience message
        gw.broadcast_tick()
    """

    BROADCAST_THRESHOLD = 0.4    # only propagate if salience >= this

    def __init__(self, capacity: int = 200):
        self._queue: List[WorkspaceMessage] = []          # min-heap (salience desc)
        self._log: deque = deque(maxlen=capacity)          # full log
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._broadcast_log: List[WorkspaceMessage] = []
        self._stats: Dict[str, int] = defaultdict(int)
        self._attention_focus: Optional[WorkspaceMessage] = None

    # ── Post ────────────────────────────────────────────────────────────────────

    def post(self, msg: WorkspaceMessage):
        """Post a message to the workspace queue."""
        heapq.heappush(self._queue, msg)
        self._log.append(msg)
        self._stats[msg.msg_type] += 1
        log.debug(f"GW ← [{msg.source_module}] {msg.msg_type} (s={msg.salience:.2f})")

    def post_event(self, msg_type: str, payload: dict,
                   source: str, salience: float):
        """Convenience wrapper."""
        self.post(WorkspaceMessage(
            msg_type=msg_type,
            payload=payload,
            source_module=source,
            salience=salience,
        ))

    # ── Subscribe ──────────────────────────────────────────────────────────────

    def subscribe(self, module_name: str, handler: Callable[[WorkspaceMessage], None],
                  msg_types: Optional[List[str]] = None):
        """
        Register a module to receive broadcasts.

        If msg_types is given, only those message types are delivered.
        Otherwise the handler receives all broadcasts.
        """
        key = f"{module_name}:{'|'.join(msg_types) if msg_types else '*'}"
        self._subscribers[key].append(handler)
        log.debug(f"GW: {module_name} subscribed to {msg_types or 'all'}")

    # ── Broadcast tick ─────────────────────────────────────────────────────────

    def broadcast_tick(self, max_messages: int = 5) -> int:
        """
        Broadcast the top-salience messages to all subscribers.
        Called once per cognitive cycle.
        Returns number of messages broadcast.
        """
        broadcast_count = 0
        messages_sent: List[WorkspaceMessage] = []

        # Pop top-k highest-salience messages
        while self._queue and broadcast_count < max_messages:
            msg = heapq.heappop(self._queue)
            if msg.salience < self.BROADCAST_THRESHOLD:
                heapq.heappush(self._queue, msg)  # put back
                break

            msg.broadcast_count += 1
            self._attention_focus = msg
            messages_sent.append(msg)
            broadcast_count += 1

            # Deliver to all matching subscribers
            for key, handlers in self._subscribers.items():
                module_name, type_filter = key.split(":", 1)
                if type_filter == "*" or msg.msg_type in type_filter.split("|"):
                    for handler in handlers:
                        try:
                            handler(msg)
                        except Exception as e:
                            log.debug(f"GW handler {module_name} error: {e}")

        self._broadcast_log.extend(messages_sent)
        return broadcast_count

    # ── Attention focus ────────────────────────────────────────────────────────

    @property
    def current_focus(self) -> Optional[WorkspaceMessage]:
        """The most recently broadcast (highest-salience) message."""
        return self._attention_focus

    def peek_queue(self, n: int = 3) -> List[WorkspaceMessage]:
        """Non-destructively peek at the top-n pending messages."""
        if not self._queue:
            return []
        top = heapq.nsmallest(n, self._queue)  # smallest = highest salience
        return top

    # ── Cross-module integration ────────────────────────────────────────────────

    def wire_brain(self, brain) -> None:
        """
        Wire standard Brain module events to the GlobalWorkspace.

        Each module's key outputs get mapped to workspace messages with
        appropriate salience levels:
          - Solve success: 0.7 (important, inform all modules)
          - Conjecture: 0.5 (interesting, propagate to meta-modules)
          - Weakness: 0.8 (urgent, redirect learning resources)
          - Transfer: 0.75 (valuable, share with concept layer)
          - Physics: 0.4 (informative, ground concepts)
          - Knowledge: 0.5 (enriching, populate concept graph)
        """
        # Listen to Brain EventBus and relay to GlobalWorkspace
        if hasattr(brain, 'events') and brain.events:
            ev = brain.events

            def _on_solve(data):
                self.post_event(
                    "solve_success" if data.data.get("success") else "solve_failed",
                    data.data,
                    source="engine",
                    salience=0.7 if data.data.get("success") else 0.3,
                )

            def _on_concept(data):
                self.post_event("concept_grounded", data.data,
                                source="concept_graph", salience=0.5)

            def _on_transfer(data):
                self.post_event("transfer_promoted", data.data,
                                source="transfer_engine", salience=0.75)

            def _on_conjecture(data):
                self.post_event("conjecture_born", data.data,
                                source="reasoning", salience=0.5)

            def _on_weakness(data):
                self.post_event("weakness_detected", data.data,
                                source="self_model", salience=0.8)

            def _on_goal_achieved(data):
                self.post_event("goal_achieved", data.data,
                                source="self_model", salience=0.85)

            try:
                ev.subscribe("solve_completed",     _on_solve)
                ev.subscribe("concept_grounded",    _on_concept)
                ev.subscribe("transfer_promoted",   _on_transfer)
                ev.subscribe("conjecture_generated",_on_conjecture)
                ev.subscribe("weakness_detected",   _on_weakness)
                ev.subscribe("goal_achieved",       _on_goal_achieved)
                log.info("GlobalWorkspace wired to Brain EventBus")
            except Exception as e:
                log.debug(f"GW wire_brain: {e}")

    # ── Introspection ──────────────────────────────────────────────────────────

    def summary(self) -> dict:
        pending = len(self._queue)
        total_broadcast = len(self._broadcast_log)
        focus = self._attention_focus
        return {
            "pending_messages": pending,
            "total_broadcast": total_broadcast,
            "subscribers": len(self._subscribers),
            "attention_focus": focus.to_dict() if focus else None,
            "message_type_counts": dict(self._stats),
            "recent_broadcasts": [m.to_dict() for m in self._broadcast_log[-5:]],
            "top_queue": [m.to_dict() for m in self.peek_queue(3)],
        }
