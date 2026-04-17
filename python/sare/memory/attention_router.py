"""
AttentionRouter — S28-2
Deeper Global Workspace attention routing.

Scores each broadcast event on three axes:
  relevance  — cosine-like token overlap with current focus
  recency    — exponential decay from time of posting
  novelty    — inverse frequency (rare event types get higher weight)

Top-K events form the "spotlight" — these are re-broadcast to registered
target modules with amplified salience, enabling cross-module coordination.

Cross-module routing table:
  surprise      → DreamConsolidator, AffectiveEnergy
  imagination   → TransformGenerator, ConceptGraph
  falsification → AgentSociety, RedTeam
  solve_success → TemporalIdentity, ContinuousStream
  high_energy   → AffectiveEnergy, GlobalBuffer
"""
from __future__ import annotations

import math
import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

_RECENCY_HALF_LIFE  = 30.0    # seconds; score halves every 30s
_SPOTLIGHT_TOP_K    = 5
_NOVELTY_WINDOW     = 200     # events tracked for frequency
_AMPLIFY_FACTOR     = 1.4     # salience boost for spotlight events
_RELEVANCE_TOKENS   = 6       # max tokens used for focus matching


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class RouterEvent:
    event_type: str
    payload:    dict
    source:     str
    salience:   float
    ts:         float = field(default_factory=time.time)
    routed_to:  List[str] = field(default_factory=list)

    def attention_score(self, focus_tokens: Set[str], freq: Dict[str, int]) -> float:
        age        = time.time() - self.ts
        recency    = math.exp(-age * math.log(2) / _RECENCY_HALF_LIFE)
        total_freq = max(1, freq.get(self.event_type, 1))
        novelty    = 1.0 / math.log1p(total_freq)
        relevance  = self._relevance(focus_tokens)
        return self.salience * (0.4 * relevance + 0.35 * recency + 0.25 * novelty)

    def _relevance(self, focus_tokens: Set[str]) -> float:
        if not focus_tokens:
            return 0.5
        evt_tokens = set(self.event_type.split("_"))
        pay_tokens: Set[str] = set()
        for v in self.payload.values():
            if isinstance(v, str):
                pay_tokens.update(v.lower().split())
        all_tokens = evt_tokens | pay_tokens
        if not all_tokens:
            return 0.0
        return len(focus_tokens & all_tokens) / len(focus_tokens | all_tokens)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "source":     self.source,
            "salience":   round(self.salience, 3),
            "routed_to":  self.routed_to,
            "age_s":      round(time.time() - self.ts, 1),
        }


# ── routing table ─────────────────────────────────────────────────────────────

_ROUTING_TABLE: Dict[str, List[str]] = {
    "surprise":         ["dream_consolidator", "affective_energy"],
    "dream_insight":    ["causal_graph", "world_model"],
    "imagination_solved":["transform_generator", "concept_graph"],
    "falsification":    ["agent_society", "red_team"],
    "solve_success":    ["temporal_identity", "continuous_stream"],
    "high_energy":      ["affective_energy", "global_buffer"],
    "blend_discovered": ["concept_graph", "global_buffer"],
    "belief_updated":   ["agent_society", "temporal_identity"],
}


# ── AttentionRouter ───────────────────────────────────────────────────────────

class AttentionRouter:
    """
    Sits between GlobalWorkspace and downstream modules.
    Scores all recent events, selects top-K spotlight events,
    re-broadcasts them to routed targets with amplified salience.
    """

    def __init__(self, spotlight_k: int = _SPOTLIGHT_TOP_K) -> None:
        self._spotlight_k    = spotlight_k
        self._events:  List[RouterEvent]              = []
        self._freq:    Dict[str, int]                 = defaultdict(int)
        self._focus:   Set[str]                       = set()
        self._targets: Dict[str, Callable]            = {}  # module_name → callback
        self._global_workspace = None

        self._total_routed   = 0
        self._total_received = 0
        self._spotlight_history: List[dict] = []

    # ── wiring ────────────────────────────────────────────────────────────────

    def wire(self, global_workspace=None) -> None:
        self._global_workspace = global_workspace
        if global_workspace:
            global_workspace.subscribe("attention_router", self.receive)

    def register_target(self, name: str, callback: Callable) -> None:
        """Register a downstream module to receive routed events."""
        self._targets[name] = callback

    def set_focus(self, tokens: List[str]) -> None:
        """Update current attention focus (e.g. current problem domain tokens)."""
        self._focus = set(t.lower() for t in tokens[:_RELEVANCE_TOKENS])

    # ── event ingestion ───────────────────────────────────────────────────────

    def receive(self, event_type: str, payload: dict,
                source: str = "", salience: float = 0.5) -> None:
        """Called by GlobalWorkspace.subscribe callback."""
        evt = RouterEvent(event_type, payload, source, salience)
        self._events.append(evt)
        self._freq[event_type] += 1
        self._total_received   += 1
        if len(self._events) > _NOVELTY_WINDOW:
            self._events.pop(0)

    def post(self, event_type: str, payload: dict,
             source: str = "", salience: float = 0.5) -> None:
        """Directly post an event (bypass GlobalWorkspace)."""
        self.receive(event_type, payload, source, salience)

    # ── routing tick ─────────────────────────────────────────────────────────

    def tick(self) -> List[RouterEvent]:
        """Score all events, select spotlight, route to targets. Returns spotlight."""
        if not self._events:
            return []

        scored = sorted(
            self._events,
            key=lambda e: e.attention_score(self._focus, self._freq),
            reverse=True,
        )
        spotlight = scored[:self._spotlight_k]

        for evt in spotlight:
            targets = _ROUTING_TABLE.get(evt.event_type, [])
            for t_name in targets:
                if t_name in self._targets:
                    try:
                        self._targets[t_name](
                            evt.event_type,
                            {**evt.payload, "_amplified": True,
                             "_salience": round(evt.salience * _AMPLIFY_FACTOR, 3)},
                            source="attention_router",
                            salience=min(1.0, evt.salience * _AMPLIFY_FACTOR),
                        )
                        if t_name not in evt.routed_to:
                            evt.routed_to.append(t_name)
                        self._total_routed += 1
                    except Exception as e:
                        log.debug(f"AttentionRouter route error to {t_name}: {e}")

        if spotlight:
            self._spotlight_history.extend([e.to_dict() for e in spotlight])
            if len(self._spotlight_history) > 50:
                self._spotlight_history = self._spotlight_history[-50:]

        return spotlight

    # ── accessors ─────────────────────────────────────────────────────────────

    @property
    def n_registered_targets(self) -> int:
        return len(self._targets)

    def current_spotlight(self) -> List[dict]:
        if not self._events:
            return []
        scored = sorted(
            self._events,
            key=lambda e: e.attention_score(self._focus, self._freq),
            reverse=True,
        )
        return [e.to_dict() for e in scored[:self._spotlight_k]]

    def summary(self) -> dict:
        return {
            "total_received":       self._total_received,
            "total_routed":         self._total_routed,
            "n_registered_targets": self.n_registered_targets,
            "spotlight_k":          self._spotlight_k,
            "current_focus":        list(self._focus),
            "event_freq":           dict(sorted(
                self._freq.items(), key=lambda x: -x[1])[:10]),
            "current_spotlight":    self.current_spotlight(),
            "recent_spotlight":     self._spotlight_history[-10:],
            "routing_table_size":   len(_ROUTING_TABLE),
        }
