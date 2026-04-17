"""
DreamConsolidator — S26-1
Offline hippocampal replay: during idle cycles replays recent surprise events
backwards in time, extracting latent causal edges not visible during waking.
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

log = logging.getLogger(__name__)


@dataclass
class DreamRecord:
    tick: int
    events_replayed: int
    causal_edges_found: int
    top_pattern: Optional[str]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "events_replayed": self.events_replayed,
            "causal_edges_found": self.causal_edges_found,
            "top_pattern": self.top_pattern,
            "timestamp": self.timestamp,
        }


@dataclass
class CausalDiscovery:
    antecedent: str
    consequent: str
    confidence: float
    source: str = "dream"

    def to_dict(self) -> dict:
        return {
            "antecedent": self.antecedent,
            "consequent": self.consequent,
            "confidence": round(self.confidence, 3),
            "source": self.source,
        }


class DreamConsolidator:
    """
    Subscribes to PredictiveWorldLoop._surprise_events and CausalGraph.
    On each dream_cycle(), replays events in reverse temporal order,
    computes co-occurrence windows to surface antecedent→consequent links.
    """

    _WINDOW = 3  # events before a surprise to search for antecedents
    _MIN_CONF = 0.4

    def __init__(self) -> None:
        self._predictive_loop = None
        self._causal_graph = None
        self._world_model = None

        self._dreams: List[DreamRecord] = []
        self._discoveries: List[CausalDiscovery] = []
        self._known_edges: set = set()

        self._tick_count = 0
        self._total_replayed = 0
        self._total_discovered = 0

    # ── wiring ────────────────────────────────────────────────────────────────
    def wire(self, predictive_loop=None, causal_graph=None, world_model=None):
        self._predictive_loop = predictive_loop
        self._causal_graph = causal_graph
        self._world_model = world_model

    # ── core ──────────────────────────────────────────────────────────────────
    def dream_cycle(self, max_events: int = 20) -> DreamRecord:
        """Run one consolidation pass. Returns a DreamRecord."""
        self._tick_count += 1
        events = self._collect_surprise_events(max_events)
        if not events:
            rec = DreamRecord(self._tick_count, 0, 0, None)
            self._dreams.append(rec)
            return rec

        # replay in reverse temporal order
        reversed_events = list(reversed(events))
        self._total_replayed += len(reversed_events)
        discoveries = self._extract_causal_edges(reversed_events)

        # deposit into causal graph / world model
        for d in discoveries:
            key = (d.antecedent, d.consequent)
            if key not in self._known_edges:
                self._known_edges.add(key)
                self._discoveries.append(d)
                self._total_discovered += 1
                self._deposit(d)

        top = discoveries[0].antecedent if discoveries else None
        rec = DreamRecord(
            self._tick_count,
            len(reversed_events),
            len(discoveries),
            top,
        )
        self._dreams.append(rec)
        if len(self._dreams) > 200:
            self._dreams = self._dreams[-200:]
        return rec

    def tick(self) -> None:
        """Called from Brain.learn_cycle every N cycles."""
        try:
            self.dream_cycle()
        except Exception as e:
            log.debug(f"DreamConsolidator tick: {e}")

    # ── internals ─────────────────────────────────────────────────────────────
    def _collect_surprise_events(self, n: int) -> List[Dict]:
        events = []
        if self._predictive_loop:
            raw = getattr(self._predictive_loop, "_surprise_events", [])
            events = [
                e.__dict__ if hasattr(e, "__dict__") else dict(e) for e in raw[-n:]
            ]
        # supplement with causal graph recent events
        if self._world_model:
            wm_events = getattr(self._world_model, "_recent_observations", [])
            for obs in wm_events[-n:]:
                events.append(
                    {
                        "transform": str(obs)[:40],
                        "actual_delta": 0.5,
                        "source": "world_model",
                    }
                )
        return events

    def _extract_causal_edges(self, events: List[Dict]) -> List[CausalDiscovery]:
        discoveries = []
        for i, surprise in enumerate(events):
            # window of events that immediately preceded this surprise
            window = events[i + 1 : i + 1 + self._WINDOW]
            s_transform = str(surprise.get("transform", "unknown"))
            for j, ante in enumerate(window):
                a_transform = str(ante.get("transform", "unknown"))
                if a_transform == s_transform:
                    continue
                # confidence decays with distance
                conf = 0.8 * (0.6**j)
                if conf >= self._MIN_CONF:
                    discoveries.append(
                        CausalDiscovery(
                            antecedent=a_transform,
                            consequent=s_transform,
                            confidence=conf,
                        )
                    )
        return discoveries

    def _broadcast_causal_discovery(self, d: CausalDiscovery) -> None:
        """
        Broadcast causal discoveries into the system's unified cognition medium
        if a GlobalWorkspace singleton is available. This closes the dream->live
        loop by making other modules immediately aware of new causal edges.
        """
        try:
            from sare.memory.global_workspace import get_global_workspace  # type: ignore

            gw = get_global_workspace()
            if not gw:
                return
            content = d.to_dict()
            # Use salience proportional to confidence; clamp to [0,1].
            salience = max(0.0, min(1.0, float(d.confidence)))
            gw.post(
                "CAUSAL_DISCOVERY",
                content=content,
                salience=salience,
                source=d.source,
            )
        except Exception:
            pass

    def _deposit(self, d: CausalDiscovery) -> None:
        if self._causal_graph:
            try:
                self._causal_graph.add_causal_edge(
                    d.antecedent, d.consequent, d.confidence
                )
            except Exception:
                pass

        if self._world_model:
            try:
                self._world_model.register_causal_pattern(
                    d.antecedent, d.consequent, d.confidence
                )
            except Exception:
                pass

        # Broadcast so other modules can immediately use dream-derived edges.
        self._broadcast_causal_discovery(d)

    # ── API ─────────────────────────────
    def summary(self) -> dict:
        return {
            "tick_count": self._tick_count,
            "events_replayed": self._total_replayed,
            "total_discovered": self._total_discovered,
            "recent_discoveries": [
                d.to_dict() for d in self._discoveries[-10:]
            ],
            "recent_dreams": [r.to_dict() for r in self._dreams[-10:]],
        }