"""
GlobalWorkspace — Attention-bid based broadcast system (Baars 1988).

Modules "bid" for global attention by publishing attention_bid events.
The workspace selects the highest-bid module each cycle and broadcasts
its content to all subscribers — making it the current "focus of attention."

This is the closest we can get to unified consciousness in a symbolic system.
Modules: CurriculumGenerator, WorldModel, SelfModel, Homeostasis, GoalSetter
"""
from __future__ import annotations
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class AttentionBid:
    module: str
    content: Any
    urgency: float        # 0-1, higher = more urgent
    relevance: float      # 0-1, estimated relevance to current goal
    timestamp: float = field(default_factory=time.time)

    @property
    def score(self) -> float:
        # Combine urgency and relevance, decay with age
        age = time.time() - self.timestamp
        decay = max(0.1, 1.0 - age / 30.0)  # decays over 30s
        return (self.urgency * 0.6 + self.relevance * 0.4) * decay


class GlobalWorkspace:
    """
    Attention-competition global workspace.

    Every BROADCAST_INTERVAL seconds, the highest-scored bid "wins"
    and its content is broadcast to all registered listeners.
    """

    BROADCAST_INTERVAL = 5.0  # seconds between broadcasts

    def __init__(self):
        self._bids: List[AttentionBid] = []
        self._listeners: List[Callable] = []
        self._lock = threading.Lock()
        self._broadcast_count = 0
        self._last_winner: Optional[str] = None
        self._active = True

        # Wire to event bus
        try:
            from sare.core.event_bus import get_event_bus
            eb = get_event_bus()
            eb.subscribe("attention_bid", self._on_bid)
            eb.subscribe("surprise_high", self._on_surprise)
            log.info("GlobalWorkspace: subscribed to attention_bid and surprise_high events")
        except Exception as e:
            log.debug("GlobalWorkspace event bus wiring failed: %s", e)

        # Start broadcast loop
        self._broadcast_thread = threading.Thread(
            target=self._broadcast_loop, daemon=True, name="global-workspace"
        )
        self._broadcast_thread.start()

    def submit_bid(self, module: str, content: Any, urgency: float, relevance: float):
        """Submit an attention bid from a module."""
        bid = AttentionBid(module=module, content=content,
                          urgency=min(1.0, urgency), relevance=min(1.0, relevance))
        with self._lock:
            # Keep only most recent bid per module
            self._bids = [b for b in self._bids if b.module != module]
            self._bids.append(bid)

    def register_listener(self, callback: Callable):
        """Register a function to be called when broadcast fires."""
        self._listeners.append(callback)

    def _on_bid(self, data: dict):
        """Handle attention_bid event from event bus."""
        try:
            self.submit_bid(
                module=data.get("module", "unknown"),
                content=data.get("content", {}),
                urgency=float(data.get("urgency", 0.5)),
                relevance=float(data.get("relevance", 0.5)),
            )
        except Exception as e:
            log.debug("GlobalWorkspace._on_bid error: %s", e)

    def _on_surprise(self, data: dict):
        """High-surprise events automatically submit a WorldModel bid."""
        try:
            surprise = float(data.get("avg_surprise", 0))
            if surprise > 2.0:
                self.submit_bid(
                    module="world_model",
                    content=data,
                    urgency=min(1.0, surprise / 5.0),
                    relevance=0.8,
                )
        except Exception as e:
            log.debug("GlobalWorkspace._on_surprise error: %s", e)

    def _broadcast_loop(self):
        """Periodic broadcast: select winner and notify listeners."""
        while self._active:
            time.sleep(self.BROADCAST_INTERVAL)
            try:
                with self._lock:
                    if not self._bids:
                        continue
                    # Select winner by score
                    winner = max(self._bids, key=lambda b: b.score)
                    self._bids.clear()  # consume all bids

                self._last_winner = winner.module
                self._broadcast_count += 1

                log.debug("GlobalWorkspace broadcast #%d: winner=%s (score=%.2f)",
                          self._broadcast_count, winner.module, winner.score)

                # Publish broadcast event
                try:
                    from sare.core.event_bus import get_event_bus
                    get_event_bus().publish("gw_broadcast", {
                        "winner": winner.module,
                        "content": winner.content,
                        "broadcast_id": self._broadcast_count,
                    })
                except Exception:
                    pass

                # Notify direct listeners
                for listener in self._listeners:
                    try:
                        listener(winner)
                    except Exception:
                        pass
            except Exception as e:
                log.debug("GlobalWorkspace broadcast error: %s", e)

    def stop(self):
        self._active = False

    @property
    def stats(self) -> dict:
        with self._lock:
            bids = len(self._bids)
        return {
            "broadcasts": self._broadcast_count,
            "last_winner": self._last_winner,
            "pending_bids": bids,
            "listeners": len(self._listeners),
            "active": self._active,
        }


_gw_instance: Optional[GlobalWorkspace] = None


def get_global_workspace() -> GlobalWorkspace:
    global _gw_instance
    if _gw_instance is None:
        _gw_instance = GlobalWorkspace()
    return _gw_instance
