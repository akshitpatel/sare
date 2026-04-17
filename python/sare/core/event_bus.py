"""
sare.core.event_bus — Lightweight singleton event bus for reactive wiring.

Usage:
    from sare.core.event_bus import get_event_bus

    bus = get_event_bus()
    bus.subscribe("my_event", lambda data: print(data))
    bus.publish("my_event", {"key": "value"})
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List

log = logging.getLogger(__name__)

_bus_lock = threading.Lock()
_bus_instance: "EventBus | None" = None


class EventBus:
    """Simple thread-safe synchronous event bus.

    Events are dispatched synchronously in the calling thread.  Each listener
    receives a single ``data`` argument (any dict or value).  Exceptions in
    listeners are caught and logged so they never crash the publisher.
    """

    def __init__(self) -> None:
        self._listeners: Dict[str, List[Callable[[Any], None]]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(self, event: str, handler: Callable[[Any], None]) -> None:
        """Register *handler* to be called whenever *event* is published."""
        with self._lock:
            self._listeners.setdefault(event, [])
            if handler not in self._listeners[event]:
                self._listeners[event].append(handler)

    def unsubscribe(self, event: str, handler: Callable[[Any], None]) -> None:
        """Remove a previously registered handler (no-op if not found)."""
        with self._lock:
            handlers = self._listeners.get(event, [])
            if handler in handlers:
                handlers.remove(handler)

    def publish(self, event: str, data: Any = None) -> None:
        """Publish *event* with optional *data* to all registered listeners."""
        with self._lock:
            handlers = list(self._listeners.get(event, []))
        for handler in handlers:
            try:
                handler(data)
            except Exception as exc:
                log.debug("EventBus listener error (event=%s): %s", event, exc)


def get_event_bus() -> EventBus:
    """Return the process-wide singleton EventBus, creating it on first call."""
    global _bus_instance
    if _bus_instance is None:
        with _bus_lock:
            if _bus_instance is None:
                _bus_instance = EventBus()
    return _bus_instance
