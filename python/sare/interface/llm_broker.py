"""
LLMBroker — single gateway for all LLM calls in SARE-HX.

Before: synthesis, QA, chat, world_model_hypothesis, semantic_solver,
planner, goal_decomposer each called _call_llm directly. No shared
priority, no rate limits, no cache. Timeouts cascaded.

After: all callers go through LLMBroker.request(role, prompt). The broker:
  - Deduplicates identical prompts (shared cache, 10min TTL)
  - Applies per-role rate limits
  - Exposes a simple stats() for the dashboard

Domain-general: roles are configurable.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

log = logging.getLogger(__name__)


# Role priorities (higher = more urgent)
_ROLE_PRIORITY = {
    "synthesis":    9,   # new transform generation
    "hypothesis":   8,   # world model hypothesis from high surprise
    "cache_fill":   7,   # fill KB cache for unsolved problems
    "planner":      6,   # long-horizon planning
    "qa":           5,   # question answering
    "semantic":     5,   # semantic solver
    "chat":         3,   # conversational
    "other":        1,
}


# Per-role rate limits: max requests per 60 seconds
_ROLE_RATE_LIMIT = {
    "synthesis":    4,
    "hypothesis":   2,
    "cache_fill":   10,
    "planner":      3,
    "qa":           15,
    "semantic":     10,
    "chat":         30,
    "other":        5,
}


@dataclass
class _CacheEntry:
    response: str
    ts: float


class LLMBroker:
    _CACHE_TTL_S = 600   # 10 minutes

    def __init__(self):
        self._lock = threading.Lock()
        self._cache: Dict[str, _CacheEntry] = {}
        self._role_requests: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=100))
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "rate_limited": 0,
            "errors": 0,
            "by_role": defaultdict(lambda: {"count": 0, "cache_hits": 0, "rate_limited": 0}),
        }

    def _prompt_key(self, prompt: str, role: str, system_prompt: str = "") -> str:
        h = hashlib.md5(f"{role}||{system_prompt}||{prompt}".encode()).hexdigest()
        return h[:16]

    def _check_rate(self, role: str) -> bool:
        """Return True if the role is within its rate limit."""
        limit = _ROLE_RATE_LIMIT.get(role, _ROLE_RATE_LIMIT["other"])
        now = time.time()
        q = self._role_requests[role]
        # Drop events older than 60s
        while q and (now - q[0]) > 60.0:
            q.popleft()
        return len(q) < limit

    def _record_request(self, role: str) -> None:
        self._role_requests[role].append(time.time())

    def request(
        self,
        role: str,
        prompt: str,
        system_prompt: str = "",
        use_synthesis_model: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        allow_cache: bool = True,
    ) -> Optional[str]:
        """Send a prompt through the broker.

        Returns the LLM response, a cached response, or None if rate-limited.
        """
        role = role if role in _ROLE_PRIORITY else "other"
        with self._lock:
            self._stats["total_requests"] += 1
            self._stats["by_role"][role]["count"] += 1

            # 1) Cache lookup
            if allow_cache:
                key = self._prompt_key(prompt, role, system_prompt)
                entry = self._cache.get(key)
                if entry and (time.time() - entry.ts) < self._CACHE_TTL_S:
                    self._stats["cache_hits"] += 1
                    self._stats["by_role"][role]["cache_hits"] += 1
                    return entry.response

            # 2) Rate limit check
            if not self._check_rate(role):
                self._stats["rate_limited"] += 1
                self._stats["by_role"][role]["rate_limited"] += 1
                log.debug("LLMBroker: rate-limited role=%s", role)
                return None

            self._record_request(role)

        # 3) Delegate to existing LLM bridge (outside lock to avoid blocking)
        try:
            from sare.interface.llm_bridge import _call_llm
            kwargs = {}
            if system_prompt:
                kwargs["system_prompt"] = system_prompt
            if max_tokens:
                kwargs["max_tokens_override"] = max_tokens
            kwargs["use_synthesis_model"] = use_synthesis_model
            response = _call_llm(prompt, **kwargs)
        except Exception as e:
            log.warning("LLMBroker: call failed (role=%s): %s", role, e)
            with self._lock:
                self._stats["errors"] += 1
            return None

        # 4) Cache the response
        if allow_cache and response:
            with self._lock:
                key = self._prompt_key(prompt, role, system_prompt)
                self._cache[key] = _CacheEntry(response=response, ts=time.time())
                # Prune old cache entries periodically
                if len(self._cache) > 500:
                    now = time.time()
                    expired = [k for k, v in self._cache.items()
                               if (now - v.ts) > self._CACHE_TTL_S]
                    for k in expired:
                        self._cache.pop(k, None)

        return response

    def stats(self) -> dict:
        with self._lock:
            br = {}
            for role, d in self._stats["by_role"].items():
                br[role] = dict(d)
                br[role]["priority"] = _ROLE_PRIORITY.get(role, 1)
                br[role]["rate_limit_per_min"] = _ROLE_RATE_LIMIT.get(role, _ROLE_RATE_LIMIT["other"])
            return {
                "total_requests": self._stats["total_requests"],
                "cache_hits": self._stats["cache_hits"],
                "cache_hit_rate": round(
                    self._stats["cache_hits"] / max(1, self._stats["total_requests"]), 3
                ),
                "rate_limited": self._stats["rate_limited"],
                "errors": self._stats["errors"],
                "cache_size": len(self._cache),
                "by_role": br,
            }


_SINGLETON: Optional[LLMBroker] = None


def get_llm_broker() -> LLMBroker:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = LLMBroker()
    return _SINGLETON
