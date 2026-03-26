"""
CompositeRuleLearner — mines proof traces for frequent 2-step transform sequences.

After every N successful solves, scan the proof_steps history for pairs like:
  [DistributiveExpansion, CombineLikeTerms]  — appears 15 times
  [MulOneElimination, AddZeroElimination]    — appears 12 times

These pairs get registered as ConceptRules with high initial utility, so the
heuristic model and credit assigner prioritize them.
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_PERSIST_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "composite_rules.json"
_MIN_SUPPORT = 5      # minimum co-occurrences to register as composite
_MAX_COMPOSITES = 50  # cap on number of registered composites


class CompositeRuleLearner:
    """
    Mines proof traces for frequent 2-step transform sequences and
    registers them as high-utility hints for the search engine.
    """

    def __init__(self):
        self._pair_counts: Counter = Counter()
        self._domain_pairs: Dict[str, Counter] = defaultdict(Counter)
        self._registered: Dict[str, dict] = {}  # composite_name -> {pair, utility, count}
        self._total_traces: int = 0
        self._load()

    def observe_trace(self, transforms_used: List[str], domain: str = "general") -> None:
        """Record a proof trace. Call after every successful solve."""
        if len(transforms_used) < 2:
            return
        self._total_traces += 1
        for i in range(len(transforms_used) - 1):
            pair = (transforms_used[i], transforms_used[i + 1])
            self._pair_counts[pair] += 1
            self._domain_pairs[domain][pair] += 1

    def mine_composites(self) -> List[dict]:
        """
        Find frequent pairs and register them.
        Returns list of newly registered composites.
        """
        new_composites = []
        top_pairs = self._pair_counts.most_common(100)

        for pair, count in top_pairs:
            if count < _MIN_SUPPORT:
                break
            name = f"Composite_{pair[0][:12]}_{pair[1][:12]}"
            if name in self._registered:
                # Update count
                self._registered[name]["count"] = count
                continue
            if len(self._registered) >= _MAX_COMPOSITES:
                break
            entry = {
                "name": name,
                "pair": list(pair),
                "count": count,
                "utility": min(1.0, count / 20.0),  # normalize to [0,1]
            }
            self._registered[name] = entry
            new_composites.append(entry)
            log.info("CompositeRuleLearner: registered %s (count=%d)", name, count)

        if new_composites:
            self._save()
        return new_composites

    def get_priority_pairs(self, domain: str = "general") -> List[Tuple[str, str]]:
        """Return top composite pairs for a domain (for transform pre-ordering)."""
        domain_counter = self._domain_pairs.get(domain, self._pair_counts)
        return [pair for pair, _ in domain_counter.most_common(10)]

    def get_registered(self) -> Dict[str, dict]:
        return dict(self._registered)

    def suggest_transform_order(self, transforms: list, domain: str = "general") -> list:
        """
        Re-order transforms so that the first element of high-frequency pairs
        appears earlier. This seeds the beam with more productive starting moves.
        """
        priority_pairs = self.get_priority_pairs(domain)
        priority_first = {pair[0] for pair in priority_pairs}
        priority_second = {pair[1] for pair in priority_pairs}

        def sort_key(t):
            name = type(t).__name__
            if name in priority_first:
                return 0
            if name in priority_second:
                return 1
            return 2

        return sorted(transforms, key=sort_key)

    def _save(self) -> None:
        try:
            _PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "pair_counts": {f"{k[0]}|{k[1]}": v for k, v in self._pair_counts.items()},
                "registered": self._registered,
                "total_traces": self._total_traces,
            }
            tmp = _PERSIST_PATH.parent / f"composite_rules.{os.getpid()}.tmp"
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, _PERSIST_PATH)
        except Exception as e:
            log.debug("CompositeRuleLearner save error: %s", e)

    def _load(self) -> None:
        if not _PERSIST_PATH.exists():
            return
        try:
            payload = json.loads(_PERSIST_PATH.read_text(encoding="utf-8"))
            for key, v in payload.get("pair_counts", {}).items():
                parts = key.split("|", 1)
                if len(parts) == 2:
                    self._pair_counts[(parts[0], parts[1])] = int(v)
            self._registered = payload.get("registered", {})
            self._total_traces = int(payload.get("total_traces", 0))
        except Exception as e:
            log.debug("CompositeRuleLearner load error: %s", e)


# Module-level singleton
_instance: Optional[CompositeRuleLearner] = None

def get_composite_learner() -> CompositeRuleLearner:
    global _instance
    if _instance is None:
        _instance = CompositeRuleLearner()
    return _instance
