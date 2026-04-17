"""
SchemaMatcher — Cache structural problem schemas to avoid redundant search.

When a problem has the same operator-tree structure as a previously solved problem,
replay the cached proof steps instead of running full BeamSearch.
"""
from __future__ import annotations
import atexit
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "schema_cache.json"


class SchemaMatcher:
    _SAVE_INTERVAL = 25    # save every N new schemas (was 50)
    _MAX_CACHE = 5000      # hard cap; evict oldest when exceeded

    def __init__(self, cache_path: Optional[Path] = None):
        self._cache: Dict[str, List[str]] = {}  # hash → proof_steps
        self._insertion_order: List[str] = []   # for LRU eviction
        self._hits = 0
        self._misses = 0
        self._path = Path(cache_path or CACHE_PATH)
        self._load()
        atexit.register(self.save)  # always save on clean shutdown

    def _load(self):
        if self._path.exists():
            try:
                self._cache = json.loads(self._path.read_text())
                self._insertion_order = list(self._cache.keys())
                log.debug("SchemaMatcher loaded %d cached schemas", len(self._cache))
            except Exception as e:
                log.debug("SchemaMatcher load failed: %s", e)

    def save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._cache, indent=2))
        except Exception as e:
            log.debug("SchemaMatcher save failed: %s", e)

    def _structural_hash(self, graph) -> str:
        """Compute a structural fingerprint of the problem graph.

        Two problems have the same hash iff they have the same operator-tree shape,
        same variable count, and same constant pattern (ignoring actual values).
        """
        try:
            import re
            # Try C++ binding graph first (has get_node_ids / get_node)
            if hasattr(graph, 'get_node_ids'):
                node_ids = graph.get_node_ids()
                node_types = []
                for nid in sorted(node_ids):
                    node = graph.get_node(nid)
                    label = getattr(node, 'label', str(node))
                    # Normalize: replace specific numbers with their type pattern
                    normalized = re.sub(r'\b\d+\.\d+\b', 'FLOAT', label)
                    normalized = re.sub(r'\b\d+\b', 'INT', normalized)
                    # Keep variable names as-is (they define structure)
                    node_types.append(normalized)
                fingerprint = "|".join(sorted(node_types))
            # Python Graph (has .nodes dict or list)
            elif hasattr(graph, 'nodes'):
                nodes = graph.nodes
                if isinstance(nodes, dict):
                    labels = [str(v.get('label', k)) for k, v in nodes.items()]
                elif isinstance(nodes, list):
                    labels = [str(getattr(n, 'label', n)) for n in nodes]
                else:
                    labels = [str(nodes)]
                normalized = [re.sub(r'\b\d+\b', 'NUM', l) for l in labels]
                fingerprint = "|".join(sorted(normalized))
            else:
                fingerprint = str(type(graph))
        except Exception:
            fingerprint = str(id(graph))

        return hashlib.md5(fingerprint.encode()).hexdigest()[:16]

    def match(self, graph) -> Optional[List[str]]:
        """Return cached proof steps if this graph structure was seen before, else None."""
        h = self._structural_hash(graph)
        result = self._cache.get(h)
        if result is not None:
            self._hits += 1
            log.debug("SchemaMatcher HIT: hash=%s steps=%s", h, result)
        else:
            self._misses += 1
        return result

    def record(self, graph, proof_steps: List[str]):
        """Store a successful proof for this graph structure."""
        h = self._structural_hash(graph)
        if h not in self._cache:
            self._cache[h] = list(proof_steps)
            self._insertion_order.append(h)
            # Evict oldest entries if over cap
            if len(self._cache) > self._MAX_CACHE:
                evict_count = len(self._cache) - (self._MAX_CACHE - 1000)
                to_evict = self._insertion_order[:evict_count]
                for old_h in to_evict:
                    self._cache.pop(old_h, None)
                self._insertion_order = self._insertion_order[evict_count:]
            # Save every 25 new schemas (was 50)
            if len(self._cache) % self._SAVE_INTERVAL == 0:
                self.save()

    def induce_generalizations(self) -> dict:
        """
        Scan cached schemas for structural similarities and deduplicate.

        Two schemas are "similar" if:
        1. Their hash keys share the same first 8 characters (same operator tree structure)
        2. They have identical proof_steps lists

        Merges duplicates by keeping the most recently inserted entry.
        Returns stats: {"merged": N, "total_before": M, "total_after": K}
        """
        if not self._cache:
            return {"merged": 0, "total_before": 0, "total_after": 0}

        total_before = len(self._cache)

        # Build position map once for O(1) insertion-order lookups
        order_map = {k: i for i, k in enumerate(self._insertion_order)}

        # Group by (first_8_chars_of_hash, tuple_of_proof_steps)
        groups: dict = {}
        for hash_key, proof_steps in list(self._cache.items()):
            proof = proof_steps if isinstance(proof_steps, list) else []
            group_key = (hash_key[:8], tuple(sorted(proof)))
            groups.setdefault(group_key, []).append(hash_key)

        merged = 0
        for hash_keys in groups.values():
            if len(hash_keys) <= 1:
                continue
            # Keep the most recently inserted entry
            best_key = max(hash_keys, key=lambda k: order_map.get(k, -1))
            for k in hash_keys:
                if k != best_key and k in self._cache:
                    del self._cache[k]
                    merged += 1

        if merged > 0:
            # Rebuild insertion order in one pass rather than O(n) remove per key
            self._insertion_order = [k for k in self._insertion_order if k in self._cache]
            self.save()

        total_after = len(self._cache)
        return {"merged": merged, "total_before": total_before, "total_after": total_after}

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "cached_schemas": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
        }


# Module-level singleton
_SCHEMA_MATCHER: Optional[SchemaMatcher] = None


def get_schema_matcher() -> SchemaMatcher:
    global _SCHEMA_MATCHER
    if _SCHEMA_MATCHER is None:
        _SCHEMA_MATCHER = SchemaMatcher()
    return _SCHEMA_MATCHER
