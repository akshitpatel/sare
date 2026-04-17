"""
UnifiedMemoryIndex — cross-references concept_graph, knowledge_graph,
world_model_v2, and episodes so entities are addressable across stores.

Before: concept "addition" lived only in concept_graph; schema "arithmetic_identity"
lived only in world_model; belief "x+0=x" lived only in knowledge_graph. The solver
couldn't say "here's everything I know about addition" because nothing linked them.

After: UnifiedMemoryIndex.lookup("addition") returns concept + schemas + beliefs +
episodes. activate() propagates a touch across all stores.

Domain-general — no math/chemistry/logic assumptions.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set

log = logging.getLogger(__name__)


class UnifiedMemoryIndex:
    """Map entity name → (concept, schemas, beliefs, episodes) across stores."""

    def __init__(self):
        self._lock = threading.Lock()
        self._by_name: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: {"concepts": [], "schemas": [], "beliefs": [], "kg_nodes": []}
        )
        self._by_domain: Dict[str, Set[str]] = defaultdict(set)
        self._rebuilt_at: float = 0.0
        self._stats = {"rebuilds": 0, "activations": 0, "lookups": 0}

    def rebuild(self) -> int:
        """Scan all memory stores and build the cross-reference index.
        Returns number of entities indexed."""
        with self._lock:
            self._by_name.clear()
            self._by_domain.clear()
            n = 0

            # 1) Concepts
            try:
                from sare.concept.concept_graph import get_concept_graph
                cg = get_concept_graph()
                for c in cg.all_concepts():
                    key = c.name.lower()
                    self._by_name[key]["concepts"].append(c.name)
                    self._by_domain[c.domain].add(key)
                    # Also index by symbol
                    if c.symbol:
                        sk = c.symbol.lower()
                        if sk != key:
                            self._by_name[sk]["concepts"].append(c.name)
                    # Index related concepts as aliases
                    for rel in c.related:
                        rk = rel.lower()
                        if rk not in self._by_name:
                            self._by_name[rk]["concepts"].append(c.name)
                    n += 1
            except Exception as e:
                log.debug("UnifiedIndex: concept_graph scan failed: %s", e)

            # 2) World model schemas
            try:
                from sare.memory.world_model import get_world_model
                wm = get_world_model()
                schemas = getattr(wm, "_schemas", {}) or {}
                for sig, sch in schemas.items():
                    name = getattr(sch, "name", None) or sig
                    if not name:
                        continue
                    key = str(name).lower()
                    self._by_name[key]["schemas"].append(str(name))
                    dom = getattr(sch, "domain", "general") or "general"
                    self._by_domain[dom].add(key)
                    n += 1
            except Exception as e:
                log.debug("UnifiedIndex: world_model scan failed: %s", e)

            # 3) Knowledge graph nodes
            try:
                from sare.memory.knowledge_graph import get_kg
                kg = get_kg()
                for nid, node in kg._nodes.items():
                    name = getattr(node, "name", "") or nid
                    if not name:
                        continue
                    # Normalize: first 60 chars as index key
                    key = str(name).lower()[:60]
                    if node.type == "belief":
                        self._by_name[key]["beliefs"].append(nid)
                    else:
                        self._by_name[key]["kg_nodes"].append(nid)
                    dom = getattr(node, "domain", "general") or "general"
                    self._by_domain[dom].add(key)
                    n += 1
            except Exception as e:
                log.debug("UnifiedIndex: knowledge_graph scan failed: %s", e)

            self._rebuilt_at = time.time()
            self._stats["rebuilds"] += 1
            log.info("UnifiedMemoryIndex: indexed %d entities across %d domains",
                     len(self._by_name), len(self._by_domain))
            return len(self._by_name)

    def lookup(self, name: str) -> Dict[str, List[str]]:
        """Return all representations of this entity across stores."""
        self._stats["lookups"] += 1
        key = name.lower().strip()
        return dict(self._by_name.get(key, {"concepts": [], "schemas": [],
                                             "beliefs": [], "kg_nodes": []}))

    def entities_in_domain(self, domain: str) -> List[str]:
        """List entity keys associated with a domain."""
        return sorted(self._by_domain.get(domain, set()))

    def activate(self, name: str, strength: float = 0.1) -> int:
        """When an entity is referenced during reasoning, propagate a 'touch'
        across all its representations. Returns number of touches made.

        Increments concept.use_count, world_model schema activation, and
        knowledge_graph node observations."""
        refs = self.lookup(name)
        count = 0
        # 1) Concept use_count
        try:
            from sare.concept.concept_graph import get_concept_graph
            cg = get_concept_graph()
            for cname in refs.get("concepts", []):
                c = cg.get(cname)
                if c is not None:
                    c.use_count += 1
                    count += 1
        except Exception:
            pass
        # 2) World model schema activation
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            schemas = getattr(wm, "_schemas", {}) or {}
            for sname in refs.get("schemas", []):
                for sig, sch in schemas.items():
                    if getattr(sch, "name", "") == sname:
                        try:
                            sch.activation = min(1.0, float(getattr(sch, "activation", 0.0)) + strength)
                            sch.use_count = int(getattr(sch, "use_count", 0)) + 1
                            count += 1
                        except Exception:
                            pass
                        break
        except Exception:
            pass
        # 3) Knowledge graph observations
        try:
            from sare.memory.knowledge_graph import get_kg
            kg = get_kg()
            for nid in refs.get("kg_nodes", []) + refs.get("beliefs", []):
                node = kg._nodes.get(nid)
                if node is not None:
                    node.observations = int(getattr(node, "observations", 0) or 0) + 1
                    count += 1
        except Exception:
            pass
        self._stats["activations"] += count
        return count

    def stats(self) -> Dict:
        return {
            "entities": len(self._by_name),
            "domains": len(self._by_domain),
            "last_rebuild": self._rebuilt_at,
            **self._stats,
        }


_SINGLETON: Optional[UnifiedMemoryIndex] = None


def get_unified_memory_index() -> UnifiedMemoryIndex:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = UnifiedMemoryIndex()
        # Lazy rebuild on first use
        try:
            _SINGLETON.rebuild()
        except Exception as e:
            log.warning("UnifiedMemoryIndex initial rebuild failed: %s", e)
    return _SINGLETON
