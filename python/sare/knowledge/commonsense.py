"""
Common Sense Knowledge Base (AGI Gap #4)

Gives SARE-HX access to broad world knowledge beyond formal math/logic rules.
Instead of relying on an external triple store, this module:

1. Ships with a curated set of key commonsense facts as a seed graph.
2. Can ingest additional knowledge from ConceptNet's public API (optional).
3. Exposes a `query(concept)` interface that returns semantically related nodes.
4. Integrates with the ConceptRegistry so SARE-HX can reason with common-sense
   facts alongside formal proofs.

Commonsense reasoning examples SARE-HX gains:
  - "Fire is hot" → prevents nonsensical transforms
  - "Dogs are animals" → taxonomy reasoning
  - "If raining, ground is wet" → causal entailment
  - "Heavy objects fall" → physical intuition
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional, Tuple

log = logging.getLogger(__name__)

# ── Built-in Commonsense Seed Facts ──────────────────────────────────────────
# Format: (subject, relation, object)
# Relations: IsA, HasA, PartOf, Causes, UsedFor, CapableOf, HasProperty, LocatedAt

_COMMONSENSE_SEEDS: List[Tuple[str, str, str]] = [
    # Physical world
    ("fire",       "HasProperty", "hot"),
    ("ice",        "HasProperty", "cold"),
    ("water",      "HasProperty", "wet"),
    ("rock",       "HasProperty", "hard"),
    ("cloud",      "Causes",      "rain"),
    ("rain",       "Causes",      "wet_ground"),
    ("sun",        "Causes",      "warmth"),
    ("gravity",    "Causes",      "falling"),

    # Biology
    ("dog",        "IsA",         "animal"),
    ("cat",        "IsA",         "animal"),
    ("bird",       "IsA",         "animal"),
    ("animal",     "IsA",         "living_thing"),
    ("human",      "IsA",         "animal"),
    ("human",      "HasA",        "brain"),
    ("human",      "CapableOf",   "reasoning"),
    ("plant",      "IsA",         "living_thing"),
    ("tree",       "IsA",         "plant"),

    # Artifacts
    ("knife",      "UsedFor",     "cutting"),
    ("pen",        "UsedFor",     "writing"),
    ("car",        "UsedFor",     "transportation"),
    ("phone",      "UsedFor",     "communication"),
    ("computer",   "UsedFor",     "computing"),
    ("book",       "UsedFor",     "reading"),

    # Social / abstract
    ("money",      "UsedFor",     "buying"),
    ("school",     "UsedFor",     "learning"),
    ("hospital",   "UsedFor",     "healing"),
    ("law",        "UsedFor",     "governance"),
    ("friendship", "HasProperty", "trust"),
    ("lie",        "Causes",      "distrust"),

    # Math facts (links to formal knowledge)
    ("zero",       "HasProperty", "additive_identity"),
    ("one",        "HasProperty", "multiplicative_identity"),
    ("infinity",   "HasProperty", "unbounded"),
    ("negative",   "IsA",         "number"),
    ("fraction",   "IsA",         "number"),
]


class CommonSenseBase:
    """
    In-memory commonsense knowledge graph.
    Loads seed facts + optional ConceptNet augmentation.
    Supports semantic query by concept name.
    """

    PERSIST_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "commonsense.json"

    def __init__(self):
        # Triple store: subject → list of (relation, object)
        self._forward: Dict[str, List[Tuple[str, str]]] = {}
        # Reverse index: object → list of (relation, subject)
        self._backward: Dict[str, List[Tuple[str, str]]] = {}

    def seed(self):
        """Load built-in commonsense seed facts."""
        for s, r, o in _COMMONSENSE_SEEDS:
            self._add(s, r, o)
        log.info(f"CommonSenseBase seeded: {len(_COMMONSENSE_SEEDS)} facts")

    def _add(self, subject: str, relation: str, obj: str):
        s, r, o = subject.lower(), relation, obj.lower()
        self._forward.setdefault(s, []).append((r, o))
        self._backward.setdefault(o, []).append((r, s))

    def query(self, concept: str, depth: int = 1) -> List[dict]:
        """
        Return all known facts about `concept` up to `depth` hops away.
        Returns list of {subject, relation, object, distance}.
        """
        concept = concept.lower().strip()
        seen, results, frontier = set(), [], [concept]

        for d in range(depth + 1):
            next_frontier = []
            for c in frontier:
                if c in seen:
                    continue
                seen.add(c)
                for rel, obj in self._forward.get(c, []):
                    results.append({"subject": c, "relation": rel, "object": obj, "distance": d})
                    next_frontier.append(obj)
                for rel, subj in self._backward.get(c, []):
                    results.append({"subject": subj, "relation": rel, "object": c, "distance": d})
            frontier = next_frontier

        return results

    def get_properties(self, concept: str) -> List[str]:
        """Return all HasProperty values for a concept."""
        return [o for r, o in self._forward.get(concept.lower(), []) if r == "HasProperty"]

    def is_a(self, concept: str, category: str) -> bool:
        """Check if concept IsA category (with one hop of transitivity)."""
        c, cat = concept.lower(), category.lower()
        direct = [(r, o) for r, o in self._forward.get(c, []) if r == "IsA"]
        for _, o in direct:
            if o == cat:
                return True
            # One more hop
            for r2, o2 in self._forward.get(o, []):
                if r2 == "IsA" and o2 == cat:
                    return True
        return False

    def augment_from_conceptnet(self, concepts: List[str], max_per_concept: int = 10):
        """
        Optionally fetch related edges from ConceptNet's public API.
        Gracefully skips if network is unavailable.
        """
        for concept in concepts:
            try:
                url = f"https://api.conceptnet.io/c/en/{concept.lower().replace(' ', '_')}?limit={max_per_concept}"
                req = urllib.request.Request(url, headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read())
                edges = data.get("edges", [])
                added = 0
                for edge in edges:
                    rel = edge.get("rel", {}).get("label", "")
                    subj = edge.get("start", {}).get("label", "")
                    obj = edge.get("end", {}).get("label", "")
                    if rel and subj and obj and len(subj) < 50 and len(obj) < 50:
                        self._add(subj, rel, obj)
                        added += 1
                log.info(f"ConceptNet: +{added} facts for '{concept}'")
            except Exception as e:
                log.debug(f"ConceptNet augment skipped for '{concept}': {e}")

    def save(self):
        self.PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {"forward": self._forward, "backward": self._backward}
        with open(self.PERSIST_PATH, "w") as f:
            json.dump(payload, f)
        log.info(f"CommonSenseBase saved: {sum(len(v) for v in self._forward.values())} total facts")

    def load(self):
        if not self.PERSIST_PATH.exists():
            return
        with open(self.PERSIST_PATH) as f:
            data = json.load(f)
        self._forward = {k: [tuple(x) for x in v] for k, v in data.get("forward", {}).items()}
        self._backward = {k: [tuple(x) for x in v] for k, v in data.get("backward", {}).items()}
        log.info(f"CommonSenseBase loaded: {sum(len(v) for v in self._forward.values())} facts")

    def total_facts(self) -> int:
        return sum(len(v) for v in self._forward.values())

    def to_graph_dict(self):
        """Convert the knowledge base into a SARE-HX graph for reasoning."""
        from sare.perception.world_grounder import GraphDict
        g = GraphDict()
        node_ids: Dict[str, int] = {}

        def get_node(label: str, ntype: str = "entity") -> int:
            key = label.lower()
            if key not in node_ids:
                node_ids[key] = g.add_node(label, node_type=ntype)
            return node_ids[key]

        for subj, facts in self._forward.items():
            sid = get_node(subj)
            for rel, obj in facts:
                oid = get_node(obj)
                g.add_edge(sid, oid, relation=rel)

        return g
