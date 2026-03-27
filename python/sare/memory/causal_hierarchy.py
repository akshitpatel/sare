"""
CausalHierarchy — 3-level causal abstraction model.

Level 0: Features    — raw observations (x=3, y=4, op='+')
Level 1: Concepts    — named patterns (linear_equation, distributive_property)
Level 2: Laws        — general rules (commutativity, transitivity, conservation)

The hierarchy lets SARE reason about WHY transforms work, not just THAT they work.
Example:
  Feature: x+0 → x (observed)
  Concept: additive_identity (pattern named)
  Law: identity_element (general algebraic law)

Upward inference: repeated features → concept candidate
Downward inference: a law is active → its concept instances should work → their features predicted
"""
from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

log = logging.getLogger(__name__)

PERSIST_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "causal_hierarchy.json"


@dataclass
class CausalNode:
    """A node at any level of the causal hierarchy."""
    id: str
    level: int          # 0=feature, 1=concept, 2=law
    label: str
    domain: str = "general"
    confidence: float = 0.5
    evidence_count: int = 0
    parent_ids: List[str] = field(default_factory=list)   # level N+1 nodes this is evidence for
    child_ids: List[str] = field(default_factory=list)    # level N-1 nodes this explains
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "level": self.level, "label": self.label,
            "domain": self.domain, "confidence": round(self.confidence, 3),
            "evidence_count": self.evidence_count,
            "parent_ids": self.parent_ids, "child_ids": self.child_ids,
            "created_at": self.created_at, "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CausalNode":
        n = cls(id=d["id"], level=d["level"], label=d["label"])
        n.domain = d.get("domain", "general")
        n.confidence = d.get("confidence", 0.5)
        n.evidence_count = d.get("evidence_count", 0)
        n.parent_ids = d.get("parent_ids", [])
        n.child_ids = d.get("child_ids", [])
        n.created_at = d.get("created_at", time.time())
        n.last_seen = d.get("last_seen", time.time())
        return n


# Known laws at level 2 — seeded at startup
_SEED_LAWS = [
    ("law_identity_element",    "identity_element",    "algebra"),
    ("law_commutativity",       "commutativity",       "algebra"),
    ("law_associativity",       "associativity",       "algebra"),
    ("law_distributivity",      "distributivity",      "algebra"),
    ("law_transitivity",        "transitivity",        "logic"),
    ("law_modus_ponens",        "modus_ponens",        "logic"),
    ("law_double_negation",     "double_negation",     "logic"),
    ("law_conservation",        "conservation",        "physics"),
    ("law_commutativity_logic", "commutativity_and_or","logic"),
    ("law_zero_element",        "zero_element",        "algebra"),
]

# Known concept → law mappings at level 1→2
_CONCEPT_LAW_MAP = {
    "additive_identity":        "law_identity_element",
    "multiplicative_identity":  "law_identity_element",
    "conjunctive_identity":     "law_identity_element",
    "disjunctive_identity":     "law_identity_element",
    "double_negation":          "law_double_negation",
    "distributive_mul_add":     "law_distributivity",
    "multiplicative_zero":      "law_zero_element",
    "subtractive_self":         "law_zero_element",
    "modus_ponens":             "law_modus_ponens",
    "transitivity":             "law_transitivity",
}

# Feature pattern → concept mappings
_FEATURE_CONCEPT_MAP = {
    "x + 0":    "additive_identity",
    "x * 1":    "multiplicative_identity",
    "x * 0":    "multiplicative_zero",
    "x - x":    "subtractive_self",
    "not not":  "double_negation",
    "p and True": "conjunctive_identity",
    "p or False": "disjunctive_identity",
}


class CausalHierarchy:
    """3-level causal hierarchy: features → concepts → laws."""

    def __init__(self, persist_path: Optional[Path] = None):
        self._path = Path(persist_path or PERSIST_PATH)
        self._nodes: Dict[str, CausalNode] = {}
        self._load()
        self._seed_laws()

    def _seed_laws(self):
        """Ensure all known laws exist at level 2."""
        for law_id, label, domain in _SEED_LAWS:
            if law_id not in self._nodes:
                self._nodes[law_id] = CausalNode(
                    id=law_id, level=2, label=label, domain=domain,
                    confidence=0.95, evidence_count=100
                )

    def observe_feature(self, expression: str, transform: str, domain: str = "general") -> Optional[str]:
        """
        Record a feature-level observation and propagate upward.

        Returns the concept ID if this feature was mapped to a concept.
        """
        feat_id = f"feat_{hash(expression + transform) & 0xFFFFFF:06x}"

        if feat_id not in self._nodes:
            self._nodes[feat_id] = CausalNode(
                id=feat_id, level=0,
                label=f"{transform}({expression[:30]})",
                domain=domain, confidence=0.5
            )

        node = self._nodes[feat_id]
        node.evidence_count += 1
        node.last_seen = time.time()
        # Confidence grows with evidence
        node.confidence = min(0.95, 0.5 + node.evidence_count * 0.02)

        # Map feature to concept (level 0 → 1)
        concept_name = _FEATURE_CONCEPT_MAP.get(expression[:20], None) or _FEATURE_CONCEPT_MAP.get(transform, None)
        if not concept_name:
            # Try partial match
            for pattern, concept in _FEATURE_CONCEPT_MAP.items():
                if pattern in expression or transform == concept:
                    concept_name = concept
                    break

        if concept_name:
            concept_id = f"concept_{concept_name}"
            self._ensure_concept(concept_id, concept_name, domain)
            if concept_id not in node.parent_ids:
                node.parent_ids.append(concept_id)
            concept_node = self._nodes[concept_id]
            if feat_id not in concept_node.child_ids:
                concept_node.child_ids.append(feat_id)
            concept_node.evidence_count += 1
            concept_node.confidence = min(0.95, 0.5 + concept_node.evidence_count * 0.01)
            # Propagate to law
            self._propagate_to_law(concept_id, concept_name, domain)
            return concept_id

        return None

    def _ensure_concept(self, concept_id: str, label: str, domain: str):
        if concept_id not in self._nodes:
            self._nodes[concept_id] = CausalNode(
                id=concept_id, level=1, label=label, domain=domain,
                confidence=0.5, evidence_count=1
            )

    def _propagate_to_law(self, concept_id: str, concept_name: str, domain: str):
        law_id = _CONCEPT_LAW_MAP.get(concept_name)
        if not law_id:
            return
        if law_id not in self._nodes:
            return
        law_node = self._nodes[law_id]
        concept_node = self._nodes[concept_id]
        if law_id not in concept_node.parent_ids:
            concept_node.parent_ids.append(law_id)
        if concept_id not in law_node.child_ids:
            law_node.child_ids.append(concept_id)
        law_node.evidence_count += 1
        log.debug("CausalHierarchy: %s → concept/%s → %s (law)",
                  concept_name, concept_name, law_id)

    def predict_transforms(self, domain: str, expression: str) -> List[str]:
        """
        Given domain + expression, use the hierarchy to predict useful transforms.

        Downward inference: find active laws for domain → their concepts → their features → transforms
        """
        transforms = []
        # Find active laws for this domain
        for node in self._nodes.values():
            if node.level == 2 and (node.domain == domain or node.domain == "general"):
                if node.confidence >= 0.7:
                    # Collect transforms from child concepts
                    for concept_id in node.child_ids:
                        if concept_id in self._nodes:
                            transforms.append(self._nodes[concept_id].label)
        return transforms[:5]  # top 5

    def get_active_laws(self, domain: str) -> List[dict]:
        """Return laws with high confidence for a domain."""
        laws = []
        for node in self._nodes.values():
            if node.level == 2 and (node.domain == domain or node.domain == "general"):
                laws.append({
                    "id": node.id,
                    "label": node.label,
                    "confidence": node.confidence,
                    "evidence": node.evidence_count,
                    "concepts": len(node.child_ids),
                })
        return sorted(laws, key=lambda x: x["confidence"], reverse=True)

    def summary(self) -> dict:
        counts = {0: 0, 1: 0, 2: 0}
        for n in self._nodes.values():
            counts[n.level] = counts.get(n.level, 0) + 1
        return {
            "total_nodes": len(self._nodes),
            "features": counts[0],
            "concepts": counts[1],
            "laws": counts[2],
            "seeded_laws": len(_SEED_LAWS),
        }

    def _load(self):
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for nid, nd in data.get("nodes", {}).items():
                self._nodes[nid] = CausalNode.from_dict(nd)
            log.debug("CausalHierarchy loaded %d nodes", len(self._nodes))
        except Exception as e:
            log.debug("CausalHierarchy load failed: %s", e)

    def save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {"nodes": {nid: n.to_dict() for nid, n in self._nodes.items()}}
            import tempfile, os
            tmp = str(self._path) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._path)
        except Exception as e:
            log.debug("CausalHierarchy save failed: %s", e)


_hierarchy_instance: Optional[CausalHierarchy] = None

def get_causal_hierarchy() -> CausalHierarchy:
    global _hierarchy_instance
    if _hierarchy_instance is None:
        _hierarchy_instance = CausalHierarchy()
    return _hierarchy_instance
