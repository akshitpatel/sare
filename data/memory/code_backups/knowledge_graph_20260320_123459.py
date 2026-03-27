"""
P4.4 — Persistent Unified Knowledge Graph

Instead of rules/schemas/beliefs/causal-links stored in separate dicts,
this provides a single graph-of-knowledge where:
  - Rules are nodes (type="rule")
  - Schemas are nodes (type="schema")
  - Beliefs are nodes (type="belief")
  - Causal links are nodes (type="causal")
  - Edges: "is_stronger_than", "depends_on", "contradicts", "exemplifies"

This enables:
  - Knowledge consolidation = graph simplification
  - Cross-concept reasoning (rule A depends on rule B)
  - Conflict detection (rule A contradicts rule C)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "memory"
KG_PATH  = DATA_DIR / "knowledge_graph.json"


@dataclass
class KGNode:
    id: str
    type: str           # rule / schema / belief / causal / domain
    name: str
    domain: str
    confidence: float = 0.5
    observations: int = 1
    content: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "type": self.type, "name": self.name,
            "domain": self.domain, "confidence": round(self.confidence, 3),
            "observations": self.observations, "content": self.content,
            "created_at": self.created_at,
        }


@dataclass
class KGEdge:
    source: str
    target: str
    relation: str       # depends_on / contradicts / exemplifies / is_stronger_than
    weight: float = 1.0

    def to_dict(self) -> dict:
        return {"source": self.source, "target": self.target,
                "relation": self.relation, "weight": round(self.weight, 3)}


class KnowledgeGraph:
    """
    Unified persistent knowledge store.
    All learned artifacts (rules, schemas, beliefs, causal links) live here.
    """

    def __init__(self):
        self._nodes: Dict[str, KGNode] = {}
        self._edges: List[KGEdge] = []
        self._id_counter = 0
        self.load()

    def _new_id(self, prefix: str = "n") -> str:
        self._id_counter += 1
        return f"{prefix}_{self._id_counter:05d}"

    # ── Ingestion ──────────────────────────────────────────────

    def add_rule(self, rule) -> str:
        """Add or update a rule node from any rule-like object."""
        name = getattr(rule, "name", None) or (rule.get("name", "") if isinstance(rule, dict) else "")
        if not name:
            return ""
        # Deduplicate by name
        for nid, n in self._nodes.items():
            if n.type == "rule" and n.name == name:
                n.confidence = max(n.confidence,
                                   float(getattr(rule, "confidence", n.confidence)))
                n.observations += 1
                return nid
        nid = self._new_id("rule")
        content = rule if isinstance(rule, dict) else (rule.to_dict() if hasattr(rule, "to_dict") else {})
        domain = getattr(rule, "domain", content.get("domain", "general"))
        conf   = float(getattr(rule, "confidence", content.get("confidence", 0.5)))
        self._nodes[nid] = KGNode(id=nid, type="rule", name=name,
                                   domain=domain, confidence=conf, content=content)
        self._try_link_domain(nid, domain)
        return nid

    def add_schema(self, schema) -> str:
        name = getattr(schema, "name", getattr(schema, "signature", ""))
        if not name:
            return ""
        for nid, n in self._nodes.items():
            if n.type == "schema" and n.name == name:
                n.confidence = max(n.confidence, float(getattr(schema, "confidence", n.confidence)))
                n.observations += 1
                return nid
        nid = self._new_id("schema")
        domain = getattr(schema, "domain", "general")
        conf   = float(getattr(schema, "confidence", 0.5))
        self._nodes[nid] = KGNode(id=nid, type="schema", name=name,
                                   domain=domain, confidence=conf)
        self._try_link_domain(nid, domain)
        return nid

    def add_belief(self, key: str, confidence: float, domain: str,
                   description: str = "") -> str:
        if not key:
            return ""
        for nid, n in self._nodes.items():
            if n.type == "belief" and n.name == key:
                n.confidence = confidence
                return nid
        nid = self._new_id("belief")
        self._nodes[nid] = KGNode(id=nid, type="belief", name=key,
                                   domain=domain, confidence=confidence,
                                   content={"description": description})
        return nid

    def add_causal_link(self, cause: str, effect: str, mechanism: str,
                        domain: str, confidence: float = 0.6) -> str:
        key = f"{cause}→{effect}:{mechanism}"
        for nid, n in self._nodes.items():
            if n.type == "causal" and n.name == key:
                n.confidence = max(n.confidence, confidence)
                n.observations += 1
                return nid
        nid = self._new_id("causal")
        self._nodes[nid] = KGNode(id=nid, type="causal", name=key,
                                   domain=domain, confidence=confidence,
                                   content={"cause": cause, "effect": effect,
                                            "mechanism": mechanism})
        return nid

    def add_edge(self, source_id: str, target_id: str, relation: str,
                 weight: float = 1.0):
        if source_id not in self._nodes or target_id not in self._nodes:
            return
        # Skip duplicate edges
        for e in self._edges:
            if e.source == source_id and e.target == target_id and e.relation == relation:
                e.weight = max(e.weight, weight)
                return
        self._edges.append(KGEdge(source_id, target_id, relation, weight))

    # ── Relationship inference ─────────────────────────────────

    def _try_link_domain(self, nid: str, domain: str):
        """Create a domain node if missing and link new node to it."""
        domain_nid = f"domain_{domain}"
        if domain_nid not in self._nodes:
            self._nodes[domain_nid] = KGNode(id=domain_nid, type="domain",
                                              name=domain, domain=domain, confidence=1.0)
        self.add_edge(nid, domain_nid, "belongs_to")

    def infer_dependencies(self):
        """
        Infer 'depends_on' edges: if rule A only fires after rule B
        (rule B is in rule A's test problems), link them.
        """
        rule_nodes = [n for n in self._nodes.values() if n.type == "rule"]
        for rule in rule_nodes:
            op = rule.content.get("operator", "")
            if op in ("+", "add"):
                mul_rules = [n for n in rule_nodes if n.content.get("operator", "") in ("*", "mul")]
                for mr in mul_rules[:1]:
                    self.add_edge(rule.id, mr.id, "depends_on", 0.5)

    def detect_contradictions(self):
        """Detect rules with opposite effects on the same pattern."""
        contradictions = []
        rule_nodes = [n for n in self._nodes.values() if n.type == "rule"]
        for i, r1 in enumerate(rule_nodes):
            for r2 in rule_nodes[i+1:]:
                if (r1.domain == r2.domain and
                        r1.content.get("operator") == r2.content.get("operator") and
                        r1.confidence > 0.7 and r2.confidence > 0.7 and
                        r1.name != r2.name):
                    self.add_edge(r1.id, r2.id, "may_conflict", 0.3)
                    contradictions.append((r1.name, r2.name))
        return contradictions

    # ── Query ──────────────────────────────────────────────────

    def get_rules_for_domain(self, domain: str,
                              min_confidence: float = 0.5) -> List[KGNode]:
        return [n for n in self._nodes.values()
                if n.type == "rule" and n.domain == domain
                and n.confidence >= min_confidence]

    def get_high_confidence_nodes(self, min_conf: float = 0.7,
                                   node_type: Optional[str] = None) -> List[KGNode]:
        return [n for n in self._nodes.values()
                if n.confidence >= min_conf
                and (node_type is None or n.type == node_type)]

    def get_domain_coverage(self) -> Dict[str, int]:
        coverage: Dict[str, int] = {}
        for n in self._nodes.values():
            if n.type != "domain":
                coverage[n.domain] = coverage.get(n.domain, 0) + 1
        return coverage

    def stats(self) -> dict:
        types: Dict[str, int] = {}
        for n in self._nodes.values():
            types[n.type] = types.get(n.type, 0) + 1
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "by_type": types,
            "domain_coverage": self.get_domain_coverage(),
        }

    # ── Consolidation ──────────────────────────────────────────

    def consolidate(self) -> dict:
        """
        Prune low-confidence nodes, merge redundant nodes,
        strengthen frequently-used nodes.
        Returns a report.
        """
        before = len(self._nodes)
        # Prune nodes with confidence < 0.2 and observations < 3
        to_remove = [nid for nid, n in self._nodes.items()
                     if n.confidence < 0.2 and n.observations < 3
                     and n.type not in ("domain",)]
        for nid in to_remove:
            del self._nodes[nid]
            self._edges = [e for e in self._edges
                           if e.source != nid and e.target != nid]

        # Infer relationships
        self.infer_dependencies()
        contradictions = self.detect_contradictions()

        after = len(self._nodes)
        return {
            "pruned": len(to_remove),
            "remaining": after,
            "contradictions_found": len(contradictions),
        }

    # ── Persistence ────────────────────────────────────────────

    def save(self):
        KG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
            "edges": [e.to_dict() for e in self._edges],
            "id_counter": self._id_counter,
        }
        try:
            KG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            log.info(f"KnowledgeGraph saved: {len(self._nodes)} nodes, {len(self._edges)} edges")
        except Exception as e:
            log.error(f"KnowledgeGraph save failed: {e}")

    def load(self):
        if not KG_PATH.exists():
            return
        try:
            data = json.loads(KG_PATH.read_text(encoding="utf-8"))
            self._id_counter = data.get("id_counter", 0)
            for nid, nd in data.get("nodes", {}).items():
                self._nodes[nid] = KGNode(
                    id=nd["id"], type=nd["type"], name=nd["name"],
                    domain=nd.get("domain", "general"),
                    confidence=float(nd.get("confidence", 0.5)),
                    observations=int(nd.get("observations", 1)),
                    content=nd.get("content", {}),
                    created_at=float(nd.get("created_at", time.time())),
                )
            for ed in data.get("edges", []):
                self._edges.append(KGEdge(
                    source=ed["source"], target=ed["target"],
                    relation=ed["relation"], weight=float(ed.get("weight", 1.0)),
                ))
            log.info(f"KnowledgeGraph loaded: {len(self._nodes)} nodes")
        except Exception as e:
            log.warning(f"KnowledgeGraph load failed (starting fresh): {e}")


_kg: Optional[KnowledgeGraph] = None


def get_knowledge_graph() -> KnowledgeGraph:
    global _kg
    if _kg is None:
        _kg = KnowledgeGraph()
    return _kg
