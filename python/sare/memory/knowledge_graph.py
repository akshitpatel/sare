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
        self._edges.append(KGEdge(source=source_id, target=target_id,
                                  relation=relation, weight=weight))

    # ── Query ─────────────────────────────────────────────────

    def get_related_rules(self, domain: str, rule_name: str) -> List[KGNode]:
        """
        Returns all rules in the same domain that are structurally similar
        or have overlapping operators to the given rule.

        This enables:
          - Analogy transfer between similar rules
          - Cross-domain generalization
          - Conflict detection between related rules
        """
        if not rule_name:
            return []

        # Find the target rule first
        target_rule = None
        for node in self._nodes.values():
            if node.type == "rule" and node.name == rule_name and node.domain == domain:
                target_rule = node
                break

        if not target_rule:
            return []

        related_rules = []
        target_content = target_rule.content

        # Extract operators from the target rule content
        target_operators = self._extract_operators(target_content)

        # Find other rules in the same domain with similar operators
        for node in self._nodes.values():
            if node.type == "rule" and node.domain == domain and node.id != target_rule.id:
                other_content = node.content
                other_operators = self._extract_operators(other_content)

                # Calculate similarity based on operator overlap
                overlap = len(target_operators & other_operators)
                if overlap > 0:
                    node.content["similarity_score"] = overlap / max(len(target_operators), 1)
                    related_rules.append(node)

        # Sort by similarity score (highest first)
        related_rules.sort(key=lambda x: x.content.get("similarity_score", 0), reverse=True)
        return related_rules

    def _extract_operators(self, content: dict) -> Set[str]:
        """Extract mathematical/logical operators from rule content."""
        operators = set()
        if not content:
            return operators

        # Look for common operator patterns in the content
        text = str(content.get("expression", "") or content.get("pattern", ""))
        for op in ["+", "-", "*", "/", "^", "and", "or", "not", "implies", "iff"]:
            if op in text.lower():
                operators.add(op)
        return operators

    # ── Persistence ────────────────────────────────────────────

    def load(self):
        if not KG_PATH.exists():
            return
        try:
            with open(KG_PATH, "r") as f:
                data = json.load(f)
                self._nodes = {k: KGNode(**v) for k, v in data.get("nodes", {}).items()}
                self._edges = [KGEdge(**e) for e in data.get("edges", [])]
                self._id_counter = max(int(k.split("_")[1]) for k in self._nodes.keys()) if self._nodes else 0
        except Exception as e:
            log.error(f"Failed to load knowledge graph: {e}")

    def save(self):
        data = {
            "nodes": {k: v.to_dict() for k, v in self._nodes.items()},
            "edges": [e.to_dict() for e in self._edges]
        }
        KG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(KG_PATH, "w") as f:
            json.dump(data, f, indent=2)

    def _try_link_domain(self, node_id: str, domain: str):
        """Create or update domain node and link to it."""
        domain_node_id = None
        for nid, n in self._nodes.items():
            if n.type == "domain" and n.name == domain:
                domain_node_id = nid
                n.observations += 1
                break
        if not domain_node_id:
            domain_node_id = self._new_id("domain")
            self._nodes[domain_node_id] = KGNode(id=domain_node_id, type="domain",
                                                 name=domain, domain=domain)
        self.add_edge(node_id, domain_node_id, "belongs_to")
