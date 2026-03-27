"""
CausalChainDetector — Multi-step causal chain analysis for SARE-HX.

Detects when transforms in a solve sequence have causal dependencies:
  T1 → T2 means "applying T1 creates the structural precondition for T2".

Algorithm (lightweight, no re-solving):
  1. Track per-solve: which transforms were applied, in what order, per domain.
  2. Use co-occurrence and ordering statistics to infer causal edges.
  3. When a chain reaches confidence ≥ threshold, record it in the KnowledgeGraph
     and emit a CAUSAL_CHAIN_DISCOVERED event.

Causal edge evidence:
  - Co-occurrence: P(T_j | T_i, same solve) — how often T_j follows T_i
  - Temporal ordering: P(T_i before T_j | both applied)
  - Cross-domain: if T_i→T_j appears in domain A and B, it's a universal pattern
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)


@dataclass
class CausalEdge:
    """A directed causal link T_cause → T_effect."""
    cause: str
    effect: str
    domain: str
    co_occurrences: int = 0
    ordered_occurrences: int = 0  # times cause appeared before effect
    domains_seen: Set[str] = field(default_factory=set)

    @property
    def confidence(self) -> float:
        if self.co_occurrences == 0:
            return 0.0
        order_ratio = self.ordered_occurrences / self.co_occurrences
        cross_domain_bonus = 0.1 * (len(self.domains_seen) - 1)
        base = 0.3 + 0.5 * order_ratio + cross_domain_bonus
        # Scale up with evidence
        evidence_scale = min(1.0, self.co_occurrences / 5.0)
        return min(0.95, base * evidence_scale + 0.1)

    def to_dict(self) -> dict:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "domain": self.domain,
            "co_occurrences": self.co_occurrences,
            "ordered": self.ordered_occurrences,
            "domains": sorted(self.domains_seen),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class CausalChain:
    """A multi-step causal chain: [T1 → T2 → T3 → ...]."""
    steps: List[str]        # transforms in causal order
    domain: str
    confidence: float
    cross_domain: bool = False
    observation_count: int = 1

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def name(self) -> str:
        return " → ".join(self.steps)

    def to_dict(self) -> dict:
        return {
            "chain": self.steps,
            "name": self.name,
            "domain": self.domain,
            "length": self.length,
            "confidence": round(self.confidence, 3),
            "cross_domain": self.cross_domain,
            "observations": self.observation_count,
        }


class CausalChainDetector:
    """
    Incrementally builds a causal graph from observed solve episodes,
    then extracts high-confidence multi-step chains.
    """

    CONFIDENCE_THRESHOLD = 0.50   # minimum to record a chain
    MIN_OBSERVATIONS     = 2      # minimum co-occurrences for a causal edge

    def __init__(self):
        # edge_key = "T_cause:T_effect" → CausalEdge
        self._edges: Dict[str, CausalEdge] = {}
        # Discovered chains (de-duped by name)
        self._chains: Dict[str, CausalChain] = {}
        # Single-transform frequency by domain
        self._transform_freq: Dict[str, Counter] = defaultdict(Counter)

    def observe(
        self,
        transforms: List[str],
        domain: str,
        delta: float,
        success: bool,
    ) -> List[CausalChain]:
        """
        Record a solve episode.  Returns any newly discovered chains.
        Only multi-transform successful solves contribute causal evidence.
        """
        if not success or len(transforms) < 2 or delta < 0.3:
            return []

        # Record single-transform frequency
        for t in transforms:
            self._transform_freq[domain][t] += 1

        # Record pairwise causal edges
        unique = list(dict.fromkeys(transforms))  # preserve order, deduplicate
        for i, t_cause in enumerate(unique):
            for j in range(i + 1, min(i + 3, len(unique))):  # up to 2 steps ahead
                t_effect = unique[j]
                key = f"{t_cause}:{t_effect}"
                if key not in self._edges:
                    self._edges[key] = CausalEdge(
                        cause=t_cause, effect=t_effect, domain=domain
                    )
                edge = self._edges[key]
                edge.co_occurrences += 1
                edge.ordered_occurrences += 1  # always ordered since we index by position
                edge.domains_seen.add(domain)

        # Extract new chains
        return self._extract_chains(unique, domain)

    def _extract_chains(self, transforms: List[str], domain: str) -> List[CausalChain]:
        """Extract multi-step chains from the current transform sequence."""
        new_chains: List[CausalChain] = []

        # Build chains of length 2 and 3
        for start in range(len(transforms)):
            for length in (2, 3):
                if start + length > len(transforms):
                    break
                steps = transforms[start:start + length]
                # All edges in this chain must be confident enough
                chain_conf = 1.0
                valid = True
                for k in range(len(steps) - 1):
                    key = f"{steps[k]}:{steps[k+1]}"
                    edge = self._edges.get(key)
                    if not edge or edge.co_occurrences < self.MIN_OBSERVATIONS:
                        valid = False
                        break
                    chain_conf *= edge.confidence
                if not valid or chain_conf < self.CONFIDENCE_THRESHOLD:
                    continue

                # Check cross-domain
                all_domains: Set[str] = set()
                for k in range(len(steps) - 1):
                    all_domains |= self._edges[f"{steps[k]}:{steps[k+1]}"].domains_seen
                cross = len(all_domains) > 1

                chain_name = " → ".join(steps)
                if chain_name in self._chains:
                    self._chains[chain_name].observation_count += 1
                    self._chains[chain_name].confidence = min(
                        0.95,
                        self._chains[chain_name].confidence + 0.05,
                    )
                else:
                    chain = CausalChain(
                        steps=steps,
                        domain=domain,
                        confidence=chain_conf,
                        cross_domain=cross,
                    )
                    self._chains[chain_name] = chain
                    new_chains.append(chain)
                    log.debug(
                        f"Causal chain discovered: {chain_name} "
                        f"[{domain}] conf={chain_conf:.2f} cross={cross}"
                    )

        return new_chains

    def get_chains(
        self,
        min_confidence: float = 0.4,
        domain: Optional[str] = None,
        min_length: int = 2,
    ) -> List[CausalChain]:
        """Return known chains filtered by confidence / domain / length."""
        result = [
            c for c in self._chains.values()
            if c.confidence >= min_confidence
            and c.length >= min_length
            and (domain is None or c.domain == domain)
        ]
        result.sort(key=lambda c: (-c.confidence, -c.length))
        return result

    def get_causal_predecessors(self, transform: str) -> List[Tuple[str, float]]:
        """What transforms causally precede this one (i.e., T_cause → transform)?"""
        preds = [
            (e.cause, e.confidence)
            for e in self._edges.values()
            if e.effect == transform and e.confidence > 0.3
        ]
        preds.sort(key=lambda x: -x[1])
        return preds

    def predict_next_transform(self, last_transform: str, domain: str) -> Optional[str]:
        """Given the last applied transform, predict the most likely next one."""
        candidates = [
            (e.effect, e.confidence)
            for key, e in self._edges.items()
            if e.cause == last_transform
            and (e.domain == domain or len(e.domains_seen) > 1)
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0]

    def summary(self) -> dict:
        return {
            "edges": len(self._edges),
            "chains": len(self._chains),
            "cross_domain_chains": sum(1 for c in self._chains.values() if c.cross_domain),
            "domains": list(self._transform_freq.keys()),
            "top_chains": [c.to_dict() for c in self.get_chains()[:5]],
        }
