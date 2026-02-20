"""
AttentionBeam — TODO-B: Attention-Weighted Beam Search Scoring
==============================================================
Replaces uniform beam selection with uncertainty-aware attention scoring.

Human working memory isn't a FIFO queue — it prioritises surprising,
uncertain, or high-potential states. This module adds that bias to SARE's
BeamSearch by re-scoring beam candidates using:

    attention_score(state) = energy(state) - β * uncertainty(state)
                             + γ * novelty(state)

Where:
  - energy(state)      : standard energy (lower = better, as before)
  - uncertainty(state) : mean node.uncertainty across the graph
                         (high uncertainty = under-explored region, worthy of attention)
  - novelty(state)     : structural distance from already-seen states
                         (novel states get a bonus to prevent circular search)
  - β                  : uncertainty exploration weight (default 0.3)
  - γ                  : novelty bonus weight (default 0.2)

The net effect: the beam doesn't just chase the lowest-energy path.
It also keeps "surprising" and "novel" states alive, preventing premature
commitment to a local minimum.

Usage::

    scorer = AttentionBeamScorer(beta=0.3, gamma=0.2)
    
    # Rerank a list of (graph_state, energy) tuples
    ranked = scorer.rerank(candidates)
    top_k = ranked[:beam_width]

    # Or use as a drop-in graph scorer
    score = scorer.score_state(graph, energy)
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass
class BeamState:
    """Wraps a graph state with all scoring components visible."""
    graph:           Any           # Graph object (Python or C++ wrapper)
    energy:          float         # Raw energy from EnergyEvaluator
    uncertainty:     float         # Mean node uncertainty
    novelty:         float         # Distance from seen states
    attention_score: float         # Final composite score (lower = prioritised)
    transforms_path: List[str]     # Transforms applied to reach this state


class AttentionBeamScorer:
    """
    Attention-weighted scorer for BeamSearch.

    Parameters
    ----------
    beta  : uncertainty exploration weight. Higher = more exploration of
            uncertain/novel regions (0 = standard greedy beam).
    gamma : novelty bonus weight. Higher = more diverse beam.
    dedup_threshold : cosine similarity above which two states are considered
                      duplicates (the lower-scoring is pruned).
    """

    def __init__(
        self,
        beta:  float = 0.30,   # uncertainty bonus
        gamma: float = 0.15,   # novelty bonus
        dedup_threshold: float = 0.95,
    ):
        self.beta  = beta
        self.gamma = gamma
        self.dedup_threshold = dedup_threshold
        self._seen_signatures: List[str] = []    # rolling seen-state fingerprints

    # ── Main API ──────────────────────────────────────────────────────────────

    def score_state(
        self,
        graph,
        energy: float,
        transforms_path: Optional[List[str]] = None,
    ) -> BeamState:
        """
        Compute the attention score for a single beam state.
        
        Returns BeamState with all components populated.
        """
        uncertainty = self._mean_uncertainty(graph)
        novelty     = self._novelty_score(graph)
        sig         = self._signature(graph)

        # Core formula: lower = better
        # Subtract uncertainty*beta and novelty*gamma to give bonuses
        # (uncertainty and novelty reduce the effective energy, making
        #  exploratory states more competitive)
        attention = energy - self.beta * uncertainty - self.gamma * novelty

        return BeamState(
            graph=graph,
            energy=energy,
            uncertainty=uncertainty,
            novelty=novelty,
            attention_score=attention,
            transforms_path=transforms_path or [],
        )

    def rerank(
        self,
        candidates: List[Tuple[Any, float]],
        beam_width: int = 8,
        transforms_paths: Optional[List[List[str]]] = None,
    ) -> List[BeamState]:
        """
        Re-score and rerank a list of (graph, energy) candidates.

        Parameters
        ----------
        candidates      : list of (graph, energy) pairs from the search engine.
        beam_width      : max states to return.
        transforms_paths: optional list of transform histories per state.

        Returns
        -------
        Sorted list of BeamState, best first (lowest attention_score).
        """
        paths  = transforms_paths or [[] for _ in candidates]
        states = [
            self.score_state(g, e, p)
            for (g, e), p in zip(candidates, paths)
        ]

        # Deduplicate structurally-identical states
        states = self._deduplicate(states)

        # Sort: lower attention_score = higher priority
        states.sort(key=lambda s: s.attention_score)

        # Update seen-signature cache
        for s in states[:beam_width]:
            sig = self._signature(s.graph)
            if sig not in self._seen_signatures:
                self._seen_signatures.append(sig)
                if len(self._seen_signatures) > 512:  # cap memory
                    self._seen_signatures = self._seen_signatures[-256:]

        return states[:beam_width]

    def summary(self) -> Dict[str, Any]:
        """Return current scorer parameters for logging/UI."""
        return {
            "beta":             self.beta,
            "gamma":            self.gamma,
            "dedup_threshold":  self.dedup_threshold,
            "seen_states":      len(self._seen_signatures),
            "type":             "AttentionBeamScorer",
        }

    def reset(self):
        """Clear seen-state cache (call between independent problems)."""
        self._seen_signatures.clear()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _mean_uncertainty(self, graph) -> float:
        """
        Compute mean node.uncertainty across all graph nodes.
        Returns 0.0 if the graph doesn't expose uncertainty.
        """
        try:
            nodes = getattr(graph, "nodes", [])
            if not nodes:
                return 0.0
            uncertainties = []
            for n in nodes:
                u = getattr(n, "uncertainty", None)
                if u is not None:
                    uncertainties.append(float(u))
            if not uncertainties:
                return 0.0
            return sum(uncertainties) / len(uncertainties)
        except Exception:
            return 0.0

    def _novelty_score(self, graph) -> float:
        """
        Estimate novelty as mean min-distance from seen signatures.
        Simple token-overlap distance over node-type sequences.
        Returns a score in [0, 1] — higher = more novel.
        """
        if not self._seen_signatures:
            return 1.0  # first state is maximally novel

        sig = self._signature(graph)
        if sig in self._seen_signatures:
            return 0.0  # exact duplicate — no novelty

        # Token Jaccard distance to nearest seen state
        sig_tokens = set(sig.split("|"))
        min_sim = 1.0
        for seen in self._seen_signatures[-64:]:  # check last 64
            seen_tokens = set(seen.split("|"))
            union = sig_tokens | seen_tokens
            if not union:
                continue
            sim = len(sig_tokens & seen_tokens) / len(union)
            min_sim = min(min_sim, sim)

        return 1.0 - min_sim   # sim->1 = low novelty, sim->0 = high novelty

    def _signature(self, graph) -> str:
        """
        Cheap structural fingerprint for deduplication.
        Uses node-type sequence sorted for order-invariance.
        """
        try:
            nodes = getattr(graph, "nodes", [])
            types = sorted(getattr(n, "type", "?") for n in nodes)
            edges = getattr(graph, "edges", [])
            edge_sigs = sorted(
                f"{getattr(e,'source','')}>{getattr(e,'target','')}"
                for e in edges
            )
            return "|".join(types + edge_sigs)
        except Exception:
            return "unknown"

    def _deduplicate(self, states: List[BeamState]) -> List[BeamState]:
        """Remove structurally near-duplicate states (keep lower attention_score)."""
        kept: List[BeamState] = []
        seen_sigs: List[str]  = []

        for state in sorted(states, key=lambda s: s.attention_score):
            sig  = self._signature(state.graph)
            s_tokens = set(sig.split("|"))
            is_dup = False
            for ks in seen_sigs:
                k_tokens = set(ks.split("|"))
                union = s_tokens | k_tokens
                if not union:
                    continue
                sim = len(s_tokens & k_tokens) / len(union)
                if sim >= self.dedup_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(state)
                seen_sigs.append(sig)

        return kept


# ── Default singleton (import and use directly) ───────────────────────────────
_default_scorer: Optional[AttentionBeamScorer] = None

def get_default_scorer() -> AttentionBeamScorer:
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = AttentionBeamScorer()
    return _default_scorer
