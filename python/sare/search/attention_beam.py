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

# ── Optional GNN embedding (torch required) ───────────────────────────────────
_GNN_AVAILABLE = False
_graph_embed_model = None

# Eagerly try to load at import time so web server picks it up
def _try_load_gnn_eager():
    """Called at module import — non-blocking, best-effort."""
    try:
        import torch
        from sare.heuristics.graph_embedding import GraphEmbedding as _GE
        global _GNN_AVAILABLE, _graph_embed_model
        _graph_embed_model = _GE(hidden_dim=64, num_layers=2)
        _graph_embed_model.eval()
        _GNN_AVAILABLE = True
    except Exception:
        pass

_try_load_gnn_eager()


def _try_load_gnn():
    """Lazily load the GNN graph embedding model (torch may not always be present)."""
    global _GNN_AVAILABLE, _graph_embed_model
    if _graph_embed_model is not None:
        return True
    try:
        import torch
        from sare.heuristics.graph_embedding import GraphEmbedding
        _graph_embed_model = GraphEmbedding(hidden_dim=64, num_layers=2)
        _graph_embed_model.eval()
        _GNN_AVAILABLE = True
        log.info("[AttentionBeam] GNN GraphEmbedding loaded (torch %s)", torch.__version__)
        return True
    except Exception as e:
        log.debug("[AttentionBeam] GNN unavailable, using Jaccard fallback: %s", e)
        return False


def _gnn_embed(graph) -> Optional[List[float]]:
    """Compute a GNN embedding vector for a graph. Returns None on failure.
    Encodes type+label combined key so structurally-isomorphic but label-different
    graphs get distinct embeddings."""
    if not _GNN_AVAILABLE and not _try_load_gnn():
        return None
    try:
        import torch
        nodes = getattr(graph, "nodes", [])
        if not nodes:
            return None
        encoder = _graph_embed_model.encoder
        # Encode type+label together for richer discrimination
        type_indices = torch.tensor(
            [encoder.get_type_idx(f"{getattr(n,'type','op')}:{getattr(n,'label','')}") for n in nodes],
            dtype=torch.long
        )
        id_to_idx = {getattr(n, "id", i): i for i, n in enumerate(nodes)}
        edges = getattr(graph, "edges", [])
        adjacency = []
        for e in edges:
            s = id_to_idx.get(getattr(e, "source", None))
            t = id_to_idx.get(getattr(e, "target", None))
            if s is not None and t is not None:
                adjacency.append((s, t))
        with torch.no_grad():
            emb = _graph_embed_model(type_indices, adjacency)
        return emb.tolist()
    except Exception as exc:
        log.debug("[AttentionBeam] GNN embed failed: %s", exc)
        return None


def _cosine_sim(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two float lists."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


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
        self._seen_signatures: List[str] = []    # rolling seen-state fingerprints (Jaccard fallback)
        self._seen_embeddings: List[List[float]] = []  # GNN embedding cache
        self._current_problem_sig: Optional[str] = None  # tracks current problem root
        # Try to load GNN at construction time (non-blocking)
        _try_load_gnn()

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

    def begin_problem(self, root_graph=None) -> None:
        """Reset seen-state cache for a new independent problem."""
        self.reset()
        if root_graph is not None:
            self._current_problem_sig = self._signature(root_graph)

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
            "seen_states":      len(self._seen_signatures) + len(self._seen_embeddings),
            "novelty_mode":     "gnn_cosine" if _GNN_AVAILABLE else "jaccard_fallback",
            "type":             "AttentionBeamScorer",
        }

    def reset(self):
        """Clear seen-state cache (call between independent problems)."""
        self._seen_signatures.clear()
        self._seen_embeddings.clear()

    def sync_dopamine(self):
        """Pull curiosity_bonus from DopamineSystem and update gamma live."""
        try:
            from sare.neuro.dopamine import get_dopamine_system
            ds = get_dopamine_system()
            # curiosity_bonus ∈ [0, 0.5]; scale to gamma range [0.05, 0.60]
            self.gamma = 0.05 + ds.curiosity_bonus * 1.10
        except Exception:
            pass

    def set_transform_uncertainty(self, uncertainty_map: dict) -> None:
        """Update per-transform uncertainty estimates from credit assigner."""
        self._transform_uncertainty = dict(uncertainty_map)

    def _get_node_uncertainty(self, graph) -> float:
        """Compute mean uncertainty across transforms that recently touched this graph type."""
        if not hasattr(self, '_transform_uncertainty') or not self._transform_uncertainty:
            return 0.0
        vals = list(self._transform_uncertainty.values())
        return sum(vals) / len(vals) if vals else 0.0

    # ── Private helpers ───────────────────────────────────────────────────────

    def _mean_uncertainty(self, graph) -> float:
        """
        Compute mean node.uncertainty across all graph nodes.
        Falls back to _get_node_uncertainty (credit-assigner-based) when
        nodes do not expose uncertainty attributes.
        """
        try:
            nodes = getattr(graph, "nodes", [])
            if not nodes:
                return self._get_node_uncertainty(graph)
            uncertainties = []
            for n in nodes:
                u = getattr(n, "uncertainty", None)
                if u is not None:
                    uncertainties.append(float(u))
            if not uncertainties:
                return self._get_node_uncertainty(graph)
            return sum(uncertainties) / len(uncertainties)
        except Exception:
            return 0.0

    def _novelty_score(self, graph) -> float:
        """
        Estimate novelty as structural distance from seen states.
        Uses GNN cosine similarity when torch is available (learned representation),
        falls back to token Jaccard on node-type sequences.
        Returns a score in [0, 1] — higher = more novel.
        """
        # ── GNN path (learned structural similarity) ──────────────────────────
        emb = _gnn_embed(graph)
        if emb is not None:
            if not self._seen_embeddings:
                self._seen_embeddings.append(emb)
                return 1.0  # first state is maximally novel
            # Find max cosine similarity to any seen embedding (most similar = least novel)
            max_sim = max(_cosine_sim(emb, seen) for seen in self._seen_embeddings[-64:])
            novelty = 1.0 - max(0.0, min(1.0, max_sim))
            # Cache this embedding
            self._seen_embeddings.append(emb)
            if len(self._seen_embeddings) > 512:
                self._seen_embeddings = self._seen_embeddings[-256:]
            return novelty

        # ── Jaccard fallback (no torch) ────────────────────────────────────────
        if not self._seen_signatures:
            return 1.0

        sig = self._signature(graph)
        if sig in self._seen_signatures:
            return 0.0

        sig_tokens = set(sig.split("|"))
        min_sim = 1.0
        for seen in self._seen_signatures[-64:]:
            seen_tokens = set(seen.split("|"))
            union = sig_tokens | seen_tokens
            if not union:
                continue
            sim = len(sig_tokens & seen_tokens) / len(union)
            min_sim = min(min_sim, sim)

        return 1.0 - min_sim

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
