"""
FactCompositionEngine — Compositional reasoning over belief triples.

Derives new facts by chaining Beliefs from WorldModel:
  A --pred1--> B  +  B --pred2--> C  →  A --compose(pred1,pred2)--> C
  confidence = min(conf1, conf2) × (0.95 ** (hops - 1))

This is not retrieval — it's inference over what the system already knows.
Novel queries ("Does fire melt ice?") are answered by chaining
  fire→causes→heat  +  heat→melts→ice  →  fire causes-then-melts ice
even though the derived fact was never explicitly stored.
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

# ── Predicate composition table ───────────────────────────────────────────────
# compose(pred1, pred2) → derived predicate
_PRED_TABLE: Dict[Tuple[str, str], str] = {
    ("causes",    "causes"):    "causes",
    ("causes",    "enables"):   "enables",
    ("causes",    "prevents"):  "prevents",
    ("causes",    "produces"):  "produces",
    ("causes",    "melts"):     "melts",
    ("causes",    "boils"):     "boils",
    ("causes",    "destroys"):  "destroys",
    ("causes",    "kills"):     "kills",
    ("causes",    "increases"): "increases",
    ("causes",    "decreases"): "decreases",
    ("enables",   "enables"):   "enables",
    ("enables",   "requires"):  "enables",
    ("enables",   "causes"):    "causes",
    ("requires",  "requires"):  "requires",
    ("requires",  "enables"):   "requires",
    ("produces",  "causes"):    "causes",
    ("produces",  "enables"):   "enables",
    ("produces",  "is"):        "produces",
    ("is-a",      "has"):       "has",
    ("is-a",      "is-a"):      "is-a",
    ("is-a",      "can"):       "can",
    ("is-a",      "causes"):    "causes",
    ("is-a",      "produces"):  "produces",
    ("is_a",      "has"):       "has",
    ("is_a",      "is_a"):      "is_a",
    ("is_a",      "can"):       "can",
    ("is_a",      "causes"):    "causes",
    ("is_a",      "produces"):  "produces",
    ("has",       "causes"):    "causes",
    ("has",       "enables"):   "enables",
    ("has",       "is"):        "has",
    ("type_of",   "has"):       "has",
    ("type_of",   "can"):       "can",
    ("part_of",   "causes"):    "causes",
    ("part_of",   "enables"):   "enables",
    ("leads_to",  "leads_to"):  "leads_to",
    ("leads_to",  "causes"):    "causes",
    ("results_in","results_in"):"results_in",
    ("results_in","causes"):    "causes",
}

# Structural/type predicates — less useful for answer generation
_STRUCTURAL_PREDS = {
    "is_a", "is-a", "type_of", "part_of", "subset_of",
    "dominant_transform", "recommended_action",
}

# Question → (subject, target) extraction patterns
_Q_PATTERNS = [
    # "Does X <verb> Y?" / "Can X <verb> Y?"
    (re.compile(r"(?:does|can|will|could)\s+(\w[\w\s]{0,20}?)\s+"
                r"(\w+)\s+([\w\s]{0,30}?)\??$", re.IGNORECASE),
     lambda m: (m.group(1).strip(), m.group(3).strip())),
    # "What does X cause/produce/enable?"
    (re.compile(r"what\s+(?:does|do)\s+(\w[\w\s]{0,20}?)\s+"
                r"(?:cause|produce|enable|require|prevent|result in|lead to)\??",
                re.IGNORECASE),
     lambda m: (m.group(1).strip(), None)),
    # "What causes/produces/enables X?"
    (re.compile(r"what\s+(?:causes?|produces?|enables?|requires?)\s+"
                r"([\w\s]{0,30}?)\??$", re.IGNORECASE),
     lambda m: (None, m.group(1).strip())),
    # "How does X affect/change Y?"
    (re.compile(r"how\s+does\s+(\w[\w\s]{0,20}?)\s+\w+\s+([\w\s]{0,30}?)\??$",
                re.IGNORECASE),
     lambda m: (m.group(1).strip(), m.group(2).strip())),
    # "What is the relationship between X and Y?"
    (re.compile(r"relationship\s+between\s+([\w\s]{1,20}?)\s+and\s+([\w\s]{1,20}?)\??",
                re.IGNORECASE),
     lambda m: (m.group(1).strip(), m.group(2).strip())),
    # "Why does X <verb>?"
    (re.compile(r"why\s+(?:does|do|is)\s+(\w[\w\s]{0,25}?)\??$", re.IGNORECASE),
     lambda m: (m.group(1).strip(), None)),
]


@dataclass
class DerivedFact:
    subject: str
    predicate: str
    obj: str
    confidence: float
    explanation: str
    path_length: int


class FactCompositionEngine:
    """Compositional reasoning engine — derives new facts by chaining belief triples."""

    _CACHE_MAX = 2000
    _MIN_CONF = 0.30
    _HOP_DECAY = 0.95

    def __init__(self) -> None:
        self._cache: Dict[str, List[DerivedFact]] = {}
        self._cache_order: deque = deque()
        self._graph_ts: float = 0.0
        self._graph: Dict[str, List[Tuple[str, str, float]]] = {}  # subject → [(pred, obj, conf)]
        self._graph_size: int = 0

    # ── Graph building ────────────────────────────────────────────────────────

    def _rebuild_graph(self) -> None:
        """Build adjacency list from all world-model beliefs. Cached until WM changes."""
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            beliefs = wm.get_beliefs()  # List[dict]
        except Exception as e:
            log.debug("[FCE] Cannot read world_model: %s", e)
            return

        graph: Dict[str, List[Tuple[str, str, float]]] = {}
        for b in beliefs:
            subj = str(b.get("subject", "") or "").lower().strip()
            pred = str(b.get("predicate", "") or "").lower().strip()
            val  = str(b.get("value", "") or "").lower().strip()
            conf = float(b.get("confidence", 0.5) or 0.5)
            if not subj or not pred or not val:
                continue
            # Skip structural/administrative predicates
            if pred in _STRUCTURAL_PREDS:
                continue
            graph.setdefault(subj, []).append((pred, val, conf))

        self._graph = graph
        self._graph_size = sum(len(v) for v in graph.values())
        self._graph_ts = time.time()
        log.debug("[FCE] Graph rebuilt: %d nodes, %d edges",
                  len(graph), self._graph_size)

    def _get_graph(self) -> Dict[str, List[Tuple[str, str, float]]]:
        """Return (possibly cached) belief graph. Rebuilt if older than 60s."""
        if time.time() - self._graph_ts > 60.0 or not self._graph:
            self._rebuild_graph()
        return self._graph

    # ── Predicate composition ─────────────────────────────────────────────────

    @staticmethod
    def _compose_predicates(p1: str, p2: str) -> str:
        result = _PRED_TABLE.get((p1, p2))
        if result:
            return result
        # Same predicate composes to itself
        if p1 == p2:
            return p1
        # Asymmetric fallback
        return "related-to"

    @staticmethod
    def _decay_confidence(confidences: List[float]) -> float:
        """min(confs) × 0.95^(hops-1)"""
        if not confidences:
            return 0.0
        return min(confidences) * (0.95 ** max(0, len(confidences) - 1))

    # ── Core BFS composition ──────────────────────────────────────────────────

    def compose(
        self,
        subject: str,
        target: Optional[str] = None,
        max_hops: int = 3,
        min_conf: float = _MIN_CONF,
    ) -> List[DerivedFact]:
        """
        BFS from `subject`, following belief edges up to `max_hops`.
        Compose predicates along each path to derive new facts.
        If `target` given, only return facts where obj matches target.
        Returns results sorted by confidence descending.
        """
        cache_key = f"{subject}|{target}|{max_hops}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        graph = self._get_graph()
        subject_lc = subject.lower().strip()

        if subject_lc not in graph and not any(
            subject_lc in node for node in graph
        ):
            return []

        # BFS state: (current_node, path_so_far)
        # path_so_far: list of (predicate, obj, confidence) tuples
        results: List[DerivedFact] = []
        seen: Set[Tuple[str, ...]] = set()

        # Start from subject node (or any node starting with subject_lc)
        start_nodes = [subject_lc]
        if subject_lc not in graph:
            start_nodes = [n for n in graph if n.startswith(subject_lc)][:3]

        queue: deque = deque()
        for sn in start_nodes:
            queue.append((sn, []))  # (current_node, path)

        while queue:
            node, path = queue.popleft()
            depth = len(path)

            if depth >= max_hops:
                continue

            edges = graph.get(node, [])
            for pred, obj, conf in edges:
                # Build path key to avoid cycles
                path_key = tuple(p[0] + ":" + p[1] for p in path) + (pred + ":" + obj,)
                if path_key in seen:
                    continue
                seen.add(path_key)

                new_path = path + [(pred, obj, conf)]
                all_confs = [c for _, _, c in new_path]
                derived_conf = self._decay_confidence(all_confs)

                if derived_conf < min_conf:
                    continue

                # If depth > 0, we have a multi-hop derived fact
                if depth > 0:
                    # Compose the predicate chain
                    composed_pred = new_path[0][0]
                    for i in range(1, len(new_path)):
                        composed_pred = self._compose_predicates(composed_pred, new_path[i][0])

                    # Build human-readable explanation
                    steps = []
                    cur = subject_lc
                    for p, o, _ in new_path:
                        steps.append(f"{cur} {p} {o}")
                        cur = o
                    explanation = " → ".join(steps)

                    if target is None or target.lower() in obj:
                        derived = DerivedFact(
                            subject=subject_lc,
                            predicate=composed_pred,
                            obj=obj,
                            confidence=round(derived_conf, 3),
                            explanation=explanation,
                            path_length=len(new_path),
                        )
                        results.append(derived)

                # Continue BFS from this object
                if obj in graph:
                    queue.append((obj, new_path))

        # Sort by confidence descending, deduplicate by (pred, obj)
        seen_answers: Set[Tuple[str, str]] = set()
        unique: List[DerivedFact] = []
        for df in sorted(results, key=lambda x: x.confidence, reverse=True):
            k = (df.predicate, df.obj)
            if k not in seen_answers:
                seen_answers.add(k)
                unique.append(df)
            if len(unique) >= 20:
                break

        # Cache
        if len(self._cache) >= self._CACHE_MAX:
            old_key = self._cache_order.popleft()
            self._cache.pop(old_key, None)
        self._cache[cache_key] = unique
        self._cache_order.append(cache_key)

        return unique

    # ── Question answering ────────────────────────────────────────────────────

    def answer_query(self, question: str, domain: str) -> Optional[str]:
        """
        Parse question text → extract subject/target → run compose() → return best answer string.
        Returns None if no composition found with conf >= min_conf.
        """
        q = re.sub(r"\n?(Choices?|Options?)\s*:.*", "", question,
                   flags=re.IGNORECASE | re.DOTALL).strip()

        subject: Optional[str] = None
        target: Optional[str] = None

        for pattern, extractor in _Q_PATTERNS:
            m = pattern.search(q)
            if m:
                try:
                    extracted = extractor(m)
                    if isinstance(extracted, tuple):
                        subject, target = extracted
                    break
                except Exception:
                    continue

        # Fallback: extract first noun phrase (first 2 words after question word)
        if subject is None:
            words = re.sub(r"[^\w\s]", "", q.lower()).split()
            skip = {"what", "who", "why", "how", "when", "where", "does",
                    "do", "is", "are", "can", "will", "the", "a", "an"}
            content_words = [w for w in words if w not in skip]
            if content_words:
                subject = " ".join(content_words[:2])

        if not subject:
            return None

        # Clean subject/target
        subject = re.sub(r"\s+", " ", subject.lower().strip())[:40]
        if target:
            target = re.sub(r"\s+", " ", target.lower().strip())[:40]

        derived = self.compose(subject, target=target, max_hops=3)
        if not derived:
            return None

        best = derived[0]
        # Format as readable answer
        if best.predicate in ("causes", "enables", "produces", "prevents",
                              "requires", "leads_to", "results_in"):
            return f"{best.subject} {best.predicate} {best.obj} ({best.explanation})"
        return best.obj

    def invalidate_cache(self) -> None:
        """Force graph rebuild on next call (call after world_model updates)."""
        self._graph_ts = 0.0
        self._cache.clear()
        self._cache_order.clear()


# ── Module-level singleton ────────────────────────────────────────────────────
_FCE_SINGLETON: Optional[FactCompositionEngine] = None


def get_fact_composer() -> FactCompositionEngine:
    global _FCE_SINGLETON
    if _FCE_SINGLETON is None:
        _FCE_SINGLETON = FactCompositionEngine()
    return _FCE_SINGLETON
