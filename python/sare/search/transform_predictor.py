import json, time, logging, threading
from pathlib import Path
from typing import List, Dict, Optional

log = logging.getLogger(__name__)

_STATS_PATH = Path(__file__).resolve().parents[3] / "data/memory/transform_stats.json"

class TransformPredictor:
    """
    Learns (graph_signature → best_transforms) purely from observed outcomes.
    Graph signature = sorted operator labels + node count (e.g. "NOT,+|5")
    """

    def __init__(self):
        self._stats: Dict[str, Dict[str, List[float]]] = {}
        # _stats[graph_sig][transform_name] = [delta1, delta2, ...]
        self._observations = 0
        self._lock = threading.Lock()
        self.load()

    def _graph_signature(self, graph) -> str:
        """Compact key: sorted operator labels + node count"""
        try:
            ops = sorted(set(
                n.label for n in graph.nodes
                if n.type == "operator"
            ))
            size = len(graph.nodes)
            return f"{','.join(ops)}|{size}"
        except Exception:
            return "unknown|0"

    def observe(self, graph, transform_name: str, delta: float):
        """Record that applying transform_name to graph gave energy delta."""
        sig = self._graph_signature(graph)
        with self._lock:
            if sig not in self._stats:
                self._stats[sig] = {}
            if transform_name not in self._stats[sig]:
                self._stats[sig][transform_name] = []
            # Cap at 50 observations per (sig, transform) pair
            lst = self._stats[sig][transform_name]
            if len(lst) < 50:
                lst.append(delta)
            else:
                lst[self._observations % 50] = delta  # circular buffer
            self._observations += 1

    def _llm_bootstrap_order(self, graph_sig: str, transforms: list) -> list:
        """Ask LLM to suggest transform ordering for an unseen graph signature."""
        try:
            from sare.interface.llm_bridge import _call_llm
            names = [t.name() for t in transforms]
            prompt = (
                f"Graph signature: {graph_sig}\n"
                f"Available transforms: {', '.join(names)}\n"
                f"Order these transforms best-first for simplifying this expression type. "
                f"Reply with ONLY a comma-separated list of transform names in the order you recommend."
            )
            resp = _call_llm(prompt).strip()
            ordered_names = [n.strip() for n in resp.split(',') if n.strip()]
            name_to_t = {t.name(): t for t in transforms}
            result = [name_to_t[n] for n in ordered_names if n in name_to_t]
            # Append any not mentioned by LLM at the end
            result += [t for t in transforms if t not in result]
            return result if len(result) == len(transforms) else transforms
        except Exception:
            return transforms

    def predict_best_transforms(self, graph, transforms: list, top_k: int = None) -> list:
        """
        Reorder transforms by predicted energy delta for this graph type.
        Returns transforms sorted best-first (highest predicted delta = most simplification).
        Falls back to LLM bootstrap for unseen signatures, then original order.
        """
        sig = self._graph_signature(graph)
        sig_data = self._stats.get(sig, {})

        if not sig_data:
            return self._llm_bootstrap_order(sig, transforms)

        def avg_delta(t) -> float:
            deltas = sig_data.get(t.name(), [])
            return sum(deltas) / len(deltas) if deltas else 0.0

        scored = sorted(transforms, key=avg_delta, reverse=True)
        if top_k:
            return scored[:top_k] + [t for t in transforms if t not in scored[:top_k]]
        return scored

    def get_stats(self) -> dict:
        """Return summary statistics."""
        total_sig = len(self._stats)
        total_obs = sum(
            sum(len(v) for v in t.values())
            for t in self._stats.values()
        )
        top_transforms = {}
        for sig, transforms in self._stats.items():
            for tname, deltas in transforms.items():
                avg = sum(deltas) / len(deltas) if deltas else 0
                if avg > 0:
                    top_transforms[tname] = top_transforms.get(tname, [])
                    top_transforms[tname].append(avg)
        best = {k: sum(v)/len(v) for k, v in top_transforms.items() if v}
        return {
            "graph_signatures_seen": total_sig,
            "total_observations": total_obs,
            "best_transforms_overall": sorted(best.items(), key=lambda x: -x[1])[:5]
        }

    def save(self) -> bool:
        try:
            _STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_STATS_PATH, "w") as f:
                json.dump(self._stats, f)
            return True
        except Exception as e:
            log.error("TransformPredictor save failed (stat loss): %s", e)
            return False

    def load(self):
        try:
            if _STATS_PATH.exists():
                with open(_STATS_PATH) as f:
                    self._stats = json.load(f)
        except Exception:
            self._stats = {}

_predictor_singleton = None

def get_transform_predictor() -> TransformPredictor:
    global _predictor_singleton
    if _predictor_singleton is None:
        _predictor_singleton = TransformPredictor()
    return _predictor_singleton
