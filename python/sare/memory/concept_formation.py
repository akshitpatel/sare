"""
Grounded Concept Formation (Pillar 3 — Human Brain Architecture)

Rather than waiting for humans to label knowledge seeds, SARE-HX discovers
its own concept ontology by clustering graph patterns it has actually solved.

Each time SARE-HX succeeds at a problem:
  1. The solved graph is fingerprinted into a sparse feature vector.
  2. The vector is stored in `ConceptMemory`.
  3. Periodically, K-Means clustering groups similar fingerprints together.
  4. Each cluster is given an auto-name by the LLM ("distributivity", "identity_add").
  5. These named clusters become first-class concepts in the ConceptRegistry.

This mirrors how human children form the concept of "dog" — not from a dictionary
definition, but from seeing hundreds of dogs and clustering their features.
"""

import logging
import json
import math
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from sare.engine import Graph

log = logging.getLogger(__name__)

# ── Graph Fingerprinting ──────────────────────────────────────────────────────

_OPERATOR_VOCAB = ["+", "-", "*", "/", "^", "=", "and", "or", "not", "implies"]
_TYPE_VOCAB = ["operator", "variable", "constant", "predicate"]

def fingerprint(graph: Graph) -> List[float]:
    """
    Converts a Graph into a fixed-length feature vector for clustering.
    
    Features:
     - Operator frequency histogram (10 dims)
     - Node type histogram (4 dims)
     - Graph depth / breadth ratios (2 dims)
     - Leaf ratio (1 dim)
     = 17-dimensional vector
    """
    vec = [0.0] * (len(_OPERATOR_VOCAB) + len(_TYPE_VOCAB) + 3)

    n_nodes = len(graph.nodes)
    if n_nodes == 0:
        return vec

    # Operator histogram
    for node in graph.nodes:
        label = getattr(node, "label", "") or ""
        if label in _OPERATOR_VOCAB:
            vec[_OPERATOR_VOCAB.index(label)] += 1.0

    # Node type histogram
    for node in graph.nodes:
        t = getattr(node, "type", "") or ""
        if t in _TYPE_VOCAB:
            vec[len(_OPERATOR_VOCAB) + _TYPE_VOCAB.index(t)] += 1.0

    # Structural stats
    n_edges = len(graph.edges)
    leaf_count = sum(1 for n in graph.nodes if not any(e.source == n.id for e in graph.edges))

    offset = len(_OPERATOR_VOCAB) + len(_TYPE_VOCAB)
    vec[offset]     = n_edges / max(n_nodes, 1)       # edge density
    vec[offset + 1] = leaf_count / max(n_nodes, 1)    # leaf ratio
    vec[offset + 2] = float(n_nodes)                   # scale

    # L2 normalize
    mag = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / mag for x in vec]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot  # Already L2-normalized


# ── In-Memory Concept Store ───────────────────────────────────────────────────

class ConceptMemory:
    """
    Rolling buffer of (fingerprint, problem_id, transforms) tuples.
    Acts as the raw experience stream for concept formation.
    """
    PERSIST_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "concept_memory.json"
    MAX_CAPACITY = 500

    def __init__(self):
        self._episodes: List[dict] = []

    def record(self, graph: Graph, problem_id: str, transforms: List[str]):
        """Store a solved episode's graph fingerprint."""
        fp = fingerprint(graph)
        self._episodes.append({
            "problem_id": problem_id,
            "fingerprint": fp,
            "transforms": transforms,
        })
        # Rolling window: keep only the most recent MAX_CAPACITY episodes
        if len(self._episodes) > self.MAX_CAPACITY:
            self._episodes = self._episodes[-self.MAX_CAPACITY:]

    def save(self):
        self.PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.PERSIST_PATH, "w") as f:
            json.dump(self._episodes, f)
        log.info(f"ConceptMemory saved: {len(self._episodes)} episodes.")

    def load(self):
        if not self.PERSIST_PATH.exists():
            return
        with open(self.PERSIST_PATH) as f:
            self._episodes = json.load(f)
        log.info(f"ConceptMemory loaded: {len(self._episodes)} episodes.")

    def __len__(self):
        return len(self._episodes)

    def retrieve_similar(self, graph, top_k: int = 3) -> List[dict]:
        """
        Find the most similar past episodes to the given graph footprint.
        Returns a list of episodes sorted by similarity descending.
        """
        if not self._episodes:
            return []
        
        target = fingerprint(graph)
        scored = []
        for ep in self._episodes:
            if not ep.get("fingerprint"):
                continue
            sim = cosine_similarity(target, ep["fingerprint"])
            scored.append((sim, ep))
        
        # Sort by similarity descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"similarity": float(s), "episode": ep} for s, ep in scored[:top_k] if s > 0.5]


# ── Concept Formation (K-Means style clustering) ─────────────────────────────

class ConceptFormation:
    """
    Discovers named concepts by clustering solved graph fingerprints.
    Runs asynchronously after each batch of solves.
    """
    MIN_EPISODES_TO_CLUSTER = 10  # Don't bother until we have enough data
    N_CLUSTERS = 6

    def __init__(self, memory: ConceptMemory, concept_registry=None, llm_namer=None):
        self.memory = memory
        self.registry = concept_registry
        self.llm_namer = llm_namer  # Optional: callable(centroid_info) -> str
        self._concepts: List[dict] = []  # {name, centroid, member_problems}

    def run(self) -> List[dict]:
        """
        Run concept formation. Returns list of discovered concept dicts.
        Requires at least MIN_EPISODES_TO_CLUSTER episodes in memory.
        """
        if len(self.memory) < self.MIN_EPISODES_TO_CLUSTER:
            log.info(f"ConceptFormation: Not enough data yet ({len(self.memory)} episodes, need {self.MIN_EPISODES_TO_CLUSTER}).")
            return []

        episodes = self.memory._episodes
        fingerprints = [e["fingerprint"] for e in episodes]

        # Try sklearn K-Means first; fall back to our own mini K-Means
        try:
            import numpy as np
            from sklearn.cluster import KMeans
            X = np.array(fingerprints)
            k = min(self.N_CLUSTERS, len(fingerprints))
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            centroids = km.cluster_centers_.tolist()
        except ImportError:
            labels, centroids = self._mini_kmeans(fingerprints, self.N_CLUSTERS)

        # Group episodes by cluster
        clusters: Dict[int, List[dict]] = {}
        for i, ep in enumerate(episodes):
            c = int(labels[i])
            clusters.setdefault(c, []).append(ep)

        # Generate concept names
        self._concepts = []
        for c_id, members in clusters.items():
            centroid = centroids[c_id]
            name = self._name_cluster(c_id, members, centroid)
            concept = {
                "name": name,
                "centroid": centroid,
                "member_count": len(members),
                "member_problems": [m["problem_id"] for m in members[:5]],
                "common_transforms": self._common_transforms(members),
            }
            self._concepts.append(concept)
            log.info(f"Concept discovered: '{name}' ({len(members)} members)")

            # Register in ConceptRegistry if available
            if self.registry and hasattr(self.registry, "add_rule"):
                rule = {
                    "name": name,
                    "domain": "auto_discovered",
                    "confidence": min(0.5 + len(members) * 0.05, 0.95),
                    "observations": len(members),
                }
                self.registry.add_rule(rule)

        return self._concepts

    def _name_cluster(self, c_id: int, members: List[dict], centroid: List[float]) -> str:
        """Ask LLM to give this cluster a human-readable concept name (Language Grounding)."""
        sample_problems = [m["problem_id"] for m in members[:4]]
        common_t = self._common_transforms(members)

        # Try LLM naming first (Feature 3: Online Concept Naming)
        try:
            if self.llm_namer:
                # If a custom callable is provided, use it
                name = self.llm_namer(sample_problems, common_t)
                if name:
                    return name
            else:
                from sare.interface.llm_bridge import _call_llm
                prompt = (
                    "You are the Teacher/Oracle of a self-learning symbolic AI (SARE-HX).\n"
                    "SARE-HX has just formed a new cognitive concept by clustering the following mathematical problems "
                    "that it solved using similar structural transformations.\n\n"
                    f"Sample problems in this cluster: {sample_problems}\n"
                    f"Common structural transforms used: {common_t}\n\n"
                    "Teach SARE the established human natural language vocabulary for this concept.\n"
                    "Return ONLY a JSON object with two keys:\n"
                    '{"concept_name": "short_snake_case_name", "description": "one sentence"}\n'
                    "Examples: 'additive_identity', 'distributive_property', 'modus_ponens', 'cancellation_pattern'"
                )
                raw = _call_llm(prompt)
                raw = re.sub(r"```(?:json)?", "", raw).strip("`").strip()
                
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    data = json.loads(m.group(0))
                    name = str(data.get("concept_name", "")).strip().lower().replace(" ", "_")
                    if name:
                        log.info(f"LLM Teacher grounded cluster {c_id} as: '{name}'")
                        return name
        except Exception as e:
            log.warning(f"LLM Language Grounding failed for cluster {c_id}: {e}")

        # Structural fallback
        op_idx = centroid[:len(_OPERATOR_VOCAB)]
        if op_idx and max(op_idx) > 0:
            dominant_op = _OPERATOR_VOCAB[op_idx.index(max(op_idx))]
            return f"auto_concept_{dominant_op}_cluster_{c_id}"
        return f"auto_concept_{c_id}"

    @staticmethod
    def _common_transforms(members: List[dict]) -> List[str]:
        """Find transforms that appear in >50% of cluster members."""
        from collections import Counter
        all_t = [t for m in members for t in m.get("transforms", [])]
        counts = Counter(all_t)
        threshold = len(members) * 0.5
        return [t for t, n in counts.items() if n >= threshold]

    @staticmethod
    def _mini_kmeans(fingerprints: List[List[float]], k: int) -> Tuple[List[int], List[List[float]]]:
        """Pure-Python mini K-Means for when sklearn is unavailable."""
        import random
        k = min(k, len(fingerprints))
        centroids = random.sample(fingerprints, k)

        for _ in range(20):  # Max iterations
            labels = []
            for fp in fingerprints:
                sims = [cosine_similarity(fp, c) for c in centroids]
                labels.append(sims.index(max(sims)))

            # Update centroids
            new_centroids = []
            for c in range(k):
                members = [fingerprints[i] for i, l in enumerate(labels) if l == c]
                if members:
                    mean = [sum(f[d] for f in members) / len(members) for d in range(len(members[0]))]
                    mag = math.sqrt(sum(x * x for x in mean)) or 1.0
                    new_centroids.append([x / mag for x in mean])
                else:
                    new_centroids.append(centroids[c])
            centroids = new_centroids

        return labels, centroids
