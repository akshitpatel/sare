"""
NeuralPerception — Lightweight embedding-based concept recognition.

Uses TF-IDF style embeddings (numpy only, no GPU) to match raw observations
against known WorldModel concepts. This gives SARE a form of perceptual
grounding without requiring a full vision model.

For images: expects text captions/descriptions (alt text, OCR output)
For text: direct embedding
For structured data: field names + values as text

This is the "sensory cortex" analogy — raw inputs → symbolic beliefs.
"""
from __future__ import annotations
import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

VOCAB_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "perception_vocab.json"

# Concept vocabulary with their key terms (used for embedding matching)
_CONCEPT_VOCAB = {
    "addition":         ["add", "plus", "sum", "total", "combine", "+"],
    "subtraction":      ["subtract", "minus", "difference", "remove", "-"],
    "multiplication":   ["multiply", "times", "product", "×", "*"],
    "division":         ["divide", "quotient", "ratio", "split", "/"],
    "equality":         ["equal", "same", "equivalent", "=", "balance"],
    "inequality":       ["greater", "less", "unequal", ">", "<", "!="],
    "variable":         ["unknown", "variable", "x", "y", "z", "solve"],
    "equation":         ["equation", "formula", "expression", "solve"],
    "linear":           ["linear", "straight", "slope", "line"],
    "quadratic":        ["square", "quadratic", "parabola", "x²", "power"],
    "logic":            ["if", "then", "and", "or", "not", "implies", "true", "false"],
    "causation":        ["cause", "effect", "because", "therefore", "leads"],
    "pattern":          ["pattern", "sequence", "rule", "repeat", "series"],
    "identity":         ["identity", "unchanged", "same", "itself"],
    "zero":             ["zero", "nothing", "empty", "null", "0"],
    "one":              ["one", "unit", "single", "1", "unity"],
    "commute":          ["commute", "order", "swap", "reverse", "rearrange"],
    "distribute":       ["distribute", "expand", "factor", "spread"],
    "simplify":         ["simplify", "reduce", "cancel", "combine"],
    "physics":          ["force", "mass", "velocity", "energy", "motion", "gravity"],
    "geometry":         ["angle", "triangle", "circle", "area", "perimeter", "shape"],
    "probability":      ["probability", "chance", "random", "likely", "odds"],
}


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric."""
    return re.findall(r'[a-z0-9]+', text.lower())


def _embed(tokens: List[str], vocab: Dict[str, List[str]]) -> np.ndarray:
    """
    Create a concept vector from tokens.

    Each dimension corresponds to a concept in _CONCEPT_VOCAB.
    Value = number of vocab terms for that concept found in tokens.
    """
    concepts = list(vocab.keys())
    vec = np.zeros(len(concepts), dtype=float)
    token_set = set(tokens)
    token_counter = Counter(tokens)

    for i, concept in enumerate(concepts):
        terms = vocab[concept]
        score = sum(token_counter.get(term, 0) for term in terms)
        # Bonus for exact term matches
        score += sum(2 for term in terms if term in token_set)
        vec[i] = score

    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class NeuralPerception:
    """
    Lightweight neural-style perception using TF-IDF concept embeddings.

    No GPU, no transformers — just numpy. Fast enough to run in the daemon loop.
    """

    def __init__(self, world_model=None):
        self._wm = world_model
        self._vocab = _CONCEPT_VOCAB.copy()
        self._history: List[dict] = []
        self._total_perceptions = 0
        self._concept_list = list(self._vocab.keys())
        # Pre-compute concept embeddings
        self._concept_embeddings = {
            c: _embed(_tokenize(" ".join(terms)), self._vocab)
            for c, terms in self._vocab.items()
        }

    def perceive(self, text: str, source: str = "text", domain: str = "general") -> List[dict]:
        """
        Perceive raw text and return a list of activated concept beliefs.

        Returns: list of {concept, confidence, domain} dicts — same format as WorldModel beliefs.
        """
        if not text or not text.strip():
            return []

        tokens = _tokenize(text)
        if len(tokens) < 2:
            return []

        query_vec = _embed(tokens, self._vocab)

        # Match against all concepts
        activations = []
        for concept, cvec in self._concept_embeddings.items():
            sim = _cosine_similarity(query_vec, cvec)
            if sim > 0.15:  # threshold
                activations.append({
                    "concept": concept,
                    "confidence": round(float(sim), 3),
                    "domain": domain,
                    "source": source,
                    "text": text[:80],
                })

        # Sort by confidence descending
        activations.sort(key=lambda x: x["confidence"], reverse=True)
        top = activations[:5]

        # Update WorldModel with perceived beliefs
        if self._wm is not None and top:
            try:
                for belief in top:
                    self._wm.update_belief(
                        f"perceived_{belief['concept']}",
                        belief["concept"],
                        belief["confidence"],
                        domain=domain,
                        evidence=f"perception:{source}",
                    )
            except Exception as e:
                log.debug("NeuralPerception WorldModel update failed: %s", e)

        self._history.append({
            "text": text[:80],
            "source": source,
            "domain": domain,
            "activations": len(top),
            "top_concept": top[0]["concept"] if top else None,
            "timestamp": time.time(),
        })
        self._total_perceptions += 1

        return top

    def perceive_problem(self, expression: str, domain: str = "general") -> List[str]:
        """
        Perceive a math expression and return suggested concept names.
        Used by curriculum generator to enrich problem metadata.
        """
        results = self.perceive(expression, source="problem", domain=domain)
        return [r["concept"] for r in results]

    def get_stats(self) -> dict:
        return {
            "total_perceptions": self._total_perceptions,
            "vocab_size": len(self._vocab),
            "concept_count": len(self._concept_list),
            "history_size": len(self._history),
        }


_perception_instance: Optional[NeuralPerception] = None

def get_neural_perception(world_model=None) -> NeuralPerception:
    global _perception_instance
    if _perception_instance is None:
        _perception_instance = NeuralPerception(world_model)
    return _perception_instance
