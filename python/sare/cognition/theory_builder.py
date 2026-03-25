"""
TheoryBuilder — aggregates world model hypotheses into coherent mini-theories.

Reads data/memory/world_hypotheses.json (written by world_model.py when surprise > 2.0),
groups by domain, and builds theory objects with evidence counts and summaries.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

_SINGLETON: Optional["TheoryBuilder"] = None

# Resolve the hypotheses file relative to the project root (4 levels up from this file).
_HYPOTHESES_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "world_hypotheses.json"


class Theory:
    """A domain-level theory assembled from multiple hypotheses."""

    def __init__(self, domain: str, hypotheses: list):
        self.domain = domain
        self.hypotheses = hypotheses
        self.hypothesis_count = len(hypotheses)
        texts = [
            h.get("text") or h.get("hypothesis") or h.get("summary") or str(h)
            for h in hypotheses[:3]
        ]
        self.summary = "; ".join(t[:80] for t in texts if t)

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "hypothesis_count": self.hypothesis_count,
            "summary": self.summary,
            "supporting_hypotheses": self.hypotheses[:5],  # cap at 5 for response size
        }


class TheoryBuilder:
    """Builds theories from accumulated hypotheses."""

    def build_theories(self, max_theories: int = 5) -> List[dict]:
        """
        Read world_hypotheses.json, cluster by domain, return top-N theories.
        Returns [] gracefully if the file doesn't exist yet.
        """
        hypotheses = self._load_hypotheses()
        if not hypotheses:
            return []

        # Group by domain
        by_domain: Dict[str, list] = {}
        for h in hypotheses:
            domain = (h.get("domain") or h.get("category") or "general").lower()
            by_domain.setdefault(domain, []).append(h)

        # Sort by hypothesis count (richest domains first)
        sorted_domains = sorted(
            by_domain.items(), key=lambda x: len(x[1]), reverse=True
        )

        theories = []
        for domain, hyps in sorted_domains[:max_theories]:
            theory = Theory(domain=domain, hypotheses=hyps)
            theories.append(theory.to_dict())

        return theories

    def get_theory_for_domain(self, domain: str) -> dict:
        """Get theory for a specific domain."""
        hypotheses = self._load_hypotheses()
        domain_hyps = [
            h
            for h in hypotheses
            if (h.get("domain") or h.get("category") or "general").lower()
            == domain.lower()
        ]
        if not domain_hyps:
            return {
                "domain": domain,
                "hypothesis_count": 0,
                "summary": "No hypotheses yet",
                "supporting_hypotheses": [],
            }
        return Theory(domain=domain, hypotheses=domain_hyps).to_dict()

    def _load_hypotheses(self) -> list:
        """Load hypotheses from disk. Returns [] on any error."""
        if not _HYPOTHESES_PATH.exists():
            return []
        try:
            raw = json.loads(_HYPOTHESES_PATH.read_text())
            # The file may be a list or a dict with a "hypotheses" key
            if isinstance(raw, list):
                return raw
            if isinstance(raw, dict):
                return raw.get("hypotheses") or list(raw.values()) or []
            return []
        except Exception:
            return []


def get_theory_builder() -> TheoryBuilder:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = TheoryBuilder()
    return _SINGLETON
