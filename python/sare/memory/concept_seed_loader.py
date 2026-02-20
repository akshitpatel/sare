"""
ConceptSeedLoader — Load foundational knowledge rules at startup.

Reads configs/knowledge_seeds.json and populates the ConceptRegistry
with pre-verified, high-confidence rules. This is equivalent to
"innate knowledge" in cognitive science — laws so fundamental they
don't need to be discovered from experience.

Without seeds: system needs 500+ solves before learning anything.
With seeds:    system starts reasoning correctly from episode 1.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

SEEDS_PATH = Path(__file__).resolve().parents[3] / "configs" / "knowledge_seeds.json"


def load_seeds(concept_registry, seeds_path: Optional[Path] = None) -> int:
    """
    Load knowledge seeds into the ConceptRegistry.

    Seeds are loaded as AbstractRule objects with high confidence.
    Uses the Python-level ConceptRegistry wrapper if C++ bindings
    aren't available (graceful degradation).

    Returns: number of seeds loaded
    """
    path = Path(seeds_path or SEEDS_PATH)
    if not path.exists():
        log.warning("Knowledge seeds file not found: %s", path)
        return 0

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log.error("Failed to load knowledge seeds: %s", e)
        return 0

    seeds = data.get("seeds", [])
    loaded = 0

    for seed in seeds:
        try:
            _inject_seed(concept_registry, seed)
            loaded += 1
        except Exception as e:
            log.debug("Seed '%s' injection failed: %s", seed.get("name", "?"), e)

    log.info("Knowledge seeds loaded: %d/%d into ConceptRegistry", loaded, len(seeds))
    return loaded


def _inject_seed(concept_registry, seed: dict):
    """Inject a single seed dict into the ConceptRegistry."""
    # Try C++ path: concept_registry is a sare_bindings.ConceptRegistry instance
    # which expects an AbstractRule object
    try:
        import sare.sare_bindings as sb  # type: ignore
        rule = sb.AbstractRule()
        rule.name        = seed["name"]
        rule.domain      = seed.get("domain", "general")
        rule.confidence  = float(seed.get("confidence", 0.9))
        rule.observations = int(seed.get("observations", 1000))
        # Pattern and replacement are stub graphs — they exist as metadata only.
        # The actual transform logic lives in the C++ transform registry.
        concept_registry.add_rule(rule)
        return
    except Exception:
        pass

    # Fallback: concept_registry is a Python dict-based store
    if hasattr(concept_registry, "_seeds"):
        concept_registry._seeds[seed["name"]] = seed
    elif isinstance(concept_registry, dict):
        concept_registry[seed["name"]] = seed
    else:
        log.debug("Unknown concept_registry type: %s, skipping seed", type(concept_registry))


class SeededConceptRegistry:
    """
    Pure-Python ConceptRegistry that works when C++ bindings are unavailable.
    Provides the same interface so web.py can call it generically.
    """

    def __init__(self):
        self._seeds: dict = {}
        self._learned: list = []

    def add_rule(self, rule):
        """Accept a rule object or dict."""
        name = getattr(rule, "name", None) or rule.get("name", "")
        if name:
            self._learned.append(rule)

    def get_rules(self):
        seed_rules = list(self._seeds.values())
        return seed_rules + self._learned

    def get_consolidated_rules(self, min_confidence: float = 0.8):
        return [
            r for r in self.get_rules()
            if (getattr(r, "confidence", r.get("confidence", 0)) if isinstance(r, dict)
                else getattr(r, "confidence", 0)) >= min_confidence
        ]

    def __len__(self):
        return len(self._seeds) + len(self._learned)
