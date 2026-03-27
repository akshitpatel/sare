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
from typing import Dict, List, Optional, Set

log = logging.getLogger(__name__)

SEEDS_PATH = Path(__file__).resolve().parents[3] / "configs" / "knowledge_seeds.json"
SUPPORTED_DOMAINS = {"arithmetic", "algebra", "logic", "calculus", "geometry", "physics"}


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
            if _validate_seed_domain(seed):
                _inject_seed(concept_registry, seed)
                loaded += 1
            else:
                log.debug("Seed '%s' skipped: unsupported domain '%s'", seed.get("name", "?"), seed.get("domain", "unknown"))
        except Exception as e:
            log.debug("Seed '%s' injection failed: %s", seed.get("name", "?"), e)

    log.info("Knowledge seeds loaded: %d/%d into ConceptRegistry", loaded, len(seeds))
    return loaded


def _validate_seed_domain(seed: dict) -> bool:
    """Validate that a seed's domain is supported by the system."""
    domain = seed.get("domain", "general")
    if domain == "general":
        return True
    return domain in SUPPORTED_DOMAINS


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
        # NOTE: Seeds are informational metadata only — pattern/replacement graphs
        # are intentionally not populated here. The actual transform logic lives
        # in the C++ transform registry (registered by name, not by graph pattern).
        # Seeds bootstrap confidence and observation counts so the system doesn't
        # need to re-discover fundamental rules from experience.
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

    Epic 22: Supports save/load for full concept persistence across reboots.
    """

    PERSIST_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "concept_registry.json"
    SYNTH_CODE_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "hallucinated_rules.py"

    def __init__(self):
        self._seeds: dict = {}
        self._learned: list = []
        self._synthetic_code: List[Dict] = []  # {name, code, problem, timestamp}
        self.load()

    def load(self, path: Optional[Path] = None) -> None:
        """Restore learned rules and synthetic code from disk."""
        import json
        p = Path(path or self.PERSIST_PATH)
        if not p.exists():
            return
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            learned = payload.get("learned", [])
            existing_names = {
                getattr(r, "name", None) or (r.get("name", "") if isinstance(r, dict) else "")
                for r in self._learned
            }
            for r in learned:
                name = r.get("name", "") if isinstance(r, dict) else getattr(r, "name", "")
                if name and name not in existing_names:
                    self._learned.append(r)
                    existing_names.add(name)
            self._synthetic_code = payload.get("synthetic", [])
            log.debug("SeededConceptRegistry loaded %d rules from disk", len(self._learned))
        except Exception as e:
            log.debug("SeededConceptRegistry load failed: %s", e)

    def add_rule(self, rule):
        """Accept a rule object or dict."""
        name = getattr(rule, "name", None) or (rule.get("name", "") if isinstance(rule, dict) else "")
        if name:
            # EWC-lite: check if overwrite is allowed
            try:
                from sare.learning.forgetting_prevention import get_forgetting_prevention
                fp = get_forgetting_prevention()
                new_conf = getattr(rule, "confidence", rule.get("confidence", 0.5) if isinstance(rule, dict) else 0.5)
                if not fp.should_overwrite(name, float(new_conf)):
                    log.debug("ForgettingPrevention: blocked overwrite of consolidated rule '%s'", name)
                    return  # Don't overwrite
            except Exception:
                pass
            self._learned.append(rule)

    def add_synthetic_rule(self, name: str, code: str, problem: str):
        """Persist a newly hallucinated Python transform to memory."""
        import time
        entry = {"name": name, "code": code, "problem": problem, "timestamp": time.time()}
        self._synthetic_code.append(entry)
        log.info(f"Adopted synthetic rule '{name}' into persistent memory.")

    def get_rules(self):
        seed_rules = list(self._seeds.values())
        return seed_rules + self._learned

    def get_all_rules(self):
        """Alias for get_rules() — used by external callers."""
        return self.get_rules()

    def get_synthetic_rules(self) -> List[Dict]:
        return list(self._synthetic_code)

    def get_consolidated_rules(self, min_confidence: float = 0.8):
        return [
            r for r in self.get_rules()
            if (getattr(r, "confidence", r.get("confidence", 0)) if isinstance(r, dict)
                else getattr(r, "confidence", 0)) >= min_confidence
        ]

    def save(self, path: Optional[Path] = None):
        """Persist learned rules and synthetic code to disk."""
        p = Path(path or self.PERSIST_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            payload = {
                "learned": [
                    r if isinstance(r, dict) else {
                        "name": getattr(r, "name", ""),
                        "domain": getattr(r, "domain", "general"),
                        "confidence": getattr(r, "confidence", 0.9),
                        "observations": getattr(r, "observations", 1),
                    }
                    for r in self._learned
                ],
                "synthetic": self._synthetic_code,
            }
            import json
            with open(p, "w") as f:
                json.dump(payload, f, indent=2)

            # Also write synthetic code as importable Python
            sp = self.SYNTH_CODE_PATH
            sp.parent.mkdir(parents=True, exist_ok=True)
            with open(sp, "w") as f:
                f.write("# Synthetic rules generated by SARE-HX\n")
                for entry in self._synthetic_code:
                    f.write(f"\n# Rule: {entry['name']}\n")
                    f.write(entry['code'])
                    f.write("\n")
        except Exception as e:
            log.error("Failed to save concept registry: %s", e)
