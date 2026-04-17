"""
Cross-Domain Transfer Engine — The hallmark of true intelligence.

When SARE-HX learns x+0=x (additive identity in arithmetic), it should
*automatically* hypothesize p∧True=p (identity in logic) and X∪∅=X (sets).

This engine discovers structural parallels across domains by:
1. Abstracting concrete rules into structural roles (identity, annihilation, involution...)
2. Finding domains that share the same abstract structure
3. Generating transfer hypotheses (concrete rule proposals in new domains)
4. Testing hypotheses via the solve engine
5. Promoting verified transfers as new rules

No hardcoded analogy tables. Everything discovered from experience.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

_PERSIST_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "learned_transfers.json"
_singleton: Optional["TransferEngine"] = None


def get_transfer_engine() -> "TransferEngine":
    global _singleton
    if _singleton is None:
        _singleton = TransferEngine()
        _singleton.load()
    return _singleton


# ═══════════════════════════════════════════════════════════════════════════════
#  Structural Roles — The universal algebra of transforms
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StructuralRole:
    """An abstract role that a transform plays, independent of domain."""
    name: str                    # e.g. "identity", "annihilation", "involution"
    description: str
    exemplars: Dict[str, List[str]] = field(default_factory=dict)  # domain → [transform_names]
    confidence: float = 0.5
    observation_count: int = 0

    def add_exemplar(self, domain: str, transform: str):
        self.exemplars.setdefault(domain, [])
        if transform not in self.exemplars[domain]:
            self.exemplars[domain].append(transform)
        self.observation_count += 1
        self.confidence = min(0.99, 0.3 + 0.05 * self.observation_count)

    @property
    def domains(self) -> Set[str]:
        return set(self.exemplars.keys())

    @property
    def is_cross_domain(self) -> bool:
        return len(self.domains) > 1

    def to_dict(self) -> dict:
        return {
            "name": self.name, "description": self.description,
            "exemplars": self.exemplars, "confidence": round(self.confidence, 3),
            "domains": sorted(self.domains), "cross_domain": self.is_cross_domain,
        }


@dataclass
class TransferHypothesis:
    """A concrete proposal to apply a rule from one domain to another."""
    source_domain: str
    source_transform: str
    source_role: str
    target_domain: str
    proposed_transform: str     # name of the proposed new transform
    proposed_pattern: str       # what it would look like
    confidence: float
    status: str = "untested"    # untested | verified | rejected
    created_at: float = field(default_factory=time.time)
    test_results: List[dict] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.source_domain}:{self.source_transform}→{self.target_domain}:{self.proposed_transform}"

    def to_dict(self) -> dict:
        return {
            "source_domain": self.source_domain,
            "source_transform": self.source_transform,
            "source_role": self.source_role,
            "target_domain": self.target_domain,
            "proposed_transform": self.proposed_transform,
            "proposed_pattern": self.proposed_pattern,
            "confidence": round(self.confidence, 3),
            "status": self.status,
            "test_results": self.test_results[-5:],
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Role Classifier — Maps concrete transforms to abstract roles
# ═══════════════════════════════════════════════════════════════════════════════

# These patterns detect structural roles from transform behavior.
# The RULES are not hardcoded — the CLASSIFIER is (minimal bootstrap).
_ROLE_PATTERNS = {
    "identity": {
        "indicators": ["identity", "zero_elim", "one_elim", "union_identity",
                        "intersect_identity", "and_true", "or_false"],
        "description": "op(x, e) = x where e is the identity element",
        "structural_pattern": "binary_op_with_neutral_element",
    },
    "annihilation": {
        "indicators": ["zero", "mul_zero", "and_false", "or_true",
                        "intersect_empty"],
        "description": "op(x, z) = z where z absorbs everything",
        "structural_pattern": "binary_op_with_absorbing_element",
    },
    "involution": {
        "indicators": ["double_neg", "double_not", "neg_neg", "not_not"],
        "description": "f(f(x)) = x — applying twice cancels",
        "structural_pattern": "double_application_cancels",
    },
    "self_inverse": {
        "indicators": ["self_subtraction", "self_division", "x_minus_x",
                        "x_div_x", "self_cancel"],
        "description": "op(x, x) = identity_element",
        "structural_pattern": "same_operand_cancellation",
    },
    "evaluation": {
        "indicators": ["const_fold", "constant_fold", "evaluate", "compute"],
        "description": "op(c1, c2) = c3 — compute concrete values",
        "structural_pattern": "concrete_computation",
    },
    "distribution": {
        "indicators": ["distribut", "expand"],
        "description": "a*(b+c) = a*b + a*c — one op distributes over another",
        "structural_pattern": "outer_op_distributes_over_inner",
    },
    "combination": {
        "indicators": ["combine", "like_terms", "collect"],
        "description": "a*x + b*x = (a+b)*x — combine similar structures",
        "structural_pattern": "factor_common_substructure",
    },
    "equation_solving": {
        "indicators": ["equation", "solve", "isolate"],
        "description": "Isolate unknown by inverse operations",
        "structural_pattern": "inverse_op_to_isolate",
    },
    "commutativity": {
        "indicators": ["commut", "swap", "canonical"],
        "description": "op(a, b) = op(b, a) — order doesn't matter",
        "structural_pattern": "operand_order_irrelevant",
    },
    "cancellation": {
        "indicators": ["cancel", "additive_cancel"],
        "description": "(x + c) - c = x — inverse ops cancel",
        "structural_pattern": "inverse_operation_cancellation",
    },
    "exponent_rule": {
        "indicators": ["power", "exp", "exponent"],
        "description": "Rules about exponentiation",
        "structural_pattern": "exponent_simplification",
    },
}


class RoleClassifier:
    """Classifies transforms into structural roles."""

    @staticmethod
    def classify(transform_name: str) -> Optional[str]:
        """Map a concrete transform name to its structural role."""
        t_lower = transform_name.lower()
        best_role = None
        best_score = 0

        for role, spec in _ROLE_PATTERNS.items():
            score = 0
            for indicator in spec["indicators"]:
                if indicator in t_lower:
                    score += len(indicator)  # longer matches = more specific
            if score > best_score:
                best_score = score
                best_role = role

        return best_role

    @staticmethod
    def get_role_description(role: str) -> str:
        spec = _ROLE_PATTERNS.get(role, {})
        return spec.get("description", f"Unknown role: {role}")

    @staticmethod
    def get_structural_pattern(role: str) -> str:
        spec = _ROLE_PATTERNS.get(role, {})
        return spec.get("structural_pattern", "unknown")


# ═══════════════════════════════════════════════════════════════════════════════
#  Transfer Engine
# ═══════════════════════════════════════════════════════════════════════════════

class TransferEngine:
    """
    Discovers and applies structural analogies across domains.

    Workflow:
    1. observe() — record each successful solve's transforms + domain
    2. discover_roles() — classify transforms into structural roles
    3. generate_hypotheses() — propose transfers from role-rich to role-poor domains
    4. test_hypothesis() — verify a transfer by trying the analogous operation
    5. promote_transfer() — add verified transfer as a new rule
    """

    def __init__(self):
        self._roles: Dict[str, StructuralRole] = {}
        self._hypotheses: Dict[str, TransferHypothesis] = {}
        self._domain_transforms: Dict[str, Set[str]] = defaultdict(set)
        self._domain_roles: Dict[str, Set[str]] = defaultdict(set)
        self._transfer_history: List[dict] = []
        self._stats = {
            "observations": 0,
            "roles_discovered": 0,
            "hypotheses_generated": 0,
            "hypotheses_verified": 0,
            "hypotheses_rejected": 0,
            "transfers_promoted": 0,
        }

    def observe(self, transforms: List[str], domain: str, success: bool):
        """Record a solve episode for transfer analysis."""
        if not success or not transforms:
            return

        self._stats["observations"] += 1

        for t in transforms:
            self._domain_transforms[domain].add(t)
            role = RoleClassifier.classify(t)
            if role:
                self._domain_roles[domain].add(role)

                # Update structural role
                if role not in self._roles:
                    self._roles[role] = StructuralRole(
                        name=role,
                        description=RoleClassifier.get_role_description(role),
                    )
                    self._stats["roles_discovered"] += 1
                self._roles[role].add_exemplar(domain, t)

    def discover_roles(self) -> Dict[str, StructuralRole]:
        """Return all discovered structural roles."""
        return self._roles

    def generate_hypotheses(self) -> List[TransferHypothesis]:
        """
        Generate transfer hypotheses by finding roles present in one
        domain but missing in another.

        Logic: If 'identity' role exists in arithmetic (via add_zero_elim)
        but NOT in logic, propose: "logic probably has an identity rule too".
        """
        new_hypotheses = []

        for role_name, role in self._roles.items():
            if not role.is_cross_domain and role.observation_count < 1:
                continue  # Need at least one observation

            # For each domain that HAS this role
            for source_domain, source_transforms in role.exemplars.items():
                # Check every OTHER known domain
                for target_domain in self._domain_transforms:
                    if target_domain == source_domain:
                        continue

                    # Does target domain already have this role?
                    target_has_role = role_name in self._domain_roles.get(target_domain, set())
                    if target_has_role:
                        continue  # Already has it

                    # Generate hypothesis: target domain should have this role too
                    source_t = source_transforms[0]  # canonical exemplar
                    proposed_name = f"{target_domain}_{role_name}"
                    proposed_pattern = (
                        f"{role_name} rule in {target_domain} "
                        f"(analogous to {source_t} in {source_domain})"
                    )

                    hyp = TransferHypothesis(
                        source_domain=source_domain,
                        source_transform=source_t,
                        source_role=role_name,
                        target_domain=target_domain,
                        proposed_transform=proposed_name,
                        proposed_pattern=proposed_pattern,
                        confidence=role.confidence * 0.7,
                    )

                    if hyp.key not in self._hypotheses:
                        self._hypotheses[hyp.key] = hyp
                        new_hypotheses.append(hyp)
                        self._stats["hypotheses_generated"] += 1

        return new_hypotheses

    def test_hypothesis(self, hypothesis: TransferHypothesis,
                        solve_fn=None, test_problems: List[str] = None) -> bool:
        """
        Test a transfer hypothesis by attempting to solve problems
        in the target domain using the source strategy.

        Returns True if the transfer appears valid.
        """
        if not solve_fn or not test_problems:
            # Can't test without a solver — mark for later
            return False

        successes = 0
        total = 0

        for problem in test_problems[:5]:
            try:
                result = solve_fn(problem)
                total += 1
                if result.get("success"):
                    # Check if the structural role was actually used
                    transforms_used = result.get("transforms", [])
                    for t in transforms_used:
                        if RoleClassifier.classify(t) == hypothesis.source_role:
                            successes += 1
                            break

                hypothesis.test_results.append({
                    "problem": problem,
                    "success": result.get("success", False),
                    "delta": result.get("delta", 0),
                })
            except Exception as e:
                hypothesis.test_results.append({
                    "problem": problem, "error": str(e)
                })

        # Verdict
        if total >= 3 and successes / total >= 0.5:
            hypothesis.status = "verified"
            hypothesis.confidence = min(0.95, hypothesis.confidence + 0.2)
            self._stats["hypotheses_verified"] += 1
            # Publish event so learn_daemon can register this as a ConceptRule
            try:
                from sare.core.event_bus import get_event_bus
                get_event_bus().publish("transfer_verified", {
                    "name": hypothesis.proposed_transform,
                    "domain": hypothesis.target_domain,
                    "source_domain": hypothesis.source_domain,
                    "confidence": hypothesis.confidence,
                    "pattern": hypothesis.proposed_pattern,
                })
            except Exception:
                pass
            return True
        elif total >= 3:
            hypothesis.status = "rejected"
            hypothesis.confidence *= 0.3
            self._stats["hypotheses_rejected"] += 1
            return False

        return False

    def get_transfer_suggestions(self, domain: str) -> List[dict]:
        """
        Get suggested transfers INTO a domain.
        Returns hypotheses sorted by confidence.
        """
        suggestions = []
        for hyp in self._hypotheses.values():
            if hyp.target_domain == domain and hyp.status != "rejected":
                suggestions.append(hyp.to_dict())
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)
        return suggestions

    def get_cross_domain_map(self) -> dict:
        """
        Return a map of cross-domain structural parallels.
        Used for visualization.
        """
        cross_roles = {
            name: role.to_dict()
            for name, role in self._roles.items()
            if role.is_cross_domain
        }

        domain_coverage = {}
        for domain in self._domain_transforms:
            roles = self._domain_roles.get(domain, set())
            domain_coverage[domain] = {
                "transforms": len(self._domain_transforms[domain]),
                "roles": sorted(roles),
                "role_count": len(roles),
                "missing_roles": sorted(
                    set(self._roles.keys()) - roles
                ),
            }

        return {
            "cross_domain_roles": cross_roles,
            "domain_coverage": domain_coverage,
            "hypotheses": {
                "total": len(self._hypotheses),
                "verified": sum(1 for h in self._hypotheses.values() if h.status == "verified"),
                "untested": sum(1 for h in self._hypotheses.values() if h.status == "untested"),
                "rejected": sum(1 for h in self._hypotheses.values() if h.status == "rejected"),
            },
            "stats": self._stats,
        }

    def save(self) -> None:
        """Persist observations and hypotheses to disk (atomic write)."""
        try:
            _PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "domain_transforms": {d: list(ts) for d, ts in self._domain_transforms.items()},
                "domain_roles": {d: list(rs) for d, rs in self._domain_roles.items()},
                "roles": {
                    name: {
                        "name": r.name,
                        "description": r.description,
                        "observation_count": r.observation_count,
                        "confidence": r.confidence,
                        "exemplars": {d: list(ts) for d, ts in r.exemplars.items()},
                    }
                    for name, r in self._roles.items()
                },
                "hypotheses": {k: h.to_dict() for k, h in self._hypotheses.items()},
                "transfer_history": self._transfer_history[-200:],
                "stats": self._stats,
            }
            tmp = _PERSIST_PATH.with_name(_PERSIST_PATH.name + ".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(tmp, _PERSIST_PATH)
            log.debug("TransferEngine saved: %d domains, %d roles, %d hypotheses",
                      len(self._domain_transforms), len(self._roles), len(self._hypotheses))
        except Exception as e:
            log.debug("TransferEngine save error: %s", e)

    def load(self) -> None:
        """Load persisted transfer state from disk."""
        if not _PERSIST_PATH.exists():
            return
        try:
            data = json.loads(_PERSIST_PATH.read_text())
            for d, ts in data.get("domain_transforms", {}).items():
                self._domain_transforms[d] = set(ts)
            for d, rs in data.get("domain_roles", {}).items():
                self._domain_roles[d] = set(rs)
            for name, rd in data.get("roles", {}).items():
                role = StructuralRole(name=rd["name"], description=rd.get("description", ""))
                role.observation_count = rd.get("observation_count", 0)
                role.confidence = rd.get("confidence", 0.5)
                role.exemplars = defaultdict(list, {d: list(ts) for d, ts in rd.get("exemplars", {}).items()})
                self._roles[name] = role
            for k, hd in data.get("hypotheses", {}).items():
                hyp = TransferHypothesis(
                    source_domain=hd.get("source_domain", ""),
                    source_transform=hd.get("source_transform", ""),
                    source_role=hd.get("source_role", ""),
                    target_domain=hd.get("target_domain", ""),
                    proposed_transform=hd.get("proposed_transform", ""),
                    proposed_pattern=hd.get("proposed_pattern", ""),
                    confidence=hd.get("confidence", 0.5),
                )
                hyp.status = hd.get("status", "untested")
                self._hypotheses[k] = hyp
            self._transfer_history = data.get("transfer_history", [])
            self._stats.update(data.get("stats", {}))
            log.debug("TransferEngine loaded: %d domains, %d roles, %d hypotheses",
                      len(self._domain_transforms), len(self._roles), len(self._hypotheses))
        except Exception as e:
            log.debug("TransferEngine load error: %s", e)

    def summary(self) -> dict:
        return {
            "roles": len(self._roles),
            "cross_domain_roles": sum(1 for r in self._roles.values() if r.is_cross_domain),
            "domains_analyzed": len(self._domain_transforms),
            "hypotheses": len(self._hypotheses),
            "verified": sum(1 for h in self._hypotheses.values() if h.status == "verified"),
            "stats": self._stats,
        }
