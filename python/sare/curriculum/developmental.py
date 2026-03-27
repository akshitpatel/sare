"""
Developmental Curriculum — The "School" for SARE-HX

Structured progression from basic pattern matching to research-level reasoning.
Mirrors how humans learn: counting → arithmetic → algebra → calculus → analysis.

Each domain has:
  - Prerequisites (must master before unlocking)
  - Staged problems (easy → medium → hard → challenge)
  - Target rules to discover
  - Competence threshold to "pass"

The curriculum adapts:
  - Zone of Proximal Development (ZPD): problems just beyond current ability
  - Spaced repetition: revisit mastered domains periodically
  - Failure-driven: stuck domains get extra attention
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "memory"
CURRICULUM_STATE_PATH = DATA_DIR / "developmental_curriculum_state.json"


# ═══════════════════════════════════════════════════════════════════════════════
#  Domain & Problem Definitions
# ═══════════════════════════════════════════════════════════════════════════════

class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    CHALLENGE = "challenge"


@dataclass
class CurriculumProblem:
    """A single problem in the curriculum."""
    expression: str
    domain: str
    difficulty: Difficulty
    target_rules: List[str]    # rules expected to be used/discovered
    hint: str = ""             # optional scaffold hint
    max_steps: int = 10        # expected max steps to solve

    def to_dict(self) -> dict:
        return {
            "expression": self.expression,
            "domain": self.domain,
            "difficulty": self.difficulty.value,
            "target_rules": self.target_rules,
            "hint": self.hint,
            "max_steps": self.max_steps,
        }


@dataclass
class Domain:
    """A knowledge domain with prerequisite dependencies."""
    name: str
    display_name: str
    prerequisites: List[str]       # domain names that must be mastered first
    competence_threshold: float    # 0-1, required to "pass" this domain
    problems: List[CurriculumProblem] = field(default_factory=list)
    unlocked: bool = False
    mastered: bool = False
    current_difficulty: Difficulty = Difficulty.EASY
    attempts: int = 0
    successes: int = 0
    last_attempted: float = 0.0
    last_reviewed: float = 0.0     # for spaced repetition

    @property
    def solve_rate(self) -> float:
        return self.successes / max(self.attempts, 1)

    @property
    def needs_review(self) -> bool:
        """Spaced repetition: review if mastered but not practiced recently."""
        if not self.mastered:
            return False
        hours_since = (time.time() - self.last_reviewed) / 3600
        return hours_since > 2.0  # review every 2 hours

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "prerequisites": self.prerequisites,
            "threshold": self.competence_threshold,
            "unlocked": self.unlocked,
            "mastered": self.mastered,
            "difficulty": self.current_difficulty.value,
            "attempts": self.attempts,
            "successes": self.successes,
            "solve_rate": round(self.solve_rate, 3),
            "problems_count": len(self.problems),
            "needs_review": self.needs_review,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Built-in Curriculum Data
# ═══════════════════════════════════════════════════════════════════════════════

def _build_domains() -> Dict[str, Domain]:
    """Build the full developmental curriculum."""
    domains = {}

    # ── Stage 1: Infant / Toddler ────────────────────────────────────────────

    domains["identity_basics"] = Domain(
        name="identity_basics",
        display_name="Identity Rules",
        prerequisites=[],
        competence_threshold=0.8,
        unlocked=True,  # always available
        problems=[
            CurriculumProblem("x + 0", "identity_basics", Difficulty.EASY,
                              ["add_zero_elim"], "Adding zero changes nothing"),
            CurriculumProblem("0 + x", "identity_basics", Difficulty.EASY,
                              ["add_zero_elim"], "Zero plus anything is that thing"),
            CurriculumProblem("x * 1", "identity_basics", Difficulty.EASY,
                              ["mul_one_elim"], "Multiplying by one changes nothing"),
            CurriculumProblem("1 * x", "identity_basics", Difficulty.EASY,
                              ["mul_one_elim"]),
            CurriculumProblem("y + 0", "identity_basics", Difficulty.MEDIUM,
                              ["add_zero_elim"]),
            CurriculumProblem("a * 1", "identity_basics", Difficulty.MEDIUM,
                              ["mul_one_elim"]),
            CurriculumProblem("(x + 0) * 1", "identity_basics", Difficulty.HARD,
                              ["add_zero_elim", "mul_one_elim"], max_steps=3),
            CurriculumProblem("1 * (0 + y)", "identity_basics", Difficulty.HARD,
                              ["add_zero_elim", "mul_one_elim"], max_steps=3),
        ],
    )

    domains["annihilation"] = Domain(
        name="annihilation",
        display_name="Annihilation Rules",
        prerequisites=["identity_basics"],
        competence_threshold=0.75,
        problems=[
            CurriculumProblem("x * 0", "annihilation", Difficulty.EASY,
                              ["mul_zero_elim"], "Anything times zero is zero"),
            CurriculumProblem("0 * x", "annihilation", Difficulty.EASY,
                              ["mul_zero_elim"]),
            CurriculumProblem("x - x", "annihilation", Difficulty.MEDIUM,
                              ["self_subtraction"], "A thing minus itself is zero"),
            CurriculumProblem("y * 0 + x", "annihilation", Difficulty.HARD,
                              ["mul_zero_elim", "add_zero_elim"], max_steps=3),
            CurriculumProblem("(a - a) * b", "annihilation", Difficulty.HARD,
                              ["self_subtraction", "mul_zero_elim"], max_steps=3),
        ],
    )

    domains["constant_arithmetic"] = Domain(
        name="constant_arithmetic",
        display_name="Constant Folding",
        prerequisites=["identity_basics"],
        competence_threshold=0.8,
        problems=[
            CurriculumProblem("3 + 4", "constant_arithmetic", Difficulty.EASY,
                              ["const_fold"]),
            CurriculumProblem("10 - 3", "constant_arithmetic", Difficulty.EASY,
                              ["const_fold"]),
            CurriculumProblem("5 * 6", "constant_arithmetic", Difficulty.EASY,
                              ["const_fold"]),
            CurriculumProblem("12 / 4", "constant_arithmetic", Difficulty.MEDIUM,
                              ["const_fold"]),
            CurriculumProblem("(3 + 4) * 2", "constant_arithmetic", Difficulty.MEDIUM,
                              ["const_fold"], max_steps=3),
            CurriculumProblem("(2 + 3) * (4 + 5) + 0", "constant_arithmetic", Difficulty.HARD,
                              ["const_fold", "add_zero_elim"], max_steps=5),
        ],
    )

    # ── Stage 2: Child ───────────────────────────────────────────────────────

    domains["negation"] = Domain(
        name="negation",
        display_name="Negation & Double Negation",
        prerequisites=["identity_basics"],
        competence_threshold=0.75,
        problems=[
            CurriculumProblem("neg neg x", "negation", Difficulty.EASY,
                              ["double_neg"], "Two negatives cancel out"),
            CurriculumProblem("not not x", "negation", Difficulty.EASY,
                              ["double_neg"]),
            CurriculumProblem("neg neg y + 0", "negation", Difficulty.MEDIUM,
                              ["double_neg", "add_zero_elim"], max_steps=3),
            CurriculumProblem("neg neg (x + 0)", "negation", Difficulty.HARD,
                              ["double_neg", "add_zero_elim"], max_steps=3),
        ],
    )

    domains["power_rules"] = Domain(
        name="power_rules",
        display_name="Exponent Rules",
        prerequisites=["identity_basics", "annihilation"],
        competence_threshold=0.75,
        problems=[
            CurriculumProblem("x ^ 0", "power_rules", Difficulty.EASY,
                              ["power_zero_elim"], "Anything to the power of zero is 1"),
            CurriculumProblem("x ^ 1", "power_rules", Difficulty.EASY,
                              ["power_one_elim"], "Anything to the power of one is itself"),
            CurriculumProblem("y ^ 0 + x ^ 1", "power_rules", Difficulty.MEDIUM,
                              ["power_zero_elim", "power_one_elim"], max_steps=4),
            CurriculumProblem("(x ^ 0) * (y ^ 1)", "power_rules", Difficulty.HARD,
                              ["power_zero_elim", "power_one_elim", "mul_one_elim"], max_steps=5),
        ],
    )

    # ── Stage 3: Pre-teen ────────────────────────────────────────────────────

    domains["combining_terms"] = Domain(
        name="combining_terms",
        display_name="Combining Like Terms",
        prerequisites=["identity_basics", "constant_arithmetic"],
        competence_threshold=0.7,
        problems=[
            CurriculumProblem("x + x", "combining_terms", Difficulty.EASY,
                              ["combine_like_terms"]),
            CurriculumProblem("2 * x + 3 * x", "combining_terms", Difficulty.MEDIUM,
                              ["combine_like_terms"]),
            CurriculumProblem("x + 2 * x", "combining_terms", Difficulty.MEDIUM,
                              ["combine_like_terms"]),
            CurriculumProblem("3 * y + 4 * y + 0", "combining_terms", Difficulty.HARD,
                              ["combine_like_terms", "add_zero_elim"], max_steps=4),
        ],
    )

    domains["distribution"] = Domain(
        name="distribution",
        display_name="Distributive Property",
        prerequisites=["combining_terms", "constant_arithmetic"],
        competence_threshold=0.65,
        problems=[
            CurriculumProblem("2 * (x + 3)", "distribution", Difficulty.EASY,
                              ["distributive_expand"]),
            CurriculumProblem("a * (b + 0)", "distribution", Difficulty.MEDIUM,
                              ["add_zero_elim"], max_steps=2),
            CurriculumProblem("x * (y + 1)", "distribution", Difficulty.MEDIUM,
                              ["distributive_expand"], max_steps=3),
            CurriculumProblem("2 * (x + 0) + 3", "distribution", Difficulty.HARD,
                              ["add_zero_elim", "distributive_expand"], max_steps=5),
        ],
    )

    domains["factoring"] = Domain(
        name="factoring",
        display_name="Algebraic Factoring",
        prerequisites=["distribution", "combining_terms"],
        competence_threshold=0.6,
        problems=[
            CurriculumProblem("2 * x + 2 * y", "factoring", Difficulty.MEDIUM,
                              ["algebraic_factor"]),
            CurriculumProblem("3 * a + 3 * b", "factoring", Difficulty.MEDIUM,
                              ["algebraic_factor"]),
            CurriculumProblem("x * a + x * b", "factoring", Difficulty.HARD,
                              ["algebraic_factor"]),
        ],
    )

    # ── Stage 4: Teenager ────────────────────────────────────────────────────

    domains["linear_equations"] = Domain(
        name="linear_equations",
        display_name="Solving Linear Equations",
        prerequisites=["combining_terms", "annihilation"],
        competence_threshold=0.7,
        problems=[
            CurriculumProblem("x + 3 = 7", "linear_equations", Difficulty.EASY,
                              ["linear_equation_solve"]),
            CurriculumProblem("x - 5 = 10", "linear_equations", Difficulty.EASY,
                              ["linear_equation_solve"]),
            CurriculumProblem("2 * x = 10", "linear_equations", Difficulty.MEDIUM,
                              ["multiply_equation_solve"]),
            CurriculumProblem("3 * x = 15", "linear_equations", Difficulty.MEDIUM,
                              ["multiply_equation_solve"]),
            CurriculumProblem("x + 2 = 8", "linear_equations", Difficulty.EASY,
                              ["linear_equation_solve"]),
        ],
    )

    domains["logic_basics"] = Domain(
        name="logic_basics",
        display_name="Propositional Logic",
        prerequisites=["identity_basics", "negation"],
        competence_threshold=0.7,
        problems=[
            # These require logic transforms (boolean AND/OR identity rules)
            # Using NL expressions that the parser can handle
            CurriculumProblem("not not x", "logic_basics", Difficulty.EASY,
                              ["double_neg"]),
            CurriculumProblem("not not (not not y)", "logic_basics", Difficulty.MEDIUM,
                              ["double_neg"], max_steps=3),
        ],
    )

    domains["set_theory"] = Domain(
        name="set_theory",
        display_name="Set Operations",
        prerequisites=["identity_basics", "logic_basics"],
        competence_threshold=0.65,
        problems=[
            # Set identity rules (handled by existing transforms)
            CurriculumProblem("A ∪ ∅", "set_theory", Difficulty.EASY,
                              ["set_union_identity"]),
            CurriculumProblem("A ∩ U", "set_theory", Difficulty.EASY,
                              ["set_intersect_identity"]),
            CurriculumProblem("∅ ∪ A", "set_theory", Difficulty.MEDIUM,
                              ["set_union_identity"]),
        ],
    )

    # ── Stage 5+: Undergraduate / Graduate ───────────────────────────────────

    domains["complex_simplification"] = Domain(
        name="complex_simplification",
        display_name="Multi-Step Simplification",
        prerequisites=["distribution", "linear_equations", "negation"],
        competence_threshold=0.6,
        problems=[
            CurriculumProblem("(x + 0) * 1 + (3 + 4)", "complex_simplification",
                              Difficulty.MEDIUM,
                              ["add_zero_elim", "mul_one_elim", "const_fold"], max_steps=5),
            CurriculumProblem("((x + 0) * 1) + ((y * 1) + 0)", "complex_simplification",
                              Difficulty.HARD,
                              ["add_zero_elim", "mul_one_elim"], max_steps=6),
            CurriculumProblem("neg neg (x * 1 + 0) + (y - y)", "complex_simplification",
                              Difficulty.CHALLENGE,
                              ["double_neg", "mul_one_elim", "add_zero_elim", "self_subtraction"],
                              max_steps=8),
            CurriculumProblem("(x + 0) * (1 + 0) + (y * 0)", "complex_simplification",
                              Difficulty.CHALLENGE,
                              ["add_zero_elim", "mul_one_elim", "mul_zero_elim"],
                              max_steps=8),
        ],
    )

    # ── Stage 6+: Advanced Domains ─────────────────────────────────────────────

    domains["calculus"] = Domain(
        name="calculus",
        display_name="Basic Calculus",
        prerequisites=["constant_arithmetic", "power_rules", "combining_terms"],
        competence_threshold=0.6,
        problems=[
            CurriculumProblem("derivative(5)", "calculus", Difficulty.EASY,
                              ["deriv_const_zero"]),
            CurriculumProblem("derivative(x)", "calculus", Difficulty.EASY,
                              ["deriv_linear"]),
            CurriculumProblem("derivative(x^2)", "calculus", Difficulty.MEDIUM,
                              ["deriv_power_rule"], max_steps=3),
            CurriculumProblem("derivative(x^3)", "calculus", Difficulty.MEDIUM,
                              ["deriv_power_rule"], max_steps=3),
            CurriculumProblem("sin(0) + derivative(x)", "calculus", Difficulty.HARD,
                              ["trig_zero", "deriv_linear", "add_zero_elim"], max_steps=4),
            CurriculumProblem("derivative(x^2) + 0", "calculus", Difficulty.HARD,
                              ["deriv_power_rule", "add_zero_elim"], max_steps=4),
        ],
    )

    domains["advanced_calculus"] = Domain(
        name="advanced_calculus",
        display_name="Advanced Calculus",
        prerequisites=["calculus", "distribution", "combining_terms"],
        competence_threshold=0.6,
        problems=[
            # Trig derivatives
            CurriculumProblem("derivative(sin(x))", "advanced_calculus", Difficulty.EASY,
                              ["deriv_sin"]),
            CurriculumProblem("derivative(cos(x))", "advanced_calculus", Difficulty.EASY,
                              ["deriv_cos"]),
            # Exp / Ln derivatives
            CurriculumProblem("derivative(exp(x))", "advanced_calculus", Difficulty.EASY,
                              ["deriv_exp"]),
            CurriculumProblem("derivative(ln(x))", "advanced_calculus", Difficulty.EASY,
                              ["deriv_ln"]),
            # Sum rule
            CurriculumProblem("derivative(x^2 + x)", "advanced_calculus", Difficulty.MEDIUM,
                              ["deriv_sum_rule", "deriv_power_rule", "deriv_linear"], max_steps=4),
            CurriculumProblem("derivative(x^3 + sin(x))", "advanced_calculus", Difficulty.MEDIUM,
                              ["deriv_sum_rule", "deriv_power_rule", "deriv_sin"], max_steps=4),
            # Product rule
            CurriculumProblem("derivative(x * x)", "advanced_calculus", Difficulty.MEDIUM,
                              ["deriv_product_rule", "deriv_linear", "algebraic_factor"], max_steps=5),
            # Chain rules
            CurriculumProblem("derivative(sin(x^2))", "advanced_calculus", Difficulty.HARD,
                              ["chain_rule_sin", "deriv_power_rule"], max_steps=5),
            CurriculumProblem("derivative(cos(x^2))", "advanced_calculus", Difficulty.HARD,
                              ["chain_rule_cos", "deriv_power_rule"], max_steps=5),
            CurriculumProblem("derivative(exp(x^2))", "advanced_calculus", Difficulty.HARD,
                              ["chain_rule_exp", "deriv_power_rule"], max_steps=5),
            # Quotient rule
            CurriculumProblem("derivative(sin(x) / x)", "advanced_calculus", Difficulty.HARD,
                              ["deriv_quotient_rule", "deriv_sin", "deriv_linear"], max_steps=6),
            # Mixed
            CurriculumProblem("derivative(x^2 + sin(x))", "advanced_calculus", Difficulty.HARD,
                              ["deriv_sum_rule", "deriv_power_rule", "deriv_sin"], max_steps=5),
        ],
    )

    domains["integration"] = Domain(
        name="integration",
        display_name="Integration",
        prerequisites=["advanced_calculus"],
        competence_threshold=0.6,
        problems=[
            CurriculumProblem("integral(3)", "integration", Difficulty.EASY,
                              ["integ_constant"]),
            CurriculumProblem("integral(x)", "integration", Difficulty.EASY,
                              ["integ_linear"]),
            CurriculumProblem("integral(x^2)", "integration", Difficulty.MEDIUM,
                              ["integ_power_rule"]),
            CurriculumProblem("integral(x^3)", "integration", Difficulty.MEDIUM,
                              ["integ_power_rule"]),
            CurriculumProblem("integral(x + x^2)", "integration", Difficulty.HARD,
                              ["integ_sum_rule", "integ_linear", "integ_power_rule"], max_steps=4),
            CurriculumProblem("integral(x^2 + x^3)", "integration", Difficulty.HARD,
                              ["integ_sum_rule", "integ_power_rule"], max_steps=4),
        ],
    )

    domains["probability_statistics"] = Domain(
        name="probability_statistics",
        display_name="Probability & Statistics",
        prerequisites=["constant_arithmetic", "combining_terms", "distribution"],
        competence_threshold=0.6,
        problems=[
            CurriculumProblem("P(empty)", "probability_statistics", Difficulty.EASY,
                              ["prob_empty_zero"]),
            CurriculumProblem("P(Omega)", "probability_statistics", Difficulty.EASY,
                              ["prob_universal_one"]),
            CurriculumProblem("P(A) + P(not A)", "probability_statistics", Difficulty.MEDIUM,
                              ["prob_complement_sum"], max_steps=2),
            CurriculumProblem("P(empty) + P(Omega)", "probability_statistics", Difficulty.MEDIUM,
                              ["prob_empty_zero", "prob_universal_one", "const_fold"], max_steps=4),
        ],
    )

    domains["matrix_operations"] = Domain(
        name="matrix_operations",
        display_name="Linear Algebra Basics",
        prerequisites=["constant_arithmetic", "distribution", "combining_terms"],
        competence_threshold=0.6,
        problems=[
            CurriculumProblem("2 * (a + b)", "matrix_operations", Difficulty.EASY,
                              ["distributive_expand"]),
            CurriculumProblem("a * 0 + b", "matrix_operations", Difficulty.EASY,
                              ["mul_zero_elim", "add_zero_elim"]),
            CurriculumProblem("3 * a + 3 * b", "matrix_operations", Difficulty.MEDIUM,
                              ["algebraic_factor"]),
            CurriculumProblem("2 * (a + b) + 0", "matrix_operations", Difficulty.MEDIUM,
                              ["distributive_expand", "add_zero_elim"], max_steps=4),
            CurriculumProblem("c * a + c * b + 0", "matrix_operations", Difficulty.HARD,
                              ["algebraic_factor", "add_zero_elim"], max_steps=4),
        ],
    )

    domains["cancellation_patterns"] = Domain(
        name="cancellation_patterns",
        display_name="Additive & Multiplicative Cancellation",
        prerequisites=["linear_equations", "combining_terms"],
        competence_threshold=0.65,
        problems=[
            CurriculumProblem("(x + 5) - 5", "cancellation_patterns", Difficulty.MEDIUM,
                              ["additive_cancellation"]),
            CurriculumProblem("(x - 3) + 3", "cancellation_patterns", Difficulty.MEDIUM,
                              ["additive_cancellation"]),
        ],
    )

    return domains


# ═══════════════════════════════════════════════════════════════════════════════
#  Prerequisite Graph (for visualization)
# ═══════════════════════════════════════════════════════════════════════════════

PREREQUISITE_GRAPH = {
    "identity_basics": [],
    "annihilation": ["identity_basics"],
    "constant_arithmetic": ["identity_basics"],
    "negation": ["identity_basics"],
    "power_rules": ["identity_basics", "annihilation"],
    "combining_terms": ["identity_basics", "constant_arithmetic"],
    "distribution": ["combining_terms", "constant_arithmetic"],
    "factoring": ["distribution", "combining_terms"],
    "linear_equations": ["combining_terms", "annihilation"],
    "logic_basics": ["identity_basics", "negation"],
    "set_theory": ["identity_basics", "logic_basics"],
    "complex_simplification": ["distribution", "linear_equations", "negation"],
    "cancellation_patterns": ["linear_equations", "combining_terms"],
    "calculus": ["constant_arithmetic", "power_rules", "combining_terms"],
    "advanced_calculus": ["calculus", "distribution", "combining_terms"],
    "probability_statistics": ["constant_arithmetic", "combining_terms", "distribution"],
    "matrix_operations": ["constant_arithmetic", "distribution", "combining_terms"],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Piaget Milestone Tracking
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PiagetMilestone:
    """
    Maps a Piaget-style cognitive stage to required domain mastery.
    When all required_domains are mastered, the milestone is reached
    and brain._check_and_advance_stage() is triggered.
    """
    stage_name:                 str          # maps to DevelopmentalStage enum value
    required_domains:           List[str]    # domains that must be mastered
    cognitive_abilities_unlocked: List[str]  # human-readable list of new capabilities

    def to_dict(self) -> dict:
        return {
            "stage_name": self.stage_name,
            "required_domains": self.required_domains,
            "cognitive_abilities_unlocked": self.cognitive_abilities_unlocked,
        }


PIAGET_MILESTONES: List[PiagetMilestone] = [
    PiagetMilestone(
        stage_name="toddler",
        required_domains=["identity_basics"],
        cognitive_abilities_unlocked=["additive/multiplicative identity", "basic arithmetic"],
    ),
    PiagetMilestone(
        stage_name="child",
        required_domains=["identity_basics", "annihilation", "constant_arithmetic"],
        cognitive_abilities_unlocked=["annihilation rules", "constant folding", "multi-step simplification"],
    ),
    PiagetMilestone(
        stage_name="preteen",
        required_domains=["combining_terms", "negation", "distribution"],
        cognitive_abilities_unlocked=["algebraic manipulation", "distribution", "combining like terms"],
    ),
    PiagetMilestone(
        stage_name="teenager",
        required_domains=["linear_equations", "factoring", "logic_basics"],
        cognitive_abilities_unlocked=["equation solving", "factoring", "logic", "cross-domain transfer"],
    ),
    PiagetMilestone(
        stage_name="undergrad",
        required_domains=["complex_simplification", "set_theory", "calculus"],
        cognitive_abilities_unlocked=["complex expressions", "set theory", "calculus", "formal proofs"],
    ),
    PiagetMilestone(
        stage_name="graduate",
        required_domains=["advanced_calculus", "probability_statistics", "matrix_operations"],
        cognitive_abilities_unlocked=["advanced calculus", "statistics", "linear algebra", "novel conjectures"],
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Developmental Curriculum Engine
# ═══════════════════════════════════════════════════════════════════════════════

class DevelopmentalCurriculum:
    """
    Manages the structured learning progression.

    Selects problems based on:
    1. Current developmental stage
    2. Domain unlock status (prerequisites met)
    3. Zone of Proximal Development (problems just beyond current ability)
    4. Spaced repetition (review mastered domains)
    """

    def __init__(self):
        self.domains: Dict[str, Domain] = _build_domains()
        self._problem_history: List[dict] = []  # last 500 attempts
        self._achieved_milestones: List[str] = []
        self._update_unlocks()

    def check_piaget_progress(self, brain=None) -> Optional[PiagetMilestone]:
        """
        Check if any Piaget milestone has been reached since last check.
        If so, calls brain._check_and_advance_stage() to trigger stage gating.

        Returns the newly reached milestone, or None.
        """
        mastered_domains = {
            name for name, domain in self.domains.items() if domain.mastered
        }
        for milestone in PIAGET_MILESTONES:
            if milestone.stage_name in self._achieved_milestones:
                continue
            required = set(milestone.required_domains)
            if required.issubset(mastered_domains):
                self._achieved_milestones.append(milestone.stage_name)
                log.info(
                    "[Piaget] Milestone reached: %s (domains: %s)",
                    milestone.stage_name,
                    ", ".join(milestone.required_domains),
                )
                if brain is not None:
                    try:
                        brain._check_and_advance_stage()
                    except Exception as e:
                        log.debug("[Piaget] brain._check_and_advance_stage failed: %s", e)
                return milestone
        return None

    def get_piaget_status(self) -> dict:
        """Return current Piaget milestone progress."""
        mastered = {name for name, d in self.domains.items() if d.mastered}
        milestones_status = []
        for m in PIAGET_MILESTONES:
            achieved = m.stage_name in self._achieved_milestones
            missing = [d for d in m.required_domains if d not in mastered]
            milestones_status.append({
                "stage": m.stage_name,
                "achieved": achieved,
                "required_domains": m.required_domains,
                "missing_domains": missing,
                "abilities": m.cognitive_abilities_unlocked,
            })
        return {
            "achieved_milestones": list(self._achieved_milestones),
            "milestones": milestones_status,
            "mastered_domains": sorted(mastered),
        }

    def next_problem_with_forgetting(self, stage=None, self_model=None) -> Optional[str]:
        """
        Extended next_problem that checks ForgettingCurve due reviews FIRST
        before selecting from ZPD. At-risk memories take priority.
        """
        # Check forgetting curve for due reviews
        try:
            from sare.memory.forgetting_curve import get_forgetting_curve
            fc = get_forgetting_curve()
            due = fc.get_due_reviews(limit=5)
            if due:
                # Return the weakest-strength item that is also a valid curriculum expression
                for item in due:
                    # item_id may be a problem_id or expression
                    expr = item.item_id
                    # Quick check: does this expression exist in any domain?
                    for domain in self.domains.values():
                        for prob in domain.problems:
                            if prob.expression == expr:
                                log.debug("[Curriculum] Forgetting review: '%s'", expr)
                                return expr
        except Exception:
            pass
        return self.next_problem(stage=stage, self_model=self_model)

    def _update_unlocks(self):
        """Unlock domains whose prerequisites are all mastered."""
        changed = True
        while changed:
            changed = False
            for name, domain in self.domains.items():
                if domain.unlocked:
                    continue
                prereqs_met = all(
                    self.domains.get(p, Domain(name="", display_name="", prerequisites=[], competence_threshold=1.0)).mastered
                    for p in domain.prerequisites
                )
                if prereqs_met or not domain.prerequisites:
                    domain.unlocked = True
                    changed = True
                    log.info(f"Domain unlocked: {domain.display_name}")

    def next_problem(self, stage=None, self_model=None) -> Optional[str]:
        """
        Pick the best next problem to learn from.

        Priority:
        1. Spaced repetition reviews (mastered domains needing refresh)
        2. ZPD problems (unlocked, not mastered, at appropriate difficulty)
        3. Random from available pool
        """
        self._update_unlocks()

        # 1. Spaced repetition: review mastered domains
        review_domains = [d for d in self.domains.values()
                          if d.needs_review and d.problems]
        if review_domains and random.random() < 0.2:  # 20% chance of review
            domain = random.choice(review_domains)
            domain.last_reviewed = time.time()
            problem = random.choice(domain.problems)
            return problem.expression

        # 2. ZPD: unlocked but not mastered domains
        learning_domains = [
            d for d in self.domains.values()
            if d.unlocked and not d.mastered and d.problems
        ]

        if not learning_domains:
            # Everything mastered or no problems — return None
            return None

        # Weight by exploration priority (from self_model if available)
        weights = []
        for d in learning_domains:
            if self_model and hasattr(self_model, 'get_domain_competence'):
                try:
                    comp = self_model.get_domain_competence(d.name)
                    if comp:
                        weights.append(comp.exploration_weight)
                        continue
                except Exception:
                    pass
            # Default weight based on solve rate
            # Special case: never tried domains get highest priority
            if d.attempts == 0:
                weights.append(1.5)   # untried, highest priority
                continue
            rate = d.solve_rate
            if rate < 0.1:
                weights.append(0.3)   # too hard, low priority
            elif rate < 0.3:
                weights.append(0.6)   # hard but reachable
            elif rate < 0.7:
                weights.append(1.0)   # optimal ZPD
            elif rate < 0.9:
                weights.append(0.5)   # getting easy
            else:
                weights.append(0.1)   # nearly mastered

        # Weighted random selection
        total_weight = sum(weights)
        if total_weight == 0:
            domain = random.choice(learning_domains)
        else:
            r = random.uniform(0, total_weight)
            cumulative = 0
            domain = learning_domains[-1]
            for d, w in zip(learning_domains, weights):
                cumulative += w
                if r <= cumulative:
                    domain = d
                    break

        # Pick problem at current difficulty level
        difficulty_order = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD, Difficulty.CHALLENGE]
        target_diff = domain.current_difficulty
        target_idx = difficulty_order.index(target_diff)

        # Get problems at current difficulty or one level below
        candidates = [p for p in domain.problems if p.difficulty == target_diff]
        if not candidates and target_idx > 0:
            prev_diff = difficulty_order[target_idx - 1]
            candidates = [p for p in domain.problems if p.difficulty == prev_diff]
        if not candidates:
            candidates = domain.problems

        problem = random.choice(candidates)
        return problem.expression

    def record_attempt(self, expression: str, domain_name: str, success: bool, delta: float):
        """Record a solve attempt and update domain state."""
        domain = self.domains.get(domain_name)
        if not domain:
            # Try to find by expression
            for d in self.domains.values():
                for p in d.problems:
                    if p.expression == expression:
                        domain = d
                        break
                if domain:
                    break

        if not domain:
            return

        domain.attempts += 1
        if success:
            domain.successes += 1
        domain.last_attempted = time.time()

        self._problem_history.append({
            "expression": expression,
            "domain": domain.name,
            "success": success,
            "delta": delta,
            "timestamp": time.time(),
        })
        if len(self._problem_history) > 500:
            self._problem_history = self._problem_history[-500:]

        # Update difficulty progression
        if domain.solve_rate > 0.8 and domain.attempts >= 3:
            difficulty_order = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD, Difficulty.CHALLENGE]
            curr_idx = difficulty_order.index(domain.current_difficulty)
            if curr_idx < len(difficulty_order) - 1:
                domain.current_difficulty = difficulty_order[curr_idx + 1]
                log.info(f"Domain {domain.display_name}: difficulty → {domain.current_difficulty.value}")

        # Check mastery
        if (domain.solve_rate >= domain.competence_threshold and
                domain.attempts >= 5):
            if not domain.mastered:
                domain.mastered = True
                domain.last_reviewed = time.time()
                log.info(f"🎓 Domain MASTERED: {domain.display_name} "
                         f"(rate={domain.solve_rate:.0%}, threshold={domain.competence_threshold:.0%})")
                self._update_unlocks()

    def get_curriculum_map(self) -> dict:
        """Return the full curriculum state for visualization."""
        return {
            "domains": {name: d.to_dict() for name, d in self.domains.items()},
            "prerequisite_graph": PREREQUISITE_GRAPH,
            "total_domains": len(self.domains),
            "unlocked": sum(1 for d in self.domains.values() if d.unlocked),
            "mastered": sum(1 for d in self.domains.values() if d.mastered),
            "total_problems": sum(len(d.problems) for d in self.domains.values()),
            "recent_history": self._problem_history[-20:],
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self):
        """Save curriculum state."""
        CURRICULUM_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "domains": {},
            "history": self._problem_history[-200:],
            "saved_at": time.time(),
        }
        for name, d in self.domains.items():
            state["domains"][name] = {
                "unlocked": d.unlocked,
                "mastered": d.mastered,
                "difficulty": d.current_difficulty.value,
                "attempts": d.attempts,
                "successes": d.successes,
                "last_attempted": d.last_attempted,
                "last_reviewed": d.last_reviewed,
            }
        with open(CURRICULUM_STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)

    def load(self):
        """Load curriculum state."""
        if not CURRICULUM_STATE_PATH.exists():
            return
        try:
            with open(CURRICULUM_STATE_PATH) as f:
                state = json.load(f)
            for name, dstate in state.get("domains", {}).items():
                if name in self.domains:
                    d = self.domains[name]
                    d.unlocked = dstate.get("unlocked", d.unlocked)
                    d.mastered = dstate.get("mastered", d.mastered)
                    d.current_difficulty = Difficulty(dstate.get("difficulty", "easy"))
                    d.attempts = dstate.get("attempts", 0)
                    d.successes = dstate.get("successes", 0)
                    d.last_attempted = dstate.get("last_attempted", 0)
                    d.last_reviewed = dstate.get("last_reviewed", 0)
            self._problem_history = state.get("history", [])
            self._update_unlocks()
            log.info(f"Curriculum loaded: {sum(1 for d in self.domains.values() if d.mastered)} mastered")
        except Exception as e:
            log.warning(f"Curriculum load failed: {e}")
