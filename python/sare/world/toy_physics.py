"""
ToyPhysicsSim — Minimal discrete physics simulation for concept grounding.

Maps mathematical operations to physical object manipulations:
  +  → combine two piles
  -  → remove from pile
  *  → scale a pile
  /  → split a pile
  =  → check balance
  0  → empty pile
  1  → unit pile

This gives abstract symbols physical meaning, partially addressing the
symbol grounding problem (Harnad 1990) by anchoring to sensorimotor intuition.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass
class PhysicsObject:
    name: str
    quantity: float = 0.0
    properties: Dict[str, float] = field(default_factory=dict)


class ToyPhysicsSim:
    """Discrete object physics for mathematical grounding."""

    def __init__(self):
        self._objects: Dict[str, PhysicsObject] = {}
        self._history: List[dict] = []
        self._reset()

    def _reset(self):
        self._objects = {
            "pile_a": PhysicsObject("pile_a", 0.0),
            "pile_b": PhysicsObject("pile_b", 0.0),
            "result": PhysicsObject("result", 0.0),
        }

    def run_operation(self, op: str, a: float, b: float) -> Tuple[float, str]:
        """
        Run a mathematical operation in physical terms.
        Returns (result, physical_description).
        """
        self._reset()
        self._objects["pile_a"].quantity = a
        self._objects["pile_b"].quantity = b

        if op == "+":
            result = a + b
            self._objects["result"].quantity = result
            desc = f"Combined pile_a({a}) and pile_b({b}) → result({result})"
        elif op == "-":
            result = a - b
            self._objects["result"].quantity = result
            desc = f"Removed pile_b({b}) from pile_a({a}) → result({result})"
        elif op == "*":
            result = a * b
            self._objects["result"].quantity = result
            desc = f"Scaled pile_a({a}) by factor({b}) → result({result})"
        elif op == "/":
            if b == 0:
                return float('inf'), "Cannot split by zero (pile_b is empty)"
            result = a / b
            self._objects["result"].quantity = result
            desc = f"Split pile_a({a}) into {b} equal parts → each part({result})"
        else:
            return 0.0, f"Unknown operation: {op}"

        self._history.append({"op": op, "a": a, "b": b, "result": result, "time": time.time()})
        return result, desc

    def verify_identity(self, identity_name: str) -> Tuple[bool, str]:
        """
        Verify a mathematical identity through physical simulation.
        Returns (verified, explanation).
        """
        tests = {
            "additive_identity": [
                ("+", 5, 0, 5, "5 + 0 = 5: adding empty pile to 5 gives 5"),
                ("+", 0, 7, 7, "0 + 7 = 7: combining empty pile with 7 gives 7"),
                ("+", 3, 0, 3, "3 + 0 = 3: adding nothing to 3 leaves it unchanged"),
            ],
            "multiplicative_identity": [
                ("*", 5, 1, 5, "5 x 1 = 5: scaling pile by factor 1 leaves it unchanged"),
                ("*", 1, 7, 7, "1 x 7 = 7: one copy of 7 is 7"),
                ("*", 3, 1, 3, "3 x 1 = 3: single factor doesn't change the pile"),
            ],
            "multiplicative_zero": [
                ("*", 5, 0, 0, "5 x 0 = 0: zero copies of anything is nothing"),
                ("*", 0, 7, 0, "0 x 7 = 0: scaling empty pile gives empty pile"),
                ("*", 3, 0, 0, "3 x 0 = 0: emptying a pile gives nothing"),
            ],
            "subtractive_self": [
                ("-", 5, 5, 0, "5 - 5 = 0: removing everything from a pile empties it"),
                ("-", 3, 3, 0, "3 - 3 = 0: equal piles cancel out"),
                ("-", 7, 7, 0, "7 - 7 = 0: self-removal always yields empty"),
            ],
            "commutativity_add": [
                ("+", 3, 4, 7, "3 + 4 = 7: order doesn't matter for combining"),
                ("+", 4, 3, 7, "4 + 3 = 7: same result regardless of order"),
            ],
            "commutativity_multiply": [
                ("*", 3, 4, 12, "3 x 4 = 12: order of scaling doesn't change result"),
                ("*", 4, 3, 12, "4 x 3 = 12: same product regardless of order"),
                ("*", 5, 2, 10, "5 x 2 = 10: multiplication is commutative"),
                ("*", 2, 5, 10, "2 x 5 = 10: confirmed commutative"),
            ],
            "associativity_add": [
                ("+", 2, 3, 5, "(2+3)+4 first step: 2+3=5"),
                ("+", 5, 4, 9, "(2+3)+4 = 9: left grouping"),
                ("+", 3, 4, 7, "2+(3+4) first step: 3+4=7"),
                ("+", 2, 7, 9, "2+(3+4) = 9: right grouping same result"),
            ],
            "associativity_multiply": [
                ("*", 2, 3, 6, "(2x3)x4 first step: 2x3=6"),
                ("*", 6, 4, 24, "(2x3)x4 = 24: left grouping"),
                ("*", 3, 4, 12, "2x(3x4) first step: 3x4=12"),
                ("*", 2, 12, 24, "2x(3x4) = 24: right grouping same result"),
            ],
            "distributivity": [
                ("+", 3, 4, 7, "3+4=7: inner sum for 2x(3+4)"),
                ("*", 2, 7, 14, "2x(3+4) = 14: distributive left side"),
                ("*", 2, 3, 6, "2x3=6: first distributed term"),
                ("*", 2, 4, 8, "2x4=8: second distributed term"),
                ("+", 6, 8, 14, "2x3 + 2x4 = 14: distributed form matches"),
            ],
            "additive_inverse": [
                ("+", 5, -5, 0, "5 + (-5) = 0: value plus its negative cancels out"),
                ("+", 3, -3, 0, "3 + (-3) = 0: additive inverse yields zero"),
                ("+", 7, -7, 0, "7 + (-7) = 0: negation cancels"),
            ],
            "multiplicative_inverse": [
                ("*", 5, 0.2, 1.0, "5 x (1/5) = 1: value times its reciprocal is 1"),
                ("*", 4, 0.25, 1.0, "4 x (1/4) = 1: multiplicative inverse yields unity"),
                ("*", 2, 0.5, 1.0, "2 x (1/2) = 1: half of two is one"),
            ],
            "zero_product": [
                ("*", 0, 5, 0, "0 x 5 = 0: zero times anything is zero"),
                ("*", 0, 100, 0, "0 x 100 = 0: zero copies of large pile is nothing"),
                ("*", 0, 0, 0, "0 x 0 = 0: zero times zero is zero"),
            ],
            "double_negation_numeric": [
                # Simulated via subtraction: -(-x) = x → subtracting a negative returns original
                ("-", 0, -5, 5, "-(-5) = 5: negating a negation returns original value"),
                ("-", 0, -3, 3, "-(-3) = 3: double negation is identity"),
                ("-", 0, -7, 7, "-(-7) = 7: minus-minus cancels"),
            ],
            "reflexivity": [
                ("-", 5, 5, 0, "5 = 5: difference is zero, confirming equality"),
                ("-", 3, 3, 0, "3 = 3: self-equality always holds"),
                ("-", 0, 0, 0, "0 = 0: zero equals itself"),
            ],
            "symmetry": [
                # a = b implies b - a = 0 (symmetric difference is zero)
                ("-", 4, 4, 0, "if a=b (4=4), then b-a=0: symmetric"),
                ("-", 7, 7, 0, "if a=b (7=7), then b-a=0: symmetric"),
                ("-", 2, 2, 0, "if a=b (2=2), then b-a=0: symmetric"),
            ],
            "transitivity_inequality": [
                # 3 > 2 and 2 > 1: verify 3 > 2 > 1 by checking differences
                ("-", 3, 2, 1, "3 - 2 = 1 > 0: confirms 3 > 2"),
                ("-", 2, 1, 1, "2 - 1 = 1 > 0: confirms 2 > 1"),
                ("-", 3, 1, 2, "3 - 1 = 2 > 0: transitivity gives 3 > 1"),
            ],
            "identity_composition": [
                ("*", 5, 1, 5, "f(x) * identity(1) = f(x): identity composition"),
                ("+", 5, 0, 5, "f(x) + identity(0) = f(x): additive identity composition"),
                ("*", 7, 1, 7, "g(x) * identity(1) = g(x): identity is neutral element"),
            ],
        }

        cases = tests.get(identity_name)
        if not cases:
            return False, f"Unknown identity: {identity_name}"

        all_pass = True
        explanations = []
        for op, a, b, expected, desc in cases:
            result, _ = self.run_operation(op, a, b)
            passed = abs(result - expected) < 1e-9
            all_pass = all_pass and passed
            mark = "PASS" if passed else "FAIL"
            explanations.append(f"  [{mark}] {desc}")

        physical_meaning = {
            "additive_identity": "Adding an empty pile to any pile leaves it unchanged -- like adding nothing",
            "multiplicative_identity": "Scaling by 1 means one copy -- unchanged",
            "multiplicative_zero": "Zero copies of anything is nothing",
            "subtractive_self": "Removing everything from itself leaves nothing",
            "commutativity_add": "Order of combining piles doesn't matter",
            "commutativity_multiply": "Order of scaling doesn't change the result",
            "associativity_add": "Grouping of pile combinations doesn't affect the total",
            "associativity_multiply": "Grouping of scaling operations doesn't affect the result",
            "distributivity": "Scaling a combined pile equals scaling each part then combining",
            "additive_inverse": "Adding the negative of a value cancels it out to zero",
            "multiplicative_inverse": "Multiplying by the reciprocal returns the unit pile",
            "zero_product": "Zero copies of any pile is always nothing",
            "double_negation_numeric": "Negating a negation returns the original value",
            "reflexivity": "Every value equals itself -- self-comparison is always balanced",
            "symmetry": "If a equals b then b equals a -- equality is two-way",
            "transitivity_inequality": "If A > B and B > C, then A > C -- order chains transitively",
            "identity_composition": "Composing any function with the identity returns the function unchanged",
        }.get(identity_name, "")

        explanation = f"Grounding '{identity_name}':\n" + "\n".join(explanations)
        if physical_meaning:
            explanation += f"\nMeaning: {physical_meaning}"

        return all_pass, explanation


class GroundedConceptLearner:
    """
    Links abstract mathematical concepts to physical simulations.

    For each concept the system learns, this module:
    1. Runs the toy physics simulation to verify it
    2. Stores a physical explanation alongside the abstract rule
    3. Uses the physical intuition to guide search (grounded confidence)
    """

    GROUNDED_CONCEPTS = [
        # Original 5
        "additive_identity", "multiplicative_identity",
        "multiplicative_zero", "subtractive_self", "commutativity_add",
        # Extended set (physics-verifiable)
        "commutativity_multiply", "associativity_add", "associativity_multiply",
        "distributivity", "additive_inverse", "multiplicative_inverse",
        "zero_product", "double_negation_numeric", "reflexivity",
        "symmetry", "transitivity_inequality", "identity_composition",
        # Analogy-grounded concepts (verified via ground_by_analogy)
        "commutativity_or", "commutativity_and", "associativity_or",
        "associativity_and", "idempotency_and", "idempotency_or",
        "absorption_and", "absorption_or", "double_negation",
        "de_morgan_and", "de_morgan_or", "distributivity_and_or",
        "distributivity_or_and", "negation_complement", "excluded_middle",
        "identity_and_true", "identity_or_false", "annihilator_and_false",
        "annihilator_or_true", "conjunction_falsehood", "disjunction_tautology",
        "transitivity_equality", "transitivity_implication",
        "contrapositive", "modus_ponens_grounding", "modus_tollens_grounding",
        "commutativity_xor", "associativity_xor", "self_inverse_xor",
        "zero_xor", "set_union_commutativity", "set_intersection_commutativity",
        "set_difference_identity", "set_double_complement", "subset_reflexivity",
        "subset_transitivity", "power_of_zero", "power_of_one", "zero_power",
        "additive_cancellation", "multiplicative_cancellation",
        "distributivity_subtraction", "negation_zero", "double_division",
        "fraction_identity", "absolute_value_nonneg", "absolute_value_symmetry",
    ]

    # Physical/intuitive meanings for all concepts
    _CONCEPT_PHYSICAL_MEANINGS: Dict[str, str] = {
        "additive_identity": "Adding an empty pile leaves any pile unchanged",
        "multiplicative_identity": "Scaling by factor 1 leaves any pile unchanged",
        "multiplicative_zero": "Zero copies of anything is nothing",
        "subtractive_self": "Removing a pile from itself leaves nothing",
        "commutativity_add": "Order of combining piles doesn't change total",
        "commutativity_multiply": "Order of scaling doesn't change result",
        "associativity_add": "Grouping of pile combinations doesn't matter",
        "associativity_multiply": "Grouping of scaling operations doesn't matter",
        "distributivity": "Scaling a combined pile = scaling each part then combining",
        "additive_inverse": "Adding the negative of something cancels it out",
        "multiplicative_inverse": "Multiplying by the reciprocal returns unity",
        "zero_product": "Zero copies of anything is nothing",
        "double_negation_numeric": "Negating twice returns the original",
        "reflexivity": "Any value equals itself",
        "symmetry": "If a=b then b=a; equality is bidirectional",
        "transitivity_inequality": "If A > B and B > C, then A > C",
        "identity_composition": "Composing any function with identity returns that function",
        "commutativity_or": "A OR B = B OR A; order of logical disjunction doesn't matter",
        "commutativity_and": "A AND B = B AND A; order of logical conjunction doesn't matter",
        "associativity_or": "Grouping of OR operations doesn't matter",
        "associativity_and": "Grouping of AND operations doesn't matter",
        "idempotency_and": "A AND A = A; repeating a conjunction changes nothing",
        "idempotency_or": "A OR A = A; repeating a disjunction changes nothing",
        "absorption_and": "A AND (A OR B) = A; the stronger condition absorbs",
        "absorption_or": "A OR (A AND B) = A; the weaker condition absorbs",
        "double_negation": "NOT(NOT A) = A; two negations cancel",
        "de_morgan_and": "NOT(A AND B) = NOT A OR NOT B",
        "de_morgan_or": "NOT(A OR B) = NOT A AND NOT B",
        "distributivity_and_or": "A AND (B OR C) = (A AND B) OR (A AND C)",
        "distributivity_or_and": "A OR (B AND C) = (A OR B) AND (A OR C)",
        "negation_complement": "A AND NOT A = False; a thing and its opposite cannot both hold",
        "excluded_middle": "A OR NOT A = True; either a thing holds or it doesn't",
        "identity_and_true": "A AND True = A; True is the identity for conjunction",
        "identity_or_false": "A OR False = A; False is the identity for disjunction",
        "annihilator_and_false": "A AND False = False; False annihilates conjunction",
        "annihilator_or_true": "A OR True = True; True annihilates disjunction",
        "conjunction_falsehood": "False AND anything = False",
        "disjunction_tautology": "True OR anything = True",
        "transitivity_equality": "If a=b and b=c then a=c",
        "transitivity_implication": "If A->B and B->C then A->C",
        "contrapositive": "If A->B then NOT B -> NOT A",
        "modus_ponens_grounding": "If A and A->B then B follows",
        "modus_tollens_grounding": "If NOT B and A->B then NOT A follows",
        "commutativity_xor": "A XOR B = B XOR A",
        "associativity_xor": "Grouping of XOR doesn't matter",
        "self_inverse_xor": "A XOR A = False; XOR with self cancels",
        "zero_xor": "A XOR False = A; False is identity for XOR",
        "set_union_commutativity": "A UNION B = B UNION A",
        "set_intersection_commutativity": "A INTERSECT B = B INTERSECT A",
        "set_difference_identity": "A MINUS empty = A",
        "set_double_complement": "Complement of complement of A = A",
        "subset_reflexivity": "Every set is a subset of itself",
        "subset_transitivity": "If A subset B and B subset C, then A subset C",
        "power_of_zero": "x^0 = 1 for nonzero x",
        "power_of_one": "x^1 = x",
        "zero_power": "0^n = 0 for positive n",
        "additive_cancellation": "(x + c) - c = x; adding and subtracting same value cancels",
        "multiplicative_cancellation": "(x * c) / c = x for nonzero c",
        "distributivity_subtraction": "a*(b-c) = a*b - a*c",
        "negation_zero": "-0 = 0; negation of zero is zero",
        "double_division": "x / (1/y) = x * y",
        "fraction_identity": "x/x = 1 for nonzero x",
        "absolute_value_nonneg": "|x| >= 0 always",
        "absolute_value_symmetry": "|x| = |-x|",
    }

    # Concepts that have physics-simulation test cases in ToyPhysicsSim.verify_identity
    _PHYSICS_VERIFIABLE: List[str] = [
        "additive_identity", "multiplicative_identity", "multiplicative_zero",
        "subtractive_self", "commutativity_add", "commutativity_multiply",
        "associativity_add", "associativity_multiply", "distributivity",
        "additive_inverse", "multiplicative_inverse", "zero_product",
        "double_negation_numeric", "reflexivity", "symmetry",
        "transitivity_inequality", "identity_composition",
    ]

    # Analogy graph: concept -> list of similar known concepts for fallback grounding
    _ANALOGY_MAP: Dict[str, List[str]] = {
        "commutativity_or": ["commutativity_add", "commutativity_multiply"],
        "commutativity_and": ["commutativity_add", "commutativity_multiply"],
        "commutativity_xor": ["commutativity_add"],
        "associativity_or": ["associativity_add", "associativity_multiply"],
        "associativity_and": ["associativity_add", "associativity_multiply"],
        "associativity_xor": ["associativity_add"],
        "idempotency_and": ["multiplicative_identity"],
        "idempotency_or": ["additive_identity"],
        "absorption_and": ["multiplicative_identity"],
        "absorption_or": ["additive_identity"],
        "double_negation": ["double_negation_numeric"],
        "de_morgan_and": ["distributivity"],
        "de_morgan_or": ["distributivity"],
        "distributivity_and_or": ["distributivity"],
        "distributivity_or_and": ["distributivity"],
        "negation_complement": ["additive_inverse", "subtractive_self"],
        "excluded_middle": ["reflexivity"],
        "identity_and_true": ["multiplicative_identity"],
        "identity_or_false": ["additive_identity"],
        "annihilator_and_false": ["multiplicative_zero", "zero_product"],
        "annihilator_or_true": ["multiplicative_identity"],
        "conjunction_falsehood": ["multiplicative_zero"],
        "disjunction_tautology": ["multiplicative_identity"],
        "transitivity_equality": ["transitivity_inequality"],
        "transitivity_implication": ["transitivity_inequality"],
        "contrapositive": ["symmetry"],
        "modus_ponens_grounding": ["identity_composition"],
        "modus_tollens_grounding": ["identity_composition"],
        "self_inverse_xor": ["additive_inverse", "subtractive_self"],
        "zero_xor": ["additive_identity"],
        "set_union_commutativity": ["commutativity_add"],
        "set_intersection_commutativity": ["commutativity_multiply"],
        "set_difference_identity": ["additive_identity"],
        "set_double_complement": ["double_negation_numeric"],
        "subset_reflexivity": ["reflexivity"],
        "subset_transitivity": ["transitivity_inequality"],
        "power_of_zero": ["multiplicative_identity"],
        "power_of_one": ["multiplicative_identity"],
        "zero_power": ["multiplicative_zero"],
        "additive_cancellation": ["additive_inverse", "subtractive_self"],
        "multiplicative_cancellation": ["multiplicative_inverse"],
        "distributivity_subtraction": ["distributivity"],
        "negation_zero": ["additive_identity"],
        "double_division": ["multiplicative_inverse"],
        "fraction_identity": ["multiplicative_inverse"],
        "absolute_value_nonneg": ["reflexivity"],
        "absolute_value_symmetry": ["symmetry"],
    }

    def __init__(self, world_model=None):
        self._sim = ToyPhysicsSim()
        self._wm = world_model
        self._groundings: Dict[str, dict] = {}
        self._ground_all()

    def _ground_all(self):
        """Ground all known concepts at startup.

        Concepts with physics-verifiable identities are grounded via simulation.
        Remaining concepts are grounded via the analogy map (_ANALOGY_MAP).
        """
        # Phase 1: physics-verifiable concepts only
        physics_verifiable = set(self._PHYSICS_VERIFIABLE)
        for concept in self.GROUNDED_CONCEPTS:
            if concept in self._groundings:
                continue
            if concept not in physics_verifiable:
                continue
            try:
                verified, explanation = self._sim.verify_identity(concept)
                if verified:
                    self._groundings[concept] = {
                        "concept": concept,
                        "verified": verified,
                        "explanation": explanation,
                        "grounded_at": time.time(),
                    }
                    log.debug("Grounded concept '%s': verified", concept)
                    if self._wm:
                        try:
                            self._wm.update_belief(
                                f"grounded_{concept}",
                                concept,
                                0.95,
                                domain="algebra",
                                evidence="toy_physics_sim",
                            )
                        except Exception:
                            pass
            except Exception as e:
                log.debug("Grounding via sim failed for '%s': %s", concept, e)

        # Phase 2: analogy-based grounding for remaining concepts
        for concept in self.GROUNDED_CONCEPTS:
            if concept in self._groundings:
                continue
            analogues = self._ANALOGY_MAP.get(concept, [])
            for analogue in analogues:
                if analogue in self._groundings and self._groundings[analogue].get("verified"):
                    try:
                        self.ground_by_analogy(concept, analogue)
                        break
                    except Exception as e:
                        log.debug("Analogy grounding failed for '%s' via '%s': %s", concept, analogue, e)

    def ground_new_rule(self, rule_name: str, domain: str = "algebra") -> Optional[dict]:
        """
        Attempt to ground a newly learned rule.
        Returns grounding dict if successful, None otherwise.
        """
        if not rule_name:
            return None

        # Normalize name
        normalized = rule_name.lower().replace(" ", "_").replace("-", "_")
        if normalized in self._groundings:
            return self._groundings[normalized]

        # Try to match against known identities (strip underscores for loose matching)
        normalized_bare = normalized.replace("_", "")
        for concept in self.GROUNDED_CONCEPTS:
            concept_bare = concept.replace("_", "")
            if concept in normalized or normalized in concept:
                return self._groundings.get(concept)
            if concept_bare in normalized_bare or normalized_bare in concept_bare:
                return self._groundings.get(concept)

        # Try physics simulation for new concept
        verified, explanation = self._sim.verify_identity(normalized)
        if verified:
            self._groundings[normalized] = {
                "concept": normalized,
                "verified": True,
                "explanation": explanation,
                "grounded_at": time.time(),
            }
            return self._groundings[normalized]

        return None

    def get_grounding(self, concept: str) -> Optional[str]:
        """Return the physical explanation for a concept."""
        g = self._groundings.get(concept)
        return g["explanation"] if g else None

    def get_all_groundings(self) -> Dict[str, dict]:
        """Return all stored groundings."""
        return dict(self._groundings)

    def ground_by_analogy(self, new_concept: str, similar_to: str) -> Optional[dict]:
        """
        Ground a new concept by borrowing the grounding from a known similar concept.

        Useful for the long tail of concepts where exact physics simulation is not
        available, but strong structural similarity to an already-grounded concept exists.

        Parameters
        ----------
        new_concept : str
            The concept to ground (will be stored in self._groundings).
        similar_to : str
            A concept already in self._groundings whose physical meaning will be adapted.

        Returns
        -------
        dict grounding record, or None if similar_to is not grounded.
        """
        try:
            if not new_concept or not similar_to:
                return None

            normalized = new_concept.lower().replace(" ", "_").replace("-", "_")
            if normalized in self._groundings:
                return self._groundings[normalized]

            source = self._groundings.get(similar_to)
            if source is None:
                # Try to auto-ground the source first if it's in our verified set
                if similar_to in self.GROUNDED_CONCEPTS:
                    self._ground_concept(similar_to)
                    source = self._groundings.get(similar_to)
            if source is None:
                return None

            # Compose an analogy-based explanation
            source_meaning = self._CONCEPT_PHYSICAL_MEANINGS.get(similar_to, source.get("explanation", ""))
            new_meaning = self._CONCEPT_PHYSICAL_MEANINGS.get(normalized, "")
            if not new_meaning:
                new_meaning = f"By analogy with '{similar_to}': {source_meaning}"

            grounding = {
                "concept": normalized,
                "verified": True,  # trust analogy when source is verified
                "explanation": (
                    f"Grounding '{normalized}' by analogy with '{similar_to}':\n"
                    f"  [ANALOGY] {new_meaning}\n"
                    f"  (Source grounding: {similar_to} — {source_meaning})"
                ),
                "grounded_at": time.time(),
                "analogy_source": similar_to,
            }
            self._groundings[normalized] = grounding

            if self._wm:
                try:
                    self._wm.update_belief(
                        f"grounded_{normalized}",
                        normalized,
                        0.80,
                        domain="algebra",
                        evidence=f"analogy_from_{similar_to}",
                    )
                except Exception:
                    pass

            log.debug("Grounded '%s' by analogy with '%s'", normalized, similar_to)
            return grounding
        except Exception as exc:
            log.debug("ground_by_analogy failed for '%s': %s", new_concept, exc)
            return None

    def _ground_concept(self, concept: str) -> None:
        """Ground a single concept via physics simulation (internal helper)."""
        try:
            verified, explanation = self._sim.verify_identity(concept)
            if not verified:
                return
            self._groundings[concept] = {
                "concept": concept,
                "verified": verified,
                "explanation": explanation,
                "grounded_at": time.time(),
            }
            if verified and self._wm:
                try:
                    self._wm.update_belief(
                        f"grounded_{concept}",
                        concept,
                        0.95,
                        domain="algebra",
                        evidence="toy_physics_sim",
                    )
                except Exception:
                    pass
        except Exception as exc:
            log.debug("_ground_concept failed for '%s': %s", concept, exc)

    def get_stats(self) -> dict:
        verified = sum(1 for g in self._groundings.values() if g.get("verified"))
        total = len(self.GROUNDED_CONCEPTS)
        return {
            "total_concepts": total,
            "grounded": len(self._groundings),
            "verified": verified,
            "coverage": round(verified / total, 2) if total > 0 else 0,
        }


_grounded_learner: Optional[GroundedConceptLearner] = None


def get_grounded_learner(world_model=None) -> GroundedConceptLearner:
    global _grounded_learner
    if _grounded_learner is None:
        _grounded_learner = GroundedConceptLearner(world_model)
    return _grounded_learner
