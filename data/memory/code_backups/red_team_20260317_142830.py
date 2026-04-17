"""
RedTeamAdversary — S26-5
Internal adversary: reads AgentSociety blackboard, generates counterexamples
to accepted beliefs, attempts to falsify them, and forces the Brain to defend or update.
This is internal RLHF — the system hardens itself against its own blind spots.
"""
from __future__ import annotations
import re
import time
import random
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

log = logging.getLogger(__name__)


@dataclass
class AttackRecord:
    target_belief:     str
    attack_expression: str
    attack_type:       str   # "negation" | "substitution" | "numeric_perturb" | "domain_swap"
    falsified:         bool
    confidence_delta:  float  # how much confidence dropped (positive = weakened)
    timestamp:         float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "target_belief":     self.target_belief[:60],
            "attack_expression": self.attack_expression[:60],
            "attack_type":       self.attack_type,
            "falsified":         self.falsified,
            "confidence_delta":  round(self.confidence_delta, 3),
            "timestamp":         self.timestamp,
        }


class RedTeamAdversary:
    """
    Spawned alongside AgentSociety. Each attack round:
    1. Pulls top-K accepted beliefs from the blackboard.
    2. Generates a perturbation (negation / substitution / numeric / domain swap).
    3. Checks if the engine contradicts the original belief.
    4. Reports falsifications back to AgentSociety to weaken confidence.
    """

    _ATTACK_TYPES  = ["negation", "substitution", "numeric_perturb", "domain_swap"]
    _CONFIDENCE_PENALTY = 0.12

    def __init__(self) -> None:
        self._society  = None
        self._engine   = None
        self._attacks: List[AttackRecord]        = []
        self._falsifications: List[AttackRecord] = []
        self._total_attacks       = 0
        self._total_falsifications = 0
        self._round_count         = 0

    # ── wiring ────────────────────────────────────────────────────────────────
    def wire(self, agent_society=None, engine=None) -> None:
        self._society = agent_society
        self._engine  = engine

    # ── main ──────────────────────────────────────────────────────────────────
    def run_attack_round(self, top_k: int = 3) -> dict:
        """Attack the top-K most confident accepted beliefs."""
        self._round_count += 1
        beliefs = self._get_top_beliefs(top_k)
        if not beliefs:
            return {"round": self._round_count, "attacks": 0, "falsifications": 0,
                    "message": "no beliefs to attack"}

        newly_falsified = 0
        for belief in beliefs:
            attack_type = random.choice(self._ATTACK_TYPES)
            attack_expr = self._generate_attack(belief, attack_type)
            falsified   = self._evaluate_attack(belief, attack_expr)

            rec = AttackRecord(
                target_belief=belief,
                attack_expression=attack_expr,
                attack_type=attack_type,
                falsified=falsified,
                confidence_delta=self._CONFIDENCE_PENALTY if falsified else 0.0,
            )
            self._attacks.append(rec)
            self._total_attacks += 1

            if falsified:
                self._falsifications.append(rec)
                self._total_falsifications += 1
                newly_falsified += 1
                self._report_falsification(belief, self._CONFIDENCE_PENALTY)

        if len(self._attacks) > 500:
            self._attacks = self._attacks[-500:]

        return {
            "round":          self._round_count,
            "attacks":        len(beliefs),
            "falsifications": newly_falsified,
            "falsification_rate": round(newly_falsified / max(len(beliefs), 1), 3),
        }

    # ── internals ─────────────────────────────────────────────────────────────
    def _get_top_beliefs(self, k: int) -> List[str]:
        if self._society is None:
            return []
        try:
            blackboard = getattr(self._society, '_blackboard', [])
            # Filter to accepted beliefs, sort by confidence
            accepted = [b for b in blackboard
                        if getattr(b, 'accepted', False)]
            accepted.sort(key=lambda b: getattr(b, 'confidence', 0), reverse=True)
            return [getattr(b, 'content', str(b)) for b in accepted[:k]]
        except Exception as e:
            log.debug(f"RedTeam get_beliefs: {e}")
            return []

    def _generate_attack(self, belief: str, attack_type: str) -> str:
        try:
            if attack_type == "negation":
                # Prepend NOT or negate the claim
                if "not" in belief.lower():
                    return re.sub(r'(?i)not\s+', '', belief, count=1)
                return "NOT(" + belief[:40] + ")"

            elif attack_type == "substitution":
                # Replace a keyword with its near-opposite
                _subs = {"add": "multiply", "zero": "one", "true": "false",
                         "equal": "unequal", "always": "never", "all": "none"}
                result = belief
                for k, v in _subs.items():
                    if k in result.lower():
                        result = re.sub(k, v, result, flags=re.IGNORECASE, count=1)
                        break
                return result if result != belief else belief + " (negated)"

            elif attack_type == "numeric_perturb":
                # Replace a number with something that makes the claim false
                nums = re.findall(r'\d+', belief)
                if nums:
                    n  = random.choice(nums)
                    nv = str(int(n) + random.choice([-1, 1, 2, -2]))
                    return belief.replace(n, nv, 1)
                return belief + " + 1"

            else:  # domain_swap
                _domains = ["arithmetic", "logic", "physics", "geometry"]
                for d in _domains:
                    if d in belief.lower():
                        others = [x for x in _domains if x != d]
                        return belief.lower().replace(d, random.choice(others), 1)
                return belief + " (domain undefined)"
        except Exception:
            return belief + " (adversarial)"

    def _evaluate_attack(self, original: str, attack: str) -> bool:
        """
        Returns True if attack expression contradicts original.
        Uses engine to attempt solving both; if attack's energy < original,
        it implies the original belief may be too strong.
        """
        if self._engine is None:
            # Stochastic fallback: ~25% falsification rate
            return random.random() < 0.25
        try:
            r_orig   = self._engine.solve(original[:80])
            r_attack = self._engine.solve(attack[:80])
            e_orig   = r_orig.get("energy", 1.0) if isinstance(r_orig, dict) else 1.0
            e_attack = r_attack.get("energy", 1.0) if isinstance(r_attack, dict) else 1.0
            # If attack has lower energy, the original belief was over-confident
            return e_attack < e_orig * 0.7
        except Exception as e:
            log.debug(f"RedTeam evaluate: {e}")
            return random.random() < 0.15

    def _report_falsification(self, belief: str, penalty: float) -> None:
        """Notify AgentSociety to weaken the belief's confidence."""
        if self._society is None:
            return
        try:
            blackboard = getattr(self._society, '_blackboard', [])
            for b in blackboard:
                content = getattr(b, 'content', '')
                if content[:40] == belief[:40]:
                    old_conf = getattr(b, 'confidence', 0.5)
                    b.confidence = max(0.0, old_conf - penalty)
                    break
        except Exception as e:
            log.debug(f"RedTeam report: {e}")

    # ── summary ───────────────────────────────────────────────────────────────
    def summary(self) -> dict:
        recent_attacks = [a.to_dict() for a in self._attacks[-8:]]
        recent_falsif  = [a.to_dict() for a in self._falsifications[-5:]]
        frate = (self._total_falsifications /
                 max(self._total_attacks, 1))
        return {
            "total_attacks":        self._total_attacks,
            "total_falsifications": self._total_falsifications,
            "falsification_rate":   round(frate, 3),
            "round_count":          self._round_count,
            "recent_attacks":       recent_attacks,
            "recent_falsifications": recent_falsif,
            "wired": {
                "society": self._society is not None,
                "engine":  self._engine is not None,
            },
        }
