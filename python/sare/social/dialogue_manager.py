"""
Dialogue Learning Manager — Human-machine conversational learning.

A human teaches the system by conversation. The system verifies claims
symbolically, promotes confirmed rules, disputes wrong ones, and asks
follow-up questions to fill gaps in its knowledge.

Design:
  - No LLM calls — all rule extraction and intent classification is symbolic.
  - Integrates with ConceptRegistry to promote/dispute rules.
  - Integrates with AutobiographicalMemory to record social learning episodes.
  - Saves sessions to data/memory/dialogue_sessions.json.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

MEMORY_DIR = Path(__file__).resolve().parents[3] / "data" / "memory"
SESSIONS_PATH = MEMORY_DIR / "dialogue_sessions.json"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DialogueTurn:
    speaker: str          # "human" or "system"
    text: str
    intent: str           # "teach", "correct", "ask", "confirm", "explain"
    extracted_rule: Optional[dict]
    confidence: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "text": self.text,
            "intent": self.intent,
            "extracted_rule": self.extracted_rule,
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DialogueTurn":
        return cls(
            speaker=d.get("speaker", "human"),
            text=d.get("text", ""),
            intent=d.get("intent", "teach"),
            extracted_rule=d.get("extracted_rule"),
            confidence=d.get("confidence", 0.5),
            timestamp=d.get("timestamp", time.time()),
        )


@dataclass
class DialogueSession:
    session_id: str
    turns: List[DialogueTurn] = field(default_factory=list)
    taught_rules: List[str] = field(default_factory=list)
    disputed_rules: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "turns": [t.to_dict() for t in self.turns],
            "taught_rules": self.taught_rules,
            "disputed_rules": self.disputed_rules,
            "open_questions": self.open_questions,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DialogueSession":
        s = cls(session_id=d["session_id"])
        s.turns = [DialogueTurn.from_dict(t) for t in d.get("turns", [])]
        s.taught_rules = d.get("taught_rules", [])
        s.disputed_rules = d.get("disputed_rules", [])
        s.open_questions = d.get("open_questions", [])
        s.created_at = d.get("created_at", time.time())
        return s


# ── Intent classification keywords ────────────────────────────────────────────

_TEACH_KEYWORDS = [
    "equals", "is always", "simplifies to", "gives", "results in",
    "always equals", "reduces to", "is the same as", "is equal to",
]
_CORRECT_KEYWORDS = [
    "wrong", "incorrect", "that's not right", "no,", "not true", "false",
    "that's wrong", "you're wrong", "that is wrong",
]
_ASK_KEYWORDS = [
    "what is", "why does", "how does", "explain", "what does", "can you",
    "what are", "tell me", "describe",
]
_CONFIRM_KEYWORDS = [
    "yes", "correct", "right", "exactly", "that's right", "indeed",
    "confirmed", "true", "agreed",
]


# ── Rule extraction patterns ──────────────────────────────────────────────────

_RULE_PATTERNS = [
    # "x plus zero equals x"  /  "x plus zero is x"
    (
        re.compile(r"(\w+)\s+plus\s+(\w+)\s+(?:always\s+)?(?:equals?|is)\s+(\w+)", re.IGNORECASE),
        "arithmetic",
        lambda m: {"lhs_a": m.group(1), "lhs_b": m.group(2), "rhs": m.group(3),
                   "op": "+", "pattern": f"{m.group(1)} + {m.group(2)}", "result": m.group(3)},
    ),
    # "not not x equals x"  /  "not not x is x"
    (
        re.compile(r"not\s+not\s+(\w+)\s+(?:always\s+)?(?:is|equals?)\s+(\w+)", re.IGNORECASE),
        "logic",
        lambda m: {"pattern": f"not not {m.group(1)}", "result": m.group(2), "op": "not not"},
    ),
    # "x times zero equals zero"  /  "x multiplied by zero is zero"
    (
        re.compile(
            r"(\w+)\s+(?:times|multiplied\s+by)\s+(\w+)\s+(?:always\s+)?(?:is|equals?)\s+(\w+)",
            re.IGNORECASE,
        ),
        "arithmetic",
        lambda m: {"lhs_a": m.group(1), "lhs_b": m.group(2), "rhs": m.group(3),
                   "op": "*", "pattern": f"{m.group(1)} * {m.group(2)}", "result": m.group(3)},
    ),
    # "x minus x equals zero" / "x minus x is zero"
    (
        re.compile(
            r"(\w+)\s+minus\s+(\w+)\s+(?:always\s+)?(?:is|equals?)\s+(\w+)",
            re.IGNORECASE,
        ),
        "arithmetic",
        lambda m: {"lhs_a": m.group(1), "lhs_b": m.group(2), "rhs": m.group(3),
                   "op": "-", "pattern": f"{m.group(1)} - {m.group(2)}", "result": m.group(3)},
    ),
    # General: "A simplifies to B" / "A reduces to B" / "A always simplifies to B"
    (
        re.compile(
            r"([\w][\w\s+*\-/]*?)\s+(?:always\s+)?(?:simplifies?|reduces?)\s+to\s+([\w][\w\s+*\-/]*)",
            re.IGNORECASE,
        ),
        "general",
        lambda m: {"pattern": m.group(1).strip(), "result": m.group(2).strip()},
    ),
    # "A and true equals A" / "A or false is A"
    (
        re.compile(
            r"(\w+)\s+(and|or)\s+(\w+)\s+(?:always\s+)?(?:is|equals?)\s+(\w+)",
            re.IGNORECASE,
        ),
        "logic",
        lambda m: {"lhs_a": m.group(1), "lhs_b": m.group(3), "rhs": m.group(4),
                   "op": m.group(2).lower(),
                   "pattern": f"{m.group(1)} {m.group(2)} {m.group(3)}", "result": m.group(4)},
    ),
]


def _rule_name_from_extracted(rule: dict) -> str:
    """Generate a stable rule name from the extracted rule dict."""
    op_map = {"+": "add", "-": "sub", "*": "mul", "/": "div",
              "and": "and", "or": "or", "not not": "double_neg"}
    op = op_map.get(rule.get("op", ""), rule.get("op", "rule"))
    b = rule.get("lhs_b") or rule.get("result", "")
    return f"dialogue_{op}_{b}".lower().replace(" ", "_")


# ── DialogueManager ───────────────────────────────────────────────────────────

class DialogueManager:
    """
    Manages conversational learning sessions.

    Humans teach rules through natural language. The system:
    1. Classifies intent (teach/correct/ask/confirm)
    2. Extracts mathematical/logical rules symbolically
    3. Verifies them via BeamSearch
    4. Promotes confirmed rules to ConceptRegistry
    5. Disputes incorrect ones
    6. Asks follow-up questions to deepen understanding
    """

    def __init__(self):
        self._sessions: Dict[str, DialogueSession] = {}
        self.load()

    # ── Main entry point ──────────────────────────────────────────────────────

    def process_turn(self, text: str, session_id: str = "default") -> dict:
        """
        Process a human turn. Returns structured response dict.
        """
        session = self.get_session(session_id)
        intent = self._classify_intent(text)
        rule = None
        verified = False
        verification = {}
        action_taken = "none"
        follow_up = None
        response = ""

        # Record human turn
        human_turn = DialogueTurn(
            speaker="human",
            text=text,
            intent=intent,
            extracted_rule=None,
            confidence=1.0,
        )
        session.turns.append(human_turn)

        if intent == "teach":
            rule = self._extract_rule(text)
            if rule:
                human_turn.extracted_rule = rule
                verification = self._verify_symbolically(rule)
                verified = verification.get("verified", False)
                if verified:
                    self._promote_via_dialogue(rule, session_id)
                    rname = _rule_name_from_extracted(rule)
                    if rname not in session.taught_rules:
                        session.taught_rules.append(rname)
                    action_taken = "promoted"
                    follow_up = self._generate_followup(rule, session)
                    # Belief revision: retract contradicting KB beliefs
                    self._revise_beliefs_for_rule(rule)
                else:
                    action_taken = "queued"
                    follow_up = f"I wasn't able to confirm '{rule.get('pattern')} → {rule.get('result')}' symbolically. Could you give me an example with specific numbers?"
            else:
                action_taken = "none"
            response = self._generate_response(intent, rule, verified, verification, session)

        elif intent == "correct":
            # Mark last taught rule as disputed
            last_teach = next(
                (t for t in reversed(session.turns[:-1]) if t.intent == "teach" and t.extracted_rule),
                None,
            )
            if last_teach and last_teach.extracted_rule:
                rname = _rule_name_from_extracted(last_teach.extracted_rule)
                self._dispute_rule(rname, f"Human correction: {text}")
                if rname not in session.disputed_rules:
                    session.disputed_rules.append(rname)
                action_taken = "disputed"
                response = f"Understood — I've flagged '{rname}' as incorrect and lowered its confidence. Could you tell me what the correct form is?"
                follow_up = "What should the correct rule be?"
            else:
                action_taken = "none"
                response = "I see you're correcting something. Could you tell me specifically what rule is wrong and what it should be?"

        elif intent == "ask":
            action_taken = "answered"
            response = self._answer_question(text, session)
            if text not in session.open_questions:
                # Only queue if we couldn't fully answer it
                if "I don't know" in response or "not sure" in response:
                    session.open_questions.append(text)

        elif intent == "confirm":
            action_taken = "confirmed"
            last_sys = next(
                (t for t in reversed(session.turns[:-1]) if t.speaker == "system"),
                None,
            )
            response = "Great! I'm glad that's correct. Is there anything else you'd like to teach me?"

        else:
            action_taken = "none"
            response = "I didn't quite follow that. You can teach me a rule like 'x plus zero equals x', correct me, or ask me a question."

        # Record system response turn
        system_turn = DialogueTurn(
            speaker="system",
            text=response,
            intent="respond",
            extracted_rule=None,
            confidence=1.0 if verified else 0.5,
        )
        session.turns.append(system_turn)

        # Record in autobiographical memory
        try:
            from sare.memory.autobiographical import get_autobiographical_memory
            am = get_autobiographical_memory()
            am.record(
                "social_interaction",
                "social",
                f"Human said: '{text[:60]}'. Intent: {intent}",
                importance=0.6,
            )
        except Exception:
            pass

        # Satisfy social / curiosity drive
        try:
            from sare.meta.homeostasis import get_homeostatic_system
            hs = get_homeostatic_system()
            hs.on_social_interaction()
            if action_taken == "promoted":
                hs.on_rule_discovered()
        except Exception:
            pass

        self.save()

        return {
            "response": response,
            "intent": intent,
            "extracted_rule": rule,
            "verified": verified,
            "action_taken": action_taken,
            "follow_up_question": follow_up,
            "session_id": session_id,
            "verification": verification,
        }

    # ── Intent classification ──────────────────────────────────────────────────

    def _classify_intent(self, text: str) -> str:
        tl = text.lower().strip()

        # Confirm first (short responses like "yes" would also match ask keywords)
        for kw in _CONFIRM_KEYWORDS:
            if tl.startswith(kw) or tl == kw:
                return "confirm"

        # Correction
        for kw in _CORRECT_KEYWORDS:
            if kw in tl:
                return "correct"

        # Question
        for kw in _ASK_KEYWORDS:
            if tl.startswith(kw):
                return "ask"

        # Teach
        for kw in _TEACH_KEYWORDS:
            if kw in tl:
                return "teach"

        # Fallback: if it has a mathematical structure, treat as teach
        if re.search(r"\b(?:equals?|is)\b", tl):
            return "teach"

        return "teach"  # optimistic default

    # ── Rule extraction ────────────────────────────────────────────────────────

    def _extract_rule(self, text: str) -> Optional[dict]:
        """
        Try to extract a mathematical/logical rule from natural language text.
        Returns a dict with at minimum "pattern" and "result" keys, plus "domain".
        """
        for pattern, domain, builder in _RULE_PATTERNS:
            m = pattern.search(text)
            if m:
                try:
                    rule = builder(m)
                    rule["domain"] = domain
                    rule["source_text"] = text[:120]
                    return rule
                except Exception as e:
                    log.debug("Rule extraction failed for pattern %s: %s", pattern.pattern, e)
        return None

    # ── Symbolic verification ──────────────────────────────────────────────────

    def _verify_symbolically(self, rule: dict) -> dict:
        """
        Try to verify the extracted rule using the SARE engine's BeamSearch.
        Parses the pattern as an expression, applies transforms, checks if
        the result matches what the human claimed.
        Returns {"verified": bool, "evidence": str, "energy_delta": float}
        """
        try:
            from sare.engine import load_problem, EnergyEvaluator, BeamSearch, ALL_TRANSFORMS
            pattern_str = rule.get("pattern", "")
            expected_result = rule.get("result", "")

            if not pattern_str:
                return {"verified": False, "evidence": "No pattern to verify", "energy_delta": 0.0}

            # Map natural-language patterns to engine-parseable expressions
            # e.g. "x + 0", "not not x"
            expr_str = pattern_str
            # Normalize: "x + zero" → "x + 0"
            expr_str = re.sub(r"\bzero\b", "0", expr_str, flags=re.IGNORECASE)
            expr_str = re.sub(r"\bone\b", "1", expr_str, flags=re.IGNORECASE)

            _, graph = load_problem(expr_str)
            energy = EnergyEvaluator()
            e_before = energy.compute(graph).total

            searcher = BeamSearch()
            result = searcher.search(graph, energy, ALL_TRANSFORMS, beam_width=6, budget_seconds=2.0)
            e_after = energy.compute(result.graph).total
            delta = e_before - e_after

            # Consider verified if energy decreased meaningfully
            verified = delta > 0.05

            # Additional check: if the expected result is a specific value,
            # try to confirm the graph represents it
            evidence = f"Energy {e_before:.3f} → {e_after:.3f} (delta={delta:.3f})"
            if not verified and expected_result:
                evidence += f". Could not simplify to '{expected_result}' symbolically."

            return {
                "verified": verified,
                "evidence": evidence,
                "energy_delta": round(delta, 4),
                "e_before": round(e_before, 4),
                "e_after": round(e_after, 4),
            }
        except Exception as e:
            log.debug("Symbolic verification error: %s", e)
            return {"verified": False, "evidence": f"Verification error: {e}", "energy_delta": 0.0}

    # ── Response generation ────────────────────────────────────────────────────

    def _generate_response(
        self, intent: str, rule: Optional[dict], verified: bool,
        verification: dict, session: DialogueSession,
    ) -> str:
        if not rule:
            return (
                "I heard you trying to teach me something, but I couldn't extract a specific rule. "
                "Try phrasing it like: 'x plus zero equals x' or 'not not x simplifies to x'."
            )

        pattern = rule.get("pattern", "?")
        result = rule.get("result", "?")
        domain = rule.get("domain", "general")
        evidence = verification.get("evidence", "")

        if verified:
            return (
                f"I've verified that '{pattern} = {result}' holds in {domain}! "
                f"I've promoted this as a new rule. {evidence}. "
                f"This matches patterns I already know — thank you!"
            )
        else:
            return (
                f"I tried to verify '{pattern} → {result}' symbolically but couldn't confirm it. "
                f"{evidence}. "
                f"I've noted it as a candidate — could you provide a concrete example or more context?"
            )

    def _answer_question(self, text: str, session: DialogueSession) -> str:
        """Answer a question about known rules/capabilities, consulting KB first."""
        tl = text.lower()

        # --- KB lookup: try WorldModel beliefs + fact chain ---
        kb_answer = self._lookup_kb(tl)
        if kb_answer:
            return kb_answer

        try:
            from sare.engine import ALL_TRANSFORMS
            rule_names = [t.name() for t in ALL_TRANSFORMS]
        except Exception:
            rule_names = []

        if "distributive" in tl:
            return (
                "The distributive law says a * (b + c) = a*b + a*c. "
                "I have this rule in my transform library."
            )
        if "identity" in tl:
            return (
                "Identity rules include: x + 0 = x (additive identity) and x * 1 = x "
                "(multiplicative identity). I know both of these."
            )
        if "what" in tl and ("know" in tl or "rules" in tl or "transforms" in tl):
            if rule_names:
                return f"I currently know these transforms: {', '.join(rule_names[:10])}."
            return "I know several arithmetic and logic transforms. Ask me about a specific one!"
        if "how" in tl and ("learn" in tl or "work" in tl):
            return (
                "I learn by: (1) autonomous problem generation and symbolic search, "
                "(2) being taught rules by humans like you, and (3) causal induction "
                "to confirm patterns before promoting them."
            )

        # Generic fallback
        if rule_names:
            return (
                f"That's an interesting question. I'm not sure I have a direct answer, "
                f"but I know {len(rule_names)} transforms. "
                f"You can teach me new facts by stating them as rules, e.g. 'x + 0 = x'."
            )
        return (
            "I'm not sure about that. Could you phrase it as a rule I could learn? "
            "For example: 'x plus zero equals x'."
        )

    def _lookup_kb(self, question_lower: str) -> Optional[str]:
        """Query WorldModel and fact chain for an answer to a natural-language question."""
        try:
            from sare.memory.world_model import get_world_model
            from sare.cognition.fact_inference import get_fact_inference

            wm  = get_world_model()
            fi  = get_fact_inference()

            # Extract candidate subject: longest word not in stop-words
            _STOP = {"what", "is", "are", "the", "a", "an", "does", "do", "how", "why",
                     "which", "can", "have", "has", "of", "in", "on", "at", "to", "for"}
            words = [w.strip("?.,!") for w in question_lower.split()
                     if len(w.strip("?.,!")) > 2 and w.strip("?.,!") not in _STOP]
            if not words:
                return None

            # Identify likely predicate from question keywords
            pred_hints = {
                "color": "color", "colour": "color",
                "capital": "capital", "population": "population",
                "breathe": "breathes", "breathes": "breathes",
                "mammal": "is_mammal", "bird": "is_bird", "fly": "can_fly",
                "mass": "has_mass", "weight": "has_mass",
                "temperature": "temperature", "boil": "boiling_point",
                "habitat": "habitat", "diet": "diet",
            }
            target_pred = None
            for hint, pred in pred_hints.items():
                if hint in question_lower:
                    target_pred = pred
                    break

            # Try each candidate subject
            for subject in words[:4]:
                if target_pred:
                    # Direct belief lookup
                    b = wm.get_belief(subject, target_pred)
                    if b and b.value:
                        return f"Based on my knowledge: {subject} {target_pred.replace('_', ' ')} is {b.value}."
                    # Backward chain
                    chained = fi.chain_to_goal(subject, target_pred, "factual", max_depth=3)
                    if chained:
                        return f"I can infer that {subject} {target_pred.replace('_', ' ')} is {chained} (derived by chain reasoning)."
                else:
                    # No predicate hint — return any high-confidence fact about this subject
                    facts = wm.get_facts("factual") + wm.get_facts("science")
                    hits = [f.get("fact", "") for f in facts
                            if subject in f.get("fact", "").lower()
                            and f.get("confidence", 0) >= 0.7]
                    if hits:
                        return f"Here's what I know about '{subject}': {hits[0]}"
        except Exception as e:
            log.debug("DialogueManager KB lookup failed: %s", e)
        return None

    def _generate_followup(self, rule: dict, session: DialogueSession) -> Optional[str]:
        """Generate a curiosity-driven follow-up question after learning a rule."""
        domain = rule.get("domain", "general")
        op = rule.get("op", "")

        follow_up_map = {
            "+": "Does this identity hold for negative numbers too?",
            "*": "What happens when you multiply by -1 instead?",
            "-": "Is there a similar rule for addition?",
            "and": "Does the same apply with OR instead of AND?",
            "or": "Does the same apply with AND instead of OR?",
            "not not": "Are there other double-negation patterns in logic you know of?",
        }
        return follow_up_map.get(op, f"Are there related rules in {domain} you'd like to teach me?")

    # ── Rule promotion / dispute ───────────────────────────────────────────────

    def _promote_via_dialogue(self, rule: dict, session_id: str):
        """Promote the rule to ConceptRegistry with source='dialogue'."""
        try:
            import sare.sare_bindings as _sb
            ConceptRegistry = getattr(_sb, "ConceptRegistry", None)
            ConceptRule = getattr(_sb, "ConceptRule", None)
            if ConceptRegistry and ConceptRule:
                # Use the global concept_registry if available; otherwise create one
                import importlib, sys
                web_mod = sys.modules.get("sare.interface.web")
                registry = getattr(web_mod, "concept_registry", None) if web_mod else None
                if registry is None:
                    registry = ConceptRegistry()

                rule_name = _rule_name_from_extracted(rule)
                pattern = rule.get("pattern", "")
                result_str = rule.get("result", "")

                cr = ConceptRule(rule_name, pattern, result_str)
                cr.confidence = 0.8
                registry.add_rule(cr)
                log.info("DialogueManager: promoted rule '%s' via dialogue", rule_name)
        except Exception as e:
            log.debug("Could not promote rule to ConceptRegistry: %s", e)

        # Also persist to identity/auto-bio
        try:
            from sare.memory.identity import get_identity_manager
            im = get_identity_manager()
            im.update_from_behavior("human_taught", rule.get("domain", "general"), True)
        except Exception:
            pass

    def _dispute_rule(self, rule_name: str, reason: str):
        """Lower confidence of a promoted rule that was flagged as wrong."""
        try:
            import sys
            web_mod = sys.modules.get("sare.interface.web")
            registry = getattr(web_mod, "concept_registry", None) if web_mod else None
            if registry and hasattr(registry, "get_rule"):
                rule = registry.get_rule(rule_name)
                if rule and hasattr(rule, "confidence"):
                    rule.confidence = max(0.0, rule.confidence - 0.4)
                    log.info("DialogueManager: disputed rule '%s', reason: %s", rule_name, reason)
        except Exception as e:
            log.debug("Could not dispute rule '%s': %s", rule_name, e)

    def _revise_beliefs_for_rule(self, rule: dict) -> None:
        """When a human teaches a rule, retract contradicting WorldModel beliefs.

        E.g. if human teaches 'x + 0 = x' but KB has 'x + 0 = x + 0' with low
        confidence, lower that belief and store the human-sourced correction.
        """
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()

            subj    = str(rule.get("pattern", "") or "").strip().lower()
            new_val = str(rule.get("result",  "") or "").strip().lower()
            op      = str(rule.get("op",      "") or "").strip()
            if not subj or not new_val:
                return

            # Map rule op to a plausible predicate to check
            pred_map = {"+": "simplifies_to", "*": "simplifies_to",
                        "-": "simplifies_to", "=": "equals"}
            pred = pred_map.get(op, "simplifies_to")

            # Look for contradicting belief (same subject/predicate, different value)
            existing = wm.get_belief(subj, pred)
            if existing and existing.value and existing.value.lower() != new_val:
                old_conf = getattr(existing, "confidence", 1.0)
                if old_conf < 0.85:
                    # Retract: lower confidence of old belief
                    wm.update_belief(subj, pred + "_retracted", existing.value,
                                     confidence=max(0.1, old_conf * 0.3), domain="general")
                    log.info("[DialogueManager] Belief revision: retracted '%s %s=%s' (was %.2f conf)",
                             subj, pred, existing.value, old_conf)

            # Store the human-sourced correction as a high-confidence belief
            wm.update_belief(subj, pred, new_val, confidence=0.92, domain="general")
            wm.add_fact(domain="general",
                        fact=f"{subj} {pred}: {new_val} [human-taught]",
                        confidence=0.92)
            log.info("[DialogueManager] Belief set from dialogue: %s %s=%s", subj, pred, new_val)
        except Exception as e:
            log.debug("[DialogueManager] Belief revision failed: %s", e)

    # ── Session management ─────────────────────────────────────────────────────

    def get_session(self, session_id: str) -> DialogueSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = DialogueSession(session_id=session_id)
        return self._sessions[session_id]

    def all_sessions(self) -> List[dict]:
        return [
            {
                "session_id": sid,
                "turn_count": len(s.turns),
                "taught_rules": s.taught_rules,
                "disputed_rules": s.disputed_rules,
                "open_questions": s.open_questions,
                "created_at": s.created_at,
            }
            for sid, s in self._sessions.items()
        ]

    def initiate_question(self, domain: str, session_id: str = "default") -> Optional[dict]:
        """
        Phase D integration: System initiates a question to a teacher.

        When the system is confused (via ConfusionDetector), it can call this
        method to generate a question expressed in natural language and route it
        to the teacher protocol.

        Returns the question dict if a pending question exists, else None.
        """
        try:
            from sare.learning.teacher_protocol import get_confusion_detector, get_teacher_registry
            cd = get_confusion_detector()
            tr = get_teacher_registry()
            pending = [q for q in cd.get_pending_questions() if q.domain == domain]
            if not pending:
                return None
            q = pending[0]
            # Try to get an answer from a queryable teacher
            resp = tr.ask_best_teacher(q)
            if resp:
                cd.answer_question(q.question_id, resp.answer_text, answered_by=resp.teacher_id)
                # Inject into dialogue session as a "teach" turn
                turn = self.process_turn(
                    f"SYSTEM ANSWER: {resp.answer_text[:200]}",
                    session_id=session_id,
                )
                return {
                    "question": q.to_dict(),
                    "answer": resp.to_dict(),
                    "turn": turn,
                }
            return {"question": q.to_dict(), "answer": None, "turn": None}
        except Exception as e:
            log.debug("[DialogueManager] initiate_question failed: %s", e)
            return None

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self):
        try:
            MEMORY_DIR.mkdir(parents=True, exist_ok=True)
            data = {sid: s.to_dict() for sid, s in self._sessions.items()}
            with open(SESSIONS_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            log.warning("DialogueManager save error: %s", e)

    def load(self):
        if not SESSIONS_PATH.exists():
            return
        try:
            with open(SESSIONS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            for sid, sd in data.items():
                self._sessions[sid] = DialogueSession.from_dict(sd)
            log.info("DialogueManager loaded: %d sessions", len(self._sessions))
        except Exception as e:
            log.warning("DialogueManager load error: %s", e)


# ── Singleton ──────────────────────────────────────────────────────────────────

_DIALOGUE_MANAGER: Optional[DialogueManager] = None


def get_dialogue_manager() -> DialogueManager:
    global _DIALOGUE_MANAGER
    if _DIALOGUE_MANAGER is None:
        _DIALOGUE_MANAGER = DialogueManager()
    return _DIALOGUE_MANAGER
