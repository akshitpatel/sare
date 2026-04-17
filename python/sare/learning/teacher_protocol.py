"""
TeacherProtocol — Unified Active Learning from Any Knowledge Source
====================================================================

Human analogy: a child stuck for 10 minutes raises their hand.
All knowledge sources (human teacher, LLM, textbook, database) speak the
same protocol. The system can initiate questions — not just answer them.

Architecture:
  ConfusionDetector  → monitors prediction errors per domain, generates
                        LearningQuestions when persistently stuck
  TeacherRegistry    → routes question to highest-trust queryable teacher
  Teacher (ABC)      → base class; LLMTeacher, HumanTeacher, DatabaseTeacher
  LearningQuestion   → what the system wants to know (encoded in internal lang)

Integration points:
  experiment_runner.py:  observe_failure() + check_teacher_responses()
  web.py endpoints:
    GET  /api/questions/pending
    POST /api/questions/answer     (human submits)
    POST /api/teachers/register
    GET  /api/teachers
"""
from __future__ import annotations

import json
import os
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_MEMORY = Path(__file__).resolve().parents[4] / "data" / "memory"


# ═══════════════════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LearningQuestion:
    """A question the system generates when persistently stuck."""
    question_id:       str
    question_text:     str
    domain:            str
    confusion_type:    str   # "transform_gap", "rule_unknown", "domain_unknown"
    stuck_expressions: List[str]
    transforms_tried:  List[str]
    urgency:           float  # [0, 1] based on how many times stuck
    status:            str = "pending"   # "pending", "answered", "failed"
    answer:            Optional[str] = None
    answered_by:       Optional[str] = None
    created_at:        float = field(default_factory=time.time)
    answered_at:       Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "question_id":      self.question_id,
            "question_text":    self.question_text,
            "domain":           self.domain,
            "confusion_type":   self.confusion_type,
            "stuck_expressions": self.stuck_expressions,
            "transforms_tried": self.transforms_tried,
            "urgency":          round(self.urgency, 3),
            "status":           self.status,
            "answer":           self.answer,
            "answered_by":      self.answered_by,
            "created_at":       self.created_at,
            "answered_at":      self.answered_at,
        }


@dataclass
class TeacherResponse:
    """Response from a teacher, may contain rule suggestions."""
    teacher_id:       str
    question_id:      str
    answer_text:      str
    suggested_rules:  List[dict]   # [{name, pattern, domain, confidence}]
    confidence:       float        # teacher's self-reported confidence
    timestamp:        float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "teacher_id":     self.teacher_id,
            "question_id":    self.question_id,
            "answer_text":    self.answer_text,
            "suggested_rules": self.suggested_rules,
            "confidence":     round(self.confidence, 3),
            "timestamp":      self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Teacher base class & implementations
# ═══════════════════════════════════════════════════════════════════════════════

class Teacher(ABC):
    """Abstract teacher — anything that can answer LearningQuestions."""

    def __init__(self, teacher_id: str, domains: Optional[List[str]] = None):
        self.teacher_id = teacher_id
        self.domains = domains or []   # empty = handles all domains
        self.is_queryable: bool = False   # can system initiate?
        self._trust_scores: Dict[str, float] = {}  # domain → trust
        self.total_answers: int = 0
        self.correct_answers: int = 0

    def trust_score(self, domain: str) -> float:
        return self._trust_scores.get(domain, self._trust_scores.get("general", 0.5))

    def update_trust(self, domain: str, correct: bool):
        t = self._trust_scores.get(domain, 0.5)
        if correct:
            self._trust_scores[domain] = min(1.0, t + 0.05)
        else:
            self._trust_scores[domain] = max(0.0, t - 0.10)

    def can_answer(self, question: LearningQuestion) -> bool:
        if not self.domains:
            return True
        return question.domain in self.domains

    @abstractmethod
    def ask(self, question: LearningQuestion) -> Optional[TeacherResponse]:
        ...

    def to_dict(self) -> dict:
        return {
            "teacher_id":    self.teacher_id,
            "teacher_type":  self.__class__.__name__,
            "domains":       self.domains,
            "is_queryable":  self.is_queryable,
            "trust_scores":  {d: round(v, 3) for d, v in self._trust_scores.items()},
            "total_answers": self.total_answers,
            "correct_answers": self.correct_answers,
        }


class LLMTeacher(Teacher):
    """Teacher backed by the LLM bridge (can be queried autonomously)."""

    def __init__(self, teacher_id: str = "llm_default", domains: Optional[List[str]] = None):
        super().__init__(teacher_id, domains)
        self.is_queryable = True  # LLM can be queried without human interaction

    def ask(self, question: LearningQuestion) -> Optional[TeacherResponse]:
        try:
            from sare.interface.llm_bridge import get_llm_bridge
            llm = get_llm_bridge()
        except Exception:
            return None

        exprs = "\n".join(f"  - {e}" for e in question.stuck_expressions[:5])
        transforms = ", ".join(question.transforms_tried[:8]) or "none"
        prompt = (
            f"An AI reasoning system is stuck on these math expressions:\n{exprs}\n\n"
            f"Domain: {question.domain}\n"
            f"Transforms already tried: {transforms}\n\n"
            f"Question: {question.question_text}\n\n"
            "Please provide:\n"
            "1. A brief explanation of why these are hard.\n"
            "2. One or two transform rules that could help (in format: RULE: pattern → result).\n"
            "3. Your confidence: HIGH / MEDIUM / LOW\n"
            "Keep your answer concise (under 200 words)."
        )
        try:
            response_text = llm.complete(prompt)
        except Exception as e:
            log.debug("[LLMTeacher] LLM call failed: %s", e)
            return None

        # Parse suggested rules from response
        rules = _parse_rules_from_text(response_text, question.domain)
        confidence = 0.7
        if "LOW" in response_text.upper():
            confidence = 0.4
        elif "MEDIUM" in response_text.upper():
            confidence = 0.65
        elif "HIGH" in response_text.upper():
            confidence = 0.85

        self.total_answers += 1
        return TeacherResponse(
            teacher_id=self.teacher_id,
            question_id=question.question_id,
            answer_text=response_text,
            suggested_rules=rules,
            confidence=confidence,
        )


class HumanTeacher(Teacher):
    """Teacher answered by a human via the web UI."""

    def __init__(self, teacher_id: str = "human_0", domains: Optional[List[str]] = None):
        super().__init__(teacher_id, domains)
        self.is_queryable = False  # requires human to visit /api/questions/pending
        self._trust_scores["general"] = 0.95   # humans trusted by default

    def ask(self, question: LearningQuestion) -> Optional[TeacherResponse]:
        # Questions appear at /api/questions/pending; human submits via /api/questions/answer
        # This method returns None — answers arrive asynchronously
        return None


class DatabaseTeacher(Teacher):
    """Teacher backed by structured knowledge files (JSON)."""

    def __init__(
        self,
        teacher_id: str = "db_default",
        domains: Optional[List[str]] = None,
        db_path: Optional[str] = None,
    ):
        super().__init__(teacher_id, domains)
        self.is_queryable = True
        self._db: Dict[str, List[dict]] = {}
        self._db_path = Path(db_path) if db_path else None
        self._load_db()

    def _load_db(self):
        if self._db_path and self._db_path.exists():
            try:
                self._db = json.loads(self._db_path.read_text())
            except Exception:
                pass

    def ask(self, question: LearningQuestion) -> Optional[TeacherResponse]:
        domain_entries = self._db.get(question.domain, [])
        if not domain_entries:
            return None
        # Return the first entry as a rule suggestion
        rules = domain_entries[:2]
        self.total_answers += 1
        return TeacherResponse(
            teacher_id=self.teacher_id,
            question_id=question.question_id,
            answer_text=f"Database: {len(rules)} rules found for domain {question.domain}",
            suggested_rules=rules,
            confidence=0.8,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  ConfusionDetector
# ═══════════════════════════════════════════════════════════════════════════════

class ConfusionDetector:
    """
    Monitors failures per domain. When persistently stuck, generates
    a LearningQuestion to route to a teacher.

    Generates a question when:
      - ≥5 consecutive failures in the same domain
      - social_drive ≥ 0.4 (system is in a social/learning orientation)
    """

    MIN_FAILURES_TO_QUESTION = 5
    MIN_SOCIAL_DRIVE = 0.4
    COOLDOWN_SECONDS = 600   # don't generate the same-domain question within 10 min

    def __init__(self):
        self._failure_counts: Dict[str, int] = {}
        self._failure_exprs: Dict[str, List[str]] = {}
        self._failure_transforms: Dict[str, List[str]] = {}
        self._last_question_time: Dict[str, float] = {}
        self._pending_questions: List[LearningQuestion] = []
        self._answered_questions: List[LearningQuestion] = []
        self._persist_path = _MEMORY / "confusion_detector.json"
        self._load()

    def observe_failure(self, domain: str, expression: str, transforms_tried: List[str]):
        """Called by experiment_runner after a failed solve attempt."""
        self._failure_counts[domain] = self._failure_counts.get(domain, 0) + 1
        self._failure_exprs.setdefault(domain, []).append(expression)
        for t in transforms_tried:
            self._failure_transforms.setdefault(domain, []).append(t)

        # Keep lists bounded
        if len(self._failure_exprs[domain]) > 20:
            self._failure_exprs[domain] = self._failure_exprs[domain][-20:]
        if len(self._failure_transforms[domain]) > 30:
            self._failure_transforms[domain] = self._failure_transforms[domain][-30:]

        # Check if we should generate a question
        self._maybe_generate_question(domain)

    def _get_social_drive(self) -> float:
        try:
            from sare.meta.homeostasis import get_homeostatic_system
            state = get_homeostatic_system().get_state()
            drives = state.get("drives", {})
            return float(drives.get("social", 0.5))
        except Exception:
            return 0.5

    def _maybe_generate_question(self, domain: str):
        count = self._failure_counts.get(domain, 0)
        if count < self.MIN_FAILURES_TO_QUESTION:
            return

        # Check cooldown
        last = self._last_question_time.get(domain, 0)
        if time.time() - last < self.COOLDOWN_SECONDS:
            return

        # Check social drive
        if self._get_social_drive() < self.MIN_SOCIAL_DRIVE:
            return

        # Generate the question
        stuck_exprs = list(set(self._failure_exprs.get(domain, [])))[:5]
        transforms_tried = list(set(self._failure_transforms.get(domain, [])))[:8]
        urgency = min(1.0, count / 20.0)

        question_text = (
            f"I've failed {count} times in the '{domain}' domain and cannot find a "
            f"simplification strategy. The expressions that stop me are: "
            + ", ".join(stuck_exprs[:3]) +
            ". What transform or rule am I missing?"
        )

        q = LearningQuestion(
            question_id=str(uuid.uuid4())[:8],
            question_text=question_text,
            domain=domain,
            confusion_type="transform_gap",
            stuck_expressions=stuck_exprs,
            transforms_tried=transforms_tried,
            urgency=urgency,
        )
        self._pending_questions.append(q)
        self._last_question_time[domain] = time.time()
        # Reset failure count after generating question
        self._failure_counts[domain] = 0
        log.info("[ConfusionDetector] Generated question for domain '%s' (urgency=%.2f)", domain, urgency)
        self._save()
        # Immediately route to LLM — no human input required
        self._auto_answer_via_llm(q)

    def _auto_answer_via_llm(self, q: "LearningQuestion"):
        """Route the question to the best available queryable teacher (LLM) immediately."""
        import threading
        def _do():
            try:
                registry = get_teacher_registry()
                response = registry.ask_best_teacher(q)
                if response:
                    self.answer_question(q.question_id, response.answer_text,
                                        answered_by=response.teacher_id)
                    log.info("[ConfusionDetector] Auto-answered q=%s by %s (conf=%.2f, rules=%d)",
                             q.question_id, response.teacher_id,
                             response.confidence, len(response.suggested_rules))
                    # Promote suggested rules into concept registry
                    if response.suggested_rules:
                        self._promote_rules(response.suggested_rules, q.domain, response.confidence)
                else:
                    log.debug("[ConfusionDetector] No queryable teacher answered q=%s", q.question_id)
            except Exception as e:
                log.warning("[ConfusionDetector] Auto-answer failed: %s", e)
        threading.Thread(target=_do, daemon=True, name="auto-answer").start()

    def _promote_rules(self, rules: list, domain: str, confidence: float):
        """Push teacher-suggested rules into the concept registry."""
        try:
            from sare.memory.concept_formation import get_concept_registry
            registry = get_concept_registry()
            for rule in rules:
                name = rule.get("name", "") if isinstance(rule, dict) else str(rule)
                pattern = rule.get("pattern", name) if isinstance(rule, dict) else str(rule)
                if name:
                    registry.add_rule(name=name, domain=domain, pattern=pattern,
                                      confidence=min(confidence, 0.8), source="teacher")
                    log.info("[ConfusionDetector] Promoted teacher rule '%s' (domain=%s)", name, domain)
        except Exception as e:
            log.debug("[ConfusionDetector] Rule promotion failed: %s", e)

    def get_pending_questions(self) -> List[LearningQuestion]:
        return [q for q in self._pending_questions if q.status == "pending"]

    def answer_question(self, question_id: str, answer_text: str, answered_by: str):
        """Called when a teacher (human or LLM) provides an answer."""
        for q in self._pending_questions:
            if q.question_id == question_id and q.status == "pending":
                q.status = "answered"
                q.answer = answer_text
                q.answered_by = answered_by
                q.answered_at = time.time()
                self._answered_questions.append(q)
                self._save()
                return q
        return None

    def _save(self):
        try:
            _MEMORY.mkdir(parents=True, exist_ok=True)
            data = {
                "pending": [q.to_dict() for q in self._pending_questions[-50:]],
                "answered": [q.to_dict() for q in self._answered_questions[-100:]],
                "failure_counts": self._failure_counts,
            }
            tmp = self._persist_path.parent / f"{self._persist_path.stem}.{os.getpid()}.{threading.get_ident()}.tmp"
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self._persist_path)
        except OSError:
            pass

    def _load(self):
        if not self._persist_path.exists():
            return
        try:
            d = json.loads(self._persist_path.read_text())
            self._failure_counts = d.get("failure_counts", {})
            for qd in d.get("pending", []):
                q = LearningQuestion(**{k: v for k, v in qd.items()
                                       if k in LearningQuestion.__dataclass_fields__})
                self._pending_questions.append(q)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  TeacherRegistry
# ═══════════════════════════════════════════════════════════════════════════════

class TeacherRegistry:
    """
    Routes questions to the highest-trust queryable teacher.
    Updates trust scores after symbolic validation.
    """

    PERSIST_PATH = _MEMORY / "teacher_registry.json"

    def __init__(self):
        self._teachers: Dict[str, Teacher] = {}
        self._response_history: List[dict] = []
        # Register default teachers
        self.register(LLMTeacher("llm_default"))
        self.register(HumanTeacher("human_0"))
        self._load_trust_scores()

    def register(self, teacher: Teacher):
        self._teachers[teacher.teacher_id] = teacher
        log.info("[TeacherRegistry] Registered teacher '%s' (queryable=%s)",
                 teacher.teacher_id, teacher.is_queryable)

    def ask_best_teacher(self, question: LearningQuestion) -> Optional[TeacherResponse]:
        """Route to highest-trust queryable teacher for this domain."""
        candidates = [
            t for t in self._teachers.values()
            if t.is_queryable and t.can_answer(question)
        ]
        if not candidates:
            return None

        # Sort by trust score for this domain
        candidates.sort(key=lambda t: t.trust_score(question.domain), reverse=True)
        for teacher in candidates:
            try:
                response = teacher.ask(question)
                if response is not None:
                    self._response_history.append({
                        "teacher_id": teacher.teacher_id,
                        "question_id": question.question_id,
                        "timestamp": time.time(),
                    })
                    return response
            except Exception as e:
                log.debug("[TeacherRegistry] Teacher '%s' failed: %s", teacher.teacher_id, e)
        return None

    def update_trust(self, teacher_id: str, domain: str, correct: bool):
        teacher = self._teachers.get(teacher_id)
        if teacher:
            teacher.update_trust(domain, correct)
            self._save_trust_scores()

    def get_all_teachers(self) -> List[dict]:
        return [t.to_dict() for t in self._teachers.values()]

    def _save_trust_scores(self):
        try:
            _MEMORY.mkdir(parents=True, exist_ok=True)
            data = {
                tid: {"trust_scores": t._trust_scores, "total_answers": t.total_answers}
                for tid, t in self._teachers.items()
            }
            tmp = self.PERSIST_PATH.parent / f"{self.PERSIST_PATH.stem}.{os.getpid()}.{threading.get_ident()}.tmp"
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self.PERSIST_PATH)
        except OSError:
            pass

    def _load_trust_scores(self):
        if not self.PERSIST_PATH.exists():
            return
        try:
            data = json.loads(self.PERSIST_PATH.read_text())
            for tid, info in data.items():
                teacher = self._teachers.get(tid)
                if teacher:
                    teacher._trust_scores = info.get("trust_scores", {})
                    teacher.total_answers = info.get("total_answers", 0)
        except Exception:
            pass


# ── Helper functions ───────────────────────────────────────────────────────────

def _parse_rules_from_text(text: str, domain: str) -> List[dict]:
    """Extract RULE: pattern → result lines from LLM response."""
    import re
    rules = []
    for line in text.split("\n"):
        m = re.search(r"RULE\s*:\s*(.+?)\s*→\s*(.+)", line)
        if m:
            rules.append({
                "name":       f"teacher_rule_{len(rules)}",
                "pattern":    m.group(1).strip(),
                "result":     m.group(2).strip(),
                "domain":     domain,
                "confidence": 0.6,
                "source":     "teacher",
            })
    return rules[:3]   # cap at 3 suggestions per response


# ── Singletons ────────────────────────────────────────────────────────────────
_confusion_detector: Optional[ConfusionDetector] = None
_teacher_registry: Optional[TeacherRegistry] = None


def get_confusion_detector() -> ConfusionDetector:
    global _confusion_detector
    if _confusion_detector is None:
        _confusion_detector = ConfusionDetector()
    return _confusion_detector


def get_teacher_registry() -> TeacherRegistry:
    global _teacher_registry
    if _teacher_registry is None:
        _teacher_registry = TeacherRegistry()
    return _teacher_registry
