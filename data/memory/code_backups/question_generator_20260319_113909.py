"""
QuestionGenerator — SARE-HX generates its own investigation questions.

Watches for surprises, pattern gaps, and contradictions, then produces
self-directed questions that drive autonomous exploration. Questions are
injected into the curriculum as priority experiments.

Sources of questions:
  1. Surprise events (world model prediction error > 2.5)
  2. Rule gaps (induction returns None repeatedly on same domain)
  3. Analogy opportunities (similar structures in different domains)
  4. Contradiction detection (two rules seem to conflict)
"""

from __future__ import annotations

import json
import logging
import time
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

_MEMORY_DIR = Path(__file__).resolve().parents[3] / "data" / "memory"
_QUESTIONS_PATH = _MEMORY_DIR / "active_questions.json"


@dataclass
class Question:
    """A self-generated investigation question."""
    question_id: str
    text: str                # "Why does X always happen when Y?"
    source: str              # "surprise" | "gap" | "analogy" | "contradiction"
    domain: str
    priority: float          # 0-1
    generated_at: float = field(default_factory=time.time)
    investigated: bool = False
    investigation_result: str = ""

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "text": self.text,
            "source": self.source,
            "domain": self.domain,
            "priority": round(self.priority, 3),
            "generated_at": self.generated_at,
            "investigated": self.investigated,
            "investigation_result": self.investigation_result,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Question":
        return cls(
            question_id=d.get("question_id", str(uuid.uuid4())[:8]),
            text=d.get("text", ""),
            source=d.get("source", "gap"),
            domain=d.get("domain", "general"),
            priority=d.get("priority", 0.5),
            generated_at=d.get("generated_at", time.time()),
            investigated=d.get("investigated", False),
            investigation_result=d.get("investigation_result", ""),
        )


class QuestionGenerator:
    """
    Watches for surprises, patterns, and gaps, then generates investigation targets.
    """

    MAX_QUESTIONS = 100  # rolling window

    def __init__(self):
        self._questions: List[Question] = []
        self._lock = threading.RLock()
        self._last_generated: float = 0.0
        self._generation_interval = 300.0  # seconds
        self.load()

    # ── Generation ─────────────────────────────────────────────────────────────

    def generate_questions(self) -> List[Question]:
        """Scan current state and return newly generated questions."""
        new_questions: List[Question] = []

        # Source 1: World model surprises
        new_questions.extend(self._scan_world_model_surprises())

        # Source 2: Rule induction gaps
        new_questions.extend(self._scan_rule_gaps())

        # Source 3: Analogy opportunities
        new_questions.extend(self._scan_analogy_opportunities())

        # Source 4: Self-improver bottlenecks
        new_questions.extend(self._scan_bottleneck_gaps())

        with self._lock:
            # Avoid duplicates by checking similar text
            existing_texts = {q.text.lower()[:60] for q in self._questions}
            added = []
            for q in new_questions:
                if q.text.lower()[:60] not in existing_texts:
                    self._questions.append(q)
                    existing_texts.add(q.text.lower()[:60])
                    added.append(q)

            # Trim to MAX_QUESTIONS (keep highest priority)
            if len(self._questions) > self.MAX_QUESTIONS:
                self._questions.sort(key=lambda q: (-q.priority, q.generated_at))
                self._questions = self._questions[:self.MAX_QUESTIONS]

        self._last_generated = time.time()
        if added:
            self.save()
            log.info("[QuestionGenerator] Generated %d new questions", len(added))
        return added

    def _scan_world_model_surprises(self) -> List[Question]:
        """Generate questions from high-surprise world model events."""
        questions = []
        try:
            wm_path = _MEMORY_DIR / "world_model.json"
            if not wm_path.exists():
                return []
            with open(wm_path, encoding="utf-8") as f:
                wm = json.load(f)

            beliefs = wm.get("beliefs", {})
            hypotheses_path = _MEMORY_DIR / "world_hypotheses.json"
            hypotheses = []
            if hypotheses_path.exists():
                with open(hypotheses_path, encoding="utf-8") as f:
                    hypotheses = json.load(f)

            # If avg_surprise is high, ask what's causing it
            avg_surprise = wm.get("avg_surprise", 0.0)
            if avg_surprise > 2.5:
                q = Question(
                    question_id=str(uuid.uuid4())[:8],
                    text=f"Why is prediction error so high (surprise={avg_surprise:.2f})? What pattern am I missing?",
                    source="surprise",
                    domain="meta",
                    priority=min(1.0, avg_surprise / 5.0),
                )
                questions.append(q)

            # Questions from recent hypotheses
            for hyp in hypotheses[-3:]:
                if isinstance(hyp, dict) and not hyp.get("verified", False):
                    domain = hyp.get("domain", "general")
                    text = hyp.get("hypothesis", "")
                    if text:
                        q = Question(
                            question_id=str(uuid.uuid4())[:8],
                            text=f"Is this hypothesis true: '{text[:100]}'?",
                            source="surprise",
                            domain=domain,
                            priority=0.6,
                        )
                        questions.append(q)
        except Exception as e:
            log.debug("[QuestionGenerator] world model scan error: %s", e)
        return questions[:2]

    def _scan_rule_gaps(self) -> List[Question]:
        """Generate questions from induction failures."""
        questions = []
        try:
            synth_path = _MEMORY_DIR / "synth_attempts.json"
            if not synth_path.exists():
                return []
            with open(synth_path, encoding="utf-8") as f:
                attempts = json.load(f)

            # Find domains with many failed attempts
            domain_failures: dict = {}
            for attempt in attempts if isinstance(attempts, list) else []:
                domain = attempt.get("domain", "general")
                if not attempt.get("success", True):
                    domain_failures[domain] = domain_failures.get(domain, 0) + 1

            for domain, count in sorted(domain_failures.items(), key=lambda x: -x[1])[:2]:
                if count >= 3:
                    q = Question(
                        question_id=str(uuid.uuid4())[:8],
                        text=f"Why do I keep failing in '{domain}'? What rule am I missing?",
                        source="gap",
                        domain=domain,
                        priority=min(0.9, 0.4 + count * 0.1),
                    )
                    questions.append(q)
        except Exception as e:
            log.debug("[QuestionGenerator] rule gap scan error: %s", e)
        return questions

    def _scan_analogy_opportunities(self) -> List[Question]:
        """Generate questions from cross-domain transfer patterns."""
        questions = []
        try:
            transfer_path = _MEMORY_DIR / "learned_transfers.json"
            if not transfer_path.exists():
                return []
            with open(transfer_path, encoding="utf-8") as f:
                transfers = json.load(f)

            # Look for successful transfers that suggest unexplored analogies
            if isinstance(transfers, list):
                for t in transfers[-5:]:
                    src = t.get("source_domain", "")
                    tgt = t.get("target_domain", "")
                    if src and tgt and src != tgt:
                        q = Question(
                            question_id=str(uuid.uuid4())[:8],
                            text=f"What other rules from '{src}' might apply to '{tgt}'?",
                            source="analogy",
                            domain=tgt,
                            priority=0.55,
                        )
                        questions.append(q)
            elif isinstance(transfers, dict):
                for rule_name, info in list(transfers.items())[:3]:
                    domains = info.get("domains", []) if isinstance(info, dict) else []
                    if len(domains) >= 2:
                        q = Question(
                            question_id=str(uuid.uuid4())[:8],
                            text=f"Rule '{rule_name}' works across {domains}. Where else might it apply?",
                            source="analogy",
                            domain="general",
                            priority=0.5,
                        )
                        questions.append(q)
        except Exception as e:
            log.debug("[QuestionGenerator] analogy scan error: %s", e)
        return questions[:2]

    def _scan_bottleneck_gaps(self) -> List[Question]:
        """Generate questions from bottleneck analyzer report."""
        questions = []
        try:
            report_path = _MEMORY_DIR / "bottleneck_report.json"
            if not report_path.exists():
                return []
            with open(report_path, encoding="utf-8") as f:
                report = json.load(f)

            targets = report.get("targets", [])
            for t in targets[:2]:
                fname = t.get("file_path", "")
                reason = t.get("reason", "")
                if fname and reason:
                    q = Question(
                        question_id=str(uuid.uuid4())[:8],
                        text=f"How can I improve {Path(fname).name}? Issue: {reason[:80]}",
                        source="gap",
                        domain="meta",
                        priority=t.get("score", 0.4),
                    )
                    questions.append(q)
        except Exception as e:
            log.debug("[QuestionGenerator] bottleneck scan error: %s", e)
        return questions

    # ── Investigation ──────────────────────────────────────────────────────────

    def investigate(self, question: Question) -> None:
        """Mark a question as being investigated (curriculum will pick it up)."""
        with self._lock:
            for q in self._questions:
                if q.question_id == question.question_id:
                    q.investigated = True
                    break
        self.save()

    def mark_answered(self, question_id: str, result: str) -> bool:
        """Record the result of investigating a question."""
        with self._lock:
            for q in self._questions:
                if q.question_id == question_id:
                    q.investigated = True
                    q.investigation_result = result
                    self.save()
                    return True
        return False

    def get_pending_questions(self) -> List[Question]:
        """Return uninvestigated questions sorted by priority."""
        with self._lock:
            return sorted(
                [q for q in self._questions if not q.investigated],
                key=lambda q: -q.priority,
            )

    def get_all(self) -> List[dict]:
        with self._lock:
            return [q.to_dict() for q in sorted(self._questions, key=lambda q: -q.priority)]

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self):
        try:
            _QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = [q.to_dict() for q in self._questions]
            with open(_QUESTIONS_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            log.warning("[QuestionGenerator] save error: %s", e)

    def load(self):
        if not _QUESTIONS_PATH.exists():
            return
        try:
            with open(_QUESTIONS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            self._questions = [Question.from_dict(d) for d in data]
            log.info("[QuestionGenerator] Loaded %d questions", len(self._questions))
        except Exception as e:
            log.warning("[QuestionGenerator] load error: %s", e)


# ── Singleton ──────────────────────────────────────────────────────────────────

_QUESTION_GENERATOR: Optional[QuestionGenerator] = None


def get_question_generator() -> QuestionGenerator:
    global _QUESTION_GENERATOR
    if _QUESTION_GENERATOR is None:
        _QUESTION_GENERATOR = QuestionGenerator()
    return _QUESTION_GENERATOR
