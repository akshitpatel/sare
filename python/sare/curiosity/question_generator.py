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
from typing import List

log = logging.getLogger(__name__)

_MEMORY_DIR = Path(__file__).resolve().parents[3] / "data" / "memory"
_QUESTIONS_PATH = _MEMORY_DIR / "active_questions.json"
# Bottleneck signal produced by meta/bottleneck_analyzer.py
_BOTTLENECK_REPORT_PATH = _MEMORY_DIR / "bottleneck_report.json"
# Optional auxiliary stats (for more context when present)
_RUN_REPORT_PATH = _MEMORY_DIR / "run_report.json"


@dataclass
class Question:
    """A self-generated investigation question."""
    question_id: str
    text: str  # "Why does X always happen when Y?"
    source: str  # "surprise" | "gap" | "analogy" | "contradiction" | "bottleneck"
    domain: str
    priority: float  # 0-1
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
            priority=float(d.get("priority", 0.5)),
            generated_at=float(d.get("generated_at", time.time())),
            investigated=bool(d.get("investigated", False)),
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

    # ── Persistence ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        with self._lock:
            self._questions = []
            if not _QUESTIONS_PATH.exists():
                return
            try:
                with open(_QUESTIONS_PATH, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, list):
                    self._questions = [Question.from_dict(x) for x in raw if isinstance(x, dict)]
                elif isinstance(raw, dict) and "questions" in raw and isinstance(raw["questions"], list):
                    self._questions = [Question.from_dict(x) for x in raw["questions"] if isinstance(x, dict)]
            except Exception:
                log.exception("[QuestionGenerator] Failed to load %s", _QUESTIONS_PATH)
                self._questions = []

            # Keep bounded window by priority then recency
            if len(self._questions) > self.MAX_QUESTIONS:
                self._questions.sort(key=lambda q: (-q.priority, -q.generated_at))
                self._questions = self._questions[: self.MAX_QUESTIONS]

    def save(self) -> None:
        with self._lock:
            _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
            data = [q.to_dict() for q in self._questions]
            tmp_path = _QUESTIONS_PATH.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp_path.replace(_QUESTIONS_PATH)

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
            # Avoid duplicates by checking similar text prefix
            existing_texts = {q.text.lower()[:80] for q in self._questions}
            added: List[Question] = []
            for q in new_questions:
                k = q.text.lower()[:80]
                if k not in existing_texts and q.text.strip():
                    self._questions.append(q)
                    existing_texts.add(k)
                    added.append(q)

            # Trim to MAX_QUESTIONS (keep highest priority)
            if len(self._questions) > self.MAX_QUESTIONS:
                self._questions.sort(key=lambda q: (-q.priority, -q.generated_at))
                self._questions = self._questions[: self.MAX_QUESTIONS]

        self._last_generated = time.time()
        if "added" in locals() and added:
            self.save()
            log.info("[QuestionGenerator] Generated %d new questions", len(added))
            return added
        return []

    def _scan_world_model_surprises(self) -> List[Question]:
        """Generate questions from high-surprise world model events."""
        questions: List[Question] = []
        try:
            wm_path = _MEMORY_DIR / "world_model.json"
            if not wm_path.exists():
                return []
            with open(wm_path, encoding="utf-8") as f:
                wm = json.load(f)

            hypotheses_path = _MEMORY_DIR / "world_hypotheses.json"
            hypotheses = []
            if hypotheses_path.exists():
                with open(hypotheses_path, encoding="utf-8") as f:
                    hypotheses = json.load(f)

            # If avg_surprise is high, ask what's causing it
            avg_surprise = float(wm.get("avg_surprise", 0.0) or 0.0)
            if avg_surprise > 2.5:
                q = Question(
                    question_id=str(uuid.uuid4())[:8],
                    text=(
                        f"Why is prediction error so high (surprise={avg_surprise:.2f})? "
                        f"What consistent pattern am I failing to model?"
                    ),
                    source="surprise",
                    domain="meta",
                    priority=min(1.0, avg_surprise / 5.0),
                )
                questions.append(q)

            # Questions from recent hypotheses
            for hyp in hypotheses[-3:]:
                if isinstance(hyp, dict) and not hyp.get("verified", False):
                    domain = hyp.get("domain", "general") or "general"
                    text = str(hyp.get("hypothesis", "") or "")
                    if text:
                        q = Question(
                            question_id=str(uuid.uuid4())[:8],
                            text=f"Is this hypothesis true: '{text[:120]}'?",
                            source="surprise",
                            domain=domain if domain else "general",
                            priority=0.65,
                        )
                        questions.append(q)

        except Exception:
            log.exception("[QuestionGenerator] _scan_world_model_surprises failed")
        return questions

    def _scan_rule_gaps(self) -> List[Question]:
        """Generate questions from induction gaps (heuristic signal)."""
        questions: List[Question] = []
        try:
            # Induction failures are not guaranteed to exist in a single file;
            # use "self_model" or "run_report" if available as proxies.
            # Keep conservative: only ask if we have explicit stuck stats.
            stuck_path = _MEMORY_DIR / "synth_attempts.json"
            if stuck_path.exists():
                with open(stuck_path, encoding="utf-8") as f:
                    attempts = json.load(f)
                if isinstance(attempts, dict):
                    # Expected: {domain: {stuck_count: int, fail_count: int, ...}, ...}
                    for domain, stats in list(attempts.items())[:20]:
                        if not isinstance(stats, dict):
                            continue
                        stuck_count = int(stats.get("stuck_count", 0) or 0)
                        fail_count = int(stats.get("fail_count", 0) or 0)
                        if stuck_count >= 3 or (fail_count >= 10 and stuck_count >= 1):
                            domain_str = str(domain)
                            priority = min(1.0, 0.35 + (stuck_count * 0.08) + (fail_count * 0.01))
                            questions.append(
                                Question(
                                    question_id=str(uuid.uuid4())[:8],
                                    text=(
                                        f"Which rule(s) or transform(s) are missing for solving domain '{domain_str}'? "
                                        f"(stuck_count={stuck_count}, fail_count={fail_count})"
                                    ),
                                    source="gap",
                                    domain=domain_str if domain_str else "general",
                                    priority=priority,
                                )
                            )

        except Exception:
            log.exception("[QuestionGenerator] _scan_rule_gaps failed")
        return questions

    def _scan_analogy_opportunities(self) -> List[Question]:
        """Generate questions from analogy opportunities (heuristic signal)."""
        questions: List[Question] = []
        try:
            # Use learned transfers if present.
            transfers_path = _MEMORY_DIR / "learned_transfers.json"
            if not transfers_path.exists():
                return []
            with open(transfers_path, encoding="utf-8") as f:
                transfers = json.load(f)
            if isinstance(transfers, list):
                # Look for unverified / low-confidence transfers to investigate.
                for tr in transfers[-30:]:
                    if not isinstance(tr, dict):
                        continue
                    conf = float(tr.get("confidence", 0.0) or 0.0)
                    verified = bool(tr.get("verified", False))
                    if not verified and conf < 0.7:
                        src = str(tr.get("source_domain", "") or "source")
                        tgt = str(tr.get("target_domain", "") or "target")
                        questions.append(
                            Question(
                                question_id=str(uuid.uuid4())[:8],
                                text=(
                                    f"Can a structural rule learned in '{src}' be reliably transferred to '{tgt}'? "
                                    f"(current confidence={conf:.2f})"
                                ),
                                source="analogy",
                                domain=tgt if tgt else "general",
                                priority=min(1.0, 0.5 + (0.7 - conf)),
                            )
                        )
            elif isinstance(transfers, dict) and "transfers" in transfers and isinstance(transfers["transfers"], list):
                # Alternate structure
                for tr in transfers["transfers"][-30:]:
                    if not isinstance(tr, dict):
                        continue
                    conf = float(tr.get("confidence", 0.0) or 0.0)
                    verified = bool(tr.get("verified", False))
                    if not verified and conf < 0.7:
                        src = str(tr.get("source_domain", "") or "source")
                        tgt = str(tr.get("target_domain", "") or "target")
                        questions.append(
                            Question(
                                question_id=str(uuid.uuid4())[:8],
                                text=(
                                    f"Can a structural rule learned in '{src}' be reliably transferred to '{tgt}'? "
                                    f"(current confidence={conf:.2f})"
                                ),
                                source="analogy",
                                domain=tgt if tgt else "general",
                                priority=min(1.0, 0.5 + (0.7 - conf)),
                            )
                        )

        except Exception:
            log.exception("[QuestionGenerator] _scan_analogy_opportunities failed")
        return questions

    def _scan_bottleneck_gaps(self) -> List[Question]:
        """
        Generate questions from self-improvement bottleneck reports.

        This closes the loop:
          meta/bottleneck_analyzer.py -> data/memory/bottleneck_report.json -> curriculum question injection
        """
        questions: List[Question] = []
        try:
            if not _BOTTLENECK_REPORT_PATH.exists():
                return []

            with open(_BOTTLENECK_REPORT_PATH, encoding="utf-8") as f:
                report = json.load(f)

            # Expected shapes (tolerant):
            # 1) {"targets": [ {file, score, reason, domain, subsystem}, ... ]}
            # 2) {"bottlenecks": [...]}
            # 3) {"top_targets": [...]}
            targets = None
            if isinstance(report, dict):
                for key in ("targets", "bottlenecks", "top_targets"):
                    if key in report and isinstance(report[key], list):
                        targets = report[key]
                        break
            if targets is None:
                return []

            # Optional extra context for nicer text
            run_report = {}
            if _RUN_REPORT_PATH.exists():
                try:
                    with open(_RUN_REPORT_PATH, encoding="utf-8") as f:
                        run_report = json.load(f)
                except Exception:
                    run_report = {}

            # Only create a small number per scan to avoid spamming curriculum
            # Rank by descending score if available.
            def _score_of(t):
                if not isinstance(t, dict):
                    return 0.0
                s = t.get("score", t.get("priority", 0.0))
                try:
                    return float(s or 0.0)
                except Exception:
                    return 0.0

            sorted_targets = sorted(targets, key=_score_of, reverse=True)[:8]

            # Use already-investigated targets to avoid redundant questions.
            with self._lock:
                investigated_texts = {q.text.lower() for q in self._questions if q.investigated}

            for t in sorted_targets:
                if not isinstance(t, dict):
                    continue
                file_name = str(t.get("file", "") or t.get("target_file", "") or t.get("path", "") or "")
                domain = str(t.get("domain", "") or "general")
                subsystem = str(t.get("subsystem", "") or t.get("module", "") or "")
                reason = str(t.get("reason", "") or t.get("description", "") or "")
                raw_score = _score_of(t)

                # Normalize score to priority in 0..1
                # If score already looks like 0..1, keep; else map typical ranges.
                if raw_score <= 1.0:
                    priority = max(0.35, min(0.95, 0.35 + raw_score * 0.6))
                else:
                    priority = max(0.35, min(1.0, raw_score / 10.0))

                if not file_name and not reason:
                    continue

                # Craft question text
                if file_name and reason:
                    q_text = (
                        f"What specific failure mode causes high bottleneck in '{file_name}' "
                        f"(domain='{domain}'): {reason[:180]}?"
                    )
                elif file_name:
                    q_text = (
                        f"What failure mode is driving persistent difficulty in '{file_name}' "
                        f"(domain='{domain}')?"
                    )
                else:
                    q_text = (
                        f"What critical missing capability is blocking improvement in domain '{domain}'?"
                    )

                key = q_text.lower().strip()
                if any(key in it or it in key for it in investigated_texts):
                    continue

                questions.append(
                    Question(
                        question_id=str(uuid.uuid4())[:8],
                        text=q_text,
                        source="bottleneck",
                        domain=domain if domain else "general",
                        priority=priority,
                    )
                )

            # Keep capped, sorted by priority
            if len(questions) > 4:
                questions.sort(key=lambda q: (-q.priority, -q.generated_at))
                questions = questions[:4]

        except Exception:
            log.exception("[QuestionGenerator] _scan_bottleneck_gaps failed")
        return questions

    # ── Curriculum integration hooks ───────────────────────────────────────────

    def get_pending_questions(self) -> List[Question]:
        """Return non-investigated questions (highest priority first)."""
        with self._lock:
            pending = [q for q in self._questions if not q.investigated]
            pending.sort(key=lambda q: (-q.priority, q.generated_at))
            # Return at most a small window to avoid crowding curriculum.
            return pending[:20]

    def mark_answered(self, question_id: str, result: str) -> None:
        with self._lock:
            for q in self._questions:
                if q.question_id == question_id:
                    q.investigated = True
                    q.investigation_result = str(result or "")
                    break
            self._questions.sort(key=lambda q: (-q.priority, -q.generated_at))
            self.save()


def get_question_generator() -> QuestionGenerator:
    global _QUESTION_GENERATOR_SINGLETON  # type: ignore
    try:
        return _QUESTION_GENERATOR_SINGLETON
    except NameError:
        _QUESTION_GENERATOR_SINGLETON = QuestionGenerator()  # type: ignore
        return _QUESTION_GENERATOR_SINGLETON