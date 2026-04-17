"""
FrontierManager — TODO-04 Implementation

Tracks the boundary between solved and unsolved problems.
This is the system's internal map of "what I can and cannot do."

The frontier is a ring-buffer of problems classified as:
  - SOLVED   : energy reduced by > threshold
  - UNSOLVED : tried, failed
  - PENDING  : generated, not yet attempted

The CurriculumGenerator should draw NEW problems from near the frontier
(slightly harder than current frontier) rather than random mutations.
The SelfModel reads the frontier to estimate per-domain competence.

Data flow:
  ExperimentRunner ──after solve──► FrontierManager.record(problem, result)
  CurriculumGenerator ──queries──► FrontierManager.sample_near_frontier()
  SelfModel ──queries──► FrontierManager.domain_stats()
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


class ProblemStatus(str, Enum):
    PENDING = "pending"
    SOLVED = "solved"
    UNSOLVED = "unsolved"


@dataclass
class FrontierProblem:
    """A problem on the learning frontier."""
    problem_id: str
    expression: str
    domain: str            # "arithmetic", "logic", "general"
    difficulty: float      # 0.0 (easy) → 1.0 (hard) — estimated from energy
    status: ProblemStatus = ProblemStatus.PENDING
    attempts: int = 0
    initial_energy: float = 0.0
    best_delta: float = 0.0     # best energy reduction achieved
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "expression": self.expression,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "status": self.status.value,
            "attempts": self.attempts,
            "initial_energy": self.initial_energy,
            "best_delta": self.best_delta,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FrontierProblem":
        return cls(
            problem_id=d["problem_id"],
            expression=d.get("expression", ""),
            domain=d.get("domain", "general"),
            difficulty=d.get("difficulty", 0.5),
            status=ProblemStatus(d.get("status", "pending")),
            attempts=d.get("attempts", 0),
            initial_energy=d.get("initial_energy", 0.0),
            best_delta=d.get("best_delta", 0.0),
            timestamp=d.get("timestamp", 0.0),
        )


@dataclass
class DomainStats:
    domain: str
    total: int = 0
    solved: int = 0
    unsolved: int = 0
    avg_difficulty: float = 0.0
    avg_delta: float = 0.0

    @property
    def solve_rate(self) -> float:
        return self.solved / self.total if self.total > 0 else 0.0

    @property
    def frontier_difficulty(self) -> float:
        """Estimated difficulty of the current skill frontier."""
        if self.solve_rate < 0.2:
            return max(0.0, self.avg_difficulty - 0.2)  # back off — too hard
        elif self.solve_rate > 0.8:
            return min(1.0, self.avg_difficulty + 0.1)  # push harder
        return self.avg_difficulty  # in the zone


class FrontierManager:
    """
    Tracks the boundary between what SARE can and cannot solve.

    Design: ring-buffer with max_size entries. Older entries are evicted
    when the buffer fills. This keeps memory bounded while maintaining
    a fresh picture of the current skill boundary.
    """

    DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "frontier.jsonl"

    def __init__(self, persist_path: Optional[Path] = None, max_size: int = 2000):
        self._path = Path(persist_path or self.DEFAULT_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max_size = max_size
        self._problems: Dict[str, FrontierProblem] = {}  # problem_id → FrontierProblem
        self._order: List[str] = []  # insertion order for ring-buffer eviction
        self._dirty = False

        self._load()

    # ── Persistence ─────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        fp = FrontierProblem.from_dict(d)
                        if fp.problem_id in self._problems:
                            continue
                        self._problems[fp.problem_id] = fp
                        self._order.append(fp.problem_id)
                    except Exception:
                        continue
        except Exception as e:
            log.warning("FrontierManager: failed to load %s: %s", self._path, e)

        self._evict_if_needed()

    def _save(self) -> None:
        if not self._dirty:
            return
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                for pid in self._order:
                    fp = self._problems.get(pid)
                    if not fp:
                        continue
                    f.write(json.dumps(fp.to_dict(), ensure_ascii=False) + "\n")
            tmp_path.replace(self._path)
            self._dirty = False
        except Exception as e:
            log.warning("FrontierManager: failed to save %s: %s", tmp_path, e)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    def shutdown(self) -> None:
        self._save()

    # ── Utilities ───────────────────────────────────────────────

    def _estimate_difficulty(self, initial_energy: float) -> float:
        if initial_energy <= 0:
            return 0.5
        # Map initial energy to [0,1] with soft saturation.
        # Higher energy indicates harder initial state.
        # Clamp to avoid extreme values.
        x = max(0.0, min(100.0, float(initial_energy)))
        return max(0.0, min(1.0, x / (x + 10.0)))

    def _infer_domain(self, problem_id: str) -> str:
        s = problem_id.lower()
        if any(k in s for k in ("logic", "bool", "and", "or", "neg")):
            return "logic"
        if any(k in s for k in ("calc", "deriv", "integral", "x^", "dx", "dy")):
            return "calculus"
        if any(k in s for k in ("algebra", "factor", "expand", "solve")):
            return "algebra"
        if any(k in s for k in ("arith", "add", "mul", "sum", "prod")):
            return "arithmetic"
        return "general"

    def _insert(self, fp: FrontierProblem) -> None:
        if fp.problem_id in self._problems:
            return
        self._problems[fp.problem_id] = fp
        self._order.append(fp.problem_id)
        self._dirty = True
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        while len(self._order) > self._max_size:
            oldest = self._order.pop(0)
            self._problems.pop(oldest, None)
            self._dirty = True

    def _domain_problems(self, domain: Optional[str]) -> List[FrontierProblem]:
        if domain is None:
            return [self._problems[pid] for pid in self._order if pid in self._problems]
        return [fp for fp in self._problems.values() if fp.domain == domain]

    # ── Core API ───────────────────────────────────────────────

    def add_pending(
        self,
        problem_id: str,
        expression: str,
        domain: str = "general",
        initial_energy: float = 0.0
    ) -> FrontierProblem:
        """Register a newly generated problem before attempting it."""
        if problem_id in self._problems:
            return self._problems[problem_id]

        difficulty = self._estimate_difficulty(initial_energy)
        fp = FrontierProblem(
            problem_id=problem_id,
            expression=expression,
            domain=domain,
            difficulty=difficulty,
            initial_energy=initial_energy,
        )
        self._insert(fp)
        return fp

    def record(
        self,
        problem_id: str,
        success: bool,
        delta: float,
        num_transforms: int = 0
    ):
        """Update a problem's status after a solve attempt."""
        fp = self._problems.get(problem_id)
        if not fp:
            # Auto-create if not pre-registered
            fp = FrontierProblem(
                problem_id=problem_id,
                expression=problem_id,
                domain=self._infer_domain(problem_id),
                difficulty=0.5,
            )
            self._insert(fp)

        fp.attempts += 1
        fp.best_delta = max(fp.best_delta, delta)
        fp.status = (
            ProblemStatus.SOLVED if success
            else ProblemStatus.UNSOLVED if fp.attempts >= 3
            else ProblemStatus.PENDING
        )
        fp.timestamp = time.time()
        self._dirty = True

    def domain_stats(self, domain: Optional[str] = None) -> Dict[str, DomainStats]:
        """Compute per-domain stats; if domain is provided, return only that domain."""
        per_domain: Dict[str, DomainStats] = {}
        for fp in self._problems.values():
            d = fp.domain
            if d not in per_domain:
                per_domain[d] = DomainStats(domain=d)
            st = per_domain[d]
            st.total += 1
            if fp.status == ProblemStatus.SOLVED:
                st.solved += 1
            elif fp.status == ProblemStatus.UNSOLVED:
                st.unsolved += 1
            st.avg_difficulty += fp.difficulty
            st.avg_delta += fp.best_delta

        for st in per_domain.values():
            if st.total > 0:
                st.avg_difficulty /= st.total
                st.avg_delta /= st.total

        if domain is not None:
            return {domain: per_domain.get(domain, DomainStats(domain=domain))}
        return per_domain

    def sample_near_frontier(self, n: int = 5, domain: Optional[str] = None) -> List[FrontierProblem]:
        """
        Sample problems near the skill frontier — slightly harder than current mastery.
        Prioritizes UNSOLVED problems with difficulty near the domain's frontier_difficulty.
        """
        stats_map = self.domain_stats(domain)
        chosen_domain_stats = None
        if domain is not None:
            chosen_domain_stats = stats_map.get(domain) or DomainStats(domain=domain)

        candidates: List[FrontierProblem] = []
        if domain is None:
            # Combine across domains but keep difficulty filtering per-domain.
            for fp in self._problems.values():
                # Prefer unsolved near frontier
                st = stats_map.get(fp.domain)
                if not st or st.total == 0:
                    continue
                if fp.status == ProblemStatus.SOLVED:
                    continue
                # slight harder bias
                if abs(fp.difficulty - st.frontier_difficulty) <= 0.25 or fp.attempts == 0:
                    candidates.append(fp)
        else:
            st = chosen_domain_stats or stats_map.get(domain) or DomainStats(domain=domain)
            for fp in self._problems.values():
                if fp.domain != domain:
                    continue
                if fp.status == ProblemStatus.SOLVED:
                    continue
                if abs(fp.difficulty - st.frontier_difficulty) <= 0.25 or fp.attempts == 0:
                    candidates.append(fp)

        if not candidates:
            # Fallback: sample any pending/unsolved problem (or create a soft random target).
            all_candidates = [fp for fp in self._problems.values() if fp.status != ProblemStatus.SOLVED]
            if domain is not None:
                all_candidates = [fp for fp in all_candidates if fp.domain == domain]
            random.shuffle(all_candidates)
            return all_candidates[:n]

        def score(fp: FrontierProblem) -> float:
            # Higher score = more likely to sample
            if domain is None:
                st = stats_map.get(fp.domain)
            else:
                st = chosen_domain_stats
            if not st:
                st = DomainStats(domain=fp.domain)
            # closeness to frontier + stuckness
            closeness = 1.0 - min(1.0, abs(fp.difficulty - (st.frontier_difficulty + 0.05)) / 0.5)
            attempts_penalty = 1.0 / (1.0 + fp.attempts)
            status_boost = 1.0 if fp.status == ProblemStatus.UNSOLVED else 0.6
            # Prioritize harder-to-solve (high difficulty near frontier) but not hopeless
            return max(0.0, closeness) * status_boost + 0.15 * attempts_penalty

        # Weighted sampling without replacement
        scored = [(score(fp), fp) for fp in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [fp for _, fp in scored[:max(n * 4, n)]]

        if len(top) <= n:
            return top[:n]

        weights = [score(fp) + 1e-6 for fp in top]
        weights_sum = sum(weights)
        if weights_sum <= 0:
            random.shuffle(top)
            return top[:n]

        selected: List[FrontierProblem] = []
        pool = top[:]
        wpool = weights[:]
        for _ in range(min(n, len(pool))):
            wsum = sum(wpool)
            if wsum <= 0:
                break
            r = random.random() * wsum
            acc = 0.0
            idx = 0
            for i, w in enumerate(wpool):
                acc += w
                if acc >= r:
                    idx = i
                    break
            selected.append(pool.pop(idx))
            wpool.pop(idx)

        # If underfilled, fill deterministically from top scores
        if len(selected) < n:
            for fp in top:
                if fp not in selected:
                    selected.append(fp)
                    if len(selected) >= n:
                        break

        return selected[:n]

    # ── Approved Change: get_stuck_problems ──────────────────────

    def get_stuck_problems(
        self,
        n: int = 5,
        domain: Optional[str] = None,
        min_attempts: int = 3,
        min_best_delta: float = 0.0,
        max_avg_age_seconds: Optional[float] = None,
    ) -> List[FrontierProblem]:
        """
        Return problems that appear "stuck": many attempts but little progress.

        Selection heuristics:
          - Prefer non-solved problems (PENDING/UNSOLVED), within attempt threshold.
          - Prefer low best_delta (little energy reduction achieved so far).
          - Prefer high difficulty when best_delta is low (hard-but-not-learning).
          - Prefer recency (newer timestamps) unless max_avg_age_seconds is set.
          - Optionally filter by minimum best_delta (e.g., exclude already-progressing).
        """
        now = time.time()
        candidates = []
        for fp in self._problems.values():
            if domain is not None and fp.domain != domain:
                continue
            if fp.status == ProblemStatus.SOLVED:
                continue
            if fp.attempts < min_attempts:
                continue
            if fp.best_delta > min_best_delta:
                # If caller sets min_best_delta > 0, filter out problems with progress beyond threshold.
                continue
            if max_avg_age_seconds is not None:
                age = max(0.0, now - fp.timestamp)
                if age > max_avg_age_seconds:
                    continue
            candidates.append(fp)

        if not candidates:
            return []

        # Score: higher = more stuck (higher attempts, lower best_delta, high difficulty, recent)
        def stuck_score(fp: FrontierProblem) -> float:
            # Attempts saturation
            att = fp.attempts
            attempt_term = min(1.0, att / max(1.0, float(min_attempts)))

            # Progress term (lower best_delta => more stuck). Clamp.
            # best_delta can be negative in some energy formulations; treat <=0 as worst.
            delta = fp.best_delta
            if delta <= 0:
                progress_term = 1.0
            else:
                # Map delta to (0,1) progress using soft saturation
                progress_term = max(0.0, 1.0 - (delta / (delta + 0.75)))

            # Difficulty term: harder problems being stuck is more informative
            diff_term = max(0.0, min(1.0, fp.difficulty))

            # Recency term: exponential decay with ~3 days half-life by default
            age = max(0.0, now - fp.timestamp)
            half_life = 60.0 * 60.0 * 24.0 * 3.0
            recency_term = 2.0 ** (-age / half_life) if half_life > 0 else 1.0

            # Combine
            return 0.45 * attempt_term + 0.40 * progress_term + 0.15 * diff_term + 0.10 * recency_term

        candidates.sort(key=stuck_score, reverse=True)
        return candidates[:n] if n > 0 else []

    # ── Convenience ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "max_size": self._max_size,
            "count": len(self._problems),
            "problems": [self._problems[pid].to_dict() for pid in self._order if pid in self._problems],
        }

    def __len__(self) -> int:
        return len(self._problems)