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
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


class ProblemStatus(str, Enum):
    PENDING  = "pending"
    SOLVED   = "solved"
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
            "problem_id":    self.problem_id,
            "expression":    self.expression,
            "domain":        self.domain,
            "difficulty":    self.difficulty,
            "status":        self.status.value,
            "attempts":      self.attempts,
            "initial_energy": self.initial_energy,
            "best_delta":    self.best_delta,
            "timestamp":     self.timestamp,
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

    # ── Core API ───────────────────────────────────────────────

    def add_pending(self, problem_id: str, expression: str,
                    domain: str = "general", initial_energy: float = 0.0) -> FrontierProblem:
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

    def record(self, problem_id: str, success: bool,
               delta: float, num_transforms: int = 0):
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
            ProblemStatus.SOLVED   if success   else
            ProblemStatus.UNSOLVED if fp.attempts >= 3 else
            ProblemStatus.PENDING
        )
        fp.timestamp = time.time()

    def sample_near_frontier(self, n: int = 5, domain: Optional[str] = None) -> List[FrontierProblem]:
        """
        Sample problems near the skill frontier — slightly harder than current mastery.
        Prioritizes UNSOLVED problems with difficulty near the domain's frontier_difficulty.
        """
        stats = self.domain_stats()
        domain_frontier = {}
        for ds in stats.values():
            domain_frontier[ds.domain] = ds.frontier_difficulty

        # Score each problem by how close it is to the frontier
        candidates = [
            fp for fp in self._problems.values()
            if fp.status == ProblemStatus.UNSOLVED
            and (domain is None or fp.domain == domain)
            and fp.attempts < 10
        ]

        if not candidates:
            # Fall back to pending problems
            candidates = [
                fp for fp in self._problems.values()
                if fp.status == ProblemStatus.PENDING
                and (domain is None or fp.domain == domain)
            ]

        target_diff = domain_frontier.get(domain or "general", 0.5)
        scored = sorted(
            candidates,
            key=lambda fp: abs(fp.difficulty - target_diff)
        )
        return scored[:n]

    def domain_stats(self) -> Dict[str, DomainStats]:
        """Compute per-domain statistics."""
        stats: Dict[str, DomainStats] = defaultdict(lambda: DomainStats(domain=""))
        for fp in self._problems.values():
            ds = stats[fp.domain]
            ds.domain = fp.domain
            ds.total += 1
            if fp.status == ProblemStatus.SOLVED:
                ds.solved += 1
                ds.avg_delta = (ds.avg_delta * (ds.solved - 1) + fp.best_delta) / ds.solved
            elif fp.status == ProblemStatus.UNSOLVED:
                ds.unsolved += 1
            ds.avg_difficulty = (
                (ds.avg_difficulty * (ds.total - 1) + fp.difficulty) / ds.total
            )
        return dict(stats)

    # ── Statistics ─────────────────────────────────────────────

    def stats(self) -> dict:
        total = len(self._problems)
        solved = sum(1 for fp in self._problems.values() if fp.status == ProblemStatus.SOLVED)
        unsolved = sum(1 for fp in self._problems.values() if fp.status == ProblemStatus.UNSOLVED)
        return {
            "total":    total,
            "solved":   solved,
            "unsolved": unsolved,
            "pending":  total - solved - unsolved,
            "solve_rate": round(solved / total, 3) if total else 0.0,
            "domains":  {d: vars(ds) for d, ds in self.domain_stats().items()},
        }

    @property
    def size(self) -> int:
        return len(self._problems)

    # ── Persistence ────────────────────────────────────────────

    def save(self):
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                for fp in self._problems.values():
                    f.write(json.dumps(fp.to_dict()) + "\n")
        except OSError as e:
            log.warning("FrontierManager save error: %s", e)

    def load(self):
        if not self._path.exists():
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        fp = FrontierProblem.from_dict(json.loads(line))
                        self._insert(fp)
            log.info("FrontierManager loaded %d problems", len(self._problems))
        except Exception as e:
            log.warning("FrontierManager load error: %s", e)

    # ── Private helpers ────────────────────────────────────────

    def _insert(self, fp: FrontierProblem):
        """Insert with ring-buffer eviction."""
        if fp.problem_id not in self._problems:
            if len(self._problems) >= self._max_size:
                oldest = self._order.pop(0)
                self._problems.pop(oldest, None)
            self._order.append(fp.problem_id)
        self._problems[fp.problem_id] = fp

    @staticmethod
    def _estimate_difficulty(initial_energy: float) -> float:
        """Map raw energy to a 0-1 difficulty estimate."""
        # Energy range: ~1.0 (trivial) to ~20.0 (very hard)
        return min(1.0, max(0.0, (initial_energy - 1.0) / 19.0))

    @staticmethod
    def _infer_domain(expr: str) -> str:
        expr = expr.lower()
        if any(op in expr for op in ("not", "and", "or", "true", "false", "¬", "∧", "∨")):
            return "logic"
        if any(op in expr for op in ("+", "-", "*", "/", "^")):
            return "arithmetic"
        return "general"
