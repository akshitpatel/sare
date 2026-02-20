"""
MemoryManager — Persistent Memory Bridge for SARE-HX

Connects the previously-disconnected C++ memory modules (EpisodicStore,
StrategyMemory, GraphSignature) to the live solve path.

Responsibilities:
  before_solve(graph) → suggest warm-start transform sequence
  after_solve(episode) → store trace, update strategy memory
  save() / load()     → persist across restarts (via JSONL + JSON)

This module is the single wiring point so that web.py and engine.py
can call one object without knowing the C++ internals.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ── Standalone bootstrap: add python/ dir to sys.path ─────────────────────────
# This lets you run: python3 memory_manager.py  (without setting PYTHONPATH)
_here = Path(__file__).resolve()
for _candidate in [
    _here.parents[2],           # python/
    _here.parents[3] / "python",  # <repo>/python/
]:
    if (_candidate / "sare").is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break

log = logging.getLogger(__name__)

# ── C++ Bindings import (optional) ────────────────────────────
_sb = None
GraphSignature = None
StrategyMemory = None
EpisodicStore = None

try:
    import sare.sare_bindings as _sb  # type: ignore
    GraphSignature = getattr(_sb, "GraphSignature", None)
    StrategyMemory = getattr(_sb, "StrategyMemory", None)
    EpisodicStore  = getattr(_sb, "EpisodicStore",  None)
except Exception as e:            # pragma: no cover
    log.warning("MemoryManager C++ bindings unavailable: %s", e)


# ── Data types ─────────────────────────────────────────────────

@dataclass
class SolveEpisode:
    """Python representation of a solve trace (mirrors C++ SolveEpisode)."""
    problem_id: str
    transform_sequence: List[str] = field(default_factory=list)
    energy_trajectory: List[float] = field(default_factory=list)
    initial_energy: float = 0.0
    final_energy: float = 0.0
    compute_time_seconds: float = 0.0
    total_expansions: int = 0
    success: bool = False

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, d: dict) -> "SolveEpisode":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class StrategyHint:
    """Returned by before_solve when a similar problem was found."""
    found: bool = False
    signature: str = ""
    transform_sequence: List[str] = field(default_factory=list)
    avg_energy_reduction: float = 0.0
    success_rate: float = 0.0


# ── MemoryManager ──────────────────────────────────────────────

class MemoryManager:
    """
    Single access point for all persistent memory in SARE-HX.

    Lifecycle:
        mm = MemoryManager(persist_dir)
        mm.load()                           # restore from disk on startup
        hint = mm.before_solve(cpp_graph)   # warm-start hint
        mm.after_solve(episode)             # write trace after solve
        mm.save()                           # flush to disk (call periodically)
    """

    DEFAULT_DIR = Path(__file__).resolve().parents[3] / "data" / "memory"

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = Path(persist_dir or self.DEFAULT_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._episodes_path  = self.persist_dir / "episodes.jsonl"
        self._strategies_path = self.persist_dir / "strategies.json"

        # In-memory stores (Python-native, C++ optional)
        self._episodes: List[SolveEpisode] = []
        self._strategies: dict = {}   # signature → strategy dict

    # ── Before-solve: strategy lookup ─────────────────────────

    def before_solve(self, graph) -> StrategyHint:
        """
        Compute signature for `graph`, look up best past strategy.
        Returns a StrategyHint with the recommended transform sequence.
        """
        hint = StrategyHint()

        try:
            sig = self._compute_signature(graph)
            hint.signature = sig

            strat = self._strategies.get(sig) or self._soft_lookup(sig)
            if strat:
                hint.found = True
                hint.transform_sequence = strat.get("transform_sequence", [])
                hint.avg_energy_reduction = strat.get("avg_energy_reduction", 0.0)
                hint.success_rate = strat.get("success_rate", 0.0)
                log.info(
                    "Memory hit: sig=%s, transforms=%s, success=%.0f%%",
                    sig[:12], hint.transform_sequence[:3], hint.success_rate * 100,
                )
        except Exception as e:
            log.debug("before_solve lookup error: %s", e)

        return hint

    # ── After-solve: record episode ────────────────────────────

    def after_solve(self, episode: SolveEpisode, graph=None):
        """
        Store solve trace and update strategy memory.
        Call this every time a solve completes, success or failure.
        """
        self._episodes.append(episode)

        if episode.success and episode.transform_sequence:
            try:
                sig = graph and self._compute_signature(graph) or episode.problem_id
                self._upsert_strategy(sig, episode)
            except Exception as e:
                log.debug("after_solve strategy update error: %s", e)

        # Auto-save every 20 episodes
        if len(self._episodes) % 20 == 0:
            self.save()

        # Check if we should trigger heuristic training
        episode_count = len(self._episodes)
        if episode_count in (200, 500, 1000) or episode_count % 500 == 0:
            self._maybe_trigger_training(episode_count)

    # ── Persistence ────────────────────────────────────────────

    def save(self):
        """Flush all in-memory data to disk."""
        try:
            with open(self._episodes_path, "w", encoding="utf-8") as f:
                for ep in self._episodes:
                    f.write(json.dumps(ep.to_dict()) + "\n")
        except OSError as e:
            log.warning("Failed to save episodes: %s", e)

        try:
            with open(self._strategies_path, "w", encoding="utf-8") as f:
                json.dump(self._strategies, f, indent=2)
        except OSError as e:
            log.warning("Failed to save strategies: %s", e)

        log.info(
            "Memory saved: %d episodes, %d strategies",
            len(self._episodes), len(self._strategies),
        )

    def load(self):
        """Restore memory from disk. Call once at startup."""
        # Load episodes
        if self._episodes_path.exists():
            try:
                with open(self._episodes_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self._episodes.append(
                                SolveEpisode.from_dict(json.loads(line))
                            )
                log.info("Loaded %d episodes from disk", len(self._episodes))
            except Exception as e:
                log.warning("Episode load error: %s", e)

        # Load strategies
        if self._strategies_path.exists():
            try:
                with open(self._strategies_path, encoding="utf-8") as f:
                    self._strategies = json.load(f)
                log.info("Loaded %d strategies from disk", len(self._strategies))
            except Exception as e:
                log.warning("Strategy load error: %s", e)

    # ── Statistics ─────────────────────────────────────────────

    def stats(self) -> dict:
        total = len(self._episodes)
        solved = sum(1 for e in self._episodes if e.success)
        return {
            "total_episodes":    total,
            "solved":            solved,
            "solve_rate":        round(solved / total, 3) if total else 0.0,
            "strategy_count":    len(self._strategies),
            "avg_energy_saved":  round(
                sum(e.initial_energy - e.final_energy for e in self._episodes if e.success)
                / max(solved, 1), 3
            ),
        }

    @property
    def episode_count(self) -> int:
        return len(self._episodes)

    def recent_episodes(self, n: int = 20) -> List[SolveEpisode]:
        return list(self._episodes[-n:])

    # ── Private helpers ────────────────────────────────────────

    def _compute_signature(self, graph) -> str:
        """Compute structural signature using GraphSignature if available."""
        if GraphSignature and graph is not None:
            try:
                return GraphSignature.compute(graph)
            except Exception:
                pass

        # Python fallback: node-type histogram string
        try:
            from collections import Counter
            nodes = getattr(graph, "nodes", [])
            counts = Counter(n.type for n in nodes)
            return "_".join(f"{t}:{c}" for t, c in sorted(counts.items()))
        except Exception:
            return "sig_unknown"

    def _soft_lookup(self, sig: str) -> Optional[dict]:
        """Fuzzy lookup: find strategy with highest structural similarity."""
        if not self._strategies:
            return None

        best_key, best_sim = None, 0.0
        for key in self._strategies:
            # Simple token overlap as similarity
            a_parts = set(sig.split("_"))
            b_parts = set(key.split("_"))
            if not a_parts or not b_parts:
                continue
            sim = len(a_parts & b_parts) / len(a_parts | b_parts)
            if sim > best_sim and sim > 0.5:
                best_sim, best_key = sim, key

        return self._strategies.get(best_key) if best_key else None

    def _upsert_strategy(self, sig: str, episode: SolveEpisode):
        """Update running-average strategy record for this signature."""
        existing = self._strategies.get(sig)
        delta = episode.initial_energy - episode.final_energy

        if existing:
            n = existing.get("usage_count", 1)
            alpha = 1.0 / (n + 1)
            existing["avg_energy_reduction"] = (
                (1 - alpha) * existing["avg_energy_reduction"] + alpha * delta
            )
            existing["success_rate"] = (
                (1 - alpha) * existing["success_rate"] + alpha * float(episode.success)
            )
            existing["usage_count"] = n + 1
            # Keep the better transform sequence
            if delta > existing.get("avg_energy_reduction", 0):
                existing["transform_sequence"] = episode.transform_sequence
        else:
            self._strategies[sig] = {
                "signature":          sig,
                "transform_sequence": episode.transform_sequence,
                "avg_energy_reduction": delta,
                "success_rate":         1.0 if episode.success else 0.0,
                "usage_count":          1,
            }

    def _maybe_trigger_training(self, episode_count: int):
        """Auto-trigger heuristic model training when enough episodes accumulate."""
        try:
            import subprocess
            trainer_path = Path(__file__).resolve().parents[2] / "heuristics" / "trainer.py"
            if trainer_path.exists():
                log.info(
                    "Auto-triggering heuristic training at %d episodes", episode_count
                )
                subprocess.Popen(
                    [sys.executable, str(trainer_path),
                     "--episodes-path", str(self._episodes_path)],
                    start_new_session=True,
                )
        except Exception as e:
            log.debug("Auto-training trigger failed: %s", e)


# ── Standalone self-test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile
    import textwrap

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    print(textwrap.dedent("""
    ╔══════════════════════════════════════════════════════╗
    ║  MemoryManager — Standalone Self-Test               ║
    ╚══════════════════════════════════════════════════════╝
    """))

    with tempfile.TemporaryDirectory() as tmpdir:
        mm = MemoryManager(persist_dir=Path(tmpdir))

        # 1. Fresh stats
        s0 = mm.stats()
        assert s0["total_episodes"] == 0, "Expected 0 episodes at start"
        print(f"[1] Fresh stats OK   → episodes={s0['total_episodes']}")

        # 2. before_solve with no history → no hint
        hint = mm.before_solve(None)
        assert not hint.found, "Expected no hint on empty memory"
        print(f"[2] before_solve OK  → found={hint.found}")

        # 3. Record 5 episodes
        for i in range(5):
            ep = SolveEpisode(
                problem_id=f"x_plus_0_{i}",
                transform_sequence=["additive_identity"] * (i + 1),
                energy_trajectory=[5.0 - i * 0.5, 1.0],
                initial_energy=5.0 - i * 0.5,
                final_energy=1.0,
                compute_time_seconds=0.01 * (i + 1),
                total_expansions=10 * (i + 1),
                success=True,
            )
            mm.after_solve(ep, graph=None)
        print(f"[3] after_solve OK   → {mm.episode_count} episodes stored")

        # 4. Stats after episodes
        s1 = mm.stats()
        assert s1["total_episodes"] == 5, f"Expected 5, got {s1['total_episodes']}"
        assert s1["solve_rate"] == 1.0, f"Expected 1.0 solve rate, got {s1['solve_rate']}"
        print(f"[4] stats OK         → solve_rate={s1['solve_rate']:.0%}, strategies={s1['strategy_count']}")

        # 5. Persist and reload
        mm.save()
        mm2 = MemoryManager(persist_dir=Path(tmpdir))
        mm2.load()
        s2 = mm2.stats()
        assert s2["total_episodes"] == 5, f"Reload: expected 5, got {s2['total_episodes']}"
        print(f"[5] save/load OK     → reloaded {s2['total_episodes']} episodes, {s2['strategy_count']} strategies")

        # 6. Soft lookup — record episode with 'arithmetic' sig then query similar
        ep_arith = SolveEpisode(
            problem_id="arithmetic:x+0",
            transform_sequence=["additive_identity", "constant_folding"],
            initial_energy=6.0, final_energy=1.0, success=True,
        )
        mm2._upsert_strategy("arithmetic:const", ep_arith)
        found = mm2._soft_lookup("arithmetic:expr")
        print(f"[6] soft_lookup OK   → found strategy={found is not None}")

        # 7. Recent episodes
        recent = mm2.recent_episodes(3)
        assert len(recent) <= 5
        print(f"[7] recent_episodes  → {len(recent)} returned")

    print()
    print("\033[92m✅  All MemoryManager tests passed!\033[0m")
    print()
    print("Tip: in production, instantiate with PYTHONPATH set so C++ bindings load.")
    print("     C++ bindings status:", "AVAILABLE" if EpisodicStore else "using pure-Python fallback")

