"""
Self-Learning Daemon — Autonomous continuous learning for SARE-HX.

Uses the Brain orchestrator to run the full cognitive loop:
  1. Pick problem from developmental curriculum (based on stage + ZPD)
  2. Solve it (BeamSearch / MCTS)
  3. Store episode, update memory
  4. Reflect on successes (extract abstract rules)
  5. Transfer rules across domains
  6. Sleep-consolidate periodically (GNN training, macro mining)
  7. Advance developmental stage when ready

This replaces the old learn_daemon.py with a cleaner, Brain-integrated design.

Usage:
    python -m sare.self_learner [--cycles 100] [--interval 2] [--verbose]
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

log = logging.getLogger("sare.self_learner")


class SelfLearner:
    """
    Autonomous learning daemon.

    Modes:
      - batch: Run N cycles then stop
      - continuous: Run until interrupted
      - sleep_wake: Alternate between active learning and sleep consolidation
    """

    def __init__(self, brain=None):
        self.brain = brain
        self._running = False
        self._stats = {
            "cycles_completed": 0,
            "total_solves": 0,
            "total_successes": 0,
            "rules_learned": 0,
            "domains_mastered": 0,
            "sleep_cycles": 0,
            "started_at": 0,
        }

    def boot(self):
        """Initialize the brain if not provided."""
        if self.brain is None:
            from sare.brain import Brain
            self.brain = Brain()
            self.brain.boot()

    def run_batch(self, n_cycles: int = 10, problems_per_cycle: int = 5,
                  sleep_every: int = 5) -> dict:
        """
        Run N learning cycles, sleeping periodically.

        Each cycle:
          1. Pick problems from curriculum
          2. Solve them
          3. Reflect and learn
          4. Every `sleep_every` cycles, run sleep consolidation
        """
        self.boot()
        self._running = True
        self._stats["started_at"] = time.time()

        log.info(f"╔══════════════════════════════════════════════╗")
        log.info(f"║  Self-Learner: {n_cycles} cycles × {problems_per_cycle} problems  ║")
        log.info(f"║  Stage: {self.brain.stage.value:<38}║")
        log.info(f"╚══════════════════════════════════════════════╝")

        for cycle in range(n_cycles):
            if not self._running:
                log.info("Learning interrupted.")
                break

            cycle_start = time.time()
            log.info(f"\n── Cycle {cycle+1}/{n_cycles} ──")

            # Run learning problems
            results = self.brain.learn_cycle(n=problems_per_cycle)

            # Tally results
            successes = sum(1 for r in results if r.get("success"))
            self._stats["cycles_completed"] += 1
            self._stats["total_solves"] += len(results)
            self._stats["total_successes"] += successes

            # Log progress
            cycle_time = time.time() - cycle_start
            solve_rate = successes / max(len(results), 1)
            log.info(f"  Results: {successes}/{len(results)} solved "
                     f"({solve_rate:.0%}) in {cycle_time:.1f}s")

            # Report curriculum progress
            if self.brain.developmental_curriculum:
                cmap = self.brain.developmental_curriculum.get_curriculum_map()
                mastered = cmap.get("mastered", 0)
                total = cmap.get("total_domains", 0)
                log.info(f"  Curriculum: {mastered}/{total} domains mastered")
                self._stats["domains_mastered"] = mastered

            # Update curriculum with results
            if self.brain.developmental_curriculum:
                for r in results:
                    try:
                        self.brain.developmental_curriculum.record_attempt(
                            expression=r.get("expression", ""),
                            domain_name=r.get("domain", "general"),
                            success=r.get("success", False),
                            delta=r.get("delta", 0),
                        )
                    except Exception:
                        pass

            # Report transforms learned
            for r in results:
                if r.get("success") and r.get("transforms"):
                    transforms_str = " → ".join(r["transforms"][:5])
                    log.info(f"    ✓ {r['expression']}: {transforms_str}")

            log.info(f"  Stage: {self.brain.stage.value}")

            # Sleep consolidation
            if (cycle + 1) % sleep_every == 0:
                self._sleep_consolidation()

            # Save state periodically
            if (cycle + 1) % 10 == 0:
                self._save_all()

        # Final save
        self._save_all()

        elapsed = time.time() - self._stats["started_at"]
        log.info(f"\n{'='*50}")
        log.info(f"Learning complete: {self._stats['cycles_completed']} cycles in {elapsed:.1f}s")
        log.info(f"Solve rate: {self._stats['total_successes']}/{self._stats['total_solves']} "
                 f"({self._stats['total_successes']/max(self._stats['total_solves'],1):.0%})")
        log.info(f"Stage: {self.brain.stage.value}")
        log.info(f"{'='*50}")

        return self._stats

    def run_continuous(self, interval: float = 2.0, problems_per_cycle: int = 3):
        """Run continuously until interrupted."""
        self.boot()
        self._running = True
        self._stats["started_at"] = time.time()

        log.info("Self-Learner running continuously (Ctrl+C to stop)")
        cycle = 0

        while self._running:
            cycle += 1
            try:
                results = self.brain.learn_cycle(n=problems_per_cycle)
                successes = sum(1 for r in results if r.get("success"))
                self._stats["cycles_completed"] += 1
                self._stats["total_solves"] += len(results)
                self._stats["total_successes"] += successes

                rate = successes / max(len(results), 1)
                log.info(f"Cycle {cycle}: {successes}/{len(results)} ({rate:.0%}) | "
                         f"Stage: {self.brain.stage.value}")

                # Update curriculum
                if self.brain.developmental_curriculum:
                    for r in results:
                        try:
                            self.brain.developmental_curriculum.record_attempt(
                                r.get("expression", ""), r.get("domain", "general"),
                                r.get("success", False), r.get("delta", 0))
                        except Exception:
                            pass

                # Sleep every 10 cycles
                if cycle % 10 == 0:
                    self._sleep_consolidation()

                # Save every 20 cycles
                if cycle % 20 == 0:
                    self._save_all()

            except KeyboardInterrupt:
                log.info("\nInterrupted by user.")
                break
            except Exception as e:
                log.error(f"Cycle {cycle} error: {e}")

            time.sleep(interval)

        self._save_all()
        log.info(f"Self-Learner stopped after {cycle} cycles.")

    def _sleep_consolidation(self):
        """
        Sleep cycle: offline consolidation.
        - Mine frequent transform patterns → promote macros
        - Train heuristic GNN (if torch available)
        - Strengthen confident rules
        - Decay unused strategies
        """
        log.info("  💤 Sleep consolidation...")
        self._stats["sleep_cycles"] += 1

        from sare.brain import Event
        self.brain.events.emit(Event.SLEEP_STARTED, {}, "self_learner")

        # 1. Mine patterns and promote macros
        try:
            from sare.learning.abstraction_learning import mine_frequent_patterns, propose_macros
            from sare.meta.macro_registry import upsert_macros
            from sare.sare_logging.logger import SareLogger

            logger = SareLogger(str(REPO_ROOT / "logs" / "solves.jsonl"))
            entries = logger.read_all()
            traces = [e.transform_sequence for e in entries
                      if e.solve_success and isinstance(e.transform_sequence, list)]

            if traces:
                patterns = mine_frequent_patterns(traces, min_freq=2, max_len=4)
                if patterns:
                    new_macros = propose_macros(patterns)
                    if new_macros:
                        upsert_macros(new_macros)
                        log.info(f"    Promoted {len(new_macros)} macro transforms")
                        # Refresh transforms in brain
                        self.brain._refresh_transforms()
        except Exception as e:
            log.debug(f"    Macro mining failed: {e}")

        # 2. Train heuristic GNN
        try:
            from sare.heuristics.trainer import train_epoch
            train_epoch(epochs=1)
            log.info("    GNN heuristic updated")
        except Exception:
            pass

        # 3. Memory decay
        if self.brain.memory_manager:
            try:
                if hasattr(self.brain.memory_manager, 'decay_strategies'):
                    self.brain.memory_manager.decay_strategies(factor=0.95)
            except Exception:
                pass

        self.brain.events.emit(Event.SLEEP_ENDED, {}, "self_learner")
        log.info("  💤 Sleep complete.")

    def _save_all(self):
        """Save all state."""
        try:
            self.brain.save_state()
            if self.brain.developmental_curriculum:
                self.brain.developmental_curriculum.save()
        except Exception as e:
            log.warning(f"Save failed: {e}")

    def stop(self):
        self._running = False


def main():
    parser = argparse.ArgumentParser(description="SARE-HX Self-Learning Daemon")
    parser.add_argument("--cycles", type=int, default=20,
                        help="Number of learning cycles (0 = continuous)")
    parser.add_argument("--problems", type=int, default=5,
                        help="Problems per cycle")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Seconds between cycles (continuous mode)")
    parser.add_argument("--sleep-every", type=int, default=5,
                        help="Sleep consolidation every N cycles")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    learner = SelfLearner()

    # Handle Ctrl+C gracefully
    def sigint_handler(sig, frame):
        log.info("\nShutdown signal received...")
        learner.stop()
    signal.signal(signal.SIGINT, sigint_handler)

    if args.cycles == 0:
        learner.run_continuous(interval=args.interval, problems_per_cycle=args.problems)
    else:
        learner.run_batch(
            n_cycles=args.cycles,
            problems_per_cycle=args.problems,
            sleep_every=args.sleep_every,
        )

    # Shutdown
    if learner.brain:
        learner.brain.shutdown()


if __name__ == "__main__":
    main()
