#!/usr/bin/env python3
"""
SARE-HX Auto Runner — Continuous intelligence optimization loop.

Monitors solve_rate, rule_promotions, and energy_delta every 30s.
Auto-adjusts beam_width (4-16) and budget_seconds (5-15) using a simple
PID-like controller. Runs benchmark suite every 10 minutes.
Prints a live dashboard to stdout.

Usage:
    python3 auto_runner.py [--port 8080] [--interval 30] [--no-bench]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent
MEMORY_DIR = REPO_ROOT / "data" / "memory"

# ── ANSI colours ────────────────────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
DIM    = "\033[2m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ── Target metrics ───────────────────────────────────────────────────────────
TARGET_SOLVE_RATE   = 0.70   # aim for 70% solve rate
TARGET_ENERGY_DELTA = 0.40   # aim for avg ΔE ≥ 0.40 per problem

# ── PID constants ───────────────────────────────────────────────────────────
KP_RATE  = 2.0   # proportional gain for solve rate
KP_DELTA = 1.5   # proportional gain for energy delta

BEAM_MIN, BEAM_MAX = 4, 16
BUDGET_MIN, BUDGET_MAX = 5.0, 15.0


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _fetch_metric(port: int, endpoint: str) -> dict:
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://localhost:{port}{endpoint}", timeout=5) as r:
            return json.loads(r.read())
    except Exception:
        return {}


def _post_json(port: int, endpoint: str, payload: dict) -> dict:
    import urllib.request, urllib.error
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}{endpoint}",
        data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())
    except Exception:
        return {}


class AutoRunner:
    def __init__(self, port: int = 8080, interval: float = 30.0, run_bench: bool = True):
        self.port = port
        self.interval = interval
        self.run_bench = run_bench

        self._beam_width: int = 8
        self._budget: float = 8.0
        self._running: bool = False

        self._history: List[dict] = []
        self._last_bench_time: float = 0.0
        self._cycle: int = 0

        # PID integral terms
        self._rate_integral: float = 0.0
        self._delta_integral: float = 0.0

    def _gather_metrics(self) -> dict:
        """Collect current system metrics from web API and disk files."""
        trainer = _fetch_metric(self.port, "/api/brain/trainer")
        arena   = _fetch_metric(self.port, "/api/brain/arena")
        causal  = _fetch_metric(self.port, "/api/causal/hierarchy")
        health  = _fetch_metric(self.port, "/api/health")

        # Read daemon progress from disk
        progress  = _read_json(MEMORY_DIR / "progress.json")
        run_report = _read_json(MEMORY_DIR / "run_report.json")

        solve_rate   = float(trainer.get("recent_rate", 0.0))
        total_probs  = int(trainer.get("total_problems", 0))
        total_succ   = int(trainer.get("total_successes", 0))
        promotions   = int(causal.get("total_promoted", 0)) if causal else 0
        arena_races  = int(arena.get("total_races", 0)) if arena else 0
        energy_delta = float(run_report.get("avg_delta", 0.0)) if run_report else 0.0

        # Health subsystems
        subsystems = health.get("subsystems", {}) if isinstance(health.get("subsystems"), dict) else {}
        healthy_count = sum(1 for v in subsystems.values() if v == "healthy")

        return {
            "solve_rate":    solve_rate,
            "total_problems": total_probs,
            "total_successes": total_succ,
            "energy_delta":  energy_delta,
            "promotions":    promotions,
            "arena_races":   arena_races,
            "healthy":       healthy_count,
            "total_subsys":  len(subsystems),
            "ts":            time.time(),
        }

    def _compute_adjustments(self, m: dict) -> Tuple[int, float]:
        """PID-like controller: adjust beam_width and budget based on metrics."""
        rate_err  = TARGET_SOLVE_RATE - m["solve_rate"]
        delta_err = TARGET_ENERGY_DELTA - m["energy_delta"]

        # Accumulate integral (capped)
        self._rate_integral  = max(-3.0, min(3.0, self._rate_integral  + rate_err))
        self._delta_integral = max(-3.0, min(3.0, self._delta_integral + delta_err))

        # Beam width: increase if solve rate too low, decrease if already high
        beam_adj = -KP_RATE * rate_err  # negative error → increase beam
        new_beam = int(round(self._beam_width + beam_adj))
        new_beam = max(BEAM_MIN, min(BEAM_MAX, new_beam))

        # Budget: increase if energy delta too low (need deeper search)
        budget_adj = KP_DELTA * delta_err
        new_budget = self._budget + budget_adj
        new_budget = max(BUDGET_MIN, min(BUDGET_MAX, new_budget))

        return new_beam, round(new_budget, 1)

    def _apply_adjustments(self, beam: int, budget: float) -> bool:
        """Push new beam_width and budget to the running system via API."""
        changed = False
        if beam != self._beam_width:
            _post_json(self.port, "/api/config/beam_width", {"value": beam})
            self._beam_width = beam
            changed = True
        if abs(budget - self._budget) > 0.4:
            _post_json(self.port, "/api/config/budget_seconds", {"value": budget})
            self._budget = budget
            changed = True
        return changed

    def _run_benchmark(self) -> dict:
        """Trigger benchmark run and return scores."""
        result = _fetch_metric(self.port, "/api/benchmark/all")
        return result

    def _print_dashboard(self, m: dict, new_beam: int, new_budget: float, bench: Optional[dict]) -> None:
        os.system("clear")
        print(f"{BOLD}{CYAN}╔══════════════════════════════════════════════════════╗{RESET}")
        print(f"{BOLD}{CYAN}║         SARE-HX Auto Runner  Cycle #{self._cycle:<5}          ║{RESET}")
        print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════╝{RESET}")
        print()

        rate_color = GREEN if m["solve_rate"] >= TARGET_SOLVE_RATE else YELLOW
        delta_color = GREEN if m["energy_delta"] >= TARGET_ENERGY_DELTA else YELLOW

        print(f"  {BOLD}Solve Rate   {rate_color}{m['solve_rate']*100:.1f}%{RESET}  "
              f"(target {TARGET_SOLVE_RATE*100:.0f}%)")
        print(f"  {BOLD}Energy ΔAvg  {delta_color}{m['energy_delta']:.3f}{RESET}  "
              f"(target {TARGET_ENERGY_DELTA:.2f})")
        print(f"  {BOLD}Problems     {WHITE}{m['total_problems']}{RESET}  "
              f"({m['total_successes']} solved)")
        print(f"  {BOLD}Promotions   {WHITE}{m['promotions']}{RESET} rules learned")
        print(f"  {BOLD}Arena races  {WHITE}{m['arena_races']}{RESET}")
        print(f"  {BOLD}Health       {GREEN if m['healthy'] == m['total_subsys'] else YELLOW}"
              f"{m['healthy']}/{m['total_subsys']}{RESET} subsystems")
        print()
        print(f"  {DIM}Current params:{RESET}  beam_width={self._beam_width}  budget={self._budget}s")
        print(f"  {DIM}Next params:   {RESET}  beam_width={new_beam}  budget={new_budget}s")
        print()

        if self._history:
            rates = [h["solve_rate"] for h in self._history[-10:]]
            bar = " ".join(
                f"{GREEN}▇{RESET}" if r >= TARGET_SOLVE_RATE else f"{YELLOW}▄{RESET}" if r > 0.4 else f"{RED}▂{RESET}"
                for r in rates
            )
            print(f"  {DIM}Solve rate trend (last 10):{RESET} {bar}")
            print()

        if bench:
            score = bench.get("overall_score", bench.get("score", 0))
            print(f"  {BOLD}Benchmark{RESET}  overall={score:.1%}" if isinstance(score, float) else
                  f"  {BOLD}Benchmark{RESET}  {bench}")
            print()

        print(f"  {DIM}Next check in {self.interval:.0f}s  |  Ctrl+C to stop{RESET}")

    def run(self) -> None:
        self._running = True
        bench_result: Optional[dict] = None

        def _handle_sigterm(sig, frame):
            print(f"\n{YELLOW}Auto runner stopping...{RESET}")
            self._running = False

        signal.signal(signal.SIGTERM, _handle_sigterm)
        signal.signal(signal.SIGINT, _handle_sigterm)

        print(f"{GREEN}SARE-HX Auto Runner started (port={self.port}, interval={self.interval}s){RESET}")

        while self._running:
            self._cycle += 1
            m = self._gather_metrics()
            self._history.append(m)
            if len(self._history) > 200:
                self._history = self._history[-100:]

            new_beam, new_budget = self._compute_adjustments(m)
            self._apply_adjustments(new_beam, new_budget)

            # Run benchmark every 10 minutes
            if self.run_bench and (time.time() - self._last_bench_time) > 600:
                bench_result = self._run_benchmark()
                self._last_bench_time = time.time()

            self._print_dashboard(m, new_beam, new_budget, bench_result)

            # Sleep in small increments so Ctrl+C is responsive
            t_end = time.time() + self.interval
            while self._running and time.time() < t_end:
                time.sleep(1.0)

        print(f"{GREEN}Auto runner stopped after {self._cycle} cycles.{RESET}")


def main():
    parser = argparse.ArgumentParser(description="SARE-HX Auto Runner")
    parser.add_argument("--port",     type=int,   default=8080)
    parser.add_argument("--interval", type=float, default=30.0,
                        help="Seconds between metric checks")
    parser.add_argument("--no-bench", action="store_true",
                        help="Disable periodic benchmark runs")
    args = parser.parse_args()

    runner = AutoRunner(port=args.port, interval=args.interval,
                        run_bench=not args.no_bench)
    runner.run()


if __name__ == "__main__":
    main()
