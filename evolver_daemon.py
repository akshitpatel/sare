#!/usr/bin/env python3
"""
evolver_daemon.py — Continuous self-improvement daemon for SARE-HX.

Runs self-improvement debates in a loop, autonomously selecting the
highest-priority target files using BottleneckAnalyzer, then patching
them through the full 7-turn debate pipeline.

Differs from run_evolver.py (single shot) by:
  - Running forever in a loop (SIGINT/SIGTERM to stop)
  - Auto-rotating targets across all bottleneck modules
  - Tracking a configurable daily USD budget cap
  - Persisting run history to data/memory/evolver_daemon.json
  - Integrating with homeostasis (pauses in consolidation mode)
  - Live colored status updates between debates

Usage:
    python3 evolver_daemon.py                           # default: 15-min interval, $2/day
    python3 evolver_daemon.py --interval 30             # 30-min between debates
    python3 evolver_daemon.py --budget 5.0              # $5/day cap (0 = unlimited)
    python3 evolver_daemon.py --target sare/memory/world_model.py --type optimize
    python3 evolver_daemon.py --dry-run                 # pick targets, don't patch
    python3 evolver_daemon.py --once                    # run one debate then exit
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import date
from pathlib import Path
from typing import List, Optional

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parent
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))
os.chdir(PYTHON)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
for noisy in ["urllib3", "httpx", "httpcore", "httpx._client"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("evolver_daemon")

# ── Paths ──────────────────────────────────────────────────────────────────────
MEMORY_DIR  = ROOT / "data" / "memory"
STATE_PATH  = MEMORY_DIR / "evolver_daemon.json"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# ── Terminal colors ────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RED    = "\033[91m"
PURPLE = "\033[95m"
DIM    = "\033[2m"
DIV    = "═" * 72


def _c(color: str, text: str) -> str:
    return f"{color}{text}{RESET}"


def _banner(title: str, color: str = CYAN) -> None:
    print(f"\n{color}{BOLD}{DIV}")
    print(f"  {title}")
    print(f"{DIV}{RESET}")


def _fmt_cost(usd: float) -> str:
    if usd == 0:
        return _c(GREEN, "$0.000000 (FREE)")
    elif usd < 0.01:
        return _c(YELLOW, f"${usd:.6f}")
    else:
        return _c(RED, f"${usd:.4f}")


# ── State persistence ──────────────────────────────────────────────────────────

class EvolverState:
    """Persists daemon run history and daily cost across restarts."""

    def __init__(self):
        self.runs: List[dict] = []          # all debate records
        self.daily_spend: float = 0.0       # today's USD spend
        self.daily_date: str = ""           # date string for today
        self.total_applied: int = 0
        self.total_rolled_back: int = 0
        self.total_debates: int = 0
        self.started_at: float = time.time()
        self._load()

    def _load(self) -> None:
        if STATE_PATH.exists():
            try:
                d = json.loads(STATE_PATH.read_text())
                self.runs            = d.get("runs", [])[-200:]
                self.total_applied   = d.get("total_applied", 0)
                self.total_rolled_back = d.get("total_rolled_back", 0)
                self.total_debates   = d.get("total_debates", 0)
                # Reset daily spend if it's a new day
                saved_date = d.get("daily_date", "")
                today = str(date.today())
                if saved_date == today:
                    self.daily_spend = d.get("daily_spend", 0.0)
                    self.daily_date  = today
                else:
                    self.daily_spend = 0.0
                    self.daily_date  = today
                log.info("EvolverState loaded: %d total debates, $%.4f today",
                         self.total_debates, self.daily_spend)
            except Exception as e:
                log.debug("EvolverState load error: %s", e)

    def save(self) -> None:
        try:
            data = {
                "runs":               self.runs[-200:],
                "total_applied":      self.total_applied,
                "total_rolled_back":  self.total_rolled_back,
                "total_debates":      self.total_debates,
                "daily_spend":        round(self.daily_spend, 6),
                "daily_date":         str(date.today()),
                "daemon_started_at":  self.started_at,
                "last_saved":         time.time(),
            }
            tmp = STATE_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(tmp, STATE_PATH)
        except Exception as e:
            log.warning("EvolverState save error: %s", e)

    def record_run(self, run: dict) -> None:
        self.total_debates += 1
        outcome = run.get("outcome", "")
        cost    = run.get("cost_usd", 0.0)

        # Reset daily counter on date change
        today = str(date.today())
        if self.daily_date != today:
            self.daily_spend = 0.0
            self.daily_date  = today

        self.daily_spend += cost
        if outcome == "applied":
            self.total_applied += 1
        elif "rolled_back" in outcome:
            self.total_rolled_back += 1

        self.runs.append(run)
        self.save()

    def today_spend(self) -> float:
        today = str(date.today())
        if self.daily_date != today:
            return 0.0
        return self.daily_spend

    def success_rate(self) -> float:
        if not self.total_debates:
            return 0.0
        return round(self.total_applied / self.total_debates, 3)


# ── Main daemon ────────────────────────────────────────────────────────────────

def _check_homeostasis_pause() -> bool:
    """Return True if homeostasis recommends pausing (consolidation mode)."""
    try:
        from sare.meta.homeostasis import get_homeostatic_system
        hs = get_homeostatic_system()
        rec = hs.get_behavior_recommendation()
        if rec == "consolidate_memory":
            log.info("Homeostasis recommends consolidation — skipping this cycle")
            return True
    except Exception:
        pass
    return False


def _get_cost_since(t0: float) -> float:
    """Get total LLM cost incurred since timestamp t0."""
    try:
        from sare.interface.llm_bridge import get_cost_summary
        return get_cost_summary().get("total_usd", 0.0)
    except Exception:
        return 0.0


def _print_run_summary(run: dict, elapsed: float, cycle: int, state: EvolverState) -> None:
    outcome = run.get("outcome", "unknown")
    target  = run.get("target_file", "?").split("/")[-1]
    cost    = run.get("cost_usd", 0.0)
    score   = run.get("critic_score", 0)
    improve = run.get("improvement_type", "")

    outcome_color = GREEN if outcome == "applied" else (YELLOW if "reject" in outcome else RED)
    print(f"\n{BOLD}  Cycle {cycle:>3}  │  {target}  [{improve}]{RESET}")
    print(f"           │  Outcome : {outcome_color}{BOLD}{outcome.upper()}{RESET}")
    print(f"           │  Critic  : {score}/10")
    print(f"           │  Cost    : {_fmt_cost(cost)}")
    print(f"           │  Elapsed : {elapsed:.1f}s")
    print(f"           │  Daily $ : {_fmt_cost(state.today_spend())}  "
          f"│  Total applied: {state.total_applied}  "
          f"│  Win rate: {state.success_rate()*100:.0f}%")


def run_daemon(
    interval_minutes: float = 15.0,
    budget_usd: float = 2.0,
    target_file: Optional[str] = None,
    improvement_type: Optional[str] = None,
    dry_run: bool = False,
    once: bool = False,
    verbose: bool = False,
) -> int:
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    _banner("SARE-HX EVOLVER DAEMON", PURPLE)
    print(f"  Interval   : {interval_minutes:.0f} min between debates")
    print(f"  Budget     : {'unlimited' if budget_usd == 0 else f'${budget_usd:.2f}/day'}")
    print(f"  Mode       : {'DRY RUN (no patches)' if dry_run else 'LIVE (patches applied)'}")
    if target_file:
        print(f"  Pinned     : {target_file}  [{improvement_type or 'optimize'}]")
    else:
        print(f"  Target     : auto (BottleneckAnalyzer)")
    print(f"  State file : {STATE_PATH}")
    print(f"  Stop with  : CTRL-C or SIGTERM\n")

    state = EvolverState()

    # Graceful shutdown
    _stop = [False]

    def _handler(sig, frame):
        log.info("Stop signal received — finishing current debate then exiting.")
        _stop[0] = True

    signal.signal(signal.SIGINT,  _handler)
    signal.signal(signal.SIGTERM, _handler)

    # Import core modules once
    try:
        from sare.meta.self_improver import SelfImprover
        from sare.interface.llm_bridge import get_cost_summary, llm_available
    except Exception as e:
        log.error("Import failed: %s", e)
        return 1

    if not llm_available():
        log.error("LLM not reachable — check configs/llm.json and network.")
        return 1

    si = SelfImprover()
    cost_before_session = get_cost_summary().get("total_usd", 0.0)

    cycle = 0
    while not _stop[0]:
        cycle += 1
        _banner(f"CYCLE {cycle}  —  {time.strftime('%H:%M:%S')}", CYAN)

        # ── Budget check ────────────────────────────────────────────────────
        today_spend = state.today_spend()
        if budget_usd > 0 and today_spend >= budget_usd:
            print(_c(YELLOW,
                f"  Daily budget exhausted (${today_spend:.4f} / ${budget_usd:.2f}) "
                "— sleeping until midnight"))
            # Sleep until next day (check every 5 min)
            while not _stop[0]:
                time.sleep(300)
                if str(date.today()) != state.daily_date:
                    break
            continue

        # ── Homeostasis check ───────────────────────────────────────────────
        if _check_homeostasis_pause():
            print(_c(DIM, f"  [homeostasis] consolidation mode — skipping debate"))
            if once:
                break
            if not _stop[0]:
                time.sleep(interval_minutes * 60 / 2)
            continue

        # ── Dry-run: just show what would be targeted ───────────────────────
        if dry_run:
            try:
                from sare.meta.bottleneck_analyzer import BottleneckAnalyzer
                targets = BottleneckAnalyzer().analyze()
                print(f"  {DIM}[dry-run] Top targets:{RESET}")
                for i, t in enumerate(targets[:5]):
                    print(f"    {i+1}. {t.module_name:45s} score={t.score:.2f}  {t.reason[:50]}")
            except Exception as e:
                print(_c(YELLOW, f"  [dry-run] BottleneckAnalyzer error: {e}"))
            if once:
                break
            log.info("Dry-run cycle done. Sleeping %.0f min.", interval_minutes)
            if not _stop[0]:
                time.sleep(interval_minutes * 60)
            continue

        # ── Run one debate ──────────────────────────────────────────────────
        cost_snap = get_cost_summary().get("total_usd", 0.0)
        t0 = time.time()

        try:
            if target_file:
                result = si.run_once(target_file=target_file,
                                     improvement_type=improvement_type or "optimize")
            else:
                result = si.run_once()
        except Exception as e:
            log.error("Debate crashed: %s", e)
            result = {"outcome": f"error: {e}", "cost_usd": 0.0}

        elapsed = time.time() - t0
        cost_this_run = get_cost_summary().get("total_usd", 0.0) - cost_snap

        # ── Record and print ────────────────────────────────────────────────
        run_record = {
            "cycle":            cycle,
            "outcome":          result.get("outcome", "unknown"),
            "target_file":      result.get("target_file", target_file or "auto"),
            "improvement_type": result.get("improvement_type", improvement_type or "optimize"),
            "critic_score":     result.get("critic_score", 0),
            "cost_usd":         round(cost_this_run, 6),
            "elapsed_s":        round(elapsed, 1),
            "timestamp":        time.time(),
        }
        state.record_run(run_record)
        _print_run_summary(run_record, elapsed, cycle, state)

        # ── Homeostasis feedback ────────────────────────────────────────────
        try:
            from sare.meta.homeostasis import get_homeostatic_system
            hs = get_homeostatic_system()
            if run_record["outcome"] == "applied":
                hs.on_rule_discovered()
            hs.on_problem_solved()
        except Exception:
            pass

        # ── World model feedback ────────────────────────────────────────────
        if run_record["outcome"] == "applied":
            try:
                from sare.memory.world_model import get_world_model
                wm = get_world_model()
                fname = run_record.get("target_file", "unknown")
                wm.add_fact("meta",
                            f"Self-improver patched {Path(fname).name} successfully",
                            confidence=0.9, source="evolver_daemon")
            except Exception:
                pass

        if once:
            break

        # ── Sleep until next cycle ──────────────────────────────────────────
        if not _stop[0]:
            sleep_s = interval_minutes * 60
            log.info("Next debate in %.0f min.  (Ctrl-C to stop)", interval_minutes)
            # Sleep in small chunks so we can respond to SIGINT quickly
            chunks = int(sleep_s / 10)
            for _ in range(chunks):
                if _stop[0]:
                    break
                time.sleep(10)

    # ── Session summary ─────────────────────────────────────────────────────
    session_cost = get_cost_summary().get("total_usd", 0.0) - cost_before_session
    _banner("SESSION SUMMARY", GREEN)
    print(f"  Cycles run       : {cycle}")
    print(f"  Total debates    : {state.total_debates}")
    print(f"  Applied patches  : {state.total_applied}")
    print(f"  Win rate         : {state.success_rate()*100:.0f}%")
    print(f"  Session cost     : {_fmt_cost(session_cost)}")
    print(f"  Today total      : {_fmt_cost(state.today_spend())}")
    print(f"  State saved      : {STATE_PATH}\n")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SARE-HX continuous self-improvement daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--interval",  type=float, default=15.0,
                        help="Minutes between debates (default: 15)")
    parser.add_argument("--budget",    type=float, default=2.0,
                        help="Daily USD budget cap, 0=unlimited (default: 2.0)")
    parser.add_argument("--target",    type=str,   default=None,
                        help="Pin a specific file (e.g. sare/memory/world_model.py)")
    parser.add_argument("--type",      type=str,   default=None,
                        choices=["optimize", "extend", "fix"],
                        help="Improvement type (default: auto)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="List targets but do not apply any patches")
    parser.add_argument("--once",      action="store_true",
                        help="Run a single debate then exit")
    parser.add_argument("--verbose",   action="store_true",
                        help="Enable DEBUG logging")
    args = parser.parse_args()

    return run_daemon(
        interval_minutes  = args.interval,
        budget_usd        = args.budget,
        target_file       = args.target,
        improvement_type  = args.type,
        dry_run           = args.dry_run,
        once              = args.once,
        verbose           = args.verbose,
    )


if __name__ == "__main__":
    raise SystemExit(main())
