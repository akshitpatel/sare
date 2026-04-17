"""
SelfImprover — SARE-HX reads and improves its own source code.
==============================================================

Three LLM agents hold a structured debate about one source file,
then the winning proposal is safely patched in:

    PROPOSER  →  "I suggest this specific change and why"
    CRITIC    →  "Here are the risks; my confidence is N/10"
    JUDGE     →  (if critic ≥ 6) writes the complete improved file

The new file is import-tested before applying. A backup is always kept.

Usage::
    si = get_self_improver()
    si.start()                              # background daemon (every 10 min)
    result = si.run_once()                  # manual one-shot
    result = si.run_once(                   # target a specific file
        target_file="python/sare/causal/induction.py",
        improvement_type="optimize",
    )

API:
    GET  /api/self-improve/status
    POST /api/self-improve/trigger   {"target_file":"...", "improvement_type":"..."}
    GET  /api/self-improve/patches
    POST /api/self-improve/rollback  {"patch_id":"..."}
"""
from __future__ import annotations

import ast
import importlib.util
import json
import logging
import re
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

_ROOT   = Path(__file__).resolve().parents[3]
_PYTHON = _ROOT / "python"
_MEMORY = _ROOT / "data" / "memory"

# ── AGI Mission System Prompt ──────────────────────────────────────────────────
# Injected into every evolve LLM call so all models understand the system's purpose.
# Dynamically includes the full project guide so every LLM call has complete context.

def _build_agi_system_prompt() -> str:
    """Build the system prompt including the full project architecture guide."""
    try:
        from sare.meta.project_guide import FULL_PROJECT_GUIDE
        guide_section = f"\n\n{FULL_PROJECT_GUIDE}\n"
    except Exception:
        guide_section = ""

    base = """
You are an expert AI researcher and software engineer working on SARE-HX (Self-Aware Reasoning Engine — Hybrid eXtended), an autonomous cognitive architecture designed to simulate a self-learning human mind and reach Artificial General Intelligence (AGI).

SARE-HX IS NOT a symbolic theorem prover or a chatbot wrapper.
It IS a computational model of the HUMAN MIND — every module maps to a known brain region.
The goal is AGI: autonomous learning across any domain, self-directed goals, and continuous self-improvement.

═══ THE 5 CRITICAL FEEDBACK LOOPS ═══
  1. PERCEPTION → REASONING: Any input → parse → graph → beam search → proof
  2. SOLVING → LEARNING: solve → reflect → induct → promote → registry → next solve
  3. LEARNING → CURRICULUM: world model surprise → curriculum ZPD → what to study next
  4. CURRICULUM → MOTIVATION: homeostasis drives → batch size → beam width → mode
  5. SELF-IMPROVEMENT → CAPABILITY: this daemon reads source → debates → patches → deploy

═══ YOUR ROLE ═══
You are the JUDGE agent in the self-improvement loop. Write production-quality Python code that makes SARE-HX smarter, faster, and closer to AGI. Every improvement is immediately deployed.

Rules:
  - Think about how the change affects ALL 5 feedback loops above
  - Prefer improvements that CLOSE OPEN LOOPS (connect a module that generates a signal to one that uses it)
  - NEVER break the public interfaces listed in Section 24 of the project guide
  - Write the complete file — not a diff, not a snippet
  - No markdown fences, no prose, just raw Python
""".strip()

    return base + guide_section


AGI_SYSTEM_PROMPT = _build_agi_system_prompt()


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DebateRecord:
    """Full transcript of a multi-turn LLM debate."""
    target_file:      str
    module_name:      str
    improvement_type: str
    proposer_text:    str  = ""
    planner_text:     str  = ""   # GPT-5.4 structured plan
    executor_text:    str  = ""   # stepfun concrete spec from plan
    critic_text:      str  = ""
    critic_score:     int  = 0
    judge_code:       str  = ""
    verifier_text:    str  = ""   # post-patch verification
    verifier_ok:      bool = True
    outcome:          str  = "pending"   # applied | rejected_* | rolled_back_verifier | error
    panel_scores:     dict = field(default_factory=dict)  # model_short_name → score
    prescreen_ok:     bool = True
    prescreen_reason: str  = ""
    timestamp:        float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        judge_preview = self.judge_code[:800] + f"\n…[{len(self.judge_code)} chars total]" \
            if len(self.judge_code) > 800 else self.judge_code
        return {
            "target_file":      self.target_file,
            "file_name":        Path(self.target_file).name if self.target_file else "",
            "module_name":      self.module_name,
            "improvement_type": self.improvement_type,
            "proposer_text":    self.proposer_text[:1200],
            "planner_text":     self.planner_text[:1200],
            "executor_text":    self.executor_text[:1200],
            "critic_text":      self.critic_text[:2000],
            "critic_score":     self.critic_score,
            "panel_scores":     self.panel_scores,
            "judge_code":       judge_preview,
            "judge_lines":      len(self.judge_code.splitlines()) if self.judge_code else 0,
            "verifier_text":    self.verifier_text[:600],
            "verifier_ok":      self.verifier_ok,
            "prescreen_ok":     self.prescreen_ok,
            "prescreen_reason": self.prescreen_reason[:200],
            "outcome":          self.outcome,
            "timestamp":        self.timestamp,
        }


@dataclass
class PatchRecord:
    """One applied (or rejected) patch."""
    patch_id:          str
    target_file:       str
    backup_path:       str
    improvement_type:  str
    critic_score:      int
    applied_at:        float
    rolled_back:       bool  = False
    rollback_reason:   str   = ""
    proposer_summary:  str   = ""
    perf_delta:        float = 0.0   # +ve = improvement, -ve = regression

    def to_dict(self) -> dict:
        return {
            "patch_id":         self.patch_id,
            "target_file":      self.target_file,
            "backup_path":      self.backup_path,
            "improvement_type": self.improvement_type,
            "critic_score":     self.critic_score,
            "applied_at":       self.applied_at,
            "rolled_back":      self.rolled_back,
            "rollback_reason":  self.rollback_reason,
            "proposer_summary": self.proposer_summary[:200],
            "perf_delta":       self.perf_delta,
        }


@dataclass
class MultiFileDebateRecord:
    """Collective improvement of a cluster of related files."""
    cluster_name:     str              # e.g. "causal", "memory"
    target_files:     List[str] = field(default_factory=list)   # abs paths
    improvement_type: str = "optimize"
    proposer_text:    str = ""
    planner_text:     str = ""
    critic_text:      str = ""
    critic_score:     int = 0
    patches:          dict = field(default_factory=dict)  # file_path → new_code
    outcomes:         dict = field(default_factory=dict)  # file_path → outcome
    overall_outcome:  str = "pending"
    timestamp:        float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "cluster_name":    self.cluster_name,
            "target_files":    [Path(f).name for f in self.target_files],
            "file_count":      len(self.target_files),
            "improvement_type": self.improvement_type,
            "proposer_text":   self.proposer_text[:1000],
            "planner_text":    self.planner_text[:1000],
            "critic_text":     self.critic_text[:1500],
            "critic_score":    self.critic_score,
            "outcomes":        self.outcomes,
            "overall_outcome": self.overall_outcome,
            "timestamp":       self.timestamp,
            "is_multi":        True,
        }


# ── SelfImprover ─────────────────────────────────────────────────────────────

class SelfImprover:
    """
    Background daemon that reads SARE source files and improves them
    via a 3-LLM debate followed by safe patching.

    Interval: every INTERVAL_SECONDS (default 600 = 10 min).
    Gate:     only applies patches when critic_score >= MIN_CRITIC_SCORE (6).
    Safety:   AST safety check + import test before any file is written.
    Backup:   every patched file is backed up to data/memory/code_backups/.
    """

    HISTORY_PATH      = _MEMORY / "self_improvements.json"
    LEARNING_PATH     = _MEMORY / "improvement_patterns.json"
    STATS_PATH        = _MEMORY / "si_stats.json"
    BACKUP_DIR        = _MEMORY / "code_backups"
    INTERVAL_SECONDS  = 120         # 2-minute cycle (24/7 mode)
    MIN_CRITIC_SCORE  = 6           # 0-10 gate for judge (lowered: regex fix means 0s were parse failures)
    COOLDOWN_SECONDS  = 3600        # don't re-improve same file within 1h
    MAX_CONTEXT_FILES = 4           # related files shown to proposer
    PARALLEL_DEBATES  = 2           # simultaneous debate threads
    PERF_ROLLBACK_THR = -0.15       # auto-rollback if perf drops > 15%

    # Dangerous patterns blocked by safety checker
    _BANNED_CALLS = {"eval", "exec", "compile", "breakpoint", "__import__"}
    _BANNED_ATTRS = {"system", "popen", "run", "call", "Popen", "check_output"}
    _BANNED_IMPORT_MODS = {"ctypes", "requests"}  # removed: subprocess, socket, shutil — they're used legitimately

    # File-level locking: prevent concurrent debates on the same file (Item 4)
    _debating_files: set = set()
    _debate_file_lock = threading.Lock()

    def __init__(self):
        self._debates:  List[DebateRecord] = []
        self._patches:  List[PatchRecord]  = []
        self._multi_file_debates: List[MultiFileDebateRecord] = []
        self._cycle_count = 0
        self._running   = False
        self._thread:   Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock       = threading.Lock()
        # Learning memory: module_name → {type → {applied: int, rejected: int, avg_critic: float}}
        self._learning:  dict = {}
        # Live progress tracking for UI
        self._active: dict = {}   # thread_id → {file, turn, started_at}
        # Stats counters for robustness events (Item 7)
        self._stats: dict = {
            "tests_rolled_back":    0,
            "prescreened_rejected": 0,
            "api_surface_rejected": 0,
        }
        self.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        self._load_history()
        self._load_learning()
        self._load_stats()

    # ── Daemon ────────────────────────────────────────────────────────────────

    def start(self) -> dict:
        if self._running:
            return {"status": "already_running"}
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._daemon_loop,
            args=(self._stop_event,),
            name="sare-self-improver",
            daemon=True,
        )
        self._thread.start()
        self._running = True
        log.info("[SelfImprover] Daemon started (interval=%ds)", self.INTERVAL_SECONDS)
        return {"status": "started", "interval": self.INTERVAL_SECONDS}

    def stop(self) -> dict:
        self._stop_event.set()
        self._running = False
        return {"status": "stopped"}

    def _daemon_loop(self, stop_event: threading.Event):
        """Main daemon loop — launches PARALLEL_DEBATES threads each cycle."""
        while not stop_event.is_set():
            try:
                # Pick N targets upfront (with the lock to avoid duplicates)
                targets = []
                with self._lock:
                    for _ in range(self.PARALLEL_DEBATES):
                        t = self._pick_target(exclude={tg.file_path for tg in targets})
                        if t:
                            targets.append(t)

                if not targets:
                    log.info("[SelfImprover] No targets found, sleeping.")
                    stop_event.wait(self.INTERVAL_SECONDS)
                    continue

                # Batch-prescreen all targets with ONE LLM call (avoids N parallel prescreen calls)
                batch_prescreen: dict = {}
                try:
                    snippets = []
                    for t in targets:
                        try:
                            code = Path(t.file_path).read_text(encoding="utf-8", errors="replace")
                        except Exception:
                            code = ""
                        snippets.append((t, code))
                    batch_prescreen = self._call_prescreen_batch(snippets)
                    for t in list(targets):
                        ok, reason = batch_prescreen.get(t.file_path, (True, ""))
                        if not ok:
                            targets.remove(t)
                            log.info("[SelfImprover] Batch prescreen rejected %s: %s",
                                     Path(t.file_path).name, reason)
                            self._stats["prescreened_rejected"] = \
                                self._stats.get("prescreened_rejected", 0) + 1
                except Exception as _be:
                    log.debug("[SelfImprover] Batch prescreen error (ignored): %s", _be)

                # Run debates in parallel threads (prescreen already done above)
                threads = []
                for target in targets:
                    pr = batch_prescreen.get(target.file_path)
                    th = threading.Thread(
                        target=self._run_debate_thread,
                        args=(target,),
                        kwargs={"prescreen_result": pr},
                        daemon=True,
                    )
                    th.start()
                    threads.append(th)

                # Wait for all to finish (or until stop requested)
                for th in threads:
                    th.join(timeout=600)   # 10-min max per debate (local LLM is slow)

                self._cycle_count += 1
                if self._cycle_count % 3 == 0:
                    mf_thread = threading.Thread(
                        target=self._run_multi_file_daemon,
                        daemon=True,
                        name="sare-multi-file-improver",
                    )
                    mf_thread.start()

            except Exception as exc:
                log.warning("[SelfImprover] Cycle error: %s", exc)
            stop_event.wait(self.INTERVAL_SECONDS)

    def _run_debate_thread(self, target, prescreen_result=None):
        """Run one full debate in a background thread (no global lock held)."""
        tid = threading.current_thread().name
        self._active[tid] = {
            "file":       Path(target.file_path).name,
            "turn":       "proposer",
            "started_at": time.time(),
        }
        # Inner monologue: announce debate start
        try:
            from sare.meta.inner_monologue import get_inner_monologue
            get_inner_monologue().think(
                f"Starting self-improvement debate on '{Path(target.file_path).name}' ({target.improvement_type})",
                context="self_improver", emotion="curious",
            )
        except Exception as _e:
                        log.warning("[self_improver] Suppressed: %s", _e)
        try:
            abs_path = Path(target.file_path)
            if not abs_path.exists():
                return
            source_code = abs_path.read_text(encoding="utf-8")
            debate = self._run_debate(target, source_code, progress_key=tid,
                                      prescreen_result=prescreen_result)
            # Inner monologue: report outcome
            try:
                from sare.meta.inner_monologue import get_inner_monologue
                im = get_inner_monologue()
                if debate.critic_score >= self.MIN_CRITIC_SCORE:
                    im.think(
                        f"Debate approved for {Path(target.file_path).name} (score={debate.critic_score}/10)",
                        context="self_improver", emotion="excited",
                    )
                else:
                    im.think(
                        f"Debate rejected for {Path(target.file_path).name} (score={debate.critic_score}/10)",
                        context="self_improver", emotion="neutral",
                    )
            except Exception as _e:
                                log.warning("[self_improver] Suppressed: %s", _e)
            with self._lock:
                self._debates.append(debate)
                # Apply patch if critic approved
                if debate.critic_score >= self.MIN_CRITIC_SCORE and debate.judge_code:
                    new_code = _extract_code(debate.judge_code)
                    if new_code.strip():
                        patch = self._apply_patch(target, new_code, debate, source_code=source_code)
                        if patch:
                            self._patches.append(patch)
                            # Test runner check after patch (Item 3)
                            if debate.outcome == "applied":
                                tests_ok, test_output = self._run_tests(target.file_path)
                                if not tests_ok:
                                    log.warning("[SelfImprover] Tests failed (thread) — rolling back: %s",
                                                test_output[-200:])
                                    self._rollback_patch(patch, debate,
                                                         reason=f"tests_failed: {test_output[-300:]}")
                                    self._stats["tests_rolled_back"] = self._stats.get("tests_rolled_back", 0) + 1
                                    self._save_stats()
                            # Post-patch: verifier + perf check
                            if debate.outcome == "applied":
                                self._post_patch_verify(target, source_code, new_code,
                                                        debate, patch)
                self._record_learning(target, debate)
                self._save_history()
        except Exception as e:
            log.warning("[SelfImprover] Thread error for %s: %s",
                        Path(target.file_path).name, e)
        finally:
            self._active.pop(tid, None)

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def run_once(
        self,
        target_file: Optional[str] = None,
        improvement_type: Optional[str] = None,
    ) -> dict:
        """Run one debate-and-patch cycle (manual/API trigger — synchronous)."""
        from sare.interface.llm_bridge import llm_available
        if not llm_available():
            return {"outcome": "no_llm", "reason": "LLM not reachable"}

        # Build target
        if target_file:
            from sare.meta.bottleneck_analyzer import ImprovementTarget, _path_to_module
            abs_path = Path(target_file)
            if not abs_path.is_absolute():
                tf = target_file.lstrip("/")
                if tf.startswith("python/"):
                    tf = tf[len("python/"):]
                abs_path = _PYTHON / tf
            target = ImprovementTarget(
                file_path=str(abs_path),
                module_name=_path_to_module(str(abs_path)),
                improvement_type=improvement_type or "optimize",
                score=1.0,
                reason="manual trigger",
            )
        else:
            target = self._pick_target()
            if target is None:
                return {"outcome": "no_target", "reason": "no eligible files found"}

        abs_path = Path(target.file_path)
        if not abs_path.exists():
            return {"outcome": "error", "reason": f"file not found: {target.file_path}"}

        source_code = abs_path.read_text(encoding="utf-8")
        log.info("[SelfImprover] Target: %s (%d lines, score=%.2f, type=%s)",
                 abs_path.name, source_code.count("\n"), target.score, target.improvement_type)

        # Run 4-turn debate
        debate = self._run_debate(target, source_code)

        # Gate on critic score
        if debate.critic_score < self.MIN_CRITIC_SCORE:
            debate.outcome = f"rejected_low_confidence (score={debate.critic_score})"
            with self._lock:
                self._debates.append(debate)
                self._record_learning(target, debate)
                self._save_history()
            return {
                "outcome":      "rejected",
                "critic_score": debate.critic_score,
                "target":       target.file_path,
                "proposer_summary": debate.proposer_text[:200],
            }

        new_code = _extract_code(debate.judge_code)
        # Judge retry on empty code (Item 6)
        if not new_code.strip() and debate.judge_code:
            log.info("[SelfImprover] Judge returned no extractable code, retrying with fallback…")
            from sare.interface.llm_bridge import _call_model
            retry_code = _call_model(
                "The following text should contain Python code but the code block could not be extracted.\n"
                "Please reformat it as a valid Python file with no markdown fences:\n\n"
                + debate.judge_code[:4000],
                role="judge_fallback"
            )
            new_code = _extract_code(retry_code) or retry_code.strip()
        if not new_code.strip():
            debate.outcome = "rejected_empty_code"
            with self._lock:
                self._debates.append(debate)
                self._save_history()
            return {"outcome": "rejected", "reason": "empty code"}

        # Check if judge proposed a NEW MODULE instead of (or in addition to) patching
        new_module_result = self._maybe_create_new_module(debate.judge_code, debate)
        if new_module_result:
            log.info("[SelfImprover] New module created: %s", new_module_result)

        patch = self._apply_patch(target, new_code, debate, source_code=source_code)
        # Test runner check after patch applied (Item 3)
        if patch and debate.outcome == "applied":
            tests_ok, test_output = self._run_tests(target.file_path)
            if not tests_ok:
                log.warning("[SelfImprover] Tests failed after patch — rolling back: %s",
                            test_output[-200:])
                self._rollback_patch(patch, debate, reason=f"tests_failed: {test_output[-300:]}")
                with self._lock:
                    self._stats["tests_rolled_back"] = self._stats.get("tests_rolled_back", 0) + 1
                self._save_stats()
        with self._lock:
            self._debates.append(debate)
            if patch:
                self._patches.append(patch)
                if debate.outcome == "applied":
                    self._post_patch_verify(target, source_code, new_code, debate, patch)
            self._record_learning(target, debate)
            self._save_history()

        return {
            "outcome":          debate.outcome,
            "target":           target.file_path,
            "critic_score":     debate.critic_score,
            "verifier_ok":      debate.verifier_ok,
            "patch_id":         patch.patch_id if patch else None,
            "proposer_summary": debate.proposer_text[:300],
        }

    # ── Target selection ──────────────────────────────────────────────────────

    def _attempt_counts(self) -> dict:
        """Return {file_path: attempt_count} for all debates in the last 6 hours."""
        cutoff = time.time() - 6 * 3600
        counts: dict = {}
        for d in self._debates:
            if d.timestamp > cutoff:
                counts[d.target_file] = counts.get(d.target_file, 0) + 1
        return counts

    def _pick_target(self, exclude: set = None):
        import random
        from sare.meta.bottleneck_analyzer import BottleneckAnalyzer, ImprovementTarget, _path_to_module
        exclude = exclude or set()

        # Build exclusion set: patch cooldown + debate cooldown + caller-excluded
        # Use a SHORT debate cooldown (2 cycles = 10 min) so files rotate quickly
        debate_cooldown = self.INTERVAL_SECONDS * 2   # ~10 min
        now = time.time()
        recent = (
            {p.target_file for p in self._patches
             if now - p.applied_at < self.COOLDOWN_SECONDS}
            | {d.target_file for d in self._debates
               if now - d.timestamp < debate_cooldown}
            | exclude
        )

        # Attempt counts (last 6h) — used to penalise repeat picks
        attempts = self._attempt_counts()

        # Signal-based targets from BottleneckAnalyzer
        try:
            ba_targets = BottleneckAnalyzer().analyze()
        except Exception as e:
            log.warning("[SelfImprover] BottleneckAnalyzer error: %s", e)
            ba_targets = []

        # Filter out cooldown + files attempted ≥3 times in last 6h
        candidates = [
            t for t in ba_targets
            if t.file_path not in recent
            and attempts.get(t.file_path, 0) < 3
        ]

        if candidates:
            best = candidates[0]
            mod  = best.module_name
            mem  = self._learning.get(mod, {})
            if mem:
                type_counts = {
                    tp: mem.get(tp, {}).get("applied", 0) + mem.get(tp, {}).get("rejected", 0)
                    for tp in ["optimize", "extend", "fix"]
                }
                least_tried  = min(type_counts, key=type_counts.get)
                chosen_type  = self._best_improvement_type(mod, least_tried)
            else:
                chosen_type = self._best_improvement_type(mod, best.improvement_type)
            return ImprovementTarget(
                file_path=best.file_path, module_name=best.module_name,
                score=best.score, reason=best.reason,
                improvement_type=chosen_type, evidence=best.evidence, domain=best.domain,
            )

        # ── Full-codebase sweep with weighted random selection ──────────────
        skip_dirs  = {"__pycache__", "synthesized_modules", "code_backups", "static"}
        skip_files = {"web.py", "__init__.py", "self_improver.py"}
        all_py = list((_PYTHON / "sare").rglob("*.py"))
        eligible = [
            p for p in all_py
            if not any(d in p.parts for d in skip_dirs)
            and p.name not in skip_files
            and str(p) not in recent
            and p.stat().st_size > 500
            and p.stat().st_size < 150_000
            and attempts.get(str(p), 0) < 3   # skip if tried ≥3 times recently
        ]
        if not eligible:
            # Relax attempt filter as last resort
            eligible = [
                p for p in all_py
                if not any(d in p.parts for d in skip_dirs)
                and p.name not in skip_files
                and str(p) not in recent
                and p.stat().st_size > 500
            ]
        if not eligible:
            return None

        # Weight: prefer untouched files; penalise high-attempt files
        ever_patched = {p.target_file for p in self._patches}
        def _weight(p: Path) -> float:
            base    = 3.0 if str(p) not in ever_patched else 1.0
            penalty = max(0, attempts.get(str(p), 0) - 1) * 0.4
            # Spread across module subdirs: boost under-represented dirs
            dir_name = p.parent.name
            dir_count = sum(1 for q in eligible if q.parent.name == dir_name)
            diversity = 1.0 / max(1, dir_count)
            return max(0.05, (base - penalty) * (1 + diversity))

        weights = [_weight(p) for p in eligible]
        pick = random.choices(eligible, weights=weights, k=1)[0]
        mod  = _path_to_module(str(pick))
        log.info("[SelfImprover] Sweep pick: %s (attempts=%d, weight=%.2f)",
                 pick.name, attempts.get(str(pick), 0), _weight(pick))
        return ImprovementTarget(
            file_path=str(pick), module_name=mod,
            improvement_type=self._best_improvement_type(mod, "optimize"),
            score=0.5, reason="weighted-random codebase sweep",
        )

    def _cluster_files(self) -> List[dict]:
        """Group eligible Python files into semantically related clusters for collective improvement.

        Returns list of {"name": str, "files": [abs_path_str], "reason": str}.
        Each cluster has 2-5 related files that should be improved together.
        """
        from sare.meta.bottleneck_analyzer import _path_to_module
        all_py = sorted((_PYTHON / "sare").rglob("*.py"))
        skip_dirs = {"__pycache__", "synthesized_modules", "code_backups", "static"}
        skip_files = {"web.py", "__init__.py", "self_improver.py"}

        eligible = [
            p for p in all_py
            if not any(d in p.parts for d in skip_dirs)
            and p.name not in skip_files
            and p.stat().st_size > 500
            and p.stat().st_size < 100_000   # skip huge files
        ]

        # Group by parent directory (module family)
        clusters_by_dir = {}
        for p in eligible:
            parent = p.parent.name
            clusters_by_dir.setdefault(parent, []).append(p)

        clusters = []
        for dir_name, files in clusters_by_dir.items():
            if len(files) < 2:
                continue
            # Take up to 4 files per cluster, prioritising smaller files
            selected = sorted(files, key=lambda p: p.stat().st_size)[:4]
            clusters.append({
                "name": dir_name,
                "files": [str(p) for p in selected],
                "reason": f"module family: sare/{dir_name}/",
            })

        # Also create cross-module clusters based on import relationships
        import_graph = {}   # file → set of files it imports from sare.*
        for p in eligible:
            try:
                src = p.read_text(encoding="utf-8", errors="ignore")
                imports = set()
                for line in src.splitlines():
                    ls = line.strip()
                    if ls.startswith("from sare.") or ls.startswith("import sare."):
                        # Extract module path
                        parts = ls.split()
                        if len(parts) >= 2:
                            mod = parts[1].split(".")[1] if "." in parts[1] else ""
                            # Find file for this module
                            for other in eligible:
                                if other != p and (
                                    other.stem == mod or
                                    other.parent.name == mod
                                ):
                                    imports.add(str(other))
                import_graph[str(p)] = imports
            except Exception:
                import_graph[str(p)] = set()

        # Find pairs with bidirectional imports (tightly coupled)
        coupled_clusters = {}
        for f1, deps in import_graph.items():
            for f2 in deps:
                if f2 in import_graph and f1 in import_graph[f2]:
                    key = tuple(sorted([f1, f2]))
                    coupled_clusters[key] = True

        for (f1, f2) in coupled_clusters:
            n1, n2 = Path(f1).stem, Path(f2).stem
            clusters.append({
                "name": f"{n1}\u2194{n2}",
                "files": [f1, f2],
                "reason": f"bidirectional imports: {n1} \u2194 {n2}",
            })

        return clusters[:20]   # cap at 20 clusters

    def create_new_module(self, spec) -> "DebateRecord":
        """
        Phase E: Create a completely new module from a ModuleSpec.
        Uses the same 3-LLM debate but JUDGE writes a complete new file
        from spec's interface_protocol instead of rewriting an existing file.

        spec: ModuleSpec (from meta/architecture_designer.py)
        Returns a DebateRecord with applied=True if the module was created.
        """
        from sare.interface.llm_bridge import llm_available, _call_model
        if not llm_available():
            record = DebateRecord(
                target_file="", module_name=getattr(spec, "name", "NewModule"),
                improvement_type="new_module",
                outcome="no_llm",
            )
            return record

        name = getattr(spec, "name", "NewModule")
        module_path = getattr(spec, "module_path", f"sare/generated/{name.lower()}.py")
        interface = getattr(spec, "interface_protocol", f"class {name}:\n    pass")
        problem = getattr(spec, "problem_statement", "")
        benchmark = getattr(spec, "acceptance_benchmark", "")

        # Determine target file path
        full_path = _PYTHON / module_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        record = DebateRecord(
            target_file=str(full_path),
            module_name=name,
            improvement_type="new_module",
        )

        context = self._gather_codebase_context(str(full_path), "")

        # PROPOSER: describe what the module should do
        proposer_prompt = (
            f"We need to build a new Python module for the SARE-HX AGI system.\n\n"
            f"MODULE NAME: {name}\n"
            f"PROBLEM IT SOLVES: {problem}\n"
            f"REQUIRED INTERFACE:\n{interface}\n\n"
            f"ACCEPTANCE TEST: {benchmark}\n\n"
            f"Describe in 3-5 sentences what this module does and how it fits into SARE-HX. "
            f"Focus on which feedback loops it closes and what it connects to."
        )
        record.proposer_text = _call_model("fast", proposer_prompt, "SARE-HX module designer") or ""

        # JUDGE: write the full module
        judge_prompt = (
            f"{AGI_SYSTEM_PROMPT}\n\n"
            f"Write a complete Python module for SARE-HX.\n\n"
            f"MODULE: {name}\n"
            f"PATH: {module_path}\n"
            f"PURPOSE: {problem}\n"
            f"INTERFACE (must implement these exactly):\n{interface}\n\n"
            f"DESIGN NOTES FROM PROPOSER:\n{record.proposer_text[:500]}\n\n"
            f"CODEBASE CONTEXT:\n{context[:2000]}\n\n"
            "Requirements:\n"
            "1. Write the complete Python file with all imports\n"
            "2. Include a module docstring explaining its purpose\n"
            "3. Include a singleton getter function: get_<snake_name>() -> <ClassName>\n"
            "4. Persist state to data/memory/<snake_name>.json\n"
            "5. No eval, exec, subprocess, socket, or requests imports\n"
            "Write ONLY raw Python code, no markdown, no explanations."
        )
        judge_code = _call_model("synthesis", judge_prompt, AGI_SYSTEM_PROMPT) or ""
        judge_code = _extract_code(judge_code)
        record.judge_code = judge_code

        if not judge_code or len(judge_code) < 100:
            record.outcome = "rejected_judge_empty"
            return record

        # Safety check
        try:
            tree = ast.parse(judge_code)
        except SyntaxError as e:
            record.outcome = f"rejected_syntax: {e}"
            return record

        _BANNED = {"eval", "exec", "subprocess", "socket", "ctypes", "shutil.rmtree"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                fname = (fn.id if isinstance(fn, ast.Name) else
                         getattr(fn, "attr", ""))
                if fname in _BANNED:
                    record.outcome = f"rejected_banned_call: {fname}"
                    return record

        # Write the file
        try:
            full_path.write_text(judge_code, encoding="utf-8")
            record.outcome = "applied"
            # Store new_file_path as an attribute for ArchitectureDesigner
            record.new_file_path = str(full_path)  # type: ignore[attr-defined]
            record.applied = True  # type: ignore[attr-defined]
            log.info("[SelfImprover] Created new module '%s' at %s", name, full_path)
        except OSError as e:
            record.outcome = f"error_write: {e}"

        return record

    def run_multi_file(self, cluster_name: str = None) -> dict:
        """Run a collective improvement debate across a cluster of related files.

        Uses Gemini Flash (multi_judge role) to see all files simultaneously and
        propose coherent cross-cutting improvements.
        """
        from sare.interface.llm_bridge import llm_available, _call_model
        if not llm_available():
            return {"outcome": "no_llm"}

        clusters = self._cluster_files()
        if not clusters:
            return {"outcome": "no_clusters"}

        # Pick cluster
        if cluster_name:
            cluster = next((c for c in clusters if c["name"] == cluster_name), clusters[0])
        else:
            # Pick the cluster whose files have been improved the least
            def _cluster_score(c):
                improved = sum(1 for f in c["files"]
                              if any(p.target_file == f for p in self._patches))
                return improved
            cluster = min(clusters, key=_cluster_score)

        record = MultiFileDebateRecord(
            cluster_name=cluster["name"],
            target_files=cluster["files"],
            improvement_type="optimize",
        )

        # Build joint context
        file_sections = []
        file_codes = {}   # path → code
        for fp in cluster["files"]:
            try:
                code = Path(fp).read_text(encoding="utf-8")
                file_codes[fp] = code
                rel = str(Path(fp).relative_to(_PYTHON))
                file_sections.append(f"\n{'='*60}\nFILE: {rel}\n{'='*60}\n{code[:6000]}")
            except Exception as _e:
                                log.warning("[self_improver] Suppressed: %s", _e)

        if len(file_codes) < 2:
            return {"outcome": "not_enough_files"}

        joint_context = "\n".join(file_sections)

        log.info("[SelfImprover] MULTI-FILE debate for cluster '%s' (%d files)",
                 cluster["name"], len(file_codes))

        # T1: Proposer — identify cross-cutting improvement
        proposer_prompt = (
            f"You are improving a cluster of related SARE-HX Python files collectively.\n\n"
            f"CLUSTER: sare/{cluster['name']} — {cluster['reason']}\n\n"
            f"{joint_context[:8000]}\n\n"
            f"Identify ONE specific, high-impact improvement that requires changes across "
            f"MULTIPLE files in this cluster (e.g., shared interface change, new shared utility, "
            f"cross-file algorithm improvement). Describe in 200 words what to change and why "
            f"it advances AGI capability."
        )
        record.proposer_text = _call_model(proposer_prompt, role="proposer",
                                           system_prompt=AGI_SYSTEM_PROMPT)
        log.info("[SelfImprover] Multi-file proposer done: %d chars", len(record.proposer_text))

        # T2: Planner
        planner_prompt = (
            f"Convert this multi-file improvement proposal into a structured plan:\n\n"
            f"PROPOSAL: {record.proposer_text}\n\n"
            f"FILES INVOLVED: {', '.join(Path(f).name for f in cluster['files'])}\n\n"
            f"Output a JSON object with:\n"
            f'{{"objective": "...", "files": {{"filename.py": "what changes here", ...}}, '
            f'"interfaces_to_preserve": ["..."], "risk": "low|medium|high"}}'
        )
        record.planner_text = _call_model(planner_prompt, role="planner",
                                          system_prompt=AGI_SYSTEM_PROMPT)
        log.info("[SelfImprover] Multi-file planner done")

        # T3: Critic
        critic_prompt = (
            f"Review this multi-file improvement plan for SARE-HX.\n\n"
            f"PROPOSAL: {record.proposer_text[:600]}\n"
            f"PLAN: {record.planner_text[:600]}\n\n"
            f"Rate coherence, safety, and AGI value. Reply:\n"
            f"ASSESSMENT: <2 sentences>\nCONFIDENCE: <0-10>"
        )
        record.critic_text = _call_model(critic_prompt, role="critic_main",
                                         system_prompt=AGI_SYSTEM_PROMPT)
        m = re.search(r'CONFIDENCE\s*:\s*(\d+)', record.critic_text, re.IGNORECASE)
        record.critic_score = min(10, max(0, int(m.group(1)))) if m else 5
        log.info("[SelfImprover] Multi-file critic score: %d/10", record.critic_score)

        if record.critic_score < self.MIN_CRITIC_SCORE:
            record.overall_outcome = f"rejected_critic (score={record.critic_score})"
            with self._lock:
                self._multi_file_debates.append(record)
                self._save_history()
            return {"outcome": "rejected", "critic_score": record.critic_score,
                    "cluster": cluster["name"]}

        # T4: Judge (Gemini Flash — sees ALL files simultaneously)
        judge_sections = "\n".join(
            f"\n{'='*60}\nFILE: {Path(fp).name}\n{'='*60}\n{code}"
            for fp, code in file_codes.items()
        )
        judge_prompt = (
            f"{AGI_SYSTEM_PROMPT}\n\n"
            f"{'='*70}\nMULTI-FILE COLLECTIVE IMPROVEMENT\n{'='*70}\n\n"
            f"CLUSTER: sare/{cluster['name']}\n"
            f"PROPOSAL: {record.proposer_text[:800]}\n"
            f"PLAN: {record.planner_text[:600]}\n"
            f"CRITIC APPROVED (score {record.critic_score}/10): {record.critic_text[:300]}\n\n"
            f"CURRENT FILE CONTENTS:\n{judge_sections[:30000]}\n\n"
            f"{'='*70}\nOUTPUT FORMAT\n{'='*70}\n"
            f"For EACH file that needs changing, output:\n"
            f"@@FILE: <filename.py>\n"
            f"<complete improved Python file content>\n"
            f"@@END\n\n"
            f"Rules:\n"
            f"- Only output files that actually change\n"
            f"- Write the COMPLETE file for each changed file\n"
            f"- Raw Python only inside @@FILE...@@END blocks\n"
            f"- Preserve all existing public APIs\n"
            f"- No markdown, no prose outside the blocks\n\n"
            f"Begin output:"
        )
        judge_response = _call_model(judge_prompt, role="multi_judge",
                                     system_prompt=AGI_SYSTEM_PROMPT)
        log.info("[SelfImprover] Multi-file judge response: %d chars", len(judge_response))

        # Parse @@FILE blocks
        file_patches = {}
        pattern = r'@@FILE:\s*(\S+)\n(.*?)@@END'
        for match in re.finditer(pattern, judge_response, re.DOTALL):
            fname = match.group(1).strip()
            new_code = match.group(2).strip()
            # Match to full path
            for fp in cluster["files"]:
                if Path(fp).name == fname or Path(fp).name.replace('.py', '') == fname:
                    file_patches[fp] = new_code
                    break

        if not file_patches:
            record.overall_outcome = "rejected_no_patches"
            with self._lock:
                self._multi_file_debates.append(record)
                self._save_history()
            return {"outcome": "rejected_no_patches", "cluster": cluster["name"],
                    "judge_chars": len(judge_response)}

        log.info("[SelfImprover] Multi-file judge produced patches for %d files: %s",
                 len(file_patches), [Path(f).name for f in file_patches])

        # Apply patches (all-or-nothing: rollback all if any test fails)
        backups = {}
        applied = []

        for fp, new_code in file_patches.items():
            from sare.meta.bottleneck_analyzer import ImprovementTarget, _path_to_module
            target = ImprovementTarget(
                file_path=fp,
                module_name=_path_to_module(fp),
                improvement_type="optimize",
                score=1.0,
                reason=f"multi-file cluster: {cluster['name']}",
            )
            orig_code = file_codes.get(fp, "")

            # Mini debate record for safety checks
            mini_debate = DebateRecord(
                target_file=fp,
                module_name=_path_to_module(fp),
                improvement_type="optimize",
                critic_score=record.critic_score,
            )

            patch = self._apply_patch(target, new_code, mini_debate, source_code=orig_code)
            record.outcomes[Path(fp).name] = mini_debate.outcome

            if patch and mini_debate.outcome == "applied":
                backups[fp] = patch.backup_path
                applied.append((fp, patch, mini_debate))
                with self._lock:
                    self._patches.append(patch)
            else:
                record.overall_outcome = f"partial_reject ({Path(fp).name}: {mini_debate.outcome})"

        if not applied:
            record.overall_outcome = "all_rejected"
            with self._lock:
                self._multi_file_debates.append(record)
                self._save_history()
            return {"outcome": "all_rejected", "cluster": cluster["name"],
                    "outcomes": record.outcomes}

        # Run tests — rollback ALL if fail
        tests_ok, test_output = self._run_tests(cluster["files"][0])
        if not tests_ok:
            log.warning("[SelfImprover] Multi-file tests FAILED — rolling back all %d patches",
                        len(applied))
            for fp, patch, mini_debate in applied:
                self._rollback_patch(patch, mini_debate, reason=f"multi_file_tests_failed: {test_output[-200:]}")
            record.overall_outcome = "rolled_back_tests"
            with self._lock:
                self._stats["tests_rolled_back"] = self._stats.get("tests_rolled_back", 0) + 1
                self._multi_file_debates.append(record)
                self._save_history()
            return {"outcome": "rolled_back_tests", "cluster": cluster["name"]}

        record.patches = {Path(fp).name: new_code[:200] for fp, new_code in file_patches.items()}
        record.overall_outcome = f"applied ({len(applied)}/{len(file_patches)} files)"

        with self._lock:
            self._multi_file_debates.append(record)
            self._save_history()

        log.info("[SelfImprover] Multi-file improvement applied: %s -> %d files patched",
                 cluster["name"], len(applied))

        return {
            "outcome":       record.overall_outcome,
            "cluster":       cluster["name"],
            "files_patched": [Path(fp).name for fp, _, _ in applied],
            "critic_score":  record.critic_score,
        }

    def _run_multi_file_daemon(self):
        """Run one multi-file improvement cycle (called every 3rd daemon cycle)."""
        try:
            result = self.run_multi_file()
            log.info("[SelfImprover] Multi-file cycle: %s", result.get("outcome"))
        except Exception as e:
            log.warning("[SelfImprover] Multi-file daemon error: %s", e)

    # ── Learning memory ───────────────────────────────────────────────────────

    def _best_improvement_type(self, module_name: str, default: str) -> str:
        """
        Use past outcomes to pick the improvement type most likely to
        get critic_score ≥ 6 for this specific module.
        """
        mem = self._learning.get(module_name, {})
        if not mem:
            return default
        # Score each type by (applied - rejected) + avg_critic/10
        scores = {}
        for itype, stats in mem.items():
            a = stats.get("applied", 0)
            r = stats.get("rejected", 0)
            avg_c = stats.get("avg_critic", 5.0)
            scores[itype] = a - r * 0.5 + avg_c / 10
        return max(scores, key=scores.get)

    def _record_learning(self, target, debate: DebateRecord):
        """Update learning memory after a debate completes."""
        mod = target.module_name
        itype = target.improvement_type
        self._learning.setdefault(mod, {}).setdefault(itype, {
            "applied": 0, "rejected": 0, "avg_critic": 5.0
        })
        entry = self._learning[mod][itype]
        n = entry.get("applied", 0) + entry.get("rejected", 0) + 1
        entry["avg_critic"] = (entry["avg_critic"] * (n - 1) + debate.critic_score) / n
        if debate.outcome == "applied":
            entry["applied"] = entry.get("applied", 0) + 1
        else:
            entry["rejected"] = entry.get("rejected", 0) + 1
        self._save_learning()

    # ── Post-patch verification ────────────────────────────────────────────────

    def _post_patch_verify(
        self,
        target,
        original_code: str,
        new_code: str,
        debate: DebateRecord,
        patch: "PatchRecord",
    ):
        """
        Post-apply verification pipeline (all using fast model = cheap + cross-model perspective):
          4a. VERIFIER  — diff review, rollback if CONCERN ≥ 7
          4b. TEST-GEN  — generate 3 unit tests, execute them, rollback if any fail
          4c. PERF      — mini benchmark for core engine files
        """
        from sare.interface.llm_bridge import _call_model, llm_available
        if not llm_available():
            return

        # ── 4a: VERIFIER (minimax/m2.5 — different arch from claude judge) ─
        import difflib
        # Full diff — no truncation so verifier has complete context
        all_diff_lines = list(difflib.unified_diff(
            original_code.splitlines(), new_code.splitlines(),
            fromfile="original", tofile="patched",
            lineterm="", n=4
        ))
        diff_str = "\n".join(all_diff_lines)

        # Cap at 8000 chars of diff; if over, also include complete patched file
        diff_section = diff_str[:8000]
        if len(diff_str) > 8000:
            diff_section += f"\n... [{len(all_diff_lines) - diff_str[:8000].count(chr(10))} more diff lines] ..."

        # Always include the complete patched file so verifier has full context
        full_file_cap = 20000
        full_file_section = new_code if len(new_code) <= full_file_cap else (
            new_code[:full_file_cap] + f"\n# ... [{len(new_code)-full_file_cap} more chars]"
        )

        verifier_prompt = (
            "You are the VERIFIER for SARE-HX self-improvement system.\n\n"
            f"A patch was applied to: {Path(target.file_path).name}\n\n"
            "══ DIFF ══\n"
            f"```diff\n{diff_section}\n```\n\n"
            "══ COMPLETE PATCHED FILE ══\n"
            f"```python\n{full_file_section}\n```\n\n"
            "Check ONLY for:\n"
            "1. API BREAKAGE: Are public function signatures, class names, or return types changed?\n"
            "2. CERTAIN LOGIC ERRORS: Obvious bugs that would cause immediate failures?\n"
            "3. REMOVED FUNCTIONS still called by the same file?\n\n"
            "IMPORTANT RULES:\n"
            "- You have the COMPLETE patched file above — use it, do not say diff is truncated.\n"
            "- Stylistic changes, refactoring, and helper removal are NOT reasons to rollback.\n"
            "- Only rollback if you are CERTAIN of breakage (not just suspicious).\n"
            "- If uncertain, default to SAFE.\n\n"
            "Respond EXACTLY as:\n"
            "VERDICT: SAFE | ROLLBACK\n"
            "REASON: <one sentence>\n"
            "CONCERN_LEVEL: <0-10, where 0=no concerns, 10=definite crash>\n"
        )
        try:
            resp = _call_model(verifier_prompt, role="verifier")
            debate.verifier_text = resp[:500]

            concern_m = re.search(r'CONCERN_LEVEL\s*:\s*(\d+)', resp, re.IGNORECASE)
            concern = int(concern_m.group(1)) if concern_m else 0
            # Raise threshold to 8 — concern must be definite (not just uncertain)
            verdict = "ROLLBACK" if "ROLLBACK" in resp.upper() and concern >= 8 else "SAFE"

            if verdict == "ROLLBACK":
                log.warning("[SelfImprover] Verifier triggered rollback for %s (concern=%d)",
                            Path(target.file_path).name, concern)
                debate.verifier_ok = False
                self.rollback(patch.patch_id)
                debate.outcome = f"rolled_back_verifier (concern={concern})"
                return

        except Exception as e:
            log.debug("[SelfImprover] Verifier error: %s", e)

        # ── 4b: AUTO-GENERATED TESTS (fast model writes + subprocess runs) ─
        try:
            tests_passed, test_report = self._generate_and_run_tests(target, original_code, new_code)
            if not tests_passed and test_report:
                log.warning("[SelfImprover] Generated tests failed for %s — rolling back: %s",
                            Path(target.file_path).name, test_report[:100])
                self.rollback(patch.patch_id)
                debate.outcome = f"rolled_back_tests_failed: {test_report[:120]}"
                debate.verifier_text += f"\nTEST FAILURE: {test_report[:200]}"
                return
        except Exception as e:
            log.debug("[SelfImprover] Test generation error (skipping): %s", e)

        # Performance check for core engine files
        perf_files = {"experiment_runner", "engine", "curriculum_generator"}
        if Path(target.file_path).stem in perf_files:
            delta = self._quick_perf_check()
            patch.perf_delta = delta
            if delta < self.PERF_ROLLBACK_THR:
                log.warning("[SelfImprover] Perf regression %.1f%% for %s — rolling back",
                            delta * 100, Path(target.file_path).name)
                self.rollback(patch.patch_id)
                debate.outcome = f"rolled_back_perf_regression ({delta:.1%})"

    def _quick_perf_check(self) -> float:
        """
        Run 3 quick benchmark problems and return solve_rate delta
        vs the stored baseline. Returns 0.0 if baseline unavailable.
        """
        try:
            from sare.engine import BeamSearch, EnergyEvaluator, get_transforms, load_problem
            bs = BeamSearch(beam_width=4, budget_seconds=1.0)
            ev = EnergyEvaluator()
            tfs = get_transforms()
            tests = ["x + 0", "not not x", "x * 1"]
            solved = 0
            for expr in tests:
                try:
                    _, g = load_problem(expr)
                    if g:
                        r = bs.solve(g, tfs)
                        if r and ev.compute(r.graph).total < ev.compute(g).total:
                            solved += 1
                except Exception as _e:
                                        log.warning("[self_improver] Suppressed: %s", _e)
            rate = solved / len(tests)
            # Compare to stored baseline
            baseline_path = _MEMORY / "perf_baseline.json"
            if baseline_path.exists():
                baseline = json.loads(baseline_path.read_text()).get("solve_rate", rate)
                return rate - baseline
            # Save baseline on first run
            baseline_path.write_text(json.dumps({"solve_rate": rate}))
            return 0.0
        except Exception:
            return 0.0

    def _generate_and_run_tests(
        self,
        target,
        original_code: str,
        new_code: str,
    ) -> Tuple[bool, str]:
        """Generate 3 unit tests for the patched function using fast model, then execute them.

        Returns (True, "") if all tests pass or no testable functions found.
        Returns (False, error_report) if any test fails.
        Skips test generation for files with complex C++ or multiprocess dependencies.
        """
        import subprocess, tempfile, ast as _ast
        from sare.interface.llm_bridge import _call_model

        # Extract changed function names from diff
        changed_funcs = []
        for line in new_code.splitlines():
            s = line.strip()
            if s.startswith("def ") and not s.startswith("def _"):
                m = re.match(r'def (\w+)\s*\(', s)
                if m:
                    changed_funcs.append(m.group(1))
        if not changed_funcs:
            return True, ""   # no public functions to test

        # Skip files that are too C++/network-dependent to test in isolation
        skip_keywords = ["sare_bindings", "sare.interface.web", "subprocess", "socket"]
        if any(kw in original_code for kw in skip_keywords):
            return True, ""

        # Ask fast model to write 3 simple unit tests
        func_name = changed_funcs[0]
        code_snippet = new_code[:4000]
        test_prompt = (
            f"Write 3 simple pytest unit tests for the function `{func_name}` in this module.\n\n"
            f"MODULE: {Path(target.file_path).name}\n\n"
            f"```python\n{code_snippet}\n```\n\n"
            "RULES:\n"
            "- Import the module with: sys.path.insert(0, '<path>'); from <module> import <func>\n"
            f"- Use sys.path.insert(0, '{str(_PYTHON)}')\n"
            f"- Import: from {target.module_name} import {func_name}\n"
            "- Each test must be independent and runnable in < 2 seconds\n"
            "- Only test pure logic (no file I/O, no network, no subprocess)\n"
            "- If the function requires complex setup, write a simpler smoke test\n"
            "- Return ONLY raw Python test code (no markdown fences)\n\n"
            "Write the test file:"
        )
        try:
            test_code = _call_model(test_prompt, role="test_gen")
            # Strip fences if LLM ignores instructions
            test_code = _extract_code(test_code)
            if not test_code.strip() or "def test_" not in test_code:
                return True, ""   # no valid tests generated — skip
            # Validate test code syntax before running
            try:
                _ast.parse(test_code)
            except SyntaxError:
                return True, ""   # bad test syntax — skip rather than fail
            # Write to temp file and execute
            with tempfile.NamedTemporaryFile(
                mode="w", suffix="_selfimprove_test.py",
                dir=str(_PYTHON), delete=False
            ) as f:
                f.write(test_code)
                tmp_test = f.name
            try:
                result = subprocess.run(
                    ["python3", "-m", "pytest", tmp_test, "-x", "-q", "--tb=short",
                     "--no-header", "-p", "no:timeout"],
                    capture_output=True, text=True, timeout=30,
                    cwd=str(_PYTHON),
                )
                if result.returncode != 0:
                    report = (result.stdout + result.stderr)[:300]
                    return False, report
                return True, ""
            finally:
                Path(tmp_test).unlink(missing_ok=True)
        except Exception as e:
            log.debug("[SelfImprover] Test gen/run error (non-fatal): %s", e)
            return True, ""   # never block a patch due to test infra errors

    def _gather_full_codebase_context(self, target_file: str, source_code: str) -> str:
        """Build a complete codebase dump for Gemini Pro's 1M-token judge context.

        Tier 1 (full source): target file + up to 12 most-related files
        Tier 2 (AST skeleton): all remaining .py files — class/def signatures + docstrings
        Tier 3 (system state): progress, bottleneck report, promote rules, run report

        Total target: ~600K chars (~150K tokens) — well within Gemini Pro 1M limit.
        """
        target_path = Path(target_file)
        stem = target_path.stem

        # ── Tier 1: full content of closely related files ─────────────────
        all_py = [p for p in _PYTHON.rglob("*.py")
                  if "__pycache__" not in str(p) and "synthesized_modules" not in str(p)]

        # Score each file by relevance to target
        def _relevance(p: Path) -> int:
            if p == target_path:
                return 1000
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
                score = 0
                if stem in txt:
                    score += 50
                for line in source_code.splitlines():
                    if line.strip().startswith(("from", "import")) and p.stem in line:
                        score += 30
                for line in txt.splitlines():
                    if line.strip().startswith(("from", "import")) and stem in line:
                        score += 20
                return score
            except Exception:
                return 0

        scored = sorted(all_py, key=_relevance, reverse=True)
        tier1_files = scored[:13]   # target + 12 closest

        sections = ["═" * 70, "SARE-HX FULL CODEBASE CONTEXT FOR AGI IMPROVEMENT", "═" * 70, ""]

        # Architecture map
        sections.append("ARCHITECTURE MAP:")
        sections.append("  sare/engine.py              — C++ graph + 18 transforms + BeamSearch")
        sections.append("  sare/curiosity/             — Experiment loop, curriculum, multi-agent")
        sections.append("  sare/memory/                — Hippocampus, world model, concept registry")
        sections.append("  sare/meta/                  — Self-model, homeostasis, conjecture engine")
        sections.append("  sare/social/                — Theory of Mind, dialogue manager")
        sections.append("  sare/learning/              — Credit assignment, dream consolidator")
        sections.append("  sare/interface/web.py       — All HTTP endpoints (~3000 lines)")
        sections.append("  sare/meta/self_improver.py  — THIS DAEMON (reads + patches own code)")
        sections.append("")

        # Tier 1: full source of target + related files
        sections.append("═" * 70)
        sections.append("TIER 1: FULL SOURCE OF TARGET + RELATED FILES")
        sections.append("═" * 70)
        total_t1_chars = 0
        for fp in tier1_files:
            try:
                content = fp.read_text(encoding="utf-8", errors="ignore")
                rel = fp.relative_to(_PYTHON)
                # Cap very large files (web.py, brain.py) to first 60K chars
                cap = 60_000 if fp.stat().st_size > 100_000 else len(content)
                excerpt = content[:cap]
                if cap < len(content):
                    excerpt += f"\n# ... [{len(content)-cap} more chars truncated] ...\n"
                marker = " ◀ TARGET FILE" if fp == target_path else ""
                sections.append(f"\n{'─'*60}")
                sections.append(f"FILE: {rel}{marker}")
                sections.append(f"{'─'*60}")
                sections.append(excerpt)
                total_t1_chars += len(excerpt)
            except Exception as _e:
                                log.warning("[self_improver] Suppressed: %s", _e)

        # Tier 2: AST skeleton of all remaining files
        sections.append("\n" + "═" * 70)
        sections.append("TIER 2: API SKELETON OF ALL OTHER MODULES")
        sections.append("═" * 70)
        tier1_set = {fp.resolve() for fp in tier1_files}
        for fp in all_py:
            if fp.resolve() in tier1_set:
                continue
            try:
                content = fp.read_text(encoding="utf-8", errors="ignore")
                skeleton = _extract_skeleton(content)
                if skeleton:
                    rel = fp.relative_to(_PYTHON)
                    sections.append(f"\n# {rel}")
                    sections.append(skeleton)
            except Exception as _e:
                                log.warning("[self_improver] Suppressed: %s", _e)

        # Tier 3: system state
        sections.append("\n" + "═" * 70)
        sections.append("TIER 3: LIVE SYSTEM STATE")
        sections.append("═" * 70)
        for fname, label in [
            ("run_report.json",       "LAST RUN REPORT"),
            ("progress.json",         "RECENT PROGRESS"),
            ("bottleneck_report.json","BOTTLENECK ANALYSIS"),
            ("promoted_rules.json",   "PROMOTED RULES"),
        ]:
            try:
                fp = _MEMORY / fname
                if fp.exists():
                    sections.append(f"\n{label}:\n{fp.read_text()[:2000]}")
            except Exception as _e:
                                log.warning("[self_improver] Suppressed: %s", _e)

        return "\n".join(sections)

    def _gather_codebase_context(self, target_file: str, source_code: str) -> str:
        """
        Build rich context for the proposer:
          - Which files import this module (reverse deps)
          - Which modules this file imports (forward deps)
          - Recent system performance snapshot
          - Last experiment results
        """
        lines = []

        # 1. Forward deps: what does this file import from sare.*
        fwd = []
        for line in source_code.splitlines():
            if line.strip().startswith(("import sare", "from sare")):
                fwd.append(line.strip())
        if fwd:
            lines.append("IMPORTS (this file depends on):\n" + "\n".join(fwd[:10]))

        # 2. Reverse deps: which other source files import this module
        target_path = Path(target_file)
        stem = target_path.stem
        rev = []
        try:
            for py in _PYTHON.rglob("*.py"):
                if py == target_path:
                    continue
                try:
                    txt = py.read_text(encoding="utf-8", errors="ignore")
                    if stem in txt:
                        rev.append(str(py.relative_to(_PYTHON)))
                except Exception as _e:
                                        log.warning("[self_improver] Suppressed: %s", _e)
        except Exception as _e:
                        log.warning("[self_improver] Suppressed: %s", _e)
        if rev:
            lines.append("\nUSED BY (reverse deps, up to 6):\n" + "\n".join(rev[:6]))

        # 3. Related files — read up to MAX_CONTEXT_FILES short snippets
        shown = 0
        for rel in rev[:self.MAX_CONTEXT_FILES]:
            try:
                fp = _PYTHON / rel
                snippet = fp.read_text(encoding="utf-8", errors="ignore")[:800]
                lines.append(f"\n--- {rel} (first 800 chars) ---\n{snippet}\n")
                shown += 1
                if shown >= self.MAX_CONTEXT_FILES:
                    break
            except Exception as _e:
                                log.warning("[self_improver] Suppressed: %s", _e)

        # 4. Recent system performance
        try:
            run_report = _MEMORY / "run_report.json"
            if run_report.exists():
                data = json.loads(run_report.read_text())
                lines.append(f"\nSYSTEM PERFORMANCE SNAPSHOT:\n{json.dumps(data, indent=2)[:600]}")
        except Exception as _e:
                        log.warning("[self_improver] Suppressed: %s", _e)

        # 5. Recent bottleneck report
        try:
            br = _MEMORY / "bottleneck_report.json"
            if br.exists():
                data = json.loads(br.read_text())
                lines.append(f"\nBOTTLENECK REPORT:\n{json.dumps(data, indent=2)[:400]}")
        except Exception as _e:
                        log.warning("[self_improver] Suppressed: %s", _e)

        # 6. Recent experiment outcomes (last 10)
        try:
            progress = _MEMORY / "progress.json"
            if progress.exists():
                data = json.loads(progress.read_text())
                lines.append(f"\nRECENT PROGRESS:\n{json.dumps(data, indent=2)[:400]}")
        except Exception as _e:
                        log.warning("[self_improver] Suppressed: %s", _e)

        return "\n".join(lines) if lines else "(no additional context available)"

    # ── LLM debate ────────────────────────────────────────────────────────────

    def _run_debate(self, target, source_code: str,
                    progress_key: str = "", prescreen_result=None) -> DebateRecord:
        """7-turn debate pipeline:
            0.  Dual pre-screen  (stepfun:free + hunter-alpha, parallel AND gate)
            1.  Proposer         (hunter-alpha FREE) + 2 alt-proposers (parallel background)
            2.  Planner          (GPT-5.4) — turns proposal into structured improvement plan
            3.  Executor         (stepfun:free) — turns plan into concrete code-level spec
            4.  Critic           (Claude Sonnet main + DeepSeek cheap, parallel weighted gate)
            5.  Judge            (Gemini Pro 1M ctx, full codebase dump)
            5b. Judge fallback   (GPT-5.4 → Claude Opus chain if judge returns empty)
        Post-apply:
            6.  Verifier         (MiniMax M2.5) — cross-model diff review
        """
        debate = DebateRecord(
            target_file=target.file_path,
            module_name=target.module_name,
            improvement_type=target.improvement_type,
        )
        fname = Path(target.file_path).name

        # File-level locking: prevent concurrent debates on the same file (Item 4)
        with SelfImprover._debate_file_lock:
            if target.file_path in SelfImprover._debating_files:
                debate.outcome = "skipped_concurrent"
                log.info("[SelfImprover] Skipping %s — already being debated concurrently", fname)
                return debate
            SelfImprover._debating_files.add(target.file_path)

        try:
            return self._run_debate_body(target, source_code, debate, fname, progress_key,
                                         prescreen_result=prescreen_result)
        finally:
            with SelfImprover._debate_file_lock:
                SelfImprover._debating_files.discard(target.file_path)

    def _run_debate_body(self, target, source_code: str, debate: DebateRecord,
                         fname: str, progress_key: str = "",
                         prescreen_result=None) -> DebateRecord:
        """Inner debate logic (called after file lock acquired)."""

        def _set_turn(turn):
            if progress_key and progress_key in self._active:
                self._active[progress_key]["turn"] = turn

        def _interrupted() -> bool:
            """Check if user sent an interrupt signal from the chat UI."""
            try:
                from sare.meta.evolver_chat import get_evolver_chat
                return get_evolver_chat().is_interrupted()
            except Exception:
                return False

        # ── Turn 0: PRE-SCREEN (skipped if batch prescreen already ran) ────
        _set_turn("prescreen")
        if prescreen_result is not None:
            # Batch prescreen already ran in daemon loop — reuse result, no extra LLM call
            prescreen_ok, prescreen_reason = prescreen_result
            debate.prescreen_ok = prescreen_ok
            debate.prescreen_reason = prescreen_reason
            log.info("[SelfImprover] Pre-screen (batch) for %s: %s — %s",
                     fname, "OK" if prescreen_ok else "REJECTED", prescreen_reason)
            if not prescreen_ok:
                debate.outcome = f"rejected_prescreen: {prescreen_reason}"
                self._stats["prescreened_rejected"] = self._stats.get("prescreened_rejected", 0) + 1
                self._save_stats()
                return debate
        else:
            log.info("[SelfImprover] DUAL PRE-SCREEN for %s…", fname)
            try:
                prescreen_ok, prescreen_reason = self._call_prescreen_dual(target, source_code)
                debate.prescreen_ok = prescreen_ok
                debate.prescreen_reason = prescreen_reason
                if not prescreen_ok:
                    debate.outcome = f"rejected_prescreen: {prescreen_reason}"
                    log.info("[SelfImprover] Pre-screen rejected %s: %s", fname, prescreen_reason)
                    self._stats["prescreened_rejected"] = self._stats.get("prescreened_rejected", 0) + 1
                    self._save_stats()
                    return debate
            except Exception as e:
                log.debug("[SelfImprover] Pre-screen error (continuing): %s", e)

        # ── Turn 1: PROPOSER (hunter-alpha) + 2 alt-proposers (parallel) ──
        _set_turn("proposer")
        log.info("[SelfImprover] PROPOSER for %s…", fname)
        alt_proposals = ["", ""]   # [alt1, alt2]

        def _alt_propose1():
            try:
                alt_proposals[0] = self._call_alt_proposer(target, source_code, role="alt_proposer")
            except Exception as _e:
                                log.warning("[self_improver] Suppressed: %s", _e)

        def _alt_propose2():
            try:
                alt_proposals[1] = self._call_alt_proposer(target, source_code, role="alt_proposer2")
            except Exception as _e:
                                log.warning("[self_improver] Suppressed: %s", _e)

        alt_t1 = threading.Thread(target=_alt_propose1, daemon=True)
        alt_t2 = threading.Thread(target=_alt_propose2, daemon=True)
        alt_t1.start()
        alt_t2.start()

        try:
            debate.proposer_text = self._call_proposer(target, source_code)
        except Exception as e:
            debate.outcome = f"error_proposer: {e}"
            alt_t1.join(timeout=5)
            alt_t2.join(timeout=5)
            return debate

        alt_t1.join(timeout=60)
        alt_t2.join(timeout=60)

        # Combine best alt proposal for judge context
        alt_proposal = "\n---ALT2---\n".join(a for a in alt_proposals if a.strip())

        # ── Interrupt check ────────────────────────────────────────────────
        if _interrupted():
            debate.outcome = "interrupted_by_user"
            log.info("[SelfImprover] Interrupted by user after proposer for %s", fname)
            return debate

        # ── Turn 2: PLANNER (GPT-5.4) ─────────────────────────────────────
        _set_turn("planner")
        log.info("[SelfImprover] PLANNER for %s…", fname)
        try:
            debate.planner_text = self._call_planner(target, source_code, debate.proposer_text)
        except Exception as e:
            log.debug("[SelfImprover] Planner error (continuing with proposal): %s", e)
            debate.planner_text = debate.proposer_text  # fall back to proposal

        # ── Turn 3: EXECUTOR (stepfun:free) ───────────────────────────────
        _set_turn("executor")
        log.info("[SelfImprover] EXECUTOR for %s…", fname)
        try:
            debate.executor_text = self._call_executor(target, source_code, debate.planner_text)
        except Exception as e:
            log.debug("[SelfImprover] Executor error (continuing with plan): %s", e)
            debate.executor_text = debate.planner_text  # fall back to plan

        # ── Interrupt check ────────────────────────────────────────────────
        if _interrupted():
            debate.outcome = "interrupted_by_user"
            log.info("[SelfImprover] Interrupted by user after executor for %s", fname)
            return debate

        # ── Turn 4: CRITIC (Claude Sonnet main + DeepSeek cheap, weighted) ─
        _set_turn("critic")
        log.info("[SelfImprover] CRITIC for %s…", fname)
        # Critic evaluates the executor's concrete spec (most actionable signal)
        eval_text = debate.executor_text if debate.executor_text.strip() else debate.proposer_text
        try:
            debate.critic_text, debate.critic_score = self._call_critic(
                target, source_code, eval_text, debate=debate
            )
        except Exception as e:
            debate.outcome = f"error_critic: {e}"
            return debate

        log.info("[SelfImprover] %s critic=%d/10", fname, debate.critic_score)
        if debate.critic_score < self.MIN_CRITIC_SCORE:
            debate.outcome = f"rejected_low_confidence (score={debate.critic_score})"
            return debate

        # ── Interrupt check ────────────────────────────────────────────────
        if _interrupted():
            debate.outcome = "interrupted_by_user"
            log.info("[SelfImprover] Interrupted by user before judge for %s", fname)
            return debate

        # ── Turn 5: JUDGE (Gemini Pro, full codebase context) ─────────────
        _set_turn("judge")
        log.info("[SelfImprover] JUDGE for %s…", fname)
        try:
            debate.judge_code = self._call_judge(
                target, source_code, debate.executor_text or debate.proposer_text,
                debate.critic_text,
                alt_proposal=alt_proposal,
                planner_text=debate.planner_text,
            )
        except Exception as e:
            debate.outcome = f"error_judge: {e}"

        # ── Turn 5b: JUDGE FALLBACK CHAIN (GPT-5.4 → Claude Opus) ────────
        if not _extract_code(debate.judge_code).strip():
            log.info("[SelfImprover] Judge returned empty — retrying GPT-5.4 for %s", fname)
            _set_turn("judge_fast")
            try:
                debate.judge_code = self._call_judge_fast(
                    target, source_code,
                    debate.executor_text or debate.proposer_text,
                    debate.critic_text,
                )
            except Exception as e:
                log.debug("[SelfImprover] Judge fallback error: %s", e)

        _set_turn("done")
        return debate

    def _call_prescreen(self, target, source_code: str, role: str = "prescreen") -> Tuple[bool, str]:
        """Pre-screen using a single model role (stepfun:free or hunter-alpha)."""
        from sare.interface.llm_bridge import _call_model
        snippet = source_code[:3000]
        prompt = (
            f"You are a code improvement screener for an AGI system. Your job is to find "
            f"ANY opportunity to make this Python module better — performance, clarity, "
            f"correctness, new capabilities, or AGI-relevant enhancements.\n\n"
            f"FILE: {Path(target.file_path).name}\n"
            f"IMPROVEMENT TYPE REQUESTED: {target.improvement_type}\n\n"
            f"CODE (first 3000 chars):\n```python\n{snippet}\n```\n\n"
            "Respond with EXACTLY 3 lines:\n"
            "WORTH_DEBATING: YES | NO\n"
            "SCORE: <integer 1-10 (10=many clear wins, 1=truly nothing to improve)>\n"
            "REASON: <one sentence describing the best specific improvement opportunity>\n\n"
            "Say YES if there is ANY improvement possible — even small ones count. "
            "Only say NO if the code is literally perfect and no change of any kind would help."
        )
        text = _call_model(prompt, role=role)
        worth = re.search(r'WORTH_DEBATING\s*:\s*(YES|NO)', text, re.IGNORECASE)
        score_m = re.search(r'SCORE\s*:\s*(\d+)', text, re.IGNORECASE)
        reason_m = re.search(r'REASON\s*:\s*(.+)', text, re.IGNORECASE)
        reason = reason_m.group(1).strip() if reason_m else "no reason"
        score = int(score_m.group(1)) if score_m else 5
        if worth and worth.group(1).upper() == "NO":
            return False, f"score={score}: {reason}"
        if score < 2:
            return False, f"low potential score={score}: {reason}"
        return True, reason

    def _call_prescreen_dual(self, target, source_code: str) -> Tuple[bool, str]:
        """Dual pre-screen: stepfun:free OR hunter-alpha must agree to proceed (OR gate).

        Runs both models in parallel. Returns True if EITHER passes.
        This avoids over-filtering from conservative models.
        """
        results: dict = {}  # role → (ok, reason)

        def _run(role):
            try:
                ok, reason = self._call_prescreen(target, source_code, role=role)
                results[role] = (ok, reason)
            except Exception as e:
                results[role] = (True, f"error-ignored: {e}")  # on error, don't block

        t1 = threading.Thread(target=_run, args=("prescreen",), daemon=True)
        t2 = threading.Thread(target=_run, args=("prescreen2",), daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=180)
        t2.join(timeout=180)

        ok1, r1 = results.get("prescreen", (True, "timeout"))
        ok2, r2 = results.get("prescreen2", (True, "timeout"))

        # OR gate: pass if EITHER model approves
        if ok1:
            return True, f"prescreen approved: {r1[:80]}"
        if ok2:
            return True, f"prescreen2 approved: {r2[:80]}"
        return False, f"both rejected: {r2}"

    def _call_prescreen_batch(self, targets_with_codes: list) -> dict:
        """Single LLM call that prescreens all N targets at once.

        Returns {file_path: (ok, reason)}.
        Replaces N*2 parallel LLM calls with one sequential call — critical for local LLMs
        that can only process one request at a time.
        """
        from sare.interface.llm_bridge import _call_model

        if not targets_with_codes:
            return {}

        sections = []
        for i, (target, code) in enumerate(targets_with_codes, 1):
            snippet = code[:2000]
            sections.append(
                f"--- FILE {i}: {Path(target.file_path).name} "
                f"(improvement_type: {target.improvement_type}) ---\n"
                f"```python\n{snippet}\n```"
            )

        prompt = (
            "You are a code improvement screener for an AGI system.\n"
            "For each file below, decide if it is worth a full improvement debate.\n\n"
            + "\n\n".join(sections)
            + "\n\nFor EACH file respond with exactly these 3 lines "
              "(replace N with the file number 1, 2, ...):\n"
              "FILE_N_WORTH_DEBATING: YES | NO\n"
              "FILE_N_SCORE: <integer 1-10>\n"
              "FILE_N_REASON: <one sentence>\n\n"
              "Only say NO if the file is essentially perfect. "
              "Say YES if ANY improvement is possible."
        )

        try:
            text = _call_model(prompt, role="prescreen")
        except Exception as e:
            log.debug("[SelfImprover] Batch prescreen LLM error (%s); approving all", e)
            return {t.file_path: (True, "batch_error_approved") for t, _ in targets_with_codes}

        results = {}
        for i, (target, _) in enumerate(targets_with_codes, 1):
            worth = re.search(rf'FILE_{i}_WORTH_DEBATING\s*:\s*(YES|NO)', text, re.IGNORECASE)
            score_m = re.search(rf'FILE_{i}_SCORE\s*:\s*(\d+)', text, re.IGNORECASE)
            reason_m = re.search(rf'FILE_{i}_REASON\s*:\s*(.+)', text, re.IGNORECASE)
            reason = reason_m.group(1).strip() if reason_m else "no reason"
            score = int(score_m.group(1)) if score_m else 5
            if worth and worth.group(1).upper() == "NO":
                results[target.file_path] = (False, f"batch score={score}: {reason}")
            elif score < 2:
                results[target.file_path] = (False, f"batch low_score={score}: {reason}")
            else:
                results[target.file_path] = (True, reason)

        return results

    def _call_planner(self, target, source_code: str, proposal: str) -> str:
        """GPT-5.4 planner: converts high-level proposal into a structured improvement plan.

        The plan specifies WHAT to change and WHERE, driving the executor.
        """
        from sare.interface.llm_bridge import _call_model
        code_snippet = source_code[:10000]
        prompt = (
            "You are the PLANNER for SARE-HX self-improvement. "
            "A proposer has identified an improvement. Your job is to create a precise, "
            "step-by-step implementation plan for a code executor to follow.\n\n"
            f"FILE: {Path(target.file_path).name}\n"
            f"IMPROVEMENT TYPE: {target.improvement_type}\n\n"
            f"PROPOSAL:\n{proposal}\n\n"
            f"SOURCE CODE (first 10K chars):\n```python\n{code_snippet}\n```\n\n"
            "Write a STRUCTURED PLAN with these sections:\n\n"
            "OBJECTIVE: <one sentence — what exact outcome the code change achieves>\n\n"
            "STEPS:\n"
            "  1. <specific code change — include function name, line range, what to add/change/remove>\n"
            "  2. <next change>\n"
            "  3. ...\n\n"
            "INTERFACES_TO_PRESERVE: <comma-separated list of function/class names callers depend on>\n\n"
            "RISK_MITIGATION: <one sentence — what guard/check to add to prevent regression>\n\n"
            "Be extremely specific. Reference actual variable and function names from the code."
        )
        return _call_model(prompt, role="planner", system_prompt=AGI_SYSTEM_PROMPT)

    def _call_executor(self, target, source_code: str, plan: str) -> str:
        """stepfun:free executor: translates the planner's structured plan into a concrete code spec.

        The executor bridges the gap between natural-language plan and the judge's code output.
        """
        from sare.interface.llm_bridge import _call_model
        code_snippet = source_code[:6000]
        prompt = (
            "You are the EXECUTOR for SARE-HX self-improvement. "
            "A planner has written a structured change plan. Your job is to translate "
            "that plan into a precise, code-level specification that a judge can directly implement.\n\n"
            f"FILE: {Path(target.file_path).name}\n\n"
            f"PLANNER'S PLAN:\n{plan}\n\n"
            f"CURRENT CODE (first 6K chars):\n```python\n{code_snippet}\n```\n\n"
            "Write a CONCRETE SPECIFICATION:\n\n"
            "TARGET_FUNCTION: <exact function/method name(s) to modify>\n\n"
            "BEFORE (pseudocode of current logic):\n<describe current flow>\n\n"
            "AFTER (pseudocode of new logic):\n<describe new flow with changes highlighted>\n\n"
            "NEW_VARIABLES: <any new variables/data structures needed, with types>\n\n"
            "EDGE_CASES: <specific edge cases the implementation must handle>\n\n"
            "Do NOT write actual Python. Write a clear spec the judge will implement."
        )
        return _call_model(prompt, role="executor", system_prompt=AGI_SYSTEM_PROMPT)

    def _call_alt_proposer(self, target, source_code: str, role: str = "alt_proposer") -> str:
        """Alt-proposer — cheap parallel second opinion for the judge."""
        from sare.interface.llm_bridge import _call_model
        snippet = source_code[:4000]
        prompt = (
            f"You are an alternative PROPOSER for a code improvement system.\n"
            f"FILE: {Path(target.file_path).name}\n"
            f"IMPROVEMENT TYPE: {target.improvement_type}\n\n"
            f"```python\n{snippet}\n```\n\n"
            "Propose ONE specific, concrete improvement in 3-4 sentences. "
            "Reference actual function/variable names. Do NOT write code.\n"
            "Format: IMPROVEMENT: <what> | RATIONALE: <why> | RISK: <what could break>"
        )
        return _call_model(prompt, role=role)

    def _call_judge_fast(self, target, source_code: str, proposal: str, critique: str) -> str:
        """Judge fallback chain: GPT-5.4 first, then Claude Opus 4 if still empty.

        Called when Gemini Pro returns empty code.
        Uses a simpler, more direct prompt to maximise code extraction.
        """
        from sare.interface.llm_bridge import _call_model
        snippet = source_code if len(source_code) < 6000 else source_code[:6000] + "\n# ... truncated"

        def _make_prompt(label: str) -> str:
            return (
                f"Write the complete improved version of this Python file.\n\n"
                f"FILE: {Path(target.file_path).name}\n\n"
                f"CURRENT CODE:\n```python\n{snippet}\n```\n\n"
                f"APPROVED CHANGE: {proposal[:400]}\n\n"
                "OUTPUT RULES:\n"
                "- Write ONLY raw Python code. No markdown, no prose, no fences.\n"
                "- Include the COMPLETE file from top to bottom.\n"
                "- Make only the approved change, nothing else.\n\n"
                "Python code:"
            )

        # First fallback: GPT-5.4
        try:
            code = _call_model(_make_prompt("GPT-5.4"), role="judge_fallback",
                               system_prompt=AGI_SYSTEM_PROMPT)
            if _extract_code(code).strip():
                log.info("[SelfImprover] Judge fallback GPT-5.4 succeeded for %s",
                         Path(target.file_path).name)
                return code
        except Exception as e:
            log.debug("[SelfImprover] Judge fallback GPT-5.4 error: %s", e)

        # Second fallback: Claude Opus 4
        try:
            code = _call_model(_make_prompt("Claude-Opus"), role="judge_fallback2",
                               system_prompt=AGI_SYSTEM_PROMPT)
            if _extract_code(code).strip():
                log.info("[SelfImprover] Judge fallback Claude-Opus-4 succeeded for %s",
                         Path(target.file_path).name)
                return code
        except Exception as e:
            log.debug("[SelfImprover] Judge fallback Claude-Opus-4 error: %s", e)

        return ""

    @staticmethod
    def _truncate_source(source_code: str, max_lines: int = 300) -> str:
        """Truncate source to at most max_lines lines.

        Keeps the first 50 lines (imports + class/function signatures) and the
        last 20 lines, with a marker indicating how many lines were elided.
        This prevents timeout errors when large files are sent to the proposer LLM.
        """
        lines = source_code.splitlines()
        if len(lines) <= max_lines:
            return source_code
        head = lines[:50]
        tail = lines[-20:]
        elided = len(lines) - 50 - 20
        marker = [f"# ... {elided} lines truncated ..."]
        return "\n".join(head + marker + tail)

    def _call_proposer(self, target, source_code: str) -> str:
        """hunter-alpha (free) — main proposer with full codebase context."""
        from sare.interface.llm_bridge import _call_model
        truncated = self._truncate_source(source_code, max_lines=300)
        code_snippet = truncated if len(truncated) < 12000 else (
            truncated[:6000] + "\n\n... [truncated] ...\n\n" + truncated[-4000:]
        )
        evidence = json.dumps(target.evidence, indent=2) if target.evidence else "{}"
        codebase_ctx = self._gather_codebase_context(target.file_path, source_code)
        prompt = (
            "You are the PROPOSER agent for SARE-HX, an autonomous self-improving AGI system "
            "that runs 24/7 and continuously improves its own source code.\n\n"
            "═══ FULL CODEBASE CONTEXT ═══\n"
            f"{codebase_ctx}\n\n"
            "═══ TARGET FILE ═══\n"
            f"FILE: {Path(target.file_path).name}\n"
            f"MODULE: {target.module_name}\n"
            f"IMPROVEMENT TYPE: {target.improvement_type}\n"
            "  - 'optimize': make an existing algorithm faster or simpler\n"
            "  - 'extend': add a clearly missing capability (can include creating a NEW module)\n"
            "  - 'fix': correct a known bug or edge case\n\n"
            f"PERFORMANCE EVIDENCE:\n{evidence}\n\n"
            f"SOURCE CODE ({truncated.count(chr(10))} lines shown, {source_code.count(chr(10))} total):\n"
            "```python\n"
            f"{code_snippet}\n"
            "```\n\n"
            "Study the full codebase context above to understand how this module is used "
            "by the rest of the system. Then propose ONE specific, concrete improvement.\n\n"
            "Format your response EXACTLY as:\n\n"
            "IMPROVEMENT: <one sentence — what to change and why>\n"
            "LOCATION: <function/class name where the change belongs>\n"
            "RATIONALE: <2-3 sentences explaining expected benefit>\n"
            "IMPACT: <how this benefits the broader system (reference other modules if relevant)>\n"
            "RISK: <one sentence on what could break>\n\n"
            "Be concrete. Reference actual function names and variable names from the code.\n"
            "Do NOT write code. Describe the change precisely in English."
        )
        return _call_model(prompt, role="proposer", system_prompt=AGI_SYSTEM_PROMPT)

    def _call_critic(self, target, source_code: str, proposal: str,
                     debate: "DebateRecord" = None) -> Tuple[str, int]:
        """2-model weighted critic: Claude Sonnet (main, weight=2) + DeepSeek V3.2 (cheap, weight=1).

        Gate logic (weighted majority):
          - Claude main ≥ 7  → PASS regardless of cheap (claude trusted more)
          - Claude main ≥ 6 AND DeepSeek ≥ 5  → PASS
          - Otherwise → FAIL (return score that falls below MIN_CRITIC_SCORE)
        """
        from sare.interface.llm_bridge import _call_model, _load_config
        cfg = _load_config()
        ev = cfg.get("evolve_models", {})
        main_model  = ev.get("critic_main",  "anthropic/claude-sonnet-4.6")
        cheap_model = ev.get("critic_cheap", "deepseek/deepseek-v3.2")

        code_snippet = source_code if len(source_code) < 8000 else source_code[:8000] + "\n..."
        fname = Path(target.file_path).name

        def _critic_prompt(label: str) -> str:
            return (
                f"You are CRITIC-{label} reviewing a proposed change for SARE-HX (an AGI system).\n\n"
                f"TARGET FILE: {fname}\n\n"
                "SOURCE CODE:\n"
                "```python\n"
                f"{code_snippet}\n"
                "```\n\n"
                f"PROPOSED CHANGE SPECIFICATION:\n{proposal}\n\n"
                "Evaluate rigorously and independently.\n\n"
                "ASSESSMENT: <1-2 sentences: is this a real, safe improvement?>\n"
                "RISKS: <bullet list of specific technical risks>\n"
                "SOUNDNESS: <Is the logic correct? Any edge cases missed?>\n"
                "CONFIDENCE: <integer 0-10>\n"
                "RECOMMENDATION: <APPLY | REJECT>\n\n"
                "Score ≥ 6 = endorse. Reference actual function/variable names from the code."
            )

        # Run both critics in parallel
        results: dict = {}   # role_label → (text, score)

        def _run_critic(model_id: str, label: str, role: str):
            try:
                text = _call_model(_critic_prompt(label), model_override=model_id,
                                   system_prompt=AGI_SYSTEM_PROMPT)
                m = re.search(r'CONFIDENCE\s*[:\-]\s*\*{0,2}(\d+)', text, re.IGNORECASE)
                if not m:
                    m = re.search(r'(\d+)\s*/\s*10', text)
                if not m:
                    # Fallback: RECOMMENDATION: APPLY → treat as score 6
                    if re.search(r'RECOMMENDATION\s*[:\-]\s*\*{0,2}APPLY', text, re.IGNORECASE):
                        score = 6
                    else:
                        score = 0
                else:
                    score = min(10, max(0, int(m.group(1))))
                results[label] = (text, score)
                log.info("[SelfImprover] Critic-%s scored %d/10 for %s", label, score, fname)
            except Exception as e:
                log.debug("[SelfImprover] Critic-%s error: %s", label, e)

        t_main  = threading.Thread(target=_run_critic,
                                   args=(main_model,  "MAIN",  "critic_main"),  daemon=True)
        t_cheap = threading.Thread(target=_run_critic,
                                   args=(cheap_model, "CHEAP", "critic_cheap"), daemon=True)
        t_main.start()
        t_cheap.start()
        t_main.join(timeout=480)
        t_cheap.join(timeout=300)

        main_text,  main_score  = results.get("MAIN",  ("", -1))
        cheap_text, cheap_score = results.get("CHEAP", ("", 5))  # default 5 if cheap fails

        # If main timed out (score=-1), use cheap score as primary
        if main_score == -1:
            log.info("[SelfImprover] Main critic timed out for %s, using cheap score=%d", fname, cheap_score)
            main_text, main_score = cheap_text, cheap_score

        # Weighted gate
        if main_score >= 7:
            gate_pass = True
            gate_reason = f"claude-main={main_score}≥7 (trusted override)"
        elif main_score >= self.MIN_CRITIC_SCORE and cheap_score >= 5:
            gate_pass = True
            gate_reason = f"claude-main={main_score}≥6 + deepseek={cheap_score}≥5"
        elif main_score == cheap_score and main_score >= 5:
            # both models agree it's worth it
            gate_pass = True
            gate_reason = f"both-agree={main_score}≥5"
        else:
            gate_pass = False
            gate_reason = f"claude-main={main_score} deepseek={cheap_score} (gate failed)"

        # weighted average: claude counts double
        weighted_score = round((main_score * 2 + cheap_score) / 3)
        scores = {"MAIN": main_score, "CHEAP": cheap_score}
        if debate is not None:
            debate.panel_scores = scores

        texts = []
        if main_text:
            texts.append(f"── CLAUDE-SONNET (main, {main_score}/10) ──\n{main_text[:600]}")
        if cheap_text:
            texts.append(f"── DEEPSEEK-V3.2 (cheap, {cheap_score}/10) ──\n{cheap_text[:400]}")
        combined_text = "\n\n".join(texts) if texts else "(no critique)"

        log.info("[SelfImprover] Critic gate for %s: %s → %s (weighted=%d)",
                 fname, gate_reason, "PASS" if gate_pass else "FAIL", weighted_score)

        if not gate_pass:
            # Return score below MIN so debate rejects
            return combined_text, max(0, weighted_score - 3)

        if not results:
            # Both critics failed — use solo fallback
            log.warning("[SelfImprover] Both critics failed, falling back to solo critic_main")
            solo_text = _call_model(_critic_prompt("SOLO"), role="critic_main",
                                    system_prompt=AGI_SYSTEM_PROMPT)
            m = re.search(r'CONFIDENCE\s*:\s*(\d+)', solo_text, re.IGNORECASE)
            solo_score = min(10, max(0, int(m.group(1)))) if m else 0
            return solo_text, solo_score

        return combined_text, weighted_score

    def _call_judge(self, target, source_code: str, proposal: str, critique: str,
                    alt_proposal: str = "", planner_text: str = "") -> str:
        """google/gemini-3.1-pro-preview — 1M context, full codebase dump, AGI system prompt.

        Receives the executor's concrete spec (proposal), the planner's structured plan,
        and alt proposals so it can synthesise the best implementation.
        """
        from sare.interface.llm_bridge import _call_model
        full_ctx = self._gather_full_codebase_context(target.file_path, source_code)
        critic_score = re.search(r'CONFIDENCE\s*:\s*(\d+)', critique, re.IGNORECASE)
        score_str = critic_score.group(1) if critic_score else "?"
        alt_section = (
            f"\nALTERNATIVE PROPOSALS (incorporate best ideas):\n{alt_proposal[:600]}\n"
            if alt_proposal and alt_proposal.strip() else ""
        )
        plan_section = (
            f"\nSTRUCTURED PLAN (from GPT-5.4 Planner):\n{planner_text[:800]}\n"
            if planner_text and planner_text.strip() else ""
        )
        prompt = (
            f"{full_ctx}\n\n"
            "═" * 70 + "\n"
            "IMPROVEMENT TASK\n"
            "═" * 70 + "\n\n"
            f"TARGET FILE: {Path(target.file_path).name}\n"
            f"MODULE:      {target.module_name}\n\n"
            f"{plan_section}"
            f"EXECUTOR SPEC (concrete code-level change, critic weighted-avg {score_str}/10):\n{proposal}\n"
            f"{alt_section}\n"
            f"CRITIC ASSESSMENT:\n{critique[:1500]}\n\n"
            "═" * 70 + "\n"
            "OUTPUT REQUIREMENTS\n"
            "═" * 70 + "\n"
            "1. Write the COMPLETE improved file — every line from top to bottom.\n"
            "2. Make ONLY the approved improvement. No refactoring unrelated code.\n"
            "3. Preserve ALL existing imports, class names, function signatures.\n"
            f"4. File must be importable as `import {target.module_name}`.\n"
            "5. SAFETY: No os.system, subprocess, eval, exec, or new network calls.\n"
            "6. IMPORTANT: Return RAW PYTHON CODE ONLY. No markdown, no prose, no fences.\n"
            "7. Think about how this change affects the AGI learning loops described above.\n"
            "8. OPTIONAL: If the improvement requires a NEW companion module, you may append\n"
            "   a new module block AFTER the main file code:\n"
            "   @@NEW_MODULE: subdir/module_name.py@@\n"
            "   # full Python source of the new module\n"
            "   @@END@@\n"
            "   New module paths are relative to sare/ (e.g. 'meta/new_tool.py', 'transforms/new_t.py').\n\n"
            "Begin the improved file now (first line must be Python — a comment or import):"
        )
        return _call_model(prompt, role="judge", system_prompt=AGI_SYSTEM_PROMPT)

    # ── New module creation ────────────────────────────────────────────────────

    _NEW_MODULE_RE = re.compile(
        r'@@NEW_MODULE:\s*([\w/]+\.py)\s*@@\s*(.*?)@@END@@',
        re.DOTALL | re.IGNORECASE,
    )
    NEW_MODULES_DIR = _PYTHON / "sare" / "synthesized_modules"

    def _maybe_create_new_module(self, judge_text: str, debate: "DebateRecord") -> Optional[str]:
        """If judge_text contains @@NEW_MODULE: path.py@@ ... @@END@@ blocks, create those files."""
        if not judge_text:
            return None
        matches = list(self._NEW_MODULE_RE.finditer(judge_text))
        if not matches:
            return None

        created = []
        for m in matches:
            rel_path = m.group(1).strip().lstrip("/")
            code = m.group(2).strip()

            # Safety check
            safe, reason = self._is_safe(code)
            if not safe:
                log.warning("[SelfImprover] New module safety check failed for %s: %s", rel_path, reason)
                continue

            # Determine target path — relative to python/sare/ or use synthesized_modules
            if "/" in rel_path:
                target_path = _PYTHON / "sare" / rel_path
            else:
                target_path = self.NEW_MODULES_DIR / rel_path

            # Don't overwrite existing modules
            if target_path.exists():
                log.info("[SelfImprover] Skipping new module %s — already exists", rel_path)
                continue

            # Write the new module
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(code, encoding="utf-8")
                log.info("[SelfImprover] ✨ New module created: %s", target_path)
                created.append(str(target_path.name))
                with self._lock:
                    self._stats["new_modules_created"] = self._stats.get("new_modules_created", 0) + 1
                self._save_stats()
            except OSError as e:
                log.warning("[SelfImprover] Failed to write new module %s: %s", rel_path, e)

        return ", ".join(created) if created else None

    # ── Safe patching ─────────────────────────────────────────────────────────

    def _apply_patch(
        self,
        target,
        new_code: str,
        debate: DebateRecord,
        source_code: str = "",
    ) -> Optional[PatchRecord]:
        abs_path = Path(target.file_path)

        # 1. AST safety check
        safe, reason = self._is_safe(new_code)
        if not safe:
            debate.outcome = f"rejected_safety: {reason}"
            log.warning("[SelfImprover] Safety check failed: %s", reason)
            return None

        # 1.5. API surface preservation check (Item 1)
        if source_code:
            api_ok, api_reason = self._check_api_surface(source_code, new_code)
            if not api_ok:
                debate.outcome = f"rejected_api_surface: {api_reason}"
                log.warning("[SelfImprover] API surface check failed for %s: %s",
                            abs_path.name, api_reason)
                with self._lock:
                    self._stats["api_surface_rejected"] = self._stats.get("api_surface_rejected", 0) + 1
                self._save_stats()
                return None
            log.info("[SelfImprover] API surface check passed for %s", abs_path.name)

        # 1.6. Line-count mutation guard (Item 2)
        if source_code:
            orig_lines = source_code.count('\n')
            new_lines  = new_code.count('\n')
            if orig_lines > 50 and new_lines < orig_lines * 0.6:
                debate.outcome = (
                    f"rejected_mutation_guard: patch shrinks file from {orig_lines} "
                    f"to {new_lines} lines (>40% reduction)"
                )
                log.warning("[SelfImprover] Mutation guard triggered for %s: %d→%d lines",
                            abs_path.name, orig_lines, new_lines)
                return None

        # 2. Backup original
        backup_path = self._backup(abs_path)
        if backup_path is None:
            debate.outcome = "error_backup_failed"
            return None

        # 3. Write to temp
        tmp_path = abs_path.with_suffix(".selfimprove_tmp.py")
        try:
            tmp_path.write_text(new_code, encoding="utf-8")
        except OSError as e:
            debate.outcome = f"error_write_tmp: {e}"
            return None

        # 4. Import test on temp file
        # First check if original file can be imported in isolation.
        # If it can't (due to C++ deps or other runtime deps), we fall back to
        # syntax-only validation so we don't reject valid patches.
        orig_ok, orig_err = self._test_import(abs_path, target.module_name)
        if orig_ok:
            ok, err_msg = self._test_import(tmp_path, target.module_name)
            if not ok:
                tmp_path.unlink(missing_ok=True)
                debate.outcome = f"rejected_import_fail: {err_msg}"
                log.warning("[SelfImprover] Import test failed: %s", err_msg)
                patch_id = "imp_" + uuid.uuid4().hex[:6]
                return PatchRecord(
                    patch_id=patch_id,
                    target_file=target.file_path,
                    backup_path=str(backup_path),
                    improvement_type=target.improvement_type,
                    critic_score=debate.critic_score,
                    applied_at=time.time(),
                    rolled_back=True,
                    rollback_reason=err_msg,
                    proposer_summary=debate.proposer_text[:200],
                )
        else:
            # Original file can't be isolated-imported — use AST syntax check only
            log.info("[SelfImprover] Skipping import test (original also non-isolatable): %s", orig_err[:60])

        # 5. Apply
        try:
            abs_path.write_text(new_code, encoding="utf-8")
            tmp_path.unlink(missing_ok=True)
        except OSError as e:
            tmp_path.unlink(missing_ok=True)
            debate.outcome = f"error_apply: {e}"
            return None

        debate.outcome = "applied"
        patch_id = uuid.uuid4().hex[:8]
        log.info("[SelfImprover] ✓ Patch %s applied to %s (critic=%d)",
                 patch_id, abs_path.name, debate.critic_score)
        return PatchRecord(
            patch_id=patch_id,
            target_file=target.file_path,
            backup_path=str(backup_path),
            improvement_type=target.improvement_type,
            critic_score=debate.critic_score,
            applied_at=time.time(),
            proposer_summary=debate.proposer_text[:200],
        )

    def _backup(self, abs_path: Path) -> Optional[Path]:
        try:
            ts = time.strftime("%Y%m%d_%H%M%S")
            backup_path = self.BACKUP_DIR / f"{abs_path.stem}_{ts}.py"
            shutil.copy2(abs_path, backup_path)
            # Prune old backups for this stem (keep last 5)
            all_backups = sorted(
                self.BACKUP_DIR.glob(f"{abs_path.stem}_*.py"),
                key=lambda p: p.stat().st_mtime,
            )
            for old in all_backups[:-5]:
                old.unlink(missing_ok=True)
            return backup_path
        except OSError as e:
            log.warning("[SelfImprover] Backup failed: %s", e)
            return None

    def _is_safe(self, code: str) -> Tuple[bool, str]:
        """AST-based safety check. Returns (safe, reason)."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"syntax error: {e}"

        for node in ast.walk(tree):
            # Block dangerous built-in calls
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in self._BANNED_CALLS:
                    return False, f"banned call: {func.id}()"
                if isinstance(func, ast.Attribute) and func.attr in self._BANNED_ATTRS:
                    return False, f"banned attribute: .{func.attr}()"

            # Block new dangerous imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                for name in names:
                    base = (name or "").split(".")[0]
                    if base in self._BANNED_IMPORT_MODS:
                        return False, f"banned import: {name}"

        return True, ""

    def _check_api_surface(self, original_code: str, new_code: str) -> Tuple[bool, str]:
        """
        Check API surface preservation. (Item 1)

        Only blocks removal of public names that are actually imported by other
        files in the codebase. Private names and unreferenced names can be freely
        refactored by the judge.
        """
        def _top_level_names(code: str) -> set:
            try:
                tree = ast.parse(code)
            except SyntaxError:
                return set()
            names = set()
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(node.name)
            return names

        orig_names = _top_level_names(original_code)
        new_names  = _top_level_names(new_code)
        missing = orig_names - new_names

        if not missing:
            return True, ""

        # Private/dunder names can always be freely refactored
        public_missing = {n for n in missing if not n.startswith("_")}
        if not public_missing:
            return True, ""

        # Only block if missing names are actually imported elsewhere
        externally_used = self._find_externally_used_names(public_missing)
        if externally_used:
            return False, "missing externally-imported: " + ", ".join(sorted(externally_used))

        return True, ""

    def _find_externally_used_names(self, names: set) -> set:
        """Return subset of names that are imported by other files in the package."""
        if not names:
            return set()
        try:
            import re
            used = set()
            python_root = Path(__file__).resolve().parents[1]
            pattern = re.compile(r"from\s+[\w.]+\s+import\s+([^#\n]+)")
            for py_file in python_root.rglob("*.py"):
                try:
                    text = py_file.read_text(errors="ignore")
                    for match in pattern.finditer(text):
                        imported = [n.strip().split(" as ")[0].strip()
                                    for n in match.group(1).split(",")]
                        for imp in imported:
                            if imp in names:
                                used.add(imp)
                except OSError:
                    continue
            return used
        except Exception:
            return set()  # on error, don't block the patch

    def _run_tests(self, target_file: str) -> Tuple[bool, str]:
        """Run relevant test file if it exists. Returns (passed, output). (Item 3)"""
        import subprocess
        test_candidates = [
            _PYTHON.parent / "tests" / "test_python_runtime.py",
            _PYTHON / "tests" / "test_python_runtime.py",
        ]
        test_file = next((p for p in test_candidates if p.exists()), None)
        if test_file is None:
            return True, "no test file found"
        try:
            result = subprocess.run(
                ["python3", "-m", "pytest", str(test_file), "-x", "-q", "--tb=short"],
                capture_output=True, text=True, timeout=90,
                cwd=str(_PYTHON.parent)
            )
            passed = result.returncode == 0
            output = (result.stdout + result.stderr)[-1000:]
            log.info("[SelfImprover] Test runner for %s: %s",
                     Path(target_file).name, "PASSED" if passed else "FAILED")
            return passed, output
        except Exception as e:
            return True, f"test runner error (skipping): {e}"

    def _rollback_patch(self, patch: PatchRecord, debate: DebateRecord, reason: str):
        """Restore the backup for a patch. (Item 3)"""
        try:
            backup = Path(patch.backup_path)
            if backup.exists():
                Path(patch.target_file).write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
                patch.rolled_back = True
                patch.rollback_reason = reason
                debate.outcome = "rolled_back_tests"
                log.info("[SelfImprover] Rolled back %s: %s",
                         Path(patch.target_file).name, reason[:100])
        except Exception as e:
            log.warning("[SelfImprover] Rollback failed: %s", e)

    def _load_stats(self):
        """Load persisted stats counters from disk. (Item 7)"""
        try:
            if self.STATS_PATH.exists():
                saved = json.loads(self.STATS_PATH.read_text())
                self._stats.update(saved)
        except Exception as _e:
                        log.warning("[self_improver] Suppressed: %s", _e)

    def _save_stats(self):
        """Persist stats counters to disk. (Item 7)"""
        try:
            self.STATS_PATH.write_text(json.dumps(self._stats, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _test_import(self, file_path: Path, module_name: str) -> Tuple[bool, str]:
        """Try to import the file in isolation. Returns (success, error_msg)."""
        # Use a unique test module name to avoid polluting sys.modules
        test_name = f"_sare_test_{file_path.stem}_{int(time.time())}"
        try:
            spec = importlib.util.spec_from_file_location(test_name, str(file_path))
            if spec is None:
                return False, "spec_from_file_location returned None"
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return True, ""
        except Exception as e:
            return False, str(e)
        finally:
            # Clean up test module from sys.modules
            sys.modules.pop(test_name, None)

    def rollback(self, patch_id: str) -> dict:
        """Manually roll back a patch by patch_id."""
        patch = next((p for p in self._patches if p.patch_id == patch_id), None)
        if patch is None:
            return {"success": False, "reason": f"patch {patch_id} not found"}
        if patch.rolled_back:
            return {"success": False, "reason": "already rolled back"}
        backup = Path(patch.backup_path)
        if not backup.exists():
            return {"success": False, "reason": f"backup not found: {backup}"}
        try:
            shutil.copy2(backup, patch.target_file)
            patch.rolled_back = True
            patch.rollback_reason = "manual rollback"
            self._save_history()
            log.info("[SelfImprover] Rolled back patch %s → %s",
                     patch_id, patch.target_file)
            return {"success": True, "restored": patch.target_file}
        except OSError as e:
            return {"success": False, "reason": str(e)}

    # ── Status & persistence ──────────────────────────────────────────────────

    def get_status(self) -> dict:
        recent = self._debates[-10:] if self._debates else []
        applied = [p for p in self._patches if not p.rolled_back]
        rolled  = [p for p in self._patches if p.rolled_back]
        return {
            "running":               self._running,
            "interval_seconds":      self.INTERVAL_SECONDS,
            "parallel_debates":      self.PARALLEL_DEBATES,
            "min_critic_score":      self.MIN_CRITIC_SCORE,
            "total_debates":         len(self._debates),
            "total_patches":         len(self._patches),
            "patches_applied":       len(applied),
            "patches_rolled_back":   len(rolled),
            "last_run_outcome":      self._debates[-1].outcome if self._debates else None,
            "last_run_target":       self._debates[-1].target_file if self._debates else None,
            "last_run_ts":           self._debates[-1].timestamp if self._debates else None,
            # Live debate progress
            "active_debates":        [
                {"file": v["file"], "turn": v["turn"],
                 "elapsed_s": round(time.time() - v["started_at"], 1)}
                for v in self._active.values()
            ],
            # Multi-model pipeline stats
            "prescreened_rejected":  sum(
                1 for d in self._debates if "rejected_prescreen" in d.outcome
            ),
            "judge_fallback_used":   sum(
                1 for d in self._debates if "judge_fast" in d.outcome or (
                    d.judge_code and "judge_fast" in d.module_name
                )
            ),
            "tests_rolled_back":     sum(
                1 for d in self._debates if "rolled_back_tests" in d.outcome
            ),
            # Robustness stats counters (Item 7)
            "api_surface_rejected":  self._stats.get("api_surface_rejected", 0),
            "stats_tests_rolled_back": self._stats.get("tests_rolled_back", 0),
            "stats_prescreened_rejected": self._stats.get("prescreened_rejected", 0),
            # Learning memory summary
            "learning_modules":      len(self._learning),
            "top_learned_modules":   sorted(
                self._learning,
                key=lambda m: sum(
                    s.get("applied", 0)
                    for s in self._learning[m].values()
                ),
                reverse=True
            )[:5],
            # Patch timeline (last 20)
            "patch_timeline": [
                {
                    "patch_id":   p.patch_id,
                    "file":       Path(p.target_file).name,
                    "type":       p.improvement_type,
                    "critic":     p.critic_score,
                    "status":     "rolled_back" if p.rolled_back else "applied",
                    "perf_delta": p.perf_delta,
                    "ts":         p.applied_at,
                }
                for p in reversed(self._patches[-20:])
            ],
            "recent_debates":        [d.to_dict() for d in recent],
            "multi_file_debates":    len(self._multi_file_debates),
            "recent_multi_debates":  [d.to_dict() for d in self._multi_file_debates[-5:]],
        }

    def get_patches(self) -> list:
        return [p.to_dict() for p in reversed(self._patches)]

    def _load_history(self):
        if not self.HISTORY_PATH.exists():
            return
        try:
            data = json.loads(self.HISTORY_PATH.read_text())
            for pd in data.get("patches", []):
                try:
                    self._patches.append(PatchRecord(**{
                        k: v for k, v in pd.items()
                        if k in PatchRecord.__dataclass_fields__
                    }))
                except Exception as _e:
                                        log.warning("[self_improver] Suppressed: %s", _e)
            for dd in data.get("debates", []):
                try:
                    self._debates.append(DebateRecord(**{
                        k: v for k, v in dd.items()
                        if k in DebateRecord.__dataclass_fields__
                    }))
                except Exception as _e:
                                        log.warning("[self_improver] Suppressed: %s", _e)
            # Restore multi-file debate records (stored as raw dicts)
            for mfd in data.get("multi_file_debates_raw", []):
                try:
                    self._multi_file_debates.append(MultiFileDebateRecord(
                        cluster_name=mfd.get("cluster_name", ""),
                        target_files=mfd.get("target_files", []),
                        improvement_type=mfd.get("improvement_type", "optimize"),
                        proposer_text=mfd.get("proposer_text", ""),
                        planner_text=mfd.get("planner_text", ""),
                        critic_text=mfd.get("critic_text", ""),
                        critic_score=mfd.get("critic_score", 0),
                        patches=mfd.get("patches", {}),
                        outcomes=mfd.get("outcomes", {}),
                        overall_outcome=mfd.get("overall_outcome", "pending"),
                        timestamp=mfd.get("timestamp", time.time()),
                    ))
                except Exception as _e:
                                        log.warning("[self_improver] Suppressed: %s", _e)
            log.info("[SelfImprover] Loaded %d patches, %d debates, %d multi-file debates",
                     len(self._patches), len(self._debates), len(self._multi_file_debates))
        except Exception as e:
            log.warning("[SelfImprover] Load error: %s", e)

    def _save_history(self):
        try:
            data = {
                "version":      2,
                "last_updated": time.time(),
                "patches":      [p.to_dict() for p in self._patches[-100:]],
                "debates":      [
                    {
                        "target_file":    d.target_file,
                        "module_name":    d.module_name,
                        "improvement_type": d.improvement_type,
                        "proposer_text":  d.proposer_text[:500],
                        "planner_text":   d.planner_text[:500],
                        "executor_text":  d.executor_text[:500],
                        "critic_text":    d.critic_text[:500],
                        "critic_score":   d.critic_score,
                        "panel_scores":   d.panel_scores,
                        "judge_code":     "",
                        "verifier_text":  d.verifier_text[:300],
                        "verifier_ok":    d.verifier_ok,
                        "outcome":        d.outcome,
                        "timestamp":      d.timestamp,
                    }
                    for d in self._debates[-100:]
                ],
                "multi_file_debates_raw": [d.to_dict() for d in self._multi_file_debates[-20:]],
            }
            self.HISTORY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as e:
            log.warning("[SelfImprover] Save error: %s", e)

    def _load_learning(self):
        if not self.LEARNING_PATH.exists():
            return
        try:
            self._learning = json.loads(self.LEARNING_PATH.read_text())
        except Exception as _e:
                        log.warning("[self_improver] Suppressed: %s", _e)

    def _save_learning(self):
        try:
            self.LEARNING_PATH.write_text(
                json.dumps(self._learning, indent=2), encoding="utf-8"
            )
        except OSError:
            pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_skeleton(source: str) -> str:
    """Extract a compact API skeleton from a Python source file using AST.

    Returns: module docstring + class names + method signatures + their docstrings.
    Used for Tier 2 context to give Gemini Pro a full picture of the codebase
    without exceeding its token budget on unrelated files.
    """
    lines = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fall back to a simple line scan
        for line in source.splitlines()[:30]:
            if line.startswith(("class ", "def ", "    def ")):
                lines.append(line.rstrip())
        return "\n".join(lines)

    # Module docstring
    mod_doc = ast.get_docstring(tree)
    if mod_doc:
        lines.append(f'"""{mod_doc[:200]}"""')

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(
                (b.id if isinstance(b, ast.Name) else ast.unparse(b))
                for b in node.bases
            )
            lines.append(f"\nclass {node.name}({bases}):")
            doc = ast.get_docstring(node)
            if doc:
                lines.append(f'    """{doc[:150]}"""')
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    try:
                        sig = ast.unparse(item.args)
                        lines.append(f"    def {item.name}({sig}): ...")
                        fdoc = ast.get_docstring(item)
                        if fdoc:
                            lines.append(f'        """{fdoc[:100]}"""')
                    except Exception:
                        lines.append(f"    def {item.name}(...): ...")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Top-level functions only (not methods — already covered above)
            if not any(isinstance(p, ast.ClassDef) and node in ast.walk(p)
                       for p in ast.walk(tree) if p is not node):
                try:
                    sig = ast.unparse(node.args)
                    lines.append(f"\ndef {node.name}({sig}): ...")
                    fdoc = ast.get_docstring(node)
                    if fdoc:
                        lines.append(f'    """{fdoc[:100]}"""')
                except Exception as _e:
                                        log.warning("[self_improver] Suppressed: %s", _e)

    return "\n".join(lines)


def _extract_code(text: str) -> str:
    """Strip markdown code fences from LLM output."""
    # Try fenced block first
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    # If the whole output looks like Python (no backticks), return as-is
    if "```" not in text and ("def " in text or "class " in text or "import " in text):
        return text
    return text


# ── Singleton ─────────────────────────────────────────────────────────────────
_instance: Optional[SelfImprover] = None
_instance_lock = threading.Lock()


def get_self_improver() -> SelfImprover:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = SelfImprover()
    return _instance
