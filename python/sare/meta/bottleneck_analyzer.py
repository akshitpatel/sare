"""
BottleneckAnalyzer — identifies which SARE source files most need improvement.

Reads persisted memory files (self_model.json, synth_attempts.json, etc.) and
scores eligible Python source files by how urgently they need work.

Returns a ranked List[ImprovementTarget] — highest score = most urgent.
"""
from __future__ import annotations

import json
import os
import threading as _thr
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_ROOT   = Path(__file__).resolve().parents[3]     # /Users/.../sare
_PYTHON = _ROOT / "python"
_MEMORY = _ROOT / "data" / "memory"


@dataclass
class ImprovementTarget:
    """One candidate file for self-improvement."""
    file_path: str            # absolute path
    module_name: str          # e.g. "sare.causal.induction"
    score: float              # 0-1, higher = more urgent
    reason: str               # human-readable
    improvement_type: str     # "optimize" | "extend" | "fix"
    evidence: dict = field(default_factory=dict)
    domain: str = "general"

    def to_dict(self) -> dict:
        return {
            "file_path":        self.file_path,
            "module_name":      self.module_name,
            "score":            round(self.score, 3),
            "reason":           self.reason,
            "improvement_type": self.improvement_type,
            "evidence":         self.evidence,
            "domain":           self.domain,
        }


# ── Domain → source file mapping ─────────────────────────────────────────────
_DOMAIN_FILE_MAP: Dict[str, str] = {
    "arithmetic":  "sare/curiosity/curriculum_generator.py",
    "algebra":     "sare/curiosity/experiment_runner.py",
    "logic":       "sare/transforms/logic_transforms.py",
    "calculus":    "sare/transforms/code_transforms.py",
    "causal":      "sare/causal/induction.py",
    "general":     "sare/meta/self_model.py",
    "planning":    "sare/agent/domains/task_scheduler_domain.py",
    "code":        "sare/transforms/code_transforms.py",
    "qa":          "sare/agent/qa_pipeline.py",
    "analogy":     "sare/causal/analogy_transfer.py",
    "memory":      "sare/memory/memory_manager.py",
    "learning":    "sare/learning/credit_assignment.py",
    "social":      "sare/social/dialogue_manager.py",
    "language":    "sare/interface/nl_parser_v2.py",
    "transfer":    "sare/transfer/synthesizer.py",
    "world":       "sare/world/predictive_loop.py",
    "neuro":       "sare/neuro/algorithm_inventor.py",
    "curriculum":  "sare/curiosity/multi_agent_learner.py",
}

# Files that must never be modified by self-improver
_EXCLUDE_FILES = {
    "web.py",
    "__init__.py",
    "sare_bindings.cpython-313-darwin.so.bak",
}
_EXCLUDE_DIRS = {"static", "__pycache__", ".git", "synthesized_modules", "code_backups"}
_MAX_LINES = 2000


class BottleneckAnalyzer:
    """
    Scans data/memory/ files + sare source tree to find the best
    improvement targets.

    Usage::
        analyzer = BottleneckAnalyzer()
        targets  = analyzer.analyze()   # sorted best-first
        print(targets[0].reason)
    """

    REPORT_PATH = _MEMORY / "bottleneck_report.json"

    # Priority files for general improvement (always offered when no signal targets found)
    _PRIORITY_MODULES = [
        # Core reasoning
        "sare/causal/induction.py",
        "sare/causal/chain_detector.py",
        "sare/causal/analogy_transfer.py",
        # Learning
        "sare/learning/credit_assignment.py",
        "sare/curiosity/curriculum_generator.py",
        "sare/curiosity/multi_agent_learner.py",
        # Memory
        "sare/memory/concept_formation.py",
        "sare/memory/memory_manager.py",
        "sare/memory/concept_rule.py",
        "sare/memory/world_model.py",
        # Meta / self
        "sare/meta/temporal_identity.py",
        "sare/meta/self_model.py",
        "sare/meta/homeostasis.py",
        "sare/meta/conjecture_engine.py",
        # Agent / planning
        "sare/agent/qa_pipeline.py",
        "sare/agent/agent_society.py",
        "sare/agent/domains/task_scheduler_domain.py",
        # Transforms
        "sare/transforms/logic_transforms.py",
        "sare/transforms/code_transforms.py",
        # Social / language
        "sare/social/dialogue_manager.py",
        "sare/interface/nl_parser_v2.py",
        # Transfer
        "sare/transfer/synthesizer.py",
        # World
        "sare/world/predictive_loop.py",
        "sare/neuro/algorithm_inventor.py",
    ]

    def analyze(self) -> List[ImprovementTarget]:
        """Main entry: return ranked improvement targets."""
        perf = self._load_perf_data()
        eligible = self._scan_eligible_files()
        targets = []
        for path in eligible:
            t = self._score_file(path, perf)
            if t is not None:
                targets.append(t)
        targets.sort(key=lambda t: -t.score)

        # Fallback: if no signal-driven targets, suggest priority modules
        if not targets:
            targets = self._fallback_targets(eligible)

        self.save_report(targets)
        return targets

    def _fallback_targets(self, eligible: List[Path]) -> List[ImprovementTarget]:
        """When no performance signals fire, suggest known-good improvement targets."""
        result = []
        for rel_suffix in self._PRIORITY_MODULES:
            for path in eligible:
                rel = str(path.relative_to(_PYTHON))
                if rel.endswith(rel_suffix):
                    try:
                        lines = path.read_text(encoding="utf-8").count("\n")
                    except OSError:
                        lines = 999
                    score = max(0.1, 0.4 - lines / 1000.0)  # smaller = easier = slightly higher
                    result.append(ImprovementTarget(
                        file_path=str(path),
                        module_name=_path_to_module(str(path)),
                        score=score,
                        reason="general hygiene improvement opportunity",
                        improvement_type="optimize",
                        evidence={"lines": lines},
                    ))
                    break
        result.sort(key=lambda t: -t.score)
        return result

    # ── File scanning ─────────────────────────────────────────────────────────

    def _scan_eligible_files(self) -> List[Path]:
        """All .py files in sare/ that are < MAX_LINES and not excluded."""
        result = []
        for p in (_PYTHON / "sare").rglob("*.py"):
            # Skip excluded dirs
            if any(ex in p.parts for ex in _EXCLUDE_DIRS):
                continue
            if p.name in _EXCLUDE_FILES:
                continue
            # Skip very large files (too risky to auto-patch)
            try:
                lines = p.read_text(encoding="utf-8").count("\n")
                if lines > _MAX_LINES:
                    continue
            except OSError:
                continue
            result.append(p)
        return result

    # ── Performance data loading ──────────────────────────────────────────────

    def _load_perf_data(self) -> dict:
        """Load and merge all relevant memory files into one perf dict."""
        perf = {
            "domain_solve_rates":   {},   # domain → float
            "low_utility_transforms": [],
            "synth_failed_domains": [],   # domains where synth failed recently
            "error_modules":        {},   # module_stem → error_count
            "world_activity":       [],
        }
        self._read_self_model(perf)
        self._read_synth_attempts(perf)
        self._read_world_model(perf)
        return perf

    def _read_self_model(self, perf: dict):
        path = _MEMORY / "self_model.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for domain, info in data.get("domains", {}).items():
                # Prefer recent window rate (recent_successes/recent_attempts) over
                # all-time solve_rate, which inflates to 1.0 after many easy problems.
                recent_attempts = info.get("recent_attempts", 0)
                recent_successes = info.get("recent_successes", 0)
                if recent_attempts >= 5:
                    rate = recent_successes / recent_attempts
                else:
                    rate = info.get("solve_rate", 1.0)
                perf["domain_solve_rates"][domain] = rate
            perf["low_utility_transforms"] = data.get("low_utility", [])
        except Exception as e:
            log.debug("BottleneckAnalyzer._read_self_model: %s", e)

    def _read_synth_attempts(self, perf: dict):
        path = _MEMORY / "synth_attempts.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            now = time.time()
            # Count failures per domain with time-decay: only last 3h counts at full weight,
            # 3-24h counts at 0.25 weight.  Cap per-domain contribution at 3 to prevent
            # stale failures from monopolising the target list.
            domain_counts: dict = {}
            for attempt in data if isinstance(data, list) else []:
                age = now - attempt.get("timestamp", 0)
                if age > 86400:
                    continue
                if attempt.get("promoted", True):
                    continue
                dom = attempt.get("domain", "general")
                weight = 1.0 if age < 3 * 3600 else 0.25
                domain_counts[dom] = domain_counts.get(dom, 0.0) + weight
            # Expand into list but cap per domain at 3 to limit signal dominance
            for dom, count in domain_counts.items():
                for _ in range(min(3, int(count))):
                    perf["synth_failed_domains"].append(dom)
        except Exception as e:
            log.debug("BottleneckAnalyzer._read_synth_attempts: %s", e)

    def _read_world_model(self, perf: dict):
        path = _MEMORY / "world_model.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for entry in data.get("activity_log", [])[-200:]:
                if entry.get("event_type") == "error":
                    module = entry.get("domain", "")
                    perf["error_modules"][module] = (
                        perf["error_modules"].get(module, 0) + 1
                    )
        except Exception as e:
            log.debug("BottleneckAnalyzer._read_world_model: %s", e)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_file(self, path: Path, perf: dict) -> Optional[ImprovementTarget]:
        """Compute a priority score for one file."""
        score = 0.0
        reasons = []
        evidence = {}
        improvement_type = "optimize"
        domain = "general"

        stem = path.stem
        rel   = str(path.relative_to(_PYTHON))
        module = _path_to_module(str(path))

        # 1. Domain solve-rate signal (uses recent window rates, threshold=0.8)
        for dom, file_suffix in _DOMAIN_FILE_MAP.items():
            if rel.endswith(file_suffix):
                rate = perf["domain_solve_rates"].get(dom, 1.0)
                evidence[f"{dom}_solve_rate"] = rate
                if rate < 0.8:
                    contrib = (0.8 - rate) * 0.4
                    score += contrib
                    reasons.append(f"{dom} solve rate {rate:.0%}")
                    improvement_type = "fix" if rate < 0.4 else "optimize"
                    domain = dom

        # 2. Synthesis failure signal (capped at 3 per domain after decay)
        failed_count = perf["synth_failed_domains"].count(
            _file_to_domain(rel)
        )
        if failed_count > 1:
            score += min(failed_count * 0.08, 0.15)   # max 0.15, not 0.3
            reasons.append(f"{failed_count} synth failures (weighted)")
            evidence["synth_failures"] = failed_count
            improvement_type = "extend"

        # 3. Low-utility transform signal
        try:
            src = path.read_text(encoding="utf-8")
            low_ut = perf.get("low_utility_transforms", [])
            matched = [t for t in low_ut if t in src]
            if matched:
                score += min(len(matched) * 0.07, 0.2)
                reasons.append(f"defines {len(matched)} low-utility transforms")
                evidence["low_utility_transforms"] = matched
        except OSError:
            pass

        # 4. Error signal from world model activity log
        err_count = perf["error_modules"].get(stem, 0)
        if err_count > 0:
            score += min(err_count * 0.08, 0.25)
            reasons.append(f"{err_count} errors logged")
            evidence["error_count"] = err_count
            if err_count >= 3:
                improvement_type = "fix"

        # Only include files with a meaningful score
        if score < 0.05:
            return None

        return ImprovementTarget(
            file_path=str(path),
            module_name=module,
            score=min(score, 1.0),
            reason="; ".join(reasons) if reasons else "general hygiene",
            improvement_type=improvement_type,
            evidence=evidence,
            domain=domain,
        )

    def generate_capability_gap_report(self) -> dict:
        """
        Identify capability gaps for the ArchitectureDesigner.

        Finds:
          (a) Domains with consistently low solve rate (< 30%)
          (b) EventBus event types that never fire (unused capabilities)
          (c) Systematic blind spots in world model predictions

        Returns a dict with a "capability_gaps" list.
        """
        gaps = []

        # (a) Low solve-rate domains
        try:
            solve_rates_path = _MEMORY / "progress.json"
            if solve_rates_path.exists():
                data = json.loads(solve_rates_path.read_text())
                for domain, stats in data.items():
                    if isinstance(stats, dict):
                        attempts = stats.get("attempts", 0)
                        successes = stats.get("successes", 0)
                        if attempts >= 10:
                            sr = successes / max(attempts, 1)
                            if sr < 0.30:
                                gaps.append({
                                    "gap_type": "low_solve_rate",
                                    "description": f"Domain '{domain}' has solve_rate={sr:.0%} over {attempts} attempts",
                                    "domain": domain,
                                    "severity": 1.0 - sr,
                                    "evidence": {"attempts": attempts, "solve_rate": round(sr, 3)},
                                })
        except Exception:
            pass

        # (b) Check EventBus for events that never fire
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and hasattr(brain, "events"):
                from sare.brain import Event
                history_events = {e.event for e in brain.events.recent(1000)}
                all_events = set(Event)
                never_fired = all_events - history_events
                for evt in never_fired:
                    if "SLEEP" in evt.value or "BOOTED" in evt.value:
                        continue  # Expected to be rare
                    gaps.append({
                        "gap_type": "unused_capability",
                        "description": f"Event '{evt.value}' has never fired — capability may be disconnected",
                        "domain": "meta",
                        "severity": 0.4,
                        "evidence": {"event": evt.value},
                    })
        except Exception:
            pass

        # (c) World model blind spots: domains where avg_surprise is very high
        try:
            from sare.cognition.predictive_engine import get_predictive_engine
            pe = get_predictive_engine()
            status = pe.get_status()
            for domain, surprise in status.get("domain_avg_surprise", {}).items():
                if surprise > 4.0:
                    gaps.append({
                        "gap_type": "prediction_blind_spot",
                        "description": f"World model cannot predict outcomes in '{domain}' (avg_surprise={surprise:.2f})",
                        "domain": domain,
                        "severity": min(1.0, surprise / 8.0),
                        "evidence": {"avg_surprise": round(surprise, 3)},
                    })
        except Exception:
            pass

        # Sort by severity
        gaps.sort(key=lambda g: g.get("severity", 0), reverse=True)

        report = {
            "generated_at": time.time(),
            "capability_gaps": gaps[:15],
            "total_gaps_found": len(gaps),
        }

        # Persist
        try:
            gap_path = _MEMORY / "capability_gaps.json"
            tmp = gap_path.parent / f"{gap_path.stem}.{os.getpid()}.{_thr.get_ident()}.tmp"
            tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
            tmp.replace(gap_path)
        except OSError:
            pass

        return report

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_report(self, targets: List[ImprovementTarget]):
        try:
            self.REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "generated_at": time.time(),
                "targets": [t.to_dict() for t in targets[:20]],
            }
            self.REPORT_PATH.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except OSError as e:
            log.warning("BottleneckAnalyzer.save_report: %s", e)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _path_to_module(file_path: str) -> str:
    """python/sare/causal/induction.py → sare.causal.induction
    Handles the case where 'sare' appears twice in the path (repo dir + package dir).
    """
    p = Path(file_path)
    parts = list(p.with_suffix("").parts)
    # Find the 'sare' that follows 'python' (the package root, not the repo dir)
    for i, part in enumerate(parts):
        if part == "python" and i + 1 < len(parts) and parts[i + 1] == "sare":
            return ".".join(parts[i + 1:])
    # Fallback: use last occurrence of 'sare'
    indices = [i for i, p in enumerate(parts) if p == "sare"]
    if indices:
        return ".".join(parts[indices[-1]:])
    return p.stem


def _file_to_domain(rel_path: str) -> str:
    """Reverse lookup: file path → domain name."""
    for dom, suffix in _DOMAIN_FILE_MAP.items():
        if rel_path.endswith(suffix):
            return dom
    return "general"


# ── Singleton ─────────────────────────────────────────────────────────────────
_analyzer: Optional[BottleneckAnalyzer] = None


def get_bottleneck_analyzer() -> BottleneckAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = BottleneckAnalyzer()
    return _analyzer
