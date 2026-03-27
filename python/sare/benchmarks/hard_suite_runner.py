"""HardSuiteRunner — Runs the 50-problem hard benchmark, tracks score history."""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

log = logging.getLogger(__name__)

_SUITE_PATH   = Path(__file__).resolve().parents[3] / "benchmarks" / "hard_suite" / "hard_problems.json"
_HISTORY_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "hard_suite_history.json"


class HardSuiteRunner:
    def __init__(self):
        self._suite   = self._load_suite()
        self._history = self._load_history()

    def _load_suite(self) -> dict:
        try:
            return json.loads(_SUITE_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("HardSuiteRunner: suite load failed: %s", e)
            return {"categories": {}}

    def _load_history(self) -> list:
        try:
            if _HISTORY_PATH.exists():
                return json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
        return []

    def run(self, solve_fn: Callable, budget_seconds: float = 5.0) -> dict:
        t0 = time.time()
        categories = self._suite.get("categories", {})
        cat_results: Dict[str, dict] = {}
        total_correct = 0
        total_problems = 0

        for cat, problems in categories.items():
            correct = 0
            details = []
            for prob in problems:
                expr      = prob.get("expression", "")
                min_delta = float(prob.get("expected_delta_min", 0.1))
                try:
                    r     = solve_fn(expr)
                    delta = float(r.get("delta", 0.0))
                    ok    = delta >= min_delta
                except Exception:
                    delta, ok = 0.0, False
                if ok:
                    correct += 1
                details.append({"id": prob.get("id"), "expression": expr,
                                 "delta": round(delta, 4), "success": ok,
                                 "description": prob.get("description", "")})
            cat_results[cat] = {"score": round(correct / max(len(problems), 1), 3),
                                 "correct": correct, "total": len(problems),
                                 "problems": details}
            total_correct  += correct
            total_problems += len(problems)

        overall = total_correct / max(total_problems, 1)
        result  = {
            "overall_score": round(overall, 3),
            "overall_pct":   f"{overall*100:.1f}%",
            "correct":       total_correct,
            "total":         total_problems,
            "elapsed_s":     round(time.time() - t0, 1),
            "categories":    cat_results,
            "timestamp":     time.time(),
        }

        self._history.append({"overall_score": overall, "correct": total_correct,
                               "total": total_problems, "timestamp": result["timestamp"]})
        if len(self._history) > 100:
            self._history = self._history[-50:]
        self._save_history()
        return result

    def trend(self) -> dict:
        if not self._history:
            return {"trend": [], "best": 0.0, "latest": 0.0, "runs": 0}
        scores = [h["overall_score"] for h in self._history[-10:]]
        return {"trend": scores, "best": round(max(scores), 3),
                "latest": round(scores[-1], 3), "runs": len(self._history)}

    def _save_history(self) -> None:
        try:
            _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _HISTORY_PATH.parent / f"hard_suite_history.{os.getpid()}.tmp"
            tmp.write_text(json.dumps(self._history, indent=2), encoding="utf-8")
            os.replace(tmp, _HISTORY_PATH)
        except Exception as e:
            log.debug("HardSuiteRunner save error: %s", e)


_instance: Optional[HardSuiteRunner] = None

def get_hard_suite_runner() -> HardSuiteRunner:
    global _instance
    if _instance is None:
        _instance = HardSuiteRunner()
    return _instance
