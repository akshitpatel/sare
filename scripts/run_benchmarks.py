#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from sare.engine import (  # noqa: E402
    BeamSearch,
    EnergyEvaluator,
    get_transforms,
    load_problem,
)


def _load_suite(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_case(case: dict, beam_width: int, max_depth: int, budget: float) -> dict:
    expression = case.get("expression", "")
    if not expression:
        return {
            "id": case.get("id", "unknown"),
            "status": "failed",
            "reason": "missing expression",
        }

    _, graph = load_problem(expression)
    energy = EnergyEvaluator()
    initial = energy.compute(graph)
    search = BeamSearch()
    result = search.search(
        graph,
        energy,
        get_transforms(include_macros=True),
        beam_width=beam_width,
        max_depth=max_depth,
        budget_seconds=budget,
    )

    final_total = result.energy.total
    initial_total = initial.total
    delta = initial_total - final_total
    reduction_pct = (delta / initial_total * 100.0) if initial_total > 0 else 0.0

    failures: list[str] = []
    min_reduction = case.get("min_reduction_pct")
    if min_reduction is not None and reduction_pct < float(min_reduction):
        failures.append(f"reduction_pct<{min_reduction}")

    max_final = case.get("max_final_energy")
    if max_final is not None and final_total > float(max_final):
        failures.append(f"final_energy>{max_final}")

    max_increase = case.get("max_energy_increase")
    increase = max(0.0, final_total - initial_total)
    if max_increase is not None and increase > float(max_increase):
        failures.append(f"energy_increase>{max_increase}")

    return {
        "id": case.get("id", expression),
        "expression": expression,
        "initial_energy": round(initial_total, 6),
        "final_energy": round(final_total, 6),
        "reduction_pct": round(reduction_pct, 3),
        "transforms": result.transforms_applied,
        "status": "passed" if not failures else "failed",
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SARE benchmark and verifier suites.")
    parser.add_argument("--benchmarks-dir", default=str(REPO_ROOT / "benchmarks"))
    parser.add_argument("--output", default=str(REPO_ROOT / "logs" / "benchmark_report.json"))
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=40)
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--strict", action="store_true", default=False)
    args = parser.parse_args()

    benchmarks_dir = Path(args.benchmarks_dir)
    suite_paths = sorted(benchmarks_dir.rglob("*.json"))
    if not suite_paths:
        print(f"No benchmark suites found in {benchmarks_dir}")
        return 1

    suites_report = []
    total_cases = 0
    failed_cases = 0

    for suite_path in suite_paths:
        suite = _load_suite(suite_path)
        suite_name = suite.get("suite", suite_path.stem)
        cases = suite.get("cases", [])
        case_reports = []
        for case in cases:
            case_result = _run_case(case, args.beam_width, args.max_depth, args.budget)
            case_reports.append(case_result)
            total_cases += 1
            if case_result["status"] == "failed":
                failed_cases += 1

        suites_report.append(
            {
                "suite": suite_name,
                "path": str(suite_path.relative_to(REPO_ROOT)),
                "description": suite.get("description", ""),
                "total_cases": len(case_reports),
                "failed_cases": sum(1 for c in case_reports if c["status"] == "failed"),
                "cases": case_reports,
            }
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": total_cases,
        "failed_cases": failed_cases,
        "pass_rate": round((total_cases - failed_cases) / total_cases * 100.0, 2) if total_cases else 0.0,
        "suites": suites_report,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote benchmark report to {output_path}")
    print(f"Total cases: {total_cases}, failed: {failed_cases}")

    if failed_cases and args.strict:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
