#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from sare.engine import (  # noqa: E402
    BeamSearch,
    EnergyEvaluator,
    EXAMPLE_PROBLEMS,
    get_transforms,
    load_heuristic_scorer,
    load_problem,
    reload_transforms,
)
from sare.learning.abstraction_learning import mine_frequent_patterns, propose_macros  # noqa: E402
from sare.logging.logger import SareLogger, SolveLog  # noqa: E402
from sare.meta.macro_registry import list_macros, macro_steps_set, upsert_macros  # noqa: E402


def graph_features(graph) -> tuple[list[str], list[tuple[int, int]]]:
    nodes = graph.nodes
    id_to_idx = {n.id: idx for idx, n in enumerate(nodes)}
    node_types = [n.type for n in nodes]
    adjacency: list[tuple[int, int]] = []
    for edge in graph.edges:
        src = id_to_idx.get(edge.source)
        tgt = id_to_idx.get(edge.target)
        if src is not None and tgt is not None:
            adjacency.append((src, tgt))
    return node_types, adjacency


def run_solve_round(log_path: Path, beam_width: int, max_depth: int, budget: float, kappa: float) -> dict:
    logger = SareLogger(str(log_path))
    energy = EnergyEvaluator()
    search = BeamSearch()
    transforms = get_transforms(include_macros=True)
    heuristic_fn = load_heuristic_scorer()

    solved = 0
    for name in EXAMPLE_PROBLEMS:
        expr, graph = load_problem(name)
        initial = energy.compute(graph)
        result = search.search(
            graph,
            energy,
            transforms,
            beam_width=beam_width,
            max_depth=max_depth,
            budget_seconds=budget,
            kappa=kappa,
            heuristic_fn=heuristic_fn,
        )
        delta = initial.total - result.energy.total
        if delta > 0.01:
            solved += 1
        node_types, adjacency = graph_features(graph)
        logger.log(
            SolveLog(
                problem_id=expr,
                initial_energy=initial.total,
                final_energy=result.energy.total,
                energy_breakdown=result.energy.components,
                search_depth=result.steps_taken,
                transform_sequence=result.transforms_applied,
                compute_time_seconds=result.elapsed_seconds,
                total_expansions=result.expansions,
                energy_trajectory=result.energy_trajectory,
                abstractions_used=[t for t in result.transforms_applied if t.startswith("macro_")],
                node_types=node_types,
                adjacency=adjacency,
                budget_exhausted=False,
                solve_success=(delta > 0.01),
            )
        )

    return {"episodes": len(EXAMPLE_PROBLEMS), "solved": solved}


def promote_macros_from_logs(log_path: Path) -> dict:
    logger = SareLogger(str(log_path))
    entries = logger.read_all()
    traces = [
        e.transform_sequence
        for e in entries
        if e.solve_success and isinstance(e.transform_sequence, list) and e.transform_sequence
    ]

    patterns = mine_frequent_patterns(traces, min_frequency=2, min_length=2, max_length=4)
    existing = list_macros()
    existing_steps = macro_steps_set(existing)
    proposed = propose_macros(patterns, existing_steps, max_new=5)

    from sare import engine as engine_mod

    base = engine_mod.get_transforms(include_macros=False)
    by_name = {t.name(): t for t in base}
    energy = EnergyEvaluator()

    promoted = []
    for spec in proposed:
        steps = []
        missing = False
        for step_name in spec.steps:
            t = by_name.get(step_name)
            if not t:
                missing = True
                break
            steps.append(t)
        if missing or len(steps) < 2:
            continue

        macro = engine_mod.MacroTransform(spec.name, steps)
        improved = 0
        applicable = 0
        total_delta = 0.0

        for ex_name in EXAMPLE_PROBLEMS:
            _, g = load_problem(ex_name)
            ctxs = macro.match(g)
            if not ctxs:
                continue
            applicable += 1
            before = energy.compute(g).total
            after_g, _ = macro.apply(g, ctxs[0])
            after = energy.compute(after_g).total
            delta = before - after
            total_delta += delta
            if delta > 0.01:
                improved += 1

        avg_delta = (total_delta / applicable) if applicable else 0.0
        if improved >= 2 and avg_delta >= 0.25:
            promoted.append(spec)

    upserted = upsert_macros(promoted)
    engine_mod.reload_transforms(include_macros=True)
    return {
        "patterns": len(patterns),
        "proposed": len(proposed),
        "promoted": len(promoted),
        "total_macros": len(upserted.get("macros", [])),
    }


def run_benchmark_gate(report_path: Path) -> bool:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_benchmarks.py"),
        "--output",
        str(report_path),
        "--strict",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        return False
    return True


def run_cycle(args) -> dict:
    macros_path = REPO_ROOT / "configs" / "abstractions.json"
    macros_backup = macros_path.with_suffix(".json.bak")
    if macros_path.exists():
        shutil.copy2(macros_path, macros_backup)

    solve_summary = run_solve_round(
        log_path=Path(args.log_path),
        beam_width=args.beam_width,
        max_depth=args.max_depth,
        budget=args.budget,
        kappa=args.kappa,
    )
    learning_summary = promote_macros_from_logs(Path(args.log_path))
    reload_transforms(include_macros=True)

    gate_ok = run_benchmark_gate(Path(args.report_path))
    if not gate_ok and macros_backup.exists():
        shutil.copy2(macros_backup, macros_path)
        reload_transforms(include_macros=True)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "solve": solve_summary,
        "learning": learning_summary,
        "benchmark_gate_passed": gate_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run autonomous SARE improvement loop.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=30)
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--kappa", type=float, default=0.1)
    parser.add_argument("--log-path", default=str(REPO_ROOT / "logs" / "solves.jsonl"))
    parser.add_argument("--report-path", default=str(REPO_ROOT / "logs" / "benchmark_report.json"))
    parser.add_argument("--cycle-report", default=str(REPO_ROOT / "logs" / "autonomy_cycle_report.jsonl"))
    args = parser.parse_args()

    report_path = Path(args.cycle_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    for index in range(args.iterations):
        summary = run_cycle(args)
        with report_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary) + "\n")
        print(
            f"cycle={index + 1} gate={summary['benchmark_gate_passed']} "
            f"solved={summary['solve']['solved']}/{summary['solve']['episodes']} "
            f"promoted={summary['learning']['promoted']}"
        )
        if index < args.iterations - 1 and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
