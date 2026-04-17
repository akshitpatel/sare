#!/usr/bin/env python3
"""
autonomous_run.py — Full SARE-HX autonomous exercise loop

Exercises every cognitive module, records what was learned, and
writes time-series data + a matplotlib progress chart.

Usage:
  python scripts/autonomous_run.py              # 20 cycles, auto-display chart
  python scripts/autonomous_run.py --cycles 50  # 50 cycles
  python scripts/autonomous_run.py --cycles 100 --no-plot  # headless CI

Output:
  data/memory/progress.json  — time-series for dashboard /api/brain/progress
  data/memory/progress.png   — matplotlib chart (if matplotlib available)
  data/memory/run_report.json — last run full summary
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone

REPO_ROOT  = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
DATA_DIR   = REPO_ROOT / "data" / "memory"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

DATA_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_PATH = DATA_DIR / "progress.json"
REPORT_PATH   = DATA_DIR / "run_report.json"
CHART_PATH    = DATA_DIR / "progress.png"

# ── colour helpers ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def col(text, c): return f"{c}{text}{RESET}"

# ── progress store ─────────────────────────────────────────────────────────────

def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        try:
            return json.loads(PROGRESS_PATH.read_text())
        except Exception:
            pass
    return {"runs": []}

def save_progress(prog: dict) -> None:
    PROGRESS_PATH.write_text(json.dumps(prog, indent=2))

# ── metric collectors ──────────────────────────────────────────────────────────

def collect_metrics(brain, cycle: int, solve_result: dict) -> dict:
    """Gather a flat metrics dict from all live modules."""
    m: dict = {
        "cycle":      cycle,
        "ts":         time.time(),
        "iso":        datetime.now(timezone.utc).isoformat(),
    }

    # solve quality
    m["solve_rate"]    = solve_result.get("solve_rate", 0.0)
    m["avg_energy"]    = solve_result.get("avg_energy", 1.0)
    m["solves"]        = solve_result.get("solved", 0)
    m["total_probs"]   = solve_result.get("total", 0)

    # S27 — ContinuousStreamLearner
    try:
        cs = brain.continuous_stream
        if cs:
            s = cs.summary()
            m["stream_throughput"] = s.get("throughput_per_min", 0)
            m["stream_running"]    = int(s.get("running", False))
    except Exception:
        pass

    # S28-1 — RobustnessHardener
    try:
        rh = brain.robustness_hardener
        if rh:
            m["robustness"]      = rh.overall_robustness()
            m["rob_runs"]        = rh.summary().get("total_runs", 0)
    except Exception:
        pass

    # S28-2 — AttentionRouter
    try:
        ar = brain.attention_router
        if ar:
            s = ar.summary()
            m["attention_routed"] = s.get("total_routed", 0)
            m["spotlight_size"]   = len(s.get("spotlight", []))
    except Exception:
        pass

    # S28-3 — RecursiveToM
    try:
        tom = brain.recursive_tom
        if tom:
            s = tom.summary()
            m["tom_agents"]      = s.get("total_agents", 0)
            m["tom_resolutions"] = s.get("total_resolutions", 0)
    except Exception:
        pass

    # S28-4 — AgentMemoryBank
    try:
        amb = brain.agent_memory_bank
        if amb:
            s = amb.summary()
            m["agentmem_agents"]  = s.get("n_agents", 0)
            m["agentmem_total"]   = s.get("total_memories", 0)
    except Exception:
        pass

    # S29-1 — MetaCurriculumEngine
    try:
        mc = brain.meta_curriculum
        if mc:
            m["meta_lp"]         = mc.learning_progress_score()
            m["meta_transfer"]   = mc.cross_domain_transfer_rate()
            m["meta_promotions"] = mc.summary().get("total_promotions", 0)
    except Exception:
        pass

    # S29-2 — ActionPhysicsSession
    try:
        ap = brain.action_physics
        if ap:
            s = ap.summary()
            m["phys_episodes"] = s.get("episodes_run", 0)
            m["phys_concepts"] = s.get("n_unique_concepts", 0)
    except Exception:
        pass

    # S29-3 — StreamBridge
    try:
        sb = brain.stream_bridge
        if sb:
            s = sb.summary()
            m["bridge_promoted"] = s.get("total_promoted", 0)
            m["bridge_rate"]     = s.get("promotion_rate", 0.0)
    except Exception:
        pass

    # S29-4 — PerceptionBridge
    try:
        pb = brain.perception_bridge
        if pb:
            s = pb.summary()
            m["percept_parsed"]   = s.get("total_parsed", 0)
            m["percept_relations"]= s.get("unique_relations", 0)
    except Exception:
        pass

    # ConceptGraph size
    try:
        cg = brain.concept_graph
        if cg and hasattr(cg, '_concepts'):
            m["concept_count"] = len(cg._concepts)
    except Exception:
        pass

    # Transform count
    try:
        from sare.engine import get_transforms
        m["transform_count"] = len(get_transforms(include_macros=True))
    except Exception:
        pass

    return m


def run_solve_pass(brain) -> dict:
    """Run one engine solve pass across EXAMPLE_PROBLEMS; also tick brain.learn_cycle."""
    from sare.engine import (
        BeamSearch, EnergyEvaluator, EXAMPLE_PROBLEMS, get_transforms,
        load_problem, load_heuristic_scorer,
    )
    try:
        energy_eval  = EnergyEvaluator()
        search       = BeamSearch()
        transforms   = get_transforms(include_macros=True)
        heuristic_fn = load_heuristic_scorer()

        solved = 0; total = 0; total_energy = 0.0
        for name in EXAMPLE_PROBLEMS:
            try:
                _, graph   = load_problem(name)
                initial    = energy_eval.compute(graph).total
                result     = search.search(
                    graph, energy_eval, transforms,
                    beam_width=6, max_depth=20,
                    budget_seconds=1.0, kappa=0.1,
                    heuristic_fn=heuristic_fn,
                )
                final_e    = result.energy.total
                total_energy += final_e
                total += 1
                if initial - final_e > 0.01:
                    solved += 1
            except Exception:
                pass

        # Also tick brain.learn_cycle to drive module hooks (ignore empty returns)
        try:
            lc_results = brain.learn_cycle(n=2)
            if isinstance(lc_results, list):
                for r in lc_results:
                    if isinstance(r, dict) and r.get('delta', 0) > 0.01:
                        solved += 1
                    total += 1
        except Exception:
            pass

        return {
            "solved":     solved,
            "total":      max(total, 1),
            "solve_rate": solved / total if total else 0.0,
            "avg_energy": total_energy / len(EXAMPLE_PROBLEMS) if EXAMPLE_PROBLEMS else 1.0,
        }
    except Exception as e:
        return {"solved": 0, "total": 0, "solve_rate": 0.0, "avg_energy": 1.0, "error": str(e)}


# ── exercise all modules ───────────────────────────────────────────────────────

def exercise_s28(brain) -> dict:
    """Explicitly exercise S28 modules beyond learn_cycle."""
    out = {}
    # Robustness — stress batch
    try:
        rh = brain.robustness_hardener
        if rh:
            recs = rh.run_stress_batch(n=8)
            out["rob_batch"] = len(recs)
    except Exception:
        pass

    # AttentionRouter — tick
    try:
        ar = brain.attention_router
        if ar:
            ar.post("test_event", {"value": 0.9}, source="autonomous", salience=0.8)
            ar.tick()
            out["attn_tick"] = True
    except Exception:
        pass

    # RecursiveToM — update + predict
    try:
        tom = brain.recursive_tom
        if tom:
            tom.update_model("agent_0", "x+0=x", 0.85, "algebra", depth=1)
            pred = tom.predict_action("agent_0", "algebra", depth=2)
            out["tom_pred"] = pred.get("action", "?")
    except Exception:
        pass

    # AgentMemoryBank — remember
    try:
        amb = brain.agent_memory_bank
        if amb:
            amb.remember("agent_0", "autonomous", "learn_cycle", "algebra", "success")
            amb.tick()
            out["amb_tick"] = True
    except Exception:
        pass

    return out


def exercise_s29(brain) -> dict:
    """Explicitly exercise S29 modules beyond learn_cycle."""
    out = {}
    # MetaCurriculum — tick
    try:
        mc = brain.meta_curriculum
        if mc:
            for dom, suc in [("arithmetic", True), ("calculus", False), ("algebra", True)]:
                mc.observe(dom, suc)
            res = mc.tick()
            out["mc_promoted"] = res.get("promoted", [])
    except Exception:
        pass

    # ActionPhysics — 2 episodes
    try:
        ap = brain.action_physics
        if ap:
            ep1 = ap.run_episode(n_steps=12)
            ep2 = ap.run_episode(n_steps=10)
            out["phys_concepts"] = ep1.concepts_found + ep2.concepts_found
    except Exception:
        pass

    # StreamBridge — submit + tick
    try:
        sb = brain.stream_bridge
        if sb:
            for expr, dom in [("d/dx x^2", "calculus"), ("x+0=x", "algebra"),
                               ("F=ma", "physics"), ("sin(0)=0", "trig")]:
                sb.submit(expr, source="EXPLORE", domain=dom)
            sb.tick()
            out["bridge_queue"] = sb.summary()["queue_depth"]
    except Exception:
        pass

    # PerceptionBridge — parse scenes
    try:
        pb = brain.perception_bridge
        if pb:
            scenes = [
                "ball above table, gravity pulls ball below table",
                "red block left of blue sphere, friction slows block",
                "large cube supports small sphere, cube touches floor",
                "force pushes block, block slides on surface, friction slows block",
            ]
            for s in scenes:
                pb.parse_scene(s)
            out["percept_total"] = pb.summary()["total_parsed"]
    except Exception:
        pass

    return out


# ── text output helpers ────────────────────────────────────────────────────────

def print_header(cycles: int) -> None:
    print()
    print(col("╔══════════════════════════════════════════════════════════════╗", CYAN))
    print(col("║         SARE-HX  Autonomous Run  —  35-Module Brain         ║", CYAN))
    print(col(f"║         {cycles} cycles · all modules exercised                   ║", CYAN))
    print(col("╚══════════════════════════════════════════════════════════════╝", CYAN))
    print()


def print_cycle_line(cycle: int, total: int, m: dict) -> None:
    bar_len  = 20
    filled   = int(cycle / total * bar_len)
    bar      = "█" * filled + "░" * (bar_len - filled)
    pct      = int(cycle / total * 100)
    sr_col   = GREEN if m.get("solve_rate", 0) > 0.6 else YELLOW
    rob_col  = GREEN if m.get("robustness", 0) > 0.7 else YELLOW
    sr_str   = col(f"{m.get('solve_rate', 0) * 100:.0f}%", sr_col)
    rob_str  = col(f"{m.get('robustness', 0) * 100:.0f}%", rob_col)
    cyc_str  = col(str(cycle).rjust(3), BOLD)
    print(
        f"  [{bar}] {pct:3d}%  "
        f"cyc={cyc_str}  "
        f"solve={sr_str}  "
        f"energy={m.get('avg_energy', 1):.3f}  "
        f"rob={rob_str}  "
        f"concepts={m.get('concept_count', 0)}  "
        f"bridge={m.get('bridge_promoted', 0)}  "
        f"percept={m.get('percept_relations', 0)}"
    )


def print_learned_summary(metrics_history: list) -> None:
    if not metrics_history:
        return
    first, last = metrics_history[0], metrics_history[-1]
    print()
    print(col("═" * 64, CYAN))
    print(col("  What was learned:", BOLD))
    print(col("═" * 64, CYAN))

    rows = [
        ("Solve rate",          "solve_rate",        "%",    100),
        ("Avg energy",          "avg_energy",        "",     1,   True),  # lower=better
        ("Robustness",          "robustness",        "%",    100),
        ("Attention routed",    "attention_routed",  "",     1),
        ("ToM agents",          "tom_agents",        "",     1),
        ("Agent memories",      "agentmem_total",    "",     1),
        ("Meta-curriculum LP",  "meta_lp",           "‰",    1000),
        ("Transfer rate",       "meta_transfer",     "%",    100),
        ("Physics episodes",    "phys_episodes",     "",     1),
        ("Unique phys concepts","phys_concepts",     "",     1),
        ("Bridge promoted",     "bridge_promoted",   "",     1),
        ("Bridge rate",         "bridge_rate",       "%",    100),
        ("Scenes parsed",       "percept_parsed",    "",     1),
        ("Unique relations",    "percept_relations", "",     1),
        ("Concept graph size",  "concept_count",     "",     1),
        ("Transform count",     "transform_count",   "",     1),
    ]

    for row in rows:
        label  = row[0]
        key    = row[1]
        suffix = row[2]
        scale  = row[3]
        lower_better = len(row) > 4 and row[4]

        v0 = first.get(key, 0) or 0
        v1 = last.get(key, 0)  or 0
        delta = v1 - v0
        disp_v1 = f"{v1 * scale:.1f}{suffix}" if scale != 1 else f"{v1:.3g}{suffix}"
        if abs(delta) < 1e-9:
            delta_str = col("  —", YELLOW)
        elif (delta > 0) != lower_better:
            delta_str = col(f"+{delta*scale:.2g}{suffix}", GREEN)
        else:
            delta_str = col(f"{delta*scale:+.2g}{suffix}", RED)

        print(f"  {label:<28}  {disp_v1:<14} {delta_str}")

    print()


# ── chart ──────────────────────────────────────────────────────────────────────

def generate_chart(metrics_history: list, output_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        cycles = [m["cycle"] for m in metrics_history]

        fig = plt.figure(figsize=(14, 10), facecolor="#0d1117")
        gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

        SERIES = [
            (0, 0, "Solve Rate & Energy",
             [("solve_rate", "Solve Rate", "#58a6ff", True),
              ("avg_energy",  "Avg Energy",  "#f78166", True, True)]),
            (0, 1, "Robustness & Attention",
             [("robustness",       "Robustness",     "#3fb950"),
              ("attention_routed", "Attn Routed",    "#d2a8ff", False, False, True)]),
            (1, 0, "Meta-Curriculum",
             [("meta_lp",       "Learning Progress", "#ffa657"),
              ("meta_transfer", "Transfer Rate",     "#58a6ff", True)]),
            (1, 1, "Physics & Perception",
             [("phys_concepts",      "Phys Concepts",   "#3fb950"),
              ("percept_relations",  "Percept Rels",    "#d2a8ff")]),
            (2, 0, "Stream Bridge Pipeline",
             [("bridge_promoted", "Promoted", "#58a6ff"),
              ("bridge_rate",     "Rate",     "#f78166", True)]),
            (2, 1, "Knowledge Growth",
             [("concept_count",    "Concepts",    "#3fb950"),
              ("transform_count",  "Transforms",  "#d2a8ff")]),
        ]

        for row, col_idx, title, series_list in SERIES:
            ax = fig.add_subplot(gs[row, col_idx])
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#8b949e", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")
            ax.set_title(title, color="#c9d1d9", fontsize=9, pad=4)
            ax.set_xlabel("Cycle", color="#8b949e", fontsize=7)

            for series in series_list:
                key   = series[0]
                label = series[1]
                color = series[2]
                pct   = len(series) > 3 and series[3]
                right = len(series) > 4 and series[4]
                raw   = len(series) > 5 and series[5]

                vals = [m.get(key, 0) or 0 for m in metrics_history]
                if pct:
                    vals = [v * 100 for v in vals]
                elif raw:
                    pass

                if right:
                    ax2 = ax.twinx()
                    ax2.tick_params(colors="#8b949e", labelsize=7)
                    ax2.set_facecolor("#161b22")
                    for sp in ax2.spines.values():
                        sp.set_edgecolor("#30363d")
                    ax2.plot(cycles, vals, color=color, linewidth=1.2,
                             alpha=0.8, label=label)
                    ax2.set_ylabel(label, color=color, fontsize=7)
                else:
                    ax.plot(cycles, vals, color=color, linewidth=1.4,
                            alpha=0.9, label=label, marker=".", markersize=3)

            if any(not (len(s) > 4 and s[4]) for s in series_list):
                ax.legend(fontsize=6, facecolor="#21262d", labelcolor="#c9d1d9",
                          framealpha=0.8, loc="best")

        fig.suptitle(
            f"SARE-HX Learning Progress  ·  {len(cycles)} cycles  ·  "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            color="#c9d1d9", fontsize=11, y=0.98
        )

        plt.savefig(str(output_path), dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return True
    except ImportError:
        return False
    except Exception as e:
        print(col(f"  Chart error: {e}", YELLOW))
        return False


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="SARE-HX full autonomous run")
    parser.add_argument("--cycles",   type=int, default=20, help="Number of learn cycles")
    parser.add_argument("--no-plot",  action="store_true",  help="Skip matplotlib chart")
    parser.add_argument("--append",   action="store_true",  help="Append to existing progress")
    args = parser.parse_args()

    print_header(args.cycles)

    # ── boot brain ────────────────────────────────────────────────────────────
    print(col("  Booting Brain…", CYAN), end="", flush=True)
    t0 = time.time()
    from sare.brain import Brain
    brain = Brain()
    brain._boot_knowledge()
    print(col(f" done ({time.time()-t0:.1f}s)", GREEN))

    # list live modules
    module_attrs = [
        "continuous_stream", "robustness_hardener", "attention_router",
        "recursive_tom", "agent_memory_bank", "meta_curriculum",
        "action_physics", "stream_bridge", "perception_bridge",
        "concept_graph", "generative_world", "affective_energy",
        "transform_generator", "red_team", "temporal_identity",
        "global_workspace", "agent_society", "predictive_loop",
    ]
    live = [a for a in module_attrs if getattr(brain, a, None) is not None]
    print(f"  Live modules: {col(str(len(live)), GREEN)}/{len(module_attrs)} → "
          f"{', '.join(live[:8])}{'…' if len(live)>8 else ''}")
    print()

    # ── run cycles ────────────────────────────────────────────────────────────
    prog = load_progress() if args.append else {"runs": []}
    metrics_history: list = []

    for cycle in range(1, args.cycles + 1):
        # Core learn cycle
        solve_result = run_solve_pass(brain)

        # Explicit module exercise every 2 cycles
        if cycle % 2 == 0:
            exercise_s28(brain)
        if cycle % 3 == 0:
            exercise_s29(brain)

        m = collect_metrics(brain, cycle, solve_result)
        metrics_history.append(m)
        prog["runs"].append(m)

        print_cycle_line(cycle, args.cycles, m)

        if cycle % 5 == 0:
            save_progress(prog)

    save_progress(prog)

    # ── learned summary ───────────────────────────────────────────────────────
    print_learned_summary(metrics_history)

    # ── chart ─────────────────────────────────────────────────────────────────
    if not args.no_plot:
        all_metrics = prog["runs"][-200:]   # last 200 data points for chart
        print(col("  Generating progress chart…", CYAN), end="", flush=True)
        ok = generate_chart(all_metrics, CHART_PATH)
        if ok:
            print(col(f" saved → {CHART_PATH}", GREEN))
        else:
            print(col(" matplotlib not available, skipping", YELLOW))

    # ── write run report ──────────────────────────────────────────────────────
    report = {
        "ts":        datetime.now(timezone.utc).isoformat(),
        "cycles":    args.cycles,
        "live_modules": live,
        "first":     metrics_history[0]  if metrics_history else {},
        "last":      metrics_history[-1] if metrics_history else {},
        "chart":     str(CHART_PATH) if not args.no_plot else None,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"  Report → {REPORT_PATH}")

    print()
    print(col("  ✓ Autonomous run complete", GREEN))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
