"""
SARE-HX Command Line Interface

Commands:
    sare solve "x + 0"          Solve an expression
    sare solve complex_simplify Use a named example
    sare solve -f problem.json  Load from file
    sare examples               List all built-in examples
    sare inspect "x * 1 + 0"   Inspect graph without solving
    sare demo                   Run all examples
    sare train                  Train heuristic model
"""

import argparse
import json
import sys
import time
from types import SimpleNamespace
from pathlib import Path

from sare.engine import (
    Graph, EnergyEvaluator, BeamSearch, MCTSSearch,
    ALL_TRANSFORMS, EXAMPLE_PROBLEMS,
    load_problem, load_problem_from_file, SearchResult, load_heuristic_scorer,
)
from sare.logging.logger import SareLogger, SolveLog

_BINDINGS_ERROR = None
try:
    import sare.sare_bindings as _sb  # type: ignore
    CppGraph = getattr(_sb, "Graph", None)  # type: ignore
    CppSearchConfig = getattr(_sb, "SearchConfig", None)  # type: ignore
    run_beam_search = getattr(_sb, "run_beam_search", None)  # type: ignore
    run_mcts_search = getattr(_sb, "run_mcts_search", None)  # type: ignore
except Exception as e:  # pragma: no cover
    CppGraph = None  # type: ignore
    CppSearchConfig = None  # type: ignore
    run_beam_search = None  # type: ignore
    run_mcts_search = None  # type: ignore
    _BINDINGS_ERROR = str(e)


# ── ANSI Colors ─────────────────────────────────────────────

class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    WHITE = "\033[97m"
    RESET = "\033[0m"
    CHECKMARK = "✓"
    ARROW = "→"
    BULLET = "•"
    LINE = "─"
    DOUBLE = "═"


def header():
    print(f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════════╗
║  SARE-HX  ·  Graph-Native Cognitive Architecture v0.1  ║
║  Structural Algebra of Realizable Expressions           ║
╚══════════════════════════════════════════════════════════╝{C.RESET}
""")


def separator(label: str = ""):
    if label:
        print(f"  {C.DIM}{C.LINE * 5}{C.RESET} {C.BOLD}{label}{C.RESET} {C.DIM}{C.LINE * (45 - len(label))}{C.RESET}")
    else:
        print(f"  {C.DIM}{C.LINE * 55}{C.RESET}")


def _graph_features(graph: Graph) -> tuple[list[str], list[tuple[int, int]]]:
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


def _py_graph_to_cpp_graph(py_graph: Graph):
    if not CppGraph:
        raise RuntimeError("C++ Graph binding unavailable")

    g = CppGraph()
    for n in py_graph.nodes:
        g.add_node_with_id(int(n.id), str(n.type))
        cn = g.get_node(int(n.id))
        if not cn:
            continue
        cn.uncertainty = float(getattr(n, "uncertainty", 0.0))
        attrs = getattr(n, "attributes", None) or {}
        for k, v in attrs.items():
            cn.set_attribute(str(k), str(v))
        label = getattr(n, "label", "") or ""
        if label:
            cn.set_attribute("label", str(label))
            if n.type in ("constant", "literal"):
                cn.set_attribute("value", str(label))
            elif n.type == "variable":
                cn.set_attribute("name", str(label))
            elif n.type == "operator":
                op_map = {
                    "+": "add", "add": "add",
                    "-": "sub", "sub": "sub",
                    "*": "mul", "mul": "mul",
                    "/": "div", "div": "div",
                    "neg": "neg",
                }
                if label in op_map:
                    cn.set_attribute("op", op_map[label])

    for e in py_graph.edges:
        g.add_edge_with_id(int(e.id), int(e.source), int(e.target), str(e.relationship_type), 1.0)
    return g


def _cpp_graph_to_py_graph(cpp_graph) -> Graph:
    g = Graph()
    id_map = {}
    op_to_label = {"add": "+", "mul": "*", "sub": "-", "div": "/", "neg": "neg"}

    for nid in cpp_graph.get_node_ids():
        n = cpp_graph.get_node(nid)
        if not n:
            continue
        ntype = str(getattr(n, "type", "")) or "unknown"
        label = n.get_attribute("label", "")
        if not label:
            if ntype in ("constant", "literal"):
                label = n.get_attribute("value", "")
            elif ntype == "variable":
                label = n.get_attribute("name", "")
            elif ntype == "operator":
                label = op_to_label.get(n.get_attribute("op", ""), "")

        attrs = {}
        for k in ("label", "value", "name", "op"):
            v = n.get_attribute(k, "")
            if v:
                attrs[k] = v

        py_id = g.add_node(ntype, label=label or "", attributes=(attrs or None))
        id_map[int(nid)] = py_id
        pn = g.get_node(py_id)
        if pn:
            pn.uncertainty = float(getattr(n, "uncertainty", 0.0))

    for eid in cpp_graph.get_edge_ids():
        e = cpp_graph.get_edge(eid)
        if not e:
            continue
        s = id_map.get(int(e.source))
        t = id_map.get(int(e.target))
        if s is not None and t is not None:
            g.add_edge(s, t, str(e.relationship_type))

    return g


def _solve_with_cpp_bindings(graph: Graph, algorithm: str, beam_width: int, max_depth: int,
                             budget: float, kappa: float):
    if not (CppGraph and CppSearchConfig and run_beam_search and run_mcts_search):
        return None

    cpp_graph = _py_graph_to_cpp_graph(graph)
    cfg = CppSearchConfig()
    cfg.beam_width = int(beam_width)
    cfg.max_depth = int(max_depth)
    cfg.budget_seconds = float(budget)
    cfg.kappa = float(kappa)

    cpp_result = run_mcts_search(cpp_graph, cfg) if algorithm == "mcts" else run_beam_search(cpp_graph, cfg)
    best_state = cpp_result.best_state
    best_energy = best_state.energy
    components = {
        "syntax": best_energy.syntax,
        "constraint": best_energy.constraint,
        "test_failure": best_energy.test_failure,
        "complexity": best_energy.complexity,
        "resource": best_energy.resource,
        "uncertainty": best_energy.uncertainty,
    }
    return {
        "graph": _cpp_graph_to_py_graph(cpp_result.best_graph),
        "energy_total": float(best_energy.total()),
        "energy_components": components,
        "transforms": list(best_state.transform_trace),
        "steps": int(best_state.depth if best_state.depth else len(best_state.transform_trace)),
        "expansions": int(cpp_result.total_expansions),
        "elapsed": float(cpp_result.elapsed_seconds),
        "trajectory": [float(best_energy.total())],
    }


# ── Solve Command ───────────────────────────────────────────

def cmd_solve(args):
    header()

    # Load problem
    if args.file:
        expr, graph = load_problem_from_file(args.file)
    elif args.problem:
        expr, graph = load_problem(args.problem)
    else:
        print(f"  {C.RED}Error: provide a problem expression or --file{C.RESET}")
        return 1

    energy = EnergyEvaluator()
    initial_energy = energy.compute(graph)

    separator("Problem")
    print(f"  {C.BULLET} Expression: {C.BOLD}{expr}{C.RESET}")
    print(f"  {C.BULLET} Graph:      {C.CYAN}{graph.node_count}{C.RESET} nodes, {C.CYAN}{graph.edge_count}{C.RESET} edges")
    print(f"  {C.BULLET} Energy:     {C.YELLOW}{initial_energy.total:.3f}{C.RESET}")
    for comp, val in initial_energy.components.items():
        if val > 0:
            print(f"    {C.DIM}{C.ARROW} {comp}: {val:.3f}{C.RESET}")
    print()

    separator("Graph Structure")
    print(f"  {graph.pretty_print()}")
    print()

    separator("Search")
    search_algo = args.algorithm if hasattr(args, 'algorithm') and args.algorithm else "beam"
    heuristic_fn = load_heuristic_scorer()
    print(f"  {C.BULLET} Algorithm:  {C.CYAN}{search_algo}{C.RESET}")
    print(f"  {C.BULLET} Beam width: {C.CYAN}{args.beam_width}{C.RESET}")
    print(f"  {C.BULLET} Max depth:  {C.CYAN}{args.max_depth}{C.RESET}")
    print(f"  {C.BULLET} Budget:     {C.CYAN}{args.budget}s{C.RESET}")
    print(f"  {C.BULLET} Kappa:      {C.CYAN}{args.kappa:.2f}{C.RESET}")
    print(f"  {C.BULLET} Heuristic:  {C.CYAN}{'model' if heuristic_fn else 'none'}{C.RESET}")
    print(f"  {C.BULLET} Transforms: {C.CYAN}{len(ALL_TRANSFORMS)}{C.RESET} ({', '.join(t.name() for t in ALL_TRANSFORMS)})")
    print()

    # Run search
    cpp_result = _solve_with_cpp_bindings(
        graph=graph,
        algorithm=search_algo,
        beam_width=args.beam_width,
        max_depth=args.max_depth,
        budget=args.budget,
        kappa=args.kappa,
    )
    if cpp_result is not None:
        result = SimpleNamespace(
            graph=cpp_result["graph"],
            energy=SimpleNamespace(total=cpp_result["energy_total"], components=cpp_result["energy_components"]),
            transforms_applied=cpp_result["transforms"],
            steps_taken=cpp_result["steps"],
            expansions=cpp_result["expansions"],
            elapsed_seconds=cpp_result["elapsed"],
            energy_trajectory=cpp_result["trajectory"],
        )
    elif search_algo == "mcts":
        searcher = MCTSSearch()
        result = searcher.search(
            graph, energy, ALL_TRANSFORMS,
            iterations=args.max_depth * 10,
            budget_seconds=args.budget,
        )
    else:
        searcher = BeamSearch()
        result = searcher.search(
            graph, energy, ALL_TRANSFORMS,
            beam_width=args.beam_width,
            max_depth=args.max_depth,
            budget_seconds=args.budget,
            kappa=args.kappa,
            heuristic_fn=heuristic_fn,
        )

    # Display results
    separator("Result")
    delta = initial_energy.total - result.energy.total
    pct = (delta / initial_energy.total * 100) if initial_energy.total > 0 else 0

    if delta > 0.01:
        status_color = C.GREEN
        status_icon = C.CHECKMARK
        status_text = "SIMPLIFIED"
    elif delta < -0.01:
        status_color = C.RED
        status_icon = "✗"
        status_text = "WORSENED"
    else:
        status_color = C.YELLOW
        status_icon = "="
        status_text = "UNCHANGED"

    print(f"  {status_color}{C.BOLD}[{status_icon}] {status_text}{C.RESET}")
    print()
    print(f"  {C.BULLET} Energy:  {C.YELLOW}{initial_energy.total:.3f}{C.RESET} {C.ARROW} {status_color}{C.BOLD}{result.energy.total:.3f}{C.RESET}  ({status_color}{'-' if delta > 0 else '+'}{abs(delta):.3f}, {pct:.1f}%{C.RESET})")
    for comp, val in result.energy.components.items():
        if val > 0:
            print(f"    {C.DIM}{C.ARROW} {comp}: {val:.3f}{C.RESET}")
    print()

    if result.transforms_applied:
        print(f"  {C.BULLET} Transforms applied ({len(result.transforms_applied)} steps):")
        for i, t in enumerate(result.transforms_applied, 1):
            print(f"    {C.DIM}{i}.{C.RESET} {C.MAGENTA}{t}{C.RESET}")
    else:
        print(f"  {C.BULLET} No transforms applied (already optimal)")
    print()

    print(f"  {C.BULLET} Expansions: {C.CYAN}{result.expansions}{C.RESET}")
    print(f"  {C.BULLET} Time:       {C.CYAN}{result.elapsed_seconds:.3f}s{C.RESET}")
    print()

    separator("Simplified Graph")
    print(f"  {result.graph.pretty_print()}")
    print()

    # Energy trajectory sparkline
    if len(result.energy_trajectory) > 1:
        separator("Energy Trajectory")
        _print_trajectory(result.energy_trajectory)
        print()

    # Log
    if args.output:
        node_types, adjacency = _graph_features(graph)
        logger = SareLogger(args.output)
        logger.log(SolveLog(
            problem_id=expr,
            initial_energy=initial_energy.total,
            final_energy=result.energy.total,
            search_depth=result.steps_taken,
            transform_sequence=result.transforms_applied,
            compute_time_seconds=result.elapsed_seconds,
            total_expansions=result.expansions,
            solve_success=(delta > 0.01),
            node_types=node_types,
            adjacency=adjacency,
        ))
        print(f"  {C.DIM}Log written to {args.output}{C.RESET}")
        print()

    return 0


def _print_trajectory(trajectory: list):
    """Print a simple ASCII energy trajectory chart."""
    if not trajectory:
        return

    max_e = max(trajectory)
    min_e = min(trajectory)
    width = 40
    height = 6

    if max_e == min_e:
        print(f"  {C.DIM}(flat at {max_e:.3f}){C.RESET}")
        return

    # Normalize
    for row in range(height):
        threshold = max_e - (max_e - min_e) * row / (height - 1)
        line = "  "
        label = f"{threshold:6.2f} │ "
        line += f"{C.DIM}{label}{C.RESET}"
        for i, e in enumerate(trajectory):
            step = i * width // max(len(trajectory) - 1, 1) if len(trajectory) > 1 else 0
            if row == 0 and i == 0:
                pass  # header
            if e >= threshold:
                line += f"{C.GREEN}█{C.RESET}"
            else:
                line += " "
        print(line)

    # Bottom axis
    print(f"  {'':>8}└{'─' * len(trajectory)}")
    print(f"  {'':>8} {'step 0':}{' ' * max(0, len(trajectory) - 12)}{'step ' + str(len(trajectory)-1):>6}")



# ── Inspect Command ─────────────────────────────────────────

def cmd_inspect(args):
    header()
    expr, graph = load_problem(args.expression)
    energy = EnergyEvaluator()
    e = energy.compute(graph)

    separator("Inspection")
    print(f"  {C.BULLET} Expression: {C.BOLD}{expr}{C.RESET}")
    print(f"  {C.BULLET} Nodes:      {C.CYAN}{graph.node_count}{C.RESET}")
    print(f"  {C.BULLET} Edges:      {C.CYAN}{graph.edge_count}{C.RESET}")
    print(f"  {C.BULLET} Energy:     {C.YELLOW}{e.total:.3f}{C.RESET}")
    for comp, val in e.components.items():
        print(f"    {C.DIM}{C.ARROW} {comp}: {val:.3f}{C.RESET}")
    print()

    separator("Graph")
    print(f"  {graph.pretty_print()}")
    print()

    separator("Applicable Transforms")
    for t in ALL_TRANSFORMS:
        matches = t.match(graph)
        if matches:
            print(f"  {C.GREEN}{C.CHECKMARK}{C.RESET} {C.MAGENTA}{t.name()}{C.RESET} — {len(matches)} match(es)")
        else:
            print(f"  {C.DIM}  {t.name()} — no matches{C.RESET}")
    print()

    return 0


# ── Examples Command ────────────────────────────────────────

def cmd_examples(args):
    header()
    separator("Built-in Example Problems")
    print()
    for name, expr in EXAMPLE_PROBLEMS.items():
        energy = EnergyEvaluator()
        _, graph = load_problem(name)
        e = energy.compute(graph)
        print(f"  {C.CYAN}{name:20s}{C.RESET} {C.DIM}│{C.RESET} {expr:30s} {C.DIM}│{C.RESET} E={C.YELLOW}{e.total:.2f}{C.RESET}  {C.DIM}({graph.node_count}N, {graph.edge_count}E){C.RESET}")
    print()
    print(f"  {C.DIM}Usage: sare solve <name>  or  sare solve \"<expression>\"{C.RESET}")
    print()
    return 0


# ── Demo Command ────────────────────────────────────────────

def cmd_demo(args):
    header()
    print(f"  {C.BOLD}Running all examples...{C.RESET}")
    print()

    energy = EnergyEvaluator()
    searcher = BeamSearch()
    total_reduction = 0.0

    for name, expr in EXAMPLE_PROBLEMS.items():
        _, graph = load_problem(name)
        initial = energy.compute(graph)

        result = searcher.search(
            graph, energy, ALL_TRANSFORMS,
            beam_width=8, max_depth=20, budget_seconds=5.0
        )

        delta = initial.total - result.energy.total
        total_reduction += delta
        pct = (delta / initial.total * 100) if initial.total > 0 else 0

        if delta > 0.01:
            icon = f"{C.GREEN}{C.CHECKMARK}{C.RESET}"
        else:
            icon = f"{C.YELLOW}={C.RESET}"

        transforms_str = " → ".join(result.transforms_applied[:3])
        if len(result.transforms_applied) > 3:
            transforms_str += f" (+{len(result.transforms_applied)-3})"

        print(f"  {icon} {C.CYAN}{name:20s}{C.RESET} {C.YELLOW}{initial.total:7.2f}{C.RESET} {C.ARROW} {C.GREEN}{result.energy.total:7.2f}{C.RESET}  {C.DIM}({pct:+.0f}%){C.RESET}  {C.MAGENTA}{transforms_str}{C.RESET}")

    print()
    separator("Summary")
    print(f"  {C.BULLET} Total energy reduction: {C.GREEN}{C.BOLD}{total_reduction:.2f}{C.RESET}")
    print(f"  {C.BULLET} Examples solved: {C.CYAN}{len(EXAMPLE_PROBLEMS)}{C.RESET}")
    print()
    return 0


# ── Train Command ───────────────────────────────────────────

def cmd_train(args):
    header()
    separator("Heuristic Training")

    try:
        from sare.learning.credit_assignment import CreditAssigner
        print(f"  {C.GREEN}{C.CHECKMARK}{C.RESET} Credit assignment module loaded")
    except ImportError:
        print(f"  {C.RED}Error: Could not import learning modules{C.RESET}")
        return 1

    # Generate training data from demo runs
    print(f"  {C.BULLET} Generating training data from built-in examples...")
    ca = CreditAssigner(alpha=0.1)
    energy = EnergyEvaluator()
    searcher = BeamSearch()

    for name, expr in EXAMPLE_PROBLEMS.items():
        _, graph = load_problem(name)
        result = searcher.search(graph, energy, ALL_TRANSFORMS,
                                  beam_width=8, max_depth=20, budget_seconds=5.0)
        if result.transforms_applied and len(result.energy_trajectory) > 1:
            credits = ca.assign_credit(result.transforms_applied,
                                        result.energy_trajectory)
            print(f"    {C.DIM}{name}: {len(credits)} credits assigned{C.RESET}")

    print()
    separator("Transform Utilities")
    utilities = ca.get_all_utilities()
    for name, u in sorted(utilities.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * max(1, int(abs(u) * 2))
        color = C.GREEN if u > 0 else C.RED
        print(f"  {color}{bar:15s}{C.RESET} {C.MAGENTA}{name:20s}{C.RESET} U={color}{u:+.3f}{C.RESET}")

    print()

    # Train and persist PyTorch heuristic model from solve logs.
    try:
        from sare.heuristics.trainer import HeuristicTrainer
        trainer = HeuristicTrainer()
        trainer.load_traces(args.log_file)
        if not trainer.samples:
            print(f"  {C.YELLOW}! No trace samples found in {args.log_file}{C.RESET}")
            print()
            return 0

        losses = trainer.train(epochs=args.epochs, lr=args.lr)
        out_path = Path(args.model_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(out_path))
        print(f"  {C.GREEN}{C.CHECKMARK}{C.RESET} Trained heuristic on {len(trainer.samples)} samples")
        print(f"  {C.BULLET} Final loss: {C.CYAN}{losses[-1]:.6f}{C.RESET}" if losses else f"  {C.BULLET} Final loss: {C.CYAN}n/a{C.RESET}")
        print(f"  {C.BULLET} Saved model: {C.CYAN}{out_path}{C.RESET}")
    except ImportError:
        print(f"  {C.YELLOW}! PyTorch not available. Credit assignment works, neural model skipped.{C.RESET}")

    print()
    return 0


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SARE-HX: Graph-Native Cognitive Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sare solve "x + 0"              Simplify an expression
  sare solve complex_simplify     Use a named example
  sare solve -f problem.json      Load from file
  sare inspect "(x + 0) * 1"     Inspect graph structure
  sare examples                   List all examples
  sare demo                       Run all examples
  sare train                      Train heuristic model
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # solve
    solve_parser = subparsers.add_parser("solve", help="Solve/simplify an expression")
    solve_parser.add_argument("problem", type=str, nargs="?", default=None,
                               help="Expression or example name")
    solve_parser.add_argument("-f", "--file", type=str, default=None,
                               help="Load problem from JSON file")
    solve_parser.add_argument("-a", "--algorithm", type=str, default="beam",
                               choices=["beam", "mcts"],
                               help="Search algorithm (default: beam)")
    solve_parser.add_argument("-w", "--beam-width", type=int, default=8,
                               help="Beam width (default: 8)")
    solve_parser.add_argument("-d", "--max-depth", type=int, default=50,
                               help="Max search depth (default: 50)")
    solve_parser.add_argument("-b", "--budget", type=float, default=10.0,
                               help="Time budget in seconds (default: 10)")
    solve_parser.add_argument("--kappa", type=float, default=0.1,
                               help="Heuristic weight in beam scoring (default: 0.1)")
    solve_parser.add_argument("-o", "--output", type=str, default=None,
                               help="Output JSONL log file")
    solve_parser.add_argument("-v", "--verbose", action="store_true")

    # inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect graph structure")
    inspect_parser.add_argument("expression", type=str,
                                 help="Expression to inspect")

    # examples
    subparsers.add_parser("examples", help="List built-in examples")

    # demo
    subparsers.add_parser("demo", help="Run all example problems")

    # train
    train_parser = subparsers.add_parser("train", help="Train heuristic model")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--log-file", type=str, default="logs/solves.jsonl")
    train_parser.add_argument("--model-out", type=str, default="models/heuristic_v1.pt")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "solve": cmd_solve,
        "inspect": cmd_inspect,
        "examples": cmd_examples,
        "demo": cmd_demo,
        "train": cmd_train,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
