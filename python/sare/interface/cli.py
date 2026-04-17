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
        print(
            f"  {C.DIM}{C.LINE * 5}{C.RESET} {C.BOLD}{label}{C.RESET} "
            f"{C.DIM}{C.LINE * (45 - len(label))}{C.RESET}"
        )
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

    algorithm_l = (algorithm or "").strip().lower()
    start_t = time.time()

    try:
        if algorithm_l in ("beam", "beam_search", "beamsearch"):
            if not hasattr(_sb, "run_beam_search") and run_beam_search is None:
                return None
            cpp_result = run_beam_search(cpp_graph, cfg)
        elif algorithm_l in ("mcts", "mctss", "mcts_search", "montecarlo", "monte-carlo"):
            if not hasattr(_sb, "run_mcts_search") and run_mcts_search is None:
                return None
            cpp_result = run_mcts_search(cpp_graph, cfg)
        else:
            return None
    except Exception:
        return None

    elapsed = time.time() - start_t

    # Best-effort normalization into a Python SearchResult-like object.
    # We preserve the public ExperimentResult expectations by mirroring the fields
    # used in this CLI: solved, proof_steps, energy_before/after.
    solved = bool(getattr(cpp_result, "solved", False))
    proof_steps = getattr(cpp_result, "proof_steps", None) or getattr(cpp_result, "steps", None) or []
    energy_before = getattr(cpp_result, "energy_before", None)
    energy_after = getattr(cpp_result, "energy_after", None)

    out_graph = getattr(cpp_result, "graph", None) or getattr(cpp_result, "final_graph", None)
    if out_graph is not None:
        try:
            out_py_graph = _cpp_graph_to_py_graph(out_graph)
        except Exception:
            out_py_graph = None
    else:
        out_py_graph = None

    res = SearchResult(
        solved=solved,
        energy_before=float(energy_before) if energy_before is not None else None,
        energy_after=float(energy_after) if energy_after is not None else None,
        rule_promoted=getattr(cpp_result, "rule_promoted", None),
        proof_steps=proof_steps,
        proof_nl=getattr(cpp_result, "proof_nl", None),
        problem_graph=out_py_graph,
        elapsed_ms=int(elapsed * 1000),
    )
    return res


# ... truncated

def _default_examples():
    return EXAMPLE_PROBLEMS


def cmd_examples(args):
    header()
    separator("Built-in examples")
    for name in sorted(_default_examples().keys()):
        print(f"  {C.BOLD}{name}{C.RESET}")


def cmd_inspect(args):
    header()
    sep_label = f'Inspect graph (no solve) for: {args.expr}'
    separator(sep_label)

    expr, graph = load_problem(args.expr)
    node_types, adjacency = _graph_features(graph)

    print(f"{C.DIM}Expression{C.RESET}: {expr}")
    print(f"{C.DIM}Nodes ({len(graph.nodes)}){C.RESET}:")
    for i, t in enumerate(node_types):
        print(f"  {C.BULLET} #{i}: {t}")

    print(f"{C.DIM}Edges ({len(graph.edges)}){C.RESET}:")
    for (src, tgt) in adjacency[:200]:
        print(f"  {C.ARROW} {src} -> {tgt}")
    if len(adjacency) > 200:
        print(f"  {C.DIM}… ({len(adjacency) - 200} more){C.RESET}")


def cmd_train(args):
    header()
    separator("Training")
    scorer = load_heuristic_scorer()
    print(f"{C.DIM}Loaded heuristic scorer{C.RESET}: {type(scorer).__name__}")
    if hasattr(scorer, "train"):
        res = scorer.train()
        print(f"{C.CHECKMARK} Train result: {res}")
    else:
        print(f"{C.YELLOW}No 'train' method found on heuristic scorer.{C.RESET}")


def cmd_demo(args):
    header()
    separator("Demo")
    examples = _default_examples()
    names = sorted(examples.keys())
    max_n = int(getattr(args, "max_examples", 0) or 0)
    if max_n <= 0:
        max_n = len(names)

    for i, name in enumerate(names[:max_n], start=1):
        expr = examples[name]
        print(f"\n{C.BOLD}[{i}/{max_n}] {name}{C.RESET}: {expr}")
        class _A: pass
        a = _A()
        a.expr = expr
        a.algorithm = getattr(args, "algorithm", "beam")
        a.beam_width = getattr(args, "beam_width", 6)
        a.max_depth = getattr(args, "max_depth", 16)
        a.budget = getattr(args, "budget", 3.0)
        a.kappa = getattr(args, "kappa", 1.2)
        a.no_cpp = getattr(args, "no_cpp", False)
        a.print_proof = True
        cmd_solve(a)


def cmd_solve(args):
    header()

    algorithm = (getattr(args, "algorithm", "beam") or "beam").strip().lower()
    beam_width = int(getattr(args, "beam_width", 6))
    max_depth = int(getattr(args, "max_depth", 16))
    budget = float(getattr(args, "budget", 3.0))
    kappa = float(getattr(args, "kappa", 1.2))
    print_proof = bool(getattr(args, "print_proof", True))

    expr_input = getattr(args, "expr", None)
    file_input = getattr(args, "file", None)

    if (expr_input is None or str(expr_input).strip() == "") and file_input is None:
        print(f"{C.RED}Missing expression or file.{C.RESET}")
        return 2

    if file_input is not None:
        expr, graph = load_problem_from_file(str(file_input))
        expr_label = str(file_input)
    else:
        s = str(expr_input).strip()
        examples = _default_examples()
        if s in examples:
            expr = examples[s]
        else:
            expr = s
        expr, graph = load_problem(expr)
        expr_label = s

    separator(f"Solve: {expr_label} ({algorithm})")

    # Evaluate energy before
    evaluator = EnergyEvaluator()
    try:
        energy_before = evaluator.compute(graph)
        eb = getattr(energy_before, "total", None)
        if eb is None and isinstance(energy_before, (int, float)):
            eb = float(energy_before)
        energy_before_total = eb if eb is not None else None
    except Exception:
        energy_before_total = None

    # Prefer C++ bindings if available, and only fallback to Python if they fail.
    use_cpp = not bool(getattr(args, "no_cpp", False))
    cpp_res = None
    if use_cpp:
        try:
            cpp_res = _solve_with_cpp_bindings(
                graph=graph,
                algorithm=algorithm,
                beam_width=beam_width,
                max_depth=max_depth,
                budget=budget,
                kappa=kappa,
            )
        except Exception:
            cpp_res = None

    if cpp_res is not None:
        solved = bool(getattr(cpp_res, "solved", False))
        energy_after = getattr(cpp_res, "energy_after", None)
        elapsed_ms = getattr(cpp_res, "elapsed_ms", None)
        rule_promoted = getattr(cpp_res, "rule_promoted", None)
        proof_steps = getattr(cpp_res, "proof_steps", None) or []
        proof_nl = getattr(cpp_res, "proof_nl", None)

        if solved:
            print(f"{C.GREEN}{C.CHECKMARK} Solved{C.RESET}")
        else:
            print(f"{C.YELLOW}{C.BOLD}Not solved (C++ search){C.RESET}")

        if energy_before_total is not None or energy_after is not None:
            print(f"{C.DIM}Energy{C.RESET}: before={energy_before_total if energy_before_total is not None else 'n/a'} after={energy_after if energy_after is not None else 'n/a'}")

        if elapsed_ms is not None:
            print(f"{C.DIM}Elapsed{C.RESET}: {elapsed_ms} ms")

        if rule_promoted:
            print(f"{C.DIM}Rule promoted{C.RESET}: {rule_promoted}")

        if print_proof:
            if proof_nl:
                print(f"\n{C.BOLD}Proof (NL){C.RESET}:\n{proof_nl}")
            if proof_steps:
                print(f"\n{C.BOLD}Proof steps{C.RESET}:")
                for idx, step in enumerate(proof_steps, start=1):
                    print(f"  {idx}. {step}")

        return 0 if solved else 1

    # Python fallback
    transforms = list(ALL_TRANSFORMS)
    # Ensure heuristic scorer is loaded/available (even if not used by all search impls).
    try:
        _ = load_heuristic_scorer()
    except Exception:
        pass

    try:
        energy_eval = EnergyEvaluator()
    except Exception:
        energy_eval = None

    start = time.time()

    # Run BeamSearch or MCTSSearch depending on algorithm.
    if algorithm in ("beam", "beam_search", "beamsearch"):
        search = BeamSearch(
            energy_fn=energy_eval.compute if energy_eval else None,
            transforms=transforms,
            beam_width=beam_width,
            budget_seconds=budget,
            max_depth=max_depth,
            kappa=kappa,
        )
    elif algorithm in ("mcts", "mctss", "mcts_search", "montecarlo", "monte-carlo"):
        search = MCTSSearch(
            energy_fn=energy_eval.compute if energy_eval else None,
            transforms=transforms,
            budget_seconds=budget,
            max_depth=max_depth,
            kappa=kappa,
        )
    else:
        print(f"{C.RED}Unknown algorithm '{algorithm}'. Use 'beam' or 'mcts'.{C.RESET}")
        return 2

    # Use engine load_problem output graph as-is.
    result = search.search(graph=graph, energy=energy_before, transforms=transforms, beam_width=beam_width, budget_seconds=budget)

    elapsed_ms = int((time.time() - start) * 1000)

    solved = bool(getattr(result, "solved", False))
    energy_after_val = getattr(result, "energy_after", None)
    proof_steps = getattr(result, "proof_steps", None) or []
    proof_nl = getattr(result, "proof_nl", None)
    rule_promoted = getattr(result, "rule_promoted", None)

    if solved:
        print(f"{C.GREEN}{C.CHECKMARK} Solved{C.RESET}")
    else:
        print(f"{C.YELLOW}{C.BOLD}Not solved{C.RESET}")

    if energy_before_total is not None or energy_after_val is not None:
        print(f"{C.DIM}Energy{C.RESET}: before={energy_before_total if energy_before_total is not None else 'n/a'} after={energy_after_val if energy_after_val is not None else 'n/a'}")

    print(f"{C.DIM}Elapsed{C.RESET}: {elapsed_ms} ms")

    if rule_promoted:
        print(f"{C.DIM}Rule promoted{C.RESET}: {rule_promoted}")

    if print_proof:
        if proof_nl:
            print(f"\n{C.BOLD}Proof (NL){C.RESET}:\n{proof_nl}")
        if proof_steps:
            print(f"\n{C.BOLD}Proof steps{C.RESET}:")
            for idx, step in enumerate(proof_steps, start=1):
                print(f"  {idx}. {step}")

    # If C++ bindings exist but failed, mention only in verbose way.
    if use_cpp and _BINDINGS_ERROR:
        # Avoid spamming; only show brief.
        print(f"{C.DIM}Note{C.RESET}: C++ bindings unavailable/failed; using Python. ({_BINDINGS_ERROR})")

    return 0 if solved else 1


def build_parser():
    p = argparse.ArgumentParser(prog="sare", description="SARE-HX Command Line Interface")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("solve", help="Solve an expression or a named example")
    ps.add_argument("expr", nargs="?", help='Expression to solve (e.g., "x + 0") or named example key')
    ps.add_argument("-f", "--file", dest="file", help="Load problem from a JSON file")
    ps.add_argument("--algorithm", default="beam", choices=["beam", "mcts"], help="Search algorithm")
    ps.add_argument("--beam-width", default=6, type=int, help="Beam width (beam only)")
    ps.add_argument("--max-depth", default=16, type=int, help="Max search depth")
    ps.add_argument("--budget", default=3.0, type=float, help="Time budget in seconds")
    ps.add_argument("--kappa", default=1.2, type=float, help="Exploration/uncertainty parameter")
    ps.add_argument("--no-cpp", action="store_true", help="Disable C++ bindings (force Python)")
    ps.add_argument("--no-proof", action="store_true", help="Do not print proof steps")
    ps.set_defaults(func=cmd_solve)

    pi = sub.add_parser("inspect", help="Inspect graph without solving")
    pi.add_argument("expr", help='Expression to inspect (e.g., "x * 1 + 0")')
    pi.set_defaults(func=cmd_inspect)

    pe = sub.add_parser("examples", help="List built-in examples")
    pe.set_defaults(func=cmd_examples)

    pd = sub.add_parser("demo", help="Run all built-in examples")
    pd.add_argument("--max-examples", default=0, type=int, help="Limit number of examples (0=all)")
    pd.add_argument("--algorithm", default="beam", choices=["beam", "mcts"], help="Search algorithm")
    pd.add_argument("--beam-width", default=6, type=int, help="Beam width (beam only)")
    pd.add_argument("--max-depth", default=16, type=int, help="Max search depth")
    pd.add_argument("--budget", default=3.0, type=float, help="Time budget in seconds")
    pd.add_argument("--kappa", default=1.2, type=float, help="Exploration parameter")
    pd.add_argument("--no-cpp", action="store_true", help="Disable C++ bindings")
    pd.set_defaults(func=cmd_demo)

    pt = sub.add_parser("train", help="Train heuristic model")
    pt.set_defaults(func=cmd_train)

    return p


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    # Normalize proof flag
    if getattr(args, "cmd", None) == "solve":
        setattr(args, "print_proof", not bool(getattr(args, "no_proof", False)))

    try:
        rc = args.func(args)
    except KeyboardInterrupt:
        print(f"{C.YELLOW}\nInterrupted.{C.RESET}")
        return 130

    return int(rc) if rc is not None else 0


if __name__ == "__main__":
    sys.exit(main())