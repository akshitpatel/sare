"""
SARE-HX Web GUI Server

Zero-dependency web server using Python's built-in http.server.
Serves the frontend and handles API requests for the SARE engine.

Usage:
    python -m sare.interface.web [--port 8080]
    Then open http://localhost:8080
"""

import json
import sys
import os
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path

from sare.engine import (
    Graph, EnergyEvaluator, BeamSearch, MCTSSearch, EXAMPLE_PROBLEMS,
    load_problem, load_heuristic_scorer,
)
from sare.logging.logger import SareLogger, SolveLog
from sare.learning.abstraction_learning import mine_frequent_patterns, propose_macros
from sare.meta.macro_registry import list_macros, macro_steps_set, upsert_macros
_BINDINGS_ERROR = None
try:
    import sare.sare_bindings as _sb  # type: ignore
    ReflectionEngine = getattr(_sb, "ReflectionEngine", None)  # type: ignore
    ConceptRegistry = getattr(_sb, "ConceptRegistry", None)  # type: ignore
    CppGraph = getattr(_sb, "Graph", None)  # type: ignore
    CppSearchConfig = getattr(_sb, "SearchConfig", None)  # type: ignore
    run_beam_search = getattr(_sb, "run_beam_search", None)  # type: ignore
    run_mcts_search = getattr(_sb, "run_mcts_search", None)  # type: ignore
except Exception as e:  # pragma: no cover
    ReflectionEngine = None  # type: ignore
    ConceptRegistry = None  # type: ignore
    CppGraph = None  # type: ignore
    CppSearchConfig = None  # type: ignore
    run_beam_search = None  # type: ignore
    run_mcts_search = None  # type: ignore
    _BINDINGS_ERROR = str(e)

_CURRICULUM_ERROR = None
try:
    from sare.curiosity.curriculum_generator import CurriculumGenerator  # type: ignore
except Exception as e:  # pragma: no cover
    CurriculumGenerator = None  # type: ignore
    _CURRICULUM_ERROR = str(e)

_EXPERIMENT_ERROR = None
try:
    from sare.curiosity.experiment_runner import ExperimentRunner  # type: ignore
except Exception as e:  # pragma: no cover
    ExperimentRunner = None  # type: ignore
    _EXPERIMENT_ERROR = str(e)

# ── Optional Self-Learning Components (require working C++ bindings) ──
CausalInduction_ = getattr(_sb, "CausalInduction", None) if '_sb' in dir() else None  # type: ignore
reflection_engine = ReflectionEngine() if ReflectionEngine else None
concept_registry  = ConceptRegistry()  if ConceptRegistry  else None
causal_induction  = CausalInduction_() if CausalInduction_  else None
curriculum_gen    = CurriculumGenerator() if CurriculumGenerator else None
_energy_for_runner = EnergyEvaluator()
experiment_runner = (
    ExperimentRunner(
        curriculum_gen=curriculum_gen,
        searcher=BeamSearch(),
        energy=_energy_for_runner,
        reflection_engine=reflection_engine,
        causal_induction=causal_induction,
        concept_registry=concept_registry,
        transforms=None,  # filled lazily in run_batch
    )
    if (ExperimentRunner and curriculum_gen)
    else None
)

# ── Memory Manager (persistent episodic + strategy memory) ──
try:
    from sare.memory.memory_manager import MemoryManager, SolveEpisode as _MemEpisode  # type: ignore
except Exception as _mem_err:  # pragma: no cover
    MemoryManager = None  # type: ignore
    _MemEpisode = None  # type: ignore
    print(f"[sare] MemoryManager unavailable: {_mem_err}")

try:
    from sare.memory.concept_seed_loader import load_seeds  # type: ignore
except Exception:  # pragma: no cover
    load_seeds = None  # type: ignore

# ── SelfModel + FrontierManager ──
try:
    from sare.meta.self_model import SelfModel  # type: ignore
except Exception as _sm_err:  # pragma: no cover
    SelfModel = None  # type: ignore
    print(f"[sare] SelfModel unavailable: {_sm_err}")

try:
    from sare.curiosity.frontier_manager import FrontierManager  # type: ignore
except Exception as _fm_err:  # pragma: no cover
    FrontierManager = None  # type: ignore
    print(f"[sare] FrontierManager unavailable: {_fm_err}")

try:
    from sare.meta.goal_setter import GoalSetter  # type: ignore
except Exception as _gs_err:  # pragma: no cover
    GoalSetter = None  # type: ignore
    print(f"[sare] GoalSetter unavailable: {_gs_err}")

try:
    from sare.causal.abductive_ranker import AbductiveRanker  # type: ignore
    _abductive_ranker = AbductiveRanker()   # registry injected lazily after startup
except Exception as _ar_err:  # pragma: no cover
    AbductiveRanker = None  # type: ignore
    _abductive_ranker = None
    print(f"[sare] AbductiveRanker unavailable: {_ar_err}")

try:
    from sare.search.attention_beam import AttentionBeamScorer  # type: ignore
    _attention_scorer = AttentionBeamScorer(beta=0.30, gamma=0.15)
except Exception as _ab_err:  # pragma: no cover
    AttentionBeamScorer = None  # type: ignore
    _attention_scorer = None
    print(f"[sare] AttentionBeamScorer unavailable: {_ab_err}")

try:
    from sare.meta.proof_builder import ProofBuilder  # type: ignore
    _proof_builder = ProofBuilder()
except Exception as _pb_err:  # pragma: no cover
    ProofBuilder = None  # type: ignore
    _proof_builder = None
    print(f"[sare] ProofBuilder unavailable: {_pb_err}")

try:
    from sare.interface.nl_parser_v2 import EnhancedNLParser as BasicNLParser  # type: ignore
    _nl_parser = BasicNLParser()
    print("[sare] EnhancedNLParser (v2) loaded")
except Exception:  # pragma: no cover
    try:
        from sare.interface.nl_parser import BasicNLParser  # type: ignore
        _nl_parser = BasicNLParser()
        print("[sare] Fallback BasicNLParser loaded")
    except Exception as _np_err:
        BasicNLParser = None  # type: ignore
        _nl_parser = None
        print(f"[sare] NLParser unavailable: {_np_err}")

try:
    from sare.causal.analogy_transfer import AnalogyTransfer  # type: ignore
    _analogy_transfer = AnalogyTransfer()
except Exception as _at_err:  # pragma: no cover
    AnalogyTransfer = None  # type: ignore
    _analogy_transfer = None
    print(f"[sare] AnalogyTransfer unavailable: {_at_err}")

try:
    from sare.heuristics.trainer_bootstrap import TrainerBootstrap  # type: ignore
    _bootstrapper = TrainerBootstrap()
except Exception as _tb_err:  # pragma: no cover
    TrainerBootstrap = None  # type: ignore
    _bootstrapper = None
    print(f"[sare] TrainerBootstrap unavailable: {_tb_err}")

try:
    from sare.meta.active_questioner import ActiveQuestioner
    _active_questioner = ActiveQuestioner()
except Exception as _aq_err:
    ActiveQuestioner = None
    _active_questioner = None
    print(f"[sare] ActiveQuestioner unavailable: {_aq_err}")

try:
    from sare.memory.hippocampus import HippocampusDaemon
    _hippocampus = HippocampusDaemon()
    _hippocampus.start()
except Exception as _hc_err:
    HippocampusDaemon = None
    _hippocampus = None
    print(f"[sare] HippocampusDaemon unavailable: {_hc_err}")

# Instantiate memory manager and restore from disk
memory_manager = MemoryManager() if MemoryManager else None
if memory_manager:
    try:
        memory_manager.load()
    except Exception as _e:
        print(f"[sare] Memory restore failed: {_e}")

# Instantiate SelfModel + FrontierManager and restore
self_model = SelfModel() if SelfModel else None
if self_model:
    try:
        self_model.load()
    except Exception as _e:
        print(f"[sare] SelfModel restore failed: {_e}")

frontier_manager = FrontierManager() if FrontierManager else None
if frontier_manager:
    try:
        frontier_manager.load()
    except Exception as _e:
        print(f"[sare] FrontierManager restore failed: {_e}")

goal_setter = GoalSetter() if GoalSetter else None
if goal_setter:
    try:
        goal_setter.load()
    except Exception as _e:
        print(f"[sare] GoalSetter restore failed: {_e}")

# Pre-load foundational knowledge into ConceptRegistry
_seeds_loaded = 0
if concept_registry and load_seeds:
    try:
        _seeds_loaded = load_seeds(concept_registry)
        print(f"[sare] Loaded {_seeds_loaded} knowledge seeds into ConceptRegistry")
    except Exception as _e:
        print(f"[sare] Seed loading failed: {_e}")

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINEERING_CHECKLIST_PATH = REPO_ROOT / "configs" / "engineering_checklist.json"
SOLVE_LOG_PATH = REPO_ROOT / "logs" / "solves.jsonl"


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


def load_engineering_checklist() -> dict:
    """Load engineering checklist from config file with safe fallback."""
    fallback = {
        "title": "SARE Engineering Checklist",
        "last_updated": "",
        "tasks": [],
    }

    if not ENGINEERING_CHECKLIST_PATH.exists():
        return fallback

    try:
        with open(ENGINEERING_CHECKLIST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return fallback

    tasks = data.get("tasks", [])
    status_counts = {"done": 0, "in_progress": 0, "pending": 0}
    for task in tasks:
        status = task.get("status", "pending")
        if status not in status_counts:
            status = "pending"
        status_counts[status] += 1

    data["summary"] = {
        "total": len(tasks),
        "done": status_counts["done"],
        "in_progress": status_counts["in_progress"],
        "pending": status_counts["pending"],
        "completion_pct": (
            round((status_counts["done"] / len(tasks)) * 100, 1) if tasks else 0.0
        ),
    }
    return data


def _py_graph_to_cpp_graph(py_graph: Graph):
    """
    Best-effort bridge: convert pure-Python engine graphs into C++ binding graphs.
    This lets us run reflection/curriculum on the same solve episodes even before
    the GUI fully switches to the C++ runtime.
    """
    if not CppGraph:
        raise RuntimeError("C++ bindings Graph is unavailable")

    g = CppGraph()

    # Nodes
    for n in py_graph.nodes:
        g.add_node_with_id(int(n.id), str(n.type))
        cn = g.get_node(int(n.id))
        if not cn:
            continue

        # Preserve uncertainty if present.
        try:
            cn.uncertainty = float(getattr(n, "uncertainty", 0.0))
        except Exception:
            pass

        # Copy explicit attributes.
        attrs = getattr(n, "attributes", None) or {}
        for k, v in attrs.items():
            cn.set_attribute(str(k), str(v))

        # Bridge the Python engine's "label" into attributes to interop with
        # curriculum + potential C++ transforms.
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

    # Edges
    for e in py_graph.edges:
        g.add_edge_with_id(int(e.id), int(e.source), int(e.target), str(e.relationship_type), 1.0)

    return g


def _cpp_graph_to_py_graph(cpp_graph) -> Graph:
    """
    Convert C++ binding graphs (used by curriculum generator) into the
    pure-Python engine Graph (used by the current GUI solver).
    """
    g = Graph()
    id_map = {}

    op_to_label = {
        "add": "+",
        "mul": "*",
        "sub": "-",
        "div": "/",
        "neg": "neg",
    }

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
                op = n.get_attribute("op", "")
                label = op_to_label.get(op, op)

        attrs = {}
        for k in ("label", "value", "name", "op"):
            v = n.get_attribute(k, "")
            if v:
                attrs[k] = v

        py_id = g.add_node(ntype, label=label or "", attributes=(attrs or None))
        id_map[int(nid)] = py_id

        try:
            pn = g.get_node(py_id)
            if pn:
                pn.uncertainty = float(getattr(n, "uncertainty", 0.0))
        except Exception:
            pass

    for eid in cpp_graph.get_edge_ids():
        e = cpp_graph.get_edge(eid)
        if not e:
            continue

        s = id_map.get(int(e.source))
        t = id_map.get(int(e.target))
        if s is None or t is None:
            continue

        g.add_edge(s, t, str(e.relationship_type))

    return g


def _solve_with_cpp_bindings(
    graph: Graph,
    algorithm: str,
    beam_width: int,
    max_depth: int,
    budget: float,
    kappa: float,
):
    if not (CppGraph and CppSearchConfig and run_beam_search and run_mcts_search):
        return None

    cpp_graph = _py_graph_to_cpp_graph(graph)
    config = CppSearchConfig()
    config.beam_width = int(beam_width)
    config.max_depth = int(max_depth)
    config.budget_seconds = float(budget)
    config.kappa = float(kappa)

    cpp_result = run_mcts_search(cpp_graph, config) if algorithm == "mcts" else run_beam_search(cpp_graph, config)
    best_state = cpp_result.best_state
    best_energy = best_state.energy
    best_graph = _cpp_graph_to_py_graph(cpp_result.best_graph)

    components = {
        "syntax": best_energy.syntax,
        "constraint": best_energy.constraint,
        "test_failure": best_energy.test_failure,
        "complexity": best_energy.complexity,
        "resource": best_energy.resource,
        "uncertainty": best_energy.uncertainty,
    }

    return {
        "graph": best_graph,
        "energy_total": float(best_energy.total()),
        "energy_components": components,
        "transforms": list(best_state.transform_trace),
        "steps": int(best_state.depth if best_state.depth else len(best_state.transform_trace)),
        "expansions": int(cpp_result.total_expansions),
        "elapsed": float(cpp_result.elapsed_seconds),
        "trajectory": [float(best_energy.total())],
    }


class SareAPIHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves static files + API endpoints."""

    STATIC_DIR = Path(__file__).parent / "static"

    def do_GET(self):
        parsed = urlparse(self.path)

        poll_endpoints = {"/api/hippocampus/status", "/api/memory/stats", "/api/self"}
        if _hippocampus and parsed.path not in poll_endpoints:
            _hippocampus.ping_active()

        if parsed.path == "/" or parsed.path == "/index.html":
            self._serve_file("index.html", "text/html")
        elif parsed.path == "/api/examples":
            self._api_examples()
        elif parsed.path in ("/api/engineering-checklist", "/api/engineering_checklist"):
            self._api_engineering_checklist()
        elif parsed.path in ("/api/learning", "/api/learn"):
            self._api_learning()
        elif parsed.path == "/api/inspect":
            params = parse_qs(parsed.query)
            expr = params.get("expr", [""])[0]
            self._api_inspect(expr)
        elif parsed.path == "/api/concepts":
            self._api_concepts()
        elif parsed.path == "/api/curiosity":
            self._api_curiosity_get()
        elif parsed.path == "/api/experiment/stats":
            self._api_experiment_stats()
        elif parsed.path == "/api/memory/stats":
            self._api_memory_stats()
        elif parsed.path == "/api/self":
            self._api_self_report()
        elif parsed.path == "/api/frontier":
            self._api_frontier()
        elif parsed.path == "/api/goals":
            self._api_goals()
        elif parsed.path == "/api/parse":
            params = parse_qs(parsed.query)
            text = params.get("q", [""])[0]
            self._api_nlparse(text)
        elif parsed.path == "/api/explain":
            params  = parse_qs(parsed.query)
            transforms = params.get("t", [])
            delta   = float(params.get("delta", ["0"])[0])
            domain  = params.get("domain", ["general"])[0]
            self._api_explain(transforms, delta, domain)
        elif parsed.path == "/api/analogy":
            self._api_analogy()
        elif parsed.path == "/api/bootstrap":
            self._api_bootstrap()
        elif parsed.path == "/api/hippocampus/status":
            self._api_hippocampus_status()
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        
        if _hippocampus:
            _hippocampus.ping_active()

        if parsed.path == "/api/solve":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_solve(body)
        elif parsed.path == "/api/learn":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_learn(body)
        elif parsed.path == "/api/curiosity/generate":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_curiosity_generate(body)
        elif parsed.path == "/api/curiosity/solve":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_curiosity_solve(body)
        elif parsed.path == "/api/experiment/run":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_experiment_run(body)
        elif parsed.path == "/api/parse":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_nlparse(body.get("text", ""))
        elif parsed.path == "/api/explain":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_explain(
                transforms=body.get("transforms", []),
                delta=float(body.get("delta", 0)),
                domain=body.get("domain", "general"),
                top_k=int(body.get("top_k", 5)),
            )
        elif parsed.path == "/api/teach":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_teach(body)
        else:
            self.send_error(404)

    def _serve_file(self, filename, content_type):
        filepath = self.STATIC_DIR / filename
        if filepath.exists():
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.end_headers()
            self.wfile.write(filepath.read_bytes())
        else:
            self.send_error(404, f"File not found: {filename}")

    def _json_response(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _api_examples(self):
        energy = EnergyEvaluator()
        examples = []
        for name, expr in EXAMPLE_PROBLEMS.items():
            _, graph = load_problem(name)
            e = energy.compute(graph)
            examples.append({
                "name": name,
                "expression": expr,
                "energy": round(e.total, 3),
                "nodes": graph.node_count,
                "edges": graph.edge_count,
            })
        self._json_response(examples)

    def _api_engineering_checklist(self):
        self._json_response(load_engineering_checklist())

    def _api_learning(self):
        logger = SareLogger(str(SOLVE_LOG_PATH))
        summary = logger.summary()
        macros = list_macros()
        self._json_response({
            "logs": summary,
            "macros": [m.__dict__ for m in macros],
            "macro_count": len(macros),
        })

    def _api_inspect(self, expr: str):
        if not expr:
            self._json_response({"error": "No expression"}, 400)
            return

        expr_str, graph = load_problem(expr)
        energy = EnergyEvaluator()
        e = energy.compute(graph)

        # Find applicable transforms
        from sare import engine as engine_mod
        transforms = engine_mod.get_transforms(
            include_macros=True,
            concept_registry=concept_registry,  # TODO-06
        )
        applicable = []
        for t in transforms:
            matches = t.match(graph)
            if matches:
                applicable.append({"name": t.name(), "matches": len(matches)})

        self._json_response({
            "expression": expr_str,
            "graph": graph.to_dict(),
            "energy": {
                "total": round(e.total, 3),
                "components": {k: round(v, 3) for k, v in e.components.items()},
            },
            "applicable_transforms": applicable,
        })

    def _api_solve(self, body: dict):
        expr = body.get("expression", "")
        algorithm = body.get("algorithm", "beam")
        beam_width = body.get("beam_width", 8)
        max_depth = body.get("max_depth", 30)
        budget = body.get("budget", 10.0)
        kappa = float(body.get("kappa", 0.1))

        if not expr:
            self._json_response({"error": "No expression"}, 400)
            return

        expr_str, graph = load_problem(expr)
        energy = EnergyEvaluator()
        initial = energy.compute(graph)

        # ── MEMORY: warm-start from past similar problems ──
        strategy_hint = None
        if memory_manager:
            try:
                strategy_hint = memory_manager.before_solve(graph)
            except Exception:
                pass

        cpp_result = _solve_with_cpp_bindings(
            graph=graph,
            algorithm=algorithm,
            beam_width=beam_width,
            max_depth=max_depth,
            budget=budget,
            kappa=kappa,
        )
        if cpp_result is not None:
            result_graph = cpp_result["graph"]
            result_energy_total = cpp_result["energy_total"]
            result_energy_components = cpp_result["energy_components"]
            result_transforms = cpp_result["transforms"]
            result_steps = cpp_result["steps"]
            result_expansions = cpp_result["expansions"]
            result_elapsed = cpp_result["elapsed"]
            result_trajectory = cpp_result["trajectory"]
        else:
            from sare import engine as engine_mod
            transforms = engine_mod.get_transforms(include_macros=True)
            heuristic_fn = load_heuristic_scorer()
            if algorithm == "mcts":
                searcher = MCTSSearch()
                result = searcher.search(
                    graph, energy, transforms,
                    iterations=max_depth * 10,
                    budget_seconds=budget,
                )
            else:
                searcher = BeamSearch()
                result = searcher.search(
                    graph, energy, transforms,
                    beam_width=beam_width,
                    max_depth=max_depth,
                    budget_seconds=budget,
                    kappa=kappa,
                    heuristic_fn=heuristic_fn,
                )
            result_graph = result.graph
            result_energy_total = result.energy.total
            result_energy_components = result.energy.components
            result_transforms = result.transforms_applied
            result_steps = result.steps_taken
            result_expansions = result.expansions
            result_elapsed = result.elapsed_seconds
            result_trajectory = result.energy_trajectory

        delta = initial.total - result_energy_total

        # ── ACTIVE QUESTIONER: if no progress made, ask for help ──
        if delta <= 0.01 and _active_questioner:
            _domain = self_model.infer_domain(expr_str) if self_model else "general"
            q_data = _active_questioner.formulate_question(graph, domain=_domain)
            if q_data:
                self._json_response({
                    "status": "needs_help",
                    "question_data": q_data,
                    "expression": expr_str
                })
                return

        abstractions_used = [t for t in result_transforms if t.startswith("macro_")]
        node_types, adjacency = _graph_features(graph)

        SareLogger(str(SOLVE_LOG_PATH)).log(SolveLog(
            problem_id=expr_str,
            initial_energy=initial.total,
            final_energy=result_energy_total,
            energy_breakdown=result_energy_components,
            search_depth=result_steps,
            transform_sequence=result_transforms,
            compute_time_seconds=result_elapsed,
            total_expansions=result_expansions,
            energy_trajectory=result_trajectory,
            abstractions_used=abstractions_used,
            node_types=node_types,
            adjacency=adjacency,
            budget_exhausted=False,
            solve_success=(delta > 0.01),
        ))

        # ── MEMORY: record this episode after solve ──
        if memory_manager and _MemEpisode:
            try:
                episode = _MemEpisode(
                    problem_id=expr_str,
                    transform_sequence=result_transforms,
                    energy_trajectory=result_trajectory,
                    initial_energy=initial.total,
                    final_energy=result_energy_total,
                    compute_time_seconds=result_elapsed,
                    total_expansions=result_expansions,
                    success=(delta > 0.01),
                )
                memory_manager.after_solve(episode, graph)
            except Exception as _mem_e:
                pass

        # Add solved problems to curriculum seeds (even if reflection is disabled).
        if delta > 0.01 and curriculum_gen:
            try:
                curriculum_gen.add_seed(_py_graph_to_cpp_graph(graph))
            except Exception:
                pass

        # ── Self-Learning (Reflection + CausalInduction gating) ──
        learned_concepts = []
        if delta > 0.01 and reflection_engine and concept_registry:
            try:
                cpp_before = _py_graph_to_cpp_graph(graph)
                cpp_after  = _py_graph_to_cpp_graph(result_graph)
                rule = reflection_engine.reflect(cpp_before, cpp_after)
                if rule and rule.valid():
                    # Gate through CausalInduction before accepting
                    if causal_induction:
                        induction_result = causal_induction.evaluate(
                            rule, EnergyEvaluator()
                        )
                        promoted = induction_result.promoted
                        reasoning = induction_result.reasoning
                    else:
                        promoted = True
                        reasoning = "CausalInduction unavailable — accepted directly"

                    if promoted:
                        concept_registry.add_rule(rule)
                    learned_concepts.append({
                        "name":         rule.name,
                        "domain":       rule.domain,
                        "confidence":   rule.confidence,
                        "promoted":     promoted,
                        "reasoning":    reasoning,
                    })
            except Exception as e:
                print(f"Reflection failed: {e}")

        # Confidence: normalized energy reduction
        reduction_pct = (delta / initial.total * 100) if initial.total > 0 else 0.0
        confidence = min(1.0, reduction_pct / 100.0)
        memory_stats = memory_manager.stats() if memory_manager else {}

        # ── METACOGNITION: update SelfModel + FrontierManager ──
        if self_model:
            try:
                _domain = self_model.infer_domain(expr_str)
                self_model.observe(
                    domain=_domain,
                    success=(delta > 0.01),
                    delta=delta,
                    steps=result_steps,
                    transforms_used=result_transforms,
                    predicted_confidence=confidence,
                )
            except Exception:
                pass
        if frontier_manager:
            try:
                frontier_manager.record(
                    problem_id=expr_str,
                    success=(delta > 0.01),
                    delta=delta,
                    num_transforms=len(result_transforms),
                )
            except Exception:
                pass

        # ── Build proof (TODO-08) ──
        proof_dict = self._api_proof(
            transforms_applied=result_transforms,
            initial_energy=initial.total,
            final_energy=result_energy_total,
            expression=expr_str,
        )

        self._json_response({
            "expression": expr_str,
            "initial": {
                "graph": graph.to_dict(),
                "energy": {
                    "total": round(initial.total, 3),
                    "components": {k: round(v, 3) for k, v in initial.components.items()},
                },
            },
            "result": {
                "graph": result_graph.to_dict(),
                "energy": {
                    "total": round(result_energy_total, 3),
                    "components": {k: round(v, 3) for k, v in result_energy_components.items()},
                },
                "transforms": result_transforms,
                "steps": result_steps,
                "expansions": result_expansions,
                "elapsed": round(result_elapsed, 4),
                "trajectory": [round(e, 3) for e in result_trajectory],
            },
            "delta":              round(delta, 3),
            "reduction_pct":      round(reduction_pct, 1),
            "confidence":         round(confidence, 3),
            "strategy_hit":       bool(strategy_hint and strategy_hint.found),
            "learned_concepts":   learned_concepts,
            "memory":             memory_stats,
            # Flattened convenience fields for UI
            "transforms_applied": result_transforms,
            "steps_taken":        result_steps,
            "expansions":         result_expansions,
            "elapsed_seconds":    round(result_elapsed, 4),
            "proof":              proof_dict,
        })


    def _api_learn(self, body: dict):
        min_frequency = int(body.get("min_frequency", 2))
        min_length = int(body.get("min_length", 2))
        max_length = int(body.get("max_length", 4))
        max_new = int(body.get("max_new", 5))

        logger = SareLogger(str(SOLVE_LOG_PATH))
        entries = logger.read_all()
        traces = [
            e.transform_sequence
            for e in entries
            if e.solve_success and isinstance(e.transform_sequence, list) and e.transform_sequence
        ]

        patterns = mine_frequent_patterns(
            traces,
            min_frequency=min_frequency,
            min_length=min_length,
            max_length=max_length,
        )

        existing = list_macros()
        existing_steps = macro_steps_set(existing)
        proposed = propose_macros(patterns, existing_steps, max_new=max_new)

        # Validate macro candidates on built-in examples by direct application.
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

            for ex_name in EXAMPLE_PROBLEMS.keys():
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

        self._json_response({
            "episodes": len(entries),
            "successful_traces": len(traces),
            "patterns_found": len(patterns),
            "proposed": [s.__dict__ for s in proposed],
            "promoted": [s.__dict__ for s in promoted],
            "macro_count": len(upserted.get("macros", [])),
        })

    def _api_concepts(self):
        if not concept_registry:
            self._json_response({
                "error": "Concept learning is disabled (C++ bindings not available).",
                "bindings_error": _BINDINGS_ERROR,
            }, 503)
            return
        rules = concept_registry.get_rules()
        self._json_response({
            "count": len(rules),
            "concepts": [
                {
                    "name": r.name,
                    "pattern": f"{r.pattern.node_count()}n/{r.pattern.edge_count()}e",
                    "replacement": f"{r.replacement.node_count()}n/{r.replacement.edge_count()}e",
                    "confidence": r.confidence,
                    "observations": r.observations
                } 
                for r in rules
            ]
        })

    def _api_curiosity_get(self):
        if not curriculum_gen:
            self._json_response({
                "error": "Curiosity/Curriculum is disabled (bindings not available).",
                "bindings_error": _BINDINGS_ERROR,
                "curriculum_error": _CURRICULUM_ERROR,
            }, 503)
            return
        # Return current curriculum state (pending problems)
        pending = curriculum_gen.pending_problems()
        problems = []
        for p in pending:
            problems.append({
                "id": p.id,
                "origin": p.origin,
                "nodes": p.graph.node_count(),
                "edges": p.graph.edge_count(),
            })
        
        self._json_response({
            "pending_count": len(pending),
            "problems": problems
        })

    def _api_curiosity_generate(self, body):
        if not curriculum_gen:
            self._json_response({
                "error": "Curiosity/Curriculum is disabled (bindings not available).",
                "bindings_error": _BINDINGS_ERROR,
                "curriculum_error": _CURRICULUM_ERROR,
            }, 503)
            return
        count = int(body.get("count", 5))
        new_batch = curriculum_gen.generate_batch(count)
        self._json_response({
            "generated": len(new_batch),
            "total_pending": len(curriculum_gen.pending_problems())
        })

    def _api_curiosity_solve(self, body: dict):
        if not curriculum_gen:
            self._json_response({
                "error": "Curiosity/Curriculum is disabled (bindings not available).",
                "bindings_error": _BINDINGS_ERROR,
                "curriculum_error": _CURRICULUM_ERROR,
            }, 503)
            return

        pid = body.get("id", "") or body.get("problem_id", "")
        algorithm = body.get("algorithm", "beam")
        beam_width = body.get("beam_width", 8)
        max_depth = body.get("max_depth", 30)
        budget = body.get("budget", 10.0)
        kappa = float(body.get("kappa", 0.1))

        if not pid:
            self._json_response({"error": "No curiosity problem id"}, 400)
            return

        entry = curriculum_gen.get_problem(pid)
        if not entry or entry.status != "pending":
            self._json_response({"error": f"Unknown or already-solved curiosity id: {pid}"}, 404)
            return

        try:
            graph = _cpp_graph_to_py_graph(entry.graph)
        except Exception as e:
            self._json_response({"error": f"Failed to convert curiosity graph: {e}"}, 500)
            return

        energy = EnergyEvaluator()
        initial = energy.compute(graph)
        cpp_result = _solve_with_cpp_bindings(
            graph=graph,
            algorithm=algorithm,
            beam_width=beam_width,
            max_depth=max_depth,
            budget=budget,
            kappa=kappa,
        )
        if cpp_result is not None:
            result_graph = cpp_result["graph"]
            result_energy_total = cpp_result["energy_total"]
            result_energy_components = cpp_result["energy_components"]
            result_transforms = cpp_result["transforms"]
            result_steps = cpp_result["steps"]
            result_expansions = cpp_result["expansions"]
            result_elapsed = cpp_result["elapsed"]
            result_trajectory = cpp_result["trajectory"]
        else:
            from sare import engine as engine_mod
            transforms = engine_mod.get_transforms(include_macros=True)
            heuristic_fn = load_heuristic_scorer()
            if algorithm == "mcts":
                searcher = MCTSSearch()
                result = searcher.search(
                    graph, energy, transforms,
                    iterations=max_depth * 10,
                    budget_seconds=budget,
                )
            else:
                searcher = BeamSearch()
                result = searcher.search(
                    graph, energy, transforms,
                    beam_width=beam_width,
                    max_depth=max_depth,
                    budget_seconds=budget,
                    kappa=kappa,
                    heuristic_fn=heuristic_fn,
                )
            result_graph = result.graph
            result_energy_total = result.energy.total
            result_energy_components = result.energy.components
            result_transforms = result.transforms_applied
            result_steps = result.steps_taken
            result_expansions = result.expansions
            result_elapsed = result.elapsed_seconds
            result_trajectory = result.energy_trajectory

        delta = initial.total - result_energy_total
        abstractions_used = [t for t in result_transforms if t.startswith("macro_")]
        success = (delta > 0.01)
        node_types, adjacency = _graph_features(graph)

        SareLogger(str(SOLVE_LOG_PATH)).log(SolveLog(
            problem_id=f"curiosity:{pid}",
            initial_energy=initial.total,
            final_energy=result_energy_total,
            energy_breakdown=result_energy_components,
            search_depth=result_steps,
            transform_sequence=result_transforms,
            compute_time_seconds=result_elapsed,
            total_expansions=result_expansions,
            energy_trajectory=result_trajectory,
            abstractions_used=abstractions_used,
            node_types=node_types,
            adjacency=adjacency,
            budget_exhausted=False,
            solve_success=success,
        ))

        learned_concepts = []
        status = "pending"
        if success:
            curriculum_gen.mark_solved(pid)
            status = "solved"
            try:
                curriculum_gen.add_seed(_py_graph_to_cpp_graph(result_graph))
            except Exception:
                pass
        elif not result_transforms:
            curriculum_gen.mark_stuck(pid)
            status = "stuck"

        if success and reflection_engine and concept_registry:
            try:
                cpp_after = _py_graph_to_cpp_graph(result_graph)
                rule = reflection_engine.reflect(entry.graph, cpp_after)
                if rule and rule.name:
                    concept_registry.add_rule(rule)
                    learned_concepts.append({
                        "name": rule.name,
                        "pattern": f"{rule.pattern.node_count()}n/{rule.pattern.edge_count()}e",
                        "replacement": f"{rule.replacement.node_count()}n/{rule.replacement.edge_count()}e",
                        "confidence": rule.confidence
                    })
            except Exception as e:
                print(f"Reflection failed: {e}")

        self._json_response({
            "expression": f"[curiosity:{pid}]",
            "curiosity": {
                "id": pid,
                "origin": entry.origin,
                "status": status,
            },
            "initial": {
                "graph": graph.to_dict(),
                "energy": {
                    "total": round(initial.total, 3),
                    "components": {k: round(v, 3) for k, v in initial.components.items()},
                },
            },
            "result": {
                "graph": result_graph.to_dict(),
                "energy": {
                    "total": round(result_energy_total, 3),
                    "components": {k: round(v, 3) for k, v in result_energy_components.items()},
                },
                "transforms": result_transforms,
                "steps": result_steps,
                "expansions": result_expansions,
                "elapsed": round(result_elapsed, 4),
                "trajectory": [round(e, 3) for e in result_trajectory],
            },
            "delta": round(delta, 3),
            "reduction_pct": round((delta / initial.total * 100) if initial.total > 0 else 0, 1),
            "learned_concepts": learned_concepts,
        })

    def _api_experiment_stats(self):
        """GET /api/experiment/stats — return ExperimentRunner stats and history."""
        if not experiment_runner:
            self._json_response({
                "error": "ExperimentRunner unavailable",
                "experiment_error": globals().get("_EXPERIMENT_ERROR", ""),
            }, 503)
            return
        stats = experiment_runner.stats()
        history = [
            {
                "problem_id":    r.problem_id,
                "solved":        r.solved,
                "energy_before": round(r.energy_before, 3),
                "energy_after":  round(r.energy_after, 3),
                "rule_name":     r.rule_name,
                "rule_promoted": r.rule_promoted,
                "elapsed_ms":    round(r.elapsed_ms, 1),
                "reasoning":     r.reasoning,
            }
            for r in experiment_runner.history[-20:]  # last 20 experiments
        ]
        self._json_response({"stats": stats, "history": history})

    def _api_experiment_run(self, body: dict):
        """POST /api/experiment/run — run a batch of self-learning experiments."""
        if not experiment_runner:
            self._json_response({
                "error": "ExperimentRunner unavailable",
                "experiment_error": globals().get("_EXPERIMENT_ERROR", ""),
            }, 503)
            return
        n = int(body.get("n", 5))
        # Lazily wire the current transform list
        from sare.engine import get_transforms
        experiment_runner.transforms = get_transforms(include_macros=True)
        results = experiment_runner.run_batch(n=n)
        self._json_response({
            "ran": len(results),
            "solved": sum(1 for r in results if r.solved),
            "rules_promoted": sum(1 for r in results if r.rule_promoted),
            "results": [
                {
                    "problem_id":    r.problem_id,
                    "solved":        r.solved,
                    "delta":         round(r.energy_before - r.energy_after, 3),
                    "rule_name":     r.rule_name,
                    "rule_promoted": r.rule_promoted,
                    "reasoning":     r.reasoning,
                    "elapsed_ms":    round(r.elapsed_ms, 1),
                }
                for r in results
            ],
            "registry_size": len(concept_registry.get_rules()) if concept_registry else 0,
        })

    def _api_self_report(self):
        """GET /api/self — SelfModel self-assessment: domain competence, curiosity weights."""
        if not self_model:
            self._json_response({"error": "SelfModel unavailable"}, 503)
            return
        self._json_response(self_model.self_report())

    def _api_frontier(self):
        """GET /api/frontier — FrontierManager stats: solved/unsolved boundary."""
        if not frontier_manager:
            self._json_response({"error": "FrontierManager unavailable"}, 503)
            return
        near_frontier = [fp.to_dict() for fp in frontier_manager.sample_near_frontier(5)]
        self._json_response({
            "stats":        frontier_manager.stats(),
            "near_frontier": near_frontier,
        })

    def _api_memory_stats(self):
        """GET /api/memory/stats — live memory statistics and recent episodes."""
        if not memory_manager:
            self._json_response({"error": "MemoryManager unavailable"}, 503)
            return
        stats = memory_manager.stats()
        recent = [
            {
                "problem_id":    ep.problem_id,
                "success":       ep.success,
                "initial_energy": round(ep.initial_energy, 3),
                "final_energy":   round(ep.final_energy, 3),
                "energy_delta":   round(ep.initial_energy - ep.final_energy, 3),
                "transforms":     ep.transform_sequence[:5],  # first 5
                "elapsed_ms":     round(ep.compute_time_seconds * 1000, 1),
                "steps":          len(ep.transform_sequence),
            }
            for ep in memory_manager.recent_episodes(20)
        ]
        registry_size = len(concept_registry.get_rules()) if concept_registry else 0
        self._json_response({
            "stats":            stats,
            "recent_episodes":  recent,
            "concept_count":    registry_size,
            "seeds_loaded":     _seeds_loaded,
        })

    def _api_goals(self):
        """GET /api/goals — GoalSetter report: active/achieved goals, next_goal."""
        if not goal_setter:
            self._json_response({"error": "GoalSetter unavailable"}, 503)
            return
        # Refresh goals from self_model if available
        if self_model and goal_setter:
            try:
                goal_setter.refresh_from_self_model(self_model.self_report())
                goal_setter.save()
            except Exception:
                pass
        self._json_response(goal_setter.report())

    def _api_nlparse(self, text: str):
        """GET/POST /api/parse — BasicNLParser: convert NL text → expression."""
        if not _nl_parser:
            self._json_response({"error": "NL Parser unavailable"}, 503)
            return
        if not text:
            self._json_response({"error": "Provide ?q=<text> or POST {\"text\":\"...\"}"}, 400)
            return
        result = _nl_parser.parse(text)
        self._json_response(result.to_dict())

    def _api_proof(self, transforms_applied, initial_energy, final_energy, expression, domain="general"):
        """Helper: build a proof dict for a solve result."""
        if not _proof_builder:
            return None
        try:
            proof = _proof_builder.build(
                expression=expression,
                transforms_applied=transforms_applied,
                initial_energy=initial_energy,
                final_energy=final_energy,
                domain=domain,
            )
            return proof.to_dict()
        except Exception:
            return None

    def _api_explain(
        self,
        transforms: list,
        delta: float,
        domain: str = "general",
        top_k: int = 5,
    ):
        """GET|POST /api/explain — AbductiveRanker: explain an observed solve outcome."""
        if not _abductive_ranker:
            self._json_response({"error": "AbductiveRanker unavailable"}, 503)
            return

        # Lazily inject live ConceptRegistry into ranker
        if _abductive_ranker.registry is None and concept_registry:
            _abductive_ranker.registry = concept_registry

        try:
            result = _abductive_ranker.explain_to_dict(
                observed_transforms=list(transforms),
                observed_delta=float(delta),
                domain=domain,
                top_k=top_k,
            )
            # Add scorer info
            if _attention_scorer:
                result["attention_scorer"] = _attention_scorer.summary()
            self._json_response(result)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)



    def _api_analogy(self):
        """GET /api/analogy — Cross-domain analogy transfer summary + suggestions."""
        if not _analogy_transfer:
            self._json_response({"error": "AnalogyTransfer unavailable"}, 503)
            return
        try:
            summary = _analogy_transfer.summary()
            transfers = _analogy_transfer.transfer_all_domains()
            suggestions = [t.to_dict() for t in transfers[:20]]
            self._json_response({
                "summary":     summary,
                "suggestions": suggestions,
                "total":       len(suggestions),
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_bootstrap(self):
        """GET /api/bootstrap — Heuristic model training bootstrap stats."""
        if not _bootstrapper:
            self._json_response({"error": "TrainerBootstrap unavailable"}, 503)
            return
        try:
            self._json_response(_bootstrapper.stats())
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_hippocampus_status(self):
        """GET /api/hippocampus/status — Hippocampus daemon status."""
        if not _hippocampus:
            self._json_response({"error": "HippocampusDaemon unavailable"}, 503)
            return
        self._json_response(_hippocampus.status())

    def _api_teach(self, body: dict):
        """POST /api/teach — Human answers an ActiveQuestioner prompt."""
        rule_text = body.get("rule_text", "")
        domain = body.get("domain", "general")
        
        if not rule_text or not _nl_parser or not concept_registry:
            self._json_response({"error": "Missing parser or concept registry"}, 503)
            return
            
        try:
            # Parse the English rule into Left = Right
            # Expecting format something like "x and false is false"
            parts = rule_text.lower().split(" is ", 1)
            if len(parts) != 2:
                parts = rule_text.lower().split(" equals ", 1)
                
            if len(parts) == 2:
                left_expr, _, _ = _nl_parser._translate(parts[0], domain, [], [])
                right_expr, _, _ = _nl_parser._translate(parts[1], domain, [], [])
            else:
                self._json_response({"error": "Rule must contain 'is' or 'equals' (e.g. 'A plus 0 is A')"}, 400)
                return
                
            # Create the ConceptRule
            import time
            import sare.sare_bindings as sb
            
            rule_name = "human_taught_" + str(int(time.time()))
            
            # Create C++ rule object and add it directly
            rule = sb.AbstractRule()
            rule.name = rule_name
            rule.domain = domain
            rule.pattern = left_expr
            rule.replacement = right_expr
            rule.confidence = 0.51
            rule.observations = 1
            
            concept_registry.add_rule(rule)
            
            self._json_response({"status": "success", "rule_added": rule_name})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def log_message(self, format, *args):
        # Suppress default logging noise, only show API calls
        if "/api/" in str(args[0]):
            print(f"  API: {args[0]}")





def run_server(port=8080):
    print(f"""
\033[96m\033[1m╔══════════════════════════════════════════════════════════╗
║  SARE-HX Web GUI                                       ║
╚══════════════════════════════════════════════════════════╝\033[0m

  Server running at: \033[92m\033[1mhttp://localhost:{port}\033[0m
  Press Ctrl+C to stop.
""")

    server = HTTPServer(("localhost", port), SareAPIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SARE-HX Web GUI")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    args = parser.parse_args()
    run_server(args.port)
