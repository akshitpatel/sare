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
from sare.sare_logging.logger import SareLogger, SolveLog
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

# Inject registry into causal modules now that it exists
if '_analogy_transfer' in dir() and _analogy_transfer:
    _analogy_transfer.concept_registry = concept_registry
if '_abductive_ranker' in dir() and _abductive_ranker:
    _abductive_ranker.concept_registry = concept_registry

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
        # Bridge: convert C++ sare_bindings.Graph → Python engine Graph before solve
        graph_converter=lambda g: _cpp_graph_to_py_graph(g),
        # Pillar 3 + 4: credit_assigner and self_model patched in after they are initialized
        credit_assigner=None,
        self_model=None,
        # Tier 1A + 1B: cross-domain analogies and abductive reasoning
        analogy_transfer=_analogy_transfer if '_analogy_transfer' in dir() else None,
        abductive_ranker=_abductive_ranker if '_abductive_ranker' in dir() else None,
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
    from sare.learning.credit_assignment import CreditAssigner  # type: ignore
except Exception as _ca_err:  # pragma: no cover
    CreditAssigner = None  # type: ignore
    print(f"[sare] CreditAssigner unavailable: {_ca_err}")

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
    from sare.interface.universal_parser import UniversalParser
    _nl_parser = UniversalParser()
    print("[sare] UniversalParser (Epic 10) loaded")
except Exception as e:
    print(f"[sare] UniversalParser failed: {e}")
    try:
        from sare.interface.nl_parser_v2 import EnhancedNLParser as BasicNLParser  # type: ignore
        _nl_parser = BasicNLParser()
        print("[sare] EnhancedNLParser (v2) loaded fallback")
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

# ── Tier 5: LLM Hybrid Bridge ──────────────────────────────────────────────
try:
    from sare.interface.llm_bridge import (
        parse_nl_problem, explain_solve_trace, llm_status, llm_available
    )
    print(f"[sare] LLMBridge loaded (LLM available: {llm_available()})")
except Exception as _llm_err:
    parse_nl_problem = None  # type: ignore
    explain_solve_trace = None  # type: ignore
    llm_status = None  # type: ignore
    llm_available = None  # type: ignore
    print(f"[sare] LLMBridge unavailable: {_llm_err}")

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

# Pillar 4: patch self_model into experiment_runner (created before self_model above)
if experiment_runner and self_model:
    experiment_runner.self_model = self_model
    print("[sare] SelfModel wired into ExperimentRunner (Pillar 4 active)")

# Tier 1B: patch self_model into curriculum_gen for abductive domain targeting
if curriculum_gen and self_model:
    curriculum_gen._self_model = self_model
    print("[sare] SelfModel wired into CurriculumGenerator (Tier 1B active)")

frontier_manager = FrontierManager() if FrontierManager else None
if frontier_manager:
    try:
        frontier_manager.load()
    except Exception as _e:
        print(f"[sare] FrontierManager restore failed: {_e}")

# CreditAssigner — tracks per-transform utility from solve traces
# Now persists directly to data/memory/credit_assigner.json via save()/load()
credit_assigner = CreditAssigner() if CreditAssigner else None
if credit_assigner:
    try:
        credit_assigner.load()  # warm-start from disk (falls back to SelfModel below)
        if not credit_assigner.utilities and self_model and hasattr(self_model, 'get_transform_utilities'):
            saved = self_model.get_transform_utilities()
            if saved:
                credit_assigner.utilities.update(saved)
                print(f"[sare] CreditAssigner warmed from SelfModel ({len(saved)} utilities)")
        if credit_assigner.utilities:
            print(f"[sare] CreditAssigner: {len(credit_assigner.utilities)} utilities loaded")
    except Exception as _e:
        print(f"[sare] CreditAssigner restore failed: {_e}")

# Pillar 3: patch credit_assigner into experiment_runner after it's loaded
if experiment_runner and credit_assigner:
    experiment_runner.credit_assigner = credit_assigner
    print("[sare] CreditAssigner wired into ExperimentRunner (Pillar 3 active)")

# Tier 1: patch analogy_transfer and abductive_ranker and inject registry
if experiment_runner:
    if _analogy_transfer:
        _analogy_transfer.concept_registry = concept_registry
        experiment_runner.analogy_transfer = _analogy_transfer
        print("[sare] AnalogyTransfer wired into ExperimentRunner (Tier 1A active)")
    if _abductive_ranker:
        _abductive_ranker.concept_registry = concept_registry
        experiment_runner.abductive_ranker = _abductive_ranker
        print("[sare] AbductiveRanker wired into ExperimentRunner (Tier 1B active)")

# CurriculumGenerator — restore past problem counter + history
if curriculum_gen:
    try:
        curriculum_gen.load()
    except Exception as _e:
        print(f"[sare] CurriculumGenerator restore failed: {_e}")

goal_setter = GoalSetter() if GoalSetter else None
if goal_setter:
    try:
        goal_setter.load()
    except Exception as _e:
        print(f"[sare] GoalSetter restore failed: {_e}")

# Pillar 3: Grounded Concept Formation — initialize experience memory
try:
    from sare.memory.concept_formation import ConceptMemory, ConceptFormation
    concept_memory = ConceptMemory()
    concept_memory.load()
    print(f"[sare] ConceptMemory loaded ({len(concept_memory)} episodes)")
except Exception as _e:
    concept_memory = None
    print(f"[sare] ConceptMemory init failed: {_e}")

# AGI Gap #4: Common Sense Knowledge Base
try:
    from sare.knowledge.commonsense import CommonSenseBase
    common_sense = CommonSenseBase()
    common_sense.load()
    if common_sense.total_facts() == 0:
        common_sense.seed()  # Load built-in facts on first run
    print(f"[sare] CommonSenseBase ready ({common_sense.total_facts()} facts)")
except Exception as _e:
    common_sense = None
    print(f"[sare] CommonSenseBase init failed: {_e}")

# AGI Gap #5: Theory of Mind Engine
try:
    from sare.social.theory_of_mind import TheoryOfMindEngine
    tom_engine = TheoryOfMindEngine()
    tom_engine.load()
    print(f"[sare] TheoryOfMindEngine ready ({len(tom_engine._agents)} agents)")
except Exception as _e:
    tom_engine = None
    print(f"[sare] TheoryOfMindEngine init failed: {_e}")

# ── Graceful shutdown: persist everything to disk on exit ──────────────────
import atexit as _atexit

def _on_shutdown():
    """Flush all learning state to disk when the server exits."""
    if self_model:
        try: self_model.save(); print("[sare] SelfModel saved.")
        except Exception as _e: print(f"[sare] SelfModel save error: {_e}")
    if frontier_manager:
        try: frontier_manager.save(); print("[sare] FrontierManager saved.")
        except Exception as _e: print(f"[sare] FrontierManager save error: {_e}")
    if credit_assigner:
        try: credit_assigner.save(); print("[sare] CreditAssigner saved.")
        except Exception as _e: print(f"[sare] CreditAssigner save error: {_e}")
    if curriculum_gen:
        try: curriculum_gen.save(); print("[sare] CurriculumGenerator saved.")
        except Exception as _e: print(f"[sare] CurriculumGenerator save error: {_e}")
    if memory_manager:
        try: memory_manager.save(); print("[sare] MemoryManager saved.")
        except Exception: pass
    # Epic 22: Save learned + synthetic rules so they survive reboots
    if concept_registry and hasattr(concept_registry, "save"):
        try: concept_registry.save(); print("[sare] ConceptRegistry saved.")
        except Exception as _e: print(f"[sare] ConceptRegistry save error: {_e}")
    # Pillar 3: save concept memory episodes
    if concept_memory and hasattr(concept_memory, "save"):
        try: concept_memory.save(); print("[sare] ConceptMemory saved.")
        except Exception as _e: print(f"[sare] ConceptMemory save error: {_e}")
    # AGI Gap #4: Save commonsense facts
    if common_sense and hasattr(common_sense, "save"):
        try: common_sense.save(); print("[sare] CommonSenseBase saved.")
        except Exception as _e: print(f"[sare] CommonSenseBase save error: {_e}")
    # AGI Gap #5: Save Theory of Mind agent models
    if tom_engine and hasattr(tom_engine, "save"):
        try: tom_engine.save(); print("[sare] TheoryOfMindEngine saved.")
        except Exception as _e: print(f"[sare] TheoryOfMindEngine save error: {_e}")

_atexit.register(_on_shutdown)

# Pre-load foundational knowledge into ConceptRegistry
_seeds_loaded = 0
if concept_registry and load_seeds:
    try:
        _seeds_loaded = load_seeds(concept_registry)
        print(f"[sare] Loaded {_seeds_loaded} knowledge seeds into ConceptRegistry")
    except Exception as _e:
        print(f"[sare] Seed loading failed: {_e}")

# Epic 22: Load persisted learned/synthetic rules on startup
if concept_registry and hasattr(concept_registry, "load"):
    try:
        concept_registry.load()
        _n_synth = len(concept_registry.get_synthetic_rules()) if hasattr(concept_registry, "get_synthetic_rules") else 0
        print(f"[sare] Concept persistence loaded ({_n_synth} synthetic transforms restored)")
    except Exception as _e:
        print(f"[sare] Concept persistence load error: {_e}")

# ── Bootstrap CurriculumGenerator with seed graphs ──
# Build minimal C++ Graph objects so ExperimentRunner can generate problems
# immediately without needing prior human solves.
# Each seed = (op x y) structure that maps to a solvable algebraic expression.
_BOOTSTRAP_SEEDS = [
    # (op_label, arg1, arg2) — simple binary expressions
    ("add",  "x", "zero"),   # x + 0  → identity_add
    ("mul",  "x", "one"),    # x * 1  → identity_mul
    ("mul",  "x", "zero"),   # x * 0  → annihilator
    ("add",  "x", "x"),      # x + x  → idempotent add
    ("sub",  "x", "x"),      # x - x  → self_inverse
    ("and",  "x", "true"),   # x ∧ T  → identity_and
    ("or",   "x", "false"),  # x ∨ F  → identity_or
    ("and",  "x", "x"),      # x ∧ x  → idempotent
]
if curriculum_gen:
    try:
        from sare.sare_bindings import Graph as _CG  # type: ignore
        _boot_added = 0
        for _op, _a1, _a2 in _BOOTSTRAP_SEEDS:
            try:
                _g = _CG()
                _op_id = _g.add_node("OP");  _g.get_node(_op_id).set_attribute("label", _op)
                _a1_id = _g.add_node("VAR"); _g.get_node(_a1_id).set_attribute("label", _a1)
                _a2_id = _g.add_node("VAR"); _g.get_node(_a2_id).set_attribute("label", _a2)
                _g.add_edge(_op_id, _a1_id, "arg")
                _g.add_edge(_op_id, _a2_id, "arg")
                curriculum_gen.add_seed(_g)
                _boot_added += 1
            except Exception as _be:
                print(f"[sare] bootstrap seed '{_op}' failed: {_be}")
        if _boot_added:
            print(f"[sare] CurriculumGenerator seeded with {_boot_added} bootstrap C++ graphs")
    except ImportError:
        pass

# Inject transforms into experiment_runner so it can solve from day 1
if experiment_runner:
    from sare.engine import ALL_TRANSFORMS as _all_t  # type: ignore
    experiment_runner.transforms = _all_t

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
        elif parsed.path == "/api/analogies":
            self._api_analogies()
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
        elif parsed.path == "/api/autolearn":
            self._api_autolearn_status()
        elif parsed.path == "/api/llm-status":
            self._api_llm_status()
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
                delta=body.get("delta", 0.0),
                domain=body.get("domain", "general"),
                top_k=body.get("top_k", 5),
                expression=body.get("expression", "")
            )
        elif parsed.path == "/api/teach":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_teach(body)
        elif parsed.path == "/api/autolearn":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_autolearn_control(body)
        elif parsed.path == "/api/solve-nl":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_solve_nl(body)
        elif parsed.path == "/api/ground":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_ground(body)
        elif parsed.path == "/api/solve-full":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_solve_full(body)
        elif parsed.path == "/api/commonsense":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_commonsense(body)
        elif parsed.path == "/api/tom":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_tom(body)
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
        force_python = bool(body.get("force_python", False))

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

        # Pillars 3 (Human Brain): Retrieve past episodes for this exact graph signature
        recalled_memory = []
        if concept_memory:
            try:
                similar = concept_memory.retrieve_similar(graph, top_k=3)
                for s in similar:
                    ep = s["episode"]
                    if ep.get("transforms"):
                        recalled_memory.append({
                            "similarity": round(s["similarity"], 2),
                            "problem_id": ep.get("problem_id", ""),
                            "transforms": ep.get("transforms", [])[:5]
                        })
            except Exception as e:
                print(f"[sare] Memory retrieval error: {e}")

        cpp_result = None
        if not force_python:
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
                    attention_scorer=_attention_scorer,
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

        # ── TIER 7 & EPIC 24: METACOGNITIVE LOOP + PROGRAM SYNTHESIS ──
        if delta <= 0.01:
            from sare.memory.program_synthesizer import ProgramSynthesizer
            from sare.memory.metacognition import MetacognitiveController
            
            try:
                controller = MetacognitiveController()
                plan = controller.generate_plan(expr_str)
                synth = ProgramSynthesizer(concept_registry)
                
                print(f"MCTS stuck. Triggering Metacognitive Loop with {len(plan)} sub-goals...")
                
                for subgoal in plan:
                    goal_prompt = f"Sub-goal for {expr_str}: {subgoal}. Write a graph mutation function."
                    print(f"Executing Metacognitive Sub-Goal: {subgoal}")
                    
                    # Pillar 2: Use attention to select the most relevant sub-graph
                    try:
                        from sare.memory.attention import AttentionSelector, WorkingMemoryWorkspace
                        window_nodes = AttentionSelector.select_window(result_graph)
                        workspace = WorkingMemoryWorkspace(result_graph, window_nodes)
                        focused_graph = workspace.extract_subgraph()
                        print(f"  Attention: Focusing on {len(focused_graph.nodes)}/{len(result_graph.nodes)} nodes")
                    except Exception:
                        focused_graph = result_graph
                        workspace = None
                    
                    s_success, s_code, s_mutated = synth.generate_transform(focused_graph, goal_prompt)
                    
                    # Re-integrate the mutated sub-graph back into the full parent graph
                    if s_success and s_mutated and workspace:
                        try:
                            s_graph = workspace.graft(s_mutated)
                        except Exception:
                            s_graph = s_mutated
                    elif s_success and s_mutated:
                        s_graph = s_mutated
                    else:
                        s_graph = None
                    
                    if s_success and s_graph:
                        s_energy = energy.compute(s_graph)
                        if s_energy.total < result_energy_total:
                            print(f"Synthesizer succeeded for sub-goal! Adopted transform.")
                            result_graph = s_graph
                            result_energy_total = s_energy.total
                            result_energy_components = s_energy.components
                            result_transforms.append(f"synthetic_llm_subgoal")
                            result_steps += 1
                            result_trajectory.append(float(s_energy.total))
                            
                            # Epic 22: Permanently adopt into persistent memory
                            rule_name = f"synth_{hash(subgoal) % 100000}"
                            if hasattr(concept_registry, "add_synthetic_rule"):
                                concept_registry.add_synthetic_rule(rule_name, s_code, expr_str)
                            
                    controller.mark_goal_completed()
                    
                delta = initial.total - result_energy_total
            except Exception as e:
                print(f"Synthesizer/Metacognition failed: {e}")

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

        # Pillar 3: Record this solve episode into concept memory for clustering
        if concept_memory and delta > 0.01:
            try:
                concept_memory.record(result_graph, expr_str, result_transforms)
                # Run concept formation every 10 new episodes
                if len(concept_memory) % 10 == 0:
                    from sare.memory.concept_formation import ConceptFormation
                    former = ConceptFormation(concept_memory, concept_registry)
                    new_concepts = former.run()
                    if new_concepts:
                        print(f"[Pillar 3] Discovered {len(new_concepts)} new concepts from experience!")
            except Exception:
                pass  # Non-critical

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

        # ── CREDIT ASSIGNMENT: update per-transform utility from this trace ──
        if credit_assigner and result_transforms and result_trajectory:
            try:
                credit_assigner.assign_credit(result_transforms, result_trajectory)
                # Persist updated utilities back into SelfModel for warm-start
                if self_model and hasattr(self_model, 'update_transform_utilities'):
                    self_model.update_transform_utilities(credit_assigner.get_all_utilities())
                # ── PERSIST immediately after every human solve ─────────────
                credit_assigner.save()      # data/memory/credit_assigner.json
                if self_model:
                    self_model.save()       # data/memory/self_model.json
                if frontier_manager:
                    frontier_manager.save() # data/memory/frontier.jsonl
                # Reload transforms with updated utility ordering for next solve
                from sare.engine import reload_transforms  # type: ignore
                reload_transforms(
                    include_macros=True,
                    concept_registry=concept_registry,
                    utility_scores=credit_assigner.get_all_utilities(),
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

        # ── Epic 20: LLM Explanation Writer ──────────────────────────────
        nl_explanation = None
        if explain_solve_trace and delta > 0.01:
            try:
                # Infer domain from expression string heuristically
                _domain = "general"
                if self_model:
                    try: _domain = self_model.infer_domain(expr_str)
                    except Exception: pass
                nl_explanation = explain_solve_trace(
                    problem=expr_str,
                    transforms_applied=result_transforms,
                    energy_before=initial.total,
                    energy_after=result_energy_total,
                    final_expression=result_graph.to_expression() if hasattr(result_graph, 'to_expression') else expr_str,
                    expression=expr_str,
                    domain=_domain,
                    goal="simplify",
                )
            except Exception as _nl_e:
                print(f"[sare] LLM explanation failed: {_nl_e}")

        self._json_response({
            "expression": expr_str,
            "nl_explanation": nl_explanation,
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
            "recalled_memory":    recalled_memory,
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
        out = []
        for r in rules:
            try:
                out.append({
                    "name": getattr(r, "name", "unknown"),
                    "domain": getattr(r, "domain", "general"),
                    "confidence": getattr(r, "confidence", 1.0),
                    "observations": getattr(r, "observations", 0)
                })
            except Exception:
                pass

        self._json_response({
            "count": len(out),
            "concepts": out,
            "rules": out
        })

    def _api_analogies(self):
        if '_analogy_transfer' not in globals() or not _analogy_transfer:
            self._json_response({"error": "Analogy transfer is disabled."}, 503)
            return
        
        transfers = []
        for domain in ["arithmetic", "logic", "algebra", "sets"]:
            transfers.extend(_analogy_transfer.transfer_from_domain(domain))
            
        # Deduplicate by rule name
        unique = {}
        for tr in transfers:
            unique[tr.name] = tr.to_dict()
            
        self._json_response({
            "count": len(unique),
            "analogies": list(unique.values())
        })

    def _api_curiosity_get(self):
        if not curriculum_gen:
            self._json_response({
                "error": "Curiosity/Curriculum is disabled (bindings not available).",
                "bindings_error": _BINDINGS_ERROR,
                "curriculum_error": _CURRICULUM_ERROR,
            }, 503)
            return
        # Return current curriculum state (pending problems — skip ghost entries)
        pending = [p for p in curriculum_gen.pending_problems() if p.graph is not None]
        problems = []
        for p in pending:
            try:
                nc = p.graph.node_count() if hasattr(p.graph, 'node_count') else len(p.graph.get_node_ids())
                ec = p.graph.edge_count() if hasattr(p.graph, 'edge_count') else len(p.graph.get_edge_ids())
            except Exception:
                nc, ec = 0, 0
            problems.append({
                "id": p.id,
                "origin": p.origin,
                "nodes": nc,
                "edges": ec,
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
        expression: str = "",
    ):
        """GET|POST /api/explain — AbductiveRanker/CausalEngine: explain an observed solve outcome."""
        try:
            # ── Tier 4: Phase 5 Causal Modeling ──
            # Try to build a causal counterfactual mathematical proof first.
            if expression and _nl_parser:
                try:
                    from sare.sare_bindings import CounterfactualSimulator, Intervention
                    parse_res = _nl_parser.parse(expression)
                    g = parse_res.graph if hasattr(parse_res, "graph") else parse_res
                    sim = CounterfactualSimulator()
                    
                    interventions = []
                    # Create some counterfactual interventions (e.g. "What if we didn't apply these transforms?")
                    # For a rigid causal simulation, we modify the root node's type.
                    iv = Intervention()
                    iv.node_id = g.root()
                    if iv.node_id != 0:
                        iv.attribute = "type"
                        iv.value = "causal_intervention_test"
                        interventions.append(iv)
                        
                    results = sim.compare_interventions(g, interventions)
                    
                    causal_hypotheses = []
                    for r in results:
                        causal_hypotheses.append({
                            "name": "Counterfactual: do(Node {} {}={})".format(r.intervention.node_id, r.intervention.attribute, r.intervention.value),
                            "posterior": min(1.0, abs(r.delta) / 10.0), # normalize 
                            "occam_score": r.delta,
                            "domain": domain,
                            "recommended_action": "accept" if r.delta < 0 else "verify"
                        })
                        
                    if len(causal_hypotheses) > 0:
                        best = causal_hypotheses[0]
                        result = {
                            "confidence_level": "mathematical_proof (Phase 5 Causal Engine)",
                            "best_explanation": {
                                "name": best["name"],
                                "domain": domain,
                                "posterior": best["posterior"],
                                "recommended_action": best["recommended_action"],
                                "reasoning_chain": [
                                    f"Original Energy: {results[0].energy_original:.3f}",
                                    f"Counterfactual Energy: {results[0].energy_counterfactual:.3f}",
                                    f"Causal Delta (ΔE): {results[0].delta:.3f}",
                                    f"Mathematical conclusion: Intervening on the graph yields ΔE={results[0].delta:.3f}."
                                ]
                            },
                            "hypotheses": causal_hypotheses
                        }
                        if _attention_scorer:
                            result["attention_scorer"] = _attention_scorer.summary()
                        self._json_response(result)
                        return
                except ImportError as e:
                    print(f"[sare] Causal Simulator ImportError: {e}")
                except Exception as e:
                    print(f"[sare] Causal Simulator failed: {e}")

            # ── Fallback to AbductiveRanker ──
            if not _abductive_ranker:
                self._json_response({"error": "AbductiveRanker unavailable"}, 503)
                return

            # Lazily inject live ConceptRegistry into ranker
            if _abductive_ranker.registry is None and concept_registry:
                _abductive_ranker.registry = concept_registry

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

    def _api_llm_status(self):
        """GET /api/llm-status — LLM Bridge health and config."""
        if llm_status:
            self._json_response(llm_status())
        else:
            self._json_response({"available": False, "error": "LLMBridge not loaded"})

    def _api_solve_nl(self, body: dict):
        """POST /api/solve-nl — Accept free-form English, parse via LLM, then solve.

        Body: {"text": "Simplify 3x + 0", "algorithm": "beam", ...}
        Returns: same as /api/solve, plus 'parsed' showing what the LLM extracted.
        """
        nl_text = body.get("text", "").strip()
        if not nl_text:
            self._json_response({"error": "No text provided"}, 400)
            return

        if not parse_nl_problem:
            self._json_response({"error": "LLMBridge unavailable — set GEMINI_API_KEY"}, 503)
            return

        # Step 1: Parse NL → structured problem
        try:
            parsed = parse_nl_problem(nl_text)
        except Exception as e:
            self._json_response({"error": f"LLM parsing failed: {e}"}, 500)
            return

        # Step 2: Solve the parsed expression via the existing /api/solve logic
        solve_body = {
            "expression": parsed.expression,
            "algorithm":  body.get("algorithm", "beam"),
            "beam_width": body.get("beam_width", 8),
            "max_depth":  body.get("max_depth", 30),
            "budget":     body.get("budget", 10.0),
            "kappa":      body.get("kappa", 0.1),
        }

        # Capture the response by temporarily monkey-patching _json_response
        _captured: list = []
        _orig_jr = self._json_response

        def _capture(data, status=200):
            _captured.append((data, status))

        self._json_response = _capture  # type: ignore
        try:
            self._api_solve(solve_body)
        finally:
            self._json_response = _orig_jr  # type: ignore

        if not _captured:
            self._json_response({"error": "Solve produced no response"}, 500)
            return

        result_data, status_code = _captured[0]
        # Inject the parsed info so the UI can display it
        result_data["parsed"] = parsed.to_dict()
        result_data["raw_input"] = nl_text
        self._json_response(result_data, status_code)

    def _api_ground(self, body: dict):
        """
        POST /api/ground — World Grounding endpoint (AGI Gap #1).

        Converts real-world data into a SARE-HX typed graph.
        Accepts: { "kind": "csv"|"text"|"json"|"url", "payload": "..." }
        Returns: { "graph": {...}, "node_count": N, "edge_count": M }
        """
        try:
            from sare.perception.world_grounder import WorldGrounder, RawPercept
            kind = body.get("kind", "text")
            payload = body.get("payload", "")
            source = body.get("source", "")

            if not payload:
                self._json_response({"error": "No payload provided"}, 400)
                return

            grounder = WorldGrounder()
            percept = RawPercept(kind=kind, payload=payload, source=source)
            graph_dict = grounder.ground(percept)
            engine_graph = grounder.to_engine_graph(graph_dict)

            self._json_response({
                "status": "grounded",
                "kind": kind,
                "node_count": len(engine_graph.nodes),
                "edge_count": len(engine_graph.edges),
                "graph": graph_dict.to_engine_dict(),
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_solve_full(self, body: dict):
        """
        POST /api/solve-full — Full NL Hybrid Pipeline (Step 1).

        Complete pipeline: Natural Language → LLM Parse → SARE-HX Solve →
                           LLM Explain → Natural Language Answer

        Accepts: { "question": "What is x if 2x + 1 = 5?" }
        Returns: { "question", "expression", "answer_nl", "proof_trace", "energy" }
        """
        from sare.interface.llm_bridge import parse_nl_problem, explain_solve_trace, _call_llm

        question = body.get("question", "").strip()
        if not question:
            self._json_response({"error": "No question provided"}, 400)
            return

        # Step 1: LLM parses NL into canonical expression
        try:
            parsed = parse_nl_problem(question)
            expr_str = parsed.expression
        except Exception as e:
            self._json_response({"error": f"NL parsing failed: {e}"}, 500)
            return

        # Step 2: SARE-HX solves the structured expression
        _captured = []
        _orig_jr = self._json_response

        def _capture(data, status=200):
            _captured.append((data, status))
        self._json_response = _capture  # type: ignore

        try:
            self._api_solve({"expression": expr_str, "budget": body.get("budget", 15.0)})
        finally:
            self._json_response = _orig_jr  # type: ignore

        if not _captured:
            self._json_response({"error": "Solve produced no response"}, 500)
            return

        result, _ = _captured[0]

        # Step 3: LLM converts proof trace back to natural language
        try:
            transforms = result.get("transforms_applied", [])
            initial_energy = result.get("initial", {}).get("energy", {}).get("total", 1.0)
            final_energy = result.get("result", {}).get("energy", {}).get("total", 0.0)
            
            explanation = explain_solve_trace(
                problem=question,
                transforms_applied=transforms,
                energy_before=initial_energy,
                energy_after=final_energy,
                final_expression=result.get("expression", expr_str)
            )
        except Exception as e:
            print(f"Error generating explanation: {e}")
            explanation = f"SARE-HX solved '{expr_str}' using {len(result.get('transforms_applied', []))} steps."

        self._json_response({
            "question": question,
            "expression": expr_str,
            "answer_nl": explanation,
            "proof_trace": result.get("transforms_applied", []),
            "energy": {
                "initial": result.get("initial", {}).get("energy", {}).get("total", 1.0),
                "final": result.get("result", {}).get("energy", {}).get("total", 0.0)
            },
            "transforms": result.get("transforms_applied", []),
            "parsed": parsed.to_dict(),
        })

    def _api_commonsense(self, body: dict):
        """
        POST /api/commonsense — Query or augment the CommonSenseBase.

        Actions:
          { "action": "query", "concept": "fire" }
          { "action": "augment", "concepts": ["fire", "water"] }
          { "action": "stats" }
        """
        action = body.get("action", "query")
        try:
            if not common_sense:
                self._json_response({"error": "CommonSenseBase not initialized"}, 503)
                return

            if action == "query":
                concept = body.get("concept", "")
                depth = int(body.get("depth", 1))
                if not concept:
                    self._json_response({"error": "concept required"}, 400)
                    return
                facts = common_sense.query(concept, depth=depth)
                props = common_sense.get_properties(concept)
                self._json_response({
                    "concept": concept,
                    "facts": facts,
                    "properties": props,
                    "total_matched": len(facts),
                })
            elif action == "augment":
                concepts = body.get("concepts", [])
                if concepts:
                    common_sense.augment_from_conceptnet(concepts)
                self._json_response({"status": "augmented", "total_facts": common_sense.total_facts()})
            else:  # stats
                self._json_response({
                    "total_facts": common_sense.total_facts(),
                    "status": "healthy",
                })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_tom(self, body: dict):
        """
        POST /api/tom — Theory of Mind reasoning.

        Actions:
          { "action": "register", "agent_id": "alice", "description": "..." }
          { "action": "add_belief", "agent_id": "bob", "belief": "the ball is in the basket" }
          { "action": "false_belief", "agent_id": "bob", "reality": "ball is in box", "belief": "ball is in basket" }
          { "action": "predict", "agent_id": "alice", "situation": "she sees smoke" }
          { "action": "infer", "agent_id": "bob", "text": "Bob wants to win the race..." }
          { "action": "summary" }
        """
        action = body.get("action", "summary")
        try:
            if not tom_engine:
                self._json_response({"error": "TheoryOfMindEngine not initialized"}, 503)
                return

            if action == "register":
                agent_id = body.get("agent_id", "")
                desc = body.get("description", "")
                ms = tom_engine.register_agent(agent_id, desc)
                self._json_response({"registered": agent_id, "state": ms.to_dict()})

            elif action == "add_belief":
                agent_id = body.get("agent_id", "")
                belief = body.get("belief", "")
                truth = body.get("truth", True)
                ms = tom_engine.register_agent(agent_id)
                ms.add_belief(belief, truth=truth)
                self._json_response({"agent": agent_id, "belief_added": belief})

            elif action == "false_belief":
                agent_id = body.get("agent_id", "")
                reality = body.get("reality", "")
                belief = body.get("belief", "")
                result = tom_engine.false_belief_test(agent_id, reality, belief)
                self._json_response(result)

            elif action == "predict":
                agent_id = body.get("agent_id", "")
                situation = body.get("situation", "")
                prediction = tom_engine.predict_action_llm(agent_id, situation)
                self._json_response({"agent": agent_id, "situation": situation, "predicted_action": prediction})

            elif action == "infer":
                agent_id = body.get("agent_id", "")
                text = body.get("text", "")
                tom_engine.infer_beliefs_from_text(agent_id, text)
                ms = tom_engine.get_agent(agent_id)
                self._json_response({"agent": agent_id, "state": ms.to_dict() if ms else {}})

            else:  # summary
                self._json_response(tom_engine.summary())

        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_autolearn_status(self):
        """GET /api/autolearn — AutoLearn daemon status and live stats."""
        running = bool(
            experiment_runner and
            experiment_runner._daemon_thread and
            experiment_runner._daemon_thread.is_alive()
        )
        stats = experiment_runner.stats() if experiment_runner else {}
        recent = []
        if experiment_runner:
            for r in experiment_runner.history[-10:]:
                recent.append({
                    "id":       r.problem_id,
                    "solved":   r.solved,
                    "e_before": round(r.energy_before, 3),
                    "e_after":  round(r.energy_after, 3),
                    "rule":     r.rule_name,
                    "promoted": r.rule_promoted,
                    "ms":       round(r.elapsed_ms, 1),
                })
        self._json_response({
            "running":    running,
            "stats":      stats,
            "recent":     recent,
            "top_transforms": credit_assigner.get_top_transforms(5)
                              if credit_assigner else [],
        })

    def _api_autolearn_control(self, body: dict):
        """POST /api/autolearn — Start or stop the autonomous learning daemon.

        Body:
            {"action": "start", "interval": 20, "batch_size": 5}
            {"action": "stop"}
        """
        action     = body.get("action", "start")
        interval   = float(body.get("interval", 20))
        batch_size = int(body.get("batch_size", 5))

        if not experiment_runner:
            self._json_response({"error": "ExperimentRunner unavailable"}, 503)
            return

        if action == "start":
            # Build a hook that persists all learning state after every batch
            def _autolearn_save_hook(results):
                try:
                    if credit_assigner: credit_assigner.save()
                    if curriculum_gen:  curriculum_gen.save()
                    if self_model:      self_model.save()
                except Exception as _hook_e:
                    pass  # don't interrupt the daemon

            experiment_runner.start_daemon(
                interval_seconds=interval,
                batch_size=batch_size,
                post_batch_hook=_autolearn_save_hook,
            )
            self._json_response({
                "status":     "started",
                "interval":   interval,
                "batch_size": batch_size,
            })
        elif action == "stop":
            experiment_runner.stop_daemon()
            self._json_response({"status": "stopped"})
        else:
            self._json_response({"error": f"Unknown action: {action}"}, 400)

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
