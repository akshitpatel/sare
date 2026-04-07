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
import logging
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
from pathlib import Path

log = logging.getLogger(__name__)

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
reflection_engine = None
concept_registry = None
causal_induction = None
curriculum_gen = None
_energy_for_runner = None
experiment_runner = None

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
except Exception as _hc_err:
    HippocampusDaemon = None
    print(f"[sare] HippocampusDaemon unavailable: {_hc_err}")
_hippocampus = None

# Session dialogue contexts for multi-turn /api/ask
_SESSION_CONTEXTS: dict = {}


def _ensure_hippocampus_started():
    global _hippocampus
    if _hippocampus or not HippocampusDaemon:
        return _hippocampus
    _ensure_legacy_runtime()
    try:
        _hippocampus = HippocampusDaemon(
            memory_manager=memory_manager,
            experiment_runner=experiment_runner,
            reflection_engine=reflection_engine,
            curriculum_gen=curriculum_gen,
        )
        _hippocampus.start()
    except Exception as _hc_start_err:
        _hippocampus = None
        print(f"[sare] HippocampusDaemon start failed: {_hc_start_err}")
    return _hippocampus


def _ensure_self_improver_started():
    """Self-evolver disabled — focus on self-learning loop."""
    print("[sare] SelfImprover auto-start DISABLED (self-learning mode)")


def _restore_component(name, component):
    if component is None:
        return None
    loader = getattr(component, "load", None)
    if callable(loader):
        try:
            loader()
        except Exception as exc:
            print(f"[sare] {name} restore failed: {exc}")
    return component


def _save_component(name, component):
    if component is None:
        return
    saver = getattr(component, "save", None)
    if not callable(saver):
        return
    try:
        saver()
        print(f"[sare] {name} saved.")
    except Exception as exc:
        print(f"[sare] {name} save error: {exc}")


def _stop_component(name, component):
    if component is None:
        return
    stopper = getattr(component, "stop", None)
    if not callable(stopper):
        return
    try:
        stopper()
        print(f"[sare] {name} stopped.")
    except Exception as exc:
        print(f"[sare] {name} stop error: {exc}")

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

memory_manager = None
self_model = None
frontier_manager = None
credit_assigner = None
goal_setter = None
concept_memory = None
common_sense = None
tom_engine = None
_legacy_runtime_ready = False
import threading as _threading
_runtime_lock = _threading.Lock()

# ── Bootstrap CurriculumGenerator with seed graphs ──
# Build minimal C++ Graph objects so ExperimentRunner can generate problems
# immediately without needing prior human solves.
# Each seed = (op x y) structure that maps to a solvable algebraic expression.
_BOOTSTRAP_SEEDS = [
    ("add",  "x", "zero"),
    ("mul",  "x", "one"),
    ("mul",  "x", "zero"),
    ("add",  "x", "x"),
    ("sub",  "x", "x"),
    ("and",  "x", "true"),
    ("or",   "x", "false"),
    ("and",  "x", "x"),
]


def _ensure_legacy_runtime():
    if _legacy_runtime_ready:
        return
    with _runtime_lock:
        if _legacy_runtime_ready:  # double-checked locking
            return
        _ensure_legacy_runtime_locked()


def _ensure_legacy_runtime_locked():
    """Inner init — called only while holding _runtime_lock."""
    global _legacy_runtime_ready
    global reflection_engine, concept_registry, causal_induction, curriculum_gen
    global _energy_for_runner, experiment_runner
    global memory_manager, self_model, frontier_manager, credit_assigner, goal_setter
    global concept_memory, common_sense, tom_engine

    # Prefer Python reflection engine (always available) over C++ one (usually missing)
    try:
        from sare.reflection.py_reflection import get_reflection_engine
        reflection_engine = get_reflection_engine()
    except Exception:
        reflection_engine = ReflectionEngine() if ReflectionEngine else None
    # Use Python ConceptRegistry (accepts Python AbstractRule objects)
    from sare.memory.concept_seed_loader import SeededConceptRegistry
    concept_registry = SeededConceptRegistry()
    if _analogy_transfer:
        _analogy_transfer.concept_registry = concept_registry
    if _abductive_ranker:
        _abductive_ranker.concept_registry = concept_registry

    # Prefer Python CausalInduction (always available) over C++ one
    try:
        from sare.causal.induction import CausalInduction
        causal_induction = CausalInduction()
    except Exception:
        causal_induction = CausalInduction_() if CausalInduction_ else None
    curriculum_gen = CurriculumGenerator() if CurriculumGenerator else None
    _energy_for_runner = EnergyEvaluator()
    experiment_runner = (
        ExperimentRunner(
            curriculum_gen=curriculum_gen,
            searcher=BeamSearch(),
            energy=_energy_for_runner,
            reflection_engine=reflection_engine,
            causal_induction=causal_induction,
            concept_registry=concept_registry,
            transforms=None,
            graph_converter=lambda g: _cpp_graph_to_py_graph(g),
            credit_assigner=None,
            self_model=None,
            analogy_transfer=_analogy_transfer,
            abductive_ranker=_abductive_ranker,
        )
        if (ExperimentRunner and curriculum_gen)
        else None
    )

    memory_manager = _restore_component("MemoryManager", MemoryManager() if MemoryManager else None)
    try:
        self_model = _restore_component("SelfModel", SelfModel(Path("data/memory")) if SelfModel else None)
    except Exception as _sm_init_err:
        print(f"[sare] SelfModel init failed: {_sm_init_err}")
        self_model = None
    frontier_manager = _restore_component("FrontierManager", FrontierManager() if FrontierManager else None)
    credit_assigner = _restore_component("CreditAssigner", CreditAssigner() if CreditAssigner else None)
    goal_setter = _restore_component("GoalSetter", GoalSetter() if GoalSetter else None)

    if experiment_runner and self_model:
        experiment_runner.self_model = self_model
        print("[sare] SelfModel wired into ExperimentRunner (Pillar 4 active)")
    if curriculum_gen and self_model:
        curriculum_gen._self_model = self_model
        print("[sare] SelfModel wired into CurriculumGenerator (Tier 1B active)")

    if credit_assigner:
        if not credit_assigner.utilities and self_model and hasattr(self_model, "get_transform_utilities"):
            try:
                saved = self_model.get_transform_utilities()
            except Exception as _e:
                saved = None
                print(f"[sare] CreditAssigner warm-start failed: {_e}")
            if saved:
                credit_assigner.utilities.update(saved)
                print(f"[sare] CreditAssigner warmed from SelfModel ({len(saved)} utilities)")
        if credit_assigner.utilities:
            print(f"[sare] CreditAssigner: {len(credit_assigner.utilities)} utilities loaded")

    if experiment_runner and credit_assigner:
        experiment_runner.credit_assigner = credit_assigner
        print("[sare] CreditAssigner wired into ExperimentRunner (Pillar 3 active)")

    if experiment_runner:
        if _analogy_transfer:
            experiment_runner.analogy_transfer = _analogy_transfer
            print("[sare] AnalogyTransfer wired into ExperimentRunner (Tier 1A active)")
        if _abductive_ranker:
            experiment_runner.abductive_ranker = _abductive_ranker
            print("[sare] AbductiveRanker wired into ExperimentRunner (Tier 1B active)")

    curriculum_gen = _restore_component("CurriculumGenerator", curriculum_gen)

    try:
        from sare.memory.concept_formation import ConceptMemory
        concept_memory = ConceptMemory()
        concept_memory.load()
        print(f"[sare] ConceptMemory loaded ({len(concept_memory)} episodes)")
    except Exception as _e:
        concept_memory = None
        print(f"[sare] ConceptMemory init failed: {_e}")

    try:
        from sare.knowledge.commonsense import CommonSenseBase
        common_sense = CommonSenseBase()
        common_sense.load()
        if common_sense.total_facts() == 0:
            common_sense.seed()
        print(f"[sare] CommonSenseBase ready ({common_sense.total_facts()} facts)")
    except Exception as _e:
        common_sense = None
        print(f"[sare] CommonSenseBase init failed: {_e}")

    try:
        from sare.social.theory_of_mind import TheoryOfMindEngine
        tom_engine = TheoryOfMindEngine()
        tom_engine.load()
        print(f"[sare] TheoryOfMindEngine ready ({len(tom_engine._agents)} agents)")
    except Exception as _e:
        tom_engine = None
        print(f"[sare] TheoryOfMindEngine init failed: {_e}")

    if concept_registry and load_seeds:
        try:
            seeds_loaded = load_seeds(concept_registry)
            print(f"[sare] Loaded {seeds_loaded} knowledge seeds into ConceptRegistry")
        except Exception as _e:
            print(f"[sare] Seed loading failed: {_e}")

    if concept_registry and hasattr(concept_registry, "load"):
        try:
            concept_registry.load()
            restored = len(concept_registry.get_synthetic_rules()) if hasattr(concept_registry, "get_synthetic_rules") else 0
            print(f"[sare] Concept persistence loaded ({restored} synthetic transforms restored)")
        except Exception as _e:
            print(f"[sare] Concept persistence load error: {_e}")

    if curriculum_gen:
        from sare.engine import load_problem as _lp
        _seed_exprs = [
            "x + 0", "x * 1", "x * 0", "x + x", "x - x",
            "not not x", "neg neg x", "x * (y + 0)", "2 * (x + 0)",
        ]
        boot_added = 0
        for _expr in _seed_exprs:
            try:
                _, _g = _lp(_expr)
                if _g:
                    curriculum_gen.add_seed(_g)
                    boot_added += 1
            except Exception as _be:
                print(f"[sare] bootstrap seed '{_expr}' failed: {_be}")
        if boot_added:
            print(f"[sare] CurriculumGenerator seeded with {boot_added} Python graphs")

    if experiment_runner:
        from sare.engine import ALL_TRANSFORMS as _all_t  # type: ignore
        _domain_transforms = list(_all_t)
        # Augment with code-domain transforms
        try:
            from sare.transforms.code_transforms import (
                IfTrueElimTransform, IfFalseElimTransform,
                NotTrueTransform, NotFalseTransform,
                AndSelfTransform, OrSelfTransform, SelfAssignElimTransform,
            )
            _domain_transforms = [
                IfTrueElimTransform(), IfFalseElimTransform(),
                NotTrueTransform(), NotFalseTransform(),
                AndSelfTransform(), OrSelfTransform(), SelfAssignElimTransform(),
            ] + _domain_transforms
        except Exception as _ce:
            print(f"[sare] code transforms skipped: {_ce}")
        # Augment with QA/inference transforms
        try:
            from sare.knowledge.commonsense import CommonSenseBase as _CSB
            from sare.transforms.logic_transforms import (
                FillUnknownTransform, ChainInferenceTransform,
                ModusPonensTransform, DoubleNegRemoveTransform, ImpliesElimTransform,
            )
            _kb = _CSB()
            _kb.load()
            if _kb.total_facts() == 0:
                _kb.seed()
            _domain_transforms = [
                FillUnknownTransform(_kb),
                ChainInferenceTransform(_kb),
                ModusPonensTransform(),
                DoubleNegRemoveTransform(),
                ImpliesElimTransform(),
            ] + _domain_transforms
            print(f"[sare] QA+logic transforms wired (KB: {_kb.total_facts()} facts)")
        except Exception as _qe:
            print(f"[sare] QA transforms skipped: {_qe}")
        experiment_runner.transforms = _domain_transforms

    _legacy_runtime_ready = True


def _path_requires_legacy_runtime(path: str) -> bool:
    if not path.startswith("/api/"):
        return False
    if path.startswith("/api/brain"):
        return False
    if path in {
        "/api/examples",
        "/api/engineering-checklist",
        "/api/engineering_checklist",
        "/api/llm-status",
        "/api/llm/rate-limits",
        "/api/learning/live",
        "/api/daemon/heartbeat",
        "/api/daemon/livelog",
    }:
        return False
    return True

# ── Graceful shutdown: persist everything to disk on exit ──────────────────
import atexit as _atexit

def _on_shutdown():
    """Flush all learning state to disk when the server exits."""
    global _hippocampus
    _save_component("SelfModel", self_model)
    _save_component("FrontierManager", frontier_manager)
    _save_component("CreditAssigner", credit_assigner)
    _save_component("CurriculumGenerator", curriculum_gen)
    _save_component("MemoryManager", memory_manager)
    _save_component("ConceptRegistry", concept_registry)
    _save_component("ConceptMemory", concept_memory)
    _save_component("CommonSenseBase", common_sense)
    _save_component("TheoryOfMindEngine", tom_engine)
    if _hippocampus:
        _stop_component("HippocampusDaemon", _hippocampus)
        _hippocampus = None

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
    from sare.engine import load_problem as _lp2
    _seed_exprs2 = [
        "x + 0", "x * 1", "x * 0", "0 + x", "1 * x",
        "x - x", "not not x", "neg neg x",
        "x * (y + 0)", "2 * (x + 0)", "x + x",
    ]
    _boot_added = 0
    for _expr2 in _seed_exprs2:
        try:
            _, _g2 = _lp2(_expr2)
            if _g2:
                curriculum_gen.add_seed(_g2)
                _boot_added += 1
        except Exception as _be:
            pass
    if _boot_added:
        print(f"[sare] CurriculumGenerator seeded with {_boot_added} Python graphs")

# Inject transforms into experiment_runner so it can solve from day 1
if experiment_runner:
    from sare.engine import ALL_TRANSFORMS as _all_t  # type: ignore
    experiment_runner.transforms = _all_t

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINEERING_CHECKLIST_PATH = REPO_ROOT / "configs" / "engineering_checklist.json"
SOLVE_LOG_PATH = REPO_ROOT / "logs" / "solves.jsonl"


# ── Graph bridge utilities (canonical implementation in core.graph_bridge) ────
from sare.core.graph_bridge import (
    graph_features as _graph_features,
    py_graph_to_cpp_graph as _py_graph_to_cpp_graph,
    cpp_graph_to_py_graph as _cpp_graph_to_py_graph,
)


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

        if _path_requires_legacy_runtime(parsed.path):
            _ensure_legacy_runtime()

        poll_endpoints = {"/api/hippocampus/status", "/api/memory/stats", "/api/self"}
        if _hippocampus and parsed.path not in poll_endpoints:
            _hippocampus.ping_active()

        if parsed.path == "/" or parsed.path == "/index.html":
            self._serve_file("index.html", "text/html")
        elif parsed.path == "/favicon.ico":
            # Serve a minimal 1×1 transparent ICO to silence browser 404s
            ico = (b"\x00\x00\x01\x00\x01\x00\x01\x01\x00\x00\x01\x00\x18\x00"
                   b"\x30\x00\x00\x00\x16\x00\x00\x00\x28\x00\x00\x00\x01\x00"
                   b"\x00\x00\x02\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00"
                   b"\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                   b"\x00\x00\x00\x00\x00\x00\x00\x00\x1a\x1a\x2a\x00\x00\x00"
                   b"\x00\x00")
            self.send_response(200)
            self.send_header("Content-Type", "image/x-icon")
            self.send_header("Content-Length", str(len(ico)))
            self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            self.wfile.write(ico)
        elif parsed.path == "/api/evolver/daemon":
            self._api_evolver_daemon()
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
        elif parsed.path == "/api/self-improve/status":
            self._api_self_improve_status()
        elif parsed.path == "/api/self-improve/patches":
            self._api_self_improve_patches()
        elif parsed.path == "/api/bottleneck":
            self._api_bottleneck()
        elif parsed.path == "/api/forgetting":
            self._api_forgetting()
        elif parsed.path == "/api/mind/stream":
            self._api_mind_stream()
        elif parsed.path == "/api/mind/questions":
            self._api_mind_questions()
        elif parsed.path == "/api/agi/evolution":
            self._api_agi_evolution()
        elif parsed.path == "/api/learning/monitor":
            self._api_learning_monitor()
        elif parsed.path == "/api/agi/value-net":
            self._api_agi_value_net()
        # ── Evolver Chat endpoints ─────────────────────────────
        elif parsed.path == "/api/evolve/logs":
            self._api_evolve_logs()
        elif parsed.path == "/api/evolve/logs/stream":
            self._api_evolve_logs_stream()
        elif parsed.path == "/api/evolve/messages":
            self._api_evolve_messages()
        elif parsed.path == "/api/evolve/messages/stream":
            self._api_evolve_messages_stream()
        elif parsed.path == "/api/evolve/suggestions":
            self._api_evolve_suggestions()
        # ── Neuro / AGI mind endpoints ────────────────────────
        elif parsed.path == "/api/neuro/dopamine":
            self._api_neuro_dopamine()
        elif parsed.path == "/api/neuro/symbols":
            self._api_neuro_symbols()
        elif parsed.path == "/api/neuro/algorithms":
            self._api_neuro_algorithms()
        elif parsed.path == "/api/neuro/creativity":
            self._api_neuro_creativity()
        elif parsed.path == "/api/neuro/htm":
            self._api_neuro_htm()
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
        elif parsed.path == "/api/daemon/status":
            self._api_daemon_status()
        elif parsed.path == "/api/daemon/activity":
            self._api_daemon_activity()
        elif parsed.path == "/api/daemon/heartbeat":
            self._api_daemon_heartbeat()
        elif parsed.path == "/api/daemon/livelog":
            params = parse_qs(parsed.query)
            lines = int(params.get("lines", ["60"])[0])
            self._api_daemon_livelog(lines)
        elif parsed.path == "/api/llm-status":
            self._api_llm_status()
        elif parsed.path == "/api/llm/rate-limits":
            self._api_llm_rate_limits()
        elif parsed.path == "/api/compose":
            params = parse_qs(parsed.query)
            subject = params.get("subject", [""])[0]
            target  = params.get("target",  [None])[0]
            hops    = int(params.get("hops", ["3"])[0])
            self._api_compose(subject, target, hops)
        elif parsed.path == "/api/world":
            self._api_world_summary()
        elif parsed.path == "/api/world/imagine":
            params = parse_qs(parsed.query)
            seed = params.get("seed", ["addition"])[0]
            depth = int(params.get("depth", ["2"])[0])
            self._api_world_imagine(seed, depth)
        elif parsed.path == "/api/world/simulate":
            params = parse_qs(parsed.query)
            scenario = params.get("scenario", [""])[0]
            steps = int(params.get("steps", ["3"])[0])
            self._api_world_simulate(scenario, steps)
        elif parsed.path == "/api/world/analogy":
            params = parse_qs(parsed.query)
            source = params.get("source", [""])[0]
            target = params.get("target", [""])[0]
            self._api_world_analogy(source, target)
        elif parsed.path == "/api/world/counterfactual":
            params = parse_qs(parsed.query)
            rule = params.get("rule", [""])[0]
            negated = params.get("negated", ["true"])[0].lower() != "false"
            self._api_world_counterfactual(rule, negated)
        elif parsed.path == "/api/world/predictions":
            self._api_world_predictions()
        elif parsed.path == "/api/world/hypotheses":
            self._api_world_hypotheses()
        elif parsed.path == "/api/world/schema/learn":
            domain = params.get("domain", ["general"])[0]
            self._api_world_schema_learn(domain)
        elif parsed.path == "/api/world/consistency":
            self._api_world_consistency()
        elif parsed.path == "/api/world/beliefs":
            params = parse_qs(parsed.query)
            domain = params.get("domain", [None])[0]
            self._api_world_beliefs(domain)
        elif parsed.path == "/api/world/analogies":
            params = parse_qs(parsed.query)
            domain = params.get("domain", [None])[0]
            self._api_world_analogies(domain)
        elif parsed.path == "/api/world/predict":
            params = parse_qs(parsed.query)
            expression = params.get("expression", ["x + 0"])[0]
            domain = params.get("domain", ["arithmetic"])[0]
            self._api_world_predict(expression, domain)
        elif parsed.path == "/api/world/live":
            self._api_world_live()
        elif parsed.path == "/api/synth/stats":
            self._api_synth_stats()
        elif parsed.path == "/api/synth/review":
            self._api_synth_review()
        elif parsed.path == "/api/llm/few-shot":
            self._api_llm_few_shot_stats()
        elif parsed.path == "/api/autolearn/log":
            self._api_autolearn_log()
        elif parsed.path == "/api/agents/status":
            self._api_agents_status()
        elif parsed.path == "/api/agents/feed":
            self._api_agents_feed()
        elif parsed.path == "/api/agents/stream":
            self._api_agents_stream()
        elif parsed.path == "/api/agi/score":
            self._api_agi_score()
        elif parsed.path == "/api/llm_teacher/log":
            self._api_llm_teacher_log()
        elif parsed.path == "/api/llm_teacher/seek":
            self._api_llm_teacher_seek()
        elif parsed.path == "/api/benchmark/symbolic":
            self._api_benchmark_symbolic()
        elif parsed.path == "/api/benchmark/logic":
            self._api_benchmark_logic()
        elif parsed.path == "/api/benchmark/coding":
            self._api_benchmark_coding()
        elif parsed.path == "/api/benchmark/arc":
            self._api_benchmark_arc()
        elif parsed.path == "/api/benchmark/all":
            self._api_benchmark_all()
        elif parsed.path == "/api/benchmark/agi":
            self._api_benchmark_agi()
        elif parsed.path == "/api/search/predictor":
            self._api_search_predictor()
        # ── Human-mind modules ────────────────────────────────
        elif parsed.path == "/api/dialogue/session":
            params = parse_qs(parsed.query)
            session_id = params.get("session_id", ["default"])[0]
            self._api_dialogue_session(session_id)
        elif parsed.path == "/api/dialogue/sessions":
            self._api_dialogue_sessions()
        elif parsed.path == "/api/identity":
            self._api_identity()
        elif parsed.path == "/api/autobiography":
            self._api_autobiography()
        elif parsed.path == "/api/homeostasis":
            self._api_homeostasis()
        elif parsed.path == "/api/dream/status":
            self._api_dream_status()
        # ── Brain Orchestrator API ────────────────────────────
        elif parsed.path == "/api/brain/status":
            self._api_brain_status()
        elif parsed.path == "/api/brain/curriculum":
            self._api_brain_curriculum()
        elif parsed.path == "/api/brain/events":
            self._api_brain_events()
        elif parsed.path == "/api/brain/world":
            self._api_brain_world()
        elif parsed.path == "/api/brain/transfer":
            self._api_brain_transfer()
        elif parsed.path == "/api/brain/autolearn":
            self._api_brain_autolearn_status()
        elif parsed.path == "/api/brain/conjectures":
            self._api_brain_conjectures()
        elif parsed.path == "/api/brain/concepts":
            self._api_brain_concepts()
        elif parsed.path == "/api/brain/goals":
            self._api_brain_goals_status()
        elif parsed.path == "/api/brain/environment":
            self._api_brain_environment_status()
        elif parsed.path == "/api/brain/selfmodel":
            self._api_brain_selfmodel()
        elif parsed.path == "/api/brain/metalearner":
            self._api_brain_metalearner()
        elif parsed.path == "/api/brain/physics":
            self._api_brain_physics()
        elif parsed.path == "/api/brain/knowledge":
            self._api_brain_knowledge()
        elif parsed.path == "/api/brain/arena":
            self._api_brain_arena()
        elif parsed.path == "/api/brain/workspace":
            self._api_brain_workspace()
        elif parsed.path == "/api/brain/multimodal":
            self._api_brain_multimodal_demo()
        elif parsed.path == "/api/brain/predictive":
            self._api_brain_predictive()
        elif parsed.path == "/api/brain/trainer":
            self._api_brain_trainer()
        elif parsed.path == "/api/brain/composite":
            self._api_brain_composite()
        elif parsed.path == "/api/brain/astar":
            self._api_brain_astar_stats()
        elif parsed.path == "/api/brain/hypothesis":
            self._api_brain_hypothesis()
        elif parsed.path == "/api/datasets":
            self._api_datasets()
        elif parsed.path == "/api/benchmark/hard":
            self._api_benchmark_hard()
        elif parsed.path == "/api/brain/society":
            self._api_brain_society()
        elif parsed.path == "/api/brain/metacognition":
            self._api_brain_metacognition()
        elif parsed.path == "/api/brain/buffer":
            self._api_brain_buffer()
        elif parsed.path == "/api/brain/blender":
            self._api_brain_blender()
        elif parsed.path == "/api/brain/dialogue":
            self._api_brain_dialogue()
        elif parsed.path == "/api/brain/sensory":
            self._api_brain_sensory()
        elif parsed.path == "/api/brain/dream":
            self._api_brain_dream()
        elif parsed.path == "/api/brain/affective":
            self._api_brain_affective()
        elif parsed.path == "/api/brain/transforms":
            self._api_brain_transforms()
        elif parsed.path == "/api/brain/imagine":
            self._api_brain_imagine()
        elif parsed.path == "/api/brain/redteam":
            self._api_brain_redteam()
        elif parsed.path == "/api/brain/identity":
            self._api_brain_identity()
        elif parsed.path == "/api/brain/stream":
            self._api_brain_stream()
        elif parsed.path == "/api/brain/robust":
            self._api_brain_robust()
        elif parsed.path == "/api/brain/attention":
            self._api_brain_attention()
        elif parsed.path == "/api/brain/tom":
            self._api_brain_tom()
        elif parsed.path == "/api/brain/agentmem":
            self._api_brain_agentmem()
        elif parsed.path == "/api/brain/progress":
            self._api_brain_progress()
        elif parsed.path == "/api/brain/metacurr":
            self._api_brain_metacurr()
        elif parsed.path == "/api/brain/actionphys":
            self._api_brain_actionphys()
        elif parsed.path == "/api/brain/streambridge":
            self._api_brain_streambridge()
        elif parsed.path == "/api/brain/percept":
            self._api_brain_percept()
        elif parsed.path == "/api/domains":
            self._api_domains_status()
        elif parsed.path == "/api/books":
            self._api_books_status()
        elif parsed.path == "/api/knowledge/commonsense":
            self._api_knowledge_commonsense()
        elif parsed.path == "/api/knowledge/lookup":
            params = parse_qs(parsed.query)
            q      = params.get("q",      [""])[0]
            domain = params.get("domain", ["general"])[0]
            self._api_knowledge_lookup(q, domain)
        elif parsed.path == "/api/knowledge/stats":
            self._api_knowledge_stats()
        elif parsed.path == "/api/questions":
            self._api_curiosity_questions()
        elif parsed.path == "/api/metacognition/plan":
            self._api_metacognition_plan()
        # ── Phase D: Teacher Protocol ─────────────────────────
        elif parsed.path == "/api/questions/pending":
            self._api_questions_pending()
        elif parsed.path == "/api/teachers":
            self._api_teachers_list()
        # ── Phase E: Architecture Designer ───────────────────
        elif parsed.path == "/api/architecture/gaps":
            self._api_architecture_gaps()
        elif parsed.path == "/api/architecture/proposals":
            self._api_architecture_proposals()
        # ── Phase F: Forgetting Curve ─────────────────────────
        elif parsed.path == "/api/memory/forgetting":
            self._api_memory_forgetting()
        # ── Phase A: Stage capabilities ───────────────────────
        elif parsed.path == "/api/brain/stage":
            self._api_brain_stage()
        elif parsed.path == "/api/brain/piaget":
            self._api_brain_piaget()
        elif parsed.path == "/api/brain/promoted-rules":
            self._api_brain_promoted_rules()
        # ── Phase B: Predictive Engine ────────────────────────
        elif parsed.path == "/api/predictive/status":
            self._api_predictive_status()
        # ── Phase C: Internal Grammar ─────────────────────────
        elif parsed.path == "/api/grammar/status":
            self._api_grammar_status()
        # ── New wired endpoints ───────────────────────────────
        elif parsed.path == "/api/solve/stream":
            params = parse_qs(parsed.query)
            expr = params.get("expr", ["x + 0"])[0]
            budget = float(params.get("budget", ["10"])[0])
            self._api_solve_stream(expr, budget)
        elif parsed.path == "/api/world/graph":
            self._api_world_graph()
        elif parsed.path == "/api/weakness":
            self._api_weakness()
        elif parsed.path == "/api/world/simulator":
            self._api_world_simulator()
        elif parsed.path == "/api/transfer":
            self._api_transfer()
        elif parsed.path == "/api/grounding":
            self._api_grounding()
        elif parsed.path == "/api/transfer/engine":
            self._api_transfer_engine(qs)
        elif parsed.path == "/api/causal/hierarchy":
            self._api_causal_hierarchy()
        elif parsed.path == "/api/curriculum/map":
            self._api_curriculum_map()
        elif parsed.path == "/api/multitask/scheduler":
            self._api_multitask_scheduler()
        elif parsed.path == "/api/algorithm/selector":
            self._api_algorithm_selector()
        elif parsed.path == "/api/meta/controller":
            self._api_meta_controller()
        elif parsed.path == "/api/consciousness":
            self._api_consciousness()
        elif parsed.path == "/api/intelligence/domains":
            self._api_intelligence_domains()
        elif parsed.path == "/api/intelligence/learning-progress":
            self._api_intelligence_learning_progress()
        elif parsed.path == "/api/intelligence/learning-dashboard":
            self._api_intelligence_learning_dashboard()
        elif parsed.path == "/api/perception/neural":
            self._api_neural_perception_stats()
        elif parsed.path == "/api/novelty":
            self._api_novelty_stats()
        elif parsed.path == "/api/schema/stats":
            self._api_schema_stats()
        elif parsed.path == "/api/physics":
            self._api_physics()
        elif parsed.path == "/api/chemistry":
            self._api_chemistry()
        elif parsed.path == "/api/science/hypothesis":
            self._api_science_hypothesis()
        elif parsed.path == "/api/science/theory":
            self._api_science_theory()
        elif parsed.path == "/api/benchmark/agi":
            self._api_benchmark_agi()
        elif parsed.path == "/api/learning/trend":
            self._api_learning_trend()
        elif parsed.path == "/api/learning/summary":
            self._api_learning_summary()
        elif parsed.path == "/api/learning/live":
            self._api_learning_live()
        elif parsed.path == "/dashboard":
            self._serve_file("dashboard.html", "text/html")
        elif parsed.path in ("/learning-dashboard", "/learning-dashboard/"):
            self._serve_file("learning_dashboard.html", "text/html")
        elif parsed.path in ("/evolve-chat", "/evolve-chat/"):
            self._serve_file("evolver_chat.html", "text/html")
        elif parsed.path in ("/todo.html", "/todo"):
            self._serve_file("todo.html", "text/html")
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)

        if _path_requires_legacy_runtime(parsed.path):
            _ensure_legacy_runtime()
        
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
        elif parsed.path == "/api/daemon/control":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_daemon_control(body)
        elif parsed.path == "/api/solve-nl":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_solve_nl(body)
        elif parsed.path == "/api/search/learn":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_search_learn(body)
        elif parsed.path == "/api/learning/chat":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_learning_chat(body)
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
        elif parsed.path == "/api/world/update":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_world_update(body)
        elif parsed.path == "/api/world/ingest":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_world_ingest(body)
        elif parsed.path == "/api/code/run":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_code_run(body)
        elif parsed.path == "/api/concepts/extract":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_concepts_extract(body)
        elif parsed.path == "/api/llm/chat":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_llm_chat(body)
        elif parsed.path == "/api/llm/config":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_llm_config_update(body)
        elif parsed.path == "/api/autolearn/reflect":
            self._api_autolearn_reflect()
        elif parsed.path == "/api/agents/start":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_agents_start(body)
        elif parsed.path == "/api/agents/stop":
            self._api_agents_stop()
        elif parsed.path == "/api/self-improve/trigger":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_self_improve_trigger(body)
        elif parsed.path == "/api/self-improve/start":
            self._api_self_improve_start()
        elif parsed.path == "/api/self-improve/stop":
            self._api_self_improve_stop()
        elif parsed.path == "/api/self-improve/rollback":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_self_improve_rollback(body)
        elif parsed.path == "/api/self-improve/multi":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_self_improve_multi(body)
        # ── Evolver Chat POST ──────────────────────────────────
        elif parsed.path == "/api/evolve/chat":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_evolve_chat(body)
        elif parsed.path == "/api/evolve/interrupt":
            self._api_evolve_interrupt()
        elif parsed.path == "/api/evolve/feedback":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_evolve_feedback(body)
        elif parsed.path == "/api/evolve/apply":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_evolve_apply(body)
        elif parsed.path == "/api/neuro/dream":
            self._api_neuro_dream()
        elif parsed.path == "/api/neuro/invent-symbol":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_neuro_invent_symbol(body)
        elif parsed.path == "/api/neuro/invent-algorithm":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_neuro_invent_algorithm(body)
        # ── Human-mind modules ────────────────────────────────
        elif parsed.path == "/api/dialogue/chat":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_dialogue_chat(body)
        # ── Brain Orchestrator POST ───────────────────────────
        elif parsed.path == "/api/brain/solve":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_solve(body)
        elif parsed.path == "/api/brain/learn":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_learn(body)
        elif parsed.path == "/api/brain/autolearn":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_autolearn_control(body)
        elif parsed.path == "/api/brain/goals":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_goals_control(body)
        elif parsed.path == "/api/brain/environment":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_environment_run(body)
        elif parsed.path == "/api/brain/selfmodel":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_selfmodel_action(body)
        elif parsed.path == "/api/brain/metalearner":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_metalearner_control(body)
        elif parsed.path == "/api/brain/arena":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_arena_race(body)
        elif parsed.path == "/api/brain/physics":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_physics_run(body)
        elif parsed.path == "/api/brain/multimodal":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_multimodal_parse(body)
        elif parsed.path == "/api/brain/predictive":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_predictive_run(body)
        elif parsed.path == "/api/brain/trainer":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_trainer_control(body)
        elif parsed.path in ("/api/config/beam_width", "/api/config/budget_seconds"):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_config_update(parsed.path, body)
        elif parsed.path == "/api/benchmark/hard":
            self._api_benchmark_hard_run()
        elif parsed.path == "/api/brain/society":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_society_deliberate(body)
        elif parsed.path == "/api/brain/metacognition":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_metacognition_think(body)
        elif parsed.path == "/api/brain/blender":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_blender_action(body)
        elif parsed.path == "/api/brain/dialogue":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_dialogue_turn(body)
        elif parsed.path == "/api/brain/sensory":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_sensory_observe(body)
        elif parsed.path == "/api/brain/dream":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_dream_cycle(body)
        elif parsed.path == "/api/brain/transforms":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_transforms_action(body)
        elif parsed.path == "/api/brain/imagine":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_imagine_action(body)
        elif parsed.path == "/api/brain/redteam":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_redteam_attack(body)
        elif parsed.path == "/api/brain/stream":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_stream_action(body)
        elif parsed.path == "/api/brain/robust":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_robust_batch(body)
        elif parsed.path == "/api/brain/tom":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_tom_action(body)
        elif parsed.path == "/api/brain/agentmem":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_agentmem_action(body)
        elif parsed.path == "/api/brain/progress":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_progress_action(body)
        elif parsed.path == "/api/brain/metacurr":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_metacurr_action(body)
        elif parsed.path == "/api/brain/actionphys":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_actionphys_run(body)
        elif parsed.path == "/api/brain/streambridge":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_streambridge_action(body)
        elif parsed.path == "/api/brain/percept":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_percept_parse(body)
        elif parsed.path == "/api/brain/instruct":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_brain_instruct(body)
        elif parsed.path == "/api/books/ingest":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_books_ingest(body)
        elif parsed.path == "/api/knowledge/expand":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_knowledge_expand(body)
        elif parsed.path == "/api/domains/seed":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_domains_seed(body)
        elif parsed.path == "/api/qa":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_qa(body)
        elif parsed.path == "/api/ask":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_ask(body)
        # ── Phase D: Teacher Protocol (POST) ──────────────────
        elif parsed.path == "/api/questions/answer":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_questions_answer(body)
        elif parsed.path == "/api/teachers/register":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_teachers_register(body)
        # ── Phase E: Architecture Designer (POST) ─────────────
        elif parsed.path == "/api/architecture/trigger":
            self._api_architecture_trigger()
        elif parsed.path == "/api/think":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_think(body)
        else:
            self.send_error(404)

    def _serve_file(self, filename, content_type):
        filepath = self.STATIC_DIR / filename
        if filepath.exists():
            data = filepath.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(data)
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
        if not expr:
            self._json_response({"error": "No expression"}, 400)
            return
        try:
            from sare.brain import get_brain
            brain = get_brain()
            result = brain.solve(
                expression=expr,
                algorithm=body.get("algorithm", "beam"),
                beam_width=body.get("beam_width", 8),
                max_depth=body.get("max_depth", 30),
                budget=body.get("budget", 10.0),
                domain=body.get("domain", "general"),
                kappa=float(body.get("kappa", 0.1)),
                force_python=bool(body.get("force_python", False)),
            )
        except Exception as e:
            self._json_response({"error": str(e)}, 500)
            return

        # Build a reliable explanation: symbolic answer first, LLM enhancement if available
        answer = result.get("answer", "")
        steps_text = result.get("steps_text", "")
        transforms = result.get("transforms_applied", [])
        e_before = result.get("initial", {}).get("energy", {}).get("total", 0)
        e_after  = result.get("result",  {}).get("energy", {}).get("total", 0)
        delta    = result.get("delta", e_before - e_after)
        reduction = result.get("reduction_pct", 0)

        # Always-correct symbolic explanation (no LLM needed)
        symbolic_explain = (
            f"<b style='color:#4fc3f7;font-size:1.3em'>{answer}</b><br><br>"
            f"<span style='color:#aaa'>Input:</span> <code>{expr}</code><br>"
            f"<span style='color:#aaa'>Steps:</span> {len(transforms)}"
            + (f" &nbsp;|&nbsp; <span style='color:#aaa'>Rules:</span> "
               + ", ".join(f"<code>{t}</code>" for t in transforms) if transforms else "")
            + f"<br><span style='color:#aaa'>Energy:</span> {e_before:.1f} → {e_after:.1f}"
              f" &nbsp;<span style='color:#69f0ae'>▼ {delta:.1f} ({reduction:.0f}% reduction)</span>"
        )

        # LLM explanation: only for novel / long proof traces (≥4 steps), never for trivial solves
        nl_explanation = None
        _n_steps_web = len(transforms) if transforms else 0
        if explain_solve_trace and result.get("solve_success") and _n_steps_web >= 4:
            import threading as _thr_llm
            def _call_llm():
                try:
                    explain_solve_trace(
                        problem=expr, transforms_applied=transforms,
                        energy_before=e_before, energy_after=e_after,
                        final_expression=answer, expression=answer,
                        domain=result.get("domain", "general"), goal="simplify",
                    )
                except Exception:
                    pass
            _thr_llm.Thread(target=_call_llm, daemon=True).start()
            # nl_explanation stays None — symbolic_explain is already complete

        result["nl_explanation"] = symbolic_explain + (
            f"<br><br><span style='color:#aaa;font-size:0.9em'>💬 {nl_explanation}</span>"
            if nl_explanation else ""
        )
        result["answer_display"] = answer  # clean field for easy frontend access
        self._json_response(result)


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
            # Fallback: load promoted rules from disk
            out = []
            try:
                import json as _j
                from pathlib import Path as _P
                p = _P(__file__).resolve().parents[3] / "data" / "memory" / "promoted_rules.json"
                if p.exists():
                    data = _j.loads(p.read_text())
                    counts = data.get("pattern_counts", {})
                    for rule in data.get("promoted_rules", []):
                        name = rule.get("name", "unknown")
                        out.append({
                            "name": name,
                            "domain": rule.get("domain", "general"),
                            "confidence": rule.get("confidence", 0.8),
                            "observations": counts.get(name, 1),
                        })
            except Exception as _e:
                                log.debug("[web] Suppressed: %s", _e)
            self._json_response({
                "count": len(out),
                "concepts": out,
                "rules": out,
                "source": "promoted_rules_fallback",
            })
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
            except Exception as _e:
                                log.debug("[web] Suppressed: %s", _e)

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
            except Exception as _e:
                                log.debug("[web] Suppressed: %s", _e)
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
        # Add surprise/prediction stats
        try:
            surprise = experiment_runner.surprise_stats()
            stats["prediction"] = surprise
        except Exception as _e:
                        log.debug("[web] Suppressed: %s", _e)
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
        # Lazily wire transforms: use already-wired domain transforms if available,
        # otherwise fall back to math-only transforms
        if not getattr(experiment_runner, 'transforms', None):
            from sare.engine import get_transforms
            experiment_runner.transforms = get_transforms(include_macros=True)
        if not experiment_runner.reflection_engine:
            try:
                from sare.reflection.py_reflection import get_reflection_engine
                experiment_runner.reflection_engine = get_reflection_engine()
            except Exception as _e:
                                log.debug("[web] Suppressed: %s", _e)
        if not experiment_runner.causal_induction:
            try:
                from sare.causal.induction import CausalInduction
                experiment_runner.causal_induction = CausalInduction()
            except Exception as _e:
                                log.debug("[web] Suppressed: %s", _e)
        if not experiment_runner.concept_registry:
            from sare.memory.concept_seed_loader import SeededConceptRegistry
            experiment_runner.concept_registry = SeededConceptRegistry()
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
                    "proof_steps":   getattr(r, "proof_steps", []),
                    "proof_nl":      getattr(r, "proof_nl", ""),
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
            except Exception as _e:
                                log.debug("[web] Suppressed: %s", _e)
        self._json_response(goal_setter.report())

    def _api_nlparse(self, text: str):
        """GET/POST /api/parse — BasicNLParser: convert NL text → expression."""
        if not _nl_parser:
            self._json_response({"error": "NL Parser unavailable"}, 503)
            return
        if not text:
            self._json_response({"error": "Provide ?q=<text> or POST {\"text\":\"...\"}"}, 400)
            return
        try:
            params = parse_qs(urlparse(self.path).query)
            multi = params.get("multi", ["0"])[0] == "1"

            if multi and hasattr(_nl_parser, "parse_multi"):
                results = _nl_parser.parse_multi(text)
                dicts = []
                for r in results:
                    if hasattr(r, "to_dict"):
                        dicts.append(r.to_dict())
                    elif isinstance(r, dict):
                        dicts.append(r)
                    else:
                        dicts.append({"result": str(r)})
                self._json_response({"results": dicts, "count": len(dicts), "multi": True})
            else:
                result = _nl_parser.parse(text)
                if hasattr(result, "to_dict"):
                    self._json_response(result.to_dict())
                elif isinstance(result, dict):
                    self._json_response(result)
                else:
                    self._json_response({"result": str(result)})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

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

    def _api_weakness(self):
        """GET /api/weakness — WeaknessDetector failure pattern report."""
        try:
            from sare.meta.weakness_detector import get_weakness_detector
            wd = get_weakness_detector()
            report = wd.analyze()
            self._json_response({
                "stats": wd.get_stats(),
                "report": report,
                "synthesis_requests": wd.get_synthesis_requests(top_k=3),
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_world_simulator(self):
        """GET /api/world/simulator — WorldSimulator causal transform ranking stats."""
        try:
            from sare.memory.world_simulator import get_world_simulator
            ws = get_world_simulator()
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            causal_count = len(wm._causal_links)
            domain_stats = {}
            for link in wm._causal_links.values():
                d = link.domain
                domain_stats.setdefault(d, {"links": 0, "avg_conf": 0.0, "_sum": 0.0})
                domain_stats[d]["links"] += 1
                domain_stats[d]["_sum"] += link.confidence
            for d, s in domain_stats.items():
                s["avg_conf"] = round(s["_sum"] / max(s["links"], 1), 3)
                del s["_sum"]
            self._json_response({
                "simulator_stats": ws.get_stats(),
                "causal_links_total": causal_count,
                "domain_breakdown": domain_stats,
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_transfer(self):
        """GET /api/transfer — Cross-domain analogy transfer summary (alias for /api/analogy)."""
        self._api_analogy()

    def _api_transfer_engine(self, qs: dict):
        """GET /api/transfer/engine?domain=<domain> — cross-domain hypothesis map from TransferEngine."""
        domain = (qs.get("domain", [""])[0] or "").strip() or None
        try:
            from sare.causal.analogy_transfer import TransferEngine
            te = TransferEngine()
            te.load()
            if domain:
                suggestions = te.get_transfer_suggestions(domain)
                self._json_response({"domain": domain, "suggestions": suggestions or []})
            else:
                # Return summary across all known domains
                summary: dict = {}
                for dom in ["algebra", "factual", "science", "reasoning", "analogy", "coding", "logic"]:
                    try:
                        sugg = te.get_transfer_suggestions(dom)
                        if sugg:
                            summary[dom] = sugg[:5]
                    except Exception:
                        pass
                self._json_response({"transfer_map": summary})
        except ImportError:
            self._json_response({"error": "TransferEngine not available"}, 503)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_grounding(self):
        """GET /api/grounding — Grounded concept learning stats and physical explanations (T2-2)."""
        try:
            from sare.world.toy_physics import get_grounded_learner
            gcl = get_grounded_learner()
            stats = gcl.get_stats()
            groundings = {}
            for concept, g in gcl.get_all_groundings().items():
                groundings[concept] = {
                    "verified": g.get("verified", False),
                    "explanation": g.get("explanation", ""),
                    "grounded_at": g.get("grounded_at", 0),
                }
            self._json_response({
                "stats": stats,
                "groundings": groundings,
                "status": "ok",
            })
        except ImportError:
            self._json_response({"error": "GroundedConceptLearner not available", "wired": False}, 503)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_causal_hierarchy(self):
        """GET /api/causal/hierarchy — 3-level causal hierarchy summary + active algebra laws."""
        try:
            from sare.memory.causal_hierarchy import get_causal_hierarchy
            ch = get_causal_hierarchy()
            self._json_response({
                "summary": ch.summary(),
                "active_laws_algebra": ch.get_active_laws("algebra"),
                "active_laws_logic": ch.get_active_laws("logic"),
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_curriculum_map(self):
        """GET /api/curriculum/map — current curriculum stage, Leitner boxes, and ZPD per domain."""
        result: dict = {}
        try:
            from sare.brain import get_brain
            b = get_brain()
            dc = getattr(b, "developmental_curriculum", None)
            if dc is not None:
                stage = getattr(b, "stage", None)
                stage_name = stage.value if stage and hasattr(stage, "value") else str(stage)
                domains_data: dict = {}
                for dom_name, dom_obj in getattr(dc, "domains", {}).items():
                    box_counts = []
                    if hasattr(dom_obj, "leitner_boxes"):
                        box_counts = [len(box) for box in dom_obj.leitner_boxes]
                    domains_data[dom_name] = {
                        "difficulty": getattr(dom_obj, "current_difficulty", None),
                        "solve_rate": getattr(dom_obj, "solve_rate", None),
                        "leitner_boxes": box_counts,
                    }
                result = {
                    "stage": stage_name,
                    "domains": domains_data,
                }
            else:
                result = {"error": "DevelopmentalCurriculum not loaded"}
        except Exception as e:
            result = {"error": str(e)}
        self._json_response(result)

    def _api_multitask_scheduler(self):
        """GET /api/multitask/scheduler — Multi-task batch allocation status."""
        try:
            from sare.curiosity.multi_task_scheduler import MultiTaskScheduler
            sched = MultiTaskScheduler()
            status = sched.get_status() if hasattr(sched, "get_status") else {"status": "unavailable"}
            self._json_response(status)
        except ImportError:
            self._json_response({"status": "module_not_found", "message": "MultiTaskScheduler not available"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_algorithm_selector(self):
        """GET /api/algorithm/selector — Best algorithm per task type (epsilon-greedy stats)."""
        try:
            from sare.meta.algorithm_selector import get_algorithm_selector
            selector = get_algorithm_selector()
            stats = selector.get_stats() if hasattr(selector, "get_stats") else {"status": "unavailable"}
            self._json_response(stats)
        except ImportError:
            self._json_response({"status": "module_not_found", "message": "AlgorithmSelector not available"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_meta_controller(self):
        """GET /api/meta/controller — MetaController stats (T3-6 meta-meta-learning)."""
        try:
            from sare.meta.algorithm_selector import get_algorithm_selector
            selector = get_algorithm_selector()
            meta_stats = selector.meta_controller.stats if hasattr(selector, "meta_controller") else {}
            self._json_response({
                "meta_controller": meta_stats,
                "current_epsilon": getattr(selector, "epsilon", 0.1),
                "status": "active",
            })
        except ImportError:
            self._json_response({"status": "module_not_found", "message": "AlgorithmSelector not available"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_consciousness(self):
        """GET /api/consciousness — GlobalWorkspace attention broadcast stats (T3-4)."""
        try:
            from sare.cognition.global_workspace import get_global_workspace
            gw = get_global_workspace()
            self._json_response(gw.stats)
        except ImportError:
            self._json_response({"status": "module_not_found", "message": "GlobalWorkspace not available"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_qa(self, body: dict):
        """POST /api/qa — Question answering from context."""
        question = body.get("question", "")
        context = body.get("context", {})
        try:
            from sare.agent.qa_pipeline import QAPipeline
            pipeline = QAPipeline()
            ctx_dict = context if isinstance(context, dict) else {"text": str(context)}
            q_graph = pipeline.build_question_graph(question, ctx_dict)
            answer = None
            if q_graph is not None:
                answer = pipeline.extract_answer(q_graph, question)
            self._json_response({"question": question, "answer": answer or "No answer found"})
        except ImportError:
            self._json_response({"error": "QAPipeline not available"}, 503)
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
            _ensure_hippocampus_started()
        if not _hippocampus:
            self._json_response({"error": "HippocampusDaemon unavailable"}, 503)
            return
        self._json_response(_hippocampus.status())

    def _api_llm_status(self):
        """GET /api/llm-status — LLM Bridge health and config."""
        if llm_status:
            status = llm_status()
            # Add synthesis_model from config
            try:
                from sare.interface.llm_bridge import _load_config
                cfg = _load_config()
                status["synthesis_model"] = cfg.get("synthesis_model", status.get("model", ""))
            except Exception as _e:
                                log.debug("[web] Suppressed: %s", _e)
            self._json_response(status)
        else:
            self._json_response({"available": False, "error": "LLMBridge not loaded"})

    def _api_llm_rate_limits(self):
        """GET /api/llm/rate-limits — Per-model rate limit tracker for planning."""
        import time as _time
        from sare.interface.llm_bridge import get_rate_limit_report, get_llm_stats
        now = _time.time()
        rl = get_rate_limit_report()
        stats = get_llm_stats()
        recent = stats.get("recent", [])

        # Compute per-model hit/success rate from recent call log
        model_counts: dict = {}
        for c in recent:
            m = c.get("model", "unknown")
            entry = model_counts.setdefault(m, {"ok": 0, "fail": 0, "roles": {}})
            if c.get("ok"):
                entry["ok"] += 1
            else:
                entry["fail"] += 1
            role = c.get("role", "?")
            entry["roles"][role] = entry["roles"].get(role, 0) + 1

        # Build summary per model
        summary = {}
        for model, counts in model_counts.items():
            total = counts["ok"] + counts["fail"]
            rl_info = rl.get(model, {})
            summary[model] = {
                "calls": total,
                "success_rate": round(counts["ok"] / max(total, 1), 2),
                "top_roles": sorted(counts["roles"].items(), key=lambda x: -x[1])[:5],
                "rate_limit_hits": rl_info.get("total_hits", 0),
                "available_now": rl_info.get("available", True),
                "cooldown_remaining_s": rl_info.get("cooldown_remaining_s", 0),
                "hits_per_hour_24h": rl_info.get("hits_per_hour_24h", 0),
                "median_gap_s": rl_info.get("median_gap_s"),
                "last_hit_ago_s": rl_info.get("last_hit_ago_s"),
            }

        # Planning advice
        advice = []
        for model, info in summary.items():
            if "free" in model and info["success_rate"] < 0.5:
                gap = info.get("median_gap_s")
                advice.append({
                    "model": model,
                    "issue": "high fallback rate",
                    "success_rate": info["success_rate"],
                    "recommendation": f"Free tier rate-limiting heavily. Median gap between hits: {gap}s. "
                                      f"Consider spacing calls or raising _MIN_CALL_GAP_S.",
                })
            elif "free" in model and info.get("hits_per_hour_24h", 0) > 5:
                advice.append({
                    "model": model,
                    "issue": "frequent rate limits",
                    "hits_per_hour": info["hits_per_hour_24h"],
                    "recommendation": "Rate limited >5x/hour. Free quota may reset hourly — "
                                      "consider batching calls in the first minute after reset.",
                })

        self._json_response({
            "generated_at": now,
            "models": summary,
            "rate_limit_detail": rl,
            "advice": advice,
            "total_calls_recorded": len(recent),
        })

    def _api_ask(self, body: dict):
        """POST /api/ask — Conversational math endpoint.

        Accepts natural language or symbolic expressions.  Extracts the expression,
        solves it with SARE-HX, and returns a plain-English answer.

        Body: {"question": "what is 2+2"} or {"question": "solve x^2 + 5 = 9"}
        Returns: {"question": ..., "expression": ..., "answer": ..., "steps_text": ...}
        """
        import re as _re
        question = (body.get("question") or body.get("text") or body.get("expression") or "").strip()
        if not question:
            self._json_response({"error": "No question provided"}, 400)
            return

        # Multi-turn context resolution
        session_id = body.get("session_id", "default")
        try:
            from sare.interface.dialogue_context import DialogueContext
            if session_id not in _SESSION_CONTEXTS:
                _SESSION_CONTEXTS[session_id] = DialogueContext()
            _ctx = _SESSION_CONTEXTS[session_id]
            resolved = _ctx.resolve(question)
            if resolved and resolved != question:
                question = resolved
            _ctx.add_turn("user", question)
        except Exception:
            _ctx = None

        # --- Pattern-based NL → expression extraction (no LLM needed) ---
        expr = question
        # "what is X" / "calculate X" / "compute X" / "evaluate X"
        _m = _re.match(
            r"^(?:what\s+is|calculate|compute|evaluate|find|simplify|solve)[\s:]+(.+)$",
            question, _re.IGNORECASE
        )
        if _m:
            expr = _m.group(1).strip()
        # "find the value of x if / where / when X"
        _m2 = _re.match(
            r"^find\s+(?:the\s+)?value\s+of\s+\w+\s+(?:if|where|when)\s+(.+)$",
            question, _re.IGNORECASE
        )
        if _m2:
            expr = _m2.group(1).strip()
        # Strip trailing question marks / periods
        expr = expr.rstrip("?.")

        # --- Try SARE-HX solve ---
        try:
            from sare.brain import get_brain
            brain = get_brain()
            result = brain.solve(
                expression=expr,
                algorithm="beam",
                beam_width=8,
                max_depth=30,
                budget=10.0,
            )
        except Exception as e:
            self._json_response({"error": f"Solve failed: {e}"}, 500)
            return

        answer = result.get("answer")
        steps_text = result.get("steps_text")
        success = result.get("solve_success", False)

        # Format a conversational reply
        if success and answer:
            reply = f"{answer}"
        elif answer:
            reply = f"≈ {answer} (partial simplification)"
        else:
            reply = "I couldn't find a closed-form answer for that expression."

        # Record brain reply in dialogue context
        try:
            if _ctx is not None and reply:
                _ctx.add_turn("brain", reply)
        except Exception:
            pass

        self._json_response({
            "question": question,
            "expression": expr,
            "answer": answer,
            "reply": reply,
            "steps_text": steps_text,
            "transforms_applied": result.get("transforms_applied", []),
            "success": success,
            "session_id": session_id,
        })

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

    def _api_search_learn(self, body: dict):
        """POST /api/search/learn — Search Wikipedia to learn about a topic.

        Body: {"topic": "water molecule", "question": "...", "answer": "..."}
        Returns: {facts_added, source, extract, answer_found}
        """
        topic = body.get("topic", "").strip()
        question = body.get("question", "").strip()
        answer = body.get("answer", "").strip() or None
        if not topic and not question:
            self._json_response({"error": "Provide 'topic' or 'question'"}, 400)
            return
        try:
            from sare.learning.web_learner import get_web_learner
            wl = get_web_learner()
            if question:
                result = wl.learn(question, expected_answer=answer, domain="general")
            else:
                result = wl.learn_topic(topic)
            self._json_response({
                "facts_added": result.get("facts_added", 0),
                "source": result.get("source", ""),
                "extract": result.get("extract", "")[:400],
                "answer_found": result.get("answer_found", False),
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_learning_chat(self, body: dict):
        """POST /api/learning/chat — Chat with the learning monitor about system progress."""
        import time as _time
        import json as _json
        from pathlib import Path as _Path

        message = body.get("message", "").strip()
        if not message:
            self._json_response({"error": "empty message"}, 400)
            return

        # ── Build context snapshot ────────────────────────────────────────────
        context_lines = []

        # Domain accuracy
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            acc_lines = []
            for key, hist in wm._belief_accuracy.items():
                if key.startswith("domain_solve_acc:") and hist:
                    dom = key[len("domain_solve_acc:"):]
                    window = min(50, len(hist))
                    pct = round(sum(hist[-window:]) / window * 100, 1)
                    acc_lines.append(f"  {dom}: {pct}%")
            if acc_lines:
                context_lines.append("Domain accuracy (rolling 50):\n" + "\n".join(sorted(acc_lines, reverse=True)))
        except Exception:
            pass

        # KB stats
        try:
            from sare.knowledge.commonsense import get_commonsense_base
            cs = get_commonsense_base()
            answer_to = sum(1 for triples in cs._forward.values() for r, _ in triples if r == "AnswerTo")
            total = sum(len(v) for v in cs._forward.values())
            context_lines.append(f"Knowledge base: {answer_to} AnswerTo triples, {total} total triples")
        except Exception:
            pass

        # Recent web searches
        try:
            from sare.learning.web_learner import get_web_learner
            searches = get_web_learner().get_search_log(10)
            if searches:
                s_lines = [f"  [{s['domain']}] {s['title']} (+{s['facts_added']} facts)" for s in searches[:5]]
                context_lines.append("Recent Wikipedia searches:\n" + "\n".join(s_lines))
        except Exception:
            pass

        # LLM stats
        try:
            from sare.interface.llm_bridge import get_llm_stats
            stats = get_llm_stats()
            context_lines.append(f"LLM calls this session: {stats['total_calls']} total")
            by_role = stats.get("by_role", {})
            if by_role:
                role_str = ", ".join(f"{k}:{v}" for k, v in sorted(by_role.items(), key=lambda x: -x[1])[:5])
                context_lines.append(f"  By role: {role_str}")
        except Exception:
            pass

        context = "\n\n".join(context_lines) if context_lines else "No data available yet."

        # ── Ask LLM ───────────────────────────────────────────────────────────
        system_prompt = (
            "You are the SARE-HX AGI system's learning monitor assistant. "
            "Answer questions about the system's current learning progress concisely and factually. "
            "Use the data provided. Keep responses under 3 sentences unless detail is requested."
        )
        prompt = f"Current system state:\n{context}\n\nUser question: {message}"

        try:
            from sare.interface.llm_bridge import _call_model, llm_available
            if llm_available and not llm_available():
                raise RuntimeError("LLM offline")
            response = _call_model(prompt, role="learning_monitor", system_prompt=system_prompt)
            self._json_response({
                "response": response.strip(),
                "context": context,
                "llm_used": True,
                "ts": _time.time(),
            })
        except Exception as _llm_err:
            # Fallback: answer directly from data without LLM
            self._json_response({
                "response": f"System state summary:\n{context}",
                "context": context,
                "llm_used": False,
                "llm_error": str(_llm_err),
                "ts": _time.time(),
            })

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

    def _api_llm_teacher_log(self):
        """GET /api/llm_teacher/log — Return LLM teacher interaction history."""
        import json as _json
        from pathlib import Path as _Path
        log_path = _Path(__file__).resolve().parents[3] / "data" / "memory" / "llm_teacher_log.json"
        try:
            entries = _json.loads(log_path.read_text()) if log_path.exists() else []
            self._json_response({
                "total_seeks": len(entries),
                "total_rules_injected": sum(e.get("rules_injected", 0) for e in entries),
                "total_seeds_added": sum(e.get("seeds_added", 0) for e in entries),
                "recent": entries[-10:],
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_llm_teacher_seek(self):
        """GET /api/llm_teacher/seek — Trigger an LLM teacher seek (async, returns immediately)."""
        import threading as _threading
        try:
            from sare.meta.llm_teacher import get_llm_teacher
            teacher = get_llm_teacher()
            # Collect stuck expressions from global experiment_runner
            stuck = []
            if experiment_runner:
                try:
                    stuck = [
                        getattr(r, "expression", None) or ""
                        for r in (experiment_runner._history or [])[-30:]
                        if not getattr(r, "solved", True)
                    ]
                    stuck = [e for e in stuck if e][:8]
                except Exception:
                    pass
            if not stuck:
                stuck = ["x + 3 = 7", "sin(0) + 0", "x + x + x", "2 * x = 6"]

            def _run():
                try:
                    teacher.seek_and_apply(stuck_exprs=stuck, cycle=9999)
                except Exception as _e:
                    log.warning("[web] llm_teacher seek error: %s", _e)

            _threading.Thread(target=_run, daemon=True, name="llm-teacher-web").start()
            self._json_response({"status": "started", "stuck_exprs": stuck})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

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

    # ── WorldModel endpoints ──────────────────────────────────────

    def _api_compose(self, subject: str, target, hops: int):
        """GET /api/compose?subject=fire&target=ice&hops=3 — derive new facts via belief chaining."""
        try:
            from sare.cognition.fact_composer import get_fact_composer
            if not subject:
                self._json_response({"error": "subject parameter required"}, 400)
                return
            fce = get_fact_composer()
            derived = fce.compose(subject, target=target or None, max_hops=min(hops, 4))
            self._json_response({
                "subject": subject,
                "target": target,
                "graph_nodes": len(fce._graph),
                "graph_edges": fce._graph_size,
                "derived": [
                    {
                        "predicate":   df.predicate,
                        "object":      df.obj,
                        "confidence":  df.confidence,
                        "path_length": df.path_length,
                        "explanation": df.explanation,
                    }
                    for df in derived
                ],
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_world_summary(self):
        """GET /api/world — full world model summary."""
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            self._json_response(wm.summary())
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_world_predictions(self):
        """GET /api/world/predictions — prediction accuracy and surprise stats."""
        try:
            stats = {}
            if experiment_runner:
                try:
                    stats = experiment_runner.surprise_stats()
                except Exception as _e:
                                        log.debug("[web] Suppressed: %s", _e)
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            try:
                stats.update(wm.prediction_stats())
            except Exception as _e:
                                log.debug("[web] Suppressed: %s", _e)
            self._json_response(stats)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_world_imagine(self, seed: str, depth: int = 2):
        """GET /api/world/imagine?seed=... — generate hypotheses from seed concept."""
        if not seed:
            self._json_response({"error": "Missing ?seed= parameter"}, 400)
            return
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            hypotheses = wm.imagine(seed, depth=depth)
            self._json_response({
                "seed": seed,
                "depth": depth,
                "count": len(hypotheses),
                "hypotheses": hypotheses,
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_world_simulate(self, scenario: str, steps: int = 3):
        """GET /api/world/simulate?scenario=... — trace causal consequences."""
        if not scenario:
            self._json_response({"error": "Missing ?scenario= parameter"}, 400)
            return
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            consequences = wm.simulate(scenario, steps=steps)
            self._json_response({
                "scenario": scenario,
                "steps": steps,
                "consequences": consequences,
                "count": len(consequences),
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_world_analogy(self, source: str, target: str):
        """GET /api/world/analogy?source=...&target=... — structural analogy."""
        if not source or not target:
            self._json_response({"error": "Missing ?source= or ?target= parameter"}, 400)
            return
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            result = wm.generate_analogy(source, target)
            if result:
                self._json_response(result)
            else:
                self._json_response({
                    "source_rule": source,
                    "target_domain": target,
                    "analogy": None,
                    "message": "No structural analogy found",
                })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_world_counterfactual(self, rule: str, negated: bool = True):
        """GET /api/world/counterfactual?rule=...&negated=true — what-if reasoning."""
        if not rule:
            self._json_response({"error": "Missing ?rule= parameter"}, 400)
            return
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            result = wm.counterfactual(rule, negated=negated)
            self._json_response(result)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_world_update(self, body: dict):
        """POST /api/world/update — add a fact or causal link to the world model."""
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()

            added = []
            # Add a plain fact
            if "fact" in body:
                domain = body.get("domain", "general")
                fact = body["fact"]
                confidence = float(body.get("confidence", 0.9))
                wm.add_fact(domain, fact, confidence, source="api")
                added.append(f"fact: {fact}")

            # Add a causal link
            if "cause" in body and "effect" in body:
                cause = body["cause"]
                effect = body["effect"]
                mechanism = body.get("mechanism", "user-specified")
                domain = body.get("domain", "general")
                confidence = float(body.get("confidence", 0.85))
                wm.add_causal_link(cause, effect, mechanism, domain, confidence)
                added.append(f"causal link: {cause} → {effect}")

            if added:
                wm.save()
                self._json_response({"status": "ok", "added": added})
            else:
                self._json_response({
                    "status": "no_op",
                    "hint": "Provide 'fact' (with optional 'domain', 'confidence') "
                            "or 'cause'+'effect' (with 'mechanism', 'domain')",
                })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_world_hypotheses(self):
        """GET /api/world/hypotheses — LLM-generated hypotheses from high-surprise events."""
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            hyps = getattr(wm, "_hypotheses", [])
            # Also load persisted hypotheses file if in-memory is empty
            if not hyps:
                from pathlib import Path
                _p = Path(__file__).resolve().parents[3] / "data" / "memory" / "world_hypotheses.json"
                if _p.exists():
                    import json as _j
                    hyps = _j.loads(_p.read_text())
            self._json_response({
                "count": len(hyps),
                "hypotheses": sorted(hyps, key=lambda h: -h.get("evidence", 1)),
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_world_schema_learn(self, domain: str):
        """GET /api/world/schema/learn?domain=... — trigger LLM schema synthesis."""
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            schema = wm.learn_schema_from_llm(domain)
            if schema:
                wm.save()
                self._json_response({"status": "learned", "domain": domain, "schema": schema})
            else:
                self._json_response({
                    "status": "insufficient_data",
                    "domain": domain,
                    "hint": "Need 10+ successful predictions in this domain first",
                })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_world_beliefs(self, domain: str = None):
        """GET /api/world/beliefs[?domain=...] — Bayesian belief tracking (v3)."""
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            beliefs = wm.get_beliefs(domain)
            self._json_response({
                "domain": domain or "all",
                "count": len(beliefs),
                "beliefs": beliefs[:100],
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_world_analogies(self, domain: str = None):
        """GET /api/world/analogies[?domain=...] — discovered structural analogies (v3)."""
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            # Also trigger discovery if few analogies
            if len(wm._analogies) < 5 and len(wm._solve_history) >= 30:
                wm.discover_analogies()
            analogies = wm.get_analogies(domain)
            self._json_response({
                "domain": domain or "all",
                "count": len(analogies),
                "analogies": analogies,
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_world_predict(self, expression: str, domain: str = "arithmetic"):
        """GET /api/world/predict?expression=...&domain=... — v3-style ranked prediction."""
        try:
            from sare.memory.world_model import get_world_model
            from sare.engine import get_transforms
            wm = get_world_model()
            transforms = [t.name() if hasattr(t, "name") else str(t)
                          for t in get_transforms()]
            result = wm.predict(expression, domain, transforms)
            self._json_response(result)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_world_consistency(self):
        """GET /api/world/consistency — run LLM pairwise consistency check on promoted rules."""
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            # Gather promoted rule names from concept registry if available
            rule_names = []
            try:
                if concept_registry is not None:
                    rule_names = [r.name for r in concept_registry.get_rules()]
            except Exception as _e:
                                log.debug("[web] Suppressed: %s", _e)
            if len(rule_names) < 2:
                self._json_response({"status": "insufficient_rules", "rule_count": len(rule_names)})
                return
            conflicts = wm.check_all_rules_consistency(rule_names)
            self._json_response({
                "rules_checked": len(rule_names),
                "conflicts": conflicts,
                "consistent": len(conflicts) == 0,
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_world_graph(self):
        """GET /api/world/graph — causal link graph for visualization."""
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            links = list(wm._causal_links.values()) if hasattr(wm, "_causal_links") else []
            # Build node set from domains
            domain_counts: dict = {}
            for link in links:
                for d in (getattr(link, "domain", "general"),):
                    domain_counts[d] = domain_counts.get(d, 0) + 1
            nodes = [{"id": d, "domain": d, "link_count": c} for d, c in domain_counts.items()]
            edges = [
                {
                    "cause":          getattr(l, "cause", ""),
                    "effect":         getattr(l, "effect", ""),
                    "mechanism":      getattr(l, "mechanism", ""),
                    "domain":         getattr(l, "domain", "general"),
                    "confidence":     round(getattr(l, "confidence", 0.5), 3),
                    "evidence_count": getattr(l, "evidence_count", 1),
                }
                for l in links[:500]  # cap at 500 edges
            ]
            self._json_response({
                "node_count": len(nodes),
                "edge_count": len(edges),
                "nodes": nodes,
                "edges": edges,
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_world_ingest(self, body: dict):
        """POST /api/world/ingest — ingest free-form text into the world model via LLM."""
        text = body.get("text", "").strip()
        domain = body.get("domain", "general")
        if not text:
            self._json_response({"error": "text required"}, 400)
            return
        try:
            from sare.interface.llm_bridge import ingest_text_for_world_model
            result = ingest_text_for_world_model(text, domain)
            self._json_response({"status": "ok", "domain": domain, **result})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_world_live(self):
        """GET /api/world/live — LiveWorld persistent state + recent discoveries."""
        try:
            from sare.world.live_world import get_live_world
            lw = get_live_world()
            summary = lw.summary()
            recent = lw.get_recent(20)
            self._json_response({
                "ok": True,
                "summary": summary,
                "recent_discoveries": recent,
                "objects": list(lw._objects.keys()),
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_code_run(self, body: dict):
        """POST /api/code/run — execute code in sandbox, return stdout."""
        code = body.get("code", "").strip()
        if not code:
            self._json_response({"error": "Missing 'code' field"}, 400)
            return
        try:
            from sare.execution.code_executor import get_executor
            result = get_executor().execute(code)
            self._json_response({
                "ok": not result.blocked and result.exit_code == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "elapsed_ms": round(result.elapsed_ms, 1),
                "blocked": result.blocked,
                "block_reason": result.block_reason,
                "timed_out": result.timed_out,
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _build_sare_system_context(self) -> str:
        """Build a rich system context string from live SARE state for LLM prompts."""
        lines = [
            "You are SARE-HX, a self-improving symbolic reasoning AI.",
            "You have direct awareness of your own internal state. Here is your current status:",
        ]
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            wm_links = len(wm._causal_links)
            wm_facts = sum(len(v) for v in wm._facts.values())
            wm_schemas = len(wm._schemas)
            solve_counts = getattr(wm, "_solve_counts", {})
            total_solves = sum(solve_counts.values())
            pred_stats = wm.prediction_stats() if hasattr(wm, "prediction_stats") else {}
            accuracy = pred_stats.get("accuracy", 0)
            lines.append(f"- World model: {wm_links} causal links, {wm_facts} facts, {wm_schemas} schemas")
            lines.append(f"- Total problems solved: {total_solves}")
            if solve_counts:
                top = sorted(solve_counts.items(), key=lambda x: -x[1])[:5]
                lines.append(f"- Domain solve counts: {', '.join(f'{d}={n}' for d, n in top)}")
            if accuracy:
                lines.append(f"- World model prediction accuracy: {accuracy:.1%}")
            recent = wm.get_activity_log()[:5]
            if recent:
                lines.append("- Recent learning events:")
                for e in recent:
                    lines.append(f"  • [{e.get('type','')}] {e.get('message','')[:80]}")
        except Exception as _e:
                        log.debug("[web] Suppressed: %s", _e)
        try:
            from sare.curiosity.experiment_runner import get_experiment_runner
            er = get_experiment_runner()
            if er and hasattr(er, "_domain_stats"):
                domain_stats = er._domain_stats
                if domain_stats:
                    struggling = [(d, s) for d, s in domain_stats.items()
                                  if s.get("total", 0) > 0 and s.get("solved", 0) / s["total"] < 0.5]
                    if struggling:
                        lines.append(f"- Struggling domains (< 50% solve rate): "
                                     + ", ".join(f"{d}({s['solved']}/{s['total']})" for d, s in struggling[:4]))
        except Exception as _e:
                        log.debug("[web] Suppressed: %s", _e)
        try:
            from sare.memory.concept_seed_loader import SeededConceptRegistry
            reg = SeededConceptRegistry()
            rules = reg.get_rules()
            if rules:
                rule_names = [getattr(r, "name", str(r)) for r in rules[:10]]
                lines.append(f"- Promoted rules ({len(rules)} total): {', '.join(rule_names)}")
        except Exception as _e:
                        log.debug("[web] Suppressed: %s", _e)
        lines.append("\nAnswer concisely and specifically based on the above data.")
        return "\n".join(lines)

    def _api_llm_chat(self, body: dict):
        """POST /api/llm/chat — direct LLM chat with live SARE context."""
        message = body.get("message", "").strip()
        history = body.get("history", [])  # list of {role, content}
        if not message:
            self._json_response({"error": "message required"}, 400)
            return
        try:
            from sare.interface.llm_bridge import _call_llm, llm_available as _llm_avail
            if not _llm_avail():
                self._json_response({"error": "LLM offline — ensure Ollama is running"}, 503)
                return
            # Build system context from live SARE state (as system role message)
            sys_ctx = self._build_sare_system_context()
            # Build conversation history as user prompt
            context_lines = []
            for turn in history[-6:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                context_lines.append(f"{'User' if role == 'user' else 'SARE'}: {content}")
            history_str = "\n".join(context_lines) + "\n" if context_lines else ""
            prompt = f"{history_str}{message}"
            reply = _call_llm(prompt, system_prompt=sys_ctx)
            # Strip "SARE:" prefix if model echoed it
            reply = reply.strip()
            if reply.startswith("SARE:"):
                reply = reply[5:].strip()
            self._json_response({"reply": reply, "model": "sare-hx"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_autolearn_reflect(self):
        """POST /api/autolearn/reflect — trigger immediate LLM reflection with full SARE context."""
        try:
            from sare.interface.llm_bridge import reflect_and_plan, llm_available as _llm_avail
            if not _llm_avail():
                self._json_response({"error": "LLM offline"}, 503)
                return
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            # Gather stats
            solve_counts = getattr(wm, "_solve_counts", {})
            total = sum(solve_counts.values())
            wm_links = len(wm._causal_links)
            wm_facts = sum(len(v) for v in wm._facts.values())
            stats = {
                "total_solves": total,
                "domain_counts": solve_counts,
                "wm_links": wm_links,
                "wm_facts": wm_facts,
            }
            # Find high-surprise domains (low solve rate)
            high_surprise = []
            try:
                from sare.curiosity.experiment_runner import get_experiment_runner
                er = get_experiment_runner()
                if er and hasattr(er, "_domain_stats"):
                    for d, s in er._domain_stats.items():
                        if s.get("total", 0) >= 3 and s.get("solved", 0) / s["total"] < 0.5:
                            high_surprise.append(d)
            except Exception as _e:
                                log.debug("[web] Suppressed: %s", _e)
            # Recent promoted rules
            recent_rules = []
            try:
                from sare.memory.concept_seed_loader import SeededConceptRegistry
                reg = SeededConceptRegistry()
                rules = reg.get_rules()
                recent_rules = [getattr(r, "name", str(r)) for r in rules[-5:]]
            except Exception as _e:
                                log.debug("[web] Suppressed: %s", _e)
            result = reflect_and_plan(stats, high_surprise, recent_rules)
            wm.log_activity("reflect", "all", f"Manual reflection: focus={result.get('curriculum_focus','?')} gaps={result.get('knowledge_gaps',[])[:2]}")
            self._json_response({
                "status": "ok",
                "result": result,
                "stats": stats,
                "high_surprise_domains": high_surprise,
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_llm_config_update(self, body: dict):
        """POST /api/llm/config — update LLM config (provider, model, etc.)."""
        import json as _j
        from pathlib import Path as _Path
        cfg_path = _Path(__file__).resolve().parents[3] / "configs" / "llm.json"
        try:
            cfg = _j.loads(cfg_path.read_text()) if cfg_path.exists() else {}
            allowed_keys = {"provider", "model", "synthesis_model", "temperature",
                           "max_tokens", "api_key", "ollama_url"}
            for k, v in body.items():
                if k in allowed_keys:
                    cfg[k] = v
            cfg_path.write_text(_j.dumps(cfg, indent=4))
            # Force reload
            from sare.interface.llm_bridge import _load_config
            _load_config(force_reload=True)
            self._json_response({"status": "ok", "config": {k: cfg[k] for k in allowed_keys if k in cfg}})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_concepts_extract(self, body: dict):
        """POST /api/concepts/extract — extract concepts from text via LLM."""
        text = body.get("text", "").strip()
        if not text:
            self._json_response({"error": "text required"}, 400)
            return
        try:
            from sare.interface.llm_bridge import extract_concepts
            concepts = extract_concepts(text)
            self._json_response({"concepts": concepts, "count": len(concepts)})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_synth_stats(self):
        """GET /api/synth/stats — LLM transform synthesizer statistics."""
        try:
            from sare.learning.llm_transform_synthesizer import get_llm_synthesizer
            synth = get_llm_synthesizer()
            stats = synth.stats()
            # Also list synthesized module files
            from pathlib import Path
            synth_dir = Path(__file__).resolve().parents[3] / "data" / "memory" / "synthesized_modules"
            files = [f.name for f in synth_dir.glob("*.py")] if synth_dir.exists() else []
            stats["synthesized_files"] = files
            self._json_response(stats)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_synth_review(self):
        """GET /api/synth/review — last 20 synthesis attempts with validation and inventiveness stats."""
        try:
            from sare.learning.llm_transform_synthesizer import get_llm_synthesizer
            synth = get_llm_synthesizer()
            if synth is None:
                self._json_response({"pending_review": [], "recently_promoted": [], "stats": {}})
                return

            stats = synth.stats() if hasattr(synth, "stats") else {}

            attempts = synth._attempts[-20:]
            pending = [a for a in attempts if not a.get("promoted", False)]
            promoted = [a for a in attempts if a.get("promoted", False)]

            self._json_response({
                "pending_review": pending,
                "recently_promoted": promoted,
                "total_attempts": len(attempts),
                "stats": stats,
            })
        except Exception as e:
            self._json_response({"error": str(e), "pending_review": [], "recently_promoted": []})

    def _api_llm_few_shot_stats(self):
        """GET /api/llm/few-shot — FewShotAdapter statistics."""
        try:
            from sare.learning.few_shot_adapter import get_few_shot_adapter
            adapter = get_few_shot_adapter()
            self._json_response(adapter.get_stats())
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_novelty_stats(self):
        """GET /api/novelty — NoveltyDetector statistics."""
        try:
            from sare.cognition.novelty_detector import get_novelty_detector
            self._json_response(get_novelty_detector().stats)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_schema_stats(self):
        """GET /api/schema/stats — schema cache statistics and recent induction results."""
        try:
            from sare.cognition.schema_matcher import get_schema_matcher
            self._json_response(get_schema_matcher().stats)
        except Exception as e:
            self._json_response({"error": str(e), "cached_schemas": 0}, 500)

    def _api_physics(self):
        """GET /api/physics — physics domain stats."""
        try:
            from sare.engine import get_transforms
            transforms = get_transforms()
            phys_transforms = [t.name() for t in transforms if "physics" in t.name() or "kinematic" in t.name() or "ohm" in t.name() or "newton" in t.name()]
            self._json_response({
                "domain": "physics",
                "transforms": phys_transforms,
                "transform_count": len(phys_transforms),
                "status": "active",
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_chemistry(self):
        """GET /api/chemistry — chemistry domain stats."""
        try:
            from sare.engine import get_transforms
            from sare.knowledge.problem_factory import ProblemFactory
            transforms = get_transforms()
            chem_transforms = [
                t.name() for t in transforms
                if "chemistry" in t.name() or "stoich" in t.name()
                or "avogadro" in t.name() or "conservation_mass" in t.name()
            ]
            factory = ProblemFactory()
            problems = factory.generate_chemistry(n=5) if hasattr(factory, "generate_chemistry") else []
            self._json_response({
                "domain": "chemistry",
                "transforms": chem_transforms,
                "transform_count": len(chem_transforms),
                "sample_problems": [p.get("expression", "") for p in problems[:3]],
                "status": "active",
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_autolearn_log(self):
        """GET /api/autolearn/log — recent auto-learning events from world model."""
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            events = wm.get_activity_log()
            # Also include solve count per domain
            solve_counts = getattr(wm, "_solve_counts", {})
            total_links = len(wm._causal_links)
            total_facts = sum(len(v) for v in wm._facts.values())
            self._json_response({
                "events": events,
                "total_events": len(events),
                "solve_counts": solve_counts,
                "wm_links": total_links,
                "wm_facts": total_facts,
                "wm_schemas": len(wm._schemas),
            })
        except Exception as e:
            self._json_response({"error": str(e), "events": []})

    # ── Multi-Agent Learning endpoints ───────────────────────────────────────

    def _api_agents_status(self):
        """GET /api/agents/status — all agent statuses."""
        try:
            from sare.curiosity.multi_agent_learner import get_multi_agent_learner
            self._json_response(get_multi_agent_learner().get_status())
        except Exception as e:
            self._json_response({"error": str(e), "running": False, "agents": []})

    def _api_agents_feed(self):
        """GET /api/agents/feed?since=0 — paginated agent event feed."""
        try:
            import urllib.parse as _up
            params = _up.parse_qs(_up.urlparse(self.path).query)
            since = int(params.get("since", ["0"])[0])
            from sare.curiosity.multi_agent_learner import get_multi_agent_learner
            mal = get_multi_agent_learner()
            self._json_response({
                "events": mal.get_feed(since_id=since),
                "running": mal._running,
            })
        except Exception as e:
            self._json_response({"error": str(e), "events": []})

    def _api_agents_stream(self):
        """GET /api/agents/stream — SSE real-time agent event stream."""
        import time as _time
        from sare.curiosity.multi_agent_learner import get_multi_agent_learner
        mal = get_multi_agent_learner()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        last_id = 0
        try:
            while True:
                new_evs = mal.get_feed_since(last_id)
                for ev in new_evs:
                    data = json.dumps(ev.to_dict())
                    self.wfile.write(f"id: {ev.id}\ndata: {data}\n\n".encode())
                    last_id = ev.id
                if new_evs:
                    self.wfile.flush()
                _time.sleep(0.4)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _api_agents_start(self, body: dict):
        """POST /api/agents/start — start N parallel agents."""
        try:
            from sare.curiosity.multi_agent_learner import get_multi_agent_learner
            n = int(body.get("n", 3))
            result = get_multi_agent_learner().start(n_agents=n)
            self._json_response(result)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_agents_stop(self):
        """POST /api/agents/stop — stop all agents."""
        try:
            from sare.curiosity.multi_agent_learner import get_multi_agent_learner
            result = get_multi_agent_learner().stop()
            self._json_response(result)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── Self-Improvement endpoints ────────────────────────────────────────────

    def _api_self_improve_status(self):
        """GET /api/self-improve/status — debate history, applied patches, daemon state."""
        try:
            from sare.meta.self_improver import get_self_improver
            self._json_response(get_self_improver().get_status())
        except Exception as e:
            self._json_response({"error": str(e), "running": False, "patches": []})

    def _api_self_improve_patches(self):
        """GET /api/self-improve/patches — list all patches (applied + rejected)."""
        try:
            from sare.meta.self_improver import get_self_improver
            self._json_response({"patches": get_self_improver().get_patches()})
        except Exception as e:
            self._json_response({"error": str(e), "patches": []})

    def _api_bottleneck(self):
        """GET /api/bottleneck — ranked list of files that most need improvement."""
        try:
            from sare.meta.bottleneck_analyzer import BottleneckAnalyzer
            targets = BottleneckAnalyzer().analyze()
            self._json_response({
                "targets": [t.to_dict() for t in targets[:10]],
                "total":   len(targets),
            })
        except Exception as e:
            self._json_response({"error": str(e), "targets": []})

    def _api_forgetting(self):
        """GET /api/forgetting — EWC-lite forgetting prevention stats."""
        try:
            from sare.learning.forgetting_prevention import get_forgetting_prevention
            self._json_response(get_forgetting_prevention().stats)
        except Exception as e:
            self._json_response({"error": str(e), "tracked_rules": 0, "consolidated": 0})

    def _api_mind_stream(self):
        """GET /api/mind/stream — last 50 inner monologue thoughts."""
        try:
            from sare.meta.inner_monologue import get_inner_monologue
            im = get_inner_monologue()
            self._json_response({
                "stream": im.get_stream(last_n=50),
                "stats": im.get_stats(),
                "narrate": im.narrate(last_n=10),
            })
        except Exception as e:
            self._json_response({"error": str(e), "stream": []})

    def _api_mind_questions(self):
        """GET /api/mind/questions — active self-generated questions."""
        try:
            from sare.curiosity.question_generator import get_question_generator
            qg = get_question_generator()
            self._json_response({
                "questions": qg.get_all(),
                "pending_count": len(qg.get_pending_questions()),
                "total": len(qg._questions),
            })
        except Exception as e:
            self._json_response({"error": str(e), "questions": []})

    def _api_agi_evolution(self):
        """GET /api/agi/evolution — real-time AGI evolution velocity across all subsystems."""
        try:
            from sare.meta.evolution_monitor import get_evolution_monitor
            report = get_evolution_monitor().get_report()
            self._json_response(report)
        except Exception as e:
            self._json_response({"error": str(e), "velocity_score": 0.0})

    def _api_learning_monitor(self):
        """GET /api/learning/monitor — internal self-evaluation: pattern mastery, velocity."""
        try:
            from sare.meta.learning_monitor import get_learning_monitor
            self._json_response(get_learning_monitor().summary())
        except Exception as e:
            self._json_response({"error": str(e)})

    def _api_agi_value_net(self):
        """GET /api/agi/value-net — MLX value network stats (M1 GPU training)."""
        try:
            from sare.heuristics.mlx_value_net import get_value_net
            self._json_response(get_value_net().get_stats())
        except Exception as e:
            self._json_response({"error": str(e), "ready": False})

    def _api_self_improve_trigger(self, body: dict):
        """POST /api/self-improve/trigger — run one debate-and-patch cycle.
        Body (optional): {"target_file": "...", "improvement_type": "optimize|extend|fix"}
        """
        import threading as _th
        try:
            from sare.meta.self_improver import get_self_improver
            si = get_self_improver()
            target_file     = body.get("target_file")
            improvement_type = body.get("improvement_type")
            # Run in background thread (may take 30-60s for LLM calls)
            result_box = {}
            def _run():
                result_box["result"] = si.run_once(
                    target_file=target_file,
                    improvement_type=improvement_type,
                )
            t = _th.Thread(target=_run, daemon=True, name="SelfImproveTrigger")
            t.start()
            t.join(timeout=120)  # up to 2 min
            result = result_box.get("result", {"outcome": "timeout"})
            self._json_response(result)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_self_improve_start(self):
        """POST /api/self-improve/start — start the improvement daemon."""
        try:
            from sare.meta.self_improver import get_self_improver
            self._json_response(get_self_improver().start())
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_self_improve_stop(self):
        """POST /api/self-improve/stop — stop the improvement daemon."""
        try:
            from sare.meta.self_improver import get_self_improver
            self._json_response(get_self_improver().stop())
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_self_improve_rollback(self, body: dict):
        """POST /api/self-improve/rollback — roll back a patch by patch_id."""
        patch_id = body.get("patch_id", "")
        if not patch_id:
            self._json_response({"error": "patch_id required"}, 400)
            return
        try:
            from sare.meta.self_improver import get_self_improver
            self._json_response(get_self_improver().rollback(patch_id))
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_self_improve_multi(self, body: dict):
        """POST /api/self-improve/multi — run a collective multi-file improvement cycle."""
        cluster_name = body.get("cluster_name") or None
        try:
            from sare.meta.self_improver import get_self_improver
            si = get_self_improver()
            # Run in background thread — respond immediately
            import threading
            result_box = {}
            ev = threading.Event()
            def _run():
                try:
                    result_box["r"] = si.run_multi_file(cluster_name=cluster_name)
                except Exception as exc:
                    result_box["r"] = {"outcome": f"error: {exc}"}
                ev.set()
            t = threading.Thread(target=_run, daemon=True)
            t.start()
            # Wait up to 180s for result
            ev.wait(timeout=180)
            self._json_response(result_box.get("r", {"outcome": "timeout"}))
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── Evolver Chat endpoints ────────────────────────────────────────────────

    def _ensure_log_buffer(self):
        """Install log buffer singleton on first call (lazy)."""
        from sare.meta.evolver_chat import get_log_buffer
        return get_log_buffer()

    def _api_solve_stream(self, expr: str, budget: float = 10.0):
        """GET /api/solve/stream?expr=...&budget=... — SSE stream of BeamSearch steps."""
        import queue as _queue
        import threading as _threading
        import time as _time
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        step_q: _queue.Queue = _queue.Queue()

        def _on_step(ev: dict):
            step_q.put(ev)

        def _run_search():
            try:
                from sare.engine import load_problem, BeamSearch, EnergyEvaluator
                from sare.brain import get_brain
                _brain = get_brain()
                transforms = list(_brain.transforms) if _brain else []
                if not transforms:
                    from sare.engine import _base_transforms
                    transforms = _base_transforms()
                energy = EnergyEvaluator()
                _prob = load_problem(expr)
                graph = _prob[1] if isinstance(_prob, tuple) else _prob
                searcher = BeamSearch()
                searcher.search(
                    graph, energy, transforms,
                    beam_width=4, max_depth=20,
                    budget_seconds=min(budget, 30.0),
                    on_step=_on_step,
                )
            except Exception as exc:
                step_q.put({"type": "error", "msg": str(exc)})
            finally:
                step_q.put(None)  # sentinel

        t = _threading.Thread(target=_run_search, daemon=True)
        t.start()

        try:
            init_ev = json.dumps({"type": "init", "expr": expr})
            self.wfile.write(f"data: {init_ev}\n\n".encode())
            self.wfile.flush()
            step_count = 0
            while True:
                try:
                    item = step_q.get(timeout=1.0)
                except _queue.Empty:
                    self.wfile.write(b"data: {\"type\":\"heartbeat\"}\n\n")
                    self.wfile.flush()
                    continue
                if item is None:
                    done_ev = json.dumps({"type": "done", "steps": step_count})
                    self.wfile.write(f"data: {done_ev}\n\n".encode())
                    self.wfile.flush()
                    break
                item["type"] = "step"
                step_count += 1
                self.wfile.write(f"data: {json.dumps(item)}\n\n".encode())
                if step_count % 5 == 0:
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _api_evolver_daemon(self):
        """GET /api/evolver/daemon — read evolver_daemon.json state file."""
        import json as _json
        path = REPO_ROOT / "data" / "memory" / "evolver_daemon.json"
        try:
            if path.exists():
                self._json_response(_json.loads(path.read_text()))
            else:
                self._json_response({})
        except Exception as e:
            self._json_response({"error": str(e)})

    def _api_evolve_logs(self):
        """GET /api/evolve/logs?since=<id> — recent evolver log entries."""
        from urllib.parse import parse_qs, urlparse
        qs    = parse_qs(urlparse(self.path).query)
        since = int(qs.get("since", ["0"])[0])
        buf   = self._ensure_log_buffer()
        self._json_response({"logs": buf.get_since(since)})

    def _api_evolve_logs_stream(self):
        """GET /api/evolve/logs/stream — SSE stream of live evolver logs."""
        import time as _time
        import logging as _logging
        buf = self._ensure_log_buffer()
        self.send_response(200)
        self.send_header("Content-Type",  "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection",    "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        # Flush any buffered logs immediately on connect
        last_id = 0
        existing = buf.get_all()
        try:
            for e in existing:
                self.wfile.write(f"id: {e['id']}\ndata: {json.dumps(e)}\n\n".encode())
                last_id = e["id"]
            if existing:
                self.wfile.flush()

            heartbeat_every = 5   # seconds
            last_hb = _time.time()
            while True:
                new = buf.get_since(last_id)
                for e in new:
                    self.wfile.write(f"id: {e['id']}\ndata: {json.dumps(e)}\n\n".encode())
                    last_id = e["id"]
                if new:
                    self.wfile.flush()
                # Heartbeat — inject a visible entry so user sees stream is live
                if _time.time() - last_hb >= heartbeat_every:
                    from sare.meta.self_improver import get_self_improver as _gsi
                    try:
                        st = _gsi().get_status()
                        active = st.get("active_debates", [])
                        if active:
                            hb_msg = "  ".join(f"[{a['file']} → {a['turn']} {a['elapsed_s']}s]" for a in active)
                        else:
                            hb_msg = f"daemon={'running' if st.get('running') else 'stopped'}  applied={st.get('patches_applied',0)}  debates={st.get('total_debates',0)}"
                    except Exception:
                        hb_msg = "alive"
                    hb_entry = {
                        "id":    -1,
                        "ts":    _time.time(),
                        "level": "INFO",
                        "tag":   "dim",
                        "msg":   f"♡ {hb_msg}",
                        "src":   "heartbeat",
                    }
                    self.wfile.write(f"data: {json.dumps(hb_entry)}\n\n".encode())
                    self.wfile.flush()
                    last_hb = _time.time()
                _time.sleep(0.25)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _api_evolve_messages(self):
        """GET /api/evolve/messages?since=<msg_id> — chat message history."""
        from urllib.parse import parse_qs, urlparse
        from sare.meta.evolver_chat import get_evolver_chat
        qs    = parse_qs(urlparse(self.path).query)
        since = qs.get("since", [""])[0]
        chat  = get_evolver_chat()
        state = chat.get_state()
        msgs  = state.get("messages", [])
        if since:
            # Filter to messages after the given id
            ids = [m.get("id", "") for m in msgs]
            if since in ids:
                msgs = msgs[ids.index(since) + 1:]
        self._json_response({"messages": msgs})

    def _api_evolve_messages_stream(self):
        """GET /api/evolve/messages/stream — SSE stream of new chat messages."""
        import time as _time
        from sare.meta.evolver_chat import get_evolver_chat
        chat = get_evolver_chat()
        self.send_response(200)
        self.send_header("Content-Type",  "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection",    "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        last_id = ""
        try:
            while True:
                state = chat.get_state()
                all_msgs = state.get("messages", [])
                if last_id:
                    ids = [m.get("id", "") for m in all_msgs]
                    new = all_msgs[ids.index(last_id) + 1:] if last_id in ids else []
                else:
                    new = all_msgs[-10:]  # send last 10 on first connect
                for m in new:
                    data = json.dumps(m)
                    self.wfile.write(f"id: {m['id']}\ndata: {data}\n\n".encode())
                    last_id = m["id"]
                if new:
                    self.wfile.flush()
                _time.sleep(0.5)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _api_evolve_suggestions(self):
        """GET /api/evolve/suggestions — pending improvement suggestions."""
        from sare.meta.evolver_chat import get_evolver_chat
        chat = get_evolver_chat()
        state = chat.get_state()
        self._json_response({"suggestions": state.get("pending_suggestions", [])})

    def _api_evolve_chat(self, body: dict):
        """POST /api/evolve/chat — send a message to the evolver assistant."""
        text = (body.get("message") or body.get("text") or "").strip()
        if not text:
            self._json_response({"error": "message required"}, 400)
            return
        try:
            from sare.meta.evolver_chat import get_evolver_chat
            result = get_evolver_chat().process_message(text)
            self._json_response(result)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_evolve_interrupt(self):
        """POST /api/evolve/interrupt — interrupt all running debates."""
        try:
            from sare.meta.evolver_chat import get_evolver_chat
            result = get_evolver_chat().process_message("interrupt all debates")
            self._json_response(result)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_evolve_feedback(self, body: dict):
        """POST /api/evolve/feedback — record feedback {"debate_id":"..","rating":1|-1,"comment":".."}."""
        try:
            from sare.meta.evolver_chat import get_evolver_chat
            comment = body.get("comment", "")
            rating  = body.get("rating", 0)
            # Synthesise a feedback message
            mood = "good great excellent" if rating >= 0 else "bad wrong terrible"
            text = f"feedback: {mood} {comment}"
            result = get_evolver_chat().process_message(text)
            self._json_response(result)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_evolve_apply(self, body: dict):
        """POST /api/evolve/apply — apply a specific suggestion by sid."""
        sid = body.get("sid", "")
        try:
            from sare.meta.evolver_chat import get_evolver_chat
            msg = f"yes {sid}" if sid else "yes"
            result = get_evolver_chat().process_message(msg)
            self._json_response(result)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── Neuro / AGI mind module endpoints ────────────────────────────────────

    def _api_neuro_dopamine(self):
        """GET /api/neuro/dopamine — DopamineSystem state + behavior mode."""
        try:
            from sare.neuro.dopamine import get_dopamine_system
            self._json_response(get_dopamine_system().get_state())
        except Exception as e:
            self._json_response({"error": str(e), "tonic_level": 0.5,
                                  "behavior_mode": "learn"})

    def _api_neuro_symbols(self):
        """GET /api/neuro/symbols — invented symbols registry."""
        try:
            from sare.neuro.symbol_creator import get_symbol_creator
            self._json_response(get_symbol_creator().get_status())
        except Exception as e:
            self._json_response({"error": str(e), "total_invented": 0, "promoted": 0})

    def _api_neuro_algorithms(self):
        """GET /api/neuro/algorithms — invented algorithms registry."""
        try:
            from sare.neuro.algorithm_inventor import get_algorithm_inventor
            self._json_response(get_algorithm_inventor().get_status())
        except Exception as e:
            self._json_response({"error": str(e), "total_invented": 0})

    def _api_neuro_creativity(self):
        """GET /api/neuro/creativity — creativity engine dream log."""
        try:
            from sare.neuro.creativity_engine import get_creativity_engine
            self._json_response(get_creativity_engine().get_status())
        except Exception as e:
            self._json_response({"error": str(e), "total_dreams": 0})

    def _api_neuro_htm(self):
        """GET /api/neuro/htm — HTM Predictor stats + top sequences per domain."""
        try:
            from sare.neuro.htm_predictor import get_htm_predictor
            htm = get_htm_predictor()
            stats = htm.get_stats()
            top_by_domain = {}
            for dom in stats.get("domains", []):
                top_by_domain[dom] = htm.top_sequences(dom, n=8)
            self._json_response({**stats, "top_sequences_by_domain": top_by_domain})
        except Exception as e:
            self._json_response({
                "error": str(e),
                "total_sequences": 0,
                "total_bigrams": 0,
                "total_trigrams": 0,
                "domains": [],
                "prediction_accuracy": 0.0,
                "top_sequences_by_domain": {},
            })

    def _api_neuro_dream(self):
        """POST /api/neuro/dream — trigger one creativity cycle (background)."""
        import threading as _th
        try:
            from sare.neuro.creativity_engine import get_creativity_engine
            result_box = {}
            def _run():
                result_box["r"] = get_creativity_engine().dream(force=True)
            t = _th.Thread(target=_run, daemon=True, name="CreativityTrigger")
            t.start()
            t.join(timeout=90)
            self._json_response(result_box.get("r", {"outcome": "timeout"}))
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_neuro_invent_symbol(self, body: dict):
        """POST /api/neuro/invent-symbol — invent new primitive.
        Body: {"stuck_exprs":["sin²(x)+cos²(x)"], "domain":"calculus"}
        """
        import threading as _th
        stuck_exprs = body.get("stuck_exprs", [])
        domain      = body.get("domain", "general")
        if not stuck_exprs:
            self._json_response({"error": "stuck_exprs required"}, 400)
            return
        try:
            from sare.neuro.symbol_creator import get_symbol_creator
            result_box = {}
            def _run():
                result_box["r"] = get_symbol_creator().invent(
                    stuck_exprs=stuck_exprs, domain=domain,
                    existing_transforms=[], force=True
                )
            t = _th.Thread(target=_run, daemon=True, name="SymbolInvent")
            t.start()
            t.join(timeout=90)
            self._json_response(result_box.get("r", {"outcome": "timeout"}))
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_neuro_invent_algorithm(self, body: dict):
        """POST /api/neuro/invent-algorithm — invent new search algorithm.
        Body: {"failure_patterns":["local_minimum"], "domain":"algebra"}
        """
        import threading as _th
        patterns = body.get("failure_patterns", ["general_stuck"])
        domain   = body.get("domain", "general")
        exprs    = body.get("failed_exprs", [])
        try:
            from sare.neuro.algorithm_inventor import get_algorithm_inventor
            result_box = {}
            def _run():
                result_box["r"] = get_algorithm_inventor().invent(
                    failure_patterns=patterns, domain=domain,
                    failed_exprs=exprs, force=True
                )
            t = _th.Thread(target=_run, daemon=True, name="AlgInvent")
            t.start()
            t.join(timeout=120)
            self._json_response(result_box.get("r", {"outcome": "timeout"}))
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── Human-mind module endpoints ───────────────────────────────────────────

    def _api_dialogue_chat(self, body: dict):
        """POST /api/dialogue/chat — process a conversational turn."""
        message = body.get("message", "").strip()
        session_id = body.get("session_id", "default")
        if not message:
            self._json_response({"error": "No message provided"}, 400)
            return
        try:
            from sare.social.dialogue_manager import get_dialogue_manager
            dm = get_dialogue_manager()
            result = dm.process_turn(message, session_id)
            self._json_response(result)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_dialogue_session(self, session_id: str):
        """GET /api/dialogue/session?session_id= — return session history."""
        try:
            from sare.social.dialogue_manager import get_dialogue_manager
            dm = get_dialogue_manager()
            session = dm.get_session(session_id)
            self._json_response(session.to_dict())
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_dialogue_sessions(self):
        """GET /api/dialogue/sessions — return all sessions summary."""
        try:
            from sare.social.dialogue_manager import get_dialogue_manager
            dm = get_dialogue_manager()
            self._json_response({"sessions": dm.all_sessions()})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_identity(self):
        """GET /api/identity — return identity traits and self-description."""
        try:
            from sare.memory.identity import get_identity_manager
            im = get_identity_manager()
            self._json_response({
                "self_description": im.get_self_description(),
                "traits": {name: t.to_dict() for name, t in im.traits.items()},
                "core_values": im.core_values,
                "milestones": [m.to_dict() for m in im.milestones],
                "learning_style": im.get_learning_style(),
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_autobiography(self):
        """GET /api/autobiography — return narrative, trajectory, and emotional state."""
        try:
            from sare.memory.autobiographical import get_autobiographical_memory
            am = get_autobiographical_memory()
            important = [e.to_dict() for e in am.get_most_important_episodes(top_k=10)]
            self._json_response({
                "narrative": am.get_narrative(),
                "emotional_arc": am.get_emotional_arc(),
                "episodes": important,
                "recent_episodes": important,
                "total_episodes": len(am._episodes),
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_homeostasis(self):
        """GET /api/homeostasis — return drive state and behavior recommendation."""
        try:
            from sare.meta.homeostasis import get_homeostatic_system
            hs = get_homeostatic_system()
            self._json_response(hs.get_state())
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_dream_status(self):
        """GET /api/dream/status — last consolidation time and causal discoveries."""
        try:
            from sare.learning.dream_consolidator import DreamConsolidator
            dc = DreamConsolidator()
            self._json_response(dc.summary())
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_agi_score(self):
        """GET /api/agi/score — Compute current AGI capability score across all dimensions."""
        score = {}
        _acc = [0.0, 0.0]   # [total, max_total] — mutable to avoid nonlocal issues

        def _s(key, val, weight=10.0):
            score[key] = {"value": round(float(val), 2), "weight": weight}
            _acc[0] += float(val) * weight
            _acc[1] += weight

        # 1. Symbolic reasoning (0-10): benchmark solve rate
        try:
            from pathlib import Path as _P
            import json as _j
            cache = _P(__file__).resolve().parents[3] / "data/memory/benchmark_cache.json"
            if cache.exists():
                d = _j.loads(cache.read_text())
                rate = d.get("last_symbolic_pass_rate") or d.get("solve_rate", 0)
                _s("symbolic_reasoning", float(rate) * 10, 15)
            else:
                _s("symbolic_reasoning", 5.0, 15)
        except Exception:
            _s("symbolic_reasoning", 0.0, 15)

        # 2. Learning loop (0-10): solve rate from recent runs
        try:
            from pathlib import Path as _P
            import json as _j
            prog = _P(__file__).resolve().parents[3] / "data/memory/progress.json"
            if prog.exists():
                d = _j.loads(prog.read_text())
                runs = d.get("runs", [])
                if runs:
                    recent = runs[-5:]
                    avg_solve = sum(r.get("solve_rate", 0) for r in recent) / len(recent)
                    _s("learning_loop", avg_solve * 10, 10)
                else:
                    _s("learning_loop", 0.0, 10)
            else:
                _s("learning_loop", 0.0, 10)
        except Exception:
            _s("learning_loop", 0.0, 10)

        # 3. Memory & knowledge (0-10): rules promoted
        try:
            from pathlib import Path as _P
            import json as _j
            pr = _P(__file__).resolve().parents[3] / "data/memory/promoted_rules.json"
            if pr.exists():
                d = _j.loads(pr.read_text())
                rules = d.get("promoted_rules", [])
                n = len(rules) if isinstance(rules, list) else len(rules)
                _s("memory_knowledge", min(10, n * 1.25), 10)
            else:
                _s("memory_knowledge", 0.0, 10)
        except Exception:
            _s("memory_knowledge", 0.0, 10)

        # 4. Self-model accuracy (0-10): domains tracked
        try:
            from pathlib import Path as _P
            import json as _j
            sm_path = _P(__file__).resolve().parents[3] / "data/memory/self_model.json"
            if sm_path.exists():
                d = _j.loads(sm_path.read_text())
                domains = len(d.get("domains", {}))
                attempts = d.get("total_attempts", 0)
                _s("self_awareness", min(10, domains * 1.5 + min(2, attempts / 50)), 10)
            else:
                _s("self_awareness", 0.0, 10)
        except Exception:
            _s("self_awareness", 0.0, 10)

        # 5. Creativity (0-10): symbol + dream inventory
        try:
            from sare.neuro.symbol_creator import get_symbol_creator
            from sare.neuro.creativity_engine import get_creativity_engine
            sc = get_symbol_creator(); ce = get_creativity_engine()
            n = sc.get_status()["total_invented"] + ce.get_status()["total_dreams"]
            _s("creativity", min(10, n * 2.0), 8)
        except Exception:
            _s("creativity", 0.0, 8)

        # 6. Self-improvement (0-10): patches applied
        try:
            from sare.meta.self_improver import get_self_improver
            si = get_self_improver()
            st = si.get_status()
            applied = st["patches_applied"]
            _s("self_improvement", min(10, applied * 2.5), 10)
        except Exception:
            _s("self_improvement", 0.0, 10)

        # 7. Theory of Mind (0-10): agent models + false-belief benchmark
        try:
            from pathlib import Path as _P
            import json as _j
            tom_path = _P(__file__).resolve().parents[3] / "data/memory/theory_of_mind.json"
            if tom_path.exists():
                d = _j.loads(tom_path.read_text())
                # File format: top-level dict of agent_id → mental_state
                # (not {"agents": {...}})
                agents = len([k for k in d if not k.startswith("_")])
                total_beliefs = sum(
                    len(v.get("beliefs", [])) + len(v.get("desires", []))
                    for v in d.values() if isinstance(v, dict)
                )
                _s("theory_of_mind", min(10, agents * 2.5 + total_beliefs * 0.2), 8)
            else:
                _s("theory_of_mind", 0.0, 8)
        except Exception:
            _s("theory_of_mind", 0.0, 8)

        # 8. Planning (0-10): plan traces + success
        try:
            from pathlib import Path as _P
            import json as _j
            traces = _P(__file__).resolve().parents[3] / "data/memory/plan_traces.jsonl"
            if traces.exists():
                lines = traces.read_text().strip().splitlines()
                total = len(lines)
                if total:
                    solved = sum(1 for l in lines if '"solved": true' in l or '"success": true' in l)
                    _s("planning", min(10, (solved / total) * 10 + min(2, total * 0.1)), 8)
                else:
                    _s("planning", 0.0, 8)
            else:
                _s("planning", 0.0, 8)
        except Exception:
            _s("planning", 0.0, 8)

        # 9. Homeostasis / drive (0-10): tonic level
        try:
            from sare.neuro.dopamine import get_dopamine_system
            ds = get_dopamine_system()
            _s("motivation", ds.tonic_level * 10, 7)
        except Exception:
            _s("motivation", 0.0, 7)

        # 10. Language understanding (0-10): world model hypotheses + dialogue
        try:
            from pathlib import Path as _P
            import json as _j
            hyp_path = _P(__file__).resolve().parents[3] / "data/memory/world_hypotheses.json"
            dial_path = _P(__file__).resolve().parents[3] / "data/memory/dialogue_sessions.json"
            n = 0
            if hyp_path.exists():
                hyp = _j.loads(hyp_path.read_text())
                n += len(hyp) if isinstance(hyp, list) else 0
            if dial_path.exists():
                dial = _j.loads(dial_path.read_text())
                sessions = len(dial) if isinstance(dial, list) else len(dial.get("sessions", []))
                n += sessions
            _s("language_grounding", min(10, n * 0.5), 7)
        except Exception:
            _s("language_grounding", 0.0, 7)

        # 11. Embodiment (0-10): grid-world episodes + concepts discovered
        try:
            from sare.world.embodied_agent import EmbodiedAgent
            _ea = EmbodiedAgent()
            _ea_sum = _ea.summary() if hasattr(_ea, "summary") else {}
            _episodes = _ea_sum.get("episodes_run", 0)
            _concepts = len(_ea_sum.get("concepts_learned", []))
            _s("embodiment", min(10, _episodes * 0.1 + _concepts * 0.5), 8)
        except Exception:
            _s("embodiment", 0.0, 8)

        # 12. World simulation (0-10): CausalRollout horizon accuracy
        try:
            from sare.world.causal_rollout import CausalRollout
            _cr = CausalRollout()
            _cr_sum = _cr.summary() if hasattr(_cr, "summary") else {}
            _pred_acc = _cr_sum.get("avg_accuracy", 0.0)
            _observations = _cr_sum.get("total_sequences", 0)
            _s("world_simulation", min(10, _pred_acc * 10 + min(2, _observations * 0.01)), 8)
        except Exception:
            _s("world_simulation", 0.0, 8)

        # 13. Causal reasoning (0-10): CounterfactualReasoner analyses
        try:
            from sare.causal.counterfactual_reasoner import CounterfactualReasoner
            _cfr = CounterfactualReasoner()
            _cfr_sum = _cfr.summary() if hasattr(_cfr, "summary") else {}
            _analyses = _cfr_sum.get("total_analyses", 0)
            _insights = _cfr_sum.get("top_transforms_count", 0)
            _s("causal_reasoning", min(10, _analyses * 0.2 + _insights * 0.5), 8)
        except Exception:
            _s("causal_reasoning", 0.0, 8)

        final = (_acc[0] / (_acc[1] * 10) * 100) if _acc[1] > 0 else 0.0
        self._json_response({
            "agi_score": round(final, 1),
            "max_score": 100,
            "dimensions": score,
            "grade": "A" if final >= 85 else "B" if final >= 70 else "C" if final >= 55 else "D",
            "timestamp": __import__("time").time(),
        })

    def _api_benchmark_symbolic(self):
        """GET /api/benchmark/symbolic — Run symbolic math benchmark suite."""
        import json as _json
        from pathlib import Path as _Path
        from sare.engine import get_transforms

        bench_path = _Path(__file__).resolve().parents[3] / "benchmarks/algebra/symbolic_math.json"
        cache_path = _Path(__file__).resolve().parents[3] / "data/memory/benchmark_cache.json"
        hard_path  = _Path(__file__).resolve().parents[3] / "data/hard_problems.json"

        try:
            problems = _json.loads(bench_path.read_text())
        except Exception as e:
            self._json_response({"error": f"Cannot load benchmark: {e}"}, 500)
            return

        energy_fn  = EnergyEvaluator()
        searcher   = BeamSearch()
        transforms = get_transforms()

        total  = 0
        passed = 0
        failed = 0
        by_cat: dict = {}
        failures = []

        for prob in problems:
            if not prob.get("solvable", True):
                continue  # skip unsolvable entries

            total += 1
            expr = prob["expression"]
            cat  = prob.get("category", "unknown")
            by_cat.setdefault(cat, {"total": 0, "passed": 0})
            by_cat[cat]["total"] += 1

            try:
                _, problem_graph = load_problem(expr)

                e_before = energy_fn.compute(problem_graph).total
                result   = searcher.search(
                    problem_graph,
                    energy_fn,
                    transforms,
                    beam_width=8,
                    budget_seconds=3.0,
                )
                e_after = result.energy.total
                delta   = e_before - e_after

                # Find root node (not pointed to by any edge)
                target_ids = {e.target for e in result.graph.edges}
                root_nodes  = [n for n in result.graph.nodes if n.id not in target_ids]
                got = root_nodes[0].label if root_nodes else ""

                ok = (
                    delta > 0.5
                    or len(result.graph.nodes) < len(problem_graph.nodes)
                    or (got and got == prob.get("expected_result", ""))
                )

                if ok:
                    passed += 1
                    by_cat[cat]["passed"] += 1
                else:
                    failed += 1
                    failures.append({
                        "id": prob["id"],
                        "expression": expr,
                        "expected": prob.get("expected_result"),
                        "got": got,
                        "delta": round(delta, 4),
                    })
            except Exception as exc:
                failed += 1
                failures.append({"id": prob["id"], "expression": expr, "error": str(exc)})

        # Append new failures as hard problems (feedback loop)
        if failures:
            try:
                existing = []
                if hard_path.exists():
                    existing = _json.loads(hard_path.read_text())
                existing_exprs = {p.get("expression") for p in existing}
                next_id = len(existing) + 1
                for f in failures:
                    if f.get("expression") and f["expression"] not in existing_exprs:
                        existing.append({
                            "id": f"hp_{next_id:03d}",
                            "expression": f["expression"],
                            "domain": "algebra",
                            "difficulty": 4,
                        })
                        next_id += 1
                hard_path.parent.mkdir(parents=True, exist_ok=True)
                hard_path.write_text(_json.dumps(existing, indent=2))
            except Exception as _e:
                                log.debug("[web] Suppressed: %s", _e)

        result_data = {
            "total":       total,
            "passed":      passed,
            "failed":      failed,
            "pass_rate":   round(passed / total, 4) if total else 0.0,
            "by_category": by_cat,
            "failures":    failures[:20],
        }

        # Cache result
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(_json.dumps(result_data, indent=2))
        except Exception as _e:
                        log.debug("[web] Suppressed: %s", _e)

        self._json_response(result_data)

    def _api_benchmark_generic(self, bench_path_rel: str) -> dict:
        """Shared logic: load a benchmark JSON, run each case, return result dict."""
        import json as _json
        from pathlib import Path as _Path

        bench_path = _Path(__file__).resolve().parents[3] / bench_path_rel
        try:
            suite = _json.loads(bench_path.read_text())
        except Exception as e:
            return {"error": f"Cannot load benchmark {bench_path_rel}: {e}"}

        from sare.engine import get_transforms
        energy_fn  = EnergyEvaluator()
        searcher   = BeamSearch()
        transforms = get_transforms()

        total = 0; passed = 0; failed = 0
        by_cat: dict = {}
        failures = []

        for prob in suite.get("cases", []):
            total += 1
            expr = prob["expression"]
            cat  = prob.get("category", suite.get("suite", "unknown"))
            by_cat.setdefault(cat, {"total": 0, "passed": 0})
            by_cat[cat]["total"] += 1
            min_pct = prob.get("min_reduction_pct", 10.0)

            try:
                _, problem_graph = load_problem(expr)
                e_before = energy_fn.compute(problem_graph).total
                result = searcher.search(
                    problem_graph, energy_fn, transforms,
                    beam_width=8, budget_seconds=3.0,
                )
                e_after = result.energy.total
                delta = e_before - e_after
                pct = (delta / e_before * 100.0) if e_before > 0 else 0.0

                target_ids = {e.target for e in result.graph.edges}
                root_nodes = [n for n in result.graph.nodes if n.id not in target_ids]
                got = root_nodes[0].label if root_nodes else ""

                ok = (
                    pct >= min_pct
                    or delta > 0.5
                    or len(result.graph.nodes) < len(problem_graph.nodes)
                    or (got and got == prob.get("expected_result", ""))
                )
                if ok:
                    passed += 1
                    by_cat[cat]["passed"] += 1
                else:
                    failed += 1
                    failures.append({
                        "id": prob["id"], "expression": expr,
                        "expected": prob.get("expected_result"),
                        "got": got, "delta": round(delta, 4), "pct": round(pct, 2),
                    })
            except Exception as exc:
                failed += 1
                failures.append({"id": prob["id"], "expression": expr, "error": str(exc)})

        return {
            "suite": suite.get("suite", bench_path_rel),
            "total": total, "passed": passed, "failed": failed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "by_category": by_cat, "failures": failures[:20],
        }

    def _api_benchmark_logic(self):
        """GET /api/benchmark/logic — Run logic simplification benchmark suite."""
        self._json_response(self._api_benchmark_generic("benchmarks/logic/smoke.json"))

    def _api_benchmark_coding(self):
        """GET /api/benchmark/coding — Run code simplification benchmark suite."""
        self._json_response(self._api_benchmark_generic("benchmarks/coding/simplify.json"))

    def _api_benchmark_arc(self):
        """GET /api/benchmark/arc — Run ARC-Lite abstract reasoning benchmark."""
        try:
            from sare.benchmarks.arc_runner import ARCRunner
            runner = ARCRunner()
            result = runner.run()
            self._json_response(result)
        except Exception as e:
            self._json_response({"error": str(e), "total": 0, "correct": 0, "accuracy": 0.0})

    def _api_benchmark_agi(self):
        """GET /api/benchmark/agi — unified 10-category AGI benchmark."""
        try:
            from sare.benchmarks.agi_suite import AGISuite
            suite = AGISuite()
            self._json_response(suite.run())
        except Exception as e:
            self._json_response({
                "error": str(e),
                "total_score": 0.0,
                "categories": [],
                "benchmark_version": "2.0",
            }, 500)

    def _api_learning_trend(self):
        """GET /api/learning/trend — historical benchmark scores + progress metrics for trend chart."""
        import json as _json
        from pathlib import Path as _Path
        _data_dir = _Path(__file__).resolve().parents[3] / "data" / "memory"

        # Benchmark history (timestamped runs)
        benchmark_history = []
        hist_path = _data_dir / "benchmark_history.json"
        if hist_path.exists():
            try:
                raw = _json.loads(hist_path.read_text())
                if isinstance(raw, list):
                    benchmark_history = [
                        {
                            "timestamp": e.get("timestamp", ""),
                            "total_score": e.get("total_score") or e.get("pass_rate", 0),
                            "total_problems": e.get("total_problems") or e.get("total", 0),
                            "by_category": e.get("by_category", {}),
                            "version": e.get("version", "1.0"),
                        }
                        for e in raw
                    ]
            except Exception:
                pass

        # Progress cycles (solve rate / throughput per cycle)
        progress_cycles = []
        prog_path = _data_dir / "progress.json"
        if prog_path.exists():
            try:
                raw = _json.loads(prog_path.read_text())
                if isinstance(raw, list):
                    progress_cycles = [
                        {
                            "cycle": e.get("cycle", i),
                            "solve_rate": e.get("solve_rate", 0),
                            "avg_energy": e.get("avg_energy", 0),
                            "bridge_rate": e.get("bridge_rate", 0),
                            "throughput": e.get("throughput", 0),
                        }
                        for i, e in enumerate(raw)
                    ]
            except Exception:
                pass

        # Transform stats (top transforms by utility)
        top_transforms = []
        ts_path = _data_dir / "transform_stats.json"
        if ts_path.exists():
            try:
                raw = _json.loads(ts_path.read_text())
                if isinstance(raw, dict):
                    entries = [(k, v) for k, v in raw.items() if isinstance(v, (int, float))]
                    entries.sort(key=lambda x: x[1], reverse=True)
                    top_transforms = [{"name": k, "utility": round(v, 3)} for k, v in entries[:15]]
            except Exception:
                pass

        # Rule promotion history (rules promoted over time)
        promoted_rules = []
        pr_path = _data_dir / "promoted_rules.json"
        if pr_path.exists():
            try:
                raw = _json.loads(pr_path.read_text())
                if isinstance(raw, list):
                    promoted_rules = raw[-20:]
                elif isinstance(raw, dict):
                    promoted_rules = list(raw.keys())[-20:]
            except Exception:
                pass

        # Self-improvement stats
        si_stats = {}
        si_path = _data_dir / "si_stats.json"
        if si_path.exists():
            try:
                si_stats = _json.loads(si_path.read_text())
            except Exception:
                pass

        # Synthesized transforms count
        synth_count = 0
        synth_dir = _Path(__file__).resolve().parents[3] / "data" / "memory" / "synthesized_modules"
        if synth_dir.exists():
            synth_count = len(list(synth_dir.glob("*.py")))

        # Is learning? Compute if recent scores are rising
        is_learning = False
        score_delta = None
        if len(benchmark_history) >= 2:
            recent   = [e["total_score"] for e in benchmark_history[-5:] if e.get("total_score")]
            if len(recent) >= 2:
                score_delta  = round(recent[-1] - recent[0], 4)
                is_learning  = score_delta > 0

        self._json_response({
            "benchmark_history":  benchmark_history[-50:],
            "progress_cycles":    progress_cycles[-50:],
            "top_transforms":     top_transforms,
            "promoted_rules":     promoted_rules,
            "si_stats":           si_stats,
            "synthesized_count":  synth_count,
            "is_learning":        is_learning,
            "score_delta":        score_delta,
        })

    def _api_benchmark_all(self):
        """GET /api/benchmark/all — Run all benchmark suites and combine results."""
        import json as _json
        from pathlib import Path as _Path

        # Algebra (existing detailed runner)
        cache_path = _Path(__file__).resolve().parents[3] / "data/memory/benchmark_cache.json"
        algebra = None
        try:
            if cache_path.exists():
                cached = _json.loads(cache_path.read_text())
                # Use cache if recent (< 30 min old)
                import time as _time
                mtime = cache_path.stat().st_mtime
                if _time.time() - mtime < 1800:
                    algebra = cached
        except Exception:
            pass
        if algebra is None:
            # Re-run symbolic benchmark inline (reuse logic without HTTP round-trip)
            algebra = self._api_benchmark_generic("benchmarks/algebra/symbolic_math.json")

        logic  = self._api_benchmark_generic("benchmarks/logic/smoke.json")
        coding = self._api_benchmark_generic("benchmarks/coding/simplify.json")

        domains = [
            {"domain": "algebra",  **{k: v for k, v in algebra.items()  if k != "failures"}},
            {"domain": "logic",    **{k: v for k, v in logic.items()    if k != "failures"}},
            {"domain": "coding",   **{k: v for k, v in coding.items()   if k != "failures"}},
        ]
        total_all = sum(d["total"] for d in domains)
        passed_all = sum(d["passed"] for d in domains)

        results: dict = {
            "total_pass_rate": round(passed_all / total_all, 4) if total_all else 0.0,
            "total": total_all, "passed": passed_all,
            "domains": domains,
        }

        try:
            from sare.benchmarks.agi_suite import AGISuite
            _agi = AGISuite()
            results["agi_suite"] = _agi.run()
        except Exception as _e:
            results["agi_suite"] = {"error": str(_e)}

        self._json_response(results)

    def _api_search_predictor(self):
        """GET /api/search/predictor — Return TransformPredictor statistics."""
        try:
            from sare.search.transform_predictor import get_transform_predictor
            stats = get_transform_predictor().get_stats()
            self._json_response(stats)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── Brain Orchestrator API Handlers ──────────────────────────────────────

    def _api_brain_status(self):
        """GET /api/brain/status — Full brain status report."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            self._json_response(brain.status())
        except Exception as e:
            self._json_response({"error": str(e), "brain_available": False}, 500)

    def _api_brain_curriculum(self):
        """GET /api/brain/curriculum — Developmental curriculum map."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain.developmental_curriculum:
                self._json_response(brain.developmental_curriculum.get_curriculum_map())
            else:
                self._json_response({"error": "Curriculum not loaded", "domains": {}})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_events(self):
        """GET /api/brain/events — Recent brain events."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            events = brain.events.recent(50)
            self._json_response({
                "events": [
                    {"event": e.event.value, "timestamp": e.timestamp,
                     "source": e.source, "data": {k: str(v)[:200] for k, v in e.data.items()}}
                    for e in events
                ]
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_solve(self, body: dict):
        """POST /api/brain/solve — Alias for /api/solve (same backend)."""
        self._api_solve(body)

    def _api_brain_learn(self, body: dict):
        """POST /api/brain/learn — Run N autonomous learning cycles."""
        n = body.get("cycles", 5)
        try:
            from sare.brain import get_brain
            brain = get_brain()
            results = brain.learn_cycle(n=n)
            successes = sum(1 for r in results if r.get("success"))

            # Update curriculum with results
            if brain.developmental_curriculum:
                for r in results:
                    try:
                        brain.developmental_curriculum.record_attempt(
                            r.get("expression", ""), r.get("domain", "general"),
                            r.get("success", False), r.get("delta", 0))
                    except Exception as _e:
                                                log.debug("[web] Suppressed: %s", _e)

            # Run transfer discovery after learning
            transfer_hyps = []
            if brain.transfer_engine:
                try:
                    hyps = brain.transfer_engine.generate_hypotheses()
                    transfer_hyps = [h.to_dict() for h in hyps[:5]]
                except Exception as _e:
                                        log.debug("[web] Suppressed: %s", _e)

            # Run analogy discovery in world model
            analogies_found = 0
            if brain.world_model_v3:
                try:
                    new_analogies = brain.world_model_v3.discover_analogies()
                    analogies_found = len(new_analogies)
                except Exception as _e:
                                        log.debug("[web] Suppressed: %s", _e)

            self._json_response({
                "cycles": len(results),
                "successes": successes,
                "solve_rate": round(successes / max(len(results), 1), 3),
                "stage": brain.stage.value,
                "transfer_hypotheses": transfer_hyps,
                "analogies_found": analogies_found,
                "results": [
                    {
                        "expression": r.get("expression", ""),
                        "success": r.get("success", False),
                        "delta": r.get("delta", 0),
                        "transforms": r.get("transforms", [])[:5],
                        "domain": r.get("domain", ""),
                    }
                    for r in results
                ],
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_autolearn_status(self):
        """GET /api/brain/autolearn — Auto-learn daemon status."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            data = brain.auto_learn_status()
            if brain.developmental_curriculum:
                cmap = brain.developmental_curriculum.get_curriculum_map()
                data["mastered"] = cmap.get("mastered", 0)
                data["unlocked"] = cmap.get("unlocked", 0)
                data["total_domains"] = cmap.get("total_domains", 0)
            data["stage"] = brain.stage.value
            self._json_response(data)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_autolearn_control(self, body: dict):
        """POST /api/brain/autolearn — Start/stop auto-learn."""
        action = body.get("action", "status")
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if action == "start":
                interval = body.get("interval", 5.0)
                problems = body.get("problems", 3)
                brain.start_auto_learn(interval=interval, problems_per_cycle=problems)
                self._json_response({"status": "started", "interval": interval, "problems": problems})
            elif action == "stop":
                brain.stop_auto_learn()
                self._json_response({"status": "stopped"})
            else:
                self._json_response(brain.auto_learn_status())
        except Exception as e:
            self._json_response({"error": str(e)}, 500)


    def _api_daemon_status(self):
        """GET /api/daemon/status - Check if learn_daemon is running and in which mode."""
        import subprocess
        try:
            result = subprocess.run(
                ["pgrep", "-f", "learn_daemon.py"],
                capture_output=True,
                text=True
            )
            running = result.returncode == 0
            turbo = False
            pid = None
            if running:
                try:
                    result = subprocess.run(
                        ["pgrep", "-f", "learn_daemon.py.*--turbo"],
                        capture_output=True,
                        text=True
                    )
                    turbo = result.returncode == 0
                except:
                    pass
                try:
                    result = subprocess.run(
                        ["pgrep", "-f", "learn_daemon.py"],
                        capture_output=True,
                        text=True
                    )
                    pid = result.stdout.strip().split('\n')[0] if result.stdout else None
                except:
                    pass
            self._json_response({
                "running": running,
                "turbo": turbo,
                "pid": pid,
                "mode": "turbo" if turbo else "normal"
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_daemon_control(self, body: dict):
        """POST /api/daemon/control - Start/stop/restart the learn_daemon."""
        import subprocess
        import time
        action = body.get("action", "status")
        mode = body.get("mode", "normal")
        try:
            if action == "start":
                result = subprocess.run(
                    ["pgrep", "-f", "learn_daemon.py"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    self._json_response({"error": "Daemon already running"}, 400)
                    return
                cmd = ["/opt/homebrew/Cellar/python@3.13/3.13.5/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python", "/Users/akshitpatel/sare/learn_daemon.py"]
                if mode == "turbo":
                    cmd.append("--turbo")
                subprocess.Popen(
                    cmd,
                    stdout=open("/Users/akshitpatel/sare/daemon.log", "a"),
                    stderr=subprocess.STDOUT,
                    start_new_session=True
                )
                self._json_response({"status": "started", "mode": mode})
            elif action == "stop":
                subprocess.run(["pkill", "-f", "learn_daemon.py"], capture_output=True)
                self._json_response({"status": "stopped"})
            elif action == "restart":
                subprocess.run(["pkill", "-f", "learn_daemon.py"], capture_output=True)
                time.sleep(1)
                cmd = ["/opt/homebrew/Cellar/python@3.13/3.13.5/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python", "/Users/akshitpatel/sare/learn_daemon.py"]
                if mode == "turbo":
                    cmd.append("--turbo")
                subprocess.Popen(
                    cmd,
                    stdout=open("/Users/akshitpatel/sare/daemon.log", "a"),
                    stderr=subprocess.STDOUT,
                    start_new_session=True
                )
                self._json_response({"status": "restarted", "mode": mode})
            elif action == "status":
                self._api_daemon_status()
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_daemon_activity(self):
        """GET /api/daemon/activity - Get recent daemon activity."""
        import os
        try:
            log_path = "/Users/akshitpatel/sare/daemon.log"
            if not os.path.exists(log_path):
                self._json_response({"activity": [], "log_file": log_path})
                return
            with open(log_path, "r") as f:
                lines = f.readlines()
            activity = []
            for line in lines[-100:]:
                line = line.strip()
                if any(k in line for k in ["Cycle", "Solved", "ERROR", "WARNING", "domain", "batch"]):
                    activity.append(line)
            self._json_response({
                "activity": activity[-30:] if len(activity) > 30 else activity,
                "total_lines": len(lines),
                "log_file": log_path
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_daemon_heartbeat(self):
        """GET /api/daemon/heartbeat — Latest heartbeat written by the daemon each cycle."""
        import time as _time
        import json as _json
        from pathlib import Path as _Path
        hb_path = _Path(__file__).parent.parent.parent.parent / "data" / "memory" / "daemon_heartbeat.json"
        try:
            if not hb_path.exists():
                self._json_response({"alive": False, "reason": "no heartbeat file yet"})
                return
            hb = _json.loads(hb_path.read_text())
            age = _time.time() - hb.get("ts", 0)
            # >15 min = stuck; >60 s = slow; else ok
            if age > 900:
                status = "stuck"
            elif age > 60:
                status = "slow"
            else:
                status = "active"
            self._json_response({
                "alive": age < 900,
                "status": status,
                "age_s": round(age, 1),
                **hb,
            })
        except Exception as e:
            self._json_response({"alive": False, "status": "error", "error": str(e)})

    def _api_daemon_livelog(self, lines: int = 60):
        """GET /api/daemon/livelog?lines=60 — Last N lines from the daemon log file."""
        import os as _os
        _DAEMON_LOG = "/tmp/sare_daemon.log"
        try:
            if not _os.path.exists(_DAEMON_LOG):
                self._json_response({"lines": [], "log_file": _DAEMON_LOG, "exists": False})
                return
            # Efficient tail: read last ~4KB and split lines
            with open(_DAEMON_LOG, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                read_bytes = min(size, lines * 200)  # estimate ~200 bytes/line
                f.seek(max(0, size - read_bytes))
                raw = f.read().decode("utf-8", errors="replace")
            all_lines = raw.splitlines()
            tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
            self._json_response({
                "lines": tail,
                "total_read": len(all_lines),
                "log_file": _DAEMON_LOG,
                "exists": True,
            })
        except Exception as e:
            self._json_response({"error": str(e), "lines": []}, 500)

    def _api_brain_conjectures(self):
        """GET /api/brain/conjectures — Generate conjectures from learned knowledge."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            conjectures = brain.generate_conjectures(n=8)
            self._json_response({"conjectures": conjectures})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_instruct(self, body: dict):
        """POST /api/brain/instruct — Solve using natural language instruction."""
        instruction = body.get("instruction", "")
        expression = body.get("expression", "")
        try:
            from sare.brain import get_brain
            brain = get_brain()

            if not brain.language_grounding:
                self._json_response({"error": "Language grounding not available"}, 500)
                return

            # Parse the instruction
            result = brain.language_grounding.parse_instruction(instruction)
            if not result:
                # Try to suggest transforms for the goal
                suggestions = brain.language_grounding.suggest_transforms_for_goal(instruction)
                self._json_response({
                    "instruction": instruction,
                    "transform": None,
                    "confidence": 0.0,
                    "suggestions": suggestions,
                    "message": "Could not parse instruction. Here are some suggestions:"
                })
                return

            # Apply the suggested transform to the expression
            transform_name = result["transform"]
            if not expression:
                self._json_response({
                    "instruction": instruction,
                    "transform": transform_name,
                    "confidence": result["confidence"],
                    "message": "Transform identified, but no expression provided to apply it to"
                })
                return

            # Solve the expression
            solve_result = brain.solve(expression)

            # Check if the suggested transform was used
            transforms_used = solve_result.get("transforms", [])
            transform_used = transform_name in transforms_used

            self._json_response({
                "instruction": instruction,
                "expression": expression,
                "suggested_transform": transform_name,
                "confidence": result["confidence"],
                "transform_used": transform_used,
                "solve_result": solve_result
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_world(self):
        """GET /api/brain/world — WorldModel v3 status."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain.world_model_v3:
                self._json_response(brain.world_model_v3.summary())
            else:
                self._json_response({"error": "WorldModel v3 not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_transfer(self):
        """GET /api/brain/transfer — Transfer Engine status + cross-domain map."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain.transfer_engine:
                data = brain.transfer_engine.get_cross_domain_map()
                data["summary"] = brain.transfer_engine.summary()
                # Add synthesized transforms
                if brain.transform_synthesizer:
                    data["synthesized_transforms"] = {
                        "total": len(brain.transform_synthesizer._synthesized),
                        "promoted": len(brain.transform_synthesizer.get_promoted()),
                        "transforms": [
                            {
                                "name": name,
                                "domain": spec.domain,
                                "role": spec.role,
                                "confidence": spec.confidence,
                                "promoted": spec.promoted
                            }
                            for name, spec in brain.transform_synthesizer._synthesized.items()
                        ]
                    }
                self._json_response(data)
            else:
                self._json_response({"error": "Transfer engine not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_concepts(self):
        """GET /api/brain/concepts — Concept graph status + all concepts."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.concept_graph:
                cg = brain.concept_graph
                summary = cg.summary()
                concepts = []
                for name, c in cg._concepts.items():
                    concepts.append({
                        "name": c.name,
                        "meaning": c.meaning,
                        "symbol": c.symbol,
                        "domain": c.domain,
                        "symbolic_rules": c.symbolic_rules[:3],
                        "examples_count": c.ground_count(),
                        "well_grounded": c.is_well_grounded(),
                        "confidence": round(c.confidence, 2),
                        "related": c.related[:3],
                    })
                self._json_response({"summary": summary, "concepts": concepts})
            else:
                self._json_response({"summary": {}, "concepts": [],
                                     "error": "Concept graph not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_goals_status(self):
        """GET /api/brain/goals — Goal planner status."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.goal_planner:
                self._json_response({"summary": brain.goal_planner.summary()})
            else:
                self._json_response({"summary": {"total_plans": 0, "active_plans": 0,
                                                  "total_nodes": 0, "plans": []}})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_goals_control(self, body: dict):
        """POST /api/brain/goals — Create goal plans."""
        action = body.get("action", "status")
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if not brain or not brain.goal_planner:
                self._json_response({"error": "Goal planner not available"})
                return
            gp = brain.goal_planner
            if action == "plan_concept":
                concept = body.get("concept", "identity_addition")
                domain = body.get("domain", "arithmetic")
                plan = gp.plan_learn_concept(concept, domain)
                self._json_response({"plan_id": plan.goal_id,
                                     "plan": plan.to_dict(),
                                     "summary": gp.summary()})
            elif action == "plan_domain":
                domain = body.get("domain", "arithmetic")
                plan = gp.plan_master_domain(domain)
                self._json_response({"plan_id": plan.goal_id,
                                     "plan": plan.to_dict(),
                                     "summary": gp.summary()})
            else:
                self._json_response({"summary": gp.summary()})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_environment_status(self):
        """GET /api/brain/environment — Environment simulator summary."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.environment_simulator:
                env = brain.environment_simulator
                summary = env.summary()
                self._json_response(summary)
            else:
                self._json_response({"total_observations": 0, "concepts_discovered": 0})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_environment_run(self, body: dict):
        """POST /api/brain/environment — Run environment experiments."""
        exp_type = body.get("type", "identity")
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if not brain or not brain.environment_simulator:
                self._json_response({"error": "Environment simulator not available"})
                return
            env = brain.environment_simulator
            if exp_type == "identity":
                obs = env.run_identity_discovery(6)
            elif exp_type == "annihilation":
                obs = env.run_annihilation_discovery(5)
            elif exp_type == "commutativity":
                obs = env.run_commutativity_discovery(4)
            else:
                obs = env.run_full_discovery_session()
            # Feed new observations into concept graph
            if brain.concept_graph:
                for o in obs:
                    brain.concept_graph.ground_example(
                        concept_name=o.concept_hint,
                        text=o.description,
                        operation=o.operation,
                        symbolic=o.symbolic,
                        domain="arithmetic",
                    )
            summary = env.summary()
            self._json_response({
                "observations": [o.to_dict() for o in obs[:8]],
                "total": summary["total_observations"],
                "concepts_discovered": summary["concepts_discovered"],
                "symbolic_rules": summary["symbolic_rules"][:4],
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── S25-1: Global Buffer ───────────────────────────────────────────────────
    def _api_brain_buffer(self):
        """GET /api/brain/buffer — GlobalBuffer working memory summary."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.global_buffer:
                self._json_response(brain.global_buffer.summary())
            else:
                self._json_response({"active_items": 0, "capacity": 7,
                                     "total_received": 0, "attention_focus": None,
                                     "error": "GlobalBuffer not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── S25-2: Concept Blender ─────────────────────────────────────────────────
    def _api_brain_blender(self):
        """GET /api/brain/blender — ConceptBlender summary."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.concept_blender:
                self._json_response(brain.concept_blender.summary())
            else:
                self._json_response({"total_blends": 0, "accepted_blends": 0,
                                     "error": "ConceptBlender not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_blender_action(self, body: dict):
        """POST /api/brain/blender — discover blends or blend a specific pair."""
        action = body.get("action", "discover")
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if not brain or not brain.concept_blender:
                self._json_response({"error": "ConceptBlender not available"})
                return
            cb = brain.concept_blender
            if action == "discover":
                blends = cb.discover_blends(max_new=5)
                if brain.concept_graph:
                    cb.feed_to_concept_graph(brain.concept_graph)
                self._json_response({
                    "new_blends": len(blends),
                    "blends": [b.to_dict() for b in blends],
                    "summary": cb.summary(),
                })
            elif action == "blend_pair":
                a = body.get("concept_a", "addition")
                b_c = body.get("concept_b", "conjunction")
                blend = cb.blend_pair(a, b_c)
                self._json_response(blend.to_dict() if blend
                                    else {"error": "concepts not found"})
            else:
                self._json_response(cb.summary())
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── S25-3: Dialogue Context ────────────────────────────────────────────────
    def _api_brain_dialogue(self):
        """GET /api/brain/dialogue — DialogueContext summary."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.dialogue_context:
                self._json_response(brain.dialogue_context.summary())
            else:
                self._json_response({"total_turns": 0, "active_turns": 0,
                                     "error": "DialogueContext not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_dialogue_turn(self, body: dict):
        """POST /api/brain/dialogue — add turn and get pronoun resolution."""
        speaker = body.get("speaker", "user")
        text    = body.get("text", "")
        domain  = body.get("domain")
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if not brain or not brain.dialogue_context:
                self._json_response({"error": "DialogueContext not available"})
                return
            dc       = brain.dialogue_context
            turn     = dc.add_turn(speaker, text, domain)
            resolved = dc.resolve(text)
            context  = dc.get_context_for_parse()
            self._json_response({
                "turn":     turn.to_dict(),
                "resolved": resolved,
                "context":  context,
                "summary":  dc.summary(),
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── S25-4: Sensory Bridge ──────────────────────────────────────────────────
    def _api_brain_sensory(self):
        """GET /api/brain/sensory — SensoryBridge calibration summary."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.sensory_bridge:
                self._json_response(brain.sensory_bridge.summary())
            else:
                self._json_response({"total_grounded": 0, "wired": False,
                                     "error": "SensoryBridge not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_sensory_observe(self, body: dict):
        """POST /api/brain/sensory — run a physics-grounded observation cycle."""
        expression = body.get("expression", "mass=2.0 velocity=3.0")
        domain     = body.get("domain", "mechanics")
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if not brain or not brain.sensory_bridge:
                self._json_response({"error": "SensoryBridge not available"})
                return
            result = brain.sensory_bridge.run_grounded_cycle(expression, domain)
            self._json_response({
                "cycle":   result,
                "summary": brain.sensory_bridge.summary(),
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── S26-1: Dream Consolidator ──────────────────────────────────────────────
    def _api_brain_dream(self):
        """GET /api/brain/dream — DreamConsolidator summary."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.dream_consolidator:
                self._json_response(brain.dream_consolidator.summary())
            else:
                self._json_response({"tick_count": 0, "total_discovered": 0,
                                     "error": "DreamConsolidator not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_dream_cycle(self, body: dict):
        """POST /api/brain/dream — trigger one dream consolidation cycle."""
        max_events = body.get("max_events", 20)
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if not brain or not brain.dream_consolidator:
                self._json_response({"error": "DreamConsolidator not available"})
                return
            rec = brain.dream_consolidator.dream_cycle(max_events=max_events)
            self._json_response({
                "dream":   rec.to_dict(),
                "summary": brain.dream_consolidator.summary(),
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── S26-2: Affective Energy ────────────────────────────────────────────────
    def _api_brain_affective(self):
        """GET /api/brain/affective — AffectiveEnergy summary + curiosity bias."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.affective_energy:
                self._json_response(brain.affective_energy.summary())
            else:
                self._json_response({"total_computed": 0, "interest_rate": 0,
                                     "error": "AffectiveEnergy not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── S26-3: Transform Generator ────────────────────────────────────────────
    def _api_brain_transforms(self):
        """GET /api/brain/transforms — TransformGenerator summary."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.transform_generator:
                self._json_response(brain.transform_generator.summary())
            else:
                self._json_response({"total_promoted": 0, "total_candidates": 0,
                                     "error": "TransformGenerator not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_transforms_action(self, body: dict):
        """POST /api/brain/transforms — generate + promote candidates."""
        action = body.get("action", "generate")
        n      = body.get("n", 5)
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if not brain or not brain.transform_generator:
                self._json_response({"error": "TransformGenerator not available"})
                return
            tg = brain.transform_generator
            if action == "generate":
                batch   = tg.generate_candidates(n=n)
                promoted = tg.promote_best()
                self._json_response({
                    "generated":  len(batch),
                    "promoted":   len(promoted),
                    "new_transforms": [t.to_dict() for t in promoted],
                    "summary":    tg.summary(),
                })
            elif action == "apply":
                expr   = body.get("expression", "x + 0")
                result = tg.apply(expr)
                self._json_response({"original": expr, "result": result,
                                     "changed": result != expr})
            else:
                self._json_response(tg.summary())
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── S26-4: Generative World Model ─────────────────────────────────────────
    def _api_brain_imagine(self):
        """GET /api/brain/imagine — GenerativeWorldModel summary."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.generative_world:
                self._json_response(brain.generative_world.summary())
            else:
                self._json_response({"total_imagined": 0, "total_solved": 0,
                                     "error": "GenerativeWorldModel not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_imagine_action(self, body: dict):
        """POST /api/brain/imagine — imagine problems or run an exploration cycle."""
        action = body.get("action", "explore")
        domain = body.get("domain")
        n      = body.get("n", 3)
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if not brain or not brain.generative_world:
                self._json_response({"error": "GenerativeWorldModel not available"})
                return
            gw = brain.generative_world
            if action == "imagine":
                problems = gw.imagine(domain=domain, n=n)
                self._json_response({
                    "imagined": [p.to_dict() for p in problems],
                    "summary":  gw.summary(),
                })
            else:
                entry = gw.explore_cycle()
                self._json_response({
                    "cycle":   entry,
                    "summary": gw.summary(),
                })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── S26-5: Red Team Adversary ─────────────────────────────────────────────
    def _api_brain_redteam(self):
        """GET /api/brain/redteam — RedTeamAdversary summary."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.red_team:
                self._json_response(brain.red_team.summary())
            else:
                self._json_response({"total_attacks": 0, "total_falsifications": 0,
                                     "error": "RedTeamAdversary not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_redteam_attack(self, body: dict):
        """POST /api/brain/redteam — run one attack round."""
        top_k = body.get("top_k", 3)
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if not brain or not brain.red_team:
                self._json_response({"error": "RedTeamAdversary not available"})
                return
            result = brain.red_team.run_attack_round(top_k=top_k)
            self._json_response({
                "round":   result,
                "summary": brain.red_team.summary(),
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── Progress / chart endpoint ──────────────────────────────────────────
    def _api_brain_progress_action(self, body: dict):
        """POST /api/brain/progress — run autonomous loop or clear data."""
        import pathlib, subprocess, sys
        action = body.get("action", "run")
        try:
            p = pathlib.Path(__file__).resolve().parents[3] / "data" / "memory" / "progress.json"
            repo = pathlib.Path(__file__).resolve().parents[4]
            if action == "clear":
                if p.exists(): p.unlink()
                self._json_response({"cleared": True})
            elif action == "run":
                cycles = int(body.get("cycles", 20))
                script = repo / "scripts" / "autonomous_run.py"
                if not script.exists():
                    self._json_response({"error": f"Script not found: {script}"})
                    return
                proc = subprocess.Popen(
                    [sys.executable, str(script),
                     "--cycles", str(cycles), "--no-plot"],
                    cwd=str(repo),
                    stdout=open("/tmp/sare_autorun.log", "w"),
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                self._json_response({"started": True, "pid": proc.pid, "cycles": cycles})
            else:
                self._json_response({"error": f"Unknown action: {action}"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_progress(self):
        """GET /api/brain/progress — time-series from data/memory/progress.json."""
        import pathlib
        try:
            p = pathlib.Path(__file__).resolve().parents[3] / "data" / "memory" / "progress.json"
            if p.exists():
                data = json.loads(p.read_text())
                runs = data.get("runs", [])
                # Return last 200 points plus live snapshot from brain
                self._json_response({
                    "runs":  runs[-200:],
                    "total": len(runs),
                    "has_data": len(runs) > 0,
                })
            else:
                # Return live snapshot from currently running brain
                from sare.brain import get_brain
                brain = get_brain()
                snapshot = {"cycle": 0, "ts": __import__("time").time(),
                            "solve_rate": 0, "avg_energy": 1.0}
                if brain:
                    try:
                        if brain.robustness_hardener:
                            snapshot["robustness"] = brain.robustness_hardener.overall_robustness()
                        if brain.meta_curriculum:
                            snapshot["meta_lp"] = brain.meta_curriculum.learning_progress_score()
                        if brain.concept_graph and hasattr(brain.concept_graph, '_concepts'):
                            snapshot["concept_count"] = len(brain.concept_graph._concepts)
                    except Exception as _e:
                                                log.debug("[web] Suppressed: %s", _e)
                self._json_response({"runs": [snapshot], "total": 1, "has_data": False})
        except Exception as e:
            self._json_response({"error": str(e), "runs": [], "total": 0, "has_data": False})

    # ── S29-1: Meta-Curriculum Engine ────────────────────────────────────────
    def _api_brain_metacurr(self):
        """GET /api/brain/metacurr — MetaCurriculumEngine summary."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.meta_curriculum:
                self._json_response(brain.meta_curriculum.summary())
            else:
                self._json_response({"tick_count": 0, "error": "MetaCurriculumEngine not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_metacurr_action(self, body: dict):
        """POST /api/brain/metacurr — observe/transfer/tick."""
        action = body.get("action", "tick")
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if not brain or not brain.meta_curriculum:
                self._json_response({"error": "MetaCurriculumEngine not available"})
                return
            mc = brain.meta_curriculum
            if action == "observe":
                mc.observe(body.get("domain", "arithmetic"), body.get("success", True))
                self._json_response(mc.summary())
            elif action == "transfer":
                src = body.get("src", "arithmetic")
                dst = body.get("dst", "algebra")
                res = mc.run_transfer_test(src, dst)
                self._json_response({"result": res.to_dict(), "summary": mc.summary()})
            else:
                result = mc.tick()
                self._json_response({"tick": result, "summary": mc.summary()})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── S29-2: Action Physics Session ─────────────────────────────────────────
    def _api_brain_actionphys(self):
        """GET /api/brain/actionphys — ActionPhysicsSession summary."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.action_physics:
                self._json_response(brain.action_physics.summary())
            else:
                self._json_response({"episodes_run": 0, "error": "ActionPhysicsSession not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_actionphys_run(self, body: dict):
        """POST /api/brain/actionphys — run a physics episode."""
        n_steps = body.get("n_steps", 15)
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if not brain or not brain.action_physics:
                self._json_response({"error": "ActionPhysicsSession not available"})
                return
            ep = brain.action_physics.run_episode(n_steps=n_steps)
            self._json_response({
                "episode": ep.to_dict(),
                "summary": brain.action_physics.summary(),
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── S29-3: Stream Bridge ──────────────────────────────────────────────────
    def _api_brain_streambridge(self):
        """GET /api/brain/streambridge — StreamBridge summary."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.stream_bridge:
                self._json_response(brain.stream_bridge.summary())
            else:
                self._json_response({"queue_depth": 0, "error": "StreamBridge not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_streambridge_action(self, body: dict):
        """POST /api/brain/streambridge — submit item or tick pipeline."""
        action = body.get("action", "tick")
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if not brain or not brain.stream_bridge:
                self._json_response({"error": "StreamBridge not available"})
                return
            sb = brain.stream_bridge
            if action == "submit":
                item = sb.submit(body.get("content", "x+1"),
                                 body.get("source", "EXPLORE"),
                                 body.get("domain", "algebra"))
                self._json_response({"item": item.to_dict(), "summary": sb.summary()})
            else:
                result = sb.tick()
                self._json_response({"tick": result, "summary": sb.summary()})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── S29-4: Perception Bridge ──────────────────────────────────────────────
    def _api_brain_percept(self):
        """GET /api/brain/percept — PerceptionBridge summary."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.perception_bridge:
                self._json_response(brain.perception_bridge.summary())
            else:
                self._json_response({"total_parsed": 0, "error": "PerceptionBridge not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_percept_parse(self, body: dict):
        """POST /api/brain/percept — parse a scene description."""
        description = body.get("description", "")
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if not brain or not brain.perception_bridge:
                self._json_response({"error": "PerceptionBridge not available"})
                return
            pb = brain.perception_bridge
            scene = pb.parse_scene(description)
            self._json_response({
                "scene":   scene.to_dict(),
                "summary": pb.summary(),
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── S28-1: Robustness Hardener ────────────────────────────────────────────
    def _api_brain_robust(self):
        """GET /api/brain/robust — RobustnessHardener summary."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.robustness_hardener:
                self._json_response(brain.robustness_hardener.summary())
            else:
                self._json_response({"overall_robustness": 0, "total_runs": 0,
                                     "error": "RobustnessHardener not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_robust_batch(self, body: dict):
        """POST /api/brain/robust — run a stress batch."""
        domain = body.get("domain")
        n      = body.get("n", 10)
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if not brain or not brain.robustness_hardener:
                self._json_response({"error": "RobustnessHardener not available"})
                return
            rh = brain.robustness_hardener
            records = rh.run_stress_batch(domain=domain, n=n)
            self._json_response({
                "ran":     len(records),
                "passed":  sum(1 for r in records if r.passed),
                "records": [r.to_dict() for r in records],
                "summary": rh.summary(),
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── S28-2: Attention Router ───────────────────────────────────────────────
    def _api_brain_attention(self):
        """GET /api/brain/attention — AttentionRouter summary."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.attention_router:
                self._json_response(brain.attention_router.summary())
            else:
                self._json_response({"total_received": 0, "total_routed": 0,
                                     "error": "AttentionRouter not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── S28-3: Recursive ToM ──────────────────────────────────────────────────
    def _api_brain_tom(self):
        """GET /api/brain/tom — RecursiveToM summary."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.recursive_tom:
                self._json_response(brain.recursive_tom.summary())
            else:
                self._json_response({"total_agents": 0, "max_depth": 3,
                                     "error": "RecursiveToM not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_tom_action(self, body: dict):
        """POST /api/brain/tom — predict_action or resolve_disagreement."""
        action   = body.get("action", "predict")
        agent_id = body.get("agent_id", "agent_0")
        context  = body.get("context", "general")
        depth    = body.get("depth", 1)
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if not brain or not brain.recursive_tom:
                self._json_response({"error": "RecursiveToM not available"})
                return
            tom = brain.recursive_tom
            if action == "predict":
                result = tom.predict_action(agent_id, context, depth=depth)
                self._json_response(result)
            elif action == "resolve":
                agent_b = body.get("agent_b", "agent_1")
                result  = tom.resolve_disagreement(agent_id, agent_b, context)
                self._json_response(result)
            elif action == "update":
                claim = body.get("claim", context)
                conf  = body.get("confidence", 0.5)
                tom.update_model(agent_id, claim, conf, body.get("domain","general"), depth)
                self._json_response(tom.summary())
            else:
                self._json_response(tom.summary())
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── S28-4: Agent Memory Bank ──────────────────────────────────────────────
    def _api_brain_agentmem(self):
        """GET /api/brain/agentmem — AgentMemoryBank summary."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.agent_memory_bank:
                self._json_response(brain.agent_memory_bank.summary())
            else:
                self._json_response({"n_agents": 0,
                                     "error": "AgentMemoryBank not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_agentmem_action(self, body: dict):
        """POST /api/brain/agentmem — remember/learn/recall/trust."""
        action   = body.get("action", "summary")
        agent_id = body.get("agent_id", "agent_0")
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if not brain or not brain.agent_memory_bank:
                self._json_response({"error": "AgentMemoryBank not available"})
                return
            amb = brain.agent_memory_bank
            if action == "remember":
                amb.remember(agent_id, body.get("event_type","interaction"),
                             body.get("description",""), body.get("domain","general"),
                             body.get("outcome","unknown"))
            elif action == "learn":
                amb.learn(agent_id, body.get("claim",""), body.get("confidence",0.5),
                          body.get("domain","general"), body.get("source","api"))
            elif action == "recall":
                facts = amb.recall(agent_id, body.get("query",""), n=body.get("n",5))
                self._json_response({"facts": [f.to_dict() for f in facts],
                                     "agent_id": agent_id})
                return
            elif action == "trust":
                amb.update_trust(agent_id, body.get("outcome","success"))
            self._json_response(amb.summary())
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── S27-1: Continuous Stream Learner ──────────────────────────────────────
    def _api_brain_stream(self):
        """GET /api/brain/stream — ContinuousStreamLearner summary."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.continuous_stream:
                self._json_response(brain.continuous_stream.summary())
            else:
                self._json_response({"running": False, "n_streams": 0,
                                     "total_solved": 0, "error": "ContinuousStream not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_stream_action(self, body: dict):
        """POST /api/brain/stream — start/stop/pause/resume a stream."""
        action    = body.get("action", "summary")
        stream_id = body.get("stream_id")
        n_streams = body.get("n_streams", 4)
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if not brain or not brain.continuous_stream:
                self._json_response({"error": "ContinuousStream not available"})
                return
            csl = brain.continuous_stream
            if action == "start":
                csl.start(n_streams=n_streams)
            elif action == "stop":
                csl.stop()
            elif action == "pause" and stream_id:
                csl.pause_stream(stream_id)
            elif action == "resume" and stream_id:
                csl.resume_stream(stream_id)
            self._json_response(csl.summary())
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── S26-6: Temporal Identity ──────────────────────────────────────────────
    def _api_brain_identity(self):
        """GET /api/brain/identity — TemporalIdentity summary."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.temporal_identity:
                self._json_response(brain.temporal_identity.summary())
            else:
                self._json_response({"session_count": 0, "total_solves": 0,
                                     "error": "TemporalIdentity not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_selfmodel(self):
        """GET /api/brain/selfmodel — full self-report: skills, weaknesses, strategies, goals."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.self_model:
                self._json_response(brain.self_model.self_report())
            else:
                self._json_response({
                    "total_solves": 0, "skill_snapshot": {}, "domains": {},
                    "strategies": {}, "best_strategy": "beam_search",
                    "weaknesses": [], "learning_goals": [],
                    "error": "SelfModel not loaded"
                })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_metalearner(self):
        """GET /api/brain/metalearner — current config, experiment history, summary."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.meta_learner:
                self._json_response(brain.meta_learner.summary())
            else:
                self._json_response({
                    "experiments_run": 0, "current_config": None,
                    "best_config": None, "improvement_history": [],
                    "recent_results": [], "error": "MetaLearner not loaded"
                })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── Upgrade 1: Physics Simulator ──────────────────────────────────────────
    def _api_brain_physics(self):
        """GET /api/brain/physics — current physics simulator summary."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.physics_simulator:
                self._json_response(brain.physics_simulator.summary())
            else:
                self._json_response({"objects": 0, "total_events": 0,
                                     "ticks": 0, "error": "PhysicsSimulator not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_physics_run(self, body: dict):
        """POST /api/brain/physics — run a physics simulation session."""
        n = body.get("n_events", 5)
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if not brain or not brain.physics_simulator:
                self._json_response({"error": "PhysicsSimulator not available"})
                return
            events = brain.physics_simulator.run_session(n)
            fed = 0
            if brain.concept_graph:
                fed = brain.physics_simulator.feed_to_concept_graph(brain.concept_graph)
            self._json_response({
                "events_generated": len(events),
                "concepts_fed": fed,
                "events": [e.to_dict() for e in events[:6]],
                "summary": brain.physics_simulator.summary(),
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── Upgrade 2: Knowledge Ingester ──────────────────────────────────────────
    def _api_brain_knowledge(self):
        """GET /api/brain/knowledge — ingested knowledge base summary."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.knowledge_ingester:
                self._json_response(brain.knowledge_ingester.summary())
            else:
                self._json_response({"titles_ingested": 0, "concepts_extracted": 0,
                                     "error": "KnowledgeIngester not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── Upgrade 3: Multi-Agent Arena ───────────────────────────────────────────
    def _api_brain_arena(self):
        """GET /api/brain/arena — agent fleet summary + race history."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.multi_agent_arena:
                self._json_response(brain.multi_agent_arena.summary())
            else:
                self._json_response({"n_agents": 0, "total_races": 0,
                                     "error": "MultiAgentArena not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_arena_race(self, body: dict):
        """POST /api/brain/arena — race all agents on a given expression."""
        expression = body.get("expression", "x + 0")
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if not brain or not brain.multi_agent_arena:
                self._json_response({"error": "MultiAgentArena not available"})
                return

            def engine_fn(expr, beam_width=8, budget_seconds=2.0,
                          strategy="beam_search", max_depth=12):
                return brain.solve(expr)

            winner, all_results = brain.multi_agent_arena.race(expression, engine_fn)
            self._json_response({
                "winner": winner.to_dict(),
                "all_results": [r.to_dict() for r in all_results],
                "summary": brain.multi_agent_arena.summary(),
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── Upgrade 5: Global Workspace ────────────────────────────────────────────
    def _api_brain_workspace(self):
        """GET /api/brain/workspace — global workspace attention focus + broadcast log."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.global_workspace:
                self._json_response(brain.global_workspace.summary())
            else:
                self._json_response({"pending_messages": 0, "total_broadcast": 0,
                                     "error": "GlobalWorkspace not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── Upgrade 4: Multi-Modal Parser ──────────────────────────────────────────
    def _api_brain_multimodal_demo(self):
        """GET /api/brain/multimodal — show supported modalities."""
        self._json_response({
            "modalities": ["nl", "code", "table", "latex", "csv"],
            "description": "POST /api/brain/multimodal with {text, domain} to parse any modality",
        })

    def _api_brain_multimodal_parse(self, body: dict):
        """POST /api/brain/multimodal — parse table/code/csv/latex/nl to expressions."""
        text = body.get("text", "")
        domain = body.get("domain", "general")
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.multi_modal_parser:
                result = brain.multi_modal_parser.parse(text, domain)
            else:
                from sare.interface.nl_parser_v2 import MultiModalParser
                result = MultiModalParser().parse(text, domain)
            self._json_response({
                "modality": result.modality,
                "expressions": result.expressions,
                "domain": result.domain,
                "confidence": result.confidence,
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── Gap 1: Predictive World Loop ───────────────────────────────────────────
    def _api_brain_predictive(self):
        """GET /api/brain/predictive — predictive loop summary."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.predictive_loop:
                self._json_response(brain.predictive_loop.summary())
            else:
                self._json_response({"total_cycles": 0, "avg_prediction_error": 0,
                                     "error": "PredictiveLoop not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_predictive_run(self, body: dict):
        """POST /api/brain/predictive — run one predict→act→observe→update cycle."""
        expression = body.get("expression", "x + 0")
        domain = body.get("domain", "arithmetic")
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if not brain or not brain.predictive_loop:
                self._json_response({"error": "PredictiveLoop not available"})
                return
            result = brain.predictive_loop.run_cycle(
                expression, domain,
                lambda e: brain.solve(e),
            )
            self._json_response({
                "cycle": result,
                "summary": brain.predictive_loop.summary(),
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── Gap 2: Autonomous Trainer ──────────────────────────────────────────────
    def _api_brain_trainer(self):
        """GET /api/brain/trainer — autonomous trainer summary + live stats."""
        import json as _json
        from pathlib import Path as _Path
        _stats_path = _Path(__file__).resolve().parents[3] / "data" / "memory" / "autonomous_trainer_stats.json"
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.autonomous_trainer and brain.autonomous_trainer._running:
                self._json_response(brain.autonomous_trainer.summary())
                return
            # Fall back to last persisted stats written by the daemon process
            if _stats_path.exists():
                try:
                    data = _json.loads(_stats_path.read_text(encoding="utf-8"))
                    data["from_disk"] = True
                    data["running"] = False
                    self._json_response(data)
                    return
                except Exception:
                    pass
            self._json_response({"running": False, "total_problems": 0,
                                 "error": "Trainer not running (daemon not active)"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_trainer_control(self, body: dict):
        """POST /api/brain/trainer — start/stop trainer or inject a problem."""
        action = body.get("action", "status")
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if not brain or not brain.autonomous_trainer:
                self._json_response({"error": "AutonomousTrainer not available"})
                return
            at = brain.autonomous_trainer
            if action == "start":
                started = at.start(brain)
                self._json_response({"started": started, "running": at._running})
            elif action == "stop":
                stopped = at.stop()
                self._json_response({"stopped": stopped, "running": at._running})
            elif action == "inject":
                expr = body.get("expression", "x + 0")
                dom = body.get("domain", "arithmetic")
                at.add_problem(expr, dom)
                self._json_response({"injected": expr, "domain": dom})
            else:
                self._json_response(at.summary())
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_composite(self):
        """GET /api/brain/composite — CompositeRuleLearner stats."""
        try:
            from sare.learning.composite_rule_learner import get_composite_learner
            cl = get_composite_learner()
            self._json_response({
                "total_traces": cl._total_traces,
                "pair_count": len(cl._pair_counts),
                "registered_composites": len(cl._registered),
                "top_pairs": [{"pair": list(k), "count": v}
                              for k, v in cl._pair_counts.most_common(10)],
                "registered": list(cl._registered.values())[:20],
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_astar_stats(self):
        """GET /api/brain/astar — AStarSearch availability check."""
        try:
            from sare.search.astar_search import AStarSearch
            self._json_response({"available": True, "max_open_set": 512})
        except Exception as e:
            self._json_response({"available": False, "error": str(e)})

    def _api_config_update(self, path: str, body: dict):
        """POST /api/config/beam_width or /api/config/budget_seconds — live param update."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            value = body.get("value")
            if value is None:
                self._json_response({"error": "missing value"}, 400)
                return
            if path == "/api/config/beam_width" and brain:
                brain.beam_width = int(value)
                self._json_response({"beam_width": int(value)})
            elif path == "/api/config/budget_seconds" and brain:
                if hasattr(brain, 'experiment_runner') and brain.experiment_runner:
                    brain.experiment_runner.budget_seconds = float(value)
                self._json_response({"budget_seconds": float(value)})
            else:
                self._json_response({"error": "not found"}, 404)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_hypothesis(self):
        """GET /api/brain/hypothesis — HypothesisEngine stats."""
        try:
            from sare.search.hypothesis_engine import get_hypothesis_engine
            self._json_response(get_hypothesis_engine().summary())
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_datasets(self):
        """GET /api/datasets — external dataset ingester summary."""
        try:
            from sare.knowledge.external_datasets import get_dataset_ingester
            self._json_response(get_dataset_ingester().summary())
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_benchmark_hard(self):
        """GET /api/benchmark/hard — hard suite score trend."""
        try:
            from sare.benchmarks.hard_suite_runner import get_hard_suite_runner
            self._json_response(get_hard_suite_runner().trend())
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_benchmark_hard_run(self):
        """POST /api/benchmark/hard — run hard suite and return full results."""
        try:
            from sare.benchmarks.hard_suite_runner import get_hard_suite_runner
            from sare.brain import get_brain
            brain = get_brain()
            if not brain:
                self._json_response({"error": "Brain not available"}, 503)
                return
            runner = get_hard_suite_runner()
            result = runner.run(lambda expr: brain.solve(expr), budget_seconds=5.0)
            self._json_response(result)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── Gap 3: Agent Society ───────────────────────────────────────────────────
    def _api_brain_society(self):
        """GET /api/brain/society — agent society summary."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if brain and brain.agent_society:
                summary = brain.agent_society.summary()
                if hasattr(brain, 'agent_negotiator') and brain.agent_negotiator:
                    summary["negotiated_truths"] = [
                        {
                            "signature": t.get("signature", ""),
                            "winner": t.get("winning_agent", ""),
                            "cost": round(t.get("energy_cost", 0.0), 2),
                            "competing": t.get("competing_proofs", 0)
                        } for t in brain.agent_negotiator.get_recent_truths(10)
                    ]
                self._json_response(summary)
            else:
                self._json_response({"n_agents": 0, "blackboard_size": 0,
                                     "error": "AgentSociety not loaded"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_society_deliberate(self, body: dict):
        """POST /api/brain/society — run a deliberation cycle or seed a belief."""
        action = body.get("action", "deliberate")
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if not brain or not brain.agent_society:
                self._json_response({"error": "AgentSociety not available"})
                return
            society = brain.agent_society
            if action == "deliberate":
                result = society.deliberation_cycle()
                if brain.concept_graph:
                    society.feed_to_concept_graph(brain.concept_graph)
                self._json_response({
                    "deliberation": result,
                    "summary": society.summary(),
                })
            elif action == "broadcast":
                content = body.get("content", "x + 0 = x")
                domain = body.get("domain", "arithmetic")
                agent_id = body.get("agent", list(society._agents.keys())[0])
                from sare.agent.agent_society import AgentMessage
                msg = AgentMessage(sender=agent_id, msg_type="belief",
                                   content=content, domain=domain, confidence=0.8)
                society.broadcast(msg)
                self._json_response({"broadcast": content, "sender": agent_id})
            else:
                self._json_response(society.summary())
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_metacognition(self):
        """GET /api/brain/metacognition — rich self-awareness snapshot."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            result: dict = {}

            # ── Conjecture engine ─────────────────────────────────────────────
            if brain and hasattr(brain, 'conjecture_engine') and brain.conjecture_engine:
                result["conjecture_engine"] = brain.conjecture_engine.summary()
            else:
                result["conjecture_engine"] = {"total_conjectures": 0, "status_breakdown": {}}

            # ── Meta curriculum ───────────────────────────────────────────────
            if brain and brain.meta_curriculum:
                result["meta_curriculum"] = brain.meta_curriculum.summary()
            else:
                result["meta_curriculum"] = {}

            # ── Analogy transfer ──────────────────────────────────────────────
            if brain and hasattr(brain, 'analogy_transfer') and brain.analogy_transfer:
                at = brain.analogy_transfer
                result["analogy_transfer"] = {
                    "transfers_applied": getattr(at, '_transfers_applied', 0),
                    "domains_covered": list(getattr(at, '_covered_domains', set())),
                }
            else:
                result["analogy_transfer"] = {"transfers_applied": 0}

            # ── Homeostasis drives ────────────────────────────────────────────
            try:
                hs = brain.homeostasis if brain else None
                if hs:
                    hs_state = hs.get_state()
                    result["drives"] = hs_state.get("drives", {})
                    result["dominant_drive"] = hs_state.get("dominant_drive", "")
                    result["behavior_recommendation"] = hs_state.get("behavior_recommendation", "")
            except Exception:
                result["drives"] = {}

            # ── Self-model: domain skills, top transforms ─────────────────────
            try:
                sm = brain.self_model if brain else None
                if sm:
                    snap = sm.snapshot() if hasattr(sm, 'snapshot') else {}
                    result["domain_skills"] = snap.get("domains", {})
                    result["top_transforms"] = snap.get("top_transforms", [])[:8]
                    result["calibration_error"] = snap.get("calibration_error", 0)
                    result["total_solves"] = snap.get("total_solves", 0)
                    result["learning_goals"] = snap.get("learning_goals", [])[:5]
            except Exception as _e:
                                log.debug("[web] Suppressed: %s", _e)

            # ── Identity traits ───────────────────────────────────────────────
            try:
                identity = brain.identity if brain else None
                if identity:
                    result["traits"] = [
                        {"name": name, "strength": round(t.strength, 3)}
                        for name, t in getattr(identity, '_traits', {}).items()
                    ]
                    result["identity_description"] = identity.get_self_description() if hasattr(identity, 'get_self_description') else ""
                    result["learning_style"] = identity.get_learning_style() if hasattr(identity, 'get_learning_style') else {}
            except Exception:
                result["traits"] = []

            # ── Active goals ──────────────────────────────────────────────────
            try:
                gs = brain.goal_setter if brain else None
                if gs:
                    goals = getattr(gs, '_goals', []) or []
                    result["active_goals"] = [
                        {"desc": getattr(g, 'description', str(g)), "domain": getattr(g, 'domain', ''),
                         "priority": getattr(g, 'priority', 0), "status": getattr(g, 'status', '')}
                        for g in goals[:6]
                    ]
            except Exception:
                result["active_goals"] = []

            # ── Credit assigner: best/worst transforms ────────────────────────
            try:
                ca = brain.credit_assigner if brain else None
                if ca:
                    credits = getattr(ca, '_transform_credits', {})
                    if credits:
                        sorted_c = sorted(credits.items(), key=lambda x: -x[1])
                        result["top_credited"] = [{"name": k, "credit": round(v, 2)} for k, v in sorted_c[:5]]
                        result["bottom_credited"] = [{"name": k, "credit": round(v, 2)} for k, v in sorted_c[-3:] if v < 1.0]
            except Exception as _e:
                                log.debug("[web] Suppressed: %s", _e)

            # ── World model surprise domains ──────────────────────────────────
            try:
                from sare.memory.world_model import get_world_model
                wm = get_world_model()
                if hasattr(wm, 'get_high_surprise_domains'):
                    result["high_surprise_domains"] = [
                        {"domain": d, "surprise": round(s, 3)}
                        for d, s in wm.get_high_surprise_domains(top_n=5)
                    ]
                wm_state = wm.prediction_stats() if hasattr(wm, 'prediction_stats') else {}
                result["wm_accuracy"] = round(wm_state.get("accuracy", 0), 3)
                result["wm_links"] = len(wm._causal_links)
            except Exception as _e:
                                log.debug("[web] Suppressed: %s", _e)

            # ── Conjectures (from world model / concept graph) ────────────────
            try:
                conjectures = []
                if brain and hasattr(brain, 'world_model_v3') and brain.world_model_v3:
                    conjectures += getattr(brain.world_model_v3, '_conjectures', [])[:4]
                if brain and hasattr(brain, 'concept_graph') and brain.concept_graph:
                    cg = brain.concept_graph
                    conjectures += getattr(cg, '_analogy_conjectures', [])[:4]
                result["conjectures"] = [
                    {"hypothesis": getattr(c, 'hypothesis', str(c))[:80],
                     "type": getattr(c, 'type', ''),
                     "plausibility": round(getattr(c, 'plausibility', 0.5), 2),
                     "domain": getattr(c, 'domain', '')}
                    for c in conjectures[:8]
                ]
            except Exception:
                result["conjectures"] = []

            # ── Focus recommendation (combined signal) ────────────────────────
            try:
                mc = brain.meta_curriculum if brain else None
                focus_domain = ""
                focus_reason = ""
                if mc and hasattr(mc, '_domains'):
                    # pick domain with highest weight and non-zero attempts
                    domains = getattr(mc, '_domains', {})
                    by_weight = sorted(domains.items(), key=lambda x: -x[1].weight if hasattr(x[1], 'weight') else 0)
                    if by_weight:
                        focus_domain = by_weight[0][0]
                        focus_reason = "highest curriculum weight"
                # override with surprise domains
                if result.get("high_surprise_domains"):
                    focus_domain = result["high_surprise_domains"][0]["domain"]
                    focus_reason = f"highest surprise ({result['high_surprise_domains'][0]['surprise']:.2f}x)"
                result["focus_domain"] = focus_domain
                result["focus_reason"] = focus_reason
            except Exception as _e:
                                log.debug("[web] Suppressed: %s", _e)

            self._json_response(result)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_metacognition_think(self, body: dict):
        """POST /api/brain/metacognition — trigger a metacognition tick (reflect + conjecture)."""
        try:
            from sare.brain import get_brain
            brain = get_brain()
            actions_taken = []

            # Generate learning goals from self-model
            if brain and brain.self_model and hasattr(brain.self_model, 'generate_learning_goals'):
                try:
                    goals = brain.self_model.generate_learning_goals()
                    actions_taken.append(f"generated {len(goals)} learning goals")
                except Exception as e:
                    actions_taken.append(f"goals failed: {e}")

            # Tick homeostasis
            if brain and brain.homeostasis:
                try:
                    brain.homeostasis.tick()
                    actions_taken.append("homeostasis ticked")
                except Exception as _e:
                                        log.debug("[web] Suppressed: %s", _e)

            # Run conjecture engine tick
            if brain and hasattr(brain, 'conjecture_engine') and brain.conjecture_engine:
                try:
                    ce = brain.conjecture_engine
                    if hasattr(ce, 'tick'):
                        ce.tick()
                        actions_taken.append("conjecture engine ticked")
                except Exception as e:
                    actions_taken.append(f"conjecture tick failed: {e}")

            # Update meta curriculum from recent solve history
            if brain and brain.meta_curriculum and brain.self_model:
                try:
                    sm = brain.self_model
                    snap = sm.snapshot() if hasattr(sm, 'snapshot') else {}
                    for domain_name, info in snap.get("domains", {}).items():
                        rate = info.get("solve_rate", 0) if isinstance(info, dict) else 0
                        success = rate >= 0.5
                        brain.meta_curriculum.observe(domain_name, success)
                    actions_taken.append("meta_curriculum updated from self-model")
                except Exception as e:
                    actions_taken.append(f"meta_curriculum update failed: {e}")

            self._json_response({"status": "ok", "actions": actions_taken})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_selfmodel_action(self, body: dict):
        """POST /api/brain/selfmodel — generate goals or detect weaknesses."""
        action = body.get("action", "snapshot")
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if not brain or not brain.self_model:
                self._json_response({"error": "SelfModel not available"})
                return
            sm = brain.self_model
            if action == "generate_goals":
                try:
                    goals = sm.generate_learning_goals(n=5)
                    self._json_response({"generated": len(goals),
                                         "goals": [g.__dict__ if hasattr(g, '__dict__') else str(g)
                                                   for g in goals]})
                except Exception as e:
                    self._json_response({"error": str(e)})
            elif action == "detect_weaknesses":
                try:
                    weak = sm.detect_weaknesses(threshold=0.5)
                    self._json_response({"weaknesses": weak})
                except Exception as e:
                    self._json_response({"error": str(e)})
            else:
                self._api_brain_selfmodel()
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_metalearner_control(self, body: dict):
        """POST /api/brain/metalearner — trigger tune_beam_width or full_tune."""
        action = body.get("action", "tune_beam_width")
        try:
            from sare.brain import get_brain
            brain = get_brain()
            if not brain or not brain.meta_learner:
                self._json_response({"error": "MetaLearner not available"})
                return
            ml = brain.meta_learner
            if action == "tune_beam_width":
                best = ml.tune_beam_width()
                ml.apply_to_brain(brain)
                self._json_response({
                    "status": "tuned",
                    "best_config": best.to_dict(),
                    "summary": ml.summary(),
                })
            elif action == "full_tune":
                best = ml.full_tune()
                ml.apply_to_brain(brain)
                self._json_response({
                    "status": "full_tuned",
                    "best_config": best.to_dict(),
                    "summary": ml.summary(),
                })
            else:
                self._json_response({"error": f"Unknown action: {action}"})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── Multi-Domain Knowledge Endpoints ─────────────────────────────────────

    def _api_knowledge_commonsense(self):
        """GET /api/knowledge/commonsense — current commonsense KB stats."""
        try:
            from sare.knowledge.commonsense import CommonSenseBase
            kb = CommonSenseBase()
            kb.load()
            if kb.total_facts() == 0:
                kb.seed()
            subjects = list(kb._forward.keys())
            sample = kb.query(subjects[0], depth=1)[:5] if subjects else []
            self._json_response({
                "total_facts": kb.total_facts(),
                "subjects": len(subjects),
                "sample_subject": subjects[0] if subjects else "",
                "sample_facts": sample,
                "top_subjects": subjects[:20],
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_knowledge_lookup(self, q: str, domain: str):
        """GET /api/knowledge/lookup?q=<question>&domain=<domain> — KB fan-out lookup."""
        if not q:
            self._json_response({"error": "q parameter required"}, 400)
            return
        try:
            from sare.memory.knowledge_lookup import KnowledgeLookup, DIRECT_THRESHOLD
            kl  = KnowledgeLookup()
            hit = kl.lookup(q, domain)
            self._json_response({
                "hit":        hit is not None,
                "answer":     hit.answer if hit else None,
                "confidence": hit.confidence if hit else 0.0,
                "source":     hit.source if hit else None,
                "direct":     hit.confidence >= DIRECT_THRESHOLD if hit else False,
                "context_facts": hit.context_facts if hit else [],
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_books_status(self):
        """GET /api/books — list ingested books and totals."""
        try:
            from sare.knowledge.book_ingester import get_book_ingester
            self._json_response(get_book_ingester().status())
        except Exception as e:
            self._json_response({"error": str(e), "total_books": 0, "books": []})

    def _api_books_ingest(self, body: dict):
        """POST /api/books/ingest {"path": "/absolute/path/to/book.pdf"} — ingest a book file."""
        import threading as _bt
        path_str = body.get("path", "").strip()
        if not path_str:
            self._json_response({"error": "path required"}, 400)
            return
        from pathlib import Path as _P
        fpath = _P(path_str)
        if not fpath.exists():
            self._json_response({"error": f"File not found: {path_str}"}, 404)
            return

        def _do_ingest():
            try:
                from sare.knowledge.book_ingester import get_book_ingester
                get_book_ingester().ingest_file(fpath, force=body.get("force", False))
            except Exception:
                pass

        _bt.Thread(target=_do_ingest, daemon=True).start()
        self._json_response({"status": "ingesting", "path": str(fpath),
                             "message": "Ingestion started in background"})

    def _api_knowledge_stats(self):
        """GET /api/knowledge/stats — aggregate KB statistics."""
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            wm_facts = sum(len(v) for v in wm._facts.values())

            kg_nodes = 0
            try:
                from sare.memory.knowledge_graph import KnowledgeGraph
                kg = KnowledgeGraph()
                kg_nodes = len(kg._nodes)
            except Exception:
                pass

            cs_facts = 0
            answer_to_count = 0
            try:
                from sare.knowledge.commonsense import get_commonsense_base
                cs = get_commonsense_base()
                cs_facts = cs.total_facts()
                # Count AnswerTo triples from live in-memory singleton
                answer_to_count = sum(
                    1 for triples in cs._forward.values()
                    for rel, _ in triples if rel == "AnswerTo"
                )
            except Exception:
                pass

            wm_sessions = 0
            kb_hit_rate = 0.0
            try:
                from sare.memory.working_memory import WorkingMemory
                wmem = WorkingMemory()
                wm_sessions = wmem._session_count
            except Exception:
                pass

            # Prefer Brain's live kb_lookup for hit rate if Brain is running
            try:
                from sare.brain import get_brain
                _b = get_brain()
                if _b and _b.kb_lookup is not None:
                    kb_hit_rate = _b.kb_lookup.get_hit_rate()
                    wm_sessions = _b.working_memory._session_count if _b.working_memory else wm_sessions
                else:
                    from sare.cognition.general_solver import get_general_solver
                    gs = get_general_solver()
                    if gs._kb_lookup is not None:
                        kb_hit_rate = gs._kb_lookup.get_hit_rate()
            except Exception:
                pass

            self._json_response({
                "world_model_facts":       wm_facts,
                "knowledge_graph_nodes":   kg_nodes,
                "commonsense_facts":       cs_facts,
                "answer_to_triples":       answer_to_count,
                "working_memory_sessions": wm_sessions,
                "kb_hit_rate_last_100":    kb_hit_rate,
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_learning_summary(self):
        """GET /api/learning/summary — general intelligence KB metrics."""
        import time as _time
        try:
            from sare.memory.world_model import get_world_model
            from sare.knowledge.commonsense import get_commonsense_base

            wm = get_world_model()
            cs = get_commonsense_base()

            # AnswerTo triple count (live in-memory)
            answer_to_count = sum(
                1 for triples in cs._forward.values()
                for rel, _ in triples if rel == "AnswerTo"
            )

            # World model facts per domain
            wm_facts_by_domain = {d: len(v) for d, v in wm._facts.items() if v}
            wm_facts_total = sum(wm_facts_by_domain.values())

            # Domain failure counts (last 500 per domain)
            domain_failures: dict = {}
            try:
                with wm._domain_failures_lock:
                    for d, entries in wm._domain_failures.items():
                        domain_failures[d] = len(entries)
            except Exception:
                pass

            # Rolling solve accuracy per domain (via _belief_accuracy keys)
            domain_accuracy: dict = {}
            try:
                for key, hist in wm._belief_accuracy.items():
                    if key.startswith("domain_solve_acc:") and hist:
                        dom = key[len("domain_solve_acc:"):]
                        window = min(50, len(hist))
                        domain_accuracy[dom] = round(sum(hist[-window:]) / window, 3)
            except Exception:
                pass

            # Concept synthesis domains triggered (each has had at least one LLM synthesis call)
            synthesis_count = 0
            try:
                synthesis_count = len(wm._last_concept_synthesis)
            except Exception:
                pass

            # KB hit rate from general solver
            kb_hit_rate = 0.0
            try:
                from sare.cognition.general_solver import get_general_solver
                gs = get_general_solver()
                if gs._kb_lookup is not None:
                    kb_hit_rate = gs._kb_lookup.get_hit_rate()
            except Exception:
                pass

            self._json_response({
                "answer_to_triples":    answer_to_count,
                "wm_facts_total":       wm_facts_total,
                "wm_facts_by_domain":   wm_facts_by_domain,
                "domain_failures":      domain_failures,
                "domain_accuracy":      domain_accuracy,
                "synthesis_count":      synthesis_count,
                "kb_hit_rate":          kb_hit_rate,
                "ts":                   _time.time(),
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_learning_live(self):
        """GET /api/learning/live — live snapshot: domain accuracy, web search log, LLM call stats."""
        import time as _time
        import json as _json
        from pathlib import Path as _Path

        result: dict = {"ts": _time.time()}

        # ── Domain accuracy (rolling window from world model) ─────────────────
        domain_accuracy: dict = {}
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            for key, hist in wm._belief_accuracy.items():
                if key.startswith("domain_solve_acc:") and hist:
                    dom = key[len("domain_solve_acc:"):]
                    window = min(50, len(hist))
                    domain_accuracy[dom] = round(sum(hist[-window:]) / window, 3)
        except Exception:
            pass

        # Also read from world_model_v2 facts count as proxy if belief_accuracy empty
        wm_facts: dict = {}
        try:
            wm_path = _Path(__file__).parent.parent.parent.parent / "data" / "memory" / "world_model_v2.json"
            if wm_path.exists():
                raw = _json.loads(wm_path.read_text())
                facts = raw.get("facts", {})
                wm_facts = {d: len(v) for d, v in facts.items() if v}
        except Exception:
            pass

        result["domain_accuracy"] = domain_accuracy
        result["wm_facts"] = wm_facts

        # ── Web search log — read from disk so we see daemon's searches ─────────
        web_searches: list = []
        try:
            log_path = _Path(__file__).parent.parent.parent.parent / "data" / "memory" / "web_learned.json"
            if log_path.exists():
                raw = _json.loads(log_path.read_text())
                entries = raw.get("entries", [])
                web_searches = list(reversed(entries[-20:]))
        except Exception:
            pass

        result["web_searches"] = web_searches

        # ── LLM call stats ────────────────────────────────────────────────────
        llm_stats: dict = {"total_calls": 0, "by_role": {}, "recent": []}
        try:
            from sare.interface.llm_bridge import get_llm_stats
            llm_stats = get_llm_stats()
        except Exception:
            pass

        result["llm_stats"] = llm_stats

        # ── KB growth snapshot ────────────────────────────────────────────────
        kb_snapshot: dict = {}
        try:
            from sare.knowledge.commonsense import get_commonsense_base
            cs = get_commonsense_base()
            answer_to = sum(1 for triples in cs._forward.values() for r, _ in triples if r == "AnswerTo")
            total_triples = sum(len(v) for v in cs._forward.values())
            kb_snapshot = {"answer_to": answer_to, "total_triples": total_triples}
        except Exception:
            pass

        result["kb_snapshot"] = kb_snapshot

        self._json_response(result)

    def _api_knowledge_expand(self, body: dict):
        """POST /api/knowledge/expand — expand KB via LLM (async)."""
        n = body.get("n", 300)
        import threading

        def _do_expand():
            try:
                from sare.knowledge.commonsense import CommonSenseBase
                kb = CommonSenseBase()
                kb.load()
                if kb.total_facts() == 0:
                    kb.seed()
                added = kb.augment_from_llm(n_facts=n)
                log.info("Knowledge expansion complete: +%d facts", added)
            except Exception as e:
                log.warning("Knowledge expansion failed: %s", e)

        t = threading.Thread(target=_do_expand, daemon=True)
        t.start()
        self._json_response({"status": "expanding", "requested_facts": n,
                             "message": "LLM knowledge expansion started in background"})

    def _api_curiosity_questions(self):
        """GET /api/questions — ActiveQuestioner: what is the AI curious about?"""
        _ensure_legacy_runtime()
        questions = []
        if experiment_runner and hasattr(experiment_runner, "_curiosity_questions"):
            questions = list(experiment_runner._curiosity_questions)
        # Also gather from multi-agent runners
        try:
            from sare.curiosity.multi_agent_learner import get_multi_agent_learner
            mal = get_multi_agent_learner()
            for agent_data in mal._agents.values():
                runner = getattr(agent_data, "_runner", None)
                if runner and hasattr(runner, "_curiosity_questions"):
                    questions.extend(runner._curiosity_questions)
        except Exception as _e:
                        log.debug("[web] Suppressed: %s", _e)
        # Sort by timestamp, most recent first
        questions = sorted(questions, key=lambda q: q.get("timestamp", 0), reverse=True)[:20]
        self._json_response({
            "count": len(questions),
            "questions": questions,
            "note": "These are domains/problems the AI is stuck on and asking about"
        })

    def _api_metacognition_plan(self):
        """GET /api/metacognition/plan — inner monologue sub-goal plans from stuck detection."""
        _ensure_legacy_runtime()
        plans = []
        if experiment_runner and hasattr(experiment_runner, "_meta_plans"):
            plans = list(experiment_runner._meta_plans)
        plans = sorted(plans, key=lambda p: p.get("timestamp", 0), reverse=True)[:10]
        self._json_response({
            "count": len(plans),
            "plans": plans,
            "description": "MetacognitiveController sub-goals generated when stuck on a domain"
        })

    def _api_domains_status(self):
        """GET /api/domains — status of all learning domains."""
        try:
            # Gather stats from experiment runner
            _ensure_legacy_runtime()
            domains_info = {}
            runner = experiment_runner
            if runner:
                stats = runner.stats() if hasattr(runner, 'stats') else {}
                # Domain breakdown from curriculum
                curriculum = getattr(runner, 'curriculum_gen', None)
                if curriculum:
                    for p in getattr(curriculum, 'generated_problems', []):
                        d = p.domain or "general"
                        if d not in domains_info:
                            domains_info[d] = {"pending": 0, "solved": 0, "stuck": 0}
                        domains_info[d][p.status] = domains_info[d].get(p.status, 0) + 1

            # Add multi-agent domain stats
            try:
                from sare.curiosity.multi_agent_learner import get_multi_agent_learner
                mal = get_multi_agent_learner()
                agent_status = mal.get_status()
                for aid, ast in agent_status.get("agents", {}).items():
                    d = ast.get("domain", "general")
                    if d not in domains_info:
                        domains_info[d] = {"pending": 0, "solved": 0, "stuck": 0}
                    domains_info[d]["agent_solved"] = domains_info[d].get("agent_solved", 0) + ast.get("solved", 0)
                    domains_info[d]["agent_attempted"] = domains_info[d].get("agent_attempted", 0) + ast.get("attempted", 0)
            except Exception as _e:
                                log.debug("[web] Suppressed: %s", _e)

            # Add known domains with 0 stats if not present
            for domain in ["arithmetic", "algebra", "logic", "calculus", "code", "qa", "planning"]:
                if domain not in domains_info:
                    domains_info[domain] = {"pending": 0, "solved": 0, "stuck": 0}

            self._json_response({"domains": domains_info})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_domains_seed(self, body: dict):
        """POST /api/domains/seed — seed problems for a specific domain."""
        domain = body.get("domain", "")
        if not domain:
            self._json_response({"error": "domain required"}, 400)
            return
        try:
            _ensure_legacy_runtime()
            runner = experiment_runner
            if not runner:
                self._json_response({"error": "runner not initialized"}, 503)
                return

            curriculum = getattr(runner, 'curriculum_gen', None)
            if not curriculum:
                self._json_response({"error": "curriculum not available"}, 503)
                return

            added = 0
            if domain == "code":
                from sare.perception.graph_builders import CodeGraphBuilder
                builder = CodeGraphBuilder()
                code_seeds = [
                    "x if True else y", "a if False else b",
                    "not not x", "True and x", "x or False",
                    "x and x", "x or x", "not True", "not False",
                ]
                for expr in code_seeds:
                    try:
                        g = builder.build(expr)
                        curriculum.add_problem(g, domain="code")
                        added += 1
                    except Exception as _e:
                                                log.debug("[web] Suppressed: %s", _e)

            elif domain == "qa":
                from sare.knowledge.commonsense import CommonSenseBase
                from sare.agent.qa_pipeline import get_qa_pipeline
                kb = CommonSenseBase()
                kb.load()
                if kb.total_facts() == 0:
                    kb.seed()
                qa = get_qa_pipeline()
                problems = qa.generate_qa_problems(kb, n=body.get("n", 20))
                for p in problems:
                    curriculum.add_problem(p)
                    added += 1

            elif domain == "planning":
                from sare.agent.domains.task_scheduler_domain import get_task_domain
                td = get_task_domain()
                problems = td.generate_problems(n=body.get("n", 10))
                for p in problems:
                    curriculum.add_problem(p)
                    added += 1

            else:
                self._json_response({"error": f"Unknown domain: {domain}. Try code, qa, planning"}, 400)
                return

            self._json_response({"status": "seeded", "domain": domain, "problems_added": added})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── Phase A: Stage + Piaget endpoints ────────────────────────────────────

    def _api_brain_stage(self):
        """GET /api/brain/stage — current developmental stage and capability gates."""
        try:
            import json as _json
            from pathlib import Path as _P
            from sare.brain import get_brain, STAGE_CAPABILITY_GATES
            brain = get_brain()

            # Read persisted state from daemon (cross-process mastery updates)
            _state_path = _P(__file__).resolve().parents[3] / "data" / "memory" / "brain_state.json"
            _persisted = {}
            try:
                if _state_path.exists():
                    _persisted = _json.loads(_state_path.read_text())
            except Exception:
                pass
            _p_stats = _persisted.get("stats", {})
            _p_stage = _persisted.get("stage", "infant")

            # Prefer in-process brain, fall back to persisted file
            if brain:
                stage_val = brain.stage.value
                stage_level = brain.stage.level
                caps = brain.get_stage_capabilities()
                rules_promoted = brain._stats.get("rules_promoted", 0)
                domains_mastered = brain._stats.get("domains_mastered", [])
                solve_rate = round(brain._stats.get("solves_succeeded", 0) /
                                   max(brain._stats.get("solves_attempted", 1), 1), 3)
            else:
                stage_val = _p_stage
                _stage_levels = ["infant","toddler","child","preteen","teenager","undergrad","graduate","researcher"]
                stage_level = _stage_levels.index(stage_val) if stage_val in _stage_levels else 0
                caps = STAGE_CAPABILITY_GATES.get(stage_val, STAGE_CAPABILITY_GATES.get("infant", {}))
                rules_promoted = _p_stats.get("rules_promoted", 0)
                domains_mastered = _p_stats.get("domains_mastered", [])
                solve_rate = 0.0

            # Merge persisted domains_mastered with in-process (daemon may have more)
            for d in _p_stats.get("domains_mastered", []):
                if d not in domains_mastered:
                    domains_mastered.append(d)

            self._json_response({
                "stage": stage_val,
                "stage_level": stage_level,
                "capabilities": caps,
                "rules_promoted": rules_promoted,
                "domains_mastered": domains_mastered,
                "solve_rate": solve_rate,
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_promoted_rules(self):
        """GET /api/brain/promoted-rules — promoted rules with usage counts."""
        try:
            import json as _json
            from pathlib import Path as _P
            p = _P(__file__).resolve().parents[3] / "data" / "memory" / "promoted_rules.json"
            if p.exists():
                d = _json.loads(p.read_text())
                counts = d.get("pattern_counts", {})
                rules = d.get("promoted_rules", [])
                enriched = []
                for r in rules:
                    name = r.get("name", "")
                    enriched.append({
                        "name": name,
                        "domain": r.get("domain", "general"),
                        "confidence": round(r.get("confidence", 0.0), 3),
                        "use_count": counts.get(name, 0),
                    })
                enriched.sort(key=lambda x: x["use_count"], reverse=True)
                self._json_response({
                    "rules": enriched,
                    "total": len(enriched),
                    "pattern_counts": counts,
                })
            else:
                self._json_response({"rules": [], "total": 0, "pattern_counts": {}})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_brain_piaget(self):
        """GET /api/brain/piaget — Piaget milestone progress."""
        try:
            from sare.curriculum.developmental import DevelopmentalCurriculum
            dc = DevelopmentalCurriculum()
            self._json_response(dc.get_piaget_status())
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── Phase B: Predictive Engine endpoint ──────────────────────────────────

    def _api_predictive_status(self):
        """GET /api/predictive/status — predictive engine accuracy and domain surprises."""
        try:
            from sare.cognition.predictive_engine import get_predictive_engine
            self._json_response(get_predictive_engine().get_status())
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── Phase C: Internal Grammar endpoint ───────────────────────────────────

    def _api_grammar_status(self):
        """GET /api/grammar/status — internal grammar vocabulary size and top symbols."""
        try:
            from sare.language.internal_grammar import get_internal_grammar
            self._json_response(get_internal_grammar().get_status())
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── Phase D: Teacher Protocol endpoints ──────────────────────────────────

    def _api_questions_pending(self):
        """GET /api/questions/pending — questions waiting for teacher answers."""
        try:
            from sare.learning.teacher_protocol import get_confusion_detector
            cd = get_confusion_detector()
            questions = cd.get_pending_questions()
            self._json_response({
                "count": len(questions),
                "questions": [q.to_dict() for q in questions],
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_questions_answer(self, body: dict):
        """POST /api/questions/answer — human submits answer to a pending question."""
        question_id = body.get("question_id", "")
        answer_text = body.get("answer", "")
        if not question_id or not answer_text:
            self._json_response({"error": "question_id and answer required"}, 400)
            return
        try:
            from sare.learning.teacher_protocol import get_confusion_detector
            cd = get_confusion_detector()
            q = cd.answer_question(question_id, answer_text, answered_by="human")
            if q:
                self._json_response({"status": "answered", "question": q.to_dict()})
            else:
                self._json_response({"error": "question not found or already answered"}, 404)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_teachers_list(self):
        """GET /api/teachers — list all registered teachers with trust scores."""
        try:
            from sare.learning.teacher_protocol import get_teacher_registry
            self._json_response({"teachers": get_teacher_registry().get_all_teachers()})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_teachers_register(self, body: dict):
        """POST /api/teachers/register — register a new teacher."""
        teacher_type = body.get("type", "llm")
        teacher_id = body.get("teacher_id", f"{teacher_type}_{int(__import__('time').time())}")
        domains = body.get("domains", [])
        try:
            from sare.learning.teacher_protocol import get_teacher_registry, LLMTeacher, DatabaseTeacher
            tr = get_teacher_registry()
            if teacher_type == "llm":
                teacher = LLMTeacher(teacher_id=teacher_id, domains=domains)
            elif teacher_type == "database":
                db_path = body.get("db_path")
                teacher = DatabaseTeacher(teacher_id=teacher_id, domains=domains, db_path=db_path)
            else:
                self._json_response({"error": f"Unknown teacher type: {teacher_type}. Use llm or database"}, 400)
                return
            tr.register(teacher)
            self._json_response({"status": "registered", "teacher": teacher.to_dict()})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── Phase E: Architecture Designer endpoints ──────────────────────────────

    def _api_architecture_gaps(self):
        """GET /api/architecture/gaps — identified capability gaps."""
        try:
            from sare.meta.architecture_designer import get_architecture_designer
            ad = get_architecture_designer()
            gaps = ad.get_gaps()
            if not gaps:
                gaps = ad.identify_gaps()
            self._json_response({"gaps": gaps, "count": len(gaps)})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_architecture_proposals(self):
        """GET /api/architecture/proposals — pending and implemented module proposals."""
        try:
            from sare.meta.architecture_designer import get_architecture_designer
            ad = get_architecture_designer()
            self._json_response({
                "proposals": ad.get_proposals(),
                "by_status": {
                    "proposed":    len(ad.get_proposals("proposed")),
                    "implemented": len(ad.get_proposals("implemented")),
                    "deployed":    len(ad.get_proposals("deployed")),
                    "failed":      len(ad.get_proposals("failed")),
                },
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_architecture_trigger(self):
        """POST /api/architecture/trigger — manually trigger gap analysis + design cycle."""
        try:
            from sare.meta.architecture_designer import get_architecture_designer
            ad = get_architecture_designer()
            gaps = ad.identify_gaps()
            self._json_response({
                "status": "analysis_complete",
                "gaps_found": len(gaps),
                "gaps": gaps[:5],
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── Phase F: Forgetting Curve endpoint ───────────────────────────────────

    def _api_memory_forgetting(self):
        """GET /api/memory/forgetting — forgetting curve stats and due reviews."""
        try:
            from sare.memory.forgetting_curve import get_forgetting_curve
            fc = get_forgetting_curve()
            stats = fc.get_stats()
            due = fc.get_due_reviews(limit=10)
            stats["due_reviews"] = [item.to_dict() for item in due]
            self._json_response(stats)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── General Intelligence endpoints ────────────────────────────────────────

    def _api_think(self, body: dict):
        """POST /api/think — solve any problem with the general intelligence engine.
        Body: {"problem": "...", "context": "...", "type": "auto|math|logic|factual|..."}
        """
        problem = body.get("problem", "").strip()
        if not problem:
            self._json_response({"error": "missing 'problem' field"}, 400)
            return
        context      = body.get("context", "")
        problem_type = body.get("type") or None
        try:
            from sare.cognition.general_solver import get_general_solver
            solver = get_general_solver()
            result = solver.solve(problem, context=context, problem_type=problem_type)
            self._json_response({
                "problem_id":    result.problem_id,
                "problem_type":  result.problem_type,
                "answer":        result.answer,
                "confidence":    round(result.confidence, 3),
                "solved":        result.solved,
                "solver_used":   result.solver_used,
                "elapsed_ms":    round(result.elapsed_ms, 1),
                "reasoning":     result.reasoning[:1500] if result.reasoning else "",
                "lesson":        result.lesson,
                "sub_steps":     result.sub_steps,
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_intelligence_domains(self):
        """GET /api/intelligence/domains — per-domain performance stats."""
        try:
            from sare.cognition.general_solver import get_general_solver
            solver = get_general_solver()
            stats  = solver.get_stats()
            # Annotate with domain metadata
            domain_meta = {
                "math":      {"label": "Symbolic Math",        "icon": "∑"},
                "logic":     {"label": "Formal Logic",         "icon": "⊢"},
                "code":      {"label": "Code Generation",      "icon": "💻"},
                "factual":   {"label": "Factual Knowledge",    "icon": "📚"},
                "reasoning": {"label": "Logical Reasoning",    "icon": "🧠"},
                "analogy":   {"label": "Analogical Thinking",  "icon": "🔗"},
                "science":   {"label": "Science",              "icon": "🔬"},
                "language":  {"label": "Language",             "icon": "💬"},
                "planning":  {"label": "Planning",             "icon": "📋"},
                "social":    {"label": "Social Cognition",     "icon": "👥"},
            }
            domains = []
            for ptype, s in stats.items():
                meta = domain_meta.get(ptype, {"label": ptype, "icon": "?"})
                domains.append({
                    "type":           ptype,
                    "label":          meta["label"],
                    "icon":           meta["icon"],
                    "attempts":       s["attempts"],
                    "solve_rate":     s["solve_rate"],
                    "avg_confidence": s["avg_confidence"],
                })
            # Add zeros for domains not yet attempted
            attempted = {d["type"] for d in domains}
            for ptype, meta in domain_meta.items():
                if ptype not in attempted:
                    domains.append({
                        "type": ptype, "label": meta["label"], "icon": meta["icon"],
                        "attempts": 0, "solve_rate": 0.0, "avg_confidence": 0.0,
                    })
            domains.sort(key=lambda x: x["type"])
            overall_attempts = sum(d["attempts"] for d in domains)
            overall_solved   = sum(int(d["solve_rate"] * d["attempts"]) for d in domains)
            self._json_response({
                "domains":          domains,
                "total_attempts":   overall_attempts,
                "overall_solve_rate": round(overall_solved / max(overall_attempts, 1), 3),
                "active_domains":   len([d for d in domains if d["attempts"] > 0]),
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_intelligence_learning_progress(self):
        """GET /api/intelligence/learning-progress — understanding vs memorization report."""
        try:
            from sare.meta.learning_progress_report import generate_learning_progress_report
            report = generate_learning_progress_report()
            self._json_response(report)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_intelligence_learning_dashboard(self):
        """GET /api/intelligence/learning-dashboard — bundled payload for learning ops UI."""
        import json as _json
        import subprocess as _subprocess
        from pathlib import Path as _Path

        try:
            from sare import brain as _brain_module
            from sare.meta import learning_progress_report as _lpr

            report = _lpr.generate_learning_progress_report()

            # Append snapshot to history (throttled: at most once per 5 min)
            history_path = getattr(_lpr, "DEFAULT_HISTORY_PATH", None)
            if history_path:
                try:
                    _hp = _Path(history_path)
                    _should_append = True
                    if _hp.exists():
                        _hist_data = _json.loads(_hp.read_text(encoding="utf-8"))
                        if isinstance(_hist_data, list) and _hist_data:
                            import time as _time_mod
                            _last_snap = _hist_data[-1]
                            _last_ts_str = _last_snap.get("timestamp", "")
                            try:
                                from datetime import datetime, timezone
                                _last_dt = datetime.fromisoformat(_last_ts_str.replace("Z", "+00:00"))
                                _age_s = (datetime.now(timezone.utc) - _last_dt).total_seconds()
                                _should_append = _age_s >= 300  # 5 minutes
                            except Exception:
                                _should_append = True
                    if _should_append and report.get("snapshot"):
                        _lpr.append_history(_Path(history_path), report["snapshot"])
                except Exception:
                    pass

            history = []
            if history_path and _Path(history_path).exists():
                try:
                    payload = _json.loads(_Path(history_path).read_text(encoding="utf-8"))
                    if isinstance(payload, list):
                        history = payload[-96:]
                except Exception:
                    history = []

            brain = getattr(_brain_module, "_brain", None)
            autolearn = {"running": False, "resident": bool(brain)}
            if brain:
                autolearn = brain.auto_learn_status()
                autolearn = dict(autolearn)
                autolearn["resident"] = True
                autolearn["stage"] = brain.stage.value
                if brain.developmental_curriculum:
                    cmap = brain.developmental_curriculum.get_curriculum_map()
                    autolearn["mastered"] = cmap.get("mastered", 0)
                    autolearn["unlocked"] = cmap.get("unlocked", 0)
                    autolearn["total_domains"] = cmap.get("total_domains", 0)

            trainer = {"running": False, "total_problems": 0}
            trainer_stats_path = _Path(__file__).resolve().parents[3] / "data" / "memory" / "autonomous_trainer_stats.json"
            if brain and getattr(brain, "autonomous_trainer", None) and brain.autonomous_trainer._running:
                trainer = brain.autonomous_trainer.summary()
            elif trainer_stats_path.exists():
                try:
                    trainer = _json.loads(trainer_stats_path.read_text(encoding="utf-8"))
                    if isinstance(trainer, dict):
                        trainer["from_disk"] = True
                        trainer["running"] = False
                except Exception:
                    trainer = {"running": False, "total_problems": 0}

            daemon = {"running": False, "turbo": False, "pid": None, "mode": "normal"}
            try:
                result = _subprocess.run(
                    ["pgrep", "-f", "learn_daemon.py"],
                    capture_output=True,
                    text=True,
                )
                daemon["running"] = result.returncode == 0
                if daemon["running"]:
                    try:
                        turbo_result = _subprocess.run(
                            ["pgrep", "-f", "learn_daemon.py.*--turbo"],
                            capture_output=True,
                            text=True,
                        )
                        daemon["turbo"] = turbo_result.returncode == 0
                    except Exception:
                        daemon["turbo"] = False
                    lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
                    daemon["pid"] = lines[0] if lines else None
                    daemon["mode"] = "turbo" if daemon["turbo"] else "normal"
            except Exception:
                pass

            self._json_response(
                {
                    "generated_at": report.get("generated_at"),
                    "report": report,
                    "history": history,
                    "autolearn": autolearn,
                    "trainer": trainer,
                    "daemon": daemon,
                }
            )
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_neural_perception_stats(self):
        """GET /api/perception/neural — NeuralPerception stats and recent history."""
        try:
            from sare.perception.neural_perception import get_neural_perception
            np_inst = get_neural_perception()
            stats = np_inst.get_stats()
            recent = list(reversed(np_inst._history[-10:])) if np_inst._history else []
            self._json_response({
                "status": "ok",
                "stats": stats,
                "concepts": np_inst._concept_list,
                "recent_perceptions": recent,
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_science_hypothesis(self):
        """GET /api/science/hypothesis — propose new hypotheses from world model surprises."""
        try:
            from sare.cognition.hypothesis_maker import get_hypothesis_maker
            proposals = get_hypothesis_maker().propose(max_proposals=5)
            normalized = []
            for p in proposals:
                if isinstance(p, dict):
                    normalized.append(p)
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    normalized.append({
                        "subject": str(p[0]),
                        "predicate": str(p[1]),
                        "value": str(p[2]) if len(p) > 2 else "",
                    })
                else:
                    normalized.append({"hypothesis": str(p)})
            self._json_response({"hypotheses": normalized, "count": len(normalized)})
        except ImportError:
            self._json_response({"hypotheses": [], "count": 0, "note": "HypothesisMaker not available"})
        except Exception as e:
            self._json_response({"hypotheses": [], "count": 0, "error": str(e)})

    def _api_science_theory(self):
        """GET /api/science/theory — aggregate hypotheses into mini-theories by domain."""
        try:
            from sare.cognition.theory_builder import get_theory_builder
            theories = get_theory_builder().build_theories(max_theories=5)
            self._json_response({"theories": theories, "count": len(theories)})
        except Exception as e:
            self._json_response({"theories": [], "count": 0, "error": str(e)})

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

    _ensure_hippocampus_started()
    _ensure_self_improver_started()

    # ── Install evolver log buffer early so no messages are missed ────────
    try:
        from sare.meta.evolver_chat import get_log_buffer
        import logging as _logging
        _buf = get_log_buffer()
        # Attach to root logger so ALL sare.* and llm_bridge logs are captured
        _root = _logging.getLogger()
        if _buf not in _root.handlers:
            _root.addHandler(_buf)
        _logging.getLogger(__name__).info("[EvolverChat] Log buffer installed ✓")
    except Exception as _e:
        pass

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    # ── Daemon watchdog: auto-restart if stuck for 15 minutes ─────────────
    import threading as _threading
    import subprocess as _subprocess
    import os as _wd_os

    _REPO_ROOT_WD = str(__file__).split("/python/")[0]
    _HB_FILE_WD   = _wd_os.path.join(_REPO_ROOT_WD, "data", "memory", "daemon_heartbeat.json")
    _DAEMON_PID   = _wd_os.path.join(_REPO_ROOT_WD, ".pid_daemon")
    _STUCK_THRESH = 15 * 60   # 15 minutes

    def _daemon_is_running():
        try:
            pid = int(open(_DAEMON_PID).read().strip())
            _wd_os.kill(pid, 0)
            return True
        except Exception:
            return False

    def _restart_daemon():
        import logging as _wdlog
        _wdlog.getLogger("watchdog").warning(
            "[Watchdog] Daemon appears stuck (no heartbeat for >15 min). Restarting…"
        )
        # Kill old process if any
        try:
            pid = int(open(_DAEMON_PID).read().strip())
            _wd_os.kill(pid, 15)   # SIGTERM
            import time as _wt; _wt.sleep(3)
            try: _wd_os.kill(pid, 9)
            except Exception: pass
        except Exception:
            pass
        # Start new daemon
        _subprocess.Popen(
            ["python3", "learn_daemon.py", "--verbose"],
            cwd=_REPO_ROOT_WD,
            stdout=open("/tmp/sare_daemon.log", "a"),
            stderr=_subprocess.STDOUT,
            start_new_session=True,
        )
        _wdlog.getLogger("watchdog").info("[Watchdog] Daemon restarted.")

    def _watchdog_loop():
        import time as _wt
        import json as _wj
        _wt.sleep(120)  # wait 2 min before first check (allow daemon startup)
        while True:
            try:
                if _daemon_is_running():
                    try:
                        hb = _wj.loads(open(_HB_FILE_WD).read())
                        age = _wt.time() - hb.get("ts", 0)
                        if age > _STUCK_THRESH:
                            _restart_daemon()
                    except FileNotFoundError:
                        pass   # heartbeat not written yet — daemon just started
                    except Exception:
                        pass
            except Exception:
                pass
            _wt.sleep(60)   # check every minute

    _wd_thread = _threading.Thread(target=_watchdog_loop, daemon=True, name="daemon-watchdog")
    _wd_thread.start()

    server = ThreadedHTTPServer(("localhost", port), SareAPIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SARE-HX Web GUI")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    args = parser.parse_args()
    run_server(args.port)
