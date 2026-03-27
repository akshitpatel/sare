"""
Brain Orchestrator — The Central Nervous System of SARE-HX

Replaces the ad-hoc try/except import soup in web.py with a clean,
event-driven architecture that wires all cognitive subsystems together.

The Brain manages:
  1. Module lifecycle (init, load, save, shutdown)
  2. Event bus (SOLVE_COMPLETED, RULE_DISCOVERED, DOMAIN_MASTERED, etc.)
  3. Cognitive loop: Perceive → Plan → Act → Reflect → Learn
  4. Developmental stage tracking (infant → child → student → researcher)

Usage:
    brain = Brain()
    brain.boot()
    result = brain.solve("x + 0")
    brain.learn_cycle(n=10)
    brain.shutdown()
"""

from __future__ import annotations

import json
import logging
import time
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from sare.sare_logging.logger import SareLogger, SolveLog

log = logging.getLogger("sare.brain")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "memory"
CONFIGS_DIR = REPO_ROOT / "configs"
SOLVE_LOG_PATH = REPO_ROOT / "logs" / "solves.jsonl"

try:
    import sare.sare_bindings as _sb  # type: ignore
except Exception:
    _sb = None

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


_DEFAULT_BRAIN_CONFIG = {
    "phases": {
        "graph": True,
        "energy": True,
        "transforms": True,
        "search": True,
        "verification": True,
        "memory": True,
        "heuristics": True,
        "abstraction": True,
        "plasticity": True,
        "causal": True,
        "reflection": True,
        "curiosity": True,
        "curriculum": True,
    },
    "brain_boot": {
        "memory": True,
        "metacognition": True,
        "knowledge": True,
        "world_model": True,
        "perception": True,
        "language": True,
        "social": True,
        "learning": True,
        "curriculum": True,
        "warmup_environment_discovery": True,
        "warmup_physics_session": True,
        "warmup_knowledge_ingestion": True,
        "warmup_transform_generator": True,
        "warmup_action_physics": True,
        "warmup_robustness_batch": True,
        "autostart_continuous_stream": True,
        "autostart_hippocampus": True,
        "seed_perception_priors": True,
    },
}


def _merge_config(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_config(base[key], value)
        else:
            base[key] = value
    return base


def _py_graph_to_cpp_graph(py_graph):
    if not _sb or not getattr(_sb, "Graph", None):
        raise RuntimeError("C++ bindings unavailable")

    g = _sb.Graph()
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
                    "^": "pow", "**": "pow", "pow": "pow",
                    "=": "eq", "eq": "eq",
                    "neg": "neg",
                    "and": "and",
                    "or": "or",
                    "not": "not",
                }
                mapped = op_map.get(label)
                if mapped:
                    cn.set_attribute("op", mapped)

    for e in py_graph.edges:
        g.add_edge_with_id(int(e.id), int(e.source), int(e.target), str(e.relationship_type), 1.0)
    return g


def _cpp_graph_to_py_graph(cpp_graph):
    from sare.engine import Graph

    g = Graph()
    id_map = {}
    op_to_label = {
        "add": "+",
        "mul": "*",
        "sub": "-",
        "div": "/",
        "pow": "^",
        "eq": "=",
        "neg": "neg",
        "and": "and",
        "or": "or",
        "not": "not",
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


def _graph_features(graph) -> Tuple[List[str], List[Tuple[int, int]]]:
    nodes = graph.nodes
    id_to_idx = {n.id: idx for idx, n in enumerate(nodes)}
    node_types = [n.type for n in nodes]
    adjacency: List[Tuple[int, int]] = []
    for edge in graph.edges:
        src = id_to_idx.get(edge.source)
        tgt = id_to_idx.get(edge.target)
        if src is not None and tgt is not None:
            adjacency.append((src, tgt))
    return node_types, adjacency


class _MinimalConceptRegistry:
    """Fallback Python concept registry when C++ bindings unavailable."""

    def __init__(self):
        self._rules: List[dict] = []

    def add_rule(self, rule):
        if isinstance(rule, dict):
            self._rules.append(rule)
        elif hasattr(rule, '__dict__'):
            self._rules.append(rule.__dict__)

    def get_rules(self):
        return self._rules

    def get_consolidated_rules(self, min_confidence=0.8):
        return [r for r in self._rules if r.get("confidence", 0) >= min_confidence]

    def save(self):
        path = DATA_DIR / "concept_registry_py.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._rules, f, indent=2, default=str)

    def load(self):
        path = DATA_DIR / "concept_registry_py.json"
        if path.exists():
            try:
                with open(path) as f:
                    self._rules = json.load(f)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Event System
# ═══════════════════════════════════════════════════════════════════════════════

class Event(str, Enum):
    # Solve events
    SOLVE_STARTED = "solve_started"
    SOLVE_COMPLETED = "solve_completed"
    SOLVE_FAILED = "solve_failed"

    # Learning events
    RULE_DISCOVERED = "rule_discovered"
    RULE_PROMOTED = "rule_promoted"
    RULE_DEMOTED = "rule_demoted"
    MACRO_CREATED = "macro_created"

    # Curriculum events
    DOMAIN_UNLOCKED = "domain_unlocked"
    DOMAIN_MASTERED = "domain_mastered"
    STAGE_ADVANCED = "stage_advanced"

    # Memory events
    EPISODE_STORED = "episode_stored"
    STRATEGY_UPDATED = "strategy_updated"
    MEMORY_CONSOLIDATED = "memory_consolidated"

    # Metacognition events
    GOAL_ACHIEVED = "goal_achieved"
    COMPETENCE_UPDATED = "competence_updated"
    SURPRISE_DETECTED = "surprise_detected"

    # Transfer events
    ANALOGY_FOUND = "analogy_found"
    TRANSFER_ATTEMPTED = "transfer_attempted"
    TRANSFER_SUCCEEDED = "transfer_succeeded"

    # Creativity events
    CREATIVITY_SPARK = "creativity_spark"
    CONJECTURE_VERIFIED = "conjecture_verified"

    # System events
    BRAIN_BOOTED = "brain_booted"
    BRAIN_SHUTDOWN = "brain_shutdown"
    SLEEP_STARTED = "sleep_started"
    SLEEP_ENDED = "sleep_ended"


@dataclass
class EventData:
    """Payload for an event."""
    event: Event
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""


class EventBus:
    """Simple synchronous event bus for inter-module communication."""

    def __init__(self):
        self._listeners: Dict[Event, List[Callable[[EventData], None]]] = {}
        self._history: List[EventData] = []
        self._max_history = 1000

    def subscribe(self, event: Event, callback: Callable[[EventData], None]):
        self._listeners.setdefault(event, []).append(callback)

    def emit(self, event: Event, data: Dict[str, Any] = None, source: str = ""):
        ed = EventData(event=event, data=data or {}, source=source)
        self._history.append(ed)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        for cb in self._listeners.get(event, []):
            try:
                cb(ed)
            except Exception as e:
                log.error(f"Event handler error for {event}: {e}")

    def recent(self, n: int = 50, event_type: Event = None) -> List[EventData]:
        if event_type:
            filtered = [e for e in self._history if e.event == event_type]
            return filtered[-n:]
        return self._history[-n:]


# ═══════════════════════════════════════════════════════════════════════════════
#  Developmental Stages
# ═══════════════════════════════════════════════════════════════════════════════

class DevelopmentalStage(str, Enum):
    INFANT = "infant"           # 0: basic pattern matching, identity rules
    TODDLER = "toddler"        # 1: simple arithmetic, logic basics
    CHILD = "child"             # 2: multi-step simplification, equation solving
    PRETEEN = "preteen"         # 3: algebraic manipulation, proof basics
    TEENAGER = "teenager"       # 4: cross-domain transfer, metacognition
    UNDERGRADUATE = "undergrad" # 5: domain specialization, formal proofs
    GRADUATE = "graduate"       # 6: novel conjectures, teaching others
    RESEARCHER = "researcher"   # 7: paradigm creation, self-improvement

    @property
    def level(self) -> int:
        return list(DevelopmentalStage).index(self)


STAGE_REQUIREMENTS = {
    DevelopmentalStage.INFANT: {"min_rules": 0, "min_domains": 0, "min_solve_rate": 0.0},
    DevelopmentalStage.TODDLER: {"min_rules": 3, "min_domains": 1, "min_solve_rate": 0.3},
    DevelopmentalStage.CHILD: {"min_rules": 8, "min_domains": 2, "min_solve_rate": 0.5},
    DevelopmentalStage.PRETEEN: {"min_rules": 20, "min_domains": 3, "min_solve_rate": 0.65},
    DevelopmentalStage.TEENAGER: {"min_rules": 40, "min_domains": 5, "min_solve_rate": 0.75},
    DevelopmentalStage.UNDERGRADUATE: {"min_rules": 80, "min_domains": 8, "min_solve_rate": 0.85},
    DevelopmentalStage.GRADUATE: {"min_rules": 150, "min_domains": 12, "min_solve_rate": 0.90},
    DevelopmentalStage.RESEARCHER: {"min_rules": 300, "min_domains": 15, "min_solve_rate": 0.95},
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Brain — The Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class Brain:
    """
    Central orchestrator for the SARE-HX cognitive architecture.

    Manages all subsystem lifecycle, event routing, and the cognitive loop.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = self._load_config(config)
        self.phase_flags = dict(self.config.get("phases", {}))
        self.boot_flags = dict(self.config.get("brain_boot", {}))

        # Event bus
        self.events = EventBus()

        # Developmental state
        self.stage = DevelopmentalStage.INFANT
        self.boot_time: float = 0.0
        self.total_solves: int = 0
        self.total_rules_learned: int = 0
        self.total_transfers: int = 0

        # Core engine modules (always available)
        self.engine = None       # sare.engine module
        self.energy = None       # EnergyEvaluator
        self.searcher = None     # BeamSearch
        self.transforms = []     # List[Transform]
        self.cpp_enabled = False
        self.cpp_bindings_available = False
        self.cpp_run_beam_search = None
        self.cpp_run_mcts_search = None
        self.cpp_search_config_cls = None
        self.cpp_reflection_engine = None
        self.py_reflection_engine = None
        self.cpp_module_generator = None

        # Memory modules
        self.memory_manager = None
        self.episodic_store = None
        self.concept_memory = None
        self.working_memory = None      # P4.2 working memory + attention
        self.knowledge_graph = None     # P4.4 unified knowledge graph

        # Learning modules
        self.reflection_engine = None
        self.causal_induction = None
        self.concept_registry = None
        self.abstraction_registry = None

        # Metacognition modules
        self.self_model = None
        self.goal_setter = None
        self.homeostasis = None
        self.frontier_manager = None
        self.credit_assigner = None

        # Curiosity modules
        self.curriculum_gen = None
        self.experiment_runner = None

        # World model
        self.world_model = None
        self.world_model_v3 = None

        # Transfer engine
        self.transfer_engine = None
        self.transform_synthesizer = None

        # Perception
        self.perception_engine = None

        # Social modules
        self.dialogue_manager = None
        self.theory_of_mind = None
        self.identity = None
        self.autobiography = None

        # Knowledge
        self.commonsense = None

        # Concept Layer (perception → concepts → symbols → reasoning)
        self.concept_graph = None
        self.environment_simulator = None
        self.goal_planner = None

        # Self-Model + Meta-Learning Engine (recursive self-improvement)
        self.meta_learner = None

        # 5 Major Upgrades (Session 23)
        self.physics_simulator = None    # Upgrade 1: True World Simulation
        self.knowledge_ingester = None   # Upgrade 2: Massive Knowledge Base
        self.multi_agent_arena = None    # Upgrade 3: Multi-Agent Learning
        self.multi_modal_parser = None   # Upgrade 4: Multi-Modal Perception
        self.global_workspace = None     # Upgrade 5 / The Key Change: Global Workspace

        # 3 Gap Closers (Session 24)
        self.predictive_loop = None      # Gap 1: Sensorimotor predict→act→observe→update
        self.autonomous_trainer = None   # Gap 2: Continuous 24/7 background learning
        self.agent_society = None        # Gap 3: Rich multi-agent with beliefs + goals
        self.agent_negotiator = None     # Gap 4: Agent Negotiation Arena

        # 4 Gap Closers (Session 25)
        self.global_buffer = None        # S25-1: Cross-session broadcast working memory
        self.concept_blender = None      # S25-2: Cross-domain novel concept synthesis
        self.dialogue_context = None     # S25-3: Multi-turn conversation tracker
        self.sensory_bridge = None       # S25-4: Physics-grounded sensorimotor observations

        # 6 Crazy Gap Closers (Session 26)
        self.dream_consolidator = None   # S26-1: Offline hippocampal replay
        self.affective_energy = None     # S26-2: Multi-component curiosity-driven energy
        self.transform_generator = None  # S26-3: Self-modifying transform synthesis
        self.generative_world = None     # S26-4: Imagination engine — novel problem sampler
        self.red_team = None             # S26-5: Internal adversary — belief falsification
        self.temporal_identity = None    # S26-6: Persistent self across sessions
        self.continuous_stream = None    # S27-1: Parallel async streams + interference shield

        # Session 28 — Close 4 remaining gaps
        self.robustness_hardener = None  # S28-1: Systematic adversarial stress testing
        self.attention_router    = None  # S28-2: Deep Global Workspace attention routing
        self.recursive_tom       = None  # S28-3: Recursive Theory of Mind (depth 3)
        self.agent_memory_bank   = None  # S28-4: Persistent cross-session agent memory

        # Session 29 — Close 4 final gaps
        self.meta_curriculum   = None   # S29-1: Meta-level domain discovery + transfer testing
        self.action_physics    = None   # S29-2: Multi-step physics with action consequences
        self.stream_bridge     = None   # S29-3: Cross-stream EXPLORE→IMAGINE→EXPLOIT→CURRICULUM
        self.perception_bridge = None   # S29-4: Scene description → symbolic objects + relations

        # AGI Leap — Cross-domain cognition
        self.analogy_transfer  = None   # AGI-1: Cross-domain structural analogy transfer
        self.conjecture_engine = None   # AGI-2: Proactive hypothesis generator

        # Language
        self.nl_parser = None
        self.llm_bridge = None
        self.language_grounding = None

        # Infrastructure
        self.hippocampus = None
        self.proof_builder = None

        # Curriculum
        self.developmental_curriculum = None

        # Track module load status
        self._module_status: Dict[str, str] = {}

        # Statistics
        self._stats = {
            "solves_attempted": 0,
            "solves_succeeded": 0,
            "rules_discovered": 0,
            "rules_promoted": 0,
            "transfers_attempted": 0,
            "transfers_succeeded": 0,
            "domains_mastered": [],
            "sleep_cycles": 0,
            "runtime_errors": 0,
        }
        self._diagnostics: List[dict] = []
        self._max_diagnostics = 200
        self._last_problem_selection: Optional[dict] = None

    @staticmethod
    def _load_config(config: Optional[dict] = None) -> dict:
        merged = deepcopy(_DEFAULT_BRAIN_CONFIG)
        path = CONFIGS_DIR / "default.yaml"
        if yaml and path.exists():
            try:
                payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                if isinstance(payload, dict):
                    _merge_config(merged, payload)
            except Exception as exc:
                log.debug("Brain config load skipped: %s", exc)
        if config:
            _merge_config(merged, config)
        return merged

    def _phase_enabled(self, name: str, default: bool = True) -> bool:
        return bool(self.phase_flags.get(name, default))

    def _boot_enabled(self, name: str, default: bool = True) -> bool:
        return bool(self.boot_flags.get(name, default))

    def _mark_skipped(self, *module_names: str, reason: str = "disabled by config") -> None:
        for module_name in module_names:
            self._module_status[module_name] = f"⏭️ {reason}"

    # ─────────────────────────────────────────────────────────────────────────
    #  Boot Sequence
    # ─────────────────────────────────────────────────────────────────────────

    def boot(self):
        """Initialize all subsystems in dependency order."""
        self.boot_time = time.time()
        log.info("╔══════════════════════════════════════════╗")
        log.info("║       SARE-HX Brain Booting...           ║")
        log.info("╚══════════════════════════════════════════╝")

        # Layer 0: Core engine (must succeed)
        self._boot_core()

        # Layer 1: Memory
        if self._boot_enabled("memory", True):
            self._boot_memory()
        else:
            self._mark_skipped(
                "memory_manager", "concept_memory", "working_memory", "knowledge_graph",
                reason="memory boot disabled",
            )

        # Layer 2: Metacognition
        if self._boot_enabled("metacognition", True):
            self._boot_metacognition()
        else:
            self._mark_skipped(
                "self_model", "goal_setter", "homeostasis", "frontier_manager",
                "credit_assigner", "proof_builder",
                reason="metacognition boot disabled",
            )

        # Layer 3: Knowledge & Concepts
        if self._boot_enabled("knowledge", True):
            self._boot_knowledge()
        else:
            self._mark_skipped(
                "concept_registry", "commonsense", "concept_graph", "environment_simulator",
                "goal_planner", "meta_learner", "physics_simulator", "knowledge_ingester",
                "multi_agent_arena", "multi_modal_parser", "global_workspace",
                "predictive_loop", "autonomous_trainer", "agent_society", "global_buffer",
                "concept_blender", "dialogue_context", "sensory_bridge", "dream_consolidator",
                "affective_energy", "transform_generator", "generative_world", "red_team",
                "temporal_identity", "continuous_stream", "robustness_hardener",
                "attention_router", "recursive_tom", "agent_memory_bank", "meta_curriculum",
                "action_physics", "stream_bridge", "perception_bridge",
                reason="knowledge boot disabled",
            )

        # Layer 4: World Model & Reasoning
        if self._boot_enabled("world_model", True):
            self._boot_world_model()
        else:
            self._mark_skipped(
                "world_model", "world_model_v3", "transfer_engine", "py_reflection_engine",
                "cpp_reflection_engine", "causal_induction", "cpp_module_generator",
                reason="world_model boot disabled",
            )

        # Layer 5: Perception
        if self._boot_enabled("perception", True):
            self._boot_perception()
        else:
            self._mark_skipped(
                "perception_engine", "transform_synthesizer",
                reason="perception boot disabled",
            )

        # Seed transfer synthesizer (needs both transfer_engine + transform_synthesizer)
        self._seed_synthesizer()

        # Layer 5b: Language
        if self._boot_enabled("language", True):
            self._boot_language()
            self._boot_language_grounding()
        else:
            self._mark_skipped("nl_parser", "llm_bridge", "language_grounding", reason="language boot disabled")

        # Layer 6: Social
        if self._boot_enabled("social", True):
            self._boot_social()
        else:
            self._mark_skipped("dialogue_manager", "theory_of_mind", "identity", "autobiography", reason="social boot disabled")

        # Layer 7: Learning & Curiosity
        if self._boot_enabled("learning", True):
            self._boot_learning()
        else:
            self._mark_skipped("curriculum_gen", "experiment_runner", "hippocampus", reason="learning boot disabled")

        # Layer 8: Curriculum
        if self._boot_enabled("curriculum", True):
            self._boot_curriculum()
        else:
            self._mark_skipped("dev_curriculum", reason="curriculum boot disabled")

        # Wire event handlers
        self._wire_events()

        # Load persisted state
        self._load_state()

        # Determine developmental stage
        self._update_stage()

        elapsed = time.time() - self.boot_time
        log.info(f"Brain booted in {elapsed:.1f}s | Stage: {self.stage.value}")
        log.info(f"Module status: {self._module_summary()}")
        self.events.emit(Event.BRAIN_BOOTED, {"stage": self.stage.value, "elapsed": elapsed})

    def _load_module(self, name: str, loader: Callable) -> Any:
        """Safely load a module, recording status."""
        try:
            result = loader()
            self._module_status[name] = "✅"
            return result
        except Exception as e:
            self._module_status[name] = f"❌ {e}"
            self._record_runtime_error(f"boot.{name}", e, "boot")
            log.warning(f"Module {name} failed: {e}")
            return None

    def _module_summary(self) -> str:
        ok = sum(1 for v in self._module_status.values() if v.startswith("✅"))
        total = len(self._module_status)
        return f"{ok}/{total} modules loaded"

    def _load_problem_graph(self, expression: str):
        """Best-effort expression -> Graph loader for background modules."""
        try:
            engine_mod = self.engine
            if engine_mod and hasattr(engine_mod, "load_problem"):
                _, graph = engine_mod.load_problem(expression)
                return graph
        except Exception as exc:
            log.debug("Brain problem loader failed for %s: %s", expression, exc)
        return None

    @staticmethod
    def _graph_preview(graph) -> str:
        if graph is None:
            return ""
        if hasattr(graph, "to_dict"):
            try:
                data = graph.to_dict()
                labels = [n.get("label") or n.get("type", "") for n in data.get("nodes", [])[:6]]
                return " ".join(filter(None, labels))
            except Exception:
                pass
        if hasattr(graph, "pretty_print"):
            try:
                return graph.pretty_print().splitlines()[0][:120]
            except Exception:
                pass
        return str(graph)[:120]

    def _record_runtime_error(self, component: str, exc: Exception, context: str = "") -> None:
        entry = {
            "time": time.time(),
            "component": component,
            "error": str(exc),
            "context": context,
        }
        self._diagnostics.append(entry)
        if len(self._diagnostics) > self._max_diagnostics:
            self._diagnostics = self._diagnostics[-self._max_diagnostics:]
        self._stats["runtime_errors"] = self._stats.get("runtime_errors", 0) + 1
        log.debug("%s failed%s: %s", component, f" ({context})" if context else "", exc)

    @staticmethod
    def _serialize_recalled_memory(similar: List[dict]) -> List[dict]:
        recalled_memory = []
        for match in similar:
            episode = match.get("episode", {}) or {}
            transforms = episode.get("transforms", []) or []
            if not transforms:
                continue
            recalled_memory.append({
                "similarity": round(float(match.get("similarity", 0.0)), 2),
                "problem_id": str(episode.get("problem_id", "")),
                "transforms": list(transforms[:5]),
            })
        return recalled_memory

    def _persist_post_solve_state(self) -> None:
        for name, component in (
            ("self_model", self.self_model),
            ("frontier_manager", self.frontier_manager),
            ("credit_assigner", self.credit_assigner),
        ):
            saver = getattr(component, "save", None)
            if not callable(saver):
                continue
            try:
                saver()
            except Exception as exc:
                self._record_runtime_error(f"{name}.save", exc, "post_solve")

    def _boot_core(self):
        """Boot core engine — this must succeed."""
        from sare.engine import (
            EnergyEvaluator, BeamSearch, get_transforms, build_expression_graph
        )
        import sare.engine as engine_mod

        self.engine = engine_mod
        self.energy = EnergyEvaluator()
        self.searcher = BeamSearch()
        self.transforms = get_transforms(include_macros=True)
        self.cpp_run_beam_search = getattr(_sb, "run_beam_search", None) if _sb else None
        self.cpp_run_mcts_search = getattr(_sb, "run_mcts_search", None) if _sb else None
        self.cpp_search_config_cls = getattr(_sb, "SearchConfig", None) if _sb else None
        self.cpp_bindings_available = bool(
            _sb and self.cpp_run_beam_search and self.cpp_run_mcts_search and self.cpp_search_config_cls
        )
        self.cpp_enabled = False
        self._module_status["engine"] = "✅"
        self._module_status["energy"] = "✅"
        self._module_status["search"] = "✅"
        self._module_status["cpp_core"] = "✅" if self.cpp_bindings_available else "❌ unavailable"
        self._module_status["transforms"] = f"✅ ({len(self.transforms)} loaded)"
        log.info(f"Core engine: {len(self.transforms)} transforms loaded")

    def _boot_memory(self):
        if not self._phase_enabled("memory", True):
            self._mark_skipped(
                "memory_manager", "concept_memory", "working_memory", "knowledge_graph",
                reason="memory phase disabled",
            )
            return

        def load_mm():
            from sare.memory.memory_manager import MemoryManager
            mm = MemoryManager()
            mm.load()
            return mm
        self.memory_manager = self._load_module("memory_manager", load_mm)

        def load_cm():
            from sare.memory.concept_formation import ConceptMemory
            cm = ConceptMemory()
            cm.load()
            return cm
        self.concept_memory = self._load_module("concept_memory", load_cm)

        def load_wm():
            from sare.memory.working_memory import WorkingMemory
            return WorkingMemory()
        self.working_memory = self._load_module("working_memory", load_wm)

        def load_kg():
            from sare.memory.knowledge_graph import KnowledgeGraph
            return KnowledgeGraph()
        self.knowledge_graph = self._load_module("knowledge_graph", load_kg)

    def _boot_metacognition(self):
        def load_sm():
            from sare.meta.self_model import SelfModel
            sm = SelfModel()
            sm.load()
            return sm
        self.self_model = self._load_module("self_model", load_sm)

        def load_gs():
            from sare.meta.goal_setter import GoalSetter
            gs = GoalSetter()
            gs.load()
            return gs
        self.goal_setter = self._load_module("goal_setter", load_gs)

        def load_ho():
            from sare.meta.homeostasis import HomeostaticSystem
            hs = HomeostaticSystem()
            hs.load()
            return hs
        self.homeostasis = self._load_module("homeostasis", load_ho)

        def load_fm():
            from sare.curiosity.frontier_manager import FrontierManager
            fm = FrontierManager()
            fm.load()
            return fm
        self.frontier_manager = self._load_module("frontier_manager", load_fm)

        def load_ca():
            from sare.learning.credit_assignment import CreditAssigner
            ca = CreditAssigner()
            if hasattr(ca, 'load'):
                ca.load()
            return ca
        self.credit_assigner = self._load_module("credit_assigner", load_ca)

        def load_pb():
            from sare.meta.proof_builder import ProofBuilder
            return ProofBuilder()
        self.proof_builder = self._load_module("proof_builder", load_pb)

    def _boot_knowledge(self):
        abstraction_enabled = self._phase_enabled("abstraction", True)

        def load_cr():
            # Try C++ ConceptRegistry first, fall back to None
            try:
                import sare.sare_bindings as sb
                CR = getattr(sb, 'ConceptRegistry', None)
                if CR:
                    cr = CR()
                    # Try to load seeds
                    try:
                        from sare.memory.concept_seed_loader import load_seeds
                        load_seeds(cr)
                    except Exception:
                        pass
                    return cr
            except Exception:
                pass
            # Return a minimal Python registry
            cr = _MinimalConceptRegistry()
            cr.load()
            return cr
        if abstraction_enabled:
            self.concept_registry = self._load_module("concept_registry", load_cr)
        else:
            self._mark_skipped("concept_registry", reason="abstraction phase disabled")

        # P1.4: Reload high-confidence rules as live transforms immediately
        if self.concept_registry and self.transforms:
            try:
                rules = self.concept_registry.get_consolidated_rules(min_confidence=0.7)
                loaded = 0
                for rule in rules:
                    t = self._rule_to_transform(rule) if hasattr(self, '_rule_to_transform') else None
                    if t:
                        self.transforms.append(t)
                        loaded += 1
                if loaded:
                    log.info(f"Boot: reloaded {loaded} high-confidence rules as live transforms")
            except Exception as _e:
                log.debug(f"Rule reload skipped: {_e}")

        def load_cs():
            from sare.knowledge.commonsense import CommonSenseBase
            cs = CommonSenseBase()
            cs.load()
            if cs.total_facts() == 0:
                cs.seed()
            return cs
        self.commonsense = self._load_module("commonsense", load_cs)

        # Concept Graph — the biological intelligence layer
        def load_cg():
            from sare.concept.concept_graph import ConceptGraph
            cg = ConceptGraph(seed=True)
            return cg
        if abstraction_enabled:
            self.concept_graph = self._load_module("concept_graph", load_cg)
        else:
            self._mark_skipped("concept_graph", reason="abstraction phase disabled")

        # Environment Simulator — grounded learning from observations
        def load_env():
            from sare.concept.environment import EnvironmentSimulator
            env = EnvironmentSimulator()
            if self._boot_enabled("warmup_environment_discovery", True):
                env.run_full_discovery_session()  # bootstrap with initial observations
            return env
        self.environment_simulator = self._load_module("environment_simulator", load_env)

        # Hierarchical Goal Planner
        def load_gp():
            from sare.concept.goal_planner import GoalPlanner
            gp = GoalPlanner()
            return gp
        if abstraction_enabled:
            self.goal_planner = self._load_module("goal_planner", load_gp)
        else:
            self._mark_skipped("goal_planner", reason="abstraction phase disabled")

        # Meta-Learning Engine — experiments with internal search configuration
        def load_ml():
            from sare.meta.meta_learner import MetaLearningEngine
            return MetaLearningEngine()
        self.meta_learner = self._load_module("meta_learner", load_ml)

        # ── Upgrade 1: PhysicsSimulator ──────────────────────────────────────────
        def load_physics():
            from sare.world.physics_simulator import PhysicsSimulator
            ps = PhysicsSimulator()
            if self._boot_enabled("warmup_physics_session", True):
                ps.run_session(5)
                if self.concept_graph:
                    ps.feed_to_concept_graph(self.concept_graph)
            return ps
        self.physics_simulator = self._load_module("physics_simulator", load_physics)

        # ── Upgrade 2: KnowledgeIngester ─────────────────────────────────────────
        def load_knowledge():
            from sare.knowledge.knowledge_ingester import KnowledgeIngester
            ki = KnowledgeIngester()
            if self._boot_enabled("warmup_knowledge_ingestion", True):
                ki.ingest_knowledge_base()
                if self.concept_graph:
                    ki.feed_to_concept_graph(self.concept_graph)
            return ki
        self.knowledge_ingester = self._load_module("knowledge_ingester", load_knowledge)

        # ── Upgrade 3: MultiAgentArena ───────────────────────────────────────────
        def load_arena():
            from sare.agent.multi_agent_arena import MultiAgentArena
            return MultiAgentArena(n_agents=3)
        self.multi_agent_arena = self._load_module("multi_agent_arena", load_arena)

        # ── Upgrade 4: MultiModalParser ──────────────────────────────────────────
        def load_mm_parser():
            from sare.interface.nl_parser_v2 import MultiModalParser
            return MultiModalParser()
        self.multi_modal_parser = self._load_module("multi_modal_parser", load_mm_parser)

        # ── Upgrade 5 (The Key Change): GlobalWorkspace ──────────────────────────
        def load_gw():
            from sare.memory.global_workspace import GlobalWorkspace
            gw = GlobalWorkspace()
            gw.wire_brain(self)
            return gw
        self.global_workspace = self._load_module("global_workspace", load_gw)

        # ── Gap 1: PredictiveWorldLoop ────────────────────────────────────────
        def load_pl():
            from sare.world.predictive_loop import PredictiveWorldLoop
            return PredictiveWorldLoop()
        self.predictive_loop = self._load_module("predictive_loop", load_pl)

        # ── Gap 2: AutonomousTrainer ──────────────────────────────────────────
        def load_at():
            from sare.learning.autonomous_trainer import AutonomousTrainer
            return AutonomousTrainer(interval_seconds=0.5, batch_size=20)
        self.autonomous_trainer = self._load_module("autonomous_trainer", load_at)

        # ── Gap 3: AgentSociety ───────────────────────────────────────────────
        def load_as():
            from sare.agent.agent_society import AgentSociety
            society = AgentSociety(n_agents=3)
            # Seed society with concept graph knowledge
            if self.concept_graph:
                for name, c in list(self.concept_graph._concepts.items())[:10]:
                    for rule in c.symbolic_rules[:2]:
                        for agent in society._agents.values():
                            if agent.specialization == c.domain or c.domain == "general":
                                agent.add_belief(rule, c.domain, 0.7, "concept_graph")
            return society
        self.agent_society = self._load_module("agent_society", load_as)

        # ── Gap 4: AgentNegotiator (Multi-Agent Arena) ───────────────────────
        def load_an():
            from sare.learning.agent_negotiator import NegotiationArena
            return NegotiationArena(self.energy)
        self.agent_negotiator = self._load_module("agent_negotiator", load_an)

        # ── S25-1: GlobalBuffer ───────────────────────────────────────────────
        def load_gb():
            from sare.memory.global_buffer import GlobalBuffer
            buf = GlobalBuffer(capacity=7)
            if self.global_workspace:
                self.global_workspace.subscribe("global_buffer", buf.receive)
            return buf
        self.global_buffer = self._load_module("global_buffer", load_gb)

        # ── S25-2: ConceptBlender ─────────────────────────────────────────────
        def load_cb():
            from sare.concept.concept_blender import ConceptBlender
            blender = ConceptBlender()
            if self.concept_graph:
                for name, c in list(self.concept_graph._concepts.items())[:10]:
                    blender.add_space(
                        concept=name,
                        domain=c.domain,
                        properties={"operation": "symbolic", "domain": c.domain,
                                    "rules": len(c.symbolic_rules)},
                        relations=c.symbolic_rules[:2],
                        examples=c.examples[:2] if hasattr(c, 'examples') else [],
                    )
            blender.discover_blends(max_new=5)
            return blender
        self.concept_blender = self._load_module("concept_blender", load_cb)

        # ── S25-3: DialogueContext ────────────────────────────────────────────
        def load_dc():
            from sare.interface.dialogue_context import DialogueContext
            return DialogueContext(window=8)
        self.dialogue_context = self._load_module("dialogue_context", load_dc)

        # ── S25-4: SensoryBridge ──────────────────────────────────────────────
        def load_sb():
            from sare.world.sensory_bridge import SensoryBridge
            bridge = SensoryBridge()
            bridge.wire(self.physics_simulator, self.predictive_loop)
            return bridge
        self.sensory_bridge = self._load_module("sensory_bridge", load_sb)

        # ── S26-1: DreamConsolidator ──────────────────────────────────────────
        def load_dream():
            from sare.learning.dream_consolidator import DreamConsolidator
            dc = DreamConsolidator()
            dc.wire(
                predictive_loop=self.predictive_loop,
                causal_graph=getattr(self, 'causal_graph', None),
                world_model=getattr(self, 'world_model', None),
            )
            return dc
        self.dream_consolidator = self._load_module("dream_consolidator", load_dream)

        # ── S26-2: AffectiveEnergy ────────────────────────────────────────────
        def load_affect():
            from sare.energy.affective_energy import AffectiveEnergy
            ae = AffectiveEnergy()
            if self.concept_graph:
                ae.calibrate_from_concepts(self.concept_graph)
            return ae
        self.affective_energy = self._load_module("affective_energy", load_affect)

        # ── S26-3: TransformGenerator ─────────────────────────────────────────
        def load_tgen():
            from sare.meta.transform_generator import TransformGenerator
            tg = TransformGenerator()
            tg.wire(self)  # pass Brain as engine proxy
            if self._boot_enabled("warmup_transform_generator", True):
                tg.generate_candidates(n=8)
                tg.promote_best()
            return tg
        self.transform_generator = self._load_module("transform_generator", load_tgen)

        # ── S26-4: GenerativeWorldModel ───────────────────────────────────────
        def load_genworld():
            from sare.world.generative_world import GenerativeWorldModel
            gw = GenerativeWorldModel()
            gw.wire(engine=self, affective_energy=self.affective_energy)
            return gw
        self.generative_world = self._load_module("generative_world", load_genworld)

        # ── S26-5: RedTeamAdversary ───────────────────────────────────────────
        def load_red():
            from sare.agent.red_team import RedTeamAdversary
            rt = RedTeamAdversary()
            rt.wire(agent_society=self.agent_society, engine=self)
            return rt
        self.red_team = self._load_module("red_team", load_red)

        # ── S26-6: TemporalIdentity ───────────────────────────────────────────
        def load_identity():
            from sare.meta.temporal_identity import TemporalIdentity
            return TemporalIdentity()
        self.temporal_identity = self._load_module("temporal_identity", load_identity)

        # ── S27-1: ContinuousStreamLearner ───────────────────────────────────────
        def load_stream():
            from sare.learning.continuous_stream import ContinuousStreamLearner
            csl = ContinuousStreamLearner()
            csl.wire(
                engine=self,
                affective_energy=self.affective_energy,
                generative_world=self.generative_world,
                trainer=getattr(self, 'autonomous_trainer', None),
            )
            if self._boot_enabled("autostart_continuous_stream", True):
                csl.start(n_streams=4)
            return csl
        self.continuous_stream = self._load_module("continuous_stream", load_stream)

        # ── S28-1: RobustnessHardener ─────────────────────────────────────
        def load_robust():
            from sare.meta.robustness_hardener import RobustnessHardener
            rh = RobustnessHardener()
            rh.wire(engine=self)
            if self._boot_enabled("warmup_robustness_batch", True):
                rh.run_stress_batch(n=8)  # warm-up
            return rh
        self.robustness_hardener = self._load_module("robustness_hardener", load_robust)

        # ── S28-2: AttentionRouter ───────────────────────────────────────
        def load_attn():
            from sare.memory.attention_router import AttentionRouter
            ar = AttentionRouter()
            ar.wire(global_workspace=self.global_workspace)
            # register downstream modules as targets
            for name, mod in [
                ("dream_consolidator", self.dream_consolidator),
                ("affective_energy",   self.affective_energy),
                ("transform_generator",self.transform_generator),
                ("agent_society",      self.agent_society),
                ("global_buffer",      self.global_buffer),
            ]:
                if mod and hasattr(mod, 'receive'):
                    ar.register_target(name, mod.receive)
            return ar
        self.attention_router = self._load_module("attention_router", load_attn)

        # ── S28-3: RecursiveToM ─────────────────────────────────────────
        def load_tom():
            from sare.agent.recursive_tom import RecursiveToM
            tom = RecursiveToM()
            tom.wire(agent_society=self.agent_society)
            return tom
        self.recursive_tom = self._load_module("recursive_tom", load_tom)

        # ── S28-4: AgentMemoryBank ──────────────────────────────────────
        def load_agentmem():
            from sare.agent.agent_memory import AgentMemoryBank
            amb = AgentMemoryBank()
            amb.wire(agent_society=self.agent_society)
            return amb
        self.agent_memory_bank = self._load_module("agent_memory_bank", load_agentmem)

        # ── S29-1: MetaCurriculumEngine ─────────────────────────────────────
        def load_meta_curr():
            from sare.learning.meta_curriculum import MetaCurriculumEngine
            mc = MetaCurriculumEngine()
            mc.wire(
                curriculum=self.curriculum_gen,
                engine=self,
                stream=self.continuous_stream,
            )
            return mc
        self.meta_curriculum = self._load_module("meta_curriculum", load_meta_curr)

        # ── S29-2: ActionPhysicsSession ──────────────────────────────────────
        def load_action_phys():
            from sare.world.action_physics import ActionPhysicsSession
            ap = ActionPhysicsSession()
            ap.wire(
                concept_graph=self.concept_graph,
                global_workspace=self.global_workspace,
            )
            if self._boot_enabled("warmup_action_physics", True):
                ap.run_episode(n_steps=12)  # warm-up
            return ap
        self.action_physics = self._load_module("action_physics", load_action_phys)

        # ── S29-3: StreamBridge ──────────────────────────────────────────────
        def load_stream_bridge():
            from sare.learning.stream_bridge import StreamBridge
            sb = StreamBridge()
            sb.wire(
                world_model=self.generative_world,
                affective=self.affective_energy,
                meta_curriculum=self.meta_curriculum,
                transform_gen=self.transform_generator,
                engine=self,
            )
            return sb
        self.stream_bridge = self._load_module("stream_bridge", load_stream_bridge)

        # ── S29-4: PerceptionBridge ──────────────────────────────────────────
        def load_percept():
            from sare.perception.perception_bridge import PerceptionBridge
            pb = PerceptionBridge()
            pb.wire(
                concept_graph=self.concept_graph,
                global_workspace=self.global_workspace,
            )
            # Seed with basic scene priors
            if self._boot_enabled("seed_perception_priors", True):
                for desc in [
                    "ball above table",
                    "block left of wall",
                    "friction slows block",
                    "gravity pulls ball below table",
                    "large cube supports small sphere",
                ]:
                    pb.parse_scene(desc)
            return pb
        self.perception_bridge = self._load_module("perception_bridge", load_percept)

        # Seed concept graph with environment observations
        if self.concept_graph and self.environment_simulator and self._boot_enabled("warmup_environment_discovery", True):
            try:
                sym_rules = self.environment_simulator.generate_symbolic_rules()
                for concept_name, symbolic, evidence in sym_rules:
                    if concept_name in self.concept_graph._concepts:
                        c = self.concept_graph._concepts[concept_name]
                        if symbolic not in c.symbolic_rules:
                            c.symbolic_rules.append(symbolic)
                obs_by_concept = self.environment_simulator.extract_concepts()
                for concept_name, obs_list in obs_by_concept.items():
                    for obs in obs_list[:3]:
                        self.concept_graph.ground_example(
                            concept_name=concept_name,
                            text=obs.description,
                            operation=obs.operation,
                            symbolic=obs.symbolic,
                            domain="arithmetic",
                        )
                log.info(f"Concept graph seeded: {self.concept_graph.summary()}")
            except Exception as _e:
                log.debug(f"Concept graph seeding: {_e}")

        # ── AGI-1: AnalogyTransfer ─────────────────────────────────────────────
        def load_analogy():
            from sare.causal.analogy_transfer import AnalogyTransfer
            at = AnalogyTransfer(concept_registry=self.concept_registry)
            # Immediately fire all known cross-domain transfers
            transfers = at.transfer_all_domains()
            applied, skipped = at.apply_to_registry(transfers)
            log.info(f"AnalogyTransfer: {applied} initial cross-domain rules applied ({skipped} skipped)")
            return at
        self.analogy_transfer = self._load_module("analogy_transfer", load_analogy)

        # ── AGI-2: ConjectureEngine ────────────────────────────────────────────
        def load_conjecture():
            from sare.meta.conjecture_engine import ConjectureEngine
            ce = ConjectureEngine(max_conjectures=200)
            ce.wire(
                negotiation_arena=self.agent_negotiator,
                engine=self,
            )
            # Seed with known rules from the concept registry
            if self.concept_registry:
                try:
                    for rule in self.concept_registry.get_consolidated_rules(min_confidence=0.7):
                        name = rule.get("name", "") if isinstance(rule, dict) else getattr(rule, "name", "")
                        domain = rule.get("domain", "arithmetic") if isinstance(rule, dict) else getattr(rule, "domain", "arithmetic")
                        conf = rule.get("confidence", 0.7) if isinstance(rule, dict) else getattr(rule, "confidence", 0.7)
                        if name:
                            ce.observe_rule(name, domain, float(conf))
                except Exception as e:
                    log.debug(f"ConjectureEngine seed: {e}")
            return ce
        self.conjecture_engine = self._load_module("conjecture_engine", load_conjecture)

    def _boot_world_model(self):
        def load_wm():
            from sare.memory.world_model import WorldModel
            return WorldModel()
        self.world_model = self._load_module("world_model", load_wm)

        # WorldModel v3 — self-learning, no hardcoding
        def load_wm3():
            from sare.memory.world_model_v3 import WorldModelV3
            return WorldModelV3()
        self.world_model_v3 = self._load_module("world_model_v3", load_wm3)

        # Transfer Engine
        if self._phase_enabled("causal", True):
            def load_te():
                from sare.transfer.engine import TransferEngine
                return TransferEngine()
            self.transfer_engine = self._load_module("transfer_engine", load_te)

            def load_ci():
                if not _sb:
                    raise RuntimeError("C++ bindings unavailable")
                cls = getattr(_sb, "CausalInduction", None)
                if not cls:
                    raise RuntimeError("CausalInduction binding unavailable")
                return cls()
            self.causal_induction = self._load_module("causal_induction", load_ci)
        else:
            self._mark_skipped("transfer_engine", "causal_induction", reason="causal phase disabled")

        if self._phase_enabled("reflection", True):
            def load_py_re():
                from sare.reflection.py_reflection import PyReflectionEngine
                return PyReflectionEngine()
            self.py_reflection_engine = self._load_module("py_reflection_engine", load_py_re)

            def load_cpp_re():
                if not _sb:
                    raise RuntimeError("C++ bindings unavailable")
                cls = getattr(_sb, "ReflectionEngine", None)
                if not cls:
                    raise RuntimeError("ReflectionEngine binding unavailable")
                return cls()
            self.cpp_reflection_engine = self._load_module("cpp_reflection_engine", load_cpp_re)
        else:
            self._mark_skipped("py_reflection_engine", "cpp_reflection_engine", reason="reflection phase disabled")

        if self._phase_enabled("plasticity", True):
            def load_mg():
                if not _sb:
                    raise RuntimeError("C++ bindings unavailable")
                cls = getattr(_sb, "ModuleGenerator", None)
                if not cls:
                    raise RuntimeError("ModuleGenerator binding unavailable")
                return cls()
            self.cpp_module_generator = self._load_module("cpp_module_generator", load_mg)
        else:
            self._mark_skipped("cpp_module_generator", reason="plasticity phase disabled")

        self.reflection_engine = self.cpp_reflection_engine or self.py_reflection_engine

    def _seed_synthesizer(self):
        """Seed transfer synthesizer so synthesis fires from the first auto-learn cycle."""
        if not (self.transfer_engine and self.transform_synthesizer):
            return
        try:
            seeded = self.transform_synthesizer.synthesize_all_missing(self.transfer_engine)
            if seeded:
                log.info(f"Boot synthesis: {len(seeded)} new transforms generated for domains: "
                         f"{list({s.domain for s in seeded})}")
                self._refresh_transforms()
        except Exception as _seed_err:
            log.debug(f"Boot synthesis skipped: {_seed_err}")

    def _boot_perception(self):
        def load_pe():
            from sare.perception.perception_engine import PerceptionEngine
            return PerceptionEngine()
        self.perception_engine = self._load_module("perception_engine", load_pe)

        if self._phase_enabled("plasticity", True):
            def load_ts():
                from sare.transfer.synthesizer import TransformSynthesizer
                return TransformSynthesizer()
            self.transform_synthesizer = self._load_module("transform_synthesizer", load_ts)
        else:
            self._mark_skipped("transform_synthesizer", reason="plasticity phase disabled")

    def _boot_language_grounding(self):
        def load_lg():
            from sare.language.grounding import LanguageGrounding
            return LanguageGrounding()
        self.language_grounding = self._load_module("language_grounding", load_lg)

    def _boot_language(self):
        def load_parser():
            try:
                from sare.interface.universal_parser import UniversalParser
                return UniversalParser()
            except Exception:
                from sare.interface.nl_parser_v2 import EnhancedNLParser
                return EnhancedNLParser()
        self.nl_parser = self._load_module("nl_parser", load_parser)

        def load_llm():
            from sare.interface import llm_bridge
            return llm_bridge
        self.llm_bridge = self._load_module("llm_bridge", load_llm)

    def _boot_social(self):
        def load_dm():
            from sare.social.dialogue_manager import DialogueManager
            return DialogueManager()
        self.dialogue_manager = self._load_module("dialogue_manager", load_dm)

        def load_tom():
            from sare.social.theory_of_mind import TheoryOfMindEngine
            tom = TheoryOfMindEngine()
            tom.load()
            return tom
        self.theory_of_mind = self._load_module("theory_of_mind", load_tom)

        def load_id():
            from sare.memory.identity import IdentityManager
            im = IdentityManager()
            im.load()
            return im
        self.identity = self._load_module("identity", load_id)

        def load_auto():
            from sare.memory.autobiographical import AutobiographicalMemory
            ab = AutobiographicalMemory()
            ab.load()
            return ab
        self.autobiography = self._load_module("autobiography", load_auto)

    def _boot_learning(self):
        if not self._phase_enabled("curiosity", True):
            self._mark_skipped("curriculum_gen", "experiment_runner", "hippocampus", reason="curiosity phase disabled")
            return

        def load_cg():
            from sare.curiosity.curriculum_generator import CurriculumGenerator
            cg = CurriculumGenerator()
            try:
                cg.load()
            except Exception:
                pass
            return cg
        self.curriculum_gen = self._load_module("curriculum_gen", load_cg)

        def load_er():
            from sare.curiosity.experiment_runner import ExperimentRunner
            er = ExperimentRunner(
                curriculum_gen=self.curriculum_gen,
                searcher=self.searcher,
                energy=self.energy,
                reflection_engine=self.py_reflection_engine or self.reflection_engine,
                causal_induction=self.causal_induction,
                concept_registry=self.concept_registry,
                transforms=self.transforms,
            )
            # Patch in optional modules
            if self.self_model:
                er.self_model = self.self_model
            if self.credit_assigner:
                er.credit_assigner = self.credit_assigner
            return er
        self.experiment_runner = self._load_module("experiment_runner", load_er)

        if self.meta_curriculum:
            try:
                self.meta_curriculum.wire(
                    curriculum=self.curriculum_gen,
                    engine=self,
                    stream=self.continuous_stream,
                )
                self._module_status["meta_curriculum"] = "✅"
            except Exception as e:
                self._module_status["meta_curriculum"] = f"❌ {e}"

        def load_hippo():
            from sare.memory.hippocampus import HippocampusDaemon
            hd = HippocampusDaemon(
                memory_manager=self.memory_manager,
                experiment_runner=self.experiment_runner,
                reflection_engine=self.py_reflection_engine or self.reflection_engine,
                curriculum_gen=self.curriculum_gen,
                problem_loader=self._load_problem_graph,
            )
            if self._boot_enabled("autostart_hippocampus", True):
                hd.start()
            return hd
        self.hippocampus = self._load_module("hippocampus", load_hippo)

    def _boot_curriculum(self):
        """Boot the developmental curriculum system."""
        if not self._phase_enabled("curriculum", True):
            self._mark_skipped("dev_curriculum", reason="curriculum phase disabled")
            return

        def load_dc():
            from sare.curriculum.developmental import DevelopmentalCurriculum
            dc = DevelopmentalCurriculum()
            dc.load()
            return dc
        self.developmental_curriculum = self._load_module("dev_curriculum", load_dc)

    # ─────────────────────────────────────────────────────────────────────────
    #  Event Wiring
    # ─────────────────────────────────────────────────────────────────────────

    def _wire_events(self):
        """Connect modules via event bus."""

        # When a solve completes, update memory + self model + world model
        self.events.subscribe(Event.SOLVE_COMPLETED, self._on_solve_completed)
        self.events.subscribe(Event.SOLVE_FAILED, self._on_solve_failed)
        self.events.subscribe(Event.RULE_DISCOVERED, self._on_rule_discovered)
        self.events.subscribe(Event.RULE_PROMOTED, self._on_rule_promoted)
        self.events.subscribe(Event.DOMAIN_MASTERED, self._on_domain_mastered)
        self.events.subscribe(Event.COMPETENCE_UPDATED, self._on_competence_updated)

    def _on_solve_completed(self, ed: EventData):
        """Post-solve processing: store episode, update competence, reflect."""
        data = ed.data
        problem_id = data.get("problem_id", "")
        transforms_used = data.get("transforms", [])
        energy_before = data.get("energy_before", 0)
        energy_after = data.get("energy_after", 0)
        domain = data.get("domain", "general")
        elapsed = data.get("elapsed", 0)
        delta = energy_before - energy_after
        success = delta > 0.01

        self._stats["solves_attempted"] += 1
        if success:
            self._stats["solves_succeeded"] += 1

        # WorldModel v3: observe every solve (learns causal links, schemas, beliefs)
        if self.world_model_v3:
            try:
                self.world_model_v3.observe_solve(
                    expression=problem_id, transforms=transforms_used,
                    delta=delta, domain=domain, success=success,
                )
            except Exception as e:
                log.debug(f"WorldModelV3 observe failed: {e}")

        # Transfer Engine: observe all solves to discover roles
        if self.transfer_engine:
            try:
                self.transfer_engine.observe(transforms_used, domain, success)
            except Exception as exc:
                self._record_runtime_error("transfer_engine.observe", exc, "post_solve")

        # 1. Store episode
        if self.memory_manager:
            try:
                from sare.memory.memory_manager import SolveEpisode
                episode = SolveEpisode(
                    problem_id=problem_id,
                    transform_sequence=transforms_used,
                    energy_trajectory=[energy_before, energy_after],
                    initial_energy=energy_before,
                    final_energy=energy_after,
                    compute_time_seconds=elapsed,
                    total_expansions=int(data.get("expansions", 0) or 0),
                    success=success,
                )
                self.memory_manager.after_solve(episode, data.get("initial_graph"))
                self.events.emit(Event.EPISODE_STORED, {"problem_id": problem_id}, "memory")
            except Exception as exc:
                self._record_runtime_error("memory_manager.after_solve", exc, "post_solve")

        # 1b. Record solved traces for concept clustering and later replay.
        if self.concept_memory is not None and success and data.get("final_graph") is not None:
            try:
                self.concept_memory.record(data["final_graph"], problem_id, transforms_used)
                if len(self.concept_memory) % 10 == 0:
                    from sare.memory.concept_formation import ConceptFormation
                    ConceptFormation(self.concept_memory, self.concept_registry).run()
            except Exception as exc:
                self._record_runtime_error("concept_memory.record", exc, "post_solve")

        # 2. Update self model (API: observe, not record_solve)
        if self.self_model:
            try:
                _elapsed_ms = elapsed * 1000
                _strategy = data.get('strategy', 'beam_search')
                self.self_model.observe(
                    domain=domain,
                    success=success,
                    delta=delta,
                    steps=len(transforms_used),
                    transforms_used=transforms_used,
                    predicted_confidence=0.5,
                    strategy=_strategy,
                    elapsed_ms=_elapsed_ms,
                )
                self.events.emit(Event.COMPETENCE_UPDATED, {
                    "domain": domain, "success": success
                }, "self_model")
            except Exception as exc:
                self._record_runtime_error("self_model.observe", exc, "post_solve")

        # 2b. Update developmental curriculum (track domain mastery)
        if self.developmental_curriculum:
            try:
                self.developmental_curriculum.record_attempt(
                    problem_id, domain, success, delta
                )
            except Exception as exc:
                self._record_runtime_error("developmental_curriculum.record_attempt", exc, "post_solve")

        # 3. Credit assignment (API: assign_credit)
        if self.credit_assigner and transforms_used:
            try:
                traj = data.get("energy_trajectory", [energy_before, energy_after])
                if len(traj) < 2:
                    traj = [energy_before, energy_after]
                self.credit_assigner.assign_credit(transforms_used, traj)
                updater = getattr(self.self_model, "update_transform_utilities", None) if self.self_model else None
                if callable(updater) and hasattr(self.credit_assigner, "get_all_utilities"):
                    updater(self.credit_assigner.get_all_utilities())
            except Exception as exc:
                self._record_runtime_error("credit_assigner.assign_credit", exc, "post_solve")

        # 4. Frontier update
        if self.frontier_manager:
            try:
                self.frontier_manager.record(
                    problem_id=problem_id,
                    success=success,
                    delta=delta,
                    num_transforms=len(transforms_used),
                )
            except Exception as exc:
                self._record_runtime_error("frontier_manager.record", exc, "post_solve")

        # 5. World model prediction tracking
        if self.world_model and hasattr(self.world_model, 'record_outcome'):
            try:
                self.world_model.record_outcome(
                    transforms_used, delta, domain
                )
            except Exception as exc:
                self._record_runtime_error("world_model.record_outcome", exc, "post_solve")

        # 6. Autobiographical memory (significant events)
        if self.autobiography and success and delta > 2.0:
            try:
                self.autobiography.record(
                    event_type="solve_success",
                    domain=domain,
                    description=f"Solved {problem_id} with delta={delta:.2f}",
                    importance=min(1.0, delta / 10.0),
                    related_rules=transforms_used[:3],
                )
            except Exception as exc:
                self._record_runtime_error("autobiography.record", exc, "solve_success")

        # 7. Reflect on successful solves (extract abstract rules)
        if success and self.reflection_engine and data.get("initial_graph") and data.get("final_graph"):
            self._reflect(data["initial_graph"], data["final_graph"], domain, transforms_used)

        # 7b. Update working memory with outcome
        if self.working_memory:
            try:
                self.working_memory.record_outcome(
                    domain, transforms_used, success, delta
                )
            except Exception as exc:
                self._record_runtime_error("working_memory.record_outcome", exc, "post_solve")

        # 7c. Feed into KnowledgeGraph
        if self.knowledge_graph and success and transforms_used:
            try:
                for t_name in set(transforms_used):
                    self.knowledge_graph.add_causal_link(
                        cause=t_name, effect="energy_reduction",
                        mechanism=domain, domain=domain, confidence=0.6,
                    )
            except Exception as exc:
                self._record_runtime_error("knowledge_graph.add_causal_link", exc, "post_solve")

        # 8. Ground language: link each applied transform to NL explanation
        if success and transforms_used and self.language_grounding:
            try:
                for t_name in set(transforms_used):
                    self.language_grounding.ground_from_transform(t_name, domain)
            except Exception as exc:
                self._record_runtime_error("language_grounding.ground_from_transform", exc, "post_solve")

        # 8a-ii. Concept grounding: perception → concept → symbol linkage
        # Every successful solve is a grounded observation that strengthens concepts
        if success and transforms_used and self.concept_graph:
            try:
                expression_text = data.get("expression") or problem_id
                result_text = self._graph_preview(data.get("final_graph"))
                self.concept_graph.ground_solve_episode(
                    expression=expression_text,
                    result=result_text,
                    transforms_used=transforms_used,
                    domain=domain,
                    delta=delta,
                )
                # Trigger abstraction for any well-observed concepts
                for t_name in transforms_used:
                    c = self.concept_graph.concept_for_transform(t_name)
                    if c and c.ground_count() >= 3:
                        self.concept_graph.abstract_from_examples(c.name)
            except Exception as exc:
                self._record_runtime_error("concept_graph.ground_solve_episode", exc, "post_solve")

        # 8b-i. Multi-step causal chain detection
        if success and len(transforms_used) >= 2 and delta > 0.3:
            try:
                if not hasattr(self, '_causal_chain_detector'):
                    from sare.causal.chain_detector import CausalChainDetector
                    self._causal_chain_detector = CausalChainDetector()
                new_chains = self._causal_chain_detector.observe(
                    transforms=transforms_used,
                    domain=domain,
                    delta=delta,
                    success=True,
                )
                for chain in new_chains:
                    if self.knowledge_graph:
                        try:
                            # Record each step of the chain as a causal link
                            for i in range(len(chain.steps) - 1):
                                self.knowledge_graph.add_causal_link(
                                    cause=chain.steps[i],
                                    effect=chain.steps[i + 1],
                                    mechanism="transform_sequence",
                                    domain=domain,
                                    confidence=chain.confidence,
                                )
                        except Exception as exc:
                            self._record_runtime_error("knowledge_graph.add_causal_link", exc, "causal_chain")
                    self.events.emit(Event.RULE_DISCOVERED, {
                        "type": "causal_chain",
                        "chain": chain.steps,
                        "name": chain.name,
                        "domain": domain,
                        "confidence": chain.confidence,
                        "cross_domain": chain.cross_domain,
                        "length": chain.length,
                    }, "causal_chain_detector")
                    log.debug(
                        f"Causal chain: {chain.name} [{domain}] "
                        f"conf={chain.confidence:.2f} cross={chain.cross_domain}"
                    )
            except Exception as exc:
                self._record_runtime_error("causal_chain_detector.observe", exc, "post_solve")

        # 8b. Abductive causal explanation — WHY did this solve work?
        if success and transforms_used and delta > 0.5:
            try:
                if not hasattr(self, '_abductive_ranker'):
                    from sare.causal.abductive_ranker import AbductiveRanker
                    self._abductive_ranker = AbductiveRanker(
                        concept_registry=self.concept_registry,
                        lambda_occam=1.2,
                    )
                hyps = self._abductive_ranker.explain(
                    observed_transforms=transforms_used,
                    observed_delta=delta,
                    domain=domain,
                    top_k=2,
                )
                for h in hyps[:1]:  # store best hypothesis
                    if self.knowledge_graph and h.posterior > 0.3:
                        try:
                            for step in h.reasoning_chain[:2]:
                                self.knowledge_graph.add_causal_link(
                                    cause=h.name, effect=step,
                                    mechanism=domain, domain=domain,
                                    confidence=h.posterior,
                                )
                        except Exception as exc:
                            self._record_runtime_error("knowledge_graph.add_causal_link", exc, "abductive_explanation")
                    # Emit analogy events for co-occurring rules
                    if h.recommended_action == "accept" and len(h.supporting_transforms) > 1:
                        self.events.emit(Event.ANALOGY_FOUND, {
                            "hypothesis": h.name,
                            "transforms": h.supporting_transforms,
                            "domain": domain,
                            "posterior": h.posterior,
                        }, "abductive_ranker")
            except Exception as exc:
                self._record_runtime_error("abductive_ranker.explain", exc, "post_solve")

    def _on_solve_failed(self, ed: EventData):
        """Handle solve failures: LEARN from them, don't just record them."""
        data = ed.data
        self._stats["solves_attempted"] += 1
        expression = data.get("expression", "")
        domain = data.get("domain", "general")
        transforms_tried = data.get("transforms", [])
        energy_before = data.get("energy_before", 0)
        energy_after = data.get("energy_after", 0)

        # ── 1. Failure Analysis: understand WHY it failed ──
        failure_reason = self._analyze_failure(data)
        self._stats.setdefault("failure_reasons", {})
        reason = failure_reason.get("reason", "unknown")
        self._stats["failure_reasons"][reason] = self._stats["failure_reasons"].get(reason, 0) + 1

        # ── 2. WorldModel v3: learn from failure (critical!) ──
        if self.world_model_v3:
            try:
                self.world_model_v3.observe_solve(
                    expression=expression, transforms=transforms_tried,
                    delta=energy_before - energy_after, domain=domain, success=False,
                )
                # Record negative belief: these transforms DON'T work here
                for t in transforms_tried:
                    bkey = f"transform:{t}:fails_on:{self._expression_class(expression)}"
                    if bkey not in self.world_model_v3._beliefs:
                        from sare.memory.world_model_v3 import Belief
                        self.world_model_v3._beliefs[bkey] = Belief(
                            subject=t, predicate=f"fails on {self._expression_class(expression)}",
                            confidence=0.5, domain=domain
                        )
                    self.world_model_v3._beliefs[bkey].update(supports=True)
            except Exception as exc:
                self._record_runtime_error("world_model_v3.observe_solve", exc, "solve_failed")

        # ── 3. Transfer Engine: observe failures too ──
        if self.transfer_engine:
            try:
                self.transfer_engine.observe(transforms_tried, domain, False)
            except Exception as exc:
                self._record_runtime_error("transfer_engine.observe", exc, "solve_failed")

        # ── 4. Retry with different strategy (MCTS if beam failed) ──
        # Guard against infinite recursion: only retry if not already retrying
        retry_result = None
        if failure_reason.get("retryable") and expression and not getattr(self, '_in_retry', False):
            self._in_retry = True
            try:
                retry_result = self._retry_with_alternative(expression, domain, failure_reason)
                if retry_result and retry_result.get("success"):
                    self._stats.setdefault("retry_successes", 0)
                    self._stats["retry_successes"] += 1
            finally:
                self._in_retry = False

        # ── 5. Homeostasis & Identity ──
        if self.homeostasis:
            try:
                self.homeostasis.tick()
            except Exception as exc:
                self._record_runtime_error("homeostasis.tick", exc, "solve_failed")
        if self.identity:
            try:
                self.identity.update_from_behavior("stuck_period", domain, success=False)
            except Exception as exc:
                self._record_runtime_error("identity.update_from_behavior", exc, "solve_failed")

        # ── 6. Store failure in autobiographical memory (important events) ──
        if self.autobiography:
            try:
                self.autobiography.record(
                    event_type="stuck_period",
                    domain=domain,
                    description=f"Failed: {expression[:40]} — {reason}",
                    importance=0.4,
                    related_rules=transforms_tried[:3],
                )
            except Exception as exc:
                self._record_runtime_error("autobiography.record", exc, "solve_failed")

    def _analyze_failure(self, solve_data: dict) -> dict:
        """Analyze WHY a solve failed. Returns structured failure report."""
        expression = solve_data.get("expression", "")
        transforms = solve_data.get("transforms", [])
        energy_before = solve_data.get("energy_before", 0)
        energy_after = solve_data.get("energy_after", 0)
        delta = energy_before - energy_after
        graph = solve_data.get("initial_graph")

        # Classify the failure
        if not transforms:
            # No transforms matched at all
            reason = "no_matching_transforms"
            retryable = True
            suggestion = "Need new transform types for this pattern"
        elif delta < 0:
            # Transforms made it worse
            reason = "negative_delta"
            retryable = True
            suggestion = "Transforms increased energy — try different order"
        elif abs(delta) < 0.01:
            # Transforms had no effect
            reason = "zero_delta"
            retryable = True
            suggestion = "Transforms matched but didn't reduce energy"
        else:
            reason = "insufficient_reduction"
            retryable = False
            suggestion = "Partial progress — may need multi-step approach"

        # Check if this expression class has failed before
        expr_class = self._expression_class(expression)
        repeated = self._stats.get("failure_reasons", {}).get(reason, 0) > 3

        return {
            "reason": reason,
            "retryable": retryable,
            "suggestion": suggestion,
            "expression_class": expr_class,
            "transforms_tried": transforms,
            "delta": delta,
            "repeated_failure": repeated,
        }

    def _expression_class(self, expression: str) -> str:
        """Classify an expression into a structural class for failure tracking."""
        e = expression.lower().strip()
        if "=" in e:
            return "equation"
        if any(op in e for op in ["not ", "and ", "or "]):
            return "logic"
        if any(op in e for op in ["union", "intersect"]):
            return "set_theory"
        ops = sum(1 for c in e if c in "+-*/^")
        if ops >= 3:
            return "complex_arithmetic"
        if ops >= 1:
            return "simple_arithmetic"
        return "atomic"

    def _retry_with_alternative(self, expression: str, domain: str,
                                 failure_info: dict) -> Optional[dict]:
        """Retry a failed problem with a different strategy."""
        try:
            # Strategy 1: Try MCTS if beam search failed
            result = self.solve(
                expression, algorithm="mcts",
                beam_width=12, max_depth=50, budget=5.0, domain=domain
            )
            if result.get("success"):
                return result

            # Strategy 2: Try with wider beam
            result = self.solve(
                expression, algorithm="beam",
                beam_width=16, max_depth=60, budget=8.0, domain=domain
            )
            return result
        except Exception as exc:
            self._record_runtime_error("retry_with_alternative", exc, expression[:80])
            return None

    def _on_rule_discovered(self, ed: EventData):
        self._stats["rules_discovered"] += 1
        self.total_rules_learned += 1

        if self.identity:
            try:
                self.identity.update_from_behavior("rule_discovered", ed.data.get("domain", "general"), success=True)
            except Exception as exc:
                self._record_runtime_error("identity.update_from_behavior", exc, "rule_discovered")

    def _on_rule_promoted(self, ed: EventData):
        self._stats["rules_promoted"] += 1
        log.info(f"Rule promoted: {ed.data.get('name', '?')} → refreshing transforms")
        self._refresh_transforms()

    def _on_domain_mastered(self, ed: EventData):
        domain = ed.data.get("domain", "")
        if domain and domain not in self._stats["domains_mastered"]:
            self._stats["domains_mastered"].append(domain)

        if self.autobiography:
            try:
                self.autobiography.record(
                    event_type="domain_mastered",
                    domain=domain,
                    description=f"Mastered domain: {domain}",
                    importance=0.9,
                    related_rules=[],
                )
            except Exception as exc:
                self._record_runtime_error("autobiography.record", exc, "domain_mastered")

        self._update_stage()

    def _on_competence_updated(self, ed: EventData):
        """Check if stage should advance."""
        self._update_stage()

    def _refresh_transforms(self):
        """Reload transforms including newly learned rules + synthesized transforms."""
        try:
            self.transforms = self.engine.get_transforms(
                include_macros=True,
                concept_registry=self.concept_registry,
            )
            # Add synthesized transforms from deep transfer
            if self.transform_synthesizer:
                try:
                    synth = self.transform_synthesizer.get_live_transforms()
                    if synth:
                        self.transforms = synth + self.transforms
                except Exception:
                    pass
            # Add transforms from discovered reflection rules
            learned_rule_source = self.py_reflection_engine or self.reflection_engine
            if learned_rule_source and hasattr(learned_rule_source, 'get_high_confidence_rules'):
                try:
                    for rule in learned_rule_source.get_high_confidence_rules(min_confidence=0.65):
                        t = self._rule_to_transform(rule)
                        if t:
                            self.transforms.insert(0, t)
                except Exception:
                    pass
            if self.experiment_runner:
                self.experiment_runner.transforms = self.transforms
        except Exception as e:
            log.error(f"Transform refresh failed: {e}")

    def _rule_to_transform(self, rule) -> Optional[Any]:
        """Convert a discovered AbstractRule into a live Transform for search."""
        try:
            from sare.engine import Transform, Graph
            name = rule.name if hasattr(rule, 'name') else rule.get('name', '')
            op = rule.operator_involved if hasattr(rule, 'operator_involved') else rule.get('operator', '')
            pattern = rule.pattern_description if hasattr(rule, 'pattern_description') else rule.get('pattern', '')

            if not name or not op:
                return None

            # Check if this rule maps to a known pattern type
            if 'identity' in name and op:
                # Find what constant is the identity element
                for label in ['0', '1', 'true', 'false', 'empty']:
                    if label in pattern:
                        from sare.transfer.synthesizer import SynthesizedTransform, _make_runtime_transform
                        spec = SynthesizedTransform(
                            name=f"learned_{name}",
                            domain=rule.domain if hasattr(rule, 'domain') else 'general',
                            role="identity",
                            operator_labels=[op],
                            element_label=label,
                            rewrite_action="replace_with_other_child",
                            confidence=rule.confidence if hasattr(rule, 'confidence') else 0.7,
                        )
                        return _make_runtime_transform(spec)

            if 'annihilation' in name and op:
                for label in ['0', 'false', 'empty']:
                    if label in pattern:
                        from sare.transfer.synthesizer import SynthesizedTransform, _make_runtime_transform
                        spec = SynthesizedTransform(
                            name=f"learned_{name}",
                            domain=rule.domain if hasattr(rule, 'domain') else 'general',
                            role="annihilation",
                            operator_labels=[op],
                            element_label=label,
                            rewrite_action="replace_with_absorbing",
                            confidence=rule.confidence if hasattr(rule, 'confidence') else 0.7,
                        )
                        return _make_runtime_transform(spec)

            if 'double' in name or 'elimination' in name:
                from sare.transfer.synthesizer import SynthesizedTransform, _make_runtime_transform
                spec = SynthesizedTransform(
                    name=f"learned_{name}",
                    domain=rule.domain if hasattr(rule, 'domain') else 'general',
                    role="involution",
                    operator_labels=[op],
                    element_label="",
                    rewrite_action="unwrap_double",
                    confidence=rule.confidence if hasattr(rule, 'confidence') else 0.7,
                )
                return _make_runtime_transform(spec)

        except Exception:
            pass
        return None

    def _is_cpp_search_candidate(self, graph, domain: str) -> bool:
        if not (self.cpp_enabled and self.cpp_bindings_available):
            return False
        if domain not in ("arithmetic", "logic", "general"):
            return False
        allowed = {"+", "*", "neg", "not", "and", "or"}
        for n in graph.nodes:
            if n.type != "operator":
                continue
            if n.label not in allowed:
                return False
        return True

    def _solve_with_cpp_bindings(self, graph, algorithm: str, beam_width: int, max_depth: int, budget: float, kappa: float = 0.1):
        if not self._is_cpp_search_candidate(graph, self._detect_domain(graph, "general")):
            return None
        try:
            cpp_graph = _py_graph_to_cpp_graph(graph)
            cfg = self.cpp_search_config_cls()
            cfg.beam_width = int(beam_width)
            cfg.max_depth = int(max_depth)
            cfg.budget_seconds = float(budget)
            if hasattr(cfg, "kappa"):
                cfg.kappa = float(kappa)
            cpp_result = self.cpp_run_mcts_search(cpp_graph, cfg) if algorithm == "mcts" else self.cpp_run_beam_search(cpp_graph, cfg)
            best_graph = _cpp_graph_to_py_graph(cpp_result.best_graph)
            initial_total = self.energy.compute(graph).total
            best_energy = self.energy.compute(best_graph)
            if best_energy.total >= initial_total - 0.01:
                return None
            return self.engine.SearchResult(
                graph=best_graph,
                energy=best_energy,
                transforms_applied=list(cpp_result.best_state.transform_trace),
                steps_taken=int(cpp_result.best_state.depth or len(cpp_result.best_state.transform_trace)),
                expansions=int(cpp_result.total_expansions),
                elapsed_seconds=float(cpp_result.elapsed_seconds),
                energy_trajectory=[initial_total, best_energy.total],
            )
        except Exception as e:
            log.debug(f"C++ solve fast path unavailable: {e}")
            return None

    def enable_cpp_fast_path(self, enabled: bool = True):
        self.cpp_enabled = bool(enabled) and self.cpp_bindings_available

    def _emit_rule_events(self, rule_name: str, domain: str, confidence: float, pattern: str = ""):
        self.events.emit(Event.RULE_DISCOVERED, {
            "name": rule_name,
            "domain": domain,
            "confidence": confidence,
            "pattern": pattern,
        }, "reflection")

    def _promote_python_rule(self, rule, domain: str):
        confidence = getattr(rule, "confidence", 0.0)
        if not self.concept_registry or confidence <= 0.6:
            return False
        try:
            payload = rule.to_dict() if hasattr(rule, "to_dict") else rule
            self.concept_registry.add_rule(payload)
            self.events.emit(Event.RULE_PROMOTED, {
                "name": getattr(rule, "name", payload.get("name", "")),
                "domain": domain,
            }, "reflection")
            # P4.4: Mirror promoted rule into KnowledgeGraph
            if self.knowledge_graph:
                try:
                    self.knowledge_graph.add_rule(rule)
                except Exception:
                    pass
            return True
        except Exception:
            return False

    def _promote_cpp_rule(self, rule, domain: str):
        if not self.concept_registry:
            return False
        promoted = False
        try:
            if self.causal_induction:
                try:
                    induction = self.causal_induction.evaluate(rule=rule, num_tests=8)
                    promoted = bool(getattr(induction, "promoted", False))
                    rule.confidence = float(getattr(induction, "evidence_score", rule.confidence))
                except Exception as _ci_err:
                    log.debug(f"CausalInduction.evaluate failed: {_ci_err}")
                    promoted = bool(getattr(rule, "confidence", 0.0) > 0.6)
            else:
                promoted = bool(getattr(rule, "confidence", 0.0) > 0.6)
            if promoted:
                payload = rule
                if isinstance(self.concept_registry, _MinimalConceptRegistry):
                    payload = {
                        "name": getattr(rule, "name", ""),
                        "domain": getattr(rule, "domain", domain),
                        "confidence": float(getattr(rule, "confidence", 0.0)),
                        "observations": int(getattr(rule, "observations", 1)),
                    }
                self.concept_registry.add_rule(payload)
                self.events.emit(Event.RULE_PROMOTED, {
                    "name": getattr(rule, "name", ""),
                    "domain": domain,
                }, "reflection")
            return promoted
        except Exception as e:
            log.debug(f"C++ rule promotion failed: {e}")
            return False

    def _verification_expressions_for(self, spec) -> List[str]:
        role = getattr(spec, "role", "")
        domain = getattr(spec, "domain", "")
        domain_defaults = {
            "logic":       ["not not x", "x and true", "x or false", "x and false", "x or true"],
            "propositional": ["not not x", "x and true", "x or false", "x and false", "x or true"],
            "calculus":    ["derivative(x)", "derivative(5)", "derivative(x^2)", "derivative(1)", "derivative(x^3)"],
            "trigonometry":["sin(0)", "cos(0)", "log(1)", "sqrt(x^2)", "sin(0) + cos(0)"],
            "probability": ["p(empty)", "p(Omega)"],
            "geometry":    ["angle_sum(triangle)", "3^2 + 4^2"],
            "arithmetic":  ["x + 0", "0 + x", "x * 1", "1 * x", "x * 0"],
            "algebra":     ["x + 0", "x * 1", "x + 2 = 5"],
        }
        if role == "identity":
            if domain in ("logic", "propositional"):
                return ["x and true", "true and x", "x or false", "false or x", "x and true and true"]
            return ["x + 0", "0 + x", "x * 1", "1 * x", "(x + 0) + 0"]
        if role == "annihilation":
            if domain in ("logic", "propositional"):
                return ["x and false", "false and x", "x or true", "true or x", "(x and false) or y"]
            return ["x * 0", "0 * x", "x and false", "x or true", "(x * 0) + y"]
        if role == "involution":
            return ["-(-x)", "-(-y)", "not not x", "not(not y)", "-(-3)"]
        if role == "self_inverse":
            return ["x - x", "y - y", "a - a", "5 - 5", "z - z"]
        return domain_defaults.get(domain, ["x + 0", "0 + x", "x * 1", "1 * x", "x * 0"])

    def _map_cpp_step_name(self, name: str) -> str:
        return {
            "algebra_add_zero": "add_zero_elim",
            "algebra_mul_one": "mul_one_elim",
            "algebra_mul_zero": "mul_zero_elim",
            "logic_double_negation": "double_negation",
            "logic_and_true": "bool_and_true",
            "ast_constant_fold": "constant_fold",
        }.get(name, name)

    def _run_cpp_plasticity(self) -> int:
        if not (self.cpp_module_generator and self.memory_manager and self.cpp_search_config_cls):
            return 0
        try:
            recent = self.memory_manager.recent_episodes(80)
            failures = [ep for ep in recent if not getattr(ep, "success", False)]
            if len(failures) < 5:
                return 0
            cpp_failures = []
            test_graphs = []
            for ep in failures[-30:]:
                try:
                    cpp_ep = _sb.SolveEpisode()
                    cpp_ep.problem_id = str(ep.problem_id)
                    cpp_ep.transform_sequence = list(ep.transform_sequence)
                    cpp_ep.energy_trajectory = list(ep.energy_trajectory)
                    cpp_ep.initial_energy = float(ep.initial_energy)
                    cpp_ep.final_energy = float(ep.final_energy)
                    cpp_ep.compute_time_seconds = float(ep.compute_time_seconds)
                    cpp_ep.total_expansions = int(ep.total_expansions)
                    cpp_ep.success = bool(ep.success)
                    cpp_failures.append(cpp_ep)
                    test_graphs.append(_py_graph_to_cpp_graph(self.engine.build_expression_graph(str(ep.problem_id))))
                except Exception:
                    continue
            if len(cpp_failures) < 5 or len(test_graphs) < 3:
                return 0
            cfg = self.cpp_search_config_cls()
            cfg.beam_width = 8
            cfg.max_depth = 20
            cfg.budget_seconds = 2.0
            proposals = self.cpp_module_generator.propose_from_failures(
                cpp_failures, test_graphs[:10], cfg, 2, 0.01
            )
            if not proposals:
                return 0
            from sare.meta.macro_registry import MacroSpec, upsert_macros
            promoted = []
            for item in proposals:
                if not item.get("promoted"):
                    continue
                steps = [self._map_cpp_step_name(str(s)) for s in item.get("steps", [])]
                if len(steps) < 2:
                    continue
                promoted.append(MacroSpec(name=str(item.get("name", "cpp_macro")), steps=steps))
            if not promoted:
                return 0
            upsert_macros(promoted)
            for spec in promoted:
                self.events.emit(Event.MACRO_CREATED, {"name": spec.name, "steps": spec.steps}, "plasticity")
            self._stats["sleep_cycles"] = self._stats.get("sleep_cycles", 0)
            return len(promoted)
        except Exception as e:
            log.debug(f"C++ plasticity failed: {e}")
            return 0

    # ─────────────────────────────────────────────────────────────────────────
    #  Cognitive Loop
    # ─────────────────────────────────────────────────────────────────────────

    def solve(self, expression: str, algorithm: str = "beam",
              beam_width: int = 8, max_depth: int = 30,
              budget: float = 10.0, domain: str = "general",
              kappa: float = 0.1, force_python: bool = False) -> dict:
        """
        Full cognitive solve: perceive → recall → plan → act → reflect.

        Returns a result dict with graph, energy, transforms, proof, etc.
        """
        self.events.emit(Event.SOLVE_STARTED, {
            "expression": expression, "domain": domain
        }, "brain")

        # 1. PERCEIVE: Parse expression into graph
        expr_str, graph = self.engine.load_problem(expression)

        # 3. PLAN: Determine domain first (needed for domain-aware energy)
        detected_domain = self._detect_domain(graph, domain)

        # P3.3: Use domain-aware EnergyEvaluator for calculus/trig/distribution/factoring domains
        if detected_domain in ("calculus", "trigonometry", "distribution", "factoring"):
            from sare.engine import EnergyEvaluator as EE
            active_energy = EE(domain=detected_domain)
        else:
            active_energy = self.energy

        initial_energy = active_energy.compute(graph)

        # 2. RECALL: Check memory for warm-start strategy
        strategy_hint = None
        if self.memory_manager:
            try:
                strategy_hint = self.memory_manager.before_solve(graph)
            except Exception as exc:
                self._record_runtime_error("memory_manager.before_solve", exc, "solve")

        recalled = []
        recalled_memory = []
        if self.concept_memory is not None:
            try:
                similar = self.concept_memory.retrieve_similar(graph, top_k=3)
                recalled = [s for s in similar if s.get("similarity", 0) > 0.7]
                recalled_memory = self._serialize_recalled_memory(similar)
            except Exception as exc:
                self._record_runtime_error("concept_memory.retrieve_similar", exc, "solve")

        # 4. ACT: Run search with synthesized transforms included
        heuristic_fn = self.engine.load_heuristic_scorer()
        transforms = list(self.transforms)

        # Add synthesized transforms from deep transfer
        if self.transform_synthesizer:
            try:
                synth = self.transform_synthesizer.get_live_transforms()
                if synth:
                    transforms = synth + transforms
            except Exception as exc:
                self._record_runtime_error("transform_synthesizer.get_live_transforms", exc, "solve")

        # P4.2 + Causal search: re-rank transforms using working memory attention
        if self.working_memory:
            try:
                self.working_memory.focus(detected_domain)
                transforms = self.working_memory.get_prioritized_transforms(
                    transforms, detected_domain
                )
            except Exception as exc:
                self._record_runtime_error("working_memory.get_prioritized_transforms", exc, "solve")

        # Compositional planning: for complex problems (>3 operators),
        # run iterative deepening — solve, check, solve again on result
        operators = [n for n in graph.nodes if n.type == "operator"]
        max_iterations = min(3, len(operators)) if len(operators) > 2 else 1

        working_graph = graph
        all_transforms_applied = []
        total_expansions = 0
        total_elapsed = 0.0
        trajectory = [active_energy.compute(graph).total]

        cpp_fast_path_result = None
        if max_iterations == 1 and not force_python:
            cpp_fast_path_result = self._solve_with_cpp_bindings(
                working_graph, algorithm, beam_width, max_depth, budget, kappa
            )

        if cpp_fast_path_result is not None:
            result = cpp_fast_path_result
            working_graph = result.graph
            total_expansions = result.expansions
            total_elapsed = result.elapsed_seconds
            trajectory = list(result.energy_trajectory or trajectory)
            all_transforms_applied = list(result.transforms_applied)
        else:
            for iteration in range(max_iterations):
                remaining_budget = max(1.0, budget - total_elapsed)

                if algorithm == "mcts":
                    from sare.engine import MCTSSearch
                    searcher = MCTSSearch()
                    result = searcher.search(
                        working_graph, active_energy, transforms,
                        iterations=max_depth * 10,
                        budget_seconds=remaining_budget,
                    )
                else:
                    result = self.searcher.search(
                        working_graph, active_energy, transforms,
                        beam_width=beam_width,
                        max_depth=max_depth,
                        budget_seconds=remaining_budget,
                        kappa=kappa,
                        heuristic_fn=heuristic_fn,
                    )

                all_transforms_applied.extend(result.transforms_applied)
                total_expansions += result.expansions
                total_elapsed += result.elapsed_seconds
                trajectory.extend(result.energy_trajectory[1:])

                iter_delta = active_energy.compute(working_graph).total - result.energy.total
                if iter_delta < 0.01:
                    break
                working_graph = result.graph

            final_energy = active_energy.compute(working_graph)
            from sare.engine import SearchResult as SR
            result = SR(
                graph=working_graph, energy=final_energy,
                transforms_applied=all_transforms_applied,
                steps_taken=len(all_transforms_applied),
                expansions=total_expansions,
                elapsed_seconds=total_elapsed,
                energy_trajectory=trajectory,
            )

        delta = initial_energy.total - result.energy.total
        success = delta > 0.01

        # 5. REFLECT: Emit solve event (triggers all post-processing via event bus)
        self.events.emit(
            Event.SOLVE_COMPLETED if success else Event.SOLVE_FAILED,
            {
                "problem_id": expr_str,
                "expression": expr_str,
                "domain": detected_domain,
                "transforms": result.transforms_applied,
                "energy_before": initial_energy.total,
                "energy_after": result.energy.total,
                "delta": delta,
                "elapsed": result.elapsed_seconds,
                "initial_graph": graph,
                "final_graph": result.graph,
                "steps": result.steps_taken,
                "expansions": result.expansions,
                "energy_trajectory": result.energy_trajectory,
            },
            "brain",
        )

        abstractions_used = [t for t in result.transforms_applied if t.startswith("macro_")]
        node_types, adjacency = _graph_features(graph)
        try:
            SareLogger(str(SOLVE_LOG_PATH)).log(SolveLog(
                problem_id=expr_str,
                initial_energy=initial_energy.total,
                final_energy=result.energy.total,
                search_depth=result.steps_taken,
                compute_time_seconds=result.elapsed_seconds,
                total_expansions=result.expansions,
                solve_success=success,
                transform_sequence=result.transforms_applied,
                abstractions_used=abstractions_used,
                modules_activated=["brain"],
                domain=detected_domain,
                energy_breakdown=result.energy.components,
                energy_trajectory=result.energy_trajectory,
                node_types=node_types,
                adjacency=adjacency,
                budget_exhausted=(result.elapsed_seconds >= budget and not success),
            ))
        except Exception as exc:
            self._record_runtime_error("solve_logger.log", exc, "post_solve")

        self._persist_post_solve_state()

        # 6. BUILD PROOF
        proof = None
        if self.proof_builder and success:
            try:
                proof = self.proof_builder.build(
                    expression=expr_str,
                    transforms_applied=result.transforms_applied,
                    initial_energy=initial_energy.total,
                    final_energy=result.energy.total,
                )
            except Exception as exc:
                self._record_runtime_error("proof_builder.build", exc, "solve")

        reduction_pct = (delta / initial_energy.total * 100) if initial_energy.total > 0 else 0.0
        proof_payload = proof.to_dict() if proof and hasattr(proof, "to_dict") else None
        initial_payload = {
            "graph": graph.to_dict(),
            "energy": {
                "total": round(initial_energy.total, 3),
                "components": {k: round(v, 3) for k, v in initial_energy.components.items()},
            },
        }
        result_payload = {
            "graph": result.graph.to_dict(),
            "energy": {
                "total": round(result.energy.total, 3),
                "components": {k: round(v, 3) for k, v in result.energy.components.items()},
            },
            "transforms": list(result.transforms_applied),
            "steps": result.steps_taken,
            "expansions": result.expansions,
            "elapsed": round(result.elapsed_seconds, 4),
            "trajectory": [round(e, 3) for e in result.energy_trajectory],
        }

        return {
            "expression": expr_str,
            "graph": result_payload["graph"],
            "energy": result_payload["energy"],
            "initial_energy": round(initial_energy.total, 3),
            "transforms": list(result.transforms_applied),
            "transforms_used": list(result.transforms_applied),
            "transforms_applied": list(result.transforms_applied),
            "steps": result.steps_taken,
            "steps_taken": result.steps_taken,
            "expansions": result.expansions,
            "elapsed": round(result.elapsed_seconds, 3),
            "elapsed_seconds": round(result.elapsed_seconds, 3),
            "delta": round(delta, 3),
            "reduction_pct": round(reduction_pct, 1),
            "confidence": round(min(1.0, reduction_pct / 100.0), 3),
            "success": success,
            "solve_success": success,
            "domain": detected_domain,
            "trajectory": [round(e, 3) for e in result.energy_trajectory],
            "proof": proof_payload,
            "strategy_hint": strategy_hint.__dict__ if strategy_hint and hasattr(strategy_hint, "__dict__") else None,
            "strategy_hit": bool(strategy_hint and getattr(strategy_hint, "found", False)),
            "recalled_memories": len(recalled),
            "recalled_memory": recalled_memory,
            "memory": self.memory_manager.stats() if self.memory_manager else {},
            "initial": initial_payload,
            "result": result_payload,
            "learned_concepts": [],
            "stage": self.stage.value,
        }

    def _reflect(self, initial_graph, final_graph, domain: str,
                 transforms_applied: List[str] = None):
        """
        P2.1: Verified rule promotion pipeline.
        Every solve → reflect → test on 3 problems → if passes → persist → live transforms.
        """
        if not (self.cpp_reflection_engine or self.py_reflection_engine or self.reflection_engine):
            return
        try:
            # Try C++ reflection first for richer pattern extraction
            if self.cpp_reflection_engine:
                try:
                    cpp_before = _py_graph_to_cpp_graph(initial_graph)
                    cpp_after = _py_graph_to_cpp_graph(final_graph)
                    cpp_rule = self.cpp_reflection_engine.reflect(cpp_before, cpp_after)
                    if cpp_rule and cpp_rule.valid():
                        self._emit_rule_events(
                            getattr(cpp_rule, "name", ""),
                            domain,
                            float(getattr(cpp_rule, "confidence", 0.5)),
                            "",
                        )
                        if self._promote_cpp_rule(cpp_rule, domain):
                            return
                except Exception as e:
                    log.debug(f"C++ reflection failed: {e}")

            py_engine = self.py_reflection_engine or self.reflection_engine
            if py_engine:
                rule = py_engine.reflect(
                    initial_graph, final_graph,
                    transforms_applied=transforms_applied or [],
                    domain=domain,
                )
                if rule and rule.valid():
                    self._emit_rule_events(
                        rule.name,
                        domain,
                        float(getattr(rule, "confidence", 0.5)),
                        getattr(rule, "pattern_description", ""),
                    )
                    # P2.1: Submit to Negotiation Arena before promoting
                    if hasattr(self, 'agent_negotiator') and self.agent_negotiator:
                        self.agent_negotiator.submit_discovery(
                            problem_signature=f"{domain}_{rule.name}",
                            agent_id="brain_reflector",
                            transform=rule,
                            derivation_path=transforms_applied or []
                        )
                    else:
                        self._reflect_and_promote(rule, domain)
        except Exception as e:
            log.debug(f"Reflection failed: {e}")

    def _reflect_and_promote(self, rule, domain: str):
        """P2.1: Test rule on 3 domain problems before promoting. Persist on success."""
        try:
            test_problems = self._get_test_problems_for_domain(domain, n=3)
            passes = self._test_rule_on_problems(rule, test_problems)
            if passes >= 2:  # 2/3 threshold
                if self._promote_python_rule(rule, domain):
                    # Persist: save concept registry to disk
                    if self.concept_registry and hasattr(self.concept_registry, 'save'):
                        try:
                            self.concept_registry.save()
                        except Exception:
                            pass
                    # Immediately refresh live transforms
                    self._refresh_transforms()
                    log.info(f"Rule PROMOTED+PERSISTED: {rule.name} ({passes}/3 tests)")
            else:
                log.debug(f"Rule REJECTED: {rule.name} ({passes}/3 tests)")
        except Exception as e:
            log.debug(f"_reflect_and_promote failed: {e}")
            self._promote_python_rule(rule, domain)  # fallback: promote anyway

    def _get_test_problems_for_domain(self, domain: str, n: int = 3,
                                       rule=None) -> List[str]:
        """Get n test problems from the same domain for rule verification.
        When rule is given, generates problems specifically matching the rule pattern."""
        # Rule-specific test problems (highest quality)
        if rule is not None:
            pattern_desc = getattr(rule, 'pattern_description', '') or ''
            op = getattr(rule, 'operator_involved', '') or ''
            role_problems = self._problems_for_rule_pattern(op, pattern_desc, domain, n)
            if len(role_problems) >= 2:
                return role_problems

        # Domain-aware problem generator
        if not hasattr(self, '_problem_gen'):
            try:
                from sare.curriculum.problem_generator import ProblemGenerator
                self._problem_gen = ProblemGenerator()
            except Exception:
                return []
        try:
            batch = self._problem_gen.generate_for_domain(domain, n=n)
            problems = [p['expression'] for p in batch]
            if len(problems) >= 2:
                return problems
        except Exception:
            pass

        # Developmental curriculum fallback
        if self.developmental_curriculum:
            try:
                d = self.developmental_curriculum.domains.get(domain)
                if d and d.problems:
                    return [p.expression for p in list(d.problems)[:n]]
            except Exception:
                pass

        return list(self.engine.EXAMPLE_PROBLEMS.values())[:n]

    def _problems_for_rule_pattern(self, op: str, pattern: str, domain: str,
                                    n: int) -> List[str]:
        """Generate test problems that specifically exercise a discovered rule pattern."""
        vars = ['x', 'y', 'a']
        problems = []
        # Identity: op(x, element) → x
        if 'identity' in pattern or '0' in pattern:
            for v in vars[:n]:
                for elem in ['0']:
                    if op in ('+', 'add'):
                        problems.append(f"{v} + {elem}")
                        problems.append(f"{elem} + {v}")
                    elif op in ('*', 'mul'):
                        problems.append(f"{v} * 1")
        # Annihilation: op(x, absorb) → absorb
        if 'annihilation' in pattern:
            for v in vars[:n]:
                if op in ('*', 'mul'):
                    problems.append(f"{v} * 0")
                elif op in ('and',):
                    problems.append(f"{v} and false")
        # Involution: op(op(x)) → x
        if 'double' in pattern or 'elimination' in pattern:
            for v in vars[:n]:
                problems.append(f"not not {v}")
                problems.append(f"-(-{v})")
        # Constant fold
        if 'fold' in pattern:
            problems = ["3 + 4", "2 * 5", "10 - 3"]
        return problems[:n]

    def _test_rule_on_problems(self, rule, problems: List[str]) -> int:
        """Test a rule: returns count of problems where rule reduces energy."""
        t = self._rule_to_transform(rule)
        # If rule can't become a transform, try matching it directly via its pattern
        if not t:
            return 0
        successes = 0
        for expr in problems:
            try:
                g = self.engine.build_expression_graph(expr)
                matches = t.match(g)
                if matches:
                    ng, _ = t.apply(g, matches[0])
                    if self.energy.compute(ng).total < self.energy.compute(g).total:
                        successes += 1
            except Exception:
                pass
        return successes

    def _detect_domain(self, graph, hint: str = "general") -> str:
        """Detect the problem domain from graph structure."""
        if hint != "general":
            return hint

        # Heuristic: check operator labels
        labels = set()
        for n in graph.nodes:
            if n.type == "operator":
                labels.add(n.label)

        # Check specific domains BEFORE generic logic
        if labels & {"d/dx", "derivative", "diff", "integral"}:
            return "calculus"
        if labels & {"sin", "cos", "tan", "log", "sqrt"}:
            return "trigonometry"
        if labels & {"P", "prob", "p"}:
            return "probability"
        if labels & {"angle_sum", "sum_angles"}:
            return "geometry"
        if labels & {"→", "implies", "==>"}:
            return "propositional"
        if labels & {"union", "intersect", "∪", "∩"}:
            return "set_theory"
        if labels & {"and", "or", "not", "implies", "AND", "OR", "NOT"}:
            return "logic_basics"

        # Structural detection: factoring pattern a*b + a*c
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("+", "add"):
                children = graph.outgoing(n.id)
                if len(children) == 2:
                    c1 = graph.get_node(children[0].target)
                    c2 = graph.get_node(children[1].target)
                    if (c1 and c2 and c1.type == "operator" and c1.label in ("*", "mul") and
                        c2.type == "operator" and c2.label in ("*", "mul")):
                        # Check if they share a common factor
                        c1_children = graph.outgoing(c1.id)
                        c2_children = graph.outgoing(c2.id)
                        if len(c1_children) == 2 and len(c2_children) == 2:
                            c1_labels = [graph.get_node(e.target).label for e in c1_children]
                            c2_labels = [graph.get_node(e.target).label for e in c2_children]
                            if any(l in c2_labels for l in c1_labels):
                                return "factoring"

        # Structural detection: distribution pattern a * (b + c)
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("*", "mul"):
                children = graph.outgoing(n.id)
                if len(children) == 2:
                    c1 = graph.get_node(children[0].target)
                    c2 = graph.get_node(children[1].target)
                    for a_node, plus_node in [(c1, c2), (c2, c1)]:
                        if a_node and plus_node and plus_node.type == "operator" and plus_node.label in ("+", "add"):
                            return "distribution"

        if labels & {"="}:
            return "algebra"
        return "arithmetic"

    # ─────────────────────────────────────────────────────────────────────────
    #  Hierarchical Planner (sub-goal decomposition without LLM)
    # ─────────────────────────────────────────────────────────────────────────

    def _plan_subgoals(self, graph, domain: str) -> List[str]:
        """
        Break a complex problem into sub-goals using own knowledge.
        No LLM — uses structural analysis of the graph.

        Strategy: identify reducible sub-expressions and order them
        from leaves inward (bottom-up simplification).
        """
        subgoals = []
        # Find operator nodes by depth (leaves first)
        operators = [n for n in graph.nodes if n.type == "operator"]
        if len(operators) <= 1:
            return []  # Simple problem, no decomposition needed

        # Score by depth: operators with only leaf children go first
        def leaf_depth(op_node):
            children = graph.outgoing(op_node.id)
            child_nodes = [graph.get_node(e.target) for e in children]
            leaf_count = sum(1 for c in child_nodes if c and c.type in ("constant", "variable"))
            return -leaf_count  # more leaves = shallower = process first

        operators.sort(key=leaf_depth)
        for op in operators[:5]:
            children = graph.outgoing(op.id)
            child_labels = []
            for e in children:
                c = graph.get_node(e.target)
                if c:
                    child_labels.append(c.label or c.type)
            expr = f"{op.label}({', '.join(child_labels)})"
            subgoals.append(f"simplify {expr}")

        return subgoals

    # ─────────────────────────────────────────────────────────────────────────
    #  Conjecture Generator
    # ─────────────────────────────────────────────────────────────────────────

    def generate_conjectures(self, n: int = 5) -> List[dict]:
        """
        Generate conjectures from learned schemas and test them.

        A conjecture is: "Pattern P always reduces energy via transform T."
        Conjectures that pass testing become promoted rules.
        """
        conjectures = []

        # Source 1: WorldModel v3 imagination
        if self.world_model_v3:
            try:
                domains = set()
                for ep in self.world_model_v3._solve_history[-100:]:
                    domains.add(ep.get("domain", "general"))
                for domain in list(domains)[:3]:
                    hyps = self.world_model_v3.imagine(domain)
                    for h in hyps[:2]:
                        conjectures.append({
                            "type": "imagination",
                            "hypothesis": h.get("hypothesis", ""),
                            "plausibility": h.get("plausibility", 0),
                            "domain": domain,
                            "source": "world_model_v3",
                        })
            except Exception:
                pass

        # Source 2: Transfer engine hypotheses
        if self.transfer_engine:
            try:
                hyps = self.transfer_engine.generate_hypotheses()
                for h in hyps[:3]:
                    conjectures.append({
                        "type": "transfer",
                        "hypothesis": h.proposed_pattern,
                        "plausibility": h.confidence,
                        "domain": h.target_domain,
                        "source": "transfer_engine",
                        "source_domain": h.source_domain,
                    })
            except Exception:
                pass

        # Source 3: Schema-based — if a schema has high confidence,
        # conjecture it applies to unseen problems
        if self.world_model_v3:
            try:
                for sig, schema in self.world_model_v3._schemas.items():
                    if schema.confidence > 0.7 and schema.use_count >= 3:
                        conjectures.append({
                            "type": "schema_generalization",
                            "hypothesis": f"Schema '{schema.name}' generalizes to new problems in {schema.domain}",
                            "plausibility": schema.confidence * 0.8,
                            "domain": schema.domain,
                            "source": "schema_induction",
                            "invariants": schema.invariants,
                        })
            except Exception:
                pass

        # Source 4: CausalChainDetector — chains with high confidence suggest
        # composite rules: "T1 always enables T2 → there may be a rule that does both"
        if hasattr(self, '_causal_chain_detector'):
            try:
                chains = self._causal_chain_detector.get_chains(min_confidence=0.55, min_length=2)
                for chain in chains[:3]:
                    plausibility = chain.confidence * (1.1 if chain.cross_domain else 1.0)
                    conjectures.append({
                        "type": "causal_chain_conjecture",
                        "hypothesis": (
                            f"Chain {chain.name} in {chain.domain}: "
                            f"applying {chain.steps[0]} always enables {chain.steps[-1]}"
                        ),
                        "plausibility": min(0.9, plausibility),
                        "domain": chain.domain,
                        "source": "causal_chain_detector",
                        "chain": chain.steps,
                        "cross_domain": chain.cross_domain,
                    })
            except Exception:
                pass

        # Source 5: WorldModel beliefs with high confidence — generalize to conjectures
        if self.world_model_v3:
            try:
                strong_beliefs = [
                    (k, b) for k, b in self.world_model_v3._beliefs.items()
                    if b.confidence > 0.75 and b.evidence_for >= 3
                    and k.startswith("causal_sequence:")
                ]
                for key, belief in sorted(strong_beliefs, key=lambda x: -x[1].confidence)[:2]:
                    bigram = key.replace("causal_sequence:", "")
                    val = belief.value if hasattr(belief, 'value') else {}
                    t1 = val.get("cause", bigram.split("→")[0] if "→" in bigram else bigram)
                    t2 = val.get("effect", "")
                    domain = val.get("domain", "general")
                    conjectures.append({
                        "type": "belief_generalization",
                        "hypothesis": (
                            f"Frequently observed: {t1} precedes {t2} "
                            f"in {domain} — may be a composite rule"
                        ),
                        "plausibility": belief.confidence * 0.8,
                        "domain": domain,
                        "source": "world_model_beliefs",
                        "t1": t1,
                        "t2": t2,
                    })
            except Exception:
                pass

        # Source 6: Analogy — if concept A has symbolic rules in domain D1,
        # conjecture the same rule pattern applies to analogous concept in D2
        if self.concept_graph:
            try:
                for c_name, concept in self.concept_graph._concepts.items():
                    if concept.is_well_grounded() and concept.symbolic_rules:
                        for rule in concept.symbolic_rules[:2]:
                            # Find analogous concepts in other domains
                            for other_name, other in self.concept_graph._concepts.items():
                                if other_name != c_name and other.domain != concept.domain:
                                    conjectures.append({
                                        "type": "analogy_conjecture",
                                        "hypothesis": (
                                            f"By analogy: rule '{rule}' from '{c_name}' ({concept.domain}) "
                                            f"may generalise to '{other_name}' ({other.domain})"
                                        ),
                                        "plausibility": concept.confidence * 0.6,
                                        "domain": other.domain,
                                        "source": "concept_graph_analogy",
                                        "source_concept": c_name,
                                        "target_concept": other_name,
                                        "rule": rule,
                                    })
            except Exception:
                pass

        # Source 7: Counterfactual — perturb a known rule to generate testable variant
        # e.g. "x + 0 = x" → conjecture: "x + 1 = x + 1" or "x * 0 = x" (falsifiable)
        if self.concept_graph:
            try:
                import random as _rand
                for c_name, concept in self.concept_graph._concepts.items():
                    if concept.symbolic_rules:
                        rule = _rand.choice(concept.symbolic_rules)
                        # Perturb: replace identity element with a different constant
                        perturbed = rule
                        for orig, alt in [(' 0 ', ' 1 '), (' 1 ', ' 0 '),
                                          (' true ', ' false '), (' false ', ' true ')]:
                            if orig in rule:
                                perturbed = rule.replace(orig, alt, 1)
                                break
                        if perturbed != rule:
                            conjectures.append({
                                "type": "counterfactual_conjecture",
                                "hypothesis": (
                                    f"Counterfactual: does '{perturbed}' hold? "
                                    f"(perturbed from '{rule}' in '{c_name}')"
                                ),
                                "plausibility": 0.15,  # deliberately low — test and falsify
                                "domain": concept.domain,
                                "source": "counterfactual_perturbation",
                                "original_rule": rule,
                                "perturbed_rule": perturbed,
                                "concept": c_name,
                            })
            except Exception:
                pass

        conjectures.sort(key=lambda x: x.get("plausibility", 0), reverse=True)
        return conjectures[:n]

    # ─────────────────────────────────────────────────────────────────────────
    #  Perception: Ingest real-world data
    # ─────────────────────────────────────────────────────────────────────────

    def ingest(self, data: str, kind: str = "text", source: str = "user") -> dict:
        """
        Ingest real-world data, extract problems, feed into learning loop.
        Supports: text, csv, json, textbook, file path.
        """
        if not self.perception_engine:
            return {"error": "Perception engine not loaded"}

        if kind == "csv":
            result = self.perception_engine.ingest_csv(data, source)
        elif kind == "json":
            import json as _j
            try:
                parsed = _j.loads(data) if isinstance(data, str) else data
            except Exception:
                parsed = {"raw": data}
            result = self.perception_engine.ingest_json(parsed, source)
        elif kind == "textbook":
            result = self.perception_engine.ingest_textbook(data, source)
        elif kind == "file":
            result = self.perception_engine.ingest_file(data)
        else:
            result = self.perception_engine.ingest_text(data, source)

        # Feed extracted problems into developmental curriculum
        problems_added = 0
        if result.problems_extracted and self.developmental_curriculum:
            from sare.curriculum.developmental import CurriculumProblem, Difficulty
            for ep in result.problems_extracted:
                diff = Difficulty.EASY if ep.difficulty < 0.3 else (
                    Difficulty.MEDIUM if ep.difficulty < 0.6 else Difficulty.HARD
                )
                # Add to the most relevant domain, or create "ingested" bucket
                target_domain = None
                for d in self.developmental_curriculum.domains.values():
                    if d.name == ep.domain:
                        target_domain = d
                        break
                if target_domain and target_domain.unlocked:
                    target_domain.problems.append(CurriculumProblem(
                        expression=ep.expression, domain=target_domain.name,
                        difficulty=diff, target_rules=[], hint=ep.context,
                    ))
                    problems_added += 1

        # Feed extracted facts into WorldModel v3
        facts_added = 0
        if result.facts_extracted and self.world_model_v3:
            for fact in result.facts_extracted:
                try:
                    from sare.memory.world_model_v3 import CausalLink
                    link = CausalLink(
                        cause=fact["subject"], effect=fact["object"],
                        mechanism=fact["relation"], domain=fact.get("domain", "general"),
                        confidence=fact.get("confidence", 0.7),
                    )
                    self.world_model_v3._causal_links[link.key] = link
                    facts_added += 1
                except Exception:
                    pass

        return {
            "source": result.source, "kind": result.kind,
            "problems_extracted": len(result.problems_extracted),
            "problems_added_to_curriculum": problems_added,
            "facts_extracted": len(result.facts_extracted),
            "facts_added_to_worldmodel": facts_added,
            "graph_nodes": result.graph_nodes, "graph_edges": result.graph_edges,
            "elapsed": round(result.elapsed, 3),
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  Deep Transfer: Synthesize new transforms
    # ─────────────────────────────────────────────────────────────────────────

    def synthesize_transforms(self) -> dict:
        """
        Use the TransferEngine to find missing roles per domain,
        then synthesize working transforms for them.
        """
        if not self.transform_synthesizer or not self.transfer_engine:
            return {"error": "Transfer engine or synthesizer not loaded"}

        new = self.transform_synthesizer.synthesize_all_missing(self.transfer_engine)
        if new:
            # Add synthesized transforms to the live transform list
            self._refresh_transforms()
            log.info(f"Synthesized {len(new)} new transforms: {[t.name for t in new]}")

        return {
            "synthesized": len(new),
            "total": self.transform_synthesizer.stats()["total_synthesized"],
            "transforms": [t.to_dict() for t in new],
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  Knowledge Reorganization
    # ─────────────────────────────────────────────────────────────────────────

    def reorganize_knowledge(self) -> dict:
        """
        Periodically restructure the knowledge base:
        - Merge redundant causal links
        - Prune low-confidence beliefs
        - Discover new analogies
        - Promote high-confidence schemas
        """
        report = {"merged": 0, "pruned": 0, "analogies": 0, "promoted": 0}

        # 1. Prune stale/weak beliefs in WorldModel v3
        if self.world_model_v3:
            try:
                weak = [k for k, b in self.world_model_v3._beliefs.items()
                        if b.confidence < 0.15 and b.evidence_for + b.evidence_against > 5]
                for k in weak:
                    del self.world_model_v3._beliefs[k]
                report["pruned"] = len(weak)
            except Exception:
                pass

        # 2. Merge redundant causal links (same mechanism + domain)
        if self.world_model_v3:
            try:
                by_mechanism: Dict[str, List[str]] = {}
                for key, link in self.world_model_v3._causal_links.items():
                    mkey = f"{link.mechanism}:{link.domain}"
                    by_mechanism.setdefault(mkey, []).append(key)
                for mkey, keys in by_mechanism.items():
                    if len(keys) > 3:
                        # Keep top 3 by confidence, remove rest
                        sorted_keys = sorted(
                            keys, key=lambda k: self.world_model_v3._causal_links[k].confidence,
                            reverse=True)
                        for k in sorted_keys[3:]:
                            del self.world_model_v3._causal_links[k]
                            report["merged"] += 1
            except Exception:
                pass

        # 3. Discover new analogies
        if self.world_model_v3:
            try:
                new = self.world_model_v3.discover_analogies()
                report["analogies"] = len(new)
            except Exception:
                pass

        # 4. Generate transfer hypotheses
        if self.transfer_engine:
            try:
                hyps = self.transfer_engine.generate_hypotheses()
                report["transfer_hypotheses"] = len(hyps)
            except Exception:
                pass

        # 5. VERIFIED TRANSFER: test hypotheses on actual problems
        if self.transfer_engine and self.transform_synthesizer:
            try:
                verified = self._verify_transfer_hypotheses()
                report["transfers_verified"] = verified
            except Exception:
                pass

        # C++ plasticity (if bindings available)
        if self.cpp_module_generator:
            try:
                report["plasticity_macros_cpp"] = self._run_cpp_plasticity()
            except Exception:
                pass

        # P2.2: Python plasticity — analyze failure patterns, propose macros
        try:
            report["plasticity_macros_py"] = self._python_plasticity()
        except Exception:
            pass

        # P3.4: GoalSetter — generate proactive learning goals
        try:
            self._generate_proactive_goals()
        except Exception:
            pass

        # P4.3: StuckDetector — check if system is looping
        try:
            stuck = self._detect_stuck()
            if stuck:
                report["stuck_detected"] = stuck
        except Exception:
            pass

        # 6. Refresh transforms with anything new
        self._refresh_transforms()

        # P4.4: Consolidate KnowledgeGraph
        if self.knowledge_graph:
            try:
                kg_report = self.knowledge_graph.consolidate()
                report["kg_consolidation"] = kg_report
            except Exception:
                pass

        # P4.5: Memory consolidation (hippocampus-style sleep replay)
        try:
            mem_report = self._consolidate_memory()
            report["memory_consolidation"] = mem_report
        except Exception:
            pass

        return report

    def _consolidate_memory(self) -> dict:
        """
        Hippocampus-style memory consolidation — standalone, no web.py dependency.

        Three phases:
          1. Sleep-replay: replay recent successful episodes, boost transform confidence
          2. Episodic abstraction: find high-frequency transform sequences → store as macro hints
          3. Causal strengthening: use CausalChainDetector to reinforce KG causal links
        """
        report = {"replayed": 0, "abstractions": 0, "chains_strengthened": 0, "rules_boosted": 0}

        # ── Phase 1: Sleep-replay rule confidence boosting ────────────────────
        if self.memory_manager:
            try:
                from collections import Counter
                recent = self.memory_manager.recent_episodes(200)
                successes = [ep for ep in recent if getattr(ep, 'success', False)]
                t_freq: Counter = Counter()
                for ep in successes:
                    for t in getattr(ep, 'transform_sequence', []):
                        t_freq[t] += 1
                report["replayed"] = len(successes)

                # Boost confidence for frequently-used successful transforms
                if self.concept_registry:
                    for t_name, freq in t_freq.most_common(10):
                        try:
                            rule = self.concept_registry.get_rule(t_name)
                            if rule:
                                old_conf = getattr(rule, 'confidence', 0.7)
                                boost = min(0.02 * (freq / max(1, len(successes))), 0.05)
                                rule.confidence = min(1.0, old_conf + boost)
                                report["rules_boosted"] += 1
                        except Exception:
                            pass
            except Exception:
                pass

        # ── Phase 2: Episodic abstraction — frequent 2-grams in solve sequences ─
        if self.memory_manager and self.world_model_v3:
            try:
                from collections import Counter
                recent = self.memory_manager.recent_episodes(300)
                bigram_counts: Counter = Counter()
                domain_for_bigram: dict = {}
                for ep in recent:
                    if not getattr(ep, 'success', False):
                        continue
                    seq = getattr(ep, 'transform_sequence', [])
                    domain = getattr(ep, 'domain', 'general')
                    for i in range(len(seq) - 1):
                        key = f"{seq[i]}→{seq[i+1]}"
                        bigram_counts[key] += 1
                        domain_for_bigram[key] = domain

                # Store frequent bigrams as causal abstractions in WorldModel
                for bigram, count in bigram_counts.most_common(5):
                    if count >= 3:
                        t1, t2 = bigram.split("→")
                        domain = domain_for_bigram.get(bigram, 'general')
                        try:
                            self.world_model_v3.update_belief(
                                key=f"causal_sequence:{bigram}",
                                value={"cause": t1, "effect": t2, "count": count, "domain": domain},
                                confidence=min(0.9, 0.4 + 0.05 * count),
                                domain=domain,
                            )
                            report["abstractions"] += 1
                        except Exception:
                            pass
            except Exception:
                pass

        # ── Phase 3: Causal chain strengthening via CausalChainDetector ──────
        if hasattr(self, '_causal_chain_detector') and self.knowledge_graph:
            try:
                chains = self._causal_chain_detector.get_chains(min_confidence=0.5)
                for chain in chains[:10]:
                    for i in range(len(chain.steps) - 1):
                        try:
                            self.knowledge_graph.add_causal_link(
                                cause=chain.steps[i],
                                effect=chain.steps[i + 1],
                                mechanism="consolidation_replay",
                                domain=chain.domain,
                                confidence=min(0.95, chain.confidence + 0.05),
                            )
                            report["chains_strengthened"] += 1
                        except Exception:
                            pass
            except Exception:
                pass

        log.debug(
            f"Memory consolidation: replayed={report['replayed']} "
            f"abstractions={report['abstractions']} "
            f"chains={report['chains_strengthened']} "
            f"rules_boosted={report['rules_boosted']}"
        )
        return report

    def _consolidate_concepts(self) -> dict:
        """
        Concept consolidation cycle — runs every 5 learn cycles.

        Three phases:
          1. Rule abstraction: for all concepts with new examples, run abstract_from_examples
          2. Novel discovery: map unseen transform names → new concept candidates
          3. Goal propagation: mark GoalPlanner goals complete when concepts become well-grounded
        """
        if not self.concept_graph:
            return {}
        report = {"abstractions": 0, "novel_concepts": 0, "goals_completed": 0}

        # ── Phase 1: Abstract rules from accumulated grounded examples ────────
        for concept_name, concept in self.concept_graph._concepts.items():
            if concept.ground_count() >= 2:
                try:
                    new_rules = self.concept_graph.abstract_from_examples(concept_name)
                    if new_rules:
                        report["abstractions"] += len(new_rules)
                        log.debug(f"Concept '{concept_name}': abstracted {len(new_rules)} rules")
                except Exception:
                    pass

        # ── Phase 2: Novel concept discovery from recently used transforms ────
        # Find transforms used in last N solves that don't map to any concept
        if self.memory_manager:
            try:
                from collections import Counter
                recent = self.memory_manager.recent_episodes(50)
                t_freq: Counter = Counter()
                for ep in recent:
                    for t in getattr(ep, 'transform_sequence', []):
                        t_freq[t] += 1

                for t_name, freq in t_freq.most_common(10):
                    if freq >= 3:
                        existing = self.concept_graph.concept_for_transform(t_name)
                        if not existing:
                            # Discover a novel concept from this transform
                            parts = t_name.replace('_', ' ').split()
                            meaning = f"learned from transform: {t_name} (used {freq}×)"
                            self.concept_graph._concepts[t_name] = \
                                __import__('sare.concept.concept_graph', fromlist=['Concept'])\
                                .Concept(
                                    name=t_name,
                                    meaning=meaning,
                                    symbol='?',
                                    domain='general',
                                    confidence=0.3 + 0.05 * freq,
                                )
                            report["novel_concepts"] += 1
                            log.debug(f"Novel concept discovered: '{t_name}' (used {freq}×)")
            except Exception:
                pass

        # ── Phase 3: Mark GoalPlanner goals complete for well-grounded concepts ─
        if self.goal_planner:
            try:
                for plan in list(self.goal_planner.active_plans()):
                    self._check_goal_completion(plan)
                report["goals_completed"] = len(
                    [g for g in self.goal_planner._all_nodes.values()
                     if g.status.value == 'completed']
                )
            except Exception:
                pass

        # ── Phase 4: Cross-domain concept merging ────────────────────────────────
        try:
            merges = self.concept_graph.run_cross_domain_merge()
            if merges > 0:
                report["cross_domain_merges"] = merges
                log.info(f"Concept consolidation: {merges} cross-domain merges")
        except Exception as _e:
            log.debug(f"Cross-domain merge: {_e}")

        # ── Phase 5: Auto-achieve SelfModel goals when domain crosses 70% ────────
        if self.self_model:
            try:
                for goal in list(self.self_model._learning_goals):
                    if goal.status != 'active' or goal.reason not in ('weakness', 'curiosity'):
                        continue
                    dc = self.self_model._domains.get(goal.domain)
                    if dc and dc.recent_rate >= 0.70:
                        self.self_model.mark_goal_achieved(goal.domain, goal.reason)
                        log.info(f"SelfModel: goal '{goal.domain}' achieved "
                                 f"(rate={dc.recent_rate:.0%})")
                        self.events.emit(
                            'goal_achieved',
                            {"domain": goal.domain, "rate": dc.recent_rate},
                            "self_model",
                        )
            except Exception as _e:
                log.debug(f"Goal auto-achieve: {_e}")

        # Persist updated concept graph
        try:
            self.concept_graph.save()
        except Exception:
            pass

        return report

    def _check_goal_completion(self, node) -> bool:
        """Recursively check if a goal's completion condition is met by current concept state."""
        if node.status.value in ('completed', 'failed'):
            return True
        # Check if this goal's concept is now well-grounded
        concept_name = node.metadata.get('concept')
        if concept_name and self.concept_graph:
            c = self.concept_graph._concepts.get(concept_name)
            if c:
                from sare.concept.goal_planner import GoalType
                if node.goal_type == GoalType.RUN_EXPERIMENT and c.ground_count() >= 3:
                    self.goal_planner.mark_complete(node.goal_id)
                    return True
                elif node.goal_type == GoalType.ABSTRACT_RULE and len(c.symbolic_rules) >= 1:
                    self.goal_planner.mark_complete(node.goal_id)
                    return True
                elif node.goal_type == GoalType.VERIFY_RULE and \
                        c.ground_count() >= 5 and c.is_well_grounded():
                    self.goal_planner.mark_complete(node.goal_id)
                    return True
        # Recurse into subgoals
        for sg in node.subgoals:
            self._check_goal_completion(sg)
        return False

    def _python_plasticity(self) -> int:
        """P2.2: Analyze failure patterns and propose new macro transforms."""
        if not self.memory_manager:
            return 0
        try:
            recent = self.memory_manager.recent_episodes(100)
            failures = [ep for ep in recent if not getattr(ep, 'success', True)]
            if len(failures) < 10:
                return 0

            from collections import Counter
            t_freq: Counter = Counter()
            for ep in failures:
                for t in getattr(ep, 'transform_sequence', []):
                    t_freq[t] += 1

            # Top transforms tried repeatedly before failures
            common = [t for t, n in t_freq.most_common(6) if n >= 3]
            if len(common) < 2:
                return 0

            # Get existing transform names to only combine valid ones
            valid_names = {t.name() for t in self.transforms}
            common = [t for t in common if t in valid_names]
            if len(common) < 2:
                return 0

            from sare.meta.macro_registry import MacroSpec, upsert_macros
            macros = []
            for i in range(min(2, len(common) - 1)):
                name = f"plastic_{common[i]}_then_{common[i+1]}"
                macros.append(MacroSpec(name=name, steps=[common[i], common[i + 1]]))

            if macros:
                upsert_macros(macros)
                self._refresh_transforms()
                for spec in macros:
                    self.events.emit(Event.MACRO_CREATED,
                                     {"name": spec.name, "steps": spec.steps,
                                      "source": "python_plasticity"}, "plasticity")
                log.info(f"Python plasticity: proposed {len(macros)} macros")
            return len(macros)
        except Exception as e:
            log.debug(f"_python_plasticity failed: {e}")
            return 0

    def _generate_proactive_goals(self):
        """P3.4: Generate proactive learning goals based on competence gaps."""
        if not self.goal_setter:
            return
        try:
            from sare.meta.goal_setter import GoalType

            # Avoid duplicate goals — check active goals first
            try:
                active_descs = {g.description for g in (self.goal_setter.active_goals() or [])}
            except Exception:
                active_descs = set()

            # Find domains with low competence → generate mastery goals
            if self.self_model:
                try:
                    competence = self.self_model.get_all_competence()
                    low = [(d, c) for d, c in competence.items() if c < 0.4]
                    for domain, comp in sorted(low, key=lambda x: x[1])[:2]:
                        desc = f"Master {domain} (competence: {comp:.0%})"
                        if desc in active_descs:
                            continue
                        self.goal_setter.add_goal(
                            type=GoalType.DOMAIN_MASTERY,
                            description=desc,
                            target=0.7,
                            domain=domain,
                            priority=1 if comp < 0.2 else 3,
                        )
                        log.debug(f"GoalSetter: mastery goal for '{domain}'")
                except Exception:
                    pass

            # Generate rule discovery goal if not enough rules known
            rules_known = self._stats.get("rules_promoted", 0)
            if rules_known < 5:
                desc = f"Discover 5 new rules (currently {rules_known})"
                if desc not in active_descs:
                    try:
                        self.goal_setter.add_goal(
                            type=GoalType.RULE_DISCOVERY,
                            description=desc,
                            target=5.0,
                            domain=None,
                            priority=2,
                        )
                    except Exception:
                        pass
        except Exception as e:
            log.debug(f"_generate_proactive_goals failed: {e}")

    def _detect_stuck(self) -> Optional[dict]:
        """P4.3: Detect when the system is looping on the same problem type."""
        if not self.memory_manager:
            return None
        try:
            recent = self.memory_manager.recent_episodes(30)
            if len(recent) < 10:
                return None
            failures = [ep for ep in recent if not getattr(ep, 'success', True)]
            if len(failures) < 8:
                return None
            # All recent failures — check if they share a structural feature
            problem_ids = [getattr(ep, 'problem_id', '') for ep in failures]
            from collections import Counter
            # Check if same expression repeated
            repeated = [(expr, cnt) for expr, cnt in Counter(problem_ids).items() if cnt >= 3]
            if repeated:
                stuck_on = repeated[0][0]
                log.warning(f"STUCK: repeated failure on '{stuck_on}' ({repeated[0][1]} times)")
                # Auto-response: add to curriculum skip list
                return {"reason": "repeated_failure", "expression": stuck_on,
                        "count": repeated[0][1]}
            return None
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────────────────────
    #  Background Self-Learning Thread
    # ─────────────────────────────────────────────────────────────────────────

    def start_auto_learn(self, interval: float = 2.0, problems_per_cycle: int = 5):
        """Start background autonomous learning thread."""
        if hasattr(self, '_auto_learn_thread') and self._auto_learn_thread and self._auto_learn_thread.is_alive():
            log.info("Auto-learn already running")
            return

        self._auto_learn_stop = threading.Event()
        self._auto_learn_stats = {
            "running": True, "cycles": 0, "total_solved": 0,
            "total_attempted": 0, "interval": interval,
            "retry_successes": 0, "avg_surprise": 0.0,
            "failure_reasons": {},
        }

        def _learn_loop():
            cycle = 0
            while not self._auto_learn_stop.is_set():
                cycle += 1
                try:
                    results = self.learn_cycle(n=problems_per_cycle)
                    successes = sum(1 for r in results if r.get("success"))
                    self._auto_learn_stats["cycles"] = cycle
                    self._auto_learn_stats["total_solved"] += successes
                    self._auto_learn_stats["total_attempted"] += len(results)

                    # Update curriculum
                    if self.developmental_curriculum:
                        for r in results:
                            try:
                                self.developmental_curriculum.record_attempt(
                                    r.get("expression", ""),
                                    r.get("domain", "general"),
                                    r.get("success", False),
                                    r.get("delta", 0))
                            except Exception:
                                pass

                    # Every 10 cycles: sleep consolidation + deep transfer
                    if cycle % 10 == 0:
                        self.reorganize_knowledge()
                        # Deep transfer: synthesize new transforms
                        try:
                            self.synthesize_transforms()
                        except Exception:
                            pass
                        self._refresh_transforms()

                    # Save periodically
                    if cycle % 20 == 0:
                        self.save_state()
                        if self.developmental_curriculum:
                            try:
                                self.developmental_curriculum.save()
                            except Exception:
                                pass
                        if self.world_model_v3:
                            try:
                                self.world_model_v3.save()
                            except Exception:
                                pass

                except Exception as e:
                    log.error(f"Auto-learn cycle {cycle} error: {e}")

                self._auto_learn_stop.wait(interval)

            self._auto_learn_stats["running"] = False
            log.info("Auto-learn stopped")

        self._auto_learn_thread = threading.Thread(
            target=_learn_loop, daemon=True, name="AutoLearn"
        )
        self._auto_learn_thread.start()
        log.info(f"Auto-learn started: every {interval}s, {problems_per_cycle} problems/cycle")

    def stop_auto_learn(self):
        """Stop background learning."""
        if hasattr(self, '_auto_learn_stop'):
            self._auto_learn_stop.set()
            log.info("Auto-learn stop requested")

    def auto_learn_status(self) -> dict:
        """Get background learning status."""
        if hasattr(self, '_auto_learn_stats'):
            return dict(self._auto_learn_stats)
        return {"running": False, "cycles": 0, "total_solved": 0, "total_attempted": 0}

    # ─────────────────────────────────────────────────────────────────────────
    #  Learning Cycle
    # ─────────────────────────────────────────────────────────────────────────

    def learn_cycle(self, n: int = 5) -> List[dict]:
        """
        Run n autonomous learning cycles with failure-driven adaptation:
        1. Pick problem (WorldModel-guided if available)
        2. Predict outcome (builds prediction model)
        3. Solve
        4. Compare prediction vs reality (learn from surprise)
        5. On failure: analyze, retry with alternative, record what to avoid
        """
        results = []

        for i in range(n):
            problem_expr = self._pick_learning_problem()
            if not problem_expr:
                break

            # WorldModel prediction BEFORE solving (tracks surprise)
            prediction = None
            if self.world_model_v3:
                try:
                    domain_guess = self._detect_domain(
                        self.engine.build_expression_graph(problem_expr), "general"
                    )
                    t_names = [t.name() for t in self.transforms[:10]]
                    prediction = self.world_model_v3.predict(
                        problem_expr, domain_guess, t_names
                    )
                except Exception:
                    pass

            result = self.solve(problem_expr)
            results.append(result)

            # WorldModel: compare prediction to actual (learn from surprise)
            if prediction and self.world_model_v3:
                try:
                    surprise = self.world_model_v3.record_outcome(
                        prediction,
                        result.get("transforms", []),
                        result.get("delta", 0),
                    )
                    result["surprise"] = surprise
                    # High surprise = most learning value
                    if surprise > 2.0 and self.autobiography:
                        try:
                            self.autobiography.record(
                                event_type="surprise",
                                domain=result.get("domain", "general"),
                                description=f"Surprise {surprise:.1f}x on {problem_expr[:30]}",
                                importance=min(1.0, surprise / 5.0),
                                related_rules=result.get("transforms", [])[:3],
                            )
                        except Exception:
                            pass
                except Exception:
                    pass

            # Homeostasis
            if self.homeostasis:
                try:
                    self.homeostasis.satisfy_drive("curiosity", 0.1)
                except Exception:
                    pass

        # Every 5 cycles: generate + test conjectures (Creativity loop)
        if n >= 5 and n % 5 == 0:
            try:
                promoted = self._test_and_promote_conjectures()
                if promoted > 0:
                    log.info(f"Conjecture cycle: {promoted} conjectures promoted")
                    self._refresh_transforms()
            except Exception as e:
                log.debug(f"Conjecture cycle failed: {e}")

        # Every 5 cycles: concept consolidation — abstract rules from grounded examples
        # and mark GoalPlanner goals complete when concepts become well-grounded
        if n >= 5 and self.concept_graph:
            try:
                self._consolidate_concepts()
            except Exception as e:
                log.debug(f"Concept consolidation: {e}")

        # Every 10 cycles: verify transfers and synthesize new transforms
        if n >= 10 and n % 10 == 0:
            try:
                verified = self._verify_transfer_hypotheses()
                if verified > 0:
                    log.info(f"Transfer verification: {verified} new transfers verified")
                    # Refresh transforms to include newly synthesized ones
                    self._refresh_transforms()
            except Exception as e:
                log.debug(f"Transfer verification failed: {e}")

        # Every 10 cycles: run meta-learning beam-width tuning
        if n >= 10 and n % 10 == 0 and self.meta_learner:
            try:
                if self.meta_learner.should_tune(min_interval_seconds=60.0):
                    best = self.meta_learner.tune_beam_width()
                    self.meta_learner.apply_to_brain(self)
                    log.info(f"MetaLearn tuned: best config = {best.name} "
                             f"(bw={best.beam_width})")
            except Exception as e:
                log.debug(f"MetaLearn tuning failed: {e}")

        # After every cycle: refresh self-model learning goals
        if self.self_model:
            try:
                goals = self.self_model.generate_learning_goals(max_goals=5)
                if goals:
                    log.debug(f"Self-model: {len(goals)} active learning goals")
            except Exception:
                pass

        # Every 5 cycles: run physics simulation → feed new grounded examples
        if n >= 5 and n % 5 == 0 and self.physics_simulator and self.concept_graph:
            try:
                events = self.physics_simulator.run_session(3)
                fed = self.physics_simulator.feed_to_concept_graph(self.concept_graph)
                if fed > 0:
                    log.debug(f"PhysicsSim: {len(events)} events → {fed} concept examples")
                    if self.global_workspace:
                        self.global_workspace.post_event(
                            "physics_event",
                            {"events": len(events), "concepts_fed": fed},
                            source="physics_simulator", salience=0.4,
                        )
            except Exception as e:
                log.debug(f"PhysicsSim cycle: {e}")

        # Every 20 cycles: re-ingest knowledge base (picks up any new entries)
        if n >= 20 and n % 20 == 0 and self.knowledge_ingester and self.concept_graph:
            try:
                result = self.knowledge_ingester.ingest_and_feed(self.concept_graph)
                log.info(f"KnowledgeIngester: {result['concepts_fed']} concepts fed")
                if self.global_workspace:
                    self.global_workspace.post_event(
                        "knowledge_ingested", result,
                        source="knowledge_ingester", salience=0.5,
                    )
            except Exception as e:
                log.debug(f"KnowledgeIngester cycle: {e}")

        # Every cycle: GlobalWorkspace broadcast tick
        if self.global_workspace:
            try:
                self.global_workspace.broadcast_tick(max_messages=3)
            except Exception:
                pass

        # Every cycle: PredictiveWorldLoop — run one predict→act→observe→update cycle
        if self.predictive_loop and results:
            try:
                last = results[-1]
                expr = last.get("expression", "x + 0")
                domain = last.get("domain", "general")
                transforms = last.get("transforms_used", None) or \
                             last.get("transforms", None) or \
                             ["add_zero_elim", "mul_one_elim"]
                self.predictive_loop.run_cycle(
                    expr, domain,
                    lambda e: self.solve(e),
                    transforms if isinstance(transforms, list) else [transforms],
                )
            except Exception as e:
                log.debug(f"PredictiveLoop cycle: {e}")

        # First cycle only: start AutonomousTrainer background thread
        if n == 1 and self.autonomous_trainer and not self.autonomous_trainer._running:
            try:
                self.autonomous_trainer.start(self)
                log.info("AutonomousTrainer background learning started")
                if self.global_workspace:
                    self.global_workspace.post_event(
                        "knowledge_ingested",
                        {"source": "autonomous_trainer", "status": "started"},
                        source="autonomous_trainer", salience=0.5,
                    )
            except Exception as e:
                log.debug(f"AutonomousTrainer start: {e}")

        # Every 3 cycles: AgentSociety deliberation + feed to concept graph
        if n >= 3 and n % 3 == 0 and self.agent_society:
            try:
                result = self.agent_society.deliberation_cycle(
                    engine_fn=lambda e: self.solve(e)
                )
                if result.get("newly_accepted", 0) > 0 and self.concept_graph:
                    fed = self.agent_society.feed_to_concept_graph(self.concept_graph)
                    log.debug(f"AgentSociety: {result['newly_accepted']} accepted, "
                              f"{fed} fed to CG")
                if self.global_workspace and result.get("newly_accepted", 0) > 0:
                    self.global_workspace.post_event(
                        "conjecture_verified",
                        result,
                        source="agent_society", salience=0.6,
                    )
            except Exception as e:
                log.debug(f"AgentSociety cycle: {e}")

        # P4: AgentNegotiator deliberation (every 5 cycles)
        if hasattr(self, 'agent_negotiator') and self.agent_negotiator and n % 5 == 0:
            try:
                truths = self.agent_negotiator.deliberate()
                for t in truths:
                    transform = t["transform"]
                    domain = t["signature"].split("_")[0] if "_" in t["signature"] else "general"
                    self._reflect_and_promote(transform, domain)
                    
                    if self.global_workspace:
                        self.global_workspace.post_event(
                            "network_truth_established",
                            {"signature": t["signature"], "winner": t["winning_agent"], "cost": t["energy_cost"]},
                            source="agent_negotiator", salience=0.8
                        )
            except Exception as e:
                log.debug(f"AgentNegotiator deliberation: {e}")

        # AGI-2: ConjectureEngine tick (every 10 cycles — validate hypotheses)
        if n % 10 == 0 and hasattr(self, 'conjecture_engine') and self.conjecture_engine:
            try:
                ce_result = self.conjecture_engine.tick()
                if ce_result.get("proven_this_tick", 0) > 0:
                    log.info(f"ConjectureEngine: {ce_result['proven_this_tick']} conjectures proven this tick")
            except Exception as e:
                log.debug(f"ConjectureEngine tick: {e}")

        # S29-1: MetaCurriculumEngine tick (every 10 cycles — adaptive domain reweighting)
        if n % 10 == 0 and self.meta_curriculum:
            try:
                mc_result = self.meta_curriculum.tick()
                if mc_result.get("promoted"):
                    log.debug(f"MetaCurriculum promoted: {mc_result['promoted']}")
            except Exception as e:
                log.debug(f"MetaCurriculum tick: {e}")

        if self.global_buffer:
            try:
                self.global_buffer.tick()
            except Exception as e:
                log.debug(f"GlobalBuffer tick: {e}")

        # S25-2: ConceptBlender — discover new blends every 5 cycles
        if n % 5 == 0 and self.concept_blender:
            try:
                new_blends = self.concept_blender.discover_blends(max_new=3)
                if new_blends and self.concept_graph:
                    fed = self.concept_blender.feed_to_concept_graph(self.concept_graph)
                    if fed and self.global_workspace:
                        self.global_workspace.post_event(
                            "cross_domain_merge",
                            {"blends_fed": fed, "blend": new_blends[0].blend_expression()},
                            source="concept_blender", salience=0.65,
                        )
            except Exception as e:
                log.debug(f"ConceptBlender cycle: {e}")

        # S25-4: SensoryBridge — run one grounded physics observation every 2 cycles
        if n % 2 == 0 and self.sensory_bridge:
            try:
                self.sensory_bridge.tick()
            except Exception as e:
                log.debug(f"SensoryBridge tick: {e}")

        # S26-1: DreamConsolidator — offline replay every 7 cycles
        if n % 7 == 0 and self.dream_consolidator:
            try:
                rec = self.dream_consolidator.dream_cycle(max_events=15)
                if rec.causal_edges_found > 0 and self.global_workspace:
                    self.global_workspace.post_event(
                        "dream_insight",
                        {"edges": rec.causal_edges_found, "pattern": rec.top_pattern},
                        source="dream_consolidator", salience=0.55,
                    )
            except Exception as e:
                log.debug(f"DreamConsolidator cycle: {e}")

        # S26-2: AffectiveEnergy — score current problem; bias future selection
        if self.affective_energy and results:
            try:
                last_expr = str(results[-1]) if results else ""
                if last_expr:
                    self.affective_energy.compute(last_expr, domain="general")
            except Exception as e:
                log.debug(f"AffectiveEnergy tick: {e}")

        # S26-3: TransformGenerator — discover + promote every 10 cycles
        if n % 10 == 0 and self.transform_generator:
            try:
                self.transform_generator.generate_candidates(n=3)
                newly = self.transform_generator.promote_best()
                if newly:
                    log.debug(f"TransformGenerator promoted {len(newly)} transforms")
            except Exception as e:
                log.debug(f"TransformGenerator cycle: {e}")

        # S26-4: GenerativeWorldModel — imagination cycle every 4 cycles
        if n % 4 == 0 and self.generative_world:
            try:
                entry = self.generative_world.explore_cycle()
                if entry.get("solved", 0) > 0 and self.global_workspace:
                    self.global_workspace.post_event(
                        "imagination_solved",
                        entry,
                        source="generative_world", salience=0.5,
                    )
            except Exception as e:
                log.debug(f"GenerativeWorldModel cycle: {e}")

        # S26-5: RedTeamAdversary — attack round every 8 cycles
        if n % 8 == 0 and self.red_team:
            try:
                result = self.red_team.run_attack_round(top_k=3)
                if result.get("falsifications", 0) > 0:
                    log.debug(f"RedTeam falsified {result['falsifications']} beliefs")
            except Exception as e:
                log.debug(f"RedTeam cycle: {e}")

        # S26-6: TemporalIdentity — tick every cycle; update session stats every 20
        if self.temporal_identity:
            try:
                self.temporal_identity.tick()
                if n % 20 == 0:
                    stats = self._collect_session_stats()
                    self.temporal_identity.update(stats)
            except Exception as e:
                log.debug(f"TemporalIdentity tick: {e}")

        # S27-1: ContinuousStreamLearner — streams run async; log throughput every 30 cycles
        if n % 30 == 0 and self.continuous_stream:
            try:
                s = self.continuous_stream.summary()
                log.debug(
                    f"ContinuousStream: {s['total_solved']} solved "
                    f"{s['throughput_per_min']}/min "
                    f"pauses={s['interference_pauses']}"
                )
            except Exception as e:
                log.debug(f"ContinuousStream tick: {e}")

        # S28-1: RobustnessHardener — stress batch every 12 cycles
        if n % 12 == 0 and self.robustness_hardener:
            try:
                self.robustness_hardener.run_stress_batch(n=6)
            except Exception as e:
                log.debug(f"RobustnessHardener cycle: {e}")

        # S28-2: AttentionRouter — tick every cycle; update focus from current problem
        if self.attention_router:
            try:
                if results:
                    domain = getattr(results[-1], 'domain', 'general') \
                             if hasattr(results[-1], 'domain') else 'general'
                    self.attention_router.set_focus([domain, 'solve', 'energy'])
                self.attention_router.tick()
            except Exception as e:
                log.debug(f"AttentionRouter tick: {e}")

        # S28-3: RecursiveToM — decay tick every 5 cycles
        if n % 5 == 0 and self.recursive_tom:
            try:
                self.recursive_tom.tick()
                # feed recent agent debate outcomes into ToM models
                if self.agent_society:
                    try:
                        bb = getattr(self.agent_society, 'blackboard', [])
                        for belief in bb[-3:]:
                            claim = getattr(belief, 'claim', str(belief))
                            conf  = getattr(belief, 'confidence', 0.5)
                            src   = getattr(belief, 'source', 'unknown')
                            self.recursive_tom.update_model(
                                src, claim, conf, "general", depth=1)
                    except Exception:
                        pass
            except Exception as e:
                log.debug(f"RecursiveToM tick: {e}")

        # S28-4: AgentMemoryBank — tick every cycle; save every 60 cycles
        if self.agent_memory_bank:
            try:
                self.agent_memory_bank.tick()
            except Exception as e:
                log.debug(f"AgentMemoryBank tick: {e}")

        # S29-1: MetaCurriculumEngine — full tick every 8 cycles
        if n % 8 == 0 and self.meta_curriculum:
            try:
                # Feed solve results into meta-curriculum
                for res in results:
                    domain  = getattr(res, 'domain', 'general') \
                              if hasattr(res, 'domain') else 'general'
                    success = getattr(res, 'solved', False) or \
                              (hasattr(res, 'energy') and res.energy < 0.5)
                    self.meta_curriculum.observe(domain, success)
                self.meta_curriculum.tick()
            except Exception as e:
                log.debug(f"MetaCurriculum tick: {e}")

        # S29-2: ActionPhysicsSession — one episode every 20 cycles
        if n % 20 == 0 and self.action_physics:
            try:
                ep = self.action_physics.run_episode(n_steps=10)
                if ep.concepts_found:
                    log.debug(f"ActionPhysics concepts: {ep.concepts_found}")
            except Exception as e:
                log.debug(f"ActionPhysics tick: {e}")

        # S29-3: StreamBridge — tick every cycle; submit stream results
        if self.stream_bridge:
            try:
                if self.continuous_stream:
                    streams = getattr(self.continuous_stream, '_streams', {})
                    for stype, stream in streams.items():
                        recent = getattr(stream, '_recent_results', [])
                        for item in recent[-1:]:
                            expr   = getattr(item, 'expression', str(item))
                            domain = getattr(item, 'domain', 'general')
                            self.stream_bridge.submit(expr, source=stype, domain=domain)
                self.stream_bridge.tick()
            except Exception as e:
                log.debug(f"StreamBridge tick: {e}")

        # S29-4: PerceptionBridge — parse world-model observations every 15 cycles
        if n % 15 == 0 and self.perception_bridge:
            try:
                if self.generative_world and hasattr(self.generative_world, 'last_problem'):
                    desc = getattr(self.generative_world, 'last_problem', '')
                    if desc and isinstance(desc, str) and len(desc) > 5:
                        self.perception_bridge.parse_scene(desc)
            except Exception as e:
                log.debug(f"PerceptionBridge tick: {e}")

        return results

    def _collect_session_stats(self) -> dict:
        """Gather stats for TemporalIdentity.update()."""
        try:
            stats = getattr(self, '_stats', {})
            solves = stats.get('solves_succeeded', 0)
            fails  = max(0, stats.get('solves_attempted', 0) - solves)
            domain_rates = {}
            if self.self_model:
                try:
                    snap = self.self_model.skill_snapshot()
                    domain_rates = {k: v for k, v in snap.items()
                                    if isinstance(v, (int, float))}
                except Exception:
                    pass
            best = None
            if self.self_model:
                try:
                    best_rec = self.self_model.best_strategy()
                    best = getattr(best_rec, 'name', None)
                except Exception:
                    pass
            discoveries = []
            if self.concept_blender:
                try:
                    recent = self.concept_blender.summary().get('top_blends', [])
                    discoveries = [b.get('name', '') for b in recent[:2]]
                except Exception:
                    pass
            return {
                "solves_this_session": solves,
                "fails_this_session":  fails,
                "domain_rates":        domain_rates,
                "best_strategy":       best,
                "new_discoveries":     discoveries,
                "primary_domain":      "arithmetic",
            }
        except Exception:
            return {"solves_this_session": 0, "fails_this_session": 0,
                    "domain_rates": {}, "best_strategy": None, "new_discoveries": []}

    def _pick_learning_problem(self) -> Optional[str]:
        """
        P2.3: Surprise-driven + ZPD + failure-replay + open-ended problem selection.
        """
        candidates: List[dict] = []

        def _add_candidate(expression: Optional[str], source: str, score: float,
                           domain: str = "general", reason: str = "") -> None:
            if not expression:
                return
            candidates.append({
                "expression": expression,
                "source": source,
                "score": round(max(0.0, min(1.0, float(score))), 4),
                "domain": domain or "general",
                "reason": reason,
            })

        def _choose_from_domain(target_domain: str) -> Optional[str]:
            if self._problem_gen:
                batch = self._problem_gen.generate_for_domain(target_domain, n=1)
                if batch:
                    return batch[0].get("expression")
            if self.developmental_curriculum:
                for dom in getattr(self.developmental_curriculum, "domains", {}).values():
                    if getattr(dom, "name", None) == target_domain and getattr(dom, "unlocked", False):
                        problems = getattr(dom, "problems", [])
                        if problems:
                            problem = problems[0]
                            return getattr(problem, "expression", None) or str(problem)
            return None

        # Ensure problem generator exists
        if not hasattr(self, "_problem_gen"):
            try:
                from sare.curriculum.problem_generator import ProblemGenerator
                self._problem_gen = ProblemGenerator()
            except Exception as exc:
                self._problem_gen = None
                self._record_runtime_error("problem_generator.init", exc, "learn_pick")

        if self.world_model_v3:
            try:
                failed_eps = [
                    ep for ep in getattr(self.world_model_v3, "_solve_history", [])[-50:]
                    if not ep.get("success") and ep.get("expression")
                ]
                for rank, episode in enumerate(reversed(failed_eps[-10:])):
                    surprise = float(episode.get("surprise", 0.0) or 0.0)
                    recency_bonus = max(0.0, 0.09 - rank * 0.01)
                    _add_candidate(
                        episode.get("expression"),
                        "failure_replay",
                        0.92 + min(surprise, 1.0) * 0.05 + recency_bonus,
                        episode.get("domain", "general"),
                        "recent failed problem replay",
                    )
            except Exception as exc:
                self._record_runtime_error("world_model_v3.failure_replay", exc, "learn_pick")

        if self.goal_planner and self._problem_gen:
            try:
                next_goal = self.goal_planner.next_actionable()
                if next_goal and next_goal.domain and next_goal.domain != "general":
                    _add_candidate(
                        _choose_from_domain(next_goal.domain),
                        "goal_planner",
                        0.88,
                        next_goal.domain,
                        f"next actionable goal: {getattr(next_goal, 'description', '')[:60]}",
                    )
            except Exception as exc:
                self._record_runtime_error("goal_planner.next_actionable", exc, "learn_pick")

        if self.self_model and self._problem_gen:
            try:
                goals = [
                    g for g in getattr(self.self_model, "_learning_goals", [])
                    if getattr(g, "status", "") == "active" and getattr(g, "domain", "general") != "general"
                ]
                if goals:
                    top_goal = max(goals, key=lambda g: getattr(g, "priority", 0.0))
                    priority = float(getattr(top_goal, "priority", 0.0) or 0.0)
                    _add_candidate(
                        _choose_from_domain(top_goal.domain),
                        "self_model",
                        0.84 + min(max(priority, 0.0), 1.0) * 0.08,
                        top_goal.domain,
                        f"active learning goal: {getattr(top_goal, 'reason', '')[:60]}",
                    )
            except Exception as exc:
                self._record_runtime_error("self_model.learning_goals", exc, "learn_pick")

        if self.goal_setter and self._problem_gen:
            try:
                next_goal = self.goal_setter.suggest_next_goal()
                if next_goal and next_goal.domain:
                    _add_candidate(
                        _choose_from_domain(next_goal.domain),
                        "goal_setter",
                        0.8,
                        next_goal.domain,
                        f"suggested goal: {getattr(next_goal, 'description', '')[:60]}",
                    )
            except Exception as exc:
                self._record_runtime_error("goal_setter.suggest_next_goal", exc, "learn_pick")

        if self.world_model_v3:
            try:
                high_surprise = self.world_model_v3.get_high_surprise_domains(3)
                for idx, item in enumerate(high_surprise):
                    target_domain = item[0]
                    surprise = float(item[1] if len(item) > 1 else 0.0)
                    _add_candidate(
                        _choose_from_domain(target_domain),
                        "surprise_domain",
                        0.72 + min(surprise, 1.0) * 0.1 - idx * 0.03,
                        target_domain,
                        f"high-surprise domain score={surprise:.2f}",
                    )
            except Exception as exc:
                self._record_runtime_error("world_model_v3.get_high_surprise_domains", exc, "learn_pick")

        if self.developmental_curriculum:
            try:
                problem = self.developmental_curriculum.next_problem(self.stage, self.self_model)
                if problem:
                    expression = getattr(problem, "expression", None) or str(problem)
                    _add_candidate(
                        expression,
                        "curriculum",
                        0.68,
                        getattr(problem, "domain", "general"),
                        "developmental curriculum next problem",
                    )
            except Exception as exc:
                self._record_runtime_error("developmental_curriculum.next_problem", exc, "learn_pick")

        if self._problem_gen:
            try:
                batch = self._problem_gen.generate_batch(n=1, max_difficulty=0.7)
                if batch:
                    _add_candidate(
                        batch[0].get("expression"),
                        "generator",
                        0.55,
                        batch[0].get("domain", "general"),
                        "open-ended generator exploration",
                    )
            except Exception as exc:
                self._record_runtime_error("problem_generator.generate_batch", exc, "learn_pick")

        # --- NEW: LLM Teacher Curriculum Generation ---
        # If all candidates have a low score (e.g., < 0.6), SARE is stagnating.
        # Ask the LLM Teacher to generate a specific exercise to force a breakthrough.
        best_score_so_far = max([c["score"] for c in candidates]) if candidates else 0.0
        if best_score_so_far < 0.6:
            try:
                from sare.interface.llm_bridge import _call_llm
                recent_rules = [r.name for r in getattr(self, "concept_registry", None).get_consolidated_rules()] if hasattr(self, "concept_registry") and self.concept_registry else []
                # Provide context of what it knows
                prompt = (
                    "You are the Teacher of an autonomous symbolic reasoning AI (SARE-HX).\n"
                    "The AI is currently stuck and stagnating. It has mastered the following concepts:\n"
                    f"{recent_rules[:15]}\n\n"
                    "Generate a single, specific mathematical or logical problem that is *just beyond* its current mastery.\n"
                    "The goal is to force the AI to discover a new, unlearned structural rule (like 'distributive property', 'De Morgan's laws', or 'factoring').\n"
                    "Return a JSON object:\n"
                    "{\n"
                    '  "expression": "the mathematical expression to solve",\n'
                    '  "domain": "algebra|logic|arithmetic",\n'
                    '  "target_concept": "the rule it forces the AI to learn"\n'
                    "}\n"
                    "Return ONLY JSON."
                )
                raw = _call_llm(prompt)
                import re, json
                raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`").strip()
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    data = json.loads(m.group(0))
                    llm_expr = str(data.get("expression", ""))
                    llm_domain = str(data.get("domain", "general"))
                    target_concept = str(data.get("target_concept", "unknown"))
                    if llm_expr:
                        _add_candidate(
                            llm_expr,
                            "llm_teacher",
                            0.95, # High priority because the Teacher prescribed it
                            llm_domain,
                            f"LLM Teacher customized exercise for ZPD to learn: {target_concept}"
                        )
                        log.info(f"LLM Teacher intervened to break stagnation. Target: {target_concept}")
            except Exception as e:
                log.debug(f"LLM Curriculum Generation failed: {e}")

        if candidates:
            best = max(candidates, key=lambda item: (item["score"], item["source"]))
            self._last_problem_selection = {
                **best,
                "candidates_considered": len(candidates),
                "time": time.time(),
            }
            log.debug(
                "Learning problem selected: %s via %s (score=%.3f, reason=%s)",
                best["expression"],
                best["source"],
                best["score"],
                best["reason"],
            )
            return best["expression"]

        examples = list(getattr(self.engine, "EXAMPLE_PROBLEMS", {}).values()) if self.engine else []
        fallback = examples[0] if examples else None
        self._last_problem_selection = {
            "expression": fallback,
            "source": "fallback",
            "score": 0.1,
            "domain": "general",
            "reason": "deterministic fallback example",
            "candidates_considered": 0,
            "time": time.time(),
        }
        return fallback

    def _test_and_promote_conjectures(self) -> int:
        """
        Creativity loop: generate conjectures, test each on actual problems,
        promote those that pass. Returns count of promoted conjectures.
        """
        promoted = 0
        try:
            conjectures = self.generate_conjectures(n=3)
        except Exception:
            return 0

        for conj in conjectures:
            plausibility = conj.get("plausibility", 0.0)
            domain = conj.get("domain", "general")
            hypothesis = conj.get("hypothesis", "")
            if not hypothesis or plausibility < 0.4:
                continue

            # Find any transform whose name appears in the hypothesis
            matching_t = None
            for t in self.transforms:
                if t.name().lower() in hypothesis.lower():
                    matching_t = t
                    break

            if matching_t is None:
                continue

            # Test conjecture on 3 domain problems — count energy reductions
            test_problems = self._get_test_problems_for_domain(domain, n=3)
            passes = 0
            for expr in test_problems:
                try:
                    g = self.engine.build_expression_graph(expr)
                    ee = self.energy
                    e0 = ee.compute(g).total
                    matches = matching_t.match(g)
                    if matches:
                        g2, _ = matching_t.apply(g, matches[0])
                        e1 = ee.compute(g2).total
                        if e1 < e0 - 0.01:
                            passes += 1
                except Exception:
                    pass

            if passes >= 2:
                # Emit rule events so it enters the discovery/promotion pipeline
                self._emit_rule_events(
                    name=f"conjecture_{matching_t.name()}",
                    domain=domain,
                    confidence=plausibility,
                    pattern=hypothesis[:100],
                )
                self.events.emit(Event.CREATIVITY_SPARK, {
                    "hypothesis": hypothesis[:80],
                    "transform": matching_t.name(),
                    "domain": domain,
                    "passes": passes,
                    "plausibility": plausibility,
                }, "conjecture_engine")
                promoted += 1
                log.debug(f"Conjecture verified: {matching_t.name()} in {domain} ({passes}/3)")

        return promoted

    def _verify_transfer_hypotheses(self) -> int:
        """
        Test untested TransferEngine hypotheses and synthesize transforms for
        any that are verified.  Called every 10 learn_cycle iterations.
        Returns the number of newly verified transfers.
        """
        if not self.transfer_engine or not self.transform_synthesizer:
            return 0

        verified = 0
        try:
            # Generate fresh hypotheses from observed domain patterns
            hyps = self.transfer_engine.generate_hypotheses()

            # Also pick up existing untested ones
            all_hyps = [
                h for h in self.transfer_engine._hypotheses.values()
                if h.status == "untested" and h.confidence > 0.3
            ]
            # Limit to top-5 by confidence to avoid long waits
            all_hyps.sort(key=lambda h: h.confidence, reverse=True)
            to_test = all_hyps[:5]

            for hyp in to_test:
                try:
                    # Build a lightweight solve_fn that the TransferEngine can call
                    target = hyp.target_domain
                    test_problems = self._get_test_problems_for_domain(target, n=5)
                    if not test_problems:
                        continue

                    from sare.engine import load_problem, EnergyEvaluator, BeamSearch
                    _ee = EnergyEvaluator(domain=target)
                    _sr = BeamSearch()
                    _tf = list(self.transforms)

                    def _solve_fn(expr, _ee=_ee, _sr=_sr, _tf=_tf):
                        try:
                            _, g = load_problem(expr)
                            e0 = _ee.compute(g).total
                            r = _sr.search(g, _ee, _tf, beam_width=5, max_depth=8, budget_seconds=1.5)
                            delta = e0 - r.energy.total
                            t_used = [t.name() for t in _tf
                                      if hasattr(t, '_last_applied') and t._last_applied]
                            return {
                                "success": delta > 0.01,
                                "delta": delta,
                                "transforms": getattr(r, 'transforms_applied', []),
                            }
                        except Exception:
                            return {"success": False, "delta": 0, "transforms": []}

                    ok = self.transfer_engine.test_hypothesis(hyp, _solve_fn, test_problems)
                    if ok:
                        verified += 1
                        # Synthesize new transforms for the verified target domain
                        if self.transform_synthesizer:
                            try:
                                new_specs = self.transform_synthesizer.synthesize_for_domain(
                                    target, missing_roles=[hyp.source_role]
                                )
                                if new_specs:
                                    log.info(
                                        f"Transfer verified: {hyp.source_role} "
                                        f"{hyp.source_domain}→{target}, "
                                        f"synthesized {len(new_specs)} transforms"
                                    )
                                    self.events.emit(Event.TRANSFER_SUCCEEDED, {
                                        "source": hyp.source_domain,
                                        "target": target,
                                        "role": hyp.source_role,
                                        "new_transforms": len(new_specs),
                                    }, "transfer_engine")
                            except Exception:
                                pass
                except Exception as _e:
                    log.debug(f"Hypothesis test failed: {_e}")

        except Exception as e:
            log.debug(f"_verify_transfer_hypotheses failed: {e}")

        return verified

    # ─────────────────────────────────────────────────────────────────────────
    #  Developmental Stage Management
    # ─────────────────────────────────────────────────────────────────────────

    def _update_stage(self):
        """Check if the system should advance to the next developmental stage."""
        stages = list(DevelopmentalStage)
        current_idx = stages.index(self.stage)

        if current_idx >= len(stages) - 1:
            return  # Already at max

        next_stage = stages[current_idx + 1]
        reqs = STAGE_REQUIREMENTS[next_stage]

        # Check requirements
        rules_count = self._stats["rules_promoted"] + self._stats["rules_discovered"]
        domains_count = len(self._stats.get("domains_mastered", []))
        solve_rate = (
            self._stats["solves_succeeded"] / max(self._stats["solves_attempted"], 1)
        )

        if (rules_count >= reqs["min_rules"] and
                domains_count >= reqs["min_domains"] and
                solve_rate >= reqs["min_solve_rate"]):
            old_stage = self.stage
            self.stage = next_stage
            log.info(f"🎓 Stage advanced: {old_stage.value} → {next_stage.value}")
            self.events.emit(Event.STAGE_ADVANCED, {
                "from": old_stage.value,
                "to": next_stage.value,
                "rules": rules_count,
                "domains": domains_count,
                "solve_rate": solve_rate,
            }, "brain")

    # ─────────────────────────────────────────────────────────────────────────
    #  State Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _load_state(self):
        """Load persisted brain state."""
        state_path = DATA_DIR / "brain_state.json"
        if state_path.exists():
            try:
                with open(state_path) as f:
                    state = json.load(f)
                self.stage = DevelopmentalStage(state.get("stage", "infant"))
                self._stats.update(state.get("stats", {}))
                self.total_solves = state.get("total_solves", 0)
                self.total_rules_learned = state.get("total_rules_learned", 0)
                self.cpp_enabled = bool(state.get("cpp_enabled", self.cpp_bindings_available)) and self.cpp_bindings_available
                log.info(f"Restored brain state: stage={self.stage.value}")
            except Exception as e:
                log.warning(f"Brain state load failed: {e}")

    def save_state(self):
        """Persist brain state to disk."""
        state_path = DATA_DIR / "brain_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "stage": self.stage.value,
            "stats": self._stats,
            "total_solves": self.total_solves,
            "total_rules_learned": self.total_rules_learned,
            "cpp_enabled": self.cpp_enabled,
            "boot_time": self.boot_time,
            "saved_at": time.time(),
        }
        try:
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            log.error(f"Brain state save failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Status & Introspection
    # ─────────────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Full brain status report."""
        uptime = time.time() - self.boot_time if self.boot_time else 0

        # Competence map
        competence = {}
        if self.self_model:
            try:
                competence = self.self_model.get_all_competence()
            except Exception:
                pass

        # Drives
        drives = {}
        if self.homeostasis:
            try:
                drives = self.homeostasis.get_all_drives()
            except Exception:
                pass

        # Goals
        goals = []
        if self.goal_setter:
            try:
                goals = self.goal_setter.active_goals()
            except Exception:
                pass

        return {
            "stage": self.stage.value,
            "stage_level": self.stage.level,
            "uptime_seconds": round(uptime, 1),
            "modules": self._module_status,
            "modules_loaded": sum(1 for v in self._module_status.values() if v.startswith("✅")),
            "modules_total": len(self._module_status),
            "stats": self._stats,
            "cpp": {
                "bindings_available": self.cpp_bindings_available,
                "enabled": self.cpp_enabled,
            },
            "competence": competence,
            "drives": drives,
            "goals": [g.to_dict() if hasattr(g, "to_dict") else str(g) for g in goals],
            "transforms_count": len(self.transforms),
            "recent_events": [
                {"event": e.event.value, "time": e.timestamp, "source": e.source}
                for e in self.events.recent(20)
            ],
            "last_problem_selection": self._last_problem_selection,
            "diagnostics": self._diagnostics[-20:],
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  Shutdown
    # ─────────────────────────────────────────────────────────────────────────

    def shutdown(self):
        """Gracefully save all state and stop daemons."""
        log.info("Brain shutting down...")
        self.events.emit(Event.BRAIN_SHUTDOWN, {}, "brain")

        # Save all modules
        for name, saver in [
            ("brain", self.save_state),
            ("self_model", lambda: self.self_model.save() if self.self_model else None),
            ("frontier", lambda: self.frontier_manager.save() if self.frontier_manager else None),
            ("credit", lambda: self.credit_assigner.save() if self.credit_assigner else None),
            ("curriculum", lambda: self.curriculum_gen.save() if self.curriculum_gen else None),
            ("memory", lambda: self.memory_manager.save() if self.memory_manager else None),
            ("concepts", lambda: self.concept_memory.save() if self.concept_memory is not None else None),
            ("concept_registry", lambda: self.concept_registry.save() if self.concept_registry and hasattr(self.concept_registry, "save") else None),
            ("synthesized_transforms", lambda: self.transform_synthesizer.save() if self.transform_synthesizer and hasattr(self.transform_synthesizer, "save") else None),
            ("goals", lambda: self.goal_setter.save() if self.goal_setter else None),
            ("identity", lambda: self.identity.save() if self.identity else None),
            ("autobiography", lambda: self.autobiography.save() if self.autobiography else None),
            ("commonsense", lambda: self.commonsense.save() if self.commonsense else None),
            ("tom", lambda: self.theory_of_mind.save() if self.theory_of_mind else None),
            ("world_model", lambda: self.world_model.save() if self.world_model else None),
            ("world_model_v3", lambda: self.world_model_v3.save() if self.world_model_v3 else None),
            ("homeostasis", lambda: self.homeostasis.save() if self.homeostasis else None),
            ("knowledge_graph", lambda: self.knowledge_graph.save() if self.knowledge_graph else None),
        ]:
            try:
                saver()
                log.info(f"  Saved: {name}")
            except Exception as e:
                log.warning(f"  Save failed: {name} — {e}")

        # Stop daemons
        self.stop_auto_learn()
        if self.continuous_stream and hasattr(self.continuous_stream, "stop"):
            try:
                self.continuous_stream.stop()
            except Exception:
                pass
        if self.hippocampus:
            try:
                self.hippocampus.stop()
            except Exception:
                pass

        log.info("Brain shutdown complete.")


# ─────────────────────────────────────────────────────────────────────────────
#  Singleton
# ─────────────────────────────────────────────────────────────────────────────

_brain: Optional[Brain] = None


def get_brain() -> Brain:
    """Get or create the global Brain instance."""
    global _brain
    if _brain is None:
        _brain = Brain()
        _brain.boot()
    return _brain
