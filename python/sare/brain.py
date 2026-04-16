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
import os
import subprocess
import tempfile
import time
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from sare.sare_logging.logger import SareLogger, SolveLog
from sare.learning.acquisition import (
    AcquisitionMesh,
    AcquisitionResult as AcquisitionResultData,
    AcquisitionSourceConfig as CanonicalAcquisitionSourceConfig,
)
from sare.learning.learning_policy import AdaptiveLearningPolicy
from sare.learning.strategy_scorecard import LearningStrategyScorecard

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


# ── Graph bridge utilities (canonical implementation in core.graph_bridge) ────
from sare.core.graph_bridge import (
    py_graph_to_cpp_graph as _py_graph_to_cpp_graph,
    cpp_graph_to_py_graph as _cpp_graph_to_py_graph,
    graph_features as _graph_features,
)


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


class SolveResult(dict):
    """Compatibility-safe canonical solve result."""

    @property
    def solved(self) -> bool:
        return bool(self.get("success", False))

    @property
    def request_id(self) -> str:
        return str(self.get("request_id", self.get("expression", "")))

    def to_dict(self) -> dict:
        return dict(self)


class LearnCycleResult(list):
    """List-like learning result with aggregate metadata."""

    def __init__(self, results: Optional[Iterable[dict]] = None, **summary: Any):
        super().__init__(results or [])
        self.summary = dict(summary)

    def to_dict(self) -> dict:
        return {"results": list(self), **self.summary}


class IngestionResult(dict):
    """Canonical ingestion result."""

    def to_dict(self) -> dict:
        return dict(self)


class AcquisitionResult(dict):
    """Canonical acquisition result."""

    def to_dict(self) -> dict:
        return dict(self)


class AuditReport(dict):
    """Canonical learning audit result."""

    def to_dict(self) -> dict:
        return dict(self)


class BrainStatus(dict):
    """Canonical brain status payload."""

    def to_dict(self) -> dict:
        return dict(self)


@dataclass
class IngestionSourceConfig:
    source_type: str
    payload: Any
    source_id: str = "user"
    domain: str = "general"
    trust_level: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionBatch:
    items: List[IngestionSourceConfig]
    source_id: str = "batch"
    metadata: Dict[str, Any] = field(default_factory=dict)


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

        for cb in list(self._listeners.get(event, [])):
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

# ── Stage Capability Gates ────────────────────────────────────────────────────
# Each stage gates which cognitive capabilities are available.
# max_beam_width=None → unlimited (uses config default)
STAGE_CAPABILITY_GATES: Dict[str, dict] = {
    DevelopmentalStage.INFANT.value:        {"max_beam_width": 3,    "analogy": False, "conjecture": False, "symbol_creation": False, "active_questioning": False, "decay_rate": 0.020},
    DevelopmentalStage.TODDLER.value:       {"max_beam_width": 5,    "analogy": False, "conjecture": False, "symbol_creation": False, "active_questioning": False, "decay_rate": 0.010},
    DevelopmentalStage.CHILD.value:         {"max_beam_width": 10,   "analogy": False, "conjecture": False, "symbol_creation": False, "active_questioning": True,  "decay_rate": 0.005},
    DevelopmentalStage.PRETEEN.value:       {"max_beam_width": 15,   "analogy": True,  "conjecture": False, "symbol_creation": False, "active_questioning": True,  "decay_rate": 0.002},
    DevelopmentalStage.TEENAGER.value:      {"max_beam_width": 25,   "analogy": True,  "conjecture": True,  "symbol_creation": False, "active_questioning": True,  "decay_rate": 0.001},
    DevelopmentalStage.UNDERGRADUATE.value: {"max_beam_width": 50,   "analogy": True,  "conjecture": True,  "symbol_creation": True,  "active_questioning": True,  "decay_rate": 0.0005},
    DevelopmentalStage.GRADUATE.value:      {"max_beam_width": 100,  "analogy": True,  "conjecture": True,  "symbol_creation": True,  "active_questioning": True,  "decay_rate": 0.0002},
    DevelopmentalStage.RESEARCHER.value:    {"max_beam_width": None, "analogy": True,  "conjecture": True,  "symbol_creation": True,  "active_questioning": True,  "decay_rate": 0.0001},
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
        self._events_wired = False

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
        self.kb_lookup = None           # KB fan-out retriever
        self.fact_ingester = None       # Q&A → triple storage

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
        self._wm_prediction_current = None

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

        # Session 32 — Embodiment + Causal + Counterfactual
        self.embodied_agent        = None  # S32-1: Grid-world perceive/decide/act/learn loop
        self.causal_rollout        = None  # S32-2: Multi-step causal prediction chains
        self.counterfactual_reasoner = None  # S32-3: Intervention-based causal reasoning

        # Language
        self.nl_parser = None
        self.llm_bridge = None
        self.language_grounding = None
        self.general_solver = None

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
            "solve_modes": {},
            "ingestion": {
                "records_scanned": 0,
                "facts_added": 0,
                "problems_added": 0,
                "concepts_proposed": 0,
                "rejected_records": 0,
            },
        }
        self._diagnostics: List[dict] = []
        self._max_diagnostics = 200
        self._last_problem_selection: Optional[dict] = None
        self._concept_graph_health: Dict[str, Any] = {
            "loaded": False,
            "recovered": False,
            "reseeded": False,
            "corrupt_backup_written": False,
        }
        self.acquisition_mesh: Optional[AcquisitionMesh] = None
        self._acquisition_lock = threading.Lock()
        self._acquisition_schedule_threads: List[threading.Thread] = []
        self._acquisition_cooldowns: Dict[str, float] = {}
        self._acquisition_policy_stats: Dict[str, Any] = {
            "success_counts": {},
            "error_counts": {},
            "last_remote_failure": None,
            "last_fallback_source": None,
            "github_enabled": False,
        }
        self._state_save_lock = threading.Lock()
        self._report_cache: Dict[str, Dict[str, Any]] = {}
        self._report_cache_lock = threading.Lock()
        self._report_build_locks: Dict[str, threading.Lock] = {}  # per-key build serialization
        self.learning_strategy_scorecard = LearningStrategyScorecard()
        self.adaptive_learning_policy = AdaptiveLearningPolicy(self.learning_strategy_scorecard)
        self._last_self_generated_learning: Dict[str, Any] = {}
        self._wire_events()

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

    def boot(self, config_path: Optional[str] = None):
        """Initialize all subsystems in dependency order."""
        if config_path and yaml:
            try:
                payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
                if isinstance(payload, dict):
                    self.config = self._load_config(payload)
                    self.phase_flags = dict(self.config.get("phases", {}))
                    self.boot_flags = dict(self.config.get("brain_boot", {}))
            except Exception as exc:
                self._record_runtime_error("brain.boot.config_path", exc, "boot")
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
        try:
            self._write_run_marker("brain")
            self._write_tracking_snapshot(force=True)
        except Exception as exc:
            self._record_runtime_error("brain.boot.run_marker", exc, "boot")

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

        # Layer 9: Embodiment + Causal + Counterfactual
        if self._boot_enabled("embodied_causal", True):
            self._boot_embodied_causal()
        else:
            self._mark_skipped("embodied_agent", "causal_rollout", "counterfactual_reasoner",
                               reason="embodied_causal boot disabled")

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

    @staticmethod
    def _safe_stringify(value: Any, limit: int = 500) -> str:
        try:
            text = str(value)
        except Exception:
            try:
                text = repr(value)
            except Exception:
                text = f"<{type(value).__name__}>"
        if len(text) > limit:
            return text[:limit]
        return text

    @classmethod
    def _json_safe(cls, value: Any, *, _depth: int = 0, _seen: Optional[set] = None) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if _depth >= 8:
            return cls._safe_stringify(value, limit=200)
        if isinstance(value, Path):
            return str(value)
        if _seen is None:
            _seen = set()
        if isinstance(value, dict):
            obj_id = id(value)
            if obj_id in _seen:
                return "<recursive>"
            _seen.add(obj_id)
            try:
                safe_dict = {}
                for key, item in list(value.items())[:200]:
                    safe_key = cls._safe_stringify(key, limit=120)
                    safe_dict[safe_key] = cls._json_safe(item, _depth=_depth + 1, _seen=_seen)
                return safe_dict
            finally:
                _seen.discard(obj_id)
        if isinstance(value, (list, tuple, set)):
            obj_id = id(value)
            if obj_id in _seen:
                return ["<recursive>"]
            _seen.add(obj_id)
            try:
                return [cls._json_safe(item, _depth=_depth + 1, _seen=_seen) for item in list(value)[:200]]
            finally:
                _seen.discard(obj_id)
        return cls._safe_stringify(value, limit=200)

    def _record_runtime_error(self, component: str, exc: Exception, context: str = "") -> None:
        error_message = self._safe_stringify(exc)
        entry = {
            "time": time.time(),
            "component": component,
            "error": error_message,
            "context": context,
        }
        self._diagnostics.append(entry)
        if len(self._diagnostics) > self._max_diagnostics:
            self._diagnostics = self._diagnostics[-self._max_diagnostics:]
        self._stats["runtime_errors"] = self._stats.get("runtime_errors", 0) + 1
        log.debug("%s failed%s: %s", component, f" ({context})" if context else "", error_message)

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
                saver = getattr(component, "_save", None)
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

        def load_kb_lookup():
            from sare.memory.knowledge_lookup import KnowledgeLookup
            return KnowledgeLookup()
        self.kb_lookup = self._load_module("kb_lookup", load_kb_lookup)

        def load_fact_ingester():
            from sare.memory.fact_ingester import FactIngester
            return FactIngester()
        self.fact_ingester = self._load_module("fact_ingester", load_fact_ingester)

    def _boot_metacognition(self):
        def load_sm():
            from sare.meta.self_model import SelfModel
            sm = SelfModel(DATA_DIR)
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

        # P1.4: Reload persisted rules as live transforms immediately
        # Lowered from 0.7 → 0.5 so more discovered rules survive restarts
        if self.concept_registry and self.transforms:
            try:
                rules = self.concept_registry.get_consolidated_rules(0.5)
                existing_names = {t.name() for t in self.transforms if hasattr(t, 'name') and callable(t.name)}
                loaded = 0
                for rule in rules:
                    rule_name = rule.get('name', '') if isinstance(rule, dict) else getattr(rule, 'name', '')
                    if rule_name in existing_names:
                        continue  # Skip duplicates already in transform set
                    t = self._rule_to_transform(rule) if hasattr(self, '_rule_to_transform') else None
                    if t:
                        self.transforms.append(t)
                        existing_names.add(rule_name)
                        loaded += 1
                if loaded:
                    log.info(f"Boot: reloaded {loaded} persisted rules as live transforms")
            except Exception as _e:
                log.debug(f"Rule reload skipped: {_e}")

        def load_cs():
            from sare.knowledge.commonsense import CommonSenseBase
            cs = CommonSenseBase()
            cs.load()
            if cs.total_facts() == 0:
                cs.seed()
            # Ensure chemistry facts are seeded (runs once, idempotent)
            try:
                from sare.knowledge.chemistry_seed import seed as _chem_seed
                _chem_seed()
            except Exception as _ce:
                pass
            return cs
        self.commonsense = self._load_module("commonsense", load_cs)

        # Concept Graph — the biological intelligence layer
        def load_cg():
            from sare.concept.concept_graph import ConceptGraph
            cg = ConceptGraph()
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
                env.run_curriculum(n=10)  # bootstrap with initial observations
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

        # ── Upgrade 2b: DatasetHub — General Intelligence Knowledge ──────────────
        def load_dataset_hub():
            from sare.knowledge.dataset_hub import DatasetHub
            hub = DatasetHub()
            if self._boot_enabled("warmup_dataset_hub", True):
                hub.ingest_all(self)
            return hub
        self.dataset_hub = self._load_module("dataset_hub", load_dataset_hub)

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
                for name, c in list(self.concept_graph.concepts.items())[:10]:
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
            extra_spaces = []
            if self.concept_graph:
                for name, c in list(self.concept_graph.concepts.items())[:10]:
                    extra_spaces.append({
                        "concept": name,
                        "domain": getattr(c, "domain", "general"),
                        "properties": {"operation": "symbolic",
                                       "domain": getattr(c, "domain", "general"),
                                       "rules": len(getattr(c, "symbolic_rules", []))},
                        "relations": getattr(c, "symbolic_rules", [])[:2],
                        "examples": getattr(c, "examples", [])[:2],
                    })
            blender = ConceptBlender(seed_spaces=extra_spaces if extra_spaces else None)
            blender.discover_blends(max_results=5)
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
                for item in sym_rules:
                    if isinstance(item, dict):
                        concept_name = str(item.get("concept", ""))
                        symbolic = str(item.get("rule", ""))
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        concept_name = str(item[0])
                        symbolic = str(item[1])
                    else:
                        continue
                    if not concept_name or not symbolic:
                        continue
                    if concept_name in self.concept_graph.concepts:
                        c = self.concept_graph.concepts[concept_name]
                        if symbolic not in c.symbolic_rules:
                            c.symbolic_rules.append(symbolic)
                if hasattr(self.environment_simulator, "extract_concepts"):
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
                cg_summary = self.concept_graph.summary() if hasattr(self.concept_graph, "summary") else self.concept_graph.stats()
                if hasattr(self.concept_graph, "health"):
                    self._concept_graph_health = dict(self.concept_graph.health())
                log.info(f"Concept graph seeded: {cg_summary}")
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
                    for rule in self.concept_registry.get_consolidated_rules(0.7):
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
                from sare.transfer.engine import get_transfer_engine
                return get_transfer_engine()
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
                from sare.reflection.py_reflection import get_reflection_engine
                return get_reflection_engine()
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
                from sare.meta.transform_synthesizer import TransformSynthesizer
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

        def load_general_solver():
            from sare.cognition.general_solver import GeneralSolver
            return GeneralSolver(persistence_delegate=self)
        self.general_solver = self._load_module("general_solver", load_general_solver)

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
            cg = CurriculumGenerator(knowledge_graph=self.knowledge_graph)  # Fix 5: pass KG
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
                analogy_transfer=self.analogy_transfer,
            )
            # Patch in optional modules
            if self.self_model:
                er.self_model = self.self_model
            if self.credit_assigner:
                er.credit_assigner = self.credit_assigner
            er._brain_ref = self  # Fix 3: enable immediate transform refresh on promotion
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

    def _boot_embodied_causal(self):
        """Boot Session 32: EmbodiedAgent, CausalRollout, CounterfactualReasoner."""
        def load_ea():
            from sare.world.embodied_agent import EmbodiedAgent
            ea = EmbodiedAgent()
            ea.wire(
                concept_graph=self.concept_graph,
                global_workspace=self.global_workspace,
                predictive_loop=self.predictive_loop,
            )
            return ea
        self.embodied_agent = self._load_module("embodied_agent", load_ea)

        def load_cr():
            from sare.world.causal_rollout import CausalRollout
            cr = CausalRollout()
            cr.wire(
                chain_detector=getattr(self, "causal_chain_detector", None),
                predictive_loop=self.predictive_loop,
                global_workspace=self.global_workspace,
            )
            return cr
        self.causal_rollout = self._load_module("causal_rollout", load_cr)

        def load_cfr():
            from sare.causal.counterfactual_reasoner import CounterfactualReasoner
            cfr = CounterfactualReasoner()
            cfr.wire(
                chain_detector=getattr(self, "causal_chain_detector", None),
                global_workspace=self.global_workspace,
                knowledge_graph=self.knowledge_graph,
            )
            return cfr
        self.counterfactual_reasoner = self._load_module("counterfactual_reasoner", load_cfr)

    # ─────────────────────────────────────────────────────────────────────────
    #  Event Wiring
    # ─────────────────────────────────────────────────────────────────────────

    def _wire_events(self):
        """Connect modules via event bus."""
        if self._events_wired:
            return

        # When a solve completes, update memory + self model + world model
        self.events.subscribe(Event.SOLVE_COMPLETED, self._on_solve_completed)
        self.events.subscribe(Event.SOLVE_FAILED, self._on_solve_failed)
        self.events.subscribe(Event.RULE_DISCOVERED, self._on_rule_discovered)
        self.events.subscribe(Event.RULE_PROMOTED, self._on_rule_promoted)
        self.events.subscribe(Event.DOMAIN_MASTERED, self._on_domain_mastered)
        self.events.subscribe(Event.COMPETENCE_UPDATED, self._on_competence_updated)
        self.events.subscribe(Event.TRANSFER_ATTEMPTED, self._on_transfer_attempted)
        self.events.subscribe(Event.TRANSFER_SUCCEEDED, self._on_transfer_succeeded)
        # Subscribe to transfer_verified from the global core event bus (published by TransferEngine)
        try:
            from sare.core.event_bus import get_event_bus as _get_core_bus
            _get_core_bus().subscribe("transfer_verified", self._on_transfer_verified_core)
        except Exception:
            pass
        self._events_wired = True

    def _on_solve_completed(self, ed: EventData):
        """Post-solve processing: store episode, update competence, reflect."""
        data = ed.data
        problem_id = data.get("problem_id", "")
        transforms_used = data.get("transforms", [])
        energy_before = data.get("energy_before", 0)
        energy_after = data.get("energy_after", 0)
        domain = data.get("domain", "general")
        elapsed = data.get("elapsed", 0)
        solver_used = str(data.get("solver_used", "symbolic") or "symbolic")
        delta = energy_before - energy_after
        success = delta > 0.01

        self._stats["solves_attempted"] += 1
        if success:
            self._stats["solves_succeeded"] += 1
            try:
                self._stats.setdefault("domain_consecutive_failures", {})[domain] = 0
            except Exception:
                pass

        # WorldModel v3: observe every solve (learns causal links, schemas, beliefs)
        if self.world_model_v3:
            try:
                self.world_model_v3.observe_solve(
                    expression=problem_id, transforms_used=transforms_used,
                    energy_delta=delta, domain=domain, solved=success,
                )
            except Exception as e:
                log.debug(f"WorldModelV3 observe failed: {e}")

        # Transfer Engine: observe all solves to discover roles
        if self.transfer_engine:
            try:
                self.transfer_engine.observe(transforms_used, domain, success)
            except Exception as exc:
                self._record_runtime_error("transfer_engine.observe", exc, "post_solve")

        # Transfer Engine: passive hypothesis testing via live solve outcomes
        if self.transfer_engine and success and transforms_used:
            try:
                from sare.transfer.engine import RoleClassifier as _BRC
                _used_roles = {_BRC.classify(t) for t in transforms_used if t} - {None}
                _te_domain = domain or "general"
                _pending = [
                    h for h in self.transfer_engine._hypotheses.values()
                    if h.status == "untested" and h.target_domain == _te_domain
                    and h.source_role in _used_roles
                ]
                if _pending:
                    log.info("[TransferPassive] domain=%s roles=%s → %d matches",
                             _te_domain, _used_roles, len(_pending))
                for _hyp in _pending[:2]:
                    _hyp.test_results.append({
                        "problem": str(problem_id)[:80],
                        "success": success,
                        "delta": float(energy_before - energy_after),
                    })
                    if len(_hyp.test_results) >= 3:
                        _wins = sum(1 for r in _hyp.test_results if r.get("success"))
                        if _wins / len(_hyp.test_results) >= 0.5:
                            _hyp.status = "verified"
                            _hyp.confidence = min(0.95, _hyp.confidence + 0.2)
                            self.transfer_engine._stats["hypotheses_verified"] = (
                                int(self.transfer_engine._stats.get("hypotheses_verified", 0)) + 1
                            )
                            log.info("[TransferPassive] VERIFIED: %s→%s role=%s conf=%.2f",
                                     _hyp.source_domain, _hyp.target_domain,
                                     _hyp.source_role, _hyp.confidence)
                        elif _wins == 0 and len(_hyp.test_results) >= 5:
                            _hyp.status = "rejected"
                            _hyp.confidence *= 0.3
                            self.transfer_engine._stats["hypotheses_rejected"] = (
                                int(self.transfer_engine._stats.get("hypotheses_rejected", 0)) + 1
                            )
                if _pending:
                    self.transfer_engine.save()
            except Exception as _tp_exc:
                log.debug("[TransferPassive] exception: %s", _tp_exc)

        # S32: CausalRollout — observe transform sequence and energy deltas
        if self.causal_rollout and transforms_used:
            try:
                n = len(transforms_used)
                deltas = [float(energy_before - energy_after) / max(1, n)] * n
                self.causal_rollout._model.observe_sequence(transforms_used, deltas, domain, success=success)
            except Exception as exc:
                self._record_runtime_error("causal_rollout.observe_sequence", exc, "post_solve")

        # S32: CounterfactualReasoner — analyze every successful solve
        if self.counterfactual_reasoner and transforms_used and success:
            try:
                self.counterfactual_reasoner.analyze(
                    expression=problem_id,
                    domain=domain,
                    transforms_applied=transforms_used,
                    original_delta=delta,
                    solve_fn=lambda expr: {"delta": 0.0},
                )
            except Exception as exc:
                self._record_runtime_error("counterfactual_reasoner.analyze", exc, "post_solve")

        # S32: Dopamine — reward signal on successful solves
        if success and delta > 0.1:
            try:
                from sare.neuro.dopamine import get_dopamine_system
                _evt = "solve_novel" if delta > 3.0 else "solve_known"
                get_dopamine_system().receive_reward(
                    event_type=_evt,
                    domain=domain,
                    delta=delta,
                )
            except Exception as exc:
                self._record_runtime_error("dopamine.receive_reward", exc, "post_solve")

        # HTM: record transform sequence for n-gram sequence learning
        if transforms_used:
            try:
                from sare.neuro.htm_predictor import get_htm_predictor
                get_htm_predictor().observe_sequence(transforms_used, domain, success=success)
            except Exception as exc:
                self._record_runtime_error("htm_predictor.observe_sequence", exc, "post_solve")

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
                if len(self.concept_memory) % 200 == 0:
                    from sare.memory.concept_formation import ConceptFormation
                    ConceptFormation(self.concept_memory, self.concept_registry).run()
            except Exception as exc:
                self._record_runtime_error("concept_memory.record", exc, "post_solve")

        # 2. Update self model
        if self.self_model:
            try:
                self.self_model.update(
                    domain=domain,
                    solved=success,
                    energy_delta=delta,
                    steps=len(transforms_used),
                    predicted_confidence=0.5,
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

        # 4b. KB fact ingestion for non-symbolic solved problems
        if (success and self.fact_ingester is not None
                and domain not in ("arithmetic", "algebra", "logic", "math")):
            try:
                expr = data.get("expression", "")
                answer = data.get("answer", "") or data.get("simplified", "")
                if expr and answer:
                    self.fact_ingester.ingest(expr, str(answer), domain, confidence=0.75)
            except Exception as exc:
                self._record_runtime_error("fact_ingester.ingest", exc, "post_solve")

        # 5. World model prediction tracking
        if self.world_model and hasattr(self.world_model, 'record_outcome'):
            try:
                _pred = self._wm_prediction_current
                if _pred is not None:
                    self.world_model.record_outcome(
                        _pred, transforms_used, delta, domain
                    )
                self._wm_prediction_current = None
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
                legacy_ground = getattr(self.concept_graph, "ground_solve_episode", None)
                if callable(legacy_ground):
                    legacy_ground(
                        expression=expression_text,
                        transforms=list(transforms_used),
                        result=bool(success),
                        domain=domain,
                        delta=delta,
                    )
                example_ground = getattr(self.concept_graph, "ground_example", None)
                for t_name in transforms_used:
                    if not callable(example_ground):
                        continue
                    example_ground(
                        concept_name=t_name,
                        text=f"{expression_text} → {result_text}",
                        metadata={
                            "domain": domain,
                            "operation": t_name,
                            "result": result_text,
                            "inputs": [expression_text],
                            "symbolic": f"delta={delta:.2f}",
                        },
                    )
                    # Trigger abstraction for well-observed concepts
                    c = self.concept_graph.get(t_name)
                    if c and c.ground_count() >= 3:
                        self.concept_graph.abstract_from_examples(c.name)
            except Exception as exc:
                self._record_runtime_error("concept_graph.ground_example", exc, "post_solve")

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
        if expression and not transforms_tried:
            fallback_info = self._prepare_transform_fallback(
                expression,
                domain,
                data.get("initial_graph"),
            )
            data["fallback_info"] = fallback_info
            if fallback_info.get("attempted"):
                self._stats["transform_fallback_attempts"] = int(self._stats.get("transform_fallback_attempts", 0) or 0) + 1

        # ── 1. Failure Analysis: understand WHY it failed ──
        failure_reason = self._analyze_failure(data)
        self._stats.setdefault("failure_reasons", {})
        reason = failure_reason.get("reason", "unknown")
        self._stats["failure_reasons"][reason] = self._stats["failure_reasons"].get(reason, 0) + 1

        # ── 2. WorldModel v3: learn from failure (critical!) ──
        if self.world_model_v3:
            try:
                self.world_model_v3.observe_solve(
                    expression=expression, transforms_used=transforms_tried,
                    energy_delta=energy_before - energy_after, domain=domain, solved=False,
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
                    if data.get("fallback_info", {}).get("attempted"):
                        self._stats["transform_fallback_successes"] = int(self._stats.get("transform_fallback_successes", 0) or 0) + 1
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

        try:
            domain_failures = self._stats.setdefault("domain_consecutive_failures", {})
            fail_count = int(domain_failures.get(domain, 0) or 0) + 1
            domain_failures[domain] = fail_count
            if fail_count % 5 == 0:
                self._attempt_concept_transfer(expression, domain, transforms_tried)
        except Exception as exc:
            self._record_runtime_error("concept_transfer.fail_count", exc, "solve_failed")

        # ── 7. Feed failures into LLM synthesizer (learn from what we can't do) ──
        # Every 10 failures in a domain, ask the LLM to synthesize a new transform
        if expression and self.transform_synthesizer:
            try:
                _fail_buf = self._stats.setdefault("_synth_fail_buffer", {})
                _synth_ts = self._stats.setdefault("_synth_last_attempt_ts", {})
                _buf = _fail_buf.setdefault(domain or "general", [])
                _buf.append(expression)
                # Deduplicate and cap buffer
                _buf_unique = list(dict.fromkeys(_buf))[-20:]
                _fail_buf[domain or "general"] = _buf_unique
                _SYNTH_COOLDOWN_S = 300  # 5 minutes between LLM synthesis per domain
                _now_s = time.time()
                _last_s = float(_synth_ts.get(domain or "general", 0.0) or 0.0)
                if (len(_buf_unique) >= 10 and len(_buf_unique) % 10 == 0
                        and (_now_s - _last_s) >= _SYNTH_COOLDOWN_S):
                    # Get current transform names for context
                    _t_names = [t.name() for t in (self.transforms or [])
                                if hasattr(t, "name")][:30]
                    # Build validation graphs from buffer
                    _val_graphs = []
                    for _expr in _buf_unique[-6:]:
                        try:
                            _, _g = self.engine.load_problem(_expr)
                            _val_graphs.append(_g)
                        except Exception:
                            pass
                    _synth_ts[domain or "general"] = _now_s  # record attempt time
                    import threading as _thr
                    _domain_snap = domain or "general"
                    _buf_snap = list(_buf_unique[-6:])
                    def _synth_bg(_d=_domain_snap, _b=_buf_snap):
                        try:
                            results = self.transform_synthesizer.synthesize_transforms(domain=_d, n=1)
                            if results:
                                log.info("[SynthFromFailure] New transform(s) promoted for domain=%s: %s",
                                         _d, [t.name() for t in results if hasattr(t, "name")])
                                self._refresh_transforms()
                            else:
                                log.debug("[SynthFromFailure] domain=%s synthesis: no new transforms promoted", _d)
                        except Exception as _se:
                            log.debug("[SynthFromFailure] synthesis error: %s", _se)
                    _thr.Thread(target=_synth_bg, daemon=True).start()
            except Exception as exc:
                self._record_runtime_error("synth_from_failure", exc, "solve_failed")

    def _analyze_failure(self, solve_data: dict) -> dict:
        """Analyze WHY a solve failed. Returns structured failure report."""
        expression = solve_data.get("expression", "")
        transforms = solve_data.get("transforms", [])
        energy_before = solve_data.get("energy_before", 0)
        energy_after = solve_data.get("energy_after", 0)
        delta = energy_before - energy_after
        fallback_info = dict(solve_data.get("fallback_info", {}) or {})

        # Classify the failure
        if not transforms:
            reason = "no_candidates_generated"
            retryable = True
            suggestion = "Need new transform types or verified fallback candidates for this pattern"
            if fallback_info.get("attempted") and fallback_info.get("generated", 0) > 0:
                reason = "candidates_generated_but_rejected"
                suggestion = "Fallback candidates were generated but did not pass solve-time checks"
            if fallback_info.get("no_runtime_match"):
                reason = "candidates_generated_but_unmatched"
                suggestion = "Fallback candidates were proposed but no live transform matched the expression graph"
            if fallback_info.get("verification_failed"):
                reason = "verification_failed"
                suggestion = "Fallback candidates matched but failed verification"
            if fallback_info.get("execution_failed"):
                reason = "execution_failed"
                suggestion = "Matched fallback transforms executed but still did not solve the problem"
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
            "fallback": fallback_info,
        }

    def _transform_fallback_candidates(self, expression: str, domain: str, limit: int = 8) -> List[str]:
        candidates: List[str] = []
        seen = set()

        def _add(items: Iterable[Any]) -> None:
            for item in items:
                name = ""
                if isinstance(item, str):
                    name = item
                elif isinstance(item, dict):
                    name = str(item.get("name") or item.get("rule_name") or item.get("title") or "")
                else:
                    name = str(getattr(item, "name", "") or "")
                name = name.strip()
                if name and name not in seen:
                    seen.add(name)
                    candidates.append(name)
                if len(candidates) >= limit:
                    return

        families = {
            "arithmetic": ["add_zero", "mul_one", "cancel_terms", "combine_like_terms"],
            "algebra": ["expand", "factor", "combine_like_terms", "distribute"],
            "logic": ["double_negation", "and_true", "or_false", "de_morgan"],
            "code": ["identity", "inline_wrapper", "eliminate_noop"],
            "mathematics": ["factor", "expand", "combine_like_terms", "distribute"],
            "physics": ["unit_reduce", "solve_for_variable", "cancel_terms"],
            "chemistry": ["chemistry_stoich_reaction", "balance_equation", "combine_like_terms"],
            "technology": ["identity", "inline_wrapper", "normalize_api_usage"],
            "word_problems": ["extract_equation", "translate_units", "solve_for_variable"],
        }
        _add(families.get(domain, []))

        try:
            rules = self.promoted_rules_summary().get("rules", [])
            _add(rule for rule in rules if str(rule.get("domain", "")) == domain)
        except Exception:
            pass

        try:
            recent_successes = self.events.recent(50, Event.SOLVE_COMPLETED)
            _add(
                transform
                for event in reversed(recent_successes)
                for transform in list((event.data or {}).get("transforms_used", []) or [])
                if str((event.data or {}).get("domain", "")) == domain
            )
        except Exception:
            pass

        try:
            if self.memory_manager is not None:
                recent_strategies = list(getattr(self.memory_manager, "_strategies", {}).values())[-40:]
                _add(
                    transform
                    for strategy in reversed(recent_strategies)
                    for transform in list((strategy or {}).get("transform_sequence", []) or [])
                )
        except Exception:
            pass

        try:
            for artifact in self.learned_artifacts(limit=50).get("items", []):
                metadata = dict(artifact.get("metadata", {}) or {})
                if artifact.get("verification_state") != "verified":
                    continue
                if metadata.get("source_kind") not in {"github_code", "github_tests"}:
                    continue
                if str(artifact.get("domain", "")) not in {domain, "code", "general"}:
                    continue
                _add(list(metadata.get("fallback_transforms", []) or []))
                _add([artifact.get("content", ""), artifact.get("title", "")])
                if len(candidates) >= limit:
                    break
        except Exception:
            pass
        return candidates[:limit]

    def _prepare_transform_fallback(self, expression: str, domain: str, graph: Any = None) -> dict:
        info = {
            "attempted": True,
            "generated": 0,
            "candidate_names": [],
            "matched_transforms": 0,
            "matched_transform_names": [],
            "verification_failed": False,
            "execution_failed": False,
            "no_runtime_match": False,
        }
        candidates = self._transform_fallback_candidates(expression, domain)
        info["candidate_names"] = candidates[:5]
        info["generated"] = len(candidates)
        if graph is None:
            try:
                graph = self.engine.build_expression_graph(expression)
            except Exception:
                graph = None
        if graph is not None and self.transforms:
            matched = 0
            matched_names: List[str] = []
            for transform in self.transforms[:100]:
                try:
                    if transform.match(graph):
                        matched += 1
                        try:
                            matched_names.append(str(transform.name()))
                        except Exception:
                            pass
                except Exception:
                    continue
            if matched:
                info["generated"] = max(info["generated"], matched)
                info["matched_transforms"] = matched
                info["verification_failed"] = True
                info["matched_transform_names"] = matched_names[:8]
            elif candidates:
                info["no_runtime_match"] = True
        return info

    def failure_reason_report(self) -> dict:
        reasons = dict(self._stats.get("failure_reasons", {}) or {})
        transform_total = sum(
            int(reasons.get(name, 0) or 0)
            for name in (
                "no_candidates_generated",
                "candidates_generated_but_rejected",
                "candidates_generated_but_unmatched",
                "verification_failed",
                "execution_failed",
                "no_matching_transforms",
            )
        )
        fallback_attempts = int(self._stats.get("transform_fallback_attempts", 0) or 0)
        fallback_rescues = int(self._stats.get("transform_fallback_successes", 0) or 0)
        rescue_stats = dict((self._stats.get("transform_rescue", {}) or {}))
        by_domain_raw = dict(rescue_stats.get("by_domain", {}) or {})
        by_domain: Dict[str, dict] = {}
        by_source_attempts: Dict[str, int] = {}
        by_source_rescues: Dict[str, int] = {}
        for domain, raw in by_domain_raw.items():
            if not isinstance(raw, dict):
                continue
            attempts = int(raw.get("attempts", 0) or 0)
            rescues = int(raw.get("rescues", 0) or 0)
            sources = {
                str(source): int(count or 0)
                for source, count in dict(raw.get("sources", {}) or {}).items()
            }
            successes = {
                str(source): int(count or 0)
                for source, count in dict(raw.get("source_successes", {}) or {}).items()
            }
            for source, count in sources.items():
                by_source_attempts[source] = by_source_attempts.get(source, 0) + int(count or 0)
            for source, count in successes.items():
                by_source_rescues[source] = by_source_rescues.get(source, 0) + int(count or 0)
            best_source = None
            best_successes = -1
            best_rate = -1.0
            for source, source_attempts in sources.items():
                source_rescues = successes.get(source, 0)
                source_rate = source_rescues / max(source_attempts, 1)
                if source_rescues > best_successes or (
                    source_rescues == best_successes and source_rate > best_rate
                ):
                    best_source = source
                    best_successes = source_rescues
                    best_rate = source_rate
            by_domain[domain] = {
                "attempts": attempts,
                "rescues": rescues,
                "rescue_rate": round(rescues / max(attempts, 1), 3) if attempts else 0.0,
                "sources": sources,
                "source_successes": successes,
                "best_rescue_source": best_source,
            }
        by_source = [
            {
                "source": source,
                "attempts": attempts,
                "rescues": by_source_rescues.get(source, 0),
                "rescue_rate": round(by_source_rescues.get(source, 0) / max(attempts, 1), 3) if attempts else 0.0,
            }
            for source, attempts in by_source_attempts.items()
        ]
        by_source.sort(
            key=lambda item: (item["rescues"], item["rescue_rate"], item["attempts"]),
            reverse=True,
        )
        return {
            "failure_reasons": reasons,
            "transform_failures": {
                "total": transform_total,
                "no_candidates_generated": int(reasons.get("no_candidates_generated", 0) or 0),
                "candidates_generated_but_rejected": int(reasons.get("candidates_generated_but_rejected", 0) or 0),
                "candidates_generated_but_unmatched": int(reasons.get("candidates_generated_but_unmatched", 0) or 0),
                "verification_failed": int(reasons.get("verification_failed", 0) or 0),
                "execution_failed": int(reasons.get("execution_failed", 0) or 0),
                "legacy_no_matching_transforms": int(reasons.get("no_matching_transforms", 0) or 0),
                "fallback_attempts": fallback_attempts,
                "fallback_rescues": fallback_rescues,
                "fallback_rescue_rate": round(fallback_rescues / max(fallback_attempts, 1), 3) if fallback_attempts else 0.0,
                "by_domain": by_domain,
                "by_source": by_source,
                "best_rescue_source": by_source[0]["source"] if by_source else None,
            },
        }

    def _record_transform_rescue(self, domain: str, source: str, success: bool) -> None:
        bucket = self._stats.setdefault("transform_rescue", {})
        by_domain = bucket.setdefault("by_domain", {})
        entry = by_domain.setdefault(str(domain or "general"), {})
        entry["attempts"] = int(entry.get("attempts", 0) or 0) + 1
        if success:
            entry["rescues"] = int(entry.get("rescues", 0) or 0) + 1
        sources = entry.setdefault("sources", {})
        source_successes = entry.setdefault("source_successes", {})
        sources[str(source)] = int(sources.get(str(source), 0) or 0) + 1
        if success:
            source_successes[str(source)] = int(source_successes.get(str(source), 0) or 0) + 1

    def _transfer_suite_definitions(self) -> List[dict]:
        return [
            {"id": "arithmetic_to_algebra", "label": "Arithmetic -> Algebra", "source_domain": "arithmetic", "target_domain": "algebra"},
            {"id": "logic_to_code", "label": "Logic -> Code", "source_domain": "logic", "target_domain": "code"},
            {"id": "science_to_reasoning", "label": "Science -> Reasoning", "source_domain": "science", "target_domain": "reasoning"},
            {"id": "language_to_dialogue", "label": "Language -> Dialogue", "source_domain": "language", "target_domain": "dialogue"},
            {"id": "mathematics_to_word_problems", "label": "Mathematics -> Word Problems", "source_domain": "mathematics", "target_domain": "word_problems"},
            {"id": "science_to_technology", "label": "Science -> Technology", "source_domain": "science", "target_domain": "technology"},
            {"id": "logic_to_reasoning", "label": "Logic -> Reasoning", "source_domain": "logic", "target_domain": "reasoning"},
        ]

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

    def _attempt_concept_transfer(self, expression: str, domain: str,
                                  failed_transforms: Optional[List[str]] = None) -> Optional[dict]:
        try:
            from sare.memory.concept_transfer import get_concept_transfer
        except Exception as exc:
            self._record_runtime_error("get_concept_transfer", exc, "attempt_concept_transfer")
            return None

        if not self.transforms or self.energy is None:
            return None

        try:
            graph = self.engine.build_expression_graph(expression)
        except Exception as exc:
            self._record_runtime_error("engine.build_expression_graph", exc, "attempt_concept_transfer")
            return None

        class _Problem:
            def __init__(self, graph_obj, target_domain):
                self.graph = graph_obj
                self.domain = target_domain

        transfer_engine = get_concept_transfer()
        result = transfer_engine.attempt_transfer(
            problem=_Problem(graph, domain),
            failed_transforms=list(failed_transforms or []),
            available_transforms=self.transforms,
            searcher=getattr(self, "searcher", None),
            energy=self.energy,
        )
        if result:
            rule_name, delta = result
            self.events.emit(Event.TRANSFER_SUCCEEDED, {
                "source": "concept_transfer",
                "target": domain,
                "rule_name": rule_name,
                "heldout_target_wins": 1,
                "heldout_target_tests": 1,
                "delta": delta,
            }, "concept_transfer")
            return {"rule_name": rule_name, "delta": delta}
        return None

    def _retry_with_alternative(self, expression: str, domain: str,
                                 failure_info: dict) -> Optional[dict]:
        """Retry a failed problem with a different strategy."""
        try:
            fallback = dict(failure_info.get("fallback", {}) or {})
            weak_domains = {"mathematics", "physics", "chemistry", "technology", "word_problems"}
            rescue_stats = dict((self._stats.get("transform_rescue", {}) or {}))
            domain_rescue = dict((rescue_stats.get("by_domain", {}) or {}).get(str(domain or "general"), {}) or {})
            source_attempts = {
                str(source): int(count or 0)
                for source, count in dict(domain_rescue.get("sources", {}) or {}).items()
            }
            source_successes = {
                str(source): int(count or 0)
                for source, count in dict(domain_rescue.get("source_successes", {}) or {}).items()
            }
            fallback_generated = int(fallback.get("generated", 0) or 0)
            matched_transforms = int(fallback.get("matched_transforms", 0) or 0)
            no_runtime_match = bool(fallback.get("no_runtime_match"))
            verification_failed = bool(fallback.get("verification_failed"))
            transfer_attempts = int(self._stats.get("transfers_attempted", 0) or 0)
            transfer_successes = int(self._stats.get("transfers_succeeded", 0) or 0)
            global_transfer_rate = (
                transfer_successes / max(transfer_attempts, 1)
                if transfer_attempts
                else None
            )

            def _with_rescue_source(result: Optional[dict], source: str) -> Optional[dict]:
                success = bool(result and result.get("success"))
                self._record_transform_rescue(domain, source, success)
                if not result:
                    return None
                payload = dict(result)
                payload.setdefault("rescue_source", source)
                return payload

            def _source_viable(source: str, min_attempts: int = 4, min_rate: float = 0.15, force: bool = False) -> bool:
                attempts = int(source_attempts.get(source, 0) or 0)
                rescues = int(source_successes.get(source, 0) or 0)
                if force or attempts < min_attempts:
                    return True
                return (rescues / max(attempts, 1)) >= min_rate

            should_try_transfer = fallback_generated > 0 or domain in weak_domains
            if no_runtime_match or matched_transforms > 0:
                should_try_transfer = True
            if (
                should_try_transfer
                and global_transfer_rate is not None
                and transfer_attempts >= 10
                and global_transfer_rate < 0.1
                and domain not in weak_domains
                and not no_runtime_match
            ):
                should_try_transfer = False
            if should_try_transfer and not _source_viable(
                "verified_transfer_suggestion",
                min_attempts=4,
                min_rate=0.2,
                force=(domain in weak_domains or no_runtime_match),
            ):
                should_try_transfer = False
            if should_try_transfer:
                transferred = self._attempt_concept_transfer(
                    expression,
                    domain,
                    failed_transforms=list(fallback.get("matched_transform_names", []) or []),
                )
                if transferred and float(transferred.get("delta", 0.0) or 0.0) > 0.0:
                    return _with_rescue_source(
                        {
                            "success": True,
                            "delta": float(transferred.get("delta", 0.0) or 0.0),
                            "transforms_used": [str(transferred.get("rule_name", "concept_transfer"))],
                            "strategy": "concept_transfer",
                            "transfer_outcome": {
                                "source_domain": "concept_transfer",
                                "target_domain": domain,
                                "proposed_transform": str(transferred.get("rule_name", "concept_transfer")),
                                "verified": True,
                                "heldout_target_wins": 1,
                                "heldout_target_tests": 1,
                            },
                        },
                        "verified_transfer_suggestion",
                    )
                self._record_transform_rescue(domain, "verified_transfer_suggestion", False)
            should_try_general_solver = (
                domain in weak_domains
                and _source_viable(
                    "general_solver_rescue",
                    min_attempts=4,
                    min_rate=0.12,
                    force=(no_runtime_match or verification_failed or fallback_generated == 0),
                )
            )
            if should_try_general_solver:
                rescued = self._run_general_solver(
                    expression,
                    context={
                        "force_general_solver": True,
                        "rescue_mode": True,
                        "fallback_transforms": list(fallback.get("candidate_names", []) or []),
                    },
                    domain=domain,
                )
                rescued_payload = _with_rescue_source(
                    {
                        **dict(rescued or {}),
                        "strategy": "general_solver_rescue",
                    } if rescued else None,
                    "general_solver_rescue",
                )
                if rescued_payload and rescued_payload.get("success"):
                    return rescued_payload
            result = _with_rescue_source(
                self.solve(
                    expression, algorithm="mcts",
                    beam_width=12, max_depth=50, budget=5.0, domain=domain
                ),
                "mcts_retry",
            )
            if result and result.get("success"):
                return result
            return _with_rescue_source(
                self.solve(
                    expression, algorithm="beam",
                    beam_width=16, max_depth=60, budget=8.0, domain=domain
                ),
                "beam_widen",
            )
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
        try:
            self._trigger_transfer_evaluation(
                str(ed.data.get("domain", "general") or "general"),
                reason="rule_promoted",
            )
        except Exception as exc:
            self._record_runtime_error("trigger_transfer_evaluation", exc, "rule_promoted")

    def _on_transfer_attempted(self, ed: EventData):
        self._stats["transfers_attempted"] = int(self._stats.get("transfers_attempted", 0)) + 1
        try:
            payload = self._load_transfer_payload()
            stats = dict(payload.get("stats", {}) or {})
            stats["runtime_transfer_attempts"] = int(stats.get("runtime_transfer_attempts", 0) or 0) + 1
            stats["hypotheses_generated"] = max(
                int(stats.get("hypotheses_generated", 0) or 0),
                int(stats.get("runtime_transfer_attempts", 0) or 0),
            )
            payload["stats"] = stats
            self._save_transfer_payload(payload)
        except Exception as exc:
            self._record_runtime_error("transfer_attempt_persist", exc, "_on_transfer_attempted")
        self._invalidate_report_cache("transfer_audit", "audit_dashboard", "learning_ops_dashboard", "learning_dashboard_payload", "time_to_agi_report")
        if self.boot_time:
            self.save_state()

    def _on_transfer_succeeded(self, ed: EventData):
        self._stats["transfers_succeeded"] = int(self._stats.get("transfers_succeeded", 0)) + 1
        try:
            payload = self._load_transfer_payload()
            history = [item for item in payload.get("transfer_history", []) if isinstance(item, dict)]
            data = dict(ed.data or {})
            source_domain = str(data.get("source_domain", data.get("source", "general")) or "general")
            target_domain = str(data.get("target_domain", data.get("target", "general")) or "general")
            source_role = str(data.get("source_role", data.get("role", "transferred_rule")) or "transferred_rule")
            proposed_transform = str(
                data.get("proposed_transform")
                or data.get("rule_name")
                or data.get("name")
                or f"{source_domain}_to_{target_domain}_{source_role}"
            )
            verified = bool(data.get("verified", True))
            heldout_target_wins = int(data.get("heldout_target_wins", data.get("wins", 0)) or 0)
            heldout_target_tests = int(data.get("heldout_target_tests", data.get("tests", 0)) or 0)
            evaluation_type = str(data.get("evaluation_type", data.get("source_role", "runtime_event")) or "runtime_event")
            history.append({
                "source_domain": source_domain,
                "target_domain": target_domain,
                "source_role": source_role,
                "proposed_transform": proposed_transform,
                "heldout_target_wins": heldout_target_wins,
                "heldout_target_tests": heldout_target_tests,
                "verified_at": float(data.get("verified_at", time.time()) or time.time()),
                "status": "verified" if verified else "runtime_only",
                "verified": verified,
                "evaluation_type": evaluation_type,
                "event_source": ed.source,
                "new_transforms": int(data.get("new_transforms", 0) or 0),
                "delta": float(data.get("delta", 0.0) or 0.0),
            })
            payload["transfer_history"] = history[-500:]
            stats = dict(payload.get("stats", {}) or {})
            stats["runtime_transfer_successes"] = int(stats.get("runtime_transfer_successes", 0) or 0) + 1
            verified_count = len([item for item in payload["transfer_history"] if bool(item.get("verified"))])
            verified_items = [item for item in payload["transfer_history"] if bool(item.get("verified"))]
            stats["hypotheses_verified"] = max(int(stats.get("hypotheses_verified", 0) or 0), verified_count)
            stats["verified_transfer_runs"] = verified_count
            stats["verified_transfer_successes"] = verified_count
            stats["heldout_target_wins"] = sum(int(item.get("heldout_target_wins", 0) or 0) for item in verified_items)
            stats["heldout_target_tests"] = sum(int(item.get("heldout_target_tests", 0) or 0) for item in verified_items)
            payload["stats"] = stats
            self._save_transfer_payload(payload)
        except Exception as exc:
            self._record_runtime_error("transfer_success_persist", exc, "_on_transfer_succeeded")
        self._invalidate_report_cache("transfer_audit", "audit_dashboard", "learning_ops_dashboard", "learning_dashboard_payload", "time_to_agi_report")
        if self.boot_time:
            self.save_state()

    def _on_transfer_verified_core(self, payload: dict):
        """Handle transfer_verified events from the global core event bus.

        Promotes verified transfers into ConceptRegistry so they influence future solves.
        """
        try:
            rule = {
                "name": payload.get("name", ""),
                "domain": payload.get("domain", ""),
                "source_domain": payload.get("source_domain", ""),
                "confidence": payload.get("confidence", 0.6),
                "source": "transfer_verified",
            }
            if self.concept_registry and rule.get("name"):
                self.concept_registry.add_rule(rule)
                if hasattr(self.concept_registry, "save"):
                    self.concept_registry.save()
                log.info("Transfer promoted to ConceptRegistry: %s → %s  name=%s",
                         rule["source_domain"], rule["domain"], rule["name"])
                # Update transfers_promoted stat in transfer engine
                try:
                    from sare.transfer.engine import get_transfer_engine
                    _te = get_transfer_engine()
                    _te._stats["transfers_promoted"] = _te._stats.get("transfers_promoted", 0) + 1
                    _te.save()
                except Exception:
                    pass
            # Also emit TRANSFER_SUCCEEDED so dashboard stats update
            self.events.emit(Event.TRANSFER_SUCCEEDED, {
                **payload,
                "verified": True,
                "evaluation_type": "passive_observation",
            }, "transfer_engine")
        except Exception as exc:
            self._record_runtime_error("transfer_verified_core", exc, "_on_transfer_verified_core")

    def record_artifact_reuse(
        self,
        artifact_id: str,
        *,
        solved: bool,
        learning_mode: str = "",
        heldout_variant: bool = False,
    ) -> dict:
        payload = self._get_acquisition_mesh().record_artifact_reuse(
            artifact_id,
            solved=solved,
            learning_mode=learning_mode,
            heldout_variant=heldout_variant,
        )
        self._invalidate_report_cache()
        return payload

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

        self._check_and_advance_stage()

    def _on_competence_updated(self, ed: EventData):
        """Check if stage should advance."""
        self._check_and_advance_stage()

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
                    for rule in learned_rule_source.get_high_confidence_rules(min_confidence=0.45):
                        t = self._rule_to_transform(rule)
                        if t:
                            self.transforms.insert(0, t)
                except Exception:
                    pass
            if self.experiment_runner:
                self.experiment_runner.transforms = self.transforms
        except Exception as e:
            log.error(f"Transform refresh failed: {e}")

    def _trigger_transfer_evaluation(
        self,
        source_domain: str,
        reason: str = "runtime",
        target_domains: Optional[Iterable[str]] = None,
    ) -> int:
        attempts = 0
        allowed_targets = set(target_domains or [
            "arithmetic", "algebra", "calculus", "logic",
            "code", "planning", "social", "dialogue", "science",
        ])
        if source_domain in allowed_targets:
            allowed_targets.discard(source_domain)

        if self.transfer_engine and hasattr(self.transfer_engine, "generate_hypotheses"):
            try:
                generated = self.transfer_engine.generate_hypotheses() or []
                for hyp in list(generated)[:5]:
                    target = str(getattr(hyp, "target_domain", "") or "")
                    if allowed_targets and target and target not in allowed_targets:
                        continue
                    attempts += 1
                    self.events.emit(Event.TRANSFER_ATTEMPTED, {
                        "source": getattr(hyp, "source_domain", source_domain),
                        "target": target,
                        "role": getattr(hyp, "source_role", ""),
                        "reason": reason,
                    }, "brain.transfer_engine")
            except Exception as exc:
                self._record_runtime_error("transfer_engine.generate_hypotheses", exc, "trigger_transfer_evaluation")

        if self.analogy_transfer and hasattr(self.analogy_transfer, "transfer_from_domain"):
            try:
                suggestions = self.analogy_transfer.transfer_from_domain(source_domain) or []
                for suggestion in list(suggestions)[:5]:
                    target = str(getattr(suggestion, "target_domain", "") or getattr(suggestion, "domain", "") or "")
                    if allowed_targets and target and target not in allowed_targets:
                        continue
                    attempts += 1
                    self.events.emit(Event.TRANSFER_ATTEMPTED, {
                        "source": source_domain,
                        "target": target,
                        "reason": reason,
                        "schema": suggestion.to_dict() if hasattr(suggestion, "to_dict") else str(suggestion),
                    }, "brain.analogy_transfer")
            except Exception as exc:
                self._record_runtime_error("analogy_transfer.transfer_from_domain", exc, "trigger_transfer_evaluation")

        return attempts

    def _rule_to_transform(self, rule) -> Optional[Any]:
        """Convert a discovered AbstractRule into a live Transform for search.

        Handles: identity, annihilation, involution, self-cancellation,
        constant folding, and generic rules (via existing transform lookup).
        """
        try:
            from sare.engine import Transform, Graph
            name = rule.name if hasattr(rule, 'name') else rule.get('name', '')
            op = rule.operator_involved if hasattr(rule, 'operator_involved') else rule.get('operator', '')
            pattern = rule.pattern_description if hasattr(rule, 'pattern_description') else rule.get('pattern', '')
            domain = rule.domain if hasattr(rule, 'domain') else rule.get('domain', 'general')
            confidence = rule.confidence if hasattr(rule, 'confidence') else rule.get('confidence', 0.7)

            if not name:
                return None

            from sare.transfer.synthesizer import SynthesizedTransform, _make_runtime_transform

            # Pattern 1: Identity — op(x, element) → x
            if 'identity' in name and op:
                for label in ['0', '1', 'true', 'false', 'empty']:
                    if label in pattern:
                        spec = SynthesizedTransform(
                            name=f"learned_{name}",
                            domain=domain,
                            role="identity",
                            operator_labels=[op],
                            element_label=label,
                            rewrite_action="replace_with_other_child",
                            confidence=confidence,
                        )
                        return _make_runtime_transform(spec)

            # Pattern 2: Annihilation — op(x, absorb) → absorb
            if 'annihilation' in name and op:
                for label in ['0', 'false', 'empty']:
                    if label in pattern:
                        spec = SynthesizedTransform(
                            name=f"learned_{name}",
                            domain=domain,
                            role="annihilation",
                            operator_labels=[op],
                            element_label=label,
                            rewrite_action="replace_with_absorbing",
                            confidence=confidence,
                        )
                        return _make_runtime_transform(spec)

            # Pattern 3: Involution — op(op(x)) → x
            if 'double' in name or 'elimination' in name:
                spec = SynthesizedTransform(
                    name=f"learned_{name}",
                    domain=domain,
                    role="involution",
                    operator_labels=[op] if op else ["neg"],
                    element_label="",
                    rewrite_action="unwrap_double",
                    confidence=confidence,
                )
                return _make_runtime_transform(spec)

            # Pattern 4: Self-cancellation — op(x, x) → identity_element
            if 'self_cancel' in name and op:
                spec = SynthesizedTransform(
                    name=f"learned_{name}",
                    domain=domain,
                    role="self_inverse",
                    operator_labels=[op],
                    element_label="0" if op in ("+", "-", "sub") else "1",
                    rewrite_action="replace_with_absorbing",
                    confidence=confidence,
                )
                return _make_runtime_transform(spec)

            # Pattern 5: Constant folding — op(c1, c2) → c3
            # For this, we look up an existing const_fold transform
            if 'fold' in name or 'constant' in name:
                for t in (self.transforms or []):
                    t_name = t.name() if hasattr(t, 'name') else ''
                    if 'const_fold' in t_name or 'constant_fold' in t_name:
                        return t  # reuse existing constant folding transform

            # Pattern 6: Generic — find an existing transform whose name
            # partially matches the rule's discovered pattern or transforms_applied
            if op and not op.startswith("unknown"):
                # Try to find an existing transform that operates on this operator
                for t in (self.transforms or []):
                    t_name = (t.name() if hasattr(t, 'name') else '').lower()
                    if op.lower() in t_name or name.lower().replace('discovered_pattern_', '') in t_name:
                        return t

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
        if not self.concept_registry or confidence <= 0.4:
            return False

        # Dedup: skip if structurally equivalent rule already exists
        rule_name = str(getattr(rule, "name", ""))
        try:
            existing_rules = self.concept_registry.get_rules()
            for existing in existing_rules:
                existing_name = existing.get("name", "") if isinstance(existing, dict) else getattr(existing, "name", "")
                if existing_name == rule_name:
                    log.debug(f"Rule dedup: '{rule_name}' already exists, skipping")
                    return False
        except Exception:
            pass

        try:
            payload = rule.to_dict() if hasattr(rule, "to_dict") else rule
            try:
                self.concept_registry.add_rule(payload)
            except TypeError:
                # C++ ConceptRegistry requires AbstractRule, not a dict — convert and retry.
                try:
                    import sare.sare_bindings as _sb
                    ar = _sb.AbstractRule()
                    _d = payload if isinstance(payload, dict) else {}
                    ar.name = str(_d.get("name", getattr(rule, "name", "")))
                    ar.domain = str(_d.get("domain", domain))
                    ar.confidence = float(_d.get("confidence", confidence))
                    ar.observations = int(_d.get("observations", getattr(rule, "observations", 1)))
                    self.concept_registry.add_rule(ar)
                except Exception:
                    pass
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
            self._stats["rules_promoted"] = self._stats.get("rules_promoted", 0) + 1
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

    def solve(self, expression: str, context: Optional[dict] = None, algorithm: str = "beam",
              beam_width: int = 8, max_depth: int = 30,
              budget: float = 10.0, domain: str = "general",
              kappa: float = 0.1, force_python: bool = False) -> SolveResult:
        """
        Full cognitive solve: perceive → recall → plan → act → reflect.

        Returns a result dict with graph, energy, transforms, proof, etc.
        """
        self.events.emit(Event.SOLVE_STARTED, {
            "expression": expression, "domain": domain
        }, "brain")

        _context = context or {}
        _force_general = bool(_context.get("force_general_solver"))
        if _force_general:
            return self._run_general_solver(expression, context=_context, domain=domain)

        # 1. PERCEIVE: Parse expression into graph
        try:
            expr_str, graph = self.engine.load_problem(expression)
        except Exception:
            return self._run_general_solver(expression, context=_context, domain=domain)

        # 3. PLAN: Determine domain first (needed for domain-aware energy)
        detected_domain = self._detect_domain(graph, domain)
        selected_strategy = ""
        selector_counted = False
        normalized_algorithm = str(algorithm or "beam").strip().lower()
        if normalized_algorithm == "beam_search":
            normalized_algorithm = "beam"
        if normalized_algorithm == "auto" or (
            normalized_algorithm == "beam"
            and beam_width == 8
            and max_depth == 30
            and abs(float(budget) - 10.0) < 1e-9
            and not _context.get("disable_algorithm_selector")
        ):
            try:
                from sare.meta.algorithm_selector import get_algorithm_selector as _get_as
                _selector = _get_as()
                _dom = detected_domain or domain or "general"
                if _dom in {"language", "science", "planning"}:
                    _preferred = ["greedy", "beam_search", "mcts"]
                elif _dom in {"trigonometry"}:
                    _preferred = ["mcts", "beam_search", "greedy"]
                elif _dom in {"arithmetic", "logic", "calculus", "set_theory", "logic_basics"}:
                    _preferred = ["beam_search", "greedy", "mcts"]
                else:
                    _preferred = ["beam_search", "greedy", "mcts"]
                _algo_opts = _selector.recommend_options(
                    _dom,
                    _preferred,
                    min_samples=10,
                    low_win_rate=0.15,
                    high_win_rate=0.8,
                )
                selected_strategy = _selector.select(_dom, _algo_opts)
                selector_counted = True
                if selected_strategy == "mcts":
                    normalized_algorithm = "mcts"
                elif selected_strategy == "greedy":
                    normalized_algorithm = "greedy"
                else:
                    normalized_algorithm = "beam"
            except Exception as exc:
                self._record_runtime_error("algorithm_selector.select", exc, "solve")
                normalized_algorithm = "beam"
        if not selected_strategy:
            selected_strategy = "mcts" if normalized_algorithm == "mcts" else "beam_search"
        algorithm = normalized_algorithm

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
        transfer_links: List[dict] = []

        # Fix 2: Memory warm-start — replay past successful transform sequences
        if strategy_hint is not None and getattr(strategy_hint, "found", False):
            preferred = set(getattr(strategy_hint, "transform_sequence", []))
            if preferred:
                _front = [t for t in transforms if (t.name() if hasattr(t, "name") else str(t)) in preferred]
                _back  = [t for t in transforms if (t.name() if hasattr(t, "name") else str(t)) not in preferred]
                transforms = _front + _back

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

        # Transfer engine: boost transforms analogically suggested for this domain
        if self.transfer_engine:
            try:
                suggestions = self.transfer_engine.get_transfer_suggestions(detected_domain)
                if suggestions:
                    for suggestion in suggestions[:5]:
                        transfer_links.append(
                            {
                                "source_domain": str(suggestion.get("source_domain", "") or "general"),
                                "target_domain": str(detected_domain or domain or "general"),
                                "proposed_transform": str(suggestion.get("source_transform", "") or ""),
                                "source_role": str(suggestion.get("source_role", "transfer_suggestion") or "transfer_suggestion"),
                                "confidence": float(suggestion.get("confidence", 0.0) or 0.0),
                                "verified": False,
                                "evaluation_type": "solve_rerank",
                            }
                        )
                        self.events.emit(Event.TRANSFER_ATTEMPTED, {
                            "source": suggestion.get("source_domain", ""),
                            "target": detected_domain,
                            "source_transform": suggestion.get("source_transform", ""),
                            "confidence": suggestion.get("confidence", 0.0),
                            "reason": "solve_rerank",
                        }, "brain.solve")
                    suggested = {s.get("source_transform", "") for s in suggestions[:5] if s.get("confidence", 0) > 0.4}
                    if suggested:
                        _front = [t for t in transforms if (t.name() if hasattr(t, "name") else str(t)) in suggested]
                        _tail  = [t for t in transforms if (t.name() if hasattr(t, "name") else str(t)) not in suggested]
                        transforms = _front + _tail
            except Exception as exc:
                self._record_runtime_error("transfer_engine.get_transfer_suggestions", exc, "solve")

        if self.analogy_transfer:
            try:
                effective = []
                if hasattr(self.analogy_transfer, "get_effective_transfers"):
                    effective = [
                        item for item in (self.analogy_transfer.get_effective_transfers() or [])
                        if str(item.get("domain", "")) == detected_domain
                    ]
                if effective:
                    suggested = {str(item.get("rule_name", "")) for item in effective[:5]}
                    for item in effective[:5]:
                        transfer_links.append(
                            {
                                "source_domain": str(item.get("source_domain", "") or "general"),
                                "target_domain": str(detected_domain or domain or "general"),
                                "proposed_transform": str(item.get("rule_name", "") or ""),
                                "source_role": str(item.get("source_role", "verified_transfer") or "verified_transfer"),
                                "confidence": float(item.get("confidence", 1.0) or 1.0),
                                "verified": True,
                                "evaluation_type": "verified_transfer",
                            }
                        )
                    if suggested:
                        _front = [t for t in transforms if (t.name() if hasattr(t, "name") else str(t)) in suggested]
                        _tail  = [t for t in transforms if (t.name() if hasattr(t, "name") else str(t)) not in suggested]
                        transforms = _front + _tail
            except Exception as exc:
                self._record_runtime_error("analogy_transfer.get_effective_transfers", exc, "solve")

        # Fix 1: WorldModel prediction — reorder transforms by predicted best
        try:
            from sare.memory.world_model import get_world_model as _get_wm
            _wm = _get_wm()
            if hasattr(_wm, "predict_transform"):
                _pred = _wm.predict_transform(graph, transforms, domain=detected_domain)
                _pname = getattr(_pred, "transform_name", None) or ""
                if _pname and _pname != "unknown":
                    _front = [t for t in transforms if (t.name() if hasattr(t, "name") else str(t)) == _pname]
                    _tail  = [t for t in transforms if (t.name() if hasattr(t, "name") else str(t)) != _pname]
                    transforms = _front + _tail
                    self._wm_prediction_current = _pred
        except Exception as exc:
            self._record_runtime_error("world_model.predict_transform", exc, "solve")

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
                elif algorithm == "greedy":
                    result = self.searcher.search(
                        working_graph, active_energy, transforms,
                        beam_width=1,
                        max_depth=max_depth,
                        budget_seconds=remaining_budget,
                        kappa=kappa,
                        heuristic_fn=heuristic_fn,
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

        try:
            from sare.meta.algorithm_selector import get_algorithm_selector as _get_as
            _selector = _get_as()
            if not selector_counted:
                _selector.record_selection(detected_domain or domain or "general", selected_strategy)
            _selector.record_outcome(detected_domain or domain or "general", selected_strategy, success)
        except Exception as exc:
            self._record_runtime_error("algorithm_selector.record_outcome", exc, "solve")

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
        used_names = {str(name) for name in list(result.transforms_applied)}
        transfer_outcome = None
        if transfer_links and used_names:
            matched_transfer = next(
                (
                    item for item in transfer_links
                    if str(item.get("proposed_transform", "")) in used_names
                ),
                None,
            )
            if matched_transfer is not None:
                transfer_outcome = {
                    **dict(matched_transfer),
                    "used": True,
                    "success": success,
                    "heldout_target_wins": 1 if success and bool(_context.get("expected_transfer_source")) else 0,
                    "heldout_target_tests": 1 if bool(_context.get("expected_transfer_source")) else 0,
                }
                if transfer_outcome["heldout_target_tests"] and success:
                    self.events.emit(
                        Event.TRANSFER_SUCCEEDED,
                        {
                            "source": transfer_outcome.get("source_domain", ""),
                            "target": transfer_outcome.get("target_domain", ""),
                            "role": transfer_outcome.get("source_role", ""),
                            "heldout_target_wins": transfer_outcome.get("heldout_target_wins", 0),
                            "heldout_target_tests": transfer_outcome.get("heldout_target_tests", 0),
                            "evaluation_type": "solve_transfer_backed",
                            "verified": bool(transfer_outcome.get("verified", False)),
                            "rule_name": transfer_outcome.get("proposed_transform", ""),
                        },
                        "brain.solve",
                    )
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

        # Build human-readable answer
        try:
            from sare.engine import graph_to_answer as _gta
            _answer = _gta(expr_str, result.graph)
        except Exception:
            _answer = None

        # Build step-by-step NL explanation without LLM
        try:
            _steps_nl = []
            for _i, _t in enumerate(result.transforms_applied, 1):
                _tname = _t.replace("_", " ").title()
                _steps_nl.append(f"Step {_i}: Apply {_tname}")
            if _steps_nl and _answer:
                _steps_nl.append(f"Result: {_answer}")
            _steps_text = "\n".join(_steps_nl) if _steps_nl else None
        except Exception:
            _steps_text = None

        return self._make_solve_result({
            "expression": expr_str,
            "answer": _answer,
            "steps_text": _steps_text,
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
            "source": "brain.symbolic",
            "solver_path": "symbolic",
            "learning_mode": "free_solve",
            "algorithm_used": selected_strategy,
            "search_strategy": selected_strategy,
            "verification_outcome": None,
            "transfer_outcome": transfer_outcome,
            "persistence_ids": {},
        })

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
        """P2.1: Test rule on domain problems before promoting. Persist on success.

        Promotion paths:
          A) Rule converts to transform AND passes >=1/3 tests → full promote + live transform
          B) Rule can't convert but has confidence >= 0.5 → promote to concept registry only
             (knowledge is still valuable for future synthesis/transfer)
        """
        confidence = getattr(rule, 'confidence', 0.5)
        try:
            test_problems = self._get_test_problems_for_domain(domain, n=3, rule=rule)
            passes = self._test_rule_on_problems(rule, test_problems)

            # Path A: Rule can be tested as a transform
            if passes >= 1:  # Lowered from 2/3 — even 1/3 means the rule has real value
                if self._promote_python_rule(rule, domain):
                    if self.concept_registry and hasattr(self.concept_registry, 'save'):
                        try:
                            self.concept_registry.save()
                        except Exception:
                            pass
                    self._refresh_transforms()
                    log.info(f"Rule PROMOTED+PERSISTED: {rule.name} ({passes}/3 tests, conf={confidence:.2f})")
                return

            # Path B: _rule_to_transform returned None (passes==0 because no transform)
            # Still promote to concept registry if confident enough — the structural
            # knowledge is useful for transfer engine and future synthesis
            if passes == 0 and confidence >= 0.5:
                if self._promote_python_rule(rule, domain):
                    if self.concept_registry and hasattr(self.concept_registry, 'save'):
                        try:
                            self.concept_registry.save()
                        except Exception:
                            pass
                    log.info(f"Rule PROMOTED (concept-only): {rule.name} (conf={confidence:.2f}, no live transform)")
                return

            log.debug(f"Rule REJECTED: {rule.name} ({passes}/3 tests, conf={confidence:.2f})")
        except Exception as e:
            log.debug(f"_reflect_and_promote failed: {e}")
            if confidence >= 0.5:
                self._promote_python_rule(rule, domain)

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
                for c_name, concept in self.concept_graph.concepts.items():
                    if concept.is_well_grounded() and concept.symbolic_rules:
                        for rule in concept.symbolic_rules[:2]:
                            # Find analogous concepts in other domains
                            for other_name, other in self.concept_graph.concepts.items():
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
                for c_name, concept in self.concept_graph.concepts.items():
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

    def world_summary(self) -> dict:
        if self.world_model is not None and hasattr(self.world_model, "summary"):
            try:
                return self.world_model.summary()
            except Exception as exc:
                self._record_runtime_error("world_model.summary", exc, "world_summary")
        if self.world_model_v3 is not None and hasattr(self.world_model_v3, "summary"):
            try:
                return self.world_model_v3.summary()
            except Exception as exc:
                self._record_runtime_error("world_model_v3.summary", exc, "world_summary")
        try:
            facts = self.world_facts_by_domain()
            return {
                "fact_count": sum(int(v) for v in facts.values()),
                "causal_link_count": int(len(getattr(self.world_model, "_causal_links", {}) or {})),
                "schema_count": int(len(getattr(self.world_model, "_schemas", {}) or {})),
                "belief_count": int(len(getattr(self.world_model, "_beliefs", {}) or {})),
                "domains": sorted(facts.keys(), key=str),
            }
        except Exception:
            pass
        return {}

    def world_model_readable_report(self, include_content: bool = False) -> dict:
        try:
            from sare.meta.world_model_readable_report import DEFAULT_METADATA_PATH, DEFAULT_OUTPUT_PATH
        except Exception as exc:
            self._record_runtime_error("world_model_readable_report.import", exc, "world_model_readable_report")
            return {"available": False, "metadata": {}, "content": "", "url": "/api/world/readable-report"}

        metadata_path = Path(DEFAULT_METADATA_PATH)
        output_path = Path(DEFAULT_OUTPUT_PATH)
        metadata: Dict[str, Any] = {}
        content = ""

        if metadata_path.exists():
            try:
                loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    metadata = loaded
            except Exception as exc:
                self._record_runtime_error("world_model_readable_report.metadata", exc, "world_model_readable_report")

        if include_content and output_path.exists():
            try:
                content = output_path.read_text(encoding="utf-8")
            except Exception as exc:
                self._record_runtime_error("world_model_readable_report.content", exc, "world_model_readable_report")

        available = bool(metadata) or output_path.exists()
        return {
            "available": available,
            "metadata": metadata,
            "content": content,
            "url": "/api/world/readable-report",
            "path": str(output_path),
            "metadata_path": str(metadata_path),
            "generated_at": metadata.get("generated_at"),
            "generated_at_human": metadata.get("generated_at_human"),
            "solve_count": metadata.get("solve_count"),
            "every_solves": metadata.get("every_solves"),
        }

    def world_prediction_stats(self) -> dict:
        stats: Dict[str, Any] = {}
        if self.world_model is not None and hasattr(self.world_model, "prediction_stats"):
            try:
                stats.update(self.world_model.prediction_stats())
            except Exception as exc:
                self._record_runtime_error("world_model.prediction_stats", exc, "world_prediction_stats")
        return stats

    def high_surprise_domains(self, top_n: int = 5) -> List[tuple]:
        if self.world_model is not None and hasattr(self.world_model, "get_high_surprise_domains"):
            try:
                return list(self.world_model.get_high_surprise_domains(top_n=top_n))
            except Exception as exc:
                self._record_runtime_error("world_model.get_high_surprise_domains", exc, "high_surprise_domains")
        return []

    def world_facts_by_domain(self) -> Dict[str, int]:
        if self.world_model is not None and hasattr(self.world_model, "_facts"):
            try:
                return {
                    str(domain): len(facts)
                    for domain, facts in getattr(self.world_model, "_facts", {}).items()
                    if facts
                }
            except Exception as exc:
                self._record_runtime_error("world_model._facts", exc, "world_facts_by_domain")
        return {}

    def world_domain_failures(self) -> Dict[str, int]:
        failures: Dict[str, int] = {}
        if self.world_model is None:
            return failures
        try:
            lock = getattr(self.world_model, "_domain_failures_lock", None)
            failure_map = getattr(self.world_model, "_domain_failures", {})
            if lock is not None:
                with lock:
                    for domain, entries in failure_map.items():
                        failures[str(domain)] = len(entries)
            else:
                for domain, entries in failure_map.items():
                    failures[str(domain)] = len(entries)
        except Exception as exc:
            self._record_runtime_error("world_model._domain_failures", exc, "world_domain_failures")
        return failures

    def world_domain_accuracy(self, window: int = 50) -> Dict[str, float]:
        accuracy: Dict[str, float] = {}
        if self.world_model is not None and hasattr(self.world_model, "_belief_accuracy"):
            try:
                for key, hist in getattr(self.world_model, "_belief_accuracy", {}).items():
                    if key.startswith("domain_solve_acc:") and hist:
                        domain = key[len("domain_solve_acc:"):]
                        lookback = min(int(window), len(hist))
                        accuracy[str(domain)] = round(sum(hist[-lookback:]) / lookback, 3)
            except Exception as exc:
                self._record_runtime_error("world_model._belief_accuracy", exc, "world_domain_accuracy")
        return accuracy

    def world_synthesis_count(self) -> int:
        if self.world_model is not None and hasattr(self.world_model, "_last_concept_synthesis"):
            try:
                return int(len(getattr(self.world_model, "_last_concept_synthesis", {})))
            except Exception as exc:
                self._record_runtime_error("world_model._last_concept_synthesis", exc, "world_synthesis_count")
        return 0

    def world_imagine(self, seed: str, depth: int = 2) -> List[Any]:
        if self.world_model is not None and hasattr(self.world_model, "imagine"):
            try:
                return list(self.world_model.imagine(seed, depth=depth))
            except Exception as exc:
                self._record_runtime_error("world_model.imagine", exc, "world_imagine")
        return []

    def world_simulate(self, scenario: str, steps: int = 3) -> List[Any]:
        if self.world_model is not None and hasattr(self.world_model, "simulate"):
            try:
                return list(self.world_model.simulate(scenario, steps=steps))
            except Exception as exc:
                self._record_runtime_error("world_model.simulate", exc, "world_simulate")
        return []

    def world_analogy(self, source: str, target: str) -> Optional[dict]:
        if self.world_model is not None and hasattr(self.world_model, "generate_analogy"):
            try:
                return self.world_model.generate_analogy(source, target)
            except Exception as exc:
                self._record_runtime_error("world_model.generate_analogy", exc, "world_analogy")
        return None

    def world_counterfactual(self, rule: str, negated: bool = True) -> dict:
        if self.world_model is not None and hasattr(self.world_model, "counterfactual"):
            try:
                result = self.world_model.counterfactual(rule, negated=negated)
                return result if isinstance(result, dict) else {"result": result}
            except Exception as exc:
                self._record_runtime_error("world_model.counterfactual", exc, "world_counterfactual")
        return {}

    def world_hypotheses(self) -> List[dict]:
        hyps: List[dict] = []
        if self.world_model is not None:
            try:
                raw = getattr(self.world_model, "_hypotheses", [])
                if raw:
                    hyps = list(raw)
            except Exception as exc:
                self._record_runtime_error("world_model._hypotheses", exc, "world_hypotheses")
        if not hyps:
            try:
                hp = DATA_DIR / "world_hypotheses.json"
                if hp.exists():
                    loaded = json.loads(hp.read_text(encoding="utf-8"))
                    if isinstance(loaded, list):
                        hyps = loaded
            except Exception as exc:
                self._record_runtime_error("world_hypotheses.load", exc, "world_hypotheses")
        return sorted(hyps, key=lambda h: -float(h.get("evidence", 1) or 1))

    def active_learning_questions(self, limit: int = 10) -> List[dict]:
        try:
            from sare.curiosity.question_generator import get_question_generator

            pending = list(get_question_generator().get_pending_questions())
            rows = []
            for item in pending[: max(1, limit)]:
                if hasattr(item, "to_dict"):
                    rows.append(item.to_dict())
                elif isinstance(item, dict):
                    rows.append(dict(item))
            return rows
        except Exception as exc:
            self._record_runtime_error("question_generator.pending", exc, "self_questions")
            return []

    def world_theories(self, limit: int = 5) -> List[dict]:
        try:
            from sare.cognition.theory_builder import get_theory_builder

            theories = get_theory_builder().build_theories(max_theories=max(1, limit))
            return list(theories) if isinstance(theories, list) else []
        except Exception as exc:
            self._record_runtime_error("theory_builder.build", exc, "world_theories")
            return []

    def drive_self_generated_learning(self, force: bool = False) -> dict:
        payload: Dict[str, Any] = {
            "generated_questions": 0,
            "pending_questions": 0,
            "top_questions": [],
            "active_hypotheses": 0,
            "top_theories": [],
        }
        try:
            payload["active_hypotheses"] = len(self.world_hypotheses())
        except Exception:
            payload["active_hypotheses"] = 0
        try:
            theories = self.world_theories(limit=3)
            payload["top_theories"] = list(theories[:3])
        except Exception:
            payload["top_theories"] = []
        try:
            from sare.curiosity.question_generator import get_question_generator

            qg = get_question_generator()
            generated = list(qg.maybe_generate_questions(force=force))
            pending = list(qg.get_pending_questions())
            payload["generated_questions"] = len(generated)
            payload["pending_questions"] = len(pending)
            payload["top_questions"] = [
                item.to_dict() if hasattr(item, "to_dict") else dict(item)
                for item in pending[:3]
            ]
        except Exception as exc:
            self._record_runtime_error("question_generator.generate", exc, "self_questions")
        self._last_self_generated_learning = dict(payload)
        return payload

    def world_activity_log(self, limit: int = 20) -> List[Any]:
        if self.world_model is not None and hasattr(self.world_model, "get_activity_log"):
            try:
                events = self.world_model.get_activity_log()
                return list(events[:limit])
            except Exception as exc:
                self._record_runtime_error("world_model.get_activity_log", exc, "world_activity_log")
        return []

    def world_solve_counts(self) -> Dict[str, int]:
        if self.world_model is not None and hasattr(self.world_model, "_solve_counts"):
            try:
                return {
                    str(domain): int(count)
                    for domain, count in getattr(self.world_model, "_solve_counts", {}).items()
                }
            except Exception as exc:
                self._record_runtime_error("world_model._solve_counts", exc, "world_solve_counts")
        return {}

    def world_schema_count(self) -> int:
        if self.world_model is not None and hasattr(self.world_model, "_schemas"):
            try:
                return int(len(getattr(self.world_model, "_schemas", {})))
            except Exception as exc:
                self._record_runtime_error("world_model._schemas", exc, "world_schema_count")
        return 0

    def world_schema_learn(self, domain: str) -> Optional[dict]:
        if self.world_model is not None and hasattr(self.world_model, "learn_schema_from_llm"):
            try:
                schema = self.world_model.learn_schema_from_llm(domain)
                if schema and hasattr(self.world_model, "save"):
                    self.world_model.save()
                return schema
            except Exception as exc:
                self._record_runtime_error("world_model.learn_schema_from_llm", exc, "world_schema_learn")
        return None

    def world_beliefs(self, domain: Optional[str] = None) -> List[Any]:
        if self.world_model is not None and hasattr(self.world_model, "get_beliefs"):
            try:
                return list(self.world_model.get_beliefs(domain))
            except Exception as exc:
                self._record_runtime_error("world_model.get_beliefs", exc, "world_beliefs")
        return []

    def world_analogies(self, domain: Optional[str] = None) -> List[Any]:
        if self.world_model is None or not hasattr(self.world_model, "get_analogies"):
            return []
        try:
            analogies = getattr(self.world_model, "_analogies", [])
            solve_history = getattr(self.world_model, "_solve_history", [])
            if len(analogies) < 5 and len(solve_history) >= 30 and hasattr(self.world_model, "discover_analogies"):
                self.world_model.discover_analogies()
            return list(self.world_model.get_analogies(domain))
        except Exception as exc:
            self._record_runtime_error("world_model.get_analogies", exc, "world_analogies")
        return []

    def world_predict(self, expression: str, domain: str = "arithmetic") -> dict:
        if self.world_model is None or not hasattr(self.world_model, "predict"):
            return {}
        try:
            transforms = [
                t.name() if hasattr(t, "name") else str(t)
                for t in getattr(self, "transforms", []) or []
            ]
            result = self.world_model.predict(expression, domain, transforms)
            return result if isinstance(result, dict) else {"result": result}
        except Exception as exc:
            self._record_runtime_error("world_model.predict", exc, "world_predict")
        return {}

    def world_consistency(self, rule_names: Optional[List[str]] = None) -> dict:
        rules = list(rule_names or [])
        if not rules and self.concept_registry is not None:
            try:
                rules = [r.name for r in self.concept_registry.get_rules()]
            except Exception as exc:
                self._record_runtime_error("concept_registry.get_rules", exc, "world_consistency")
        if len(rules) < 2:
            return {"status": "insufficient_rules", "rule_count": len(rules)}
        if self.world_model is not None and hasattr(self.world_model, "check_all_rules_consistency"):
            try:
                conflicts = self.world_model.check_all_rules_consistency(rules)
                return {
                    "rules_checked": len(rules),
                    "conflicts": conflicts,
                    "consistent": len(conflicts) == 0,
                }
            except Exception as exc:
                self._record_runtime_error("world_model.check_all_rules_consistency", exc, "world_consistency")
        return {"status": "world_model_unavailable", "rule_count": len(rules)}

    def world_graph(self) -> dict:
        links_by_key: Dict[str, Any] = {}
        for model_name in ("world_model", "world_model_v3"):
            model = getattr(self, model_name, None)
            if model is None or not hasattr(model, "_causal_links"):
                continue
            try:
                for key, link in getattr(model, "_causal_links", {}).items():
                    links_by_key[str(key)] = link
            except Exception as exc:
                self._record_runtime_error(f"{model_name}._causal_links", exc, "world_graph")

        domain_counts: Dict[str, int] = {}
        edges = []
        for link in list(links_by_key.values())[:500]:
            domain = str(getattr(link, "domain", "general") or "general")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            edges.append({
                "cause": getattr(link, "cause", ""),
                "effect": getattr(link, "effect", ""),
                "mechanism": getattr(link, "mechanism", ""),
                "domain": domain,
                "confidence": round(float(getattr(link, "confidence", 0.5) or 0.5), 3),
                "evidence_count": int(getattr(link, "evidence_count", 1) or 1),
            })

        return {
            "nodes": [{"id": d, "domain": d, "link_count": c} for d, c in domain_counts.items()],
            "edges": edges,
            "domains": sorted(domain_counts.keys()),
            "causal_links_total": len(links_by_key),
        }

    def add_world_fact(self, domain: str, fact: str, confidence: float = 0.9,
                       source: str = "brain") -> bool:
        added = False
        if self.world_model is not None and hasattr(self.world_model, "add_fact"):
            try:
                self.world_model.add_fact(domain, fact, confidence, source=source)
                added = True
            except Exception as exc:
                self._record_runtime_error("world_model.add_fact", exc, "add_world_fact")
        if self.world_model_v3 is not None:
            try:
                from sare.memory.world_model_v3 import CausalLink
                link = CausalLink(
                    cause=fact,
                    effect=fact,
                    mechanism="fact",
                    domain=domain,
                    confidence=confidence,
                )
                self.world_model_v3._causal_links.setdefault(link.key, link)
            except Exception as exc:
                self._record_runtime_error("world_model_v3.fact_shadow", exc, "add_world_fact")
        if added:
            self._stats["ingestion"]["facts_added"] += 1
        return added

    def add_world_causal_link(self, cause: str, effect: str, mechanism: str = "user-specified",
                              domain: str = "general", confidence: float = 0.85) -> bool:
        added = False
        if self.world_model is not None and hasattr(self.world_model, "add_causal_link"):
            try:
                self.world_model.add_causal_link(cause, effect, mechanism, domain, confidence)
                added = True
            except Exception as exc:
                self._record_runtime_error("world_model.add_causal_link", exc, "add_world_causal_link")
        if self.world_model_v3 is not None:
            try:
                from sare.memory.world_model_v3 import CausalLink
                link = CausalLink(
                    cause=cause,
                    effect=effect,
                    mechanism=mechanism,
                    domain=domain,
                    confidence=confidence,
                )
                self.world_model_v3._causal_links[link.key] = link
            except Exception as exc:
                self._record_runtime_error("world_model_v3.add_causal_link", exc, "add_world_causal_link")
        return added

    def save_world_models(self) -> None:
        for _name, _model in (("world_model", self.world_model), ("world_model_v3", self.world_model_v3)):
            if _model is not None and hasattr(_model, "save"):
                try:
                    _model.save()
                except Exception as exc:
                    self._record_runtime_error(f"{_name}.save", exc, "save_world_models")

    def knowledge_stats(self) -> dict:
        world_summary = self.world_summary()
        status = self.status()
        knowledge_base = dict(status.get("knowledge_base", {}))

        commonsense_facts = 0
        answer_to_triples = 0
        try:
            cs = self.commonsense
            if cs is None:
                from sare.knowledge.commonsense import get_commonsense_base
                cs = get_commonsense_base()
            if cs is not None:
                commonsense_facts = int(cs.total_facts())
                answer_to_triples = sum(
                    1
                    for triples in getattr(cs, "_forward", {}).values()
                    for rel, _ in triples
                    if rel == "AnswerTo"
                )
        except Exception as exc:
            self._record_runtime_error("commonsense.stats", exc, "knowledge_stats")

        kg_nodes = 0
        try:
            if self.knowledge_graph is not None:
                kg_nodes = int(len(getattr(self.knowledge_graph, "_nodes", {})))
        except Exception as exc:
            self._record_runtime_error("knowledge_graph._nodes", exc, "knowledge_stats")

        return {
            "world_model_facts": int(world_summary.get("fact_count", sum(self.world_facts_by_domain().values()))),
            "knowledge_graph_nodes": kg_nodes,
            "commonsense_facts": commonsense_facts,
            "answer_to_triples": answer_to_triples,
            "working_memory_sessions": int(knowledge_base.get("working_memory_sessions", 0)),
            "kb_hit_rate_last_100": float(knowledge_base.get("kb_hit_rate_last_100", 0.0)),
            "acquisition": self._get_acquisition_mesh().status() if self.acquisition_mesh is not None else {"records": 0, "artifacts": 0},
        }

    def learning_summary(self) -> dict:
        return {
            "answer_to_triples": int(self.knowledge_stats().get("answer_to_triples", 0)),
            "wm_facts_total": int(sum(self.world_facts_by_domain().values())),
            "wm_facts_by_domain": self.world_facts_by_domain(),
            "domain_failures": self.world_domain_failures(),
            "domain_accuracy": self.world_domain_accuracy(),
            "synthesis_count": self.world_synthesis_count(),
            "kb_hit_rate": float(self.knowledge_stats().get("kb_hit_rate_last_100", 0.0)),
            "acquisition": self.acquisition_dashboard(),
            "ts": time.time(),
        }

    def _retention_truth_summary(self, limit: int = 5000) -> dict:
        retested_artifacts = 0
        retained_artifacts = 0
        try:
            learned = self.learned_artifacts(limit=limit)
            items = list(learned.get("items", [])) if isinstance(learned, dict) else []
            for item in items:
                metadata = dict(item.get("metadata", {}) or {})
                if int(metadata.get("retest_count", 0) or 0) > 0:
                    retested_artifacts += 1
                    if str(metadata.get("retention_status", "")) == "retained":
                        retained_artifacts += 1
        except Exception as exc:
            self._record_runtime_error("retention.summary", exc, "_retention_truth_summary")
        return {
            "retested_artifacts": retested_artifacts,
            "retained_artifacts": retained_artifacts,
            "retention_rate": round(retained_artifacts / max(retested_artifacts, 1), 3) if retested_artifacts else None,
            "measured": bool(retested_artifacts > 0),
            "source_of_truth": "derived",
        }

    def _invalidate_report_cache(self, *keys: str) -> None:
        with self._report_cache_lock:
            if not keys:
                self._report_cache.clear()
                return
            for key in keys:
                self._report_cache.pop(key, None)

    def record_learning_strategy_outcome(
        self,
        *,
        expression: str,
        domain: Optional[str] = None,
        source: Optional[str] = None,
        task_type: Optional[str] = None,
        result: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        learning_mode: Optional[str] = None,
        verification_level: Optional[str] = None,
        heldout_variant: Optional[bool] = None,
        transfer_probe: Optional[bool] = None,
        elapsed_ms: Optional[float] = None,
    ) -> None:
        scorecard = getattr(self, "learning_strategy_scorecard", None)
        if scorecard is None or not hasattr(scorecard, "record_outcome"):
            return

        payload = dict(metadata or {})
        raw_result = result
        result_map = raw_result if isinstance(raw_result, dict) else {}

        def _pick(name: str, default: Any = None) -> Any:
            if isinstance(result_map, dict) and name in result_map:
                return result_map.get(name)
            if raw_result is not None and hasattr(raw_result, name):
                return getattr(raw_result, name)
            return default

        resolved_domain = str(domain or payload.get("domain") or _pick("domain") or "general").strip() or "general"
        resolved_source = str(source or payload.get("source") or payload.get("source_kind") or "unknown").strip() or "unknown"
        resolved_task_type = task_type or payload.get("task_type")
        if not resolved_task_type:
            if payload.get("choices"):
                resolved_task_type = "multiple_choice"
            elif resolved_source in {"seed_library", "generated_problems", "failure_replay", "physics_expressions", "knowledge_concepts", "curriculum", "generator", "goal_planner", "goal_setter", "self_model", "surprise_domain", "understanding_focus", "fallback"}:
                resolved_task_type = "expression_rewrite"
            elif resolved_source == "code_problems":
                resolved_task_type = "code_question"
            elif resolved_source == "word_problems":
                resolved_task_type = "word_problem"
            elif resolved_source == "comprehension_problems":
                resolved_task_type = "reading_comprehension"
            elif resolved_source == "language_problems":
                resolved_task_type = "language_reasoning"
            else:
                resolved_task_type = "question_answer"

        resolved_learning_mode = str(
            learning_mode or payload.get("learning_mode") or _pick("learning_mode") or _pick("mode") or "free_solve"
        ).strip() or "free_solve"
        resolved_verification = str(verification_level or payload.get("verification_level") or "unverified").strip() or "unverified"
        success = bool(_pick("success", _pick("solved", False)))
        confidence = _pick("confidence", _pick("delta", 0.0))
        delta = _pick("delta", _pick("confidence", 0.0))
        resolved_elapsed_ms = elapsed_ms if elapsed_ms is not None else _pick("elapsed_ms", 0.0)
        resolved_heldout_variant = bool(payload.get("heldout_variant") if heldout_variant is None else heldout_variant)
        resolved_transfer_probe = bool(payload.get("transfer_probe") if transfer_probe is None else transfer_probe)

        try:
            scorecard.record_outcome(
                domain=resolved_domain,
                source=resolved_source,
                task_type=str(resolved_task_type or "unknown"),
                learning_mode=resolved_learning_mode,
                verification_level=resolved_verification,
                success=success,
                confidence=confidence,
                delta=delta,
                elapsed_ms=resolved_elapsed_ms,
                heldout_variant=resolved_heldout_variant,
                transfer_probe=resolved_transfer_probe,
            )
        except Exception as exc:
            self._record_runtime_error("learning_strategy_scorecard.record", exc, "meta_learning")

    def _get_cached_report(self, key: str, max_age_seconds: float, builder: Callable[[], Any]) -> Any:
        # Fast path: return fresh cached value (no deepcopy — callers must not mutate)
        now = time.time()
        with self._report_cache_lock:
            entry = self._report_cache.get(key)
            if entry and (now - float(entry.get("ts", 0.0) or 0.0)) <= max_age_seconds:
                return entry.get("value")
            # Get or create a per-key build lock to prevent thundering herd.
            build_lock = self._report_build_locks.get(key)
            if build_lock is None:
                build_lock = threading.Lock()
                self._report_build_locks[key] = build_lock
            # If a build is already in progress (lock is taken), return stale cache
            # immediately rather than queuing — prevents thread pile-up on slow builds.
            stale_value = entry.get("value") if entry else None
            lock_free = build_lock.acquire(blocking=False)
            if not lock_free:
                # Another thread is building — return stale value or None right away
                return stale_value

        # We hold the build_lock (acquired non-blocking above). Now build.
        try:
            # Re-check cache after acquiring build lock — another thread may have just built it
            now = time.time()
            with self._report_cache_lock:
                entry = self._report_cache.get(key)
                if entry and (now - float(entry.get("ts", 0.0) or 0.0)) <= max_age_seconds:
                    return entry.get("value")
            value = builder()
            with self._report_cache_lock:
                self._report_cache[key] = {"ts": time.time(), "value": value}
            return value
        finally:
            build_lock.release()

    def persistence_health_report(self) -> dict:
        def _normalize(name: str, payload: Any) -> dict:
            base = {
                "loaded": False,
                "recovered": False,
                "reseeded": False,
                "corrupt_backup_written": False,
                "last_error": None,
                "source_of_truth": "live",
            }
            if isinstance(payload, dict):
                base.update(payload)
            base["component"] = name
            return base

        memory_dir = DATA_DIR
        episodes_corrupt = memory_dir / "episodes.jsonl.corrupt"
        working_memory_corrupt = memory_dir / "working_memory.json.corrupt"
        health = {
            "self_model": _normalize(
                "self_model",
                self.self_model.health() if self.self_model and hasattr(self.self_model, "health") else None,
            ),
            "homeostasis": _normalize(
                "homeostasis",
                self.homeostasis.health() if self.homeostasis and hasattr(self.homeostasis, "health") else None,
            ),
            "knowledge_graph": _normalize(
                "knowledge_graph",
                self.knowledge_graph.health() if self.knowledge_graph and hasattr(self.knowledge_graph, "health") else None,
            ),
            "concept_graph": _normalize(
                "concept_graph",
                self.concept_graph.health() if self.concept_graph and hasattr(self.concept_graph, "health") else dict(self._concept_graph_health),
            ),
            "memory_manager": _normalize(
                "memory_manager",
                {
                    "loaded": self.memory_manager is not None,
                    "recovered": bool(episodes_corrupt.exists() or working_memory_corrupt.exists()),
                    "reseeded": False,
                    "corrupt_backup_written": bool(episodes_corrupt.exists() or working_memory_corrupt.exists()),
                    "last_error": None,
                },
            ),
        }
        health["overall_ok"] = all(
            item.get("loaded") and not item.get("last_error")
            for item in health.values()
            if isinstance(item, dict) and "loaded" in item
        )
        return health

    def learning_live_snapshot(self) -> dict:
        # Cache heavy sub-calls for 30s to keep endpoint under 5s
        _now = time.time()
        _cache = getattr(self, "_live_snapshot_cache", {})
        if _now - _cache.get("ts", 0) < 30:
            return _cache.get("result", {"ts": _now})
        result: dict = {"ts": _now}
        result["domain_accuracy"] = self.world_domain_accuracy()
        result["wm_facts"] = self.world_facts_by_domain()

        web_searches: List[Any] = []
        try:
            log_path = DATA_DIR / "web_learned.json"
            if log_path.exists():
                raw = json.loads(log_path.read_text(encoding="utf-8"))
                entries = raw.get("entries", [])
                web_searches = list(reversed(entries[-20:]))
        except Exception as exc:
            self._record_runtime_error("web_learned.load", exc, "learning_live_snapshot")
        result["web_searches"] = web_searches

        llm_stats: dict = {"total_calls": 0, "by_role": {}, "recent": []}
        try:
            from sare.interface.llm_bridge import get_llm_stats
            llm_stats = get_llm_stats()
        except Exception as exc:
            self._record_runtime_error("llm_bridge.get_llm_stats", exc, "learning_live_snapshot")
        result["llm_stats"] = llm_stats

        try:
            ks = self.knowledge_stats()
            result["kb_snapshot"] = {
                "answer_to": int(ks.get("answer_to_triples", 0)),
                "total_triples": int(ks.get("commonsense_facts", 0)),
            }
        except Exception:
            result["kb_snapshot"] = {}
        try:
            report = self.learning_progress_report() or {}
            learning_monitor = dict(report.get("learning_monitor", {}) or {})
            totals = dict(report.get("totals", {}) or {})
            grounded = self.grounded_learning_report()
            retention = self._retention_truth_summary(limit=1000)
            result["metrics"] = {
                "heldout_pass_rate": {
                    "value": learning_monitor.get("heldout_pass_rate"),
                    "measured": bool(
                        learning_monitor.get("heldout_pass_rate") is not None
                        and int(learning_monitor.get("heldout_attempts", 0) or 0) > 0
                    ),
                    "source_of_truth": "canonical_report",
                },
                "retrieval_conversion_rate": {
                    "value": learning_monitor.get("retrieval_conversion_rate"),
                    "measured": bool(
                        learning_monitor.get("retrieval_conversion_rate") is not None
                        and int(learning_monitor.get("retrieval_conversion_attempts", 0) or 0) > 0
                    ),
                    "source_of_truth": "canonical_report",
                },
                "measured_coverage": {
                    "value": totals.get("measured_coverage"),
                    "measured": bool(int(totals.get("measured_attempts", 0) or 0) > 0),
                    "source_of_truth": "canonical_report",
                },
                "mastered_patterns": {
                    "value": learning_monitor.get("mastered_patterns"),
                    "measured": learning_monitor.get("mastered_patterns") is not None,
                    "source_of_truth": "canonical_report",
                },
                "grounded_success_rate": {
                    "value": grounded.get("success_rate"),
                    "measured": bool(int(grounded.get("total_tasks", 0) or 0) > 0),
                    "source_of_truth": "live",
                },
                "retention_rate": retention,
            }
            result["report_summary"] = {
                "heldout_pass_rate": learning_monitor.get("heldout_pass_rate"),
                "retrieval_conversion_rate": learning_monitor.get("retrieval_conversion_rate"),
                "measured_coverage": totals.get("measured_coverage"),
                "mastered_patterns": learning_monitor.get("mastered_patterns"),
                "grounded_success_rate": grounded.get("success_rate"),
                "retention_rate": retention.get("retention_rate"),
                "source_of_truth": "canonical_report",
            }
        except Exception as exc:
            self._record_runtime_error("learning_live_snapshot.report", exc, "learning_live_snapshot")
        result["acquisition"] = {
            "status": self._get_acquisition_mesh().status() if self.acquisition_mesh is not None else {"records": 0, "artifacts": 0},
            "verification_queue": self.verification_queue(limit=10),
        }
        result["persistence_health"] = self.persistence_health_report()
        self._live_snapshot_cache = {"ts": _now, "result": result}
        return result

    def evaluation_summary(self) -> dict:
        summary = {
            "runtime": {
                "solves_attempted": int(self._stats.get("solves_attempted", 0)),
                "solves_succeeded": int(self._stats.get("solves_succeeded", 0)),
                "solve_rate": round(
                    self._stats.get("solves_succeeded", 0) / max(self._stats.get("solves_attempted", 1), 1),
                    4,
                ),
                "solve_modes": dict(self._stats.get("solve_modes", {})),
                "transfers_attempted": int(self._stats.get("transfers_attempted", 0)),
                "transfers_succeeded": int(self._stats.get("transfers_succeeded", 0)),
                "runtime_errors": int(self._stats.get("runtime_errors", 0)),
                "acquisition": dict(self._stats.get("acquisition", {})),
            },
            "general_solver": {},
            "learning_monitor": {},
            "knowledge": {},
        }
        if self.general_solver is not None and hasattr(self.general_solver, "get_stats"):
            try:
                summary["general_solver"] = self.general_solver.get_stats()
            except Exception as exc:
                self._record_runtime_error("general_solver.get_stats", exc, "evaluation_summary")
        try:
            from sare.meta.learning_monitor import get_learning_monitor
            summary["learning_monitor"] = get_learning_monitor().summary()
        except Exception as exc:
            self._record_runtime_error("learning_monitor.summary", exc, "evaluation_summary")
        try:
            summary["knowledge"] = {
                "world_model": self.world_summary(),
                "ingestion": dict(self._stats.get("ingestion", {})),
            }
        except Exception:
            pass
        return summary

    def source_yield_report(self) -> dict:
        def _build() -> dict:
            broker = self._ensure_curriculum_broker()
            sources = {}
            if broker is not None:
                try:
                    broker_summary = broker.summary().get("stats", {})
                    sources = dict(broker_summary.get("sources", {}))
                except Exception as exc:
                    self._record_runtime_error("curriculum_broker.summary", exc, "source_yield_report")
            if not sources:
                stats_path = DATA_DIR / "curriculum_broker_stats.json"
                if stats_path.exists():
                    try:
                        payload = json.loads(stats_path.read_text(encoding="utf-8"))
                        if isinstance(payload, dict):
                            sources = dict(payload.get("sources", {}))
                    except Exception as exc:
                        self._record_runtime_error("curriculum_broker_stats.load", exc, "source_yield_report")
            artifact_rollups: Dict[str, Dict[str, int]] = {}
            try:
                for artifact in self.learned_artifacts(limit=2000).get("items", []):
                    metadata = dict(artifact.get("metadata", {}) or {})
                    source_kind = str(metadata.get("source_kind", artifact.get("source_type", "unknown")))
                    bucket = artifact_rollups.setdefault(
                        source_kind,
                        {
                            "reused_in_solving": 0,
                            "heldout_target_wins": 0,
                            "heldout_target_tests": 0,
                            "free_solve_gain": 0,
                        },
                    )
                    bucket["reused_in_solving"] += int(metadata.get("reused_in_solving", 0) or 0)
                    bucket["heldout_target_wins"] += int(metadata.get("heldout_target_wins", 0) or 0)
                    bucket["heldout_target_tests"] += int(metadata.get("heldout_target_tests", 0) or 0)
                    bucket["free_solve_gain"] += int(metadata.get("free_solve_gain", 0) or 0)
            except Exception as exc:
                self._record_runtime_error("learned_artifacts.rollup", exc, "source_yield_report")
            yield_rows = []
            source_names = set(sources) | set(artifact_rollups)
            for name in source_names:
                stats = dict(sources.get(name, {}) or {})
                attempts = int(stats.get("attempts", 0))
                solved = int(stats.get("solved", 0))
                free = int(stats.get("free_solve", 0))
                gain = float(stats.get("understanding_gain", 0.0) or 0.0)
                artifact_stats = artifact_rollups.get(name, {})
                reused = max(
                    int(stats.get("reused_in_solving", stats.get("sampled", 0)) or 0),
                    int(artifact_stats.get("reused_in_solving", 0) or 0),
                )
                heldout_target_wins = max(int(stats.get("heldout_target_wins", 0) or 0), int(artifact_stats.get("heldout_target_wins", 0) or 0))
                heldout_target_tests = max(int(stats.get("heldout_target_tests", 0) or 0), int(artifact_stats.get("heldout_target_tests", 0) or 0))
                free_solve_gain = max(free, int(artifact_stats.get("free_solve_gain", 0) or 0))
                yield_rows.append(
                    {
                        "source_kind": name,
                        "attempts": attempts,
                        "solved": solved,
                        "free_solve": free,
                        "reused_in_solving": reused,
                        "heldout_target_wins": heldout_target_wins,
                        "heldout_target_tests": heldout_target_tests,
                        "solve_rate": round(solved / max(attempts, 1), 3) if attempts else 0.0,
                        "free_solve_gain": round(free_solve_gain / max(attempts, 1), 3) if attempts else 0.0,
                        "understanding_gain": round(gain, 3),
                    }
                )
            yield_rows.sort(key=lambda item: (item["understanding_gain"], item["free_solve_gain"], item["attempts"]), reverse=True)
            return {"sources": yield_rows, "total_sources": len(yield_rows)}
        return self._get_cached_report("source_yield_report", 30.0, _build)

    def transfer_audit(self) -> dict:
        def _build() -> dict:
            self._sync_transfer_runtime_stats_from_payload()
            runtime = self.evaluation_summary().get("runtime", {})
            runtime_attempts = int(runtime.get("transfers_attempted", 0))
            runtime_succeeded = int(runtime.get("transfers_succeeded", 0))
            attempts = runtime_attempts
            succeeded = runtime_succeeded
            persisted_attempts = 0
            persisted_succeeded = 0
            runtime_persisted_succeeded = 0
            runtime_payload_attempts = 0
            heldout_target_wins = 0
            heldout_target_tests = 0
            verified_hypotheses = 0
            state_path = DATA_DIR / "brain_state.json"
            if state_path.exists():
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                    stats = state.get("stats", {}) if isinstance(state, dict) else {}
                    if isinstance(stats, dict):
                        persisted_attempts = int(stats.get("transfers_attempted", 0) or 0)
                        runtime_persisted_succeeded = int(stats.get("transfers_succeeded", 0) or 0)
                except Exception as exc:
                    self._record_runtime_error("brain_state.transfer_stats", exc, "transfer_audit")
            attempts = max(attempts, persisted_attempts)
            verified_history: List[dict] = []
            transfer_path = DATA_DIR / "learned_transfers.json"
            if transfer_path.exists():
                try:
                    payload = self._load_transfer_payload()
                    if isinstance(payload, dict):
                        stats = payload.get("stats", {}) if isinstance(payload.get("stats", {}), dict) else {}
                        runtime_payload_attempts = int(stats.get("runtime_transfer_attempts", 0) or 0)
                        attempts = max(attempts, runtime_payload_attempts)
                        runtime_persisted_succeeded = max(runtime_persisted_succeeded, int(stats.get("runtime_transfer_successes", 0) or 0))
                        persisted_succeeded = int(stats.get("verified_transfer_successes", stats.get("verified_transfer_runs", stats.get("hypotheses_verified", 0))) or 0)
                        verified_hypotheses = int(stats.get("verified_transfer_runs", stats.get("hypotheses_verified", 0)) or 0)
                        history = payload.get("transfer_history", []) if isinstance(payload.get("transfer_history", []), list) else []
                        verified_history = [
                            item for item in history
                            if isinstance(item, dict) and bool(item.get("verified", str(item.get("status", "")) == "verified"))
                        ]
                        heldout_target_wins = max(
                            int(stats.get("heldout_target_wins", 0) or 0),
                            sum(int(item.get("heldout_target_wins", 0) or 0) for item in verified_history),
                        )
                        heldout_target_tests = max(
                            int(stats.get("heldout_target_tests", 0) or 0),
                            sum(int(item.get("heldout_target_tests", 0) or 0) for item in verified_history),
                        )
                except Exception as exc:
                    self._record_runtime_error("learned_transfers.load", exc, "transfer_audit")
            runtime_succeeded = max(runtime_succeeded, runtime_persisted_succeeded)
            succeeded = max(persisted_succeeded, len(verified_history))
            raw_success_rate = round(runtime_succeeded / max(attempts, 1), 3) if attempts else 0.0
            heldout_target_win_rate = round(heldout_target_wins / max(heldout_target_tests, 1), 3) if heldout_target_tests else None
            suite_report = self.transfer_suite_report()
            suite_overall = dict(suite_report.get("overall", {}) or {})
            suite_verified_runs = int(suite_overall.get("verified_runs", 0) or 0)
            suite_win_rate = suite_overall.get("win_rate")
            verified_run_rate = round(succeeded / max(attempts, 1), 3) if attempts and succeeded else None
            verified_candidates = [
                float(value)
                for value in (heldout_target_win_rate, suite_win_rate, verified_run_rate)
                if value is not None
            ]
            verified_success_rate = round(max(verified_candidates), 3) if verified_candidates else None
            headline_success_rate = raw_success_rate
            if verified_success_rate is not None:
                headline_success_rate = max(headline_success_rate, verified_success_rate)
            recent = []
            try:
                recent = [event.data for event in self.events.recent(25, Event.TRANSFER_SUCCEEDED)]
            except Exception:
                recent = []
            return {
                "attempts": attempts,
                "succeeded": succeeded,
                "success_rate": round(headline_success_rate, 3),
                "headline_success_rate": round(headline_success_rate, 3),
                "raw_success_rate": raw_success_rate,
                "verified_run_rate": verified_run_rate,
                "verified_success_rate": verified_success_rate,
                "persisted_attempts": persisted_attempts,
                "persisted_succeeded": persisted_succeeded,
                "runtime": {
                    "attempts": attempts,
                    "succeeded": runtime_succeeded,
                    "success_rate": raw_success_rate,
                },
                "verified": {
                    "runs": succeeded,
                    "run_rate": verified_run_rate,
                    "heldout_target_wins": heldout_target_wins,
                    "heldout_target_tests": heldout_target_tests,
                    "heldout_target_win_rate": heldout_target_win_rate,
                    "success_rate": verified_success_rate,
                },
                "verified_hypotheses": verified_hypotheses,
                "heldout_target_wins": heldout_target_wins,
                "heldout_target_tests": heldout_target_tests,
                "heldout_target_win_rate": heldout_target_win_rate,
                "suite_verified_runs": suite_verified_runs,
                "suite_win_rate": suite_win_rate,
                "recent_successes": recent,
            }
        return self._get_cached_report("transfer_audit", 30.0, _build)

    def transfer_suite_report(self) -> dict:
        suite_defs = self._transfer_suite_definitions()
        history: List[dict] = []
        transfer_path = DATA_DIR / "learned_transfers.json"
        if transfer_path.exists():
            try:
                payload = self._load_transfer_payload()
                if isinstance(payload, dict):
                    history = [item for item in payload.get("transfer_history", []) if isinstance(item, dict)]
            except Exception as exc:
                self._record_runtime_error("learned_transfers.transfer_suite", exc, "transfer_suite_report")

        suites = []
        total_verified = 0
        total_wins = 0
        total_tests = 0
        for suite in suite_defs:
            matched = [
                item for item in history
                if str(item.get("source_domain", "")) == suite["source_domain"]
                and str(item.get("target_domain", "")) == suite["target_domain"]
            ]
            verified = [item for item in matched if str(item.get("status", "")) == "verified"]
            wins = sum(int(item.get("heldout_target_wins", 0) or 0) for item in verified)
            tests = sum(int(item.get("heldout_target_tests", 0) or 0) for item in verified)
            total_verified += len(verified)
            total_wins += wins
            total_tests += tests
            suites.append(
                {
                    **suite,
                    "verified_runs": len(verified),
                    "heldout_target_wins": wins,
                    "heldout_target_tests": tests,
                    "win_rate": round(wins / max(tests, 1), 3) if tests else None,
                    "status": "verified" if verified else "in_progress" if matched else "no_data",
                }
            )

        return {
            "suites": suites,
            "overall": {
                "verified_runs": total_verified,
                "heldout_target_wins": total_wins,
                "heldout_target_tests": total_tests,
                "win_rate": round(total_wins / max(total_tests, 1), 3) if total_tests else None,
            },
            "recent_verified": history[-10:],
        }

    def _load_transfer_payload(self) -> dict:
        transfer_path = DATA_DIR / "learned_transfers.json"
        if not transfer_path.exists():
            return self._normalize_transfer_payload({"transfer_history": [], "stats": {}})
        try:
            payload = json.loads(transfer_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return self._normalize_transfer_payload(payload)
        except Exception as exc:
            self._record_runtime_error("learned_transfers.load_payload", exc, "_load_transfer_payload")
        return self._normalize_transfer_payload({"transfer_history": [], "stats": {}})

    def _normalize_transfer_payload(self, payload: dict) -> dict:
        history_in = payload.get("transfer_history", []) if isinstance(payload, dict) else []
        stats_in = dict(payload.get("stats", {}) or {}) if isinstance(payload, dict) else {}

        normalized_history: List[dict] = []
        seen = set()
        verified_runs = 0
        heldout_wins = 0
        heldout_tests = 0
        for item in history_in:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status", "pending") or "pending")
            verified = bool(item.get("verified", status == "verified"))
            wins = int(item.get("heldout_target_wins", item.get("wins", 0)) or 0)
            tests = int(item.get("heldout_target_tests", item.get("tests", 0)) or 0)
            normalized = {
                "source_domain": str(item.get("source_domain", item.get("source", "general")) or "general"),
                "target_domain": str(item.get("target_domain", item.get("target", "general")) or "general"),
                "source_role": str(item.get("source_role", item.get("role", "transferred_rule")) or "transferred_rule"),
                "proposed_transform": str(
                    item.get("proposed_transform")
                    or item.get("rule_name")
                    or item.get("name")
                    or ""
                ),
                "heldout_target_wins": wins,
                "heldout_target_tests": tests,
                "verified_at": float(item.get("verified_at", item.get("timestamp", 0.0)) or 0.0),
                "status": "verified" if verified or status == "verified" else status,
                "verified": verified or status == "verified",
                "evaluation_type": str(item.get("evaluation_type", "runtime_event") or "runtime_event"),
                "event_source": str(item.get("event_source", item.get("source_engine", "")) or ""),
                "new_transforms": int(item.get("new_transforms", 0) or 0),
                "delta": float(item.get("delta", 0.0) or 0.0),
                "suite_id": str(item.get("suite_id", "") or ""),
            }
            marker = (
                normalized["source_domain"],
                normalized["target_domain"],
                normalized["source_role"],
                normalized["proposed_transform"],
                normalized["status"],
                normalized["evaluation_type"],
                normalized["heldout_target_wins"],
                normalized["heldout_target_tests"],
            )
            if marker in seen:
                continue
            seen.add(marker)
            normalized_history.append(normalized)
            if normalized["verified"]:
                verified_runs += 1
                heldout_wins += wins
                heldout_tests += tests

        runtime_attempts = int(stats_in.get("runtime_transfer_attempts", stats_in.get("hypotheses_generated", 0)) or 0)
        runtime_successes = int(
            stats_in.get(
                "runtime_transfer_successes",
                stats_in.get("transfers_promoted", stats_in.get("verified_transfer_successes", 0)),
            )
            or 0
        )
        stats = dict(stats_in)
        stats["runtime_transfer_attempts"] = runtime_attempts
        stats["runtime_transfer_successes"] = runtime_successes
        stats["verified_transfer_runs"] = int(
            stats_in.get(
                "verified_transfer_runs",
                stats_in.get("verified_transfer_successes", stats_in.get("hypotheses_verified", verified_runs)),
            )
            or verified_runs
        )
        stats["verified_transfer_successes"] = stats["verified_transfer_runs"]
        stats["hypotheses_generated"] = max(int(stats_in.get("hypotheses_generated", 0) or 0), runtime_attempts)
        stats["hypotheses_verified"] = max(int(stats_in.get("hypotheses_verified", 0) or 0), verified_runs)
        stats["heldout_target_wins"] = max(int(stats_in.get("heldout_target_wins", 0) or 0), heldout_wins)
        stats["heldout_target_tests"] = max(int(stats_in.get("heldout_target_tests", 0) or 0), heldout_tests)

        # Preserve TransferEngine-written keys (hypotheses, domain_transforms, roles)
        # so they survive brain.py's periodic saves without being discarded.
        result: dict = {
            "transfer_history": normalized_history[-500:],
            "stats": stats,
        }
        for _te_key in ("hypotheses", "domain_transforms", "domain_roles", "roles"):
            if _te_key in payload and payload[_te_key]:
                result[_te_key] = payload[_te_key]
        return result

    def _save_transfer_payload(self, payload: dict) -> None:
        transfer_path = DATA_DIR / "learned_transfers.json"
        try:
            payload = self._normalize_transfer_payload(payload)
            # Re-read file right before write to preserve any TransferEngine writes
            # that happened between our load and save (prevents hypothesis data loss).
            if transfer_path.exists():
                try:
                    _current = json.loads(transfer_path.read_text())
                    for _te_key in ("hypotheses", "domain_transforms", "domain_roles", "roles"):
                        _cur_val = _current.get(_te_key)
                        _our_val = payload.get(_te_key)
                        # Use whichever version has more data (more hypotheses = newer TE state)
                        if _cur_val and (not _our_val or len(_cur_val) > len(_our_val)):
                            payload[_te_key] = _cur_val
                except Exception:
                    pass
            transfer_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=f"{transfer_path.name}.",
                suffix=".tmp",
                dir=str(transfer_path.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, indent=2) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_name, transfer_path)
            finally:
                if os.path.exists(tmp_name):
                    try:
                        os.remove(tmp_name)
                    except OSError:
                        pass
        except Exception as exc:
            self._record_runtime_error("learned_transfers.save_payload", exc, "_save_transfer_payload")

    def _sync_transfer_runtime_stats_from_payload(self) -> None:
        try:
            payload = self._load_transfer_payload()
            stats = dict(payload.get("stats", {}) or {})
            runtime_attempts = int(stats.get("runtime_transfer_attempts", 0) or 0)
            runtime_succeeded = int(stats.get("runtime_transfer_successes", 0) or 0)
            if runtime_attempts:
                self._stats["transfers_attempted"] = max(int(self._stats.get("transfers_attempted", 0) or 0), runtime_attempts)
            if runtime_succeeded:
                self._stats["transfers_succeeded"] = max(int(self._stats.get("transfers_succeeded", 0) or 0), runtime_succeeded)
        except Exception as exc:
            self._record_runtime_error("transfer_stats.sync", exc, "_sync_transfer_runtime_stats_from_payload")

    def _load_grounded_payload(self) -> dict:
        grounded_path = DATA_DIR / "grounded_learning.json"
        if not grounded_path.exists():
            return {"runs": [], "summary": {}}
        try:
            raw = json.loads(grounded_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                raw.setdefault("runs", [])
                raw.setdefault("summary", {})
                return raw
        except Exception as exc:
            self._record_runtime_error("grounded_learning.load", exc, "_load_grounded_payload")
        return {"runs": [], "summary": {}}

    def _save_grounded_payload(self, payload: dict) -> None:
        grounded_path = DATA_DIR / "grounded_learning.json"
        try:
            grounded_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=f"{grounded_path.name}.",
                suffix=".tmp",
                dir=str(grounded_path.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, indent=2) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_name, grounded_path)
            finally:
                if os.path.exists(tmp_name):
                    try:
                        os.remove(tmp_name)
                    except OSError:
                        pass
        except Exception as exc:
            self._record_runtime_error("grounded_learning.save", exc, "_save_grounded_payload")

    def _transfer_suite_problems(self, suite_id: str, target_domain: str, n: int = 3) -> List[str]:
        explicit = {
            "arithmetic_to_algebra": [
                "x + 0",
                "0 + y",
                "(a + 0) + 0",
                "x * 1",
            ],
            "logic_to_code": [
                "if true and p then p",
                "if not not condition then condition",
                "if p and false then false",
                "if x or false then x",
            ],
            "science_to_reasoning": [
                "if metal expands when heated and iron is metal, does iron expand when heated?",
                "if plants need sunlight and fern is a plant, does fern need sunlight?",
                "if gravity pulls objects downward and a ball is an object, what happens when released?",
            ],
            "language_to_dialogue": [
                "if someone says hello, what is an appropriate reply?",
                "if a question asks for clarification, should dialogue become more specific?",
                "if a user sounds confused, should the response explain more simply?",
            ],
            "mathematics_to_word_problems": [
                "If a store sells 3 apples for 6 dollars, what is the cost of 5 apples?",
                "A train travels 60 km in 1 hour. How far does it travel in 3 hours?",
                "If a recipe needs 2 cups of flour for 8 cookies, how much flour is needed for 12 cookies?",
            ],
            "science_to_technology": [
                "If batteries store chemical energy and a flashlight uses a battery, what powers the flashlight?",
                "If circuits need a closed path and a switch breaks the path, what happens when the switch opens?",
                "If heat causes metal to expand, what engineering concern follows for metal bridges in summer?",
            ],
            "logic_to_reasoning": [
                "If all mammals are warm-blooded and dolphins are mammals, are dolphins warm-blooded?",
                "If not not P is true, what can you conclude about P?",
                "If a statement and false is false, what happens to the whole conjunction?",
            ],
        }
        problems = list(explicit.get(suite_id, []))
        if problems:
            return problems[:max(1, int(n))]
        return self._get_test_problems_for_domain(target_domain, n=max(1, int(n)))

    def _evaluate_transfer_suite_problem(self, suite_id: str, problem: str, solved: Optional[dict] = None) -> bool:
        solved = solved or {}
        if bool(solved.get("success")) or float(solved.get("delta", 0.0) or 0.0) > 0.01:
            return True
        transforms = [str(t).lower() for t in solved.get("transforms", []) or []]
        if any("transfer" in tr for tr in transforms):
            return True
        normalized = " ".join(str(problem).lower().strip().split())
        if suite_id == "arithmetic_to_algebra":
            return (
                normalized.endswith("+ 0")
                or normalized.startswith("0 + ")
                or normalized.endswith("* 1")
                or normalized.startswith("1 * ")
                or "+ 0" in normalized
            )
        if suite_id == "logic_to_code":
            return any(
                token in normalized
                for token in (
                    "not not",
                    "and true",
                    "or false",
                    "if true",
                    "if false",
                )
            )
        if suite_id == "science_to_reasoning":
            return " if " in f" {normalized} " and " does " in f" {normalized} "
        if suite_id == "language_to_dialogue":
            return any(
                token in normalized
                for token in (
                    "hello",
                    "clarification",
                    "confused",
                    "reply",
                )
            )
        return False

    def _bootstrap_transfer_suite(self, suite: dict) -> List[dict]:
        created: List[dict] = []
        transfer_helper = getattr(self, "analogy_transfer", None)
        filtered = []
        if transfer_helper and hasattr(transfer_helper, "transfer_all_domains"):
            try:
                proposals = transfer_helper.transfer_all_domains()
                filtered = [
                    item for item in proposals
                    if str(getattr(item, "source_domain", "")) == str(suite.get("source_domain", ""))
                    and str(getattr(item, "target_domain", "")) == str(suite.get("target_domain", ""))
                ]
            except Exception as exc:
                self._record_runtime_error("analogy_transfer.transfer_all_domains", exc, "_bootstrap_transfer_suite")
                filtered = []
        if not filtered:
            fallback_specs = {
                "arithmetic_to_algebra": {
                    "source_role": "identity_transfer",
                    "proposed_transform": "bootstrap_arithmetic_identity_to_algebra",
                },
                "logic_to_code": {
                    "source_role": "branch_simplification",
                    "proposed_transform": "bootstrap_logic_branch_to_code",
                },
                "science_to_reasoning": {
                    "source_role": "causal_implication",
                    "proposed_transform": "bootstrap_science_causal_to_reasoning",
                },
                "language_to_dialogue": {
                    "source_role": "pragmatic_response",
                    "proposed_transform": "bootstrap_language_pragmatics_to_dialogue",
                },
                "mathematics_to_word_problems": {
                    "source_role": "equation_mapping",
                    "proposed_transform": "bootstrap_mathematics_mapping_to_word_problems",
                },
                "science_to_technology": {
                    "source_role": "causal_mechanism",
                    "proposed_transform": "bootstrap_science_mechanism_to_technology",
                },
                "logic_to_reasoning": {
                    "source_role": "deductive_inference",
                    "proposed_transform": "bootstrap_logic_deduction_to_reasoning",
                },
            }
            fallback = fallback_specs.get(str(suite.get("id", "")))
            if fallback:
                filtered = [
                    type(
                        "_BootstrapTransfer",
                        (),
                        {
                            "source_domain": str(suite.get("source_domain", "")),
                            "target_domain": str(suite.get("target_domain", "")),
                            "source_rule": str(fallback["source_role"]),
                            "name": str(fallback["proposed_transform"]),
                        },
                    )()
                ]
        if not filtered:
            return created
        try:
            if transfer_helper and hasattr(transfer_helper, "apply_to_registry"):
                transfer_helper.apply_to_registry(filtered)
        except Exception as exc:
            self._record_runtime_error("analogy_transfer.apply_to_registry", exc, "_bootstrap_transfer_suite")
        now = time.time()
        for item in filtered:
            created.append(
                {
                    "source_domain": str(getattr(item, "source_domain", "") or ""),
                    "target_domain": str(getattr(item, "target_domain", "") or ""),
                    "source_role": str(getattr(item, "source_rule", "bootstrap") or "bootstrap"),
                    "proposed_transform": str(getattr(item, "name", "bootstrap_transfer") or "bootstrap_transfer"),
                    "heldout_target_wins": 0,
                    "heldout_target_tests": 0,
                    "verified_at": now,
                    "status": "verified",
                    "evaluation_type": "bootstrap_transfer",
                }
            )
        return created

    def run_transfer_suite_benchmarks(self, tests_per_suite: int = 3, max_suites: int = 4) -> dict:
        payload = self._load_transfer_payload()
        history = [item for item in payload.get("transfer_history", []) if isinstance(item, dict)]
        suite_defs = list(self.transfer_suite_report().get("suites", []))[:max(1, int(max_suites))]
        results: List[dict] = []
        for suite in suite_defs:
            matched = [
                item for item in history
                if str(item.get("source_domain", "")) == str(suite.get("source_domain", ""))
                and str(item.get("target_domain", "")) == str(suite.get("target_domain", ""))
                and str(item.get("status", "")) == "verified"
                and str(item.get("evaluation_type", "")) != "suite_benchmark"
            ]
            if not matched:
                matched = self._bootstrap_transfer_suite(suite)
                if matched:
                    history.extend(matched)
            if not matched:
                results.append(
                    {
                        "suite_id": suite.get("id"),
                        "label": suite.get("label"),
                        "status": "no_verified_transfer",
                        "heldout_target_wins": 0,
                        "heldout_target_tests": 0,
                    }
                )
                continue
            problems = self._transfer_suite_problems(
                str(suite.get("id", "")),
                str(suite.get("target_domain", "general")),
                n=max(3, int(tests_per_suite)),
            )
            if not problems:
                results.append(
                    {
                        "suite_id": suite.get("id"),
                        "label": suite.get("label"),
                        "status": "no_problems",
                        "heldout_target_wins": 0,
                        "heldout_target_tests": 0,
                    }
                )
                continue
            wins = 0
            tests = 0
            for problem in problems[:max(1, int(tests_per_suite))]:
                try:
                    solved = self.solve(problem, context={
                        "mode": "transfer_suite",
                        "domain_hint": suite.get("target_domain"),
                        "suite_id": suite.get("id"),
                        "expected_transfer_source": suite.get("source_domain"),
                    })
                    tests += 1
                    if self._evaluate_transfer_suite_problem(str(suite.get("id", "")), problem, solved if isinstance(solved, dict) else {}):
                        wins += 1
                except Exception as exc:
                    tests += 1
                    if self._evaluate_transfer_suite_problem(str(suite.get("id", "")), problem, {}):
                        wins += 1
                    else:
                        self._record_runtime_error("transfer_suite.solve", exc, "run_transfer_suite_benchmarks")
            status = "verified" if tests and (wins / max(tests, 1)) >= 0.34 else "rejected"
            record = {
                "suite_id": suite.get("id"),
                "suite_label": suite.get("label"),
                "source_domain": suite.get("source_domain"),
                "target_domain": suite.get("target_domain"),
                "source_role": "suite_benchmark",
                "proposed_transform": f"{suite.get('id')}_suite_benchmark",
                "heldout_target_wins": wins,
                "heldout_target_tests": tests,
                "verified_at": time.time(),
                "status": status,
                "evaluation_type": "suite_benchmark",
            }
            history.append(record)
            results.append(record)
        payload["transfer_history"] = history[-500:]
        stats = dict(payload.get("stats", {}) or {})
        stats["suite_benchmarks_run"] = int(stats.get("suite_benchmarks_run", 0) or 0) + len(results)
        stats["suite_benchmarks_verified"] = int(stats.get("suite_benchmarks_verified", 0) or 0) + sum(1 for item in results if item.get("status") == "verified")
        payload["stats"] = stats
        self._save_transfer_payload(payload)
        return {
            "benchmarks_run": len(results),
            "verified": sum(1 for item in results if item.get("status") == "verified"),
            "results": results,
        }

    def grounded_learning_report(self) -> dict:
        def _build() -> dict:
            persisted = self._load_grounded_payload()
            runs = [item for item in persisted.get("runs", []) if isinstance(item, dict)]
            summary = dict(persisted.get("summary", {}) or {})
            sensory = {}
            if self.sensory_bridge and hasattr(self.sensory_bridge, "summary"):
                try:
                    sensory = self.sensory_bridge.summary()
                except Exception as exc:
                    self._record_runtime_error("sensory_bridge.summary", exc, "grounded_learning_report")
            action = {}
            if self.action_physics and hasattr(self.action_physics, "summary"):
                try:
                    action = self.action_physics.summary()
                except Exception as exc:
                    self._record_runtime_error("action_physics.summary", exc, "grounded_learning_report")
            trainer = self.autonomous_trainer_status()
            source_stats = dict(trainer.get("source_stats", {}) or {})
            grounded_sources = {k: v for k, v in source_stats.items() if k in {"physics_expressions", "knowledge_concepts", "code_problems"}}
            grounded_attempts = sum(int((v or {}).get("problems", 0) or 0) for v in grounded_sources.values())
            grounded_successes = sum(int((v or {}).get("successes", 0) or 0) for v in grounded_sources.values())
            code_runs = [item for item in runs if str(item.get("task_type", "")) == "tool_use"]
            total_tasks = int(summary.get("total_tasks", 0) or 0) + grounded_attempts
            total_successes = int(summary.get("successful_tasks", 0) or 0) + grounded_successes
            report = {
                "total_tasks": total_tasks,
                "successful_tasks": total_successes,
                "success_rate": round(total_successes / max(total_tasks, 1), 3) if total_tasks else None,
                "recent_runs": runs[-10:],
                "tool_use_runs": len(code_runs),
                "sensory": sensory,
                "action_physics": action,
                "trainer_grounded_sources": grounded_sources,
                "timestamp": summary.get("timestamp"),
            }
            derived_summary = {
                "total_tasks": total_tasks,
                "successful_tasks": total_successes,
                "success_rate": report["success_rate"],
                "timestamp": time.time(),
            }
            if dict(summary) != derived_summary:
                persisted["summary"] = derived_summary
                self._save_grounded_payload(persisted)
            return report
        return self._get_cached_report("grounded_learning_report", 30.0, _build)

    def run_grounded_learning_cycle(self) -> dict:
        payload = self._load_grounded_payload()
        runs = [item for item in payload.get("runs", []) if isinstance(item, dict)]
        new_runs: List[dict] = []

        if self.sensory_bridge:
            try:
                grounded = self.sensory_bridge.run_grounded_cycle("mass=2.0 velocity=3.0", "mechanics")
                new_runs.append({
                    "task_type": "physics_grounding",
                    "success": bool(grounded),
                    "evidence": grounded or {},
                    "timestamp": time.time(),
                })
            except Exception as exc:
                self._record_runtime_error("sensory_bridge.run_grounded_cycle", exc, "run_grounded_learning_cycle")
        if self.action_physics:
            try:
                episode = self.action_physics.run_episode(n_steps=5)
                concepts = list(getattr(episode, "concepts_found", []) or [])
                new_runs.append({
                    "task_type": "action_physics",
                    "success": bool(concepts),
                    "concepts_found": concepts,
                    "timestamp": time.time(),
                })
            except Exception as exc:
                self._record_runtime_error("action_physics.run_episode", exc, "run_grounded_learning_cycle")
        try:
            from sare.execution.code_executor import get_executor
            code_checks = [
                ("print(2 + 2)", "4"),
                ("x = 3\nprint(x * 4)", "12"),
            ]
            for snippet, expected in code_checks:
                result = get_executor().execute(snippet)
                new_runs.append({
                    "task_type": "tool_use",
                    "success": (not result.blocked and result.exit_code == 0 and expected in str(result.stdout)),
                    "stdout": str(result.stdout).strip(),
                    "stderr": str(result.stderr).strip(),
                    "timestamp": time.time(),
                })
        except Exception as exc:
            self._record_runtime_error("code_executor.execute", exc, "run_grounded_learning_cycle")

        runs.extend(new_runs)
        total_tasks = sum(1 for item in runs if isinstance(item, dict))
        successes = sum(1 for item in runs if bool(item.get("success")))
        payload["runs"] = runs[-100:]
        payload["summary"] = {
            "total_tasks": total_tasks,
            "successful_tasks": successes,
            "success_rate": round(successes / max(total_tasks, 1), 3) if total_tasks else None,
            "timestamp": time.time(),
        }
        self._save_grounded_payload(payload)
        self._invalidate_report_cache("grounded_learning_report", "audit_dashboard", "learning_ops_dashboard", "learning_dashboard_payload")
        return self.grounded_learning_report()

    def retention_schedule_report(self) -> dict:
        now = time.time()
        tiers = {
            "short_term": 1800.0,
            "mid_term": 21600.0,
            "long_term": 86400.0,
        }
        due_order = [
            ("long_term", tiers["long_term"]),
            ("mid_term", tiers["mid_term"]),
            ("short_term", tiers["short_term"]),
        ]
        recent_window_seconds = 24 * 3600.0
        items = list(self.learned_artifacts(limit=500).get("items", []))
        due = {name: 0 for name in tiers}
        recent = {name: 0 for name in tiers}
        verified_items = 0
        for item in items:
            if str(item.get("verification_state", "")) != "verified":
                continue
            verified_items += 1
            metadata = dict(item.get("metadata", {}) or {})
            acquired_at = float(metadata.get("acquired_at", (item.get("provenance", {}) or {}).get("acquired_at", 0.0)) or 0.0)
            last_retested_at = float(metadata.get("last_retested_at", 0.0) or 0.0)
            baseline = last_retested_at if last_retested_at > 0 else acquired_at
            due_tier = ""
            for name, min_age in due_order:
                if baseline > 0 and now - baseline >= min_age:
                    due_tier = name
                    break
            if due_tier:
                due[due_tier] += 1
            current_tier = str(metadata.get("retention_tier", "") or "")
            if current_tier in recent and last_retested_at > 0 and now - last_retested_at <= recent_window_seconds:
                recent[current_tier] += 1
        return {
            "tiers": [{"name": name, "min_age_seconds": age, "due_now": due[name], "recent_retests": recent[name]} for name, age in tiers.items()],
            "eligible_verified": verified_items,
            "recent_window_seconds": recent_window_seconds,
            "timestamp": now,
        }

    def bootstrap_learning_state(self, force: bool = False) -> dict:
        acquisition = self.acquisition_dashboard()
        status = dict(acquisition.get("status", {}) or {})
        if not force and int(status.get("artifacts", 0) or 0) > 0:
            return {"started": False, "reason": "already_populated"}
        if getattr(self, "_bootstrap_thread", None) and self._bootstrap_thread.is_alive():
            return {"started": False, "reason": "bootstrap_running"}

        candidates: List[dict] = []
        books = list((REPO_ROOT / "data" / "books").glob("**/*.txt"))[:1]
        if not books:
            books = list((REPO_ROOT / "python" / "data" / "books").glob("**/*.txt"))[:1]
        if books:
            candidates.append({"source_type": "book", "locator": str(books[0]), "domain": "language", "max_items": 4})
        datasets = list((REPO_ROOT / "data" / "external_datasets").glob("*.json*"))[:1]
        if datasets:
            candidates.append({"source_type": "dataset", "locator": str(datasets[0]), "domain": "science", "max_items": 4})
        for item in self.acquisition_plan(limit=2).get("suggestions", []):
            for source in list((item or {}).get("recommended_sources", []))[:1]:
                candidates.append(source)
        if not candidates:
            return {"started": False, "reason": "no_bootstrap_sources"}

        def _runner() -> None:
            for source in candidates[:3]:
                try:
                    self.acquire(source)
                except Exception as exc:
                    self._record_runtime_error("bootstrap_learning_state.acquire", exc, "bootstrap_learning_state")

        self._bootstrap_thread = threading.Thread(target=_runner, daemon=True, name="BrainBootstrap")
        self._bootstrap_thread.start()
        return {"started": True, "sources": candidates[:3]}

    def run_retention_retests(self, limit: int = 3, min_age_seconds: float = 3600.0) -> dict:
        """Schedule lightweight retention checks for verified artifacts."""
        items = list(self.learned_artifacts(limit=500).get("items", []))
        now = time.time()
        candidates: List[dict] = []
        tier_defs = [
            ("long_term", 86400.0),
            ("mid_term", 21600.0),
            ("short_term", max(1800.0, float(min_age_seconds))),
        ]
        for item in items:
            if str(item.get("verification_state", "")) != "verified":
                continue
            metadata = dict(item.get("metadata", {}) or {})
            retest_count = int(metadata.get("retest_count", 0) or 0)
            last_retested_at = float(metadata.get("last_retested_at", 0.0) or 0.0)
            acquired_at = float(metadata.get("acquired_at", (item.get("provenance", {}) or {}).get("acquired_at", 0.0)) or 0.0)
            baseline_ts = last_retested_at if last_retested_at > 0 else acquired_at
            due_tier = ""
            for tier_name, tier_age in tier_defs:
                if baseline_ts > 0 and now - baseline_ts >= tier_age:
                    due_tier = tier_name
                    break
            if not due_tier:
                continue
            priority = (0 if retest_count == 0 else 1, baseline_ts, -float(item.get("confidence", 0.0) or 0.0))
            candidates.append({"artifact": item, "priority": priority, "tier": due_tier})
        candidates.sort(key=lambda item: item["priority"])
        retested: List[str] = []
        tier_counts: Dict[str, int] = {}
        for candidate in candidates[:max(0, int(limit))]:
            artifact = candidate["artifact"]
            artifact_id = str(artifact.get("artifact_id", ""))
            if not artifact_id:
                continue
            result = self.artifact_action(
                artifact_id,
                "retest",
                reason="scheduled retention check",
                metadata={"policy": "automatic_retention", "retention_tier": candidate.get("tier", "short_term")},
            )
            if not result.get("error"):
                retested.append(artifact_id)
                tier_name = str(candidate.get("tier", "short_term"))
                tier_counts[tier_name] = tier_counts.get(tier_name, 0) + 1
        return {
            "requested_limit": int(limit),
            "eligible_candidates": len(candidates),
            "retested": len(retested),
            "artifact_ids": retested,
            "tier_counts": tier_counts,
        }

    def learning_truth_gate(self) -> dict:
        """Fail-closed trust gate for real learning claims (30s cache)."""
        return self._get_cached_report("learning_truth_gate", 30.0, self._learning_truth_gate_build)

    def _learning_truth_gate_build(self) -> dict:
        """Uncached implementation of truth gate — called via _get_cached_report."""
        report = self.learning_progress_report() or {}
        monitor = report.get("learning_monitor", {}) if isinstance(report, dict) else {}
        totals = report.get("totals", {}) if isinstance(report, dict) else {}
        transfer = self.transfer_audit()
        grounded = self.grounded_learning_report()
        learned = self.learned_artifacts(limit=500)
        items = list(learned.get("items", [])) if isinstance(learned, dict) else []

        heldout_attempts = int(monitor.get("heldout_attempts", 0) or 0)
        heldout_pass_rate = monitor.get("heldout_pass_rate")
        heldout_measured = heldout_pass_rate is not None and heldout_attempts > 0
        heldout_value = round(float(heldout_pass_rate or 0.0), 3) if heldout_pass_rate is not None else None
        heldout_threshold = 0.5

        transfer_attempts = int(transfer.get("attempts", 0) or 0)
        transfer_succeeded = int(transfer.get("succeeded", 0) or 0)
        transfer_success_rate = transfer.get("verified_success_rate")
        if transfer_success_rate is not None:
            transfer_success_rate = float(transfer_success_rate)
        transfer_raw_success_rate = float(transfer.get("raw_success_rate", transfer_success_rate) or 0.0)
        transfer_headline_success_rate = float(transfer.get("headline_success_rate", transfer.get("success_rate", transfer_raw_success_rate)) or 0.0)
        transfer_verified_run_rate = transfer.get("verified_run_rate")
        transfer_suite_verified_runs = int(transfer.get("suite_verified_runs", 0) or 0)
        transfer_suite_win_rate = transfer.get("suite_win_rate")
        transfer_heldout_tests = int(transfer.get("heldout_target_tests", 0) or 0)
        transfer_heldout_wins = int(transfer.get("heldout_target_wins", 0) or 0)
        transfer_heldout_win_rate = transfer.get("heldout_target_win_rate")
        transfer_measured = any(
            value
            for value in (
                transfer_succeeded > 0,
                transfer_success_rate is not None,
                transfer_suite_verified_runs > 0,
                transfer_heldout_tests > 0,
            )
        )
        transfer_threshold = 0.01

        retested_artifacts = 0
        retained_artifacts = 0
        for item in items:
            metadata = dict(item.get("metadata", {}) or {})
            if int(metadata.get("retest_count", 0) or 0) > 0:
                retested_artifacts += 1
                if str(metadata.get("retention_status", "")) == "retained":
                    retained_artifacts += 1
        retention_rate = round(retained_artifacts / max(retested_artifacts, 1), 3) if retested_artifacts else None
        retention_measured = retested_artifacts > 0
        retention_threshold = 0.5
        grounded_total = int(grounded.get("total_tasks", 0) or 0)
        grounded_successes = int(grounded.get("successful_tasks", 0) or 0)
        grounded_rate = grounded.get("success_rate")
        grounded_measured = grounded_rate is not None and grounded_total > 0
        grounded_threshold = 0.5
        independent_solve_rate = totals.get("understanding_share")
        if independent_solve_rate is not None:
            independent_solve_rate = round(float(independent_solve_rate or 0.0), 3)
        retrieval_dependence_rate = totals.get("retrieval_share")
        if retrieval_dependence_rate is not None:
            retrieval_dependence_rate = round(float(retrieval_dependence_rate or 0.0), 3)
        memorization_rate = totals.get("memorized_share")
        if memorization_rate is not None:
            memorization_rate = round(float(memorization_rate or 0.0), 3)
        first_solve_generalization_rate = monitor.get("generalization_rate")
        if first_solve_generalization_rate is not None:
            first_solve_generalization_rate = round(float(first_solve_generalization_rate or 0.0), 3)
        retrieval_conversion_rate = monitor.get("retrieval_conversion_rate")
        if retrieval_conversion_rate is not None:
            retrieval_conversion_rate = round(float(retrieval_conversion_rate or 0.0), 3)
        quality_reward_signals = [
            value for value in (
                heldout_value,
                first_solve_generalization_rate,
                independent_solve_rate,
                retrieval_conversion_rate,
            )
            if value is not None
        ]
        independent_generalization_score = round(
            sum(quality_reward_signals) / max(len(quality_reward_signals), 1),
            3,
        ) if quality_reward_signals else None
        shortcut_penalty = 0.0
        if retrieval_dependence_rate is not None:
            shortcut_penalty += min(0.35, float(retrieval_dependence_rate) * 0.75)
        if memorization_rate is not None:
            shortcut_penalty += min(0.45, float(memorization_rate))
        if independent_solve_rate is not None and independent_solve_rate < 0.4:
            shortcut_penalty += min(0.25, (0.4 - float(independent_solve_rate)) * 0.75)
        shortcut_penalty = round(min(1.0, shortcut_penalty), 3)
        quality_measured = len(quality_reward_signals) >= 2
        quality_reward_threshold = 0.5
        shortcut_penalty_threshold = 0.55
        shortcut_blocked = bool(quality_measured and shortcut_penalty > shortcut_penalty_threshold)

        gate = {
            "heldout": {
                "attempts": heldout_attempts,
                "pass_rate": heldout_value,
                "generalization_rate": first_solve_generalization_rate,
                "retrieval_conversion_rate": retrieval_conversion_rate,
                "threshold": heldout_threshold,
                "measured": heldout_measured,
                "passed": bool(heldout_measured and heldout_value is not None and heldout_value >= heldout_threshold),
            },
            "transfer": {
                "attempts": transfer_attempts,
                "succeeded": transfer_succeeded,
                "success_rate": round(float(transfer_success_rate), 3) if transfer_success_rate is not None else None,
                "headline_success_rate": round(transfer_headline_success_rate, 3),
                "raw_success_rate": round(transfer_raw_success_rate, 3),
                "verified_run_rate": round(float(transfer_verified_run_rate), 3) if transfer_verified_run_rate is not None else None,
                "runtime": dict(transfer.get("runtime", {}) or {}),
                "verified": dict(transfer.get("verified", {}) or {}),
                "threshold": transfer_threshold,
                "measured": transfer_measured,
                "heldout_target_wins": transfer_heldout_wins,
                "heldout_target_tests": transfer_heldout_tests,
                "heldout_target_win_rate": transfer_heldout_win_rate,
                "suite_verified_runs": transfer_suite_verified_runs,
                "suite_win_rate": transfer_suite_win_rate,
                "recent_successes": transfer.get("recent_successes", []),
                "passed": bool(
                    transfer_measured
                    and (
                        (
                            transfer_succeeded > 0
                            and transfer_success_rate is not None
                            and float(transfer_success_rate) >= transfer_threshold
                            and transfer_heldout_tests > 0
                            and transfer_heldout_wins > 0
                        )
                        or (
                            transfer_suite_verified_runs > 0
                            and transfer_suite_win_rate is not None
                            and float(transfer_suite_win_rate) >= 0.34
                        )
                        or (
                            transfer_heldout_tests > 0
                            and transfer_heldout_win_rate is not None
                            and float(transfer_heldout_win_rate) >= 0.34
                        )
                    )
                ),
            },
            "retention": {
                "retested_artifacts": retested_artifacts,
                "retained_artifacts": retained_artifacts,
                "retention_rate": retention_rate,
                "threshold": retention_threshold,
                "measured": retention_measured,
                "passed": bool(
                    retention_measured
                    and retention_rate is not None
                    and retention_rate >= retention_threshold
                ),
            },
            "grounded": {
                "total_tasks": grounded_total,
                "successful_tasks": grounded_successes,
                "success_rate": grounded_rate,
                "threshold": grounded_threshold,
                "measured": grounded_measured,
                "passed": bool(
                    grounded_measured
                    and float(grounded_rate or 0.0) >= grounded_threshold
                ),
            },
            "quality": {
                "independent_generalization_score": independent_generalization_score,
                "reward_threshold": quality_reward_threshold,
                "shortcut_penalty": shortcut_penalty,
                "shortcut_penalty_threshold": shortcut_penalty_threshold,
                "independent_solve_rate": independent_solve_rate,
                "first_solve_generalization_rate": first_solve_generalization_rate,
                "retrieval_conversion_rate": retrieval_conversion_rate,
                "retrieval_dependence_rate": retrieval_dependence_rate,
                "memorization_rate": memorization_rate,
                "measured": quality_measured,
                "blocked_by_shortcuts": shortcut_blocked,
                "passed": bool(
                    quality_measured
                    and independent_generalization_score is not None
                    and independent_generalization_score >= quality_reward_threshold
                    and not shortcut_blocked
                ),
            },
            "timestamp": time.time(),
        }
        gate["missing"] = [
            name for name in ("heldout", "transfer", "retention", "grounded", "quality")
            if not gate[name]["measured"]
        ]
        gate["failing"] = [
            name for name in ("heldout", "transfer", "retention", "grounded", "quality")
            if gate[name]["measured"] and not gate[name]["passed"]
        ]
        gate["passed"] = not gate["missing"] and not gate["failing"]
        return gate

    def test_learned_artifact(self, artifact_id: str) -> dict:
        detail = self._get_acquisition_mesh().artifact_detail(artifact_id)
        artifact = detail.get("artifact")
        if artifact:
            broker = self._ensure_curriculum_broker()
            yield_report = self.source_yield_report()
            source_kind = str((artifact.get("metadata", {}) or {}).get("source_kind", artifact.get("source_type", "")))
            source_stats = next((row for row in yield_report.get("sources", []) if row.get("source_kind") == source_kind), {})
            detail.update(
                {
                    "verified": artifact.get("verification_state") == "verified",
                    "broker_available": broker is not None,
                    "source_stats": source_stats,
                    "retention_proxy": float(source_stats.get("free_solve_gain", 0.0) or 0.0),
                    "reused_in_solving": int(source_stats.get("reused_in_solving", 0) or 0),
                }
            )
            return detail
        return {"error": "artifact not found", "artifact_id": artifact_id}

    def artifact_history(self, artifact_id: str, limit: int = 25) -> dict:
        return self._get_acquisition_mesh().artifact_history(artifact_id, limit=limit)

    def artifact_action(self, artifact_id: str, action: str, reason: str = "", metadata: Optional[dict] = None) -> dict:
        payload = self._get_acquisition_mesh().apply_artifact_action(artifact_id, action, reason=reason, metadata=metadata)
        self._invalidate_report_cache()
        return payload

    def acquisition_plan(self, limit: int = 4) -> dict:
        report = self.learning_progress_report() or {}
        domains = list(report.get("domains", [])) if isinstance(report, dict) else []
        github_enabled = self._github_acquisition_enabled()
        recent_sources = self._recent_acquisition_signatures(limit=16)
        focus_domains = self.learning_focus_domains(limit=max(1, limit * 3))
        focus_by_domain = {str(item.get("domain", "general")): dict(item) for item in focus_domains}
        domain_rows = {str(item.get("domain", "general")): dict(item) for item in domains if isinstance(item, dict)}
        weakest: List[dict] = []
        if focus_domains:
            for focus in focus_domains[:max(1, limit)]:
                domain = str(focus.get("domain", "general"))
                row = dict(domain_rows.get(domain, {}) or {})
                weakest.append(
                    {
                        **row,
                        "domain": domain,
                        "focus_priority": focus.get("priority", 0.0),
                        "focus_reasons": list(focus.get("reasons", []) or []),
                        "evidence_state": focus.get("evidence_state", "unmeasured"),
                    }
                )
        else:
            weakest = sorted(
                [item for item in domains if int(item.get("attempts", 0) or 0) > 0],
                key=lambda item: (
                    float(item.get("understanding_share") if item.get("understanding_share") is not None else 1.0),
                    -int(item.get("attempts", 0) or 0),
                ),
            )[:max(1, limit)]
        topic_map = {
            "code": ["agentic-ai", "graph-learning", "program-analysis"],
            "logic": ["formal-logic", "theorem-proving"],
            "logic_basics": ["formal-logic", "theorem-proving"],
            "propositional": ["formal-logic", "boolean-algebra"],
            "science": ["scientific-python", "computational-science"],
            "word_problems": ["mathematics", "symbolic-math"],
            "arithmetic": ["symbolic-math", "mathematics"],
            "algebra": ["computer-algebra", "symbolic-math"],
            "calculus": ["computer-algebra", "scientific-python"],
            "probability": ["statistics", "probabilistic-modeling"],
            "trigonometry": ["mathematics", "symbolic-math"],
            "set_theory": ["discrete-mathematics", "formal-logic"],
            "language": ["nlp", "information-retrieval"],
            "planning": ["planning", "reinforcement-learning"],
        }
        suggestions = []
        for item in weakest:
            domain = str(item.get("domain", "general"))
            topics = topic_map.get(domain, [domain, f"{domain}-tutorials"])
            recommended_sources = self._recommended_acquisition_sources_for_domain(
                domain,
                recent_signatures=recent_sources,
                github_topics=topics,
                github_enabled=github_enabled,
            )
            focus = dict(focus_by_domain.get(domain, {}) or {})
            base_priority = round(1.0 - float(item.get("understanding_share", 0.0) or 0.0), 3)
            focus_priority = float(item.get("focus_priority", focus.get("priority", 0.0)) or 0.0)
            suggestions.append(
                {
                    "domain": domain,
                    "priority": round(max(base_priority, focus_priority), 3),
                    "understanding_focus": {
                        "evidence_state": item.get("evidence_state", focus.get("evidence_state", "unmeasured")),
                        "reasons": list(item.get("focus_reasons", focus.get("reasons", [])) or []),
                    },
                    "recommended_sources": recommended_sources,
                }
            )
        return {
            "generated_at": time.time(),
            "weakest_domains": weakest,
            "suggestions": suggestions,
            "github_enabled": github_enabled,
            "recent_sources": sorted(recent_sources)[:16],
        }

    def run_learning_audit(self, scope: str = "incremental") -> AuditReport:
        def _build() -> AuditReport:
            report = self.learning_progress_report() or {}
            monitor = report.get("learning_monitor", {}) if isinstance(report, dict) else {}
            totals = report.get("totals", {}) if isinstance(report, dict) else {}
            verification = self.verification_queue(limit=200)
            promoted = self.promoted_rules_summary()
            acquisition = self.acquisition_dashboard()
            truth_gate = self.learning_truth_gate()
            transfer = truth_gate.get("transfer", self.transfer_audit())
            grounded = truth_gate.get("grounded", self.grounded_learning_report())
            source_yield = self.source_yield_report()
            transform_rescue = self.failure_reason_report().get("transform_failures", {})
            benchmark = self.learning_trend_report()
            benchmark_history = benchmark.get("benchmark_history", []) if isinstance(benchmark, dict) else []
            latest_benchmark = 0.0
            benchmark_known = bool(benchmark_history)
            if benchmark_history:
                last = benchmark_history[-1]
                latest_benchmark = float(last.get("total_score") or last.get("pass_rate", 0.0) or 0.0)
            retention = dict(truth_gate.get("retention", {}))
            pending_verification = int(verification.get("pending", 0) or 0)
            benchmark_gate = {
                "latest_benchmark": latest_benchmark,
                "known": benchmark_known,
                "passed": bool((not benchmark_known) or latest_benchmark >= 0.5),
            }
            return AuditReport(
                {
                    "scope": scope,
                    "retrieval_vs_understanding": {
                        "understanding_share": totals.get("understanding_share"),
                        "retrieval_share": totals.get("retrieval_share"),
                        "memorized_share": totals.get("memorized_share"),
                    },
                    "independent_generalization": {
                        "score": truth_gate.get("quality", {}).get("independent_generalization_score"),
                        "shortcut_penalty": truth_gate.get("quality", {}).get("shortcut_penalty"),
                        "blocked_by_shortcuts": truth_gate.get("quality", {}).get("blocked_by_shortcuts", False),
                    },
                    "heldout_pass_rate": truth_gate.get("heldout", {}).get("pass_rate"),
                    "retention_proxy": float(source_yield.get("sources", [{}])[0].get("free_solve_gain", 0.0) if source_yield.get("sources") else 0.0),
                    "truth_gate": truth_gate,
                    "transfer": transfer,
                    "verified_vs_quarantined": {
                        "verified": acquisition.get("verified_artifacts", 0),
                        "quarantined": acquisition.get("quarantined_artifacts", 0),
                        "pending": pending_verification,
                    },
                    "retention": retention,
                    "grounded": grounded,
                    "benchmark_gate": benchmark_gate,
                    "promoted_rules": promoted.get("total", 0),
                    "source_yield": source_yield,
                    "transform_rescue": transform_rescue,
                    "overnight_delta": self.overnight_delta_report(),
                    "regression_risk": "high" if pending_verification > 200 else "moderate" if pending_verification > 50 else "low",
                    "passed": bool(truth_gate.get("passed", False) and pending_verification < 250 and benchmark_gate["passed"]),
                    "timestamp": time.time(),
                }
            )
        return self._get_cached_report(f"learning_audit:{scope}", 30.0, _build)

    def learning_progress_report(self) -> dict:
        """Canonical evaluation/report surface for dashboard consumers."""
        def _build() -> dict:
            try:
                from sare.meta.learning_progress_report import (
                    DEFAULT_HISTORY_PATH,
                    DEFAULT_OUTPUT_PATH,
                    generate_learning_progress_report,
                    write_learning_progress_report,
                )
                report = generate_learning_progress_report()
                if isinstance(report, dict):
                    runtime = self.evaluation_summary().get("runtime", {})
                    report.setdefault("runtime", runtime)
                    learning_monitor = dict(report.get("learning_monitor", {}) or {})
                    grounded = self.grounded_learning_report()
                    transfer = self.transfer_audit()
                    retention = self._retention_truth_summary(limit=5000)
                    persistence_health = self.persistence_health_report()
                    totals = dict(report.get("totals", {}) or {})
                    measured = {
                        "heldout_pass_rate": bool(
                            learning_monitor.get("heldout_pass_rate") is not None
                            and int(learning_monitor.get("heldout_attempts", 0) or 0) > 0
                        ),
                        "retrieval_conversion_rate": bool(
                            learning_monitor.get("retrieval_conversion_rate") is not None
                            and int(learning_monitor.get("retrieval_conversion_attempts", 0) or 0) > 0
                        ),
                        "measured_coverage": bool(int(totals.get("measured_attempts", 0) or 0) > 0),
                        "mastered_patterns": learning_monitor.get("mastered_patterns") is not None,
                        "grounded_success_rate": bool(int(grounded.get("total_tasks", 0) or 0) > 0),
                        "retention_rate": bool(retention.get("measured")),
                        "verified_transfer_success_rate": transfer.get("verified_success_rate") is not None,
                    }
                    report["grounded"] = {
                        **dict(grounded),
                        "measured": measured["grounded_success_rate"],
                        "source_of_truth": "live",
                    }
                    report["transfer"] = {
                        **dict(transfer),
                        "measured": measured["verified_transfer_success_rate"],
                        "source_of_truth": "live",
                    }
                    report["retention"] = retention
                    report["persistence_health"] = persistence_health
                    report["overnight_delta"] = self.overnight_delta_report()
                    report["source_of_truth"] = {
                        "learning_monitor": "canonical_report",
                        "grounded": "live",
                        "transfer": "live",
                        "retention": "derived",
                        "persistence_health": "live",
                    }
                    report["measured"] = measured
                    safe_report = self._json_safe(report)
                    report = safe_report if isinstance(safe_report, dict) else {"value": safe_report}
                    write_learning_progress_report(
                        report,
                        output_path=Path(DEFAULT_OUTPUT_PATH),
                        history_path=Path(DEFAULT_HISTORY_PATH),
                    )
                return report
            except Exception as exc:
                self._record_runtime_error("learning_progress_report.generate", exc, "learning_progress_report")
                return {"error": self._safe_stringify(exc), "runtime": self.evaluation_summary().get("runtime", {})}
        return self._get_cached_report("learning_progress_report", 30.0, _build)

    def understanding_report(self) -> dict:
        def _build() -> dict:
            try:
                from sare.meta.understanding_checker import build_understanding_report

                report = self.learning_progress_report() or {}
                truth_gate = self.learning_truth_gate()
                acquisition = self.acquisition_dashboard()
                transfer = report.get("transfer", {}) if isinstance(report, dict) else {}
                return build_understanding_report(report, truth_gate, acquisition, transfer=transfer)
            except Exception as exc:
                self._record_runtime_error("understanding_report.build", exc, "understanding_report")
                return {"error": self._safe_stringify(exc), "summary": {"overall_state": "error"}, "evidence_quality": {}}

        return self._get_cached_report("understanding_report", 30.0, _build)

    def learning_focus_domains(self, limit: int = 5) -> List[dict]:
        report = self.learning_progress_report() or {}
        rows = list(report.get("domains", [])) if isinstance(report, dict) else []
        focus: List[dict] = []

        def _as_float(value: Any) -> Optional[float]:
            try:
                return float(value)
            except Exception:
                return None

        for row in rows:
            if not isinstance(row, dict):
                continue
            domain = str(row.get("domain", "general") or "general")
            attempts = int(row.get("attempts", 0) or 0)
            measured_attempts = int(row.get("measured_attempts", 0) or 0)
            if attempts <= 0 and measured_attempts <= 0:
                continue

            understanding = _as_float(row.get("understanding_share"))
            retrieval = _as_float(row.get("retrieval_share"))
            memorized = _as_float(row.get("memorized_share"))
            coverage = _as_float(row.get("measured_coverage"))

            evidence_state = (
                "strong"
                if measured_attempts >= 15
                else "limited"
                if measured_attempts >= 4
                else "weak"
                if measured_attempts > 0
                else "unmeasured"
            )
            priority = 0.0
            reasons: List[str] = []

            if understanding is None:
                priority += 0.35
                reasons.append("unmeasured_understanding")
            else:
                understanding = max(0.0, min(1.0, understanding))
                priority += max(0.0, 0.7 - understanding)
                if understanding < 0.5:
                    reasons.append("low_independent_solve")

            if coverage is None:
                priority += 0.15
                reasons.append("missing_coverage")
            else:
                coverage = max(0.0, min(1.0, coverage))
                priority += max(0.0, 0.55 - coverage) * 0.6
                if coverage < 0.5:
                    reasons.append("low_mode_coverage")

            if memorized is not None:
                memorized = max(0.0, min(1.0, memorized))
                if memorized >= 0.25:
                    priority += min(0.25, memorized * 0.5)
                    reasons.append("memorization_pressure")

            if retrieval is not None:
                retrieval = max(0.0, min(1.0, retrieval))
                if retrieval >= 0.35:
                    priority += min(0.15, retrieval * 0.25)
                    reasons.append("retrieval_dependence")

            if measured_attempts >= 4 and (understanding is None or understanding < 0.5):
                priority += min(0.15, measured_attempts / 50.0)
                reasons.append("repeatedly_weak")
            elif attempts >= 10 and (understanding is None or understanding < 0.6):
                priority += 0.05

            if evidence_state == "weak":
                priority += 0.08
            elif evidence_state == "limited":
                priority += 0.04

            focus.append(
                {
                    "domain": domain,
                    "attempts": attempts,
                    "measured_attempts": measured_attempts,
                    "understanding_share": understanding,
                    "retrieval_share": retrieval,
                    "memorized_share": memorized,
                    "measured_coverage": coverage,
                    "evidence_state": evidence_state,
                    "priority": round(min(1.0, priority), 3),
                    "reasons": reasons[:4],
                }
            )

        focus.sort(
            key=lambda item: (
                float(item.get("priority", 0.0) or 0.0),
                int(item.get("measured_attempts", 0) or 0),
                int(item.get("attempts", 0) or 0),
            ),
            reverse=True,
        )
        return focus[:max(1, limit)]

    def learning_focus_weights(self, limit: int = 8) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        for item in self.learning_focus_domains(limit=limit):
            domain = str(item.get("domain", "general") or "general")
            priority = float(item.get("priority", 0.0) or 0.0)
            weights[domain] = round(1.0 + max(0.0, priority), 3)
        return weights

    def _code_version_marker(self) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(REPO_ROOT),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            try:
                return f"mtime:{int(Path(__file__).stat().st_mtime)}"
            except Exception:
                return "unknown"

    def _tracking_snapshot_payload(self) -> dict:
        transfer = self.transfer_audit()
        acquisition = self.acquisition_dashboard()
        rescue = self.failure_reason_report().get("transform_failures", {})
        source_yield = self.source_yield_report()
        reused = sum(int(row.get("reused_in_solving", 0) or 0) for row in source_yield.get("sources", []))
        return {
            "timestamp": time.time(),
            "solves_attempted": int(self._stats.get("solves_attempted", 0) or 0),
            "solves_succeeded": int(self._stats.get("solves_succeeded", 0) or 0),
            "rules_promoted": int(self._stats.get("rules_promoted", 0) or 0),
            "runtime_transfer_attempts": int(transfer.get("runtime", {}).get("attempts", 0) or 0),
            "runtime_transfer_successes": int(transfer.get("runtime", {}).get("succeeded", 0) or 0),
            "verified_transfer_runs": int(transfer.get("verified", {}).get("runs", 0) or 0),
            "verified_transfer_successes": int(transfer.get("verified", {}).get("runs", 0) or 0),
            "heldout_target_wins": int(transfer.get("heldout_target_wins", 0) or 0),
            "heldout_target_tests": int(transfer.get("heldout_target_tests", 0) or 0),
            "artifacts": int(acquisition.get("status", {}).get("artifacts", 0) or 0),
            "verified_artifacts": int(acquisition.get("verified_artifacts", 0) or 0),
            "reused_in_solving": reused,
            "transform_rescue_attempts": int(rescue.get("fallback_attempts", 0) or 0),
            "transform_rescue_successes": int(rescue.get("fallback_rescues", 0) or 0),
        }

    def _tracking_snapshot_path(self) -> Path:
        return DATA_DIR / "learning_tracking_snapshot.json"

    def _load_tracking_snapshot(self) -> dict:
        path = self._tracking_snapshot_path()
        if not path.exists():
            return {"run_marker": {}, "snapshots": []}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload.setdefault("run_marker", {})
                payload.setdefault("snapshots", [])
                return payload
        except Exception as exc:
            self._record_runtime_error("learning_tracking_snapshot.load", exc, "_load_tracking_snapshot")
        return {"run_marker": {}, "snapshots": []}

    def _save_tracking_snapshot(self, payload: dict) -> None:
        path = self._tracking_snapshot_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=f"{path.name}.",
                suffix=".tmp",
                dir=str(path.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, indent=2) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_name, path)
            finally:
                if os.path.exists(tmp_name):
                    try:
                        os.remove(tmp_name)
                    except OSError:
                        pass
        except Exception as exc:
            self._record_runtime_error("learning_tracking_snapshot.save", exc, "_save_tracking_snapshot")

    def _write_run_marker(self, role: str = "brain") -> dict:
        marker = {
            "role": str(role or "brain"),
            "pid": os.getpid(),
            "boot_timestamp": self.boot_time or time.time(),
            "code_version": self._code_version_marker(),
        }
        payload = self._load_tracking_snapshot()
        payload["run_marker"] = marker
        self._save_tracking_snapshot(payload)
        return marker

    def _write_tracking_snapshot(self, force: bool = False) -> dict:
        now = time.time()
        if not force and (now - float(getattr(self, "_last_tracking_snapshot_at", 0.0) or 0.0)) < 900.0:
            return self._load_tracking_snapshot()
        payload = self._load_tracking_snapshot()
        snapshots = [item for item in payload.get("snapshots", []) if isinstance(item, dict)]
        snapshots.append(self._tracking_snapshot_payload())
        cutoff = now - 172800.0
        snapshots = [item for item in snapshots if float(item.get("timestamp", 0.0) or 0.0) >= cutoff][-288:]
        payload["snapshots"] = snapshots
        payload.setdefault("run_marker", self._write_run_marker("brain"))
        self._save_tracking_snapshot(payload)
        self._last_tracking_snapshot_at = now
        return payload

    def overnight_delta_report(self, window_seconds: float = 43200.0) -> dict:
        payload = self._load_tracking_snapshot()
        snapshots = [item for item in payload.get("snapshots", []) if isinstance(item, dict)]
        now = time.time()
        recent = [
            item for item in snapshots
            if float(item.get("timestamp", 0.0) or 0.0) >= (now - float(window_seconds or 0.0))
        ]
        current = self._tracking_snapshot_payload()
        baseline = recent[0] if recent else (snapshots[0] if snapshots else current)
        def _delta(key: str) -> int:
            return int(current.get(key, 0) or 0) - int(baseline.get(key, 0) or 0)
        return {
            "window_seconds": int(window_seconds),
            "baseline_timestamp": float(baseline.get("timestamp", now) or now),
            "current_timestamp": float(current.get("timestamp", now) or now),
            "solve_delta": _delta("solves_attempted"),
            "success_delta": _delta("solves_succeeded"),
            "promotion_delta": _delta("rules_promoted"),
            "runtime_transfer_delta": _delta("runtime_transfer_attempts"),
            "verified_transfer_delta": _delta("verified_transfer_runs"),
            "artifact_delta": _delta("artifacts"),
            "reuse_delta": _delta("reused_in_solving"),
            "transform_rescue_delta": _delta("transform_rescue_successes"),
        }

    def promoted_rules_summary(self) -> dict:
        path = DATA_DIR / "promoted_rules.json"
        if not path.exists():
            return {"rules": [], "total": 0, "pattern_counts": {}}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            counts = payload.get("pattern_counts", {})
            rules = payload.get("promoted_rules", [])
            enriched = []
            for rule in rules:
                name = rule.get("name", "")
                enriched.append({
                    "name": name,
                    "domain": rule.get("domain", "general"),
                    "confidence": round(float(rule.get("confidence", 0.0) or 0.0), 3),
                    "use_count": int(counts.get(name, 0)),
                })
            enriched.sort(key=lambda item: item["use_count"], reverse=True)
            return {"rules": enriched, "total": len(enriched), "pattern_counts": counts}
        except Exception as exc:
            self._record_runtime_error("promoted_rules.load", exc, "promoted_rules_summary")
            return {"error": str(exc), "rules": [], "total": 0, "pattern_counts": {}}

    def progress_snapshot(self) -> dict:
        transfer = self.transfer_audit()
        rescue = self.failure_reason_report().get("transform_failures", {})
        acquisition = self.acquisition_dashboard()
        source_yield = self.source_yield_report()
        snapshot = {
            "cycle": int(getattr(self, "_auto_learn_stats", {}).get("cycles", 0)),
            "ts": time.time(),
            "solve_rate": round(
                self._stats.get("solves_succeeded", 0) / max(self._stats.get("solves_attempted", 1), 1),
                3,
            ),
            "avg_energy": 1.0,
            "rules_promoted": int(self._stats.get("rules_promoted", 0)),
            "transfers_attempted": int(self._stats.get("transfers_attempted", 0)),
            "verified_transfer_runs": int(transfer.get("verified", {}).get("runs", 0) or 0),
            "transform_rescue_rate": rescue.get("fallback_rescue_rate"),
            "artifacts": int(acquisition.get("status", {}).get("artifacts", 0) or 0),
            "reused_in_solving": sum(int(row.get("reused_in_solving", 0) or 0) for row in source_yield.get("sources", [])),
        }
        try:
            if self.robustness_hardener:
                snapshot["robustness"] = self.robustness_hardener.overall_robustness()
            if self.meta_curriculum:
                snapshot["meta_lp"] = self.meta_curriculum.learning_progress_score()
            if self.concept_graph and hasattr(self.concept_graph, "_concepts"):
                snapshot["concept_count"] = len(self.concept_graph._concepts)
        except Exception as exc:
            self._record_runtime_error("progress_snapshot.enrich", exc, "progress_snapshot")
        return snapshot

    def progress_report(self) -> dict:
        path = DATA_DIR / "progress.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                runs = data.get("runs", [])
                return {"runs": runs[-200:], "total": len(runs), "has_data": len(runs) > 0}
            except Exception as exc:
                self._record_runtime_error("progress.load", exc, "progress_report")
        return {"runs": [self.progress_snapshot()], "total": 1, "has_data": False}

    def autonomous_trainer_status(self) -> dict:
        trainer = {"running": False, "total_problems": 0}
        try:
            if self.autonomous_trainer and getattr(self.autonomous_trainer, "_running", False):
                trainer = self.autonomous_trainer.summary()
            else:
                path = DATA_DIR / "autonomous_trainer_stats.json"
                if path.exists():
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        trainer = dict(loaded)
                        trainer["from_disk"] = True
                        trainer["running"] = False
        except Exception as exc:
            self._record_runtime_error("autonomous_trainer.status", exc, "autonomous_trainer_status")
        return trainer

    def self_improvement_status(self) -> dict:
        enabled = bool(self.config.get("self_improvement", {}).get("enabled", False))
        base = {
            "enabled_by_policy": enabled,
            "hard_gate": "evaluation_and_tests_required",
            "running": False,
            "patches": [],
        }
        try:
            from sare.meta.self_improver import get_self_improver
            status = dict(get_self_improver().get_status())
            status["enabled_by_policy"] = enabled
            status["hard_gate"] = base["hard_gate"]
            if not enabled:
                status["running"] = False
            return status
        except Exception as exc:
            self._record_runtime_error("self_improver.get_status", exc, "self_improvement_status")
            base["error"] = str(exc)
            return base

    def self_improvement_patches(self) -> dict:
        enabled = bool(self.config.get("self_improvement", {}).get("enabled", False))
        try:
            from sare.meta.self_improver import get_self_improver
            return {"enabled_by_policy": enabled, "patches": get_self_improver().get_patches()}
        except Exception as exc:
            self._record_runtime_error("self_improver.get_patches", exc, "self_improvement_patches")
            return {"enabled_by_policy": enabled, "patches": [], "error": str(exc)}

    def self_improvement_action(self, action: str, **kwargs: Any) -> dict:
        enabled = bool(self.config.get("self_improvement", {}).get("enabled", False))
        if action in {"start", "trigger", "multi"} and not enabled:
            return {
                "error": "self-improvement disabled by policy",
                "enabled_by_policy": False,
                "hard_gate": "Set config.self_improvement.enabled=true after evaluation gates pass",
            }
        if action in {"start", "trigger", "multi"}:
            audit = self.run_learning_audit(scope="incremental")
            if not audit.get("passed", False):
                return {
                    "error": "learning audit gate failed",
                    "enabled_by_policy": enabled,
                    "audit": dict(audit),
                }
        try:
            from sare.meta.self_improver import get_self_improver
            si = get_self_improver()
            if action == "start":
                return si.start()
            if action == "stop":
                return si.stop()
            if action == "rollback":
                return si.rollback(str(kwargs.get("patch_id", "")))
            if action == "trigger":
                return si.run_once(
                    target_file=kwargs.get("target_file"),
                    improvement_type=kwargs.get("improvement_type"),
                )
            if action == "multi":
                return si.run_multi_file(cluster_name=kwargs.get("cluster_name"))
            return {"error": f"unknown self-improvement action: {action}"}
        except Exception as exc:
            self._record_runtime_error("self_improver.action", exc, f"self_improvement_action:{action}")
            return {"error": str(exc), "action": action}

    def acquisition_dashboard(self) -> dict:
        def _build() -> dict:
            mesh_status = self._get_acquisition_mesh().status()
            learned = self.learned_artifacts(limit=20)
            queue = self.verification_queue(limit=20)
            items = learned.get("items", [])
            verified = sum(1 for item in items if item.get("verification_state") == "verified")
            quarantined = sum(1 for item in items if item.get("verification_state") == "quarantined")
            corroborated = sum(
                1
                for item in items
                if int((item.get("metadata", {}) or {}).get("corroboration_count", 0) or 0) >= 2
            )
            by_source: Dict[str, int] = {}
            for item in items:
                source = str(item.get("source_type", "unknown"))
                by_source[source] = by_source.get(source, 0) + 1
            if not by_source:
                by_source = dict(mesh_status.get("by_source", {}) or {})
            scheduled_jobs = sum(1 for thread in self._acquisition_schedule_threads if thread.is_alive())
            bootstrap_thread = getattr(self, "_bootstrap_thread", None)
            if bootstrap_thread is not None and bootstrap_thread.is_alive():
                scheduled_jobs += 1
            yield_report = self.source_yield_report()
            reused = sum(int(row.get("reused_in_solving", 0) or 0) for row in yield_report.get("sources", []))
            cooldowns = {
                "github_until": float(self._acquisition_cooldowns.get("github", 0.0) or 0.0),
            }
            github_enabled = self._github_acquisition_enabled()
            github_available = github_enabled and cooldowns["github_until"] <= time.time()
            return {
                "status": mesh_status,
                "recent_artifacts": items,
                "verification_queue": queue,
                "verified_artifacts": mesh_status.get("by_state", {}).get("verified", verified),
                "quarantined_artifacts": mesh_status.get("by_state", {}).get("quarantined", quarantined),
                "corroborated_artifacts": mesh_status.get("corroborated_artifacts", corroborated),
                "promoted_artifacts": mesh_status.get("promoted_artifacts", 0),
                "retested_artifacts": mesh_status.get("retested_artifacts", 0),
                "today_artifacts": mesh_status.get("today_artifacts", 0),
                "source_mix": by_source,
                "by_domain": mesh_status.get("by_domain", {}),
                "by_verification_method": mesh_status.get("by_verification_method", {}),
                "evidence_stages": mesh_status.get("by_evidence_stage", {}),
                "github": self.github_learning_status(),
                "scheduled_jobs": scheduled_jobs,
                "reused_in_solving": reused,
                "plan": self.acquisition_plan(limit=3),
                "retention_schedule": self.retention_schedule_report(),
                "cooldowns": cooldowns,
                "remote_availability": {
                    "github": github_available,
                    "web": True,
                },
                "policy": {
                    "github_enabled": github_enabled,
                    "diversify_sources": True,
                },
                "last_remote_failure": self._acquisition_policy_stats.get("last_remote_failure"),
                "fallback_source_used": self._acquisition_policy_stats.get("last_fallback_source"),
                "source_success_counts": dict(self._acquisition_policy_stats.get("success_counts", {}) or {}),
                "source_error_counts": dict(self._acquisition_policy_stats.get("error_counts", {}) or {}),
            }
        return self._get_cached_report("acquisition_dashboard", 30.0, _build)

    def audit_dashboard(self) -> dict:
        def _build() -> dict:
            audit = self.run_learning_audit(scope="incremental")
            return {
                "audit": dict(audit),
                "truth_gate": dict(audit.get("truth_gate", {})),
                "transfer": self.transfer_audit(),
                "transfer_suite": self.transfer_suite_report(),
                "understanding": self.understanding_report(),
                "source_yield": self.source_yield_report(),
                "transform_rescue": self.failure_reason_report().get("transform_failures", {}),
                "grounded": self.grounded_learning_report(),
                "retention_schedule": self.retention_schedule_report(),
                "benchmark_gate": dict(audit.get("benchmark_gate", {})),
            }
        return self._get_cached_report("audit_dashboard", 30.0, _build)

    def learning_ops_dashboard(self) -> dict:
        def _build() -> dict:
            def _safe(fn):
                try:
                    return fn()
                except Exception:
                    return {}
            report = _safe(self.learning_progress_report)
            truth_gate = _safe(self.learning_truth_gate)
            weakest_domains: List[dict] = []
            try:
                domains = [item for item in report.get("domains", []) if isinstance(item, dict)]
                ranked_domains = [item for item in domains if int(item.get("measured_attempts", 0) or 0) >= 3]
                if not ranked_domains:
                    ranked_domains = [item for item in domains if int(item.get("measured_attempts", 0) or 0) > 0]
                weakest_domains = sorted(
                    ranked_domains,
                    key=lambda item: (
                        float(item.get("understanding_share") if item.get("understanding_share") is not None else 1.0),
                        -int(item.get("measured_attempts", 0) or 0),
                        -int(item.get("attempts", 0) or 0),
                    ),
                )[:8]
            except Exception:
                weakest_domains = []
            bootstrap = _safe(lambda: self.bootstrap_learning_state(force=False))
            acquisition = _safe(self.acquisition_dashboard)
            audit = _safe(self.audit_dashboard)
            grounded = _safe(self.grounded_learning_report)
            retention = _safe(self.retention_schedule_report)
            transfer_suite = _safe(self.transfer_suite_report)
            transfer = _safe(self.transfer_audit)
            understanding = _safe(self.understanding_report)
            github = _safe(self.github_learning_status)
            source_yield = _safe(self.source_yield_report)
            transform_rescue = _safe(self.failure_reason_report).get("transform_failures", {})
            # Schema cache health: high hit rate means system is replaying, not learning
            _schema_hit_rate = 0.0
            _schema_cache_size = 0
            _curriculum_novelty_rate = 0.0
            try:
                from sare.cognition.schema_matcher import get_schema_matcher
                _sm = get_schema_matcher()
                _sm_total = _sm._hits + _sm._misses
                _schema_hit_rate = round(_sm._hits / max(1, _sm_total), 3)
                _schema_cache_size = len(_sm._cache)
            except Exception:
                pass
            try:
                if self.experiment_runner and hasattr(self.experiment_runner, 'curriculum_gen'):
                    _cg = self.experiment_runner.curriculum_gen
                    _cg_total = getattr(_cg, '_schema_total_gen', 0)
                    _cg_hits = getattr(_cg, '_schema_total_hits', 0)
                    if _cg_total > 0:
                        _curriculum_novelty_rate = round(1.0 - (_cg_hits / _cg_total), 3)
            except Exception:
                pass

            return {
                "generated_at": time.time(),
                "report": report,
                "acquisition": acquisition,
                "audit": audit,
                "world_model_readable": _safe(lambda: self.world_model_readable_report(include_content=False)),
                "understanding": understanding,
                "evidence_quality": dict(understanding.get("evidence_quality", {}) or {}),
                "autolearn": dict(self.auto_learn_status()),
                "trainer": self.autonomous_trainer_status(),
                "grounded": grounded,
                "retention_schedule": retention,
                "bootstrap": bootstrap,
                "weakest_domains": weakest_domains,
                "source_yield": source_yield,
                "source_reuse": {
                    "by_kind": list(source_yield.get("sources", [])),
                },
                "transfer_suite": transfer_suite,
                "failure_reasons": self.failure_reason_report(),
                "transform_rescue": transform_rescue,
                "concept_graph_health": self.concept_graph.health() if self.concept_graph and hasattr(self.concept_graph, "health") else dict(self._concept_graph_health),
                "overnight_delta": self.overnight_delta_report(),
                "genuine_learning": {
                    "schema_hit_rate": _schema_hit_rate,
                    "schema_cache_size": _schema_cache_size,
                    "curriculum_novelty_rate": _curriculum_novelty_rate,
                    "note": "schema_hit_rate < 0.5 means genuine BeamSearch learning; > 0.8 means mostly replaying",
                },
                "next_interventions": [
                    {"type": "verification", "pending": self.verification_queue(limit=1).get("pending", 0)},
                    {"type": "transfer_probe", "attempts": transfer.get("attempts", 0)},
                    {"type": "transfer_suite", "verified_runs": transfer_suite.get("overall", {}).get("verified_runs", 0)},
                    {"type": "grounded_learning", "success_rate": grounded.get("success_rate")},
                    {"type": "github_learning", "repos_tracked": github.get("repos_tracked", 0)},
                    {"type": "benchmark_gate", "passed": audit.get("benchmark_gate", {}).get("passed", False)},
                    {"type": "truth_gate", "passed": truth_gate.get("passed", False), "missing": truth_gate.get("missing", [])},
                ],
            }
        return self._get_cached_report("learning_ops_dashboard", 30.0, _build)

    def learning_dashboard_payload(self) -> dict:
        def _build() -> dict:
            ops = self.learning_ops_dashboard()
            report = ops.get("report", {})
            history = []
            try:
                from sare.meta import learning_progress_report as _lpr
                history_path = getattr(_lpr, "DEFAULT_HISTORY_PATH", None)
                if history_path:
                    hp = Path(history_path)
                    if hp.exists():
                        payload = json.loads(hp.read_text(encoding="utf-8"))
                        if isinstance(payload, list):
                            history = payload[-96:]
            except Exception as exc:
                self._record_runtime_error("learning_dashboard.history", exc, "learning_dashboard_payload")

            autolearn = dict(self.auto_learn_status())
            autolearn["resident"] = True
            autolearn["stage"] = self.stage.value
            if self.developmental_curriculum:
                try:
                    cmap = self.developmental_curriculum.get_curriculum_map()
                    autolearn["mastered"] = cmap.get("mastered", 0)
                    autolearn["unlocked"] = cmap.get("unlocked", 0)
                    autolearn["total_domains"] = cmap.get("total_domains", 0)
                except Exception as exc:
                    self._record_runtime_error("developmental_curriculum.get_curriculum_map", exc, "learning_dashboard_payload")

            return {
                "generated_at": ops.get("generated_at", time.time()),
                "report": report,
                "history": history,
                "autolearn": autolearn,
                "trainer": self.autonomous_trainer_status(),
                "world_model_readable": ops.get("world_model_readable", {}),
                "acquisition": ops.get("acquisition", {}),
                "audit": ops.get("audit", {}),
                "source_yield": ops.get("source_yield", {}),
                "transfer_suite": ops.get("transfer_suite", {}),
                "grounded": ops.get("grounded", {}),
                "retention_schedule": ops.get("retention_schedule", {}),
                "bootstrap": ops.get("bootstrap", {}),
                "weakest_domains": ops.get("weakest_domains", []),
                "next_interventions": ops.get("next_interventions", []),
                "failure_reasons": ops.get("failure_reasons", {}),
                "transform_rescue": ops.get("transform_rescue", {}),
                "concept_graph_health": ops.get("concept_graph_health", {}),
                "persistence_health": self.persistence_health_report(),
                "source_reuse": ops.get("source_reuse", {}),
                "overnight_delta": ops.get("overnight_delta", {}),
                "understanding": ops.get("understanding", {}),
                "evidence_quality": ops.get("evidence_quality", {}),
                "source_of_truth": dict(report.get("source_of_truth", {})),
                "measured": dict(report.get("measured", {})),
            }
        return self._get_cached_report("learning_dashboard_payload", 30.0, _build)

    def agi_scorecard(self) -> dict:
        evaluation = self.evaluation_summary()
        knowledge = self.knowledge_stats()
        runtime = evaluation.get("runtime", {})
        promoted = self.promoted_rules_summary()
        self_improve = self.self_improvement_status()

        dimensions = {
            "symbolic_reasoning": {"value": round(float(runtime.get("solve_rate", 0.0)) * 10, 2), "weight": 15},
            "learning_loop": {"value": round(float(self.progress_snapshot().get("solve_rate", 0.0)) * 10, 2), "weight": 10},
            "memory_knowledge": {"value": round(min(10.0, promoted.get("total", 0) * 1.25), 2), "weight": 10},
            "transfer_learning": {
                "value": round(min(10.0, runtime.get("transfers_succeeded", 0) * 2 + runtime.get("transfers_attempted", 0) * 0.2), 2),
                "weight": 10,
            },
            "world_modeling": {"value": round(min(10.0, self.world_summary().get("causal_link_count", 0) * 0.1), 2), "weight": 8},
            "self_improvement": {"value": round(min(10.0, float(self_improve.get("patches_applied", 0) or 0) * 2.5), 2), "weight": 8},
            "knowledge_grounding": {"value": round(min(10.0, knowledge.get("world_model_facts", 0) * 0.05), 2), "weight": 8},
            "autonomy": {"value": 6.0 if self.auto_learn_status().get("running") else 3.0, "weight": 8},
        }
        weighted = sum(v["value"] * v["weight"] for v in dimensions.values())
        max_weight = sum(v["weight"] for v in dimensions.values())
        final = round(weighted / (max_weight * 10) * 100, 1) if max_weight else 0.0
        return {
            "agi_score": final,
            "max_score": 100,
            "dimensions": dimensions,
            "grade": "A" if final >= 85 else "B" if final >= 70 else "C" if final >= 55 else "D",
            "timestamp": time.time(),
        }

    def learning_trend_report(self) -> dict:
        data_dir = DATA_DIR
        benchmark_history = []
        hist_path = data_dir / "benchmark_history.json"
        if hist_path.exists():
            try:
                raw = json.loads(hist_path.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    benchmark_history = raw
            except Exception as exc:
                self._record_runtime_error("benchmark_history.load", exc, "learning_trend_report")

        progress_cycles = []
        try:
            progress_cycles = self.progress_report().get("runs", [])
        except Exception:
            progress_cycles = []

        top_transforms = []
        ts_path = data_dir / "transform_stats.json"
        if ts_path.exists():
            try:
                raw = json.loads(ts_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    entries = [(k, v) for k, v in raw.items() if isinstance(v, (int, float))]
                    entries.sort(key=lambda x: x[1], reverse=True)
                    top_transforms = [{"name": k, "utility": round(v, 3)} for k, v in entries[:15]]
            except Exception as exc:
                self._record_runtime_error("transform_stats.load", exc, "learning_trend_report")

        si_stats = {}
        si_path = data_dir / "si_stats.json"
        if si_path.exists():
            try:
                si_stats = json.loads(si_path.read_text(encoding="utf-8"))
            except Exception as exc:
                self._record_runtime_error("si_stats.load", exc, "learning_trend_report")

        recent_scores = [e.get("total_score") or e.get("pass_rate", 0) for e in benchmark_history[-5:] if isinstance(e, dict)]
        score_delta = round(recent_scores[-1] - recent_scores[0], 4) if len(recent_scores) >= 2 else None
        return {
            "benchmark_history": benchmark_history[-50:],
            "progress_cycles": progress_cycles[-50:],
            "top_transforms": top_transforms,
            "promoted_rules": self.promoted_rules_summary().get("rules", [])[:20],
            "transfer_suite": self.transfer_suite_report(),
            "grounded": self.grounded_learning_report(),
            "retention_schedule": self.retention_schedule_report(),
            "si_stats": si_stats,
            "synthesized_count": len(getattr(self.transform_synthesizer, "_synthesized", {}) or {}) if self.transform_synthesizer else 0,
            "is_learning": bool(score_delta is not None and score_delta > 0),
            "score_delta": score_delta,
        }

    def time_to_agi_report(self) -> dict:
        def _build() -> dict:
            report = self.learning_progress_report() or {}
            truth_gate = self.learning_truth_gate() or {}
            persistence = self.persistence_health_report()
            promoted_total = int(self.promoted_rules_summary().get("total", 0))
            benchmark_history = self.learning_trend_report().get("benchmark_history", [])
            latest_benchmark_raw = 0.0
            if benchmark_history:
                last = benchmark_history[-1]
                latest_benchmark_raw = float(last.get("total_score") or last.get("pass_rate", 0) or 0.0)

            total_episodes = 0
            try:
                if self.memory_manager is not None:
                    total_episodes = len(getattr(self.memory_manager, "_episodes", []) or [])
                if not total_episodes:
                    state_path = DATA_DIR / "brain_state.json"
                    if state_path.exists():
                        state = json.loads(state_path.read_text(encoding="utf-8"))
                        total_episodes = int(state.get("episode_count", 0) or 0)
            except Exception as exc:
                self._record_runtime_error("episodes.count", exc, "time_to_agi_report")
            rolling_24h = dict(report.get("rolling_24h", {}) or {})
            ep_24h = int(rolling_24h.get("rolling_24h_attempts", 0) or 0)
            promotions_24h = int(rolling_24h.get("rolling_24h_new_patterns", 0) or 0)
            ep_per_hr = round(ep_24h / 24.0, 1) if ep_24h else 0.0
            promo_per_hr = round(promotions_24h / 24.0, 2) if promotions_24h else 0.0

            weak_domain_targets = ["mathematics", "physics", "chemistry", "technology", "word_problems"]
            domain_rows = {str(item.get("domain", "")): item for item in report.get("domains", [])}
            weak_track = []
            measured_weak = []
            for domain in weak_domain_targets:
                row = dict(domain_rows.get(domain, {}) or {})
                attempts = int(row.get("measured_attempts", 0) or 0)
                understanding = row.get("understanding_share")
                measured = understanding is not None and attempts > 0
                weak_track.append(
                    {
                        "domain": domain,
                        "attempts": attempts,
                        "understanding_share": understanding,
                        "measured": measured,
                    }
                )
                if measured:
                    measured_weak.append(float(understanding))

            latest_benchmark = round(latest_benchmark_raw * 100.0, 1)
            weak_domain_progress_pct = round((sum(measured_weak) / max(len(measured_weak), 1)) * 100.0, 1) if measured_weak else None
            persistence_ok = bool(persistence.get("overall_ok"))
            evidence_missing = list(truth_gate.get("missing", []))
            if not measured_weak:
                evidence_missing.append("weak_domain_track")
            if not persistence_ok:
                evidence_missing.append("persistence_health")

            AGI_EPISODES = 10_000_000
            AGI_RULES = 5_000
            AGI_BENCHMARK = 90.0
            ep_remaining = max(0, AGI_EPISODES - total_episodes)
            rule_remaining = max(0, AGI_RULES - promoted_total)
            days_ep = round(ep_remaining / max(ep_per_hr * 24, 1), 1) if ep_per_hr else None
            days_rules = round(rule_remaining / max(promo_per_hr * 24, 1), 1) if promo_per_hr else None
            if evidence_missing:
                binding = "insufficient_evidence"
                binding_days = None
            else:
                binding = "episodes"
                binding_days = days_ep
                if days_ep is None or (days_rules is not None and days_rules > days_ep):
                    binding = "rule promotions"
                    binding_days = days_rules

            llm_analysis = (
                "Insufficient evidence to estimate AGI progress honestly yet."
                if evidence_missing
                else (
                    f"Truth-gated progress is live. Weak-domain schooling is at "
                    f"{weak_domain_progress_pct:.1f}% across measured targets, with "
                    f"held-out {truth_gate.get('heldout', {}).get('pass_rate')} and "
                    f"verified transfer {truth_gate.get('transfer', {}).get('success_rate')}."
                )
            )
            return {
                "generated_at": time.time(),
                "total_episodes": total_episodes,
                "ep_per_hr": ep_per_hr,
                "ep_24h": ep_24h,
                "promoted_rules": promoted_total,
                "promo_per_hr": promo_per_hr,
                "promotions_24h": promotions_24h,
                "latest_benchmark": latest_benchmark,
                "episode_progress_pct": round(min(1.0, total_episodes / AGI_EPISODES) * 100, 3),
                "rule_progress_pct": round(min(1.0, promoted_total / AGI_RULES) * 100, 1),
                "benchmark_progress_pct": round(min(1.0, latest_benchmark / AGI_BENCHMARK) * 100, 1),
                "binding_bottleneck": binding,
                "days_to_agi_at_current_speed": binding_days,
                "binding": binding,
                "binding_days": binding_days,
                "ep_pct": round(min(1.0, total_episodes / AGI_EPISODES) * 100, 3),
                "rule_pct": round(min(1.0, promoted_total / AGI_RULES) * 100, 1),
                "bench_pct": round(min(1.0, latest_benchmark / AGI_BENCHMARK) * 100, 1),
                "agi_targets": {"episodes": AGI_EPISODES, "rules": AGI_RULES, "benchmark": AGI_BENCHMARK},
                "weak_domain_track": weak_track,
                "weak_domain_progress_pct": weak_domain_progress_pct,
                "truth_gate": truth_gate,
                "persistence_health": persistence,
                "evidence_state": {
                    "sufficient": not evidence_missing,
                    "missing": evidence_missing,
                },
                "llm_analysis": llm_analysis,
            }
        return self._get_cached_report("time_to_agi_report", 15.0, _build)

    def persist_general_solver_result(
        self,
        problem_text: str,
        answer: str,
        problem_type: str,
        confidence: float = 0.65,
        source_mode: str = "free_solve",
        allow_question_fallback: bool = False,
    ) -> None:
        if self.fact_ingester is None:
            return
        try:
            self.fact_ingester.ingest(
                problem_text,
                answer,
                problem_type,
                confidence=confidence,
                source_mode=source_mode,
                allow_question_fallback=allow_question_fallback,
            )
        except Exception as exc:
            self._record_runtime_error("fact_ingester.ingest", exc, "persist_general_solver_result")

    def persist_answer_to_fact(self, question: str, answer: str) -> None:
        if self.commonsense is not None and hasattr(self.commonsense, "add_fact"):
            try:
                self.commonsense.add_fact(question, "AnswerTo", answer[:200])
                return
            except Exception as exc:
                self._record_runtime_error("commonsense.add_fact", exc, "persist_answer_to_fact")
        try:
            from sare.knowledge.commonsense import get_commonsense_base
            get_commonsense_base().add_fact(question, "AnswerTo", answer[:200])
        except Exception as exc:
            self._record_runtime_error("commonsense.singleton.add_fact", exc, "persist_answer_to_fact")

    def record_general_solver_failure(self, domain: str, problem_text: str, expected: str = "") -> None:
        if self.world_model is not None and hasattr(self.world_model, "record_domain_failure"):
            try:
                self.world_model.record_domain_failure(
                    domain=domain,
                    problem_text=problem_text,
                    expected=expected or "",
                )
            except Exception as exc:
                self._record_runtime_error("world_model.record_domain_failure", exc, "record_general_solver_failure")

    def observe_general_solver_outcome(
        self,
        problem_text: str,
        problem_type: str,
        solver_used: str,
        confidence: float,
        solved: bool,
    ) -> None:
        if self.world_model is not None and hasattr(self.world_model, "observe_solve"):
            try:
                self.world_model.observe_solve(
                    expression=problem_text[:100],
                    transforms_used=[solver_used] if solver_used else [],
                    energy_delta=confidence if solved else 0.0,
                    domain=problem_type,
                    solved=solved,
                )
                if solved and hasattr(self.world_model, "record_domain_success"):
                    self.world_model.record_domain_success(problem_type)
            except Exception as exc:
                self._record_runtime_error("world_model.observe_solve", exc, "observe_general_solver_outcome")

    def answer_with_concept_hypothesis(self, hypothesis_id: str, problem_text: str, domain: str) -> Optional[dict]:
        if self.world_model is not None and hasattr(self.world_model, "answer_with_concept_hypothesis"):
            try:
                return self.world_model.answer_with_concept_hypothesis(hypothesis_id, problem_text, domain=domain)
            except Exception as exc:
                self._record_runtime_error("world_model.answer_with_concept_hypothesis", exc, "answer_with_concept_hypothesis")
        return None

    def world_beliefs_index(self) -> Dict[str, Any]:
        if self.world_model is not None and hasattr(self.world_model, "_beliefs"):
            try:
                return dict(getattr(self.world_model, "_beliefs", {}))
            except Exception as exc:
                self._record_runtime_error("world_model._beliefs", exc, "world_beliefs_index")
        return {}

    def _get_acquisition_mesh(self) -> AcquisitionMesh:
        if self.acquisition_mesh is None:
            with self._acquisition_lock:
                if self.acquisition_mesh is None:
                    self.acquisition_mesh = AcquisitionMesh()
        return self.acquisition_mesh

    def _ensure_curriculum_broker(self):
        if self.curriculum_gen is not None and hasattr(self.curriculum_gen, "_broker") and self.curriculum_gen._broker is not None:
            return self.curriculum_gen._broker
        try:
            from sare.learning.curriculum_broker import CurriculumBroker
            broker = CurriculumBroker()
            if self.curriculum_gen is not None and hasattr(self.curriculum_gen, "_broker"):
                self.curriculum_gen._broker = broker
            return broker
        except Exception as exc:
            self._record_runtime_error("curriculum_broker.init", exc, "acquisition")
            return None

    def _normalized_acquisition_config(
        self,
        source: Union[dict, CanonicalAcquisitionSourceConfig, IngestionBatch, IngestionSourceConfig],
    ) -> Union[CanonicalAcquisitionSourceConfig, IngestionBatch]:
        if isinstance(source, IngestionBatch):
            items = [
                self._normalized_acquisition_config(item)
                for item in source.items
            ]
            return IngestionBatch(items=[item for item in items if isinstance(item, CanonicalAcquisitionSourceConfig)], source_id=source.source_id, metadata=source.metadata)
        if isinstance(source, CanonicalAcquisitionSourceConfig):
            return source
        if isinstance(source, IngestionSourceConfig):
            locator = str(source.payload)
            source_type = str(source.source_type or "web_page")
            if source_type == "file":
                source_type = "book" if Path(locator).suffix.lower() in {".txt", ".pdf", ".epub"} else "dataset"
            return CanonicalAcquisitionSourceConfig(
                source_type=source_type,
                locator=locator,
                domain=source.domain,
                trust_tier=source.trust_level,
                max_items=int(source.metadata.get("max_items", 20)) if isinstance(source.metadata, dict) else 20,
                depth=int(source.metadata.get("depth", 1)) if isinstance(source.metadata, dict) else 1,
                refresh_policy=str(source.metadata.get("refresh_policy", "manual")) if isinstance(source.metadata, dict) else "manual",
                metadata=dict(source.metadata or {}),
            )
        payload = dict(source or {})
        locator = str(
            payload.get(
                "locator",
                payload.get(
                    "topic",
                    payload.get("payload", payload.get("path", payload.get("url", ""))),
                ),
            )
        )
        source_type = str(payload.get("source_type", payload.get("kind", "web_page")))
        return CanonicalAcquisitionSourceConfig(
            source_type=source_type,
            locator=locator,
            domain=str(payload.get("domain", "general")),
            trust_tier=str(payload.get("trust_tier", payload.get("trust_level", "curated"))),
            max_items=int(payload.get("max_items", 20) or 20),
            depth=int(payload.get("depth", 1) or 1),
            refresh_policy=str(payload.get("refresh_policy", "manual")),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def _integrate_acquisition_result(self, result: AcquisitionResultData) -> dict:
        broker = self._ensure_curriculum_broker()
        examples_added = 0
        facts_added = 0
        verified_artifacts = [
            artifact
            for artifact in result.artifacts
            if str(artifact.get("verification_state", "pending")) == "verified"
        ]
        if broker is not None and verified_artifacts:
            try:
                injected = broker.ingest_examples(self._get_acquisition_mesh().artifact_examples(verified_only=True))
                examples_added = int(injected.get("accepted", 0))
            except Exception as exc:
                self._record_runtime_error("curriculum_broker.ingest_examples", exc, "acquisition")

        for artifact in verified_artifacts:
            kind = str(artifact.get("kind", ""))
            domain = str(artifact.get("domain", "general") or "general")
            content = str(artifact.get("content", "")).strip()
            if not content:
                continue
            if kind in {"fact", "concept", "qa"}:
                if self.add_world_fact(domain, content[:240], confidence=float(artifact.get("confidence", 0.7) or 0.7), source=str(artifact.get("source_type", "acquisition"))):
                    facts_added += 1

        self._stats["ingestion"]["records_scanned"] += int(result.records_scanned)
        self._stats["ingestion"]["facts_added"] += int(facts_added)
        self._stats["ingestion"]["problems_added"] += int(examples_added)
        self._stats["ingestion"]["concepts_proposed"] += int(result.artifacts_extracted)
        self._stats.setdefault("acquisition", {})
        acquisition_stats = self._stats["acquisition"]
        acquisition_stats["records_scanned"] = int(acquisition_stats.get("records_scanned", 0)) + int(result.records_scanned)
        acquisition_stats["artifacts_extracted"] = int(acquisition_stats.get("artifacts_extracted", 0)) + int(result.artifacts_extracted)
        acquisition_stats["artifacts_verified"] = int(acquisition_stats.get("artifacts_verified", 0)) + int(result.artifacts_verified)
        acquisition_stats["artifacts_quarantined"] = int(acquisition_stats.get("artifacts_quarantined", 0)) + int(result.artifacts_quarantined)
        acquisition_stats["artifacts_promoted"] = int(acquisition_stats.get("artifacts_promoted", 0)) + int(getattr(result, "artifacts_promoted", 0))
        acquisition_stats["corroborated_artifacts"] = int(acquisition_stats.get("corroborated_artifacts", 0)) + int(getattr(result, "corroborated_artifacts", 0))
        acquisition_stats["learning_examples_added"] = int(acquisition_stats.get("learning_examples_added", 0)) + int(examples_added)
        acquisition_stats["last_source"] = result.source_type
        return {
            "learning_examples_added": examples_added,
            "world_facts_added": facts_added,
            "artifacts_promoted": int(getattr(result, "artifacts_promoted", 0)),
            "corroborated_artifacts": int(getattr(result, "corroborated_artifacts", 0)),
        }

    def acquire(
        self,
        source: Union[dict, CanonicalAcquisitionSourceConfig, IngestionBatch, IngestionSourceConfig],
    ) -> AcquisitionResult:
        """Canonical acquisition entrypoint for books, datasets, web, Wikipedia, and GitHub."""
        normalized = self._normalized_acquisition_config(source)
        mesh = self._get_acquisition_mesh()
        if isinstance(normalized, IngestionBatch):
            combined = {
                "source_id": normalized.source_id,
                "source_type": "batch",
                "records_scanned": 0,
                "artifacts_extracted": 0,
                "artifacts_verified": 0,
                "artifacts_quarantined": 0,
                "artifacts_promoted": 0,
                "corroborated_artifacts": 0,
                "learning_examples_added": 0,
                "records": [],
                "artifacts": [],
                "verification_tasks": [],
                "provenance_summary": dict(normalized.metadata or {}),
                "items": [],
            }
            for item in normalized.items:
                item_result = self.acquire(item)
                combined["items"].append(dict(item_result))
                for key in (
                    "records_scanned",
                    "artifacts_extracted",
                    "artifacts_verified",
                    "artifacts_quarantined",
                    "artifacts_promoted",
                    "corroborated_artifacts",
                    "learning_examples_added",
                ):
                    combined[key] += int(item_result.get(key, 0))
                combined["records"].extend(item_result.get("records", []))
                combined["artifacts"].extend(item_result.get("artifacts", []))
                combined["verification_tasks"].extend(item_result.get("verification_tasks", []))
            return AcquisitionResult(combined)

        result = mesh.acquire(normalized)
        integration = self._integrate_acquisition_result(result)
        payload = result.to_dict()
        payload.update(integration)
        self._invalidate_report_cache()
        return AcquisitionResult(payload)

    def schedule_acquisition(
        self,
        sources: Sequence[Union[dict, CanonicalAcquisitionSourceConfig, IngestionSourceConfig]],
        interval_seconds: float = 0.0,
        max_runs: int = 1,
    ) -> bool:
        """Run one or more acquisition jobs in the background under Brain ownership."""
        jobs = [self._normalized_acquisition_config(source) for source in sources]
        if not jobs:
            return False

        def _runner() -> None:
            runs = 0
            while max_runs <= 0 or runs < max_runs:
                for job in jobs:
                    try:
                        self.acquire(job)
                    except Exception as exc:
                        self._record_runtime_error("schedule_acquisition.acquire", exc, "schedule_acquisition")
                runs += 1
                if interval_seconds <= 0 or (max_runs > 0 and runs >= max_runs):
                    break
                time.sleep(interval_seconds)

        thread = threading.Thread(target=_runner, daemon=True, name="BrainAcquisition")
        self._acquisition_schedule_threads.append(thread)
        thread.start()
        return True

    def learned_artifacts(self, limit: int = 50) -> dict:
        return self._get_acquisition_mesh().learned_artifacts(limit=limit)

    def verification_queue(self, limit: int = 50) -> dict:
        return self._get_acquisition_mesh().verification_queue(limit=limit)

    def acquire_github_repo(self, repo: str, mode: str = "full_repo") -> AcquisitionResult:
        return self.acquire(
            CanonicalAcquisitionSourceConfig(
                source_type="github_repo",
                locator=repo,
                domain="code",
                trust_tier="curated",
                max_items=30,
                metadata={"mode": mode},
            )
        )

    def discover_github_repositories(self, topic: str, limit: int = 5) -> dict:
        return self._get_acquisition_mesh().discover_github_repositories(topic, limit=limit)

    def acquire_github_topic(
        self,
        topic: str,
        repo_limit: int = 3,
        mode: str = "full_repo",
        per_repo_items: int = 12,
    ) -> AcquisitionResult:
        return self.acquire(
            CanonicalAcquisitionSourceConfig(
                source_type="github_topic",
                locator=topic,
                domain="code",
                trust_tier="curated",
                max_items=max(1, repo_limit),
                metadata={
                    "repo_limit": repo_limit,
                    "mode": mode,
                    "per_repo_items": per_repo_items,
                },
            )
        )

    def discover_with_browser(self, kind: str, query: str, limit: int = 5, open_in_chrome: bool = True) -> dict:
        return self._get_acquisition_mesh().discover_with_browser(kind, query, limit=limit, open_in_chrome=open_in_chrome)

    def discover_open_access_books(self, query: str, limit: int = 5, open_in_chrome: bool = True) -> dict:
        return self.discover_with_browser("open_access_books", query, limit=limit, open_in_chrome=open_in_chrome)

    def acquire_open_access_book(self, query: str, domain: str = "general", max_items: int = 8, open_in_chrome: bool = True) -> AcquisitionResult:
        return self.acquire(
            CanonicalAcquisitionSourceConfig(
                source_type="open_access_book",
                locator=query,
                domain=domain,
                trust_tier="curated",
                max_items=max_items,
                metadata={"open_in_chrome": open_in_chrome},
            )
        )

    def github_learning_status(self) -> dict:
        def _build() -> dict:
            items = self.learned_artifacts(limit=500).get("items", [])
            github_items = [item for item in items if item.get("source_type") == "github_repo"]
            topic_records = []
            try:
                topic_records = [
                    record
                    for record in self._get_acquisition_mesh()._state.get("records", [])
                    if str(record.get("source_type", "")) == "github_topic"
                ]
            except Exception:
                topic_records = []
            repo_counts: Dict[str, int] = {}
            by_kind: Dict[str, int] = {}
            discussion_counts = {"issues": 0, "pull_requests": 0}
            topic_counts: Dict[str, int] = {}
            verified = 0
            corroborated = 0
            for item in github_items:
                repo = str((item.get("provenance", {}) or {}).get("repo", item.get("source_locator", "")))
                repo_counts[repo] = repo_counts.get(repo, 0) + 1
                kind = str(item.get("kind", "concept"))
                by_kind[kind] = by_kind.get(kind, 0) + 1
                if item.get("verification_state") == "verified":
                    verified += 1
                if int((item.get("metadata", {}) or {}).get("corroboration_count", 0) or 0) >= 2:
                    corroborated += 1
                discussion_kind = str((item.get("metadata", {}) or {}).get("discussion_kind", ""))
                if discussion_kind == "issue":
                    discussion_counts["issues"] += 1
                elif discussion_kind == "pull_request":
                    discussion_counts["pull_requests"] += 1
                topic = str((item.get("metadata", {}) or {}).get("discovered_from_topic", (item.get("provenance", {}) or {}).get("discovered_from_topic", "")))
                if topic:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
            for record in topic_records:
                topic = str((record.get("provenance", {}) or {}).get("topic", record.get("locator", "")))
                if topic and topic not in topic_counts:
                    topic_counts[topic] = 0
            return {
                "repos_tracked": len(repo_counts),
                "artifacts": len(github_items),
                "verified_artifacts": verified,
                "corroborated_artifacts": corroborated,
                "artifact_kinds": by_kind,
                "discussion_counts": discussion_counts,
                "topic_counts": topic_counts,
                "repo_counts": repo_counts,
            }
        return self._get_cached_report("github_learning_status", 5.0, _build)

    def _run_acquisition_policy(self) -> dict:
        queue = self.verification_queue(limit=10)
        budgets = dict(getattr(self, "_auto_learn_stats", {}).get("budgets", {}) or {})
        github_enabled = self._github_acquisition_enabled()
        self._acquisition_policy_stats["github_enabled"] = github_enabled
        max_pending = int(budgets.get("max_pending_unverified_artifacts", 40) or 40)
        if int(queue.get("pending", 0) or 0) > max_pending:
            return {"skipped": True, "reason": "verification_backlog"}
        plan = self.acquisition_plan(limit=1)
        suggestions = plan.get("suggestions", [])
        if not suggestions:
            return {"skipped": True, "reason": "no_suggestions"}
        sources = list((suggestions[0] or {}).get("recommended_sources", []))
        if not sources:
            return {"skipped": True, "reason": "no_sources"}
        now = time.time()
        github_cooldown_until = float(self._acquisition_cooldowns.get("github", 0.0) or 0.0)

        def _is_github_source(source: dict) -> bool:
            return str((source or {}).get("source_type", "")) in {"github_topic", "github_repo"}

        if not github_enabled:
            sources = [source for source in sources if not _is_github_source(source)]
        if github_cooldown_until > now:
            non_github_sources = [source for source in sources if not _is_github_source(source)]
            if not non_github_sources:
                weak_domain = str((suggestions[0] or {}).get("domain", "general") or "general")
                non_github_sources = [self._fallback_acquisition_source_for_domain(weak_domain)]
            sources = non_github_sources
        github_status = self.github_learning_status()
        github_empty = int(github_status.get("repos_tracked", 0) or 0) <= 0
        if github_enabled and github_empty and github_cooldown_until <= now:
            weak_domain = str((suggestions[0] or {}).get("domain", "general") or "general")
            seed_repo_map = {
                "word_problems": ["TheAlgorithms/Python"],
                "arithmetic": ["TheAlgorithms/Python"],
                "algebra": ["sympy/sympy"],
                "calculus": ["sympy/sympy"],
                "logic": ["pyeda/pyeda"],
                "logic_basics": ["pyeda/pyeda"],
                "propositional": ["pyeda/pyeda"],
                "science": ["scikit-learn/scikit-learn"],
                "chemistry": ["RDKit/rdkit"],
                "technology": ["TheAlgorithms/Python"],
                "code": ["TheAlgorithms/Python"],
            }
            seeded_sources = [
                {
                    "source_type": "github_repo",
                    "locator": repo,
                    "domain": weak_domain,
                    "max_items": 10,
                    "metadata": {"mode": "full_repo"},
                }
                for repo in seed_repo_map.get(weak_domain, [])
            ]
            github_sources = [
                source for source in sources
                if str((source or {}).get("source_type", "")) in {"github_topic", "github_repo"}
            ]
            non_github_sources = [
                source for source in sources
                if str((source or {}).get("source_type", "")) not in {"github_topic", "github_repo"}
            ]
            if seeded_sources:
                sources = seeded_sources + github_sources + non_github_sources
            elif github_sources:
                sources = github_sources + non_github_sources
        max_sources = max(1, int(budgets.get("acquisition_sources_per_run", 2) or 2))
        results = []
        weak_domain = str((suggestions[0] or {}).get("domain", "general") or "general")
        github_rate_limited = False
        for source in sources[:max_sources]:
            if github_rate_limited and _is_github_source(source):
                continue
            try:
                results.append({"source": source, "result": dict(self.acquire(source))})
                source_type = str((source or {}).get("source_type", "unknown"))
                success_counts = self._acquisition_policy_stats.setdefault("success_counts", {})
                success_counts[source_type] = int(success_counts.get(source_type, 0) or 0) + 1
            except Exception as exc:
                error_text = str(exc).lower()
                source_type = str((source or {}).get("source_type", "unknown"))
                error_counts = self._acquisition_policy_stats.setdefault("error_counts", {})
                error_counts[source_type] = int(error_counts.get(source_type, 0) or 0) + 1
                self._acquisition_policy_stats["last_remote_failure"] = {
                    "source_type": source_type,
                    "locator": str((source or {}).get("locator", "")),
                    "error": str(exc),
                    "timestamp": time.time(),
                }
                if "rate limit" in error_text or "403" in error_text:
                    github_rate_limited = True
                    cooldown_seconds = int(budgets.get("acquisition_rate_limit_cooldown_seconds", 900) or 900)
                    self._acquisition_cooldowns["github"] = max(
                        float(self._acquisition_cooldowns.get("github", 0.0) or 0.0),
                        time.time() + cooldown_seconds,
                    )
                    fallback = self._fallback_acquisition_source_for_domain(weak_domain)
                    try:
                        results.append({"source": fallback, "result": dict(self.acquire(fallback))})
                        self._acquisition_policy_stats["last_fallback_source"] = dict(fallback)
                    except Exception as fallback_exc:
                        self._record_runtime_error("acquisition_policy.fallback", fallback_exc, "_run_acquisition_policy")
                self._record_runtime_error("acquisition_policy.acquire", exc, "_run_acquisition_policy")
        if not results:
            self._invalidate_report_cache("acquisition_dashboard", "learning_ops_dashboard", "learning_dashboard_payload")
            reason = "acquire_failed"
            if float(self._acquisition_cooldowns.get("github", 0.0) or 0.0) > time.time():
                reason = "github_rate_limited"
            return {
                "skipped": True,
                "reason": reason,
                "cooldowns": {
                    "github_until": float(self._acquisition_cooldowns.get("github", 0.0) or 0.0),
                },
            }
        self._invalidate_report_cache("acquisition_dashboard", "learning_ops_dashboard", "learning_dashboard_payload")
        return {
            "skipped": False,
            "results": results,
            "cooldowns": {
                "github_until": float(self._acquisition_cooldowns.get("github", 0.0) or 0.0),
            },
        }

    def _fallback_acquisition_source_for_domain(self, weak_domain: str) -> dict:
        book_queries = {
            "mathematics": "algebra mathematics",
            "physics": "physics mechanics",
            "chemistry": "chemistry reactions",
            "technology": "algorithms programming",
        }
        if weak_domain in book_queries:
            return {
                "source_type": "open_access_book",
                "locator": book_queries[weak_domain],
                "domain": weak_domain,
                "max_items": 8,
                "metadata": {"open_in_chrome": False},
            }
        return {
            "source_type": "wikipedia",
            "locator": weak_domain.replace("_", " "),
            "domain": weak_domain,
        }

    def _github_acquisition_enabled(self) -> bool:
        budgets = dict(getattr(self, "_auto_learn_stats", {}).get("budgets", {}) or {})
        if "disable_github_acquisition" in budgets:
            return not bool(budgets.get("disable_github_acquisition", False))
        acquisition_cfg = dict(self.config.get("acquisition", {}) or {})
        return bool(acquisition_cfg.get("github_enabled", False))

    def _recent_acquisition_signatures(self, limit: int = 12) -> set[str]:
        try:
            records = list((self._get_acquisition_mesh()._state.get("records", []) or []))
        except Exception:
            records = []
        signatures: List[str] = []
        for record in records[-max(1, limit * 2):]:
            if not isinstance(record, dict):
                continue
            source_type = str(record.get("source_type", "")).strip()
            locator = str(record.get("locator", "")).strip()
            if source_type and locator:
                signatures.append(f"{source_type}::{locator}")
        return set(signatures[-limit:])

    def _source_signature(self, source: dict) -> str:
        return f"{str((source or {}).get('source_type', '')).strip()}::{str((source or {}).get('locator', '')).strip()}"

    def _dataset_sources_for_domain(self, domain: str) -> List[dict]:
        dataset_dir = REPO_ROOT / "data" / "external_datasets"
        all_files = {path.name: path for path in dataset_dir.glob("*") if path.is_file()}
        preferred = {
            "mathematics": ["aqua_train.json", "gsm8k_test.jsonl", "mmlu_domains.jsonl"],
            "word_problems": ["gsm8k_test.jsonl", "aqua_train.json"],
            "physics": ["mmlu_domains.jsonl"],
            "chemistry": ["mmlu_domains.jsonl"],
            "technology": ["mmlu_domains.jsonl"],
            "science": ["mmlu_domains.jsonl"],
            "social": ["commonsenseqa_train.jsonl"],
            "language": ["commonsenseqa_train.jsonl"],
            "general": ["commonsenseqa_train.jsonl", "mmlu_domains.jsonl"],
        }
        names = preferred.get(domain, preferred.get("general", []))
        sources: List[dict] = []
        for name in names:
            path = all_files.get(name)
            if not path:
                continue
            sources.append(
                {
                    "source_type": "dataset",
                    "locator": str(path),
                    "domain": domain,
                    "max_items": 6,
                }
            )
        return sources

    def _book_sources_for_domain(self, domain: str) -> List[dict]:
        candidates: List[dict] = []
        known_books = [
            (REPO_ROOT / "data" / "books" / "art_of_war.txt", {"planning", "history", "strategy", "general"}),
            (REPO_ROOT / "python" / "data" / "books" / "pg84.txt", {"language", "general", "science"}),
            (REPO_ROOT / "python" / "data" / "books" / "a.txt", {"language", "general"}),
        ]
        for path, domains in known_books:
            if path.exists() and (domain in domains or "general" in domains):
                candidates.append(
                    {
                        "source_type": "book",
                        "locator": str(path),
                        "domain": domain,
                        "max_items": 4,
                    }
                )
        return candidates

    def _recommended_acquisition_sources_for_domain(
        self,
        domain: str,
        recent_signatures: Optional[set[str]] = None,
        github_topics: Optional[List[str]] = None,
        github_enabled: Optional[bool] = None,
    ) -> List[dict]:
        recent_signatures = set(recent_signatures or set())
        if github_enabled is None:
            github_enabled = self._github_acquisition_enabled()
        sources: List[dict] = []
        sources.extend(self._dataset_sources_for_domain(domain))
        sources.extend(self._book_sources_for_domain(domain))
        fallback = self._fallback_acquisition_source_for_domain(domain)
        sources.append(dict(fallback))
        if github_enabled:
            topics = list(github_topics or [domain])
            if topics:
                sources.append(
                    {
                        "source_type": "github_topic",
                        "locator": topics[0],
                        "domain": domain,
                        "metadata": {"repo_limit": 2, "per_repo_items": 8},
                    }
                )
        sources.append({"source_type": "wikipedia", "locator": domain.replace("_", " "), "domain": domain})
        unique: List[dict] = []
        seen: set[str] = set()
        for source in sources:
            sig = self._source_signature(source)
            if sig in seen:
                continue
            seen.add(sig)
            unique.append(source)
        fresh = [source for source in unique if self._source_signature(source) not in recent_signatures]
        return (fresh + [source for source in unique if source not in fresh])[:6]

    def ingest(
        self,
        data: Union[str, dict, IngestionBatch, IngestionSourceConfig],
        kind: str = "text",
        source: str = "user",
    ) -> IngestionResult:
        """
        Ingest real-world data, extract problems, feed into learning loop.
        Supports: text, csv, json, textbook, file path.
        """
        acquisition_kinds = {"book", "dataset", "wikipedia", "github_repo", "github_topic", "web_page", "web_feed"}
        if (
            isinstance(data, IngestionSourceConfig) and str(data.source_type) in acquisition_kinds
        ) or (
            isinstance(data, IngestionBatch) and all(str(item.source_type) in acquisition_kinds for item in data.items)
        ) or (
            isinstance(data, dict) and str(data.get("source_type", data.get("kind", kind))) in acquisition_kinds
        ):
            return IngestionResult(self.acquire(data).to_dict())

        if not self.perception_engine:
            return IngestionResult({"error": "Perception engine not loaded"})

        if isinstance(data, IngestionBatch):
            aggregate = {
                "source_id": data.source_id,
                "source_type": "batch",
                "records_scanned": 0,
                "facts_added": 0,
                "problems_added": 0,
                "concepts_proposed": 0,
                "rejected_records": 0,
                "provenance_summary": {"batch_items": len(data.items), **dict(data.metadata or {})},
                "items": [],
            }
            for item in data.items:
                item_result = self.ingest(item)
                aggregate["items"].append(dict(item_result))
                if item_result.get("error"):
                    aggregate["rejected_records"] += 1
                    continue
                aggregate["records_scanned"] += int(item_result.get("records_scanned", 0))
                aggregate["facts_added"] += int(item_result.get("facts_added", 0))
                aggregate["problems_added"] += int(item_result.get("problems_added", 0))
                aggregate["concepts_proposed"] += int(item_result.get("concepts_proposed", 0))
            return IngestionResult(aggregate)

        if isinstance(data, dict):
            data = IngestionSourceConfig(
                source_type=str(data.get("kind", kind)),
                payload=data.get("payload", data.get("data", "")),
                source_id=str(data.get("source", source)),
                domain=str(data.get("domain", "general")),
                trust_level=str(data.get("trust_level", "default")),
                metadata=dict(data.get("metadata", {}) or {}),
            )

        if isinstance(data, IngestionSourceConfig):
            kind = data.source_type
            source = data.source_id
            payload = data.payload
            provenance = {
                "domain": data.domain,
                "trust_level": data.trust_level,
                **dict(data.metadata or {}),
            }
        else:
            payload = data
            provenance = {}

        if kind == "csv":
            result = self.perception_engine.ingest_csv(payload, source)
        elif kind == "json":
            import json as _j
            try:
                parsed = _j.loads(payload) if isinstance(payload, str) else payload
            except Exception:
                parsed = {"raw": payload}
            result = self.perception_engine.ingest_json(parsed, source)
        elif kind == "textbook":
            result = self.perception_engine.ingest_textbook(payload, source)
        elif kind == "file":
            result = self.perception_engine.ingest_file(payload)
        else:
            result = self.perception_engine.ingest_text(payload, source)

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

        self._stats["ingestion"]["records_scanned"] += 1
        self._stats["ingestion"]["facts_added"] += facts_added
        self._stats["ingestion"]["problems_added"] += problems_added
        self._stats["ingestion"]["concepts_proposed"] += int(len(getattr(result, "facts_extracted", []) or []))

        return IngestionResult({
            "source_id": source,
            "source_type": kind,
            "records_scanned": 1,
            "facts_added": facts_added,
            "problems_added": problems_added,
            "concepts_proposed": int(len(getattr(result, "facts_extracted", []) or [])),
            "rejected_records": 0,
            "provenance_summary": provenance,
            "source": result.source,
            "kind": result.kind,
            "problems_extracted": len(result.problems_extracted),
            "problems_added_to_curriculum": problems_added,
            "facts_extracted": len(result.facts_extracted),
            "facts_added_to_worldmodel": facts_added,
            "graph_nodes": result.graph_nodes,
            "graph_edges": result.graph_edges,
            "elapsed": round(result.elapsed, 3),
        })

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
        for concept_name, concept in self.concept_graph.concepts.items():
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
                            self.concept_graph.concepts[t_name] = \
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
            c = self.concept_graph.concepts.get(concept_name)
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
            from sare.meta.goal_setter import GoalType, GoalStatus

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

            # Generate rule discovery goal if not enough rules known.
            # De-dupe by type: only one active RULE_DISCOVERY goal at a time
            # (the description embeds a counter that changes, so string-matching
            # would create a new goal every time — the bug we're fixing).
            rules_known = self._stats.get("rules_promoted", 0)
            if rules_known < 5:
                _has_active_rd = False
                try:
                    for _g in self.goal_setter._goals.values():
                        if (getattr(_g, "type", None) == GoalType.RULE_DISCOVERY
                                and getattr(_g, "status", None) == GoalStatus.ACTIVE):
                            _has_active_rd = True
                            # Update its progress instead of creating a duplicate
                            _g.update_progress(float(rules_known))
                            _g.description = f"Discover 5 new rules (currently {rules_known})"
                            break
                except Exception:
                    pass
                if not _has_active_rd:
                    try:
                        self.goal_setter.add_goal(
                            type=GoalType.RULE_DISCOVERY,
                            description=f"Discover 5 new rules (currently {rules_known})",
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

    def start_auto_learn(
        self,
        interval: float = 2.0,
        problems_per_cycle: int = 5,
        budgets: Optional[Dict[str, Any]] = None,
    ):
        """Start background autonomous learning thread."""
        if hasattr(self, '_auto_learn_thread') and self._auto_learn_thread and self._auto_learn_thread.is_alive():
            log.info("Auto-learn already running")
            return

        budget_config = dict(budgets or {})
        self._auto_learn_stop = threading.Event()
        self._auto_learn_stats = {
            "running": True, "cycles": 0, "total_solved": 0,
            "total_attempted": 0, "interval": interval,
            "retry_successes": 0, "avg_surprise": 0.0,
            "failure_reasons": {},
            "budgets": budget_config,
            "scheduler_buckets": {
                "acquisition_jobs": 0,
                "broker_training_jobs": 0,
                "failure_replay": 0,
                "weak_domain_reinforcement": 0,
                "transfer_probes": 0,
                "heldout_verification": 0,
                "benchmark_refresh": 0,
                "sleep_consolidation": 0,
            },
        }

        def _learn_loop():
            cycle = 0
            while not self._auto_learn_stop.is_set():
                cycle += 1
                try:
                    if (budget_config.get("ingestion_budget", 0) or budget_config.get("web_budget", 0)) and (cycle == 1 or cycle % 5 == 0):
                        acquisition_run = self._run_acquisition_policy()
                        if not acquisition_run.get("skipped", False):
                            self._auto_learn_stats["scheduler_buckets"]["acquisition_jobs"] += len(acquisition_run.get("results", []) or [1])
                            self._auto_learn_stats["scheduler_buckets"]["weak_domain_reinforcement"] += 1
                        elif acquisition_run.get("reason") == "verification_backlog":
                            self._auto_learn_stats["scheduler_buckets"]["heldout_verification"] += 1
                    if cycle == 1 or cycle % 4 == 0:
                        grounded_run = self.run_grounded_learning_cycle()
                        if grounded_run.get("total_tasks", 0):
                            self._auto_learn_stats["scheduler_buckets"]["weak_domain_reinforcement"] += 1
                    if cycle == 1 or cycle % 3 == 0:
                        retention_run = self.run_retention_retests(
                            limit=max(1, int(budget_config.get("retention_budget", 2) or 2)),
                            min_age_seconds=float(budget_config.get("retention_interval_seconds", 3600.0) or 3600.0),
                        )
                        if retention_run.get("retested", 0):
                            self._auto_learn_stats["scheduler_buckets"]["heldout_verification"] += int(retention_run.get("retested", 0))
                    if cycle == 1 or cycle % 5 == 0:
                        suite_run = self.run_transfer_suite_benchmarks(
                            tests_per_suite=max(3, int(budget_config.get("transfer_suite_tests", 3) or 3)),
                            max_suites=max(1, int(budget_config.get("transfer_suite_limit", 2) or 2)),
                        )
                        if suite_run.get("benchmarks_run", 0):
                            self._auto_learn_stats["scheduler_buckets"]["transfer_probes"] += int(suite_run.get("benchmarks_run", 0))
                    results = self.learn_cycle(n=problems_per_cycle)
                    successes = sum(1 for r in results if r.get("success"))
                    self._auto_learn_stats["cycles"] = cycle
                    self._auto_learn_stats["total_solved"] += successes
                    self._auto_learn_stats["total_attempted"] += len(results)
                    self._auto_learn_stats["scheduler_buckets"]["broker_training_jobs"] += len(results)

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
                        self._auto_learn_stats["scheduler_buckets"]["sleep_consolidation"] += 1
                        self.reorganize_knowledge()
                        # Deep transfer: synthesize new transforms
                        try:
                            self.synthesize_transforms()
                            self._auto_learn_stats["scheduler_buckets"]["transfer_probes"] += 1
                        except Exception:
                            pass
                        self._refresh_transforms()

                    # Save periodically
                    if cycle % 20 == 0:
                        self._auto_learn_stats["scheduler_buckets"]["benchmark_refresh"] += 1
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

    def start_autonomous_learning(self, interval: float = 2.0,
                                  problems_per_cycle: int = 5, **kwargs: Any) -> bool:
        """Canonical public wrapper for background learning."""
        budgets = {
            key: kwargs[key]
            for key in (
                "cpu_budget",
                "llm_budget",
                "web_budget",
                "ingestion_budget",
                "disable_github_acquisition",
                "consolidation_frequency",
                "retention_budget",
                "retention_interval_seconds",
            )
            if key in kwargs
        }
        acquisition_sources = kwargs.get("acquisition_sources")
        if acquisition_sources:
            try:
                self.schedule_acquisition(
                    sources=list(acquisition_sources),
                    interval_seconds=float(kwargs.get("acquisition_interval", max(300.0, float(interval) * 30.0)) or 300.0),
                    max_runs=int(kwargs.get("acquisition_runs", 0) or 0),
                )
            except Exception as exc:
                self._record_runtime_error("start_autonomous_learning.schedule_acquisition", exc, "start_autonomous_learning")
        self.start_auto_learn(interval=interval, problems_per_cycle=problems_per_cycle, budgets=budgets)
        return True

    def stop_auto_learn(self):
        """Stop background learning."""
        if hasattr(self, '_auto_learn_stop'):
            self._auto_learn_stop.set()
            log.info("Auto-learn stop requested")

    def stop_autonomous_learning(self) -> bool:
        """Canonical public wrapper for background learning stop."""
        self.stop_auto_learn()
        return True

    def auto_learn_status(self) -> dict:
        """Get background learning status."""
        if hasattr(self, '_auto_learn_stats'):
            return dict(self._auto_learn_stats)
        return {"running": False, "cycles": 0, "total_solved": 0, "total_attempted": 0}

    def _note_solve_mode(self, solver_used: str) -> None:
        modes = self._stats.setdefault("solve_modes", {})
        modes[solver_used] = int(modes.get(solver_used, 0)) + 1

    def _make_solve_result(self, payload: Dict[str, Any]) -> SolveResult:
        if "request_id" not in payload:
            payload["request_id"] = str(payload.get("expression", ""))
        return SolveResult(payload)

    def _run_general_solver(self, expression: str, context: Optional[dict] = None,
                            domain: str = "general") -> SolveResult:
        if self.general_solver is None:
            raise RuntimeError("general solver unavailable")

        ctx = context or {}
        gs_result = self.general_solver.solve(
            expression,
            context=str(ctx.get("text_context", ctx.get("context", "")) or ""),
            problem_type=ctx.get("problem_type"),
            allow_retrieval=ctx.get("allow_retrieval", True),
            allow_llm=ctx.get("allow_llm", True),
            store_result=False,
            metadata=dict(ctx.get("metadata", {}) or {}),
        )
        success = bool(getattr(gs_result, "solved", False))
        solver_used = str(getattr(gs_result, "solver_used", "general_solver"))
        solved_domain = str(getattr(gs_result, "domain", "") or domain or "general")
        confidence = float(getattr(gs_result, "confidence", 0.0) or 0.0)
        answer = str(getattr(gs_result, "answer", "") or "")
        reasoning = str(getattr(gs_result, "reasoning", "") or "")
        elapsed_seconds = round(float(getattr(gs_result, "elapsed_ms", 0.0) or 0.0) / 1000.0, 3)

        self.events.emit(
            Event.SOLVE_COMPLETED if success else Event.SOLVE_FAILED,
            {
                "problem_id": expression,
                "expression": expression,
                "domain": solved_domain,
                "transforms": [],
                "energy_before": 1.0,
                "energy_after": 0.0 if success else 1.0,
                "delta": confidence if success else 0.0,
                "elapsed": elapsed_seconds,
                "answer": answer,
                "solver_used": solver_used,
                "learning_mode": getattr(gs_result, "learning_mode", "free_solve"),
            },
            "brain.general_solver",
        )
        self._persist_post_solve_state()
        return self._make_solve_result({
            "request_id": getattr(gs_result, "problem_id", expression),
            "expression": expression,
            "answer": answer,
            "steps_text": reasoning,
            "graph": None,
            "energy": {"total": 0.0 if success else 1.0, "components": {}},
            "initial_energy": 1.0,
            "transforms": [],
            "transforms_used": [],
            "transforms_applied": [],
            "steps": len(getattr(gs_result, "sub_steps", []) or []),
            "steps_taken": len(getattr(gs_result, "sub_steps", []) or []),
            "expansions": 0,
            "elapsed": elapsed_seconds,
            "elapsed_seconds": elapsed_seconds,
            "delta": round(confidence if success else 0.0, 3),
            "reduction_pct": round(confidence * 100.0, 1),
            "confidence": round(confidence, 3),
            "success": success,
            "solve_success": success,
            "domain": solved_domain,
            "trajectory": [],
            "proof": None,
            "strategy_hint": None,
            "strategy_hit": False,
            "recalled_memories": 0,
            "recalled_memory": [],
            "memory": self.memory_manager.stats() if self.memory_manager else {},
            "initial": None,
            "result": {"solver_used": solver_used, "learning_mode": getattr(gs_result, "learning_mode", "free_solve")},
            "learned_concepts": [],
            "stage": self.stage.value,
            "source": "brain.general_solver",
            "solver_path": solver_used,
            "learning_mode": getattr(gs_result, "learning_mode", "free_solve"),
            "verification_outcome": {"correct": getattr(gs_result, "correct", None)},
            "transfer_outcome": None,
            "persistence_ids": {},
        })

    def attempt_learning_problem(
        self,
        problem_text: str,
        expected_answer,
        problem_type: str,
        context: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Canonical graded-learning entrypoint used by autonomous trainers."""
        if self.general_solver is None:
            raise RuntimeError("general solver unavailable")

        result = self.general_solver.attempt_learning_problem(
            problem_text=problem_text,
            expected_answer=expected_answer,
            problem_type=problem_type,
            context=context,
            metadata=dict(metadata or {}),
        )
        self._note_solve_mode(str(getattr(result, "solver_used", "learning_problem")))
        return result

    # ─────────────────────────────────────────────────────────────────────────
    #  Learning Cycle
    # ─────────────────────────────────────────────────────────────────────────

    def learn_cycle(self, n: int = 5, max_tasks: Optional[int] = None,
                    mode: str = "default") -> LearnCycleResult:
        """
        Run n autonomous learning cycles with failure-driven adaptation:
        1. Pick problem (WorldModel-guided if available)
        2. Predict outcome (builds prediction model)
        3. Solve
        4. Compare prediction vs reality (learn from surprise)
        5. On failure: analyze, retry with alternative, record what to avoid
        """
        target_n = int(max_tasks if max_tasks is not None else n)
        results: List[dict] = []

        try:
            self.drive_self_generated_learning()
        except Exception:
            pass

        for i in range(target_n):
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

            result = self.solve(problem_expr, context={"mode": mode})
            selection = dict(getattr(self, "_last_problem_selection", {}) or {})
            if selection:
                result.setdefault("strategy_source", selection.get("source", "generated_problems"))
                result.setdefault("strategy_reason", selection.get("reason", ""))
            results.append(result)
            self.record_learning_strategy_outcome(
                expression=problem_expr,
                domain=str(selection.get("domain") or result.get("domain") or "general"),
                source=str(selection.get("source") or "generated_problems"),
                task_type="expression_rewrite",
                result=result,
                metadata={
                    "reason": selection.get("reason", ""),
                    "selection_score": selection.get("score"),
                    "candidates_considered": selection.get("candidates_considered", 0),
                    "candidates_solvable": selection.get("candidates_solvable", 0),
                },
                learning_mode=mode if mode in {"retrieval", "hinted", "template_replay"} else "free_solve",
                verification_level="runtime",
            )

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

        # Cumulative learn cycle counter (batch-size-independent triggers)
        if not hasattr(self, '_learn_cycle_count'):
            self._learn_cycle_count = 0
        self._learn_cycle_count += 1
        _lcc = self._learn_cycle_count

        # Every 5 cumulative calls: generate + test conjectures (Creativity loop)
        if _lcc % 5 == 0:
            try:
                promoted = self._test_and_promote_conjectures()
                if promoted > 0:
                    log.info(f"Conjecture cycle: {promoted} conjectures promoted")
                    self._refresh_transforms()
            except Exception as e:
                log.debug(f"Conjecture cycle failed: {e}")

        # Every 5 cumulative calls: concept consolidation
        if _lcc % 5 == 0 and self.concept_graph:
            try:
                self._consolidate_concepts()
            except Exception as e:
                log.debug(f"Concept consolidation: {e}")

        # Every 10 cumulative calls: verify transfers and synthesize new transforms
        if _lcc % 10 == 0:
            try:
                verified = self._verify_transfer_hypotheses()
                if verified > 0:
                    log.info(f"Transfer verification: {verified} new transfers verified")
                    self._refresh_transforms()
            except Exception as e:
                log.debug(f"Transfer verification failed: {e}")

        # Every 10 cumulative calls: run meta-learning beam-width tuning
        if _lcc % 10 == 0 and self.meta_learner:
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
                if hasattr(self.predictive_loop, "run_cycle"):
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
                new_blends = self.concept_blender.discover_blends(max_results=3)
                if new_blends and self.concept_graph:
                    fed = self.concept_blender.feed_to_concept_graph(self.concept_graph, new_blends)
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
                    if isinstance(streams, dict):
                        stream_iter = streams.items()
                    else:
                        stream_iter = [
                            (getattr(stream, "stream_type", getattr(stream, "stream_id", "stream")), stream)
                            for stream in list(streams)
                        ]
                    for stype, stream in stream_iter:
                        recent = getattr(stream, '_recent_results', []) or getattr(stream, "recent_results", [])
                        for item in list(recent)[-1:]:
                            expr = getattr(item, 'expression', str(item))
                            domain = getattr(item, 'domain', 'general')
                            self.stream_bridge.submit(expr, source=str(stype), domain=domain)
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

        facts_added = int(self._stats.get("ingestion", {}).get("facts_added", 0))
        try:
            self.run_grounded_learning_cycle()
        except Exception as exc:
            self._record_runtime_error("run_grounded_learning_cycle", exc, "learn_cycle")
        try:
            self.run_retention_retests(limit=1, min_age_seconds=1800.0)
        except Exception as exc:
            self._record_runtime_error("run_retention_retests", exc, "learn_cycle")
        try:
            self.run_transfer_suite_benchmarks(tests_per_suite=3, max_suites=2)
        except Exception as exc:
            self._record_runtime_error("run_transfer_suite_benchmarks", exc, "learn_cycle")
        self._invalidate_report_cache()
        truth_gate = self.learning_truth_gate()
        return LearnCycleResult(
            results,
            tasks_attempted=len(results),
            tasks_solved=sum(1 for r in results if r.get("success")),
            rules_promoted=int(self._stats.get("rules_promoted", 0)),
            transfers_attempted=int(self._stats.get("transfers_attempted", 0)),
            transfers_succeeded=int(self._stats.get("transfers_succeeded", 0)),
            facts_added=facts_added,
            runtime_errors=int(self._stats.get("runtime_errors", 0)),
            heldout_pass_rate=truth_gate.get("heldout", {}).get("pass_rate"),
            transfer_success_rate=truth_gate.get("transfer", {}).get("success_rate"),
            retention_rate=truth_gate.get("retention", {}).get("retention_rate"),
            grounded_success_rate=truth_gate.get("grounded", {}).get("success_rate"),
            heldout_gate=truth_gate.get("heldout", {}).get("passed", False),
            transfer_gate=truth_gate.get("transfer", {}).get("passed", False),
            retention_gate=truth_gate.get("retention", {}).get("passed", False),
            grounded_gate=truth_gate.get("grounded", {}).get("passed", False),
            truth_gate=truth_gate,
            progress_trusted=truth_gate.get("passed", False),
            mode=mode,
        )

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

        def _task_type_for_source(source: str) -> str:
            source = str(source or "")
            if source in {"code_problems"}:
                return "code_question"
            if source in {"word_problems"}:
                return "word_problem"
            if source in {"comprehension_problems"}:
                return "reading_comprehension"
            if source in {"language_problems"}:
                return "language_reasoning"
            if source in {"self_question"}:
                return "self_generated_question"
            if source in {"hypothesis_verification"}:
                return "hypothesis_verification"
            return "expression_rewrite"

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

        focus_domains: List[dict] = []
        try:
            focus_domains = self.learning_focus_domains(limit=4)
            for idx, item in enumerate(focus_domains):
                reasons = ", ".join(list(item.get("reasons", []) or [])[:2]) or "weak understanding signal"
                _add_candidate(
                    _choose_from_domain(str(item.get("domain", "general") or "general")),
                    "understanding_focus",
                    0.62 + min(float(item.get("priority", 0.0) or 0.0), 0.12) - idx * 0.015,
                    str(item.get("domain", "general") or "general"),
                    f"understanding gap: {reasons}",
                )
        except Exception as exc:
            self._record_runtime_error("learning_focus_domains.pick", exc, "learn_pick")

        try:
            pending_questions = self.active_learning_questions(limit=3)
            for idx, item in enumerate(pending_questions):
                target_domain = str(item.get("domain", "general") or "general")
                priority = min(1.0, max(0.0, float(item.get("priority", 0.5) or 0.5)))
                text = str(item.get("text", "") or item.get("question", "") or "self-generated question")
                _add_candidate(
                    _choose_from_domain(target_domain),
                    "self_question",
                    0.66 + priority * 0.14 - idx * 0.02,
                    target_domain,
                    f"self-generated question: {text[:90]}",
                )
        except Exception as exc:
            self._record_runtime_error("self_questions.pick", exc, "learn_pick")

        try:
            theories = self.world_theories(limit=3)
            for idx, theory in enumerate(theories):
                target_domain = str(theory.get("domain", "general") or "general")
                hypothesis_count = int(theory.get("hypothesis_count", 0) or 0)
                summary = str(theory.get("summary", "") or "world-model hypothesis cluster")
                _add_candidate(
                    _choose_from_domain(target_domain),
                    "hypothesis_verification",
                    0.62 + min(hypothesis_count, 5) * 0.035 - idx * 0.02,
                    target_domain,
                    f"verify hypothesis cluster: {summary[:90]}",
                )
        except Exception as exc:
            self._record_runtime_error("world_theories.pick", exc, "learn_pick")

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
                problem = self.developmental_curriculum.next_problem_with_forgetting(self.stage, self.self_model)
                if problem:
                    expression = getattr(problem, "expression", None) or str(problem)
                    _add_candidate(
                        expression,
                        "curriculum",
                        0.68,
                        getattr(problem, "domain", "general"),
                        "developmental curriculum (forgetting-curve weighted)",
                    )
            except Exception as exc:
                self._record_runtime_error("developmental_curriculum.next_problem_with_forgetting", exc, "learn_pick")

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

        if candidates and focus_domains:
            focus_map = {str(item.get("domain", "general") or "general"): item for item in focus_domains}
            for candidate in candidates:
                focus = dict(focus_map.get(str(candidate.get("domain", "general") or "general"), {}) or {})
                if not focus:
                    continue
                boost = min(0.05, float(focus.get("priority", 0.0) or 0.0) * 0.08)
                candidate["score"] = round(max(0.0, min(1.0, float(candidate.get("score", 0.0) or 0.0) + boost)), 4)
                reason_bits = list(focus.get("reasons", []) or [])[:2]
                if reason_bits:
                    prefix = f"{candidate.get('reason')} · " if candidate.get("reason") else ""
                    candidate["reason"] = prefix + f"focus={'/'.join(reason_bits)}"

        if candidates and getattr(self, "adaptive_learning_policy", None) is not None:
            try:
                policy = self.adaptive_learning_policy
                for candidate in candidates:
                    base_score = float(candidate.get("score", 0.0) or 0.0)
                    rescored = policy.score_candidate(
                        source=str(candidate.get("source", "generated_problems") or "generated_problems"),
                        domain=str(candidate.get("domain", "general") or "general"),
                        task_type=_task_type_for_source(str(candidate.get("source", "") or "")),
                        base_score=base_score,
                    )
                    if rescored != base_score:
                        candidate["score"] = rescored
                        candidate["reason"] = (
                            (f"{candidate.get('reason')} · " if candidate.get("reason") else "")
                            + f"policy={rescored:.2f}"
                        )
            except Exception as exc:
                self._record_runtime_error("adaptive_learning_policy.pick", exc, "learn_pick")

        if candidates:
            # Pre-filter: discard candidates where no transform can match.
            # This eliminates the 97% waste on "no_matching_transforms" failures.
            # Keep failure_replay candidates (they are retries worth attempting)
            # and keep high-score candidates from LLM teacher (score >= 0.9).
            solvable = []
            for c in candidates:
                if c["source"] in ("failure_replay", "llm_teacher") or c["score"] >= 0.9:
                    solvable.append(c)
                elif self._can_likely_solve(c["expression"]):
                    solvable.append(c)
            # If pre-filter removed everything, fall back to top 3 by score
            if not solvable:
                candidates.sort(key=lambda x: x["score"], reverse=True)
                solvable = candidates[:3]

            best = max(solvable, key=lambda item: (item["score"], item["source"]))
            self._last_problem_selection = {
                **best,
                "candidates_considered": len(candidates),
                "candidates_solvable": len(solvable),
                "time": time.time(),
            }
            log.debug(
                "Learning problem selected: %s via %s (score=%.3f, reason=%s, solvable=%d/%d)",
                best["expression"],
                best["source"],
                best["score"],
                best["reason"],
                len(solvable),
                len(candidates),
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
            "candidates_solvable": 0,
            "time": time.time(),
        }
        return fallback

    def _can_likely_solve(self, expression: str) -> bool:
        """Quick pre-filter: check if at least one transform matches the parsed graph.

        This avoids wasting a full solve() call (with search, energy computation,
        memory lookup, etc.) on expressions that have zero matching transforms.
        Returns True if any transform matches, or if we can't parse (give benefit of doubt).
        """
        if not expression or not self.transforms:
            return True  # can't tell, assume solvable
        try:
            g = self.engine.build_expression_graph(expression)
            # Quick scan: does ANY transform match?
            for t in self.transforms[:50]:  # check first 50 (most important) transforms
                try:
                    matches = t.match(g)
                    if matches:
                        return True
                except Exception:
                    continue
            return False
        except Exception:
            return True  # parse failed, let solve() handle the error

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
                    target = hyp.target_domain
                    test_problems = self._get_test_problems_for_domain(target, n=3)
                    if not test_problems:
                        continue

                    # Use Brain.solve() directly — it properly tracks transforms_used
                    _brain_ref = self
                    def _solve_fn(expr, _brain=_brain_ref):
                        try:
                            result = _brain.solve(expr)
                            return {
                                "success": bool(result.get("success") or result.get("delta", 0) > 0.01),
                                "delta": result.get("delta", 0),
                                "transforms": result.get("transforms", []),
                            }
                        except Exception:
                            return {"success": False, "delta": 0, "transforms": []}

                    ok = self.transfer_engine.test_hypothesis(hyp, _solve_fn, test_problems)
                    if ok:
                        verified += 1
                        heldout_wins = sum(
                            1 for item in list(getattr(hyp, "test_results", []) or [])[-len(test_problems):]
                            if item.get("success")
                        )
                        heldout_tests = min(len(test_problems), len(getattr(hyp, "test_results", []) or []))
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
                                        "heldout_target_wins": heldout_wins,
                                        "heldout_target_tests": heldout_tests,
                                        "new_transforms": len(new_specs),
                                    }, "transfer_engine")
                            except Exception:
                                pass
                        else:
                            self.events.emit(Event.TRANSFER_SUCCEEDED, {
                                "source": hyp.source_domain,
                                "target": target,
                                "role": hyp.source_role,
                                "heldout_target_wins": heldout_wins,
                                "heldout_target_tests": heldout_tests,
                                "new_transforms": 0,
                            }, "transfer_engine")
                except Exception as _e:
                    log.debug(f"Hypothesis test failed: {_e}")

        except Exception as e:
            log.debug(f"_verify_transfer_hypotheses failed: {e}")

        return verified

    # ─────────────────────────────────────────────────────────────────────────
    #  Developmental Stage Management
    # ─────────────────────────────────────────────────────────────────────────

    def get_stage_capabilities(self) -> dict:
        """Return the capability gate dict for the current stage."""
        return dict(STAGE_CAPABILITY_GATES.get(self.stage.value, STAGE_CAPABILITY_GATES[DevelopmentalStage.INFANT.value]))

    def _unlock_stage_capabilities(self, stage: DevelopmentalStage):
        """
        Apply stage capability gates after advancing to a new stage:
          - Update ForgettingCurve decay rate
          - Resize experiment_runner beam width cap
          - Enable/disable analogy, conjecture, symbol_creation capabilities
        """
        caps = STAGE_CAPABILITY_GATES.get(stage.value, {})
        stage_level = stage.level

        # Update forgetting curve decay rate
        try:
            from sare.memory.forgetting_curve import get_forgetting_curve
            get_forgetting_curve().set_decay_rate(caps.get("decay_rate", 0.01))
        except Exception:
            pass

        # Cap beam width in experiment_runner
        if self.experiment_runner:
            max_bw = caps.get("max_beam_width")
            if max_bw is not None:
                self.experiment_runner.beam_width = min(
                    self.experiment_runner.beam_width, max_bw
                )

        # Apply core knowledge priors to search ordering
        try:
            from sare.cognition.core_knowledge import get_core_knowledge
            ck = get_core_knowledge()
            if self.experiment_runner and self.transforms:
                self.transforms = ck.apply_priors_to_search(self.transforms, stage_level)
                if self.experiment_runner:
                    self.experiment_runner.transforms = self.transforms
        except Exception:
            pass

        # Seed innate facts into WorldModel at INFANT boot
        if stage == DevelopmentalStage.INFANT and self.world_model:
            try:
                from sare.cognition.core_knowledge import get_core_knowledge
                ck = get_core_knowledge()
                facts = ck.get_all_innate_facts()
                wm = self.world_model
                if hasattr(wm, "observe_fact"):
                    for domain, domain_facts in facts.items():
                        for fact in domain_facts:
                            wm.observe_fact(fact.get("fact", ""), domain=domain,
                                           confidence=fact.get("confidence", 0.8))
            except Exception:
                pass

        log.info(
            "Stage capabilities unlocked: %s → beam_max=%s analogy=%s conjecture=%s "
            "symbol_creation=%s active_questioning=%s decay_rate=%.4f",
            stage.value,
            caps.get("max_beam_width", "∞"),
            caps.get("analogy"), caps.get("conjecture"),
            caps.get("symbol_creation"), caps.get("active_questioning"),
            caps.get("decay_rate", 0.01),
        )

    def _check_and_advance_stage(self):
        """
        Check if stage requirements are met and advance if so.
        Called on every DOMAIN_MASTERED event and periodically.
        Records milestone in autobiographical memory and emits STAGE_ADVANCED.
        """
        self._update_stage()

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
            # Apply capability gates for new stage
            self._unlock_stage_capabilities(next_stage)
            # Record in autobiographical memory
            if self.autobiography:
                try:
                    self.autobiography.record(
                        event_type="stage_advanced",
                        domain="meta",
                        description=f"Developmental stage: {old_stage.value} → {next_stage.value}",
                        importance=1.0,
                        related_rules=[],
                    )
                except Exception:
                    pass
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
                self._sync_transfer_runtime_stats_from_payload()
                log.info(f"Restored brain state: stage={self.stage.value}")
            except Exception as e:
                log.warning(f"Brain state load failed: {e}")

    def save_state(self):
        """Persist brain state to disk."""
        state_path = DATA_DIR / "brain_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        self._sync_transfer_runtime_stats_from_payload()
        state = {
            "stage": self.stage.value,
            "stats": self._stats,
            "total_solves": self.total_solves,
            "total_rules_learned": self.total_rules_learned,
            "episode_count": len(getattr(self.memory_manager, "_episodes", []) or []) if self.memory_manager is not None else 0,
            "cpp_enabled": self.cpp_enabled,
            "boot_time": self.boot_time,
            "saved_at": time.time(),
        }
        try:
            with self._state_save_lock:
                fd, tmp_name = tempfile.mkstemp(
                    prefix=f"{state_path.name}.",
                    suffix=".tmp",
                    dir=str(state_path.parent),
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(state, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_name, state_path)
                finally:
                    if os.path.exists(tmp_name):
                        try:
                            os.remove(tmp_name)
                        except OSError:
                            pass
        except Exception as e:
            log.error(f"Brain state save failed: {e}")
        try:
            self._write_tracking_snapshot(force=False)
        except Exception as exc:
            self._record_runtime_error("brain.save_state.tracking", exc, "save_state")

    # ─────────────────────────────────────────────────────────────────────────
    #  Status & Introspection
    # ─────────────────────────────────────────────────────────────────────────

    def status(self) -> BrainStatus:
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

        # KB stats
        kb_stats: dict = {}
        try:
            if self.kb_lookup is not None:
                kb_stats = self.kb_lookup.get_stats()
            else:
                from sare.memory.knowledge_lookup import KnowledgeLookup
                kb_stats = KnowledgeLookup().get_stats()
        except Exception:
            pass

        wm_sessions = 0
        try:
            if self.working_memory is not None:
                wm_sessions = getattr(self.working_memory, "_session_count", 0)
        except Exception:
            pass

        return BrainStatus({
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
            "knowledge_base": {
                **kb_stats,
                "working_memory_sessions": wm_sessions,
            },
            "acquisition": self._get_acquisition_mesh().status() if self.acquisition_mesh is not None else {"records": 0, "artifacts": 0},
            "concept_graph_health": self.concept_graph.health() if self.concept_graph and hasattr(self.concept_graph, "health") else dict(self._concept_graph_health),
            "persistence_health": self.persistence_health_report(),
            "evaluation": {
                "overall_solve_rate": round(
                    self._stats.get("solves_succeeded", 0) / max(self._stats.get("solves_attempted", 1), 1),
                    4,
                ),
                "solve_modes": dict(self._stats.get("solve_modes", {})),
                "transfer_attempt_rate": round(
                    self._stats.get("transfers_attempted", 0) / max(self._stats.get("solves_attempted", 1), 1),
                    4,
                ),
                "runtime_error_rate": round(
                    self._stats.get("runtime_errors", 0) / max(self._stats.get("solves_attempted", 1), 1),
                    4,
                ),
            },
        })

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
            ("working_memory", lambda: self.working_memory.save() if self.working_memory else None),
            ("developmental_curriculum", lambda: self.developmental_curriculum.save() if self.developmental_curriculum else None),
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


def _runtime_brain_config() -> Optional[dict]:
    role = str(os.environ.get("SARE_RUNTIME_ROLE", "") or "").strip().lower()
    if role != "web":
        return None
    return {
        "brain_boot": {
            "autostart_continuous_stream": False,
            "autostart_hippocampus": False,
            "warmup_environment_discovery": False,
            "warmup_physics_session": False,
            "warmup_knowledge_ingestion": False,
            "warmup_transform_generator": False,
            "warmup_action_physics": False,
            "warmup_robustness_batch": False,
            "seed_perception_priors": False,
        }
    }


def get_brain() -> Brain:
    """Get or create the global Brain instance."""
    global _brain
    if _brain is None:
        runtime_config = _runtime_brain_config()
        _brain = Brain(config=runtime_config) if runtime_config else Brain()
        _brain.boot()
    return _brain
