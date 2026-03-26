#!/usr/bin/env python3
"""
learn_daemon.py — Autonomous background learning daemon for SARE-HX.

Runs ExperimentRunner in a loop: generate → solve → reflect → induce → learn.
Also pre-registers hard problems to drive the transform synthesizer.

Usage:
    python3 learn_daemon.py [--verbose] [--interval 30] [--batch-size 5]
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path


class _RunnerBrain:
    """Minimal brain-like shim so AutonomousTrainer can run against ExperimentRunner."""
    self_model = None
    predictive_loop = None
    physics_simulator = None
    knowledge_ingester = None

    def __init__(self, runner):
        self._runner = runner

    def solve(self, expression: str) -> dict:
        try:
            from sare.engine import load_problem
            _, g = load_problem(expression)
            if g is None:
                return {"success": False, "delta": 0.0}

            class _P:  # minimal problem-like object
                pass
            p = _P()
            p.graph = g
            p.id = "auto_trainer"
            p.domain = "arithmetic"
            p.expression = expression
            p.py_graph = True

            r = self._runner._run_single(p)
            return {
                "success": bool(getattr(r, "solved", False)),
                "delta": float(getattr(r, "delta", 0.0)),
                "transforms_used": [],
                "steps": getattr(r, "proof_steps", 0),
            }
        except Exception:
            return {"success": False, "delta": 0.0}

REPO_ROOT = Path(__file__).resolve().parent
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("learn_daemon")

# ── Seed expressions that bootstrap the curriculum ─────────────────────────
_SEEDS = [
    # Core algebraic identities
    "x + 0",
    "x * 1",
    "x * 0",
    "0 + x",
    "1 * x",
    "x + x",
    "x - x",
    "not not x",
    "neg neg x",
    # Extended seeds for distributive / associative patterns
    "x * (y + 0)",
    "x * (y + 1 - 1)",
    "a * (b + 0)",
    "2 * (x + 0)",
    # Trigonometry
    "sin(0)",
    "cos(0)",
    "sin(x) * sin(x) + cos(x) * cos(x)",
    "sin(2 * x)",
    "tan(0)",
    # Calculus
    "integral(x)",
    "integral(0)",
    "derivative(x * x)",
    "derivative(x + 1)",
    # Probability
    "p + (1 - p)",
    "p * 1",
    "p * 0",
    "p(A) + p(not A)",
    # Logic / set theory
    "p and true",
    "p or false",
    "p and false",
    "p or true",
    "A union empty",
    "A intersect A",
    # Number theory
    "gcd(x, x)",
    "gcd(x, 0)",
    "mod(x, 1)",
    # Linear algebra
    "det(I)",
    "transpose(transpose(A))",
    "A + zero_matrix",
    # Combinatorics
    "n! / n!",
    "C(n, 0)",
    "C(n, n)",
    # Thermodynamics
    "delta_U + 0",
    "Q - W",
]


def _build_curriculum_gen():
    """Instantiate CurriculumGenerator, falling back gracefully."""
    try:
        from sare.curiosity.curriculum_generator import CurriculumGenerator
        gen = CurriculumGenerator()
        try:
            gen.load()
        except Exception:
            pass
        return gen
    except Exception as e:
        log.warning("CurriculumGenerator unavailable: %s", e)
        return None


def _build_experiment_runner(curriculum_gen):
    """Instantiate ExperimentRunner with searcher, energy, transforms."""
    try:
        from sare.engine import BeamSearch, EnergyEvaluator, get_transforms
        from sare.curiosity.experiment_runner import ExperimentRunner

        searcher   = BeamSearch()
        energy     = EnergyEvaluator()
        transforms = get_transforms(include_macros=True)

        # Optional reflection / induction
        reflection_engine = None
        causal_induction  = None
        concept_registry  = None
        try:
            from sare.reflection.py_reflection import get_reflection_engine
            reflection_engine = get_reflection_engine()
        except Exception as exc:
            log.debug("PyReflectionEngine unavailable: %s", exc)

        try:
            from sare.memory.concept_seed_loader import SeededConceptRegistry
            concept_registry = SeededConceptRegistry()
        except Exception as exc:
            log.debug("SeededConceptRegistry unavailable: %s", exc)

        try:
            from sare.causal.induction import CausalInduction
            causal_induction = CausalInduction()
        except Exception as exc:
            log.debug("CausalInduction unavailable: %s", exc)
            causal_induction = None

        runner = ExperimentRunner(
            curriculum_gen=curriculum_gen,
            searcher=searcher,
            energy=energy,
            reflection_engine=reflection_engine,
            causal_induction=causal_induction,
            concept_registry=concept_registry,
            transforms=transforms,
            beam_width=8,
            budget_seconds=5.0,
        )
        return runner
    except Exception as e:
        log.error("ExperimentRunner unavailable: %s", e)
        return None


def _seed_curriculum(curriculum_gen):
    """Add seed graphs from _SEEDS list into the curriculum."""
    if curriculum_gen is None:
        return 0

    from sare.engine import load_problem
    added = 0
    for expr in _SEEDS:
        try:
            _, g = load_problem(expr)
            if g:
                curriculum_gen.add_seed(g)
                added += 1
        except Exception as e:
            log.debug("Seed '%s' failed: %s", expr, e)
    log.info("Seeded curriculum with %d expressions", added)
    return added


def _load_hard_problems(curriculum_gen, repo_root: Path) -> int:
    """Pre-register hard problems to drive synthesizer."""
    path = repo_root / "data" / "hard_problems.json"
    if not path.exists():
        return 0
    try:
        from sare.engine import load_problem
        problems = json.loads(path.read_text())
        added = 0
        for p in problems:
            try:
                _, g = load_problem(p["expression"])
                if g:
                    curriculum_gen.add_seed(g)
                    added += 1
            except Exception:
                pass
        log.info("Pre-registered %d hard problems from data/hard_problems.json", added)
        return added
    except Exception as e:
        log.warning("Could not load hard problems: %s", e)
        return 0


def _save_hook(curriculum_gen, experiment_runner):
    """Called after each batch to persist state."""
    try:
        if curriculum_gen:
            curriculum_gen.save()
    except Exception as e:
        log.debug("curriculum_gen save error: %s", e)
    try:
        if experiment_runner and getattr(experiment_runner, '_transform_predictor', None):
            experiment_runner._transform_predictor.save()
    except Exception as e:
        log.debug("transform_predictor save error: %s", e)


def run_daemon(interval: float = 30.0, batch_size: int = 5, verbose: bool = False):
    """Main daemon loop."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    log.info("learn_daemon starting (interval=%.0fs, batch=%d)", interval, batch_size)

    curriculum_gen = _build_curriculum_gen()
    if curriculum_gen is None:
        log.error("Cannot start: CurriculumGenerator failed to load.")
        return 1

    experiment_runner = _build_experiment_runner(curriculum_gen)
    if experiment_runner is None:
        log.error("Cannot start: ExperimentRunner failed to load.")
        return 1

    # Bootstrap seeds
    _seed_curriculum(curriculum_gen)
    _load_hard_problems(curriculum_gen, REPO_ROOT)

    # ── Static world knowledge (no LLM) ──────────────────────────────────────
    try:
        from sare.knowledge.static_knowledge import get_static_loader
        _sk_loader = get_static_loader()
        _sk_status = _sk_loader.get_load_status()
        if not _sk_status["already_loaded"]:
            _sk_count = _sk_loader.load_all()
            log.info("StaticKnowledge: loaded %d structured facts into WorldModel", _sk_count)
        else:
            log.info("StaticKnowledge: already loaded (%d facts), skipping", _sk_status["fact_count"])
    except Exception as _sk_exc:
        log.debug("StaticKnowledge load error: %s", _sk_exc)

    # ── Seed curriculum with algorithmically-generated problems (no LLM) ─────
    try:
        from sare.knowledge.problem_factory import get_problem_factory
        _pf = get_problem_factory()
        _pf_batch = _pf.generate_batch(total=500, seed=2026)
        _pf_added = 0
        for _pf_prob in _pf_batch:
            try:
                from sare.engine import load_problem
                _pg = load_problem(_pf_prob["expression"])
                if _pg:
                    curriculum_gen.add_seed(_pg)
                    _pf_added += 1
            except Exception:
                pass
        log.info("ProblemFactory: seeded %d/%d generated problems into curriculum", _pf_added, len(_pf_batch))
    except Exception as _pf_exc:
        log.debug("ProblemFactory seed error: %s", _pf_exc)

    # A.3 — Augment commonsense KB from ConceptNet (one-time at startup)
    try:
        from sare.knowledge.commonsense import CommonSenseBase
        _kb = CommonSenseBase()
        _kb.load()
        if len(_kb._forward) < 60:
            _kb.seed()
            _kb.augment_from_conceptnet(
                ["dog", "fire", "water", "energy", "force", "atom",
                 "cell", "number", "algorithm", "language"],
                max_per_concept=8,
            )
            _kb.save()
            log.info("CommonSense KB augmented: %d concepts", len(_kb._forward))
    except Exception as exc:
        log.debug("ConceptNet augment skipped: %s", exc)

    # A.1 — Start AutonomousTrainer background learning
    _auto_trainer = None
    try:
        from sare.learning.autonomous_trainer import AutonomousTrainer
        _auto_trainer = AutonomousTrainer(interval_seconds=15.0, batch_size=5, max_workers=2)
        _brain_shim = _RunnerBrain(experiment_runner)
        _auto_trainer.start(_brain_shim)
        log.info("AutonomousTrainer background learning started")
    except Exception as exc:
        log.debug("AutonomousTrainer unavailable: %s", exc)

    # A.2 — Start MultiAgentLearner (5 domain specialists)
    _mal = None
    try:
        from sare.curiosity.multi_agent_learner import get_multi_agent_learner
        _mal = get_multi_agent_learner()
        result = _mal.start(n_agents=2)
        log.info("MultiAgentLearner started: %s", result)
    except Exception as exc:
        log.debug("MultiAgentLearner unavailable: %s", exc)

    # A.3 — Boot MultiAgentArena (competitive self-play between strategy variants)
    _arena = None
    try:
        from sare.agent.multi_agent_arena import MultiAgentArena
        _arena = MultiAgentArena(n_agents=3)
        log.info("MultiAgentArena ready (3 agents, competitive self-play)")
    except Exception as exc:
        log.debug("MultiAgentArena unavailable: %s", exc)

    # A.4 — Start GeneralSolver + GeneralCurriculum (general intelligence)
    _general_solver = None
    _general_curriculum = None
    try:
        from sare.cognition.general_solver import get_general_solver
        from sare.learning.general_curriculum import GeneralCurriculum
        _general_solver     = get_general_solver()
        _general_curriculum = GeneralCurriculum()
        log.info("GeneralSolver + GeneralCurriculum ready (10-domain general intelligence)")
    except Exception as exc:
        log.debug("GeneralSolver unavailable: %s", exc)

    # Handle graceful shutdown
    _stop = [False]

    def _handler(sig, frame):
        log.info("Shutdown signal received — stopping after current batch.")
        _stop[0] = True

    signal.signal(signal.SIGINT,  _handler)
    signal.signal(signal.SIGTERM, _handler)

    # Homeostasis-driven behavior
    _homeostasis = None
    try:
        from sare.meta.homeostasis import get_homeostatic_system
        _homeostasis = get_homeostatic_system()
    except Exception:
        pass

    # LLM Teacher — routes seek_human_input → LLM
    _llm_teacher = None
    try:
        from sare.meta.llm_teacher import get_llm_teacher
        _llm_teacher = get_llm_teacher()
        log.info("LLMTeacher ready (seek_human_input → LLM)")
    except Exception as exc:
        log.debug("LLMTeacher unavailable: %s", exc)

    # DreamConsolidator singleton — created once to preserve state across cycles
    _dream_consolidator = None
    try:
        from sare.learning.dream_consolidator import DreamConsolidator
        _dream_consolidator = DreamConsolidator()
        # Wire world model so it can receive causal discoveries
        try:
            from sare.memory.world_model import get_world_model
            _dream_consolidator.wire(world_model=get_world_model())
        except Exception:
            pass
        log.info("DreamConsolidator ready")
    except Exception as exc:
        log.debug("DreamConsolidator unavailable: %s", exc)

    _experiential_learner = None
    try:
        from sare.memory.experiential_learner import get_experiential_learner
        _experiential_learner = get_experiential_learner()
        log.info("ExperientialLearner ready (predict→surprise→update loop)")
    except Exception as exc:
        log.debug("ExperientialLearner unavailable: %s", exc)

    # Wire dream consolidator to experiential learner as event source
    try:
        if _dream_consolidator is not None and _experiential_learner is not None:
            _dream_consolidator.wire(
                predictive_loop=_experiential_learner,
                world_model=_dream_consolidator._world_model,
            )
            log.info("DreamConsolidator wired to ExperientialLearner as surprise event source")
    except Exception as _wire_exc:
        log.debug("DreamConsolidator wire error: %s", _wire_exc)

    # S32: EmbodiedAgent, CausalRollout, CounterfactualReasoner
    _embodied_agent = None
    try:
        from sare.world.embodied_agent import EmbodiedAgent
        _embodied_agent = EmbodiedAgent()
        log.info("EmbodiedAgent ready (grid-world perceive/decide/act/learn)")
    except Exception as exc:
        log.debug("EmbodiedAgent unavailable: %s", exc)

    _causal_rollout = None
    try:
        from sare.world.causal_rollout import CausalRollout
        _causal_rollout = CausalRollout()
        log.info("CausalRollout ready (multi-step causal prediction)")
    except Exception as exc:
        log.debug("CausalRollout unavailable: %s", exc)

    _counterfactual_reasoner = None
    try:
        from sare.causal.counterfactual_reasoner import CounterfactualReasoner
        _counterfactual_reasoner = CounterfactualReasoner()
        log.info("CounterfactualReasoner ready (intervention-based causal reasoning)")
    except Exception as exc:
        log.debug("CounterfactualReasoner unavailable: %s", exc)

    # EmbodiedAgent warm-up: 50 episodes to build procedural memory
    if _embodied_agent is not None:
        try:
            _warm_ep = _embodied_agent.run_session(n_episodes=50, max_steps_each=30)
            _warm_sum = _embodied_agent.summary()
            log.info("EmbodiedAgent warmup: %d episodes, reward=%.2f, concepts=%s",
                     _warm_sum.get("episodes_run", 0),
                     _warm_sum.get("total_reward", 0),
                     _warm_sum.get("concepts_learned", [])[:4])
        except Exception as _ea_warm_exc:
            log.debug("EmbodiedAgent warmup error: %s", _ea_warm_exc)

    # Seed CausalRollout from existing WorldModel causal links
    if _causal_rollout is not None:
        try:
            from sare.memory.world_model import get_world_model
            _wm = get_world_model()
            _seeded = 0
            for _link in list(_wm._causal_links.values())[:200]:
                _mech = getattr(_link, 'mechanism', '')
                if _mech:
                    _causal_rollout._model.observe_sequence(
                        [_mech], [_link.confidence * 2.0], _link.domain, success=True
                    )
                    _seeded += 1
            log.info("CausalRollout seeded %d observations from WorldModel links", _seeded)
        except Exception as _cr_seed_exc:
            log.debug("CausalRollout seed error: %s", _cr_seed_exc)

    # NeuralPerception — lightweight TF-IDF concept embedding layer (T2-1)
    neural_perception = None
    try:
        from sare.perception.neural_perception import get_neural_perception
        from sare.memory.world_model import get_world_model as _gwm_np
        neural_perception = get_neural_perception(_gwm_np())
        log.info("NeuralPerception ready (%d concepts)", len(neural_perception._concept_list))
    except Exception as _np_exc:
        neural_perception = None
        log.debug("NeuralPerception init failed: %s", _np_exc)

    # Stuck-problem tracker for TransformSynthesizer
    # {domain: [expr1, expr2, ...]} — accumulates unsolved exprs; fires synthesis at _SYNTH_TRIGGER
    _stuck_by_domain: dict = {}
    _SYNTH_TRIGGER = 5    # synthesize after this many stuck problems per domain (was 15)
    _synth_cooldown: dict = {}  # {domain: last_synth_cycle}
    _SYNTH_COOLDOWN_CYCLES = 10  # don't re-synthesize a domain within 10 cycles

    # MetaLearningEngine — MAML fast_adapt + online_adapt wiring
    _meta_learner = None
    try:
        from sare.meta.meta_learner import MetaLearningEngine
        _meta_learner = MetaLearningEngine()
        log.info("MetaLearningEngine ready (MAML fast_adapt + online_adapt)")
    except Exception as exc:
        log.debug("MetaLearningEngine unavailable: %s", exc)

    # ── Reactive wiring: "rule_promoted" → immediate AnalogyTransfer sweep ──
    try:
        from sare.core.event_bus import get_event_bus as _get_event_bus
        def _on_rule_promoted(data):
            try:
                import threading as _at_thread
                def _sweep():
                    try:
                        from sare.causal.analogy_transfer import AnalogyTransfer
                        _at = AnalogyTransfer(concept_registry)
                        _transfers = _at.transfer_all_domains()
                        if _transfers:
                            _applied, _skipped = _at.apply_to_registry(_transfers)
                            if _applied:
                                log.info("[AnalogyTransfer] rule_promoted event → applied %d new rules (%d already known)",
                                         _applied, _skipped)
                    except Exception as _at_exc:
                        log.debug("AnalogyTransfer reactive sweep error: %s", _at_exc)
                _at_thread.Thread(target=_sweep, daemon=True, name="analogy-transfer-reactive").start()
            except Exception as _at_outer_exc:
                log.debug("rule_promoted handler error: %s", _at_outer_exc)
        _get_event_bus().subscribe("rule_promoted", _on_rule_promoted)
        log.info("learn_daemon: subscribed to 'rule_promoted' → AnalogyTransfer reactive sweep")

        # Item 4: transfer_verified → register as ConceptRule in registry
        def _on_transfer_verified(data):
            try:
                name = (data or {}).get("name", "")
                domain = (data or {}).get("domain", "general")
                confidence = float((data or {}).get("confidence", 0.75))
                if name:
                    concept_registry.add_rule({
                        "name": name,
                        "domain": domain,
                        "confidence": confidence,
                        "observations": 10,
                        "source": "transfer",
                    })
                    concept_registry.save()
                    log.info("[TransferRule] Registered verified transfer '%s' (domain=%s conf=%.2f)",
                             name, domain, confidence)
            except Exception as _tv_exc:
                log.debug("transfer_verified handler error: %s", _tv_exc)
        _get_event_bus().subscribe("transfer_verified", _on_transfer_verified)
        log.info("learn_daemon: subscribed to 'transfer_verified' → ConceptRule registration")

        # Bootstrap: promote already-verified hypotheses from disk (missed events)
        try:
            from sare.transfer.engine import get_transfer_engine
            _boot_te = get_transfer_engine()
            _boot_verified = [h for h in _boot_te._hypotheses.values() if h.status == "verified"]
            _boot_promoted = 0
            for _bh in _boot_verified:
                name = _bh.proposed_transform
                existing = [r for r in concept_registry.get_all_rules()
                            if (r.get("name") if isinstance(r, dict) else getattr(r, "name", "")) == name]
                if not existing and name:
                    concept_registry.add_rule({
                        "name": name,
                        "domain": _bh.target_domain,
                        "confidence": _bh.confidence,
                        "observations": 10,
                        "source": "transfer_bootstrap",
                    })
                    _boot_promoted += 1
            if _boot_promoted:
                concept_registry.save()
                log.info("[TransferBootstrap] Promoted %d already-verified hypotheses into ConceptRegistry", _boot_promoted)
        except Exception as _boot_exc:
            log.debug("Transfer bootstrap error: %s", _boot_exc)
    except Exception as _eb_exc:
        log.debug("EventBus subscription error: %s", _eb_exc)

    # T2-3: Continuous-Time Learning Stream
    continuous_learner = None
    try:
        from sare.learning.continuous_stream import ContinuousLearner
        _wm_for_cl = None
        try:
            from sare.memory.world_model import get_world_model
            _wm_for_cl = get_world_model()
        except Exception:
            pass
        continuous_learner = ContinuousLearner(
            experiment_runner=experiment_runner,
            concept_registry=getattr(experiment_runner, 'concept_registry', None),
            world_model=_wm_for_cl,
        )
        log.info("ContinuousLearner ready (episode-driven micro-learning, k=%d)", 3)
    except Exception as _cl_exc:
        log.debug("ContinuousLearner unavailable: %s", _cl_exc)

    # T2-2: Grounded Concept Learning — toy physics simulation
    grounded_learner = None
    try:
        from sare.world.toy_physics import get_grounded_learner
        from sare.memory.world_model import get_world_model as _gwm_gcl
        grounded_learner = get_grounded_learner(_gwm_gcl())
        log.info("GroundedConceptLearner ready: %s", grounded_learner.get_stats())
    except Exception as _gcl_exc:
        grounded_learner = None
        log.debug("GroundedConceptLearner init failed: %s", _gcl_exc)

    # T3-3: Catastrophic forgetting prevention
    try:
        from sare.learning.forgetting_prevention import get_forgetting_prevention
        forgetting_prevention = get_forgetting_prevention()
        log.info("ForgettingPrevention ready: %s", forgetting_prevention.stats)
    except Exception as _fp_exc:
        forgetting_prevention = None
        log.debug("ForgettingPrevention init failed: %s", _fp_exc)

    # T3-4: Global Workspace
    try:
        from sare.cognition.global_workspace import get_global_workspace
        global_workspace = get_global_workspace()
        log.info("GlobalWorkspace ready: %s", global_workspace.stats)
    except Exception as _gw_exc:
        global_workspace = None
        log.debug("GlobalWorkspace init failed: %s", _gw_exc)

    cycle = 0
    while not _stop[0]:
        cycle += 1

        # ── Homeostasis-driven batch adaptation ──────────────────────
        _effective_batch = batch_size
        _behavior = "explore_new_domain"
        if _homeostasis:
            try:
                _homeostasis.tick()
                _behavior = _homeostasis.get_behavior_recommendation()
                if _behavior == "consolidate_memory":
                    # Consolidation mode: smaller batch, save more often
                    _effective_batch = max(2, batch_size // 2)
                    log.info("Homeostasis → CONSOLIDATE (batch=%d)", _effective_batch)
                elif _behavior == "explore_new_domain":
                    # Exploration mode: try harder problems
                    _effective_batch = batch_size + 2
                    log.info("Homeostasis → EXPLORE (batch=%d)", _effective_batch)
                elif _behavior == "deepen_weak_domain":
                    # Focus mode: normal batch, curriculum will focus
                    log.info("Homeostasis → DEEPEN weak domains")
                elif _behavior == "seek_human_input":
                    log.info("Homeostasis → SEEK INPUT (social drive high) → routing to LLM")
                    if _llm_teacher is not None:
                        # Collect recent failed expressions from runner history
                        _stuck = []
                        try:
                            _stuck = [
                                getattr(r, "expression", None) or ""
                                for r in (experiment_runner._history or [])[-30:]
                                if not getattr(r, "solved", True)
                            ]
                            _stuck = [e for e in _stuck if e][:8]
                        except Exception:
                            pass
                        # Fall back to seed expressions if no recent failures recorded
                        if not _stuck:
                            _stuck = _SEEDS[:6]
                        import threading as _threading
                        _t = _threading.Thread(
                            target=_llm_teacher.seek_and_apply,
                            kwargs={
                                "stuck_exprs": _stuck,
                                "homeostasis": _homeostasis,
                                "cycle": cycle,
                                "curriculum_gen": curriculum_gen,
                            },
                            daemon=True,
                            name="llm-teacher",
                        )
                        _t.start()
                elif _behavior == "generate_analogies":
                    log.info("Homeostasis → ANALOGY mode")
                else:
                    log.debug("Homeostasis → %s", _behavior)
            except Exception as exc:
                log.debug("Homeostasis tick error: %s", exc)

        try:
            results = experiment_runner.run_batch(n=_effective_batch)
            solved  = sum(1 for r in results if r.solved)
            promoted = sum(1 for r in results if r.rule_promoted)
            log.info(
                "Cycle %d [%s]: %d/%d solved, %d rules promoted",
                cycle, _behavior, solved, len(results), promoted,
            )
            # Homeostasis satisfaction after each batch
            if _homeostasis:
                try:
                    _homeostasis.on_batch_completed(solved, len(results))
                    if promoted > 0:
                        _homeostasis.on_rule_discovered()
                except Exception as exc:
                    log.debug("Homeostasis on_batch_completed error: %s", exc)
        except Exception as e:
            log.error("Batch error on cycle %d: %s", cycle, e)

        # T2-2: Ground newly promoted rules via toy physics simulation
        if grounded_learner is not None and promoted > 0:
            try:
                for _r in results:
                    if getattr(_r, 'rule_promoted', False):
                        grounded_learner.ground_new_rule(
                            getattr(_r, 'promoted_rule_name', ''),
                            getattr(_r, 'domain', 'algebra'),
                        )
            except Exception:
                pass

        # T2-3: Feed every result into the continuous learner
        if continuous_learner is not None:
            for _r in results:
                try:
                    continuous_learner.record_episode(_r)
                except Exception:
                    pass

        # T2-5: Few-shot adaptation — collect high-confidence solves as examples
        try:
            from sare.learning.few_shot_adapter import get_few_shot_adapter as _get_fsa
            _fsa = _get_fsa()
            for _r in results:
                if getattr(_r, 'solved', False) and getattr(_r, 'expression', ''):
                    _steps = list(getattr(_r, 'proof_steps', None) or [])
                    if _steps:
                        _conf = 1.0 - float(getattr(_r, 'energy_after', 0) or 0) / max(float(getattr(_r, 'energy_before', 1) or 1), 0.001)
                        _conf = max(0.0, min(1.0, _conf))
                        _fsa.add_example(
                            _r.expression,
                            " → ".join(_steps),
                            getattr(_r, 'domain', 'general'),
                            _conf
                        )
        except Exception:
            pass

        _save_hook(curriculum_gen, experiment_runner)

        # Wire experiential learner — predict→experience→surprise loop
        if _experiential_learner is not None:
            for _r in results:
                _expr = getattr(_r, "problem_id", "") or ""
                _dom  = getattr(_r, "domain", "arithmetic") or "arithmetic"
                _solved = bool(getattr(_r, "solved", False))
                _delta = float(getattr(_r, "energy_before", 0) - getattr(_r, "energy_after", 0))
                _transforms_used = getattr(_r, "rule_name", "")
                _transforms_list = [_transforms_used] if _transforms_used else []
                _experiential_learner.record_experience(
                    prediction=None,  # no prediction was made before (first pass)
                    actual_transforms=_transforms_list,
                    actual_delta=_delta,
                    domain=_dom,
                    solved=_solved,
                )
            # Every 10 cycles: run imagination
            if cycle % 10 == 0:
                _n_counterfactuals = _experiential_learner.imagine_batch(n=30)
                if _n_counterfactuals:
                    log.info("[Experience] Imagined %d counterfactual beliefs (cycle %d)", _n_counterfactuals, cycle)

        # ── MetaLearner: online_adapt per result + fast_adapt every 15 cycles ─
        if _meta_learner is not None:
            try:
                for _r in results:
                    _dom  = getattr(_r, "domain", "arithmetic") or "arithmetic"
                    _expr = getattr(_r, "problem_id", "") or ""
                    _solved = bool(getattr(_r, "solved", False))
                    _delta = float(getattr(_r, "energy_before", 0.0) - getattr(_r, "energy_after", 0.0))
                    _meta_learner.online_adapt(
                        _expr, _dom,
                        solved=_solved,
                        delta=_delta,
                    )
            except Exception as _ma_exc:
                log.debug("online_adapt error: %s", _ma_exc)
            if cycle % 15 == 0:
                try:
                    import threading as _threading
                    _weak_domain = _experiential_learner.get_weakest_domain() if _experiential_learner else "arithmetic"
                    if hasattr(_meta_learner, "_domain_ema") and _meta_learner._domain_ema:
                        _weak_domain = min(_meta_learner._domain_ema,
                                           key=_meta_learner._domain_ema.get)
                    _t_maml = _threading.Thread(
                        target=_meta_learner.fast_adapt,
                        kwargs={"domain": _weak_domain, "n_inner_steps": 3,
                                "n_support": 3, "n_query": 3},
                        daemon=True, name="maml-fast-adapt",
                    )
                    _t_maml.start()
                    log.info("[MAML] fast_adapt launched for domain=%s", _weak_domain)
                except Exception as _fa_exc:
                    log.debug("fast_adapt error: %s", _fa_exc)

        # ── MultiAgentArena: competitive self-play every 50 cycles ──────────
        if cycle % 50 == 0 and _arena is not None:
            try:
                def _arena_engine(expr, beam_width=6, budget_seconds=3.0, **_kw):
                    from sare.engine import load_problem
                    _, _g = load_problem(expr)
                    if _g is None:
                        return {"success": False, "delta": 0.0, "result": expr, "steps": 0, "transforms_used": []}
                    _p = type("P", (), {
                        "graph": _g, "id": "arena", "domain": "arithmetic",
                        "expression": expr, "py_graph": True,
                    })()
                    _r = experiment_runner._run_single(_p)
                    return {
                        "success": bool(getattr(_r, "solved", False)),
                        "delta": float(getattr(_r, "delta", 0.0)),
                        "result": expr, "steps": len(getattr(_r, "proof_steps", None) or []),
                        "transforms_used": list(getattr(_r, "proof_steps", None) or []),
                    }
                _arena_expr = (results[0].problem_id if results else "x + 0")
                _winner, _ = _arena.race(_arena_expr, _arena_engine, max_workers=3)
                log.info("[ARENA] cycle %d: winner=%s delta=%.3f",
                         cycle, _winner.agent_id, _winner.delta)
            except Exception as _ae:
                log.debug("Arena race error: %s", _ae)

        # ── RedTeamAdversary: challenge promoted rules every 25 cycles ───────
        if cycle % 25 == 0 and promoted > 0:
            try:
                from sare.agent.red_team import RedTeamAdversary as _RTA
                _rt = _RTA()
                _rt.wire(engine=lambda expr: experiment_runner._run_single(
                    type("P", (), {"graph": __import__("sare.engine", fromlist=["load_problem"]).load_problem(expr)[1],
                                   "id": "rt", "domain": "algebra", "expression": expr, "py_graph": True})()
                ))
                _rt_res = _rt.run_attack_round(top_k=3)
                log.info("[REDTEAM] cycle %d: attacks=%d falsifications=%d rate=%.2f",
                         cycle, _rt_res.get("attacks", 0), _rt_res.get("falsifications", 0),
                         _rt_res.get("falsification_rate", 0.0))
            except Exception as _rte:
                log.debug("RedTeam error: %s", _rte)

        # ── T2-1: NeuralPerception — embed recent expressions every 15 cycles ──
        if cycle % 15 == 0 and neural_perception is not None:
            try:
                # Perceive recent expressions to update WorldModel beliefs
                _recent = [r for r in results if hasattr(r, 'expression') and r.expression][:5]
                for _r in _recent:
                    neural_perception.perceive_problem(_r.expression, getattr(_r, 'domain', 'general'))
            except Exception:
                pass

        # ── S32: EmbodiedAgent — 1 episode per cycle ─────────────────────────
        if _embodied_agent is not None:
            try:
                _ep = _embodied_agent.run_episode(max_steps=20)
                if _ep and _ep.concepts_discovered:
                    log.info("[Embodied] Episode %d: reward=%.2f concepts=%s",
                             _ep.episode_id, _ep.total_reward, _ep.concepts_discovered)
            except Exception as _ea_exc:
                log.debug("EmbodiedAgent episode error: %s", _ea_exc)

        # ── S32: CausalRollout — feed successful solves every 5 cycles ───────
        if _causal_rollout is not None and cycle % 5 == 0:
            try:
                for _r in results:
                    if getattr(_r, 'rule_name', ''):
                        _ts = [_r.rule_name]
                        _ds = [float(_r.energy_before - _r.energy_after)]
                        _dom = getattr(_r, 'domain', 'algebra')
                        _solved = bool(getattr(_r, 'solved', False))
                        _causal_rollout._model.observe_sequence(_ts, _ds, _dom, success=_solved)
            except Exception as _cr_exc:
                log.debug("CausalRollout observe error: %s", _cr_exc)

        # ── S32: CounterfactualReasoner — analyze hard cases every 10 cycles ─
        if _counterfactual_reasoner is not None and cycle % 10 == 0:
            try:
                _hard = [_r for _r in results
                         if not getattr(_r, 'solved', True)
                         and abs(getattr(_r, 'energy_before', 0) - getattr(_r, 'energy_after', 0)) < 0.01]
                for _r in _hard[:3]:
                    _counterfactual_reasoner.analyze(
                        expression=getattr(_r, 'problem_id', ''),
                        domain=getattr(_r, 'domain', 'algebra'),
                        transforms_applied=[],
                        energy_before=getattr(_r, 'energy_before', 1.0),
                        energy_after=getattr(_r, 'energy_after', 1.0),
                    )
            except Exception as _cfr_exc:
                log.debug("CounterfactualReasoner error: %s", _cfr_exc)

        # ── HTM Predictor: record transform sequences for n-gram learning ────
        try:
            from sare.neuro.htm_predictor import get_htm_predictor
            _htm_pred = get_htm_predictor()
            for _r in results:
                _ps = getattr(_r, "proof_steps", None) or []
                if _ps:
                    _dom = getattr(_r, "domain", "algebra") or "algebra"
                    _solved = bool(getattr(_r, "solved", False))
                    _htm_pred.observe_sequence(_ps, _dom, success=_solved)
        except Exception as _htm_exc:
            log.debug("HTMPredictor observe error: %s", _htm_exc)

        # ── Accumulate stuck problems + trigger TransformSynthesizer ─────
        try:
            for _r in results:
                # problem_id is the expression string in ExperimentResult
                _expr = getattr(_r, "problem_id", None) or ""
                _dom  = getattr(_r, "domain", "algebra") or "algebra"
                _e_before = getattr(_r, "energy_before", 0.0)
                _e_after  = getattr(_r, "energy_after",  0.0)
                # no_matching_transforms proxy: failed + no energy change (nothing applied)
                _no_transform = not getattr(_r, "solved", True) and abs(_e_before - _e_after) < 0.01
                # slow solve: solved but took many steps (hard problem — partial stuck credit)
                _n_steps = len(getattr(_r, "proof_steps", None) or [])
                _slow_solve = getattr(_r, "solved", False) and _n_steps >= 6
                if _no_transform or _slow_solve:
                    _stuck_by_domain.setdefault(_dom, [])
                    if _expr and _expr not in _stuck_by_domain[_dom]:
                        if _slow_solve and not _no_transform:
                            # slow solves count as 0.5 — track with a marker to halve trigger
                            _stuck_by_domain[_dom].append(f"~{_expr}")  # ~ prefix = slow
                        else:
                            _stuck_by_domain[_dom].append(_expr)

            # Only synthesize for symbolic domains — others use general solver
            _SYNTH_DOMAINS = {"algebra", "arithmetic", "calculus", "logic", "general"}
            for _dom, _stuck_exprs in list(_stuck_by_domain.items()):
                if _dom not in _SYNTH_DOMAINS:
                    _stuck_by_domain[_dom] = []  # clear non-symbolic accumulator
                    continue
                # Count stuck credits: full for unsolved, 0.5 for slow solves (~ prefix)
                _stuck_credits = sum(0.5 if e.startswith("~") else 1.0 for e in _stuck_exprs)
                if _stuck_credits < _SYNTH_TRIGGER:
                    continue
                if cycle - _synth_cooldown.get(_dom, 0) < _SYNTH_COOLDOWN_CYCLES:
                    continue
                # Fire synthesizer
                try:
                    from sare.learning.llm_transform_synthesizer import get_llm_synthesizer
                    _synth = get_llm_synthesizer()
                    _existing = [t.name() for t in experiment_runner.transforms if hasattr(t, "name")]
                    # Build validation graphs from recent SOLVED problems in this domain
                    # (not from stuck ones — a new transform shouldn't be validated against
                    # the hardest unsolvable cases; solved cases provide a positive signal)
                    _val_graphs = []
                    try:
                        from sare.engine import load_problem as _lp_val
                        _hist = getattr(experiment_runner, '_history', []) or []
                        for _hr in reversed(_hist[-80:]):
                            if len(_val_graphs) >= 8:
                                break
                            if (getattr(_hr, 'domain', '') == _dom and
                                    getattr(_hr, 'solved', False)):
                                _hr_expr = (getattr(_hr, 'problem_id', '') or
                                            getattr(_hr, 'expression', ''))
                                if _hr_expr:
                                    try:
                                        _, _vg = _lp_val(_hr_expr)
                                        if _vg is not None:
                                            _val_graphs.append(_vg)
                                    except Exception:
                                        pass
                        # Fallback: use stuck exprs unpacked correctly
                        if len(_val_graphs) < 2:
                            for _e in _stuck_exprs[:6]:
                                if len(_val_graphs) >= 6:
                                    break
                                try:
                                    _, _vg = _lp_val(_e)
                                    if _vg is not None:
                                        _val_graphs.append(_vg)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    log.info("[Synth] Triggering synthesis for domain=%s (%d stuck exprs)", _dom, len(_stuck_exprs))
                    _synth_cooldown[_dom] = cycle      # mark cooldown immediately
                    _stuck_by_domain[_dom] = []        # reset accumulator immediately
                    # Run in background — LLM call should not block main loop
                    import threading as _threading
                    def _do_synthesis(_s, _d, _stuck, _vg, _ex, _runner):
                        try:
                            _res = _s.synthesize(
                                domain=_d,
                                stuck_exprs=_stuck[:10],
                                validation_graphs=_vg,
                                existing_transform_names=_ex,
                            )
                            if _res["promoted"]:
                                log.info("[Synth] New transform synthesized: %s (score=%.2f)",
                                         _res["actual_name"], _res["score"])
                                try:
                                    _runner._load_synthesized_transforms()
                                except Exception:
                                    pass
                            else:
                                log.info("[Synth] Synthesis failed for domain=%s: %s", _d, _res["message"])
                        except Exception as _se:
                            log.debug("[Synth] Thread error: %s", _se)
                    _threading.Thread(
                        target=_do_synthesis,
                        args=(_synth, _dom, _stuck_exprs[:10], _val_graphs, _existing, experiment_runner),
                        daemon=True, name=f"synth-{_dom}",
                    ).start()
                except Exception as _se:
                    log.debug("[Synth] Error: %s", _se)
        except Exception as _stuck_err:
            log.debug("Stuck tracker error: %s", _stuck_err)

        # ── General intelligence batch (every cycle) ──────────────────────
        if _general_solver is not None and _general_curriculum is not None:
            try:
                _gen_problems = _general_curriculum.generate_batch(size=10)
                _gen_solved = 0
                _gen_lessons = 0
                _llm_problems = []  # problems that need LLM — offloaded to thread
                for _gp in _gen_problems:
                    # Fast path: template problems have known answers — no LLM needed
                    if _gp.expected:
                        _gr = _general_solver.solve_with_known_answer(
                            _gp.text, _gp.expected, _gp.domain)
                        if _gr.solved:
                            _gen_solved += 1
                        # Record outcome for AlgorithmSelector (meta-learning)
                        try:
                            from sare.meta.algorithm_selector import get_algorithm_selector
                            _task_type = {"math": "algebra", "logic": "logic", "code": "coding",
                                          "language": "language", "planning": "planning"}.get(_gp.domain, "science")
                            _strategy = {"symbolic": "beam_search", "llm": "greedy", "hybrid": "mcts",
                                         "kb_cache": "beam_search", "fact_chain": "greedy",
                                         "template": "beam_search"}.get(_gr.solver_used, "beam_search")
                            get_algorithm_selector().record_outcome(_task_type, _strategy, _gr.solved)
                        except Exception:
                            pass
                        if _gr.lesson and _gr.solved:
                            _gen_lessons += 1
                            try:
                                from sare.memory.world_model import get_world_model
                                get_world_model().add_fact(
                                    domain=_gp.domain,
                                    fact=_gr.lesson[:200],
                                    confidence=_gr.confidence,
                                )
                            except Exception:
                                pass
                        if _gp.expected and _gr.answer:
                            _correct = _gp.expected.lower() in _gr.answer.lower() or _gr.answer.lower() in _gp.expected.lower()
                            if not _correct:
                                log.debug("[General] Wrong: '%s' expected='%s' got='%s'",
                                          _gp.text[:50], _gp.expected[:30], _gr.answer[:30])
                    else:
                        # LLM path — collect and fire in a single background thread
                        _llm_problems.append(_gp)

                # Fire LLM problems in background — doesn't block main loop
                if _llm_problems:
                    import threading as _threading
                    def _run_llm_batch(_probs, _solver):
                        for _p in _probs:
                            try:
                                _r = _solver.solve(_p.text, context=_p.context or "", problem_type=_p.domain)
                                if _r.lesson and _r.solved:
                                    try:
                                        from sare.memory.world_model import get_world_model
                                        get_world_model().add_fact(
                                            domain=_p.domain,
                                            fact=_r.lesson[:200],
                                            confidence=_r.confidence,
                                        )
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                    _threading.Thread(
                        target=_run_llm_batch,
                        args=(_llm_problems, _general_solver),
                        daemon=True, name="general-llm-batch",
                    ).start()

                log.info(
                    "General batch: %d/%d template solved, %d lessons, %d LLM→bg (domains: %s)",
                    _gen_solved, len(_gen_problems) - len(_llm_problems),
                    _gen_lessons, len(_llm_problems),
                    ", ".join(set(p.domain for p in _gen_problems)),
                )
            except Exception as exc:
                log.debug("General intelligence batch error: %s", exc)

        # T3-4: Global Workspace — key modules submit attention bids every 5 cycles
        if cycle % 5 == 0 and global_workspace is not None:
            try:
                from sare.core.event_bus import get_event_bus
                # Homeostasis bids based on its drive levels
                get_event_bus().publish("attention_bid", {
                    "module": "homeostasis",
                    "content": {"cycle": cycle},
                    "urgency": 0.4, "relevance": 0.5
                })
                # SelfModel bids when in weak domain
                get_event_bus().publish("attention_bid", {
                    "module": "self_model",
                    "content": {"cycle": cycle, "focus": "weak_domain"},
                    "urgency": 0.3, "relevance": 0.6
                })
            except Exception:
                pass

        # ── FactInference + KB stats every 5 cycles ───────────────────────────
        if cycle % 5 == 0:
            try:
                from sare.cognition.fact_inference import get_fact_inference
                _fi = get_fact_inference()
                _fi_total = 0
                for _fi_domain in ["factual", "science", "reasoning", "analogy"]:
                    _n = _fi.infer_iterative(_fi_domain, max_rounds=3, max_new_per_round=20)
                    if _n:
                        _fi_total += _n
                        log.info("[Daemon] Iterative chaining: %d new facts for domain=%s", _n, _fi_domain)
                if _fi_total:
                    log.info("[Daemon] Total inferred this cycle: %d facts", _fi_total)
            except Exception as _fi_exc:
                log.debug("FactInference error: %s", _fi_exc)

            try:
                from sare.memory.knowledge_lookup import KnowledgeLookup
                _kl = KnowledgeLookup()
                _kb_stats = _kl.get_stats()
                if _general_solver is not None and hasattr(_general_solver, '_kb_lookup') and _general_solver._kb_lookup is not None:
                    _kb_stats = _general_solver._kb_lookup.get_stats()
                log.info("[Daemon] KB stats: %s", _kb_stats)
            except Exception as _kl_exc:
                log.debug("KnowledgeLookup stats error: %s", _kl_exc)

        # ── KB self-test every 10 cycles: find gaps, route to priority curriculum ─
        if cycle % 10 == 0 and _general_solver is not None and _general_curriculum is not None:
            try:
                import random as _rng
                from sare.learning.general_curriculum import (
                    _FACTUAL_TEMPLATES, _REASONING_TEMPLATES,
                    GeneralProblem, inject_priority_problems,
                )
                from sare.cognition.fact_inference import get_fact_inference
                from sare.memory.fact_ingester import _extract_triples

                _pool = list(_FACTUAL_TEMPLATES) + list(_REASONING_TEMPLATES)
                _sample = _rng.sample(_pool, min(20, len(_pool)))
                _st_hits, _st_misses = 0, []

                for _sq, _sa in _sample:
                    _kb_ans = None
                    # Try KB lookup first
                    if _general_solver._kb_lookup is not None:
                        try:
                            _hit = _general_solver._kb_lookup.lookup(_sq, "factual")
                            if _hit:
                                _kb_ans = _hit.answer
                        except Exception:
                            pass
                    # Try fact chain if KB missed
                    if not _kb_ans:
                        try:
                            _triples = _extract_triples(_sq, "")
                            if _triples:
                                _s, _p, _ = _triples[0]
                                _kb_ans = get_fact_inference().chain_to_goal(_s, _p, "factual", max_depth=3)
                        except Exception:
                            pass
                    if (_kb_ans and (
                        _sa.lower() in _kb_ans.lower() or _kb_ans.lower() in _sa.lower()
                    )):
                        _st_hits += 1
                    else:
                        _st_misses.append((_sq, _sa))

                _st_rate = _st_hits / max(len(_sample), 1)
                log.info("[Self-test] KB self-sufficiency: %d/%d (%.1f%%) — %d gaps found",
                         _st_hits, len(_sample), _st_rate * 100, len(_st_misses))

                # Route gaps back as high-priority curriculum
                if _st_misses:
                    import time as _t
                    inject_priority_problems([
                        GeneralProblem(
                            problem_id=f"selftest_{int(_t.time()*1000) % 10**7}_{_i}",
                            text=_mq, domain="factual", expected=_ma, difficulty=0.7,
                        )
                        for _i, (_mq, _ma) in enumerate(_st_misses[:5])
                    ])
                    log.info("[Self-test] Injected %d KB gaps into priority curriculum", min(5, len(_st_misses)))
            except Exception as _st_err:
                log.debug("Self-test error: %s", _st_err)

        # KB Gap Detector: every 20 cycles, find under-explored subjects and inject curiosity goals
        if cycle % 20 == 0 and _general_curriculum is not None:
            try:
                from sare.memory.world_model import get_world_model as _gwm
                from sare.learning.general_curriculum import GeneralProblem, inject_priority_problems
                _wm_gap = _gwm()
                _beliefs = _wm_gap.get_beliefs()
                # Build subject → set of predicates
                _subj_preds: dict = {}
                for _b in _beliefs:
                    _s = str(_b.get("subject", "") or "").lower().strip()
                    _p = str(_b.get("predicate", "") or "").lower().strip()
                    if _s and _p and len(_s) > 2:
                        _subj_preds.setdefault(_s, set()).add(_p)
                # Subjects known but with only 1-2 predicates are KB gaps
                _gap_subjects = [_s for _s, _ps in _subj_preds.items()
                                 if 1 <= len(_ps) <= 2 and not _s.startswith("_")]
                if _gap_subjects:
                    import random as _rng2, time as _t2
                    _gap_subjects = _rng2.sample(_gap_subjects, min(6, len(_gap_subjects)))
                    _gap_probes = [
                        ("what does {} do?".format(_s),          "factual"),
                        ("what is {} made of?".format(_s),       "science"),
                        ("where does {} live?".format(_s),       "factual"),
                        ("what is {} related to?".format(_s),    "reasoning"),
                    ]
                    _gap_problems = []
                    for _s in _gap_subjects:
                        _q, _dom = _rng2.choice(_gap_probes)
                        _q = _q.replace("{}".format(_s), _s)
                        _gap_problems.append(GeneralProblem(
                            problem_id=f"kbgap_{int(_t2.time()*1000) % 10**7}_{_s[:8]}",
                            text=_q, domain=_dom, expected=None, difficulty=0.6,
                        ))
                    inject_priority_problems(_gap_problems)
                    log.info("[KB-Gap] Injected %d open-ended goal questions from %d sparse subjects",
                             len(_gap_problems), len(_gap_subjects))
            except Exception as _gap_err:
                log.debug("KB gap detection error: %s", _gap_err)

        # Episodic Replay Loop every 25 cycles — retry failed episodes via hippocampus
        if cycle % 25 == 0:
            try:
                from sare.memory.hippocampus import HippocampusDaemon
                _hc = HippocampusDaemon(curriculum_gen=curriculum_gen)
                _injected = _hc.replay_episodes(curriculum_gen)
                if _injected:
                    log.info("[EpisodicReplay] Cycle %d: injected %d hard episodes into curriculum", cycle, _injected)
            except Exception as _er_exc:
                log.debug("EpisodicReplay error: %s", _er_exc)

        # Hypothesis generation every 15 cycles — genuine curiosity from analogy
        if cycle % 15 == 0:
            try:
                from sare.cognition.hypothesis_maker import get_hypothesis_maker
                _hyps = get_hypothesis_maker().propose(max_proposals=3)
                if _hyps:
                    log.info("[HypothesisMaker] Proposed %d hypotheses: %s",
                             len(_hyps), [f"{s} is_a {v}" for s, p, v in _hyps[:2]])
            except Exception as _hm_exc:
                log.debug("HypothesisMaker error: %s", _hm_exc)

        # Dream consolidation every 10 cycles (or when consolidation drive is high)
        _consolidation_due = (cycle % 10 == 0)
        if not _consolidation_due and _homeostasis:
            try:
                _consolidation_due = (
                    _homeostasis.drives.get("consolidation") is not None
                    and _homeostasis.drives["consolidation"].level > 0.7
                )
            except Exception:
                pass
        if _consolidation_due and _dream_consolidator is not None:
            try:
                rec = _dream_consolidator.dream_cycle()
                log.info(
                    "DreamConsolidator: replayed=%d, discovered=%d edges (total=%d)",
                    rec.events_replayed, rec.causal_edges_found,
                    _dream_consolidator._total_discovered,
                )
                if _homeostasis:
                    _homeostasis.on_sleep_cycle()
            except Exception as exc:
                log.debug("DreamConsolidator cycle error: %s", exc)

        # Save world model periodically
        if cycle % 5 == 0:
            try:
                from sare.memory.world_model import get_world_model
                get_world_model().save()
            except Exception:
                pass

        # Self-generated questions every 5 cycles
        if cycle % 5 == 0:
            try:
                from sare.curiosity.question_generator import get_question_generator
                qg = get_question_generator()
                new_qs = qg.generate_questions()
                if new_qs:
                    log.info("QuestionGenerator: %d new self-generated questions", len(new_qs))
                    for q in new_qs[:3]:
                        log.info("  ? [%s] %s", q.source, q.text[:80])
            except Exception as exc:
                log.debug("QuestionGenerator error: %s", exc)

        # C.1 — Non-math problem injection every 10 cycles
        if cycle % 10 == 0:
            try:
                from sare.knowledge.commonsense import CommonSenseBase
                from sare.perception.graph_builders import SentenceGraphBuilder, PlanGraphBuilder
                _cs_kb = CommonSenseBase()
                _cs_kb.load()
                _sent_builder = SentenceGraphBuilder()
                _plan_builder = PlanGraphBuilder()

                # Inject up to 3 commonsense inference problems
                _cs_triples = []
                for subj, edges in list(_cs_kb._forward.items())[:20]:
                    for rel, obj in edges:
                        _cs_triples.append((subj, rel, obj))
                import random as _rand
                _rand.shuffle(_cs_triples)
                for subj, rel, obj in _cs_triples[:3]:
                    try:
                        g = _sent_builder.build_from_triple(subj, rel, obj, hide="object")
                        if g and curriculum_gen:
                            curriculum_gen.add_seed(g)
                    except Exception:
                        pass

                # Inject 2 planning chain problems
                _plan_templates = [
                    [{"action": "study", "from": "ignorant", "to": "informed"},
                     {"action": "practice", "from": "informed", "to": "skilled"}],
                    [{"action": "earn", "from": "poor", "to": "has_funds"},
                     {"action": "buy", "from": "has_funds", "to": "has_item"}],
                ]
                for steps in _plan_templates[:2]:
                    try:
                        g = _plan_builder.build_plan(steps)
                        if g and curriculum_gen:
                            curriculum_gen.add_seed(g)
                    except Exception:
                        pass

                log.info("Cycle %d: injected commonsense + planning problems", cycle)
            except Exception as exc:
                log.debug("Non-math problem injection error: %s", exc)

        # Discover analogies across domains every 20 cycles
        if cycle % 20 == 0:
            try:
                from sare.memory.world_model import get_world_model
                wm = get_world_model()
                if len(wm._solve_history) >= 30:
                    new_analogies = wm.discover_analogies()
                    if new_analogies:
                        log.info("WorldModel: discovered %d new analogies", len(new_analogies))
            except Exception as exc:
                log.debug("discover_analogies error: %s", exc)

        # D.2 — Algorithmic problem injection every 20 cycles (no LLM)
        if cycle % 20 == 0:
            try:
                from sare.knowledge.problem_factory import get_problem_factory
                from sare.engine import load_problem
                _pf2 = get_problem_factory()
                _pf2_batch = _pf2.generate_batch(total=100, seed=cycle)
                _pf2_added = 0
                for _pf2_prob in _pf2_batch:
                    try:
                        _pg2 = load_problem(_pf2_prob["expression"])
                        if _pg2:
                            curriculum_gen.add_seed(_pg2)
                            _pf2_added += 1
                    except Exception:
                        pass
                if _pf2_added:
                    log.info("ProblemFactory cycle %d: injected %d new problems", cycle, _pf2_added)
            except Exception as exc:
                log.debug("ProblemFactory injection error: %s", exc)

        # Cross-domain transfer hypotheses every 20 cycles; save every 5
        if cycle % 5 == 0:
            try:
                from sare.transfer.engine import get_transfer_engine
                _te = get_transfer_engine()
                if cycle % 20 == 0:
                    _hyps = _te.generate_hypotheses()
                    _hyp_total = len(_hyps) if _hyps else 0
                    summary = _te.summary()
                    if _hyp_total:
                        log.info("[Daemon] TransferEngine: +%d new hypotheses | domains=%d roles=%d total_hyps=%d verified=%d",
                                 _hyp_total, summary["domains_analyzed"], summary["roles"],
                                 summary["hypotheses"], summary["verified"])

                # Test untested hypotheses with the live experiment runner (returns proof_steps)
                if cycle % 10 == 0:
                    _untested = [h for h in _te._hypotheses.values() if h.status == "untested"][:5]
                    _DOMAIN_TEST_PROBS = {
                        "algebra":     ["x + 0", "x * 1", "x + x", "2*x - x", "x^2 + 0"],
                        "arithmetic":  ["1 + 1", "2 * 3", "0 + 5", "4 * 1", "3 - 3"],
                        "logic":       ["p and True", "p or False", "not not p", "p and p", "p or p"],
                        "calculus":    ["x + 0", "x * 1", "x - x", "2*x - x", "x^2 * 1"],
                        "set_theory":  ["x + 0", "x * 1", "x - x", "0 + x", "x * 0"],
                        "code":        ["x + 0", "x * 1", "x - 0", "0 * x", "x + x"],
                        "qa":          ["x + 0", "1 + 1", "x * 1", "0 + x", "x - x"],
                    }
                    for _hyp in _untested:
                        def _solve_fn(expr, _runner=experiment_runner):
                            try:
                                from sare.engine import load_problem
                                _, _g = load_problem(expr)
                                if _g is None:
                                    return {"success": False, "delta": 0.0, "transforms": []}
                                class _P:
                                    pass
                                _p = _P(); _p.graph = _g; _p.id = "te_test"
                                _p.domain = "general"; _p.expression = expr; _p.py_graph = True
                                _r = _runner._run_single(_p)
                                _steps = list(getattr(_r, "proof_steps", None) or [])
                                return {"success": bool(getattr(_r, "solved", False)),
                                        "delta": float(getattr(_r, "energy_before", 0) - getattr(_r, "energy_after", 0)),
                                        "transforms": _steps}
                            except Exception:
                                return {"success": False, "delta": 0.0, "transforms": []}
                        _test_probs = _DOMAIN_TEST_PROBS.get(_hyp.target_domain, list(_SEEDS)[:5])
                        _te.test_hypothesis(_hyp, solve_fn=_solve_fn, test_problems=_test_probs)
                    _tested = sum(1 for h in _te._hypotheses.values() if h.status != "untested")
                    _verified = sum(1 for h in _te._hypotheses.values() if h.status == "verified")
                    if _untested:
                        log.info("[TransferEngine] Tested %d hypotheses → verified=%d total", _verified, len(_te._hypotheses))

                _te.save()
            except Exception as _te_exc:
                log.debug("TransferEngine sweep error: %s", _te_exc)

        # Item 5: Compositional problem generator — inject KB fact-pair inference problems every 20 cycles
        if cycle % 20 == 0:
            try:
                from sare.memory.world_model import get_world_model as _gwm_comp
                _wm_comp = _gwm_comp()
                _comp_injected = 0
                for _dom in list(getattr(_wm_comp, '_facts', {}).keys())[:3]:
                    _facts = getattr(_wm_comp, '_facts', {}).get(_dom, [])
                    if len(_facts) >= 2:
                        import random as _rand_comp
                        _f1, _f2 = _rand_comp.sample(_facts, 2)
                        # Build a compositional inference problem as an expression string
                        # Pattern: "if A has B and C is-a A, what property does C have?"
                        _expr = f"{_f1.get('subject','x')} + {_f2.get('subject','y')}"
                        try:
                            from sare.engine import load_problem as _lp_comp
                            _, _g_comp = _lp_comp(_expr)
                            if _g_comp is not None and curriculum_gen is not None:
                                class _CompProb:
                                    pass
                                _cp = _CompProb()
                                _cp.graph = _g_comp
                                _cp.id = f"comp_{_dom}_{cycle}"
                                _cp.domain = _dom
                                _cp.expression = _expr
                                _cp.py_graph = True
                                if hasattr(curriculum_gen, '_priority_queue'):
                                    curriculum_gen._priority_queue.append(_cp)
                                    _comp_injected += 1
                        except Exception:
                            pass
                if _comp_injected:
                    log.debug("[CompGen] Injected %d compositional problems from KB facts", _comp_injected)
            except Exception as _comp_exc:
                log.debug("Compositional problem generator error: %s", _comp_exc)

        # AnalogyTransfer: apply cross-domain rules to ConceptRegistry every 30 cycles
        if cycle % 30 == 0:
            try:
                from sare.causal.analogy_transfer import AnalogyTransfer
                _at = AnalogyTransfer(concept_registry)
                _at_transfers = _at.transfer_all_domains()
                if _at_transfers:
                    _at_applied, _at_skipped = _at.apply_to_registry(_at_transfers)
                    if _at_applied:
                        log.info("[AnalogyTransfer] Applied %d new cross-domain rules (%d already known)",
                                 _at_applied, _at_skipped)
            except Exception as _at_exc:
                log.debug("AnalogyTransfer sweep error: %s", _at_exc)

        # T3-3: EWC-lite consolidation every 50 cycles
        if cycle % 50 == 0 and forgetting_prevention is not None:
            try:
                forgetting_prevention.consolidate()
                forgetting_prevention.save()
            except Exception:
                pass

        # Belief expiry every 50 cycles — decay stale beliefs older than 24h
        if cycle % 50 == 0:
            try:
                from sare.memory.world_model import get_world_model as _gwm_exp
                _decayed = _gwm_exp().expire_stale_beliefs(max_age_seconds=86400)
                if _decayed:
                    log.info("[BeliefExpiry] Decayed %d stale beliefs (>24h old)", _decayed)
            except Exception as _exp_exc:
                log.debug("BeliefExpiry error: %s", _exp_exc)

        # Knowledge corpus refresh every 100 cycles — picks up new benchmark/doc files
        if cycle % 100 == 0:
            try:
                from sare.knowledge.knowledge_ingester import KnowledgeIngester
                _ki_refresh = KnowledgeIngester()
                _ki_refresh.ingest_local_corpus()  # idempotent — skips already-seen files
                log.debug("[KnowledgeIngester] Periodic corpus refresh at cycle %d", cycle)
            except Exception as _ki_exc:
                log.debug("KnowledgeIngester periodic refresh error: %s", _ki_exc)

        # Schema generalization: deduplicate similar proof patterns every 50 cycles
        if cycle % 50 == 0:
            try:
                from sare.cognition.schema_matcher import get_schema_matcher
                _sm = get_schema_matcher()
                if hasattr(_sm, "induce_generalizations"):
                    _induction_stats = _sm.induce_generalizations()
                    if _induction_stats.get("merged", 0) > 0:
                        log.info("[Schema] Induced %d generalizations (cycle %d, %d→%d schemas)",
                                 _induction_stats["merged"], cycle,
                                 _induction_stats["total_before"], _induction_stats["total_after"])
            except Exception as _sm_exc:
                log.debug("[Schema] Induction error: %s", _sm_exc)

        # Memory GC every 50 cycles — cap large files
        if cycle % 50 == 0:
            import os as _os
            _gc_targets = [
                ("data/memory/episodes.jsonl", 10_000),
                ("data/memory/plan_traces.jsonl", 5_000),
                ("data/memory/frontier.jsonl", 3_000),
            ]
            for _gc_path, _max_lines in _gc_targets:
                try:
                    _p = REPO_ROOT / _gc_path
                    if _p.exists():
                        _lines = _p.read_text(encoding="utf-8", errors="replace").splitlines()
                        if len(_lines) > _max_lines:
                            _kept = _lines[-_max_lines:]
                            _tmp = _p.with_suffix(".tmp")
                            _tmp.write_text("\n".join(_kept) + "\n", encoding="utf-8")
                            _os.replace(_tmp, _p)
                            log.info("[GC] Trimmed %s: %d → %d lines", _p.name, len(_lines), len(_kept))
                except Exception as _gc_exc:
                    log.debug("Memory GC error for %s: %s", _gc_path, _gc_exc)

        if not _stop[0]:
            time.sleep(interval)

    log.info("learn_daemon stopped after %d cycles.", cycle)

    # Shutdown parallel learners
    if _auto_trainer:
        try:
            _auto_trainer.stop()
        except Exception:
            pass
    if _mal and getattr(_mal, '_running', False):
        try:
            _mal.stop()
        except Exception:
            pass

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SARE-HX autonomous learning daemon")
    parser.add_argument("--interval",   type=float, default=30.0,
                        help="Seconds between batches (default: 30)")
    parser.add_argument("--batch-size", type=int,   default=5,
                        help="Problems per batch (default: 5)")
    parser.add_argument("--verbose",    action="store_true",
                        help="Enable DEBUG logging")
    args = parser.parse_args()

    return run_daemon(
        interval=args.interval,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    raise SystemExit(main())
