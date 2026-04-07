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

    def attempt_learning_problem(
        self,
        problem_text: str,
        expected_answer,
        problem_type: str,
        context: str = "",
        metadata=None,
    ):
        """Delegate to GeneralSolver so AutonomousTrainer general sources are enabled."""
        try:
            from sare.cognition.general_solver import GeneralSolver
            if not hasattr(self, "_general_solver"):
                self._general_solver = GeneralSolver()
            _meta = dict(metadata or {})
            _meta.setdefault("skip_llm", True)  # don't block the daemon cycle with inline LLM
            return self._general_solver.attempt_learning_problem(
                problem_text,
                expected_answer,
                problem_type,
                context=context,
                metadata=_meta,
            )
        except Exception as exc:
            # Return a minimal object so AutonomousTrainer doesn't crash
            try:
                from sare.cognition.general_solver import GeneralSolveResult
                r = GeneralSolveResult.__new__(GeneralSolveResult)
                r.solved = False
                r.correct = False
                r.confidence = 0.0
                r.elapsed_ms = 0.0
                r.answer = ""
                r.expected_answer = expected_answer
                return r
            except Exception:
                class _R:
                    solved = False; correct = False; confidence = 0.0
                    elapsed_ms = 0.0; answer = ""; expected_answer = None
                return _R()

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
    # ── General / non-math seeds (Grade 5-8 general intelligence) ────────────
    # Fractions
    "a / b + c / b",          # fraction add same denom
    "a / b - c / b",          # fraction sub same denom
    "a / 1",                  # whole divide
    "n / d",                  # generic fraction
    # Proportions / ratios
    "a / b = c / d",          # cross-multiply form
    # Basic science expressions
    "F = m * a",
    "E = m * c * c",
    "V = I * R",
    "p = m * v",
    "W = F * d",
    "P = W / t",
    # Logic / reasoning patterns
    "A and (B or C)",
    "not (A and B)",
    "not (A or B)",
    "A implies B",
    "if A then B",
    # Geometry
    "area = pi * r * r",
    "C = 2 * pi * r",
    "perimeter = 2 * l + 2 * w",
    "a * a + b * b = c * c",   # Pythagorean
    # Statistics
    "mean = sum / n",
    "variance = sum_sq / n",
    # Chemical / molar
    "n = m / M",
    "PV = nRT",
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
            budget_seconds=1.5,
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


_PID_FILE       = REPO_ROOT / "data" / "memory" / "daemon.pid"
_HEARTBEAT_FILE = REPO_ROOT / "data" / "memory" / "daemon_heartbeat.json"


def run_daemon(interval: float = 15.0, batch_size: int = 10, verbose: bool = False, turbo: bool = False, turbo_learn: bool = False):
    """Main daemon loop."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # PID guard — prevent multiple daemon instances from running simultaneously
    import os as _pid_os
    try:
        _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        if _PID_FILE.exists():
            try:
                _old_pid = int(_PID_FILE.read_text().strip())
                _pid_os.kill(_old_pid, 0)   # signal 0: check existence without sending
                log.warning("learn_daemon already running (pid=%d) — exiting to avoid conflicts.", _old_pid)
                return 0
            except (ProcessLookupError, ValueError):
                pass   # stale PID file — ok to overwrite
        _PID_FILE.write_text(str(_pid_os.getpid()))
    except Exception as _pid_exc:
        log.debug("PID guard error (non-fatal): %s", _pid_exc)

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

    # ── External datasets (curated + GSM8K) ───────────────────────────────────
    try:
        from sare.knowledge.external_datasets import get_dataset_ingester
        _ext = get_dataset_ingester()
        _ext.load_curated()
        _ext.load_gsm8k_sample(max_problems=8500)
        _ext.load_commonsenseqa(max_problems=200)
        _ext.load_aqua(max_problems=1000)
        log.info("External datasets staged for CurriculumBroker (summary: %s)", _ext.summary())
    except Exception as _ext_exc:
        log.debug("External dataset ingestion error: %s", _ext_exc)

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

    # A.2.5 — Immediate Wikipedia sweep for weakest domains (startup, background)
    try:
        import threading as _wiki_startup_thread
        from sare.learning.wikipedia_ingester import WikipediaIngester, DOMAIN_TOPICS
        def _startup_wiki_sweep():
            try:
                wi = WikipediaIngester()
                # Sweep 3 knowledge-heavy domains right at startup
                for _startup_domain in ["psychology", "history", "biology"]:
                    if _startup_domain not in DOMAIN_TOPICS:
                        continue
                    cached = len(wi.get_cached_topics(_startup_domain))
                    if cached < 5:   # only sweep if we have few cached topics
                        n = wi.sweep_domain(_startup_domain, max_topics=20)
                        log.info("[Wikipedia] Startup sweep: domain=%s facts=%d", _startup_domain, n)
            except Exception as _ws_exc:
                log.debug("Startup Wikipedia sweep error: %s", _ws_exc)
        _wiki_startup_thread.Thread(target=_startup_wiki_sweep, daemon=True, name="wiki-startup").start()
        log.info("[Wikipedia] Startup sweep launched for psychology/history/biology")
    except Exception as _wsi_exc:
        log.debug("Wikipedia startup sweep init error: %s", _wsi_exc)

    # A.2.6 — Bootstrap neural learner with domain templates (genuine learning, not memorization)
    _neural_learner = None
    try:
        from sare.neuro.neural_learner import get_neural_learner as _get_nl
        _neural_learner = _get_nl()
        _neural_learner.load()
        log.info("[NeuralLearner] Ready: %d memories, backend=%s",
                 len(_neural_learner), _neural_learner.get_stats()["backend"])
        # Bootstrap: teach from curriculum templates so it starts with basic knowledge
        if len(_neural_learner) < 50:
            from sare.learning.general_curriculum import (
                _PLANNING_TEMPLATES, _SOCIAL_TEMPLATES, _ANALOGY_TEMPLATES,
                _FACTUAL_TEMPLATES, _REASONING_TEMPLATES, _SCIENCE_TEMPLATES,
                _LANGUAGE_TEMPLATES, _HISTORY_TEMPLATES, _GEOGRAPHY_TEMPLATES,
                _BIOLOGY_TEMPLATES, _ECONOMICS_TEMPLATES, _PSYCHOLOGY_TEMPLATES,
            )
            _nl_boot_map = {
                'planning': _PLANNING_TEMPLATES, 'analogy': _ANALOGY_TEMPLATES,
                'social': _SOCIAL_TEMPLATES, 'factual': _FACTUAL_TEMPLATES,
                'reasoning': _REASONING_TEMPLATES, 'science': _SCIENCE_TEMPLATES,
                'language': _LANGUAGE_TEMPLATES, 'history': _HISTORY_TEMPLATES,
                'geography': _GEOGRAPHY_TEMPLATES, 'biology': _BIOLOGY_TEMPLATES,
                'economics': _ECONOMICS_TEMPLATES, 'psychology': _PSYCHOLOGY_TEMPLATES,
            }
            _nl_boot_count = 0
            for _nl_dom, _nl_templates in _nl_boot_map.items():
                for _nl_q, _nl_a in _nl_templates:
                    _neural_learner.learn(_nl_q, _nl_a, _nl_dom, correct=True)
                    _nl_boot_count += 1
            _neural_learner.flush()   # wait for training thread to process all queued items
            _neural_learner.save()
            log.info("[NeuralLearner] Bootstrap complete: taught %d templates, memory=%d",
                     _nl_boot_count, len(_neural_learner))
    except Exception as _nl_exc:
        log.debug("NeuralLearner startup error: %s", _nl_exc)

    # A.2.7 — Stream Wikipedia dump into world model (background, resumes from checkpoint)
    try:
        import threading as _wiki_dump_threading
        from sare.learning.wikipedia_dump_ingester import WikipediaDumpIngester as _WikiDumpIngester
        _wiki_dump = _WikiDumpIngester()
        if _wiki_dump.is_available():
            _wiki_status = _wiki_dump.get_status()
            log.info("[WikiDump] Dump available (%sMB), %d articles processed so far",
                     _wiki_status["dump_size_mb"], _wiki_status["articles_processed"])
            def _run_wiki_dump():
                try:
                    _n = _wiki_dump.ingest_batch(max_articles=1000)
                    log.info("[WikiDump] Startup batch done: %d facts ingested", _n)
                except Exception as _wde:
                    log.debug("[WikiDump] Startup batch error: %s", _wde)
            _wiki_dump_threading.Thread(target=_run_wiki_dump, daemon=True, name="wiki-dump-startup").start()
        else:
            log.debug("[WikiDump] Dump not available, skipping")
    except Exception as _wiki_dump_exc:
        log.debug("[WikiDump] Init error: %s", _wiki_dump_exc)

    # A.2.8 — LiveWorld: persistent interactive world the system explores
    _live_world = None
    _lw_wm = None
    _lw_cs = None
    try:
        from sare.world.live_world import get_live_world as _get_live_world
        from sare.memory.world_model import get_world_model as _gwm_lw
        from sare.knowledge.commonsense import get_commonsense_base as _gcb_lw
        _live_world = _get_live_world()
        _lw_wm = _gwm_lw()
        _lw_cs = _gcb_lw()
        _lw_summary = _live_world.summary()
        log.info("[LiveWorld] Ready: %d objects, %d rules, %d prior discoveries, %d pairs untried",
                 _lw_summary["objects"], _lw_summary["rules"],
                 _lw_summary["discoveries"], _lw_summary["untried_remaining"])
    except Exception as _lw_exc:
        log.debug("[LiveWorld] Init error: %s", _lw_exc)

    # A.3 — Augment commonsense KB from ConceptNet (one-time at startup)
    try:
        from sare.knowledge.commonsense import get_commonsense_base
        _kb = get_commonsense_base()
        if len(_kb._forward) < 60:
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
        _at_interval = 1.0 if turbo else 5.0
        _at_batch = 50 if turbo else 20
        _at_workers = 8 if turbo else 4
        _auto_trainer = AutonomousTrainer(
            interval_seconds=_at_interval,
            batch_size=_at_batch,
            max_workers=_at_workers,
            symbolic_only=False,   # ← enable general sources (word, code, commonsense, language)
        )
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
        # Wire integration points for concept grounding
        try:
            from sare.concept.concept_graph import get_concept_graph
            from sare.cognition.global_workspace import get_global_workspace
            _embodied_agent.wire(
                concept_graph=get_concept_graph(),
                global_workspace=get_global_workspace(),
            )
        except Exception:
            pass
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

    # EmbodiedAgent warm-up: run in background thread (was synchronous 50 episodes = ~10 min)
    if _embodied_agent is not None:
        import threading as _ea_warmup_thread
        def _ea_warmup(_ea=_embodied_agent):
            try:
                # 10 short episodes instead of 50 full ones — remainder accumulates during cycles
                for _i in range(10):
                    _ea.run_episode(max_steps=20, goal="explore")
                log.info("[EmbodiedAgent] Background warmup complete (10 episodes)")
            except Exception as _ea_warm_exc:
                log.debug("EmbodiedAgent warmup error: %s", _ea_warm_exc)
        _ea_warmup_thread.Thread(target=_ea_warmup, daemon=True, name="ea-warmup").start()
        log.info("[EmbodiedAgent] Warmup started in background (10 episodes × 20 steps)")

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

    # CreativityEngine — Default Mode Network creative insight generation
    _creativity_engine = None
    try:
        from sare.neuro.creativity_engine import get_creativity_engine
        _creativity_engine = get_creativity_engine()
        _creativity_engine.COOLDOWN_SECONDS = 300  # 5 min (was 30 min)
        log.info("CreativityEngine ready (cooldown=%ds, dreams=%d)",
                 _creativity_engine.COOLDOWN_SECONDS, len(_creativity_engine._dreams))
    except Exception as _ce_exc:
        log.debug("CreativityEngine unavailable: %s", _ce_exc)

    # SymbolCreator — invents new mathematical primitives when stuck
    _symbol_creator = None
    try:
        from sare.neuro.symbol_creator import get_symbol_creator
        _symbol_creator = get_symbol_creator()
        log.info("SymbolCreator ready (invented=%d)", _symbol_creator.get_status().get("total_invented", 0))
    except Exception as _sc_exc:
        log.debug("SymbolCreator unavailable: %s", _sc_exc)

    # TheoryOfMindEngine — social reasoning + agent modeling
    _theory_of_mind = None
    try:
        from sare.social.theory_of_mind import TheoryOfMindEngine
        _theory_of_mind = TheoryOfMindEngine()
        _theory_of_mind.load()
        # Seed canonical agents if not already present
        _canonical_agents = {
            "teacher": {
                "desc": "A human teacher who knows mathematics and wants to help",
                "beliefs": [("Mathematics has consistent rules", True), ("The student can learn", True)],
                "desires": ["teach correct rules", "verify student understanding"],
            },
            "student": {
                "desc": "A learner who makes mistakes but is eager",
                "beliefs": [("Practice improves skill", True), ("Some problems are hard", True)],
                "desires": ["solve problems correctly", "understand patterns"],
            },
            "adversary": {
                "desc": "A skeptic who challenges claims and tests robustness",
                "beliefs": [("Rules should be proven", True), ("Exceptions may exist", True)],
                "desires": ["find counter-examples", "test edge cases"],
            },
            "collaborator": {
                "desc": "A peer who shares knowledge and works together",
                "beliefs": [("Sharing knowledge helps everyone", True), ("Different perspectives are valuable", True)],
                "desires": ["share useful rules", "combine insights"],
            },
        }
        for _aid, _ainfo in _canonical_agents.items():
            if _theory_of_mind.get_agent(_aid) is None:
                _ms = _theory_of_mind.register_agent(_aid, _ainfo["desc"])
                for _b_content, _b_truth in _ainfo["beliefs"]:
                    _ms.add_belief(_b_content, _b_truth, 0.9)
                for _d in _ainfo["desires"]:
                    _ms.add_desire(_d)
        _theory_of_mind.save()
        log.info("TheoryOfMindEngine ready (%d agents)", len(_theory_of_mind._agents))
    except Exception as _tom_exc:
        log.debug("TheoryOfMindEngine unavailable: %s", _tom_exc)

    # DialogueManager — conversational learning
    _dialogue_manager = None
    try:
        from sare.social.dialogue_manager import get_dialogue_manager
        _dialogue_manager = get_dialogue_manager()
        log.info("DialogueManager ready (%d sessions)", len(_dialogue_manager._sessions))
    except Exception as _dm_exc:
        log.debug("DialogueManager unavailable: %s", _dm_exc)

    # PerceptionEngine — auto-ingest data files for broader learning
    _perception_engine = None
    try:
        from sare.perception.perception_engine import PerceptionEngine
        _perception_engine = PerceptionEngine()
        # Auto-ingest external datasets and key memory files at startup
        from pathlib import Path as _PEPath
        _data_root = _PEPath(__file__).resolve().parent / "python" / "sare"
        _data_mem = _PEPath(__file__).resolve().parent / "data"
        _pe_ingested = 0
        # Ingest external datasets
        _ext_dir = _data_mem / "external_datasets"
        if _ext_dir.exists():
            for _f in list(_ext_dir.glob("*.jsonl"))[:3]:
                try:
                    _text = _f.read_text(errors="replace")[:50000]  # limit size
                    _pr = _perception_engine.ingest_text(_text, source=str(_f.name))
                    _pe_ingested += len(_pr.problems_extracted)
                except Exception:
                    pass
        # Ingest key memory files for fact extraction
        for _mem_file in ["world_hypotheses.json", "promoted_rules.json", "concept_graph.json"]:
            _mf = _data_mem / "memory" / _mem_file
            if _mf.exists():
                try:
                    _text = _mf.read_text(errors="replace")[:30000]
                    _pr = _perception_engine.ingest_text(_text, source=_mem_file)
                    _pe_ingested += len(_pr.problems_extracted)
                except Exception:
                    pass
        log.info("PerceptionEngine ready (startup ingested %d problems from data files)", _pe_ingested)
    except Exception as _pe_exc:
        log.debug("PerceptionEngine unavailable: %s", _pe_exc)

    # ── ConceptNet bulk knowledge ingest ─────────────────────────────────────
    try:
        from sare.knowledge.commonsense import get_commonsense_base
        _cs_base = get_commonsense_base()
        if _cs_base.total_facts() < 10_000:
            import urllib.request as _cn_urllib
            from pathlib import Path as _CNPath
            _cn_cache = _CNPath(__file__).resolve().parent / "data" / "external_datasets" / "conceptnet-assertions-5.7.0.csv.gz"
            if not _cn_cache.exists():
                _cn_url = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
                log.info("Downloading ConceptNet bulk CSV (~30MB)...")
                try:
                    _cn_req = _cn_urllib.Request(_cn_url, headers={"User-Agent": "SARE-HX/1.0"})
                    with _cn_urllib.urlopen(_cn_req, timeout=120) as _cn_resp:
                        _cn_cache.parent.mkdir(parents=True, exist_ok=True)
                        _cn_cache.write_bytes(_cn_resp.read())
                    log.info("ConceptNet CSV downloaded (%d bytes)", _cn_cache.stat().st_size)
                except Exception as _cn_dl_exc:
                    log.warning("ConceptNet download failed: %s — falling back to LLM expansion", _cn_dl_exc)
                    try:
                        _llm_added = _cs_base.augment_from_llm(n_facts=500)
                        log.info("LLM fallback: +%d commonsense facts", _llm_added)
                    except Exception:
                        pass
            if _cn_cache.exists():
                import threading as _cn_threading
                def _bg_conceptnet_ingest():
                    try:
                        _added = _cs_base.bulk_ingest_conceptnet_csv(str(_cn_cache), max_facts=600_000)
                        log.info("ConceptNet background ingest done: +%d facts, total: %d", _added, _cs_base.total_facts())
                    except Exception as _e:
                        log.warning("ConceptNet background ingest failed: %s", _e)
                _cn_thread = _cn_threading.Thread(target=_bg_conceptnet_ingest, daemon=True, name="conceptnet-ingest")
                _cn_thread.start()
                log.info("ConceptNet bulk ingest started in background thread")
        else:
            log.info("CommonSenseBase already has %d facts — skipping ConceptNet ingest", _cs_base.total_facts())
    except Exception as _cs_exc:
        log.debug("ConceptNet ingest setup error: %s", _cs_exc)

    # ── Knowledge Graph bulk seed from CommonSenseBase ────────────────────────
    try:
        from sare.memory.world_model import get_world_model as _get_wm
        _wm = _get_wm()
        _wm_fact_count = sum(len(v) for v in _wm.get_all_facts().values())
        if _wm_fact_count < 2000:
            import threading as _kg_threading
            def _bg_kg_seed():
                try:
                    from sare.knowledge.commonsense import get_commonsense_base
                    _csb = get_commonsense_base()
                    _triples = _csb.to_triples()[:5000]
                    _seeded = 0
                    _REL_TO_DOMAIN = {
                        "IsA": "taxonomy", "HasA": "composition", "PartOf": "composition",
                        "Causes": "causality", "UsedFor": "function", "CapableOf": "capability",
                        "HasProperty": "properties", "LocatedAt": "spatial",
                        "RequiredFor": "prerequisites", "Enables": "causality",
                        "OppositeOf": "semantics", "RelatedTo": "semantics",
                    }
                    for subj, rel, obj in _triples:
                        domain = _REL_TO_DOMAIN.get(rel, "commonsense")
                        fact_str = f"{subj} {rel} {obj}"
                        _wm.add_fact(domain=domain, fact=fact_str, confidence=0.85, source="conceptnet")
                        _seeded += 1
                    log.info("KG bulk seed done: +%d facts from CommonSenseBase", _seeded)
                except Exception as _e:
                    log.debug("KG bulk seed error: %s", _e)
            _kg_thread = _kg_threading.Thread(target=_bg_kg_seed, daemon=True, name="kg-seed")
            _kg_thread.start()
            log.info("Knowledge graph bulk seed started in background (%d existing facts)", _wm_fact_count)
        else:
            log.info("WorldModel already has %d facts — skipping KG seed", _wm_fact_count)
    except Exception as _kg_exc:
        log.debug("KG seed setup error: %s", _kg_exc)

    # CausalChainDetector — multi-step causal chain analysis
    _chain_detector = None
    try:
        from sare.causal.chain_detector import CausalChainDetector
        _chain_detector = CausalChainDetector()
        # Wire into CausalRollout if available
        if _causal_rollout is not None:
            _causal_rollout.wire(chain_detector=_chain_detector)
        log.info("CausalChainDetector ready")
    except Exception as _cd_exc:
        log.debug("CausalChainDetector unavailable: %s", _cd_exc)

    # AbductiveRanker — abductive reasoning for causal explanation
    _abductive_ranker = None
    try:
        from sare.causal.abductive_ranker import AbductiveRanker
        _abductive_ranker = AbductiveRanker(concept_registry=concept_registry)
        log.info("AbductiveRanker ready")
    except Exception as _ar_exc:
        log.debug("AbductiveRanker unavailable: %s", _ar_exc)

    # Stuck-problem tracker for TransformSynthesizer
    # {domain: [expr1, expr2, ...]} — accumulates unsolved exprs; fires synthesis at _SYNTH_TRIGGER
    _stuck_by_domain: dict = {}
    _SYNTH_TRIGGER = 5    # synthesize after this many stuck problems per domain (was 15)
    _synth_cooldown: dict = {}  # {domain: last_synth_cycle}
    _SYNTH_COOLDOWN_CYCLES = 10  # don't re-synthesize a domain within 10 cycles

    # Book ingestion state — rolling window per file
    _book_offsets: dict = {}   # {filename: char_offset}
    _book_dir = REPO_ROOT / "python" / "data" / "books"
    if not _book_dir.exists():
        _book_dir = REPO_ROOT / "data" / "books"

    # DomainProblemGenerator for non-math curricula
    _domain_gen = None
    try:
        from sare.curiosity.domain_problem_generator import get_domain_generator
        _domain_gen = get_domain_generator()
        log.info("DomainProblemGenerator ready (domains: %s, pool: %s)",
                 _domain_gen.available_domains()[:6], _domain_gen.pool_stats())
    except Exception as _dg_exc:
        log.debug("DomainProblemGenerator unavailable: %s", _dg_exc)

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

    # ── Turbo mode: thread pool for async periodic tasks ───────────────────
    _turbo_pool = None
    if turbo:
        import concurrent.futures as _turbo_cf
        _turbo_pool = _turbo_cf.ThreadPoolExecutor(max_workers=4, thread_name_prefix="turbo-periodic")
        # B4: Increase fast-path ratio — only 1-in-200 problems get full cognitive path
        experiment_runner._fast_path_ratio = 200
        # Reduce search budget for throughput: 8s → 1s, beam 8 → 4
        experiment_runner.budget_seconds = 1.0
        experiment_runner.beam_width = 4
        log.info("TURBO MODE: interval=%.1fs, batch=%d, fast_ratio=200, budget=1.0s, beam=4, async periodic tasks enabled", interval, batch_size)
    elif turbo_learn:
        # Turbo-Learn: 3× throughput while keeping real learning on 1-in-20 solves
        experiment_runner._fast_path_ratio = 20
        experiment_runner.budget_seconds = 2.0
        experiment_runner.beam_width = 6
        log.info("TURBO-LEARN MODE: interval=%.1fs, batch=%d, fast_ratio=20, budget=2.0s, beam=6 — learning stays active", interval, batch_size)

    cycle = 0
    _benchmark_probe_interval_s = 30.0 * 60.0
    _last_benchmark_probe_at = 0.0
    while not _stop[0]:
        cycle += 1

        # Heavy tasks: skip in turbo (every 500), reduce in turbo_learn (every 15)
        # turbo_learn optimizes for KB growth — heavy symbolic tasks slow that down
        if turbo:
            _run_heavy = (cycle % 500 == 0)
        elif turbo_learn:
            _run_heavy = (cycle % 15 == 0)
        else:
            _run_heavy = True

        # ── Homeostasis-driven batch adaptation ──────────────────────
        _effective_batch = batch_size
        _behavior = "explore_new_domain"
        if _homeostasis and not turbo:
            try:
                _homeostasis.tick()
                _behavior = _homeostasis.get_behavior_recommendation()
                if _behavior == "consolidate_memory":
                    # Consolidation mode: smaller batch, save more often
                    _effective_batch = max(2, batch_size // 2)
                    log.info("Homeostasis → CONSOLIDATE (batch=%d)", _effective_batch)
                elif _behavior == "explore_new_domain":
                    # Exploration mode: try harder problems
                    _effective_batch = batch_size + 10
                    log.info("Homeostasis → EXPLORE (batch=%d)", _effective_batch)
                elif _behavior == "deepen_weak_domain":
                    # Focus mode: normal batch, curriculum will focus
                    log.info("Homeostasis → DEEPEN weak domains")
                elif _behavior == "seek_human_input":
                    # Throttle: fire LLM teacher at most once every 10 cycles
                    if _llm_teacher is not None and cycle % 10 == 0:
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
            _batch_t0 = time.time()
            results = experiment_runner.run_batch(n=_effective_batch)
            _batch_elapsed = time.time() - _batch_t0
            solved  = sum(1 for r in results if r.solved)
            promoted = sum(1 for r in results if r.rule_promoted)
            # Track C++ vs Python backend usage for throughput visibility
            _cpp_backend = getattr(experiment_runner, '_cpp_ready', False)
            _rate = len(results) / _batch_elapsed if _batch_elapsed > 0 else 0
            log.info(
                "Cycle %d [%s]: %d/%d solved, %d promoted | %.0f prob/s | cpp=%s",
                cycle, _behavior, solved, len(results), promoted,
                _rate, "on" if _cpp_backend else "off",
            )
            # ── Heartbeat: write after every cycle so watchdog can detect stuck state ──
            try:
                import os as _hb_os
                _hb = {
                    "ts": time.time(),
                    "cycle": cycle,
                    "pid": _hb_os.getpid(),
                    "solved": solved,
                    "total": len(results),
                    "promoted": promoted,
                    "rate": round(_rate, 1),
                    "behavior": _behavior,
                    "interval_s": interval,
                }
                _hb_tmp = _HEARTBEAT_FILE.with_suffix(".tmp")
                _hb_tmp.write_text(json.dumps(_hb))
                _hb_os.replace(str(_hb_tmp), str(_HEARTBEAT_FILE))
            except Exception:
                pass
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

        # Feed solve results into TransferEngine so it can discover cross-domain roles
        try:
            from sare.transfer.engine import get_transfer_engine
            _te = get_transfer_engine()
            for _r in results:
                if _r.solved and _r.proof_steps:
                    _te.observe(_r.proof_steps, getattr(_r, "domain", "general"), success=True)
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
            # Every 10 cycles: run imagination in background thread
            if _run_heavy and cycle % 10 == 0:
                import threading as _exp_thread
                def _do_imagine(_el=_experiential_learner, _cyc=cycle):
                    try:
                        _n = _el.imagine_batch(n=30)
                        if _n:
                            log.info("[Experience] Imagined %d counterfactual beliefs (cycle %d)", _n, _cyc)
                    except Exception:
                        pass
                _exp_thread.Thread(target=_do_imagine, daemon=True, name="imagine-batch").start()

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
            if _run_heavy and cycle % 15 == 0:
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

        # ── Analogy Transfer: sweep all domains every 10 cycles (background) ──
        if _run_heavy and cycle % 10 == 0:
            import threading as _at_threading
            def _do_analogy_sweep(_cyc=cycle):
                try:
                    from sare.causal.analogy_transfer import AnalogyTransfer
                    from sare.memory.concept_seed_loader import get_concept_registry as _gcr
                    _at = AnalogyTransfer(_gcr())
                    _at_results = _at.sweep_all_domains()
                    if _at_results:
                        log.info("[Transfer] cycle %d: %d new analogy transfers", _cyc, len(_at_results))
                except Exception as _ate:
                    log.debug("Analogy transfer error: %s", _ate)
            _at_threading.Thread(target=_do_analogy_sweep, daemon=True, name="analogy-sweep").start()

        # ── LiveWorld: autonomous world exploration every cycle ──────────────
        if _live_world is not None:
            import threading as _lw_threading
            def _do_live_world(_lw=_live_world, _wm=_lw_wm, _cs=_lw_cs, _cyc=cycle):
                try:
                    new_facts = _lw.explore_step()
                    if new_facts:
                        for _f in new_facts:
                            _lw.feed_to_memory(_f, world_model=_wm, commonsense=_cs)
                            if _f.novel:
                                log.info("[LiveWorld] Discovered: %s %s %s (domain=%s conf=%.2f)",
                                         _f.subject, _f.predicate, _f.obj, _f.domain, _f.confidence)
                    # Hypothesis-driven exploration every 50 cycles
                    if _cyc % 50 == 0:
                        _hyp = _lw.hypothesize_and_test(world_model=_wm)
                        for _hf in _hyp:
                            _lw.feed_to_memory(_hf, world_model=_wm, commonsense=_cs)
                        if _hyp:
                            log.info("[LiveWorld] Hypothesis verified: %d novel facts", len(_hyp))
                    _lw.save_state()
                except Exception as _lwe:
                    log.debug("[LiveWorld] Explore error: %s", _lwe)
            _lw_threading.Thread(target=_do_live_world, daemon=True, name="live-world").start()

        # ── MultiAgentArena: competitive self-play every 50 cycles ──────────
        if _run_heavy and cycle % 50 == 0 and _arena is not None:
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
        if _run_heavy and cycle % 25 == 0 and promoted > 0:
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
        if _run_heavy and cycle % 15 == 0 and neural_perception is not None:
            try:
                # Perceive recent expressions to update WorldModel beliefs
                _recent = [r for r in results if hasattr(r, 'expression') and r.expression][:5]
                for _r in _recent:
                    neural_perception.perceive_problem(_r.expression, getattr(_r, 'domain', 'general'))
            except Exception:
                pass

        # ── S33: PerceptionEngine — ingest solve results as text every 30 cycles ─
        if _run_heavy and _perception_engine is not None and cycle % 30 == 0:
            try:
                _pe_text_parts = []
                for _r in results[:10]:
                    _expr = getattr(_r, 'expression', '') or getattr(_r, 'problem_id', '')
                    _dom = getattr(_r, 'domain', 'general')
                    _solved = getattr(_r, 'solved', False)
                    _ps = getattr(_r, 'proof_steps', []) or []
                    if _expr:
                        _pe_text_parts.append(
                            f"simplify: {_expr} domain={_dom} solved={_solved} "
                            f"transforms={','.join(str(s) for s in _ps[:5])}"
                        )
                if _pe_text_parts:
                    _pe_batch = "\n".join(_pe_text_parts)
                    _pe_result = _perception_engine.ingest_text(_pe_batch, source=f"cycle_{cycle}")
                    if _pe_result.problems_extracted:
                        log.info("[Perception] Cycle %d: extracted %d problems, %d facts from solve results",
                                 cycle, len(_pe_result.problems_extracted), len(_pe_result.facts_extracted))
            except Exception as _pe_exc:
                log.debug("PerceptionEngine cycle error: %s", _pe_exc)

        # ── Book ingestion — rolling 20k-char window every 25 cycles ──────────
        if _run_heavy and _perception_engine is not None and cycle % 25 == 0 and _book_dir.exists():
            try:
                _book_files = sorted(_book_dir.glob("*.txt"))
                for _bpath in _book_files:
                    try:
                        _b_offset = _book_offsets.get(_bpath.name, 0)
                        _b_text_full = _bpath.read_text(encoding="utf-8", errors="replace")
                        _b_chunk = _b_text_full[_b_offset:_b_offset + 20_000]
                        if not _b_chunk.strip():
                            # Book exhausted — restart from beginning for continuous learning
                            _book_offsets[_bpath.name] = 0
                            _b_chunk = _b_text_full[:20_000]
                        if _b_chunk.strip():
                            _b_result = _perception_engine.ingest_textbook(
                                _b_chunk, source=_bpath.stem)
                            _b_probs = getattr(_b_result, 'problems_extracted', []) or []
                            _b_facts = getattr(_b_result, 'facts_extracted', []) or []
                            try:
                                from sare.knowledge.knowledge_ingester import KnowledgeIngester
                                _book_ingest = KnowledgeIngester().ingest_reading_chunk(
                                    f"{_bpath.stem} reading",
                                    _b_chunk,
                                    domain="language",
                                    source_path=str(_bpath.relative_to(REPO_ROOT) if REPO_ROOT in _bpath.parents else _bpath),
                                )
                                if _book_ingest.get("questions"):
                                    log.info(
                                        "[Books] %s → +%d comprehension tasks",
                                        _bpath.stem,
                                        int(_book_ingest.get("questions", 0)),
                                    )
                            except Exception as _book_q_exc:
                                log.debug("[Books] Comprehension extraction error for %s: %s", _bpath.name, _book_q_exc)
                            # Inject extracted problems into curriculum
                            for _bp in _b_probs[:10]:
                                try:
                                    from sare.engine import load_problem as _lp_book
                                    _bexpr = getattr(_bp, 'expression', str(_bp))
                                    _, _bg = _lp_book(_bexpr)
                                    if _bg is not None and curriculum_gen is not None:
                                        curriculum_gen.add_seed(_bg)
                                except Exception:
                                    pass
                            _book_offsets[_bpath.name] = _b_offset + 20_000
                            if _b_probs or _b_facts:
                                log.info("[Books] %s (+%d chars): %d problems, %d facts extracted",
                                         _bpath.stem, 20_000, len(_b_probs), len(_b_facts))
                    except Exception as _bf_exc:
                        log.debug("[Books] Error reading %s: %s", _bpath.name, _bf_exc)
            except Exception as _books_exc:
                log.debug("Book ingestion error: %s", _books_exc)

        # ── Wikipedia dump ingestion — every 100 cycles, 200 articles (background) ────
        if _run_heavy and cycle % 100 == 0:
            import threading as _wiki_dump_thread
            def _do_dump_ingest(_cyc=cycle):
                try:
                    from sare.learning.wikipedia_dump_ingester import WikipediaDumpIngester
                    _wdi = WikipediaDumpIngester()
                    if _wdi.is_available():
                        n = _wdi.ingest_batch(max_articles=200)
                        if n:
                            log.info("[WikiDump] cycle %d: ingested %d facts from Simple English Wikipedia", _cyc, n)
                    else:
                        log.debug("[WikiDump] Dump not available — skipping (run: curl -L ... simplewiki-latest.xml.bz2)")
                except Exception as _wdi_exc:
                    log.debug("[WikiDump] Error: %s", _wdi_exc)
            _wiki_dump_thread.Thread(target=_do_dump_ingest, daemon=True, name="wiki-dump").start()

        # ── Wikipedia domain sweep — every 50 cycles, sweep weakest domain ────
        if _run_heavy and cycle % 50 == 0 and _neural_learner is not None:
            try:
                _neural_learner.save()
                _nl_stats = _neural_learner.get_stats()
                log.info("[NeuralLearner] Saved: memories=%d, hit_rate=%.2f, learns=%d",
                         _nl_stats["memory_size"], _nl_stats["hit_rate"], _nl_stats["learn_calls"])
            except Exception as _nl_save_exc:
                log.debug("[NeuralLearner] Save error: %s", _nl_save_exc)

        if _run_heavy and cycle % 50 == 0:
            try:
                import threading as _wiki_threading
                from sare.learning.wikipedia_ingester import WikipediaIngester, DOMAIN_TOPICS
                _wiki_ingester = WikipediaIngester()
                # Pick the domain with fewest cached topics (least knowledge)
                _wiki_domain_counts = {
                    d: len(_wiki_ingester.get_cached_topics(d))
                    for d in DOMAIN_TOPICS
                }
                _wiki_target = min(_wiki_domain_counts, key=_wiki_domain_counts.get)
                log.info("[Wikipedia] Starting sweep of domain '%s' (cached=%d topics)",
                         _wiki_target, _wiki_domain_counts[_wiki_target])
                def _wiki_sweep(_d=_wiki_target, _wi=_wiki_ingester):
                    try:
                        n = _wi.sweep_domain(_d, max_topics=30)
                        log.info("[Wikipedia] Sweep complete: domain=%s, facts=%d", _d, n)
                    except Exception as _we:
                        log.debug("[Wikipedia] Sweep error: %s", _we)
                _wiki_thread = _wiki_threading.Thread(target=_wiki_sweep, daemon=True, name="WikiSweep")
                _wiki_thread.start()
            except Exception as _wiki_exc:
                log.debug("WikipediaIngester error: %s", _wiki_exc)

        # ── S32: EmbodiedAgent — 3 episodes per cycle (explore/deliver/mixed) ─
        if _run_heavy and _embodied_agent is not None:
            try:
                # Reset world every 100 episodes for environment diversity
                if _embodied_agent._episode_count > 0 and _embodied_agent._episode_count % 100 == 0:
                    _embodied_agent.world._reset()
                _ea_goals = ["explore", "deliver", "mixed"]
                for _ea_goal in _ea_goals:
                    _ep = _embodied_agent.run_episode(max_steps=30, goal=_ea_goal)
                    if _ep and _ep.concepts_discovered:
                        log.info("[Embodied] Episode %d (%s): reward=%.2f concepts=%s",
                                 _ep.episode_id, _ea_goal, _ep.total_reward, _ep.concepts_discovered)
            except Exception as _ea_exc:
                log.debug("EmbodiedAgent episode error: %s", _ea_exc)

        # ── S32: CausalRollout — feed successful solves every 5 cycles ───────
        if _run_heavy and _causal_rollout is not None and cycle % 5 == 0:
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

        # ── S33: CausalRollout — run actual prediction rollouts every 10 cycles (background) ─
        if _run_heavy and _causal_rollout is not None and cycle % 10 == 0:
            import threading as _cr_threading
            _cr_exprs_bg = [
                (getattr(_r, 'expression', '') or getattr(_r, 'problem_id', ''),
                 getattr(_r, 'domain', 'algebra'))
                for _r in results if getattr(_r, 'expression', '') or getattr(_r, 'problem_id', '')
            ][:3]
            def _do_causal_rollout(_cr=_causal_rollout, _exprs=_cr_exprs_bg, _cyc=cycle):
                try:
                    from sare.engine import ALL_TRANSFORMS, load_problem, EnergyEvaluator, BeamSearch
                    _avail_transforms = [t.name() for t in ALL_TRANSFORMS]
                    _cr_ev = EnergyEvaluator()
                    _cr_bs = BeamSearch()
                    def _cr_solve_fn(expr_str):
                        try:
                            _, _g = load_problem(expr_str)
                            _e_before = _cr_ev.compute(_g).total
                            _res = _cr_bs.search(_g, _cr_ev, ALL_TRANSFORMS, beam_width=4, budget_seconds=1.0)
                            _e_after = _cr_ev.compute(_res.graph).total
                            return {
                                "expression": str(_res.graph),
                                "delta": _e_before - _e_after,
                                "transforms_used": [s.transform_name for s in _res.steps] if hasattr(_res, 'steps') else [],
                                "success": _e_after < _e_before,
                            }
                        except Exception:
                            return {"expression": expr_str, "delta": 0.0, "transforms_used": [], "success": False}
                    for _cr_expr, _cr_dom in _exprs:
                        _rr = _cr.run_rollout(_cr_expr, _cr_dom, _avail_transforms, _cr_solve_fn, horizon=3)
                        if _rr.steps_completed > 0:
                            log.info("[CausalRollout] %s: accuracy=%.2f steps=%d/%d error=%.3f",
                                     _cr_dom, _rr.horizon_accuracy, _rr.steps_completed,
                                     _rr.plan.n_steps, _rr.cumulative_error)
                except Exception as _cr_run_exc:
                    log.debug("CausalRollout run error: %s", _cr_run_exc)
            _cr_threading.Thread(target=_do_causal_rollout, daemon=True, name="causal-rollout").start()

        # ── S32: CounterfactualReasoner — analyze hard cases every 10 cycles ─
        if _run_heavy and _counterfactual_reasoner is not None and cycle % 10 == 0:
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

        # ── S33: CausalChainDetector — observe every successful multi-transform solve ─
        if _chain_detector is not None:
            try:
                for _r in results:
                    _ps = getattr(_r, 'proof_steps', None) or []
                    if len(_ps) >= 2 and getattr(_r, 'solved', False):
                        _delta = float(getattr(_r, 'energy_before', 1.0) - getattr(_r, 'energy_after', 1.0))
                        _dom = getattr(_r, 'domain', 'algebra')
                        _new_chains = _chain_detector.observe(_ps, _dom, _delta, success=True)
                        for _nc in _new_chains:
                            log.info("[CausalChain] Discovered: %s [%s] conf=%.2f cross=%s",
                                     _nc.name, _nc.domain, _nc.confidence, _nc.cross_domain)
            except Exception as _cd_exc:
                log.debug("CausalChainDetector observe error: %s", _cd_exc)

        # ── S33: AbductiveRanker — explain surprising results every 15 cycles ─
        if _run_heavy and _abductive_ranker is not None and cycle % 15 == 0:
            try:
                # Find results with large unexpected deltas (surprising outcomes)
                _surprising = [
                    _r for _r in results
                    if getattr(_r, 'solved', False)
                    and abs(getattr(_r, 'energy_before', 0) - getattr(_r, 'energy_after', 0)) > 2.0
                    and getattr(_r, 'proof_steps', None)
                ]
                for _r in _surprising[:3]:
                    _ps = list(_r.proof_steps)
                    _delta = float(_r.energy_before - _r.energy_after)
                    _dom = getattr(_r, 'domain', 'algebra')
                    _hyps = _abductive_ranker.explain(_ps, _delta, _dom, top_k=3)
                    if _hyps and _hyps[0].recommended_action == "accept":
                        log.info("[Abductive] Best explanation for %s (delta=%.2f): %s (posterior=%.2f)",
                                 _dom, _delta, _hyps[0].name, _hyps[0].posterior)
                    # Feed "verify" hypotheses into curriculum as interesting problems
                    _verify_hyps = [h for h in _hyps if h.recommended_action == "verify"]
                    if _verify_hyps and hasattr(curriculum, 'add_hypothesis_problem'):
                        for _vh in _verify_hyps[:1]:
                            try:
                                curriculum.add_hypothesis_problem(_vh.name, _dom, _vh.posterior)
                            except Exception:
                                pass
            except Exception as _ar_exc:
                log.debug("AbductiveRanker error: %s", _ar_exc)

        # ── S33: CreativityEngine — dream cycle every 20 cycles (background) ─
        if _run_heavy and _creativity_engine is not None and cycle % 20 == 0:
            import threading as _ce_threading
            def _do_dream(_ce=_creativity_engine):
                try:
                    _dream_result = _ce.dream()
                    if _dream_result.get("promoted"):
                        log.info("[Creativity] Dream promoted: %s (score=%.2f) %s→%s",
                                 _dream_result.get("transform_name", "?"),
                                 _dream_result.get("score", 0),
                                 _dream_result.get("source_domain", "?"),
                                 _dream_result.get("target_domain", "?"))
                    elif _dream_result.get("message") != "cooldown":
                        log.debug("[Creativity] Dream: %s", _dream_result.get("message", ""))
                except Exception as _ce_exc:
                    log.debug("CreativityEngine dream error: %s", _ce_exc)
            _ce_threading.Thread(target=_do_dream, daemon=True, name="creativity-dream").start()

        # ── S33: CreativityEngine — analogy transfer on cross-domain solves (background) ──
        if _run_heavy and _creativity_engine is not None and cycle % 10 == 0:
            import threading as _cat_threading
            import random as _rnd
            _solved_with_proofs_bg = [
                (_r.domain if hasattr(_r, 'domain') else 'algebra', list(_r.proof_steps))
                for _r in results
                if getattr(_r, 'solved', False) and getattr(_r, 'proof_steps', None)
            ][:2]
            def _do_analogy_transfer(_ce=_creativity_engine, _swp=_solved_with_proofs_bg):
                try:
                    _tgt_candidates = ["logic", "set_theory", "probability", "geometry"]
                    for _src_dom, _proof_steps in _swp:
                        _tgt = [d for d in _tgt_candidates if d != _src_dom]
                        if _tgt:
                            _at_result = _ce.analogy_transfer(
                                source_domain=_src_dom,
                                target_domain=_rnd.choice(_tgt),
                                proof_steps=_proof_steps,
                            )
                            if _at_result.get("promoted"):
                                log.info("[Creativity] Analogy found: %s→%s",
                                         _src_dom, _at_result.get("target_domain"))
                except Exception as _at_exc:
                    log.debug("CreativityEngine analogy error: %s", _at_exc)
            _cat_threading.Thread(target=_do_analogy_transfer, daemon=True, name="creativity-analogy").start()

        # ── S33: Social Intelligence — ToM false-belief benchmarks + belief updates (background) ─
        if _run_heavy and _theory_of_mind is not None and cycle % 20 == 0:
            import threading as _tom_threading
            import random as _tom_rnd
            _tom_solve_rate = (sum(1 for _r in results if getattr(_r, 'solved', False)) / max(len(results), 1))
            def _do_tom(_tom=_theory_of_mind, _sr=_tom_solve_rate):
                try:
                    _fb_scenarios = [
                        ("teacher", "The ball is in the basket", "The ball is in the box"),
                        ("student", "x + 0 simplifies to x + 0", "x + 0 simplifies to x"),
                        ("adversary", "All rules have exceptions", "Most rules are consistent"),
                        ("teacher", "The system knows factoring", "The system is still learning factoring"),
                        ("collaborator", "Both approaches work equally", "Approach A is more efficient"),
                    ]
                    _scenario = _tom_rnd.choice(_fb_scenarios)
                    _fb_result = _tom.false_belief_test(_scenario[0], _scenario[2], _scenario[1])
                    log.info("[ToM] False-belief test: %s — discrepancy=%s",
                             _scenario[0], _fb_result.get("discrepancy_detected"))
                    _sare_self = _tom.get_agent("sare_hx")
                    if _sare_self:
                        _sare_self.add_belief(f"Current solve rate is {_sr:.0%} on recent problems",
                                              True, min(1.0, _sr + 0.3))
                    _teacher = _tom.get_agent("teacher")
                    if _teacher:
                        _teacher.add_belief(f"The student solves {_sr:.0%} of problems", True, 0.85)
                    _tom.save()
                except Exception as _tom_exc:
                    log.debug("TheoryOfMind cycle error: %s", _tom_exc)
            _tom_threading.Thread(target=_do_tom, daemon=True, name="theory-of-mind").start()

        # ── S33: DialogueManager — initiate questions when curiosity is high (background) ──
        if _run_heavy and _dialogue_manager is not None and cycle % 25 == 0:
            import threading as _dm_threading
            _unsolved_domains_bg = {}
            for _r in results:
                if not getattr(_r, 'solved', True):
                    _d = getattr(_r, 'domain', 'algebra')
                    _unsolved_domains_bg[_d] = _unsolved_domains_bg.get(_d, 0) + 1
            if _unsolved_domains_bg:
                _weak_dom_bg = max(_unsolved_domains_bg, key=_unsolved_domains_bg.get)
                def _do_dialogue(_dm=_dialogue_manager, _wd=_weak_dom_bg):
                    try:
                        _q_result = _dm.initiate_question(_wd)
                        if _q_result and _q_result.get("answer"):
                            log.info("[Dialogue] Got answer for %s: %s",
                                     _wd, str(_q_result.get("answer"))[:80])
                    except Exception as _dm_exc:
                        log.debug("DialogueManager question error: %s", _dm_exc)
                _dm_threading.Thread(target=_do_dialogue, daemon=True, name="dialogue-mgr").start()

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

        # ── Learning monitor: record every result ─────────────────────────
        try:
            from sare.meta.learning_monitor import get_learning_monitor
            _lmon = get_learning_monitor()
            for _r in results:
                _r_graph = getattr(_r, "initial_graph", None) or getattr(_r, "graph", None)
                if _r_graph is not None:
                    _lmon.record(
                        _r_graph,
                        solved=bool(getattr(_r, "solved", False)),
                        energy_before=float(getattr(_r, "energy_before", 0.0) or 0.0),
                        energy_after=float(getattr(_r, "energy_after",  0.0) or 0.0),
                        cycle=cycle,
                        domain=str(getattr(_r, "domain", "general") or "general"),
                    )
            _lmon.maybe_report(cycle)
        except Exception as _lmon_exc:
            log.debug("LearningMonitor error: %s", _lmon_exc)

        # ── Accumulate stuck problems + trigger PatternExtractor / Synthesizer ─
        try:
            for _r in results:
                _expr = getattr(_r, "problem_id", None) or ""
                _dom  = getattr(_r, "domain", "algebra") or "algebra"
                _e_before = getattr(_r, "energy_before", 0.0)
                _e_after  = getattr(_r, "energy_after",  0.0)
                _no_transform = not getattr(_r, "solved", True) and abs(_e_before - _e_after) < 0.01
                _n_steps = len(getattr(_r, "proof_steps", None) or [])
                _slow_solve = getattr(_r, "solved", False) and _n_steps >= 6
                if _no_transform or _slow_solve:
                    _stuck_by_domain.setdefault(_dom, [])
                    if _expr and _expr not in _stuck_by_domain[_dom]:
                        _stuck_by_domain[_dom].append(
                            f"~{_expr}" if (_slow_solve and not _no_transform) else _expr
                        )

            _SYNTH_DOMAINS = {"algebra", "arithmetic", "calculus", "logic", "general",
                              "chemistry", "physics"}

            # ── Phase 1: Symbolic PatternExtractor (no LLM) ───────────────────
            # Runs eagerly — try to auto-promote rules from stuck graph structure.
            try:
                from sare.learning.pattern_extractor import extract_and_promote, should_skip_llm
                _stuck_cache = getattr(experiment_runner, '_stuck_graph_cache', {})
                for _dom, _stuck_exprs in list(_stuck_by_domain.items()):
                    if _dom not in _SYNTH_DOMAINS:
                        continue
                    _dom_graphs = list(_stuck_cache.get(_dom, []))
                    if len(_dom_graphs) >= 2:
                        _n_promoted = extract_and_promote(
                            _dom_graphs, _dom,
                            getattr(experiment_runner, 'concept_registry', None),
                        )
                        if _n_promoted:
                            experiment_runner._refresh_concept_transforms()
            except Exception as _pe_exc:
                log.debug("[PatternExtractor] error: %s", _pe_exc)

            # ── Phase 2: Batch LLM Synthesis (one call for all ready domains) ─
            # Collect all domains that have enough stuck credits AND are off cooldown.
            _batch_domains = []
            _batch_exprs   = {}
            _batch_graphs  = {}
            _stuck_cache   = getattr(experiment_runner, '_stuck_graph_cache', {})
            from sare.engine import load_problem as _lp_val

            for _dom, _stuck_exprs in list(_stuck_by_domain.items()):
                if _dom not in _SYNTH_DOMAINS:
                    _stuck_by_domain[_dom] = []
                    continue
                _credits = sum(0.5 if e.startswith("~") else 1.0 for e in _stuck_exprs)
                if _credits < _SYNTH_TRIGGER:
                    continue
                if cycle - _synth_cooldown.get(_dom, 0) < _SYNTH_COOLDOWN_CYCLES:
                    continue
                # Skip if PatternExtractor already covered most patterns
                try:
                    from sare.learning.pattern_extractor import should_skip_llm
                    _dom_graphs = list(_stuck_cache.get(_dom, []))
                    if should_skip_llm(_dom_graphs, _dom):
                        log.info("[Synth] Skipping LLM for %s — patterns already extracted", _dom)
                        _synth_cooldown[_dom] = cycle
                        _stuck_by_domain[_dom] = []
                        continue
                except Exception:
                    pass

                # Build validation graphs (stuck cache primary)
                _vg = list(_stuck_cache.get(_dom, []))[:8]
                _clean = [e.lstrip("~") for e in _stuck_exprs[:8]]
                if len(_vg) < 4:
                    for _e in _clean:
                        if len(_vg) >= 8:
                            break
                        try:
                            _, _g = _lp_val(_e)
                            if _g is not None:
                                _vg.append(_g)
                        except Exception:
                            pass

                _batch_domains.append(_dom)
                _batch_exprs[_dom]  = _clean[:6]
                _batch_graphs[_dom] = _vg
                _synth_cooldown[_dom] = cycle
                _stuck_by_domain[_dom] = []

            if _batch_domains:
                _existing = [t.name() for t in experiment_runner.transforms
                             if hasattr(t, "name")]
                log.info("[Synth] Batch synthesis for %d domains: %s",
                         len(_batch_domains), _batch_domains)
                import threading as _threading
                def _do_batch_synthesis(_domains, _exprs_map, _graphs_map, _ex, _runner):
                    try:
                        from sare.learning.llm_transform_synthesizer import get_llm_synthesizer
                        _synth = get_llm_synthesizer()
                        for _d in _domains:
                            try:
                                _res = _synth.synthesize(
                                    domain=_d,
                                    stuck_exprs=_exprs_map.get(_d, []),
                                    validation_graphs=_graphs_map.get(_d, []),
                                    existing_transform_names=_ex,
                                )
                                if _res["promoted"]:
                                    log.info("[Synth] ✓ %s synthesized for domain=%s (score=%.2f)",
                                             _res["actual_name"], _d, _res["score"])
                                    try:
                                        _runner._load_synthesized_transforms()
                                        _runner._refresh_concept_transforms()
                                    except Exception:
                                        pass
                                else:
                                    log.info("[Synth] ✗ domain=%s: %s", _d, _res["message"])
                            except Exception as _de:
                                log.debug("[Synth] domain=%s error: %s", _d, _de)
                    except Exception as _be:
                        log.debug("[Synth] batch thread error: %s", _be)
                _threading.Thread(
                    target=_do_batch_synthesis,
                    args=(_batch_domains, _batch_exprs, _batch_graphs, _existing, experiment_runner),
                    daemon=True, name="synth-batch",
                ).start()

        except Exception as _stuck_err:
            log.debug("Stuck tracker error: %s", _stuck_err)

        # ── General intelligence batch (every cycle, even in turbo) ────────
        # Un-gated from _run_heavy: this is the primary source of real learning
        # across 10 domains. Smaller batch in turbo (12) vs normal (24).
        _gen_batch_size = 60 if turbo else 24
        if _general_solver is not None and _general_curriculum is not None:
            try:
                if _run_heavy and cycle % 10 == 0:
                    # Run teacher queue in background to avoid blocking on LLM calls
                    def _run_teacher_queue(_gc=_general_curriculum):
                        try:
                            _teacher = _gc.process_teacher_queue(max_items=2)
                            if _teacher.get("generated"):
                                log.info(
                                    "Teacher queue: +%d LLM gap-fill examples (%d pending)",
                                    int(_teacher.get("generated", 0)),
                                    int(_teacher.get("pending", 0)),
                                )
                        except Exception as _teacher_exc:
                            log.debug("Teacher queue error: %s", _teacher_exc)
                    import threading as _tq_t
                    _tq_t.Thread(target=_run_teacher_queue, daemon=True,
                                 name=f"teacher-queue-{cycle}").start()

                _gen_problems = _general_curriculum.generate_batch(size=_gen_batch_size)
                _gen_solved = 0
                _gen_lessons = 0
                _mode_counts = {}
                _source_counts = {}
                _backend_counts = {}
                _free_solve_debt = []
                _conversion_debt = []
                _llm_problems = []  # problems that need LLM — offloaded to thread

                def _make_conversion_variants(_problem_text: str) -> list[str]:
                    _base = str(_problem_text or "").strip()
                    if not _base:
                        return []
                    _trimmed = _base[:-1] if _base.endswith("?") else _base
                    _lowered = _trimmed[:1].lower() + _trimmed[1:] if _trimmed else _trimmed
                    _variants = [
                        f"Brief answer: {_trimmed}?",
                        f"In one short phrase, {_lowered}?",
                        f"Answer directly: {_trimmed}?",
                    ]
                    _seen = set()
                    _deduped = []
                    for _variant in _variants:
                        _key = _variant.strip().lower()
                        if _key and _key not in _seen and _key != _base.strip().lower():
                            _seen.add(_key)
                            _deduped.append(_variant)
                    return _deduped

                def _pattern_key_for_problem(_problem, _result) -> str:
                    _meta = dict(getattr(_problem, "metadata", {}) or {})
                    return str(
                        getattr(_result, "pattern_key", "")
                        or _meta.get("pattern_key", "")
                        or getattr(_problem, "example_hash", "")
                        or getattr(_problem, "source_id", "")
                        or getattr(_problem, "problem_id", "")
                    ).strip()

                for _gp in _gen_problems:
                    _source_kind = getattr(_gp, "source_kind", "unknown") or "unknown"
                    _source_counts[_source_kind] = _source_counts.get(_source_kind, 0) + 1
                    if _gp.expected:
                        _gp_meta = {
                            "source_kind": getattr(_gp, "source_kind", "data_problem"),
                            "generator": getattr(_gp, "generator", "human_data"),
                            "verification_level": getattr(_gp, "verification_level", "unverified"),
                            "backend_hint": getattr(_gp, "backend_hint", "python"),
                            "task_type": getattr(_gp, "task_type", "question_answer"),
                            "source_path": getattr(_gp, "source_path", ""),
                            "source_id": getattr(_gp, "source_id", ""),
                            "example_hash": getattr(_gp, "example_hash", ""),
                            "cycle": cycle,
                        }
                        _gp_meta.update(dict(getattr(_gp, "metadata", {}) or {}))
                        # Always skip inline LLM calls to avoid blocking the main loop.
                        # LLM teaching happens in the dedicated llm_teacher background thread
                        # (fires every cycle % 10), not inline per-problem.
                        _gp_meta["skip_llm"] = True
                        _gr = _general_solver.attempt_learning_problem(
                            _gp.text,
                            _gp.expected,
                            _gp.domain,
                            context=_gp.context or "",
                            metadata=_gp_meta,
                        )
                        try:
                            _general_curriculum.record_outcome(
                                _gp,
                                _gr,
                                new_pattern_mastered=bool(getattr(_gr, "heldout_verified", False)),
                            )
                        except Exception:
                            pass
                        if _gr.solved:
                            _gen_solved += 1
                        _mode = getattr(_gr, "learning_mode", "unknown")
                        _mode_counts[_mode] = _mode_counts.get(_mode, 0) + 1
                        _backend = getattr(_gr, "backend", "python") or "python"
                        _backend_counts[_backend] = _backend_counts.get(_backend, 0) + 1
                        # Record outcome for AlgorithmSelector (meta-learning)
                        try:
                            from sare.meta.algorithm_selector import get_algorithm_selector
                            _task_type = {"math": "algebra", "logic": "logic", "code": "coding",
                                          "language": "language", "planning": "planning"}.get(_gp.domain, "science")
                            _strategy = {"symbolic": "beam_search", "llm": "greedy", "hybrid": "mcts",
                                         "kb_cache": "beam_search", "fact_chain": "greedy",
                                         "template": "beam_search",
                                         "native_rule": "beam_search",
                                         "native_code": "beam_search"}.get(_gr.solver_used, "beam_search")
                            get_algorithm_selector().record_outcome(_task_type, _strategy, _gr.solved)
                        except Exception:
                            pass
                        _is_authoritative = getattr(_gp, "verification_level", "unverified") in {
                            "authoritative", "evidence_backed", "cross_checked"
                        }
                        if (
                            _gr.lesson and _gr.solved and _mode != "template_replay"
                            and _is_authoritative
                        ):
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
                        # ── KB accumulation: store every Q→A pair we know, win or lose ──
                        # Authoritative = high-confidence teaching; unverified = lower confidence.
                        # This is how the system builds a retrievable KB over time.
                        _is_general_domain = _gp.domain in (
                            "history", "geography", "biology", "economics", "psychology",
                            "science", "factual", "language", "social", "reasoning",
                            "commonsense", "word_problems", "analogy", "planning",
                            "general", "technology", "literature", "philosophy",
                        )
                        if _gp.expected and _is_general_domain:
                            _conf_store = 0.85 if _is_authoritative else 0.60
                            _fact_text = f"{_gp.text.rstrip('?')} → {_gp.expected}"
                            try:
                                from sare.memory.world_model import get_world_model
                                get_world_model().add_fact(
                                    domain=_gp.domain,
                                    fact=_fact_text[:200],
                                    confidence=_conf_store,
                                )
                            except Exception:
                                pass
                            # Push into commonsense KB for AnswerTo lookup on next attempt
                            try:
                                from sare.knowledge.commonsense import get_commonsense_base
                                get_commonsense_base().add_fact(
                                    domain=_gp.domain,
                                    question=_gp.text,
                                    answer=_gp.expected,
                                )
                            except Exception:
                                pass
                            # Reinforce correct retrievals so they become stronger
                            if _gr.solved and _gr.solver_used in ("kb_cache", "fact_chain", "retrieval"):
                                try:
                                    from sare.memory.world_model import get_world_model
                                    get_world_model().record_domain_success(_gp.domain)
                                except Exception:
                                    pass
                        _problem_pattern_key = _pattern_key_for_problem(_gp, _gr)
                        if _is_authoritative and _mode in ("retrieval", "web_learned"):
                            try:
                                from sare.learning.general_curriculum import GeneralProblem

                                for _idx, _variant in enumerate(_make_conversion_variants(_gp.text)[:3]):
                                    _conversion_debt.append(
                                        GeneralProblem(
                                            problem_id=f"convert_{int(time.time()*1000) % 10**7}_{_idx}",
                                            text=_variant,
                                            domain=_gp.domain,
                                            expected=_gp.expected,
                                            context=_gp.context or "",
                                            difficulty=min(1.0, max(_gp.difficulty, 0.55) + 0.05),
                                            task_type=_gp.task_type,
                                            evidence_span=_gp.evidence_span,
                                            source_kind="retrieval_to_free_solve",
                                            source_path=_gp.source_path,
                                            source_id=_gp.source_id or _gp.problem_id,
                                            generator="retrieval_conversion",
                                            verifier=_gp.verifier,
                                            backend_hint=_gp.backend_hint,
                                            verification_level=_gp.verification_level,
                                            example_hash=_gp.example_hash,
                                            metadata={
                                                **dict(_gp.metadata or {}),
                                                "priority_reason": "retrieval_to_free_solve",
                                                "heldout_variant": True,
                                                "conversion_origin_mode": _mode,
                                                "pattern_key": _problem_pattern_key,
                                            },
                                        )
                                    )
                            except Exception as _conv_exc:
                                log.debug("[General] retrieval conversion debt error: %s", _conv_exc)
                        if _gp.expected and _is_authoritative and _mode != "free_solve":
                            _free_solve_debt.append(_gp)
                        _hypothesis_id = str((_gp_meta or {}).get("concept_hypothesis_id", "") or "")
                        if _hypothesis_id:
                            try:
                                from sare.memory.world_model import get_world_model

                                _verified_now = bool(_gr.solved and _mode == "free_solve")
                                _wm = get_world_model()
                                _wm.record_concept_hypothesis_verification(_hypothesis_id, _verified_now)
                                if getattr(_gr, "heldout_verified", False):
                                    _wm.promote_concept_hypothesis(_hypothesis_id)
                            except Exception as _hyp_verify_exc:
                                log.debug("[ConceptSynth] verification record error: %s", _hyp_verify_exc)
                        if _gp.expected and _gr.answer:
                            _correct = _gp.expected.lower() in _gr.answer.lower() or _gr.answer.lower() in _gp.expected.lower()
                            if not _correct:
                                log.debug("[General] Wrong: '%s' expected='%s' got='%s'",
                              _gp.text[:50], _gp.expected[:30], _gr.answer[:30])
                            # Neural learner: feed outcome (contrastive on wrong, reinforce on correct)
                            if _neural_learner is not None and _gp.domain not in ("math","logic","code","algebra","arithmetic"):
                                try:
                                    _neural_learner.learn(_gp.text, _gr.answer, _gp.domain, correct=_correct)
                                    if not _correct and _gp.expected:
                                        # Also reinforce the correct answer
                                        _neural_learner.learn(_gp.text, _gp.expected, _gp.domain, correct=True)
                                except Exception:
                                    pass
                    else:
                        # Ungraded open-ended prompts stay off the honest learning loop.
                        _llm_problems.append(_gp)

                # Fire open-ended prompts in background without counting them as graded learning.
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

                if _free_solve_debt or _conversion_debt:
                    try:
                        from sare.learning.general_curriculum import GeneralProblem, inject_priority_problems
                        _debt_tag = int(time.time() * 1000) % 10**7
                        _debt_batch = [
                            GeneralProblem(
                                problem_id=f"freedebt_{_debt_tag}_{_i}",
                                text=_p.text,
                                domain=_p.domain,
                                expected=_p.expected,
                                context=_p.context or "",
                                difficulty=min(1.0, max(_p.difficulty, 0.6) + 0.1),
                                task_type=_p.task_type,
                                evidence_span=_p.evidence_span,
                                source_kind=_p.source_kind,
                                source_path=_p.source_path,
                                source_id=_p.source_id,
                                generator=_p.generator,
                                verifier=_p.verifier,
                                backend_hint=_p.backend_hint,
                                verification_level=_p.verification_level,
                                example_hash=_p.example_hash,
                                metadata={
                                    **dict(_p.metadata or {}),
                                    "priority_reason": "free_solve_failure",
                                },
                            )
                            for _i, _p in enumerate(_free_solve_debt[:10])
                        ]
                        _debt_batch.extend(_conversion_debt[:10])
                        inject_priority_problems(_debt_batch)
                        log.info(
                            "[General] Re-queued %d authoritative / conversion items as curriculum debt",
                            len(_debt_batch),
                        )
                    except Exception as _debt_exc:
                        log.debug("Free-solve debt injection error: %s", _debt_exc)

                log.info(
                    "General batch: %d/%d graded correct, %d lessons, %d open-ended→bg (domains: %s, sources: %s, backends: %s)",
                    _gen_solved, len(_gen_problems) - len(_llm_problems),
                    _gen_lessons, len(_llm_problems),
                    ", ".join(sorted(set(p.domain for p in _gen_problems))),
                    _source_counts,
                    _backend_counts,
                )
                if _mode_counts:
                    log.info("General learning modes: %s", _mode_counts)
            except Exception as exc:
                log.debug("General intelligence batch error: %s", exc)

        # ── General-domain concept synthesis from failure clusters every 5 cycles ──
        if _run_heavy and cycle % 5 == 0 and _general_solver is not None:
            try:
                import threading as _cs_threading
                from sare.memory.world_model import get_world_model as _gwm_cs
                _GENERAL_DOMAINS = [
                    "history", "geography", "biology", "economics", "psychology",
                    "factual", "science", "social",
                ]
                def _do_general_concept_synthesis():
                    _wm = _gwm_cs()
                    import random as _rnd
                    from sare.learning.general_curriculum import GeneralProblem, inject_priority_problems
                    _shuffled = list(_GENERAL_DOMAINS)
                    _rnd.shuffle(_shuffled)
                    _synthesized = 0
                    for _dom in _shuffled:
                        try:
                            _clusters = _wm.get_failure_patterns(_dom, min_count=3, window_seconds=3600.0)
                            for _cluster in _clusters:
                                if _synthesized >= 3:
                                    break  # max 3 LLM synthesis calls per cycle
                                _rule = _wm.synthesize_knowledge_rule(_dom, _cluster)
                                if _rule:
                                    _hypothesis_id = str(_rule.get("hypothesis_id", "") or "")
                                    _pattern_key = str(_rule.get("pattern_key", "") or "")
                                    _parent = str(_rule.get("parent", "") or "")
                                    _verifications = list(_rule.get("verification_examples", []) or [])[:5]
                                    if _verifications:
                                        _verify_tag = int(time.time() * 1000) % 10**7
                                        inject_priority_problems([
                                            GeneralProblem(
                                                problem_id=f"conceptverify_{_verify_tag}_{_i}",
                                                text=str(_item.get("question", "") or "")[:200],
                                                domain=_dom,
                                                expected=str(_item.get("answer", "") or "")[:120],
                                                difficulty=0.7,
                                                task_type="question_answer",
                                                source_kind="concept_hypothesis",
                                                generator="concept_synthesis",
                                                verification_level="authoritative",
                                                metadata={
                                                    "priority_reason": "concept_hypothesis_verification",
                                                    "heldout_variant": True,
                                                    "pattern_key": _pattern_key,
                                                    "concept_parent": _parent,
                                                    "concept_hypothesis_id": _hypothesis_id,
                                                    "conversion_origin_mode": "concept_hypothesis",
                                                },
                                            )
                                            for _i, _item in enumerate(_verifications)
                                            if str(_item.get("question", "") or "").strip()
                                            and str(_item.get("answer", "") or "").strip()
                                        ])
                                    log.info("[ConceptSynth] %s → staged hypothesis: %s",
                                             _dom, str(_rule)[:120])
                                    _synthesized += 1
                        except Exception as _cse:
                            log.debug("ConceptSynth %s error: %s", _dom, _cse)
                        if _synthesized >= 3:
                            break  # max 3 domains per run
                _cs_threading.Thread(
                    target=_do_general_concept_synthesis, daemon=True,
                    name=f"concept_synth_{cycle}"
                ).start()
            except Exception as _cs_exc:
                log.debug("ConceptSynth trigger error: %s", _cs_exc)

        # T3-4: Global Workspace — key modules submit attention bids every 5 cycles
        if _run_heavy and cycle % 5 == 0 and global_workspace is not None:
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

        # ── FactInference + KB stats every 5 cycles (background thread) ─────────
        # Moved to background: transitivity chaining is O(N^3) on world model facts
        # and can block the main loop for 10+ minutes on large knowledge bases.
        if _run_heavy and cycle % 5 == 0:
            def _run_fact_inference(_cycle=cycle):
                try:
                    from sare.cognition.fact_inference import get_fact_inference
                    _fi = get_fact_inference()
                    _fi_total = 0
                    for _fi_domain in ["factual", "science", "reasoning", "analogy"]:
                        _n = _fi.infer_iterative(_fi_domain, max_rounds=2, max_new_per_round=15)
                        if _n:
                            _fi_total += _n
                            log.info("[Daemon] Iterative chaining: %d new facts for domain=%s", _n, _fi_domain)
                    if _fi_total:
                        log.info("[Daemon] Total inferred this cycle: %d facts", _fi_total)
                except Exception as _fi_exc:
                    log.debug("FactInference error: %s", _fi_exc)
            import threading as _fi_t
            _fi_t.Thread(target=_run_fact_inference, daemon=True,
                         name=f"fact-infer-{cycle}").start()

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
        if _run_heavy and cycle % 10 == 0 and _general_solver is not None and _general_curriculum is not None:
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
                            metadata={"priority_reason": "self_test_gap"},
                        )
                        for _i, (_mq, _ma) in enumerate(_st_misses[:5])
                    ])
                    log.info("[Self-test] Injected %d KB gaps into priority curriculum", min(5, len(_st_misses)))
            except Exception as _st_err:
                log.debug("Self-test error: %s", _st_err)

        # ── Lightweight benchmark probe every 30 minutes ───────────────────────
        if (
            _general_solver is not None
            and _general_curriculum is not None
            and (time.time() - _last_benchmark_probe_at) >= _benchmark_probe_interval_s
        ):
            _last_benchmark_probe_at = time.time()
            try:
                import random as _probe_rng
                from sare.learning.general_curriculum import (
                    _TEMPLATE_BANKS,
                    GeneralProblem,
                    inject_priority_problems,
                )

                _probe_stats = _general_solver.get_stats() or {}

                def _probe_weakness(_domain: str) -> float:
                    _raw = _probe_stats.get(_domain, {}) if isinstance(_probe_stats, dict) else {}
                    if not isinstance(_raw, dict):
                        _raw = {}
                    _attempts = int(_raw.get("attempts", 0) or 0)
                    _modes = _raw.get("modes", {}) if isinstance(_raw.get("modes", {}), dict) else {}
                    _measured = 0
                    for _bucket in _modes.values():
                        if isinstance(_bucket, dict):
                            _measured += int(_bucket.get("attempts", 0) or 0)
                    _coverage = (_measured / max(_attempts, 1)) if _attempts else 0.0
                    _free = float(_raw.get("free_solve_rate", _raw.get("solve_rate", 0.0)) or 0.0)
                    _novelty = 0.30 if _attempts <= 0 else (0.12 if _attempts < 12 else 0.0)
                    return (max(0.0, 0.60 - _free) * 1.6) + (max(0.0, 0.50 - _coverage) * 1.2) + _novelty

                _ranked_probe_domains = sorted(
                    (
                        (_probe_weakness(_domain), _domain)
                        for _domain in _TEMPLATE_BANKS
                        if _domain != "math"
                    ),
                    reverse=True,
                )
                _probe_pool = []
                for _, _domain in _ranked_probe_domains[:5]:
                    _bank = list(_TEMPLATE_BANKS.get(_domain, []) or [])
                    _probe_rng.shuffle(_bank)
                    for _question, _answer in _bank[:3]:
                        _probe_pool.append((_domain, _question, _answer))
                _probe_rng.shuffle(_probe_pool)
                _probe_sample = _probe_pool[:10]
                _probe_hits = 0
                _probe_misses = []

                for _idx, (_domain, _question, _answer) in enumerate(_probe_sample):
                    _probe_result = _general_solver.attempt_learning_problem(
                        _question,
                        _answer,
                        _domain,
                        metadata={
                            "source_kind": "benchmark_probe",
                            "generator": "benchmark_probe",
                            "verification_level": "authoritative",
                            "backend_hint": "python",
                            "task_type": "question_answer",
                            "skip_llm": True,  # probes are fast checks, not learning time
                        },
                    )
                    if _probe_result.solved and getattr(_probe_result, "learning_mode", "") == "free_solve":
                        _probe_hits += 1
                        continue
                    _probe_misses.append(
                        GeneralProblem(
                            problem_id=f"benchprobe_{int(time.time()*1000) % 10**7}_{_idx}",
                            text=_question,
                            domain=_domain,
                            expected=_answer,
                            difficulty=0.75,
                            source_kind="benchmark_probe",
                            generator="benchmark_probe",
                            verification_level="authoritative",
                            metadata={
                                "priority_reason": "benchmark_probe_miss",
                                "observed_mode": getattr(_probe_result, "learning_mode", "unknown"),
                            },
                        )
                    )

                if _probe_misses:
                    inject_priority_problems(_probe_misses[:10])
                log.info(
                    "[BenchmarkProbe] %d/%d free-solved in weak categories, injected %d debt items",
                    _probe_hits,
                    len(_probe_sample),
                    min(10, len(_probe_misses)),
                )
            except Exception as _probe_exc:
                log.debug("Benchmark probe error: %s", _probe_exc)

        # KB Gap Detector: every 20 cycles, find under-explored subjects and inject curiosity goals
        if _run_heavy and cycle % 20 == 0 and _general_curriculum is not None:
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
        if _run_heavy and cycle % 25 == 0:
            try:
                from sare.memory.hippocampus import HippocampusDaemon
                _hc = HippocampusDaemon(curriculum_gen=curriculum_gen)
                _injected = _hc.replay_episodes(curriculum_gen)
                if _injected:
                    log.info("[EpisodicReplay] Cycle %d: injected %d hard episodes into curriculum", cycle, _injected)
            except Exception as _er_exc:
                log.debug("EpisodicReplay error: %s", _er_exc)

        # Hypothesis generation every 15 cycles — genuine curiosity from analogy
        if _run_heavy and cycle % 15 == 0:
            try:
                from sare.cognition.hypothesis_maker import get_hypothesis_maker
                _hyps = get_hypothesis_maker().propose(max_proposals=3)
                if _hyps:
                    log.info("[HypothesisMaker] Proposed %d hypotheses: %s",
                             len(_hyps), [f"{s} is_a {v}" for s, p, v in _hyps[:2]])
            except Exception as _hm_exc:
                log.debug("HypothesisMaker error: %s", _hm_exc)

        # Dream consolidation every 10 cycles (or when consolidation drive is high)
        _consolidation_due = (_run_heavy and cycle % 10 == 0)
        if not _consolidation_due and _run_heavy and _homeostasis:
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

        # Save world model periodically (less often in turbo mode)
        if cycle % (100 if turbo else 5) == 0:
            try:
                from sare.memory.world_model import get_world_model
                get_world_model().save()
            except Exception:
                pass

        # Self-generated questions every 5 cycles
        if _run_heavy and cycle % 5 == 0:
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
        if _run_heavy and cycle % 10 == 0:
            try:
                from sare.knowledge.commonsense import get_commonsense_base
                from sare.perception.graph_builders import SentenceGraphBuilder, PlanGraphBuilder
                _cs_kb = get_commonsense_base()
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
        if _run_heavy and cycle % 20 == 0:
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
        if _run_heavy and cycle % 20 == 0:
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
        if _run_heavy and cycle % 5 == 0:
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

                # Test untested hypotheses in a background thread to avoid blocking main loop
                if cycle % 10 == 0:
                    _untested = [h for h in _te._hypotheses.values() if h.status == "untested"][:3]  # reduced from 5
                    _DOMAIN_TEST_PROBS = {
                        "algebra":     ["x + 0", "x * 1", "x + x", "2*x - x", "x^2 + 0"],
                        "arithmetic":  ["1 + 1", "2 * 3", "0 + 5", "4 * 1", "3 - 3"],
                        "logic":       ["p and True", "p or False", "not not p", "p and p", "p or p"],
                        "calculus":    ["x + 0", "x * 1", "x - x", "2*x - x", "x^2 * 1"],
                        "set_theory":  ["x + 0", "x * 1", "x - x", "0 + x", "x * 0"],
                        "code":        ["x + 0", "x * 1", "x - 0", "0 * x", "x + x"],
                        "qa":          ["x + 0", "1 + 1", "x * 1", "0 + x", "x - x"],
                    }
                    # Run hypothesis testing in background thread to avoid blocking main cycle
                    if _untested:
                        def _run_hyp_testing(_hyps, _te_ref, _runner, _domain_probs, _seeds):
                            import threading as _t
                            for _hyp in _hyps:
                                def _solve_fn(expr, _r=_runner):
                                    try:
                                        from sare.engine import load_problem
                                        _, _g = load_problem(expr)
                                        if _g is None:
                                            return {"success": False, "delta": 0.0, "transforms": []}
                                        class _P:
                                            pass
                                        _p = _P(); _p.graph = _g; _p.id = "te_test"
                                        _p.domain = "general"; _p.expression = expr; _p.py_graph = True
                                        # Use a fast budget to avoid blocking
                                        _saved_bw = _r.beam_width
                                        _saved_bs = _r.budget_seconds
                                        _r.beam_width = min(_r.beam_width, 8)
                                        _r.budget_seconds = min(_r.budget_seconds, 1.0)
                                        try:
                                            _res = _r._run_single(_p)
                                        finally:
                                            _r.beam_width = _saved_bw
                                            _r.budget_seconds = _saved_bs
                                        _steps = list(getattr(_res, "proof_steps", None) or [])
                                        return {"success": bool(getattr(_res, "solved", False)),
                                                "delta": float(getattr(_res, "energy_before", 0) - getattr(_res, "energy_after", 0)),
                                                "transforms": _steps}
                                    except Exception:
                                        return {"success": False, "delta": 0.0, "transforms": []}
                                _tp = _domain_probs.get(_hyp.target_domain, list(_seeds)[:3])
                                _te_ref.test_hypothesis(_hyp, solve_fn=_solve_fn, test_problems=_tp[:3])
                        import threading as _te_t
                        _te_t.Thread(
                            target=_run_hyp_testing,
                            args=(_untested, _te, experiment_runner, _DOMAIN_TEST_PROBS, _SEEDS),
                            daemon=True, name="te-hyp-test",
                        ).start()
                        _tested = sum(1 for h in _te._hypotheses.values() if h.status != "untested")
                        _verified = sum(1 for h in _te._hypotheses.values() if h.status == "verified")
                        log.debug("[TransferEngine] Launched bg test for %d hypotheses", len(_untested))

                _te.save()
            except Exception as _te_exc:
                log.debug("TransferEngine sweep error: %s", _te_exc)

        # Item 5: Compositional problem generator — inject KB fact-pair inference problems every 20 cycles
        if _run_heavy and cycle % 20 == 0:
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
        if _run_heavy and cycle % 30 == 0:
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
        if _run_heavy and cycle % 50 == 0 and forgetting_prevention is not None:
            try:
                forgetting_prevention.consolidate()
                forgetting_prevention.save()
            except Exception:
                pass

        # Belief expiry every 50 cycles — decay stale beliefs older than 24h
        if _run_heavy and cycle % 50 == 0:
            try:
                from sare.memory.world_model import get_world_model as _gwm_exp
                _decayed = _gwm_exp().expire_stale_beliefs(max_age_seconds=86400)
                if _decayed:
                    log.info("[BeliefExpiry] Decayed %d stale beliefs (>24h old)", _decayed)
            except Exception as _exp_exc:
                log.debug("BeliefExpiry error: %s", _exp_exc)

        # Knowledge corpus refresh every 100 cycles — picks up new benchmark/doc files
        if _run_heavy and cycle % 100 == 0:
            try:
                from sare.knowledge.knowledge_ingester import KnowledgeIngester
                _ki_refresh = KnowledgeIngester()
                _ki_refresh.ingest_local_corpus()  # idempotent — skips already-seen files
                log.debug("[KnowledgeIngester] Periodic corpus refresh at cycle %d", cycle)
            except Exception as _ki_exc:
                log.debug("KnowledgeIngester periodic refresh error: %s", _ki_exc)

        # Save commonsense KB every 20 cycles so learned Q&A facts survive restarts
        if cycle % 20 == 0:
            try:
                from sare.knowledge.commonsense import get_commonsense_base
                _cs_save = get_commonsense_base()
                _cs_save.save()
                log.debug("[CommonsenseKB] Saved %d facts at cycle %d", _cs_save.total_facts(), cycle)
            except Exception as _cs_save_exc:
                log.debug("CommonsenseKB save error: %s", _cs_save_exc)

        # Book scan every 50 cycles — pick up any new .txt/.pdf/.epub dropped in data/books/
        if _run_heavy and cycle % 50 == 0:
            try:
                from sare.knowledge.book_ingester import get_book_ingester
                _new_books = get_book_ingester().scan_new_books()
                if _new_books:
                    for _b in _new_books:
                        log.info(
                            "[BookIngester] Ingested '%s' → %d chunks, %d concepts, %d Q&A",
                            _b.title, _b.chunks, _b.concepts, _b.questions,
                        )
            except Exception as _bi_exc:
                log.debug("BookIngester scan error: %s", _bi_exc)

        # LLM knowledge expansion every 100 cycles — generate commonsense facts
        if _run_heavy and cycle % 100 == 0:
            try:
                import threading as _llm_kg_threading
                from sare.knowledge.commonsense import get_commonsense_base as _get_csb_llm
                _csb_llm = _get_csb_llm()
                # Cap at 5K LLM-generated facts total (estimate ~300 per call)
                if _csb_llm.total_facts() < 50_000:
                    def _bg_llm_expand():
                        try:
                            _n = _csb_llm.augment_from_llm(n_facts=100)
                            log.info("[LLM-KG] +%d commonsense facts (total: %d)", _n, _csb_llm.total_facts())
                        except Exception as _e:
                            log.debug("[LLM-KG] expansion error: %s", _e)
                    _llm_kg_t = _llm_kg_threading.Thread(target=_bg_llm_expand, daemon=True, name="llm-kg-expand")
                    _llm_kg_t.start()
            except Exception as _llm_kg_exc:
                log.debug("[LLM-KG] setup error: %s", _llm_kg_exc)

        # Schema generalization: deduplicate similar proof patterns every 50 cycles
        if _run_heavy and cycle % 50 == 0:
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
        # SelfImprover: LLM-driven code improvement every 100 cycles
        if _run_heavy and cycle % 100 == 0:
            try:
                import threading as _si_threading
                from sare.meta.self_improver import SelfImprover as _SelfImprover
                _no_match = (experiment_runner._domain_consec_fails or {}) if experiment_runner else {}
                _worst_domain = max(_no_match, key=_no_match.get) if _no_match else None
                _si_target = None
                if _worst_domain:
                    _si_target = "sare/curiosity/experiment_runner.py"
                _si_goal = (
                    f"reduce no_matching_transforms failures in domain={_worst_domain}"
                    if _worst_domain else "improve overall solve rate"
                )
                def _run_self_improve():
                    try:
                        _si = _SelfImprover()
                        _res = _si.run_once(
                            target_file=_si_target,
                            improvement_type="optimize",
                        )
                        log.info("[SelfImprover] cycle %d result: %s", cycle, _res.get("outcome"))
                    except Exception as _si_inner:
                        log.debug("[SelfImprover] inner error: %s", _si_inner)
                _si_t = _si_threading.Thread(target=_run_self_improve, daemon=True, name="self-improver")
                _si_t.start()
                log.info("[SelfImprover] triggered at cycle %d (goal: %s)", cycle, _si_goal)
            except Exception as _si_exc:
                log.debug("[SelfImprover] setup error: %s", _si_exc)

        # ── WeaknessDetector: analyze failure patterns every 50 cycles ──────
        if _run_heavy and cycle % 50 == 0:
            try:
                from sare.meta.weakness_detector import get_weakness_detector as _get_wd
                _wd = _get_wd()
                _wd_report = _wd.analyze()
                _wd_patterns = _wd_report.get("patterns", [])
                if _wd_patterns:
                    log.info("[WeaknessDetector] cycle %d: %d patterns identified, top: %s",
                             cycle, len(_wd_patterns), _wd_patterns[0].get("description", "")[:60])
                    if _general_curriculum is not None:
                        try:
                            from sare.learning.general_curriculum import inject_priority_problems

                            _weak_debt = []
                            for _pattern in _wd_patterns[:3]:
                                _dom = str(_pattern.get("domain", "general") or "general")
                                for _i in range(2):
                                    _probe = _general_curriculum.generate_one(_dom if _dom != "general" else None)
                                    if _probe is None:
                                        continue
                                    _probe.metadata = {
                                        **dict(getattr(_probe, "metadata", {}) or {}),
                                        "priority_reason": "weakness_detector",
                                        "weakness_description": _pattern.get("description", ""),
                                    }
                                    _weak_debt.append(_probe)
                                _wd.mark_controller_action(_pattern.get("description", ""))
                            if _weak_debt:
                                inject_priority_problems(_weak_debt[:6])
                                log.info("[WeaknessDetector] injected %d weak-domain debt problems", len(_weak_debt[:6]))
                        except Exception as _wd_debt_exc:
                            log.debug("[WeaknessDetector] debt injection error: %s", _wd_debt_exc)
                    # If patterns exist, pass top synthesis requests to TransformSynthesizer
                    _wd_reqs = _wd.get_synthesis_requests(top_k=2)
                    if _wd_reqs and experiment_runner is not None:
                        for _req in _wd_reqs:
                            try:
                                from sare.meta.transform_synthesizer import TransformSynthesizer as _TS
                                _ts = _TS()
                                _cands = _ts.synthesize_transforms(
                                    domain=_req.get("domain", "general"),
                                    n=2,
                                    context=_req.get("pattern_description", ""),
                                )
                                if _cands:
                                    _verified = [c for c in _cands
                                                 if experiment_runner._verify_synthesized_transform(c, _req.get("domain", "general"), n_tests=3)]
                                    if _verified:
                                        experiment_runner.transforms.extend(_verified)
                                        _wd.mark_controller_action(_req.get("pattern_description", ""))
                                        log.info("[WeaknessDetector] injected %d synthesized transforms for %s",
                                                 len(_verified), _req.get("domain"))
                            except Exception as _wd_syn_exc:
                                log.debug("[WeaknessDetector] synthesis error: %s", _wd_syn_exc)
            except Exception as _wd_exc:
                log.debug("[WeaknessDetector] analysis error: %s", _wd_exc)

        # ── brain_state.json sync: compute real stage from general_solver_stats ─
        if cycle % 100 == 0:
            try:
                import json as _bsj, os as _bsos, math as _bsm
                _gs_path = REPO_ROOT / "data" / "memory" / "general_solver_stats.json"
                _bs_path = REPO_ROOT / "data" / "memory" / "brain_state.json"
                _gs = _bsj.loads(_gs_path.read_text()) if _gs_path.exists() else {}
                _bs_total_attempts = sum(v.get("attempts", 0) for v in _gs.values())
                _bs_total_solved   = sum(v.get("solved",   0) for v in _gs.values())
                _bs_solve_rate     = _bs_total_solved / max(_bs_total_attempts, 1)
                _bs_domains        = [d for d, v in _gs.items() if v.get("attempts", 0) >= 50]
                # Load existing to preserve rules_promoted / rules_discovered
                _bs_existing = {}
                if _bs_path.exists():
                    try:
                        _bs_existing = _bsj.loads(_bs_path.read_text())
                    except Exception:
                        pass
                _bs_stats = _bs_existing.get("stats", {})
                _bs_rules_promoted   = _bs_stats.get("rules_promoted", 0)
                _bs_rules_discovered = _bs_stats.get("rules_discovered", 0)
                _bs_rules_count      = _bs_rules_promoted + _bs_rules_discovered
                # Compute stage using same thresholds as STAGE_REQUIREMENTS
                _STAGES = [
                    ("infant",      0,   0,  0.0),
                    ("toddler",     3,   1,  0.3),
                    ("child",       8,   2,  0.5),
                    ("preteen",    20,   3,  0.65),
                    ("teenager",   40,   5,  0.75),
                    ("undergraduate", 80, 8, 0.85),
                    ("graduate",  150,  12,  0.90),
                    ("researcher", 300, 15,  0.95),
                ]
                _bs_stage = "infant"
                for _sname, _sr, _sd, _ssr in _STAGES:
                    if _bs_rules_count >= _sr and len(_bs_domains) >= _sd and _bs_solve_rate >= _ssr:
                        _bs_stage = _sname
                _bs_new = {
                    "stage": _bs_stage,
                    "stats": {
                        "solves_attempted": _bs_total_attempts,
                        "solves_succeeded": _bs_total_solved,
                        "rules_discovered": _bs_rules_discovered,
                        "rules_promoted":   _bs_rules_promoted,
                        "domains_mastered": _bs_domains,
                        "runtime_errors":   _bs_stats.get("runtime_errors", 0),
                    },
                    "total_solves": _bs_total_solved,
                    "total_rules_learned": _bs_rules_count,
                    "solve_rate": round(_bs_solve_rate, 4),
                    "domains": _bs_domains,
                    "cpp_enabled": _bs_existing.get("cpp_enabled", True),
                    "boot_time": _bs_existing.get("boot_time", time.time()),
                    "saved_at": time.time(),
                }
                _bs_tmp = _bs_path.with_suffix(".json.tmp")
                _bs_tmp.write_text(_bsj.dumps(_bs_new, indent=2))
                _bsos.replace(_bs_tmp, _bs_path)
                if cycle % 500 == 0:
                    log.info("[BrainState] stage=%s solves=%d rate=%.1f%% domains=%d",
                             _bs_stage, _bs_total_solved, _bs_solve_rate*100, len(_bs_domains))
            except Exception as _bse:
                log.debug("[BrainState] sync error: %s", _bse)

        if _run_heavy and cycle % 50 == 0:
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
            if turbo:
                # Zero-sleep turbo: yield to OS scheduler briefly
                time.sleep(0.001)
            else:
                time.sleep(interval)

    log.info("learn_daemon stopped after %d cycles.", cycle)

    # Remove PID file on clean shutdown
    try:
        _PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass

    # Shutdown turbo thread pool
    if _turbo_pool is not None:
        try:
            _turbo_pool.shutdown(wait=False)
        except Exception:
            pass

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
    parser.add_argument("--interval",   type=float, default=15.0,
                        help="Seconds between batches (default: 15)")
    parser.add_argument("--batch-size", type=int,   default=10,
                        help="Problems per batch (default: 10)")
    parser.add_argument("--verbose",    action="store_true",
                        help="Enable DEBUG logging")
    parser.add_argument("--fast",       action="store_true",
                        help="Fast training mode: interval=0.5s, batch=50")
    parser.add_argument("--turbo",       action="store_true",
                        help="Turbo mode: interval=0, batch=200, async periodic tasks, 50K+/hr")
    parser.add_argument("--turbo-learn", action="store_true",
                        help="Turbo-Learn mode: interval=3s, batch=20, fast_ratio=20, beam=6 — 3x throughput with full learning")
    args = parser.parse_args()

    if args.turbo:
        args.interval   = 0.0
        args.batch_size = 200
    elif getattr(args, 'turbo_learn', False):
        args.interval   = 3.0
        args.batch_size = 20
    elif args.fast:
        args.interval   = 0.5
        args.batch_size = 50

    return run_daemon(
        interval=args.interval,
        batch_size=args.batch_size,
        verbose=args.verbose,
        turbo=getattr(args, 'turbo', False),
        turbo_learn=getattr(args, 'turbo_learn', False),
    )


if __name__ == "__main__":
    raise SystemExit(main())
