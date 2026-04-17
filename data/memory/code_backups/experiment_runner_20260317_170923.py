"""
ExperimentRunner — closes the curiosity loop.

Autonomously picks pending problems from CurriculumGenerator,
solves them with BeamSearch, passes successes through ReflectionEngine
→ CausalInduction → ConceptRegistry.

This is the heartbeat of SARE's self-learning:
  Generate → Attempt → Reflect → Induce → Learn → Repeat
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from sare.curiosity.curriculum_generator import CurriculumGenerator, GeneratedProblem

log = logging.getLogger(__name__)

# ── Types ──────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    problem_id: str
    solved: bool
    energy_before: float = 0.0
    energy_after: float  = 0.0
    rule_name: str       = ""        # Rule extracted (if any)
    rule_promoted: bool  = False     # Did CausalInduction promote it?
    elapsed_ms: float    = 0.0
    reasoning: str       = ""        # CausalInduction verdict
    proof_steps: List[str] = field(default_factory=list)  # Transform chain
    proof_nl: str        = ""        # Natural-language proof narrative


# ── ExperimentRunner ───────────────────────────────────────────

class ExperimentRunner:
    """
    Closes the Curiosity → Solve → Reflect loop.

    Usage (blocking batch):
        runner = ExperimentRunner(curriculum_gen, searcher, energy, ...)
        results = runner.run_batch(n=10)

    Usage (background daemon):
        runner.start_daemon(interval_seconds=30)
        runner.stop_daemon()
    """

    def __init__(
        self,
        curriculum_gen: "CurriculumGenerator",
        searcher,
        energy,
        reflection_engine=None,
        causal_induction=None,
        concept_registry=None,
        transforms=None,
        beam_width: int = 8,
        budget_seconds: float = 5.0,
        **kwargs,  # accept extra kwargs (e.g. graph_converter) gracefully
    ):
        self.curriculum_gen    = curriculum_gen
        self.searcher          = searcher
        self.energy            = energy
        self.reflection_engine = reflection_engine
        self.causal_induction  = causal_induction
        self.concept_registry  = concept_registry
        self.transforms        = transforms or []
        self.beam_width        = beam_width
        self.budget_seconds    = budget_seconds

        self._history: List[ExperimentResult] = []
        self._daemon_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._batch_count = 0

        self._world_model = None
        try:
            from sare.memory.world_model import get_world_model
            self._world_model = get_world_model()
        except Exception:
            pass
        self._failed_predictions: list = []   # (domain, graph_sig, expected_delta, actual_delta)
        self._surprise_threshold = 3.0        # surprise > this triggers failure curriculum (was 1.5, too aggressive)

        try:
            from sare.search.transform_predictor import get_transform_predictor
            self._transform_predictor = get_transform_predictor()
        except Exception:
            self._transform_predictor = None

        # ── Stuck-domain tracking for auto-synthesis ───────────────────
        # {domain: {"count": int, "exprs": [str], "graphs": [Graph]}}
        self._stuck_domains: dict = {}
        self._synthesis_cooldown: dict = {}   # domain → last_synthesis_time
        self._synthesis_trigger = 3           # consecutive stuck before firing LLM

        # Load any synthesized transforms from data/memory/synthesized_modules/
        self._load_synthesized_transforms()

    def _load_synthesized_transforms(self):
        """Load verified synthesized transforms into the live transform set."""
        import importlib.util
        synth_dir = Path(__file__).resolve().parents[3] / "data" / "memory" / "synthesized_modules"
        if not synth_dir.exists():
            return
        existing_names = {t.name() for t in self.transforms if hasattr(t, 'name')}
        for py_file in synth_dir.glob("*.py"):
            try:
                spec = importlib.util.spec_from_file_location(py_file.stem, str(py_file))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                # Find Transform subclasses in the module
                from sare.engine import Transform
                for attr_name in dir(mod):
                    cls = getattr(mod, attr_name)
                    if (isinstance(cls, type) and issubclass(cls, Transform)
                            and cls is not Transform):
                        instance = cls()
                        if instance.name() not in existing_names:
                            self.transforms.append(instance)
                            existing_names.add(instance.name())
                            log.info("Loaded synthesized transform: %s", instance.name())
            except Exception as exc:
                log.debug("Failed to load synthesized %s: %s", py_file.name, exc)

    # ── Auto-synthesis: fire LLM when domain is repeatedly stuck ──

    def _record_stuck(self, problem, domain: str):
        """Track stuck problems per domain; trigger LLM synthesis when threshold hit."""
        d = self._stuck_domains.setdefault(domain, {"count": 0, "exprs": [], "graphs": []})
        d["count"] += 1
        # Store expression label if available
        expr = getattr(problem, "expression", None) or problem.id
        if len(d["exprs"]) < 20:
            d["exprs"].append(expr)
        if hasattr(problem, "graph") and problem.graph is not None and len(d["graphs"]) < 10:
            d["graphs"].append(problem.graph)

        if d["count"] >= self._synthesis_trigger:
            # Check cooldown: don't re-synthesize same domain within 5 minutes
            last = self._synthesis_cooldown.get(domain, 0)
            if time.time() - last > 300:
                self._synthesis_cooldown[domain] = time.time()
                d["count"] = 0  # reset counter after trigger
                self._trigger_synthesis(domain, d["exprs"][:], d["graphs"][:])
                # Also run failure analysis to inject remedial knowledge
                self._trigger_failure_analysis(domain, d["exprs"][:])
                # MetacognitiveController: inner monologue — "why am I stuck?"
                self._trigger_metacognition(domain, d["exprs"][:])
                # ActiveQuestioner: formulate a curiosity question on the stuck graph
                if d["graphs"]:
                    self._trigger_curiosity_question(d["graphs"][0], domain)
                # SymbolCreator: invent a new primitive if stuck
                self._trigger_symbol_creation(domain, d["exprs"][:])
                # CreativityEngine: dream up cross-domain insight
                self._trigger_creativity(domain)

    def _trigger_synthesis(self, domain: str, stuck_exprs: list, stuck_graphs: list):
        """Fire LLM synthesis in a background thread (non-blocking)."""
        import threading

        def _run():
            try:
                from sare.learning.llm_transform_synthesizer import get_llm_synthesizer
                synth = get_llm_synthesizer()
                existing_names = [
                    t.name() for t in self.transforms if hasattr(t, "name")
                ]
                # Use stuck graphs as validation set, fall back to curriculum seeds
                validation = stuck_graphs[:8]
                if len(validation) < 3 and self.curriculum_gen:
                    for seed in (self.curriculum_gen.seed_problems or [])[:5]:
                        if hasattr(seed, "graph"):
                            validation.append(seed.graph)

                log.info(
                    "[AutoSynth] Triggering LLM synthesis for domain=%s "
                    "(%d stuck exprs, %d validation graphs)",
                    domain, len(stuck_exprs), len(validation),
                )
                result = synth.synthesize(
                    domain=domain,
                    stuck_exprs=stuck_exprs,
                    validation_graphs=validation,
                    existing_transform_names=existing_names,
                )
                if result["promoted"]:
                    # Hot-load the new transform into the live set
                    self._load_synthesized_transforms()
                    log.info(
                        "[AutoSynth] New transform promoted: %s (score=%.2f)",
                        result["actual_name"], result["score"],
                    )
                else:
                    log.info("[AutoSynth] Synthesis failed: %s", result["message"])
            except Exception as exc:
                log.warning("[AutoSynth] Error: %s", exc)

        threading.Thread(target=_run, daemon=True, name="AutoSynth").start()

    def _trigger_failure_analysis(self, domain: str, failed_exprs: list):
        """Background thread: ask LLM to diagnose failures and inject remedial knowledge."""
        import threading

        def _run():
            try:
                from sare.interface.llm_bridge import analyze_failures
                wm = self._world_model
                known_links = wm.get_causal_links(domain)[:8] if wm else []
                result = analyze_failures(domain, failed_exprs, known_links)
                if result.get("error"):
                    return
                log.info(
                    "[FailureAnalysis] domain=%s missing_rule=%s +%d facts +%d links",
                    domain, result.get("missing_rule", "?"),
                    result.get("facts_added", 0), result.get("links_added", 0),
                )
                wm = self._world_model
                if wm:
                    wm.log_activity("failure_analysis", domain,
                        f"Missing rule: {result.get('missing_rule','?')} — added {result.get('facts_added',0)} facts, {len(result.get('new_seeds',[]))} practice seeds",
                        facts_added=result.get("facts_added",0),
                        links_added=result.get("links_added",0))
                # Seed the curriculum with LLM-suggested easier problems
                seeds = result.get("new_seeds", [])
                if seeds and self.curriculum_gen:
                    from sare.engine import load_problem as _lp
                    for expr in seeds:
                        try:
                            _, g = _lp(str(expr))
                            if g:
                                self.curriculum_gen.add_seed(g)
                        except Exception:
                            pass
                    log.info("[FailureAnalysis] Added %d practice seeds for %s", len(seeds), domain)
            except Exception as exc:
                log.debug("[FailureAnalysis] Error: %s", exc)

        threading.Thread(target=_run, daemon=True, name="FailureAnalysis").start()

    def _trigger_metacognition(self, domain: str, stuck_exprs: list):
        """Inner monologue: MetacognitiveController breaks stuck problem into sub-goals."""
        def _run():
            try:
                from sare.memory.metacognition import MetacognitiveController
                mc = MetacognitiveController()
                problem_desc = f"Stuck on {domain} domain. Failed expressions: {', '.join(stuck_exprs[:5])}"
                sub_goals = mc.generate_plan(problem_desc)
                # Store sub-goals so they can be retrieved (e.g. via /api/metacognition/plan)
                if not hasattr(self, "_meta_plans"):
                    self._meta_plans = []
                self._meta_plans.append({
                    "domain": domain,
                    "sub_goals": sub_goals,
                    "timestamp": time.time(),
                })
                self._meta_plans = self._meta_plans[-20:]  # keep last 20
                log.info("[Metacognition] Inner monologue for domain=%s: %d sub-goals: %s",
                         domain, len(sub_goals), sub_goals[:3])
                # Homeostasis: metacognitive reflection is a form of self-exploration
                try:
                    from sare.meta.homeostasis import get_homeostatic_system
                    get_homeostatic_system().on_exploration()
                except Exception:
                    pass
            except Exception as exc:
                log.debug("[Metacognition] Error: %s", exc)

        threading.Thread(target=_run, daemon=True, name="Metacognition").start()

    def _trigger_curiosity_question(self, graph, domain: str):
        """ActiveQuestioner: generate a curiosity question for the stuck graph."""
        try:
            from sare.meta.active_questioner import ActiveQuestioner
            aq = ActiveQuestioner()
            question = aq.formulate_question(graph, domain)
            if question:
                if not hasattr(self, "_curiosity_questions"):
                    self._curiosity_questions = []
                self._curiosity_questions.append({
                    "domain": domain,
                    "question": question.get("question", ""),
                    "target_expr": question.get("target_expr", ""),
                    "timestamp": time.time(),
                })
                self._curiosity_questions = self._curiosity_questions[-50:]
                log.info("[Curiosity] Question generated for %s: %s",
                         domain, question.get("question", "")[:80])
        except Exception as exc:
            log.debug("[Curiosity] Error: %s", exc)

    def _trigger_symbol_creation(self, domain: str, stuck_exprs: list):
        """Background: SymbolCreator invents a new primitive for stuck expressions."""
        import threading as _th

        existing = [t.name() for t in self.transforms if hasattr(t, "name")]

        def _run():
            try:
                from sare.neuro.symbol_creator import get_symbol_creator
                sc = get_symbol_creator()
                result = sc.invent(stuck_exprs=stuck_exprs, domain=domain,
                                   existing_transforms=existing)
                if result.get("promoted"):
                    # Hot-load the new symbol into transform list
                    new_tfs = sc.load_promoted_transforms()
                    known = {t.name() for t in self.transforms if hasattr(t, "name")}
                    for t in new_tfs:
                        if hasattr(t, "name") and t.name() not in known:
                            self.transforms.insert(0, t)
                            log.info("[SymbolCreator] Hot-loaded: %s", t.name())
            except Exception as exc:
                log.debug("[SymbolCreator] Error: %s", exc)

        _th.Thread(target=_run, daemon=True, name="SymbolCreator").start()

    def _trigger_creativity(self, domain: str):
        """Background: CreativityEngine dreams up a cross-domain insight."""
        import threading as _th

        def _run():
            try:
                from sare.neuro.creativity_engine import get_creativity_engine
                ce = get_creativity_engine()
                result = ce.dream()
                if result.get("promoted"):
                    new_tfs = ce.load_promoted_transforms()
                    known = {t.name() for t in self.transforms if hasattr(t, "name")}
                    for t in new_tfs:
                        if hasattr(t, "name") and t.name() not in known:
                            self.transforms.insert(0, t)
                            log.info("[Creativity] Hot-loaded: %s", t.name())
            except Exception as exc:
                log.debug("[Creativity] Error: %s", exc)

        _th.Thread(target=_run, daemon=True, name="CreativityDream").start()

    # ── Public: run one problem ────────────────────────────────

    def run_one(self, problem: "GeneratedProblem") -> ExperimentResult:
        t0 = time.time()
        result = ExperimentResult(problem_id=problem.id, solved=False)

        try:
            from sare.engine import EnergyEvaluator
            energy_fn = self.energy if self.energy else EnergyEvaluator()

            # 1. Evaluate initial energy
            e_before = energy_fn.compute(problem.graph).total
            result.energy_before = e_before

            # 2. Run search (use predictor to reorder transforms if available)
            if self._transform_predictor and len(self.transforms) > 3:
                transforms_to_use = self._transform_predictor.predict_best_transforms(
                    problem.graph, self.transforms
                )
            else:
                transforms_to_use = self.transforms

            # ── Pre-solve prediction (world model) ────────────────────────────
            _prediction = None
            _domain = getattr(problem, "domain", "general") or "general"
            if self._world_model:
                try:
                    _prediction = self._world_model.predict_transform(
                        problem.graph, transforms_to_use, domain=_domain
                    )
                    log.debug(
                        "Prediction for %s: transform=%s expected_delta=%.2f conf=%.2f",
                        problem.id, _prediction.transform_name,
                        _prediction.expected_delta, _prediction.confidence,
                    )
                    # Reorder transforms: put predicted-best first, then rest
                    # This gives the search a head start on the most promising path
                    predicted_name = _prediction.transform_name
                    if predicted_name and predicted_name != "unknown":
                        front = [t for t in transforms_to_use
                                 if (t.name() if hasattr(t, 'name') else '') == predicted_name]
                        rest = [t for t in transforms_to_use
                                if (t.name() if hasattr(t, 'name') else '') != predicted_name]
                        if front:
                            transforms_to_use = front + rest
                except Exception:
                    _prediction = None

            search_result = self.searcher.search(
                problem.graph,
                energy_fn,
                transforms_to_use,
                beam_width=self.beam_width,
                budget_seconds=self.budget_seconds,
            )
            e_after = search_result.energy.total
            result.energy_after = e_after
            delta = e_before - e_after
            result.solved = delta > 0.01

            # Capture proof steps (transform chain)
            _transforms_applied = (
                search_result.transforms_applied
                if hasattr(search_result, "transforms_applied")
                else []
            )
            result.proof_steps = list(_transforms_applied)

            # ── Post-solve: record outcome, compute surprise ───────────────────
            _surprise = 0.0
            if self._world_model and _prediction is not None:
                try:
                    _surprise = self._world_model.record_outcome(
                        _prediction,
                        actual_transforms=search_result.transforms_applied if hasattr(search_result, "transforms_applied") else [],
                        actual_delta=delta,
                        domain=_domain,
                    )
                    if _surprise > self._surprise_threshold:
                        log.info(
                            "HIGH SURPRISE %.2f for %s: predicted %.2f got %.2f — queueing for failure curriculum",
                            _surprise, problem.id, _prediction.expected_delta, delta,
                        )
                        self._failed_predictions.append({
                            "domain": _domain,
                            "problem_id": problem.id,
                            "expected_delta": _prediction.expected_delta,
                            "actual_delta": delta,
                            "surprise": _surprise,
                            "transforms_tried": search_result.transforms_applied if hasattr(search_result, "transforms_applied") else [],
                        })
                        # Generate a new variant of this problem for the curriculum
                        if self.curriculum_gen:
                            try:
                                self.curriculum_gen.add_failure_for_retry(problem)
                            except Exception:
                                pass
                except Exception as exc:
                    log.debug("Outcome recording error: %s", exc)

            log.info(
                "Experiment %s: energy %.2f → %.2f (Δ=%.2f) solved=%s",
                problem.id, e_before, e_after, delta, result.solved,
            )

            if result.solved:
                # Build natural-language proof narrative
                try:
                    from sare.meta.proof_builder import ProofBuilder
                    _pb = ProofBuilder()
                    _expr = getattr(problem, "expression", problem.id)
                    _proof = _pb.build(
                        expression=str(_expr),
                        transforms_applied=result.proof_steps,
                        initial_energy=e_before,
                        final_energy=e_after,
                        domain=_domain,
                    )
                    result.proof_nl = _proof.text
                except Exception:
                    pass

                # Record transform observations for predictor
                if self._transform_predictor:
                    try:
                        for step_name in (
                            search_result.transforms_applied
                            if hasattr(search_result, "transforms_applied")
                            else []
                        ):
                            self._transform_predictor.observe(
                                problem.graph, step_name, delta
                            )
                    except Exception:
                        pass

                # Auto-learn from proof: enrich world model + trigger LLM extraction
                if self._world_model:
                    try:
                        solved_expr = getattr(problem, "expression", problem.id)
                        self._world_model.enrich_from_proof(
                            result.proof_steps, _domain, solved_expr, delta
                        )
                        # Inject any LLM-generated problem variants into curriculum
                        variants = self._world_model.pop_pending_variants()
                        for expr, vdomain in variants:
                            try:
                                from sare.engine import load_problem as _lp
                                _, vg = _lp(expr)
                                if vg and self.curriculum_gen:
                                    self.curriculum_gen.add_seed(vg)
                            except Exception:
                                pass
                    except Exception as _e:
                        log.debug("enrich_from_proof error: %s", _e)

                # Mark the problem solved and add result as new seed
                self.curriculum_gen.mark_solved(problem.id)
                self.curriculum_gen.add_seed(search_result.graph)

                # 3. Reflect: extract candidate rule
                if self.reflection_engine:
                    try:
                        rule = self.reflection_engine.reflect(
                            problem.graph, search_result.graph,
                            transforms_applied=result.proof_steps,
                            domain=_domain,
                        )
                        if rule and rule.valid():
                            result.rule_name = rule.name

                            # 4. Causal Induction: test rule before accepting
                            if self.causal_induction is not None and self.concept_registry is not None:
                                try:
                                    induction = self.causal_induction.evaluate(
                                        rule, energy_fn
                                    )
                                    result.rule_promoted = induction.promoted
                                    result.reasoning     = induction.reasoning
                                except Exception as ci_exc:
                                    log.warning("CausalInduction error for %s: %s", rule.name, ci_exc)
                                    # Promote anyway if CI fails
                                    result.rule_promoted = True
                                    result.reasoning = "CI error, auto-promoted"

                                if result.rule_promoted:
                                    try:
                                        self.concept_registry.add_rule(rule)
                                    except Exception as cr_exc:
                                        log.warning("ConceptRegistry.add_rule error: %s", cr_exc)
                                    _score = getattr(induction, "evidence_score", 0.0) if "induction" in dir() else 0.0
                                    log.info(
                                        "Rule '%s' PROMOTED (score=%.2f)",
                                        rule.name, _score,
                                    )
                                    try:
                                        from sare.memory.world_model import on_rule_promoted
                                        _domain = getattr(rule, "domain", "general")
                                        _pattern = getattr(rule, "pattern", str(rule.name))
                                        consistency = on_rule_promoted(rule.name, _domain, _pattern)
                                        if not consistency["consistent"]:
                                            log.warning(
                                                "[WorldModel] Rule %s conflicts: %s",
                                                rule.name, consistency["conflicts"],
                                            )
                                    except Exception:
                                        pass
                                else:
                                    log.info(
                                        "Rule '%s' rejected: %s",
                                        rule.name, induction.reasoning,
                                    )
                            elif self.concept_registry is not None:
                                # No causal induction: add directly with prior confidence
                                self.concept_registry.add_rule(rule)
                                result.rule_promoted = True
                    except Exception as exc:
                        log.warning("Reflection failed for %s: %s", problem.id, exc)
            else:
                self.curriculum_gen.mark_stuck(problem.id)
                # Track stuck domain for auto-synthesis
                self._record_stuck(problem, _domain)

            # ── Autobiographical memory recording ─────────────────
            try:
                from sare.memory.autobiographical import get_autobiographical_memory
                _am = get_autobiographical_memory()
                _domain = getattr(problem, "domain", "general") or "general"
                if result.solved:
                    _am.record(
                        "rule_applied",
                        _domain,
                        f"Solved via {result.rule_name}" if result.rule_name else "Solved",
                        [result.rule_name] if result.rule_name else [],
                        importance=0.3,
                    )
                # Homeostasis: tick drives on each solve
                try:
                    from sare.meta.homeostasis import get_homeostatic_system
                    _hs = get_homeostatic_system()
                    if result.solved:
                        _hs.on_problem_solved()
                    if result.rule_promoted:
                        _hs.on_rule_discovered()
                    # on_domain_mastered: fire when this domain's solve rate passes 90%
                    _domain_solves = [r for r in self._history[-30:] if getattr(r, 'problem_id', '').startswith(('gen_', 'qa_', 'plan_', 'code_', 'ext_'))]
                    _domain_solved = sum(1 for r in _domain_solves if r.solved)
                    if len(_domain_solves) >= 10 and _domain_solved / len(_domain_solves) >= 0.9:
                        _hs.on_domain_mastered()
                except Exception:
                    pass

                # ── Dopamine reward signal ─────────────────────────────
                try:
                    from sare.neuro.dopamine import get_dopamine_system
                    _ds = get_dopamine_system()
                    _delta = result.energy_before - result.energy_after
                    if result.solved:
                        # Novel problem = bigger reward
                        _is_novel = _domain not in [
                            r.problem_id.split("_")[0] for r in self._history[-20:]
                            if r.solved
                        ]
                        _evt = "solve_novel" if _is_novel else "solve_known"
                        _ds.receive_reward(_evt, domain=_domain, delta=_delta)
                    else:
                        _ds.receive_reward("stuck", domain=_domain, delta=0.0)
                    if result.rule_promoted:
                        _ds.receive_reward("rule_promoted", domain=_domain, delta=_delta)
                    _ds.tick(elapsed_seconds=result.elapsed_ms / 1000.0)
                except Exception:
                    pass
            except Exception:
                pass

        except Exception as exc:
            log.error("ExperimentRunner error for %s: %s", problem.id, exc)
        finally:
            result.elapsed_ms = (time.time() - t0) * 1000
            self._history.append(result)

        return result

    # ── Public: batch run ─────────────────────────────────────

    def run_batch(self, n: int = 5) -> List[ExperimentResult]:
        """Pick up to `n` pending problems and run them."""
        # Generate new problems if the queue is low
        pending = self.curriculum_gen.pending_problems()
        if len(pending) < n:
            self.curriculum_gen.generate_batch(size=n - len(pending))
            pending = self.curriculum_gen.pending_problems()

        # AttentionBeamScorer: prioritize high-uncertainty / novel problems
        try:
            from sare.search.attention_beam import AttentionBeamScorer as _ABS
            from sare.engine import EnergyEvaluator as _EV
            if pending and len(pending) > n:
                _abs = _ABS()
                _abs.sync_dopamine()  # update gamma from current curiosity_bonus
                _ev = _EV()
                _scored = []
                for _p in pending:
                    try:
                        _e = _ev.compute(_p.graph).total
                        _state = _abs.score_state(_p.graph, _e)
                        # Lower attention_score = more interesting (uncertainty+novelty bonus)
                        _scored.append((_state.attention_score, _p))
                    except Exception:
                        _scored.append((999.0, _p))
                _scored.sort(key=lambda x: x[0])
                pending = [_p for _, _p in _scored]
        except Exception:
            pass

        results = []
        for problem in pending[:n]:
            results.append(self.run_one(problem))

        self._batch_count += 1
        if self._transform_predictor and self._batch_count % 10 == 0:
            try:
                self._transform_predictor.save()
            except Exception:
                pass

        # Every 30 batches: use WorldModel.imagine() to inject hypothetical problems
        if self._batch_count % 30 == 0 and self._world_model and self._batch_count > 0:
            try:
                from sare.engine import load_problem as _lp_img
                # Imagine from the most-surprised domain
                high_surprise = self._world_model.get_high_surprise_domains(top_n=1)
                _img_domain = high_surprise[0][0] if high_surprise else "algebra"
                hypotheses = self._world_model.imagine(_img_domain, depth=1)
                _img_added = 0
                for hyp in hypotheses[:5]:
                    expr = hyp.get("hypothesis", "")
                    # Only inject well-formed expression-like hypotheses
                    if expr and any(op in expr for op in ["+", "-", "*", "/", "^", "=", "x", "y"]):
                        try:
                            _, g = _lp_img(expr)
                            if g:
                                self.curriculum_gen.add_seed(g)
                                _img_added += 1
                        except Exception:
                            pass
                if _img_added:
                    log.info("[Imagine] Injected %d hypothetical seeds from domain '%s'", _img_added, _img_domain)
            except Exception as _img_e:
                log.debug("[Imagine] Error: %s", _img_e)

        # Every 20 batches: trigger schema learning for the most active domain
        if self._batch_count % 20 == 0 and self._world_model:
            try:
                high_surprise = self._world_model.get_high_surprise_domains(top_n=1)
                if high_surprise:
                    _learn_domain = high_surprise[0][0]
                    import threading as _th
                    _th.Thread(
                        target=self._world_model.learn_schema_from_llm,
                        args=(_learn_domain,),
                        daemon=True,
                        name="SchemaLearner",
                    ).start()
            except Exception:
                pass

        # Every 50 batches: LLM reflects on performance and restructures knowledge
        if self._batch_count % 50 == 0 and self._batch_count > 0:
            self._trigger_reflection()

        return results

    def _trigger_reflection(self):
        """Background LLM reflection: reviews stats and enriches world model."""
        import threading as _th

        def _run():
            try:
                from sare.interface.llm_bridge import reflect_and_plan
                stats = self.stats()
                wm = self._world_model
                high_surprise = wm.get_high_surprise_domains(top_n=3) if wm else []
                high_surprise_names = [d for d, _ in high_surprise]
                recent_rules = [
                    r.rule_name for r in self._history[-30:]
                    if r.rule_promoted and r.rule_name
                ]
                result = reflect_and_plan(stats, high_surprise_names, recent_rules)
                if result.get("error"):
                    return
                focus = result.get("curriculum_focus", "")
                gaps = result.get("knowledge_gaps", [])
                log.info(
                    "[Reflection] batch=%d focus='%s' +%d facts +%d links gaps=%d",
                    self._batch_count, focus[:60],
                    result.get("facts_added", 0), result.get("links_added", 0),
                    len(gaps),
                )
                wm = self._world_model
                if wm:
                    wm.log_activity("reflection", "general",
                        f"Focus: {focus[:80]}  |  {len(gaps)} knowledge gaps found",
                        facts_added=result.get("facts_added", 0),
                        links_added=result.get("links_added", 0))
            except Exception as exc:
                log.debug("[Reflection] Error: %s", exc)

        _th.Thread(target=_run, daemon=True, name="LLMReflection").start()

    # ── Public: stats ─────────────────────────────────────────

    @property
    def history(self) -> List[ExperimentResult]:
        return list(self._history)

    def stats(self) -> dict:
        total   = len(self._history)
        solved  = sum(1 for r in self._history if r.solved)
        promoted = sum(1 for r in self._history if r.rule_promoted)
        avg_ms  = (
            sum(r.elapsed_ms for r in self._history) / total if total else 0.0
        )
        return {
            "total":         total,
            "solved":        solved,
            "solve_rate":    solved / total if total else 0.0,
            "rules_promoted": promoted,
            "avg_ms":        round(avg_ms, 1),
        }

    def surprise_stats(self) -> dict:
        """Return prediction accuracy and high-surprise domains."""
        base = {
            "failed_predictions": len(self._failed_predictions),
            "recent_failures": self._failed_predictions[-5:],
        }
        if self._world_model:
            try:
                base.update(self._world_model.prediction_stats())
            except Exception:
                pass
        return base

    # ── Daemon mode ───────────────────────────────────────────

    def start_daemon(self, interval_seconds: float = 30.0, batch_size: int = 5):
        """Run experiments in a background thread."""
        if self._daemon_thread and self._daemon_thread.is_alive():
            return  # Already running
        self._stop_event.clear()

        def _loop():
            log.info("ExperimentRunner daemon started (interval=%.0fs)", interval_seconds)
            while not self._stop_event.is_set():
                try:
                    self.run_batch(n=batch_size)
                except Exception as exc:
                    log.error("Daemon batch error: %s", exc)
                self._stop_event.wait(interval_seconds)
            log.info("ExperimentRunner daemon stopped.")

        self._daemon_thread = threading.Thread(target=_loop, daemon=True, name="ExperimentRunner")
        self._daemon_thread.start()

    def stop_daemon(self):
        self._stop_event.set()
        if self._daemon_thread:
            self._daemon_thread.join(timeout=5)
