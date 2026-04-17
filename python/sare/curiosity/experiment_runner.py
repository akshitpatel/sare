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
import os
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from sare.curiosity.curriculum_generator import CurriculumGenerator

log = logging.getLogger(__name__)

# Stage advancement thresholds (mirrors brain.py STAGE_REQUIREMENTS)
_STAGE_REQS = [
    ("infant",      0,   0,  0.0),
    ("toddler",     3,   1,  0.3),
    ("child",       8,   2,  0.5),
    ("preteen",    20,   3,  0.65),
    ("teenager",   40,   5,  0.75),
    ("undergrad",  80,   8,  0.85),
    ("graduate",  150,  12,  0.90),
    ("researcher",300,  15,  0.95),
]

def _compute_stage(rules: int, domains: int, solve_rate: float) -> str:
    stage = "infant"
    for name, min_r, min_d, min_sr in _STAGE_REQS:
        if rules >= min_r and domains >= min_d and solve_rate >= min_sr:
            stage = name
    return stage


@dataclass
class ExperimentResult:
    problem_id: str
    solved: bool
    energy_before: float = 0.0
    energy_after: float = 0.0
    rule_name: str = ""
    rule_promoted: bool = False
    elapsed_ms: float = 0.0
    reasoning: str = ""
    proof_steps: List[str] = field(default_factory=list)
    proof_nl: str = ""
    domain: str = "general"
    search_strategy: str = "beam_search"
    final_graph: Any = field(default=None, repr=False)  # graph after solving


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
        budget_seconds: float = 8.0,
        analogy_transfer=None,
        **kwargs,
    ):
        self.curriculum_gen = curriculum_gen
        self.searcher = searcher
        self.energy = energy
        self.reflection_engine = reflection_engine
        self.causal_induction = causal_induction
        self.concept_registry = concept_registry
        self._domain_solve_tracker: dict = {}  # domain -> {"attempts": 0, "successes": 0}
        self.transforms = transforms or []
        self.beam_width = beam_width
        self.budget_seconds = budget_seconds
        self.analogy_transfer = analogy_transfer

        self._history: List[ExperimentResult] = []
        self._daemon_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._batch_count = 0
        # Deduplication: track names of rules already successfully promoted this session.
        # Prevents the same rule (e.g. double_negation_elimination) from being logged
        # and re-processed hundreds of times across cycles.
        self._promoted_rule_names: set = set()

        # TransformSynthesizer trigger: track consecutive failures per domain.
        # When a domain accumulates _SYNTH_FAIL_THRESHOLD consecutive failures,
        # TransformSynthesizer is called to attempt LLM-assisted transform synthesis.
        self._domain_consec_fails: dict = {}     # domain -> int
        self._SYNTH_FAIL_THRESHOLD: int = 20     # trigger synthesis after N consecutive failures
        self._synth_triggered_domains: set = set()  # domains where synthesis was triggered this session
        # Domain deprioritization: domains with too many consecutive failures get
        # a cooldown so we stop wasting cycles on problems we cannot solve.
        # Key: domain, Value: batch_count when it can be re-enabled.
        self._domain_cooldown: dict = {}          # domain -> batch_count when it re-enables
        self._DOMAIN_COOLDOWN_BATCHES: int = 30   # skip a domain for 30 batches (~3-5 min)

        self._recent_batch_stats: List[dict] = []
        self._adaptive_batch_size: Optional[int] = None

        # Heuristic pre-scoring: store successful embeddings + transform names
        # so we can cosine-rank transforms before BeamSearch.
        # Each entry: {"embedding": List[float], "transforms": List[str]}
        self._successful_embeddings: List[dict] = []
        self._embedder = None  # lazy-loaded GraphEmbedding instance
        self._value_net = None  # lazy-loaded MLXValueNet (M1 GPU)
        self._value_net_callable = None  # graph → float callable for BeamSearch

        # HTMPredictor: learns transform sequences, used to reorder transforms in search
        self._htm_predictor = None
        try:
            from sare.neuro.htm_predictor import HTMPredictor
            self._htm_predictor = HTMPredictor()
            log.info("[ExperimentRunner] HTMPredictor initialized")
        except Exception as _he:
            log.debug("[ExperimentRunner] HTMPredictor unavailable: %s", _he)

        # TransformPolicy: attention-based neural transform ranker
        self._transform_policy = None
        try:
            from sare.heuristics.transform_policy import get_transform_policy
            _t_names = [t.name() for t in (transforms or [])]
            self._transform_policy = get_transform_policy(_t_names)
            log.info("[ExperimentRunner] TransformPolicy initialized (%d transforms)", len(_t_names))
        except Exception as _tpe:
            log.debug("[ExperimentRunner] TransformPolicy unavailable: %s", _tpe)
        self._last_wm_prediction = None  # world model prediction for current problem

        # P2-G: per-transform uncertainty tracking (populated from credit assigner)
        self._transform_uncertainty: dict = {}
        self._solve_count: int = 0   # used for composite learner mine frequency

        # P3-H: A* fallback for hard problems
        self._use_astar_fallback: bool = True

        # Fix 4: cache of stuck graphs per domain for synthesis validation
        from collections import defaultdict
        self._stuck_graph_cache: dict = defaultdict(list)

        # ── Stuck-problem quarantine ────────────────────────────────────────────
        # Problems that fail repeatedly burn CPU with no progress.
        # After _QUARANTINE_FAIL_LIMIT consecutive failures the problem is
        # quarantined for _QUARANTINE_COOLDOWN_BATCHES batches.
        self._problem_consec_fails: dict = {}     # problem_id -> consecutive fail count
        self._problem_quarantine: dict = {}        # problem_id -> batch# when re-enabled
        self._QUARANTINE_FAIL_LIMIT: int = 8       # quarantine after 8 straight failures
        self._QUARANTINE_COOLDOWN_BATCHES: int = 500  # ~45 min @ 1 batch/5s

        # Load synthesized transforms from data/memory/synthesized_modules/
        synth = self._load_synthesized_transforms()
        if synth:
            self.transforms = list(self.transforms) + synth

        # Fix 2: cold-start registry load — inject concept rules from registry file
        self._refresh_concept_transforms()

        # C++ BeamSearch: load bindings for 10-50x speedup on math/logic problems
        self._cpp_run_beam_search = None
        self._cpp_search_config_cls = None
        self._cpp_graph_to_py = None
        self._cpp_py_to_graph = None
        self._cpp_ready = False
        try:
            import sare.sare_bindings as _sb
            from sare.core.graph_bridge import cpp_graph_to_py_graph, py_graph_to_cpp_graph
            self._cpp_run_beam_search = getattr(_sb, "run_beam_search", None)
            self._cpp_search_config_cls = getattr(_sb, "SearchConfig", None)
            self._cpp_graph_to_py = cpp_graph_to_py_graph
            self._cpp_py_to_graph = py_graph_to_cpp_graph
            self._cpp_ready = bool(self._cpp_run_beam_search and self._cpp_search_config_cls)
            if self._cpp_ready:
                log.info("[ExperimentRunner] C++ BeamSearch ready (10-50x speedup for math/logic)")
        except Exception as _cpp_e:
            log.debug("[ExperimentRunner] C++ bindings unavailable: %s", _cpp_e)

    def _pick_next_problem(self):
        candidates = [
            "next_problem",
            "get_next_problem",
            "sample_problem",
            "generate_problem",
            "next",
            "pop",
        ]
        for name in candidates:
            fn = getattr(self.curriculum_gen, name, None)
            if callable(fn):
                try:
                    return fn()
                except TypeError:
                    continue
        # Fall back to generate_batch(size=1) and return the first result
        gen_batch = getattr(self.curriculum_gen, "generate_batch", None)
        if callable(gen_batch):
            batch = gen_batch(size=1)
            if batch:
                return batch[0]
        raise AttributeError("CurriculumGenerator does not expose a supported problem retrieval method")

    def _load_synthesized_transforms(self) -> list:
        """Load verified Transform subclasses from data/memory/synthesized_modules/."""
        import importlib.util
        from pathlib import Path
        from sare.engine import Transform
        synth_dir = Path(__file__).resolve().parents[3] / "data" / "memory" / "synthesized_modules"
        loaded = []
        if not synth_dir.exists():
            return loaded
        for py_file in synth_dir.glob("*.py"):
            try:
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                for name, obj in vars(mod).items():
                    if (isinstance(obj, type) and issubclass(obj, Transform)
                            and obj is not Transform):
                        loaded.append(obj())
                        log.info("[ExperimentRunner] Loaded synth transform: %s", name)
            except Exception as e:
                log.warning("[ExperimentRunner] Failed to load %s: %s", py_file.name, e)
        return loaded

    def _maybe_reload_synthesized(self) -> int:
        """Hot-reload any new .py files written to synthesized_modules/ since last check.

        Called at the start of every run_batch() so transforms discovered by the Oracle
        pipeline during the same session are immediately available — not just on restart.
        """
        import importlib.util
        from pathlib import Path
        from sare.engine import Transform
        synth_dir = Path(__file__).resolve().parents[3] / "data" / "memory" / "synthesized_modules"
        if not synth_dir.exists():
            return 0
        if not hasattr(self, "_loaded_synth_files"):
            # First call: seed the known-file set without loading (already loaded at __init__)
            self._loaded_synth_files = {f.name for f in synth_dir.glob("*.py")}
            return 0
        new_files = [f for f in synth_dir.glob("*.py") if f.name not in self._loaded_synth_files]
        if not new_files:
            return 0
        known_names = {t.name() for t in self.transforms}
        added = 0
        for f in new_files:
            self._loaded_synth_files.add(f.name)
            try:
                spec = importlib.util.spec_from_file_location(f.stem, f)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                for attr, obj in vars(mod).items():
                    if (isinstance(obj, type) and issubclass(obj, Transform)
                            and obj is not Transform):
                        inst = obj()
                        if inst.name() not in known_names:
                            self.transforms = list(self.transforms) + [inst]
                            known_names.add(inst.name())
                            added += 1
                            log.info("[ExperimentRunner] Hot-loaded new transform: %s", inst.name())
            except Exception as e:
                log.debug("[ExperimentRunner] Hot-reload failed for %s: %s", f.name, e)
        return added

    def _refresh_concept_transforms(self) -> None:
        """Fix 2: Refresh concept-rule transforms without requiring Brain wrapper."""
        if self.concept_registry is None:
            return
        try:
            from sare.memory.concept_rule import concept_transforms_from_registry
            new_concept = concept_transforms_from_registry(self.concept_registry)
            # Replace existing concept_ transforms, then append new ones
            self.transforms = (
                [t for t in self.transforms if not t.name().startswith("concept_")]
                + new_concept
            )
            if new_concept:
                log.info("[ExperimentRunner] Refreshed concept transforms: +%d rules",
                         len(new_concept))
        except Exception as exc:
            log.debug("[ExperimentRunner] concept transform refresh failed: %s", exc)

    def _evaluate_energy(self, graph) -> float:
        try:
            if callable(self.energy):
                return float(self.energy(graph))
        except Exception:
            pass
        for method in ("compute", "score", "evaluate"):
            fn = getattr(self.energy, method, None)
            if callable(fn):
                try:
                    result = fn(graph)
                    # EnergyEvaluator.compute returns an Energy namedtuple with .total
                    if hasattr(result, "total"):
                        return float(result.total)
                    return float(result)
                except Exception:
                    pass
        return 0.0

    @staticmethod
    def _is_cpp_candidate(graph) -> bool:
        """Check if graph is eligible for C++ BeamSearch (simple math/logic operators only)."""
        _allowed = {"+", "-", "*", "/", "^", "neg", "and", "or", "not", "eq", "="}
        try:
            for node in graph.nodes:
                if getattr(node, "type", "") != "operator":
                    continue
                label = str(getattr(node, "label", "") or "").lower()
                if label and label not in _allowed:
                    return False
            return True
        except Exception:
            return False

    def _search(self, graph, transforms=None):
        kwargs = {
            "beam_width": self.beam_width,
            "budget_seconds": self.budget_seconds,
        }
        if transforms is None:
            transforms = self.transforms

        # P1-A: pass attention scorer to BeamSearch
        try:
            from sare.search.attention_beam import get_default_scorer
            kwargs["attention_scorer"] = get_default_scorer()
        except Exception:
            pass

        # Neural wiring: HTMPredictor (transform reordering) + MLXValueNet (state scoring)
        if self._htm_predictor is not None:
            kwargs["transform_predictor"] = self._htm_predictor
        if self._value_net_callable is None and self._value_net is not None:
            _vn = self._value_net
            _emb_fn = self._get_graph_embedding
            def _vnet_wrapper(g):
                try:
                    emb = _emb_fn(g)
                    if emb:
                        return _vn.score(emb)
                except Exception:
                    pass
                return 0.5
            self._value_net_callable = _vnet_wrapper
        if self._value_net_callable is not None:
            kwargs["value_net"] = self._value_net_callable

        # P2-F: pass heuristic_fn to BeamSearch
        try:
            from sare.brain import get_brain as _get_brain
            _br = _get_brain()
            if _br is not None and hasattr(_br, 'heuristic_model') and _br.heuristic_model is not None:
                kwargs["heuristic_fn"] = _br.heuristic_model.predict_graph
        except Exception:
            pass

        # Try C++ BeamSearch first for eligible graphs (10-50x faster)
        # Only accept the C++ result if it actually improved energy — otherwise fall through to Python.
        if self._cpp_ready and self._is_cpp_candidate(graph):
            try:
                e_before_cpp = self._evaluate_energy(graph)
                cpp_graph = self._cpp_py_to_graph(graph)
                cfg = self._cpp_search_config_cls()
                cfg.beam_width = kwargs.get("beam_width", self.beam_width)
                cfg.max_depth = 24
                cfg.budget_seconds = min(kwargs.get("budget_seconds", self.budget_seconds), 2.0)
                if hasattr(cfg, "kappa"):
                    cfg.kappa = 0.1
                cpp_result = self._cpp_run_beam_search(cpp_graph, cfg)
                best_graph = self._cpp_graph_to_py(cpp_result.best_graph)
                e_total = self._evaluate_energy(best_graph)

                # Only use C++ result if it genuinely improved energy
                if e_before_cpp - e_total > 0.1:
                    class _CppSearchResult:
                        pass
                    r = _CppSearchResult()
                    r.graph = best_graph
                    r.energy = type("_E", (), {"total": e_total})()
                    r.transforms_applied = list(
                        getattr(getattr(cpp_result, "best_state", None), "transform_trace", []) or []
                    )
                    return r
                # C++ didn't improve — fall through to Python BeamSearch
                log.debug("[ExperimentRunner] C++ search no improvement (e_before=%.2f e_after=%.2f), using Python", e_before_cpp, e_total)
            except Exception as _cpp_exc:
                log.debug("[ExperimentRunner] C++ search failed, Python fallback: %s", _cpp_exc)

        if hasattr(self.searcher, "search"):
            try:
                # Try the standard BeamSearch signature: search(graph, energy, transforms, ...)
                return self.searcher.search(graph, self.energy, transforms, **kwargs)
            except TypeError:
                pass
            try:
                return self.searcher.search(graph, **kwargs)
            except TypeError:
                return self.searcher.search(graph)
        if callable(self.searcher):
            try:
                return self.searcher(graph, self.energy, transforms, **kwargs)
            except TypeError:
                pass
            try:
                return self.searcher(graph, **kwargs)
            except TypeError:
                return self.searcher(graph)
        raise AttributeError("Searcher does not expose a supported search method")

    def _parse_search_outcome(self, outcome, original_graph, energy_before: float = 0.0):
        solved = False
        final_graph = original_graph
        proof_steps: List[str] = []
        proof_nl = ""

        if isinstance(outcome, bool):
            solved = outcome
        elif outcome is None:
            solved = False
        else:
            # BeamSearch returns SearchResult with .energy (Energy namedtuple), .graph,
            # .transforms_applied — no .solved field.
            final_graph = getattr(
                outcome,
                "best_graph",
                getattr(outcome, "graph", getattr(outcome, "result_graph", original_graph)),
            )

            # Determine solved: explicit flag first, then energy threshold
            explicit_solved = getattr(outcome, "solved", getattr(outcome, "success", None))
            if explicit_solved is not None:
                solved = bool(explicit_solved)
            else:
                # Solved if energy dropped meaningfully OR final graph is atomic
                outcome_energy = getattr(outcome, "energy", None)
                if outcome_energy is not None:
                    e_total = float(getattr(outcome_energy, "total", outcome_energy) or 0)
                    delta = energy_before - e_total
                    solved = e_total < 1.5 or delta > 0.5

            # Extract proof steps from transforms_applied or proof_steps
            steps = getattr(outcome, "proof_steps",
                            getattr(outcome, "steps",
                                    getattr(outcome, "transforms_applied", None)))
            if steps:
                proof_steps = [str(s) for s in steps]

            proof_nl = str(getattr(outcome, "proof_nl", getattr(outcome, "narrative", "")) or "")

        return solved, final_graph, proof_steps, proof_nl

    def _reflect_and_learn(self, problem, result: ExperimentResult):
        if not result.solved:
            return result

        if self.reflection_engine is not None:
            try:
                # PyReflectionEngine.reflect(initial_graph, final_graph, transforms_applied, domain)
                initial_graph = getattr(problem, "graph", problem)
                final_graph = getattr(result, "final_graph",
                                      getattr(result, "graph", initial_graph))
                transforms_applied = result.proof_steps or []
                domain = getattr(problem, "domain", "general")
                reflection = self.reflection_engine.reflect(
                    initial_graph, final_graph,
                    transforms_applied=transforms_applied, domain=domain
                )
            except TypeError:
                try:
                    reflection = self.reflection_engine.reflect(problem, result)
                except Exception:
                    reflection = None
            except Exception:
                reflection = None
        else:
            reflection = None

        if self.causal_induction is not None:
            # Always call induce() synchronously so verdict is available this cycle.
            # queue_episode() was async and caused verdict=None, blocking all promotions.
            try:
                verdict = self.causal_induction.induce(problem=problem, result=result, reflection=reflection)
            except TypeError:
                try:
                    verdict = self.causal_induction.induce(problem, result, reflection)
                except Exception:
                    verdict = None
            except Exception:
                verdict = None
        else:
            verdict = None

        if verdict is not None:
            # reflection may be the AbstractRule itself; extract name from it first
            _rule_obj = reflection if (reflection is not None and hasattr(reflection, "valid")) else None
            result.rule_name = str(
                getattr(verdict, "rule_name", None)
                or getattr(verdict, "name", None)
                or (getattr(_rule_obj, "name", None) if _rule_obj else None)
                or ""
            )
            result.rule_promoted = bool(getattr(verdict, "promoted", getattr(verdict, "rule_promoted", False)))
            result.reasoning = str(getattr(verdict, "reasoning", getattr(verdict, "verdict", "")) or "")
            # Deduplicate: only process/log a promotion once per session per rule name.
            _is_new_promotion = (
                result.rule_promoted
                and result.rule_name
                and result.rule_name not in self._promoted_rule_names
            )
            if _is_new_promotion:
                self._promoted_rule_names.add(result.rule_name)
                log.info("[ExperimentRunner] Rule promoted (NEW): %s (%s)", result.rule_name, verdict.reasoning)
            elif result.rule_promoted and result.rule_name:
                log.debug("[ExperimentRunner] Rule already promoted, skipping duplicate: %s", result.rule_name)
                result.rule_promoted = False  # suppress downstream re-processing

            if result.rule_promoted and self.concept_registry is not None:
                try:
                    rule_to_register = _rule_obj or verdict
                    if hasattr(self.concept_registry, "add_rule"):
                        self.concept_registry.add_rule(rule_to_register)
                    elif hasattr(self.concept_registry, "register"):
                        self.concept_registry.register(rule_to_register)
                    elif hasattr(self.concept_registry, "promote"):
                        self.concept_registry.promote(rule_to_register)
                except Exception:
                    pass
                # Fix 2+3: Immediately refresh transforms (brain or standalone)
                try:
                    brain = getattr(self, "_brain_ref", None)
                    if brain is not None and hasattr(brain, "_refresh_transforms"):
                        brain._refresh_transforms()
                    else:
                        self._refresh_concept_transforms()
                except Exception as exc:
                    log.debug("Immediate transform refresh failed: %s", exc)
                # Add promoted rule to ConceptHierarchy for cross-domain generalization
                if result.rule_name:
                    try:
                        from sare.memory.concept_hierarchy import get_concept_hierarchy
                        _hier = get_concept_hierarchy()
                        _rule_domain = str(getattr(problem, "domain", "general") or "general")
                        _hier.add_concept(result.rule_name, _rule_domain)
                        if len(_hier._nodes) % 10 == 0:
                            _hier.save()
                    except Exception as _hier_exc:
                        log.debug("[ConceptHierarchy] add_concept error: %s", _hier_exc)
                # Register new rule in TransformPolicy so it can learn its utility
                if self._transform_policy is not None and result.rule_name:
                    try:
                        self._transform_policy.add_transform(result.rule_name)
                    except Exception:
                        pass

            if result.rule_promoted and result.rule_name:
                try:
                    from sare.memory.world_model import get_world_model
                    _rule_domain = str(getattr(problem, "domain", "general") or "general")
                    _rule_pattern = str(
                        getattr(verdict, "pattern", None)
                        or (getattr(_rule_obj, "pattern_description", None) if _rule_obj else None)
                        or ""
                    )
                    get_world_model().observe_rule_promotion(
                        rule_name=result.rule_name,
                        domain=_rule_domain,
                        pattern=_rule_pattern,
                    )
                except Exception:
                    pass

        # T3-3: EWC-lite — record usage of transforms that contributed to solve
        try:
            from sare.learning.forgetting_prevention import get_forgetting_prevention
            _fp = get_forgetting_prevention()
            # Record usage of each transform that contributed to solve
            if result.solved and result.proof_steps:
                for _step in result.proof_steps:
                    _fp.record_usage(_step, confidence=0.8, observations=1)
            # On rule promotion: check if overwrite is allowed
            # (The actual overwrite check happens in SeededConceptRegistry.add_rule)
        except Exception:
            pass

        # Record transfer outcome if the rule used came from a transfer.
        if self.analogy_transfer is not None and result.solved:
            rule_name = result.rule_name
            is_transfer = (
                "_transfer" in rule_name
                or getattr(result, "transferred_from", None) is not None
            )
            if is_transfer:
                domain = str(getattr(problem, "domain", "") or "")
                delta = max(0.0, result.energy_before - result.energy_after)
                try:
                    self.analogy_transfer.record_transfer_outcome(
                        rule_name=rule_name,
                        domain=domain,
                        success=True,
                        delta=delta,
                    )
                except Exception:
                    pass

        # Feed every successful solve into the cross-domain TransferEngine.
        # This is how it accumulates observations to generate transfer hypotheses.
        if result.solved:
            _te_transforms = result.proof_steps or ([result.rule_name] if result.rule_name else [])
            _te_domain = getattr(result, "domain", None) or str(getattr(problem, "domain", "algebra") or "algebra")
            if _te_transforms:
                try:
                    from sare.transfer.engine import get_transfer_engine, RoleClassifier
                    _te = get_transfer_engine()
                    _te.observe(_te_transforms, _te_domain, success=True)

                    # Passive hypothesis testing: verify pending transfers using live solve outcomes.
                    # No separate solve_fn needed — we check if the roles used in THIS solve
                    # match the source_role of any untested hypothesis targeting this domain.
                    _used_roles = {RoleClassifier.classify(t) for t in _te_transforms} - {None}
                    _pending = [
                        h for h in _te._hypotheses.values()
                        if h.status == "untested"
                        and h.target_domain == _te_domain
                        and h.source_role in _used_roles
                    ]
                    if _used_roles and not _pending:
                        log.debug("[TransferPassive] domain=%s roles=%s → no pending hypotheses (total=%d)",
                                  _te_domain, _used_roles, len(_te._hypotheses))
                    elif _pending:
                        log.info("[TransferPassive] domain=%s roles=%s → %d matches, appending test",
                                 _te_domain, _used_roles, len(_pending))
                    for _hyp in _pending[:2]:  # at most 2 per solve to keep overhead low
                        _hyp.test_results.append({
                            "problem": str(getattr(problem, "expression", problem))[:80],
                            "success": result.solved,
                            "delta": float(result.energy_before - result.energy_after),
                        })
                        _n = len(_hyp.test_results)
                        if _n >= 3:
                            _wins = sum(1 for r in _hyp.test_results if r.get("success"))
                            if _wins / _n >= 0.5:
                                _hyp.status = "verified"
                                _hyp.confidence = min(0.95, _hyp.confidence + 0.2)
                                _te._stats["hypotheses_verified"] += 1
                                _te._stats["transfers_promoted"] = _te._stats.get("transfers_promoted", 0) + 1
                                log.info("Transfer verified passively: %s → %s role=%s",
                                         _hyp.source_domain, _hyp.target_domain, _hyp.source_role)
                                _te.save()
                                # Publish to core event bus so brain.py promotes to ConceptRegistry
                                try:
                                    from sare.core.event_bus import get_event_bus as _gceb
                                    _gceb().publish("transfer_verified", {
                                        "name": _hyp.proposed_transform,
                                        "domain": _hyp.target_domain,
                                        "source_domain": _hyp.source_domain,
                                        "confidence": _hyp.confidence,
                                        "pattern": _hyp.proposed_pattern,
                                    })
                                except Exception:
                                    pass
                            elif _wins == 0 and _n >= 5:
                                _hyp.status = "rejected"
                                _hyp.confidence *= 0.3
                                _te._stats["hypotheses_rejected"] += 1
                                _te.save()
                except Exception as _te_exc:
                    log.debug("[TransferPassive] exception in transfer passive testing: %s", _te_exc)

        # Per-domain credit assignment — update transform utilities with domain-specific baseline.
        if result.solved and result.proof_steps:
            try:
                from sare.learning.credit_assignment import CreditAssigner
                _ca = CreditAssigner()
                _ca.load()
                _ca_domain = str(getattr(result, "domain", None) or getattr(problem, "domain", "general") or "general")
                # Build an energy trajectory: start=energy_before, end=energy_after, uniform steps
                _n_steps = len(result.proof_steps)
                _e_start = float(result.energy_before or 0.0)
                _e_end = float(result.energy_after or 0.0)
                _step = (_e_end - _e_start) / _n_steps if _n_steps else 0.0
                _traj = [_e_start + i * _step for i in range(_n_steps + 1)]
                _ca.assign_credit(result.proof_steps, _traj, domain=_ca_domain)
                _ca.save()
                # P2-G: update transform uncertainty from credit utilities
                try:
                    _utilities = _ca.get_all_utilities()
                    for _tname, _util in _utilities.items():
                        self._transform_uncertainty[_tname] = max(0.0, 1.0 - abs(float(_util)))
                    # Push to attention scorer
                    from sare.search.attention_beam import get_default_scorer
                    get_default_scorer().set_transform_uncertainty(self._transform_uncertainty)
                except Exception:
                    pass
            except Exception:
                pass

        # P3-I: CompositeRuleLearner observation and periodic mining
        if result.solved and result.proof_steps:
            try:
                from sare.learning.composite_rule_learner import get_composite_learner
                _cl = get_composite_learner()
                _cl_domain = str(getattr(result, "domain", "general"))
                _cl.observe_trace(result.proof_steps, _cl_domain)
                self._solve_count += 1
                if self._solve_count % 20 == 0:
                    _new_composites = _cl.mine_composites()
                    if _new_composites and self.concept_registry is not None:
                        _existing_names = set()
                        try:
                            if hasattr(self.concept_registry, "get_rules"):
                                for _rule in list(self.concept_registry.get_rules() or []):
                                    _name = getattr(_rule, "name", None) or (_rule.get("name", "") if isinstance(_rule, dict) else "")
                                    if _name:
                                        _existing_names.add(str(_name))
                        except Exception:
                            pass
                        _added = False
                        for _comp in _new_composites:
                            try:
                                _name = str(_comp.get("name", "") or "")
                                if not _name or _name in _existing_names:
                                    continue
                                self.concept_registry.add_rule({
                                    "name": _name,
                                    "pair": _comp["pair"],
                                    "utility": _comp["utility"],
                                    "source": "composite_mining",
                                    "domain": _cl_domain,
                                })
                                _existing_names.add(_name)
                                _added = True
                            except Exception:
                                pass
                        if _added and hasattr(self.concept_registry, "save"):
                            try:
                                self.concept_registry.save()
                            except Exception:
                                pass
                if self._solve_count % 100 == 0:
                    _cl.mine_cross_domain_pairs()
            except Exception:
                pass

        # Self-evaluation: record in LearningMonitor (no benchmark dependency)
        try:
            from sare.meta.learning_monitor import get_learning_monitor
            _initial = getattr(result, "initial_graph", None) or getattr(result, "graph", None)
            if _initial is None:
                _initial = getattr(problem, "graph", problem)
            get_learning_monitor().record(
                _initial,
                solved=bool(getattr(result, "solved", False)),
                energy_before=float(getattr(result, "energy_before", 0.0) or 0.0),
                energy_after=float(getattr(result, "energy_after",  0.0) or 0.0),
                cycle=getattr(self, "_batch_count", 0),
                domain=str(getattr(result, "domain", "general") or "general"),
            )
        except Exception:
            pass

        # T2-3: Publish episode_complete event for ContinuousLearner
        try:
            from sare.core.event_bus import get_event_bus
            get_event_bus().publish("episode_complete", {
                "expression":    str(getattr(result, "problem_id", "")),
                "domain":        str(getattr(result, "domain", "general")),
                "solved":        bool(getattr(result, "solved", False)),
                "proof_steps":   list(getattr(result, "proof_steps", None) or []),
                "energy_before": float(getattr(result, "energy_before", 0) or 0),
                "energy_after":  float(getattr(result, "energy_after", 0) or 0),
            })
        except Exception:
            pass

        return result

    def _get_graph_embedding(self, graph) -> Optional[List[float]]:
        """Embed a graph using the GNN heuristic (MPS-accelerated if available)."""
        try:
            import torch
            from sare.heuristics.graph_embedding import GraphEmbedding
            if self._embedder is None:
                self._embedder = GraphEmbedding()
                self._embedder.eval()

            nodes = getattr(graph, "nodes", [])
            if not nodes:
                return None

            id_to_idx = {n.id: idx for idx, n in enumerate(nodes)}
            type_indices = torch.tensor(
                [self._embedder.encoder.get_type_idx(getattr(n, "type", "unknown") or "unknown")
                 for n in nodes],
                dtype=torch.long,
            )
            adjacency = []
            for edge in getattr(graph, "edges", []):
                src = id_to_idx.get(getattr(edge, "source", None))
                tgt = id_to_idx.get(getattr(edge, "target", None))
                if src is not None and tgt is not None:
                    adjacency.append((src, tgt))

            with torch.no_grad():
                emb = self._embedder(type_indices, adjacency)  # always CPU tensor
            return emb.tolist()
        except Exception as exc:
            log.debug("Graph embedding failed: %s", exc)
            return None

    def _heuristic_reorder_transforms(self, graph) -> List:
        """
        Use cosine similarity against past successful embeddings to reorder
        self.transforms so the most historically-effective ones come first.
        Also blends in the MLX value net score when available.
        Falls back to original order if embedding is unavailable.
        """
        if not self.transforms or not self._successful_embeddings:
            return self.transforms

        emb = self._get_graph_embedding(graph)
        if emb is None:
            return self.transforms

        # Find the most similar past successful episode (cosine similarity)
        best_sim = -1.0
        best_transforms: List[str] = []
        import math
        norm_emb = math.sqrt(sum(v * v for v in emb)) or 1.0

        for entry in self._successful_embeddings[-50:]:  # only last 50 for speed
            past_emb = entry["embedding"]
            dot = sum(a * b for a, b in zip(emb, past_emb))
            norm_past = entry.get("norm", math.sqrt(sum(v * v for v in past_emb)) or 1.0)
            sim = dot / (norm_emb * norm_past)
            if sim > best_sim:
                best_sim = sim
                best_transforms = entry["transforms"]

        if best_sim < 0.5 or not best_transforms:
            return self.transforms  # not similar enough — don't reorder

        # Query MLX value net for current graph state score (blended with cosine)
        value_net_score = None
        try:
            if self._value_net is None:
                from sare.heuristics.mlx_value_net import get_value_net
                self._value_net = get_value_net()
            if self._value_net is not None:
                value_net_score = self._value_net.score(emb)
        except Exception:
            pass  # value net not ready — fall back to cosine-only

        # Per-transform scoring: cosine similarity contributes 0.6,
        # value net (if available) contributes 0.4; credit utility blended in at 0.2
        best_set = set(best_transforms)
        cosine_score = best_sim  # similarity of current graph to best past episode

        # Load persisted credit utilities for domain-aware transform scoring
        _credit_utilities: dict = {}
        try:
            from sare.learning.credit_assignment import CreditAssigner as _CA
            _ca_inst = _CA()
            _ca_inst.load()
            _credit_utilities = _ca_inst.get_all_utilities()
        except Exception:
            pass

        def _transform_score(t) -> float:
            in_best = t.name() in best_set
            cs = cosine_score if in_best else 0.0
            cu = float(_credit_utilities.get(t.name(), 0.0))
            # Normalize credit utility to 0-1 range (clamp at 5.0 as practical max)
            cu_norm = min(1.0, max(0.0, cu / 5.0))
            if value_net_score is not None:
                vs = float(value_net_score) if in_best else 0.0
                return 0.5 * cs + 0.3 * vs + 0.2 * cu_norm
            return 0.8 * cs + 0.2 * cu_norm

        scored = sorted(self.transforms, key=_transform_score, reverse=True)
        # Split into those that were in best_transforms (promoted) vs the rest
        priority = [t for t in scored if t.name() in best_set]
        rest = [t for t in scored if t.name() not in best_set]
        reordered = priority + rest

        # Blend world model prediction: boost predicted-best transform to front
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            if hasattr(wm, "predict_transform"):
                pred = wm.predict_transform(graph, [t.name() for t in reordered])
                pname = getattr(pred, "transform_name", None) or ""
                if pname:
                    front = [t for t in reordered if t.name() == pname]
                    tail  = [t for t in reordered if t.name() != pname]
                    reordered = front + tail
                    # Stash prediction so _run_single can call record_outcome()
                    self._last_wm_prediction = pred
                    log.debug("WM prediction: boost '%s' (expected_delta=%.3f)",
                              pname, getattr(pred, "expected_delta", 0.0))
        except Exception:
            pass

        log.debug(
            "Heuristic reorder: sim=%.3f, vnet=%s, promoting %d/%d transforms",
            best_sim,
            f"{value_net_score:.3f}" if value_net_score is not None else "n/a",
            len(priority), len(self.transforms),
        )
        return reordered

    def _record_successful_embedding(self, graph, transform_sequence: List[str]):
        """Store graph embedding + transform sequence for future heuristic reordering."""
        if not transform_sequence:
            return
        emb = self._get_graph_embedding(graph)
        if emb is None:
            return
        import math
        norm = math.sqrt(sum(v * v for v in emb)) or 1.0
        self._successful_embeddings.append({
            "embedding": emb,
            "norm": norm,
            "transforms": transform_sequence,
        })
        # Keep memory bounded
        if len(self._successful_embeddings) > 200:
            self._successful_embeddings = self._successful_embeddings[-200:]

    def _get_emotional_bias(self) -> dict:
        """Return beam/budget modulation from homeostasis drives."""
        try:
            from sare.meta.homeostasis import get_homeostatic_system
            return get_homeostatic_system().get_search_modulation()
        except Exception:
            return {"beam_delta": 0, "budget_delta": 0.0, "domain_switch": False, "mode": "normal"}

    def _run_single(self, problem, _skip_llm_fallback: bool = False) -> ExperimentResult:
        graph = getattr(problem, "graph", problem)
        problem_id = str(getattr(problem, "id", getattr(problem, "problem_id", "unknown")))
        domain = str(getattr(problem, "domain", "general"))
        expr = str(getattr(problem, "expression", getattr(problem, "name", problem_id)))

        # Inner monologue: announce intent
        try:
            from sare.meta.inner_monologue import get_inner_monologue
            _im = get_inner_monologue()
            _im.think(f"Attempting '{expr[:60]}' in domain '{domain}'",
                      context="search", emotion="curious")
        except Exception:
            _im = None

        # Associative retrieval: hint from similar past episodes
        try:
            from sare.memory.autobiographical import get_autobiographical_memory
            _auto = get_autobiographical_memory()
            _emb_hint = self._get_graph_embedding(graph)
            _similar = _auto.retrieve_similar(_emb_hint, top_k=3)
            if _similar and _im is not None:
                ep_desc = _similar[0].description[:60]
                _im.think(f"Recall: similar to '{ep_desc}'", context="memory", emotion="neutral")
        except Exception:
            pass

        # Emotional modulation of search parameters
        emotional_bias = self._get_emotional_bias()
        beam_delta = emotional_bias.get("beam_delta", 0)
        budget_delta = emotional_bias.get("budget_delta", 0.0)
        _saved_beam = self.beam_width
        _saved_budget = self.budget_seconds
        _search_strategy_used = "beam_search"
        self.beam_width = max(2, min(32, self.beam_width + beam_delta))
        # Adaptive budget: adjust by emotional state, clamp [0.3s, 10.0s]
        self.budget_seconds = max(0.3, min(10.0, self.budget_seconds + budget_delta))

        if beam_delta != 0 or budget_delta != 0.0:
            if _im is not None:
                _im.think(
                    f"Emotional bias: beam={self.beam_width}, budget={self.budget_seconds:.1f}s "
                    f"(mode={emotional_bias.get('mode','normal')})",
                    context="homeostasis", emotion="neutral",
                )

        # Reset world model prediction slot for this problem
        self._last_wm_prediction = None

        # Phase A: Apply stage-gated beam width (caps beam to stage max)
        _stage_level = 0
        try:
            brain_ref = getattr(self, "_brain_ref", None)
            if brain_ref is not None:
                _stage_level = getattr(brain_ref.stage, "level", 0)
                caps = getattr(brain_ref, "get_stage_capabilities", lambda: {})()
                max_bw = caps.get("max_beam_width")
                if max_bw is not None:
                    self.beam_width = min(self.beam_width, max_bw)
        except Exception:
            pass

        # Phase F: Apply dopamine search temperature to effective beam width
        try:
            from sare.neuro.dopamine import get_dopamine_system
            _dopa = get_dopamine_system()
            _temp = _dopa.search_temperature   # [0.1, 1.0]
            _encoding_strength = _dopa.encoding_strength
            # Scale beam width by temperature (high dopamine = wider beam)
            _temp_beam = max(2, int(self.beam_width * _temp))
            self.beam_width = _temp_beam
        except Exception:
            _encoding_strength = 1.0

        # Phase B: Predictive engine — rank transforms by EFE before search
        _pe_prediction = None
        _pe_predicted_transform = ""
        _pe_predicted_delta = 0.0
        try:
            from sare.cognition.predictive_engine import get_predictive_engine
            _pe = get_predictive_engine()
            from sare.memory.world_model import get_world_model
            _wm_pe = get_world_model()
            ranked = _pe.select_action(
                graph=graph,
                transforms=self.transforms,
                world_model=_wm_pe,
                stage_level=_stage_level,
                domain=domain,
            )
            if ranked:
                # Reorder transforms by EFE score
                self.transforms = [t for t, _ in ranked]
                _pe_predicted_transform = ""
                best_t = ranked[0][0]
                try:
                    _pe_predicted_transform = best_t.name() if callable(getattr(best_t, "name", None)) else str(best_t)
                except Exception:
                    pass
                _pe_predicted_delta = ranked[0][1] if ranked else 0.0
        except Exception:
            pass

        # Phase D: Check for pending teacher responses and inject suggested rules
        try:
            from sare.learning.teacher_protocol import get_confusion_detector, get_teacher_registry
            _cd = get_confusion_detector()
            _tr = get_teacher_registry()
            for q in _cd.get_pending_questions():
                if q.domain == domain and q.urgency > 0.5:
                    resp = _tr.ask_best_teacher(q)
                    if resp and resp.suggested_rules:
                        _cd.answer_question(q.question_id, resp.answer_text, resp.teacher_id)
                        log.debug("[ExperimentRunner] Teacher '%s' answered for domain '%s'", resp.teacher_id, domain)
        except Exception:
            pass

        # Heuristic pre-scoring: reorder transforms based on cosine similarity
        # to past successful graph embeddings (MPS-accelerated on Apple M1).
        _saved_transforms = self.transforms
        try:
            self.transforms = self._heuristic_reorder_transforms(graph)
        except Exception:
            pass  # safe fallback: keep original order

        # HTM reranking: reorder by n-gram sequence predictions
        _htm_recent: list = []
        try:
            from sare.neuro.htm_predictor import get_htm_predictor
            _htm = get_htm_predictor()
            self.transforms = _htm.rerank_transforms(self.transforms, _htm_recent, domain)
        except Exception:
            pass

        # WorldSimulator: reorder transforms using accumulated causal knowledge
        try:
            from sare.memory.world_simulator import get_world_simulator
            self.transforms = get_world_simulator().predict_best_transforms(
                self.transforms, domain
            )
        except Exception:
            pass

        start = time.time()
        energy_before = self._evaluate_energy(graph)

        # Reset attention-beam scorer's seen-state cache for each new independent problem
        try:
            from sare.search.attention_beam import get_default_scorer
            get_default_scorer().begin_problem(graph)
        except Exception:
            pass

        # Schema matcher: check if this graph structure was solved before
        _schema_hit = False
        try:
            from sare.cognition.schema_matcher import get_schema_matcher
            _sm = get_schema_matcher()
            _cached_steps = _sm.match(graph)
            if _cached_steps:
                solved, final_graph, proof_steps, proof_nl = True, graph, _cached_steps, ""
                self.transforms = _saved_transforms
                self.beam_width = _saved_beam
                self.budget_seconds = _saved_budget
                _schema_hit = True
                log.debug("SchemaMatcher HIT for problem %s (%d cached steps)", problem_id, len(_cached_steps))
        except Exception:
            _schema_hit = False

        # KB Fast-Path: for knowledge domains, query world model BEFORE expensive BeamSearch.
        # If we have a stored Wikipedia/ingested fact that answers the question, return it
        # immediately — no graph search needed. This is how ingested internet data actually
        # gets used at solve time.
        _KB_SKIP_DOMAINS = frozenset({
            "algebra", "arithmetic", "calculus", "trigonometry", "fraction",
            "geometry", "symbolic_math", "logic", "math",
        })
        if not _schema_hit and domain not in _KB_SKIP_DOMAINS and expr and len(expr.split()) >= 3:
            try:
                from sare.memory.knowledge_lookup import KnowledgeLookup, DIRECT_THRESHOLD
                _kl_hit = KnowledgeLookup().lookup(expr, domain)
                if _kl_hit and _kl_hit.confidence >= DIRECT_THRESHOLD:
                    solved = True
                    final_graph = graph
                    proof_steps = [f"kb:{_kl_hit.source}"]
                    proof_nl = _kl_hit.answer
                    _schema_hit = True   # bypass BeamSearch
                    self.transforms = _saved_transforms
                    self.beam_width = _saved_beam
                    self.budget_seconds = _saved_budget
                    log.info("[ExperimentRunner] KB hit: domain=%s conf=%.2f source=%s — %s",
                             domain, _kl_hit.confidence, _kl_hit.source, expr[:60])
            except Exception as _kl_exc:
                log.debug("[ExperimentRunner] KB lookup error: %s", _kl_exc)

        if not _schema_hit:
            try:
                # P2-E: Transform family routing — filter to domain-relevant transforms
                _search_transforms = self.transforms
                try:
                    from sare.transforms.transform_families import get_family_transforms
                    _search_transforms = get_family_transforms(domain, self.transforms)
                except Exception:
                    pass

                # AGI wiring: activate concepts for this problem so knowledge gets USED,
                # not just stored. Domain-general — works for any domain that has concepts
                # in the concept graph (math, logic, physics, chemistry, code, commonsense).
                _concept_hints: set = set()
                try:
                    from sare.concept.concept_graph import get_concept_graph
                    _cg = get_concept_graph()
                    # Extract symbols from the problem graph
                    _problem_symbols = []
                    try:
                        if hasattr(graph, 'get_node_ids'):
                            for _nid in graph.get_node_ids():
                                _n = graph.get_node(_nid)
                                if _n:
                                    _lbl = str(getattr(_n, 'label', '') or '')
                                    if _lbl:
                                        _problem_symbols.append(_lbl)
                        elif hasattr(graph, 'nodes'):
                            for _n in graph.nodes:
                                _lbl = str(getattr(_n, 'label', '') or '')
                                if _lbl:
                                    _problem_symbols.append(_lbl)
                    except Exception:
                        pass
                    if _problem_symbols:
                        _activated = _cg.activate_for_problem(domain or "general", _problem_symbols)
                        if _activated:
                            _concept_hints = set(_cg.get_transform_hints(_activated))
                            log.debug("[ConceptActivation] %d concepts activated, %d hints",
                                      len(_activated), len(_concept_hints))
                except Exception as _ce:
                    log.debug("[ConceptActivation] error: %s", _ce)

                # Reorder transforms: boost those whose name matches concept hints
                # (symbol-level or keyword-level). Keeps all transforms; only changes order.
                if _concept_hints:
                    def _concept_match_score(t):
                        try:
                            tname = t.name().lower()
                        except Exception:
                            return 0
                        return sum(1 for h in _concept_hints if h and h.lower() in tname)
                    _scored = [(t, _concept_match_score(t)) for t in _search_transforms]
                    _scored.sort(key=lambda x: -x[1])  # higher score first
                    _search_transforms = [t for t, _s in _scored]

                # P3-I: CompositeRuleLearner — suggest transform order
                try:
                    from sare.learning.composite_rule_learner import get_composite_learner
                    _search_transforms = get_composite_learner().suggest_transform_order(_search_transforms, domain)
                except Exception:
                    pass

                # Option B: Hypothesis engine — generate forward predictions
                _hypotheses = []
                try:
                    from sare.search.hypothesis_engine import get_hypothesis_engine
                    _hyp_engine = get_hypothesis_engine()
                    _hypotheses = _hyp_engine.generate(graph, domain or "general")
                except Exception:
                    pass

                # TransformPolicy: neural attention-based pre-sort of transforms
                if self._transform_policy is not None:
                    try:
                        _graph_emb = self._get_graph_embedding(graph)
                        if _graph_emb:
                            _t_names = [t.name() for t in _search_transforms]
                            _sorted_names = self._transform_policy.sorted_transforms(_t_names, _graph_emb)
                            _name_to_t = {t.name(): t for t in _search_transforms}
                            _search_transforms = [_name_to_t[n] for n in _sorted_names if n in _name_to_t]
                    except Exception:
                        pass

                outcome = self._search(graph, transforms=_search_transforms)
                solved, final_graph, proof_steps, proof_nl = self._parse_search_outcome(outcome, graph, energy_before)

                # P3-H: A* fallback for hard problems that BeamSearch failed on
                if self._use_astar_fallback and not solved:
                    _beam_steps = getattr(outcome, 'steps_taken', getattr(outcome, 'expansions', 0))
                    _beam_delta = energy_before - self._evaluate_energy(
                        getattr(outcome, 'graph', getattr(outcome, 'result_graph', graph))
                    ) if outcome else 0.0
                    if _beam_delta < 0.01 and _beam_steps > 50:
                        try:
                            from sare.search.astar_search import AStarSearch
                            _astar = AStarSearch()
                            _astar_result = _astar.search(
                                graph, self.energy, _search_transforms,
                                beam_width=self.beam_width,
                                max_depth=30,
                                budget_seconds=min(self.budget_seconds, 10.0),
                            )
                            if _astar_result.get("success") and _astar_result.get("delta", 0) > 0.01:
                                # Rebuild a SearchResult-like object for _parse_search_outcome
                                class _AStarOutcome:
                                    pass
                                _ao = _AStarOutcome()
                                _ao.graph = _astar_result["result_graph"]
                                _ao.result_graph = _ao.graph
                                _ao.success = _astar_result["success"]
                                _ao.solved = _astar_result["success"]
                                _ao.transforms_applied = _astar_result.get("transforms_used", [])
                                _ao.steps_taken = len(_ao.transforms_applied)
                                outcome = _ao
                                solved, final_graph, proof_steps, proof_nl = self._parse_search_outcome(outcome, graph, energy_before)
                                _search_strategy_used = "astar_fallback"
                                log.debug("[ExperimentRunner] A* fallback succeeded for %s (delta=%.3f)", problem_id, _astar_result["delta"])
                        except Exception as _astar_exc:
                            log.debug("[ExperimentRunner] A* fallback error: %s", _astar_exc)

            except Exception as exc:
                log.exception("Experiment failed for problem %s: %s", problem_id, exc)
                solved, final_graph, proof_steps, proof_nl = False, graph, [], ""
            finally:
                self.transforms = _saved_transforms  # always restore original list
                self.beam_width = _saved_beam
                self.budget_seconds = _saved_budget

        # ── Semantic path: fires when symbolic BeamSearch fails on non-math ──
        # For commonsense, factual, science, reasoning, word-problem domains the
        # symbolic engine has no transforms. Instead of silently discarding the
        # problem, use the LLM to answer it and feed the knowledge to WorldModel.
        # This makes SARE learn from EVERYTHING, not just symbolic math.
        # NOTE: _skip_llm_fallback=True when called from run_batch — the
        # GeneralSolver batch handles LLM-backed general knowledge separately.
        # Allowing LLM calls here blocks the entire batch for 5-8 minutes.
        _sem_solution = None
        if not solved and expr and not _skip_llm_fallback:
            try:
                from sare.learning.semantic_solver import get_semantic_solver, _SYMBOLIC_DOMAINS
                _sem_solver = get_semantic_solver()
                _sem_domain = str(domain or "general")
                if _sem_solver.should_attempt(_sem_domain, expr):
                    _sem_solution = _sem_solver.solve(question=expr, domain=_sem_domain)
                    if _sem_solution.solved:
                        # Mark as solved via semantic path — no graph change, but knowledge captured
                        solved = True
                        proof_steps = [f"semantic_{_sem_solution.reasoning_type}"]
                        proof_nl = _sem_solution.answer
                        log.info(
                            "[ExperimentRunner] Semantic solve: domain=%s type=%s conf=%.2f — %s",
                            _sem_domain, _sem_solution.reasoning_type,
                            _sem_solution.confidence, expr[:60],
                        )
            except Exception as _sem_exc:
                log.debug("[ExperimentRunner] Semantic path error: %s", _sem_exc)

        # ── Semantic reflection: learn from LLM-solved problems ──────────────
        if _sem_solution is not None and _sem_solution.solved:
            try:
                from sare.reflection.semantic_reflection import get_semantic_reflection
                get_semantic_reflection().reflect(_sem_solution)
            except Exception as _sr_exc:
                log.debug("[ExperimentRunner] Semantic reflection error: %s", _sr_exc)

        # Inner monologue: report outcome
        if _im is not None:
            try:
                if solved:
                    _im.think(
                        f"Solved '{expr[:50]}' in {len(proof_steps)} steps",
                        context="search", emotion="excited",
                    )
                else:
                    _im.think(
                        f"Failed on '{expr[:50]}' — no transform path found",
                        context="search", emotion="frustrated",
                    )
            except Exception:
                pass

        # Record successful embedding for future heuristic reordering
        if solved and proof_steps:
            try:
                self._record_successful_embedding(graph, proof_steps)
            except Exception:
                pass

        # Schema matcher: record this solved structure for future cache hits
        if solved and proof_steps and not _schema_hit:
            try:
                from sare.cognition.schema_matcher import get_schema_matcher
                get_schema_matcher().record(graph, proof_steps)
            except Exception:
                pass

        # Causal hierarchy: observe feature→concept→law for each solve step
        if solved and proof_steps:
            try:
                from sare.memory.causal_hierarchy import get_causal_hierarchy
                _ch = get_causal_hierarchy()
                for _step in proof_steps:
                    _ch.observe_feature(getattr(problem, 'expression', ''), _step, domain)
                if len(proof_steps) % 20 == 0:
                    _ch.save()
            except Exception:
                pass

        energy_after = self._evaluate_energy(final_graph)

        # Intrinsic motivation: novelty bonus reduces effective energy cost
        try:
            from sare.cognition.novelty_detector import get_novelty_detector
            _nov = get_novelty_detector().score(
                getattr(problem, 'expression', ''),
                getattr(problem, 'domain', 'general')
            )
            # High novelty (>0.7) gives up to 10% energy bonus
            if _nov > 0.7 and solved:
                _bonus = (_nov - 0.7) / 0.3 * 0.1
                energy_after = max(0.0, energy_after - _bonus)
        except Exception:
            pass

        # Feed outcome into MLX value network for online learning (M1 GPU)
        try:
            if self._value_net is None:
                from sare.heuristics.mlx_value_net import get_value_net
                self._value_net = get_value_net()
            emb = self._get_graph_embedding(graph)
            if emb is not None:
                delta = energy_before - energy_after
                self._value_net.record_outcome(emb, delta, solved=solved)
        except Exception:
            pass

        # TransformPolicy: update neural ranker from energy deltas
        if self._transform_policy is not None and proof_steps:
            try:
                _deltas = []
                _step_energy = energy_before
                for step_name in proof_steps:
                    # Approximate per-step delta as uniform share (exact deltas not tracked)
                    _total_delta = energy_before - energy_after
                    _deltas.append(_total_delta / max(len(proof_steps), 1))
                self._transform_policy.update(proof_steps, _deltas)
                # Register any new transforms from this solve
                for name in proof_steps:
                    self._transform_policy.add_transform(name)
            except Exception:
                pass

        # HTMPredictor: observe transform sequence so future predictions improve
        if self._htm_predictor is not None and proof_steps:
            try:
                self._htm_predictor.observe_sequence(proof_steps, domain or "general", solved)
                # Also ensure value_net_callable is initialized for next solve
                if self._value_net_callable is None and self._value_net is not None:
                    _vn = self._value_net
                    _emb_fn = self._get_graph_embedding
                    def _vnet_wrapper(g):
                        try:
                            emb = _emb_fn(g)
                            if emb:
                                return _vn.score(emb)
                        except Exception:
                            pass
                        return 0.5
                    self._value_net_callable = _vnet_wrapper
            except Exception:
                pass

        # Feed solve outcome to AlgorithmSelector so strategy win rates reflect the
        # strategy that actually ran, not a post-hoc synthetic choice.
        try:
            from sare.meta.algorithm_selector import get_algorithm_selector as _get_as
            _as = _get_as()
            _strategy_used = str(_search_strategy_used or "beam_search")
            _as.record_selection(domain or "algebra", _strategy_used)
            _as.record_outcome(domain or "algebra", _strategy_used, solved)
        except Exception:
            pass

        # Record hypothesis engine outcome
        try:
            if _hypotheses:
                from sare.search.hypothesis_engine import get_hypothesis_engine
                get_hypothesis_engine().record(solved)
        except Exception:
            pass

        # Record world model prediction outcome so predictions improve over time
        if self._last_wm_prediction is not None:
            try:
                from sare.memory.world_model import get_world_model
                wm = get_world_model()
                wm.record_outcome(
                    self._last_wm_prediction,
                    actual_transforms=proof_steps,
                    actual_delta=energy_before - energy_after,
                    domain=domain,
                )
                wm.observe_solve(
                    expression=expr,
                    transforms_used=proof_steps,
                    energy_delta=energy_before - energy_after,
                    domain=domain,
                    solved=solved,
                )
            except Exception:
                pass
            finally:
                self._last_wm_prediction = None

        # Phase B: Record predictive engine outcome
        _actual_delta = energy_before - energy_after
        _actual_transform = proof_steps[0] if proof_steps else ""
        try:
            from sare.cognition.predictive_engine import get_predictive_engine
            _pe = get_predictive_engine()
            from sare.memory.world_model import get_world_model
            _wm2 = get_world_model()
            _pe.observe_outcome(
                predicted_transform=_pe_predicted_transform,
                actual_transform=_actual_transform,
                actual_delta=_actual_delta,
                predicted_delta=_pe_predicted_delta,
                domain=domain,
                world_model=_wm2,
            )
        except Exception:
            pass

        # Phase C: Encode solve step in internal grammar
        if solved and proof_steps:
            try:
                from sare.language.internal_grammar import get_internal_grammar
                _grammar = get_internal_grammar()
                for step in proof_steps[:3]:  # encode top 3 steps
                    _grammar.encode_solve_step(step, graph, final_graph)
            except Exception:
                pass

        # Fix 4: Cache stuck graphs for synthesis validation
        if not solved:
            cache = self._stuck_graph_cache[domain]
            if len(cache) < 20:
                cache.append(graph)

        # Phase D: Notify confusion detector of failure
        if not solved:
            try:
                from sare.learning.teacher_protocol import get_confusion_detector
                get_confusion_detector().observe_failure(
                    domain=domain,
                    expression=expr,
                    transforms_tried=list(set(
                        t.name() if callable(getattr(t, "name", None)) else str(t)
                        for t in self.transforms[:8]
                    )),
                )
            except Exception:
                pass

        elapsed_ms = (time.time() - start) * 1000.0

        # HTM: record transform sequence for n-gram learning
        if proof_steps:
            try:
                from sare.neuro.htm_predictor import get_htm_predictor
                get_htm_predictor().observe_sequence(proof_steps, domain, success=solved)
            except Exception:
                pass

        # Counterfactual reasoning: analyze critical vs redundant steps
        if solved and proof_steps:
            try:
                from sare.reasoning.counterfactual import get_counterfactual_reasoner
                cf = get_counterfactual_reasoner()
                cf.analyze(proof_steps, problem, self.energy)
            except Exception:
                pass
        elif not solved:
            try:
                from sare.reasoning.counterfactual import get_counterfactual_reasoner
                cf = get_counterfactual_reasoner()
                cf.hypothesize(problem, self.transforms)
                # Fix 4: Feed failed problem back to curriculum for retry
                try:
                    self.curriculum_gen.add_failure_for_retry(problem)
                except Exception as _cf_exc:
                    log.debug("Counterfactual → curriculum feedback failed: %s", _cf_exc)
            except Exception:
                pass

        # Feed failures into LLM synthesizer: every 10 unique stuck expressions
        # per domain, ask LLM to invent a new transform (no hardcoding)
        if not solved and expr:
            try:
                if not hasattr(self, '_synth_fail_buf'):
                    self._synth_fail_buf = {}
                if not hasattr(self, '_synth_last_attempt'):
                    self._synth_last_attempt = {}
                _buf = self._synth_fail_buf.setdefault(domain or "general", [])
                if expr not in _buf:
                    _buf.append(expr)
                _SYNTH_COOLDOWN_S = 300  # 5 minutes between synthesis attempts per domain
                _now = time.time()
                _last = self._synth_last_attempt.get(domain or "general", 0.0)
                if len(_buf) >= 10 and len(_buf) % 10 == 0 and (_now - _last) >= _SYNTH_COOLDOWN_S:
                    self._synth_last_attempt[domain or "general"] = _now
                    # Use brain's synthesizer if available
                    _brain_synth = None
                    try:
                        from sare.brain import get_brain as _gb
                        _brain_synth = _gb().transform_synthesizer
                    except Exception:
                        pass
                    if _brain_synth:
                        _t_names = [t.name() for t in self.transforms if hasattr(t, "name")][:30]
                        _val_graphs = []
                        for _ex in _buf[-6:]:
                            try:
                                from sare.engine import load_problem as _lp
                                _, _g = _lp(_ex)
                                _val_graphs.append(_g)
                            except Exception:
                                pass
                        if _val_graphs:
                            import threading as _thr
                            _domain_snap = domain or "general"
                            _exprs_snap = list(_buf[-6:])
                            def _synth_bg(_bs=_brain_synth, _d=_domain_snap,
                                          _e=_exprs_snap, _tn=_t_names, _vg=_val_graphs):
                                try:
                                    res = _bs.synthesize(
                                        domain=_d, stuck_exprs=_e,
                                        validation_graphs=_vg,
                                        existing_transform_names=_tn,
                                    )
                                    if res.get("promoted"):
                                        log.info("[SynthFromFailure] domain=%s new transform: %s",
                                                 _d, res.get("class_name", "?"))
                                        try:
                                            from sare.brain import get_brain as _gb2
                                            _gb2()._refresh_transforms()
                                        except Exception:
                                            pass
                                    else:
                                        log.debug("[SynthFromFailure] domain=%s score=%.2f %s",
                                                  _d, res.get("score", 0), res.get("message", ""))
                                except Exception as _se:
                                    log.debug("[SynthFromFailure] error: %s", _se)
                            _thr.Thread(target=_synth_bg, daemon=True).start()
            except Exception as _sfb_exc:
                log.debug("[SynthFromFailure] buffer error: %s", _sfb_exc)

        result = ExperimentResult(
            problem_id=problem_id,
            solved=solved,
            energy_before=energy_before,
            energy_after=energy_after,
            elapsed_ms=elapsed_ms,
            proof_steps=proof_steps,
            proof_nl=proof_nl,
            final_graph=final_graph,
            domain=domain,
            search_strategy=_search_strategy_used,
        )

        return self._reflect_and_learn(problem, result)

    def _current_system_load(self) -> float:
        load = 0.0
        try:
            if hasattr(os, "getloadavg"):
                one_min_load = os.getloadavg()[0]
                cpu_count = os.cpu_count() or 1
                load = max(0.0, min(1.0, one_min_load / max(1, cpu_count)))
        except Exception:
            load = 0.0
        return load

    def _record_batch_stats(self, requested_n: int, results: List[ExperimentResult], elapsed_s: float):
        solved_count = sum(1 for r in results if r.solved)
        attempted = len(results)
        solve_rate = (solved_count / attempted) if attempted else 0.0
        avg_elapsed_ms = (sum(r.elapsed_ms for r in results) / attempted) if attempted else 0.0
        load = self._current_system_load()

        # Track schema cache hit rate for genuine learning measurement
        _schema_hit_rate = 0.0
        try:
            from sare.cognition.schema_matcher import get_schema_matcher
            _sm = get_schema_matcher()
            _total = _sm._hits + _sm._misses
            _schema_hit_rate = round(_sm._hits / max(1, _total), 3)
        except Exception:
            pass

        self._recent_batch_stats.append(
            {
                "requested_n": requested_n,
                "attempted": attempted,
                "solved": solved_count,
                "solve_rate": solve_rate,
                "avg_elapsed_ms": avg_elapsed_ms,
                "elapsed_s": elapsed_s,
                "schema_hit_rate": _schema_hit_rate,
                "load": load,
                "timestamp": time.time(),
            }
        )
        if len(self._recent_batch_stats) > 8:
            self._recent_batch_stats = self._recent_batch_stats[-8:]

    def _adaptive_target_batch_size(self, requested_n: int) -> int:
        if requested_n <= 1:
            return requested_n
        # In turbo mode, skip adaptive throttling — always use full batch
        if getattr(self, '_fast_path_ratio', 50) >= 200:
            return requested_n

        if self._adaptive_batch_size is None:
            self._adaptive_batch_size = requested_n

        history = self._recent_batch_stats[-4:]
        if not history:
            return max(1, self._adaptive_batch_size)

        avg_solve_rate = sum(h["solve_rate"] for h in history) / len(history)
        avg_load = sum(h["load"] for h in history) / len(history)
        avg_elapsed_ms = sum(h["avg_elapsed_ms"] for h in history) / len(history)

        target = self._adaptive_batch_size

        # Only reduce on high CPU load — don't reduce on low solve rate alone
        # (hard problems naturally have low solve rate; that's expected)
        if avg_load >= 0.9:
            target = max(3, target - 2)
        elif avg_load >= 0.75:
            target = max(3, target - 1)
        elif avg_load <= 0.35 and avg_solve_rate >= 0.65:
            target = min(requested_n, target + 2)
        elif avg_load <= 0.5 and avg_solve_rate >= 0.45:
            target = min(requested_n, target + 1)

        if avg_elapsed_ms > (self.budget_seconds * 1000.0 * 0.9):
            target = max(3, target - 1)

        # Never drop below 3 — single-problem batches prevent meaningful learning
        self._adaptive_batch_size = max(3, min(requested_n, target))
        return self._adaptive_batch_size

    def _run_single_fast(self, problem) -> ExperimentResult:
        """
        Lightweight solve path — skips cognitive modules (inner monologue, autobio,
        predictive engine, dopamine, counterfactual, etc.).  Used for bulk learning
        cycles where throughput matters more than rich metadata.
        Records only the bare minimum: solve outcome + value net + world model.
        """
        graph = getattr(problem, "graph", problem)
        problem_id = str(getattr(problem, "id", "fast"))
        domain = str(getattr(problem, "domain", "general"))
        expr = str(getattr(problem, "expression", problem_id))

        start = time.time()
        energy_before = self._evaluate_energy(graph)
        transforms = self.transforms or []

        # ── Text-based curriculum problems (commonsense, factual QA) ──────────
        # Problems with _question/_answer on the graph skip symbolic BeamSearch
        # and route to GeneralSolver.attempt_learning_problem() instead.
        _q = getattr(graph, "_question", None) or (expr if domain in ("commonsense", "factual") else None)
        _a = getattr(graph, "_answer", None)
        if _q and _a:
            try:
                from sare.cognition.general_solver import GeneralSolver
                _gs = GeneralSolver()
                _lr = _gs.attempt_learning_problem(
                    problem_text=_q,
                    expected_answer=str(_a),
                    problem_type=domain,
                    context="",
                )
                solved = _lr.solved and bool(_lr.answer)
                # Check answer correctness
                if solved and _a:
                    _expected = str(_a).lower().strip()
                    _got = str(_lr.answer).lower().strip()
                    solved = (_expected in _got) or (_got in _expected) or _expected == _got
                proof_steps = [f"general_solver:{_lr.solver_used}"] if solved else []
                final_graph = graph
                energy_after = energy_before
                elapsed_ms = (time.time() - start) * 1000.0
                return ExperimentResult(
                    problem_id=problem_id, solved=solved,
                    energy_before=energy_before, energy_after=energy_after,
                    elapsed_ms=elapsed_ms, proof_steps=proof_steps, domain=domain,
                )
            except Exception as _qe:
                log.debug("[ExperimentRunner] Text QA fast path error: %s", _qe)

        try:
            outcome = self._search(graph)
            solved, final_graph, proof_steps, _ = self._parse_search_outcome(
                outcome, graph, energy_before)
        except Exception:
            solved, final_graph, proof_steps = False, graph, []

        energy_after = self._evaluate_energy(final_graph)
        delta = energy_before - energy_after

        _turbo = getattr(self, '_fast_path_ratio', 50) >= 200

        # Skip per-solve side effects in turbo mode for max throughput
        if not _turbo:
            # Feed value net (M1 GPU online learning)
            try:
                if self._value_net is None:
                    from sare.heuristics.mlx_value_net import get_value_net
                    self._value_net = get_value_net()
                emb = self._get_graph_embedding(graph)
                if emb:
                    self._value_net.record_outcome(emb, delta, solved=solved)
            except Exception:
                pass

            # World model observation (lightweight)
            try:
                from sare.memory.world_model import get_world_model
                get_world_model().observe_solve(
                    expression=expr, transforms_used=proof_steps,
                    energy_delta=delta, domain=domain, solved=solved,
                )
            except Exception:
                pass

        elapsed_ms = (time.time() - start) * 1000.0
        result = ExperimentResult(
            problem_id=problem_id, solved=solved,
            energy_before=energy_before, energy_after=energy_after,
            elapsed_ms=elapsed_ms, proof_steps=proof_steps, domain=domain,
            search_strategy="beam_search",
            final_graph=final_graph if solved else None,
        )
        # Run reflect-and-learn on a fraction of solves for rule promotion
        if solved:
            import random as _rnd_reflect
            _reflect_prob = 1.0 if not _turbo else 0.03  # 3% sampling in turbo mode
            if _rnd_reflect.random() < _reflect_prob:
                if not _turbo:
                    result = self._reflect_and_learn(problem, result)
                else:
                    # Turbo mode: non-blocking reflection via background thread (semaphore-gated)
                    # Prevents thread exhaustion while still discovering new rules
                    if not hasattr(self, '_reflect_semaphore'):
                        import threading as _th_reflect
                        self._reflect_semaphore = _th_reflect.Semaphore(1)
                    def _bg_reflect(p, r):
                        with self._reflect_semaphore:
                            try:
                                self._reflect_and_learn(p, r)
                            except Exception:
                                pass
                    import threading as _th_reflect
                    _th_reflect.Thread(target=_bg_reflect, args=(problem, result),
                                     daemon=True).start()
        return result

    def run_batch(self, n: int = 10) -> List[ExperimentResult]:
        import concurrent.futures as _cf
        batch_start = time.time()
        # Hot-reload any transforms written by Oracle pipeline since last batch
        self._maybe_reload_synthesized()
        target_n = self._adaptive_target_batch_size(n)

        # Collect problems first (curriculum gen is not thread-safe)
        problems = []
        for _ in range(target_n):
            try:
                problem = self._pick_next_problem()
                if problem is None:
                    break
                problems.append(problem)
            except Exception as exc:
                log.debug("No problem available for experiment batch: %s", exc)
                break

        # Fill remaining slots with ProblemFactory when curriculum runs dry
        _remaining = target_n - len(problems)
        if _remaining > 0:
            try:
                from sare.knowledge.problem_factory import get_problem_factory
                from sare.engine import load_problem as _lp_fb
                _pf = get_problem_factory()
                _pf_batch = _pf.generate_batch(total=_remaining, seed=int(time.time()) % 10**7)
                _pf_added = 0
                for _pf_item in _pf_batch:
                    try:
                        _res = _lp_fb(_pf_item["expression"])
                        if _res is not None:
                            _g = _res[1] if isinstance(_res, tuple) else _res
                            if _g is not None:
                                problems.append(_g)
                                _pf_added += 1
                    except Exception:
                        pass
                if _pf_added:
                    log.debug("[run_batch] ProblemFactory filled %d/%d remaining slots", _pf_added, _remaining)
            except Exception as _fb_exc:
                log.debug("[run_batch] ProblemFactory fallback error: %s", _fb_exc)

        # Use fast path for bulk problems; keep full path for 1-in-N (cognitive updates)
        _fast_ratio = getattr(self, '_fast_path_ratio', 50)
        _turbo_mode = _fast_ratio >= 200
        # In turbo mode, ALL problems use fast path (avoids LLM blocking).
        # Learning quality maintained via 5% reflect sampling in _run_single_fast.
        _n_probs = len(problems)
        _full_every = max(_n_probs, _fast_ratio) if _n_probs > 0 else 1
        def _solver(idx_problem):
            idx, p = idx_problem
            if _turbo_mode:
                return self._run_single_fast(p)
            if idx == 0:
                # Skip LLM fallback in batch context — GeneralSolver batch handles it separately.
                # Allowing synchronous LLM calls here blocks all 30-50 parallel futures.
                return self._run_single(p, _skip_llm_fallback=True)
            if _full_every > 0 and idx % _full_every == 0:
                return self._run_single(p, _skip_llm_fallback=True)
            return self._run_single_fast(p)

        # Solve in parallel threads
        results: List[ExperimentResult] = []
        # Cap to 8 workers to prevent OS thread exhaustion from timeout accumulation
        max_workers = min(len(problems), min(8, max(1, (os.cpu_count() or 4))))
        # Batch timeout: budget_seconds per problem + 30s overhead, capped at 60s.
        _batch_timeout = min(60.0, self.budget_seconds * 3 + 30.0)
        if max_workers > 1 and len(problems) > 1:
            with _cf.ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="sare-solve") as pool:
                futs = {pool.submit(_solver, (i, p)): p
                        for i, p in enumerate(problems)}
                try:
                    for fut in _cf.as_completed(futs, timeout=_batch_timeout):
                        try:
                            results.append(fut.result(timeout=30.0))
                        except Exception as exc:
                            log.debug("Parallel solve error: %s", exc)
                except _cf.TimeoutError:
                    log.warning("[run_batch] Batch timeout (%.0fs) — cancelling pending futures", _batch_timeout)
                    for _fut in futs:
                        _fut.cancel()
        else:
            for i, problem in enumerate(problems):
                results.append(_solver((i, problem)))

        for r in results:
            self._history.append(r)

        # Async LLM cache-fill: for unsolved non-symbolic problems, call LLM in background
        # and store the answer back in world model. Next time the same question arrives,
        # KB lookup will hit directly (no LLM needed). This is how the system learns from
        # internet-ingested knowledge + LLM responses over time.
        _SYMBOLIC_BATCH_SKIP = frozenset({
            "algebra", "arithmetic", "calculus", "trigonometry", "fraction",
            "geometry", "symbolic_math", "logic", "math", "physics", "chemistry",
        })
        _kb_misses = [
            r for r in results
            if not r.solved
            and str(getattr(r, 'domain', 'general') or 'general') not in _SYMBOLIC_BATCH_SKIP
            and len(str(getattr(r, 'expression', '') or getattr(r, 'problem_id', '') or '').split()) >= 3
        ]
        if _kb_misses:
            import threading as _llm_fill_thread
            def _fill_cache(misses):
                for _r in misses[:3]:   # max 3 LLM calls per batch
                    try:
                        _expr = str(getattr(_r, 'expression', '') or getattr(_r, 'problem_id', ''))
                        _dom = str(getattr(_r, 'domain', 'general') or 'general')
                        from sare.learning.semantic_solver import get_semantic_solver
                        _sem = get_semantic_solver()
                        if not _sem.should_attempt(_dom, _expr):
                            continue
                        _sol = _sem.solve(question=_expr, domain=_dom)
                        if _sol.solved and _sol.answer:
                            # Cache answer in world model for future KB lookups
                            from sare.memory.world_model import get_world_model
                            _wm = get_world_model()
                            _fact = f"{_expr}: {_sol.answer}"
                            if hasattr(_wm, 'add_fact'):
                                _wm.add_fact(_dom, _fact, confidence=0.85, source="llm_cache")
                            log.info("[LLMCache] Stored answer for domain=%s: %s → %s",
                                     _dom, _expr[:50], _sol.answer[:60])
                    except Exception as _lce:
                        log.debug("[LLMCache] Fill error: %s", _lce)
            _llm_fill_thread.Thread(target=_fill_cache, args=(_kb_misses,),
                                    daemon=True, name="llm-cache-fill").start()

        # Update per-domain solve tracker and check for mastery
        for r in results:
            domain = str(getattr(r, 'domain', '') or 'general')
            if not domain:
                domain = 'general'
            t = self._domain_solve_tracker.setdefault(domain, {"attempts": 0, "successes": 0})
            t["attempts"] += 1
            if r.solved:
                t["successes"] += 1
                self._domain_consec_fails[domain] = 0  # reset on success
                # If domain was on cooldown, lift it now that we can solve problems
                self._domain_cooldown.pop(domain, None)
                # Notify WeaknessDetector of success (for rate tracking)
                try:
                    from sare.meta.weakness_detector import get_weakness_detector
                    get_weakness_detector().record_success(domain, getattr(r, "graph", None))
                except Exception:
                    pass
            else:
                self._domain_consec_fails[domain] = self._domain_consec_fails.get(domain, 0) + 1
                # Record failure in WeaknessDetector for pattern analysis
                try:
                    from sare.meta.weakness_detector import get_weakness_detector
                    get_weakness_detector().record_failure(
                        domain=domain,
                        problem_id=getattr(r, "problem_id", ""),
                        proof_steps=getattr(r, "proof_steps", None),
                        graph=getattr(r, "graph", None),
                    )
                except Exception:
                    pass
                # Attempt cross-domain concept transfer on failure
                if self._domain_consec_fails[domain] % 5 == 0:
                    try:
                        from sare.memory.concept_transfer import get_concept_transfer
                        _ct = get_concept_transfer()
                        _ct_result = _ct.attempt_transfer(
                            problem=r,
                            failed_transforms=r.proof_steps or [],
                            available_transforms=self.transforms,
                            searcher=self.searcher,
                            energy=self.energy,
                        )
                        if _ct_result:
                            rule_name, delta = _ct_result
                            log.info("[ConceptTransfer] Successful transfer in domain=%s rule=%s delta=%.2f",
                                     domain, rule_name, delta)
                    except Exception as _ct_exc:
                        log.debug("[ConceptTransfer] attempt error: %s", _ct_exc)
                # Deprioritize domains with extreme consecutive failures (post-synthesis still failing)
                # Cooldown: 50 consecutive fails → pause for DOMAIN_COOLDOWN_BATCHES batches
                _thresh = self._SYNTH_FAIL_THRESHOLD * 3  # 60 fails = 3x synthesis threshold
                _cf = self._domain_consec_fails[domain]
                if _cf >= _thresh and domain not in self._domain_cooldown:
                    _resume = self._batch_count + self._DOMAIN_COOLDOWN_BATCHES
                    self._domain_cooldown[domain] = _resume
                    log.warning(
                        "[ExperimentRunner] Domain '%s' on cooldown for %d batches "
                        "(%d consecutive failures, synthesis not helping). "
                        "Will resume at batch %d.",
                        domain, self._DOMAIN_COOLDOWN_BATCHES, _cf, _resume,
                    )
        self._maybe_emit_domain_mastery()
        self._maybe_trigger_synthesis()

        # Wire homeostasis: solving problems satisfies drives (was never called from solve loop,
        # causing all 7 drives to saturate at 1.0 permanently).
        try:
            from sare.meta.homeostasis import get_homeostatic_system
            _hs = get_homeostatic_system()
            _total = len(results)
            _solved = sum(1 for _r in results if getattr(_r, "solved", False))
            if _total > 0:
                _hs.on_batch_completed(_solved, _total)
            # Novel domains satisfy exploration drive (domain-agnostic signal)
            _domains_this_batch = {str(getattr(_r, "domain", "general") or "general")
                                    for _r in results}
            if len(_domains_this_batch) >= 2:
                _hs.on_exploration()
            # Tick drives forward each batch (time-based growth)
            _hs.tick()
        except Exception as _hs_exc:
            log.debug("[ExperimentRunner] homeostasis update error: %s", _hs_exc)

        self._batch_count += 1
        self._record_batch_stats(requested_n=n, results=results, elapsed_s=time.time() - batch_start)
        return results

    def _maybe_emit_domain_mastery(self):
        """Emit DOMAIN_MASTERED events / persist mastery for domains crossing the threshold."""
        _MASTERY_THRESHOLD = 0.80
        _MIN_ATTEMPTS = 10
        for domain, t in self._domain_solve_tracker.items():
            if t["attempts"] < _MIN_ATTEMPTS:
                continue
            rate = t["successes"] / max(t["attempts"], 1)
            if rate >= _MASTERY_THRESHOLD:
                # Always persist to file so web server (separate process) picks it up
                self._persist_domain_mastery(domain, rate)
                # Also emit in-process event if brain is available
                try:
                    from sare.brain import get_brain, Event
                    brain = get_brain()
                    if brain is not None:
                        if domain not in (brain._stats.get("domains_mastered") or []):
                            brain.events.emit(Event.DOMAIN_MASTERED, {"domain": domain, "solve_rate": rate})
                except Exception as exc:
                    log.debug("Domain mastery emit failed: %s", exc)

    def _maybe_trigger_synthesis(self):
        """Trigger TransformSynthesizer when a domain has too many consecutive failures.

        If a domain hits _SYNTH_FAIL_THRESHOLD consecutive unsolved problems, it means
        the current transform set has a genuine gap for that domain. We ask the LLM
        to synthesize candidate transforms, validate them on known problems, and add
        the verified ones to the live transform set.
        """
        # Lift expired cooldowns so domains get another chance
        _expired = [d for d, resume in self._domain_cooldown.items()
                    if self._batch_count >= resume]
        for d in _expired:
            del self._domain_cooldown[d]
            self._domain_consec_fails[d] = 0
            log.info("[ExperimentRunner] Domain '%s' cooldown expired — re-enabling", d)

        if not hasattr(self, '_synth_last_attempt'):
            self._synth_last_attempt = {}
        _SYNTH_COOLDOWN_S = 300  # 5 minutes between synthesis attempts per domain
        _now = time.time()

        for domain, consec in list(self._domain_consec_fails.items()):
            if consec < self._SYNTH_FAIL_THRESHOLD:
                continue
            # Avoid re-triggering every batch once threshold is crossed
            trigger_key = f"{domain}:{consec // self._SYNTH_FAIL_THRESHOLD}"
            if trigger_key in self._synth_triggered_domains:
                continue
            # Time-based cooldown: don't re-synthesize within 5 minutes
            if (_now - self._synth_last_attempt.get(domain, 0.0)) < _SYNTH_COOLDOWN_S:
                continue
            self._synth_triggered_domains.add(trigger_key)
            self._synth_last_attempt[domain] = _now
            log.warning(
                "[ExperimentRunner] Domain '%s' has %d consecutive failures — triggering TransformSynthesizer",
                domain, consec,
            )
            try:
                from sare.meta.transform_synthesizer import TransformSynthesizer
                synth = TransformSynthesizer()

                # ── Pass real context: recent failed expressions for this domain ──
                # Extract stuck expressions from recent history (last 200 results)
                _stuck_exprs = []
                for _r in list(self._history)[-200:]:
                    if (not getattr(_r, "solved", True)
                            and str(getattr(_r, "domain", "")) == domain
                            and getattr(_r, "problem_id", "")):
                        _stuck_exprs.append(str(_r.problem_id))
                _stuck_exprs = list(dict.fromkeys(_stuck_exprs))[:8]  # dedup, max 8

                # Inject stuck_exprs into synthesizer before calling
                if _stuck_exprs and hasattr(synth, "_impl") and synth._impl is not None:
                    synth._impl._last_stuck_exprs = _stuck_exprs

                candidates = synth.synthesize_transforms(domain=domain, n=3)
                if not candidates:
                    log.info("[ExperimentRunner] TransformSynthesizer returned no candidates for '%s'", domain)
                    continue

                # Verification: test each candidate on 5 known problems before accepting
                verified = []
                for cand in candidates:
                    try:
                        _passes = self._verify_synthesized_transform(cand, domain, n_tests=5)
                        if _passes:
                            verified.append(cand)
                            log.info(
                                "[ExperimentRunner] Synthesized transform VERIFIED for '%s': %s",
                                domain, getattr(cand, 'name', lambda: str(cand))() if callable(getattr(cand, 'name', None)) else str(cand),
                            )
                        else:
                            log.info(
                                "[ExperimentRunner] Synthesized transform REJECTED (failed verification): %s",
                                getattr(cand, 'name', lambda: str(cand))() if callable(getattr(cand, 'name', None)) else str(cand),
                            )
                    except Exception as ve:
                        log.debug("[ExperimentRunner] Transform verification error: %s", ve)

                if verified:
                    self.transforms.extend(verified)
                    log.info(
                        "[ExperimentRunner] Added %d synthesized transform(s) for domain '%s'. Total transforms: %d",
                        len(verified), domain, len(self.transforms),
                    )
                    # Reset consecutive fail counter and lift cooldown — new capability added
                    self._domain_consec_fails[domain] = 0
                    self._domain_cooldown.pop(domain, None)
            except Exception as exc:
                log.warning("[ExperimentRunner] TransformSynthesizer error for '%s': %s", domain, exc)

    def _verify_synthesized_transform(self, transform, domain: str, n_tests: int = 5) -> bool:
        """Test a synthesized transform on a small set of known problems.

        Returns True if the transform applies and reduces energy on at least one
        test problem without raising exceptions on any of them.
        """
        try:
            problems = []
            # Try to get test problems from the curriculum generator
            if self.curriculum_gen is not None:
                try:
                    problems = [self.curriculum_gen.generate(domain=domain) for _ in range(n_tests)]
                    problems = [p for p in problems if p is not None]
                except Exception:
                    pass
            if not problems:
                return True  # Can't test — give benefit of the doubt

            energy_improvements = 0
            for prob in problems[:n_tests]:
                try:
                    from sare.engine import load_problem
                    _, g = load_problem(str(getattr(prob, 'expression', prob)))
                    matches = transform.match(g)
                    if matches:
                        new_g, delta = transform.apply(g, matches[0])
                        if delta < 0:
                            energy_improvements += 1
                except Exception:
                    return False  # Any exception during verification = reject

            return energy_improvements > 0
        except Exception:
            return True  # If verification itself errors, allow the transform

    def _persist_domain_mastery(self, domain: str, rate: float):
        """Write domain mastery to brain_state.json so the web server picks it up."""
        import json as _json, os as _os
        from pathlib import Path as _P
        state_path = _P(__file__).resolve().parents[3] / "data" / "memory" / "brain_state.json"
        try:
            state = _json.loads(state_path.read_text()) if state_path.exists() else {}
            stats = state.setdefault("stats", {})
            mastered = stats.setdefault("domains_mastered", [])
            if domain not in mastered:
                mastered.append(domain)
                # Recompute stage based on rules + domains + solve_rate
                rules = stats.get("rules_promoted", 0)
                solve_rate = rate
                new_stage = _compute_stage(rules, len(mastered), solve_rate)
                old_stage = state.get("stage", "infant")
                if new_stage != old_stage:
                    state["stage"] = new_stage
                    log.info("[ExperimentRunner] Stage advanced: %s → %s", old_stage, new_stage)
                tmp = state_path.parent / f"{state_path.stem}.{os.getpid()}.{threading.get_ident()}.tmp"
                tmp.write_text(_json.dumps(state, indent=2))
                _os.replace(tmp, state_path)
                log.info("[ExperimentRunner] Domain mastered (persisted): %s (rate=%.2f) stage=%s",
                         domain, rate, state["stage"])
        except Exception as e:
            log.warning("[ExperimentRunner] _persist_domain_mastery failed: %s", e)

    def start_daemon(self, interval_seconds: float = 30.0, batch_size: int = 10):
        if self._daemon_thread is not None and self._daemon_thread.is_alive():
            return

        self._stop_event.clear()

        def _loop():
            while not self._stop_event.is_set():
                try:
                    self.run_batch(n=batch_size)
                except Exception as exc:
                    log.exception("ExperimentRunner daemon batch failed: %s", exc)
                self._stop_event.wait(interval_seconds)

        self._daemon_thread = threading.Thread(target=_loop, name="ExperimentRunnerDaemon", daemon=True)
        self._daemon_thread.start()

    def stop_daemon(self):
        self._stop_event.set()
        if self._daemon_thread is not None:
            self._daemon_thread.join(timeout=2.0)

    @property
    def history(self) -> List[ExperimentResult]:
        return list(self._history)

    def stats(self) -> dict:
        """Return summary statistics for the web dashboard."""
        h = self._history
        total = len(h)
        solved = sum(1 for r in h if r.solved)

        # Compute real-time schema hit rate
        _schema_hit_rate = 0.0
        try:
            from sare.cognition.schema_matcher import get_schema_matcher
            _sm = get_schema_matcher()
            _total = _sm._hits + _sm._misses
            _schema_hit_rate = round(_sm._hits / max(1, _total), 3)
        except Exception:
            pass

        return {
            "total_experiments": total,
            "solved": solved,
            "solve_rate": round(solved / total, 3) if total else 0.0,
            "schema_hit_rate": _schema_hit_rate,
            "genuine_solve_rate": round(solved / total * (1 - _schema_hit_rate), 3) if total else 0.0,
            "recent_batches": self._recent_batch_stats[-4:],
            "transforms_available": len(self.transforms),
        }

    def surprise_stats(self) -> dict:
        """Return prediction/surprise stats from WorldModel if available."""
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            report = wm.get_prediction_report() if hasattr(wm, "get_prediction_report") else {}
            return report or {"predictions": 0, "accuracy": 0.0}
        except Exception:
            return {"predictions": 0, "accuracy": 0.0}
