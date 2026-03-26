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

        self._recent_batch_stats: List[dict] = []
        self._adaptive_batch_size: Optional[int] = None

        # Heuristic pre-scoring: store successful embeddings + transform names
        # so we can cosine-rank transforms before BeamSearch.
        # Each entry: {"embedding": List[float], "transforms": List[str]}
        self._successful_embeddings: List[dict] = []
        self._embedder = None  # lazy-loaded GraphEmbedding instance
        self._value_net = None  # lazy-loaded MLXValueNet (M1 GPU)
        self._last_wm_prediction = None  # world model prediction for current problem

        # P2-G: per-transform uncertainty tracking (populated from credit assigner)
        self._transform_uncertainty: dict = {}
        self._solve_count: int = 0   # used for composite learner mine frequency

        # P3-H: A* fallback for hard problems
        self._use_astar_fallback: bool = True

        # Load synthesized transforms from data/memory/synthesized_modules/
        synth = self._load_synthesized_transforms()
        if synth:
            self.transforms = list(self.transforms) + synth

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

        # P2-F: pass heuristic_fn to BeamSearch
        try:
            from sare.brain import get_brain as _get_brain
            _br = _get_brain()
            if _br is not None and hasattr(_br, 'heuristic_model') and _br.heuristic_model is not None:
                kwargs["heuristic_fn"] = _br.heuristic_model.predict_graph
        except Exception:
            pass

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
            if result.rule_promoted:
                log.info("[ExperimentRunner] Rule promoted: %s (%s)", result.rule_name or "(unnamed)", verdict.reasoning)

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
                # Fix 3: Immediately refresh transforms instead of waiting 10 cycles
                try:
                    brain = getattr(self, "_brain_ref", None)
                    if brain is not None and hasattr(brain, "_refresh_transforms"):
                        brain._refresh_transforms()
                except Exception as exc:
                    log.debug("Immediate transform refresh failed: %s", exc)

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
                    from sare.transfer.engine import get_transfer_engine
                    get_transfer_engine().observe(_te_transforms, _te_domain, success=True)
                except Exception:
                    pass

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
                    _cl.mine_composites()
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

    def _run_single(self, problem) -> ExperimentResult:
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
        self.beam_width = max(2, self.beam_width + beam_delta)
        # Adaptive budget: adjust by emotional state, never drop below 0.3s
        self.budget_seconds = max(0.3, self.budget_seconds + budget_delta)

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

        if not _schema_hit:
            try:
                # P2-E: Transform family routing — filter to domain-relevant transforms
                _search_transforms = self.transforms
                try:
                    from sare.transforms.transform_families import get_family_transforms
                    _search_transforms = get_family_transforms(domain, self.transforms)
                except Exception:
                    pass

                # P3-I: CompositeRuleLearner — suggest transform order
                try:
                    from sare.learning.composite_rule_learner import get_composite_learner
                    _search_transforms = get_composite_learner().suggest_transform_order(_search_transforms, domain)
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

        # Feed solve outcome to AlgorithmSelector so strategy win rates improve over time
        try:
            from sare.meta.algorithm_selector import get_algorithm_selector as _get_as
            _get_as().record_outcome(domain or "algebra", "beam_search", solved)
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

        self._recent_batch_stats.append(
            {
                "requested_n": requested_n,
                "attempted": attempted,
                "solved": solved_count,
                "solve_rate": solve_rate,
                "avg_elapsed_ms": avg_elapsed_ms,
                "elapsed_s": elapsed_s,
                "load": load,
                "timestamp": time.time(),
            }
        )
        if len(self._recent_batch_stats) > 8:
            self._recent_batch_stats = self._recent_batch_stats[-8:]

    def _adaptive_target_batch_size(self, requested_n: int) -> int:
        if requested_n <= 1:
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

        try:
            outcome = self._search(graph)
            solved, final_graph, proof_steps, _ = self._parse_search_outcome(
                outcome, graph, energy_before)
        except Exception:
            solved, final_graph, proof_steps = False, graph, []

        energy_after = self._evaluate_energy(final_graph)
        delta = energy_before - energy_after

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
        )
        # Still run reflect-and-learn on successful fast solves (rule promotion)
        if solved:
            result = self._reflect_and_learn(problem, result)
        return result

    def run_batch(self, n: int = 10) -> List[ExperimentResult]:
        import concurrent.futures as _cf
        batch_start = time.time()
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

        # Use fast path for bulk problems; keep full path for 1-in-50 (cognitive updates)
        _full_every = max(1, len(problems) // 50)
        def _solver(idx_problem):
            idx, p = idx_problem
            if idx % _full_every == 0:
                return self._run_single(p)
            return self._run_single_fast(p)

        # Solve in parallel threads
        results: List[ExperimentResult] = []
        max_workers = min(len(problems), max(1, (os.cpu_count() or 4)))
        if max_workers > 1 and len(problems) > 1:
            with _cf.ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="sare-solve") as pool:
                futs = {pool.submit(_solver, (i, p)): p
                        for i, p in enumerate(problems)}
                for fut in _cf.as_completed(futs):
                    try:
                        results.append(fut.result(timeout=30.0))
                    except Exception as exc:
                        log.debug("Parallel solve error: %s", exc)
        else:
            for i, problem in enumerate(problems):
                results.append(_solver((i, problem)))

        for r in results:
            self._history.append(r)

        # Update per-domain solve tracker and check for mastery
        for r in results:
            domain = str(getattr(r, 'domain', '') or 'general')
            if not domain:
                domain = 'general'
            t = self._domain_solve_tracker.setdefault(domain, {"attempts": 0, "successes": 0})
            t["attempts"] += 1
            if r.solved:
                t["successes"] += 1
        self._maybe_emit_domain_mastery()

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
                rules = stats.get("rules_promoted", 0) + stats.get("rules_discovered", 0)
                solve_rate = rate
                new_stage = _compute_stage(rules, len(mastered), solve_rate)
                old_stage = state.get("stage", "infant")
                if new_stage != old_stage:
                    state["stage"] = new_stage
                    log.info("[ExperimentRunner] Stage advanced: %s → %s", old_stage, new_stage)
                tmp = state_path.with_suffix(".tmp")
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
        return {
            "total_experiments": total,
            "solved": solved,
            "solve_rate": round(solved / total, 3) if total else 0.0,
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