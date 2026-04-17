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
        budget_seconds: float = 5.0,
        analogy_transfer=None,
        **kwargs,
    ):
        self.curriculum_gen = curriculum_gen
        self.searcher = searcher
        self.energy = energy
        self.reflection_engine = reflection_engine
        self.causal_induction = causal_induction
        self.concept_registry = concept_registry
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

    def _search(self, graph):
        kwargs = {
            "beam_width": self.beam_width,
            "budget_seconds": self.budget_seconds,
        }
        transforms = self.transforms

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

    def _parse_search_outcome(self, outcome, original_graph):
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
                # Solved if final energy is below 1.0 (complexity-only baseline)
                outcome_energy = getattr(outcome, "energy", None)
                if outcome_energy is not None:
                    e_total = float(getattr(outcome_energy, "total", outcome_energy) or 0)
                    solved = e_total < 1.0

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
            # Use async queue_episode when available; fall back to synchronous induce().
            if hasattr(self.causal_induction, "queue_episode"):
                self.causal_induction.queue_episode(
                    problem=problem,
                    result=result,
                    reflection=reflection,
                )
                verdict = None  # verdict arrives asynchronously via callback
            else:
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
            result.rule_name = str(getattr(verdict, "rule_name", getattr(verdict, "name", "")) or "")
            result.rule_promoted = bool(getattr(verdict, "promoted", getattr(verdict, "rule_promoted", False)))
            result.reasoning = str(getattr(verdict, "reasoning", getattr(verdict, "verdict", "")) or "")

            if result.rule_promoted and self.concept_registry is not None:
                try:
                    if hasattr(self.concept_registry, "register"):
                        self.concept_registry.register(verdict)
                    elif hasattr(self.concept_registry, "promote"):
                        self.concept_registry.promote(verdict)
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
        # value net (if available) contributes 0.4
        best_set = set(best_transforms)
        cosine_score = best_sim  # similarity of current graph to best past episode

        def _transform_score(t) -> float:
            in_best = t.name() in best_set
            cs = cosine_score if in_best else 0.0
            if value_net_score is not None:
                # value net scores the overall state; use it as a tie-breaker /
                # blend for transforms that were in the best episode
                vs = float(value_net_score) if in_best else 0.0
                return 0.6 * cs + 0.4 * vs
            return cs

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
        self.budget_seconds = max(1.0, self.budget_seconds + budget_delta)

        if beam_delta != 0 or budget_delta != 0.0:
            if _im is not None:
                _im.think(
                    f"Emotional bias: beam={self.beam_width}, budget={self.budget_seconds:.1f}s "
                    f"(mode={emotional_bias.get('mode','normal')})",
                    context="homeostasis", emotion="neutral",
                )

        # Reset world model prediction slot for this problem
        self._last_wm_prediction = None

        # Heuristic pre-scoring: reorder transforms based on cosine similarity
        # to past successful graph embeddings (MPS-accelerated on Apple M1).
        _saved_transforms = self.transforms
        try:
            self.transforms = self._heuristic_reorder_transforms(graph)
        except Exception:
            pass  # safe fallback: keep original order

        start = time.time()
        energy_before = self._evaluate_energy(graph)

        try:
            outcome = self._search(graph)
            solved, final_graph, proof_steps, proof_nl = self._parse_search_outcome(outcome, graph)
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

        energy_after = self._evaluate_energy(final_graph)

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

        # Record world model prediction outcome so predictions improve over time
        if self._last_wm_prediction is not None:
            try:
                from sare.memory.world_model import get_world_model
                get_world_model().record_outcome(
                    self._last_wm_prediction,
                    actual_transforms=proof_steps,
                    actual_delta=energy_before - energy_after,
                    domain=domain,
                )
            except Exception:
                pass
            finally:
                self._last_wm_prediction = None
        elapsed_ms = (time.time() - start) * 1000.0

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

    def run_batch(self, n: int = 10) -> List[ExperimentResult]:
        batch_start = time.time()
        target_n = self._adaptive_target_batch_size(n)
        results: List[ExperimentResult] = []

        for _ in range(target_n):
            try:
                problem = self._pick_next_problem()
            except Exception as exc:
                log.debug("No problem available for experiment batch: %s", exc)
                break

            if problem is None:
                break

            result = self._run_single(problem)
            self._history.append(result)
            results.append(result)

        self._batch_count += 1
        self._record_batch_stats(requested_n=n, results=results, elapsed_s=time.time() - batch_start)
        return results

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