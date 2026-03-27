"""
MetaLearningEngine — Experiments with SARE-HX's own internal settings.

Implements recursive self-improvement by:
1. Testing different search configurations (beam_width, budget_seconds)
2. Measuring actual performance (solve rate, avg energy reduction)
3. Promoting the best config as the new default
4. Logging all experiments for introspection

This is the "meta" layer: the system improving its own thinking process.
"""

from __future__ import annotations

import logging
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """A specific configuration to test."""
    name: str
    beam_width: int = 8
    budget_seconds: float = 2.0
    mcts_simulations: int = 200
    max_depth: int = 12
    strategy: str = "beam_search"   # "beam_search" | "mcts"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "beam_width": self.beam_width,
            "budget_seconds": self.budget_seconds,
            "mcts_simulations": self.mcts_simulations,
            "max_depth": self.max_depth,
            "strategy": self.strategy,
        }


@dataclass
class ExperimentResult:
    """Result of running one configuration on a benchmark."""
    config: ExperimentConfig
    solve_rate: float = 0.0
    avg_delta: float = 0.0
    avg_time_ms: float = 0.0
    n_problems: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def score(self) -> float:
        """
        Composite score balancing solve rate, energy gain, and speed.
        Higher is better.
        """
        if self.n_problems == 0:
            return 0.0
        speed_factor = max(0.1, 1.0 - self.avg_time_ms / 5000)
        return self.solve_rate * self.avg_delta * speed_factor

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "solve_rate": round(self.solve_rate, 3),
            "avg_delta": round(self.avg_delta, 3),
            "avg_time_ms": round(self.avg_time_ms, 1),
            "n_problems": self.n_problems,
            "score": round(self.score, 4),
            "timestamp": self.timestamp,
        }


# Default configuration candidates to explore
_DEFAULT_CONFIGS = [
    ExperimentConfig("narrow_fast",   beam_width=4,  budget_seconds=1.0, max_depth=8),
    ExperimentConfig("standard",      beam_width=8,  budget_seconds=2.0, max_depth=12),
    ExperimentConfig("wide_thorough", beam_width=16, budget_seconds=3.0, max_depth=16),
    ExperimentConfig("deep_search",   beam_width=8,  budget_seconds=4.0, max_depth=20),
    ExperimentConfig("mcts_light",    beam_width=4,  budget_seconds=2.0,
                     mcts_simulations=100, strategy="mcts"),
    ExperimentConfig("mcts_heavy",    beam_width=4,  budget_seconds=4.0,
                     mcts_simulations=400, strategy="mcts"),
]

# Benchmark problems covering multiple domains
_BENCHMARK_PROBLEMS = [
    "x + 0",
    "x * 1",
    "x - x",
    "not not p",
    "p and true",
    "sin(0)",
    "log(1)",
    "integral(x)",
    "x * 0",
    "p or false",
]

# Per-domain task pools for MAML inner/outer loops
_DOMAIN_TASKS: Dict[str, List[str]] = {
    "arithmetic":    ["x + 0", "x * 1", "x * 0", "x - x", "2 + 3"],
    "logic":         ["not not p", "p and true", "p or false", "p and false", "true or false"],
    "algebra":       ["x + 0", "x * 1", "a * (b + 0)", "(x + 0) * 1", "x + x - x"],
    "calculus":      ["integral(x)", "sin(0)", "log(1)", "derivative(x)", "cos(0)"],
    "trigonometry":  ["sin(0)", "cos(0)", "tan(0)", "sin(x) * sin(x) + cos(x) * cos(x)"],
    "probability":   ["p + (1 - p)", "p * 1", "p * 0"],
    "thermodynamics":["delta_U + 0", "Q - W"],
}


class MetaLearningEngine:
    """
    Self-improving search configuration optimizer.

    Runs controlled experiments: tries different beam widths,
    budget constraints, and strategies on benchmark problems.
    Promotes the best-performing config as the system default.

    Connects to the SelfModel so learning goals can include
    "improve reasoning speed" or "try wider beam search."
    """

    def __init__(self):
        self._configs: List[ExperimentConfig] = list(_DEFAULT_CONFIGS)
        self._results: List[ExperimentResult] = []
        self._best: Optional[ExperimentConfig] = None
        self._current: ExperimentConfig = ExperimentConfig("standard")
        self._experiments_run: int = 0
        self._last_tuned: float = 0.0
        self._improvement_history: List[dict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def current_config(self) -> ExperimentConfig:
        return self._best or self._current

    def should_tune(self, min_interval_seconds: float = 300.0) -> bool:
        """True if enough time has passed to run another tuning cycle."""
        return time.time() - self._last_tuned > min_interval_seconds

    def run_experiment(self, config: ExperimentConfig,
                       problems: List[str] = None) -> ExperimentResult:
        """
        Run one configuration on a set of benchmark problems.
        Returns performance metrics.
        """
        from sare.engine import (
            load_problem, get_transforms, EnergyEvaluator, BeamSearch, MCTSSearch
        )
        bench = problems or _BENCHMARK_PROBLEMS
        transforms = get_transforms(include_macros=False, include_synthesized=False)

        successes = 0
        total_delta = 0.0
        total_time_ms = 0.0

        for expr in bench:
            try:
                evaluator = EnergyEvaluator()
                _, g = load_problem(expr)
                e0 = evaluator.compute(g).total

                t0 = time.perf_counter()
                if config.strategy == "mcts":
                    searcher = MCTSSearch()
                    result = searcher.search(
                        g, evaluator, transforms,
                        n_simulations=config.mcts_simulations,
                        budget_seconds=config.budget_seconds,
                    )
                else:
                    searcher = BeamSearch()
                    result = searcher.search(
                        g, evaluator, transforms,
                        beam_width=config.beam_width,
                        max_depth=config.max_depth,
                        budget_seconds=config.budget_seconds,
                    )
                elapsed_ms = (time.perf_counter() - t0) * 1000

                delta = e0 - result.energy.total
                if delta > 0.01:
                    successes += 1
                    total_delta += delta
                total_time_ms += elapsed_ms
            except Exception:
                total_time_ms += config.budget_seconds * 1000

        n = len(bench)
        res = ExperimentResult(
            config=config,
            solve_rate=successes / n if n > 0 else 0.0,
            avg_delta=total_delta / max(successes, 1),
            avg_time_ms=total_time_ms / n if n > 0 else 0.0,
            n_problems=n,
        )
        self._results.append(res)
        self._experiments_run += 1
        log.info(
            "MetaLearn experiment '%s': rate=%.2f delta=%.3f time=%.0fms score=%.4f",
            config.name, res.solve_rate, res.avg_delta, res.avg_time_ms, res.score,
        )
        return res

    def tune_beam_width(self, problems: List[str] = None) -> ExperimentConfig:
        """
        Quick beam-width sweep: test 4, 8, 12, 16, 20.
        Returns the best config found.
        """
        candidates = [
            ExperimentConfig(f"bw_{w}", beam_width=w, budget_seconds=2.0, max_depth=12)
            for w in [4, 8, 12, 16, 20]
        ]
        results = [self.run_experiment(c, problems) for c in candidates]
        best_result = max(results, key=lambda r: r.score)
        self._promote(best_result.config, results)
        return best_result.config

    def full_tune(self, problems: List[str] = None) -> ExperimentConfig:
        """
        Full sweep across all default configs.
        More thorough but takes longer.
        """
        results = [self.run_experiment(c, problems) for c in self._configs]
        best_result = max(results, key=lambda r: r.score)
        self._promote(best_result.config, results)
        return best_result.config

    def _promote(self, new_best: ExperimentConfig,
                 all_results: List[ExperimentResult]):
        """Promote a new best config and record the improvement."""
        old_score = max(
            (r.score for r in self._results[:-len(all_results)]
             if r.config.name == (self._best.name if self._best else "standard")),
            default=0.0,
        )
        new_score = max(r.score for r in all_results)
        if self._best is None or new_score > old_score:
            prev_name = self._best.name if self._best else "none"
            self._best = new_best
            self._improvement_history.append({
                "from": prev_name,
                "to": new_best.name,
                "score_before": round(old_score, 4),
                "score_after": round(new_score, 4),
                "gain": round(new_score - old_score, 4),
                "timestamp": time.time(),
            })
            log.info(
                "MetaLearn promoted '%s' (score %.4f → %.4f)",
                new_best.name, old_score, new_score,
            )
        self._last_tuned = time.time()

    # ── MAML-style fast adaptation ────────────────────────────────────────────

    def fast_adapt(self, domain: str, n_inner_steps: int = 3,
                   n_support: int = 3, n_query: int = 3) -> Optional[ExperimentConfig]:
        """
        MAML-style fast adaptation for a target domain.

        Inner loop: run n_inner_steps of config improvement on a support set
                    (small random sample from the domain task pool).
        Outer loop: evaluate the adapted config on a held-out query set.

        Returns the best adapted config if it beats the current default,
        else returns None.
        """
        task_pool = _DOMAIN_TASKS.get(domain, _BENCHMARK_PROBLEMS)
        if len(task_pool) < n_support + n_query:
            support = task_pool
            query   = task_pool
        else:
            shuffled = random.sample(task_pool, len(task_pool))
            support  = shuffled[:n_support]
            query    = shuffled[n_support:n_support + n_query]

        log.info("MAML fast_adapt: domain=%s support=%d query=%d inner_steps=%d",
                 domain, len(support), len(query), n_inner_steps)

        # --- Inner loop: gradient-free config search on support set ---
        # Simulate 'gradient steps' by iteratively picking configs that
        # improve on the support set; learning-rate analogue = config delta.
        adapted_configs = [ExperimentConfig(f"maml_base_{domain}",
                                            beam_width=self.current_config.beam_width,
                                            budget_seconds=self.current_config.budget_seconds,
                                            max_depth=self.current_config.max_depth,
                                            strategy=self.current_config.strategy)]
        best_support_score = self.run_experiment(adapted_configs[0], support).score

        for step in range(n_inner_steps):
            # Generate a neighbourhood of configs by perturbing beam_width +/- 2
            bw = adapted_configs[-1].beam_width
            candidates = [
                ExperimentConfig(f"maml_inner_{domain}_s{step}_bw{w}",
                                 beam_width=w,
                                 budget_seconds=adapted_configs[-1].budget_seconds,
                                 max_depth=adapted_configs[-1].max_depth,
                                 strategy=adapted_configs[-1].strategy)
                for w in [max(2, bw - 2), bw, bw + 2, bw + 4]
            ]
            step_results = [self.run_experiment(c, support) for c in candidates]
            best_step = max(step_results, key=lambda r: r.score)
            if best_step.score > best_support_score:
                best_support_score = best_step.score
                adapted_configs.append(best_step.config)
                log.debug("MAML inner step %d: improved to bw=%d score=%.4f",
                          step, best_step.config.beam_width, best_step.score)

        final_adapted = adapted_configs[-1]

        # --- Outer loop: evaluate adapted config on query set ---
        query_result  = self.run_experiment(final_adapted, query)
        baseline_result = self.run_experiment(self.current_config, query)

        log.info("MAML outer eval: adapted=%.4f baseline=%.4f (domain=%s)",
                 query_result.score, baseline_result.score, domain)

        if query_result.score > baseline_result.score:
            # Adapted config generalises better — record as domain-specific winner
            adapted_name = f"maml_{domain}"
            final_adapted.name = adapted_name
            self._results.append(query_result)
            self._improvement_history.append({
                "from": self.current_config.name,
                "to": adapted_name,
                "domain": domain,
                "score_before": round(baseline_result.score, 4),
                "score_after":  round(query_result.score, 4),
                "gain":         round(query_result.score - baseline_result.score, 4),
                "method":       "maml_fast_adapt",
                "timestamp":    time.time(),
            })
            self._best = final_adapted
            self._last_tuned = time.time()
            log.info("MAML: promoted '%s' for domain '%s' (gain=+%.4f)",
                     adapted_name, domain, query_result.score - baseline_result.score)
            return final_adapted

        return None

    def online_adapt(self, expression: str, domain: str,
                     solved: bool, delta: float) -> None:
        """
        Online single-step adaptation: adjust beam_width based on
        live solve outcome without running a full experiment.
        Uses an EMA solve-rate tracker per domain.
        """
        if not hasattr(self, "_domain_ema"):
            self._domain_ema: Dict[str, float] = {}
        alpha = 0.15
        prev = self._domain_ema.get(domain, 0.5)
        self._domain_ema[domain] = (1 - alpha) * prev + alpha * (1.0 if solved else 0.0)
        ema = self._domain_ema[domain]

        # Shrink beam if doing well (faster), grow beam if struggling
        if ema > 0.85 and self.current_config.beam_width > 4:
            self.current_config.beam_width = max(4, self.current_config.beam_width - 1)
            log.debug("online_adapt [%s]: ema=%.2f → shrink bw to %d",
                      domain, ema, self.current_config.beam_width)
        elif ema < 0.35 and self.current_config.beam_width < 24:
            self.current_config.beam_width = min(24, self.current_config.beam_width + 2)
            log.debug("online_adapt [%s]: ema=%.2f → grow bw to %d",
                      domain, ema, self.current_config.beam_width)

    def apply_to_brain(self, brain) -> bool:
        """
        Apply the best found config to the Brain's search parameters.
        Returns True if an actual change was made.
        """
        cfg = self.current_config
        changed = False
        try:
            if hasattr(brain, '_beam_width') and brain._beam_width != cfg.beam_width:
                brain._beam_width = cfg.beam_width
                changed = True
            if hasattr(brain, '_budget_seconds') and brain._budget_seconds != cfg.budget_seconds:
                brain._budget_seconds = cfg.budget_seconds
                changed = True
            if changed:
                log.info(
                    "MetaLearn applied config '%s': bw=%d budget=%.1fs",
                    cfg.name, cfg.beam_width, cfg.budget_seconds,
                )
        except Exception as e:
            log.debug("MetaLearn apply error: %s", e)
        return changed

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Dict suitable for API / dashboard display."""
        recent = self._results[-20:]
        maml_history = [h for h in self._improvement_history if h.get("method") == "maml_fast_adapt"]
        return {
            "experiments_run": self._experiments_run,
            "current_config":  self.current_config.to_dict(),
            "best_config":     self._best.to_dict() if self._best else None,
            "improvement_history": self._improvement_history[-5:],
            "recent_results":  [r.to_dict() for r in recent[-5:]],
            "configs_available": len(self._configs),
            "last_tuned": self._last_tuned,
            "maml": {
                "fast_adapt_runs": len(maml_history),
                "recent_adaptations": maml_history[-3:],
                "domain_ema": getattr(self, "_domain_ema", {}),
                "weakest_domain": (
                    min(self._domain_ema, key=self._domain_ema.get)
                    if getattr(self, "_domain_ema", {}) else None
                ),
            },
        }
