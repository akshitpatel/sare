import logging
import threading
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional


log = logging.getLogger(__name__)


class _FailureReplayProblem:
    def __init__(self, problem_id: str, graph):
        self.id = problem_id
        self.graph = graph


class HippocampusDaemon(threading.Thread):
    """
    Background daemon that monitors system activity.
    If the system is idle for > SLEEP_THRESHOLD_SECONDS, it enters Sleep Mode
    and performs offline consolidation over recent experience.
    """

    SLEEP_THRESHOLD_SECONDS = 15.0

    def __init__(
        self,
        memory_manager=None,
        experiment_runner=None,
        reflection_engine=None,
        curriculum_gen=None,
        problem_loader: Optional[Callable[[str], Any]] = None,
    ):
        super().__init__(daemon=True, name="Hippocampus")
        self.last_active_time = time.time()
        self.is_sleeping = False
        self.current_task = "Booting"
        self.episodes_replayed = 0
        self.running = True
        self.learning_lock = threading.Lock()
        self.memory_manager = memory_manager
        self.experiment_runner = experiment_runner
        self.reflection_engine = reflection_engine
        self.curriculum_gen = curriculum_gen
        self.problem_loader = problem_loader or self._default_problem_loader
        self.failure_clusters: List[dict] = []
        self.last_sleep_report: Dict[str, Any] = {}

    def wire(self, memory_manager=None, experiment_runner=None, reflection_engine=None, curriculum_gen=None, problem_loader=None):
        if memory_manager is not None:
            self.memory_manager = memory_manager
        if experiment_runner is not None:
            self.experiment_runner = experiment_runner
        if reflection_engine is not None:
            self.reflection_engine = reflection_engine
        if curriculum_gen is not None:
            self.curriculum_gen = curriculum_gen
        if problem_loader is not None:
            self.problem_loader = problem_loader

    def ping_active(self):
        """Reset the idle timer when the system receives activity."""
        self.last_active_time = time.time()
        if self.is_sleeping:
            self.is_sleeping = False
            self.current_task = "Waking up due to stimulus"

    def status(self) -> Dict[str, Any]:
        """Return current sleep/consolidation state."""
        idle_time = time.time() - self.last_active_time
        return {
            "state": "sleeping" if self.is_sleeping else "awake",
            "idle_seconds": round(idle_time, 1),
            "current_task": self.current_task if self.is_sleeping else "Monitoring",
            "episodes_replayed": self.episodes_replayed,
            "failure_clusters": len(self.failure_clusters),
            "last_sleep_report": dict(self.last_sleep_report),
        }

    def run(self):
        self.current_task = "Monitoring"
        while self.running:
            time.sleep(2.0)
            idle_time = time.time() - self.last_active_time
            if not self.is_sleeping and idle_time > self.SLEEP_THRESHOLD_SECONDS:
                self.is_sleeping = True
                self._sleep_cycle()

    @staticmethod
    def _default_problem_loader(problem_id: str):
        try:
            from sare.engine import load_problem
            _, graph = load_problem(problem_id)
            return graph
        except Exception:
            return None

    def _load_problem_graph(self, problem_id: str):
        if not self.problem_loader:
            return None
        try:
            return self.problem_loader(problem_id)
        except Exception as exc:
            log.debug("Hippocampus problem load failed for %s: %s", problem_id, exc)
            return None

    @staticmethod
    def _graph_signature(graph) -> str:
        node_counts = Counter(node.type for node in getattr(graph, "nodes", []))
        edge_counts = Counter(edge.relationship_type for edge in getattr(graph, "edges", []))
        op_labels = sorted(
            (getattr(node, "label", "") or getattr(node, "attributes", {}).get("op", ""))
            for node in getattr(graph, "nodes", [])
            if getattr(node, "type", "") == "operator"
        )
        node_part = ",".join(f"{name}:{count}" for name, count in sorted(node_counts.items()))
        edge_part = ",".join(f"{name}:{count}" for name, count in sorted(edge_counts.items()))
        op_part = ",".join(label for label in op_labels if label)
        return f"nodes[{node_part}]|edges[{edge_part}]|ops[{op_part}]"

    def _cluster_failure_patterns(self, failed_episodes) -> List[dict]:
        clusters: Dict[str, dict] = {}
        signature_cache: Dict[str, str] = {}

        for episode in failed_episodes:
            problem_id = episode.problem_id
            if problem_id in signature_cache:
                signature = signature_cache[problem_id]
            else:
                graph = self._load_problem_graph(problem_id)
                if graph is None:
                    continue
                signature = self._graph_signature(graph)
                signature_cache[problem_id] = signature

            cluster = clusters.setdefault(signature, {
                "cluster_id": signature,
                "count": 0,
                "sample_problems": [],
                "avg_initial_energy": 0.0,
                "avg_final_energy": 0.0,
                "avg_transforms": 0.0,
            })

            cluster["count"] += 1
            count = cluster["count"]
            cluster["avg_initial_energy"] += (episode.initial_energy - cluster["avg_initial_energy"]) / count
            cluster["avg_final_energy"] += (episode.final_energy - cluster["avg_final_energy"]) / count
            cluster["avg_transforms"] += (len(episode.transform_sequence) - cluster["avg_transforms"]) / count
            if len(cluster["sample_problems"]) < 3:
                cluster["sample_problems"].append(problem_id)

        ordered = sorted(clusters.values(), key=lambda item: (-item["count"], item["cluster_id"]))
        self.failure_clusters = ordered[:5]
        return self.failure_clusters

    def _queue_failure_retries(self, clusters: List[dict]) -> int:
        if not self.curriculum_gen:
            return 0
        queued = 0
        for cluster in clusters:
            for problem_id in cluster["sample_problems"]:
                if self.curriculum_gen.queue_problem(problem_id, priority="high", reason="failure_replay"):
                    queued += 1
        return queued

    def _sleep_cycle(self):
        self.current_task = "Loading recent episodes"
        log.info("Hippocampus entering sleep cycle")
        if not self.memory_manager:
            log.warning("Hippocampus cannot consolidate: no memory manager wired")
            self.is_sleeping = False
            return

        recent_episodes = self.memory_manager.recent_episodes(100)
        if not recent_episodes:
            self.current_task = "No recent episodes"
            time.sleep(5.0)
            self.is_sleeping = False
            return

        failed = [ep for ep in recent_episodes if not ep.success]
        self.current_task = f"Clustering {len(failed)} failures"
        clusters = self._cluster_failure_patterns(failed)

        self.current_task = "Queueing failure retries"
        queued = self._queue_failure_retries(clusters)

        # Phase F: Forgetting curve — decay all tracked memories and get due reviews
        forgetting_at_risk = 0
        try:
            from sare.memory.forgetting_curve import get_forgetting_curve
            self.current_task = "Applying Ebbinghaus decay"
            fc = get_forgetting_curve()
            forgetting_at_risk = fc.decay_all()
            due_reviews = fc.get_due_reviews(limit=10)
            if due_reviews and self.curriculum_gen:
                for item in due_reviews:
                    try:
                        self.curriculum_gen.queue_problem(
                            item.item_id,
                            priority="high",
                            reason="forgetting_curve_review",
                        )
                    except Exception:
                        pass
        except Exception:
            pass

        # Phase C: Internal grammar — discover new composite symbols from thought history
        new_abstractions = 0
        try:
            from sare.language.internal_grammar import get_internal_grammar
            self.current_task = "Mining grammar abstractions"
            grammar = get_internal_grammar()
            abstractions = grammar.discover_abstractions()
            new_abstractions = len(abstractions)
            if new_abstractions > 0:
                log.info("Hippocampus: discovered %d new grammar abstractions", new_abstractions)
        except Exception:
            pass

        self.current_task = "Running reflection"
        if self.reflection_engine:
            self.reflection_engine.consolidate(recent_episodes)

        self.current_task = "Running experiments"
        if self.experiment_runner:
            _run = getattr(self.experiment_runner, "run_scheduled",
                           getattr(self.experiment_runner, "run_batch", None))
            if _run:
                _run()

        self.last_sleep_report = {
            "timestamp": time.time(),
            "episodes_processed": len(recent_episodes),
            "failures_clustered": len(failed),
            "clusters_found": len(clusters),
            "retries_queued": queued,
            "forgetting_at_risk": forgetting_at_risk,
            "new_grammar_abstractions": new_abstractions,
        }
        log.info("Hippocampus sleep cycle complete: %s", self.last_sleep_report)
        self.episodes_replayed += len(recent_episodes)
        self.is_sleeping = False

    def replay_episodes(self, curriculum_gen=None) -> int:
        """
        Episodic replay loop: pull hard episodes from autobiographical memory and
        inject them back into the curriculum so they get retried.

        Designed to be called periodically from the learn daemon (e.g. every 25 cycles).
        Does NOT import hippocampus from autobiographical — no circular import.

        Returns the number of problems injected.
        """
        injected = 0
        try:
            from sare.memory.autobiographical import get_autobiographical_memory
            autobio = get_autobiographical_memory()
            hard_episodes = autobio.get_hard_episodes(n=10)
        except Exception as exc:
            log.debug("replay_episodes: could not load autobiographical memory: %s", exc)
            return 0

        _curriculum = curriculum_gen or self.curriculum_gen
        if _curriculum is None:
            log.debug("replay_episodes: no curriculum_gen available, skipping injection")
            return 0

        for ep in hard_episodes:
            try:
                expression = ep.get("expression", "")
                domain = ep.get("domain", "general")
                if not expression:
                    continue

                # Try queue_problem (used by web.py wiring) then fall back to add_seed
                queued = False
                if hasattr(_curriculum, "queue_problem"):
                    try:
                        result = _curriculum.queue_problem(
                            expression, priority="high", reason="episodic_replay"
                        )
                        queued = bool(result)
                    except Exception:
                        pass

                if not queued and hasattr(_curriculum, "add_seed"):
                    try:
                        graph = self._load_problem_graph(expression)
                        if graph is not None:
                            _curriculum.add_seed(graph)
                            queued = True
                        else:
                            # Last resort: inject via _priority_queue list if present
                            pq = getattr(_curriculum, "_priority_queue", None)
                            if pq is not None and isinstance(pq, list):
                                pq.append({"expression": expression, "domain": domain})
                                queued = True
                    except Exception:
                        pass

                if queued:
                    injected += 1
                    log.debug("replay_episodes: injected expression=%r domain=%s", expression, domain)

            except Exception as ep_exc:
                log.debug("replay_episodes: error processing episode %s: %s", ep, ep_exc)

        if injected:
            log.info("Hippocampus.replay_episodes: injected %d hard episodes into curriculum", injected)
        return injected