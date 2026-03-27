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

        for episode in failed_episodes:
            graph = self._load_problem_graph(episode.problem_id)
            if graph is None:
                continue

            signature = self._graph_signature(graph)
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
                cluster["sample_problems"].append(episode.problem_id)

        ordered = sorted(clusters.values(), key=lambda item: (-item["count"], item["cluster_id"]))
        self.failure_clusters = ordered[:5]
        return self.failure_clusters

    def _queue_failure_retries(self, clusters: List[dict]) -> int:
        if not self.curriculum_gen or not hasattr(self.curriculum_gen, "add_failure_for_retry"):
            return 0

        added = 0
        for cluster in clusters[:3]:
            for problem_id in cluster.get("sample_problems", []):
                graph = self._load_problem_graph(problem_id)
                if graph is None:
                    continue
                replay_problem = _FailureReplayProblem(problem_id=f"sleep_retry:{problem_id}", graph=graph)
                try:
                    self.curriculum_gen.add_failure_for_retry(replay_problem)
                    added += 1
                    break
                except Exception as exc:
                    log.debug("Hippocampus retry queue failed for %s: %s", problem_id, exc)
        return added

    def _sleep_cycle(self):
        """The actual offline consolidation work."""
        if not self.is_sleeping:
            return

        report = {
            "episodes_replayed": 0,
            "failure_clusters": 0,
            "retry_problems_added": 0,
        }

        with self.learning_lock:
            self.current_task = "Tuning Heuristics (GNN Phase)"
            try:
                from sare.heuristics.trainer import train_epoch
                if self.memory_manager and getattr(self.memory_manager, "_episodes", None):
                    time.sleep(2.0)
                    train_epoch(epochs=1)
                    self.episodes_replayed += len(self.memory_manager._episodes)
                    report["episodes_replayed"] = len(self.memory_manager._episodes)
            except ImportError:
                pass
            except Exception as exc:
                print(f"[Hippocampus] GNN Training Error: {exc}")

            if not self.is_sleeping:
                self.last_sleep_report = report
                return

            self.current_task = "Consolidating Memory (Rule Replay)"
            try:
                if self.memory_manager and self.experiment_runner:
                    episodes = self.memory_manager.recent_episodes(100)
                    replayed_count = 0
                    promoted_rules = getattr(self.experiment_runner, "_py_promoted_rules", {})
                    for ep in episodes:
                        if not ep.success:
                            continue
                        replayed_count += 1
                        for t_name in ep.transform_sequence:
                            if t_name in promoted_rules:
                                rule_obj = promoted_rules[t_name]
                                old_conf = getattr(rule_obj, "confidence", 0.80)
                                rule_obj.confidence = min(1.0, old_conf + 0.05)
                    if replayed_count > 0:
                        print(f"[Hippocampus] Replayed {replayed_count} episodes. Strengthened known rules.")
                time.sleep(1.0)
            except Exception as exc:
                print(f"[Hippocampus] Consolidation Error: {exc}")

            if not self.is_sleeping:
                self.last_sleep_report = report
                return

            self.current_task = "Mining Failure Patterns"
            try:
                failed_episodes = []
                if self.memory_manager:
                    failed_episodes = [ep for ep in self.memory_manager.recent_episodes(200) if not ep.success]
                clusters = self._cluster_failure_patterns(failed_episodes)
                report["failure_clusters"] = len(clusters)
                if clusters:
                    print(f"[Hippocampus] Mined {len(clusters)} real failure clusters from {len(failed_episodes)} failed episodes.")
            except Exception as exc:
                print(f"[Hippocampus] Failure Pattern Mining Error: {exc}")

            if not self.is_sleeping:
                self.last_sleep_report = report
                return

            self.current_task = "Synthesizing Curriculum"
            try:
                report["retry_problems_added"] = self._queue_failure_retries(self.failure_clusters)
                if report["retry_problems_added"]:
                    print(f"[Hippocampus] Queued {report['retry_problems_added']} curriculum retries from failure clusters.")
            except Exception as exc:
                print(f"[Hippocampus] Curriculum Synthesis Error: {exc}")

        self.last_sleep_report = report
        self.current_task = "Monitoring"
        self.is_sleeping = False

    def stop(self):
        self.running = False
        if self.is_alive():
            self.join(timeout=2.0)
