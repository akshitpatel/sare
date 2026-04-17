import json
import importlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from sare.brain import Brain, Event, EventData
import sare.brain as brain_module
from sare.curiosity.curriculum_generator import CurriculumGenerator
from sare.engine import Graph
from sare.learning.credit_assignment import CreditAssigner
from sare.learning.meta_curriculum import MetaCurriculumEngine
from sare.knowledge.knowledge_ingester import KnowledgeIngester
from sare.memory.hippocampus import HippocampusDaemon
from sare.memory.memory_manager import MemoryManager, SolveEpisode
from sare.perception.perception_bridge import PerceptionBridge
from sare.world.generative_world import GenerativeWorldModel, ImaginaryProblem


def test_graph_from_dict_preserves_ids_and_edges():
    graph = Graph.from_dict(
        {
            "nodes": [
                {"id": 10, "type": "operator", "label": "+"},
                {"id": 20, "type": "variable", "label": "x"},
                {"id": 30, "type": "constant", "label": "0"},
            ],
            "edges": [
                {"id": 7, "source": 10, "target": 20, "type": "left_operand"},
                {"id": 8, "source": 10, "target": 30, "type": "right_operand"},
            ],
        }
    )

    assert sorted(node.id for node in graph.nodes) == [10, 20, 30]
    assert graph.get_edge(7).source == 10
    assert graph.get_edge(8).target == 30
    assert graph._next_node_id == 31
    assert graph._next_edge_id == 9


def test_memory_manager_after_solve_updates_strategy_store(tmp_path):
    graph = Graph()
    root = graph.add_node("operator", "+")
    var = graph.add_node("variable", "x")
    zero = graph.add_node("constant", "0")
    graph.add_edge(root, var, "left_operand")
    graph.add_edge(root, zero, "right_operand")

    mm = MemoryManager(persist_dir=tmp_path)
    episode = SolveEpisode(
        problem_id="x_plus_0",
        transform_sequence=["add_zero_elim"],
        energy_trajectory=[5.0, 1.0],
        initial_energy=5.0,
        final_energy=1.0,
        success=True,
    )

    mm.after_solve(episode, graph)
    sig = mm._compute_signature(graph)

    assert sig in mm._strategies
    assert mm._strategies[sig]["transform_sequence"] == ["add_zero_elim"]
    assert mm._strategies[sig]["attempts"] == 1

    mm.after_solve(episode, graph)
    assert mm._strategies[sig]["attempts"] == 2
    assert mm._strategies[sig]["usage_count"] == 2


class _StubSolveEngine:
    def __init__(self, result=None, error: Exception | None = None):
        self._result = result
        self._error = error

    def solve(self, _expr):
        if self._error:
            raise self._error
        return self._result


class _MappingSolveEngine:
    def __init__(self, outcomes):
        self._outcomes = outcomes

    def solve(self, expr):
        return self._outcomes.get(expr, {"success": False, "energy": 1.0})


class _CaptureConceptGraph:
    def __init__(self):
        self.calls = []

    def ground_solve_episode(self, **kwargs):
        self.calls.append(kwargs)

    def concept_for_transform(self, _transform_name):
        return None


class _FailingWorkingMemory:
    def record_outcome(self, *args, **kwargs):
        raise RuntimeError("working memory offline")


class _FailingSelfModel:
    def observe(self, *args, **kwargs):
        raise RuntimeError("self model offline")


class _CaptureConceptMemory:
    def __init__(self):
        self.records = []

    def retrieve_similar(self, _graph, top_k=3):
        return [{
            "similarity": 0.91,
            "episode": {
                "problem_id": "prior:x + 0",
                "transforms": ["add_zero_elim"],
            },
        }]

    def record(self, graph, problem_id, transforms):
        self.records.append({
            "graph": graph.to_dict(),
            "problem_id": problem_id,
            "transforms": list(transforms),
        })

    def __len__(self):
        return len(self.records)


class _FailingRecallConceptMemory:
    def retrieve_similar(self, _graph, top_k=3):
        raise RuntimeError("recall unavailable")

    def record(self, graph, problem_id, transforms):
        return None

    def __len__(self):
        return 0


class _StubProblemGenerator:
    def generate_for_domain(self, domain, n=1):
        return [{"expression": f"{domain}_generated", "domain": domain}]

    def generate_batch(self, n=1, max_difficulty=0.7):
        return [{"expression": "novel_generated", "domain": "general"}]


class _StubGoal:
    def __init__(self, domain, description, priority=0.5, reason=""):
        self.domain = domain
        self.description = description
        self.priority = priority
        self.reason = reason or description
        self.status = "active"


class _StubGoalPlanner:
    def next_actionable(self):
        return _StubGoal("logic", "practice logic simplification")


class _StubGoalSetter:
    def suggest_next_goal(self):
        return _StubGoal("algebra", "follow algebra roadmap")


class _StubWorldModelV3:
    def __init__(self):
        self._solve_history = [{
            "expression": "failed_expr",
            "success": False,
            "domain": "arithmetic",
            "surprise": 0.8,
        }]

    def get_high_surprise_domains(self, n):
        return [("logic", 0.9)]


def test_meta_curriculum_transfer_uses_success_key():
    engine = MetaCurriculumEngine()
    engine.wire(engine=_StubSolveEngine({"success": True, "energy": 0.9}))

    result = engine.run_transfer_test("arithmetic", "algebra")

    assert result.solved is True
    assert engine._statuses["algebra"].transfer_tested == 1
    assert engine._statuses["algebra"].skill > 0.0


def test_meta_curriculum_transfer_failure_on_exception():
    engine = MetaCurriculumEngine()
    engine.wire(engine=_StubSolveEngine(error=RuntimeError("boom")))

    result = engine.run_transfer_test("algebra", "physics")

    assert result.solved is False
    assert engine._statuses["physics"].transfer_tested == 1
    assert engine._statuses["physics"].skill == 0.0


def test_generative_world_model_understands_success_and_energy_dict():
    world = GenerativeWorldModel()
    world.wire(engine=_StubSolveEngine({
        "success": True,
        "energy": {"total": 0.125},
        "result": "solved cleanly",
    }))

    result = world._attempt_solve(ImaginaryProblem("x + 0", "algebra", "template"))

    assert result == {
        "solved": True,
        "energy": 0.125,
        "result": "solved cleanly",
    }


def test_generative_world_model_without_engine_is_deterministic_failure():
    world = GenerativeWorldModel()

    result = world._attempt_solve(ImaginaryProblem("x + 0", "algebra", "template"))

    assert result == {
        "solved": False,
        "energy": 1.0,
        "result": "engine_unavailable",
    }


def test_generative_world_model_uses_benchmark_seeds_when_empty(tmp_path):
    logic_dir = tmp_path / "benchmarks" / "logic"
    logic_dir.mkdir(parents=True)
    (logic_dir / "smoke.json").write_text(
        json.dumps({"cases": [{"expression": "not not x"}]}),
        encoding="utf-8",
    )

    world = GenerativeWorldModel(repo_root=tmp_path)
    imagined = world.imagine(n=1)[0]

    assert world.summary()["benchmark_domains"] == ["logic"]
    assert imagined.domain == "logic"
    assert imagined.origin.startswith("benchmark_")


def test_generative_world_model_reports_benchmark_win_rate(tmp_path):
    algebra_dir = tmp_path / "benchmarks" / "algebra"
    logic_dir = tmp_path / "benchmarks" / "logic"
    algebra_dir.mkdir(parents=True)
    logic_dir.mkdir(parents=True)
    (algebra_dir / "symbolic_math.json").write_text(
        json.dumps([
            {"expression": "x + 0", "domain": "algebra"},
            {"expression": "x * 1", "domain": "algebra"},
        ]),
        encoding="utf-8",
    )
    (logic_dir / "smoke.json").write_text(
        json.dumps({"cases": [{"expression": "not not x", "domain": "logic"}]}),
        encoding="utf-8",
    )

    world = GenerativeWorldModel(repo_root=tmp_path)
    world.wire(engine=_MappingSolveEngine({
        "x + 0": {"success": True, "energy": 0.2},
        "x * 1": {"success": True, "energy": 0.2},
        "not not x": {"success": False, "energy": 1.0},
    }))

    report = world.evaluate_benchmarks(max_per_domain=3)

    assert report["attempted"] == 3
    assert report["solved"] == 2
    assert report["solve_rate"] == 0.667
    assert report["domains"]["algebra"]["solve_rate"] == 1.0
    assert report["domains"]["logic"]["solve_rate"] == 0.0
    assert world.summary()["last_benchmark_eval"]["solve_rate"] == 0.667


def test_brain_phase_gates_skip_memory_and_learning_boot():
    brain = Brain(config={"phases": {"memory": False, "curiosity": False}})

    brain._boot_memory()
    brain._boot_learning()

    assert brain.memory_manager is None
    assert brain.experiment_runner is None
    assert brain._module_status["memory_manager"].startswith("⏭️")
    assert brain._module_status["experiment_runner"].startswith("⏭️")


def test_brain_solve_returns_legacy_compatible_payload_and_records_concepts():
    brain = Brain(config={
        "brain_boot": {
            "memory": False,
            "metacognition": False,
            "knowledge": False,
            "world_model": False,
            "perception": False,
            "language": False,
            "social": False,
            "learning": False,
            "curriculum": False,
            "warmup_environment_discovery": False,
            "warmup_physics_session": False,
            "warmup_knowledge_ingestion": False,
            "warmup_transform_generator": False,
            "warmup_action_physics": False,
            "warmup_robustness_batch": False,
            "autostart_continuous_stream": False,
            "autostart_hippocampus": False,
            "seed_perception_priors": False,
        }
    })
    brain.boot()
    concept_memory = _CaptureConceptMemory()
    brain.concept_memory = concept_memory

    result = brain.solve("x + 0", force_python=True)

    assert result["solve_success"] is True
    assert result["transforms_applied"] == result["transforms"]
    assert result["steps_taken"] == result["steps"]
    assert result["initial"]["energy"]["total"] >= result["result"]["energy"]["total"]
    assert result["recalled_memory"][0]["problem_id"] == "prior:x + 0"
    assert concept_memory.records[0]["problem_id"] == "x + 0"
    assert concept_memory.records[0]["transforms"]


def test_brain_solve_records_recall_failures_in_diagnostics():
    brain = Brain(config={
        "brain_boot": {
            "memory": False,
            "metacognition": False,
            "knowledge": False,
            "world_model": False,
            "perception": False,
            "language": False,
            "social": False,
            "learning": False,
            "curriculum": False,
            "warmup_environment_discovery": False,
            "warmup_physics_session": False,
            "warmup_knowledge_ingestion": False,
            "warmup_transform_generator": False,
            "warmup_action_physics": False,
            "warmup_robustness_batch": False,
            "autostart_continuous_stream": False,
            "autostart_hippocampus": False,
            "seed_perception_priors": False,
        }
    })
    brain.boot()
    brain.concept_memory = _FailingRecallConceptMemory()

    result = brain.solve("x + 0", force_python=True)

    assert result["solve_success"] is True
    assert brain._stats["runtime_errors"] >= 1
    assert any(entry["component"] == "concept_memory.retrieve_similar" for entry in brain._diagnostics)


def test_brain_problem_selection_prefers_recent_failures_and_records_reason():
    brain = Brain()
    brain._problem_gen = _StubProblemGenerator()
    brain.world_model_v3 = _StubWorldModelV3()
    brain.goal_planner = _StubGoalPlanner()
    brain.goal_setter = _StubGoalSetter()

    picked = brain._pick_learning_problem()

    assert picked == "failed_expr"
    assert brain._last_problem_selection["source"] == "failure_replay"
    assert brain._last_problem_selection["candidates_considered"] >= 3
    assert "failed problem" in brain._last_problem_selection["reason"]


def test_web_module_keeps_legacy_runtime_lazy_on_import():
    import sare.interface.web as web

    web = importlib.reload(web)

    assert web._legacy_runtime_ready is False
    assert web.memory_manager is None
    assert web._path_requires_legacy_runtime("/api/solve") is True
    assert web._path_requires_legacy_runtime("/api/brain/status") is False


def test_brain_solve_writes_log_entry(tmp_path):
    old_log_path = brain_module.SOLVE_LOG_PATH
    brain_module.SOLVE_LOG_PATH = tmp_path / "solves.jsonl"
    try:
        brain = Brain(config={
            "brain_boot": {
                "memory": False,
                "metacognition": False,
                "knowledge": False,
                "world_model": False,
                "perception": False,
                "language": False,
                "social": False,
                "learning": False,
                "curriculum": False,
                "warmup_environment_discovery": False,
                "warmup_physics_session": False,
                "warmup_knowledge_ingestion": False,
                "warmup_transform_generator": False,
                "warmup_action_physics": False,
                "warmup_robustness_batch": False,
                "autostart_continuous_stream": False,
                "autostart_hippocampus": False,
                "seed_perception_priors": False,
            }
        })
        brain.boot()

        result = brain.solve("x + 0", force_python=True)

        assert result["solve_success"] is True
        rows = [json.loads(line) for line in brain_module.SOLVE_LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 1
        assert rows[0]["problem_id"] == "x + 0"
        assert rows[0]["solve_success"] is True
        assert rows[0]["modules_activated"] == ["brain"]
    finally:
        brain_module.SOLVE_LOG_PATH = old_log_path


def test_brain_status_exposes_last_problem_selection():
    brain = Brain()
    brain._problem_gen = _StubProblemGenerator()
    brain.world_model_v3 = _StubWorldModelV3()
    brain.goal_planner = _StubGoalPlanner()
    brain.goal_setter = _StubGoalSetter()

    picked = brain._pick_learning_problem()
    status = brain.status()

    assert picked == "failed_expr"
    assert status["last_problem_selection"]["expression"] == "failed_expr"
    assert status["last_problem_selection"]["source"] == "failure_replay"


def test_brain_solve_completed_grounds_concepts_from_event_data():
    brain = Brain()
    concept_graph = _CaptureConceptGraph()
    brain.concept_graph = concept_graph

    graph = Graph()
    root = graph.add_node("operator", "+")
    var = graph.add_node("variable", "x")
    zero = graph.add_node("constant", "0")
    graph.add_edge(root, var, "left_operand")
    graph.add_edge(root, zero, "right_operand")

    brain._on_solve_completed(EventData(
        event=Event.SOLVE_COMPLETED,
        data={
            "problem_id": "x + 0",
            "expression": "x + 0",
            "transforms": ["add_zero_elim"],
            "energy_before": 4.0,
            "energy_after": 1.0,
            "domain": "arithmetic",
            "final_graph": graph,
        },
    ))

    assert len(concept_graph.calls) == 1
    assert concept_graph.calls[0]["expression"] == "x + 0"
    assert concept_graph.calls[0]["result"]


def test_hippocampus_clusters_real_failure_patterns():
    def make_graph(op_label, const_label, var_label):
        graph = Graph()
        root = graph.add_node("operator", op_label)
        var = graph.add_node("variable", var_label)
        const = graph.add_node("constant", const_label)
        graph.add_edge(root, var, "left_operand")
        graph.add_edge(root, const, "right_operand")
        return graph

    problem_graphs = {
        "x + 0": make_graph("+", "0", "x"),
        "y + 0": make_graph("+", "0", "y"),
        "a * 1": make_graph("*", "1", "a"),
    }
    episodes = [
        SolveEpisode(problem_id="x + 0", success=False, initial_energy=4.0, final_energy=4.0),
        SolveEpisode(problem_id="y + 0", success=False, initial_energy=5.0, final_energy=5.0),
        SolveEpisode(problem_id="a * 1", success=False, initial_energy=3.0, final_energy=3.0),
    ]

    daemon = HippocampusDaemon(problem_loader=lambda problem_id: problem_graphs.get(problem_id))
    clusters = daemon._cluster_failure_patterns(episodes)

    assert len(clusters) == 2
    assert clusters[0]["count"] == 2
    assert set(clusters[0]["sample_problems"]) == {"x + 0", "y + 0"}


def test_brain_records_runtime_diagnostics_for_post_solve_failures():
    brain = Brain()
    brain.working_memory = _FailingWorkingMemory()

    brain._on_solve_completed(EventData(
        event=Event.SOLVE_COMPLETED,
        data={
            "problem_id": "x + 0",
            "expression": "x + 0",
            "transforms": ["add_zero_elim"],
            "energy_before": 4.0,
            "energy_after": 1.0,
            "domain": "arithmetic",
        },
    ))

    assert brain._stats["runtime_errors"] >= 1
    assert any(entry["component"] == "working_memory.record_outcome" for entry in brain._diagnostics)


def test_brain_records_runtime_diagnostics_for_self_model_failures():
    brain = Brain()
    brain.self_model = _FailingSelfModel()

    brain._on_solve_completed(EventData(
        event=Event.SOLVE_COMPLETED,
        data={
            "problem_id": "x + 0",
            "expression": "x + 0",
            "transforms": ["add_zero_elim"],
            "energy_before": 4.0,
            "energy_after": 1.0,
            "domain": "arithmetic",
        },
    ))

    assert brain._stats["runtime_errors"] >= 1
    assert any(entry["component"] == "self_model.observe" for entry in brain._diagnostics)


def test_knowledge_ingester_reads_local_corpus(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.md").write_text(
        "# Sample Algebra\nIdentity element means x + 0 = x and x * 1 = x.\n",
        encoding="utf-8",
    )
    benchmarks_dir = tmp_path / "benchmarks" / "algebra"
    benchmarks_dir.mkdir(parents=True)
    (benchmarks_dir / "sample.json").write_text(
        '[{"expression":"x + 0","expected_result":"x","category":"additive_identity"}]',
        encoding="utf-8",
    )

    ingester = KnowledgeIngester()
    extracted = ingester.ingest_local_corpus(repo_root=tmp_path)

    assert extracted > 0
    assert ingester.summary()["corpus_files_ingested"] == 2


def test_knowledge_ingester_tracks_recursive_provenance_and_merges_sources(tmp_path):
    nested_docs = tmp_path / "docs" / "math"
    nested_docs.mkdir(parents=True)
    (nested_docs / "lesson.md").write_text(
        "# Logic Lesson\nDouble negation means not not p = p.\n",
        encoding="utf-8",
    )

    ingester = KnowledgeIngester()
    ingester.ingest_text(
        "Identity A",
        "x + 0 = x.",
        domain="arithmetic",
        concept_hints=["identity_addition"],
        source_path="docs/a.md",
    )
    ingester.ingest_text(
        "Identity B",
        "x + 0 = x and x * 1 = x.",
        domain="arithmetic",
        concept_hints=["identity_addition"],
        source_path="docs/math/b.md",
    )
    ingester.ingest_local_corpus(repo_root=tmp_path)

    concept = ingester._extracted["identity_addition"]
    summary = ingester.summary()

    assert concept.source_path == "docs/a.md"
    assert sorted(concept.source_refs) == ["docs/a.md", "docs/math/b.md"]
    assert summary["unique_sources"] >= 3
    assert any(item["source_path"] == "docs/a.md" for item in summary["recent_concepts"])


def test_perception_bridge_merges_descriptors_into_objects():
    bridge = PerceptionBridge()

    scene = bridge.parse_scene("red block left of door")

    assert any(obj.name == "red_block" for obj in scene.objects)
    assert not any(obj.name == "red" for obj in scene.objects)
    assert scene.relations[0].subject == "red_block"


def test_perception_bridge_parses_state_transitions():
    bridge = PerceptionBridge()

    transition = bridge.parse_transition(
        "red block left of door",
        "move block",
        "red block right of door",
    )

    assert transition.action == "move block"
    assert transition.added_relations == ["right_of(red_block,door)"]
    assert transition.removed_relations == ["left_of(red_block,door)"]
    assert transition.added_objects == []
    assert transition.removed_objects == []
    assert bridge.summary()["recent_transitions"][-1]["action"] == "move block"


def test_credit_assigner_save_and_load_round_trip(tmp_path):
    assigner = CreditAssigner()
    assigner.assign_credit(["add_zero_elim"], [3.0, 1.0])

    path = tmp_path / "credit_assigner.json"
    assigner.save(path)

    restored = CreditAssigner()
    restored.load(path)

    assert restored.get_utility("add_zero_elim") == assigner.get_utility("add_zero_elim")
    assert restored.baseline == assigner.baseline
    assert restored.baseline_count == assigner.baseline_count


def test_curriculum_generator_save_and_load_round_trip(tmp_path):
    graph = Graph()
    root = graph.add_node("operator", "+")
    var = graph.add_node("variable", "x")
    zero = graph.add_node("constant", "0")
    graph.add_edge(root, var, "left_operand")
    graph.add_edge(root, zero, "right_operand")

    generator = CurriculumGenerator()
    generator.add_seed(graph)
    generated = generator.generate_batch(size=1)
    path = tmp_path / "curriculum.json"

    generator.save(path)

    restored = CurriculumGenerator()
    restored.load(path)

    assert len(restored.seed_problems) == 1
    assert len(restored.generated_problems) == len(generated)
    assert restored.generated_problems[0].origin == "mutated_seed"
    assert restored.generated_problems[0].graph.to_dict() == generated[0].graph.to_dict()
