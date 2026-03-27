# SARE-HX: Complete Architecture & Module Connectivity Map

## High-Level Layer Stack

```
┌─────────────────────────────────────────────────────────────────┐
│  INTERFACES          web.py  cli.py  brain.py                   │
├─────────────────────────────────────────────────────────────────┤
│  META / EVOLUTION    self_improver  evolver_daemon  bottleneck  │
├─────────────────────────────────────────────────────────────────┤
│  LEARNING            transform_synthesizer  credit_assign       │
├─────────────────────────────────────────────────────────────────┤
│  CURIOSITY           experiment_runner  curriculum_generator    │
├─────────────────────────────────────────────────────────────────┤
│  MEMORY              world_model  autobiographical  homeostasis │
├─────────────────────────────────────────────────────────────────┤
│  REASONING           causal.induction  analogy_transfer  ToM   │
├─────────────────────────────────────────────────────────────────┤
│  PERCEPTION          graph_builders  multimodal  nl_parser     │
├─────────────────────────────────────────────────────────────────┤
│  SEARCH / HEURISTIC  beam_search  attention_beam  mlx_value_net│
├─────────────────────────────────────────────────────────────────┤
│  CORE ENGINE         engine.py  transforms  C++ bindings       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer 0 — Core Engine

### `python/sare/engine.py`
**Purpose**: Pure-Python graph, transforms, energy evaluator, search algorithms. Zero external dependencies.

| Export | Type | Role |
|--------|------|------|
| `Graph` | class | Directed node-edge graph; `.nodes`, `.edges`, `.add_node()`, `.add_edge()` |
| `Node` | dataclass | id, label, value, node_type |
| `Edge` | dataclass | src, dst, relation |
| `Transform` | ABC | `.name`, `.apply(graph) → Graph\|None` |
| `EnergyEvaluator` | class | Complexity score (node count, depth, operator weight) |
| `BeamSearch` | class | Beam search over transform applications |
| `MCTSSearch` | class | Monte Carlo Tree Search variant |
| `load_problem()` | fn | Expression string → Graph |
| `ALL_TRANSFORMS` | list | 18 built-in transforms |

**Imported by**: everything — web.py, experiment_runner, curriculum_generator, all transform modules, all search modules, perception/graph_builders.

**Imports**: stdlib only (json, math, time, random, dataclasses, typing, pathlib).

### `core/sare_bindings.so` (C++ via pybind11)
**Purpose**: High-performance C++ implementations bridged to Python.

| C++ Export | Python Usage |
|-----------|--------------|
| `CppGraph` | memory_manager, curriculum_generator |
| `SearchConfig` | experiment_runner (fast beam params) |
| `ReflectionEngine` | abstraction_learning |
| `ConceptRegistry` | concept_rule, experiment_runner |
| `GraphEmbedder` | heuristic_model |

**Bridge**: `core/graph_bridge.py` — `GraphBridge` converts `engine.Graph ↔ CppGraph`.

### `transforms/code_transforms.py`
Exports: `CodeConstFoldTransform`, `InlineIdentityTransform`, `DeadCodeElimTransform`, `ReturnSimplifyTransform`
Imports: `engine.Transform`, `engine.Graph`, `engine.Node`

### `transforms/logic_transforms.py`
Exports: `ModusPonens`, `ModusTollens`, `Transitivity`, `NegationElim`, `UniversalInstantiation`
Imports: `engine.Transform`, `engine.Graph`, `engine.Node`

---

## Layer 1 — Search & Heuristics

### `search/attention_beam.py`
- **Exports**: `AttentionBeamScorer`
- **Role**: Re-scores beam candidates by uncertainty + novelty to avoid premature convergence
- **Imports**: `engine.Graph`
- **Wired**: passed as `attention_scorer=` kwarg to `BeamSearch` in `experiment_runner.py:615`

### `search/transform_predictor.py`
- **Role**: Predicts which transform to try next given current graph state
- **Feeds**: `experiment_runner` reorders transform list using world model prediction

### `heuristics/mlx_value_net.py`
- **Exports**: `MLXValueNet`
- **Role**: Apple M1/M2 Metal GPU value network (~8x faster than PyTorch MPS)
- **Imports**: `mlx` library (Apple-only)
- **Used by**: `BeamSearch` for state evaluation

### `heuristics/graph_embedding.py`
- **Exports**: `GraphEmbedding` (128-dim feature vector from graph)
- **Used by**: `heuristic_model.py`, `mlx_value_net.py`

### `heuristics/heuristic_model.py`
- **Exports**: `HeuristicModel`, `HeuristicLoss`
- **Imports**: `heuristics.graph_embedding.GraphEmbedding`
- **Role**: Learned value function; trained online via `credit_assignment`

---

## Layer 2 — Perception & Parsing

### `interface/nl_parser_v2.py`
- **Exports**: `EnhancedNLParser`, `EnhancedParseResult` (has `.expression`)
- **Role**: Dependency parsing, operator precedence, multi-domain intent detection
- **Used by**: `web.py` (POST /api/solve natural language input)

### `interface/universal_parser.py`
- **Role**: Unified parser facade — dispatches to nl_parser_v2 or direct expression parser

### `perception/graph_builders.py`
- **Exports**: `CodeGraphBuilder`, `SentenceGraphBuilder`, `PlanGraphBuilder`
- **Imports**: `engine.Graph`, `engine.Node`, `engine.Edge`, `engine.build_expression_graph`
- **Role**: Converts code AST / sentences / plans into SARE graphs for symbolic reasoning

### `perception/multimodal.py`
- **Role**: Image / table / text → graph (Phase 5); calls `graph_builders` internally

---

## Layer 3 — Memory (20 modules)

### `memory/memory_manager.py`
- **Exports**: `MemoryManager`
- **Imports**: optional `sare_bindings` (C++ EpisodicStore, StrategyMemory, GraphSignature)
- **Role**: Persistent memory bridge; `before_solve()` warm-start, `after_solve()` record
- **Used by**: `experiment_runner`

### `memory/world_model.py`
- **Exports**: `WorldModel`, `Schema`, `CausalLink`, `Belief`
- **Imports**: `interface.llm_bridge` (lazy, background thread when surprise > 2.0)
- **Role**: Belief store, causal link graph, contradiction detection, counterfactual reasoning
- **Files**: `data/memory/world_model_v2.json`, `data/memory/world_hypotheses.json`
- **Used by**: `experiment_runner` (predicts best transform), `curriculum_generator` (ZPD adjustment)

### `memory/world_model_v3.py`
- **Imports**: `memory.world_model` (Schema, CausalLink)
- **Role**: Extended v3 with structural signatures, decay, self-learning fields

### `memory/autobiographical.py`
- **Exports**: `AutobiographicalMemory`, `LearningEpisode`, `NarrativeThread`
- **File**: `data/memory/autobiographical.json`
- **Role**: Tracks solve events + emotional valence; generates self-narrative
- **Used by**: `curriculum_generator` (retries failure episodes first), `dialogue_manager`

### `memory/identity.py`
- **Exports**: `IdentityManager`, `Trait`
- **File**: `data/memory/identity.json`
- **Role**: Persistent personality traits (curious, precise, persistent, collaborative)

### `memory/homeostasis.py` (also at `meta/homeostasis.py`)
- **Exports**: `HomeostaticSystem`, `Drive`
- **File**: `data/memory/homeostasis.json`
- **Drives**: curiosity, mastery, social, exploration, consolidation
- **Used by**: `learn_daemon.py` (adjusts batch size and mode), `experiment_runner`

### `memory/attention.py`
- **Exports**: `AttentionSelector`, `WorkingMemoryWindow`
- **Imports**: `engine.Graph`, `engine.Node`
- **Role**: Selects top-K nodes for focused search (activates on graphs > 15 nodes)
- **Used by**: `experiment_runner` (passed to search)

### `memory/hippocampus.py`
- **Exports**: `HippocampusDaemon`
- **Role**: Background sleep-mode consolidation; offline replay of episodes

### `memory/forgetting_curve.py`
- **Exports**: `ForgettingCurve`, `LeitnerBox`
- **Role**: Spaced repetition; API `/api/memory/forgetting`

### `memory/concept_rule.py`
- **Exports**: `ConceptRule` (wraps AbstractRule as Transform)
- **Imports**: `engine.Transform`, `engine.Graph`, `engine.Node`
- **Role**: Learned rules promoted from ConceptRegistry become Transform objects

### `memory/program_synthesizer.py`
- **Exports**: `ProgramSynthesizer`
- **Imports**: `interface.llm_bridge._call_llm`, `engine.Graph`
- **Role**: LLM-based program synthesis for stuck problems

### Other memory modules

| Module | Role |
|--------|------|
| `concept_formation.py` | Concept clustering from recurring graph patterns |
| `knowledge_graph.py` | Persistent symbolic knowledge base (KB) |
| `working_memory.py` | Short-term scratchpad for active solve state |
| `global_workspace.py` | Global broadcast bus (GWT-inspired) |
| `global_buffer.py` | Ring buffer for recent events |
| `attention_router.py` | Multi-source attention routing across memory modules |

---

## Layer 4 — Reasoning

### `causal/induction.py`
- **Exports**: `CausalInduction`, `InductionResult`
- **Imports**: `causal.knowledge_base.get_ckb()`
- **Role**: Tests candidate rules on 3-5 problems before promotion (Wall 3 fix)
- **Used by**: `experiment_runner` after reflection generates novel rules

### `causal/analogy_transfer.py`
- **Exports**: `AnalogyTransfer`
- **Imports**: `causal.knowledge_base`, `curiosity.curriculum_generator`
- **Role**: Structural analogy matching; transfers rules arithmetic → algebra → logic
- **Fires**: after rule promotion; persisted to `data/memory/learned_transfers.json`

### `causal/knowledge_base.py`
- **Exports**: `get_ckb()` (singleton), `CausalKnowledgeBase`
- **Role**: Domain-specific causal mechanisms; used by induction + transfer

### `social/theory_of_mind.py`
- **Exports**: `TheoryOfMindEngine`, `BeliefGraph`, `Proposition`
- **Role**: Models beliefs/desires/intentions of other agents; `false_belief_test()`
- **Benchmark**: `benchmarks/social/false_belief.json` (3/3 pass)

### `social/dialogue_manager.py`
- **Exports**: `DialogueManager`, `DialogueTurn`
- **Role**: teach/ask/correct intents; symbolic rule extraction; autobiographical recording
- **Used by**: `web.py` POST /api/chat

### `knowledge/commonsense.py`
- **Exports**: `CommonsenseKB`, `get_domain_hints()`
- **Role**: 60+ seed facts (IsA, HasA, PartOf, Causes, UsedFor, etc.)
- **Used by**: `curriculum_generator`, `experiment_runner`

---

## Layer 5 — Curiosity & Curriculum

### `curiosity/experiment_runner.py` ← Central Solve Loop
- **Exports**: `ExperimentRunner`, `ExperimentResult`
- **Imports**:
  - `engine` (Graph, BeamSearch, transforms)
  - `curriculum_generator` (GeneratedProblem)
  - `memory_manager` (warm-start / record)
  - `learning.abstraction_learning` (mine patterns)
  - `causal.induction` (validate rules)
  - `causal.analogy_transfer` (cross-domain)
  - `memory.attention` (AttentionSelector)
  - `search.attention_beam` (AttentionBeamScorer)
  - `meta.credit_assignment` (reward)
  - `meta.homeostasis` (behavior mode)
  - `memory.world_model` (predict transform + observe)
  - `reflection.py_reflection` (batch oracle)
  - `learning.llm_transform_synthesizer` (stuck → LLM writes transform)
- **Flow**: `run_batch()` → pick problem → solve → reflect → induce → learn → update world model

### `curiosity/curriculum_generator.py`
- **Exports**: `CurriculumGenerator`, `GeneratedProblem`
- **Imports**: optional `sare_bindings` (C++ Graph), `memory`, `commonsense`
- **Role**: Per-domain Zone of Proximal Development; generates problems just beyond current ability
- **Reads**: world model surprise + autobiographical failures

### `curiosity/frontier_manager.py`
- **Exports**: `FrontierManager`
- **File**: `data/memory/frontier.jsonl`
- **Role**: Tracks solved/unsolved/pending; competence boundary map

### `curiosity/multi_task_scheduler.py`
- **Exports**: `MultiTaskScheduler`
- **File**: `data/memory/multi_task_scheduler.json`
- **Role**: Allocates solve budget across task types by ZPD; API `/api/multitask/scheduler`

---

## Layer 6 — Learning

### `learning/llm_transform_synthesizer.py` ← Wall 1 Fix
- **Exports**: `TransformSynthesizer`
- **Imports**: `interface.llm_bridge`, `engine.Transform`
- **Role**: When stuck on N problems, LLM writes a new Transform class; validated + loaded dynamically
- **Output**: `data/memory/synthesized_modules/`

### `learning/abstraction_learning.py`
- **Exports**: `mine_frequent_patterns()`, `propose_macros()`
- **Imports**: `meta.macro_registry.MacroSpec`
- **Role**: Identifies recurring transform sequences → candidate macros

### `learning/credit_assignment.py`
- **Exports**: `CreditAssigner`, `CreditResult`
- **Role**: Per-transform reward propagation; updates utility scores for BeamSearch ordering

### `learning/teacher_protocol.py`
- **Exports**: `TeacherProtocol`, `ConfusionDetector`, `TeacherRegistry`, `LLMTeacher`
- **Imports**: `interface.llm_bridge`, `memory.concept_rule`
- **Role**: Auto-generates confusion questions → background LLM auto-answers → promotes rules
- **Flow**: confusion detected → `_auto_answer_via_llm()` (background thread) → `answer_question()` → `_promote_rules()`

---

## Layer 7 — Meta / Evolution

### `meta/self_improver.py` ← Code Self-Modification
- **Exports**: `SelfImprover`
- **Imports**: `interface.llm_bridge` (4-model debate: proposer/planner/critic/judge)
- **Role**: Reads Python source → proposes improvement → validates import → replaces file
- **Rate-limited**: `llm_bridge._rate_limit_wait()` (3s min gap); exponential backoff on 429

### `meta/evolution_monitor.py`
- **Exports**: `EvolutionMonitor`, `SubsystemHealth`
- **Role**: AGI velocity tracking across 6 subsystems; API `/api/agi/evolution`

### `meta/bottleneck_analyzer.py`
- **Exports**: `BottleneckAnalyzer`, `ImprovementTarget`
- **Role**: Scores Python files by urgency; API `/api/bottleneck`

### `meta/goal_setter.py`
- **Exports**: `GoalSetter`, `Goal`, `GoalStatus`
- **File**: `data/memory/active_goals.json`
- **Role**: Maintains goal stack; auto-generates sub-goals from weak domains

### `meta/proof_builder.py`
- **Exports**: `ProofBuilder`, `format_proof()`, `proof_to_nl()`
- **Role**: Transform sequence → natural language explanation

### `meta/architecture_designer.py`
- **Role**: Gap analysis + architectural proposals; API `/api/architecture/gaps` + `/api/architecture/proposals`

### `meta/algorithm_selector.py`
- **Role**: Epsilon-greedy strategy selection per task type; API `/api/algorithm/selector`

### `meta/macro_registry.py`
- **Exports**: `MacroSpec`, `list_macros()`, `upsert_macros()`
- **Role**: Stores frequent transform sequences as named macros

---

## Layer 8 — LLM Bridge (Cross-Cutting)

### `interface/llm_bridge.py` ← Central LLM Hub
- **Exports**: `_call_llm()`, `_call_model()`, `llm_available()`
- **Config**: `configs/llm.json`
- **Providers**: OpenRouter (primary) → deepseek/deepseek-v3.2 fallback → Ollama (offline)
- **Rate limiting**: 3s min gap (`_rate_limit_wait()`), exponential backoff on 429 (15→30→60→120s, 4 retries)

**Current model roster**:

| Role | Model |
|------|-------|
| Main | `x-ai/grok-4.1-fast` |
| Synthesis | `minimax/minimax-m2.7:free` |
| Fast | `minimax/minimax-m2.7` |
| Alt proposer | `google/gemini-3.1-flash-lite-preview` |
| Critic | `anthropic/claude-sonnet-4.6` |
| Fallback | `deepseek/deepseek-v3.2` |

**Imported by**: world_model, synthesizer, dialogue_manager, teacher_protocol, self_improver, program_synthesizer, py_reflection

---

## Layer 9 — Interface / API

### `interface/web.py` ← API Gateway (~3000 lines)
- **Server**: `ThreadedHTTPServer` (BaseHTTPRequestHandler, no frameworks)
- **Lazy-loads**: all subsystems on first request

| Endpoint | Subsystem |
|----------|-----------|
| GET /api/health | 7 subsystems |
| GET /api/brain/stage | developmental_curriculum |
| GET /api/brain/piaget | developmental_curriculum |
| GET /api/predictive/status | world_model |
| GET /api/grammar/status | internal_grammar |
| GET /api/memory/forgetting | forgetting_curve |
| GET /api/architecture/gaps | architecture_designer |
| GET /api/questions/pending | teacher_protocol |
| GET /api/identity | autobiographical + identity |
| GET /api/homeostasis | homeostasis |
| GET /api/curriculum/stage | developmental_curriculum |
| GET /api/world | world_model |
| GET /api/transfer | analogy_transfer |
| GET /api/agi/evolution | evolution_monitor |
| GET /api/evolver/daemon | evolver_daemon.json |
| GET /api/evolve/logs/stream | SSE live log tail |
| GET /api/self-improve/status | self_improver |
| POST /api/chat | dialogue_manager |
| POST /api/qa | qa_pipeline |
| GET /favicon.ico | 1×1 ICO binary |

---

## Data Flows

### Solve Loop (learn_daemon → results on disk)

```
learn_daemon.py
  └─► ExperimentRunner.run_batch()
        ├─ CurriculumGenerator.next_problem()     ← ZPD + autobiographical + world model surprise
        ├─ MemoryManager.before_solve()           ← warm-start episodic store
        ├─ AttentionSelector.select_window()      ← focus on high-info nodes (>15 nodes)
        ├─ WorldModel.predict_transform()         ← reorder transform list
        ├─ BeamSearch.run(attention_scorer=…)     ← AttentionBeamScorer + MLXValueNet
        ├─ MemoryManager.after_solve()            ← record episode
        ├─ CreditAssigner.assign()                ← per-transform reward
        ├─ PyReflection.reflect()                 ← enqueue rule (flush at 10 → 1 LLM call)
        ├─ CausalInduction.test_candidate()       ← validate on 3-5 problems
        ├─ AnalogyTransfer.sweep_all_domains()    ← cross-domain rule transfer
        ├─ WorldModel.observe_solve()             ← update causal links + beliefs
        └─ (stuck N times) TransformSynthesizer   ← LLM writes new Transform class
```

### Evolver Loop (evolver_daemon → modified source files)

```
evolver_daemon.py
  └─► SelfImprover.run_cycle()
        ├─ BottleneckAnalyzer.score_files()
        ├─ LLM Proposer (grok-4.1-fast)     → propose code change
        ├─ LLM Planner  (gpt-5.4-nano)      → implementation plan
        ├─ LLM Critic   (claude-sonnet-4.6) → review + score
        ├─ LLM Judge    (gemini-3-flash)    → accept/reject
        ├─ (accepted) write file + import test
        └─ HomeostaticSystem                → pause in consolidation mode
```

---

## Critical State Files

| File | Owner | Purpose |
|------|-------|---------|
| `data/memory/world_model_v2.json` | WorldModel | Beliefs + causal links |
| `data/memory/autobiographical.json` | AutobiographicalMemory | Episode history |
| `data/memory/identity.json` | IdentityManager | Personality traits |
| `data/memory/homeostasis.json` | HomeostaticSystem | Drive states |
| `data/memory/frontier.jsonl` | FrontierManager | Competence boundary |
| `data/memory/synthesized_modules/` | TransformSynthesizer | LLM-generated transforms |
| `data/memory/learned_transfers.json` | AnalogyTransfer | Cross-domain rules |
| `data/memory/self_model.json` | SelfModel | Competence estimates |
| `data/memory/evolver_daemon.json` | EvolverDaemon | Run history + spend |
| `configs/llm.json` | LLMBridge | Model + API key config |

---

## Entry Points

| Process | Command | Key Modules |
|---------|---------|-------------|
| Web UI | `python3 -m sare.interface.web --port 8080` | web.py → all subsystems lazy |
| Learn Daemon | `python3 learn_daemon.py` | experiment_runner → curriculum_generator → memory |
| Evolver Daemon | `python3 evolver_daemon.py` | self_improver → llm_bridge → bottleneck_analyzer |
| CLI | `python3 -m sare.interface.cli solve "expr"` | engine → search |
| Full | `./run.sh all` | web + daemon + evolver |
