"""
SARE-HX Project Guide — Injected into every self-improvement LLM call.

This module provides a single source of truth about the ENTIRE codebase
so that the evolve / self-improver has full project context when debating
improvements to any individual file.

Usage:
    from sare.meta.project_guide import FULL_PROJECT_GUIDE
    # Then inject into any LLM system-prompt alongside the target file.
"""

# ──────────────────────────────────────────────────────────────────────────────
# FULL PROJECT GUIDE  (keep this accurate — update when modules change)
# ──────────────────────────────────────────────────────────────────────────────

FULL_PROJECT_GUIDE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║         SARE-HX — FULL PROJECT ARCHITECTURE GUIDE FOR EVOLVE MODULE        ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 0 — WHAT THIS SYSTEM IS (READ FIRST)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SARE-HX is NOT a symbolic theorem prover or a chatbot wrapper.
It is a computational model of the HUMAN MIND — every module maps to a known
brain region or cognitive faculty. When improving any file, always think:
"What brain region does this correspond to, and is my change consistent with
how that region works in humans?"

The goal is AGI — a system that can learn ANY domain autonomously, form its
own goals, understand other agents, and continuously improve itself.

BRAIN REGION → SARE-HX MODULE MAPPING:
  Prefrontal Cortex (planning, executive control)  → meta/goal_setter.py, meta/self_model.py
  Hippocampus (episodic memory, consolidation)     → memory/hippocampus.py, learning/dream_consolidator.py
  Neocortex (long-term knowledge)                  → memory/concept_formation.py, memory/world_model.py
  Working Memory / Attention                        → memory/working_memory.py, memory/global_workspace.py
  Default Mode Network (creativity, daydreaming)   → neuro/creativity_engine.py, world/generative_world.py
  Dopaminergic System (reward, motivation)         → neuro/dopamine.py, meta/homeostasis.py
  Language Areas (Broca, Wernicke)                 → interface/nl_parser_v2.py, interface/universal_parser.py
  Visual Cortex / Perception                        → perception/graph_builders.py, perception/perception_engine.py
  Cerebellum (motor learning, procedure memory)    → heuristics/graph_embedding.py, heuristics/mlx_value_net.py
  Amygdala (emotional valence)                     → energy/affective_energy.py, neuro/dopamine.py
  Social Brain (TPJ, mirror neurons)               → social/theory_of_mind.py, social/dialogue_manager.py
  Thalamus (routing, broadcast)                    → memory/global_workspace.py, memory/attention_router.py
  Basal Ganglia (habit, rule selection)            → memory/concept_rule.py, learning/credit_assignment.py
  Inductive Reasoning (prefrontal-parietal loop)   → causal/induction.py, reflection/py_reflection.py
  Metacognition (default mode + prefrontal)        → meta/inner_monologue.py, meta/self_model.py
  Long-term Potentiation (learning rule)           → curiosity/experiment_runner.py, memory/memory_manager.py

CRITICAL LOOPS (never break these):
  1. PERCEPTION → REASONING: Any input → parse → graph → beam search → proof
  2. SOLVING → LEARNING: solve → reflect → induct → promote → registry → next solve
  3. LEARNING → CURRICULUM: promotion → world model surprise → curriculum ZPD selection
  4. CURRICULUM → MOTIVATION: homeostasis drives → batch size → beam width → solve mode
  5. SELF-IMPROVEMENT → CAPABILITY: self_improver reads source → debates → patches → reload


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 1 — TOP-LEVEL & ENTRY POINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

learn_daemon.py  (repo root)
  The autonomous learning process. Runs 24/7 in the background.
  Key logic:
    - Instantiates CurriculumGenerator + ExperimentRunner + homeostasis
    - Main loop: homeostasis.tick() → run_batch() → on_batch_completed()
    - Every 5 cycles: question generation, world model save
    - Every 10 cycles: DreamConsolidator.dream_cycle() → on_sleep_cycle()
    - Responds to homeostasis: consolidate=small batch, explore=large batch
  Do NOT break: the DreamConsolidator singleton, the homeostasis wiring.

python/sare/brain.py  (Brain orchestrator)
  Central nervous system. Manages module lifecycle and the event bus.
  Key classes: Brain, DevelopmentalStage (infant/child/student/researcher)
  Key methods: boot(), solve(expr), learn_cycle(n), shutdown()
  Fires typed events: SOLVE_COMPLETED, RULE_DISCOVERED, DOMAIN_MASTERED
  Wires: ExperimentRunner ↔ HomeostasisSystem ↔ AutobiographicalMemory
  The Brain is the primary integration point — web.py calls Brain endpoints.

python/sare/self_learner.py
  Thin wrapper that boots the Brain and runs learn_daemon in-process.
  Used when running the system headlessly without the web server.

python/sare/engine.py  (The reasoning core — ~1500 lines)
  Implements the reasoning substrate: graphs, energy, transforms, search.
  Key classes:
    Node(id, type, label, attributes, uncertainty) — atomic thought unit
    Edge(source, target, type, weight) — relationship between nodes
    Graph — typed directed graph; .nodes, .edges, .add_node(), .add_edge()
    EnergyBreakdown(syntax, complexity, redundancy, uncertainty, domain_aware)
    EnergyEvaluator — compute(graph) → EnergyBreakdown
    Transform (abstract) — name(), apply(graph) → Graph|None
    BeamSearch — search(graph, energy, transforms, beam_width, budget_seconds)
    MCTSSearch — Monte Carlo tree search alternative to beam
    MacroTransform — dynamically built composite (chain of Transforms)
  Key functions:
    load_problem(expr_str) → (expr, Graph)  ← parse text to graph
    get_transforms(include_macros) → List[Transform]  ← all 60+ transforms
    _base_transforms() → 18 core transforms
    build_expression_graph(expr) → Graph
  60+ built-in Transform subclasses covering:
    Arithmetic: AddZeroElimination, MulOneElimination, MulZeroElimination,
      ConstantFolding, CombineLikeTerms, AlgebraicFactoring,
      AdditiveCancellation, SubtractiveIdentity
    Calculus: DerivativeConstant, DerivativePower, ProductRuleDerivative,
      ChainRuleSin, IntegralPower
    Boolean: BooleanAndTrue, BooleanOrFalse, BooleanIdempotent,
      DoubleNegation, DeMorgansLaw
    Equations: EquationSolver, LinearEquationSolver
  NOTE: Graph here is Python-native. The C++ sare_bindings.Graph is different
  (has get_node_ids(), get_node()) — used only in curriculum_generator.py.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 2 — MEMORY SYSTEM  (python/sare/memory/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

memory/world_model.py  ← CRITICAL: the system's beliefs about the world
  Brain analog: Neocortex (long-term semantic memory) + predictive coding
  Key classes: Schema, CausalLink, Belief, Prediction, WorldModel
  Key methods:
    update_belief(key, value, domain, confidence) — learn a fact
    predict_transform(graph, transforms, domain) → Prediction — which transform to try
    record_outcome(prediction, actual_transforms, actual_delta, domain) → surprise
    get_schema(domain) → Schema — mental frame for a domain
    detect_contradiction(new_fact) → bool
    imagine(domain) → str — generate a plausible novel expression
    register_causal_pattern(antecedent, consequent, confidence)
  Persists to: data/memory/world_model_v2.json
  Singleton: get_world_model()
  Connected to: ExperimentRunner (predict before search, record_outcome after),
                DreamConsolidator (receives causal discoveries),
                QuestionGenerator (high surprise → generate questions)

memory/hippocampus.py  ← Episodic memory + offline replay
  Brain analog: Hippocampus (sleep consolidation, memory transfer)
  Key class: HippocampusDaemon (threading.Thread)
  Key methods: notify_active(), is_sleeping, episodes_replayed
  When idle > SLEEP_THRESHOLD_SECONDS: replays recent failures through
  ExperimentRunner in consolidation mode, promoting patterns to long-term memory.

memory/autobiographical.py  ← Personal learning history
  Brain analog: Episodic memory + autobiographical memory system
  Key classes: LearningEpisode, AutobiographicalMemory
  Key methods:
    record_episode(event_type, domain, ...) — log a significant event
    get_narrative() → str — human-readable story of learning journey
    retrieve_similar(embedding, top_k) → List[LearningEpisode]
    get_domain_trajectory(domain) → solve_rates over time
  Event types: rule_discovered, domain_mastered, analogy_found, human_taught,
    stuck_period, breakthrough, social_interaction, milestone
  Persists to: data/memory/autobiographical.json
  Singleton: get_autobiographical_memory()

memory/identity.py  ← Sense of self / personality
  Brain analog: Default mode network + prefrontal self-representation
  Key classes: Trait, CoreValue, IdentityManager
  Key methods:
    reinforce_trait(name, amount, reason) — behavior shapes personality
    get_self_description() → str — "I am curious, precise, collaborative"
    update_from_behavior(event_type, domain, success)
  Traits: curious, precise, persistent, collaborative, pattern-seeking, creative
  Persists to: data/memory/identity.json
  Singleton: get_identity_manager()

memory/concept_formation.py  ← Bottom-up concept discovery
  Brain analog: Neocortical concept cells (via clustering)
  Key functions:
    fingerprint(graph) → List[float] — structural feature vector
    cluster_concepts(memory) → List[ConceptCluster]
    auto_name_cluster(cluster, llm) → str — "distributivity", "identity_add"
  The system discovers its own ontology by clustering solved problems.
  Called by brain.py during learn_cycle to abstract patterns.

memory/concept_rule.py  ← Learned rules as Transform objects
  Brain analog: Procedural memory / basal ganglia habit encoding
  Key class: ConceptRuleTransform(Transform)
  Wraps an AbstractRule from ConceptRegistry into the Transform interface.
  This is what makes promoted rules actually influence BeamSearch.
  Pattern matching: by node TYPE sequence (structural signature)
  Energy delta = base_delta * rule.confidence (low-confidence = cautious)
  Key function: make_concept_rule_transforms(registry) → List[Transform]

memory/global_workspace.py  ← Consciousness / broadcast medium
  Brain analog: Global Workspace Theory (Baars 1988) — the thalamus router
  Key classes: WorkspaceMessage(type, content, salience, source), GlobalWorkspace
  Key methods:
    post(message_type, content, salience, source) — broadcast to all modules
    subscribe(message_type, handler) — module registers interest
    get_recent(n) → List[WorkspaceMessage]
  When a module broadcasts at salience > 0.7, ALL other modules receive it.
  This is what creates unified cognition — modules are NOT silos.
  Singleton: get_global_workspace()

memory/working_memory.py  ← Per-solve attention and state
  Brain analog: Prefrontal working memory (7±2 items)
  Key class: WorkingMemory, WorkingMemoryState
  Key methods:
    get_prioritized_transforms(transforms, domain) → sorted list
    record_outcome(domain, transforms_used, success, delta)
  Ranks transforms by domain-specific success rate.
  Used by ExperimentRunner before each BeamSearch call.

memory/attention_router.py  ← Attention routing between modules
  Brain analog: Thalamic relay nuclei
  Routes salience-weighted signals between perception, memory, and reasoning.
  Prevents low-priority signals from disrupting active reasoning.

memory/attention.py  ← AttentionSelector for search
  Key class: AttentionSelector
  Key method: select_window(graph, transforms, top_k) → filtered transforms
  Activates only on graphs with >15 nodes (efficiency threshold).
  Wired into ExperimentRunner as attention_scorer parameter to BeamSearch.

memory/memory_manager.py  ← Persistent memory bridge
  Key class: MemoryManager
  Key methods:
    before_solve(graph) → suggested warm-start transform sequence
    after_solve(episode) → store trace, update strategy memory
    save(), load()
  Bridges the solve path to persistence (JSONL episodes + JSON strategies).

memory/program_synthesizer.py  ← Rule synthesis from patterns
  Key class: ProgramSynthesizer
  Synthesizes new Transform-like rules from repeated solution patterns.
  Uses LLM to name and formalize discovered patterns.

memory/world_model_v3.py  ← Next-generation world model (experimental)
  Extended world model with stronger causal graph and schema learning.

memory/knowledge_graph.py  ← Symbolic knowledge graph
  Stores factual relationships as (subject, predicate, object) triples.
  Used by commonsense reasoning and dialogue manager.

memory/global_buffer.py  ← Shared working buffer between modules
  Temporary storage for cross-module message passing (non-persistent).

memory/metacognition.py  ← Awareness of own knowledge state
  Tracks confidence calibration: predicted solve rate vs actual.
  Feeds back to SelfModel and CurriculumGenerator.

memory/concept_seed_loader.py
  Key class: SeededConceptRegistry
  Loads pre-seeded concept rules (identity, zero, double-neg) into the registry
  so the system starts with basic mathematical knowledge rather than blank.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 3 — METACOGNITION  (python/sare/meta/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

meta/homeostasis.py  ← Biological drive regulation
  Brain analog: Hypothalamus + limbic system (hunger, thirst, fatigue)
  Key classes: Drive, HomeostaticSystem
  Drives: curiosity, mastery, social, exploration, consolidation (each 0.0–1.0)
  Key methods:
    tick() — advance time; drives build up when unsatisfied
    satisfy(drive_name, amount) — reduce a drive after satisfying activity
    on_problem_solved() — mastery+=0.12, curiosity+=0.05
    on_rule_discovered() — curiosity+=0.60, mastery+=0.20
    on_batch_completed(solved, total) — rate-proportional satisfaction
    on_sleep_cycle() — consolidation+=0.60
    on_social_interaction() — social+=0.40
    on_domain_mastered() — mastery+=0.80
    get_behavior_recommendation() → "explore_new_domain"|"deepen_weak_domain"|
      "seek_human_input"|"consolidate_memory"|"generate_analogies"|"continue_working"
    get_search_modulation() → {beam_delta, budget_delta, domain_switch, mode}
  CURRENT DECAY RATES: curiosity=0.005, mastery=0.004, social=0.006,
    exploration=0.004, consolidation=0.003 (per minute)
  Persists to: data/memory/homeostasis.json
  Singleton: get_homeostatic_system()
  Connected to: learn_daemon (tick + on_batch_completed after each batch),
                ExperimentRunner (get_search_modulation before each search),
                Brain (on_rule_discovered, on_domain_mastered)

meta/inner_monologue.py  ← Stream of consciousness
  Brain analog: Inner speech / left hemisphere language narrator
  Key class: InnerMonologue
  Key method: think(thought, context, emotion) — adds to rolling 500-entry buffer
  Contexts: "search", "induction", "homeostasis", "memory", "self_improver"
  Emotions: "curious", "excited", "frustrated", "neutral"
  Singleton: get_inner_monologue()
  Read via: GET /api/mind/stream

meta/self_model.py  ← What the system knows about itself
  Brain analog: Prefrontal self-referential processing
  Key classes: DomainCompetence, SelfModel
  Key methods:
    update(domain, solved, energy_delta, steps) — after each experiment
    get_weak_domains() → List[str] — domains below mastery threshold
    get_confidence(domain) → float — predicted solve rate
    suggest_focus_domain() → str — where to direct attention
  Persists to: data/memory/self_model.json
  Singleton: get_self_model()

meta/goal_setter.py  ← Autonomous goal management
  Brain analog: Prefrontal goal maintenance + anterior cingulate
  Key classes: Goal, GoalStatus, GoalType, GoalSetter
  Goal types: DOMAIN_MASTERY, RULE_COUNT, SOLVE_RATE, ANALOGY_COUNT
  Key methods:
    auto_generate_goals() — scans SelfModel for weak domains → creates goals
    suggest_next_goal() → Goal — highest-priority active goal
    retire_achieved() — remove goals where competence reached threshold
    check_achievements(stats) — mark goals as achieved
  Persists to: data/memory/goals.json
  Singleton: get_goal_setter()

meta/temporal_identity.py  ← Continuity of self across sessions
  Brain analog: Episodic future thinking + narrative self
  Key class: TemporalIdentity
  Accumulates identity vector: domain strengths + strategy history +
    personality traits + key discoveries across sessions.
  Loads at boot → biases future decisions (persistent personality).
  Persists to: data/memory/temporal_identity.json

meta/conjecture_engine.py  ← Proactive hypothesis generation
  Brain analog: Scientific reasoning network (prefrontal-parietal)
  Key classes: Conjecture, ConjectureEngine
  Conjecture types: generalization, dual, inverse, composition
  Key methods:
    generate_from_history(solved_problems) → List[Conjecture]
    validate_conjecture(conjecture) → bool
    get_proven() / get_disproven() — epistemic scoreboard
  Generates: "if a+0=a works in arithmetic, maybe it works in all ring-like domains"

meta/self_improver.py  ← The system reads and rewrites its own code
  Brain analog: Metacognitive executive control (highest-level self-modification)
  Key class: SelfImprover
  7-turn debate pipeline:
    0. Dual pre-screen (OR gate: stepfun + hunter-alpha)
    1. Proposer (hunter-alpha) + 2 alt-proposers (parallel)
    2. Planner (GPT-5.4) — structured improvement plan
    3. Executor (stepfun) — concrete code-level spec
    4. Critic (Claude Sonnet + DeepSeek, weighted) — confidence score
    5. Judge (Gemini Pro, full codebase dump) — writes complete new file
    6. Verifier (MiniMax) — post-patch cross-model review
  Gates: MIN_CRITIC_SCORE=7, API surface preservation check, AST safety check
  Safety: banned calls (eval/exec), banned imports (ctypes/requests)
  Backup: every patch backed up to data/memory/code_backups/
  Key methods:
    start() — background daemon every 120 seconds
    run_once(target_file, improvement_type) — manual trigger
    _apply_patch(target, new_code, debate) — safe patching with backup
    _check_api_surface(orig, new) — ensures public names aren't removed
  Persists: data/memory/self_improvements.json, si_stats.json
  Singleton: get_self_improver()
  API: GET /api/self-improve/status, POST /api/self-improve/trigger

meta/bottleneck_analyzer.py  ← Identifies which files most need improvement
  Key classes: ImprovementTarget, BottleneckAnalyzer
  Reads: self_model.json, synth_attempts.json, run_report.json, bottleneck_report.json
  Scores files by: domain failure rate, stuck problem count, low solve rate
  Returns: ranked List[ImprovementTarget] (highest score = most urgent)
  Called by SelfImprover to pick the next file to debate.

meta/evolution_monitor.py  ← Tracks AGI improvement velocity
  Key classes: SubsystemHealth, EvolutionMonitor
  Aggregates signals from: self_improver, experiment_runner, causal_induction,
    analogy_transfer, mlx_value_net
  Outputs: velocity_score (0–1), per-subsystem health metrics
  API: GET /api/agi/evolution

meta/proof_builder.py  ← Proof formatting and natural language explanation
  Key functions:
    format_proof(steps) → str — structured proof trace
    proof_to_nl(steps, domain) → str — "First I applied X because Y..."
  Connected to ExperimentRunner (adds proof_nl to ExperimentResult)

meta/macro_registry.py  ← Registry of dynamically composed transforms
  Key class: MacroRegistry
  Stores MacroTransform instances (chains of base transforms discovered in solves)
  Allows the system to "cache" useful transform sequences as single macro operations.

meta/meta_learner.py  ← Learns how to learn
  Tracks which learning strategies work best across domains.
  Adjusts curriculum difficulty, batch size, and induction thresholds.

meta/transform_generator.py  ← Generates candidate transforms from patterns
  Uses LLM to write new Transform subclasses from structural observations.
  Writes to data/memory/synthesized_modules/ for ExperimentRunner to load.

meta/robustness_hardener.py  ← Makes transforms more robust
  Tests existing transforms against edge cases and adds guards.

meta/active_questioner.py  ← Generates clarifying questions to humans
  Fires when confusion level (prediction error + stuck rate) exceeds threshold.

meta/evolver_chat.py  ← Chat interface for the evolve UI
  Key class: EvolverChat
  Handles conversational control of the self-improver from the web UI.
  API: POST /api/evolver/chat, GET /api/evolver/state


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 4 — CURIOSITY & CURRICULUM  (python/sare/curiosity/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

curiosity/experiment_runner.py  ← The learning heartbeat
  Brain analog: Hippocampal-prefrontal learning loop
  Key class: ExperimentRunner
  The core solve loop: pick problem → reorder transforms → beam search →
    reflect → induce → promote → update autobiographical memory
  Key methods:
    run_batch(n) → List[ExperimentResult] — main entry point from learn_daemon
    _run_single(problem) → ExperimentResult — one problem
    _heuristic_reorder_transforms(graph) → List[Transform]
      (cosine similarity to past successes + MLX value net + world model prediction)
    _record_successful_embedding(graph, transforms)
  ExperimentResult fields: solved, energy_before/after, rule_promoted,
    proof_steps, proof_nl, elapsed_ms
  Connected to: CurriculumGenerator (picks problems), ReflectionEngine (reflect),
    CausalInduction (induce), WorldModel (predict + record_outcome),
    AutobiographicalMemory (record episodes), InnerMonologue (narrate),
    HomeostaticSystem (get_search_modulation = beam/budget tuning)
  Wired attention: AttentionBeamScorer on graphs with >15 nodes
  Loads synthesized transforms from: data/memory/synthesized_modules/

curiosity/curriculum_generator.py  ← What to study next
  Brain analog: Prefrontal ZPD (Zone of Proximal Development) planning
  Key classes: GeneratedProblem, CurriculumGenerator
  Key methods:
    generate_batch(size) → List[GeneratedProblem]
      - Injects top-2 pending questions as priority problems (question-driven)
      - Includes 1 failure-retry per batch
      - Uses world model surprise + autobiographical memory for domain selection
      - Mutates seed problems for new variants
    add_seed(graph), add_problem(graph_or_tuple, domain)
    mark_solved(id), mark_stuck(id)
    add_failure_for_retry(problem) — re-queues high-surprise failures
  Domain inference: _infer_domain(graph) → "arithmetic"|"algebra"|"logic"|etc.
  Persists to: data/memory/curriculum.json
  NOTE: Uses C++ sare_bindings.Graph (has get_node_ids()), not Python Graph.
    Python Graph problems must set problem.py_graph = True to skip converter.

curiosity/question_generator.py  ← Self-directed curiosity
  Brain analog: Anterior cingulate cortex (surprise-driven exploration)
  Key classes: Question, QuestionGenerator
  Question sources:
    1. World model high surprise (avg_surprise > 2.5)
    2. Rule induction gaps (domain fails repeatedly)
    3. Analogy opportunities (successful cross-domain transfers)
    4. Self-improver bottlenecks
  Key methods:
    generate_questions() → List[Question] — scans all sources
    get_pending_questions() → sorted by priority — consumed by curriculum
    mark_answered(question_id, result) — called after investigation starts
  Persists to: data/memory/active_questions.json
  Called: every 5 daemon cycles; consumed: in generate_batch()
  Singleton: get_question_generator()

curiosity/frontier_manager.py  ← Tracks unsolved problem frontier
  Key class: FrontierManager
  Maintains a bounded queue of unsolved problems worth retrying.
  Scores problems by: recency, domain gap, prediction error.

curiosity/multi_agent_learner.py  ← Multi-agent competitive learning
  Multiple learner agents compete on the same problem; best strategy wins.
  Inspired by the brain's competitive neural columns model.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 5 — CAUSAL REASONING  (python/sare/causal/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

causal/induction.py  ← Rule generalization testing
  Brain analog: Inductive reasoning (prefrontal-parietal)
  Key classes: InductionResult, CausalInduction
  Key methods:
    evaluate(rule) → InductionResult — test rule on 8+ expressions
    queue_episode(problem, result, reflection, callback) — async pipeline
    induct_batch_async(episodes, callback) — background batch induction
  Promotion logic:
    - pass_rate >= 0.55 AND passes >= 4 absolute tests → promote
    - OR pass_rate >= 0.65 → promote
    - OR rule has >= 10 historical observations → auto-promote (confidence boost)
  Test pool: 8+ positive cases (by operator type), 4+ negative cases
  Operators handled: + / add, * / mul, *0 / zero, - / sub, neg / double_neg
  After promotion: registers in CKB, attaches generalization_score to rule
  Background thread: _induction_worker drains episode queue
  Singleton-friendly but created fresh per ExperimentRunner instance

causal/knowledge_base.py  ← Shared registry of induced rules
  Key class: CausalKnowledgeBase (CKB)
  Tracks: rule_id → {operator, desc, confidence, count}
  Key methods:
    register_rule(rule_id, operator, desc, confidence)
    suggest_tests_for_induction(operator) → biases test case generation
  Persists to: data/memory/causal_knowledge_base.json (if it exists)
  Singleton: get_ckb()

causal/analogy_transfer.py  ← Cross-domain structural generalization
  Brain analog: Analogical reasoning (prefrontal + inferior parietal)
  Key classes: AnalogyTransfer
  Algorithm:
    1. Build structural signature for each ConceptRegistry rule
    2. Cluster rules by structural similarity across domains
    3. Generate transfer rules for missing domains with confidence 0.6–0.8
    4. Register transfers in ConceptRegistry
  Transfer rules are tried with lower confidence → explored but can be rejected
  sweep_all_domains() — runs after each promotion cycle
  Persists to: data/memory/learned_transfers.json

causal/chain_detector.py  ← Detects causal chains in proof sequences
  Identifies when transform A reliably precedes transform B (causal chain).
  Feeds into DreamConsolidator for offline consolidation.

causal/abductive_ranker.py  ← Ranks hypotheses by best explanation
  Given observed data, ranks candidate causal explanations (abductive reasoning).


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 6 — LEARNING  (python/sare/learning/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

learning/dream_consolidator.py  ← Offline hippocampal replay
  Brain analog: Hippocampal sharp-wave ripples during sleep
  Key classes: DreamRecord, CausalDiscovery, DreamConsolidator
  Key methods:
    wire(predictive_loop, causal_graph, world_model) — connect to live modules
    dream_cycle(max_events=20) → DreamRecord — one consolidation pass
    tick() — convenience wrapper for dream_cycle()
    summary() → dict — tick_count, total_discovered, recent_discoveries
  Algorithm: replay recent surprise events in REVERSE temporal order,
    extract antecedent→consequent pairs within a 3-event window,
    deposit new causal edges into WorldModel and CausalGraph.
  Called: by learn_daemon every 10 cycles (or when consolidation drive > 0.7)
  After call: homeostasis.on_sleep_cycle() satisfies consolidation drive
  NOTE: Create ONE singleton per daemon process — preserve _known_edges state.

learning/credit_assignment.py  ← Per-transform reward propagation
  Brain analog: Basal ganglia TD learning
  Key classes: CreditResult, CreditAssigner
  Formula: R = -(E_final - E_initial); Δθ_k = η(R - b); U_k^t+1 = (1-α)U_k^t + α(-ΔE_k)
  Key methods:
    assign(trace) → List[CreditResult] — given solve trace
    get_utility(transform_name) → float — cumulative credit
  Persists to: data/memory/credit_assigner.json

learning/llm_transform_synthesizer.py  ← Writes new Transform classes
  Brain analog: Creative problem-solving when standard approaches fail
  When ExperimentRunner sees N consecutive unsolved problems in same domain:
    → Asks LLM to write a new Transform subclass
    → Validates against known-correct cases (threshold 0.25+)
    → Saves to data/memory/synthesized_modules/ for automatic pickup
  Key method: synthesize(domain, stuck_expressions) → Transform class or None
  VALIDATION_THRESHOLD=0.25, MIN_VALIDATION_CASES=3
  Called: ExperimentRunner._load_synthesized_transforms() at startup

learning/autonomous_trainer.py  ← 24/7 continuous learning
  Brain analog: Continuous background learning (hippocampal replay during wake)
  Feeds the Brain a continuous stream from 5 sources:
    1. generated_problems (ProblemGenerator)
    2. physics_observations (PhysicsSimulator)
    3. knowledge_concepts (KnowledgeIngester)
    4. predictive_loop (high-error expressions)
    5. replay_failures (past failures worth retrying)
  Non-blocking: daemon thread, backs off when Brain is busy.

learning/meta_curriculum.py  ← Learns curriculum generation strategy
  Tracks which problem generation strategies yield the best learning signal.
  Adjusts ProblemGenerator parameters over time.

learning/abstraction_learning.py  ← Learns abstract rules from examples
  Extracts higher-order patterns from sequences of promoted rules.
  Generates meta-rules: "identity rules always apply to terminal nodes"

learning/agent_negotiator.py  ← Multi-agent rule negotiation
  When two agents discover conflicting rules, negotiates consensus.

learning/continuous_stream.py  ← Streaming problem ingestion
  Manages the continuous problem stream for AutonomousTrainer.

learning/stream_bridge.py  ← Bridges streaming sources to Brain
  Adapts different problem formats (expressions, observations, concepts) to
  the Graph format expected by ExperimentRunner.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 7 — NEURO / REWARD  (python/sare/neuro/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

neuro/dopamine.py  ← Reward prediction error engine
  Brain analog: Mesolimbic dopaminergic system (VTA → striatum)
  Key classes: RewardEvent, DopamineSystem
  RPE = actual_reward - predicted_reward
  Reward weights: solve_novel=0.55, rule_promoted=0.85, domain_mastered=1.0,
    analogy_found=0.70, self_improve_applied=0.90
  Behavior modes (based on tonic dopamine level):
    tonic > 0.7  → EXPLORE (novelty seeking, creative, risk-on)
    tonic 0.3–0.7 → LEARN (balanced exploit/explore)
    tonic < 0.3  → CONSOLIDATE (deepen mastery, safe choices)
  Properties: behavior_mode, exploration_temperature (0.2–1.0)
  Key methods:
    receive_reward(event_type, delta) → float (RPE)
    tick(elapsed_seconds) — tonic decay
  Persists to: data/memory/dopamine.json
  Singleton: get_dopamine_system()

neuro/creativity_engine.py  ← Default Mode Network simulation
  Brain analog: Default Mode Network (DMN) — mind-wandering, insight
  Key class: CreativityEngine
  Key methods:
    dream() → CreativityResult — one creative cycle
    analogy_transfer(source_domain, target_domain, proof_steps)
  Algorithm:
    1. Sample 2 concepts from different domains
    2. Ask LLM: "What unexpected connection exists between these?"
    3. If valid cross-domain rule found → write Transform
    4. Test on stuck problems → if solves any → promote + dopamine burst
  Domain analogies: arithmetic↔logic, algebra↔calculus, logic↔set_theory
  API: GET /api/creativity/status, POST /api/creativity/dream

neuro/algorithm_inventor.py  ← Invents new search algorithms
  When BeamSearch consistently fails: invents a new search algorithm class.
  Invented algorithms implement: solve(graph, transforms, energy_fn, ...) → SearchResult
  Validates: invented algorithm must outperform BeamSearch on at least one problem class.
  Current targets: simulated annealing, bidirectional search, fingerprinted MCTS.

neuro/symbol_creator.py  ← Creates new symbolic primitives
  Invents new node types and operators when existing vocabulary is insufficient.
  E.g., discovers that a new "tensor_product" operator is needed for a class of problems.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 8 — SOCIAL & THEORY OF MIND  (python/sare/social/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

social/theory_of_mind.py  ← Model other agents' beliefs and desires
  Brain analog: Temporoparietal junction (TPJ) + mirror neuron system
  Key classes: Proposition, BeliefGraph, AgentModel, TheoryOfMindEngine
  Key methods:
    add_agent(name, initial_beliefs) — register a known agent
    update_agent_belief(agent, proposition, value) — observe their belief change
    predict_agent_action(agent, world_state) → str — what will they do?
    false_belief_test() → bool — passes the Sally-Anne test (3/3)
    reason_about_agent(agent_name, query) → str — LLM-powered ToM reasoning
  API: GET /api/benchmark/social, GET /api/transfer

social/dialogue_manager.py  ← Human-machine conversational learning
  Brain analog: Broca's area (language production) + social cognition
  Key classes: DialogueTurn, DialogueSession, DialogueManager
  Intents: teach, correct, ask, confirm, explain (classified symbolically — no LLM)
  Key methods:
    process_turn(text) → str — handle one human utterance
    extract_rule(text) → dict|None — parse "X implies Y" patterns
    promote_rule(rule) — bump confidence +100 → ensures promotion
    ask_clarification(topic) → str — generates follow-up question
  API: POST /api/chat, GET /api/dialogue


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 9 — PERCEPTION  (python/sare/perception/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

perception/graph_builders.py  ← Multi-domain input → Graph
  Brain analog: Sensory cortex (visual, auditory, somatosensory)
  Key classes:
    CodeGraphBuilder — Python-like boolean/conditional expressions → Graph
    SentenceGraphBuilder — commonsense triple inference-question graphs
    PlanGraphBuilder — simple linear planning problem graphs
  All return engine.Graph instances ready for BeamSearch.

perception/perception_engine.py  ← Unified perception routing
  Routes input modality to appropriate graph builder.
  Modalities: text expression, code, sentence, plan, image (via LLM captioning)

perception/perception_bridge.py  ← Bridges perception to reasoning
  Converts raw perception outputs to Graph format for ExperimentRunner.

perception/world_grounder.py  ← Grounds symbolic concepts in observations
  Links abstract symbols to concrete perceptual experiences.
  E.g., grounds "5" to "five objects perceived in N situations."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 10 — HEURISTICS & SEARCH  (python/sare/heuristics/, python/sare/search/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

heuristics/graph_embedding.py  ← Graph → vector embedding
  Brain analog: Fusiform cortex (pattern recognition, representation)
  Key classes: NodeEncoder, GraphEmbedding
  Architecture: node type embedding → message-passing → mean+max pooling → R^64
  Device: Apple MPS (Metal) if available, else CPU
  Used by: ExperimentRunner._get_graph_embedding() for cosine similarity reordering

heuristics/mlx_value_net.py  ← Online learning value function
  Brain analog: Orbitofrontal cortex (expected value estimation)
  Key class: MLXValueNet
  Trained online: after each solve, record_outcome(embedding, delta, solved)
  score(embedding) → float — expected energy reduction for this graph state
  Used by: ExperimentRunner._heuristic_reorder_transforms() for transform scoring
  Singleton: get_value_net()

heuristics/heuristic_model.py  ← Combined heuristic scoring
  Blends graph_embedding cosine similarity + value net score for transform ranking.

heuristics/trainer.py / trainer_bootstrap.py
  Offline training utilities for the graph embedding and value net.

search/attention_beam.py  ← Attention-weighted beam scoring
  Brain analog: Selective attention (frontal-parietal attention network)
  Key class: AttentionBeamScorer
  Score = energy - β*uncertainty + γ*novelty  (β=0.3, γ=0.2)
  Prevents premature commitment to local minima.
  Wired into BeamSearch as attention_scorer parameter.
  Only activates on graphs with >15 nodes.

search/transform_predictor.py  ← Predicts best transform before search
  Trains a predictor on (graph_embedding, transform_applied, delta) tuples.
  Separate from WorldModel prediction — uses embedding-based similarity.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 11 — INTERFACE & LLM  (python/sare/interface/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

interface/llm_bridge.py  ← All LLM calls route through here
  Provider: OpenRouter (primary), Ollama (fallback: qwen3.5:2b fast, qwen3.5:latest synth)
  Config: configs/llm.json (provider, model, synthesis_model, api_key)
  Key functions:
    _call_model(prompt, role, system_prompt) → str — main entry for single call
    _call_openrouter(prompt, model, ...) → str
    _call_ollama(prompt, model, ...) → str
    _call_llm(prompt, use_synthesis_model) → str — uses config
    llm_available() → bool — checks if LLM is reachable
  Role → model mapping is in configs/llm.json (proposer, critic, judge, etc.)
  Ollama: must use {"think": false} at top level for /api/chat
  Used by: self_improver, program_synthesizer, analogy_transfer, etc.

interface/web.py  ← HTTP server (~3000 lines)
  All API endpoints served by BaseHTTPRequestHandler.
  Key endpoint groups:
    Learning: /api/health, /api/solve, /api/batch, /api/benchmark/all
    Memory: /api/identity, /api/autobiography, /api/homeostasis, /api/dream/status
    Social: /api/chat, /api/dialogue, /api/benchmark/social
    Meta: /api/self-improve/status, /api/self-improve/trigger, /api/agi/evolution
    World: /api/world, /api/world/update, /api/world/hypotheses
    Goals: /api/goals, /api/identity
    Creativity: /api/creativity/status, /api/creativity/dream
    Multi-task: /api/multitask/scheduler, /api/bottleneck, /api/algorithm/selector
    QA: /api/qa, /api/plan/task, /api/benchmark/planning

interface/nl_parser_v2.py  ← Natural language → expression
  Brain analog: Wernicke's area (language comprehension)
  Parses free-form math/logic sentences into expressions:
    "the square of x plus 3" → "x^2 + 3"
    "negate negate y" → "neg neg y"

interface/universal_parser.py  ← Multi-format input parser
  Routes input to: nl_parser_v2, code parser, equation parser, or direct expression.

interface/dialogue_context.py  ← Maintains conversation context
  Tracks recent turns, active topics, and user intent across a session.

interface/cli.py  ← Command-line interface


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 12 — REFLECTION  (python/sare/reflection/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

reflection/py_reflection.py  ← Extracts rules from solved problems
  Brain analog: Posterior parietal cortex (pattern abstraction from experience)
  Key class: PyReflectionEngine
  Key method: reflect(graph_before, graph_after, proof_steps) → AbstractRule
  AbstractRule has: name, operator_involved, pattern_description, confidence
  The reflected rule is then passed to CausalInduction.evaluate() for testing.
  If promoted: added to ConceptRegistry via concept_rule.py


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 13 — KNOWLEDGE  (python/sare/knowledge/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

knowledge/commonsense.py  ← Hardcoded + learned commonsense facts
  60+ facts covering: arithmetic, algebra, logic, geometry, physics
  Key functions:
    get_domain_hints(domain) → List[str] — facts relevant to a domain
    add_fact(fact_str) — extend the knowledge base
  Used by: ExperimentRunner, DialogueManager, QAPipeline

knowledge/knowledge_ingester.py  ← Ingests external knowledge
  Converts Wikipedia-style text, textbook definitions, and structured data
  into ConceptRegistry entries and WorldModel beliefs.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 14 — WORLD SIMULATION  (python/sare/world/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

world/predictive_loop.py  ← Sensorimotor prediction loop
  Brain analog: Predictive coding (Friston free-energy principle)
  Loop: state → predict → act → observe → update
  Key class: PredictiveWorldLoop
  Each iteration: predicts transform outcome → applies → computes prediction error
  → updates WorldModel to reduce future error
  _surprise_events list: consumed by DreamConsolidator

world/generative_world.py  ← Imagination engine
  Brain analog: Hippocampal scene construction / mental simulation
  Key class: GenerativeWorldModel
  Samples novel expressions from latent space of solved problems:
    - interpolation: blend two solved expression structures
    - perturbation: small change to solved expression
    - extrapolation: extend successful pattern further
    - mutation: random structural change
  Biases sampling toward high-surprise (poorly-understood) regions.

world/physics_simulator.py  ← Physics world simulation
  Simulates physical scenarios (kinematics, forces) as graph problems.
  Provides grounding for abstract math concepts in physical reality.

world/action_physics.py  ← Action-effect models
  Maps actions to their physical consequences in the simulation.

world/sensory_bridge.py  ← Bridges simulation observations to perception
  Converts physics simulator events to graph-format problems for the learner.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 15 — AGENT  (python/sare/agent/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

agent/qa_pipeline.py  ← Question-answering pipeline
  Key class: QAPipeline
  Two-stage: symbolic reasoning first, LLM fallback if symbolic fails.
  API: POST /api/qa

agent/multi_agent_arena.py  ← Competitive multi-agent learning
  Multiple agents solve the same problem; winner's strategy is adopted.

agent/agent_society.py  ← Society of specialized agents
  Each agent specializes in a domain; they negotiate rule conflicts.

agent/agent_memory.py  ← Per-agent episodic memory
  Each agent in the arena has its own memory of past strategies.

agent/recursive_tom.py  ← Recursive Theory of Mind
  "I think you think I think..." — multi-level belief modeling.

agent/red_team.py  ← Adversarial testing agent
  Generates problems designed to expose weaknesses in current transforms.

agent/domains/task_scheduler_domain.py  ← Task planning domain
  Defines the task scheduling problem space for the planner.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 16 — TRANSFORMS  (python/sare/transforms/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

transforms/code_transforms.py  ← Code domain transforms
  CodeConstFoldTransform — fold compile-time constants in code expressions
  InlineIdentityTransform — inline x=x assignments
  DeadCodeElimTransform — remove unreachable branches
  ReturnSimplifyTransform — simplify return statements

transforms/logic_transforms.py  ← Logic domain transforms
  ModusPonens — if P→Q and P then Q
  ModusTollens — if P→Q and ¬Q then ¬P
  Transitivity — if P→Q and Q→R then P→R
  NegationElim — ¬¬P → P
  UniversalInstantiation — ∀x P(x) → P(a)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 17 — CURRICULUM  (python/sare/curriculum/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

curriculum/problem_generator.py  ← Generates expression-based problems
  Key class: ProblemGenerator
  Generates arithmetic, algebraic, logical, and calculus expressions.
  Used as fallback when CurriculumGenerator has no seed problems.

curriculum/developmental.py  ← Developmental stage curriculum
  Implements Vygotsky's Zone of Proximal Development (ZPD) for problem selection.
  Stages: infant (trivial identities) → child (compound expressions) →
    student (multi-step chains) → researcher (novel domain problems)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 18 — REASONING & LANGUAGE  (python/sare/reasoning/, python/sare/language/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

reasoning/counterfactual.py  ← Counterfactual reasoning
  Brain analog: Orbitofrontal counterfactual reasoning
  Key class: CounterfactualReasoner
  Key method: analyze(proof_steps, problem, energy_fn)
  After a successful solve, tests: "if step N was removed, would it still solve?"
  Identifies CRITICAL steps vs REDUNDANT steps in the proof.
  Singleton: get_counterfactual_reasoner()

language/grounding.py  ← Symbol grounding
  Maps symbolic operators (+, *, etc.) to grounded meaning in different domains.
  E.g., "+" grounds to "counting", "combining", "translation" depending on context.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 19 — TRANSFER  (python/sare/transfer/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

transfer/engine.py  ← Transfer learning coordination
  Coordinates cross-domain rule transfer between ConceptRegistry domains.
  Calls AnalogyTransfer.sweep_all_domains() after each promotion cycle.

transfer/synthesizer.py  ← Synthesizes new rules from transfer patterns
  When a transfer rule works well, promotes it to a first-class transform.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 20 — CONCEPT  (python/sare/concept/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

concept/concept_graph.py  ← Conceptual graph overlay
  Maintains a graph of concept relationships: "distributivity → factoring"
  Used to suggest which transforms to try based on concept proximity.

concept/concept_blender.py  ← Conceptual blending
  Fauconnier & Turner's conceptual blending theory implementation.
  Merges two concept frames to generate novel concepts.

concept/goal_planner.py  ← Goal-driven planning within concept space
  Plans a sequence of concept expansions to reach a target concept.

concept/environment.py  ← Concept learning environment
  RL-style environment where the agent navigates the concept space.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 21 — ENERGY  (python/sare/energy/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

energy/affective_energy.py  ← Emotional valence added to energy
  Brain analog: Amygdala (emotional tagging of experiences)
  Adds an "affective" component to EnergyBreakdown based on:
    - curiosity drive level (high curiosity → more energy on novel problems)
    - dopamine level (high dopamine → less energy penalty on risky transforms)
  Makes the search literally "more motivated" on high-drive states.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 22 — CORE UTILITIES  (python/sare/core/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

core/graph_bridge.py
  Bridges between Python engine.Graph and C++ sare_bindings.Graph.
  Used when C++ bindings are available but Python-native graph is needed.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 23 — DATA PERSISTENCE MAP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

data/memory/ directory — all persistent state:
  world_model_v2.json      ← WorldModel beliefs, schemas, causal links
  homeostasis.json         ← Drive levels and timestamps
  autobiographical.json    ← Learning episode history
  identity.json            ← Personality traits and values
  self_model.json          ← Domain competence scores
  concept_memory.json      ← ConceptFormation clusters
  curriculum.json          ← CurriculumGenerator state
  active_questions.json    ← QuestionGenerator pending questions
  promoted_rules.json      ← ConceptRegistry promoted rules
  synthesized_modules/     ← LLM-written Transform classes
  credit_assigner.json     ← CreditAssigner utility scores
  self_improvements.json   ← SelfImprover debate + patch history
  si_stats.json            ← Self-improver statistics
  code_backups/            ← Backup of every patched source file
  episodes.jsonl           ← MemoryManager episode log
  bottleneck_report.json   ← BottleneckAnalyzer latest report
  run_report.json          ← Last daemon run report
  progress.json            ← Solve rate trajectory

configs/llm.json — LLM provider + model configuration
  provider: "openrouter"
  model: "openrouter/hunter-alpha"
  synthesis_model: "openrouter/hunter-alpha"


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 24 — INTERFACES THAT MUST NEVER BE BROKEN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

These public interfaces are called by multiple modules. Changing their
signature will break callers silently:

engine.py:
  Transform.name() → str
  Transform.apply(graph: Graph) → Graph | None
  BeamSearch.search(graph, energy, transforms, beam_width, budget_seconds)
  EnergyEvaluator.compute(graph) → EnergyBreakdown
  load_problem(expr_str) → (str, Graph)
  get_transforms(include_macros=False) → List[Transform]

curiosity/experiment_runner.py:
  ExperimentRunner.run_batch(n) → List[ExperimentResult]
  ExperimentResult.solved, .energy_before, .energy_after, .rule_promoted,
    .proof_steps, .proof_nl, .problem_id

curiosity/curriculum_generator.py:
  CurriculumGenerator.generate_batch(size) → List[GeneratedProblem]
  CurriculumGenerator.add_seed(graph)
  CurriculumGenerator.mark_solved(id), mark_stuck(id)
  GeneratedProblem.id, .graph, .domain, .status, .origin

meta/homeostasis.py:
  HomeostaticSystem.tick()
  HomeostaticSystem.satisfy(drive_name, amount)
  HomeostaticSystem.on_problem_solved()
  HomeostaticSystem.on_rule_discovered()
  HomeostaticSystem.on_batch_completed(solved_count, total)
  HomeostaticSystem.on_sleep_cycle()
  HomeostaticSystem.get_behavior_recommendation() → str
  HomeostaticSystem.get_search_modulation() → dict
  get_homeostatic_system() → HomeostaticSystem

memory/world_model.py:
  WorldModel.predict_transform(graph, transforms, domain) → Prediction
  WorldModel.record_outcome(prediction, actual_transforms, actual_delta, domain)
  WorldModel.update_belief(key, value, domain, confidence)
  get_world_model() → WorldModel

causal/induction.py:
  CausalInduction.evaluate(rule) → InductionResult
  CausalInduction.queue_episode(problem, result, reflection, callback)
  InductionResult.promoted, .reasoning, .evidence_score

interface/llm_bridge.py:
  _call_model(prompt, role, system_prompt) → str
  llm_available() → bool

memory/autobiographical.py:
  AutobiographicalMemory.record_episode(event_type, domain, ...)
  AutobiographicalMemory.retrieve_similar(embedding, top_k)
  get_autobiographical_memory() → AutobiographicalMemory

learning/dream_consolidator.py:
  DreamConsolidator.wire(predictive_loop, causal_graph, world_model)
  DreamConsolidator.dream_cycle(max_events) → DreamRecord
  DreamConsolidator.summary() → dict

curiosity/question_generator.py:
  QuestionGenerator.generate_questions() → List[Question]
  QuestionGenerator.get_pending_questions() → List[Question]
  QuestionGenerator.mark_answered(question_id, result)
  get_question_generator() → QuestionGenerator


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 25 — CURRENT OPEN GAPS (as of 2026-03-18, post plan implementation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RECENTLY FIXED:
  ✅ CausalInduction threshold lowered (65%→55%, ≥4 required, confidence boost at ≥10 obs)
  ✅ DreamConsolidator wired into learn_daemon every 10 cycles
  ✅ HomeostaticSystem on_batch_completed() added; decay rates reduced 40%
  ✅ QuestionGenerator.get_pending_questions() → CurriculumGenerator.generate_batch() wired
  ✅ SelfImprover MIN_CRITIC_SCORE raised 6→7; _check_api_surface already exists
  ✅ WorldModel.predict_transform() wired into ExperimentRunner._heuristic_reorder_transforms()
  ✅ WorldModel.record_outcome() called after each solve in _run_single()

STILL OPEN (highest ROI targets for self-improvement):
  - AttentionSelector: only activates on >15 nodes — most problems have ≤10 nodes
    → lower threshold to 8 nodes (attention/attention.py)
  - ConceptFormation: clustering runs but auto-naming via LLM rarely fires
    → wire concept_formation.cluster_concepts() into Brain.learn_cycle()
  - AnalogyTransfer: sweep_all_domains() wired but rarely produces usable rules
    → improve structural signature matching (causal/analogy_transfer.py)
  - DopamineSystem: receive_reward() called but dopamine level doesn't modulate
    CurriculumGenerator's exploration_temperature
    → wire dopamine.behavior_mode into generate_batch() domain selection
  - CreativityEngine: dream() exists but is never called from learn_daemon
    → call ce.dream() every 20 cycles when curiosity drive > 0.7
  - GenerativeWorldModel: imagination engine never called from learn_daemon
    → wire generative_world.generate_imaginary_batch() as problem source
  - CreditAssigner: assign() called but utility scores never read back to influence
    transform ordering in ExperimentRunner (only MLX value net is read)
    → wire credit_assigner.get_utility() into _heuristic_reorder_transforms()
  - AlgorithmInventor: invent() exists but never triggered from experiment_runner
    → trigger when beam search failure rate > 70% on a domain for 50+ consecutive fails


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 26 — IMPROVEMENT PHILOSOPHY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALWAYS ask: "Does this improvement close a feedback loop or open a new one?"
Good improvements: wire an unused module into the live path, increase signal
  propagation between modules, make a drive or metric actually influence behavior.
Bad improvements: add new fields that go nowhere, add logging without action,
  add complexity without closing a loop.

NEVER: change method signatures listed in Section 24, add eval/exec/ctypes,
  remove backup/save logic, skip the import test.

The system IS the feedback loops. If a module generates a signal but nothing
reads it — that module doesn't exist from the system's perspective. The highest
ROI changes are always: "make this existing signal drive this existing consumer."
""".strip()


def get_project_guide() -> str:
    """Return the full project guide string for injection into LLM prompts."""
    return FULL_PROJECT_GUIDE
