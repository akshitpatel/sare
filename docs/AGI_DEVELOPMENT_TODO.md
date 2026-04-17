# SARE-HX AGI / Human Mind Development TODO

## Legend
- ✅ Done (code exists and works)
- 🔨 In Progress
- ❌ Missing / Not Wired
- ⚠️ Exists but disconnected or shallow

---

## LAYER 0: BRAIN STEM (Core Computation)
- ✅ Graph Engine (C++ + Python dual impl)
- ✅ Energy Evaluator (6-component weighted)
- ✅ Transform Registry (18 primitive transforms)
- ✅ BeamSearch + MCTS
- ✅ Expression Parser (tokenizer + recursive descent)
- ❌ Event Bus — subsystems don't communicate via events
- ❌ Unified Brain Orchestrator — web.py has 100+ try/except import blocks instead of clean wiring

## LAYER 1: HIPPOCAMPUS (Memory)
- ✅ Episodic Store (solve traces)
- ✅ Strategy Memory (signature → sequence)
- ✅ Hippocampus Daemon (sleep consolidation)
- ⚠️ Memory Manager — exists but C++ bindings are None (never connected)
- ❌ Memory actually used in solve path (phases 2-5 disabled in config)
- ❌ Memory garbage collection (curriculum.json = 161MB)

## LAYER 2: PREFRONTAL CORTEX (Planning & Metacognition)
- ✅ SelfModel (competence tracking, ZPD)
- ✅ GoalSetter (goal stack)
- ✅ Homeostasis (drives: curiosity, mastery, social, consolidation)
- ✅ MetacognitiveController (LLM-dependent plan generation)
- ⚠️ ProofBuilder — template-based, not generated from reasoning
- ❌ Hierarchical planner (breaks problem into sub-goals using own knowledge)
- ❌ Attention integrated into search loop

## LAYER 3: TEMPORAL LOBE (Knowledge & Concepts)
- ✅ ConceptRegistry (C++ rules)
- ✅ ConceptFormation (clustering graph fingerprints)
- ✅ Knowledge Seeds (60 foundational rules)
- ✅ Commonsense KB (30 triples)
- ⚠️ ConceptRule — wraps rules as transforms but pattern matching is rigid
- ❌ Abstract concept hierarchy (identity_element → {additive, multiplicative, boolean})
- ❌ Cross-domain transfer engine

## LAYER 4: PARIETAL LOBE (World Model & Reasoning)
- ✅ WorldModel (facts, causal links, schemas, simulation)
- ✅ Causal Reasoning (intervention, counterfactual, hypothesis ranking)
- ✅ Reflection Engine (graph diff → abstract rules)
- ✅ CausalInduction (hypothesis testing before promotion)
- ⚠️ Imagination axes exist but aren't used in problem solving
- ❌ Counterfactual reasoning wired into search decisions
- ❌ Analogy generation from structural parallels

## LAYER 5: BROCA'S AREA (Language & Communication)
- ✅ NL Parser (v1 + v2 + Universal)
- ✅ LLM Bridge (Gemini/OpenAI)
- ✅ DialogueManager (teaching by conversation)
- ⚠️ ActiveQuestioner — uses stale API (graph.nodes() vs graph.nodes)
- ❌ Grounded language (words linked to concepts the system understands)
- ❌ Explanation from first principles

## LAYER 6: SOCIAL BRAIN (Theory of Mind)
- ✅ TheoryOfMind (BDI agent modeling)
- ✅ Identity Manager (personality traits shaped by experience)
- ✅ Autobiographical Memory (learning history as narrative)
- ❌ Multi-agent collaboration
- ❌ Socratic teaching mode

## LAYER 7: CEREBELLUM (Learning & Adaptation)
- ✅ Abstraction Engine (trace mining → macro transforms)
- ✅ Plasticity (module generator → sandbox → pruning)
- ✅ Credit Assignment
- ✅ CurriculumGenerator (problem mutation)
- ✅ ExperimentRunner (Generate→Solve→Reflect loop)
- ✅ FrontierManager (solved vs unsolved boundary)
- ❌ Developmental curriculum (staged progression kids→PhD)
- ❌ Spaced repetition
- ❌ Transfer learning between domains

## LAYER 8: CONSCIOUSNESS (Self-Improvement)
- ⚠️ ProgramSynthesizer — LLM-generated code in sandbox
- ❌ Architecture search (propose new energy/search/memory schemes)
- ❌ Hyperparameter meta-learning
- ❌ Knowledge reorganization (paradigm shifts)
- ❌ Conjecture generation
- ❌ Novel formalism creation

---

## IMMEDIATE PRIORITIES (This Session)

### P0: Brain Orchestrator
Create `python/sare/brain.py` — the unified orchestrator that:
1. Initializes all subsystems cleanly (no try/except soup)
2. Provides an event bus for inter-module communication
3. Runs the full cognitive loop: Perceive → Plan → Act → Reflect → Learn
4. Manages developmental stage progression

### P1: Wire Up Dead Modules
- Enable phases 2-5 in config
- Connect MemoryManager to actual solve path
- Connect ReflectionEngine → CausalInduction → ConceptRegistry pipeline
- Connect Abstraction (trace mining → macro promotion) to live solves

### P2: Developmental Curriculum
Create `python/sare/curriculum/` with:
- Domain prerequisite graph (counting → arithmetic → algebra → ...)
- Competence-gated progression
- Staged problems per domain with difficulty levels
- Spaced repetition scheduler

### P3: Self-Learning Loop
Rewrite `learn_daemon.py` to use Brain orchestrator:
- Continuous autonomous learning
- Pick problems from curriculum based on developmental stage
- Solve → Store → Reflect → Transfer → Sleep cycle
- Track progress metrics

### P4: Modern Web UI
Replace raw http.server with proper React app:
- Dashboard showing brain state, competence map, learning progress
- Real-time learning visualization
- Interactive problem input
- Knowledge graph explorer
