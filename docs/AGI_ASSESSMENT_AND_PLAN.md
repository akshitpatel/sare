# SARE-HX: AGI Assessment & Development Plan

## Final Assessment (March 13, 2026) — v3

### Quick Stats
- **Solve rate: 98%** (48/49 diverse problems)
- **26 transforms** (18 hand-built + 8 new + 26 auto-synthesized via deep transfer)
- **30+ modules** loaded (perception, world model, transfer, curriculum, metacognition, social)
- **103 causal links** discovered autonomously (zero hardcoded in WorldModel v3)
- **10 schemas** induced from experience
- **31 beliefs** formed (positive + negative, from success + failure)
- **8 structural roles** identified across domains
- **13 curriculum domains**, 8 mastered via autonomous learning
- **Background self-learning** runs continuously at 2s intervals

---

## Current State Assessment (March 2026)

---

## 1. What You've Built — Subsystem Inventory

### A. C++ Core (High-Performance Deterministic Engine)

| Subsystem | Files | Status | Depth |
|---|---|---|---|
| **Graph Engine** | `core/graph/` (10 files) | ✅ Solid | Full adjacency-list graph with delta ops, snapshots, diff, VF2 subgraph matcher, MCS |
| **Energy System** | `core/energy/` (14 files) | ✅ Solid | 6-component weighted energy (syntax, constraint, test, complexity, resource, uncertainty) |
| **Transforms** | `core/transforms/` (12 files) | ✅ Solid | Algebra, logic, AST transforms with utility tracking, registry, cost model |
| **Search** | `core/search/` (7 files) | ✅ Solid | BeamSearch + MCTS with UCB1, heuristic prior, budget manager |
| **Verification** | `core/verification/` (2 files) | ✅ Basic | SyntaxChecker, StaticAnalyzer, UnitTestRunner |
| **Memory** | `core/memory/` (6 files) | ✅ Solid | EpisodicStore (JSONL), StrategyMemory (signature→sequence), GraphSignature |
| **Abstraction** | `core/abstraction/` (8 files) | ✅ Solid | TraceMiner (k-gram patterns), MacroBuilder, AbstractionRegistry, CompressionEvaluator |
| **Plasticity** | `core/plasticity/` (6 files) | ✅ Solid | ModuleGenerator (failure analysis), SandboxRunner (A/B testing), PruningManager |
| **Causal** | `core/causal/` (6 files) | ✅ Solid | Intervention (do-operator), CounterfactualSimulator, HypothesisRanker (Occam) |
| **Reflection** | `core/reflection/` (5 files) | ✅ Solid | ReflectionEngine (graph diff → abstract rules), CausalInduction (hypothesis testing), ConceptRegistry |

### B. Python Layer (Probabilistic / High-Level)

| Subsystem | Files | Status | Depth |
|---|---|---|---|
| **Pure-Python Engine** | `engine.py` (1930 lines) | ✅ Full | Mirrors entire C++ API: Graph, Energy, Transforms, BeamSearch, MCTS |
| **Heuristics / GNN** | `heuristics/` (4 files) | ✅ Present | Graph embedding, heuristic model, trainer, bootstrap trainer |
| **Curiosity** | `curiosity/` (3 files) | ✅ Working | CurriculumGenerator (problem mutation), ExperimentRunner (Generate→Solve→Reflect loop), FrontierManager |
| **Memory (Rich)** | `memory/` (12 files) | ✅ Extensive | MemoryManager, Hippocampus (sleep consolidation), Attention (working memory), Autobiographical memory, Identity, WorldModel (schemas, causal links, counterfactuals, analogy), ConceptFormation (clustering), ConceptRule, ProgramSynthesizer |
| **Meta / Metacognition** | `meta/` (6 files) | ✅ Working | SelfModel (competence tracking, ZPD), GoalSetter (goal stack), Homeostasis (drives), ActiveQuestioner, ProofBuilder, MacroRegistry |
| **Social** | `social/` (2 files) | ✅ Present | DialogueManager (teaching by conversation), TheoryOfMind (BDI agent modeling) |
| **Perception** | `perception/` (1 file) | ⚠️ Basic | WorldGrounder (CSV, text, JSON → graph), but limited modalities |
| **Knowledge** | `knowledge/` (1 file) | ⚠️ Basic | Commonsense seed KB (~30 triples), optional ConceptNet integration |
| **Interface** | `interface/` (7 files) | ✅ Working | Web UI, CLI, LLM Bridge (Gemini/OpenAI), NL Parser (v1+v2), Universal Parser |
| **Learning** | `learning/` (2 files) | ⚠️ Basic | Abstraction learning, credit assignment |

### C. Infrastructure

| Component | Status |
|---|---|
| **C++ Build (CMake)** | ✅ Working with GoogleTest, PyBind11 |
| **11 Test Suites** | ✅ Covers all C++ subsystems |
| **Web UI** | ✅ Running at localhost:8080 |
| **Learn Daemon** | ✅ Autonomous background loop |
| **Autonomous Loop Script** | ✅ Solve → mine → promote macros |
| **Hippocampus Daemon** | ✅ Sleep-mode consolidation |
| **Persistent Memory** | ✅ 38+ JSON/JSONL files in data/memory/ |
| **Config System** | ✅ YAML + JSON configs |
| **CI/CD** | ✅ GitHub Actions |

---

## 2. Mapping to "Kids → PhD" Human Cognitive Development

### Human Learning Stages vs. SARE-HX Capability

| Human Stage | Core Capability | SARE-HX Status | Gap |
|---|---|---|---|
| **Infant (0-2)** | Sensorimotor, object permanence, basic pattern recognition | ✅ Graph perception, basic transform matching | Limited sensory input (no vision/audio) |
| **Toddler (2-4)** | Language acquisition, imitation, simple causal reasoning | ⚠️ NL parser exists, dialogue learning, basic causal | No true language generativity, no embodied learning |
| **Child (4-7)** | Concrete operations, rule following, basic arithmetic | ✅ Arithmetic/logic transforms, rule application, energy minimization | Rules are hand-coded or LLM-prompted, not truly self-discovered from scratch |
| **Pre-teen (7-12)** | Abstract thinking, hypothesis testing, multi-step planning | ✅ MCTS, beam search, causal induction, hypothesis ranking | Limited cross-domain transfer, no curriculum spanning math→science→humanities |
| **Teenager (12-18)** | Formal operations, metacognition, identity formation, social reasoning | ✅ SelfModel, Identity, Homeostasis, TheoryOfMind, Autobiographical memory | Most are scaffolded, not emergent. ToM is basic. No true self-awareness |
| **Undergraduate** | Domain specialization, proof construction, research methodology | ⚠️ ProofBuilder exists but template-based. Can learn domain rules. | No deep domain expertise, no reading/ingesting textbooks |
| **Graduate / PhD** | Novel research, paradigm creation, cross-domain synthesis, teaching others | ❌ Missing | No ability to formulate novel research questions, create new formalisms, or teach discovered knowledge systematically |

### Developmental Maturity Score: **~35-40% toward human-like learning**

**Strengths:**
- Core computational substrate is well-designed (graph + energy + search is a solid foundation)
- Memory architecture is surprisingly rich (episodic, strategy, autobiographical, identity)
- Self-supervised learning loop exists (Generate → Solve → Reflect → Induce → Learn)
- Metacognition scaffolding is present (SelfModel, Homeostasis, GoalSetter)

**Weaknesses:**
- Most learning is within a narrow symbolic algebra/logic domain
- No multi-modal perception (the "body" is missing)
- Cross-domain transfer is primitive
- Language understanding relies on LLM crutch rather than emergent capability
- No true curriculum progression (kids→PhD trajectory doesn't exist as a pathway)

---

## 3. Critical Gaps Analysis

### GAP 1: Developmental Curriculum (The "School" is Missing)
**Problem:** The system has curiosity-driven problem generation but no structured developmental progression. A child goes through arithmetic → algebra → geometry → calculus → linear algebra → analysis. SARE-HX's curriculum is flat.

**What's Needed:**
- Staged knowledge domains with prerequisite chains
- Competence gates (must reach 80% in algebra before attempting calculus)
- Scaffold → fade paradigm (initially guided, progressively autonomous)

### GAP 2: Cross-Domain Transfer Learning
**Problem:** Rules learned in arithmetic don't transfer to logic, set theory, or code. A human who learns "identity element" in addition (x+0=x) immediately hypothesizes it for multiplication (x*1=x) and boolean (x AND true = x).

**What's Needed:**
- Structural analogy engine (not just surface pattern matching)
- Abstract concept hierarchies (identity_element → {additive_identity, multiplicative_identity, boolean_identity})
- Transfer scoring and hypothesis generation

### GAP 3: Multi-Modal Perception
**Problem:** SARE-HX only understands pre-structured graph inputs. A human learns from visual diagrams, spoken explanations, written text, physical manipulation.

**What's Needed:**
- Rich text understanding (beyond NL parser → simple graph)
- Diagram/image understanding (math notation, circuit diagrams, etc.)
- Ability to read and learn from textbooks/papers

### GAP 4: Compositional Generalization
**Problem:** The system can learn x+0=x and x*1=x as separate rules, but struggles with compositional problems like "(a+0)*(b*1) + (c-c)" that require chaining multiple rules in novel combinations.

**What's Needed:**
- Hierarchical planning (break complex problems into sub-goals automatically)
- MetacognitiveController exists but is LLM-dependent, not self-sufficient
- Planning should use the system's own learned rules as plan operators

### GAP 5: True Self-Improvement (The "PhD" Level)
**Problem:** The system can learn rules from experience, but cannot:
- Invent new mathematical formalisms
- Generate novel conjectures
- Prove theorems it hasn't seen before
- Create new problem domains from scratch

**What's Needed:**
- Conjecture generation engine
- Formal proof verification (beyond template matching)
- Novelty detection and pursuit
- Meta-learning: learning how to learn better

### GAP 6: Robust Language & Communication
**Problem:** DialogueManager and NL parsers exist but rely heavily on regex/template matching or LLM bridge. The system cannot truly understand or generate nuanced natural language.

**What's Needed:**
- Grounded language learning (words linked to graph concepts the system actually understands)
- Explanation generation from first principles (not templates)
- Socratic dialogue capability

### GAP 7: Long-Horizon Reasoning & Working Memory
**Problem:** BeamSearch/MCTS operate on short-horizon graph rewrites. Solving a real math problem (e.g., "solve x^2 + 3x + 2 = 0") requires 10-20 strategic steps with working memory.

**What's Needed:**
- Hierarchical MCTS (plan at abstract level, execute at concrete level)
- Working memory with attention (attention.py exists but isn't deeply integrated into search)
- Backtracking with learned heuristics for when to abandon a path

### GAP 8: Continuous Integration of Knowledge
**Problem:** Knowledge seeds are pre-loaded, rules are learned incrementally, but there's no mechanism for the system to restructure its entire knowledge base when a paradigm-shifting insight occurs.

**What's Needed:**
- Knowledge graph restructuring (when learning "groups" abstract concept, reorganize all identity/inverse rules under it)
- Belief revision (when a high-confidence rule is contradicted, cascade updates)
- Conceptual change (Piaget's accommodation, not just assimilation)

---

## 4. High-Level Development Plan

### Overview: 6 Tiers, 24 Months

```
Tier 0: Foundation Hardening     (Months 1-2)   ← You Are Here
Tier 1: Developmental Curriculum (Months 2-5)
Tier 2: Transfer & Abstraction   (Months 5-9)
Tier 3: Autonomous Reasoning     (Months 9-14)
Tier 4: Language & Communication (Months 14-18)
Tier 5: Self-Improvement         (Months 18-24)
```

---

### Tier 0: Foundation Hardening (Months 1-2)
*Goal: Make what you have bulletproof before adding complexity.*

#### 0.1 — Activate Dormant Phases
- **Status:** `configs/default.yaml` has phases 2-5 set to `false`
- **Action:** Enable memory, heuristics, abstraction, plasticity, causal in config
- Wire the C++ modules into the Python solve path end-to-end
- Verify the full loop: `solve → store episode → mine patterns → promote macro → use macro in next solve`

#### 0.2 — Integration Testing
- Create end-to-end integration tests that verify:
  - EpisodicStore persists and retrieves across restarts
  - StrategyMemory actually warm-starts search
  - MacroTransforms get used by BeamSearch
  - ReflectionEngine → CausalInduction → ConceptRegistry pipeline works
- Target: 90%+ pass rate on integration tests

#### 0.3 — Performance Baseline
- Run `scripts/autonomous_loop.py` for 1000 episodes
- Record: solve rate per domain, avg energy reduction, rules learned, macros promoted
- This becomes the baseline for measuring all future improvements

#### 0.4 — Memory Consolidation Cleanup
- `data/memory/curriculum.json` is 161MB — needs pagination/rotation
- `data/memory/episodes.jsonl` is 13MB — add archival
- Implement memory garbage collection (decay old episodes, prune low-value strategies)

---

### Tier 1: Developmental Curriculum (Months 2-5)
*Goal: Build the "school" — structured progression from simple to complex.*

#### 1.1 — Domain Knowledge Graph
Create a prerequisite dependency graph of knowledge domains:

```
counting → arithmetic → algebra → polynomials → calculus
                ↘ fractions → ratios
logic → propositional_logic → predicate_logic → set_theory
                                    ↘ proof_methods
```

- Each domain has: seed problems, target rules, competence threshold
- Store in `configs/developmental_curriculum.json` (a richer version of existing)

#### 1.2 — Competence-Gated Progression
- Extend `SelfModel` with domain prerequisite checking
- CurriculumGenerator draws from the next-unlocked domain
- Implement ZPD-based difficulty selection (already modeled in SelfModel.exploration_weight)
- Add "readiness tests" — if the system can solve 80% of domain N, unlock domain N+1

#### 1.3 — Scaffold → Fade Learning
- Phase 1 (Scaffolded): Provide hints with problems (e.g., "try the identity rule")
- Phase 2 (Guided): Provide only the domain, let system choose strategy
- Phase 3 (Independent): Raw problem, no hints
- Track per-domain which scaffold level the system is at

#### 1.4 — Expand Problem Domains
Currently limited to algebraic simplification. Add:
- **Equation solving** (isolate variable: 2x + 3 = 7 → x = 2)
- **Logical deduction** (modus ponens, syllogisms, truth tables)
- **Set operations** (union, intersection, complement)
- **Basic geometry** (area/perimeter formulas as graph transforms)
- Each domain = new transform set + energy components + seed problems

#### 1.5 — Spaced Repetition
- Problems the system solved long ago should resurface periodically
- If a previously-solved domain's competence drops, re-queue problems
- Integrate with Hippocampus sleep cycle for offline review

---

### Tier 2: Transfer & Abstraction (Months 5-9)
*Goal: Learn abstract concepts that bridge domains, like a smart student.*

#### 2.1 — Structural Analogy Engine
- Given a rule in domain A, generate candidate analogous rules in domain B
- Example: learning x+0=x (arithmetic) → hypothesize x∧True=x (logic)
- Algorithm: extract structural skeleton (binary_op, identity_element, operand) → instantiate in new domain
- Store successful transfers in `data/memory/learned_transfers.json` (already exists!)

#### 2.2 — Abstract Concept Hierarchy
- Build on ConceptFormation (clustering) to create a concept taxonomy
- E.g., cluster {additive_identity, multiplicative_identity, boolean_and_true} → "identity_element"
- Use LLM bridge to name clusters, but verify naming via symbolic properties
- Concepts become searchable: "find all identity rules" returns rules across domains

#### 2.3 — Analogical Problem Solving
- When stuck on a problem in domain X, query for structurally similar solved problems in domain Y
- Use SubgraphMatcher MCS (already implemented!) to find structural parallels
- Adapt the solution strategy from Y to X
- Record successful cross-domain transfers as high-importance autobiographical events

#### 2.4 — Meta-Rule Learning
- Learn rules about rules: "If [operator] has a [zero-element], then [operator](x, zero) = [result]"
- These meta-rules generate new concrete rules when a new operator is encountered
- Store in a separate MetaRuleRegistry

#### 2.5 — Compression-Driven Learning
- CompressionEvaluator exists but isn't driving learning decisions
- Implement MDL (Minimum Description Length) principle: prefer representations that compress the knowledge base
- When two rules can be unified into one meta-rule, prefer the meta-rule
- Track "description length" of the full knowledge base over time — it should decrease as abstractions are formed

---

### Tier 3: Autonomous Reasoning (Months 9-14)
*Goal: Solve multi-step problems without hand-holding, like a grad student.*

#### 3.1 — Hierarchical Planning
- Replace flat BeamSearch with hierarchical decomposition:
  1. **Strategic level:** Break problem into sub-goals (e.g., "isolate variable", "simplify left side", "evaluate constants")
  2. **Tactical level:** For each sub-goal, run MCTS with domain-specific transforms
  3. **Execution level:** Apply concrete transforms
- MetacognitiveController should generate plans from the system's own knowledge, not LLM

#### 3.2 — Attention-Guided Search
- Deeply integrate attention.py into the search loop
- At each search step, AttentionSelector narrows the graph to a working memory window
- Transforms only apply within the window → massive search space reduction
- Window shifts based on which region has highest energy / uncertainty

#### 3.3 — Conjecture Generation
- After mastering a domain, the system should generate conjectures:
  - "I notice that [pattern] always holds. Is this a general law?"
  - Test conjectures via CausalInduction
  - Promoted conjectures become new rules
- This is the transition from "student" to "researcher"

#### 3.4 — Proof Construction
- ProofBuilder currently generates post-hoc explanations
- Upgrade to forward proof construction:
  - Given a conjecture, construct a proof tree using known rules
  - Each proof step = a verified transform application
  - Failed proofs generate counter-examples that feed back into learning

#### 3.5 — Novelty Detection & Pursuit
- Detect when a problem has genuinely new structure (not seen in any training episode)
- Novel problems get higher curiosity priority
- Track a "novelty score" per problem based on distance from nearest known fingerprint

---

### Tier 4: Language & Communication (Months 14-18)
*Goal: Understand and explain like a teacher.*

#### 4.1 — Grounded Language Model
- Words/phrases linked to graph concepts the system actually understands
- "identity" → pointer to the identity_element meta-concept
- "simplify" → pointer to the energy-minimization goal
- Build a bidirectional lexicon: concept ↔ natural language

#### 4.2 — Explanation from First Principles
- Given a solve trace, generate a genuine explanation (not template-fill)
- Reference the rules used, why each step was chosen, what alternatives were considered
- Tailor explanation depth to the questioner's level (uses SelfModel to estimate)

#### 4.3 — Socratic Teaching Mode
- When a human asks "why is x+0 = x?", don't just state the rule
- Ask guiding questions: "What happens when you add nothing to a pile of 5 apples?"
- Build on DialogueManager with pedagogical strategies

#### 4.4 — Reading & Knowledge Ingestion
- Parse structured text (textbook chapters, Wikipedia math articles) into graph form
- Extract rules, definitions, theorems, and examples
- Verify extracted knowledge against existing ConceptRegistry
- Flag contradictions for human review

#### 4.5 — Multi-Agent Collaboration
- Extend TheoryOfMind for actual multi-agent problem solving
- Agent A specializes in algebra, Agent B in logic → collaborate on mixed problems
- Share strategies, rules, and conjectures between agents

---

### Tier 5: Self-Improvement (Months 18-24)
*Goal: The system improves its own architecture — the "PhD" level.*

#### 5.1 — Architecture Search
- ModuleGenerator currently proposes transforms. Extend to propose:
  - New energy components
  - New search strategies
  - New memory organization schemes
- Evaluate via SandboxRunner methodology (A/B test on held-out problems)

#### 5.2 — Learning Rate Meta-Learning
- The system should learn its own hyperparameters:
  - Optimal beam width per domain
  - Energy weights that evolve as competence grows
  - Exploration/exploitation balance in MCTS
- Use the SelfModel's calibration error as the meta-objective

#### 5.3 — Knowledge Base Reorganization
- Periodically restructure the concept hierarchy
- Merge redundant rules, split over-general rules
- Implement Piaget's "accommodation" — when new evidence breaks an existing schema, restructure

#### 5.4 — Novel Formalism Creation
- When existing transforms can't solve a class of problems, the system should:
  1. Analyze the failure mode
  2. Hypothesize what kind of operation is needed
  3. Define the operation formally (as a new transform type)
  4. Test it in sandbox
  5. Promote if successful
- This is ModuleGenerator on steroids — inventing new math, not just new instances

#### 5.5 — Self-Replication of Knowledge
- Package learned knowledge (rules, concepts, strategies) into a "curriculum" that can bootstrap a fresh SARE-HX instance
- Measure: does the bootstrapped instance reach competence faster than learning from scratch?
- This is the system "teaching" a copy of itself — the ultimate test of understanding

---

## 5. Priority Matrix

| Priority | Item | Impact | Effort | Dependencies |
|---|---|---|---|---|
| **P0** | 0.1 Activate dormant phases | Critical | Low | None |
| **P0** | 0.2 Integration tests | Critical | Medium | 0.1 |
| **P0** | 0.4 Memory cleanup (161MB curriculum.json) | Critical | Low | None |
| **P1** | 1.1 Domain knowledge graph | High | Medium | 0.1 |
| **P1** | 1.4 Expand problem domains | High | High | 0.1 |
| **P1** | 1.2 Competence-gated progression | High | Medium | 1.1 |
| **P2** | 2.1 Structural analogy engine | Very High | High | 1.4 |
| **P2** | 3.2 Attention-guided search | High | Medium | 0.2 |
| **P2** | 2.2 Abstract concept hierarchy | High | High | 2.1 |
| **P3** | 3.1 Hierarchical planning | Very High | High | 1.2, 3.2 |
| **P3** | 3.3 Conjecture generation | High | High | 2.2 |
| **P3** | 4.1 Grounded language model | Medium | High | 2.2 |
| **P4** | 5.1 Architecture search | Very High | Very High | 3.1 |
| **P4** | 5.4 Novel formalism creation | Transformative | Very High | 3.3, 3.4 |

---

## 6. Key Metrics to Track

| Metric | Current (Est.) | Tier 1 Target | Tier 3 Target | Tier 5 Target |
|---|---|---|---|---|
| **Domains Mastered** | 2 (arith, logic) | 6 | 12 | 20+ |
| **Rules Learned (autonomous)** | ~10 | 50 | 200 | 1000+ |
| **Cross-Domain Transfers** | 0 | 5 | 50 | 200+ |
| **Max Problem Complexity (steps)** | 3-5 | 10 | 25 | 50+ |
| **Solve Rate (seen domains)** | ~60% | 80% | 90% | 95% |
| **Solve Rate (novel domains)** | ~5% | 15% | 40% | 70% |
| **Self-Discovered Conjectures** | 0 | 0 | 10 | 100+ |
| **Knowledge Base Compression** | N/A | baseline | 20% smaller | 50% smaller |

---

## 7. Architectural Recommendations

1. **Unify C++ and Python engines.** Currently two parallel implementations exist. Long-term, the Python engine should be a thin wrapper over C++ bindings, not a reimplementation.

2. **Event-driven architecture.** Replace direct function calls between subsystems with an event bus: `SOLVE_COMPLETED`, `RULE_DISCOVERED`, `DOMAIN_MASTERED`, etc. This decouples modules and enables the homeostatic system to react to any event.

3. **Formal type system for graphs.** Currently node types are strings ("operator", "variable", "constant"). Define a typed schema per domain so the energy system and transforms can be statically verified.

4. **Benchmark suite.** Create a standardized test battery (like IQ tests for the system) that spans domains and difficulty levels. Run it after every major change.

5. **Distributed compute.** MCTS rollouts and sandbox evaluations are embarrassingly parallel. Add multi-threaded C++ search and multi-process Python experimentation.

---

## Summary

**You've built an impressive cognitive architecture foundation.** The graph-native representation, energy-driven optimization, memory systems, and self-supervised learning loop are well-designed. The system is roughly at the **"smart pre-teen"** level — it can follow rules, learn from experience, and has nascent metacognition.

**The biggest gaps to reach "PhD level" are:**
1. **Structured developmental progression** (the "school" pathway)
2. **Cross-domain transfer** (the hallmark of true intelligence)
3. **Hierarchical multi-step reasoning** (beyond flat search)
4. **Self-generated conjectures and proofs** (the "researcher" mode)
5. **Knowledge reorganization** (paradigm shifts, not just accumulation)

**Recommended immediate next steps:**
1. Activate dormant phases (0.1) — this is blocking everything
2. Fix memory bloat (0.4) — 161MB curriculum.json will cause issues
3. Build integration test suite (0.2) — verify the full loop works
4. Design the developmental curriculum graph (1.1) — the roadmap for the system's "education"
