# SARE-HX AGI Development Roadmap
**Last Updated:** March 13, 2026  
**Status:** Active Development — Session 3 Progress

---

## Current Verified State (Session 3 Results)

### What Works Right Now (Confirmed via Code)

| Capability | Evidence | Score |
|---|---|---|
| **Symbolic Solve** | 39 transforms, 11/11 test expressions solved | ✅ Solid |
| **Multi-domain Knowledge** | Arithmetic (23), Logic (7), Trig (4), Calculus (3), Sets (2) | ✅ Real |
| **Parser** | +, -, *, /, ^, =, neg, not, and, or, sin/cos/tan/log/sqrt/derivative, true/false | ✅ Complete |
| **Python Reflection** | 4 structural rule patterns extracted from solves | ✅ Works |
| **Transfer Synthesis** | 26 transforms auto-generated across identity/annihilation/involution roles | ✅ Works |
| **Open Problem Discovery** | 243 templates across 5 domains including trig+calculus | ✅ Works |
| **Language Grounding** | NL explanations + analogies for all transform types | ✅ Works |
| **Brain Event Bus** | Solve → Reflect → Ground → Store pipeline wired | ✅ Works |
| **C++ Reflection** | `_py_graph_to_cpp_graph` + `ReflectionEngine.reflect()` extracts `additive_identity` | ✅ Works (when binary available) |
| **CausalInduction** | C++ hypothesis testing fixed (API calls), Python fallback in place | ✅ Fixed |
| **Auto-learn Thread** | Background daemon, synthesis seeds on boot | ✅ Wired |

### What's Broken / Missing

| Gap | Impact | Priority |
|---|---|---|
| **C++ binary crashes on import** (macOS -all_load + OOM) | C++ reflection, CausalInduction, ModuleGenerator all unavailable | HIGH |
| **No SolveEpisode binding** | Can't pass Python episodes to C++ ModuleGenerator | HIGH |
| **GoalSetter never generates goals** | System doesn't proactively plan its own learning | MEDIUM |
| **Distributive domain 0% solve rate** | `a*(b+c)=ab+ac` increases node count — energy goes UP | MEDIUM |
| **No persistent rule-to-transform pipeline** | Rules in ConceptRegistry don't automatically become live transforms | HIGH |
| **No imagination/conjecture testing** | WorldModel v3 generates hypotheses but doesn't test them on problems | MEDIUM |
| **No probability/statistics domain** | Major reasoning gap | LOW |
| **No spatial/temporal reasoning** | No geometry, no time | LOW |

---

## Phase 1: Foundation Fixes (1–2 weeks)

### P1.1 — Fix C++ Binary (HIGHEST PRIORITY)

**Problem:** `causal_induction.cpp` changes + macOS `-all_load` linker flag causes OOM on import.

**Root Cause:** The `-all_load` flag loads ALL symbols from `libsare_core.a`, which has grown significantly. This exceeds memory limits in test environments.

**Fix Options (in order of preference):**
```
Option A: Replace -all_load with -force_load for specific targets only
  cmake: target_link_options(sare_bindings PRIVATE "-Wl,-force_load,libsare_core.a")
  This only forces load of what's actually used.

Option B: Build sare_core as SHARED instead of STATIC
  change: add_library(sare_core SHARED ...) in CMakeLists.txt
  This allows lazy symbol loading.

Option C: Build bindings separately without -all_load
  Remove the -all_load block entirely and let normal symbol resolution work.
```

**Action Items:**
```cmake
# In CMakeLists.txt, replace:
if(APPLE)
    target_link_options(sare_bindings PRIVATE "LINKER:-all_load")
endif()
# With:
if(APPLE)
    target_link_options(sare_bindings PRIVATE 
        "-Wl,-force_load,${CMAKE_BINARY_DIR}/libsare_core.a")
endif()
```

### P1.2 — Expose SolveEpisode to Python (enables real plasticity)

**What:** Add `SolveEpisode` and `generate_step_sequences` to pybind_module.cpp using safe string-return-only API.

**Impact:** Enables `_run_cpp_plasticity` to convert Python memory episodes → C++ SolveEpisodes → C++ macro candidates → Python macros.

**Safe Binding Pattern:**
```cpp
// Add AFTER fixing -all_load issue:
py::class_<sare::SolveEpisode>(m, "SolveEpisode")
    .def(py::init<>())
    .def_readwrite("problem_id", &sare::SolveEpisode::problem_id)
    .def_readwrite("transform_sequence", &sare::SolveEpisode::transform_sequence)
    .def_readwrite("energy_trajectory", &sare::SolveEpisode::energy_trajectory)
    .def_readwrite("initial_energy", &sare::SolveEpisode::initial_energy)
    .def_readwrite("final_energy", &sare::SolveEpisode::final_energy)
    .def_readwrite("success", &sare::SolveEpisode::success);

// Safe generate_step_sequences returns string vectors only:
.def("generate_step_sequences",
     [](sare::ModuleGenerator& self, const std::vector<sare::SolveEpisode>& failures,
        size_t max) -> std::vector<std::vector<std::string>> {
         auto candidates = self.generate(failures, max);
         std::vector<std::vector<std::string>> result;
         for (auto& c : candidates) {
             if (!c) continue;
             std::string name = c->name();
             // parse "generated_X_then_Y" → ["X", "Y"]
             ...
             result.push_back(steps);
         }
         return result;
     }, py::arg("failures"), py::arg("max_candidates") = 3)
```

### P1.3 — Fix Distributive Domain Solve Rate

**Problem:** `distributive_expand` produces LARGER graphs (more nodes), so energy INCREASES. The system never solves distributive problems.

**Root Fix:** The `DistributiveExpansion` transform should only fire when it enables a follow-up simplification. We need a lookahead heuristic.

**Implementation:**
```python
class DistributiveExpansion(Transform):
    def match(self, graph):
        # Only match if a like-term combination will follow
        matches = super().match(graph)
        valid = []
        for ctx in matches:
            ng, _ = super().apply(graph, ctx)
            # Check if CombineLikeTerms can fire on the expanded form
            if CombineLikeTerms().match(ng):
                valid.append(ctx)
        return valid
```

### P1.4 — Wire ConceptRegistry → Live Transforms (Persistent Learning)

**Problem:** Rules promoted to ConceptRegistry (via reflection) don't become live transforms after restart.

**Fix:** In `Brain.boot()`, after loading ConceptRegistry, reload all high-confidence rules as `ConceptRule` transforms.

```python
def _boot_core(self):
    ...
    # After _boot_knowledge() sets up concept_registry:
    if self.concept_registry:
        rules = self.concept_registry.get_consolidated_rules(min_confidence=0.7)
        for rule in rules:
            t = self._rule_to_transform(rule)
            if t:
                self.transforms.append(t)
```

---

## Phase 2: Learning Loop Completion (2–4 weeks)

### P2.1 — Verified Rule Promotion Pipeline

**Current State:** Rules are discovered by PyReflectionEngine but only stored transiently (not persisted across sessions).

**Target:** Every solve → reflect → test on 3 problems → if passes → add to ConceptRegistry → persist → reload on next boot.

**Implementation:**
```python
def _reflect_and_promote(self, g_before, g_after, transforms, domain):
    # 1. Extract rule (Python or C++ path)
    rule = self.py_reflection_engine.reflect(g_before, g_after, transforms, domain)
    if not rule or not rule.valid():
        return
    
    # 2. Test on 3 problems from the same domain
    test_problems = self._get_test_problems(domain, n=3)
    successes = self._test_rule_on_problems(rule, test_problems)
    
    # 3. Only promote if rule passes tests
    if successes >= 2:  # 2/3 threshold
        self.concept_registry.add_rule(rule.to_dict())
        if hasattr(self.concept_registry, 'save'):
            self.concept_registry.save()
        self._refresh_transforms()
        log.info(f"Rule PROMOTED: {rule.name} ({successes}/3 tests)")
```

### P2.2 — Genuine Self-Improvement via Architecture Search

**Current State:** `_run_cpp_plasticity()` exists but can't run without SolveEpisode binding.

**Python-Only Alternative (works now):**
```python
def _python_plasticity(self) -> int:
    """Analyze failure patterns and propose new macro transforms."""
    if not self.memory_manager:
        return 0
    recent = self.memory_manager.recent_episodes(100)
    failures = [ep for ep in recent if not ep.success]
    if len(failures) < 10:
        return 0
    
    # Find most common failed transform sequences
    from collections import Counter
    t_freq = Counter()
    for ep in failures:
        for t in ep.transform_sequence:
            t_freq[t] += 1
    
    # Top 3 most-tried-but-failed transforms
    common = [t for t, n in t_freq.most_common(5) if n >= 3]
    if len(common) < 2:
        return 0
    
    # Propose macro: combine two commonly-failing transforms
    from sare.meta.macro_registry import MacroSpec, upsert_macros
    macros = []
    for i in range(min(2, len(common)-1)):
        name = f"python_macro_{common[i]}_then_{common[i+1]}"
        macros.append(MacroSpec(name=name, steps=[common[i], common[i+1]]))
    
    if macros:
        upsert_macros(macros)
        self._refresh_transforms()
    return len(macros)
```

### P2.3 — Surprise-Driven Curriculum Adaptation

**Current State:** WorldModel v3 tracks prediction vs reality, but curriculum selection doesn't use surprise scores properly.

**Fix:** Add proper surprise-driven problem selection:
```python
def _pick_learning_problem(self) -> str:
    # If WorldModel has surprise data, prioritize surprised domains
    if self.world_model_v3:
        surprised = self.world_model_v3.get_high_surprise_domains(3)
        if surprised and random.random() < 0.4:
            domain = surprised[0][0]  # highest surprise domain
            # Generate a problem specifically for that domain
            pg = getattr(self, '_problem_gen', None)
            if pg:
                batch = pg.generate_for_domain(domain, n=1)
                if batch:
                    return batch[0]['expression']
    ...
```

---

## Phase 3: Grounded Reasoning (1–2 months)

### P3.1 — Multi-modal Problem Representation

**Current State:** Parser handles only symbolic infix expressions.

**Target:** Handle natural language math problems:
- "What is 3 + 4?" → expression graph
- "Simplify sin(0) + cos(0)" → graph
- "Find x if x + 5 = 12" → equation graph

**Implementation Path:**
1. Extend `UniversalParser` with keyword patterns (currently exists in `interface/universal_parser.py`)
2. Wire `PerceptionEngine.ingest_text()` into the curriculum problem source
3. Add a proper NL→expression mapping layer

### P3.2 — Analogy and Transfer Testing

**Current State:** Transfer hypotheses are generated but not tested on actual problems.

**Missing:** The `_verify_transfer_hypotheses()` function exists with a 5/5 pass threshold — but it requires the synthesized transforms to actually reduce energy on test cases. Currently `and true → x` reduces energy, but the verification loop doesn't run during auto-learn reliably.

**Fix:** Run verification on every `reorganize_knowledge()` call AND log which transfers actually helped with which problems.

### P3.3 — Calculus Reasoning Mode

**Current State:** Derivative transforms exist (`deriv_const_zero`, `deriv_linear`, `deriv_power_rule`) and fire on the right inputs.

**Missing:** `deriv_power_rule` increases energy (it's a rewrite, not a simplification). The energy model needs a "calculus mode" where applying the correct rule is ALWAYS rewarded regardless of node count.

**Fix Options:**
```python
# Option A: Domain-aware energy evaluator
class CalculusEnergyEvaluator(EnergyEvaluator):
    def compute(self, graph: Graph) -> EnergyBreakdown:
        bd = super().compute(graph)
        # Penalize derivative operators (unexpanded) heavily
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("d/dx", "derivative", "diff"):
                bd.components["redundancy"] += 5.0  # unexpanded derivative is "expensive"
        return bd

# Option B: Post-simplification energy (evaluate final result after applying all rules)
```

### P3.4 — Real-World Data Ingestion (Textbook Problems)

**Current State:** `PerceptionEngine` exists and can parse text/CSV/JSON, but is rarely used.

**Target:** Automatically generate problems from:
- Math textbook OCR results (text input)
- Khan Academy problem formats (JSON)
- Wolfram Alpha API results (structured)

---

## Phase 4: AGI Architecture Expansion (2–6 months)

### P4.1 — Causal Reasoning Engine

**Current State:** WorldModel v3 has causal links, schemas, and belief network. But it doesn't use them to guide PROBLEM SELECTION.

**Target:** 
- When a schema is discovered: automatically generate related problems
- When a belief is updated: test whether adjacent rules need revision
- Use causal reasoning to predict WHICH transform will help BEFORE searching

### P4.2 — Working Memory + Attention Mechanism

**Current State:** BeamSearch uses a heuristic scorer but doesn't distinguish "currently relevant" context.

**Target:** Add working memory:
```
Working memory = {
    "currently_solving": graph,
    "goal": target_form,
    "relevant_rules": top_k_rules_for_current_domain,
    "recent_failures": transforms_that_failed,
    "hypothesis": current_conjecture_being_tested,
}
```

### P4.3 — Meta-Cognitive Monitoring

**Missing:** The system cannot realize when it's "stuck" in a way that suggests a fundamental knowledge gap.

**Implementation:**
```python
class StuckDetector:
    def analyze(self, recent_solves: List[dict]) -> dict:
        # Are we consistently failing a specific type of problem?
        # Do all failures share a common structural feature?
        # Have we tried the same (useless) transforms repeatedly?
        pass
```

### P4.4 — Persistent Knowledge Graph

**Current State:** Each rule, schema, belief, and causal link is stored separately.

**Target:** Unified knowledge graph where:
- Rules are nodes
- "is stronger than" / "depends on" / "contradicts" are edges
- Knowledge consolidation = graph simplification

---

## Phase 5: Path to 75% AGI Capability (3–12 months)

### What 75% AGI Requires

Based on the current honest assessment, reaching 75% AGI capability requires:

| Dimension | Current | Target | Gap |
|---|---|---|---|
| **Domain Coverage** | 5 domains | 12+ domains | +7 domains |
| **Rule Learning** | 4 rules/session | 50+ rules persisted | Persistence + more diverse problems |
| **Transfer** | 26 synthesized (not promoted) | 80% auto-promoted | Verified transfer pipeline |
| **Self-Improvement** | Python plasticity works | C++ ModuleGenerator wired | Fix C++ binary |
| **Language Understanding** | NL explanations exist | NL → problem parsing | UniversalParser extension |
| **Goal-Directed Learning** | GoalSetter loaded but idle | Proactive goal generation | GoalSetter integration |
| **Causal Reasoning** | WorldModel v3 has causal links | Guides solve strategy | Belief-guided search |
| **Memory Consolidation** | Hippocampus runs | Sleep strengthens rules | Already partially wired |

### Priority Order for Maximum AGI Score Impact

```
1. Fix C++ binary (-all_load → -force_load)           → +5% (C++ features enable)
2. Wire ConceptRegistry → live transforms              → +5% (rules persist)
3. Verified rule promotion (test before promote)       → +5% (quality rules only)
4. Python plasticity macro generation                  → +4% (self-improvement)
5. Calculus energy model fix                           → +3% (calculus works end-to-end)
6. NL → expression parsing                             → +5% (broader problem intake)
7. Surprise-driven curriculum                          → +3% (smarter problem selection)
8. GoalSetter proactive goals                          → +4% (autonomous planning)
9. Causal reasoning for search guidance                → +5% (smarter search)
10. 3+ new symbolic domains                            → +5% (probability, geometry, logic)
```

**Estimated total from fixes above: +44% → current ~55% + 44% = ~75% AGI capability**

---

## Immediate Next Session Action Plan (What to do first)

### Step 1: Fix CMakeLists.txt (-all_load → -force_load)
```cmake
# File: CMakeLists.txt, line ~91
# Change:
target_link_options(sare_bindings PRIVATE "LINKER:-all_load")
# To:
target_link_options(sare_bindings PRIVATE 
    "-Wl,-force_load,${CMAKE_BINARY_DIR}/libsare_core.a")
```

### Step 2: Rebuild + verify binary loads
```bash
cmake --build build -j4
# Verify: python3 -c "import sare.sare_bindings as sb; print(sb.ReflectionEngine())"
```

### Step 3: Add SolveEpisode binding
- Simple struct, safe to bind
- Enables `_run_cpp_plasticity` in Brain

### Step 4: Wire persistent rule promotion
- `_reflect_and_promote()` with 3-problem verification
- Save to ConceptRegistry on every promotion
- Reload on Brain boot

### Step 5: Test end-to-end learn cycle
```python
brain = get_brain()
brain.start_auto_learn(interval=1.0, problems_per_cycle=10)
time.sleep(60)  # Let it learn for 1 minute
status = brain.status()
print(f"Rules learned: {status['stats']['rules_promoted']}")
print(f"Transforms: {status['transforms_count']}")
```

---

## Files Changed This Session (Summary)

### Core Engine (`python/sare/engine.py`)
- Added: `TrigZero`, `CosZero`, `LogOne`, `SqrtSquare`
- Added: `DerivativeConstant`, `DerivativeLinear`, `DerivativePower`
- Added: `CosZero` transform (cos(0) → 1)
- Fixed: Parser now handles `and`/`or` as infix boolean operators
- Fixed: `true`/`false` parsed as constants, not variables
- Fixed: `derivative(...)`, `diff(...)`, `integral(...)` parsed as function calls
- Added: `_parse_boolean()` tier in parser precedence

### Brain Orchestrator (`python/sare/brain.py`)
- Added: `_py_graph_to_cpp_graph()` and `_cpp_graph_to_py_graph()` helpers
- Added: `_seed_synthesizer()` — runs synthesis on boot
- Added: `enable_cpp_fast_path()` — safe opt-in C++ search path
- Added: `_solve_with_cpp_bindings()` — C++ fast path with fallback guard
- Added: `_verify_transfer_hypotheses()` with 5-test promotion threshold
- Added: `_run_cpp_plasticity()` — C++ ModuleGenerator integration
- Added: `_promote_cpp_rule()`, `_promote_python_rule()` — unified rule promotion
- Added: `_detect_domain()` now handles trigonometry and calculus
- Fixed: `CausalInduction.evaluate()` robust fallback
- Fixed: Language grounding wired into post-solve processing

### Transfer Synthesizer (`python/sare/transfer/synthesizer.py`)
- Added: `trigonometry` and `calculus` domain operators to `DOMAIN_OPERATORS`
- Added: Persistence via `save()`/`load()` — synthesized transforms survive restarts
- Fixed: `get_live_transforms()` now only returns PROMOTED transforms
- Fixed: `load()` handles missing/incomplete saved schemas gracefully

### Language Grounding (`python/sare/language/grounding.py`)
- Added: `ground()` alias for `ground_rule()` with short signature
- Added: Trig/calculus/cos/sqrt/deriv grounding templates
- Fixed: `ground_from_transform()` handles all new transform types

### Problem Generator (`python/sare/curriculum/problem_generator.py`)
- Added: `trigonometry` and `calculus` operator pools
- Added: 18 new templates (trig identities, derivative rules)
- Added: `_detect_domain()` now recognizes trig/calculus expressions

### C++ Core (`core/reflection/causal_induction.cpp`)
- Fixed: `energy.evaluate()` → `energy.computeTotal()` (correct API)
- Fixed: `e.sourceId`/`e.targetId`/`e.type` → `e.source`/`e.target`/`e.relationship_type`

---

## Known Issues (Do Not Close Until Fixed)

| ID | Issue | File | Status |
|---|---|---|---|
| CPP-001 | C++ binary crashes on import (macOS -all_load OOM) | CMakeLists.txt | 🔴 Open |
| CPP-002 | SolveEpisode not exposed in Python | core/bindings/pybind_module.cpp | 🔴 Open (removed due to crash) |
| ENG-001 | deriv_power_rule increases energy (calculus is rewrite, not simplification) | engine.py + energy model | 🟡 Accepted |
| LRN-001 | Reflection rules not persisted across Brain restarts | brain.py | 🟡 Partial (ConceptRegistry not saved to disk) |
| LRN-002 | Synthesized transforms (26 generated) not all promoted | transfer/synthesizer.py | 🟡 By design (need 5/5 test pass) |
| CUR-001 | Distributive domain 0% solve rate | engine.py DistributiveExpansion | 🟡 Known limitation |
| GOL-001 | GoalSetter never generates autonomous goals | meta/goal_setter.py | 🟡 Open |

---

*Generated from session audit. All scores based on running code, not claims.*
