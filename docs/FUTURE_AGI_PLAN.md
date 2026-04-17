# SARE-HX: Future AGI Plan
## System Re-Evaluation + Roadmap + Ollama/Qwen3.5 World Model Integration
*Last updated: 2026-03-15*

---

## Part 1 — Honest System Re-Evaluation

### What's Actually Working (Confirmed)

| Capability | Status | Evidence |
|---|---|---|
| Symbolic reasoning (algebra, logic, arithmetic) | ✅ SOLID | 180/180 benchmark (100%) |
| Self-learning loop | ✅ SOLID | 28/28 rules promoted, 100% solve rate |
| CausalInduction rule validation | ✅ SOLID | "4/4 cases passed" per rule |
| World model predictions | ✅ SOLID | Prediction accuracy rising from 6% → 100% |
| Transform synthesis (LLM-written transforms) | ✅ WIRED | 3 LLM synthesized transforms confirmed |
| Domain auto-detection | ✅ SOLID | algebra/logic/arithmetic inferred from graph ops |
| Homeostasis / drives | ✅ WIRED | explore/consolidate/deepen recommendations |
| Autobiographical memory | ✅ WIRED | Records solve events, narrative generation |
| Theory of Mind | ✅ BASIC | 3/3 false-belief tests pass |
| Multimodal perception | ✅ WIRED | image/table/text → graph pipeline |
| Conversational learning (Chat/Teach) | ✅ WIRED | Rule injection via dialogue |
| Long-horizon planning | ✅ WIRED | PlannerStateMachine, multi-step goals |
| Cross-domain analogy transfer | ✅ WIRED | algebra → logic pattern transfer |

### What's Weak or Missing

| Gap | Severity | Root Cause |
|---|---|---|
| Language understanding | HIGH | Symbolic only; no real NL grounding |
| World model breadth | HIGH | 43 causal links manually seeded; not learned from experience |
| Creativity / novel synthesis | HIGH | Transforms and rules bounded by what LLM writes |
| Real reasoning beyond algebra | HIGH | Physics, chemistry, history — all unsupported |
| Grounded sensorimotor loop | MEDIUM | ToyMathEnv is toy; no real environment |
| Social intelligence | MEDIUM | False-belief tests but no real dialogue partner |
| Metacognition accuracy | MEDIUM | Self-model scores don't reflect real capability |
| LLM integration (local) | MEDIUM | Gemini-only; no local Ollama support — **fixed in this session** |
| Transform synthesis trigger | LOW | Synthesizer wired but warm-up is manual |
| Domain discovery | LOW | Clusterer wired but not yet auto-triggered |

### AGI Score (Honest, March 2026)

| Dimension | Score | Notes |
|---|---|---|
| Narrow problem-solving | 85% | Algebra/logic at near-human level |
| Self-improvement | 55% | Loop works; bounded by transform space |
| Language & NL | 20% | LLM bridge exists; not grounded in meaning |
| General reasoning | 25% | Only symbolic; no causal chains beyond algebra |
| Social / emotional | 30% | ToM wired; no real social loop |
| Creativity | 20% | LLM synthesis works; constrained space |
| Embodiment | 15% | ToyMathEnv only |
| **Overall** | **~37%** | Specialized cognitive system, not general |

---

## Part 2 — Ollama / Qwen3.5 World Model Integration

### What Was Done (This Session)

1. **`llm_bridge.py`** — Added `_call_ollama()` function calling `http://localhost:11434/api/generate`
2. **`_call_llm()`** — Dispatches to Ollama when `provider == "ollama"` (no API key needed)
3. **`llm_available()` / `llm_status()`** — Ollama-aware: pings `/api/tags` to check reachability
4. **`configs/llm.json`** — Switched to `"provider": "ollama", "model": "qwen3.5:2b"`

Installed locally: `qwen3.5:2b` (2.7GB) and `qwen3.5:latest` (6.6GB)

### Current LLM Usage in SARE-HX

| Use Case | File | Quality with qwen3.5:2b |
|---|---|---|
| Parse NL problem → expression | `llm_bridge.py:parse_nl_problem()` | Good (simple JSON extraction) |
| Generate proof explanation | `llm_bridge.py:explain_solve_trace()` | Good (structured prompt) |
| Plan subgoals | `llm_bridge.py:plan_subgoals()` | Adequate (short list) |
| Synthesize new Transforms | `learning/transform_synthesizer.py` | Needs qwen3.5:latest for code quality |
| Domain naming (discovery) | `curiosity/domain_discoverer.py` | Good (naming task) |
| Chat/teach dialogue | `interface/web.py:_api_chat()` | Adequate |

### Recommended Model Selection Strategy

```json
{
    "provider": "ollama",
    "model": "qwen3.5:2b",
    "synthesis_model": "qwen3.5:latest",
    "synthesis_model_threshold": "hard_problems"
}
```

Use `qwen3.5:2b` for fast tasks (parsing, naming, explanation) and `qwen3.5:latest` for code synthesis (Transform classes). This needs a `_call_llm(prompt, model_override=None)` API change in `llm_bridge.py`.

### Phase W — World Model × LLM Integration

The current `WorldModel` (`python/sare/memory/world_model.py`) uses hand-coded causal links. Upgrading it to use Qwen3.5 for hypothesis generation would dramatically improve its predictive capability.

#### W1: LLM Hypothesis Generation (2 weeks)

**File:** `python/sare/memory/world_model.py`

When the world model encounters a surprise (predicted_delta vs actual_delta mismatch > threshold), call the LLM to generate an explanatory hypothesis:

```python
def _generate_hypothesis_with_llm(self, domain, transform_name, expected, actual):
    prompt = (
        f"In {domain} math, applying transform '{transform_name}' was expected to reduce "
        f"complexity by {expected:.1f} but actually reduced it by {actual:.1f}. "
        f"Write ONE hypothesis explaining why, as a JSON: "
        f'{{ "hypothesis": "...", "pattern": "...", "confidence": 0.0-1.0 }}'
    )
    raw = _call_llm(prompt)
    return json.loads(raw)
```

Store hypotheses in `data/memory/world_model_hypotheses.json`, accumulate evidence, graduate high-confidence hypotheses to causal links.

#### W2: LLM-Assisted Schema Learning (3 weeks)

**File:** `python/sare/memory/world_model.py`

After 10+ experiments in a domain, call LLM to synthesize a domain schema:

```python
def learn_schema_from_llm(self, domain, recent_successes):
    prompt = (
        f"Given these {domain} simplifications:\n"
        + "\n".join(f"  {s['before']} → {s['after']} via {s['transform']}" for s in recent_successes)
        + "\nWrite a JSON schema: { \"rules\": [...], \"patterns\": [...] }"
    )
    # Store as domain schema, seed future predictions
```

#### W3: Contradiction Detection via LLM (1 week)

When two promoted rules conflict (e.g., `x * 1 → x` and a new rule says `x * 1 → x * 1`), call LLM to adjudicate:

```python
def check_consistency_with_llm(self, rule_a, rule_b):
    prompt = f"Are these rules consistent? Rule A: '{rule_a}'. Rule B: '{rule_b}'. Answer JSON: {{\"consistent\": bool, \"reason\": \"...\"}}"
```

#### W4: World Model API Endpoints (1 week)

Add to `web.py`:
- `POST /api/world/hypothesize` — trigger LLM hypothesis generation for a domain
- `GET /api/world/hypotheses` — list all pending hypotheses + evidence counts
- `POST /api/world/schema/learn` — manually trigger schema synthesis for a domain
- `GET /api/world/consistency` — run contradiction check across all promoted rules

---

## Part 3 — Full Future AGI Roadmap

### Phase A — Language Grounding (6-8 weeks)

**Goal:** Move beyond symbolic math to real NL understanding.

| Task | File | Description |
|---|---|---|
| A1 | `perception/sentence_graph_builder.py` | Dependency parse → semantic graph (already exists, needs work) |
| A2 | `llm_bridge.py` | `extract_concepts(text)` → list of concept nodes + relations |
| A3 | `memory/concept_memory.py` | Link LLM concepts to ConceptRegistry rules |
| A4 | `agent/qa_pipeline.py` | Upgrade: use world model beliefs in QA answers |
| A5 | `web.py` | `/api/qa` endpoint uses grounded concepts |

**Metric:** Answer 10 simple factual questions correctly using world model beliefs.

### Phase B — Grounded World Model (8-10 weeks)

**Goal:** World model learns from text/experience, not just symbolic experiments.

| Task | File | Description |
|---|---|---|
| B1 | `memory/world_model.py` | W1-W4 above (LLM hypothesis + schema) |
| B2 | `perception/text_ingester.py` (NEW) | Read Wikipedia paragraphs → extract causal facts |
| B3 | `memory/world_model.py` | `ingest_text(text, domain)` → add beliefs |
| B4 | `curiosity/domain_discoverer.py` | Auto-trigger on 5+ consecutive stuck problems |
| B5 | `web.py` | `/api/world/ingest` endpoint |

**Metric:** World model has 200+ beliefs learned from experience (not seeded).

### Phase C — Multi-Domain Reasoning (10-12 weeks)

**Goal:** Reason across algebra, geometry, physics, basic chemistry.

| Task | File | Description |
|---|---|---|
| C1 | `transforms/physics_transforms.py` (NEW) | F=ma, energy conservation, kinematics |
| C2 | `transforms/geometry_transforms.py` (NEW) | Pythagorean theorem, angle sum, similar triangles |
| C3 | `curiosity/curriculum_generator.py` | Multi-domain ZPD with domain-specific seeds |
| C4 | `causal/analogy_transfer.py` | Transfer proofs from algebra → physics (F=ma ↔ v=s/t) |
| C5 | `benchmarks/physics/kinematics.json` (NEW) | 50 physics simplification problems |
| C6 | `benchmarks/geometry/triangles.json` (NEW) | 50 geometry problems |

**Metric:** 70%+ on physics and geometry benchmarks.

### Phase D — Improved Self-Improvement (4-6 weeks)

**Goal:** Transform synthesis fires automatically on real bottlenecks.

| Task | File | Description |
|---|---|---|
| D1 | `learning/transform_synthesizer.py` | Auto-trigger on 3+ consecutive stuck + same error pattern |
| D2 | `llm_bridge.py` | Use `qwen3.5:latest` for synthesis (vs :2b for parsing) |
| D3 | `curiosity/experiment_runner.py` | Track "stuck pattern" signature across problems |
| D4 | `meta/bottleneck_analyzer.py` | Feed bottleneck report into synthesizer prompt |
| D5 | `learning/transform_validator.py` (NEW) | Run synthesized transforms against 20 known-correct cases before accepting |

**Metric:** Synthesizer fires and creates a valid transform without human intervention.

### Phase E — Real Social Intelligence (8-10 weeks)

**Goal:** SARE can learn from multi-turn dialogue, not just single teach commands.

| Task | File | Description |
|---|---|---|
| E1 | `social/dialogue_manager.py` | Multi-turn context (currently single-turn) |
| E2 | `social/theory_of_mind.py` | Build user model: track what user knows/believes |
| E3 | `web.py:_api_chat()` | Use user model to personalize explanations |
| E4 | `social/dialogue_manager.py` | Clarification requests when ambiguous |
| E5 | `memory/autobiographical.py` | Record dialogue episodes, reference in future sessions |

**Metric:** 3-turn dialogue where SARE correctly models what the user already knows and adjusts explanation depth.

### Phase F — Genuine Creativity (12-16 weeks)

**Goal:** SARE generates novel conjectures it hasn't seen before.

| Task | File | Description |
|---|---|---|
| F1 | `curiosity/conjecture_generator.py` (NEW) | LLM generates "what if" variants of known rules |
| F2 | `causal/induction.py` | Test conjectures against 10 cases before promoting |
| F3 | `meta/proof_builder.py` | Attempt to prove conjectures using existing transforms |
| F4 | `web.py` | `/api/conjectures` endpoint — list + test conjectures |
| F5 | `memory/concept_memory.py` | Grade conjectures by novelty (cosine distance from known rules) |

**Metric:** Generate 1 correct novel mathematical identity not in any seed or transform list.

### Phase G — Metacognition Accuracy (4-6 weeks)

**Goal:** SARE's self-assessment matches its actual capability.

| Task | File | Description |
|---|---|---|
| G1 | `memory/self_model.py` | Track per-domain accuracy → update self_model.json |
| G2 | `meta/homeostasis.py` | Drive calibration: compare reported score vs benchmark score |
| G3 | `memory/identity.py` | Narrative reflects honest capability, not aspirational |
| G4 | `web.py` | `/api/identity` returns calibrated capability profile |

**Metric:** `/api/identity` AGI score within 5% of measured benchmark scores.

---

## Part 4 — Priority Order (Next 6 Months)

```
Month 1:  W1-W4 (World Model × LLM)        ← Highest leverage, infrastructure
Month 2:  D1-D5 (Auto-synthesis)            ← Closes self-improvement gap
Month 3:  A1-A5 (Language grounding)        ← Foundation for everything else
Month 4:  B1-B5 (Grounded world model)      ← Compounds with A
Month 5:  C1-C6 (Multi-domain)              ← Expands reasoning breadth
Month 6:  E1-E5 or F1-F5                   ← Social or Creative, based on results
```

**One-line summary of the northstar:** SARE-HX becomes genuinely general when its world model is grounded in language, its transforms synthesize themselves on bottlenecks, and its social loop can learn from a human partner in real-time. The Ollama/Qwen3.5 local LLM makes all three of these possible without external API dependency.

---

## Part 5 — Immediate Next Steps (This Week)

1. **W1**: Add `_generate_hypothesis_with_llm()` to `world_model.py` — triggered on surprise > 2.0
2. **D1**: Wire auto-trigger in `transform_synthesizer.py` — fire on 5 consecutive stuck problems with same domain
3. **D2**: Add `synthesis_model` config key to `llm.json`, use in `transform_synthesizer.py`
4. **B4**: Auto-trigger `domain_discoverer.py` after 5 stuck problems
5. Verify `qwen3.5:latest` synthesis quality vs `:2b` — run one synthesis comparison
