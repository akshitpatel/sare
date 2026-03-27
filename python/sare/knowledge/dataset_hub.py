"""
DatasetHub — LLMDataHub Integration for General Intelligence

SARE-HX is not just a reasoning engine — it's building general intelligence.
This module integrates open datasets from LLMDataHub and HuggingFace into
every learning dimension:

  1. Language Training     → LanguageGrounding, InternalGrammar, DialogueManager
  2. World Knowledge       → KnowledgeIngester → ConceptGraph
  3. Reasoning Problems    → Curriculum seed library, ProblemGenerator
  4. Social Understanding  → DialogueManager (conversation data)
  5. Causal Knowledge      → WorldModel (cause-effect patterns)
  6. Chain-of-Thought      → Reflection engine (step-by-step reasoning)

Supported dataset sources (from LLMDataHub):
  - HuggingFace datasets (streamed, no full download needed)
  - Local JSON/JSONL files
  - Built-in curated knowledge packs

Usage:
    hub = DatasetHub()
    hub.ingest_all(brain)           # feed everything into Brain subsystems
    hub.ingest_language(brain)      # language training only
    hub.ingest_knowledge(brain)     # world knowledge only
    hub.ingest_reasoning(brain)     # math/logic problems only
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "memory"

# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset Registry — curated entries from LLMDataHub + custom packs
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DatasetEntry:
    """A dataset that can be ingested for a specific AGI dimension."""
    name: str
    source: str                     # "huggingface", "local", "builtin"
    hf_id: str = ""                 # HuggingFace dataset ID
    url: str = ""
    dimensions: List[str] = field(default_factory=list)  # AGI dimensions it improves
    domain: str = "general"
    format: str = "json"            # json, jsonl, text, conversation
    description: str = ""
    priority: float = 0.5           # 0-1, higher = ingest first

    def to_dict(self) -> dict:
        return {
            "name": self.name, "source": self.source,
            "hf_id": self.hf_id, "dimensions": self.dimensions,
            "domain": self.domain, "priority": self.priority,
        }


# Curated datasets from LLMDataHub most relevant to AGI
_DATASET_REGISTRY: List[DatasetEntry] = [
    # ── Reasoning & Math ─────────────────────────────────────────────────────
    DatasetEntry(
        name="PRM800K",
        source="huggingface", hf_id="openai/prm800k",
        dimensions=["problem_solving", "causal_reasoning", "chain_of_thought"],
        domain="mathematics",
        description="Step-by-step math reasoning with process rewards",
        priority=0.9,
    ),
    DatasetEntry(
        name="TheoremQA",
        source="huggingface", hf_id="wenhu/TheoremQA",
        dimensions=["problem_solving", "world_knowledge", "causal_reasoning"],
        domain="mathematics",
        description="Theorem-based QA across math, physics, CS, finance",
        priority=0.85,
    ),
    DatasetEntry(
        name="SymbolicInstructionTuning",
        source="huggingface", hf_id="sail/symbolic-instruction-tuning",
        dimensions=["problem_solving", "language", "transfer"],
        domain="symbolic",
        description="Symbolic manipulation instructions with natural language",
        priority=0.9,
    ),
    DatasetEntry(
        name="AlpacaCoT",
        source="huggingface", hf_id="QingyiSi/Alpaca-CoT",
        dimensions=["chain_of_thought", "language", "causal_reasoning"],
        domain="general",
        description="Chain-of-thought instruction data across domains",
        priority=0.8,
    ),

    # ── Language & Communication ─────────────────────────────────────────────
    DatasetEntry(
        name="ShareGPT52K",
        source="huggingface", hf_id="RyokoAI/ShareGPT52K",
        dimensions=["language", "social_intelligence", "dialogue"],
        domain="conversation",
        description="Human-AI dialogue covering diverse topics",
        priority=0.7,
    ),
    DatasetEntry(
        name="OASST1",
        source="huggingface", hf_id="OpenAssistant/oasst1",
        dimensions=["language", "social_intelligence", "dialogue"],
        domain="conversation",
        description="Human-annotated multi-turn assistant conversations",
        priority=0.75,
    ),
    DatasetEntry(
        name="DollyInstruct",
        source="huggingface", hf_id="databricks/databricks-dolly-15k",
        dimensions=["language", "world_knowledge", "open_ended"],
        domain="general",
        description="Human-authored instruction-response pairs across 8 categories",
        priority=0.7,
    ),

    # ── World Knowledge ──────────────────────────────────────────────────────
    DatasetEntry(
        name="ProofPile",
        source="huggingface", hf_id="hoskinson-center/proof-pile",
        dimensions=["world_knowledge", "problem_solving", "concept_formation"],
        domain="mathematics",
        description="Mathematical proofs and formal reasoning texts",
        priority=0.8,
    ),
    DatasetEntry(
        name="ELI5",
        source="huggingface", hf_id="eli5",
        dimensions=["language", "world_knowledge", "social_intelligence"],
        domain="general",
        description="Explain Like I'm 5 — simplified explanations of complex topics",
        priority=0.7,
    ),

    # ── Code & Algorithms ────────────────────────────────────────────────────
    DatasetEntry(
        name="CodeAlpaca120K",
        source="huggingface", hf_id="iamtarun/code_instructions_120k_alpaca",
        dimensions=["problem_solving", "language", "world_knowledge"],
        domain="computer_science",
        description="Code instruction-response pairs for algorithmic reasoning",
        priority=0.65,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Built-in Knowledge Packs — Curated data that feeds all AGI dimensions
#  These don't require network access; they're baked into the module.
# ═══════════════════════════════════════════════════════════════════════════════

# Language training: sentence patterns → structural understanding
_LANGUAGE_TRAINING_DATA: List[dict] = [
    # Causal reasoning patterns
    {"text": "If you heat water to 100°C, it boils.", "domain": "physics",
     "type": "causal", "cause": "heat water to 100°C", "effect": "it boils"},
    {"text": "When supply exceeds demand, prices fall.", "domain": "economics",
     "type": "causal", "cause": "supply exceeds demand", "effect": "prices fall"},
    {"text": "Doubling the force doubles the acceleration.", "domain": "physics",
     "type": "causal", "cause": "doubling force", "effect": "doubles acceleration"},
    {"text": "Adding a catalyst lowers the activation energy.", "domain": "chemistry",
     "type": "causal", "cause": "adding catalyst", "effect": "lowers activation energy"},
    {"text": "Increasing interest rates reduces borrowing.", "domain": "economics",
     "type": "causal", "cause": "increasing interest rates", "effect": "reduces borrowing"},

    # Definitional patterns
    {"text": "A prime number is a natural number greater than 1 with no positive divisors other than 1 and itself.",
     "domain": "mathematics", "type": "definition", "concept": "prime_number"},
    {"text": "Entropy is a measure of the disorder or randomness in a system.",
     "domain": "physics", "type": "definition", "concept": "entropy"},
    {"text": "An algorithm is a finite sequence of well-defined instructions to solve a class of problems.",
     "domain": "computer_science", "type": "definition", "concept": "algorithm"},
    {"text": "Photosynthesis is the process by which plants convert light energy into chemical energy.",
     "domain": "biology", "type": "definition", "concept": "photosynthesis"},
    {"text": "A derivative measures the rate at which a function changes as its input changes.",
     "domain": "calculus", "type": "definition", "concept": "derivative"},
    {"text": "Democracy is a system of government where citizens exercise power by voting.",
     "domain": "political_science", "type": "definition", "concept": "democracy"},
    {"text": "Natural selection is the process where organisms with favorable traits are more likely to reproduce.",
     "domain": "biology", "type": "definition", "concept": "natural_selection"},
    {"text": "Opportunity cost is the value of the next best alternative foregone.",
     "domain": "economics", "type": "definition", "concept": "opportunity_cost"},

    # Analogical reasoning patterns
    {"text": "The nucleus of an atom is like the sun in our solar system — everything orbits around it.",
     "domain": "physics", "type": "analogy",
     "source": "solar system", "target": "atom", "mapping": "center of orbit"},
    {"text": "A cell membrane is like a security checkpoint — it controls what goes in and out.",
     "domain": "biology", "type": "analogy",
     "source": "security checkpoint", "target": "cell membrane", "mapping": "access control"},
    {"text": "An electric circuit is like a water pipe system — current flows like water, resistance is like a narrow pipe.",
     "domain": "physics", "type": "analogy",
     "source": "water pipes", "target": "electric circuit", "mapping": "flow and resistance"},
    {"text": "DNA is like a blueprint — it contains the instructions for building an organism.",
     "domain": "biology", "type": "analogy",
     "source": "blueprint", "target": "DNA", "mapping": "construction instructions"},
    {"text": "A computer's RAM is like a desk — it holds what you're currently working on, but clears when you stop.",
     "domain": "computer_science", "type": "analogy",
     "source": "desk", "target": "RAM", "mapping": "temporary workspace"},

    # Quantitative reasoning
    {"text": "If a car travels at 60 km/h for 2 hours, it covers 120 km.",
     "domain": "physics", "type": "quantitative",
     "formula": "distance = speed * time", "values": {"speed": 60, "time": 2, "distance": 120}},
    {"text": "A 10% discount on a $50 item saves $5, making the price $45.",
     "domain": "arithmetic", "type": "quantitative",
     "formula": "discount = price * rate", "values": {"price": 50, "rate": 0.1, "discount": 5}},
    {"text": "If you invest $1000 at 5% annual interest, after one year you have $1050.",
     "domain": "economics", "type": "quantitative",
     "formula": "amount = principal * (1 + rate)", "values": {"principal": 1000, "rate": 0.05}},

    # Logical reasoning patterns
    {"text": "All mammals are warm-blooded. A whale is a mammal. Therefore, a whale is warm-blooded.",
     "domain": "logic", "type": "syllogism",
     "major": "All mammals are warm-blooded", "minor": "A whale is a mammal",
     "conclusion": "A whale is warm-blooded"},
    {"text": "If it rains, the ground gets wet. The ground is wet. We cannot conclude it rained — something else may have caused it.",
     "domain": "logic", "type": "fallacy", "name": "affirming_the_consequent"},
    {"text": "No reptiles are warm-blooded. Some animals are reptiles. Therefore, some animals are not warm-blooded.",
     "domain": "logic", "type": "syllogism",
     "major": "No reptiles are warm-blooded", "minor": "Some animals are reptiles",
     "conclusion": "Some animals are not warm-blooded"},

    # Social / Theory of Mind
    {"text": "Alice thinks the ball is in the basket, but Bob moved it to the box while she wasn't looking.",
     "domain": "social", "type": "theory_of_mind",
     "belief_holder": "Alice", "false_belief": "ball is in basket", "reality": "ball is in box"},
    {"text": "When someone says 'Nice weather we're having' during a storm, they are being sarcastic.",
     "domain": "social", "type": "pragmatics", "literal": "nice weather", "intended": "bad weather"},
    {"text": "A teacher simplifies complex ideas for students because students have less prior knowledge.",
     "domain": "social", "type": "perspective_taking",
     "agent": "teacher", "adaptation": "simplification", "reason": "knowledge gap"},

    # Temporal reasoning
    {"text": "The Renaissance came after the Middle Ages and before the Industrial Revolution.",
     "domain": "history", "type": "temporal",
     "events": ["Middle Ages", "Renaissance", "Industrial Revolution"], "order": "sequential"},
    {"text": "Seeds must be planted before they can grow. Growth requires water and sunlight over time.",
     "domain": "biology", "type": "temporal",
     "events": ["planting", "watering", "growth"], "order": "sequential"},

    # Spatial reasoning
    {"text": "The heart is located in the chest, slightly to the left of center.",
     "domain": "biology", "type": "spatial",
     "object": "heart", "location": "chest, left of center"},
    {"text": "In a standard chessboard, the queen starts on her own color.",
     "domain": "games", "type": "spatial",
     "object": "queen", "rule": "starts on own color"},
]

# Dialogue training: multi-turn conversations for social intelligence
_DIALOGUE_TRAINING_DATA: List[dict] = [
    {
        "context": "teaching_math",
        "turns": [
            {"speaker": "human", "text": "What happens when you add zero to a number?"},
            {"speaker": "system", "text": "Adding zero to any number gives you the same number back. For example, 5 + 0 = 5. Zero is the identity element for addition."},
            {"speaker": "human", "text": "Is there something similar for multiplication?"},
            {"speaker": "system", "text": "Yes! Multiplying by one leaves any number unchanged. For example, 7 × 1 = 7. One is the identity element for multiplication. Notice the structural parallel: both are 'identity' operations."},
        ],
    },
    {
        "context": "teaching_logic",
        "turns": [
            {"speaker": "human", "text": "Explain De Morgan's law."},
            {"speaker": "system", "text": "De Morgan's law says: NOT (A AND B) = (NOT A) OR (NOT B). Negation distributes over conjunction by flipping AND to OR."},
            {"speaker": "human", "text": "Is there a version for OR?"},
            {"speaker": "system", "text": "Yes: NOT (A OR B) = (NOT A) AND (NOT B). The pattern is symmetric — negation always flips the operator."},
        ],
    },
    {
        "context": "teaching_physics",
        "turns": [
            {"speaker": "human", "text": "Why do heavy and light objects fall at the same speed?"},
            {"speaker": "system", "text": "In a vacuum, all objects accelerate at the same rate due to gravity (9.8 m/s²). While heavier objects have more gravitational force, they also have more inertia. The two effects cancel exactly: a = F/m = mg/m = g."},
            {"speaker": "human", "text": "But a feather falls slower than a rock in real life."},
            {"speaker": "system", "text": "That's due to air resistance, not gravity. Air resistance depends on surface area and shape, not mass. In a vacuum chamber, a feather and a bowling ball hit the ground at exactly the same time."},
        ],
    },
    {
        "context": "correcting_misconception",
        "turns": [
            {"speaker": "human", "text": "x times 0 equals x, right?"},
            {"speaker": "system", "text": "No — x times 0 equals 0, not x. Zero is the absorbing element for multiplication: it 'absorbs' whatever it multiplies. You may be thinking of x times 1 = x, where 1 is the identity element."},
            {"speaker": "human", "text": "Oh right, so 0 is special for addition and 1 is special for multiplication?"},
            {"speaker": "system", "text": "Exactly. 0 is the identity for addition (x + 0 = x) and the absorber for multiplication (x × 0 = 0). 1 is the identity for multiplication (x × 1 = x). Each operation has its own special elements."},
        ],
    },
    {
        "context": "socratic_questioning",
        "turns": [
            {"speaker": "human", "text": "What is the derivative of a constant?"},
            {"speaker": "system", "text": "Before I answer — what does a derivative measure?"},
            {"speaker": "human", "text": "The rate of change."},
            {"speaker": "system", "text": "Right. Now, does a constant change?"},
            {"speaker": "human", "text": "No, so the derivative must be zero!"},
            {"speaker": "system", "text": "Exactly. d/dx(c) = 0 because a constant has zero rate of change. You derived the rule yourself through reasoning."},
        ],
    },
]

# Chain-of-thought reasoning examples
_COT_TRAINING_DATA: List[dict] = [
    {
        "problem": "Simplify: (x + 3) + (2x - 1)",
        "domain": "algebra",
        "steps": [
            "Remove parentheses: x + 3 + 2x - 1",
            "Group like terms: (x + 2x) + (3 - 1)",
            "Combine like terms: 3x + 2",
        ],
        "answer": "3x + 2",
        "rules_used": ["associativity", "combine_like_terms", "constant_fold"],
    },
    {
        "problem": "Prove: NOT (A AND B) = (NOT A) OR (NOT B)",
        "domain": "logic",
        "steps": [
            "Consider all cases for A and B (truth table approach)",
            "When A=T, B=T: NOT(T AND T) = NOT(T) = F; (NOT T) OR (NOT T) = F OR F = F ✓",
            "When A=T, B=F: NOT(T AND F) = NOT(F) = T; (NOT T) OR (NOT F) = F OR T = T ✓",
            "When A=F, B=T: NOT(F AND T) = NOT(F) = T; (NOT F) OR (NOT T) = T OR F = T ✓",
            "When A=F, B=F: NOT(F AND F) = NOT(F) = T; (NOT F) OR (NOT F) = T OR T = T ✓",
            "All cases match, so the identity holds.",
        ],
        "answer": "Proved by exhaustive truth table",
        "rules_used": ["truth_table", "boolean_evaluation", "de_morgan"],
    },
    {
        "problem": "Find derivative: d/dx(3x² + 2x + 1)",
        "domain": "calculus",
        "steps": [
            "Apply linearity: d/dx(3x²) + d/dx(2x) + d/dx(1)",
            "Apply power rule to 3x²: 3 · 2 · x^(2-1) = 6x",
            "Apply power rule to 2x: 2 · 1 · x^(1-1) = 2",
            "Derivative of constant 1: 0",
            "Sum: 6x + 2 + 0 = 6x + 2",
        ],
        "answer": "6x + 2",
        "rules_used": ["linearity", "power_rule", "constant_derivative"],
    },
    {
        "problem": "If a train travels 120 km in 2 hours, what is its average speed?",
        "domain": "physics",
        "steps": [
            "Recall formula: speed = distance / time",
            "Substitute values: speed = 120 km / 2 h",
            "Compute: speed = 60 km/h",
        ],
        "answer": "60 km/h",
        "rules_used": ["speed_formula", "division"],
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
#  DatasetHub — Main integration class
# ═══════════════════════════════════════════════════════════════════════════════

class DatasetHub:
    """
    Integrates external datasets into every AGI dimension of SARE-HX.

    This is the bridge between the open dataset ecosystem (LLMDataHub,
    HuggingFace) and the internal learning subsystems (KnowledgeIngester,
    ConceptGraph, LanguageGrounding, DialogueManager, etc.).
    """

    def __init__(self):
        self._registry = list(_DATASET_REGISTRY)
        self._stats = {
            "knowledge_entries_ingested": 0,
            "language_entries_ingested": 0,
            "dialogue_sessions_ingested": 0,
            "cot_examples_ingested": 0,
            "reasoning_problems_ingested": 0,
            "hf_datasets_loaded": 0,
            "total_concepts_created": 0,
        }
        self._ingested_sources: List[str] = []

    # ── Master ingestion ─────────────────────────────────────────────────────

    def ingest_all(self, brain) -> dict:
        """Feed all built-in data packs into every Brain subsystem."""
        log.info("DatasetHub: Starting full AGI knowledge ingestion...")

        results = {}
        results["knowledge"] = self.ingest_knowledge(brain)
        results["language"] = self.ingest_language(brain)
        results["dialogue"] = self.ingest_dialogue(brain)
        results["cot"] = self.ingest_chain_of_thought(brain)
        results["reasoning"] = self.ingest_reasoning_problems(brain)
        results["hf"] = self.ingest_hf_datasets(brain)

        log.info(
            "DatasetHub: Ingestion complete — %d knowledge, %d language, "
            "%d dialogue, %d CoT, %d reasoning entries",
            self._stats["knowledge_entries_ingested"],
            self._stats["language_entries_ingested"],
            self._stats["dialogue_sessions_ingested"],
            self._stats["cot_examples_ingested"],
            self._stats["reasoning_problems_ingested"],
        )
        return results

    # ── 1. World Knowledge → KnowledgeIngester + ConceptGraph ────────────────

    def ingest_knowledge(self, brain) -> int:
        """Feed language training data into KnowledgeIngester as world knowledge."""
        count = 0
        ki = getattr(brain, "knowledge_ingester", None)
        cg = getattr(brain, "concept_graph", None)

        for entry in _LANGUAGE_TRAINING_DATA:
            text = entry["text"]
            domain = entry.get("domain", "general")
            entry_type = entry.get("type", "fact")
            concept = entry.get("concept", "")

            # Extract concept hints from the entry
            hints = []
            if concept:
                hints.append(concept)
            if entry_type == "causal":
                hints.append(f"causal_{domain}")
            if entry_type == "definition" and concept:
                hints.append(concept)
            if entry_type == "analogy":
                hints.append(entry.get("target", ""))
            if not hints:
                # Auto-extract key nouns
                words = re.findall(r'\b[a-z]{4,}\b', text.lower())
                hints = list(dict.fromkeys(words))[:3]

            # Feed to KnowledgeIngester
            if ki:
                try:
                    ki.ingest_text(
                        title=f"DatasetHub_{entry_type}_{domain}",
                        text=text,
                        domain=domain,
                        concept_hints=[h for h in hints if h],
                    )
                    count += 1
                except Exception as e:
                    log.debug(f"DatasetHub knowledge ingest failed: {e}")

            # Feed causal patterns to WorldModel
            if entry_type == "causal" and hasattr(brain, "world_model_v3") and brain.world_model_v3:
                try:
                    if hasattr(brain.world_model_v3, "add_causal_link"):
                        brain.world_model_v3.add_causal_link(
                            entry.get("cause", ""),
                            entry.get("effect", ""),
                            domain=domain,
                            confidence=0.8,
                        )
                except Exception:
                    pass

            # Feed analogies to transfer engine
            if entry_type == "analogy" and hasattr(brain, "transfer_engine") and brain.transfer_engine:
                try:
                    source = entry.get("source", "")
                    target = entry.get("target", "")
                    if source and target:
                        # Record as cross-domain observation
                        brain.transfer_engine.observe(
                            [f"analogy_{source}_to_{target}"],
                            domain, True,
                        )
                except Exception:
                    pass

        # Push accumulated knowledge to ConceptGraph
        if ki and cg:
            try:
                ki.feed_to_concept_graph(cg)
            except Exception:
                pass

        self._stats["knowledge_entries_ingested"] += count
        log.info(f"DatasetHub: Ingested {count} world knowledge entries")
        return count

    # ── 2. Language Training → LanguageGrounding + InternalGrammar ────────────

    def ingest_language(self, brain) -> int:
        """Feed language patterns into LanguageGrounding and InternalGrammar."""
        count = 0
        lg = getattr(brain, "language_grounding", None)
        grammar = getattr(brain, "internal_grammar", None)

        for entry in _LANGUAGE_TRAINING_DATA:
            text = entry["text"]
            domain = entry.get("domain", "general")
            entry_type = entry.get("type", "fact")

            # Ground definitions as formal concepts
            if entry_type == "definition" and lg:
                concept = entry.get("concept", "")
                if concept:
                    try:
                        lg.ground_rule(
                            rule_name=f"def_{concept}",
                            formal_rule=text[:80],
                            domain=domain,
                            structural_role="evaluation",
                            operator=concept,
                            element="",
                        )
                        count += 1
                    except Exception:
                        pass

            # Feed causal patterns as grammar rules
            if entry_type == "causal" and grammar:
                try:
                    cause = entry.get("cause", "")
                    effect = entry.get("effect", "")
                    if cause and effect:
                        # Create symbols for cause and effect
                        cause_sym = grammar.learn_new_symbol(
                            type("_S", (), {
                                "name": f"cause_{cause[:20].replace(' ', '_')}",
                                "domain": domain,
                            })()
                        )
                        effect_sym = grammar.learn_new_symbol(
                            type("_S", (), {
                                "name": f"effect_{effect[:20].replace(' ', '_')}",
                                "domain": domain,
                            })()
                        )
                        count += 1
                except Exception:
                    pass

            # Feed analogies as structural mapping knowledge
            if entry_type == "analogy" and lg:
                try:
                    source = entry.get("source", "")
                    target = entry.get("target", "")
                    mapping = entry.get("mapping", "")
                    if source and target:
                        lg.ground_rule(
                            rule_name=f"analogy_{source}_to_{target}",
                            formal_rule=f"{source} ≈ {target} via {mapping}",
                            domain=domain,
                            structural_role="commutativity",  # structural parallel
                            operator="analogy",
                            element=mapping,
                        )
                        count += 1
                except Exception:
                    pass

            # Feed syllogisms as logic training
            if entry_type == "syllogism" and lg:
                try:
                    conclusion = entry.get("conclusion", "")
                    major = entry.get("major", "")
                    if conclusion:
                        lg.ground_rule(
                            rule_name=f"syllogism_{conclusion[:30].replace(' ', '_')}",
                            formal_rule=f"{major} → {conclusion}",
                            domain="logic",
                            structural_role="equation_solving",
                            operator="implies",
                        )
                        count += 1
                except Exception:
                    pass

        self._stats["language_entries_ingested"] += count
        log.info(f"DatasetHub: Ingested {count} language training entries")
        return count

    # ── 3. Dialogue Training → DialogueManager ───────────────────────────────

    def ingest_dialogue(self, brain) -> int:
        """Feed conversation data into DialogueManager for social learning."""
        count = 0
        dm = getattr(brain, "dialogue_manager", None)
        if not dm:
            return 0

        for conv in _DIALOGUE_TRAINING_DATA:
            try:
                # Create a dialogue session from training data
                session_id = f"dataset_hub_{conv['context']}_{int(time.time())}"
                if hasattr(dm, "start_session"):
                    dm.start_session(session_id)

                for turn in conv["turns"]:
                    if turn["speaker"] == "human" and hasattr(dm, "human_says"):
                        dm.human_says(turn["text"], session_id=session_id)
                    elif turn["speaker"] == "system" and hasattr(dm, "system_says"):
                        dm.system_says(turn["text"], session_id=session_id)
                    elif hasattr(dm, "process_turn"):
                        dm.process_turn(turn["text"], speaker=turn["speaker"])

                count += 1
            except Exception as e:
                log.debug(f"DatasetHub dialogue ingest failed: {e}")

        self._stats["dialogue_sessions_ingested"] += count
        log.info(f"DatasetHub: Ingested {count} dialogue sessions")
        return count

    # ── 4. Chain-of-Thought → Reflection Engine ──────────────────────────────

    def ingest_chain_of_thought(self, brain) -> int:
        """Feed step-by-step reasoning traces into the system's knowledge."""
        count = 0
        ki = getattr(brain, "knowledge_ingester", None)
        cg = getattr(brain, "concept_graph", None)

        for cot in _COT_TRAINING_DATA:
            problem = cot["problem"]
            domain = cot.get("domain", "general")
            steps = cot.get("steps", [])
            answer = cot.get("answer", "")
            rules_used = cot.get("rules_used", [])

            # Feed the step-by-step trace as knowledge
            if ki:
                try:
                    trace_text = f"Problem: {problem}\n"
                    for i, step in enumerate(steps):
                        trace_text += f"Step {i+1}: {step}\n"
                    trace_text += f"Answer: {answer}"

                    ki.ingest_text(
                        title=f"CoT_{domain}_{problem[:30]}",
                        text=trace_text,
                        domain=domain,
                        concept_hints=rules_used[:5],
                    )
                    count += 1
                except Exception:
                    pass

            # Ground each used rule
            if cg:
                for rule_name in rules_used:
                    try:
                        cg.ground_example(
                            concept_name=rule_name,
                            text=f"Used in: {problem[:60]}",
                            operation="chain_of_thought",
                            inputs=[problem[:40]],
                            result=answer[:40],
                            domain=domain,
                            symbolic=answer,
                        )
                    except Exception:
                        pass

        # Push to ConceptGraph
        if ki and cg:
            try:
                ki.feed_to_concept_graph(cg)
            except Exception:
                pass

        self._stats["cot_examples_ingested"] += count
        log.info(f"DatasetHub: Ingested {count} chain-of-thought examples")
        return count

    # ── 5. Reasoning Problems → Curriculum ───────────────────────────────────

    def ingest_reasoning_problems(self, brain) -> int:
        """Extract solvable expressions from CoT data and add to curriculum."""
        count = 0

        for cot in _COT_TRAINING_DATA:
            domain = cot.get("domain", "general")
            problem = cot.get("problem", "")
            answer = cot.get("answer", "")

            # Extract the expression if it looks solvable
            expr = self._extract_solvable_expression(problem)
            if not expr:
                continue

            # Add to developmental curriculum
            dc = getattr(brain, "developmental_curriculum", None)
            if dc and hasattr(dc, "add_problem"):
                try:
                    dc.add_problem(expr, domain=domain, difficulty="medium")
                    count += 1
                except Exception:
                    pass

            # Add to problem generator seed library
            if hasattr(brain, "_problem_gen") and brain._problem_gen:
                try:
                    if hasattr(brain._problem_gen, "add_seed"):
                        brain._problem_gen.add_seed(expr, domain=domain)
                        count += 1
                except Exception:
                    pass

        self._stats["reasoning_problems_ingested"] += count
        log.info(f"DatasetHub: Ingested {count} reasoning problems into curriculum")
        return count

    # ── 6. HuggingFace Dataset Streaming ─────────────────────────────────────

    def ingest_hf_datasets(self, brain, max_per_dataset: int = 100) -> int:
        """
        Stream samples from HuggingFace datasets and ingest them.
        Requires `datasets` library. Gracefully skips if not installed.
        """
        count = 0
        try:
            from datasets import load_dataset
        except ImportError:
            log.info("DatasetHub: `datasets` library not installed — skipping HF ingestion. "
                     "Install with: pip install datasets")
            return 0

        ki = getattr(brain, "knowledge_ingester", None)
        if not ki:
            return 0

        # Sort by priority, ingest highest first
        for entry in sorted(self._registry, key=lambda e: e.priority, reverse=True):
            if entry.source != "huggingface" or not entry.hf_id:
                continue
            if entry.name in self._ingested_sources:
                continue

            try:
                log.info(f"DatasetHub: Loading HF dataset '{entry.hf_id}'...")
                ds = load_dataset(entry.hf_id, split="train", streaming=True)

                ingested = 0
                for sample in ds:
                    if ingested >= max_per_dataset:
                        break

                    text = self._extract_text_from_sample(sample)
                    if not text or len(text) < 20:
                        continue

                    # Classify and ingest
                    hints = self._extract_hints_from_text(text, entry.domain)
                    ki.ingest_text(
                        title=f"HF_{entry.name}_{ingested}",
                        text=text[:2000],
                        domain=entry.domain,
                        concept_hints=hints[:5],
                    )
                    ingested += 1

                self._ingested_sources.append(entry.name)
                count += ingested
                self._stats["hf_datasets_loaded"] += 1
                log.info(f"DatasetHub: Ingested {ingested} samples from {entry.name}")

            except Exception as e:
                log.debug(f"DatasetHub: Failed to load {entry.hf_id}: {e}")

        return count

    # ── Utility methods ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_solvable_expression(problem: str) -> Optional[str]:
        """Extract a mathematical expression from a problem statement."""
        # Look for "Simplify: expr" or "Find derivative: expr" patterns
        m = re.search(r"(?:Simplify|Find|Compute|Evaluate|Prove)[:\s]+(.+)", problem)
        if m:
            expr = m.group(1).strip()
            # Only return if it looks like a symbolic expression
            if re.search(r'[+\-*/^=()xyzABd/]', expr):
                return expr
        # Try to find equation-like patterns
        m = re.search(r'([a-zA-Z0-9\s+\-*/^()]+\s*[=<>]+\s*[a-zA-Z0-9\s+\-*/^()]+)', problem)
        if m:
            return m.group(1).strip()
        return None

    @staticmethod
    def _extract_text_from_sample(sample: dict) -> str:
        """Extract text from a HuggingFace dataset sample."""
        # Try common field names
        for key in ["text", "content", "input", "instruction", "question",
                     "prompt", "context", "passage", "document"]:
            if key in sample and isinstance(sample[key], str):
                return sample[key]
        # Try nested conversation format
        if "conversations" in sample and isinstance(sample["conversations"], list):
            texts = [turn.get("value", turn.get("text", ""))
                     for turn in sample["conversations"]
                     if isinstance(turn, dict)]
            return " ".join(texts[:3])
        # Try output/response
        for key in ["output", "response", "answer", "completion"]:
            if key in sample and isinstance(sample[key], str):
                return sample[key]
        return ""

    @staticmethod
    def _extract_hints_from_text(text: str, domain: str) -> List[str]:
        """Auto-extract concept hints from text."""
        hints = []
        if domain and domain != "general":
            hints.append(domain)

        # Find capitalized terms (likely concepts)
        caps = re.findall(r'\b[A-Z][a-z]{3,}\b', text[:500])
        for word in caps[:3]:
            hints.append(word.lower())

        # Find terms near "is", "means", "defined as" (likely definitions)
        defs = re.findall(r'(\w{4,})\s+(?:is|means|refers to|defined as)\b', text[:500])
        hints.extend(d.lower() for d in defs[:2])

        return list(dict.fromkeys(hints))[:6]  # deduplicate, max 6

    def get_registry(self) -> List[dict]:
        """Return the full dataset registry."""
        return [e.to_dict() for e in self._registry]

    def add_dataset(self, name: str, hf_id: str = "", domain: str = "general",
                    dimensions: List[str] = None, priority: float = 0.5):
        """Register a new dataset for future ingestion."""
        entry = DatasetEntry(
            name=name, source="huggingface", hf_id=hf_id,
            dimensions=dimensions or ["world_knowledge"],
            domain=domain, priority=priority,
        )
        self._registry.append(entry)

    def summary(self) -> dict:
        return {
            "registered_datasets": len(self._registry),
            "ingested_sources": self._ingested_sources,
            "stats": dict(self._stats),
            "registry": [e.to_dict() for e in sorted(
                self._registry, key=lambda e: e.priority, reverse=True
            )[:10]],
        }
