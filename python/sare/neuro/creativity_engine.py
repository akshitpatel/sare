"""
CreativityEngine — Default Mode Network simulation
===================================================
The brain's Default Mode Network (DMN) is active during:
  - "mind wandering" / daydreaming
  - creative insight
  - connecting disparate memories

SARE's CreativityEngine simulates this by:
  1. Randomly sampling two concepts from different domains
  2. Asking the LLM: "What unexpected connection exists between these?"
  3. If a valid cross-domain rule is found — write a Transform for it
  4. Test: does this insight solve any existing stuck problems?
  5. If yes: promote and fire a dopamine burst

Additionally, the engine does "analogical reasoning":
  - Takes a successful proof from domain A
  - Asks: "Could this same pattern solve problems in domain B?"
  - Generates a variant Transform tuned for the target domain

Key insight: CREATIVITY = recombination of existing knowledge in novel ways.
The engine doesn't need new data — it mines its own memory for hidden connections.

Usage::
    ce = get_creativity_engine()
    result = ce.dream()                          # one creative cycle
    result = ce.analogy_transfer(               # targeted cross-domain
        source_domain="arithmetic",
        target_domain="logic",
        proof_steps=["double_neg", "add_zero_elim"],
    )

API:
    GET /api/creativity/status
    POST /api/creativity/dream   (trigger one cycle)
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_MEMORY = Path(__file__).resolve().parents[4] / "data" / "memory"
_SYNTH  = _MEMORY / "synthesized_modules"
_DREAM_LOG = _MEMORY / "dream_log.json"

# Domain pairs with known analogical relationships
_DOMAIN_ANALOGIES: List[Tuple[str, str]] = [
    ("arithmetic",  "logic"),       # x+0=x ↔ x∧True=x
    ("algebra",     "calculus"),    # simplification ↔ differentiation rules
    ("logic",       "set_theory"),  # ∧/∨ ↔ ∩/∪
    ("arithmetic",  "string"),      # numerical identity ↔ string identity
    ("calculus",    "physics"),     # rate of change ↔ velocity
    ("algebra",     "geometry"),    # factoring ↔ decomposition
    ("logic",       "planning"),    # inference ↔ action consequence
    ("arithmetic",  "probability"), # zero ↔ impossibility
]


@dataclass
class DreamResult:
    """One creative insight cycle."""
    source_domain:   str
    target_domain:   str
    concept_a:       str
    concept_b:       str
    insight:         str            # plain-English connection found
    transform_name:  str
    transform_code:  str
    promoted:        bool
    validation_score: float
    timestamp:       float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d.pop("transform_code", None)  # don't serialize full code
        return d


class CreativityEngine:
    """
    Default Mode Network: generates creative cross-domain insights
    and turns them into concrete Transform implementations.
    """

    COOLDOWN_SECONDS = 300   # 5 min between full dream cycles
    MAX_LOG = 50

    def __init__(self):
        self._dreams:    List[DreamResult] = []
        self._last_dream: float = 0.0
        self._load()

    # ── Main API ─────────────────────────────────────────────────────────────

    def dream(self, force: bool = False) -> dict:
        """
        One creative cycle: pick two domains, find a cross-domain insight,
        write a Transform, test it.
        """
        from sare.interface.llm_bridge import _call_llm, llm_available
        if not llm_available():
            return {"promoted": False, "message": "LLM unavailable"}

        if not force and time.time() - self._last_dream < self.COOLDOWN_SECONDS:
            return {"promoted": False, "message": "cooldown"}

        self._last_dream = time.time()

        # Step 1: Choose concept pair
        src_domain, tgt_domain, concept_a, concept_b = self._pick_concept_pair()
        log.info("[Creativity] Dream: %s (%s) ↔ %s (%s)",
                 concept_a, src_domain, concept_b, tgt_domain)

        # Step 2: Find the insight
        insight = self._find_insight(concept_a, src_domain, concept_b, tgt_domain, _call_llm)
        if not insight or len(insight) < 20:
            return {"promoted": False, "message": "No insight found"}

        # Step 3: Write Transform
        transform_name = self._name_for(concept_a, concept_b)
        code = self._write_creative_transform(
            transform_name, concept_a, src_domain,
            concept_b, tgt_domain, insight, _call_llm
        )
        if not code:
            return {"promoted": False, "message": "No code generated", "insight": insight}

        from sare.neuro.symbol_creator import _strip_fences, _is_safe, _test_import
        code = _strip_fences(code)
        safe, reason = _is_safe(code)
        if not safe:
            return {"promoted": False, "message": f"Safety: {reason}", "insight": insight}

        ok, err = _test_import(transform_name, code)
        if not ok:
            return {"promoted": False, "message": f"Import fail: {err}", "insight": insight}

        # Step 4: Validate
        score = self._validate(transform_name, code)

        # Step 5: Save
        path = _SYNTH / f"{transform_name}.py"
        path.write_text(code, encoding="utf-8")

        result = DreamResult(
            source_domain=src_domain,
            target_domain=tgt_domain,
            concept_a=concept_a,
            concept_b=concept_b,
            insight=insight[:300],
            transform_name=transform_name,
            transform_code=code,
            promoted=score >= 0.1,
            validation_score=score,
        )
        self._dreams.append(result)
        self._save()

        if result.promoted:
            log.info("[Creativity] ✓ Creative transform '%s' promoted (score=%.2f)",
                     transform_name, score)
            # Dopamine burst for creative insight
            try:
                from sare.neuro.dopamine import get_dopamine_system
                get_dopamine_system().receive_reward(
                    "creative_hypothesis", domain=tgt_domain, delta=score * 8
                )
            except Exception:
                pass

        return {
            "promoted":        result.promoted,
            "transform_name":  transform_name,
            "insight":         insight[:300],
            "source_domain":   src_domain,
            "target_domain":   tgt_domain,
            "score":           score,
            "message":         "promoted" if result.promoted else "saved_not_promoted",
        }

    def analogy_transfer(
        self,
        source_domain: str,
        target_domain: str,
        proof_steps:   List[str],
    ) -> dict:
        """
        Targeted creativity: given a successful proof in source_domain,
        ask whether the same pattern applies in target_domain.
        """
        from sare.interface.llm_bridge import _call_llm, llm_available
        if not llm_available():
            return {"promoted": False, "message": "LLM unavailable"}

        proof_str = " → ".join(proof_steps) if proof_steps else "unknown"
        prompt = (
            f"You are an expert mathematician and logician.\n\n"
            f"A SARE-HX AGI system solved a problem in the {source_domain} domain "
            f"using this proof chain:\n  {proof_str}\n\n"
            f"Your task: determine if this SAME reasoning pattern could apply in the "
            f"{target_domain} domain.\n\n"
            f"If yes:\n"
            f"1. Describe the analogous pattern in {target_domain}\n"
            f"2. Give a concrete example expression in {target_domain} it would simplify\n"
            f"3. Name the analogous transform: ANALOG_NAME: <PascalCase>\n\n"
            f"If no analogous pattern exists, respond with: NO_ANALOGY\n\n"
            f"Be specific and mathematical. Reference actual operations, not vague descriptions."
        )
        response = _call_llm(prompt, use_synthesis_model=True)

        if "NO_ANALOGY" in response.upper():
            return {"promoted": False, "message": "No analogy found", "response": response[:200]}

        # Dopamine reward for finding analogy
        try:
            from sare.neuro.dopamine import get_dopamine_system
            get_dopamine_system().receive_reward(
                "analogy_found", domain=target_domain, delta=3.0
            )
        except Exception:
            pass

        return {
            "promoted":      True,
            "analogy":       response[:400],
            "source_domain": source_domain,
            "target_domain": target_domain,
            "proof_chain":   proof_str,
            "message":       "analogy found",
        }

    def get_status(self) -> dict:
        return {
            "total_dreams":       len(self._dreams),
            "promoted":           sum(1 for d in self._dreams if d.promoted),
            "last_dream":         self._last_dream,
            "cooldown_remaining": max(0.0, self.COOLDOWN_SECONDS - (time.time() - self._last_dream)),
            "recent_dreams":      [d.to_dict() for d in self._dreams[-5:]],
        }

    def load_promoted_transforms(self) -> list:
        """Load all promoted creative transforms as instances."""
        import importlib.util
        transforms = []
        for dream in self._dreams:
            if not dream.promoted:
                continue
            path = _SYNTH / f"{dream.transform_name}.py"
            if not path.exists():
                continue
            try:
                tname = f"_sare_creative_{dream.transform_name}"
                spec = importlib.util.spec_from_file_location(tname, str(path))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                cls = getattr(mod, dream.transform_name, None)
                if cls:
                    transforms.append(cls())
            except Exception as e:
                log.debug("[Creativity] Load error %s: %s", dream.transform_name, e)
        return transforms

    # ── Concept picking ───────────────────────────────────────────────────────

    def _pick_concept_pair(self) -> Tuple[str, str, str, str]:
        """Pick a (src_domain, tgt_domain, concept_a, concept_b) pair."""
        # Concept libraries per domain
        concepts: Dict[str, List[str]] = {
            "arithmetic":  ["zero", "identity", "inverse", "commutativity", "associativity"],
            "logic":       ["negation", "conjunction", "disjunction", "implication", "tautology"],
            "algebra":     ["factoring", "distribution", "substitution", "simplification"],
            "calculus":    ["differentiation", "integration", "chain rule", "limit", "continuity"],
            "set_theory":  ["union", "intersection", "complement", "subset", "empty set"],
            "probability": ["independence", "conditional", "complement", "Bayes"],
            "geometry":    ["symmetry", "transformation", "congruence", "similarity"],
            "planning":    ["precondition", "effect", "goal", "action", "state"],
            "string":      ["concatenation", "substring", "reversal", "empty string"],
        }

        # Prefer known analogy pairs
        if _DOMAIN_ANALOGIES and random.random() < 0.7:
            src_d, tgt_d = random.choice(_DOMAIN_ANALOGIES)
        else:
            all_domains = list(concepts.keys())
            src_d, tgt_d = random.sample(all_domains, 2)

        c_a = random.choice(concepts.get(src_d, ["identity"]))
        c_b = random.choice(concepts.get(tgt_d, ["identity"]))
        return src_d, tgt_d, c_a, c_b

    def _find_insight(self, c_a, src, c_b, tgt, llm) -> str:
        prompt = (
            f"You are exploring the deep connections between mathematics and logic.\n\n"
            f"Concept A: '{c_a}' from {src}\n"
            f"Concept B: '{c_b}' from {tgt}\n\n"
            "Find a GENUINE mathematical connection between these two concepts.\n"
            "The connection should be:\n"
            "  1. Non-obvious (not just 'both are mathematical')\n"
            "  2. Expressible as a simplification rule\n"
            "  3. Useful for an automated reasoning system\n\n"
            "Give a 2-3 sentence insight about their connection, then one concrete example:\n"
            "EXAMPLE: <expression> → <simplified form>\n\n"
            "If there is no genuine connection, just write: NO_CONNECTION"
        )
        resp = llm(prompt, use_synthesis_model=True)
        return "" if "NO_CONNECTION" in resp.upper() else resp

    def _write_creative_transform(self, name, c_a, src, c_b, tgt, insight, llm) -> str:
        prompt = (
            f"Write a Python Transform class named `{name}` for the SARE-HX AGI system.\n\n"
            f"This transform embodies the following cross-domain insight:\n{insight}\n\n"
            f"It connects '{c_a}' ({src}) with '{c_b}' ({tgt}).\n\n"
            "Implement:\n"
            f"class {name}:\n"
            f"    def name(self) -> str: return '{name}'\n"
            f"    def applies_to(self, graph) -> bool: ...\n"
            f"    def apply(self, graph): ...  # returns new Graph or None\n\n"
            "Rules:\n"
            "- Import only: from sare.engine import Graph\n"
            "- Use node.type, node.label, edge.relationship_type\n"
            "- Return new Graph (don't mutate input) or None\n"
            "- The result must have LOWER energy than the input\n"
            "- No eval/exec/subprocess/socket imports\n\n"
            "Return raw Python only, no markdown:"
        )
        return llm(prompt, use_synthesis_model=True)

    def _name_for(self, c_a: str, c_b: str) -> str:
        """Generate a PascalCase name from two concepts."""
        def pascal(s):
            return "".join(w.capitalize() for w in re.sub(r'[^a-zA-Z0-9]', ' ', s).split())
        import re
        name = pascal(c_a) + pascal(c_b) + "Analogy"
        return name if name else "CreativeTransform"

    def _validate(self, name: str, code: str) -> float:
        """Quick validation: can the transform improve any test expression?"""
        try:
            import importlib.util, sys
            test_name = f"_sare_creative_test_{name}_{int(time.time())}"
            path = _SYNTH / f"_ctest_{name}.py"
            path.write_text(code, encoding="utf-8")
            spec = importlib.util.spec_from_file_location(test_name, str(path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            cls = getattr(mod, name, None)
            path.unlink(missing_ok=True)
            sys.modules.pop(test_name, None)
            if not cls:
                return 0.0
            t = cls()
            # Try on standard test expressions
            from sare.engine import load_problem as lp, EnergyEvaluator
            ev = EnergyEvaluator()
            improved = 0
            for expr in ["x + 0", "not not x", "x * 1", "x and True", "x or False"]:
                try:
                    _, g = lp(expr)
                    if g and t.applies_to(g):
                        ng = t.apply(g)
                        if ng and ev.compute(ng).total < ev.compute(g).total:
                            improved += 1
                except Exception:
                    pass
            return improved / 5.0
        except Exception as e:
            log.debug("[Creativity] Validate error: %s", e)
            return 0.0

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save(self):
        try:
            data = [d.to_dict() for d in self._dreams[-self.MAX_LOG:]]
            _DREAM_LOG.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as e:
            log.debug("[Creativity] Save error: %s", e)

    def _load(self):
        if not _DREAM_LOG.exists():
            return
        try:
            data = json.loads(_DREAM_LOG.read_text())
            for d in data:
                r = DreamResult(transform_code="", **{
                    k: v for k, v in d.items()
                    if k in DreamResult.__dataclass_fields__
                })
                self._dreams.append(r)
        except Exception as e:
            log.debug("[Creativity] Load error: %s", e)


import re as _re


# ── Singleton ─────────────────────────────────────────────────────────────────
_instance: Optional[CreativityEngine] = None

def get_creativity_engine() -> CreativityEngine:
    global _instance
    if _instance is None:
        _instance = CreativityEngine()
    return _instance
