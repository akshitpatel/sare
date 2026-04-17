"""
KnowledgeIngester — Upgrade 2: Massive Knowledge Base

Reads raw text (Wikipedia articles, textbooks, papers, news) and
automatically builds concept graphs from them.

Pipeline:
  text → sentence split → concept extraction → rule mining → ConceptGraph

This gives SARE-HX a foundation of world knowledge beyond what it
discovers through problem-solving alone. Concepts emerge from reading,
just as human vocabulary grows from reading.
"""

from __future__ import annotations

import json
import re
import time
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]

_GENERIC_HINTS = {
    "agi", "development", "system", "architecture", "todo", "plan",
    "roadmap", "assessment", "suite", "description", "symbolic",
}

# ── Built-in knowledge seeds ───────────────────────────────────────────────────
# These simulate a knowledge base extract; in production you'd feed real text.

_KNOWLEDGE_BASE: List[dict] = [
    # Mathematics
    {
        "title": "Algebra — Identity Laws",
        "domain": "arithmetic",
        "text": "In algebra, an identity element leaves other elements unchanged. "
                "For addition, adding zero to any number gives that number: x + 0 = x. "
                "For multiplication, multiplying by one leaves the number unchanged: x * 1 = x. "
                "These are called additive and multiplicative identities.",
        "concepts": ["identity_addition", "identity_multiplication"],
    },
    {
        "title": "Logic — Boolean Laws",
        "domain": "logic",
        "text": "Boolean algebra governs logical operations. "
                "The law of double negation states that negating twice returns the original: not not p = p. "
                "De Morgan's law: not (A and B) = (not A) or (not B). "
                "Conjunction with true: A and true = A. Disjunction with false: A or false = A.",
        "concepts": ["double_negation", "conjunction", "negation"],
    },
    {
        "title": "Calculus — Derivatives",
        "domain": "calculus",
        "text": "The derivative measures the rate of change of a function. "
                "The derivative of a constant is zero: d/dx(c) = 0. "
                "The derivative of x is one: d/dx(x) = 1. "
                "The power rule states: d/dx(x^n) = n * x^(n-1). "
                "The chain rule: d/dx(f(g(x))) = f'(g(x)) * g'(x).",
        "concepts": ["differentiation"],
    },
    {
        "title": "Physics — Newton's Laws",
        "domain": "mechanics",
        "text": "Newton's first law: an object at rest stays at rest unless acted upon by force. "
                "Newton's second law: F = m * a, force equals mass times acceleration. "
                "Newton's third law: for every action there is an equal and opposite reaction. "
                "Conservation of momentum: m1*v1 + m2*v2 = constant in a closed system.",
        "concepts": ["momentum_conservation", "kinematics"],
    },
    {
        "title": "Thermodynamics — Heat and Energy",
        "domain": "thermodynamics",
        "text": "The first law of thermodynamics states that energy is conserved: dU = Q - W. "
                "Heat flows from hot to cold bodies: Q = m * c * delta_T. "
                "The ideal gas law: PV = nRT where P is pressure, V volume, n moles, R gas constant, T temperature. "
                "Entropy always increases in an isolated system.",
        "concepts": ["heat_transfer", "gas_law"],
    },
    {
        "title": "Set Theory — Basic Operations",
        "domain": "set_theory",
        "text": "The union of two sets A and B contains all elements in either: A union B. "
                "The intersection contains elements in both: A intersection B. "
                "The empty set has no elements. A union empty = A (identity). "
                "A intersection A = A (idempotence). The complement of A contains all elements not in A.",
        "concepts": ["set_identity", "set_operations"],
    },
    {
        "title": "Number Theory — Modular Arithmetic",
        "domain": "number_theory",
        "text": "Modular arithmetic works with remainders. a mod n gives the remainder of a divided by n. "
                "Fermat's little theorem: if p is prime, a^p ≡ a (mod p). "
                "The greatest common divisor gcd(a,b) is the largest number dividing both. "
                "If gcd(a,b) = 1, then a and b are coprime.",
        "concepts": ["modular_arithmetic", "number_theory"],
    },
    {
        "title": "Information Theory — Entropy",
        "domain": "information_theory",
        "text": "Shannon entropy measures information content: H = -sum(p * log2(p)). "
                "A uniform distribution has maximum entropy. A certain event has zero entropy. "
                "Mutual information measures shared information between two variables. "
                "The KL divergence measures how one distribution differs from another.",
        "concepts": ["entropy", "information_measure"],
    },
    {
        "title": "Graph Theory — Basic Concepts",
        "domain": "graph_theory",
        "text": "A graph consists of vertices and edges. The degree of a vertex is the number of edges. "
                "A path connects two vertices through a sequence of edges. "
                "A cycle is a path that returns to its starting vertex. "
                "A connected graph has a path between every pair of vertices. "
                "Trees are connected graphs with no cycles.",
        "concepts": ["graph_connectivity", "graph_structure"],
    },
    {
        "title": "Abstract Algebra — Groups",
        "domain": "abstract_algebra",
        "text": "A group has four properties: closure, associativity, identity, and inverses. "
                "The identity element e satisfies: e * a = a * e = a for all a. "
                "Every element a has an inverse a^-1 such that a * a^-1 = e. "
                "Abelian groups have commutativity: a * b = b * a.",
        "concepts": ["group_identity", "group_inverse"],
    },
    {
        "title": "Biology — Cell Theory",
        "domain": "biology",
        "text": (
            "All living things are made of cells. The cell is the basic unit of life. "
            "Cells come from pre-existing cells. DNA carries genetic information: A pairs with T, G pairs with C. "
            "Photosynthesis converts light energy into glucose. ATP is the energy currency of cells."
        ),
        "concepts": ["cell_division", "dna_replication", "photosynthesis"],
    },
    {
        "title": "Chemistry — Atomic Structure",
        "domain": "chemistry",
        "text": (
            "Atoms are made of protons, neutrons, and electrons. "
            "Elements in the same group have similar properties. "
            "A chemical equation must be balanced: reactants = products. "
            "pH measures acidity: pH < 7 is acidic, pH > 7 is basic, pH = 7 is neutral."
        ),
        "concepts": ["atomic_structure", "chemical_balance", "acid_base"],
    },
    {
        "title": "Computer Science — Algorithms",
        "domain": "computer_science",
        "text": (
            "An algorithm is a finite set of instructions to solve a problem. "
            "Big-O notation describes time complexity: O(1) constant, O(n) linear, O(n^2) quadratic. "
            "Sorting algorithms: merge sort O(n log n), quicksort O(n log n) average. "
            "A binary search runs in O(log n) time on a sorted array."
        ),
        "concepts": ["algorithm_complexity", "sorting", "searching"],
    },
    {
        "title": "Psychology — Learning and Memory",
        "domain": "psychology",
        "text": (
            "Classical conditioning: a neutral stimulus becomes associated with an unconditioned stimulus. "
            "Operant conditioning: behavior is shaped by consequences. "
            "Working memory holds roughly 7 items. Long-term memory stores knowledge indefinitely. "
            "The spacing effect shows distributed practice is more effective than massed practice."
        ),
        "concepts": ["conditioning", "memory_encoding", "learning_theory"],
    },
    {
        "title": "Economics — Supply and Demand",
        "domain": "economics",
        "text": (
            "When price rises, quantity demanded falls: the law of demand. "
            "When price rises, quantity supplied rises: the law of supply. "
            "Equilibrium price is where supply equals demand. "
            "GDP measures total economic output. Inflation is a general rise in price level."
        ),
        "concepts": ["supply_demand", "equilibrium", "gdp"],
    },
    {
        "title": "Geography — Earth Systems",
        "domain": "geography",
        "text": (
            "The Earth has four layers: crust, mantle, outer core, inner core. "
            "Tectonic plates move causing earthquakes and volcanoes. "
            "The water cycle: evaporation, condensation, precipitation, runoff. "
            "Latitude measures distance from equator; longitude measures east-west position."
        ),
        "concepts": ["plate_tectonics", "water_cycle", "coordinate_system"],
    },
    {
        "title": "History — Causation Patterns",
        "domain": "history",
        "text": (
            "The industrial revolution increased manufacturing efficiency through mechanization. "
            "The printing press spread information, enabling the Reformation. "
            "Revolutions occur when popular discontent exceeds the regime's capacity for repression. "
            "World War I was caused by nationalism, imperialism, alliances, and assassination."
        ),
        "concepts": ["industrial_revolution", "causation", "revolution_conditions"],
    },
    {
        "title": "Linguistics — Language Structure",
        "domain": "linguistics",
        "text": (
            "A sentence consists of a subject and a predicate. "
            "Morphemes are the smallest units of meaning. "
            "Syntax governs how words combine into sentences. "
            "All human languages have nouns, verbs, and a way to negate propositions."
        ),
        "concepts": ["sentence_structure", "morphology", "syntax_rules"],
    },
]

# Patterns for extracting symbolic rules from text
_RULE_PATTERNS = [
    (r"([a-zA-Z\s\+\-\*\/\^\(\)=]+)\s*=\s*([a-zA-Z0-9\s\+\-\*\/\^\(\)]+)", "equation"),
    (r"d/dx\([^)]+\)\s*=\s*[^\n.]+", "derivative"),
    (r"[a-zA-Z]\s*\*\s*[a-zA-Z0-9]\s*=\s*[a-zA-Z0-9]", "algebraic"),
]


@dataclass
class ExtractedConcept:
    """A concept extracted from ingested text."""
    name: str
    meaning: str
    domain: str
    symbolic_rules: List[str] = field(default_factory=list)
    source_title: str = ""
    source_path: str = ""
    source_refs: List[str] = field(default_factory=list)
    confidence: float = 0.6

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "meaning": self.meaning,
            "domain": self.domain,
            "symbolic_rules": self.symbolic_rules,
            "source": self.source_title,
            "source_path": self.source_path,
            "sources": self.source_refs,
            "source_count": len(self.source_refs),
        }


class KnowledgeIngester:
    """
    Reads text (simulated knowledge base) and auto-builds ConceptGraph entries.

    In a production system, this would consume:
      - Wikipedia API (article text)
      - arXiv / PubMed (abstract text)
      - Textbook PDFs (parsed text)
      - News feeds (recent knowledge)

    The ingester extracts:
      1. Named concepts (nouns near mathematical operators)
      2. Symbolic rules (equations and laws)
      3. Domain classification (arithmetic, logic, physics, ...)
      4. Related concept links (co-occurrence in same sentence)
    """

    def __init__(self):
        self._extracted: Dict[str, ExtractedConcept] = {}
        self._ingested_titles: List[str] = []
        self._total_rules_found: int = 0
        self._corpus_files_ingested: List[str] = []

    # ── Core ingestion ─────────────────────────────────────────────────────────

    def ingest_text(self, title: str, text: str, domain: str = "general",
                    concept_hints: List[str] = None, source_path: str = "") -> List[ExtractedConcept]:
        """
        Parse text and extract concepts + symbolic rules.
        Returns list of new ExtractedConcept objects.
        """
        extracted = []
        sentences = re.split(r'[.!?]\s+', text)

        # Extract symbolic rules from sentences
        rules_found: List[str] = []
        for sentence in sentences:
            for pattern, kind in _RULE_PATTERNS:
                matches = re.findall(pattern, sentence)
                for m in matches:
                    rule = m if isinstance(m, str) else " = ".join(m)
                    rule = rule.strip()
                    if len(rule) > 3 and len(rule) < 80:
                        rules_found.append(rule)

        # Create/update concepts from hints
        hints = concept_hints or []
        source_ref = source_path or title
        for concept_name in hints:
            # Find the sentence most relevant to this concept
            relevant_sentences = [
                s for s in sentences
                if concept_name.replace("_", " ") in s.lower() or
                   any(part in s.lower() for part in concept_name.split("_"))
            ]
            meaning = relevant_sentences[0][:120] if relevant_sentences else text[:80]

            if concept_name not in self._extracted:
                ec = ExtractedConcept(
                    name=concept_name,
                    meaning=meaning.strip(),
                    domain=domain,
                    symbolic_rules=rules_found[:3],
                    source_title=title,
                    source_path=source_path,
                    source_refs=[source_ref] if source_ref else [],
                    confidence=0.7,
                )
                self._extracted[concept_name] = ec
                extracted.append(ec)
            else:
                # Enrich existing
                existing = self._extracted[concept_name]
                for r in rules_found:
                    if r not in existing.symbolic_rules:
                        existing.symbolic_rules.append(r)
                if source_ref and source_ref not in existing.source_refs:
                    existing.source_refs.append(source_ref)
                    existing.confidence = min(0.95, existing.confidence + 0.02)
                if source_path and not existing.source_path:
                    existing.source_path = source_path

        self._ingested_titles.append(title)
        self._total_rules_found += len(rules_found)
        log.info(f"KnowledgeIngester: '{title}' → {len(hints)} concepts, {len(rules_found)} rules")
        return extracted

    def ingest_knowledge_base(self) -> int:
        """Ingest the built-in knowledge base plus selected local corpus files."""
        total = 0
        for entry in _KNOWLEDGE_BASE:
            extracted = self.ingest_text(
                title=entry["title"],
                text=entry["text"],
                domain=entry["domain"],
                concept_hints=entry["concepts"],
            )
            total += len(extracted)
        total += self.ingest_local_corpus()
        return total

    @staticmethod
    def _infer_domain_from_path(path: Path, fallback: str = "general") -> str:
        parts = {part.lower() for part in path.parts}
        if "logic" in parts:
            return "logic"
        if "algebra" in parts:
            return "algebra"
        if "calculus" in parts:
            return "calculus"
        if "physics" in parts or "mechanics" in parts:
            return "mechanics"
        return fallback

    @staticmethod
    def _infer_concept_hints(title: str, text: str, extra_hints: Optional[List[str]] = None) -> List[str]:
        hints: List[str] = []
        for raw in extra_hints or []:
            token = re.sub(r"[^a-z0-9_]+", "_", raw.lower()).strip("_")
            if token and token not in hints:
                hints.append(token)

        title_tokens = re.findall(r"[a-z][a-z0-9_]{2,}", title.lower())
        for token in title_tokens:
            if token not in _GENERIC_HINTS and token not in hints:
                hints.append(token)

        rule_like = re.findall(r"\b([a-z_]{3,})\b", text.lower()[:400])
        counts = Counter(token for token in rule_like if token not in _GENERIC_HINTS)
        for token, _count in counts.most_common(4):
            if token not in hints:
                hints.append(token)

        return hints[:8]

    def ingest_local_corpus(self, repo_root: Optional[Path] = None, max_examples_per_file: int = 24) -> int:
        """Ingest local docs/benchmarks so the knowledge base reflects real repo content."""
        root = Path(repo_root or REPO_ROOT)
        total = 0
        paths = sorted((root / "docs").glob("**/*.md"))
        paths += sorted((root / "benchmarks").glob("**/*.json"))
        hard_problems = root / "data" / "hard_problems.json"
        if hard_problems.exists():
            paths.append(hard_problems)

        for path in paths:
            if str(path) in self._corpus_files_ingested:
                continue
            try:
                if path.suffix == ".md":
                    text = path.read_text(encoding="utf-8")[:5000]
                    title = path.stem.replace("_", " ").title()
                    domain = self._infer_domain_from_path(path)
                    hints = self._infer_concept_hints(title, text)
                    total += len(self.ingest_text(
                        title,
                        text,
                        domain=domain,
                        concept_hints=hints,
                        source_path=str(path.relative_to(root)),
                    ))
                elif path.suffix == ".json":
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    title = path.stem.replace("_", " ").title()
                    domain = self._infer_domain_from_path(path)
                    examples: List[dict] = []
                    extra_hints: List[str] = []

                    if isinstance(payload, list):
                        examples = payload[:max_examples_per_file]
                    elif isinstance(payload, dict) and isinstance(payload.get("cases"), list):
                        examples = payload["cases"][:max_examples_per_file]
                        if payload.get("suite"):
                            extra_hints.append(str(payload["suite"]))

                    if not examples:
                        continue

                    rendered = []
                    for item in examples:
                        expr = str(item.get("expression", "")).strip()
                        expected = str(item.get("expected_result", "")).strip()
                        category = str(item.get("category", item.get("domain", ""))).strip()
                        if category:
                            extra_hints.append(category)
                        if expr and expected:
                            rendered.append(f"{expr} = {expected}")
                        elif expr:
                            rendered.append(expr)

                    if not rendered:
                        continue

                    text = ". ".join(rendered)
                    hints = self._infer_concept_hints(title, text, extra_hints=extra_hints)
                    total += len(self.ingest_text(
                        title,
                        text,
                        domain=domain,
                        concept_hints=hints,
                        source_path=str(path.relative_to(root)),
                    ))

                self._corpus_files_ingested.append(str(path))
            except Exception as exc:
                log.debug("KnowledgeIngester skipped %s: %s", path, exc)
        return total

    def feed_to_concept_graph(self, cg) -> int:
        """
        Push all extracted concepts into a ConceptGraph.
        Returns number of concepts enriched.
        """
        count = 0
        for name, ec in self._extracted.items():
            # Ground a textual example for each concept
            try:
                cg.ground_example(
                    concept_name=name,
                    text=ec.meaning[:100],
                    operation="knowledge_ingestion",
                    inputs=ec.source_refs[:3] or [ec.source_title],
                    result=ec.meaning[:40],
                    domain=ec.domain,
                    symbolic=ec.symbolic_rules[0] if ec.symbolic_rules else "",
                )
                # Add symbolic rules to the concept
                if name in cg._concepts:
                    for rule in ec.symbolic_rules[:3]:
                        if rule not in cg._concepts[name].symbolic_rules:
                            cg._concepts[name].symbolic_rules.append(rule)
                count += 1
            except Exception:
                pass
        log.info(f"KnowledgeIngester fed {count} concepts to ConceptGraph")
        return count

    def ingest_and_feed(self, cg) -> dict:
        """Convenience: ingest built-in KB + feed to ConceptGraph."""
        n_concepts = self.ingest_knowledge_base()
        n_fed = self.feed_to_concept_graph(cg)
        return {"concepts_extracted": n_concepts, "concepts_fed": n_fed,
                "rules_found": self._total_rules_found, "titles": len(self._ingested_titles)}

    def summary(self) -> dict:
        all_sources = sorted({ref for ec in self._extracted.values() for ref in ec.source_refs})
        return {
            "titles_ingested": len(self._ingested_titles),
            "concepts_extracted": len(self._extracted),
            "total_rules_found": self._total_rules_found,
            "corpus_files_ingested": len(self._corpus_files_ingested),
            "unique_sources": len(all_sources),
            "recent_sources": all_sources[-5:],
            "domains": list({ec.domain for ec in self._extracted.values()}),
            "recent_concepts": [ec.to_dict() for ec in
                                 list(self._extracted.values())[-5:]],
        }
