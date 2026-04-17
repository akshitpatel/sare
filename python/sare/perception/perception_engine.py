"""
Perception Engine — Multi-modal data ingestion into the Brain learning loop.

Converts real-world data (CSV, text, JSON, math textbooks, URLs) into
SARE-HX graphs, extracts learnable problems, and feeds them into the
developmental curriculum for autonomous learning.

This bridges the gap between "can solve x+0" and "understands the world"
by letting SARE-HX encounter diverse data and extract structure from it.

Pipeline:
  1. Ingest raw data (file, text, URL, API response)
  2. Ground into typed graph via WorldGrounder
  3. Extract learnable problems (expressions, equations, relations)
  4. Feed problems into curriculum / learning loop
  5. Build world model observations from non-math content
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass
class ExtractedProblem:
    """A problem extracted from real-world data."""
    expression: str
    domain: str
    source: str           # where it came from
    difficulty: float     # estimated 0-1
    context: str = ""     # surrounding text for explanation
    extracted_at: float = field(default_factory=time.time)


@dataclass
class PerceptionResult:
    """Result of perceiving raw data."""
    source: str
    kind: str             # csv, text, json, math, url
    problems_extracted: List[ExtractedProblem]
    facts_extracted: List[dict]    # for world model
    graph_nodes: int
    graph_edges: int
    elapsed: float = 0.0


class MathExtractor:
    """
    Extracts mathematical expressions and equations from text.
    Finds things like: "x + 0 = x", "2 * (3 + 4)", "solve for x: 2x + 3 = 7"
    """

    # Patterns that look like math expressions
    _MATH_PATTERNS = [
        # Equations: x + 3 = 7, 2x = 10
        r'(\w+\s*[+\-*/^]\s*\w+\s*=\s*\w+)',
        # Expressions with operators: (x + 0) * 1
        r'(\([^)]{3,30}\)\s*[+\-*/^]\s*\w+)',
        r'(\w+\s*[+\-*/^]\s*\([^)]{3,30}\))',
        # Simple binary: x + 0, y * 1, a - a
        r'(\b[a-z]\s*[+\-*/^]\s*\d+\b)',
        r'(\b\d+\s*[+\-*/^]\s*[a-z]\b)',
        r'(\b[a-z]\s*[+\-*/^]\s*[a-z]\b)',
        # Negation: not not x, --x, neg neg x
        r'(not\s+not\s+\w+)',
        r'(neg\s+neg\s+\w+)',
        # Power: x^0, x^1
        r'(\w+\s*\^\s*[01])',
    ]

    # Textbook-style problem indicators
    _PROBLEM_INDICATORS = [
        r'simplify[:\s]+(.{5,50})',
        r'evaluate[:\s]+(.{5,50})',
        r'solve[:\s]+(.{5,50})',
        r'compute[:\s]+(.{5,50})',
        r'reduce[:\s]+(.{5,50})',
        r'factor[:\s]+(.{5,50})',
        r'expand[:\s]+(.{5,50})',
    ]

    @staticmethod
    def extract(text: str, source: str = "text") -> List[ExtractedProblem]:
        """Extract mathematical problems from free text."""
        problems = []
        seen = set()

        # Clean text
        text = text.replace('\n', ' ').replace('\r', ' ')

        # 1. Find textbook-style problems
        for pattern in MathExtractor._PROBLEM_INDICATORS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                expr = m.group(1).strip().rstrip('.,:;')
                expr = re.sub(r'[^a-zA-Z0-9+\-*/^()= .]', '', expr).strip()
                if expr and len(expr) >= 3 and expr not in seen:
                    seen.add(expr)
                    # Get surrounding context
                    start = max(0, m.start() - 30)
                    end = min(len(text), m.end() + 30)
                    context = text[start:end].strip()
                    problems.append(ExtractedProblem(
                        expression=expr, domain=MathExtractor._guess_domain(expr),
                        source=source, difficulty=MathExtractor._estimate_difficulty(expr),
                        context=context,
                    ))

        # 2. Find raw math expressions
        for pattern in MathExtractor._MATH_PATTERNS:
            for m in re.finditer(pattern, text):
                expr = m.group(1).strip()
                if expr and len(expr) >= 3 and expr not in seen:
                    seen.add(expr)
                    problems.append(ExtractedProblem(
                        expression=expr, domain=MathExtractor._guess_domain(expr),
                        source=source, difficulty=MathExtractor._estimate_difficulty(expr),
                    ))

        return problems

    @staticmethod
    def _guess_domain(expr: str) -> str:
        e = expr.lower()
        if '=' in e:
            return "algebra"
        if any(w in e for w in ['not', 'and', 'or', 'true', 'false']):
            return "logic"
        if any(w in e for w in ['union', 'intersect']):
            return "sets"
        if '^' in e:
            return "power_rules"
        return "arithmetic"

    @staticmethod
    def _estimate_difficulty(expr: str) -> float:
        ops = sum(1 for c in expr if c in '+-*/^=')
        parens = expr.count('(')
        length = len(expr)
        score = min(1.0, (ops * 0.15 + parens * 0.2 + length * 0.01))
        return round(score, 2)


class FactExtractor:
    """
    Extracts world knowledge facts from text for the WorldModel.
    Finds statements like: "Fire is hot", "Dogs are animals", "2+2=4"
    """

    _FACT_PATTERNS = [
        # X is Y
        (r'(\w+)\s+is\s+(?:a\s+)?(\w+)', "IsA"),
        # X has Y
        (r'(\w+)\s+has\s+(?:a\s+)?(\w+)', "HasA"),
        # X causes Y
        (r'(\w+)\s+causes?\s+(\w+)', "Causes"),
        # X are Y
        (r'(\w+)\s+are\s+(\w+)', "IsA"),
        # If X then Y
        (r'if\s+(.{3,30})\s+then\s+(.{3,30})', "Implies"),
    ]

    @staticmethod
    def extract(text: str, domain: str = "general") -> List[dict]:
        facts = []
        seen = set()
        for pattern, relation in FactExtractor._FACT_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                subj = m.group(1).strip().lower()
                obj = m.group(2).strip().lower()
                key = f"{subj}:{relation}:{obj}"
                if key not in seen and len(subj) > 1 and len(obj) > 1:
                    seen.add(key)
                    facts.append({
                        "subject": subj,
                        "relation": relation,
                        "object": obj,
                        "domain": domain,
                        "confidence": 0.7,
                    })
        return facts


class PerceptionEngine:
    """
    Main perception engine. Ingests multi-modal data and produces
    learnable problems + world knowledge.
    """

    def __init__(self):
        from sare.perception.world_grounder import WorldGrounder
        self._grounder = WorldGrounder()
        self._math_extractor = MathExtractor()
        self._fact_extractor = FactExtractor()
        self._ingestion_history: List[dict] = []

    def ingest_text(self, text: str, source: str = "text") -> PerceptionResult:
        """Ingest free text: extract math problems + world facts."""
        start = time.time()
        from sare.perception.world_grounder import RawPercept

        # Ground into graph
        gd = self._grounder.ground(RawPercept("text", text, source))

        # Extract math problems
        problems = self._math_extractor.extract(text, source)

        # Extract world facts
        facts = self._fact_extractor.extract(text)

        elapsed = time.time() - start
        self._record(source, "text", len(problems), len(facts))

        return PerceptionResult(
            source=source, kind="text",
            problems_extracted=problems, facts_extracted=facts,
            graph_nodes=len(gd.nodes), graph_edges=len(gd.edges),
            elapsed=elapsed,
        )

    def ingest_csv(self, csv_text: str, source: str = "csv") -> PerceptionResult:
        """Ingest CSV data: extract numeric patterns as problems."""
        start = time.time()
        from sare.perception.world_grounder import RawPercept

        gd = self._grounder.ground(RawPercept("csv", csv_text, source))

        # Extract numeric relationships from CSV
        problems = []
        import csv as csv_mod
        import io
        reader = csv_mod.reader(io.StringIO(csv_text))
        rows = list(reader)
        if len(rows) > 1:
            headers = rows[0]
            for row in rows[1:6]:  # first 5 data rows
                nums = []
                for val in row:
                    try:
                        nums.append(float(val))
                    except ValueError:
                        pass
                # Generate arithmetic problems from numeric data
                if len(nums) >= 2:
                    for i in range(len(nums) - 1):
                        expr = f"{nums[i]} + {nums[i+1]}"
                        problems.append(ExtractedProblem(
                            expression=expr, domain="arithmetic",
                            source=source, difficulty=0.2,
                        ))

        facts = []
        elapsed = time.time() - start
        self._record(source, "csv", len(problems), len(facts))

        return PerceptionResult(
            source=source, kind="csv",
            problems_extracted=problems, facts_extracted=facts,
            graph_nodes=len(gd.nodes), graph_edges=len(gd.edges),
            elapsed=elapsed,
        )

    def ingest_json(self, data: Any, source: str = "json") -> PerceptionResult:
        """Ingest JSON data."""
        start = time.time()
        from sare.perception.world_grounder import RawPercept

        json_str = json.dumps(data) if not isinstance(data, str) else data
        gd = self._grounder.ground(RawPercept("json", json_str, source))

        problems = []
        facts = []

        # If the JSON has math expressions, extract them
        if isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, str):
                    problems.extend(self._math_extractor.extract(val, source))
                    facts.extend(self._fact_extractor.extract(val))
        elif isinstance(data, list):
            for item in data[:20]:
                if isinstance(item, str):
                    problems.extend(self._math_extractor.extract(item, source))

        elapsed = time.time() - start
        self._record(source, "json", len(problems), len(facts))

        return PerceptionResult(
            source=source, kind="json",
            problems_extracted=problems, facts_extracted=facts,
            graph_nodes=len(gd.nodes), graph_edges=len(gd.edges),
            elapsed=elapsed,
        )

    def ingest_textbook(self, text: str, source: str = "textbook") -> PerceptionResult:
        """
        Specialized textbook parser: extracts structured problems,
        definitions, theorems, and examples.
        """
        start = time.time()

        # Extract math problems (enhanced for textbook format)
        problems = self._math_extractor.extract(text, source)

        # Extract definitions as facts
        facts = self._fact_extractor.extract(text, "mathematics")

        # Look for theorem/definition blocks
        theorem_patterns = [
            r'(?:theorem|lemma|proposition)[:\s]+(.{10,100})',
            r'(?:definition)[:\s]+(.{10,100})',
            r'(?:example)[:\s]+(.{5,80})',
        ]
        for pattern in theorem_patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                content = m.group(1).strip()
                # Try to extract math from theorem content
                sub_problems = self._math_extractor.extract(content, source)
                problems.extend(sub_problems)
                if not sub_problems:
                    # Store as a fact instead
                    facts.append({
                        "subject": "theorem",
                        "relation": "states",
                        "object": content[:80],
                        "domain": "mathematics",
                        "confidence": 0.9,
                    })

        elapsed = time.time() - start
        self._record(source, "textbook", len(problems), len(facts))

        return PerceptionResult(
            source=source, kind="textbook",
            problems_extracted=problems, facts_extracted=facts,
            graph_nodes=0, graph_edges=0,
            elapsed=elapsed,
        )

    def ingest_file(self, path: str) -> PerceptionResult:
        """Auto-detect file type and ingest."""
        p = Path(path)
        if not p.exists():
            return PerceptionResult(
                source=path, kind="error",
                problems_extracted=[], facts_extracted=[],
                graph_nodes=0, graph_edges=0,
            )

        content = p.read_text(errors="replace")
        suffix = p.suffix.lower()

        if suffix == ".csv":
            return self.ingest_csv(content, str(p))
        elif suffix == ".json":
            try:
                data = json.loads(content)
                return self.ingest_json(data, str(p))
            except Exception:
                return self.ingest_text(content, str(p))
        else:
            return self.ingest_text(content, str(p))

    def _record(self, source: str, kind: str, n_problems: int, n_facts: int):
        self._ingestion_history.append({
            "source": source, "kind": kind,
            "problems": n_problems, "facts": n_facts,
            "timestamp": time.time(),
        })
        if len(self._ingestion_history) > 200:
            self._ingestion_history = self._ingestion_history[-200:]

    def stats(self) -> dict:
        return {
            "total_ingestions": len(self._ingestion_history),
            "total_problems": sum(h["problems"] for h in self._ingestion_history),
            "total_facts": sum(h["facts"] for h in self._ingestion_history),
            "recent": self._ingestion_history[-10:],
        }
