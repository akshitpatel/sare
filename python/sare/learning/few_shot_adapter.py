"""
FewShotAdapter — Dynamic few-shot specialization for the LLM bridge.

Collects SARE's best solved problem-solution pairs and prepends them
as examples to every LLM call. This gives the effect of domain-specialized
fine-tuning without weight updates.

The adapter learns which examples are most useful by tracking whether
LLM responses improved after adding specific examples (credit assignment
over the few-shot context).
"""
from __future__ import annotations
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "few_shot_dataset.json"
MAX_EXAMPLES = 500    # max stored examples
MAX_CONTEXT = 5       # max examples to include per LLM call
MIN_CONFIDENCE = 0.7  # minimum solve confidence to store as example


@dataclass
class FewShotExample:
    """A problem-solution pair used as a few-shot demonstration."""
    problem: str          # the input problem/expression
    solution: str         # the proof steps or answer
    domain: str
    confidence: float
    used_count: int = 0   # how many times this example was selected
    utility_score: float = 0.5  # EMA-tracked utility (did it help?)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "problem": self.problem, "solution": self.solution,
            "domain": self.domain, "confidence": round(self.confidence, 3),
            "used_count": self.used_count,
            "utility_score": round(self.utility_score, 3),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FewShotExample":
        ex = cls(
            problem=d["problem"], solution=d["solution"],
            domain=d.get("domain", "general"), confidence=d.get("confidence", 0.7)
        )
        ex.used_count = d.get("used_count", 0)
        ex.utility_score = d.get("utility_score", 0.5)
        ex.created_at = d.get("created_at", time.time())
        return ex


class FewShotAdapter:
    """
    Dynamic few-shot prompt adapter.

    Usage:
        adapter = FewShotAdapter()
        adapter.add_example("x + 0", "x", "algebra", 0.95)

        enriched_prompt = adapter.enrich_prompt(
            "Solve: 2*x + 0", domain="algebra"
        )
        # enriched_prompt now contains relevant solved examples before the question
    """

    def __init__(self, dataset_path: Optional[Path] = None):
        self._path = Path(dataset_path or DATASET_PATH)
        self._examples: List[FewShotExample] = []
        self._stats = {"total_added": 0, "total_enrichments": 0, "avg_examples_per_call": 0.0}
        self._load()

    def _load(self):
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            self._examples = [FewShotExample.from_dict(d) for d in data.get("examples", [])]
            self._stats = data.get("stats", self._stats)
            log.debug("FewShotAdapter loaded %d examples", len(self._examples))
        except Exception as e:
            log.debug("FewShotAdapter load failed: %s", e)

    def save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            import tempfile, os
            data = {
                "examples": [ex.to_dict() for ex in self._examples[-MAX_EXAMPLES:]],
                "stats": self._stats,
            }
            tmp = str(self._path) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._path)
        except Exception as e:
            log.debug("FewShotAdapter save failed: %s", e)

    def add_example(self, problem: str, solution: str, domain: str, confidence: float):
        """Add a new solved example to the dataset."""
        if confidence < MIN_CONFIDENCE:
            return
        if not problem.strip() or not solution.strip():
            return
        # Avoid exact duplicates
        for ex in self._examples:
            if ex.problem == problem and ex.domain == domain:
                # Update confidence if better
                if confidence > ex.confidence:
                    ex.confidence = confidence
                return

        self._examples.append(FewShotExample(
            problem=problem, solution=solution,
            domain=domain, confidence=confidence
        ))
        self._stats["total_added"] += 1

        # Trim if over limit
        if len(self._examples) > MAX_EXAMPLES:
            # Keep highest-confidence + highest-utility examples
            self._examples.sort(key=lambda x: x.confidence * 0.6 + x.utility_score * 0.4, reverse=True)
            self._examples = self._examples[:MAX_EXAMPLES]

        # Save every 50 new examples
        if self._stats["total_added"] % 50 == 0:
            self.save()

    def _select_examples(self, domain: str, query: str, n: int = MAX_CONTEXT) -> List[FewShotExample]:
        """Select the most relevant examples for a given domain + query."""
        if not self._examples:
            return []

        # Score each example by relevance
        query_words = set(re.findall(r'[a-z0-9]+', query.lower()))

        scored = []
        for ex in self._examples:
            score = 0.0
            # Domain match bonus
            if ex.domain == domain:
                score += 0.4
            elif ex.domain == "general":
                score += 0.1

            # Keyword overlap with query
            ex_words = set(re.findall(r'[a-z0-9]+', ex.problem.lower()))
            overlap = len(query_words & ex_words) / max(len(query_words), 1)
            score += overlap * 0.3

            # Confidence and utility
            score += ex.confidence * 0.2
            score += ex.utility_score * 0.1

            scored.append((score, ex))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:n]]

    def enrich_prompt(self, prompt: str, domain: str = "general", system_context: str = "") -> str:
        """
        Prepend relevant few-shot examples to a prompt.

        Returns the enriched prompt string.
        """
        examples = self._select_examples(domain, prompt)
        if not examples:
            return prompt

        # Build few-shot prefix
        lines = []
        if system_context:
            lines.append(system_context)
            lines.append("")

        lines.append("Here are some examples of solved problems:")
        lines.append("")

        for ex in examples:
            lines.append(f"Problem: {ex.problem}")
            lines.append(f"Solution: {ex.solution}")
            lines.append("")
            ex.used_count += 1

        lines.append("Now solve this:")
        lines.append(prompt)

        # Update stats
        self._stats["total_enrichments"] += 1
        prev_avg = self._stats.get("avg_examples_per_call", 0.0)
        n = self._stats["total_enrichments"]
        self._stats["avg_examples_per_call"] = round(
            prev_avg * (n-1)/n + len(examples)/n, 2
        )

        return "\n".join(lines)

    def record_feedback(self, problem: str, helped: bool):
        """Record whether few-shot context helped — updates utility scores."""
        for ex in self._examples:
            if ex.used_count > 0:
                alpha = 0.1
                ex.utility_score = (1 - alpha) * ex.utility_score + alpha * (1.0 if helped else 0.0)

    def get_stats(self) -> dict:
        return {
            **self._stats,
            "stored_examples": len(self._examples),
            "domains": list({ex.domain for ex in self._examples}),
            "avg_confidence": round(
                sum(ex.confidence for ex in self._examples) / max(len(self._examples), 1), 3
            ),
        }


_adapter_instance: Optional[FewShotAdapter] = None

def get_few_shot_adapter() -> FewShotAdapter:
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = FewShotAdapter()
    return _adapter_instance
