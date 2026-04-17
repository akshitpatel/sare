"""
GeneralCurriculum — generates diverse real-world problems beyond symbolic math.

Produces problems across 10 domains using LLM generation + static templates,
ensuring the system trains on a representative slice of human knowledge every cycle.

Domain mix (configurable):
  math       20%  — symbolic expressions (routed to symbolic engine)
  logic      10%  — propositional / predicate logic
  factual    20%  — geography, history, biology, chemistry facts
  science    10%  — physics / chemistry word problems
  reasoning  10%  — syllogisms, analogical inference
  analogy    10%  — "X is to Y as A is to ?"
  code        5%  — small Python tasks
  language    5%  — grammar, comprehension, paraphrase
  planning    5%  — step-by-step task decomposition
  social      5%  — theory of mind, perspective taking
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_STATS_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "general_solver_stats.json"

# ── Priority queue for self-test failures (curriculum debt) ───────────────────
_PRIORITY_QUEUE: List["GeneralProblem"] = []
_PRIORITY_LOCK = threading.Lock()


def inject_priority_problems(problems: List["GeneralProblem"]) -> None:
    """Inject problems that should be solved in the next batch (curriculum debt)."""
    with _PRIORITY_LOCK:
        _PRIORITY_QUEUE.extend(problems)

log = logging.getLogger(__name__)

# ── Problem dataclass ─────────────────────────────────────────────────────────

@dataclass
class GeneralProblem:
    problem_id:   str
    text:         str
    domain:       str
    expected:     Optional[str] = None   # known answer if available
    context:      str = ""
    difficulty:   float = 0.5            # 0.0 = trivial, 1.0 = expert


# ── Static template banks ─────────────────────────────────────────────────────

_FACTUAL_TEMPLATES: List[Tuple[str, str]] = [
    ("What is the capital of France?",               "Paris"),
    ("What element has atomic number 1?",            "Hydrogen"),
    ("What is the speed of light in m/s?",           "299,792,458"),
    ("How many bones are in the adult human body?",  "206"),
    ("What planet is closest to the Sun?",           "Mercury"),
    ("What is the chemical formula for water?",      "H2O"),
    ("What is the largest organ in the human body?", "skin"),
    ("In what year did World War II end?",           "1945"),
    ("What is the powerhouse of the cell?",          "mitochondria"),
    ("Who wrote 'Hamlet'?",                          "Shakespeare"),
    ("What is the boiling point of water at sea level?", "100 degrees Celsius"),
    ("How many continents are there?",               "7"),
    ("What is Newton's second law?",                 "F = ma"),
    ("What gas do plants absorb during photosynthesis?", "carbon dioxide"),
    ("What is the square root of 144?",              "12"),
    ("What is the hardest natural mineral?",         "diamond"),
    ("What is the currency of Japan?",               "yen"),
    ("How many sides does a hexagon have?",          "6"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is DNA an abbreviation for?",             "deoxyribonucleic acid"),
]

_REASONING_TEMPLATES: List[Tuple[str, str]] = [
    ("All mammals are warm-blooded. Dogs are mammals. Are dogs warm-blooded?", "yes"),
    ("If it rains, the ground gets wet. The ground is wet. Did it necessarily rain?", "not necessarily"),
    ("All birds have wings. Penguins are birds. Do penguins have wings?", "yes"),
    ("No reptiles are mammals. Snakes are reptiles. Are snakes mammals?", "no"),
    ("If A implies B and B implies C, does A imply C?", "yes"),
    ("All prime numbers greater than 2 are odd. Is 7 prime and odd?", "yes"),
    ("Socrates is a man. All men are mortal. Is Socrates mortal?", "yes"),
    ("If today is Monday, what day is tomorrow?", "Tuesday"),
    ("A is taller than B. B is taller than C. Who is tallest?", "A"),
    ("All cats are animals. Some animals are pets. Are all cats pets?", "not necessarily"),
]

_ANALOGY_TEMPLATES: List[Tuple[str, str]] = [
    ("Hot is to cold as day is to ?",        "night"),
    ("Dog is to puppy as cat is to ?",       "kitten"),
    ("Book is to library as painting is to ?", "museum"),
    ("Foot is to shoe as hand is to ?",      "glove"),
    ("Doctor is to hospital as teacher is to ?", "school"),
    ("Fish is to water as bird is to ?",     "air"),
    ("Paris is to France as Rome is to ?",   "Italy"),
    ("Ear is to hear as eye is to ?",        "see"),
    ("Keyboard is to type as brush is to ?", "paint"),
    ("Addition is to sum as multiplication is to ?", "product"),
]

_SCIENCE_TEMPLATES: List[Tuple[str, str]] = [
    ("A car has mass 1000 kg and acceleration 2 m/s². What is the force?", "2000 N"),
    ("What is the kinetic energy of a 2 kg object moving at 3 m/s?", "9 J"),
    ("Water is H2O. How many atoms are in one water molecule?", "3"),
    ("If voltage is 12V and resistance is 4 ohms, what is the current?", "3 A"),
    ("A wave has frequency 5 Hz. What is its period?", "0.2 s"),
    ("What happens to gas pressure when volume decreases at constant temperature?", "pressure increases"),
    ("If half-life is 10 years, after 20 years what fraction remains?", "1/4"),
]

_LOGIC_TEMPLATES: List[Tuple[str, str]] = [
    ("A AND True",       "A"),
    ("A OR False",       "A"),
    ("A AND False",      "False"),
    ("A OR True",        "True"),
    ("NOT NOT A",        "A"),
    ("A AND A",          "A"),
    ("A OR A",           "A"),
    ("NOT (NOT A OR NOT B)", "A AND B"),
    ("A AND (NOT A)",    "False"),
    ("A OR (NOT A)",     "True"),
]

_LANGUAGE_TEMPLATES: List[Tuple[str, str]] = [
    ("What is the plural of 'mouse'?",              "mice"),
    ("What is the past tense of 'run'?",            "ran"),
    ("What does 'benevolent' mean?",                "kind, well-meaning"),
    ("What is a synonym for 'happy'?",              "joyful"),
    ("What is the opposite of 'ancient'?",          "modern"),
    ("Rearrange: 'lazy the dog fox quick brown over jumps the'", "the quick brown fox jumps over the lazy dog"),
    ("What is an anagram of 'listen'?",             "silent"),
]

_PLANNING_TEMPLATES: List[Tuple[str, str]] = [
    ("What are the steps to make a cup of tea?",
     "boil water, add tea bag, pour water, steep, remove bag, add milk/sugar optionally"),
    ("What are the steps to solve a quadratic equation?",
     "write in standard form, compute discriminant, apply quadratic formula, simplify"),
    ("What are the high-level steps to train a machine learning model?",
     "collect data, preprocess, choose model, train, evaluate, tune, deploy"),
    ("What steps do you take to debug a Python error?",
     "read error message, find line number, inspect code, add print statements, fix, retest"),
]

_SOCIAL_TEMPLATES: List[Tuple[str, str]] = [
    ("Sally puts her ball in a basket and leaves. Anne moves the ball to a box. "
     "Where will Sally look for the ball when she returns?",
     "in the basket (Sally doesn't know Anne moved it)"),
    ("John thinks the meeting is at 3pm. It was moved to 4pm but he wasn't told. "
     "What time does John think the meeting is?",
     "3pm"),
    ("If someone smiles while saying 'I'm fine', what might they actually be feeling?",
     "possibly not fine; the smile may mask distress"),
]

# ── Domain weight table ───────────────────────────────────────────────────────

_DOMAIN_WEIGHTS: Dict[str, float] = {
    "math":      0.20,
    "logic":     0.10,
    "factual":   0.20,
    "science":   0.10,
    "reasoning": 0.10,
    "analogy":   0.10,
    "code":      0.05,
    "language":  0.05,
    "planning":  0.05,
    "social":    0.05,
}

_TEMPLATE_BANKS = {
    "factual":   _FACTUAL_TEMPLATES,
    "reasoning": _REASONING_TEMPLATES,
    "analogy":   _ANALOGY_TEMPLATES,
    "science":   _SCIENCE_TEMPLATES,
    "logic":     _LOGIC_TEMPLATES,
    "language":  _LANGUAGE_TEMPLATES,
    "planning":  _PLANNING_TEMPLATES,
    "social":    _SOCIAL_TEMPLATES,
}

# Math / code are generated dynamically (no fixed templates)

# ── LLM-generated problem prompts ────────────────────────────────────────────

_LLM_GENERATE_PROMPTS: Dict[str, str] = {
    "factual": (
        "Generate 5 factual trivia questions (science, geography, history, biology) "
        "with short answers. Format as JSON list: [{\"q\":\"...\",\"a\":\"...\"}]"
    ),
    "reasoning": (
        "Generate 5 logical reasoning problems (syllogisms, deductive arguments). "
        "Format as JSON list: [{\"q\":\"...\",\"a\":\"...\"}]"
    ),
    "analogy": (
        "Generate 5 analogy problems in the format 'X is to Y as A is to ?'. "
        "Format as JSON list: [{\"q\":\"...\",\"a\":\"...\"}]"
    ),
    "science": (
        "Generate 5 science word problems (physics, chemistry, biology) "
        "with numeric or short answers. Format as JSON list: [{\"q\":\"...\",\"a\":\"...\"}]"
    ),
    "language": (
        "Generate 5 language questions (grammar, vocabulary, word puzzles). "
        "Format as JSON list: [{\"q\":\"...\",\"a\":\"...\"}]"
    ),
    "code": (
        "Generate 5 short Python coding tasks with expected outputs. "
        "Format as JSON list: [{\"q\":\"Write Python to: ...\",\"a\":\"...\"}]"
    ),
    "planning": (
        "Generate 5 step-by-step planning tasks (everyday procedures, algorithms). "
        "Format as JSON list: [{\"q\":\"What are the steps to ...?\",\"a\":\"step1, step2, ...\"}]"
    ),
}

# Math seeds (just generate varied expressions)
_MATH_SEEDS = [
    "x + 0", "x * 1", "x - x", "x * 0", "0 + x", "1 * x",
    "not not x", "A AND True", "A OR False", "x + 1 - 1",
    "2 * x + 3 * x", "x * (y + 0)", "(x + y) * 1",
    "a * b * 0", "x / 1", "x - 0",
]


class GeneralCurriculum:
    """
    Generates a balanced batch of diverse problems for general intelligence training.

    Each call to generate_batch(n) returns n problems drawn proportionally from
    all 10 problem domains, mixing static templates with LLM-generated problems.
    """

    def __init__(self):
        self._llm_ready   = False
        self._call_llm    = None
        self._llm_cache: List[GeneralProblem] = []   # buffered LLM problems
        self._attempt     = 0
        self._init_llm()

    def _init_llm(self):
        try:
            from sare.interface.llm_bridge import _call_llm
            self._call_llm  = _call_llm
            self._llm_ready = True
        except Exception:
            pass

    # ── Public API ────────────────────────────────────────────────────────────

    def _get_adaptive_weights(self) -> Dict[str, float]:
        """Compute dynamic domain weights from WorldModel surprise + homeostasis."""
        weights = dict(_DOMAIN_WEIGHTS)

        # Factor 1: WorldModel per-domain surprise (boost surprising domains)
        try:
            from sare.memory.world_model import get_world_model
            for domain, surprise in (get_world_model().get_high_surprise_domains(top_n=4) or []):
                if domain in weights:
                    weights[domain] = min(0.40, weights[domain] * (1.0 + surprise * 0.15))
        except Exception:
            pass

        # Factor 2: Homeostasis recommendation
        try:
            from sare.meta.homeostasis import get_homeostasis
            rec = get_homeostasis().get_behavior_recommendation()
            if rec == "explore":
                # Flatten toward uniform
                weights = {d: (w + 0.10) / 2.0 for d, w in weights.items()}
            elif rec in ("deepen_weak_domain", "consolidate"):
                # Boost domains below ZPD sweet-spot solve rate
                if _STATS_PATH.exists():
                    try:
                        _s = json.loads(_STATS_PATH.read_text())
                        for domain, dstats in _s.items():
                            if domain in weights:
                                sr = float(dstats.get("solve_rate", 1.0) or 1.0)
                                if sr < 0.45:
                                    weights[domain] = min(0.40, weights[domain] * 1.5)
                    except Exception:
                        pass
        except Exception:
            pass

        # Normalize to sum=1
        total = sum(weights.values())
        if total > 0:
            return {d: w / total for d, w in weights.items()}
        return dict(_DOMAIN_WEIGHTS)

    def generate_batch(self, size: int = 20) -> List[GeneralProblem]:
        """Return a list of `size` diverse problems."""
        problems: List[GeneralProblem] = []

        # Drain priority queue first (up to 1/3 of batch = curriculum debt)
        _priority_slots = max(1, size // 3)
        with _PRIORITY_LOCK:
            _priority = _PRIORITY_QUEUE[:_priority_slots]
            del _PRIORITY_QUEUE[:len(_priority)]
        problems.extend(_priority)

        _remaining = size - len(problems)
        _weights = self._get_adaptive_weights()
        domains = random.choices(
            list(_weights.keys()),
            weights=list(_weights.values()),
            k=_remaining,
        )
        for domain in domains:
            p = self._generate_one(domain)
            if p is not None:
                problems.append(p)
        # Inject self-generated questions from QuestionGenerator (up to 3 per batch)
        try:
            from sare.curiosity.question_generator import get_question_generator
            _pending = get_question_generator().get_pending_questions()[:3]
            for _q in _pending:
                _dom = _q.domain if _q.domain in _DOMAIN_WEIGHTS else "factual"
                problems.append(GeneralProblem(
                    problem_id=f"selfq_{_q.question_id}",
                    text=_q.text,
                    domain=_dom,
                    difficulty=max(0.5, _q.priority),
                ))
        except Exception:
            pass

        self._attempt += 1
        return problems

    def generate_one(self, domain: str | None = None) -> Optional[GeneralProblem]:
        if domain is None:
            domain = random.choices(
                list(_DOMAIN_WEIGHTS.keys()),
                weights=list(_DOMAIN_WEIGHTS.values()),
            )[0]
        return self._generate_one(domain)

    # ── Internal generators ───────────────────────────────────────────────────

    def _generate_one(self, domain: str) -> Optional[GeneralProblem]:
        pid = f"{domain}_{int(time.time()*1000) % 10**7}"

        # Math: pick a random seed expression
        if domain == "math":
            expr = random.choice(_MATH_SEEDS)
            return GeneralProblem(problem_id=pid, text=expr, domain="math", difficulty=0.3)

        # Try LLM cache first
        cached = [p for p in self._llm_cache if p.domain == domain]
        if cached:
            p = random.choice(cached)
            self._llm_cache = [x for x in self._llm_cache if x is not p]
            return p

        # Try static template bank
        bank = _TEMPLATE_BANKS.get(domain)
        if bank:
            q, a = random.choice(bank)
            return GeneralProblem(
                problem_id=pid, text=q, domain=domain,
                expected=a, difficulty=0.4,
            )

        # Fall back to LLM generation
        return self._generate_via_llm(domain, pid)

    def _generate_via_llm(self, domain: str, pid: str) -> Optional[GeneralProblem]:
        if not self._llm_ready:
            return None
        prompt = _LLM_GENERATE_PROMPTS.get(domain)
        if not prompt:
            return None
        try:
            raw = self._call_llm(prompt)
            # Parse JSON list
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if not match:
                return None
            items = json.loads(match.group())
            # Cache all but first
            new_problems = []
            for item in items:
                q = item.get("q") or item.get("question", "")
                a = item.get("a") or item.get("answer", "")
                if q:
                    new_problems.append(GeneralProblem(
                        problem_id=f"{domain}_{int(time.time()*1000) % 10**7}",
                        text=q.strip(), domain=domain,
                        expected=a.strip() if a else None,
                        difficulty=0.5,
                    ))
            if new_problems:
                self._llm_cache.extend(new_problems[1:])
                return new_problems[0]
        except Exception as e:
            log.debug("[GeneralCurriculum] LLM generate failed for %s: %s", domain, e)
        return None


import re  # needed inside _generate_via_llm
