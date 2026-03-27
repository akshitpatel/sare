"""
GeneralSolver — Universal problem-solving engine for SARE-HX.

Routes any input (math, logic, code, factual, science, reasoning, analogy)
to the best available solver, combining symbolic reasoning with LLM intelligence.

Intelligence hierarchy:
  1. Symbolic engine  — deterministic, perfect on math/logic expressions
  2. Hybrid           — symbolic first, LLM extension on failure
  3. LLM reasoning    — chain-of-thought for factual/science/analogy
  4. LLM + execution  — code problems with optional sandbox execution
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ── Problem type constants ────────────────────────────────────────────────────
PTYPE_MATH      = "math"
PTYPE_LOGIC     = "logic"
PTYPE_CODE      = "code"
PTYPE_FACTUAL   = "factual"
PTYPE_REASONING = "reasoning"
PTYPE_ANALOGY   = "analogy"
PTYPE_SCIENCE   = "science"
PTYPE_LANGUAGE  = "language"
PTYPE_PLANNING  = "planning"
PTYPE_SOCIAL    = "social"

ALL_TYPES = [
    PTYPE_MATH, PTYPE_LOGIC, PTYPE_CODE, PTYPE_FACTUAL,
    PTYPE_REASONING, PTYPE_ANALOGY, PTYPE_SCIENCE,
    PTYPE_LANGUAGE, PTYPE_PLANNING, PTYPE_SOCIAL,
]

# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class GeneralSolveResult:
    problem_id:   str
    problem_text: str
    problem_type: str
    answer:       str
    confidence:   float          # 0.0–1.0
    reasoning:    str
    solver_used:  str            # "symbolic" | "llm" | "hybrid" | "code_gen"
    solved:       bool
    elapsed_ms:   float
    lesson:       Optional[str] = None   # extractable rule/fact for knowledge base
    domain:       Optional[str] = None
    sub_steps:    List[str]     = field(default_factory=list)

# ── Pattern classifiers ───────────────────────────────────────────────────────

_CODE_RE = re.compile(
    r'\b(def |class |return |import |for |while |elif |lambda |print\(|```)',
    re.IGNORECASE,
)
_LOGIC_RE = re.compile(
    r'\b(AND|OR|NOT|TRUE|FALSE|implies|iff|XOR|NAND|NOR)\b'
    r'|not\s+not\b'
    r'|\b[A-Z]\s+(AND|OR)\s+[A-Z]\b',
)
_MATH_RE = re.compile(
    r'\d+\s*[\+\-\*\/\^]\s*\d+'          # arithmetic
    r'|[xXyYzZ]\s*[\+\-\*\/\^]'           # algebra
    r'|\b(sin|cos|tan|log|exp|sqrt)\s*\(' # functions
    r'|^\s*[\d\+\-\*\/xXyYzZ\s\^\(\)\.]+\s*$',  # pure expression
)
_SCIENCE_RE = re.compile(
    r'\b(force|mass|velocity|acceleration|energy|momentum|gravity|quantum|'
    r'atom|molecule|cell|DNA|gene|photon|electron|proton|neutron|'
    r'Newton|Einstein|Darwin|Faraday|Bohr|Heisenberg|'
    r'F\s*=\s*m|E\s*=\s*mc)\b',
    re.IGNORECASE,
)
_ANALOGY_RE = re.compile(r'\bis to\b.{1,60}\bas\b', re.IGNORECASE)
_REASONING_RE = re.compile(
    r'\b(if all|if some|if no|every|therefore|it follows|conclude|'
    r'given that|suppose|assume|all\s+\w+\s+are|are\s+\w+\s+\w+\?|'
    r'syllogism|deduct|infer|necessarily|must be)\b',
    re.IGNORECASE,
)
_PLANNING_RE = re.compile(
    r'\b(steps to|how to|procedure for|algorithm for|plan for|'
    r'what are the steps|in what order)\b',
    re.IGNORECASE,
)
_SOCIAL_RE = re.compile(
    r'\b(thinks|believes|knows|feels|wants|expects|'
    r'theory of mind|false belief|perspective)\b',
    re.IGNORECASE,
)

# ── Stats path ────────────────────────────────────────────────────────────────
_STATE_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "general_solver_stats.json"


class GeneralSolver:
    """Universal problem solver for SARE-HX general intelligence."""

    def __init__(self):
        self._stats: Dict[str, Dict] = {}
        self._load_stats()
        self._symbolic_ready = False
        self._llm_ready = False
        self._kb_lookup = None
        self._fact_ingester = None
        self._init_subsystems()
        self._init_kb()

    # ── Init ─────────────────────────────────────────────────────────────────

    def _init_subsystems(self):
        try:
            from sare.interface.llm_bridge import _call_llm
            self._call_llm = _call_llm
            self._llm_ready = True
            log.debug("[GeneralSolver] LLM bridge ready")
        except Exception as e:
            log.debug("[GeneralSolver] LLM unavailable: %s", e)

        try:
            from sare.engine import load_problem, BeamSearch, EnergyEvaluator, get_transforms
            self._load_problem = load_problem
            self._searcher     = BeamSearch()
            self._energy       = EnergyEvaluator()
            self._transforms   = get_transforms(include_macros=True)
            self._symbolic_ready = True
            log.debug("[GeneralSolver] Symbolic engine ready")
        except Exception as e:
            log.warning("[GeneralSolver] Symbolic engine unavailable: %s", e)

    def _init_kb(self):
        try:
            from sare.memory.knowledge_lookup import KnowledgeLookup
            self._kb_lookup = KnowledgeLookup()
            log.debug("[GeneralSolver] KnowledgeLookup ready")
        except Exception as e:
            log.debug("[GeneralSolver] KnowledgeLookup unavailable: %s", e)

        try:
            from sare.memory.fact_ingester import FactIngester
            self._fact_ingester = FactIngester()
            log.debug("[GeneralSolver] FactIngester ready")
        except Exception as e:
            log.debug("[GeneralSolver] FactIngester unavailable: %s", e)

        try:
            from sare.cognition.self_verifier import get_self_verifier
            self._verifier = get_self_verifier()
            log.debug("[GeneralSolver] SelfVerifier ready")
        except Exception as e:
            self._verifier = None
            log.debug("[GeneralSolver] SelfVerifier unavailable: %s", e)

    # ── Classification ────────────────────────────────────────────────────────

    def classify(self, text: str) -> str:
        """Classify problem type via regex heuristics (fast, no LLM required)."""
        if _CODE_RE.search(text):        return PTYPE_CODE
        if _LOGIC_RE.search(text):       return PTYPE_LOGIC
        if _MATH_RE.search(text):        return PTYPE_MATH
        if _ANALOGY_RE.search(text):     return PTYPE_ANALOGY
        if _SOCIAL_RE.search(text):      return PTYPE_SOCIAL
        if _SCIENCE_RE.search(text):     return PTYPE_SCIENCE
        if _REASONING_RE.search(text):   return PTYPE_REASONING
        if _PLANNING_RE.search(text):    return PTYPE_PLANNING
        return PTYPE_FACTUAL

    # ── Main solve entry point ────────────────────────────────────────────────

    def solve_with_known_answer(
        self,
        problem_text: str,
        expected_answer: str,
        problem_type: str,
    ) -> GeneralSolveResult:
        """Fast path: we already know the correct answer (from template bank).
        Checks KB cache first, then falls back to template. Always stores result in KB."""
        t0 = time.time()
        pid = str(uuid.uuid4())[:8]
        ptype = problem_type

        # KB cache check first — skip template if same question was already seen
        if ptype not in (PTYPE_MATH, PTYPE_LOGIC) and self._kb_lookup is not None:
            try:
                from sare.memory.knowledge_lookup import DIRECT_THRESHOLD
                hit = self._kb_lookup.lookup(problem_text, ptype)
                if hit is not None and hit.confidence >= DIRECT_THRESHOLD:
                    self._record_stats(ptype, True, hit.confidence)
                    return GeneralSolveResult(
                        problem_id=pid, problem_text=problem_text,
                        problem_type=ptype, answer=hit.answer,
                        confidence=hit.confidence,
                        reasoning=f"Retrieved from KB ({hit.source})",
                        solver_used="kb_cache", solved=True,
                        elapsed_ms=(time.time() - t0) * 1000,
                        domain=ptype,
                    )
            except Exception:
                pass

        _lesson = expected_answer if ptype not in (PTYPE_MATH, PTYPE_LOGIC, PTYPE_CODE) else None
        result = GeneralSolveResult(
            problem_id=pid, problem_text=problem_text,
            problem_type=ptype, answer=expected_answer,
            confidence=0.95, reasoning="Template answer (known)",
            solver_used="template", solved=True,
            elapsed_ms=(time.time() - t0) * 1000,
            domain=ptype, lesson=_lesson,
        )
        self._record_stats(ptype, True, 0.95)

        # Store in KB so future lookups serve from cache
        if ptype not in (PTYPE_MATH, PTYPE_LOGIC) and self._fact_ingester is not None:
            try:
                self._fact_ingester.ingest(problem_text, expected_answer, ptype,
                                           confidence=0.95)
            except Exception as _fi_err:
                log.debug("[GeneralSolver] FactIngester (template) error: %s", _fi_err)

        return result

    def solve(
        self,
        problem_text: str,
        context: str = "",
        problem_type: str | None = None,
    ) -> GeneralSolveResult:
        t0 = time.time()
        pid = str(uuid.uuid4())[:8]
        ptype = problem_type or self.classify(problem_text)

        # ── KB cache lookup (non-symbolic problems only) ──────────────────────
        if ptype not in (PTYPE_MATH, PTYPE_LOGIC) and self._kb_lookup is not None:
            try:
                from sare.memory.knowledge_lookup import DIRECT_THRESHOLD, CONTEXT_THRESHOLD
                hit = self._kb_lookup.lookup(problem_text, ptype)
                if hit is not None and hit.confidence >= DIRECT_THRESHOLD:
                    result = GeneralSolveResult(
                        problem_id=pid, problem_text=problem_text,
                        problem_type=ptype, answer=hit.answer,
                        confidence=hit.confidence,
                        reasoning=f"Retrieved from KB ({hit.source})",
                        solver_used="kb_cache", solved=True, elapsed_ms=0,
                        domain=ptype,
                    )
                    result.elapsed_ms = (time.time() - t0) * 1000
                    self._record_stats(ptype, True, hit.confidence)
                    return result
                # Partial hit: inject as context prefix for LLM
                if hit is not None and hit.confidence >= CONTEXT_THRESHOLD:
                    context = f"Known facts: {'; '.join(hit.context_facts[:3])}\n" + context
            except Exception as _kb_err:
                log.debug("[GeneralSolver] KB lookup error: %s", _kb_err)

        # ── Fact-chain lookup (N-hop inference before LLM) ────────────────────
        if ptype in (PTYPE_FACTUAL, PTYPE_SCIENCE, PTYPE_REASONING):
            try:
                from sare.memory.fact_ingester import _extract_triples
                from sare.cognition.fact_inference import get_fact_inference
                _triples = _extract_triples(problem_text, "")
                if _triples:
                    _subj, _pred, _ = _triples[0]
                    _chain_ans = get_fact_inference().chain_to_goal(_subj, _pred, ptype, max_depth=3)
                    if _chain_ans:
                        result = GeneralSolveResult(
                            problem_id=pid, problem_text=problem_text,
                            problem_type=ptype, answer=_chain_ans,
                            confidence=0.75,
                            reasoning="Retrieved via N-hop fact chain",
                            solver_used="fact_chain", solved=True,
                            elapsed_ms=(time.time() - t0) * 1000,
                            domain=ptype,
                        )
                        self._record_stats(ptype, True, 0.75)
                        return result
            except Exception as _fc_err:
                log.debug("[GeneralSolver] Fact chain error: %s", _fc_err)

        # Route to solver
        if ptype in (PTYPE_MATH, PTYPE_LOGIC) and self._symbolic_ready:
            result = self._solve_symbolic(pid, problem_text, ptype)
            # Hybrid fallback: if symbolic fails, try LLM
            if not result.solved and self._llm_ready:
                llm_r = self._solve_with_llm(pid, problem_text, ptype, context)
                if llm_r.solved:
                    llm_r.solver_used = "hybrid"
                    result = llm_r
        elif ptype == PTYPE_CODE:
            result = self._solve_code(pid, problem_text, context)
        elif self._llm_ready:
            result = self._solve_with_llm(pid, problem_text, ptype, context)
        else:
            result = GeneralSolveResult(
                problem_id=pid, problem_text=problem_text,
                problem_type=ptype, answer="[no solver available]",
                confidence=0.0, reasoning="No LLM or symbolic engine available.",
                solver_used="none", solved=False, elapsed_ms=0,
            )

        result.elapsed_ms = (time.time() - t0) * 1000
        self._record_stats(ptype, result.solved, result.confidence)

        # ── Store result in KB for future cache hits ──────────────────────────
        if (result.solved and result.answer
                and result.solver_used not in ("kb_cache", "fact_chain")
                and ptype not in (PTYPE_MATH, PTYPE_LOGIC)
                and self._fact_ingester is not None):
            try:
                _store_conf = result.confidence * 0.75
                if self._verifier is not None:
                    _ok, _mult = self._verifier.verify(problem_text, result.answer, ptype)
                    if _ok:
                        self._fact_ingester.ingest(problem_text, result.answer, ptype,
                                                   confidence=_store_conf * _mult)
                    else:
                        log.debug("[GeneralSolver] Verifier rejected answer for: %s", problem_text[:60])
                else:
                    self._fact_ingester.ingest(problem_text, result.answer, ptype,
                                               confidence=_store_conf)
            except Exception as _fi_err:
                log.debug("[GeneralSolver] FactIngester error: %s", _fi_err)

        return result

    # ── Symbolic solver ───────────────────────────────────────────────────────

    def _solve_symbolic(self, pid: str, text: str, ptype: str) -> GeneralSolveResult:
        try:
            _, g = self._load_problem(text)
            if g is None:
                raise ValueError("parse returned None")

            e_before         = self._energy.compute(g).total
            orig_node_count  = g.node_count      # capture BEFORE search modifies g
            sr = self._searcher.search(
                g, self._energy, self._transforms,
                beam_width=8, budget_seconds=3.0,
            )
            e_after = sr.energy.total
            delta   = e_before - e_after
            solved  = delta > 0.5

            # Build human-readable answer from simplified graph's node labels
            answer = text
            try:
                result_nodes = list(sr.graph.nodes)
                labels = [str(getattr(n, "label", "") or "") for n in result_nodes]
                labels = [l for l in labels if l]
                if labels and len(result_nodes) < orig_node_count:
                    # Fewer nodes = simplification; join labels of result
                    answer = " ".join(labels) if len(labels) > 1 else labels[0]
                elif solved:
                    answer = f"{text} → simplified (Δ={delta:.1f})"
            except Exception:
                if solved:
                    answer = f"{text} → simplified (Δ={delta:.1f})"

            return GeneralSolveResult(
                problem_id=pid, problem_text=text, problem_type=ptype,
                answer=answer,
                confidence=min(1.0, delta / 10.0) if solved else 0.0,
                reasoning=f"Symbolic: energy {e_before:.2f} → {e_after:.2f} (Δ={delta:.2f})",
                solver_used="symbolic", solved=solved, elapsed_ms=0, domain=ptype,
            )
        except Exception as e:
            return GeneralSolveResult(
                problem_id=pid, problem_text=text, problem_type=ptype,
                answer="", confidence=0.0, reasoning=f"Symbolic failed: {e}",
                solver_used="symbolic", solved=False, elapsed_ms=0,
            )

    # ── LLM chain-of-thought solver ───────────────────────────────────────────

    def _solve_with_llm(
        self, pid: str, text: str, ptype: str, context: str = ""
    ) -> GeneralSolveResult:
        ctx_block = f"\nContext:\n{context[:600]}\n" if context else ""
        prompt = (
            f"You are a highly capable reasoning engine. Solve this {ptype} problem "
            f"step by step using clear logical reasoning.\n"
            f"{ctx_block}"
            f"Problem: {text}\n\n"
            f"Think step by step, then write 'Answer: ' followed by your final answer on the last line."
        )
        try:
            raw = self._call_llm(prompt)

            # Extract answer line
            answer = ""
            lines  = raw.strip().split("\n")
            for line in reversed(lines):
                if line.strip().lower().startswith("answer:"):
                    answer = line.split(":", 1)[1].strip()
                    break
            if not answer:
                non_empty = [l.strip() for l in lines if l.strip()]
                answer    = non_empty[-1] if non_empty else raw[:300]

            # Strip markdown and template artifacts from answer
            answer = re.sub(r'\*{1,3}', '', answer).strip()   # **bold** / *italic*
            answer = re.sub(r'^#+\s*', '', answer)             # ## headings
            answer = answer.strip('`').strip()
            answer = re.sub(r'^<(.+)>$', r'\1', answer).strip()  # <value> template literals

            # Extract lesson: the answer itself is the lesson for factual/science domains
            lesson = answer if answer and ptype not in (PTYPE_MATH, PTYPE_LOGIC, PTYPE_CODE) else None
            # Also check for explicit principle lines
            if not lesson:
                for line in lines:
                    low = line.lower()
                    if any(kw in low for kw in ["always ", "rule:", "principle:", "in general,", "therefore,"]):
                        lesson = line.strip()
                        break

            # Extract sub-steps: numbered or bulleted lines
            sub_steps = [
                l.strip() for l in lines
                if re.match(r'^\s*(\d+[\.\)]|[-•*])\s+', l) and l.strip()
            ][:8]

            return GeneralSolveResult(
                problem_id=pid, problem_text=text, problem_type=ptype,
                answer=answer, confidence=0.75 if answer else 0.0,
                reasoning=raw[:1200], solver_used="llm",
                solved=bool(answer), elapsed_ms=0,
                lesson=lesson, domain=ptype, sub_steps=sub_steps,
            )
        except Exception as e:
            log.debug("[GeneralSolver] LLM solve failed: %s", e)
            return GeneralSolveResult(
                problem_id=pid, problem_text=text, problem_type=ptype,
                answer="", confidence=0.0, reasoning=f"LLM error: {e}",
                solver_used="llm", solved=False, elapsed_ms=0,
            )

    # ── Code solver ───────────────────────────────────────────────────────────

    def _solve_code(self, pid: str, text: str, context: str = "") -> GeneralSolveResult:
        if not self._llm_ready:
            return GeneralSolveResult(
                problem_id=pid, problem_text=text, problem_type=PTYPE_CODE,
                answer="", confidence=0.0, reasoning="LLM unavailable",
                solver_used="code_gen", solved=False, elapsed_ms=0,
            )

        prompt = (
            "Write clean, correct Python code to solve the following problem. "
            "Output ONLY the code — no explanation, no markdown fences.\n\n"
            f"Problem: {text}"
        )
        try:
            code = self._call_llm(prompt, use_synthesis_model=True)
            # Strip markdown fences if present
            code = re.sub(r"^```[a-z]*\n?", "", code.strip())
            code = re.sub(r"\n?```$", "", code)

            # Attempt sandbox execution
            exec_output = ""
            try:
                import io, contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    exec(compile(code, "<sandbox>", "exec"), {})  # noqa: S102
                exec_output = buf.getvalue().strip()
            except Exception as exec_e:
                log.debug("[GeneralSolver] Code exec: %s", exec_e)

            answer = exec_output if exec_output else code[:800]

            return GeneralSolveResult(
                problem_id=pid, problem_text=text, problem_type=PTYPE_CODE,
                answer=answer, confidence=0.8 if exec_output else 0.65,
                reasoning=f"Generated code {'+ executed output' if exec_output else '(unexecuted)'}",
                solver_used="code_gen", solved=True, elapsed_ms=0,
                domain="code",
            )
        except Exception as e:
            return GeneralSolveResult(
                problem_id=pid, problem_text=text, problem_type=PTYPE_CODE,
                answer="", confidence=0.0, reasoning=f"Code gen failed: {e}",
                solver_used="code_gen", solved=False, elapsed_ms=0,
            )

    # ── Stats ─────────────────────────────────────────────────────────────────

    def _record_stats(self, ptype: str, solved: bool, confidence: float):
        if ptype not in self._stats:
            self._stats[ptype] = {"attempts": 0, "solved": 0, "total_confidence": 0.0}
        s = self._stats[ptype]
        s["attempts"]          += 1
        s["solved"]            += int(solved)
        s["total_confidence"]  += confidence

        # Save every 50 solves
        total = sum(v["attempts"] for v in self._stats.values())
        if total % 50 == 0:
            self._save_stats()

    def get_stats(self) -> Dict[str, Any]:
        return {
            ptype: {
                "attempts":       s["attempts"],
                "solve_rate":     round(s["solved"] / max(s["attempts"], 1), 3),
                "avg_confidence": round(s["total_confidence"] / max(s["attempts"], 1), 3),
            }
            for ptype, s in self._stats.items()
        }

    def _load_stats(self):
        try:
            if _STATE_PATH.exists():
                self._stats = json.loads(_STATE_PATH.read_text())
        except Exception:
            self._stats = {}

    def _save_stats(self):
        try:
            _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _STATE_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._stats, indent=2))
            os.replace(tmp, _STATE_PATH)
        except Exception as e:
            log.debug("[GeneralSolver] Stats save failed: %s", e)


# ── Singleton ─────────────────────────────────────────────────────────────────

_SINGLETON: Optional[GeneralSolver] = None


def get_general_solver() -> GeneralSolver:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = GeneralSolver()
    return _SINGLETON
