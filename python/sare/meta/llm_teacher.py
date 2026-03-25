"""
llm_teacher.py — Routes the `seek_human_input` homeostatic drive to the LLM.

When SARE-HX is stuck (social drive high, many failures), instead of waiting
for a human to visit the web UI this module calls the configured LLM and asks
it to act as a teacher.  The LLM response is parsed for `lhs → rhs` rewrite
rules which are injected back into the system as seeds + world-model facts.

Design goals:
  - One clear prompt → deterministic output format → cheap, reliable parsing
  - Throttled: at most once per _MIN_SEEK_INTERVAL_CYCLES daemon cycles
  - All interactions persisted to data/memory/llm_teacher_log.json
  - Fully self-contained; daemon just calls LLMTeacher().seek_and_apply(...)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

_LOG_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "llm_teacher_log.json"
_MIN_SEEK_INTERVAL_CYCLES = 20   # ~10 minutes at 30s daemon interval (600s / 30s = 20 cycles)
_MAX_STUCK_EXPRS = 8             # max expressions to include in prompt
_MAX_RULES_PER_CALL = 10         # cap parsing to avoid runaway injection

_SYSTEM_PROMPT = """\
You are a symbolic math and logic tutor for an AI reasoning engine.
The engine works by rewriting expression graphs using transformation rules.
It is currently stuck on the problems listed below — it cannot find a simplification.

For each expression, provide:
1. The simplified canonical form
2. The general rewrite rule that was applied

Respond ONLY with lines in this exact format (one per expression):
  <original_expression> → <simplified_form>  [rule: <short rule name>]

Examples of valid response lines:
  sin(0) + 0 → 0  [rule: trig_zero_plus_add_zero]
  x + x → 2*x  [rule: combine_like_terms]
  not not A → A  [rule: double_negation]
  3 + 5 → 8  [rule: constant_folding]

Do not add explanations, headers, or any other text — ONLY the formatted lines.
"""


# ── Parsing ────────────────────────────────────────────────────────────────────

_RULE_RE = re.compile(
    r"^\s*(.+?)\s*[→\->]+\s*(.+?)\s*\[rule:\s*([^\]]+)\]",
    re.IGNORECASE,
)
_ARROW_RE = re.compile(r"^\s*(.+?)\s*[→\->]+\s*(.+?)$")


def _parse_rules(text: str) -> List[Tuple[str, str, str]]:
    """
    Extract (lhs, rhs, rule_name) triples from LLM response text.
    Returns up to _MAX_RULES_PER_CALL entries.
    """
    results = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _RULE_RE.match(line)
        if m:
            lhs, rhs, rule_name = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            results.append((lhs, rhs, rule_name))
            continue
        # Fallback: bare arrow without [rule:] tag
        m2 = _ARROW_RE.match(line)
        if m2:
            lhs, rhs = m2.group(1).strip(), m2.group(2).strip()
            if lhs and rhs and lhs != rhs:
                results.append((lhs, rhs, "llm_taught"))
        if len(results) >= _MAX_RULES_PER_CALL:
            break
    return results


# ── Main class ─────────────────────────────────────────────────────────────────

class LLMTeacher:
    """
    Routes seek_human_input → LLM → injected rules.

    Usage (from daemon loop):
        teacher = LLMTeacher()
        teacher.seek_and_apply(
            stuck_exprs=["sin(0) + 0", "x + x + x"],
            homeostasis=_homeostasis,
            cycle=cycle,
        )
    """

    def __init__(self):
        self._log: list = []
        self._last_seek_cycle: int = -_MIN_SEEK_INTERVAL_CYCLES
        self._load_log()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_log(self):
        try:
            if _LOG_PATH.exists():
                self._log = json.loads(_LOG_PATH.read_text())
                # Restore last-seek-cycle from most recent entry
                if self._log:
                    self._last_seek_cycle = self._log[-1].get("cycle", 0)
        except Exception as e:
            log.debug("[LLMTeacher] log load error: %s", e)

    def _save_log(self):
        try:
            _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _LOG_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._log[-200:], indent=2))
            os.replace(tmp, _LOG_PATH)
        except Exception as e:
            log.debug("[LLMTeacher] log save error: %s", e)

    # ── Throttle check ────────────────────────────────────────────────────────

    def should_seek(self, cycle: int) -> bool:
        return (cycle - self._last_seek_cycle) >= _MIN_SEEK_INTERVAL_CYCLES

    # ── Core seek-and-apply ───────────────────────────────────────────────────

    def seek_and_apply(
        self,
        stuck_exprs: List[str],
        homeostasis=None,
        cycle: int = 0,
        curriculum_gen=None,
    ) -> dict:
        """
        Ask the LLM to teach us rules for the stuck expressions.
        Returns a summary dict: {rules_injected, seeds_added, raw_response}.
        """
        if not self.should_seek(cycle):
            log.debug("[LLMTeacher] throttled (last=%d, now=%d)", self._last_seek_cycle, cycle)
            return {"rules_injected": 0, "seeds_added": 0, "throttled": True}

        if not stuck_exprs:
            return {"rules_injected": 0, "seeds_added": 0, "reason": "no_stuck_exprs"}

        exprs_to_ask = stuck_exprs[:_MAX_STUCK_EXPRS]
        user_prompt = "Stuck expressions:\n" + "\n".join(f"  - {e}" for e in exprs_to_ask)

        log.info("[LLMTeacher] Seeking input for %d stuck exprs: %s",
                 len(exprs_to_ask), exprs_to_ask)

        try:
            from sare.interface.llm_bridge import _call_llm
            raw = _call_llm(user_prompt, system_prompt=_SYSTEM_PROMPT)
        except Exception as e:
            log.warning("[LLMTeacher] LLM call failed: %s", e)
            return {"rules_injected": 0, "seeds_added": 0, "error": str(e)}

        log.info("[LLMTeacher] LLM responded (%d chars)", len(raw))
        log.debug("[LLMTeacher] Raw:\n%s", raw)

        rules = _parse_rules(raw)
        rules_injected = 0
        seeds_added = 0

        for lhs, rhs, rule_name in rules:
            # 1. Add lhs as a new curriculum seed
            if curriculum_gen is not None:
                try:
                    from sare.engine import load_problem
                    _, g = load_problem(lhs)
                    if g:
                        curriculum_gen.add_seed(g)
                        seeds_added += 1
                except Exception as e:
                    log.debug("[LLMTeacher] seed add failed for '%s': %s", lhs, e)

            # 2. Record as world model fact
            try:
                from sare.memory.world_model import get_world_model
                wm = get_world_model()
                domain = _infer_domain(lhs)
                wm.add_fact(domain, f"{lhs} → {rhs}", source="llm_teacher")
            except Exception as e:
                log.debug("[LLMTeacher] world model fact failed: %s", e)

            # 3. Record in autobiographical memory
            try:
                from sare.memory.autobiographical import get_autobiographical_memory
                am = get_autobiographical_memory()
                am.record(
                    event_type="rule_discovered",
                    domain=_infer_domain(lhs),
                    description=f"LLM taught: {lhs} → {rhs}  [{rule_name}]",
                    importance=0.6,
                )
            except Exception as e:
                log.debug("[LLMTeacher] autobio record failed: %s", e)

            rules_injected += 1
            log.info("[LLMTeacher] Rule: %s → %s  [%s]", lhs, rhs, rule_name)

        # 4. Satisfy social drive
        if homeostasis is not None:
            try:
                homeostasis.satisfy("social", 0.35)
            except Exception as e:
                log.debug("[LLMTeacher] homeostasis satisfy failed: %s", e)

        # 5. Persist interaction log
        self._last_seek_cycle = cycle
        self._log.append({
            "cycle": cycle,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stuck_exprs": exprs_to_ask,
            "rules_parsed": rules,
            "rules_injected": rules_injected,
            "seeds_added": seeds_added,
            "raw_chars": len(raw),
        })
        self._save_log()

        log.info("[LLMTeacher] Done: %d rules injected, %d seeds added",
                 rules_injected, seeds_added)
        return {
            "rules_injected": rules_injected,
            "seeds_added": seeds_added,
            "raw_response": raw,
        }


# ── Domain inference (simple heuristic) ────────────────────────────────────────

def _infer_domain(expr: str) -> str:
    expr_l = expr.lower()
    if any(k in expr_l for k in ("and", "or", "not", "true", "false", "→", "⊢")):
        return "logic"
    if any(k in expr_l for k in ("sin", "cos", "tan", "log", "exp", "sqrt", "ln")):
        return "math"
    if "=" in expr_l:
        return "algebra"
    if any(k in expr_l for k in ("def ", "return", "if ", "while", "class ")):
        return "code"
    try:
        float(expr_l.replace(" ", "").replace("*", "").replace("+", "").replace("-", ""))
        return "arithmetic"
    except Exception:
        pass
    return "algebra"


# ── Singleton ─────────────────────────────────────────────────────────────────

_instance: Optional[LLMTeacher] = None


def get_llm_teacher() -> LLMTeacher:
    global _instance
    if _instance is None:
        _instance = LLMTeacher()
    return _instance
