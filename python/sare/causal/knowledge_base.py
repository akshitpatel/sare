"""
Causal Knowledge Base (CKB) — shared registry of induced rules and their
success chains, used by CausalInduction to bias test generation toward
operators that have worked well in the past.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_MEMORY = Path(__file__).resolve().parents[3] / "data" / "memory"
_CKB_PATH = _MEMORY / "causal_knowledge_base.json"


class CausalKnowledgeBase:
    """
    Lightweight shared store for successfully induced rules.

    Tracks which operators have been validated, how many times, and what
    confidence was achieved. Used by CausalInduction.suggest_tests_for_induction()
    to bias positive test case generation toward known-good operators.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._rules: Dict[str, dict] = {}   # rule_id -> {operator, desc, confidence, count}
        self._op_rules: Dict[str, List[str]] = {}  # operator -> [rule_ids]
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def register_rule(self, rule_id: str, operator: str, desc: str,
                      confidence: float) -> None:
        """Record a successfully promoted rule."""
        with self._lock:
            if rule_id in self._rules:
                self._rules[rule_id]["count"] = self._rules[rule_id].get("count", 1) + 1
                self._rules[rule_id]["confidence"] = max(
                    self._rules[rule_id]["confidence"], confidence
                )
            else:
                self._rules[rule_id] = {
                    "operator": operator,
                    "desc": desc,
                    "confidence": round(confidence, 4),
                    "count": 1,
                }
                if operator:
                    self._op_rules.setdefault(operator, []).append(rule_id)
        self._save()

    def suggest_tests_for_induction(self, operator: str) -> List[str]:
        """
        Return rule descriptions for the given operator that have been
        successfully induced before — useful for test-case biasing.

        Returns a list of desc strings (may be empty).
        """
        with self._lock:
            rule_ids = self._op_rules.get(operator or "", [])
            return [
                self._rules[rid]["desc"]
                for rid in rule_ids
                if rid in self._rules
            ]

    def get_all_rules(self) -> List[dict]:
        with self._lock:
            return list(self._rules.values())

    def rule_count(self) -> int:
        with self._lock:
            return len(self._rules)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        try:
            _CKB_PATH.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {"rules": self._rules, "op_rules": self._op_rules}
            _CKB_PATH.write_text(json.dumps(data, indent=2))
        except Exception as e:
            log.debug("CKB save error: %s", e)

    def _load(self):
        try:
            if _CKB_PATH.exists():
                data = json.loads(_CKB_PATH.read_text())
                self._rules = data.get("rules", {})
                self._op_rules = data.get("op_rules", {})
                log.debug("CKB loaded: %d rules", len(self._rules))
        except Exception as e:
            log.debug("CKB load error: %s", e)


# ── Singleton ──────────────────────────────────────────────────────────────────
_ckb: Optional[CausalKnowledgeBase] = None
_ckb_lock = threading.Lock()


def get_ckb() -> CausalKnowledgeBase:
    global _ckb
    if _ckb is None:
        with _ckb_lock:
            if _ckb is None:
                _ckb = CausalKnowledgeBase()
    return _ckb
