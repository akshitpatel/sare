"""
LLMKnowledgeExpander — asks the LLM to generate domain facts when world model
has < THRESHOLD facts for a domain AND last expansion was > MIN_INTERVAL_S ago.
Deposits into WorldModel.add_fact() + CommonSenseBase._add().
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

log = logging.getLogger(__name__)

_STATE_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "llm_expander_state.json"
THRESHOLD = 50         # expand when domain has fewer facts than this
MIN_INTERVAL_S = 3600  # 1-hour cooldown per domain

_EXPAND_DOMAINS = [
    "biology", "chemistry", "computer_science",
    "psychology", "economics", "geography", "history", "linguistics",
]


class LLMKnowledgeExpander:
    def __init__(self):
        self._last: dict = {}
        try:
            if _STATE_PATH.exists():
                self._last = json.loads(_STATE_PATH.read_text())
        except Exception:
            pass

    def should_expand(self, domain: str) -> bool:
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            facts = wm.get_facts(domain) if hasattr(wm, "get_facts") else []
            if len(facts) >= THRESHOLD:
                return False
        except Exception:
            pass
        return (time.time() - self._last.get(domain, 0)) > MIN_INTERVAL_S

    def expand_domain(self, domain: str) -> int:
        from sare.interface.llm_bridge import _call_llm
        from sare.knowledge.commonsense import CommonSenseBase

        prompt = (
            f"Generate 20 factual knowledge triples about '{domain}' as JSON. "
            'Format: [{"subject":"...","relation":"IsA|HasA|Causes|UsedFor|HasProperty",'
            '"object":"...","fact":"one-sentence fact"}] '
            "Use short lowercase words. Output ONLY the JSON array, no explanation."
        )
        try:
            raw = _call_llm(prompt, system_prompt="You are a factual knowledge graph builder.")
        except Exception as exc:
            log.debug("LLM expand_domain(%s) failed: %s", domain, exc)
            return 0

        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if not match:
            return 0
        try:
            items = json.loads(match.group())
        except Exception:
            return 0

        kb = CommonSenseBase()
        kb.load()
        added = 0
        for item in items:
            fact = item.get("fact", "")
            subj = item.get("subject", "")
            rel  = item.get("relation", "IsA")
            obj  = item.get("object", "")
            if fact:
                try:
                    from sare.memory.world_model import get_world_model
                    wm = get_world_model()
                    if hasattr(wm, "add_fact"):
                        wm.add_fact(domain, fact, confidence=0.7, source="llm_expander")
                except Exception:
                    pass
            if subj and obj:
                kb._add(subj, rel, obj)
                added += 1
        kb.save()
        try:
            from sare.memory.world_model import get_world_model
            get_world_model().save()
        except Exception:
            pass

        self._last[domain] = time.time()
        try:
            _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _STATE_PATH.write_text(json.dumps(self._last))
        except Exception:
            pass

        log.info("LLM expanded '%s': +%d triples", domain, added)
        return added
