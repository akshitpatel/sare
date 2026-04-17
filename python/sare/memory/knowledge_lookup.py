"""
KnowledgeLookup — Fan-out retriever querying all KB layers in parallel.

Queries WorldModel facts, KnowledgeGraph beliefs, and CommonSenseBase
to return the best available answer for a given question/domain pair.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

log = logging.getLogger(__name__)

DIRECT_THRESHOLD = 0.85   # return immediately, no LLM needed
CONTEXT_THRESHOLD = 0.50  # inject as context into LLM prompt

# Question extraction patterns
_WHAT_IS_RE   = re.compile(r"what is (?:the )?(\w[\w\s]*?) of (.+?)[\?\.]*$", re.IGNORECASE)
_WHAT_ARE_RE  = re.compile(r"what (?:is|are) (.+?)[\?\.]*$", re.IGNORECASE)
_DEF_RE       = re.compile(r"define (.+?)[\?\.]*$", re.IGNORECASE)
_HOW_MUCH_RE  = re.compile(r"how (?:much|many|fast|far|long) (?:is|are|does|do) (.+?)[\?\.]*$", re.IGNORECASE)


@dataclass
class KBHit:
    answer: str
    confidence: float
    source: str          # "world_model" | "commonsense" | "knowledge_graph" | "fused"
    context_facts: List[str] = field(default_factory=list)
    subject: str = ""
    domain: str = ""


class KnowledgeLookup:
    """Fan-out KB retriever. Thread-safe (reads only; no internal mutable state)."""

    def __init__(self):
        self._hit_history: List[bool] = []   # last 100 lookups: True=hit, False=miss

    # ── Public API ─────────────────────────────────────────────────────────────

    def lookup(self, question: str, domain: str = "general") -> Optional[KBHit]:
        """
        Try all KB layers for an answer. Returns KBHit if confidence ≥ CONTEXT_THRESHOLD,
        else None. Caller should check hit.confidence vs DIRECT_THRESHOLD.
        """
        subject, predicate = self._extract_subject_predicate(question)
        keywords = self._keywords(question)

        wm_hit    = self._query_world_model(question, domain, subject, predicate, keywords)
        kg_hit    = self._query_knowledge_graph(domain, subject, keywords)
        cs_hit    = self._query_commonsense(domain, subject, keywords)

        hit = self._fuse(wm_hit, kg_hit, cs_hit, domain, subject)

        is_hit = hit is not None and hit.confidence >= CONTEXT_THRESHOLD
        self._hit_history.append(is_hit)
        if len(self._hit_history) > 100:
            self._hit_history.pop(0)

        return hit if is_hit else None

    def get_stats(self) -> dict:
        total = len(self._hit_history)
        hits  = sum(self._hit_history)
        return {
            "lookups_tracked": total,
            "kb_hit_rate_last_100": round(hits / max(total, 1), 3),
            "direct_threshold": DIRECT_THRESHOLD,
            "context_threshold": CONTEXT_THRESHOLD,
        }

    def get_hit_rate(self) -> float:
        total = len(self._hit_history)
        return round(sum(self._hit_history) / max(total, 1), 3)

    # ── Subject/predicate extraction ───────────────────────────────────────────

    def _extract_subject_predicate(self, question: str):
        m = _WHAT_IS_RE.search(question)
        if m:
            return m.group(2).strip().lower(), m.group(1).strip().lower()
        m = _HOW_MUCH_RE.search(question)
        if m:
            return m.group(1).strip().lower(), "quantity"
        m = _DEF_RE.search(question)
        if m:
            return m.group(1).strip().lower(), "definition"
        m = _WHAT_ARE_RE.search(question)
        if m:
            return m.group(1).strip().lower(), "description"
        # Fallback: first non-stop word as subject
        words = [w for w in question.lower().split() if w not in {"what","is","are","the","a","an","of","how","much","many","does","do","define"}]
        return (words[0] if words else question[:20].lower()), "answer"

    def _keywords(self, question: str) -> List[str]:
        stop = {"what","is","are","the","a","an","of","how","much","many","does","do","define","please","can","you","tell","me","about"}
        return [w.lower().strip("?.,!") for w in question.split() if w.lower().strip("?.,!") not in stop and len(w) > 2]

    # ── WorldModel query ───────────────────────────────────────────────────────

    def _query_world_model(self, question: str, domain: str, subject: str, predicate: str, keywords: List[str]) -> Optional[KBHit]:
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()

            # Path 1: Direct structured belief lookup (subject + predicate → value)
            # This is O(1) and highly precise — works for facts ingested via FactIngester
            if subject and predicate and predicate not in ("answer", "description"):
                belief = wm.get_belief(subject, predicate) if hasattr(wm, 'get_belief') else None
                if belief is not None and belief.value and belief.confidence >= CONTEXT_THRESHOLD:
                    return KBHit(
                        answer=belief.value,
                        confidence=min(1.0, belief.confidence + 0.1),  # structured hits are more reliable
                        source="world_model_structured",
                        context_facts=[f"{subject} {predicate}: {belief.value}"],
                        subject=subject,
                        domain=domain,
                    )

            # Path 2: Semantic search over structured beliefs
            if hasattr(wm, 'search_beliefs') and keywords:
                belief_results = wm.search_beliefs(question, domain=domain, top_k=3)
                if belief_results:
                    b = belief_results[0]
                    val = b.get("value", "")
                    if val and b.get("confidence", 0) >= CONTEXT_THRESHOLD:
                        ctx = [f"{r.get('subject','')} {r.get('predicate','')}={r.get('value','')}" for r in belief_results]
                        return KBHit(
                            answer=val,
                            confidence=round(b.get("confidence", 0.5), 3),
                            source="world_model_belief",
                            context_facts=ctx,
                            subject=subject,
                            domain=domain,
                        )

            # Path 3: Keyword search over prose facts (original path)
            facts_to_check = wm.get_facts(domain) + (wm.get_facts("general") if domain != "general" else [])
            best_score = 0.0
            best_fact  = None

            for fd in facts_to_check:
                fact_text = fd.get("fact", "").lower()
                conf      = fd.get("confidence", 0.5)
                kw_hits = sum(1 for kw in keywords if kw in fact_text)
                if kw_hits == 0:
                    continue
                score = conf * (kw_hits / max(len(keywords), 1))
                if score > best_score:
                    best_score = score
                    best_fact  = fd

            if best_fact and best_score >= 0.3:
                fact_text = best_fact.get("fact", "")
                conf = min(best_fact.get("confidence", 0.5), best_score + 0.1)
                return KBHit(
                    answer=fact_text,
                    confidence=round(conf, 3),
                    source="world_model",
                    context_facts=[fact_text],
                    subject=subject,
                    domain=domain,
                )
        except Exception as e:
            log.debug("[KBLookup] WorldModel query failed: %s", e)
        return None

    # ── KnowledgeGraph query ───────────────────────────────────────────────────

    def _query_knowledge_graph(self, domain: str, subject: str, keywords: List[str]) -> Optional[KBHit]:
        try:
            from sare.memory.knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph()

            best_score = 0.0
            best_node  = None
            for node in kg._nodes.values():
                if node.type not in ("belief", "fact"):
                    continue
                node_text = (node.name + " " + node.content.get("description", "")).lower()
                kw_hits = sum(1 for kw in keywords if kw in node_text)
                if kw_hits == 0:
                    continue
                score = node.confidence * (kw_hits / max(len(keywords), 1))
                if score > best_score:
                    best_score = score
                    best_node  = node

            if best_node and best_score >= 0.3:
                desc = best_node.content.get("description", best_node.name)
                return KBHit(
                    answer=desc,
                    confidence=round(min(best_node.confidence, best_score + 0.1), 3),
                    source="knowledge_graph",
                    context_facts=[desc],
                    subject=subject,
                    domain=domain,
                )
        except Exception as e:
            log.debug("[KBLookup] KnowledgeGraph query failed: %s", e)
        return None

    # ── CommonSense query ──────────────────────────────────────────────────────

    def _query_commonsense(self, domain: str, subject: str, keywords: List[str]) -> Optional[KBHit]:
        try:
            from sare.knowledge.commonsense import CommonSenseBase
            cs = CommonSenseBase()
            cs.load()

            # Query by subject and each keyword
            all_results = cs.query(subject, depth=1)
            for kw in keywords[:3]:
                all_results.extend(cs.query(kw, depth=1))

            if not all_results:
                return None

            # Score by relevance: closer distance + more keyword overlap in object
            best_score = 0.0
            best_item  = None
            context_facts: List[str] = []

            for item in all_results:
                obj = item.get("object", "").lower()
                rel = item.get("relation", "")
                dist = item.get("distance", 1)
                kw_hits = sum(1 for kw in keywords if kw in obj or kw in item.get("subject",""))
                score = (kw_hits + 0.5) / (dist + 1)
                fact_str = f"{item.get('subject','')} {rel} {obj}"
                if len(context_facts) < 5:
                    context_facts.append(fact_str)
                if score > best_score:
                    best_score = score
                    best_item  = item

            if best_item:
                obj = best_item.get("object", "")
                rel = best_item.get("relation", "")
                subj = best_item.get("subject", subject)
                answer = f"{subj} {rel} {obj}"
                # Commonsense confidence is lower than verified facts
                conf = min(0.7, 0.3 + best_score * 0.1)
                return KBHit(
                    answer=answer,
                    confidence=round(conf, 3),
                    source="commonsense",
                    context_facts=context_facts[:5],
                    subject=subject,
                    domain=domain,
                )
        except Exception as e:
            log.debug("[KBLookup] CommonSense query failed: %s", e)
        return None

    # ── Score fusion ───────────────────────────────────────────────────────────

    def _fuse(self, wm: Optional[KBHit], kg: Optional[KBHit], cs: Optional[KBHit],
              domain: str, subject: str) -> Optional[KBHit]:
        candidates = [h for h in (wm, kg, cs) if h is not None]
        if not candidates:
            return None

        # Pick highest-confidence source; collect all context facts
        best = max(candidates, key=lambda h: h.confidence)
        all_context = []
        for h in candidates:
            all_context.extend(h.context_facts)
        # Deduplicate context
        seen = set()
        deduped = []
        for f in all_context:
            if f not in seen:
                seen.add(f)
                deduped.append(f)

        if len(candidates) > 1:
            # Fuse: boost confidence slightly when multiple sources agree
            best = KBHit(
                answer=best.answer,
                confidence=min(0.98, best.confidence + 0.05 * (len(candidates) - 1)),
                source="fused" if len(candidates) > 1 else best.source,
                context_facts=deduped[:8],
                subject=subject,
                domain=domain,
            )
        else:
            best.context_facts = deduped[:8]

        return best
