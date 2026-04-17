"""
HypothesisMaker — proposes candidate beliefs by analogy over existing KB.

When the system knows "dog is_a mammal" and "cat is_a mammal", it can propose
"wolf is_a mammal" if wolf appears in any nearby fact. Proposals are stored at
confidence=0.30 (hypothesis tier) and will be confirmed or decayed over time
by the belief expiry and iterative inference cycles.

This is genuine curiosity: generating novel hypotheses rather than reacting
to known gaps.
"""
from __future__ import annotations

import logging
import random
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

_HYPOTHESIS_CONF = 0.30


class HypothesisMaker:
    """Proposes plausible new beliefs by structural analogy over WorldModel facts."""

    def propose(self, max_proposals: int = 3) -> List[Tuple[str, str, str]]:
        """Generate up to max_proposals novel (subject, predicate, value) hypotheses.

        Strategy:
          1. Collect high-confidence (≥0.70) is_a/isa beliefs → category clusters.
          2. For each category with ≥2 members, check if any member has a
             'related_to' or 'similar_to' link to an entity NOT already in the cluster.
          3. Propose that linked entity is_a the same category.
          4. Store at confidence=0.30 in WorldModel.
        """
        proposed: List[Tuple[str, str, str]] = []
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()

            # Gather is_a beliefs with high confidence
            isa_predicates = {"is_a", "isa", "is", "type_of"}
            candidates = [
                b for b in wm._beliefs.values()
                if b.predicate.lower() in isa_predicates
                and b.confidence >= 0.70
                and b.value
            ]

            if len(candidates) < 2:
                return proposed

            # Build category → [members]
            category_members: dict = {}
            for b in candidates:
                cat = b.value.strip().lower()
                category_members.setdefault(cat, []).append(b.subject.strip().lower())

            # Shuffle for variety across cycles
            shuffled_cats = list(category_members.items())
            random.shuffle(shuffled_cats)

            for cat, members in shuffled_cats:
                if len(members) < 2:
                    continue

                for member in members[:3]:
                    # Look for entities related to this member
                    for rel_pred in ("related_to", "similar_to", "associated_with"):
                        rel_b = wm.get_belief(member, rel_pred)
                        if not rel_b or not rel_b.value:
                            continue
                        candidate = rel_b.value.strip().lower()
                        if not candidate or candidate in members:
                            continue
                        # Only propose if no existing is_a belief for candidate
                        existing = wm.get_belief(candidate, "is_a")
                        if existing is not None:
                            continue
                        # Propose
                        wm.update_belief(
                            subject=candidate,
                            predicate="is_a",
                            value=cat,
                            confidence=_HYPOTHESIS_CONF,
                            domain="general",
                        )
                        proposed.append((candidate, "is_a", cat))
                        log.debug("[HypothesisMaker] Proposed: %s is_a %s (via %s)",
                                  candidate, cat, member)
                        if len(proposed) >= max_proposals:
                            return proposed
        except Exception as e:
            log.debug("[HypothesisMaker] propose error: %s", e)
        return proposed


_SINGLETON: Optional[HypothesisMaker] = None


def get_hypothesis_maker() -> HypothesisMaker:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = HypothesisMaker()
    return _SINGLETON
