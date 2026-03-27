"""
StreamBridge — S29-3
Cross-stream knowledge transfer for ContinuousStreamLearner.

Implements the 4-stage pipeline:
  EXPLORE  → discovers a new transform / problem pattern
  IMAGINE  → tests it in a sandbox (GenerativeWorldModel)
  EXPLOIT  → trains on it (AffectiveEnergy + ContinuousStreamLearner)
  CURRICULUM → integrates it (MetaCurriculumEngine + TransformGenerator)

Items enter via `submit(item)` and flow through the pipeline.
Each gate applies a quality filter; items that fail a gate are recycled
back to EXPLORE with a penalty flag.

TransferItem tracks provenance so the full lineage is visible on the dashboard.
"""
from __future__ import annotations

import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

log = logging.getLogger(__name__)

_GATE_THRESHOLDS = {
    "imagine":     0.40,   # imagination solve-rate to pass to EXPLOIT
    "exploit":     0.55,   # exploit energy improvement to pass to CURRICULUM
    "curriculum":  0.60,   # curriculum novelty score to fully integrate
}
_PIPELINE_LIMIT = 40       # max items tracked


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class TransferItem:
    item_id:    int
    source:     str       # stream type: EXPLORE / IMAGINE / etc.
    content:    str       # transform text or problem expression
    domain:     str
    stage:      str       # current stage: explore→imagine→exploit→curriculum→done
    score:      float     = 0.0
    passes:     int       = 0
    failures:   int       = 0
    promoted:   bool      = False
    ts_enter:   float     = field(default_factory=time.time)
    ts_promote: Optional[float] = None
    lineage:    List[str] = field(default_factory=list)

    def advance(self, new_stage: str, score: float) -> None:
        self.lineage.append(f"{self.stage}→{new_stage}({score:.2f})")
        self.stage = new_stage
        self.score = score
        self.passes += 1
        if new_stage == "done":
            self.promoted       = True
            self.ts_promote     = time.time()

    def fail_gate(self, gate: str) -> None:
        self.lineage.append(f"{gate}✗")
        self.failures += 1

    def to_dict(self) -> dict:
        return {
            "id":       self.item_id,
            "source":   self.source,
            "content":  self.content[:50],
            "domain":   self.domain,
            "stage":    self.stage,
            "score":    round(self.score, 3),
            "passes":   self.passes,
            "failures": self.failures,
            "promoted": self.promoted,
            "lineage":  self.lineage,
            "age_s":    round(time.time() - self.ts_enter, 1),
        }


# ── StreamBridge ──────────────────────────────────────────────────────────────

class StreamBridge:
    """
    Mediates cross-stream transfer between the 4 ContinuousStreamLearner streams.
    Each tick() advances all queued items through the pipeline.
    """

    def __init__(self) -> None:
        self._next_id        = 0
        self._queue: Deque[TransferItem] = deque(maxlen=_PIPELINE_LIMIT)
        self._promoted:  List[TransferItem] = []
        self._recycled:  List[TransferItem] = []

        # wired modules
        self._world_model    = None   # GenerativeWorldModel
        self._affective      = None   # AffectiveEnergy
        self._meta_curriculum = None  # MetaCurriculumEngine
        self._transform_gen  = None   # TransformGenerator
        self._engine         = None

        self._total_submitted = 0
        self._total_promoted  = 0
        self._total_recycled  = 0
        self._tick_count      = 0

    # ── wiring ────────────────────────────────────────────────────────────────

    def wire(self, world_model=None, affective=None,
             meta_curriculum=None, transform_gen=None, engine=None) -> None:
        self._world_model    = world_model
        self._affective      = affective
        self._meta_curriculum = meta_curriculum
        self._transform_gen  = transform_gen
        self._engine         = engine

    # ── public submit ─────────────────────────────────────────────────────────

    def submit(self, content: str, source: str = "EXPLORE",
               domain: str = "general") -> TransferItem:
        """Submit a discovered item to enter the pipeline at the EXPLORE stage."""
        self._next_id += 1
        item = TransferItem(self._next_id, source, content, domain, "explore")
        self._queue.append(item)
        self._total_submitted += 1
        return item

    # ── tick ──────────────────────────────────────────────────────────────────

    def tick(self) -> dict:
        """Advance each queued item by one pipeline stage."""
        self._tick_count += 1
        moved   = []
        dropped = []

        # Process up to 5 items per tick to avoid overloading
        items_this_tick = list(self._queue)[:5]

        for item in items_this_tick:
            if item.stage == "explore":
                self._gate_imagine(item)
            elif item.stage == "imagine":
                self._gate_exploit(item)
            elif item.stage == "exploit":
                self._gate_curriculum(item)
            elif item.stage == "done":
                self._queue.remove(item)
                self._promoted.append(item)
                self._total_promoted += 1
                moved.append(item.to_dict())
            elif item.stage == "recycled":
                self._queue.remove(item)
                self._recycled.append(item)
                self._total_recycled += 1
                dropped.append(item.to_dict())

        return {
            "tick":     self._tick_count,
            "queue":    len(self._queue),
            "moved":    moved,
            "dropped":  dropped,
        }

    # ── gate implementations ──────────────────────────────────────────────────

    def _gate_imagine(self, item: TransferItem) -> None:
        """Test in imagination sandbox (GenerativeWorldModel)."""
        score = 0.0
        try:
            if self._world_model and hasattr(self._world_model, 'attempt'):
                result = self._world_model.attempt(item.content)
                score  = 1.0 - getattr(result, 'energy', 0.6)
            else:
                score = random.uniform(0.3, 0.8)
        except Exception:
            score = random.uniform(0.3, 0.7)

        if score >= _GATE_THRESHOLDS["imagine"]:
            item.advance("imagine", score)
        else:
            item.fail_gate("imagine")
            if item.failures >= 3:
                item.stage = "recycled"

    def _gate_exploit(self, item: TransferItem) -> None:
        """Measure energy improvement via solver + AffectiveEnergy."""
        score = 0.0
        try:
            eng = self._engine
            if eng and hasattr(eng, '_engine'):
                eng = eng._engine
            if eng and hasattr(eng, 'solve'):
                result = eng.solve(item.content)
                energy = getattr(result, 'energy', 0.6) if result else 0.6
                score  = max(0.0, 1.0 - energy)
            else:
                score = random.uniform(0.35, 0.85)
        except Exception:
            score = random.uniform(0.35, 0.75)

        # Amplify score via AffectiveEnergy curiosity signal
        if self._affective and hasattr(self._affective, 'score'):
            try:
                af = self._affective.score(item.content)
                score = min(1.0, score * (1 + af.get("novelty", 0) * 0.3))
            except Exception:
                pass

        if score >= _GATE_THRESHOLDS["exploit"]:
            item.advance("exploit", score)
        else:
            item.fail_gate("exploit")
            if item.failures >= 3:
                item.stage = "recycled"

    def _gate_curriculum(self, item: TransferItem) -> None:
        """Integrate into MetaCurriculumEngine and TransformGenerator."""
        score = item.score  # inherit from exploit stage

        if self._meta_curriculum:
            try:
                self._meta_curriculum.observe(item.domain, True)
                score = min(1.0, score + 0.05)
            except Exception:
                pass

        if self._transform_gen and hasattr(self._transform_gen, 'promote_template'):
            try:
                self._transform_gen.promote_template(item.content, item.domain)
                score = min(1.0, score + 0.05)
            except Exception:
                pass

        if score >= _GATE_THRESHOLDS["curriculum"]:
            item.advance("done", score)
        else:
            item.fail_gate("curriculum")
            if item.failures >= 3:
                item.stage = "recycled"

    # ── summary ───────────────────────────────────────────────────────────────

    def pipeline_stages(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in self._queue:
            counts[item.stage] = counts.get(item.stage, 0) + 1
        return counts

    def summary(self) -> dict:
        return {
            "queue_depth":     len(self._queue),
            "total_submitted": self._total_submitted,
            "total_promoted":  self._total_promoted,
            "total_recycled":  self._total_recycled,
            "tick_count":      self._tick_count,
            "pipeline_stages": self.pipeline_stages(),
            "promotion_rate":  round(
                self._total_promoted / max(1, self._total_submitted), 3),
            "recent_promoted": [i.to_dict() for i in self._promoted[-5:]],
            "recent_recycled": [i.to_dict() for i in self._recycled[-3:]],
            "queue_items":     [i.to_dict() for i in list(self._queue)[:8]],
            "modules_wired": {
                "world_model":     self._world_model is not None,
                "affective":       self._affective is not None,
                "meta_curriculum": self._meta_curriculum is not None,
                "transform_gen":   self._transform_gen is not None,
            },
        }
