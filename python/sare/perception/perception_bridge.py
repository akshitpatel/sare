"""
PerceptionBridge — S29-4
Converts structured scene descriptions and symbolic input into
ConceptGraph objects and relations for the reasoning layer.

Perception → Symbolic pipeline:
  input description  →  parse_scene()  →  {objects, relations}
                                        →  ConceptGraph nodes + edges

Supports:
  - Spatial relations:  above, below, left_of, right_of, near, far, touches, inside
  - Comparative:        larger_than, smaller_than, heavier_than, faster_than
  - Action relations:   pushes, blocks, supports, contains
  - Property tagging:   color, shape, size, material

Example:
  "ball above table"         → object(ball) + object(table) + relation(above, ball, table)
  "red block left of door"   → object(red_block) + object(door) + relation(left_of, block, door)
  "friction slows block"     → object(friction) + object(block) + relation(slows, friction, block)
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

# ── relation patterns ─────────────────────────────────────────────────────────

_SPATIAL = {
    "above":    r"\b(\w+)\s+above\s+(\w+)\b",
    "below":    r"\b(\w+)\s+below\s+(\w+)\b",
    "left_of":  r"\b(\w+)\s+(?:left of|to the left of)\s+(\w+)\b",
    "right_of": r"\b(\w+)\s+(?:right of|to the right of)\s+(\w+)\b",
    "near":     r"\b(\w+)\s+(?:near|next to|beside)\s+(\w+)\b",
    "touches":  r"\b(\w+)\s+(?:touches|touching|against)\s+(\w+)\b",
    "inside":   r"\b(\w+)\s+(?:inside|within|in)\s+(\w+)\b",
    "on":       r"\b(\w+)\s+on\s+(?:top of\s+)?(\w+)\b",
}

_COMPARATIVE = {
    "larger_than":  r"\b(\w+)\s+(?:is\s+)?(?:larger|bigger)\s+than\s+(\w+)\b",
    "smaller_than": r"\b(\w+)\s+(?:is\s+)?(?:smaller|tinier)\s+than\s+(\w+)\b",
    "heavier_than": r"\b(\w+)\s+(?:is\s+)?heavier\s+than\s+(\w+)\b",
    "faster_than":  r"\b(\w+)\s+(?:is\s+)?faster\s+than\s+(\w+)\b",
}

_ACTION_REL = {
    "pushes":    r"\b(\w+)\s+pushes\s+(\w+)\b",
    "blocks":    r"\b(\w+)\s+blocks\s+(\w+)\b",
    "supports":  r"\b(\w+)\s+supports\s+(\w+)\b",
    "contains":  r"\b(\w+)\s+contains\s+(\w+)\b",
    "slows":     r"\b(\w+)\s+slows\s+(\w+)\b",
    "moves":     r"\b(\w+)\s+moves\s+(\w+)\b",
    "hits":      r"\b(\w+)\s+hits\s+(\w+)\b",
}

_ALL_PATTERNS = {**_SPATIAL, **_COMPARATIVE, **_ACTION_REL}

# ── property extraction ───────────────────────────────────────────────────────

_COLORS    = {"red", "blue", "green", "yellow", "black", "white", "orange", "purple"}
_SHAPES    = {"ball", "block", "cube", "sphere", "cylinder", "box", "ring", "disc"}
_MATERIALS = {"wood", "metal", "rubber", "glass", "plastic", "stone", "foam"}
_SIZES     = {"small", "large", "big", "tiny", "huge", "medium"}

_STOPWORDS = {"a", "an", "the", "is", "are", "was", "of", "to", "in", "on", "and",
              "or", "it", "its", "this", "that", "there", "then", "when", "with",
              "into", "from", "at", "by", "for", "as", "so", "if", "not", "no",
              "left", "right", "above", "below", "near", "touches", "inside", "supports",
              "contains", "slows", "moves", "hits", "pushes"}

_DESCRIPTORS = _COLORS | _MATERIALS | _SIZES


def _extract_properties(token: str) -> dict:
    props: dict = {}
    t = token.lower()
    for c in _COLORS:
        if c in t:
            props["color"] = c
    for s in _SHAPES:
        if s in t:
            props["shape"] = s
    for m in _MATERIALS:
        if m in t:
            props["material"] = m
    for sz in _SIZES:
        if sz in t:
            props["size"] = sz
    return props


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class PerceptObject:
    name:       str
    properties: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "properties": self.properties}


@dataclass
class PerceptRelation:
    relation: str
    subject:  str
    object_:  str
    confidence: float = 1.0

    def to_symbolic(self) -> str:
        return f"{self.relation}({self.subject},{self.object_})"

    def to_dict(self) -> dict:
        return {
            "relation":   self.relation,
            "subject":    self.subject,
            "object":     self.object_,
            "symbolic":   self.to_symbolic(),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class ParsedScene:
    description:  str
    objects:      List[PerceptObject]
    relations:    List[PerceptRelation]
    ts:           float = field(default_factory=time.time)

    @property
    def n_objects(self) -> int:
        return len(self.objects)

    @property
    def n_relations(self) -> int:
        return len(self.relations)

    def to_dict(self) -> dict:
        return {
            "description": self.description[:80],
            "objects":     [o.to_dict() for o in self.objects],
            "relations":   [r.to_dict() for r in self.relations],
            "n_objects":   self.n_objects,
            "n_relations": self.n_relations,
            "symbolic":    [r.to_symbolic() for r in self.relations],
        }


@dataclass
class ParsedTransition:
    before: ParsedScene
    action: str
    after: ParsedScene
    added_relations: List[str]
    removed_relations: List[str]
    added_objects: List[str]
    removed_objects: List[str]
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "action": self.action[:80],
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "added_relations": self.added_relations,
            "removed_relations": self.removed_relations,
            "added_objects": self.added_objects,
            "removed_objects": self.removed_objects,
        }


# ── PerceptionBridge ──────────────────────────────────────────────────────────

class PerceptionBridge:
    """
    Parses scene descriptions into symbolic (object, relation) structures.
    Posts discovered concepts and relations to ConceptGraph.
    """

    def __init__(self) -> None:
        self._concept_graph    = None
        self._global_workspace = None

        self._total_parsed      = 0
        self._total_objects     = 0
        self._total_relations   = 0
        self._seen_relations:   Set[str]      = set()
        self._scene_history:    List[ParsedScene] = []
        self._transition_history: List[ParsedTransition] = []
        self._history_limit     = 50

    # ── wiring ────────────────────────────────────────────────────────────────

    def wire(self, concept_graph=None, global_workspace=None) -> None:
        self._concept_graph    = concept_graph
        self._global_workspace = global_workspace

    # ── parse ─────────────────────────────────────────────────────────────────

    def parse_scene(self, description: str) -> ParsedScene:
        """Parse a natural-language scene description into objects + relations."""
        text = self._normalize_description(description)

        # Extract relations first
        relations: List[PerceptRelation] = []
        seen_pairs: Set[Tuple[str, str]] = set()
        for rel_name, pattern in _ALL_PATTERNS.items():
            for m in re.finditer(pattern, text, re.IGNORECASE):
                subj = self._clean(m.group(1))
                obj  = self._clean(m.group(2))
                if subj and obj and subj != obj:
                    pair = (rel_name, subj, obj)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        relations.append(PerceptRelation(rel_name, subj, obj))

        # Extract objects from relation subjects/objects + standalone tokens
        obj_names: Set[str] = set()
        for r in relations:
            obj_names.add(r.subject)
            obj_names.add(r.object_)

        # Scan remaining tokens
        tokens = re.findall(r'\b[a-z][a-z_]*\b', text)
        for tok in tokens:
            if tok not in _STOPWORDS and len(tok) > 2:
                obj_names.add(tok)

        objects = [PerceptObject(n, _extract_properties(n)) for n in sorted(obj_names)]

        scene = ParsedScene(description, objects, relations)
        self._scene_history.append(scene)
        if len(self._scene_history) > self._history_limit:
            self._scene_history.pop(0)

        self._total_parsed    += 1
        self._total_objects   += len(objects)
        self._total_relations += len(relations)

        # Track unique relations regardless of graph availability
        for r in relations:
            self._seen_relations.add(r.to_symbolic())

        self._post_to_graph(scene)
        return scene

    def parse_transition(self, before: str, action: str, after: str) -> ParsedTransition:
        """Parse a before/action/after tuple and extract the grounded state change."""
        before_scene = self.parse_scene(before)
        after_scene = self.parse_scene(after)

        before_rel = {rel.to_symbolic() for rel in before_scene.relations}
        after_rel = {rel.to_symbolic() for rel in after_scene.relations}
        before_objects = {obj.name for obj in before_scene.objects}
        after_objects = {obj.name for obj in after_scene.objects}

        transition = ParsedTransition(
            before=before_scene,
            action=action.strip(),
            after=after_scene,
            added_relations=sorted(after_rel - before_rel),
            removed_relations=sorted(before_rel - after_rel),
            added_objects=sorted(after_objects - before_objects),
            removed_objects=sorted(before_objects - after_objects),
        )
        self._transition_history.append(transition)
        if len(self._transition_history) > self._history_limit:
            self._transition_history.pop(0)
        self._post_transition(transition)
        return transition

    def _clean(self, token: str) -> str:
        t = token.strip().lower()
        # Strip trailing 's' only for known shape/material nouns
        if t.endswith('s') and len(t) > 3:
            t = t
        return t if t not in _STOPWORDS and len(t) > 1 else ""

    @staticmethod
    def _normalize_description(description: str) -> str:
        text = description.lower().strip()
        descriptor_group = "|".join(sorted(_DESCRIPTORS))
        shape_group = "|".join(sorted(_SHAPES))
        pattern = re.compile(rf"\b((?:{descriptor_group})\s+)+(?:({shape_group}))\b")

        def _merge(match: re.Match) -> str:
            tokens = re.findall(r"[a-z]+", match.group(0))
            return "_".join(tokens)

        return pattern.sub(_merge, text)

    # ── ConceptGraph integration ──────────────────────────────────────────────

    def _post_to_graph(self, scene: ParsedScene) -> None:
        """Add parsed objects and relations to ConceptGraph."""
        if not self._concept_graph:
            return
        try:
            cg = self._concept_graph
            for obj in scene.objects:
                if hasattr(cg, 'add_concept'):
                    cg.add_concept(obj.name, {
                        "source": "perception",
                        "properties": obj.properties,
                    })
                elif hasattr(cg, '_concepts'):
                    try:
                        from sare.core.concept_graph import Concept
                        if obj.name not in cg._concepts:
                            c = Concept(name=obj.name)
                            if obj.properties:
                                c.symbolic_rules.append(
                                    f"percept:{obj.name}({','.join(f'{k}={v}' for k,v in obj.properties.items())})")
                            cg._concepts[obj.name] = c
                    except ImportError:
                        pass

            for rel in scene.relations:
                sym = rel.to_symbolic()
                if sym not in self._seen_relations:
                    self._seen_relations.add(sym)
                    if hasattr(cg, 'add_relation'):
                        cg.add_relation(rel.subject, rel.relation, rel.object_)
                    elif hasattr(cg, '_concepts'):
                        for nm in (rel.subject, rel.object_):
                            try:
                                from sare.core.concept_graph import Concept
                                if nm not in cg._concepts:
                                    cg._concepts[nm] = Concept(name=nm)
                                cg._concepts[nm].symbolic_rules.append(sym)
                            except ImportError:
                                pass

        except Exception as e:
            log.debug(f"PerceptionBridge post_to_graph: {e}")

        if self._global_workspace:
            try:
                self._global_workspace.broadcast(
                    "scene_parsed",
                    {"n_objects": len(scene.objects),
                     "n_relations": len(scene.relations),
                     "symbolic": [r.to_symbolic() for r in scene.relations[:3]]},
                    source="perception_bridge",
                    salience=0.6,
                )
            except Exception as e:
                log.debug(f"PerceptionBridge gw: {e}")

    def _post_transition(self, transition: ParsedTransition) -> None:
        if self._concept_graph:
            try:
                cg = self._concept_graph
                action_label = transition.action or "state_change"
                if hasattr(cg, "add_concept"):
                    cg.add_concept(action_label, {
                        "source": "perception_transition",
                        "added_relations": transition.added_relations,
                        "removed_relations": transition.removed_relations,
                    })
            except Exception as e:
                log.debug(f"PerceptionBridge transition graph: {e}")

        if self._global_workspace:
            try:
                self._global_workspace.broadcast(
                    "transition_parsed",
                    {
                        "action": transition.action,
                        "added_relations": transition.added_relations[:3],
                        "removed_relations": transition.removed_relations[:3],
                    },
                    source="perception_bridge",
                    salience=0.7,
                )
            except Exception as e:
                log.debug(f"PerceptionBridge transition gw: {e}")

    # ── convenience ───────────────────────────────────────────────────────────

    def image_to_symbolic(self, image_description: str) -> List[str]:
        """Parse a description and return list of symbolic relation strings."""
        scene = self.parse_scene(image_description)
        return [r.to_symbolic() for r in scene.relations]

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "total_parsed":      self._total_parsed,
            "total_objects":     self._total_objects,
            "total_relations":   self._total_relations,
            "unique_relations":  len(self._seen_relations),
            "concept_graph_wired": self._concept_graph is not None,
            "recent_scenes":     [s.to_dict() for s in self._scene_history[-5:]],
            "recent_transitions": [t.to_dict() for t in self._transition_history[-3:]],
            "known_relations":   sorted(self._seen_relations)[:15],
        }
