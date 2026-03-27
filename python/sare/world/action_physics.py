"""
ActionPhysicsSession — S29-2
Multi-step physics simulation with agent actions, state transitions,
and automatic concept extraction.

Object state: {id, x, y, vx, vy, mass, friction, radius}
Action types: push, drop, throw, collide

Each episode runs n_steps of:
  apply_action(state, action) → next_state
  detect_concept(transition) → concept name if pattern matches

Extracted concepts (e.g. "friction", "momentum", "gravity", "elastic_collision")
are posted to ConceptGraph as new nodes with evidence from the simulation.
"""
from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_GRAVITY     = 9.8
_DT          = 0.1          # simulation time step seconds
_FLOOR_Y     = 0.0
_FRICTION_MU = 0.3          # default kinetic friction coefficient


# ── physics primitives ────────────────────────────────────────────────────────

@dataclass
class PhysObject:
    obj_id:   str
    x:        float = 0.0
    y:        float = 1.0
    vx:       float = 0.0
    vy:       float = 0.0
    mass:     float = 1.0
    friction: float = _FRICTION_MU
    radius:   float = 0.5
    on_floor: bool  = False

    def copy(self) -> "PhysObject":
        return PhysObject(self.obj_id, self.x, self.y, self.vx, self.vy,
                          self.mass, self.friction, self.radius, self.on_floor)

    def to_dict(self) -> dict:
        return {
            "id": self.obj_id,
            "x": round(self.x, 3), "y": round(self.y, 3),
            "vx": round(self.vx, 3), "vy": round(self.vy, 3),
            "mass": self.mass, "on_floor": self.on_floor,
        }


def _step(obj: PhysObject) -> PhysObject:
    """Integrate one DT step: gravity + friction + floor collision."""
    o = obj.copy()
    if not o.on_floor:
        o.vy -= _GRAVITY * _DT
    o.x += o.vx * _DT
    o.y += o.vy * _DT
    if o.y <= _FLOOR_Y:
        o.y       = _FLOOR_Y
        o.vy      = max(0.0, -o.vy * 0.4)   # inelastic bounce (40% restitution)
        o.on_floor = True
    else:
        o.on_floor = False
    if o.on_floor and abs(o.vx) > 0:
        friction_decel = o.friction * _GRAVITY * _DT
        if abs(o.vx) < friction_decel:
            o.vx = 0.0
        else:
            o.vx -= math.copysign(friction_decel, o.vx)
    return o


# ── action definitions ────────────────────────────────────────────────────────

def _push(obj: PhysObject, force: float, angle_deg: float = 0.0) -> PhysObject:
    o = obj.copy()
    rad    = math.radians(angle_deg)
    ax     = (force / o.mass) * math.cos(rad)
    ay     = (force / o.mass) * math.sin(rad)
    o.vx  += ax * _DT
    o.vy  += ay * _DT
    return o


def _drop(obj: PhysObject, height: float) -> PhysObject:
    o = obj.copy()
    o.y  = height
    o.vy = 0.0
    o.vx = 0.0
    o.on_floor = False
    return o


def _throw(obj: PhysObject, speed: float, angle_deg: float) -> PhysObject:
    o = obj.copy()
    rad   = math.radians(angle_deg)
    o.vx  = speed * math.cos(rad)
    o.vy  = speed * math.sin(rad)
    o.on_floor = False
    return o


def _collide(a: PhysObject, b: PhysObject) -> Tuple[PhysObject, PhysObject]:
    """1D elastic collision on x-axis."""
    a2, b2 = a.copy(), b.copy()
    m1, m2 = a.mass, b.mass
    v1, v2 = a.vx, b.vx
    a2.vx = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
    b2.vx = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)
    return a2, b2


_ACTIONS = {
    "push":    lambda o, **kw: _push(o, kw.get("force", 5.0), kw.get("angle", 0.0)),
    "drop":    lambda o, **kw: _drop(o, kw.get("height", 5.0)),
    "throw":   lambda o, **kw: _throw(o, kw.get("speed", 8.0), kw.get("angle", 45.0)),
}


# ── concept detection ─────────────────────────────────────────────────────────

def _detect_concept(before: PhysObject, after: PhysObject,
                    action: str) -> Optional[str]:
    """Return concept name if the transition demonstrates a known pattern."""
    dy = abs(after.y - before.y)
    dv = abs(after.vx - before.vx)

    if action == "drop" and dy > 0.5 and not after.on_floor:
        return "gravity"
    if action == "drop" and after.on_floor and abs(after.vy) < 0.1:
        return "inelastic_collision"
    if action == "push" and after.on_floor and abs(after.vx) < abs(before.vx):
        return "friction"
    if action == "push" and dv > 0 and after.mass > 1.5:
        return "inertia"
    if action == "throw" and after.y > before.y:
        return "projectile_motion"
    if action == "collide" and dv > 0.5:
        return "momentum_transfer"
    return None


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class PhysStep:
    step:    int
    action:  str
    before:  dict
    after:   dict
    concept: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "step":    self.step,
            "action":  self.action,
            "concept": self.concept,
            "dy":      round(self.after.get("y", 0) - self.before.get("y", 0), 3),
            "dvx":     round(self.after.get("vx", 0) - self.before.get("vx", 0), 3),
        }


@dataclass
class PhysEpisode:
    episode_id:    int
    steps:         List[PhysStep] = field(default_factory=list)
    concepts_found: List[str]     = field(default_factory=list)
    ts:            float          = field(default_factory=time.time)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict:
        return {
            "episode_id":     self.episode_id,
            "n_steps":        self.n_steps,
            "concepts_found": self.concepts_found,
            "steps":          [s.to_dict() for s in self.steps[-5:]],
        }


# ── ActionPhysicsSession ──────────────────────────────────────────────────────

class ActionPhysicsSession:
    """
    Runs multi-step physics episodes with agent actions.
    Extracts concepts from state transitions and posts to ConceptGraph.
    """

    _ACTION_SEQ = [
        ("drop",  {"height": 5.0}),
        ("push",  {"force": 6.0, "angle": 0.0}),
        ("throw", {"speed": 7.0, "angle": 45.0}),
        ("push",  {"force": 3.0, "angle": 15.0}),
        ("drop",  {"height": 3.0}),
    ]

    def __init__(self) -> None:
        self._concept_graph  = None
        self._global_workspace = None
        self._episode_count  = 0
        self._concepts_found: Dict[str, int] = {}
        self._episodes: List[PhysEpisode]    = []
        self._episode_limit  = 30

    def wire(self, concept_graph=None, global_workspace=None) -> None:
        self._concept_graph    = concept_graph
        self._global_workspace = global_workspace

    # ── episode runner ────────────────────────────────────────────────────────

    def run_episode(self, n_steps: int = 15,
                    obj: Optional[PhysObject] = None) -> PhysEpisode:
        """Run one multi-step physics episode. Returns PhysEpisode."""
        if obj is None:
            obj = PhysObject(
                obj_id=f"obj_{self._episode_count}",
                x=0.0, y=random.uniform(1.0, 6.0),
                mass=random.uniform(0.5, 3.0),
                friction=random.uniform(0.1, 0.6),
            )

        self._episode_count += 1
        ep = PhysEpisode(episode_id=self._episode_count)

        for step_i in range(n_steps):
            action_name, kwargs = self._ACTION_SEQ[step_i % len(self._ACTION_SEQ)]
            before = obj.copy()

            # Apply action then simulate physics DT steps
            obj = _ACTIONS[action_name](obj, **kwargs)
            for _ in range(3):   # 3 physics sub-steps per action
                obj = _step(obj)

            concept = _detect_concept(before, obj, action_name)
            ps = PhysStep(step_i, action_name, before.to_dict(), obj.to_dict(), concept)
            ep.steps.append(ps)

            if concept and concept not in ep.concepts_found:
                ep.concepts_found.append(concept)
                self._concepts_found[concept] = self._concepts_found.get(concept, 0) + 1
                self._post_concept(concept, ep.episode_id)

        self._episodes.append(ep)
        if len(self._episodes) > self._episode_limit:
            self._episodes.pop(0)
        return ep

    def _post_concept(self, concept: str, episode_id: int) -> None:
        """Post discovered concept to ConceptGraph and GlobalWorkspace."""
        if self._concept_graph:
            try:
                if hasattr(self._concept_graph, 'add_concept'):
                    self._concept_graph.add_concept(concept, {
                        "source": "action_physics",
                        "episode": episode_id,
                        "symbolic": f"physics:{concept}",
                    })
                elif hasattr(self._concept_graph, '_concepts'):
                    from sare.core.concept_graph import Concept
                    if concept not in self._concept_graph._concepts:
                        c = Concept(name=concept)
                        c.symbolic_rules.append(f"physics:{concept}")
                        self._concept_graph._concepts[concept] = c
            except Exception as e:
                log.debug(f"ActionPhysics concept_graph: {e}")

        if self._global_workspace:
            try:
                self._global_workspace.broadcast(
                    "physics_concept", {"concept": concept, "episode": episode_id},
                    source="action_physics", salience=0.7,
                )
            except Exception as e:
                log.debug(f"ActionPhysics gw broadcast: {e}")

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "episodes_run":   self._episode_count,
            "concepts_found": self._concepts_found,
            "n_unique_concepts": len(self._concepts_found),
            "recent_episodes": [e.to_dict() for e in self._episodes[-5:]],
            "concept_graph_wired": self._concept_graph is not None,
        }
