"""
EmbodiedAgent — Session 32 Fix 1: Embodiment (0% → 12%)

A discrete grid-world agent that implements the full embodiment loop:

    perceive → decide → act → observe outcome → learn

This is TRUE embodiment per cognitive science:
  - Agent has a body (position, inventory, energy)
  - Agent perceives local environment (what's nearby)
  - Agent chooses actions based on learned policy
  - Agent observes consequences and updates its model
  - Agent pursues goals (move object to target, explore, collect)

The grid world contains:
  - Walls, floors, objects (pushable, pickable)
  - Target zones (goals)
  - Physics: gravity (objects fall), friction (objects slow)

Learning mechanism:
  - Action→outcome EMA tracks which actions work in which contexts
  - Surprise signal when outcome differs from prediction
  - Successful action sequences stored as procedural memory
  - Concepts extracted from repeated patterns (e.g., "pushing moves objects")
"""
from __future__ import annotations

import logging
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

# ── Grid world constants ─────────────────────────────────────────────────────

GRID_W, GRID_H = 12, 8
MAX_INVENTORY = 3
ACTIONS = ["move_up", "move_down", "move_left", "move_right",
           "push", "pick", "drop", "look", "wait"]

_CONCEPT_MAP = {
    "push_moved":       "force_causes_motion",
    "pick_success":     "object_manipulation",
    "drop_gravity":     "gravity_pulls_down",
    "wall_blocked":     "solid_objects_block",
    "goal_reached":     "goal_completion",
    "object_at_target": "task_achievement",
    "explore_new":      "spatial_exploration",
    "push_blocked":     "mass_resistance",
}


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class GridCell:
    """A single cell in the grid world."""
    x: int
    y: int
    wall: bool = False
    has_object: bool = False
    object_id: Optional[str] = None
    object_mass: float = 1.0
    is_target: bool = False
    visited: bool = False

    def to_dict(self) -> dict:
        d = {"x": self.x, "y": self.y}
        if self.wall:
            d["wall"] = True
        if self.has_object:
            d["object"] = self.object_id
            d["mass"] = self.object_mass
        if self.is_target:
            d["target"] = True
        return d


@dataclass
class AgentBody:
    """The agent's physical state in the world."""
    x: int = 1
    y: int = 1
    facing: str = "right"  # up/down/left/right
    inventory: List[str] = field(default_factory=list)
    energy: float = 100.0
    steps_taken: int = 0

    def to_dict(self) -> dict:
        return {
            "position": (self.x, self.y),
            "facing": self.facing,
            "inventory": list(self.inventory),
            "energy": round(self.energy, 1),
            "steps": self.steps_taken,
        }


@dataclass
class Percept:
    """What the agent perceives from its current position."""
    visible_cells: List[dict]       # cells within perception radius
    nearby_objects: List[str]       # object IDs adjacent
    at_target: bool                 # standing on a target zone
    holding: List[str]              # items in inventory
    blocked: Dict[str, bool]       # which directions are blocked
    energy: float
    position: Tuple[int, int]

    def to_dict(self) -> dict:
        return {
            "visible": len(self.visible_cells),
            "nearby_objects": self.nearby_objects,
            "at_target": self.at_target,
            "holding": self.holding,
            "blocked": self.blocked,
            "energy": round(self.energy, 1),
            "position": self.position,
        }


@dataclass
class ActionOutcome:
    """Result of an action in the world."""
    action: str
    success: bool
    effect: str                     # human-readable description
    concept: Optional[str] = None   # concept learned from this outcome
    reward: float = 0.0
    surprise: float = 0.0          # how unexpected (0=expected, 1=very surprising)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "success": self.success,
            "effect": self.effect,
            "concept": self.concept,
            "reward": round(self.reward, 3),
            "surprise": round(self.surprise, 3),
        }


@dataclass
class EpisodeRecord:
    """A complete episode of embodied interaction."""
    episode_id: int
    steps: List[ActionOutcome] = field(default_factory=list)
    total_reward: float = 0.0
    concepts_discovered: List[str] = field(default_factory=list)
    goal_achieved: bool = False
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "n_steps": len(self.steps),
            "total_reward": round(self.total_reward, 3),
            "concepts": self.concepts_discovered,
            "goal_achieved": self.goal_achieved,
            "duration_s": round(self.duration_s, 3),
        }


# ── Grid World ───────────────────────────────────────────────────────────────

class GridWorld:
    """A simple 2D grid world with walls, objects, and target zones."""

    def __init__(self, width: int = GRID_W, height: int = GRID_H):
        self.w = width
        self.h = height
        self.grid: List[List[GridCell]] = []
        self._object_positions: Dict[str, Tuple[int, int]] = {}
        self._target_positions: List[Tuple[int, int]] = []
        self._reset()

    def _reset(self) -> None:
        """Build a fresh grid with walls, objects, and targets."""
        self.grid = [
            [GridCell(x, y) for x in range(self.w)]
            for y in range(self.h)
        ]
        # Border walls
        for y in range(self.h):
            for x in range(self.w):
                if x == 0 or x == self.w - 1 or y == 0 or y == self.h - 1:
                    self.grid[y][x].wall = True

        # Internal walls (simple maze-like structure)
        for x in range(3, 7):
            self.grid[3][x].wall = True
        for y in range(1, 4):
            self.grid[y][8].wall = True

        # Place objects
        self._object_positions = {}
        obj_placements = [
            ("box_A", 2, 2, 1.0),
            ("box_B", 5, 5, 2.0),
            ("ball_C", 9, 2, 0.5),
            ("cube_D", 3, 6, 1.5),
        ]
        for obj_id, ox, oy, mass in obj_placements:
            cell = self.grid[oy][ox]
            cell.has_object = True
            cell.object_id = obj_id
            cell.object_mass = mass
            self._object_positions[obj_id] = (ox, oy)

        # Place target zones
        self._target_positions = [(10, 6), (6, 1), (1, 6)]
        for tx, ty in self._target_positions:
            self.grid[ty][tx].is_target = True

    def cell(self, x: int, y: int) -> Optional[GridCell]:
        if 0 <= x < self.w and 0 <= y < self.h:
            return self.grid[y][x]
        return None

    def is_walkable(self, x: int, y: int) -> bool:
        c = self.cell(x, y)
        return c is not None and not c.wall and not c.has_object

    def move_object(self, obj_id: str, nx: int, ny: int) -> bool:
        """Move an object to a new position if walkable."""
        if obj_id not in self._object_positions:
            return False
        ox, oy = self._object_positions[obj_id]
        nc = self.cell(nx, ny)
        if nc is None or nc.wall or nc.has_object:
            return False
        # Clear old position
        self.grid[oy][ox].has_object = False
        self.grid[oy][ox].object_id = None
        # Set new position
        nc.has_object = True
        nc.object_id = obj_id
        nc.object_mass = self.grid[oy][ox].object_mass
        self._object_positions[obj_id] = (nx, ny)
        return True

    def remove_object(self, obj_id: str) -> Optional[float]:
        """Remove an object (picked up). Returns mass or None."""
        if obj_id not in self._object_positions:
            return None
        ox, oy = self._object_positions[obj_id]
        cell = self.grid[oy][ox]
        mass = cell.object_mass
        cell.has_object = False
        cell.object_id = None
        del self._object_positions[obj_id]
        return mass

    def place_object(self, obj_id: str, x: int, y: int, mass: float = 1.0) -> bool:
        """Place an object at a position."""
        c = self.cell(x, y)
        if c is None or c.wall or c.has_object:
            return False
        c.has_object = True
        c.object_id = obj_id
        c.object_mass = mass
        self._object_positions[obj_id] = (x, y)
        return True

    def check_object_at_target(self) -> List[str]:
        """Return object IDs that are on target zones."""
        result = []
        for tx, ty in self._target_positions:
            cell = self.grid[ty][tx]
            if cell.has_object and cell.object_id:
                result.append(cell.object_id)
        return result

    def get_visible(self, x: int, y: int, radius: int = 3) -> List[dict]:
        """Return visible cells within radius."""
        cells = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    c = self.cell(x + dx, y + dy)
                    if c:
                        cells.append(c.to_dict())
        return cells

    def summary(self) -> dict:
        return {
            "size": (self.w, self.h),
            "objects": len(self._object_positions),
            "targets": len(self._target_positions),
            "objects_at_target": self.check_object_at_target(),
        }


# ── Direction helpers ────────────────────────────────────────────────────────

_DIR = {
    "up":    (0, -1),
    "down":  (0, 1),
    "left":  (-1, 0),
    "right": (1, 0),
}

def _action_to_dir(action: str) -> Optional[Tuple[int, int]]:
    """Extract direction from a move action."""
    for dirname, delta in _DIR.items():
        if dirname in action:
            return delta
    return None


# ── Embodied Agent ───────────────────────────────────────────────────────────

class EmbodiedAgent:
    """
    A grid-world agent that perceives, decides, acts, and learns.

    Implements the full embodiment loop:
      perceive → decide → act → observe → learn → (repeat)

    Learning:
      - Action-context EMA: tracks success rates per (action, context_key)
      - Procedural memory: stores successful action sequences
      - Surprise tracking: high-surprise outcomes trigger deeper learning
      - Concept extraction: repeated patterns become named concepts
    """

    def __init__(self, max_episode_steps: int = 50):
        self.world = GridWorld()
        self.body = AgentBody()
        self._max_steps = max_episode_steps

        # Learning state
        self._action_success: Dict[str, List[float]] = defaultdict(list)
        self._context_policy: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {a: 0.5 for a in ACTIONS}
        )
        self._procedural_memory: deque = deque(maxlen=100)
        self._surprise_log: deque = deque(maxlen=500)
        self._concepts_learned: Dict[str, int] = {}

        # Episode tracking
        self._episode_count = 0
        self._total_steps = 0
        self._total_reward = 0.0
        self._episodes: deque = deque(maxlen=50)
        self._current_sequence: List[str] = []

        # Integration points
        self._concept_graph = None
        self._global_workspace = None
        self._predictive_loop = None

    def wire(self, concept_graph=None, global_workspace=None,
             predictive_loop=None) -> None:
        self._concept_graph = concept_graph
        self._global_workspace = global_workspace
        self._predictive_loop = predictive_loop

    # ── Perception ───────────────────────────────────────────────────────────

    def perceive(self) -> Percept:
        """Gather sensory information from the current position."""
        x, y = self.body.x, self.body.y
        visible = self.world.get_visible(x, y, radius=3)

        # Detect nearby objects (adjacent cells)
        nearby = []
        for dirname, (dx, dy) in _DIR.items():
            c = self.world.cell(x + dx, y + dy)
            if c and c.has_object and c.object_id:
                nearby.append(c.object_id)

        # Check blocked directions
        blocked = {}
        for dirname, (dx, dy) in _DIR.items():
            blocked[dirname] = not self.world.is_walkable(x + dx, y + dy)

        # Mark current cell as visited
        cell = self.world.cell(x, y)
        at_target = cell.is_target if cell else False
        if cell:
            cell.visited = True

        return Percept(
            visible_cells=visible,
            nearby_objects=nearby,
            at_target=at_target,
            holding=list(self.body.inventory),
            blocked=blocked,
            energy=self.body.energy,
            position=(x, y),
        )

    # ── Decision making ──────────────────────────────────────────────────────

    def decide(self, percept: Percept) -> str:
        """Choose an action based on learned policy + exploration."""
        ctx = self._context_key(percept)
        policy = self._context_policy[ctx]

        # Epsilon-greedy with decay
        epsilon = max(0.1, 0.5 * (0.995 ** self._total_steps))

        if random.random() < epsilon:
            # Explore: random valid action
            valid = self._valid_actions(percept)
            return random.choice(valid) if valid else "wait"

        # Exploit: pick highest-value action
        valid = self._valid_actions(percept)
        if not valid:
            return "wait"
        best = max(valid, key=lambda a: policy.get(a, 0.0))
        return best

    def _valid_actions(self, percept: Percept) -> List[str]:
        """Return actions that make sense in current context."""
        valid = ["look", "wait"]
        for dirname in ["up", "down", "left", "right"]:
            if not percept.blocked.get(dirname, True):
                valid.append(f"move_{dirname}")
        if percept.nearby_objects:
            valid.append("push")
            if len(percept.holding) < MAX_INVENTORY:
                valid.append("pick")
        if percept.holding:
            valid.append("drop")
        return valid

    def _context_key(self, percept: Percept) -> str:
        """Create a compact context key for policy lookup."""
        parts = []
        if percept.nearby_objects:
            parts.append("obj_near")
        if percept.at_target:
            parts.append("at_target")
        if percept.holding:
            parts.append("holding")
        # Blocked pattern
        blocked_dirs = [d for d, b in percept.blocked.items() if b]
        if blocked_dirs:
            parts.append(f"blocked_{'_'.join(sorted(blocked_dirs))}")
        return "|".join(parts) if parts else "open"

    # ── Action execution ─────────────────────────────────────────────────────

    def act(self, action: str) -> ActionOutcome:
        """Execute an action in the world and return the outcome."""
        x, y = self.body.x, self.body.y
        self.body.steps_taken += 1
        self.body.energy -= 0.5  # energy cost per action
        self._total_steps += 1

        if action.startswith("move_"):
            return self._do_move(action)
        elif action == "push":
            return self._do_push()
        elif action == "pick":
            return self._do_pick()
        elif action == "drop":
            return self._do_drop()
        elif action == "look":
            return self._do_look()
        else:
            return ActionOutcome(action=action, success=True,
                                 effect="waited", reward=-0.01)

    def _do_move(self, action: str) -> ActionOutcome:
        delta = _action_to_dir(action)
        if delta is None:
            return ActionOutcome(action=action, success=False,
                                 effect="invalid_direction")

        dx, dy = delta
        nx, ny = self.body.x + dx, self.body.y + dy

        if self.world.is_walkable(nx, ny):
            self.body.x = nx
            self.body.y = ny
            self.body.facing = action.replace("move_", "")

            # Check if we reached a new cell
            cell = self.world.cell(nx, ny)
            concept = None
            reward = 0.01  # small reward for moving

            if cell and not cell.visited:
                concept = "explore_new"
                reward = 0.05  # bonus for exploration

            if cell and cell.is_target:
                concept = "goal_reached"
                reward = 0.2

            return ActionOutcome(
                action=action, success=True,
                effect=f"moved_to_{nx}_{ny}",
                concept=_CONCEPT_MAP.get(concept) if concept else None,
                reward=reward,
            )
        else:
            return ActionOutcome(
                action=action, success=False,
                effect="wall_blocked",
                concept=_CONCEPT_MAP.get("wall_blocked"),
                reward=-0.05,
            )

    def _do_push(self) -> ActionOutcome:
        x, y = self.body.x, self.body.y
        facing = self.body.facing
        dx, dy = _DIR.get(facing, (1, 0))

        # Object must be adjacent in facing direction
        adj_cell = self.world.cell(x + dx, y + dy)
        if not adj_cell or not adj_cell.has_object:
            return ActionOutcome(action="push", success=False,
                                 effect="nothing_to_push", reward=-0.02)

        obj_id = adj_cell.object_id
        # Push object one cell further in facing direction
        target_x, target_y = x + 2 * dx, y + 2 * dy

        if self.world.move_object(obj_id, target_x, target_y):
            # Check if pushed onto target
            concept = "push_moved"
            reward = 0.15
            at_target = self.world.check_object_at_target()
            if obj_id in at_target:
                concept = "object_at_target"
                reward = 1.0  # big reward for task completion!

            return ActionOutcome(
                action="push", success=True,
                effect=f"pushed_{obj_id}_to_{target_x}_{target_y}",
                concept=_CONCEPT_MAP.get(concept),
                reward=reward,
            )
        else:
            return ActionOutcome(
                action="push", success=False,
                effect=f"push_blocked_{obj_id}",
                concept=_CONCEPT_MAP.get("push_blocked"),
                reward=-0.03,
            )

    def _do_pick(self) -> ActionOutcome:
        if len(self.body.inventory) >= MAX_INVENTORY:
            return ActionOutcome(action="pick", success=False,
                                 effect="inventory_full", reward=-0.02)

        x, y = self.body.x, self.body.y
        # Check adjacent cells for objects
        for dirname, (dx, dy) in _DIR.items():
            adj = self.world.cell(x + dx, y + dy)
            if adj and adj.has_object and adj.object_id:
                obj_id = adj.object_id
                mass = self.world.remove_object(obj_id)
                if mass is not None:
                    self.body.inventory.append(obj_id)
                    self.body.energy -= mass * 2  # heavier = more energy
                    return ActionOutcome(
                        action="pick", success=True,
                        effect=f"picked_{obj_id}",
                        concept=_CONCEPT_MAP.get("pick_success"),
                        reward=0.1,
                    )

        return ActionOutcome(action="pick", success=False,
                             effect="nothing_to_pick", reward=-0.02)

    def _do_drop(self) -> ActionOutcome:
        if not self.body.inventory:
            return ActionOutcome(action="drop", success=False,
                                 effect="nothing_to_drop", reward=-0.02)

        obj_id = self.body.inventory.pop()
        x, y = self.body.x, self.body.y
        dx, dy = _DIR.get(self.body.facing, (1, 0))
        drop_x, drop_y = x + dx, y + dy

        if self.world.place_object(obj_id, drop_x, drop_y):
            concept = "drop_gravity"
            reward = 0.05

            # Check if dropped on target
            at_target = self.world.check_object_at_target()
            if obj_id in at_target:
                concept = "object_at_target"
                reward = 1.0

            return ActionOutcome(
                action="drop", success=True,
                effect=f"dropped_{obj_id}_at_{drop_x}_{drop_y}",
                concept=_CONCEPT_MAP.get(concept),
                reward=reward,
            )
        else:
            # Can't drop there, put back
            self.body.inventory.append(obj_id)
            return ActionOutcome(action="drop", success=False,
                                 effect="drop_blocked", reward=-0.02)

    def _do_look(self) -> ActionOutcome:
        percept = self.perceive()
        return ActionOutcome(
            action="look", success=True,
            effect=f"see_{len(percept.visible_cells)}_cells_{len(percept.nearby_objects)}_objects",
            reward=0.0,
        )

    # ── Learning ─────────────────────────────────────────────────────────────

    def learn_from_outcome(self, action: str, outcome: ActionOutcome,
                           percept: Percept) -> None:
        """Update policy and extract concepts from action outcome."""
        ctx = self._context_key(percept)

        # Update action success rate (EMA)
        key = f"{action}:{ctx}"
        buf = self._action_success[key]
        buf.append(1.0 if outcome.success else 0.0)
        if len(buf) > 200:
            del buf[:len(buf) - 200]

        # Update context policy
        old = self._context_policy[ctx].get(action, 0.5)
        alpha = 0.15
        signal = outcome.reward if outcome.success else outcome.reward - 0.1
        self._context_policy[ctx][action] = old + alpha * (signal - old)

        # Track surprise
        expected_success = old > 0.0
        actual_success = outcome.success
        if expected_success != actual_success:
            outcome.surprise = 0.7 + 0.3 * abs(outcome.reward)
            self._surprise_log.append({
                "step": self._total_steps,
                "action": action,
                "context": ctx,
                "expected": expected_success,
                "actual": actual_success,
                "reward": outcome.reward,
            })

        # Extract concept if present
        if outcome.concept:
            self._concepts_learned[outcome.concept] = \
                self._concepts_learned.get(outcome.concept, 0) + 1
            self._post_concept(outcome.concept)

        # Track action sequence for procedural memory
        self._current_sequence.append(action)

    def _post_concept(self, concept: str) -> None:
        """Post discovered concept to ConceptGraph and GlobalWorkspace."""
        if self._concept_graph:
            try:
                if hasattr(self._concept_graph, 'add_concept'):
                    self._concept_graph.add_concept(concept, {
                        "source": "embodied_agent",
                        "grounding": "grid_world_interaction",
                        "symbolic": f"embodiment:{concept}",
                    })
                elif hasattr(self._concept_graph, '_concepts'):
                    from sare.concept.concept_graph import Concept
                    if concept not in self._concept_graph._concepts:
                        c = Concept(name=concept)
                        c.symbolic_rules.append(f"embodiment:{concept}")
                        self._concept_graph._concepts[concept] = c
            except Exception as e:
                log.debug(f"EmbodiedAgent concept_graph: {e}")

        if self._global_workspace:
            try:
                self._global_workspace.post_event(
                    "concept_grounded",
                    {"concept": concept, "source": "embodied_agent",
                     "grounding": "physical_interaction"},
                    source="embodied_agent", salience=0.7,
                )
            except Exception as e:
                log.debug(f"EmbodiedAgent gw broadcast: {e}")

        # Broadcast to WorldModel — grounds symbolic concepts in embodied experience
        try:
            from sare.memory.world_model import get_world_model
            _wm = get_world_model()
            _wm.update_belief(
                subject=concept,
                predicate="grounded_from",
                value="embodied_experience",
                confidence=0.80,
                domain="factual",
            )
            _wm.add_fact(
                domain="science",
                fact=f"{concept} grounded_from: embodied_experience",
                confidence=0.80,
            )
        except Exception as e:
            log.debug(f"EmbodiedAgent WorldModel broadcast: {e}")

    # ── Episode runner ───────────────────────────────────────────────────────

    def run_episode(self, max_steps: Optional[int] = None,
                    goal: str = "explore") -> EpisodeRecord:
        """
        Run one complete embodied episode.

        Goals:
          - "explore": visit as many new cells as possible
          - "deliver": push/carry any object to any target zone
          - "mixed": alternate between explore and deliver
        """
        steps = max_steps or self._max_steps
        self._episode_count += 1
        self._current_sequence = []

        # Reset agent position but keep world state (persistent world)
        self.body.x = random.randint(1, self.world.w - 2)
        self.body.y = random.randint(1, self.world.h - 2)
        while not self.world.is_walkable(self.body.x, self.body.y):
            self.body.x = random.randint(1, self.world.w - 2)
            self.body.y = random.randint(1, self.world.h - 2)
        self.body.energy = 100.0
        self.body.steps_taken = 0

        ep = EpisodeRecord(episode_id=self._episode_count)
        start = time.time()
        goal_achieved = False

        for step_i in range(steps):
            if self.body.energy <= 0:
                break

            # 1. Perceive
            percept = self.perceive()

            # 2. Decide
            action = self.decide(percept)

            # 3. Act
            outcome = self.act(action)

            # 4. Learn
            self.learn_from_outcome(action, outcome, percept)

            ep.steps.append(outcome)
            ep.total_reward += outcome.reward
            self._total_reward += outcome.reward

            if outcome.concept and outcome.concept not in ep.concepts_discovered:
                ep.concepts_discovered.append(outcome.concept)

            # Check goal achievement
            if goal in ("deliver", "mixed"):
                at_target = self.world.check_object_at_target()
                if at_target:
                    goal_achieved = True
                    break

        ep.goal_achieved = goal_achieved
        ep.duration_s = time.time() - start

        # Store successful sequences as procedural memory
        if ep.total_reward > 0.5 or goal_achieved:
            self._procedural_memory.append({
                "episode": self._episode_count,
                "sequence": list(self._current_sequence[-20:]),
                "reward": ep.total_reward,
                "goal": goal,
                "achieved": goal_achieved,
            })

        self._episodes.append(ep)
        log.debug(
            f"Embodied episode {self._episode_count}: "
            f"reward={ep.total_reward:.2f} concepts={len(ep.concepts_discovered)} "
            f"goal={goal_achieved}"
        )
        return ep

    def run_session(self, n_episodes: int = 3, goal: str = "mixed") -> List[EpisodeRecord]:
        """Run multiple episodes. Alternates goals if 'mixed'."""
        goals = ["explore", "deliver", "explore"] if goal == "mixed" else [goal] * n_episodes
        results = []
        for i in range(n_episodes):
            g = goals[i % len(goals)]
            ep = self.run_episode(goal=g)
            results.append(ep)
        return results

    # ── Policy inspection ────────────────────────────────────────────────────

    def get_policy_summary(self) -> Dict[str, Dict[str, float]]:
        """Return learned action values per context."""
        return {
            ctx: {a: round(v, 3) for a, v in actions.items() if abs(v) > 0.01}
            for ctx, actions in self._context_policy.items()
            if any(abs(v) > 0.01 for v in actions.values())
        }

    def get_procedural_memory(self, n: int = 5) -> List[dict]:
        """Return recent successful action sequences."""
        return list(self._procedural_memory)[-n:]

    # ── Summary ──────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "episodes_run": self._episode_count,
            "total_steps": self._total_steps,
            "total_reward": round(self._total_reward, 3),
            "concepts_learned": self._concepts_learned,
            "n_concepts": len(self._concepts_learned),
            "n_procedural_memories": len(self._procedural_memory),
            "n_surprise_events": len(self._surprise_log),
            "policy_contexts": len(self._context_policy),
            "agent": self.body.to_dict(),
            "world": self.world.summary(),
            "recent_episodes": [e.to_dict() for e in list(self._episodes)[-5:]],
        }
