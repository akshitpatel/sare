import logging
from typing import List, Optional, Tuple
from sare.interface.llm_bridge import plan_subgoals
from sare.engine import Graph

log = logging.getLogger(__name__)

class MetacognitiveController:
    """
    Tier 7 / Pillar 4 Engine.
    The "Inner Monologue" of SARE-HX.
    Prevents combinatorial explosion by breaking large problems down into small semantic sub-goals
    before engaging the generic MCTS or Beam Search.
    """

    def __init__(self):
        self.sub_goals: List[str] = []
        self.current_goal_index: int = 0

    def generate_plan(self, user_problem: str) -> List[str]:
        """
        Consults the Pre-frontal Cortex (LLM Bridge) to generate a step-by-step logic plan.
        """
        log.info(f"Generating metacognitive plan for: {user_problem}")
        try:
            plan, confidence = plan_subgoals(user_problem)
            if confidence < 0.3:
                log.warning(f"LLM confidence {confidence:.2f} below threshold, falling back to direct solving")
                self.sub_goals = [user_problem]
                self.current_goal_index = 0
                return self.sub_goals
            self.sub_goals = plan
            self.current_goal_index = 0
            log.info(f"Metacognitive Plan generated: {self.sub_goals}")
            return self.sub_goals
        except Exception as e:
            log.error(f"Failed to generate plan: {e}")
            self.sub_goals = [user_problem]  # Fallback to trying to solve the whole thing at once
            return self.sub_goals

    def get_current_goal(self) -> Optional[str]:
        """Gets the active sub-goal string."""
        if self.current_goal_index < len(self.sub_goals):
            return self.sub_goals[self.current_goal_index]
        return None

    def mark_goal_completed(self):
        """Advances the state machine to the next sub-goal."""
        if self.current_goal_index < len(self.sub_goals):
            log.info(f"Metacognitive Goal Completed: {self.sub_goals[self.current_goal_index]}")
            self.current_goal_index += 1

    def is_plan_complete(self) -> bool:
        return self.current_goal_index >= len(self.sub_goals)
