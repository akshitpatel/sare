
from __future__ import annotations

import random
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from sare.sare_bindings import Graph, Node, Edge
except ImportError:
    logging.warning("SARE bindings not found. CurriculumGenerator will not function.")
    Graph = None


@dataclass
class GeneratedProblem:
    id: str
    graph: Graph
    origin: str = ""
    status: str = "pending"  # "pending" | "solved" | "stuck"
    created_at: float = field(default_factory=time.time)


class CurriculumGenerator:
    """
    The 'Explorer' engine. Autonomously generates new problems by mutating
    solved ones, pushing the boundary of the system's capabilities.
    """
    def __init__(self):
        self.seed_problems: List[Graph] = []
        self.generated_problems: List[GeneratedProblem] = []
        self.problem_history: Dict[str, str] = {}  # ID -> Origin
        self._next_id = 0

    def add_seed(self, graph: Graph):
        """Add a solved problem to the seed pool."""
        self.seed_problems.append(graph)

    def pending_problems(self) -> List[GeneratedProblem]:
        return [p for p in self.generated_problems if p.status == "pending"]

    def get_problem(self, problem_id: str) -> Optional[GeneratedProblem]:
        for p in self.generated_problems:
            if p.id == problem_id:
                return p
        return None

    def mark_solved(self, problem_id: str) -> bool:
        p = self.get_problem(problem_id)
        if not p:
            return False
        p.status = "solved"
        return True

    def mark_stuck(self, problem_id: str) -> bool:
        p = self.get_problem(problem_id)
        if not p:
            return False
        p.status = "stuck"
        return True

    def generate_batch(self, size: int = 5) -> List[GeneratedProblem]:
        """Generate a batch of new problems from seeds."""
        if not self.seed_problems:
            return []
        
        batch: List[GeneratedProblem] = []
        for _ in range(size):
            seed = random.choice(self.seed_problems)
            new_problem = self._mutate(seed)
            if new_problem:
                pid = f"gen_{self._next_id}"
                self._next_id += 1
                origin = "mutated_seed"
                self.problem_history[pid] = origin
                batch.append(GeneratedProblem(id=pid, graph=new_problem, origin=origin))
        
        self.generated_problems.extend(batch)
        return batch

    def _mutate(self, graph: Graph) -> Optional[Graph]:
        """Apply random mutations to a graph clone."""
        if not Graph: return None
        
        new_graph = graph.clone()
        
        # Developmental curriculum heuristic:
        # 1. Optionally perturb (make it slightly different).
        # 2. Always inject a *solvable* redundancy pattern so the engine has
        #    a clear learning signal (energy reduction + trace).
        if random.random() < 0.5:
            new_graph = random.choice([self._mutate_constant, self._mutate_operator])(new_graph)

        wrappers = [self._wrap_add_zero, self._wrap_mul_one, self._wrap_double_neg]
        new_graph = random.choice(wrappers)(new_graph)
        if random.random() < 0.35:
            new_graph = random.choice(wrappers)(new_graph)

        return new_graph

    def _find_root(self, graph: Graph) -> Optional[int]:
        """Heuristic root: node with no incoming edges (prefer operator)."""
        roots = []
        for nid in graph.get_node_ids():
            try:
                if len(graph.get_incoming(nid)) == 0:
                    roots.append(nid)
            except Exception:
                continue

        if not roots:
            return None

        for nid in roots:
            n = graph.get_node(nid)
            if n and n.type == "operator":
                return nid

        return roots[0]

    def _wrap_add_zero(self, graph: Graph) -> Graph:
        """Wrap root: root + 0 (solvable by add_zero)."""
        root = self._find_root(graph)
        if not root:
            return graph

        op_id = graph.add_node("operator")
        op = graph.get_node(op_id)
        if op:
            op.set_attribute("label", "+")
            op.set_attribute("op", "add")

        zero_id = graph.add_node("constant")
        z = graph.get_node(zero_id)
        if z:
            z.set_attribute("label", "0")
            z.set_attribute("value", "0")

        graph.add_edge(op_id, root, "left_operand")
        graph.add_edge(op_id, zero_id, "right_operand")
        return graph

    def _wrap_mul_one(self, graph: Graph) -> Graph:
        """Wrap root: root * 1 (solvable by mul_one)."""
        root = self._find_root(graph)
        if not root:
            return graph

        op_id = graph.add_node("operator")
        op = graph.get_node(op_id)
        if op:
            op.set_attribute("label", "*")
            op.set_attribute("op", "mul")

        one_id = graph.add_node("constant")
        o = graph.get_node(one_id)
        if o:
            o.set_attribute("label", "1")
            o.set_attribute("value", "1")

        graph.add_edge(op_id, root, "left_operand")
        graph.add_edge(op_id, one_id, "right_operand")
        return graph

    def _wrap_double_neg(self, graph: Graph) -> Graph:
        """Wrap root: --root (solvable by double_neg)."""
        root = self._find_root(graph)
        if not root:
            return graph

        inner_id = graph.add_node("operator")
        inner = graph.get_node(inner_id)
        if inner:
            inner.set_attribute("label", "neg")
            inner.set_attribute("op", "neg")

        outer_id = graph.add_node("operator")
        outer = graph.get_node(outer_id)
        if outer:
            outer.set_attribute("label", "neg")
            outer.set_attribute("op", "neg")

        graph.add_edge(inner_id, root, "operand")
        graph.add_edge(outer_id, inner_id, "operand")
        return graph

    def _mutate_constant(self, graph: Graph) -> Graph:
        """Change a constant value (e.g., 0 -> 1, 1 -> 2)."""
        nodes = []
        # graph.get_node_ids() returns list of IDs
        for nid in graph.get_node_ids():
            node = graph.get_node(nid)
            if node.type in ("constant", "literal"):
                nodes.append(node)
        
        if not nodes:
            return graph # No constants to mutate
            
        target = random.choice(nodes)
        
        # Simple mutation: change value
        current_val = target.get_attribute("value") or target.get_attribute("label") or ""
        try:
            val = float(current_val)
            new_val = val + random.choice([-1, 1])
            if new_val < 0 and random.random() < 0.5: new_val = 0 # Bias towards 0/1
            target.set_attribute("value", str(new_val))
            target.set_attribute("label", str(new_val)) # Keep both for compatibility
        except ValueError:
            pass # Non-numeric constant
            
        return graph

    def _mutate_operator(self, graph: Graph) -> Graph:
        """Change an operator type (e.g., + -> *)."""
        nodes = []
        for nid in graph.get_node_ids():
            node = graph.get_node(nid)
            if node.type == "operator":
                nodes.append(node)
        
        if not nodes:
            return graph
            
        target = random.choice(nodes)
        
        ops = ["add", "mul", "sub", "div"]  # Basic set
        label_to_op = {"+": "add", "*": "mul", "-": "sub", "/": "div"}
        op_to_label = {"add": "+", "mul": "*", "sub": "-", "div": "/"}

        current_op = target.get_attribute("op") or label_to_op.get(target.get_attribute("label"), "")
        if current_op not in ops:
            current_op = ""

        new_op = random.choice([op for op in ops if op != current_op]) if current_op else random.choice(ops)
        target.set_attribute("op", new_op)
        target.set_attribute("label", op_to_label.get(new_op, new_op))
        
        return graph
