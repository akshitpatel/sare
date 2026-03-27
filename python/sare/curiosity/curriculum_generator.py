
from __future__ import annotations

import json
import random
import time
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

try:
    from sare.sare_bindings import Graph, Node, Edge
except ImportError:
    logging.warning("SARE bindings not found. CurriculumGenerator will not function.")
    Graph = None


def _graph_to_expr(graph) -> str:
    """Best-effort conversion of a graph back to a compact expression string."""
    try:
        labels = []
        nodes = list(graph.nodes) if hasattr(graph, "nodes") else []
        for n in nodes:
            lbl = getattr(n, "label", "")
            if lbl and lbl not in labels:
                labels.append(lbl)
        return " ".join(labels[:8]) if labels else ""
    except Exception:
        return ""


@dataclass
class GeneratedProblem:
    id: str
    graph: Graph
    origin: str = ""
    status: str = "pending"  # "pending" | "solved" | "stuck"
    created_at: float = field(default_factory=time.time)
    domain: str = ""
    expression: str = ""

    def __post_init__(self):
        """Infer domain and expression from graph if not set."""
        if not self.domain and self.graph:
            self.domain = _infer_domain(self.graph)
        if not self.expression and self.graph:
            self.expression = _graph_to_expr(self.graph)


def _infer_domain(graph) -> str:
    """Infer domain from graph node labels/types."""
    try:
        labels = set()
        if hasattr(graph, "get_node_ids"):
            for nid in graph.get_node_ids():
                n = graph.get_node(nid)
                if n:
                    labels.add(getattr(n, "label", ""))
        elif hasattr(graph, "nodes"):
            for n in graph.nodes:
                labels.add(getattr(n, "label", ""))

        # Check for logic operators
        logic_ops = {"not", "and", "or", "implies", "neg", "NOT", "AND", "OR"}
        if labels & logic_ops:
            return "logic"

        # Check for calculus
        calc_ops = {"deriv", "d/dx", "integral", "∫", "sin", "cos", "tan", "ln", "exp"}
        if labels & calc_ops:
            return "calculus"

        # Check for set theory
        set_ops = {"union", "intersect", "complement", "∪", "∩"}
        if labels & set_ops:
            return "set_theory"

        # Check for code/planning/QA domains
        code_ops = {"if", "else", "then", "assign", "return", "def", "True", "False"}
        if labels & code_ops:
            return "code"

        qa_ops = {"unknown", "?", "has_relation", "concept"}
        if labels & qa_ops:
            return "qa"

        plan_ops = {"goal", "step", "precedes", "achieves", "action"}
        if labels & plan_ops:
            return "planning"

        # Check for distributive / factoring patterns
        if "*" in labels and "+" in labels:
            return "algebra"

        # Default arithmetic
        arith_ops = {"+", "-", "*", "/", "^"}
        if labels & arith_ops:
            return "arithmetic"
    except Exception:
        pass
    return "general"


class CurriculumGenerator:
    """
    The 'Explorer' engine. Autonomously generates new problems by mutating
    solved ones, pushing the boundary of the system's capabilities.
    """
    def __init__(self, knowledge_graph=None):
        self.seed_problems: List[Graph] = []
        self.generated_problems: List[GeneratedProblem] = []
        self.problem_history: Dict[str, str] = {}  # ID -> Origin
        self._next_id = 0
        self._persist_path = Path(__file__).resolve().parents[3] / "data" / "memory" / "curriculum.json"
        self._knowledge_graph = knowledge_graph  # Fix 5: KG hierarchy for domain bridging

        # Autobiographical replay: inject hard/failed episodes back into curriculum
        self._autobio = None
        self._batch_count = 0
        try:
            from sare.memory.autobiographical import AutobiographicalMemory
            self._autobio = AutobiographicalMemory()
        except Exception:
            pass

        # Reactive wiring: set when a "surprise_high" event is received
        self._priority_domain: Optional[str] = None
        try:
            from sare.core.event_bus import get_event_bus
            def _on_surprise_high(data):
                try:
                    domain = (data or {}).get("domain", "")
                    if domain:
                        self._priority_domain = domain
                        log.debug("CurriculumGenerator: priority_domain set to %r (surprise=%.2f)",
                                  domain, (data or {}).get("avg_surprise", 0.0))
                except Exception:
                    pass
            get_event_bus().subscribe("surprise_high", _on_surprise_high)
        except Exception:
            pass

    def add_seed(self, graph: Graph):
        """Add a solved problem to the seed pool."""
        self.seed_problems.append(graph)

    def add_problem(self, graph_or_tuple, domain: str = "") -> "GeneratedProblem":
        """
        Add an arbitrary graph as a pending problem (not just a seed).
        Accepts a Graph, a (expr_str, Graph) tuple, or a GeneratedProblem.
        Returns the created GeneratedProblem.
        """
        from sare.curiosity.curriculum_generator import GeneratedProblem  # self-import safe
        if isinstance(graph_or_tuple, GeneratedProblem):
            gp = graph_or_tuple
            if domain:
                gp.domain = domain
            self.generated_problems.append(gp)
            return gp
        if isinstance(graph_or_tuple, tuple):
            graph = graph_or_tuple[1]
        else:
            graph = graph_or_tuple
        pid = f"ext_{self._next_id}"
        self._next_id += 1
        gp = GeneratedProblem(id=pid, graph=graph, origin="external", domain=domain or _infer_domain(graph))
        self.generated_problems.append(gp)
        return gp

    @staticmethod
    def _node_ids(graph) -> List[int]:
        if hasattr(graph, "get_node_ids"):
            return list(graph.get_node_ids())
        return [node.id for node in graph.nodes]

    @staticmethod
    def _incoming_count(graph, node_id: int) -> int:
        if hasattr(graph, "get_incoming"):
            return len(graph.get_incoming(node_id))
        return len(graph.incoming(node_id))

    @staticmethod
    def _get_node_attr(node, key: str, default: str = "") -> str:
        if hasattr(node, "get_attribute"):
            return node.get_attribute(key, default)
        if key == "label":
            return getattr(node, "label", default)
        return getattr(node, "attributes", {}).get(key, default)

    @staticmethod
    def _set_node_attr(node, key: str, value: str) -> None:
        if hasattr(node, "set_attribute"):
            node.set_attribute(key, value)
            return
        if key == "label":
            node.label = value
        else:
            node.attributes[key] = value

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

    def add_failure_for_retry(self, problem) -> None:
        """
        When the world model is highly surprised by a problem, re-queue it
        as a high-priority pending problem. This is the failure-driven curriculum.
        Cap retry queue at 20 to prevent infinite retry flooding.
        """
        try:
            # Cap retry queue to prevent flooding
            retry_pending = sum(
                1 for p in self.generated_problems
                if getattr(p, "origin", "") == "failure_retry"
                and getattr(p, "status", "") == "pending"
            )
            if retry_pending >= 20:
                log.debug("Retry queue full (%d), skipping", retry_pending)
                return

            # If the problem has a graph, add a new mutated variant
            if hasattr(problem, "graph") and problem.graph is not None:
                try:
                    mutated = self._mutate(problem.graph)
                    if mutated is not None:
                        pid = f"retry_{uuid.uuid4().hex[:8]}"
                        origin = "failure_retry"
                        self.problem_history[pid] = origin
                        new_p = GeneratedProblem(
                            id=pid,
                            graph=mutated,
                            origin=origin,
                            status="pending",
                        )
                        self.generated_problems.append(new_p)
                        log.debug("Failure retry queued: %s", pid)
                except Exception:
                    pass
        except Exception as exc:
            log.debug("add_failure_for_retry error: %s", exc)

    def generate_batch(self, size: int = 5) -> List[GeneratedProblem]:
        """Generate a batch of new problems, influenced by learning history."""
        if not self.seed_problems:
            # Fallback: use ProblemGenerator to create expression-based problems
            try:
                from sare.curriculum.problem_generator import ProblemGenerator
                from sare.engine import load_problem as _lp
                pg = ProblemGenerator()
                candidates = pg.generate_batch(n=size)
                added = 0
                for item in candidates:
                    expr = item.get("expression", "")
                    if not expr:
                        continue
                    try:
                        _, g = _lp(expr)
                        if g:
                            self.add_seed(g)
                            added += 1
                    except Exception:
                        pass
                if added:
                    log.info("CurriculumGenerator: ProblemGenerator seeded %d problems (queue was empty)", added)
                    # Now fall through with non-empty seeds
                else:
                    return []
            except Exception as _pge:
                log.debug("ProblemGenerator fallback failed: %s", _pge)
                return []

        # ── Inject up to 2 question-driven problems as priority items ────────
        batch: List[GeneratedProblem] = []
        try:
            from sare.curiosity.question_generator import get_question_generator
            qg = get_question_generator()
            pending_qs = qg.get_pending_questions()
            q_injected = 0
            for q in pending_qs[:2]:
                if q_injected >= 2:
                    break
                # Prefer a seed from the question's domain; fall back to any seed
                domain_seeds = [s for s in self.seed_problems
                                if _infer_domain(s) == q.domain]
                seed = domain_seeds[0] if domain_seeds else (
                    self.seed_problems[0] if self.seed_problems else None)
                if seed is None:
                    continue
                mutated = self._mutate(seed)
                if mutated is None:
                    continue
                pid = f"q_{uuid.uuid4().hex[:8]}"
                self.problem_history[pid] = "question_driven"
                qp = GeneratedProblem(id=pid, graph=mutated, origin="question_driven",
                                      domain=q.domain or _infer_domain(mutated))
                self.generated_problems.append(qp)
                batch.insert(0, qp)  # priority placement
                qg.mark_answered(q.question_id, "investigating")
                q_injected += 1
                log.debug("Question-driven problem injected: domain=%s source=%s", q.domain, q.source)
        except Exception as _qe:
            log.debug("Question injection error: %s", _qe)

        # ── Autobiographical hard-episode replay every 10 batches ───────────
        self._batch_count += 1
        if self._autobio and self._batch_count % 10 == 0:
            try:
                from sare.engine import load_problem as _lp_autobio
                hard_eps = self._autobio.get_hard_episodes(n=15)
                autobio_injected = 0
                for ep in hard_eps:
                    if autobio_injected >= 3:  # cap at 3 per batch
                        break
                    expr = ep.get("expression") or ep.get("description", "")
                    domain = ep.get("domain", "general")
                    if not expr:
                        continue
                    try:
                        _, g = _lp_autobio(str(expr))
                        if g is not None:
                            pid = f"autobio_{uuid.uuid4().hex[:8]}"
                            gp = GeneratedProblem(id=pid, graph=g, origin="autobio_retry", domain=domain)
                            self.generated_problems.append(gp)
                            batch.insert(0, gp)
                            autobio_injected += 1
                    except Exception:
                        pass
                if autobio_injected:
                    log.debug("CurriculumGenerator: injected %d autobio hard episodes", autobio_injected)
            except Exception as _ae:
                log.debug("Autobiographical replay error: %s", _ae)

        # ── Progressive difficulty: inject harder problems for mastered domains ─
        if self._batch_count % 5 == 0:  # every 5 batches
            try:
                from sare.meta.self_model import get_self_model
                from sare.knowledge.problem_factory import ProblemFactory
                from sare.engine import load_problem as _lp_hard
                mastered = get_self_model().get_mastered_domains()
                if mastered:
                    _pf = ProblemFactory()
                    for _dom in mastered[:2]:  # cap at 2 domains per batch
                        hard_probs = _pf.generate_hard(_dom, n=1, complexity=3)
                        for hp in hard_probs:
                            expr = hp.get("expression", "")
                            if not expr:
                                continue
                            try:
                                _, g = _lp_hard(expr)
                                if g is not None:
                                    pid = f"hard_{uuid.uuid4().hex[:8]}"
                                    gp = GeneratedProblem(id=pid, graph=g,
                                                          origin="progressive_hard", domain=_dom)
                                    self.generated_problems.append(gp)
                                    batch.append(gp)
                            except Exception:
                                pass
            except Exception as _hde:
                log.debug("Progressive difficulty error: %s", _hde)

        # ── Include at most 1 retry per batch (prevent retry flooding) ─
        retry_problems = [p for p in self.generated_problems
                          if getattr(p, "origin", "") == "failure_retry"
                          and getattr(p, "status", "pending") == "pending"]
        for rp in retry_problems[:1]:
            batch.append(rp)
            rp.status = "in_progress"

        remaining = size - len(batch)
        if remaining <= 0:
            return batch

        # ── MultiTaskScheduler: weight remaining budget across domains ──────
        _domain_weights: dict = {}
        try:
            from sare.curiosity.multi_task_scheduler import MultiTaskScheduler
            allocation = MultiTaskScheduler().allocate_batch(remaining)
            if allocation:
                _domain_weights = allocation
        except Exception as _mts_e:
            log.debug("MultiTaskScheduler allocation skipped: %s", _mts_e)

        # ── GoalSetter weight boost: active goal domains get 1.5× weight ─
        try:
            from sare.meta.goal_setter import GoalSetter
            _gs = GoalSetter.__new__(GoalSetter)
            _gs._goals = {}
            _gs._goal_count = 0
            _gs.load()
            _goal_domains = _gs.active_goal_domains()
            if _goal_domains and _domain_weights:
                for _gd in _goal_domains:
                    if _gd in _domain_weights:
                        _domain_weights[_gd] = int(_domain_weights[_gd] * 1.5) or 1
                    else:
                        # Add the domain with a baseline weight so it gets representation
                        _avg = max(1, sum(_domain_weights.values()) // max(1, len(_domain_weights)))
                        _domain_weights[_gd] = int(_avg * 1.5) or 1
                log.debug("[CurriculumGenerator] GoalSetter weight boost applied for: %s", _goal_domains)
        except Exception:
            pass

        # ── Use world model + autobiographical memory for smart selection ─
        priority_domains = self._get_priority_domains()

        # Reactive: if a surprise_high event fired, boost that domain to front
        try:
            if self._priority_domain and self._priority_domain not in priority_domains:
                priority_domains.insert(0, self._priority_domain)
            self._priority_domain = None  # consume the flag after one batch
        except Exception:
            pass

        # Novelty detector: prefer seeds that map to novel expressions
        _novelty_detector = None
        try:
            from sare.cognition.novelty_detector import get_novelty_detector
            _novelty_detector = get_novelty_detector()
        except Exception:
            pass

        for _ in range(remaining):
            # If scheduler provided domain weights, bias seed selection
            if _domain_weights:
                import random as _rand
                domains = list(_domain_weights.keys())
                weights = [_domain_weights[d] for d in domains]
                # Apply novelty weighting to domain weights
                if _novelty_detector is not None:
                    try:
                        weights = [
                            w * _novelty_detector.score("", d)
                            for w, d in zip(weights, domains)
                        ]
                    except Exception:
                        pass
                chosen_domain = _rand.choices(domains, weights=weights, k=1)[0]
                domain_seeds = [s for s in self.seed_problems
                                if _infer_domain(s) == chosen_domain]
                seed = (self._select_seed([chosen_domain]) if domain_seeds
                        else self._select_seed(priority_domains))
            else:
                seed = self._select_seed(priority_domains)
            new_problem = self._mutate(seed)
            if new_problem:
                pid = f"gen_{self._next_id}"
                self._next_id += 1
                origin = "mutated_seed"
                self.problem_history[pid] = origin
                gp = GeneratedProblem(id=pid, graph=new_problem, origin=origin)
                batch.append(gp)
                # Record with novelty detector so repeated patterns score lower
                if _novelty_detector is not None:
                    try:
                        _novelty_detector.record(gp.expression, gp.domain)
                    except Exception:
                        pass

        # Record all batch problems (including question-driven/retry items) for novelty tracking
        if _novelty_detector is not None:
            try:
                for _bp in batch:
                    _expr = getattr(_bp, "expression", "")
                    _dom = getattr(_bp, "domain", "general")
                    if _expr and getattr(_bp, "origin", "") != "mutated_seed":
                        # mutated_seed already recorded above; record others here
                        _novelty_detector.record(_expr, _dom)
            except Exception:
                pass

        self.generated_problems.extend(
            [p for p in batch if p not in self.generated_problems])
        return batch

    def _get_priority_domains(self) -> list:
        """Ask world model + autobiographical memory which domains need work."""
        priority = []
        # High-surprise domains from world model (most to learn)
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            for domain, surprise in wm.get_high_surprise_domains(top_n=3):
                if surprise > 0.3:
                    priority.append(domain)
        except Exception:
            pass
        # Domains with recent breakthroughs (reinforce learning)
        try:
            from sare.memory.autobiographical import get_autobiographical_memory
            am = get_autobiographical_memory()
            influence = am.influence_curriculum()
            boost = influence.get("boost_domains", [])
            avoid = influence.get("avoid_domains", [])
            priority.extend(boost)
            # Don't add avoided domains
            priority = [d for d in priority if d not in avoid]
        except Exception:
            pass
        # Dopamine behavior mode influences domain selection:
        #   explore  → pick domains with low engagement (novel territories)
        #   consolidate → pick domains with high engagement (reinforce known)
        try:
            from sare.neuro.dopamine import get_dopamine_system
            ds = get_dopamine_system()
            mode = ds.behavior_mode
            known_domains = list(ds._domain_rewards.keys())
            if known_domains:
                if mode == "explore":
                    # prefer low-engagement domains
                    low_eng = sorted(known_domains, key=lambda d: ds.domain_engagement(d))[:3]
                    priority = low_eng + [p for p in priority if p not in low_eng]
                elif mode == "consolidate":
                    # prefer high-engagement domains
                    high_eng = sorted(known_domains, key=lambda d: -ds.domain_engagement(d))[:3]
                    priority = high_eng + [p for p in priority if p not in high_eng]
        except Exception:
            pass
        # Fix 5: KG hierarchy — if top-priority domain links to another domain in KG, bias toward it
        try:
            kg = self._knowledge_graph
            if kg is not None and priority:
                top_domain = priority[0]
                bridge_domains = []
                if hasattr(kg, "get_related_domains"):
                    bridge_domains = kg.get_related_domains(top_domain) or []
                elif hasattr(kg, "_causal_links"):
                    for link in kg._causal_links.values():
                        if getattr(link, "mechanism", "") == top_domain and hasattr(link, "domain"):
                            bridge_domains.append(link.domain)
                for bd in bridge_domains[:2]:
                    if bd and bd not in priority:
                        priority.append(bd)
        except Exception:
            pass
        # GoalSetter active goals: boost domains that the system has an explicit goal for
        try:
            from sare.meta.goal_setter import GoalSetter
            gs = GoalSetter.__new__(GoalSetter)
            gs._goals = {}
            gs._goal_count = 0
            gs.load()
            goal_domains = gs.active_goal_domains()
            # Insert goal-targeted domains at the front (highest priority) without duplicates
            for gd in reversed(goal_domains):
                if gd and gd not in priority:
                    priority.insert(0, gd)
                elif gd in priority:
                    # Move to front to signal boost
                    priority.remove(gd)
                    priority.insert(0, gd)
            if goal_domains:
                log.debug("[CurriculumGenerator] GoalSetter boosted domains: %s", goal_domains)
        except Exception:
            pass
        return priority

    def _select_seed(self, priority_domains: list):
        """Select a seed, biased toward priority domains if possible.

        When priority_domains is non-empty, 70% of selections come from seeds
        whose inferred domain matches one of the priority domains.  If no
        matching seeds exist the method falls back to a uniform random choice
        so that backward-compatibility is preserved.
        """
        if not priority_domains or random.random() < 0.3:
            return random.choice(self.seed_problems)

        # Build a list of seeds that belong to a priority domain.
        # Seeds may be raw Graph objects (stored in self.seed_problems) or
        # GeneratedProblem instances that already carry a .domain attribute.
        priority_seeds = []
        for seed in self.seed_problems:
            if isinstance(seed, GeneratedProblem):
                domain = seed.domain or _infer_domain(seed.graph)
            else:
                domain = _infer_domain(seed)
            if domain in priority_domains:
                priority_seeds.append(seed)

        if priority_seeds:
            return random.choice(priority_seeds)

        # No seeds match any priority domain — fall back to uniform random.
        return random.choice(self.seed_problems)

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
        for nid in self._node_ids(graph):
            try:
                if self._incoming_count(graph, nid) == 0:
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
            self._set_node_attr(op, "label", "+")
            self._set_node_attr(op, "op", "add")

        zero_id = graph.add_node("constant")
        z = graph.get_node(zero_id)
        if z:
            self._set_node_attr(z, "label", "0")
            self._set_node_attr(z, "value", "0")

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
            self._set_node_attr(op, "label", "*")
            self._set_node_attr(op, "op", "mul")

        one_id = graph.add_node("constant")
        o = graph.get_node(one_id)
        if o:
            self._set_node_attr(o, "label", "1")
            self._set_node_attr(o, "value", "1")

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
            self._set_node_attr(inner, "label", "neg")
            self._set_node_attr(inner, "op", "neg")

        outer_id = graph.add_node("operator")
        outer = graph.get_node(outer_id)
        if outer:
            self._set_node_attr(outer, "label", "neg")
            self._set_node_attr(outer, "op", "neg")

        graph.add_edge(inner_id, root, "operand")
        graph.add_edge(outer_id, inner_id, "operand")
        return graph

    def _mutate_constant(self, graph: Graph) -> Graph:
        """Change a constant value (e.g., 0 -> 1, 1 -> 2)."""
        nodes = []
        # graph.get_node_ids() returns list of IDs
        for nid in self._node_ids(graph):
            node = graph.get_node(nid)
            if node.type in ("constant", "literal"):
                nodes.append(node)
        
        if not nodes:
            return graph # No constants to mutate
            
        target = random.choice(nodes)
        
        # Simple mutation: change value
        current_val = self._get_node_attr(target, "value") or self._get_node_attr(target, "label") or ""
        try:
            val = float(current_val)
            new_val = val + random.choice([-1, 1])
            if new_val < 0 and random.random() < 0.5: new_val = 0 # Bias towards 0/1
            self._set_node_attr(target, "value", str(new_val))
            self._set_node_attr(target, "label", str(new_val)) # Keep both for compatibility
        except ValueError:
            pass # Non-numeric constant
            
        return graph

    def _mutate_operator(self, graph: Graph) -> Graph:
        """Change an operator type (e.g., + -> *)."""
        nodes = []
        for nid in self._node_ids(graph):
            node = graph.get_node(nid)
            if node.type == "operator":
                nodes.append(node)
        
        if not nodes:
            return graph
            
        target = random.choice(nodes)
        
        ops = ["add", "mul", "sub", "div"]  # Basic set
        label_to_op = {"+": "add", "*": "mul", "-": "sub", "/": "div"}
        op_to_label = {"add": "+", "mul": "*", "sub": "-", "div": "/"}

        current_op = self._get_node_attr(target, "op") or label_to_op.get(self._get_node_attr(target, "label"), "")
        if current_op not in ops:
            current_op = ""

        new_op = random.choice([op for op in ops if op != current_op]) if current_op else random.choice(ops)
        self._set_node_attr(target, "op", new_op)
        self._set_node_attr(target, "label", op_to_label.get(new_op, new_op))
        
        return graph

    @staticmethod
    def _graph_to_dict(graph: Graph) -> dict:
        if hasattr(graph, "to_dict"):
            return graph.to_dict()
        if not hasattr(graph, "get_node_ids") or not hasattr(graph, "get_edge_ids"):
            raise TypeError("Graph does not support serialization")

        nodes = []
        for node_id in graph.get_node_ids():
            node = graph.get_node(node_id)
            nodes.append({
                "id": int(node.id),
                "type": str(node.type),
                "label": CurriculumGenerator._get_node_attr(node, "label"),
                "attributes": {
                    key: CurriculumGenerator._get_node_attr(node, key)
                    for key in ("label", "op", "value")
                    if CurriculumGenerator._get_node_attr(node, key)
                },
            })

        edges = []
        for edge_id in graph.get_edge_ids():
            edge = graph.get_edge(edge_id)
            edges.append({
                "id": int(edge.id),
                "source": int(edge.source),
                "target": int(edge.target),
                "type": str(edge.relationship_type),
            })

        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def _graph_from_dict(data: dict):
        if data is None:
            return None
        if "_expr_fallback" in data:
            try:
                from sare.engine import load_problem
                _, g = load_problem(data["_expr_fallback"])
                return g
            except Exception:
                return None
        from sare.engine import Graph as PyGraph
        return PyGraph.from_dict(data)

    # Max entries to persist (keeps file manageable, ~5MB)
    _MAX_SAVED_GENERATED = 3_000
    _MAX_SAVED_SEEDS     = 2_000

    def save(self, path: Optional[Path] = None) -> None:
        """Persist curriculum state so web/bootstrap restores stop failing."""
        target = Path(path or self._persist_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        # Trim in-memory lists before persisting to prevent unbounded growth
        if len(self.generated_problems) > self._MAX_SAVED_GENERATED:
            # Keep most recent pending + all solved, trim oldest generated
            pending = [p for p in self.generated_problems if p.status == "pending"]
            solved  = [p for p in self.generated_problems if p.status != "pending"]
            combined = (solved + pending)[-self._MAX_SAVED_GENERATED:]
            self.generated_problems = combined
        if len(self.seed_problems) > self._MAX_SAVED_SEEDS:
            self.seed_problems = self.seed_problems[-self._MAX_SAVED_SEEDS:]

        def _safe_to_dict(g):
            try:
                return CurriculumGenerator._graph_to_dict(g)
            except (TypeError, AttributeError):
                expr = getattr(g, '_expression', None) or getattr(g, 'expression', None)
                return {"_expr_fallback": str(expr)} if expr else None

        payload = {
            "next_id": self._next_id,
            "problem_history": dict(list(self.problem_history.items())[-self._MAX_SAVED_GENERATED:]),
            "seed_problems": [d for d in (_safe_to_dict(g) for g in self.seed_problems) if d is not None],
            "generated_problems": [
                {
                    "id": problem.id,
                    "graph": _safe_to_dict(problem.graph),
                    "origin": problem.origin,
                    "status": problem.status,
                    "created_at": problem.created_at,
                }
                for problem in self.generated_problems
                if _safe_to_dict(problem.graph) is not None
            ],
        }
        import os as _os
        tmp = target.parent / f"{target.stem}.{_os.getpid()}.tmp"
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _os.replace(tmp, target)

    def load(self, path: Optional[Path] = None) -> None:
        """Restore curriculum state from disk."""
        target = Path(path or self._persist_path)
        if not target.exists():
            return

        payload = json.loads(target.read_text(encoding="utf-8"))
        self._next_id = int(payload.get("next_id", 0))
        self.problem_history = {
            str(problem_id): str(origin)
            for problem_id, origin in payload.get("problem_history", {}).items()
        }
        self.seed_problems = [
            self._graph_from_dict(graph_data)
            for graph_data in payload.get("seed_problems", [])
        ]
        self.generated_problems = [
            GeneratedProblem(
                id=str(problem["id"]),
                graph=self._graph_from_dict(problem["graph"]),
                origin=str(problem.get("origin", "")),
                status=str(problem.get("status", "pending")),
                created_at=float(problem.get("created_at", time.time())),
            )
            for problem in payload.get("generated_problems", [])
        ]
