"""
Graph builders for code, language, and planning domains.

- CodeGraphBuilder    — Python-like boolean/conditional expressions → Graph
- SentenceGraphBuilder — commonsense triple inference-question graphs
- PlanGraphBuilder    — simple linear planning problem graphs
"""

from __future__ import annotations

from typing import List, Dict, Optional

from sare.engine import Graph, Node, Edge, build_expression_graph


# ──────────────────────────────────────────────────────────────────────────────
#  CodeGraphBuilder
# ──────────────────────────────────────────────────────────────────────────────

class CodeGraphBuilder:
    """
    Parse Python-like boolean/conditional expressions into SARE Graphs.

    Supported patterns (checked in order):
      - ``x if <cond> else y``        → if_node connected to cond, then, else
      - ``x and True``, ``x or False`` → delegates to build_expression_graph
      - ``not not x``                  → delegates to build_expression_graph
      - ``x = x``  (self-assignment)   → assign_node with lhs/rhs pointing at
                                         the same variable node
      - everything else                → delegates to build_expression_graph
    """

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def build(self, code: str) -> Graph:
        """Parse *code* and return a Graph representation."""
        code = code.strip()
        try:
            return self._parse(code)
        except Exception:
            # Graceful fallback: wrap in an error node rather than crashing.
            g = Graph()
            g.add_node("error", code)
            return g

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _parse(self, code: str) -> Graph:
        # ── ternary: "X if COND else Y" ──────────────────────────────
        if " if " in code and " else " in code:
            return self._build_ternary(code)

        # ── self-assignment: "x = x" ──────────────────────────────────
        if "=" in code and "==" not in code:
            parts = code.split("=", 1)
            lhs = parts[0].strip()
            rhs = parts[1].strip()
            if lhs == rhs:
                return self._build_self_assign(lhs)

        # ── delegate: "and", "or", "not" expressions ─────────────────
        #    build_expression_graph handles arithmetic; we just pass through.
        return build_expression_graph(code)

    def _build_ternary(self, code: str) -> Graph:
        """``then_expr if cond_expr else else_expr`` → if-graph."""
        # Split on the FIRST " if " (left side is then-branch value)
        # and on the last " else " to get the else-branch.
        #   form: "<then> if <cond> else <else>"
        if_idx = code.index(" if ")
        else_idx = code.rindex(" else ")

        then_expr = code[:if_idx].strip()
        cond_expr = code[if_idx + 4:else_idx].strip()
        else_expr = code[else_idx + 6:].strip()

        g = Graph()

        # Central if-node
        if_id = g.add_node("if", "if")

        # Condition sub-graph — may be a literal or complex expression
        cond_id = self._embed_value(g, cond_expr)
        g.add_edge(if_id, cond_id, "condition")

        # Then branch
        then_id = self._embed_value(g, then_expr)
        g.add_edge(if_id, then_id, "then")

        # Else branch
        else_id = self._embed_value(g, else_expr)
        g.add_edge(if_id, else_id, "else")

        return g

    def _build_self_assign(self, var_name: str) -> Graph:
        """``x = x`` → assign_node with lhs and rhs edges to the same var."""
        g = Graph()
        assign_id = g.add_node("assign", "=")
        var_id = g.add_node("var", var_name)
        g.add_edge(assign_id, var_id, "lhs")
        g.add_edge(assign_id, var_id, "rhs")
        return g

    @staticmethod
    def _embed_value(g: Graph, expr: str) -> int:
        """
        Embed a simple expression into *g* inline.

        For literals ("True", "False", digits) we add a value node directly.
        For anything complex we build a sub-graph and import its nodes/edges.
        """
        expr = expr.strip()

        # Boolean / simple literal → add value node directly
        if expr in ("True", "False"):
            return g.add_node("value", expr)
        try:
            float(expr)
            return g.add_node("constant", expr)
        except ValueError:
            pass

        # Single identifier (variable)
        if expr.isidentifier():
            return g.add_node("var", expr)

        # Complex sub-expression — build separately and merge
        sub = build_expression_graph(expr)
        return _merge_subgraph(g, sub)


# ──────────────────────────────────────────────────────────────────────────────
#  SentenceGraphBuilder
# ──────────────────────────────────────────────────────────────────────────────

class SentenceGraphBuilder:
    """
    Build "inference question graphs" from commonsense triples.

    Graph layout for a question (subject, relation):

        concept_node(subject) ──[has_relation]──▶ relation_node(relation)
                                                         │[object]
                                                         ▼
                                                   unknown_node(?)

    The caller is expected to run ``FillUnknownTransform`` to resolve the
    ``?`` node using a commonsense KB.
    """

    def build_question(self, subject: str, relation: str) -> Graph:
        """
        Build a question graph hiding the object of (subject, relation, ?).
        """
        g = Graph()
        subj_id = g.add_node("concept", subject.lower())
        rel_id = g.add_node("relation", relation)
        unk_id = g.add_node("unknown", "?")

        g.add_edge(subj_id, rel_id, "has_relation")
        g.add_edge(rel_id, unk_id, "object")
        return g

    def build_from_triple(
        self,
        subj: str,
        rel: str,
        obj: str,
        hide: str = "object",
    ) -> Graph:
        """
        Build a graph from a known triple, hiding one component.

        Parameters
        ----------
        subj, rel, obj : str
            The complete triple.
        hide : "object" | "subject" | "relation"
            Which component to replace with an unknown node.
        """
        g = Graph()

        if hide == "object":
            subj_id = g.add_node("concept", subj.lower())
            rel_id = g.add_node("relation", rel)
            obj_id = g.add_node("unknown", "?", {"answer": obj.lower()})
            g.add_edge(subj_id, rel_id, "has_relation")
            g.add_edge(rel_id, obj_id, "object")

        elif hide == "subject":
            subj_id = g.add_node("unknown", "?", {"answer": subj.lower()})
            rel_id = g.add_node("relation", rel)
            obj_id = g.add_node("concept", obj.lower())
            g.add_edge(subj_id, rel_id, "has_relation")
            g.add_edge(rel_id, obj_id, "object")

        elif hide == "relation":
            subj_id = g.add_node("concept", subj.lower())
            rel_id = g.add_node("unknown", "?", {"answer": rel})
            obj_id = g.add_node("concept", obj.lower())
            g.add_edge(subj_id, rel_id, "has_relation")
            g.add_edge(rel_id, obj_id, "object")

        else:
            raise ValueError(f"hide must be 'object', 'subject', or 'relation'; got {hide!r}")

        return g


# ──────────────────────────────────────────────────────────────────────────────
#  PlanGraphBuilder
# ──────────────────────────────────────────────────────────────────────────────

class PlanGraphBuilder:
    """
    Build simple linear planning problem graphs.

    A plan step is a dict::

        {"action": "buy_milk", "from": "home", "to": "store"}

    The resulting graph looks like::

        state(home) ──[action]──▶ action(buy_milk) ──[result]──▶ state(store) …

    The first ``from`` state is the start; the last ``to`` state is the goal.
    Consecutive steps share state nodes when adjacent locations match.
    """

    def build_plan(self, steps: List[Dict[str, str]]) -> Graph:
        """
        Build a plan graph from a list of step dicts.

        Each dict must have at least ``action``.  ``from`` and ``to`` default
        to "unknown_state" when absent.
        """
        g = Graph()

        if not steps:
            g.add_node("state", "empty_plan")
            return g

        # Cache state nodes by label so shared states reuse the same node.
        state_nodes: Dict[str, int] = {}

        def _get_or_create_state(label: str, is_goal: bool = False) -> int:
            if label not in state_nodes:
                attrs = {"goal": "true"} if is_goal else {}
                nid = g.add_node("state", label, attrs)
                state_nodes[label] = nid
            return state_nodes[label]

        last_step_idx = len(steps) - 1

        for i, step in enumerate(steps):
            try:
                action_label = step.get("action", f"step_{i}")
                from_label = step.get("from", "unknown_state")
                to_label = step.get("to", "unknown_state")
                is_goal_to = (i == last_step_idx)

                from_id = _get_or_create_state(from_label)
                to_id = _get_or_create_state(to_label, is_goal=is_goal_to)

                action_id = g.add_node("action", action_label)
                g.add_edge(from_id, action_id, "action")
                g.add_edge(action_id, to_id, "result")
            except Exception:
                # Skip malformed steps gracefully
                continue

        return g


# ──────────────────────────────────────────────────────────────────────────────
#  Private utility
# ──────────────────────────────────────────────────────────────────────────────

def _merge_subgraph(target: Graph, source: Graph) -> int:
    """
    Import all nodes/edges from *source* into *target*, remapping IDs.

    Returns the ID of the first node encountered in source (heuristic root).
    """
    id_map: Dict[int, int] = {}

    for node in source.nodes:
        new_id = target.add_node(node.type, node.label, dict(node.attributes))
        id_map[node.id] = new_id

    for edge in source.edges:
        src = id_map.get(edge.source)
        tgt = id_map.get(edge.target)
        if src is not None and tgt is not None:
            target.add_edge(src, tgt, edge.relationship_type)

    # Return the remapped ID of the smallest original node ID (likely the root)
    if id_map:
        return id_map[min(id_map)]
    # Fallback: add an unknown node
    return target.add_node("unknown", "?")
