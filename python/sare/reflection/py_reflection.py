"""
Pure Python Reflection Engine — Extracts abstract rules from solve traces.

Replaces the broken C++ ReflectionEngine (which crashes with TypeError
because it expects C++ Graph objects but receives Python Graph objects).

Algorithm:
  1. Compare initial_graph and final_graph after a successful solve
  2. Find which nodes were removed/added/modified (the diff)
  3. Extract the structural pattern (what kind of simplification happened)
  4. Generalize into an abstract rule with a semantic name
  5. Return the rule for CausalInduction testing and ConceptRegistry promotion

This is the KEY missing piece that enables the system to learn NEW rules
from its own solve experiences.
"""

from __future__ import annotations

import logging
import json
import re
import threading
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class AbstractRule:
    """A discovered abstract rule from reflection."""
    name: str
    domain: str
    pattern_description: str    # human-readable: "op(x, identity) → x"
    removed_node_types: List[str]   # what was removed
    preserved_node_types: List[str] # what was kept
    operator_involved: str      # which operator was simplified
    confidence: float = 0.6
    observations: int = 1

    def valid(self) -> bool:
        return bool(self.name and self.pattern_description and self.observations > 0)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "domain": self.domain,
            "pattern": self.pattern_description,
            "removed": self.removed_node_types,
            "preserved": self.preserved_node_types,
            "operator": self.operator_involved,
            "confidence": round(self.confidence, 3),
            "observations": self.observations,
            "valid": self.valid(),
        }


_ORACLE_BATCH_SIZE = 5
_ORACLE_FLUSH_INTERVAL = 2.0  # seconds between background flush attempts


# ── Singleton pattern for PyReflectionEngine ───────────────────────────────
_reflection_engine_singleton: Optional["PyReflectionEngine"] = None
_reflection_engine_lock = threading.Lock()


def get_reflection_engine() -> Optional["PyReflectionEngine"]:
    """Get the singleton PyReflectionEngine instance."""
    global _reflection_engine_singleton
    with _reflection_engine_lock:
        if _reflection_engine_singleton is None:
            try:
                _reflection_engine_singleton = PyReflectionEngine()
                log.info("[PyReflectionEngine] Singleton instance created")
            except Exception as e:
                log.error("[PyReflectionEngine] Failed to create singleton: %s", e)
                _reflection_engine_singleton = None
        return _reflection_engine_singleton


class PyReflectionEngine:
    """
    Pure Python reflection engine. Analyzes graph diffs to extract rules.

    After a successful solve:
      initial_graph (complex) → final_graph (simpler)

    The diff tells us WHAT changed. The reflection engine generalizes
    this into a reusable rule.

    Oracle calls are batched: up to 10 rule candidates are queued and
    validated in a single LLM request instead of one request per rule.
    """

    def __init__(self, llm_bridge=None):
        self._discovered_rules: Dict[str, AbstractRule] = {}
        self._rejected_rules: set = set()  # names of Oracle-rejected rules; don't re-queue
        self._observation_count = 0
        self._llm_bridge = llm_bridge
        # Queue of (AbstractRule,) waiting for batch oracle validation
        self._oracle_queue: List[AbstractRule] = []
        self._oracle_lock = threading.Lock()
        self._oracle_event = threading.Event()
        self._oracle_thread = threading.Thread(
            target=self._oracle_worker, daemon=True, name="OracleValidator"
        )
        self._oracle_thread.start()

    def _get_llm_bridge(self):
        if not self._llm_bridge:
            try:
                from sare.interface import llm_bridge
                self._llm_bridge = llm_bridge
            except ImportError:
                pass
        return self._llm_bridge

    def _call_oracle_batch(self, candidates: List[AbstractRule]) -> List[Tuple[bool, str]]:
        """
        Validate up to 10 rule candidates in a single LLM call.
        Returns a list of (is_valid, human_name) parallel to `candidates`.
        Falls back to (True, original_name) on any error.
        """
        bridge = self._get_llm_bridge()
        if not bridge:
            log.warning("[PyReflectionEngine] LLM bridge not available, using defaults")
            return [(True, c.name) for c in candidates]

        items_json = json.dumps([
            {
                "id": i,
                "domain": c.domain,
                "pattern": c.pattern_description,
                "removed": c.removed_node_types,
                "kept": c.preserved_node_types,
            }
            for i, c in enumerate(candidates)
        ], indent=2)

        prompt = (
            "You are the Oracle of a symbolic reasoning engine.\n"
            "Validate each of the following induced structural rules.\n"
            "For each rule decide:\n"
            "  - is_valid: true if mathematically sound in general, false if overfitted/hallucinated\n"
            "  - human_name: short formal name (e.g. 'Additive Identity', 'Double Negation')\n\n"
            "Rules to validate:\n"
            + items_json
            + "\n\nReturn ONLY a JSON array (one object per rule, same order):\n"
            "[{\"id\":0,\"is_valid\":true,\"human_name\":\"...\"},...]\n"
            "Return ONLY the JSON array, no markdown."
        )

        defaults = [(True, c.name) for c in candidates]

        # Call LLM with timeout to prevent Oracle worker from hanging
        _result_holder: list = [None]

        def _llm_call():
            try:
                _result_holder[0] = bridge._call_llm(prompt)
            except Exception as _e:
                log.debug("[PyReflectionEngine] Oracle LLM call error: %s", _e)

        try:
            call_thread = threading.Thread(target=_llm_call, daemon=True)
            call_thread.start()
            call_thread.join(timeout=600)  # 10 min — LM Studio can be slow

            if call_thread.is_alive():
                log.error("[PyReflectionEngine] Oracle LLM call timeout (>600s)")
                return defaults

            raw = _result_holder[0]
            if raw is None:
                return defaults
            
            raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`").strip()
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if not m:
                log.warning("[PyReflectionEngine] Oracle LLM response: %s", raw[:200])
                return defaults
            results = json.loads(m.group(0))
            out = list(defaults)  # start with defaults
            for entry in results:
                idx = int(entry.get("id", -1))
                if 0 <= idx < len(candidates):
                    out[idx] = (bool(entry.get("is_valid", True)),
                                str(entry.get("human_name", candidates[idx].name)))
            log.info("[PyReflectionEngine] Oracle validated %d rules: %s", len(out),
                     [(i, v[0], v[1][:30]) for i, v in enumerate(out)])
            return out
        except Exception as e:
            log.error(f"Oracle batch LLM error: {e}")
            return defaults

    def _oracle_worker(self) -> None:
        """Background daemon: drains the oracle queue whenever signalled."""
        log.info("[PyReflectionEngine] Oracle worker thread started")
        while True:
            self._oracle_event.wait(timeout=_ORACLE_FLUSH_INTERVAL)
            self._oracle_event.clear()
            self._flush_oracle_queue()

    def _flush_oracle_queue(self) -> None:
        """Send queued candidates to the Oracle in batches (non-blocking safe)."""
        with self._oracle_lock:
            if not self._oracle_queue:
                return
            # Only take up to _ORACLE_BATCH_SIZE items per flush
            batch = self._oracle_queue[:_ORACLE_BATCH_SIZE]
            del self._oracle_queue[:_ORACLE_BATCH_SIZE]
        log.info("[PyReflectionEngine] Flushing Oracle queue: %d candidates (remaining: %d)",
                 len(batch), len(self._oracle_queue))
        results = self._call_oracle_batch(batch)
        for rule, (is_valid, human_name) in zip(batch, results):
            if is_valid:
                original_name = rule.name  # preserve generated name for lookup
                rule.name = human_name.replace(" ", "_").lower()
                self._discovered_rules[rule.name] = rule
                # Also index by original generated name so reflect() lookup succeeds
                if original_name != rule.name:
                    self._discovered_rules[original_name] = rule
                log.info(f"Oracle VALIDATED concept: {human_name}")
                # Generate a live Transform subclass file for this rule
                try:
                    self._write_transform_file(rule)
                except Exception as _te:
                    log.debug("[PyReflectionEngine] Transform file gen failed: %s", _te)
                try:
                    from sare.meta.inner_monologue import get_inner_monologue
                    get_inner_monologue().think(
                        f"Discovered new rule: '{human_name}' in domain '{rule.domain}'",
                        context="reflection", emotion="excited",
                    )
                except Exception:
                    pass
            else:
                self._rejected_rules.add(rule.name)  # prevent re-queueing
                log.warning(f"Oracle REJECTED rule: {rule.pattern_description}")
                try:
                    from sare.meta.inner_monologue import get_inner_monologue
                    get_inner_monologue().think(
                        f"Rule candidate rejected by Oracle: '{rule.pattern_description[:60]}'",
                        context="reflection", emotion="neutral",
                    )
                except Exception:
                    pass

    def reflect(self, initial_graph, final_graph,
                transforms_applied: List[str] = None,
                domain: str = "general") -> Optional[AbstractRule]:
        """
        Analyze a successful solve to extract an abstract rule.

        Args:
            initial_graph: Graph before solving
            final_graph: Graph after solving
            transforms_applied: List of transform names used
            domain: Problem domain

        Returns:
            AbstractRule if a clear pattern was found, else None
        """
        self._observation_count += 1

        try:
            # 1. Compute the diff
            initial_nodes = {n.id: n for n in initial_graph.nodes}
            final_nodes = {n.id: n for n in final_graph.nodes}

            initial_ids = set(initial_nodes.keys())
            final_ids = set(final_nodes.keys())

            removed_ids = initial_ids - final_ids
            added_ids = final_ids - initial_ids
            kept_ids = initial_ids & final_ids

            # If nothing changed, no rule to extract
            if not removed_ids and not added_ids:
                return None

            # 2. Classify removed nodes
            removed_types = []
            removed_labels = []
            operator_removed = None
            constant_removed = None
            for nid in removed_ids:
                n = initial_nodes[nid]
                removed_types.append(n.type)
                removed_labels.append(n.label)
                if n.type == "operator":
                    operator_removed = n.label
                elif n.type == "constant":
                    constant_removed = n.label

            # 3. Classify preserved nodes
            preserved_types = [initial_nodes[nid].type for nid in kept_ids]

            # 4. Infer the structural pattern
            rule = self._infer_rule(
                removed_types=removed_types,
                removed_labels=removed_labels,
                preserved_types=preserved_types,
                operator_removed=operator_removed,
                constant_removed=constant_removed,
                transforms_applied=transforms_applied or [],
                domain=domain,
                initial_node_count=len(initial_nodes),
                final_node_count=len(final_nodes),
            )

            if rule and rule.valid():
                # Merge with existing rule if same name
                if rule.name in self._discovered_rules:
                    existing = self._discovered_rules[rule.name]
                    existing.observations += 1
                    existing.confidence = min(0.99, existing.confidence + 0.05)
                    return existing
                elif rule.name in self._rejected_rules:
                    return None  # already rejected; skip silently
                else:
                    # New rule — enqueue for async Oracle validation (never blocks caller)
                    with self._oracle_lock:
                        log.info(f"Queuing novel rule candidate for Oracle: {rule.name} (queue={len(self._oracle_queue)+1})")
                        self._oracle_queue.append(rule)
                        if len(self._oracle_queue) >= _ORACLE_BATCH_SIZE:
                            self._oracle_event.set()  # wake worker immediately when batch full
                    # Return None here; validated rules appear in _discovered_rules after flush
                    return None

        except Exception as e:
            log.debug(f"Reflection failed: {e}")

        return None

    def flush(self) -> int:
        """Signal background worker to flush any pending Oracle queue immediately."""
        before = len(self._discovered_rules)
        self._oracle_event.set()
        return len(self._discovered_rules) - before

    def consolidate(self, episodes: list) -> dict:
        """Batch consolidation over recent episodes. Called by HippocampusDaemon."""
        by_domain: dict = {}
        for ep in episodes:
            if getattr(ep, "success", False):
                domain = getattr(ep, "domain", "general")
                by_domain[domain] = by_domain.get(domain, 0) + 1
        total = sum(by_domain.values())
        log.debug("PyReflectionEngine.consolidate: %d successful episodes, %d domains",
                  total, len(by_domain))
        # Signal background worker to drain any partial queue
        if self._oracle_queue:
            log.info(f"Signalling Oracle worker to flush {len(self._oracle_queue)} candidates after consolidate")
            self._oracle_event.set()
        return {"consolidated": total, "domains": by_domain}

    def _infer_rule(self, removed_types: List[str], removed_labels: List[str],
                    preserved_types: List[str], operator_removed: Optional[str],
                    constant_removed: Optional[str],
                    transforms_applied: List[str], domain: str,
                    initial_node_count: int, final_node_count: int) -> Optional[AbstractRule]:
        """Infer an abstract rule from the diff analysis."""

        nodes_removed = initial_node_count - final_node_count

        # Pattern: operator + constant removed, variable preserved
        # → Identity or Annihilation rule
        if operator_removed and constant_removed:
            if "variable" in preserved_types:
                # Identity pattern: op(x, c) → x
                name = f"discovered_{operator_removed}_{constant_removed}_identity"
                pattern = f"{operator_removed}(x, {constant_removed}) → x"
                return AbstractRule(
                    name=name, domain=domain,
                    pattern_description=pattern,
                    removed_node_types=removed_types,
                    preserved_node_types=preserved_types,
                    operator_involved=operator_removed,
                    confidence=0.65,
                )
            else:
                # Annihilation: op(x, c) → c
                name = f"discovered_{operator_removed}_{constant_removed}_annihilation"
                pattern = f"{operator_removed}(x, {constant_removed}) → {constant_removed}"
                return AbstractRule(
                    name=name, domain=domain,
                    pattern_description=pattern,
                    removed_node_types=removed_types,
                    preserved_node_types=preserved_types,
                    operator_involved=operator_removed,
                    confidence=0.60,
                )

        # Pattern: two operators removed (double negation / involution)
        if removed_types.count("operator") >= 2 and nodes_removed >= 2:
            ops = [l for l, t in zip(removed_labels, removed_types) if t == "operator"]
            if len(set(ops)) == 1:
                name = f"discovered_double_{ops[0]}_elimination"
                pattern = f"{ops[0]}({ops[0]}(x)) → x"
                return AbstractRule(
                    name=name, domain=domain,
                    pattern_description=pattern,
                    removed_node_types=removed_types,
                    preserved_node_types=preserved_types,
                    operator_involved=ops[0],
                    confidence=0.70,
                )

        # Pattern: operator removed, constants merged (constant folding)
        if operator_removed and "constant" in preserved_types and nodes_removed >= 2:
            name = f"discovered_{operator_removed}_constant_fold"
            pattern = f"{operator_removed}(c1, c2) → c3"
            return AbstractRule(
                name=name, domain=domain,
                pattern_description=pattern,
                removed_node_types=removed_types,
                preserved_node_types=preserved_types,
                operator_involved=operator_removed,
                confidence=0.75,
            )

        # Pattern: self-cancellation (x op x → identity)
        if operator_removed and nodes_removed >= 2:
            # Check if same-label nodes were removed
            var_labels = [l for l, t in zip(removed_labels, removed_types) if t == "variable"]
            if len(var_labels) >= 2 and len(set(var_labels)) == 1:
                name = f"discovered_{operator_removed}_self_cancel"
                pattern = f"{operator_removed}(x, x) → identity"
                return AbstractRule(
                    name=name, domain=domain,
                    pattern_description=pattern,
                    removed_node_types=removed_types,
                    preserved_node_types=preserved_types,
                    operator_involved=operator_removed,
                    confidence=0.70,
                )

        # Generic: some simplification happened
        if nodes_removed > 0 and transforms_applied:
            primary_transform = transforms_applied[0]
            name = f"discovered_pattern_{primary_transform}"
            pattern = f"Simplification via {primary_transform} (removed {nodes_removed} nodes)"
            return AbstractRule(
                name=name, domain=domain,
                pattern_description=pattern,
                removed_node_types=removed_types,
                preserved_node_types=preserved_types,
                operator_involved=operator_removed or "unknown",
                confidence=0.50,
            )

        return None

    def _write_transform_file(self, rule) -> None:
        """Generate a Python Transform subclass file from an Oracle-validated AbstractRule."""
        import re as _re
        from pathlib import Path as _P

        op = getattr(rule, "operator_involved", None) or ""
        name = rule.name  # human name like "additive_identity"
        pat = getattr(rule, "pattern_description", "") or ""
        removed = getattr(rule, "removed_node_types", []) or []
        preserved = getattr(rule, "preserved_node_types", []) or []

        # Determine rule type from pattern and node types
        # Identity: op(x, const) → x  (removes operator + constant, keeps variable)
        # Annihilation: op(x, const) → const  (removes operator + variable, keeps constant)
        # Double negation: op(op(x)) → x  (removes two operators, keeps variable)

        op_count_removed = removed.count("operator")
        const_removed = "constant" in removed
        var_removed = "variable" in removed
        var_preserved = "variable" in preserved
        const_preserved = "constant" in preserved

        # Build safe class name
        class_name = "Discovered_" + _re.sub(r"[^a-zA-Z0-9]", "_", name).title().replace("_", "")

        synth_dir = _P(__file__).resolve().parents[4] / "data" / "memory" / "synthesized_modules"
        synth_dir.mkdir(parents=True, exist_ok=True)
        out_path = synth_dir / f"rule_{_re.sub(r'[^a-z0-9]', '_', name)}.py"

        # Don't overwrite existing files
        if out_path.exists():
            return

        if op_count_removed == 2 and var_preserved and op:
            # Double negation: neg(neg(x)) → x
            code = (
                f'"""Auto-generated from Oracle-validated rule: {name} — {pat}"""\n'
                'from sare.engine import Transform, Graph\n'
                'from typing import List, Tuple\n'
                '\n'
                f'class {class_name}(Transform):\n'
                f'    """Discovered rule: {pat}"""\n'
                '\n'
                '    def name(self) -> str:\n'
                f'        return {repr(name)}\n'
                '\n'
                '    def match(self, graph: Graph) -> List[dict]:\n'
                '        matches = []\n'
                '        for outer in graph.nodes:\n'
                f'            if outer.type == "operator" and outer.label in ({repr(op)}, {repr(op.replace("neg","not").replace("not","neg"))}):\n'
                '                out_children = graph.outgoing(outer.id)\n'
                '                for e_out in out_children:\n'
                '                    inner = graph.get_node(e_out.target)\n'
                '                    if inner and inner.type == "operator" and inner.label == outer.label:\n'
                '                        in_children = graph.outgoing(inner.id)\n'
                '                        for e_in in in_children:\n'
                '                            target = graph.get_node(e_in.target)\n'
                '                            if target and target.type in ("variable", "symbol", "constant"):\n'
                '                                matches.append({"outer": outer.id, "inner": inner.id, "target": target.id})\n'
                '        return matches\n'
                '\n'
                '    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:\n'
                '        g = graph.clone()\n'
                '        outer_id = context["outer"]\n'
                '        inner_id = context["inner"]\n'
                '        target_id = context["target"]\n'
                '        for e in list(g.edges):\n'
                '            if e.target == outer_id:\n'
                '                g.remove_edge(e.id)\n'
                '                g.add_edge(e.source, target_id, e.relationship_type)\n'
                '        for e in list(g.edges):\n'
                '            if e.source == inner_id or e.target == inner_id:\n'
                '                g.remove_edge(e.id)\n'
                '        for e in list(g.edges):\n'
                '            if e.source == outer_id or e.target == outer_id:\n'
                '                g.remove_edge(e.id)\n'
                '        g.remove_node(outer_id)\n'
                '        g.remove_node(inner_id)\n'
                '        return g, -3.0\n'
            )
        elif op and const_removed and var_preserved and not var_removed:
            # Identity: op(x, const) → x
            const_val = "0"
            m = _re.search(r'\(x,\s*([^)]+)\)', pat)
            if m:
                const_val = m.group(1).strip()
            code = (
                f'"""Auto-generated from Oracle-validated rule: {name} — {pat}"""\n'
                'from sare.engine import Transform, Graph\n'
                'from typing import List, Tuple\n'
                '\n'
                f'class {class_name}(Transform):\n'
                f'    """Discovered rule: {pat}"""\n'
                '\n'
                f'    CONST_VAL = {repr(const_val)}\n'
                f'    OP_LABELS = {repr([op])}\n'
                '\n'
                '    def name(self) -> str:\n'
                f'        return {repr(name)}\n'
                '\n'
                '    def match(self, graph: Graph) -> List[dict]:\n'
                '        matches = []\n'
                '        for n in graph.nodes:\n'
                '            if n.type == "operator" and n.label in self.OP_LABELS:\n'
                '                children = graph.outgoing(n.id)\n'
                '                for e in children:\n'
                '                    child = graph.get_node(e.target)\n'
                '                    if child and child.type == "constant" and child.label == self.CONST_VAL:\n'
                '                        other = None\n'
                '                        for e2 in children:\n'
                '                            if e2.id != e.id:\n'
                '                                other = graph.get_node(e2.target)\n'
                '                        if other:\n'
                '                            matches.append({"op": n.id, "const": child.id, "other": other.id})\n'
                '        return matches\n'
                '\n'
                '    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:\n'
                '        g = graph.clone()\n'
                '        op_id = context["op"]\n'
                '        const_id = context["const"]\n'
                '        other_id = context["other"]\n'
                '        for e in list(g.edges):\n'
                '            if e.target == op_id:\n'
                '                g.remove_edge(e.id)\n'
                '                g.add_edge(e.source, other_id, e.relationship_type)\n'
                '        for e in list(g.edges):\n'
                '            if e.source == op_id or e.target == op_id:\n'
                '                g.remove_edge(e.id)\n'
                '        g.remove_node(op_id)\n'
                '        g.remove_node(const_id)\n'
                '        return g, -3.0\n'
            )
        elif op and var_removed and const_preserved and not const_removed:
            # Annihilation: op(x, const) → const  (e.g. x * 0 = 0)
            const_val = "0"
            m = _re.search(r'\(x,\s*([^)]+)\)', pat)
            if m:
                const_val = m.group(1).strip()
            code = (
                f'"""Auto-generated from Oracle-validated rule: {name} — {pat}"""\n'
                'from sare.engine import Transform, Graph\n'
                'from typing import List, Tuple\n'
                '\n'
                f'class {class_name}(Transform):\n'
                f'    """Discovered rule: {pat}"""\n'
                '\n'
                f'    CONST_VAL = {repr(const_val)}\n'
                f'    OP_LABELS = {repr([op])}\n'
                '\n'
                '    def name(self) -> str:\n'
                f'        return {repr(name)}\n'
                '\n'
                '    def match(self, graph: Graph) -> List[dict]:\n'
                '        matches = []\n'
                '        for n in graph.nodes:\n'
                '            if n.type == "operator" and n.label in self.OP_LABELS:\n'
                '                children = graph.outgoing(n.id)\n'
                '                const_node = None\n'
                '                var_node = None\n'
                '                for e in children:\n'
                '                    child = graph.get_node(e.target)\n'
                '                    if child and child.type == "constant" and child.label == self.CONST_VAL:\n'
                '                        const_node = child\n'
                '                    elif child and child.type in ("variable", "symbol"):\n'
                '                        var_node = child\n'
                '                if const_node and var_node:\n'
                '                    matches.append({"op": n.id, "const": const_node.id, "var": var_node.id})\n'
                '        return matches\n'
                '\n'
                '    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:\n'
                '        g = graph.clone()\n'
                '        op_id = context["op"]\n'
                '        const_id = context["const"]\n'
                '        var_id = context["var"]\n'
                '        for e in list(g.edges):\n'
                '            if e.target == op_id:\n'
                '                g.remove_edge(e.id)\n'
                '                g.add_edge(e.source, const_id, e.relationship_type)\n'
                '        for e in list(g.edges):\n'
                '            if e.source == op_id or e.target == op_id:\n'
                '                g.remove_edge(e.id)\n'
                '        g.remove_node(op_id)\n'
                '        g.remove_node(var_id)\n'
                '        return g, -3.0\n'
            )
        else:
            return  # Can't generate for unknown pattern type

        try:
            out_path.write_text(code)
            log.info("[PyReflectionEngine] Generated Transform file: %s", out_path.name)
        except Exception as e:
            log.debug("[PyReflectionEngine] Failed to write transform file: %s", e)

    def get_discovered_rules(self) -> List[AbstractRule]:
        return list(self._discovered_rules.values())

    def get_high_confidence_rules(self, min_confidence: float = 0.7) -> List[AbstractRule]:
        return [r for r in self._discovered_rules.values()
                if r.confidence >= min_confidence and r.observations >= 2]

    def stats(self) -> dict:
        return {
            "total_reflections": self._observation_count,
            "rules_discovered": len(self._discovered_rules),
            "high_confidence": len(self.get_high_confidence_rules()),
            "oracle_queue_pending": len(self._oracle_queue),
            "oracle_batch_size": _ORACLE_BATCH_SIZE,
            "rules": [r.to_dict() for r in self._discovered_rules.values()],
        }
