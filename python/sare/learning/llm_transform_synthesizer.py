"""
LLM Transform Synthesizer — writes new Transform subclasses when the system is stuck.

When ExperimentRunner sees N consecutive unsolved problems in the same domain,
this module asks Qwen3.5 to write a new Transform class, validates it against
known-correct cases, and saves it to data/memory/synthesized_modules/ so
ExperimentRunner._load_synthesized_transforms() will pick it up automatically.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import logging
import re
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_SYNTH_DIR = Path(__file__).resolve().parents[3] / "data" / "memory" / "synthesized_modules"
_SYNTH_DIR.mkdir(parents=True, exist_ok=True)

_VALIDATION_THRESHOLD = 0.2   # fraction of validation cases that must improve (lowered from 0.4)
_MIN_VALIDATION_CASES = 2     # minimum validation cases required (lowered from 3)


# ── Inventiveness scoring ─────────────────────────────────────────────────────

def _inventiveness_score(cls, existing_transform_names: list, validation_frac: float) -> float:
    """
    Compute inventiveness = novelty * usefulness * generality.

    - novelty: 1.0 if name is unique, 0.5 if 4-char prefix matches an existing transform
    - usefulness: validation_frac (pass rate on test cases)
    - generality: fraction of distinct node-type strings in match() source (max 1.0)
    """
    try:
        instance = cls() if callable(cls) else cls
    except Exception:
        return 0.0

    new_name = instance.name() if hasattr(instance, "name") else ""

    novelty = 1.0
    if new_name and len(new_name) >= 4:
        for existing in existing_transform_names:
            if len(existing) >= 4 and new_name[:4].lower() == existing[:4].lower():
                novelty = 0.5
                break

    generality = 0.3
    try:
        if hasattr(instance, "match"):
            src = inspect.getsource(instance.match)
            node_types = set(re.findall(r"""['\"](operator|variable|constant|function|identifier)['\"]""", src))
            generality = min(len(node_types) / 3.0, 1.0)
    except Exception:
        pass

    return round(novelty * validation_frac * generality, 3)


# ── Prompt template ───────────────────────────────────────────────────────────

_PROMPT = """\
You are writing a new Python Transform class for SARE-HX, a symbolic math reasoning engine.

TASK: The system is stuck on {domain} problems. Write a new Transform that handles patterns \
not covered by the existing transforms listed below.

STUCK PROBLEMS (these could not be solved):
{stuck_exprs}

EXISTING TRANSFORMS ALREADY IN SYSTEM (do NOT duplicate these):
{existing_names}

GRAPH API (use ONLY these methods):
- graph.nodes  → list of Node objects  (each has .id, .type, .label, .attributes dict)
- graph.edges  → list of Edge objects  (each has .id, .source, .target, .relationship_type)
- graph.get_node(node_id) → Node or None
- graph.outgoing(node_id) → list of Edge going OUT from this node
- graph.incoming(node_id) → list of Edge going INTO this node
- graph.clone() → deep copy of graph
- graph.add_node(type, label, attributes) → new node id
- graph.add_edge(source_id, target_id, rel_type) → new edge id
- graph.remove_node(node_id)  (also removes connected edges)
- graph.remove_edge(edge_id)

Node types used: "operator", "constant", "variable", "function"
Common operator labels: "+", "-", "*", "/", "^", "neg", "not", "and", "or", "=", "sin", "cos", "ln", "exp"

EXAMPLE TRANSFORM (AddZeroElimination):
```python
class AddZeroElimination(Transform):
    def name(self): return "add_zero_elim"

    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("+", "add"):
                children = graph.outgoing(n.id)
                for e in children:
                    child = graph.get_node(e.target)
                    if child and child.type == "constant" and child.label == "0":
                        other = None
                        for e2 in children:
                            if e2.id != e.id:
                                other = graph.get_node(e2.target)
                        if other:
                            matches.append({{"op": n.id, "zero": child.id, "other": other.id}})
        return matches

    def apply(self, graph, context):
        g = graph.clone()
        op_id, zero_id, other_id = context["op"], context["zero"], context["other"]
        for e in list(g.edges):
            if e.target == op_id:
                g.remove_edge(e.id)
                g.add_edge(e.source, other_id, e.relationship_type)
        g.remove_node(op_id)
        g.remove_node(zero_id)
        return g, -3.0
```

REQUIREMENTS:
1. Class name MUST be unique — use a descriptive name like "{class_name_hint}"
2. `match(self, graph)` returns a list of dicts (one per pattern match found), or []
3. `apply(self, graph, context)` returns (new_graph, delta) where delta < 0 means simpler
4. Import ONLY from: `from sare.engine import Transform, Graph`
   You MUST include this exact import at the top of the file: from typing import Tuple, List, Dict, Any, Optional
   You MAY also use `import math` and `import re`
   FORBIDDEN: `import os`, `import sys`, `import subprocess`, `import socket`, `import shutil`
   FORBIDDEN: `eval(`, `exec(`, `open(`, `__import__(`, `compile(`
5. Return ONLY the raw Python class code — NO markdown fences, NO explanation text, NO ```

Write the Transform class now:"""


# ── Safety check ─────────────────────────────────────────────────────────────

_BANNED_CALLS = {"eval", "exec", "open", "__import__", "compile", "breakpoint"}
_BANNED_IMPORTS = {"os", "sys", "subprocess", "socket", "shutil", "importlib"}


def _strip_boilerplate_imports(code: str) -> str:
    """Remove banned imports that the LLM adds as boilerplate but never uses.

    LLMs often prepend `import sys` or `import os` even when the code doesn't
    reference them.  Strip those lines so the safety check doesn't false-positive.
    Imports whose module IS used in the code body are left in place (the safety
    check will then catch them).
    """
    lines = code.splitlines(keepends=True)
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Match "import <banned>" or "from <banned> import ..."
        if stripped.startswith("import ") or stripped.startswith("from "):
            mod = ""
            if stripped.startswith("import "):
                mod = stripped[7:].split()[0].split(".")[0]
            elif stripped.startswith("from "):
                mod = stripped[5:].split()[0].split(".")[0]
            if mod in _BANNED_IMPORTS:
                # Only strip if the module identifier doesn't appear elsewhere in code
                rest = code.replace(line, "")
                if mod not in rest:
                    continue  # skip this import line
        cleaned.append(line)
    return "".join(cleaned)


def _is_safe(code: str) -> bool:
    # Strip boilerplate imports before checking — prevents false positives from
    # LLM-generated `import sys` lines that are never used.
    code = _strip_boilerplate_imports(code)
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in _BANNED_IMPORTS:
                        return False
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] in _BANNED_IMPORTS:
                    return False
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in _BANNED_CALLS:
                    return False
        return True
    except SyntaxError:
        return False


# ── Code extraction + loading ─────────────────────────────────────────────────

def _extract_code(raw: str) -> str:
    """Strip markdown fences, leading language tag, etc."""
    raw = raw.strip()
    # Remove ```python ... ``` or ``` ... ```
    m = re.search(r"```(?:python)?\s*\n(.*?)\n```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Remove leading "python\n" if LLM disobeys
    if raw.startswith("python\n"):
        raw = raw[7:]
    return raw.strip()


def _load_transform_class(code: str, class_name: str):
    """Exec code in safe namespace, return the Transform subclass or None."""
    from sare.engine import Transform
    ns: dict = {
        "Transform": Transform,
        "Tuple": Tuple,
        "List": List,
        "Dict": Dict,
        "Any": Any,
        "Optional": Optional,
    }
    try:
        exec(compile(code, "<synthesized>", "exec"), ns)
    except Exception as exc:
        log.debug("Exec failed: %s", exc)
        return None
    cls = ns.get(class_name)
    if cls is None:
        # Try to find any Transform subclass in ns
        for v in ns.values():
            if isinstance(v, type) and issubclass(v, Transform) and v is not Transform:
                cls = v
                break
    if cls is None:
        log.debug("No Transform subclass found in synthesized code")
    return cls


# ── Validation ────────────────────────────────────────────────────────────────

def _graph_hash(graph) -> int:
    """Structural hash for a graph (node types + values)."""
    try:
        return hash(tuple(sorted(
            (getattr(n, 'node_type', getattr(n, 'type', '')),
             str(getattr(n, 'value', '') or ''),
             getattr(n, 'label', ''))
            for n in graph.nodes
        )))
    except Exception:
        return 0


def _validate_transform(cls, validation_graphs: list) -> Tuple[bool, float, str]:
    """
    Run the transform against validation_graphs.
    Returns (passed, fraction_improved, message).

    A case counts as improved if ANY of:
      (a) actual energy decreases by > 0.1
      (b) reported delta < -0.1 AND graph structure changed (hash differs)
    """
    from sare.engine import EnergyEvaluator
    energy_fn = EnergyEvaluator()

    if not validation_graphs:
        return False, 0.0, "no validation graphs"

    improved = 0
    errors = 0
    matches_found = 0
    instance = cls()

    for graph in validation_graphs:
        try:
            matches = instance.match(graph)
            if not matches:
                continue
            matches_found += 1
            h_before = _graph_hash(graph)
            new_graph, delta = instance.apply(graph, matches[0])
            e_before = energy_fn.compute(graph).total
            e_after = energy_fn.compute(new_graph).total
            h_after = _graph_hash(new_graph)
            # Accept: actual energy drop OR reported delta with structural change
            if e_after < e_before - 0.1:
                improved += 1
            elif delta < -0.1 and h_after != h_before:
                improved += 1
        except Exception as exc:
            log.debug("Validation error: %s", exc)
            errors += 1

    total = len(validation_graphs)
    frac = improved / total if total > 0 else 0.0
    msg = f"{improved}/{total} improved, {errors} errors"

    # Domain mismatch: transform found no matches at all and no errors
    if matches_found == 0 and errors == 0:
        log.info("[Synth] No matches on validation graphs — possible domain mismatch; deferring")
        return False, 0.0, "no_matches_domain_mismatch"

    threshold = _VALIDATION_THRESHOLD if matches_found >= _MIN_VALIDATION_CASES else 0.2
    passed = frac >= threshold and improved >= 1
    return passed, frac, msg


# ── Main synthesizer ──────────────────────────────────────────────────────────

class LLMTransformSynthesizer:
    """
    Generates new Transform classes via Qwen3.5 when the system is stuck.

    Usage:
        synth = LLMTransformSynthesizer()
        result = synth.synthesize(
            domain="algebra",
            stuck_exprs=["sin(0) + 0", "cos(0) * 1"],
            validation_graphs=[...],
            existing_transform_names=["add_zero_elim", "mul_one_elim", ...],
        )
        if result["promoted"]:
            print("New transform saved:", result["class_name"])
    """

    def __init__(self):
        self._attempts: List[dict] = []
        self._persist_path = Path(__file__).resolve().parents[3] / "data" / "memory" / "synth_attempts.json"
        self._load_attempts()

    def _load_attempts(self):
        if self._persist_path.exists():
            try:
                self._attempts = json.loads(self._persist_path.read_text())
            except Exception:
                pass

    def _save_attempts(self):
        try:
            self._persist_path.write_text(json.dumps(self._attempts[-200:], indent=2))
        except Exception:
            pass

    def _class_name_hint(self, domain: str, stuck_exprs: list) -> str:
        """Generate a plausible class name from context."""
        domain_part = domain.replace("_", " ").title().replace(" ", "")
        # Try to infer a pattern name from the expressions
        ops = set()
        for expr in stuck_exprs:
            for token in re.findall(r"sin|cos|log|ln|exp|sqrt|abs|tan", expr):
                ops.add(token.capitalize())
        op_part = "".join(sorted(ops)[:2]) if ops else "Pattern"
        ts = str(int(time.time()))[-5:]
        return f"Synth{domain_part}{op_part}_{ts}"

    def synthesize(
        self,
        domain: str,
        stuck_exprs: List[str],
        validation_graphs: list,
        existing_transform_names: List[str],
        retries: int = 2,
    ) -> dict:
        """
        Call LLM, generate code, validate, save if good.
        Returns dict with keys: promoted, class_name, code, score, message, path.
        """
        from sare.interface.llm_bridge import _call_llm

        class_name = self._class_name_hint(domain, stuck_exprs)
        # Limit to 15 domain-relevant names to keep prompt short and avoid timeouts
        _domain_kw = domain.lower().replace("_", "")
        _relevant = [n for n in existing_transform_names if _domain_kw in n or domain[:4] in n]
        _other = [n for n in existing_transform_names if n not in _relevant]
        _trimmed = sorted(_relevant)[:10] + sorted(_other)[:5]
        existing_names_str = "\n".join(f"  - {n}" for n in _trimmed)
        stuck_str = "\n".join(f"  - {e}" for e in stuck_exprs[:6])

        prompt = _PROMPT.format(
            domain=domain,
            stuck_exprs=stuck_str,
            existing_names=existing_names_str,
            class_name_hint=class_name,
        )

        for attempt in range(retries):
            try:
                log.info("[Synth] LLM call attempt %d/%d for domain=%s", attempt + 1, retries, domain)
                raw = _call_llm(prompt, use_synthesis_model=True, max_tokens_override=800)
                code = _extract_code(raw)
                # Strip boilerplate imports before safety check and execution
                code = _strip_boilerplate_imports(code)

                if not _is_safe(code):
                    log.warning("[Synth] Code failed safety check (attempt %d)", attempt + 1)
                    self._record(domain, class_name, code, False, 0.0, "safety_fail")
                    continue

                cls = _load_transform_class(code, class_name)
                if cls is None:
                    log.warning("[Synth] Could not load class from code (attempt %d)", attempt + 1)
                    self._record(domain, class_name, code, False, 0.0, "load_fail")
                    continue

                # Use class's actual name if different
                actual_name = cls().name() if hasattr(cls(), "name") else class_name

                # Check not duplicate
                if actual_name in existing_transform_names:
                    log.info("[Synth] Generated duplicate: %s — skipping", actual_name)
                    self._record(domain, class_name, code, False, 0.0, "duplicate")
                    continue

                # Validate
                if len(validation_graphs) >= _MIN_VALIDATION_CASES:
                    passed, score, msg = _validate_transform(cls, validation_graphs)
                else:
                    # Too few graphs to validate — accept with caution
                    passed, score, msg = True, 0.5, "insufficient_validation_graphs"

                try:
                    inventiveness = _inventiveness_score(cls, existing_transform_names, score)
                except Exception:
                    inventiveness = 0.0

                log.info("[Synth] Validation for %s: %s (passed=%s, inventiveness=%.3f)",
                         actual_name, msg, passed, inventiveness)
                self._record(domain, class_name, code, passed, score, msg, inventiveness)

                if passed:
                    # Save to synthesized_modules/
                    save_path = self._save_module(class_name, code, domain, actual_name)
                    return {
                        "promoted": True,
                        "class_name": class_name,
                        "actual_name": actual_name,
                        "code": code,
                        "score": score,
                        "inventiveness": inventiveness,
                        "message": msg,
                        "path": str(save_path),
                    }

            except Exception as exc:
                log.warning("[Synth] Attempt %d error: %s", attempt + 1, exc)
                self._record(domain, class_name, "", False, 0.0, f"error:{exc}")

        return {
            "promoted": False,
            "class_name": class_name,
            "actual_name": class_name,
            "code": "",
            "score": 0.0,
            "message": f"all {retries} attempts failed",
            "path": None,
        }

    def _save_module(self, class_name: str, code: str, domain: str, actual_name: str) -> Path:
        """Write validated transform code to synthesized_modules/."""
        header = textwrap.dedent(f"""\
            # Auto-synthesized Transform — domain: {domain}
            # Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
            # Name: {actual_name}
            from typing import Tuple, List, Dict, Any, Optional
            from sare.engine import Transform, Graph

        """)
        filename = f"{class_name.lower()}.py"
        path = _SYNTH_DIR / filename
        path.write_text(header + code)
        log.info("[Synth] Saved: %s", path)
        return path

    def _record(self, domain: str, class_name: str, code: str,
                promoted: bool, score: float, message: str, inventiveness: float = 0.0):
        self._attempts.append({
            "timestamp": time.time(),
            "domain": domain,
            "class_name": class_name,
            "promoted": promoted,
            "score": score,
            "inventiveness": inventiveness,
            "message": message,
            "code_len": len(code),
        })
        self._save_attempts()

    def stats(self) -> dict:
        total = len(self._attempts)
        promoted = sum(1 for a in self._attempts if a["promoted"])
        recent = self._attempts[-5:]
        return {
            "total_attempts": total,
            "promoted": promoted,
            "success_rate": promoted / total if total else 0.0,
            "recent": recent,
        }


# ── Singleton ──────────────────────────────────────────────────────────────────

_SINGLETON: Optional[LLMTransformSynthesizer] = None


def get_llm_synthesizer() -> LLMTransformSynthesizer:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = LLMTransformSynthesizer()
    return _SINGLETON
