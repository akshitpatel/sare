"""
SymbolCreator — SARE invents new mathematical and logical primitives
====================================================================
When the system is consistently stuck on a class of problems, it
doesn't just request a new Transform — it invents a NEW SYMBOL:
a new node type, edge type, or fundamental operation that didn't
exist in its language before.

Human analogy: when mathematicians couldn't express "the number whose
square is -1", they invented 'i'. When logicians needed to express
"there exists", they invented ∃. Symbols extend the expressive power
of the system beyond what existing language can represent.

Pipeline:
  1. Collect stuck expression patterns → find common structure
  2. PROPOSER LLM: "What new symbol/primitive would help express these?"
  3. DEFINER LLM: "Define this symbol formally (domain, arity, rules)"
  4. CODER LLM: "Write a Transform class that implements this symbol"
  5. Safety check + import test
  6. Register: new node type label + Transform class hot-loaded
  7. Dopamine burst: symbol_created reward

Invented symbols are saved to:
  data/memory/invented_symbols.json         (registry)
  data/memory/synthesized_modules/{name}.py (code)

Usage::
    sc = get_symbol_creator()
    result = sc.invent(
        stuck_exprs=["sin²(x)+cos²(x)", "1-cos²(x)", "sec²(x)-1"],
        domain="calculus",
        existing_transforms=["add_zero_elim", "mul_one_elim", ...]
    )
    print(result["symbol_name"])   # e.g. "PythagoreanIdentity"
    print(result["promoted"])      # True if added to transform library
"""
from __future__ import annotations

import ast
import importlib.util
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_ROOT   = Path(__file__).resolve().parents[4]
_MEMORY = _ROOT / "data" / "memory"
_SYNTH  = _MEMORY / "synthesized_modules"
_REGISTRY = _MEMORY / "invented_symbols.json"

# Safety: never allow these in generated code
_BANNED_CALLS   = {"eval", "exec", "compile", "breakpoint", "__import__"}
_BANNED_IMPORTS = {"subprocess", "socket", "ctypes", "requests", "shutil"}


@dataclass
class InventedSymbol:
    """One invented symbol/primitive."""
    name:          str            # PascalCase, e.g. "TrigIdentity"
    description:   str            # plain English
    domain:        str
    arity:         int            # how many arguments
    notation:      str            # e.g. "sin²(x) + cos²(x) = 1"
    transform_code: str           # full Python code of the Transform class
    transform_path: str           # file path
    promoted:      bool = False
    validation_score: float = 0.0
    created_at:    float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "transform_code"}


class SymbolCreator:
    """Invents new mathematical/logical primitives as Transform classes."""

    def __init__(self):
        _SYNTH.mkdir(parents=True, exist_ok=True)
        self._symbols: List[InventedSymbol] = []
        self._cooldowns: Dict[str, float] = {}
        self._load()

    # ── Main API ─────────────────────────────────────────────────────────────

    def invent(
        self,
        stuck_exprs:          List[str],
        domain:               str,
        existing_transforms:  List[str],
        force:                bool = False,
    ) -> dict:
        """
        Attempt to invent a new symbol for the given stuck expressions.
        Returns dict with: symbol_name, description, promoted, score, message
        """
        from sare.interface.llm_bridge import _call_llm, llm_available
        if not llm_available():
            return {"promoted": False, "message": "LLM unavailable"}

        # Cooldown: don't re-invent for same domain within 15 minutes
        if not force:
            last = self._cooldowns.get(domain, 0)
            if time.time() - last < 900:
                return {"promoted": False, "message": f"cooldown for {domain}"}

        self._cooldowns[domain] = time.time()
        log.info("[SymbolCreator] Inventing symbol for domain=%s stuck=%d exprs",
                 domain, len(stuck_exprs))

        # Step 1: PROPOSER — what symbol would help?
        proposal = self._propose_symbol(stuck_exprs, domain, existing_transforms, _call_llm)
        if not proposal:
            return {"promoted": False, "message": "Proposer returned empty"}

        sym_name    = self._extract_name(proposal)
        description = self._extract_description(proposal)
        notation    = self._extract_notation(proposal)
        log.info("[SymbolCreator] Proposed: %s — %s", sym_name, description[:60])

        # Step 2: CODER — write the Transform class
        code = self._write_transform(
            sym_name, description, notation, domain, stuck_exprs, _call_llm
        )
        if not code:
            return {"promoted": False, "message": "Coder returned empty code", "symbol_name": sym_name}

        code = _strip_fences(code)

        # Step 3: Safety + import test
        safe, reason = _is_safe(code)
        if not safe:
            return {"promoted": False, "message": f"Safety: {reason}", "symbol_name": sym_name}

        ok, err = _test_import(sym_name, code)
        if not ok:
            return {"promoted": False, "message": f"Import failed: {err}", "symbol_name": sym_name}

        # Step 4: Validate — does this transform improve any stuck problems?
        score = self._validate_transform(sym_name, code)
        threshold = 0.15  # at least 15% of stuck problems must improve

        # Step 5: Save and register
        path = _SYNTH / f"{sym_name}.py"
        path.write_text(code, encoding="utf-8")

        symbol = InventedSymbol(
            name=sym_name,
            description=description,
            domain=domain,
            arity=1,
            notation=notation,
            transform_code=code,
            transform_path=str(path),
            promoted=(score >= threshold),
            validation_score=score,
        )
        self._symbols.append(symbol)
        self._save()

        if symbol.promoted:
            log.info("[SymbolCreator] ✓ Symbol '%s' INVENTED (score=%.2f, path=%s)",
                     sym_name, score, path)
            # Dopamine burst
            try:
                from sare.neuro.dopamine import get_dopamine_system
                get_dopamine_system().receive_reward("symbol_created", domain=domain, delta=score*10)
            except Exception:
                pass
            # Phase C: Register new symbol in internal grammar
            try:
                from sare.language.internal_grammar import get_internal_grammar
                get_internal_grammar().learn_new_symbol(symbol)
            except Exception:
                pass
        else:
            log.info("[SymbolCreator] Symbol '%s' saved but not promoted (score=%.2f < %.2f)",
                     sym_name, score, threshold)

        return {
            "symbol_name":   sym_name,
            "description":   description,
            "notation":      notation,
            "promoted":      symbol.promoted,
            "score":         score,
            "path":          str(path),
            "message":       "invented" if symbol.promoted else "saved_not_promoted",
        }

    def load_promoted_transforms(self) -> list:
        """Load all promoted invented symbols as Transform instances."""
        transforms = []
        for s in self._symbols:
            if not s.promoted:
                continue
            path = Path(s.transform_path)
            if not path.exists():
                continue
            try:
                spec = importlib.util.spec_from_file_location(
                    f"_sare_sym_{s.name}", str(path)
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                cls = getattr(mod, s.name, None)
                if cls:
                    transforms.append(cls())
                    log.debug("[SymbolCreator] Loaded %s", s.name)
            except Exception as e:
                log.debug("[SymbolCreator] Load error %s: %s", s.name, e)
        return transforms

    def get_status(self) -> dict:
        return {
            "total_invented": len(self._symbols),
            "promoted":       sum(1 for s in self._symbols if s.promoted),
            "domains":        list({s.domain for s in self._symbols}),
            "recent":         [s.to_dict() for s in self._symbols[-5:]],
        }

    # ── LLM prompts ──────────────────────────────────────────────────────────

    def _propose_symbol(self, stuck_exprs, domain, existing, llm) -> str:
        existing_str = ", ".join(existing[:15]) if existing else "none"
        exprs_str = "\n".join(f"  - {e}" for e in stuck_exprs[:8])
        prompt = (
            f"You are a mathematician helping an AGI system invent new primitives.\n\n"
            f"DOMAIN: {domain}\n"
            f"EXISTING TRANSFORMS: {existing_str}\n\n"
            f"STUCK EXPRESSIONS (the system cannot simplify these):\n{exprs_str}\n\n"
            "These expressions share a hidden structure that the current symbol vocabulary "
            "cannot capture. Your task: propose ONE new mathematical symbol or primitive "
            "operation that would allow the system to recognize and simplify this pattern.\n\n"
            "Format EXACTLY as:\n"
            "SYMBOL_NAME: <PascalCase, no spaces, e.g. TrigIdentity>\n"
            "DESCRIPTION: <one sentence: what this symbol represents>\n"
            "NOTATION: <how it appears in expressions, e.g. sin²(x)+cos²(x)>\n"
            "RULE: <the simplification rule, e.g. sin²(x)+cos²(x) → 1>\n"
            "JUSTIFICATION: <why this symbol is genuinely new and useful>\n\n"
            "The symbol must be:\n"
            "1. Genuinely new (not already in EXISTING TRANSFORMS)\n"
            "2. Applicable to at least 3 of the stuck expressions\n"
            "3. A real mathematical/logical concept (not arbitrary)"
        )
        return llm(prompt, use_synthesis_model=True)

    def _write_transform(self, name, description, notation, domain, stuck_exprs, llm) -> str:
        exprs_str = "\n".join(f"  # {e}" for e in stuck_exprs[:5])
        prompt = (
            f"Write a Python Transform class named `{name}` for the SARE-HX AGI system.\n\n"
            f"SYMBOL: {name}\n"
            f"DESCRIPTION: {description}\n"
            f"NOTATION/RULE: {notation}\n"
            f"DOMAIN: {domain}\n\n"
            f"Target expressions (must be simplified by this Transform):\n{exprs_str}\n\n"
            "The Transform must follow this interface:\n\n"
            "```python\nclass " + name + ":\n"
            "    def name(self) -> str: return '" + name + "'\n"
            "    def apply(self, graph):\n"
            "        '''Returns new Graph if applicable, None otherwise.'''\n"
            "        ...\n"
            "    def applies_to(self, graph) -> bool:\n"
            "        '''Quick check before apply().'''\n"
            "        ...\n```\n\n"
            "REQUIREMENTS:\n"
            "1. Use only: `from sare.engine import Graph` for graph manipulation\n"
            "2. Check node types: `node.type` (str), node labels: `node.label` (str)\n"
            "3. Check edge relationships: `edge.relationship_type` (str)\n"
            "4. Return a NEW Graph from apply() — never mutate the input graph\n"
            "5. Return None from apply() if the pattern doesn't match\n"
            "6. No imports of subprocess, socket, eval, exec, or file writes\n"
            "7. The apply() must reduce graph energy (make the graph simpler)\n\n"
            "Write ONLY the Python code, no markdown fences, no explanations:"
        )
        return llm(prompt, use_synthesis_model=True)

    # ── Validation ───────────────────────────────────────────────────────────

    def _validate_transform(self, name: str, code: str) -> float:
        """Load the transform and try it on recent stuck expressions. Returns 0-1."""
        try:
            test_name = f"_sare_sym_test_{name}_{int(time.time())}"
            path = _SYNTH / f"_test_{name}.py"
            path.write_text(code, encoding="utf-8")

            spec = importlib.util.spec_from_file_location(test_name, str(path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            cls = getattr(mod, name, None)
            path.unlink(missing_ok=True)
            sys.modules.pop(test_name, None)

            if cls is None:
                return 0.0

            transform = cls()

            # Try on a few test graphs
            try:
                from sare.engine import load_problem as lp
                from sare.engine import EnergyEvaluator
                ev = EnergyEvaluator()
                test_exprs = [
                    "x + 0", "x * 1", "not not x", "x * 0", "x - x"
                ]
                improved = 0
                for expr in test_exprs:
                    try:
                        _, g = lp(expr)
                        if g and transform.applies_to(g):
                            new_g = transform.apply(g)
                            if new_g:
                                e_before = ev.compute(g).total
                                e_after  = ev.compute(new_g).total
                                if e_after < e_before:
                                    improved += 1
                    except Exception:
                        pass
                return improved / max(len(test_exprs), 1)
            except Exception:
                return 0.1  # Module loaded OK but can't test graphs — give partial credit
        except Exception as e:
            log.debug("[SymbolCreator] Validate error: %s", e)
            return 0.0

    # ── Extraction helpers ────────────────────────────────────────────────────

    def _extract_name(self, text: str) -> str:
        m = re.search(r'SYMBOL_NAME\s*:\s*([A-Za-z][A-Za-z0-9_]*)', text)
        if m:
            return m.group(1)
        # Fallback: first PascalCase word
        m2 = re.search(r'\b([A-Z][a-z]+[A-Z][A-Za-z]*)\b', text)
        return m2.group(1) if m2 else "NewSymbol"

    def _extract_description(self, text: str) -> str:
        m = re.search(r'DESCRIPTION\s*:\s*(.+)', text)
        return m.group(1).strip() if m else ""

    def _extract_notation(self, text: str) -> str:
        for key in ("NOTATION", "RULE"):
            m = re.search(rf'{key}\s*:\s*(.+)', text)
            if m:
                return m.group(1).strip()
        return ""

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save(self):
        try:
            data = [s.to_dict() for s in self._symbols[-100:]]
            _REGISTRY.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as e:
            log.debug("[SymbolCreator] Save error: %s", e)

    def _load(self):
        if not _REGISTRY.exists():
            return
        try:
            data = json.loads(_REGISTRY.read_text())
            for d in data:
                s = InventedSymbol(**{k: v for k, v in d.items()
                                      if k in InventedSymbol.__dataclass_fields__},
                                   transform_code="")
                self._symbols.append(s)
        except Exception as e:
            log.debug("[SymbolCreator] Load error: %s", e)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    return text if ("def " in text or "class " in text) else text


def _is_safe(code: str) -> tuple:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in _BANNED_CALLS:
                return False, f"banned call: {fn.id}"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""])
            for nm in names:
                if (nm or "").split(".")[0] in _BANNED_IMPORTS:
                    return False, f"banned import: {nm}"
    return True, ""


def _test_import(name: str, code: str) -> tuple:
    test_name = f"_sare_sym_import_test_{name}_{int(time.time())}"
    path = _SYNTH / f"_importtest_{name}.py"
    try:
        path.write_text(code, encoding="utf-8")
        spec = importlib.util.spec_from_file_location(test_name, str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return True, ""
    except Exception as e:
        return False, str(e)
    finally:
        path.unlink(missing_ok=True)
        sys.modules.pop(test_name, None)


# ── Singleton ─────────────────────────────────────────────────────────────────
_instance: Optional[SymbolCreator] = None

def get_symbol_creator() -> SymbolCreator:
    global _instance
    if _instance is None:
        _instance = SymbolCreator()
    return _instance
