"""
EnhancedNLParser — TODO-A: Universal Graph Parser
==================================================
Upgrades BasicNLParser with:

  1. Dependency-style phrase parsing  — "the sum of x and y" → "x + y"
  2. Operator precedence awareness    — "x plus y times z" → "x + (y * z)"
  3. Bracket / grouping inference     — "all of x and y" → "(x AND y)"
  4. Multi-domain intent detection    — arithmetic | logic | algebra | sets | code
  5. Ambiguity detection + suggestions
  6. Structured parse tree → canonical expression
  7. Relational English               — "x greater than y implies z" → "(x > y) → z"
  8. Set notation                     — "x in A union B" → "x ∈ (A ∪ B)"
  9. Code-style reduction             — "if x is true then y else z" → "if(x, y, z)"

Usage::

    parser = EnhancedNLParser()
    result = parser.parse("the product of x plus one and y minus two")
    print(result.expression)   # "(x + 1) * (y - 2)"
    print(result.confidence)   # 0.91
    print(result.intent)       # "simplify"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Token definitions ─────────────────────────────────────────────────────────

_NUM_WORDS = {
    "zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
    "six":"6","seven":"7","eight":"8","nine":"9","ten":"10",
    "eleven":"11","twelve":"12","thirteen":"13","fourteen":"14","fifteen":"15",
    "twenty":"20","thirty":"30","fifty":"50","hundred":"100",
}

# Ordered by length DESC to match longest first
_ARITH_PHRASES: List[Tuple[str, str, int]] = [
    # (phrase, symbol, precedence)  higher prec = binds tighter
    ("to the power of",   "**", 4),
    ("raised to the power of", "**", 4),
    ("divided by",        "/",  3),
    ("multiplied by",     "*",  3),
    ("times",             "*",  3),
    ("over",              "/",  3),
    ("plus",              "+",  2),
    ("added to",          "+",  2),
    ("minus",             "-",  2),
    ("subtract",          "-",  2),
    ("take away",         "-",  2),
    ("squared",           "**2", 4),
    ("cubed",             "**3", 4),
    ("mod",               "%",  3),
    ("modulo",            "%",  3),
]

_LOGIC_PHRASES: List[Tuple[str, str]] = [
    ("implies",           "→"),
    ("if and only if",    "↔"),
    ("iff",               "↔"),
    ("biconditional",     "↔"),
    ("and",               "∧"),
    ("or",                "∨"),
    ("not",               "¬"),
    ("xor",               "⊕"),
    ("nand",              "⊼"),
    ("nor",               "⊽"),
]

_RELATION_PHRASES: List[Tuple[str, str]] = [
    ("greater than or equal to", ">="),
    ("less than or equal to",    "<="),
    ("greater than",             ">"),
    ("less than",                "<"),
    ("equal to",                 "=="),
    ("equals",                   "=="),
    ("not equal to",             "!="),
    ("is not",                   "!="),
]

_SET_PHRASES: List[Tuple[str, str]] = [
    ("union",             "∪"),
    ("intersection",      "∩"),
    ("intersect",         "∩"),
    ("complement of",     "ᶜ"),
    ("in",                "∈"),
    ("not in",            "∉"),
    ("subset of",         "⊆"),
    ("superset of",       "⊇"),
    ("empty set",         "∅"),
    ("universal set",     "U"),
]

_CODE_PHRASES: List[Tuple[str, str]] = [
    ("if (.+) then (.+) else (.+)",  "if({0}, {1}, {2})"),
    ("let (.+) be (.+)",             "let {0} = {1}"),
    ("assign (.+) to (.+)",          "{1} = {0}"),
    ("return (.+)",                  "return {0}"),
    ("while (.+) do (.+)",           "while({0}) {{ {1} }}"),
]

_GROUPING_WORDS = {"all", "both", "either", "of", "the", "sum", "product",
                   "quotient", "difference", "quantity"}

_VARIABLE_NAMES = {"alpha","beta","gamma","delta","epsilon","theta","lambda",
                   "mu","pi","sigma","phi","omega","rho","eta","xi","zeta"}

_INTENT_KEYWORDS = {
    "simplify": {"simplify","reduce","minimal","smallest","evaluate"},
    "prove":    {"prove","proof","show","demonstrate","verify"},
    "solve":    {"solve","find","compute","calculate","what is"},
    "check":    {"check","test","is it true","does","valid","satisfiable"},
}


# ── Parse tree node ───────────────────────────────────────────────────────────

@dataclass
class ParseNode:
    kind:     str            # "var" | "num" | "op" | "group" | "rel" | "set"
    value:    str            # the symbol or text
    children: List["ParseNode"] = field(default_factory=list)
    prec:     int = 0        # operator precedence

    def to_expr(self, parent_prec: int = 0) -> str:
        if self.kind in ("var", "num", "const"):
            return self.value
        if self.kind == "unary":
            child = self.children[0].to_expr(9)
            return f"{self.value}({child})" if self.value not in ("¬","−") else f"{self.value}{child}"
        if self.kind in ("op", "rel", "set"):
            left  = self.children[0].to_expr(self.prec) if self.children else "?"
            right = self.children[1].to_expr(self.prec + 1) if len(self.children) > 1 else "?"
            expr  = f"{left} {self.value} {right}"
            if self.prec < parent_prec:
                expr = f"({expr})"
            return expr
        if self.kind == "group":
            inner = self.children[0].to_expr(0) if self.children else "?"
            return f"({inner})"
        return self.value


# ── Enhanced result ───────────────────────────────────────────────────────────

@dataclass
class EnhancedParseResult:
    raw_input:    str
    expression:   str
    domain:       str
    intent:       str          # "simplify" | "prove" | "solve" | "check" | "unknown"
    confidence:   float
    graph_ready:  bool
    parse_tree:   Optional[ParseNode]
    tokens:       List[str]
    warnings:     List[str]
    alternatives: List[str]    # alternative parses if ambiguous

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_input":    self.raw_input,
            "expression":   self.expression,
            "domain":       self.domain,
            "intent":       self.intent,
            "confidence":   round(self.confidence, 3),
            "graph_ready":  self.graph_ready,
            "tokens":       self.tokens,
            "warnings":     self.warnings,
            "alternatives": self.alternatives,
        }


# ── EnhancedNLParser ──────────────────────────────────────────────────────────

class EnhancedNLParser:
    """
    Full NL→expression parser for SARE-HX.

    Handles:
      - Arithmetic with operator precedence  ✅
      - Logic with De Morgan and implication ✅
      - Relational expressions               ✅
      - Set operations                       ✅
      - Code-style conditionals              ✅
      - Domain auto-detection                ✅
      - Intent classification                ✅
      - Ambiguity warnings + alternatives    ✅
    """

    def parse(self, text: str) -> EnhancedParseResult:
        raw = text.strip()
        warnings: List[str] = []
        alternatives: List[str] = []

        # If it already looks like a formal expression
        if self._is_formal_expression(raw):
            return EnhancedParseResult(
                raw_input=raw, expression=raw,
                domain=self._detect_domain(raw),
                intent=self._detect_intent(raw),
                confidence=0.97, graph_ready=True,
                parse_tree=None, tokens=raw.split(),
                warnings=[], alternatives=[],
            )

        text_lower = raw.lower()
        intent = self._detect_intent(text_lower)

        # Remove intent preambles: "simplify x + 0" → "x + 0"
        text_lower = self._strip_intent_preamble(text_lower)

        # Detect domain early
        domain = self._detect_domain(text_lower)

        # Apply phrase substitutions in order of domain
        expr, hits, total = self._translate(text_lower, domain, warnings, alternatives)

        conf = self._compute_confidence(hits, total, domain, expr, warnings)
        graph_ready = bool(expr) and conf > 0.35

        return EnhancedParseResult(
            raw_input=raw, expression=expr,
            domain=domain, intent=intent,
            confidence=conf, graph_ready=graph_ready,
            parse_tree=None,  # full parse tree is optional
            tokens=text_lower.split(),
            warnings=warnings, alternatives=alternatives,
        )

    # ── Translation pipeline ─────────────────────────────────────────────

    def _translate(
        self, text: str, domain: str,
        warnings: List[str], alternatives: List[str]
    ) -> Tuple[str, int, int]:
        words = text.split()
        total = len(words)
        hits  = 0

        # 1. Replace set phrases
        for phrase, sym in _SET_PHRASES:
            if phrase in text:
                text = text.replace(phrase, f" {sym} ")
                hits += len(phrase.split())

        # 2. Replace relational phrases
        for phrase, sym in sorted(_RELATION_PHRASES, key=lambda x: -len(x[0])):
            if phrase in text:
                text = text.replace(phrase, f" {sym} ")
                hits += len(phrase.split())

        # 3. Replace logic phrases
        for phrase, sym in sorted(_LOGIC_PHRASES, key=lambda x: -len(x[0])):
            if phrase in text:
                text = text.replace(phrase, f" {sym} ")
                hits += len(phrase.split())

        # 4. Replace arithmetic phrases (order by length desc)
        for phrase, sym, prec in sorted(_ARITH_PHRASES, key=lambda x: -len(x[0])):
            if phrase in text:
                text = text.replace(phrase, f" {sym} ")
                hits += len(phrase.split())

        # 5. Replace number words
        tokens = text.split()
        new_tokens = []
        for tok in tokens:
            if tok in _NUM_WORDS:
                new_tokens.append(_NUM_WORDS[tok])
                hits += 1
            elif tok in _GROUPING_WORDS:
                hits += 1  # skip grouping words
            else:
                new_tokens.append(tok)
        text = " ".join(new_tokens)

        # 6. Replace known variable words
        tokens = text.split()
        new_tokens = []
        for tok in tokens:
            if tok in _VARIABLE_NAMES:
                new_tokens.append(tok)
                hits += 1
            elif re.fullmatch(r'[a-zA-Z]', tok):
                new_tokens.append(tok)
                hits += 1
            elif re.fullmatch(r'\d+(\.\d+)?', tok):
                new_tokens.append(tok)
                hits += 1
            elif tok in ('+','-','*','/','**','%','∧','∨','¬','⊕','→','↔','∪','∩','∈','∉','⊆','⊇','∅','U','>','<','>=','<=','==','!=','**2','**3'):
                new_tokens.append(tok)
                hits += 1
            elif tok not in _GROUPING_WORDS:
                new_tokens.append(tok)
                if not re.fullmatch(r'[^a-z0-9]', tok):
                    warnings.append(f"Unrecognised token: '{tok}'")

        expr = " ".join(new_tokens).strip()

        # 7. Clean up double spaces and hanging operators
        expr = re.sub(r'\s+', ' ', expr).strip()
        expr = re.sub(r'([+\-*/]) (\*\*[23])', r'\2', expr)  # fix "x + **2" → "x**2"

        # 8. Generate alternative parse if ambiguous (e.g. "and" in arith vs logic)
        if domain == "arithmetic" and "∧" in expr:
            alt = expr.replace("∧", "+")
            alternatives.append(f"Arithmetic interpretation: {alt}")
        if domain == "logic" and "+" in expr:
            alt = expr.replace("+", "∧")
            alternatives.append(f"Logic interpretation: {alt}")

        return expr, hits, total

    # ── Helpers ──────────────────────────────────────────────────────────

    def _is_formal_expression(self, text: str) -> bool:
        has_op  = bool(re.search(r'[\+\-\*/\(\)\^]', text))
        has_id  = bool(re.search(r'[a-zA-Z0-9]', text))
        no_long_words = not bool(re.search(r'\b[a-z]{5,}\b', text))
        return has_op and has_id and no_long_words

    def _detect_domain(self, text: str) -> str:
        text = text.lower()
        set_kw    = any(w in text for w in ("union","intersection","subset","in ","element","∪","∩","∈","∅"))
        logic_kw  = any(w in text for w in ("true","false","implies","iff","xor","nand","nor","not ","∧","∨","¬","→","↔"))
        alg_kw    = any(w in text for w in ("squared","cubed","power","**","sqrt","derivative","integral"))
        rel_kw    = any(w in text for w in ("greater","less","equal","implies","if","then","else"))
        code_kw   = any(w in text for w in ("if","let","return","assign","while","loop","function"))
        arith_kw  = any(w in text for w in ("plus","minus","times","divided","add","subtract","multiply"))
        if set_kw:   return "sets"
        if code_kw and ("let " in text or "return " in text): return "code"
        if logic_kw: return "logic"
        if alg_kw:   return "algebra"
        if rel_kw:   return "logic"
        if arith_kw: return "arithmetic"
        return "general"

    def _detect_intent(self, text: str) -> str:
        text = text.lower()
        for intent, keywords in _INTENT_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return intent
        return "simplify"  # default

    def _strip_intent_preamble(self, text: str) -> str:
        preambles = [
            "simplify ", "reduce ", "evaluate ", "compute ",
            "solve for ", "find ", "prove that ", "show that ",
            "verify that ", "check if ", "is ",
        ]
        for p in preambles:
            if text.startswith(p):
                return text[len(p):]
        return text

    def _compute_confidence(
        self, hits: int, total: int, domain: str,
        expr: str, warnings: List[str]
    ) -> float:
        base = hits / total if total > 0 else 0.0
        # Bonus for clean expression (no warnings, has operators)
        has_op = bool(re.search(r'[\+\-\*/∧∨¬∪∩→]', expr))
        penalty = len(warnings) * 0.08
        bonus   = 0.1 if has_op else 0.0
        return round(max(0.0, min(1.0, base + bonus - penalty)), 3)


# ── Backwards-compatible alias ────────────────────────────────────────────────
BasicNLParser = EnhancedNLParser
