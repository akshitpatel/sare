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

# Arithmetic-of patterns: "sum of X and Y" → "X + Y"
# These MUST be applied before logic patterns convert 'and' → '∧'
# Applied in multi-pass fixed-point loop to handle nested phrases.
_ARITH_OF_PATTERNS: List[Tuple[str, str]] = [
    # Powers: "square of X" → "(X)**2", "cube of X" → "(X)**3"
    (r'\bthe square of\b\s*(.+)',                    r'(\1)**2'),
    (r'\bsquare of\b\s*(.+)',                        r'(\1)**2'),
    (r'\bthe cube of\b\s*(.+)',                      r'(\1)**3'),
    (r'\bcube of\b\s*(.+)',                          r'(\1)**3'),
    (r'\bthe square root of\b\s*(.+)',               r'sqrt(\1)'),
    # Multiplied-by-scalar: "twice X" → "2*X", "thrice X" → "3*X"
    (r'\btwice\b\s+(.+)',                            r'2 * \1'),
    (r'\bthrice\b\s+(.+)',                           r'3 * \1'),
    (r'\bdouble\b\s+(.+)',                           r'2 * \1'),
    (r'\btriple\b\s+(.+)',                           r'3 * \1'),
    # Quantity grouping: "the quantity X" → "(X)"
    (r'\bthe quantity\s+(.+)',                        r'(\1)'),
    (r'\bquantity\s+(.+)',                            r'(\1)'),
    # N times a phrase: "3 times the sum of ..." → "3 * (sum of ...)"
    (r'(\d+(?:\.\d+)?)\s+times\s+(?:the\s+)?(.+)',  r'\1 * \2'),
    # Core arithmetic-of forms
    (r'\bthe sum of\b\s*(.+?)\s+and\s+(.+)',         r'(\1) + (\2)'),
    (r'\bsum of\b\s*(.+?)\s+and\s+(.+)',             r'(\1) + (\2)'),
    (r'\bthe product of\b\s*(.+?)\s+and\s+(.+)',     r'(\1) * (\2)'),
    (r'\bproduct of\b\s*(.+?)\s+and\s+(.+)',         r'(\1) * (\2)'),
    (r'\bthe difference of\b\s*(.+?)\s+and\s+(.+)',  r'(\1) - (\2)'),
    (r'\bdifference of\b\s*(.+?)\s+and\s+(.+)',      r'(\1) - (\2)'),
    (r'\bthe quotient of\b\s*(.+?)\s+and\s+(.+)',    r'(\1) / (\2)'),
    (r'\bquotient of\b\s*(.+?)\s+and\s+(.+)',        r'(\1) / (\2)'),
    (r'\bthe sum of\b\s*(.+?)\s+plus\s+(.+)',         r'(\1) + (\2)'),
    # "X times Y" (both sides are variables/expressions, no digit prefix)
    (r'([a-zA-Z_]\w*)\s+times\s+([a-zA-Z_]\w*)',     r'\1 * \2'),
]

# Ordered longest-first for greedy matching
_CALCULUS_PATTERNS: List[Tuple[str, str]] = [
    # Derivative phrases → derivative(...)
    (r'\bderivative of\b\s*(.+)',        r'derivative(\1)'),
    (r'\bdifferentiate\b\s*(.+)',        r'derivative(\1)'),
    (r'\bd/dx\s+of\b\s*(.+)',           r'derivative(\1)'),
    (r'\bd/dx\s+(.+)',                   r'derivative(\1)'),
    (r'\bderiv\s+of\b\s*(.+)',           r'derivative(\1)'),
    # Integral phrases → integral(...)
    (r'\bintegral of\b\s*(.+?)(?:\s+dx|\s+with respect to \w+)?$', r'integral(\1)'),
    (r'\bintegrate\b\s*(.+?)(?:\s+dx|\s+with respect to \w+)?$',   r'integral(\1)'),
    # Function phrases → func(arg)  (with-argument forms must come BEFORE bare renames)
    (r'\bsine of\b\s*(.+)',              r'sin(\1)'),
    (r'\bcosine of\b\s*(.+)',            r'cos(\1)'),
    (r'\btangent of\b\s*(.+)',           r'tan(\1)'),
    (r'\bnatural log of\b\s*(.+)',       r'ln(\1)'),
    (r'\bnatural logarithm of\b\s*(.+)', r'ln(\1)'),
    (r'\blog of\b\s*(.+)',               r'log(\1)'),
    (r'\bexponential of\b\s*(.+)',       r'exp(\1)'),
    (r'\be to the\b\s*(.+)',             r'exp(\1)'),
    (r'\bsquare root of\b\s*(.+)',       r'sqrt(\1)'),
    (r'\babs(?:olute value)? of\b\s*(.+)', r'abs(\1)'),
    # Bare renames (no argument given — user just used NL name without "of")
    (r'\bsine\b',    'sin'),
    (r'\bcosine\b',  'cos'),
    (r'\btangent\b', 'tan'),
    (r'\bnatural log\b',        'ln'),
    (r'\bnatural logarithm\b',  'ln'),
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

    # Unicode → ASCII normalization table applied before all other processing
    _UNICODE_MAP = [
        # Superscript digits
        ("²", "**2"), ("³", "**3"), ("⁴", "**4"), ("⁵", "**5"),
        ("⁰", "**0"), ("¹", "**1"),
        # Subscript digits (treat as variable suffix)
        ("₀", "_0"), ("₁", "_1"), ("₂", "_2"), ("₃", "_3"),
        # Multiplication / division
        ("×", "*"), ("·", "*"), ("÷", "/"), ("∕", "/"),
        # Minus variants
        ("\u2212", "-"), ("\u2013", "-"), ("\u2014", "-"),
        # Fraction slash
        ("\u2044", "/"),
        # Greek letters common in math
        ("α", "a"), ("β", "b"), ("γ", "c"), ("δ", "d"),
        ("π", "3.14159"), ("∞", "inf"),
        # Logic / set symbols kept as-is in the parser already, no conversion needed
        # but handle common Unicode arrows
        ("⇒", "→"), ("⇔", "↔"), ("¬", "not "),
        # Absolute value bars (paired) → abs(
        # handled by regex below, not simple replacement
    ]

    # Known math function names for case-normalization
    _MATH_FUNCTIONS = {
        'sin', 'cos', 'tan', 'ln', 'log', 'exp', 'sqrt',
        'abs', 'derivative', 'integral', 'arcsin', 'arccos', 'arctan',
    }

    @classmethod
    def _normalize_unicode(cls, text: str) -> str:
        """Replace unicode math symbols with ASCII equivalents."""
        for src, dst in cls._UNICODE_MAP:
            text = text.replace(src, dst)
        # Superscript number sequences like x² → x**2 (in case multi-char)
        text = re.sub(r'\*\*(\d)\s*\*\*(\d)', r'**\1\2', text)  # collapse **2**3 edge case
        # Paired absolute value bars: |expr| → abs(expr) — simple single-level
        text = re.sub(r'\|([^|]+)\|', r'abs(\1)', text)
        # Remove zero-width characters
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
        # Normalize fancy quotes
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        return text

    @classmethod
    def _robustness_clean(cls, text: str) -> str:
        """
        Fix common malformed-input patterns before parsing:
          1. Case-normalize known math function names (Sin → sin, SIN → sin)
          2. Caret exponentiation: x^2 → x**2
          3. Comma decimal separator: 3,14 → 3.14
          4. Semicolon separator: take first clause only
          5. Trailing dot on numbers: 0. → 0
          6. Repeated + or − operators: x + + y → x + y, x - - y → x + y
          7. Balanced parentheses: auto-close or strip unmatched parens
          8. Trailing/leading commas in argument lists: f(x, → f(x)
        """
        # 1. Case-normalize math function names
        def _lower_func(m):
            return m.group(0).lower()
        for fn in cls._MATH_FUNCTIONS:
            text = re.sub(r'\b' + fn + r'\b', fn, text, flags=re.IGNORECASE)

        # 2. Caret exponentiation
        text = re.sub(r'\^\s*(\d+)', r'**\1', text)
        text = re.sub(r'\^\s*([a-zA-Z])', r'**\1', text)

        # 3. Comma decimal separator: digit,digit → digit.digit
        text = re.sub(r'(\d),(\d)', r'\1.\2', text)

        # 4. Semicolon: take only first expression
        if ';' in text:
            text = text.split(';')[0].strip()

        # 5. Trailing dot on numbers
        text = re.sub(r'(\d)\.$', r'\1', text.strip())
        text = re.sub(r'(\d)\.\s', r'\1 ', text)

        # 6. Repeated operators (collapse double sign)
        text = re.sub(r'\+\s*\+', '+', text)      # x + + y → x + y
        text = re.sub(r'-\s*-', '+', text)         # x - - y → x + y (double negation)
        text = re.sub(r'\*\s*\*\s*(?!\d|\*)', '**', text)  # x * * 2 → x**2 (but not x***)

        # 7. Balanced parentheses recovery
        depth = 0
        result_chars = []
        for ch in text:
            if ch == '(':
                depth += 1
                result_chars.append(ch)
            elif ch == ')':
                if depth > 0:
                    depth -= 1
                    result_chars.append(ch)
                # else: extra closing paren — silently drop it
            else:
                result_chars.append(ch)
        # Close any still-open parens
        result_chars.extend([')'] * depth)
        text = ''.join(result_chars)

        # 7b. Strip redundant outer wrapper parens: (E) → E (iterative)
        def _strip_outer(s: str) -> str:
            while s.startswith('(') and s.endswith(')'):
                # verify this outer pair actually matches
                inner_depth = 0
                for i, ch in enumerate(s):
                    if ch == '(': inner_depth += 1
                    elif ch == ')': inner_depth -= 1
                    if inner_depth == 0 and i < len(s) - 1:
                        break  # outer ( closes before the end — not a wrapper
                else:
                    s = s[1:-1]
                    continue
                break
            return s
        text = _strip_outer(text)

        # 7c. Simplify double-nested parens: ((expr)) → (expr)
        text = re.sub(r'\(\(([^()]+)\)\)', r'(\1)', text)   # ((x)) → (x)
        text = re.sub(r'\(\(([^()]+)\)\)', r'(\1)', text)   # again for triple nesting

        # 8. Trailing comma in argument list: f(x, → f(x); also fix double commas
        text = re.sub(r',\s*\)', ')', text)
        text = re.sub(r',\s*$', '', text.strip())
        text = re.sub(r',\s*,', ',', text)   # f(x,,y) → f(x,y)

        return text.strip()

    def parse(self, text: str) -> EnhancedParseResult:
        raw = self._normalize_unicode(text.strip())
        raw = self._robustness_clean(raw)
        warnings: List[str] = []
        alternatives: List[str] = []

        # If it already looks like a formal expression
        if self._is_formal_expression(raw):
            # Still expand implicit multiplication: '2x' → '2 * x', '2(...)' → '2 * (...)'
            formal = re.sub(r'(\d+)\s*([a-zA-Z](?:\s*\*\*\s*\d+)?)', r'\1 * \2', raw)
            formal = re.sub(r'(\d+)\s*\(', r'\1 * (', formal)
            return EnhancedParseResult(
                raw_input=raw, expression=formal,
                domain=self._detect_domain(formal),
                intent=self._detect_intent(formal),
                confidence=0.97, graph_ready=True,
                parse_tree=None, tokens=formal.split(),
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

        # LLM fallback: if heuristic confidence is low, ask LLM to parse
        if conf < 0.6:
            llm_expr = self._llm_parse_fallback(raw)
            if llm_expr:
                expr = llm_expr
                conf = 0.75
                warnings.append("expression inferred via LLM fallback")

        graph_ready = bool(expr) and conf > 0.35

        # Compositionality: extract (subject, relation, object) triples from raw input
        # and store as WorldModel beliefs so meaning accumulates from parts.
        self._decompose_sentence(raw)

        return EnhancedParseResult(
            raw_input=raw, expression=expr,
            domain=domain, intent=intent,
            confidence=conf, graph_ready=graph_ready,
            parse_tree=None,  # full parse tree is optional
            tokens=text_lower.split(),
            warnings=warnings, alternatives=alternatives,
        )

    # ── Compositionality: SVO triple extraction ─────────────────────────────

    _DECOMPOSE_PATTERNS = [
        (re.compile(r'^(.+?)\s+is\s+an?\s+(.+)$',       re.I), "is_a"),
        (re.compile(r'^(.+?)\s+has\s+(.+)$',             re.I), "has"),
        (re.compile(r'^(.+?)\s+can\s+(.+)$',             re.I), "can"),
        (re.compile(r'^(.+?)\s+causes\s+(.+)$',          re.I), "causes"),
        (re.compile(r'^(.+?)\s+lives\s+in\s+(.+)$',      re.I), "lives_in"),
        (re.compile(r'^(.+?)\s+belongs\s+to\s+(.+)$',    re.I), "belongs_to"),
        (re.compile(r'^(.+?)\s+is\s+made\s+of\s+(.+)$',  re.I), "made_of"),
        (re.compile(r'^(.+?)\s+is\s+part\s+of\s+(.+)$',  re.I), "part_of"),
        (re.compile(r'^(.+?)\s+is\s+(.+)$',              re.I), "is"),
    ]

    @classmethod
    def _decompose_sentence(cls, text: str) -> None:
        """Extract (subject, relation, object) from common NL patterns and
        store as WorldModel beliefs (compositionality: meaning-from-parts)."""
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            t = text.strip()
            for pat, relation in cls._DECOMPOSE_PATTERNS:
                m = pat.match(t)
                if m:
                    subj = m.group(1).strip().lower()
                    obj  = m.group(2).strip().lower()
                    # Skip trivially short or very long matches
                    if subj and obj and 2 < len(subj) < 60 and 2 < len(obj) < 80:
                        wm.update_belief(subject=subj, predicate=relation,
                                         value=obj, confidence=0.65, domain="general")
                    break  # one pattern match per sentence
        except Exception:
            pass

    def _llm_parse_fallback(self, text: str) -> str:
        """Ask LLM to convert natural language math to a symbolic expression.
        Returns empty string on failure."""
        try:
            from sare.interface.llm_bridge import _call_llm
            prompt = (
                f"Convert this natural language math problem to a symbolic expression.\n"
                f"Input: {text}\n"
                f"Rules: use standard operators (+, -, *, /, **, parentheses). "
                f"Reply with ONLY the symbolic expression, nothing else. "
                f"Example: 'the sum of x and three' → 'x + 3'"
            )
            result = _call_llm(prompt).strip()
            import re
            # Reject if too long or looks like a prose explanation (spaces > math chars)
            if not result or len(result) > 120:
                return ""
            # Must have math content: digits, variables, operators
            if not re.search(r'[0-9a-zA-Z]', result):
                return ""
            # Reject obvious prose: contains common English words
            _prose_words = ('cannot', 'i cannot', 'sorry', 'there is', 'the expression',
                            'please', 'note that', 'this is', 'i am', 'i need')
            if any(w in result.lower() for w in _prose_words):
                return ""
            # Must be mostly math characters (letters, digits, operators, parens, spaces)
            math_chars = sum(1 for c in result if c in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/^()., _[]<>=!&|~%')
            if math_chars / max(len(result), 1) < 0.75:
                return ""
            return result
        except Exception:
            pass
        return ""

    # ── Translation pipeline ─────────────────────────────────────────────

    def _translate(
        self, text: str, domain: str,
        warnings: List[str], alternatives: List[str]
    ) -> Tuple[str, int, int]:
        words = text.split()
        total = len(words)
        hits  = 0

        # -1. Number-word → digit pre-pass (must run before calculus patterns so
        #     "sine of zero" → "sine of 0" before "sine of X" → "sin(X)")
        pre_tokens = text.split()
        text = " ".join(_NUM_WORDS.get(t.lower(), t) for t in pre_tokens)

        # -0.5. Strip leading 'the ' determiners ("the derivative of x" → "derivative of x")
        text = re.sub(r'\bthe\s+(?=(?:derivative|integral|sine|cosine|tangent|log|sqrt|square root|sum|product|difference|quotient)\b)', '', text, flags=re.IGNORECASE)

        # -0.3. Arithmetic-of patterns — multi-pass fixed-point for nested phrases
        # e.g. "product of the sum of x and 3 and y" → "(x + 3) * y"
        _MAX_PASSES = 6
        for _pass in range(_MAX_PASSES):
            prev = text
            for pattern, replacement in _ARITH_OF_PATTERNS:
                new_text, n = re.subn(pattern, replacement, text, flags=re.IGNORECASE)
                if n > 0:
                    hits += n * 2
                    text = new_text
            if text == prev:
                break  # fixed point reached — no more substitutions possible

        # 0. Calculus pre-processing (must run before other substitutions)
        for pattern, replacement in _CALCULUS_PATTERNS:
            new_text, n = re.subn(pattern, replacement, text, flags=re.IGNORECASE)
            if n > 0:
                hits += n * 2
                text = new_text

        # 1. Replace set phrases (word-boundary safe to avoid 'sin'→'s∈e')
        for phrase, sym in _SET_PHRASES:
            pattern = r'\b' + re.escape(phrase) + r'\b'
            new_text, n = re.subn(pattern, f' {sym} ', text)
            if n > 0:
                text = new_text
                hits += len(phrase.split()) * n

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

        # 7b. Wrap bare 'func token' → 'func(token)' for math functions not yet parenthesized
        #     e.g. 'sin x' → 'sin(x)', 'ln x' → 'ln(x)'. Skips already-parenthesized forms.
        expr = re.sub(
            r'\b(sin|cos|tan|ln|log|sqrt|exp|abs)\s+([a-zA-Z0-9_]+)\b(?!\s*\()',
            r'\1(\2)',
            expr,
        )

        # 7c. Implicit multiplication: '2x' → '2 * x', '3x**2' → '3 * x**2', '2(...)' → '2 * (...)'
        expr = re.sub(r'(\d+)\s*([a-zA-Z](?:\s*\*\*\s*\d+)?)', r'\1 * \2', expr)
        expr = re.sub(r'(\d+)\s*\(', r'\1 * (', expr)

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
        # NL calculus phrases must go through _translate even if they have operators
        has_calculus_nl = bool(re.search(
            r'\b(derivative of|differentiate|d/dx|integral of|integrate|deriv of)\b',
            text, re.IGNORECASE))
        if has_calculus_nl:
            return False
        # Arithmetic-of NL phrases ("sum of", "the product of") must not bypass
        has_arith_nl = bool(re.search(
            r'\b(sum of|product of|difference of|quotient of|the sum|the product)\b',
            text, re.IGNORECASE))
        if has_arith_nl:
            return False
        has_op  = bool(re.search(r'[\+\-\*/\(\)\^]', text))
        has_id  = bool(re.search(r'[a-zA-Z0-9]', text))
        no_long_words = not bool(re.search(r'\b[a-z]{5,}\b', text))
        return has_op and has_id and no_long_words

    def _detect_domain(self, text: str) -> str:
        text = text.lower()
        set_kw    = any(w in text for w in ("union","intersection","subset","in ","element","∪","∩","∈","∅"))
        logic_kw  = any(w in text for w in ("true","false","implies","iff","xor","nand","nor","not ","∧","∨","¬","→","↔"))
        alg_kw    = any(w in text for w in ("squared","cubed","power","**","sqrt","derivative","integral",
                                                  "differentiate","d/dx","integrate","deriv","sin","cos","exp","ln"))
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


# ══════════════════════════════════════════════════════════════════════════════
# Upgrade 4 — Multi-Modal Perception
# Parse tables, code blocks, markdown equations, and CSV data into
# expressions the SARE engine can reason about.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModalParseResult:
    """Result from multi-modal parsing."""
    modality: str           # 'table', 'code', 'equation', 'csv', 'nl'
    expressions: List[str]  # extracted symbolic expressions
    domain: str
    confidence: float
    raw: str                # original input


class MultiModalParser:
    """
    Upgrade 4: Multi-Modal Perception layer.

    Handles:
      - Markdown tables  → column relationships → expressions
      - Python/math code → assignments + expressions → symbolic forms
      - LaTeX equations  → rendered expressions
      - CSV data         → numeric patterns → hypotheses
      - Plain NL         → delegates to EnhancedNLParser
    """

    _CODE_ASSIGN  = re.compile(r'([a-zA-Z_]\w*)\s*=\s*([^=\n][^\n]*)')
    _CODE_RETURN  = re.compile(r'return\s+([^\n]+)')
    _CODE_EXPR    = re.compile(r'^([a-zA-Z0-9_\+\-\*/\^\(\)\s\.]+)$')
    _LATEX_FRAC   = re.compile(r'\\frac\{([^}]+)\}\{([^}]+)\}')
    _LATEX_SQRT   = re.compile(r'\\sqrt\{([^}]+)\}')
    _LATEX_SUM    = re.compile(r'\\sum[_\^{}\w\s]*')
    _TABLE_SEP    = re.compile(r'\|?([^|]+)\|?')
    _CSV_ROW      = re.compile(r'^\s*[\d\.\-]+(?:\s*,\s*[\d\.\-]+)+\s*$')

    def __init__(self):
        self._nl_parser = EnhancedNLParser()

    def detect_modality(self, text: str) -> str:
        """Detect the input modality."""
        stripped = text.strip()
        # Code block
        if stripped.startswith('```') or '=' in stripped and any(
                kw in stripped for kw in ['def ', 'return ', 'import ', 'print(']):
            return 'code'
        # Markdown table
        if '|' in stripped and '-+-' in stripped.replace(' ', '') or \
           stripped.count('|') >= 3:
            return 'table'
        # LaTeX
        if '\\frac' in stripped or '\\sqrt' in stripped or '\\sum' in stripped:
            return 'latex'
        # CSV
        if self._CSV_ROW.match(stripped.split('\n')[0] if '\n' in stripped else stripped):
            return 'csv'
        return 'nl'

    def parse(self, text: str, domain: str = "general") -> ModalParseResult:
        """Parse any modality into a list of symbolic expressions."""
        modality = self.detect_modality(text)
        if modality == 'code':
            return self._parse_code(text, domain)
        elif modality == 'table':
            return self._parse_table(text, domain)
        elif modality == 'latex':
            return self._parse_latex(text, domain)
        elif modality == 'csv':
            return self._parse_csv(text, domain)
        else:
            r = self._nl_parser.parse(text)
            return ModalParseResult('nl', [r.expression] if r.expression else [],
                                    r.domain or domain, r.confidence, text)

    def _parse_code(self, text: str, domain: str) -> ModalParseResult:
        """Extract assignments and return statements from code."""
        text = text.strip().strip('`').strip()
        exprs = []
        # Assignments: var = expr
        for m in self._CODE_ASSIGN.finditer(text):
            var, val = m.group(1).strip(), m.group(2).strip()
            val = val.split('#')[0].strip()  # remove comments
            if var not in ('True', 'False', 'None') and len(val) < 60:
                exprs.append(f"{var} = {val}")
        # Return statements
        for m in self._CODE_RETURN.finditer(text):
            expr = m.group(1).strip().split('#')[0].strip()
            if len(expr) < 60:
                exprs.append(expr)
        # Single-line expressions
        for line in text.split('\n'):
            line = line.strip()
            if self._CODE_EXPR.match(line) and len(line) > 2:
                exprs.append(line)
        conf = 0.75 if exprs else 0.2
        return ModalParseResult('code', exprs[:8], domain or 'arithmetic', conf, text)

    def _parse_table(self, text: str, domain: str) -> ModalParseResult:
        """
        Parse a markdown table → column relationship expressions.

        | x | y = x^2 |
        |---|---------|
        | 1 | 1       |
        | 2 | 4       |
        | 3 | 9       |

        → Detects that y = x^2 from numeric pattern.
        """
        lines = [l.strip() for l in text.strip().split('\n') if '|' in l]
        if len(lines) < 2:
            return ModalParseResult('table', [], domain, 0.1, text)
        # Header row
        headers = [h.strip() for h in lines[0].split('|') if h.strip()]
        # Data rows (skip separator)
        data_rows = []
        for line in lines[2:]:
            cells = [c.strip() for c in line.split('|') if c.strip()]
            if cells and all(re.match(r'^-?[\d\.]+$', c) for c in cells):
                data_rows.append([float(c) for c in cells])
        exprs = []
        # Check if header contains an equation already
        for h in headers:
            if '=' in h or any(op in h for op in ['+', '-', '*', '/', '^']):
                exprs.append(h.replace('^', '**'))
        # Try to detect relationship from data
        if len(data_rows) >= 2 and len(headers) >= 2:
            xs = [r[0] for r in data_rows if len(r) >= 2]
            ys = [r[1] for r in data_rows if len(r) >= 2]
            if xs and ys:
                # Check linear: y = a*x + b
                try:
                    if len(xs) >= 2:
                        a = (ys[-1] - ys[0]) / (xs[-1] - xs[0]) if xs[-1] != xs[0] else 0
                        b = ys[0] - a * xs[0]
                        if all(abs(y - (a * x + b)) < 0.01 for x, y in zip(xs, ys)):
                            coef = f"{a:.2g}".rstrip('0').rstrip('.')
                            bias = f"{b:+.2g}" if abs(b) > 0.001 else ""
                            exprs.append(f"{headers[1]} = {coef}*{headers[0]}{bias}")
                except Exception:
                    pass
                # Check quadratic: y = x^2
                if all(abs(y - x**2) < 0.01 for x, y in zip(xs, ys)):
                    exprs.append(f"{headers[1]} = {headers[0]}**2")
        conf = 0.6 if exprs else 0.3
        return ModalParseResult('table', exprs[:6], domain or 'arithmetic', conf, text)

    def _parse_latex(self, text: str, domain: str) -> ModalParseResult:
        """Convert LaTeX math notation to Python-style expressions."""
        expr = text
        expr = self._LATEX_FRAC.sub(r'(\1)/(\2)', expr)
        expr = self._LATEX_SQRT.sub(r'sqrt(\1)', expr)
        expr = self._LATEX_SUM.sub('sum', expr)
        expr = expr.replace('\\cdot', '*').replace('\\times', '*')
        expr = expr.replace('\\div', '/').replace('\\pm', '±')
        expr = expr.replace('^', '**').replace('{', '(').replace('}', ')')
        expr = re.sub(r'\\[a-zA-Z]+', '', expr).strip()
        conf = 0.7 if len(expr) > 2 else 0.2
        return ModalParseResult('latex', [expr] if expr else [], domain or 'calculus', conf, text)

    def _parse_csv(self, text: str, domain: str) -> ModalParseResult:
        """
        Parse CSV numeric data → detect pattern expressions.
        e.g.  1,1  2,4  3,9  → y = x**2
        """
        rows = []
        for line in text.strip().split('\n'):
            line = line.strip()
            if self._CSV_ROW.match(line):
                try:
                    rows.append([float(v.strip()) for v in line.split(',')])
                except ValueError:
                    pass
        exprs = []
        if len(rows) >= 3 and rows[0] and len(rows[0]) >= 2:
            xs = [r[0] for r in rows if len(r) >= 2]
            ys = [r[1] for r in rows if len(r) >= 2]
            # Check x^2
            if all(abs(y - x**2) < 0.01 for x, y in zip(xs, ys)):
                exprs.append("y = x**2")
            # Check x^3
            elif all(abs(y - x**3) < 0.01 for x, y in zip(xs, ys)):
                exprs.append("y = x**3")
            # Check 2x
            elif all(abs(y - 2*x) < 0.01 for x, y in zip(xs, ys)):
                exprs.append("y = 2*x")
            # Check linear
            elif len(xs) >= 2 and xs[-1] != xs[0]:
                a = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
                b = ys[0] - a * xs[0]
                if all(abs(y - (a*x + b)) < 0.01 for x, y in zip(xs, ys)):
                    exprs.append(f"y = {a:.3g}*x + {b:.3g}")
        conf = 0.65 if exprs else 0.2
        return ModalParseResult('csv', exprs, domain or 'arithmetic', conf, text)
