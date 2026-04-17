"""
Universal Parser — Epic 10: Zero-Dependency NLP Comprehension
================================================================
Replaces the old BasicNLParser with a context-aware semantic parsing 
engine capable of handling ambiguous word problems without external 
dependencies like spaCy.

Features:
- Entity extraction ("Bob", "Alice", "apples")
- Intent classification (transfer, comparison, query)
- Relational mapping ("gives 1 to Alice" -> `Alice + 1`, `Bob - 1`)
- Compilation to Canonical AST expression
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

# ─── Ontology & Vocabulary ──────────────────────────────────────────────────

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}

_TRANSFER_VERBS_OUT = {"gives": "-", "loses": "-", "drops": "-", "spends": "-"}
_TRANSFER_VERBS_IN  = {"receives": "+", "gets": "+", "finds": "+", "earns": "+"}
_QUERY_MARKERS      = {"how many", "what is", "how much", "find"}

# ── Math NL patterns ────────────────────────────────────────────
_SIMPLIFY_KEYWORDS = {"simplify", "simplify:", "evaluate", "compute", "reduce", "calculate"}
_DERIVATIVE_KEYWORDS = {"derivative", "differentiate", "d/dx", "diff"}
_INTEGRAL_KEYWORDS = {"integral", "integrate", "antiderivative"}
_TRIG_MAP = {
    "sin": "sin", "sine": "sin",
    "cos": "cos", "cosine": "cos",
    "tan": "tan", "tangent": "tan",
    "log": "log", "logarithm": "log",
    "sqrt": "sqrt", "square root": "sqrt",
}
_FIND_KEYWORDS = {"find", "solve", "what is x", "what's x"}


def _try_parse_math_nl(text: str) -> Optional[str]:
    """
    P4.1: Attempt to extract a symbolic expression from natural language math input.
    Returns a parseable expression string or None.

    Handles:
      - "simplify x + 0"            → "x + 0"
      - "what is sin(0)?"           → "sin(0)"
      - "derivative of x^3"         → "derivative(x^3)"
      - "find x if x + 5 = 12"      → "x + 5 = 12"
      - "simplify (x + 0) * 1"      → "(x + 0) * 1"
    """
    import re
    t = text.strip().rstrip("?. \t")

    # Direct simplify/evaluate keyword at start
    lower = t.lower()
    for kw in _SIMPLIFY_KEYWORDS:
        if lower.startswith(kw):
            rest = t[len(kw):].strip().lstrip(":").strip()
            if rest:
                return rest

    # "derivative of X" → "derivative(X)"  [check BEFORE generic "what is X?"]
    for kw in _DERIVATIVE_KEYWORDS:
        m = re.match(rf"(?i){re.escape(kw)}\s+(?:of\s+)?(.+)", t)
        if m:
            expr = m.group(1).strip()
            return f"derivative({expr})"

    # "integral of X" → "integral(X)"
    for kw in _INTEGRAL_KEYWORDS:
        m = re.match(rf"(?i){re.escape(kw)}\s+(?:of\s+)?(.+)", t)
        if m:
            expr = m.group(1).strip()
            return f"integral({expr})"

    # "sin of 0" → "sin(0)", "cosine of 0" → "cos(0)"
    for nl, sym in _TRIG_MAP.items():
        m = re.match(rf"(?i){re.escape(nl)}\s+(?:of\s+)?(.+)", t)
        if m:
            arg = m.group(1).strip()
            return f"{sym}({arg})"

    # "what is X?" or "what's X?" — after trig/deriv so "what is derivative of X" works
    m = re.match(r"(?i)what(?:'s| is)\s+(.+)", t)
    if m:
        inner = m.group(1).strip()
        # Recursively process the extracted expression
        processed = _try_parse_math_nl(inner)
        return processed if processed else inner

    # "find x if X" or "solve X" → extract equation
    for kw in _FIND_KEYWORDS:
        m = re.match(rf"(?i){re.escape(kw)}\s+(?:x\s+)?(?:if|when|where|such that|:)?\s*(.+)", t)
        if m:
            expr = m.group(1).strip()
            if any(op in expr for op in ("=", "+", "-", "*", "/")):
                return expr

    # Fallback: if text looks like an expression already (contains operators)
    if re.search(r'[+\-*/^()=]', t) and not re.search(r'[A-Z]', t.split()[0] if t.split() else ""):
        return t

    return None

@dataclass
class EntityData:
    name: str
    quantity: int = 0
    history: List[str] = field(default_factory=list)

@dataclass
class ParseResult:
    raw_input: str
    expression: str
    domain: str
    confidence: float
    graph_ready: bool
    entities: Dict[str, EntityData]
    tokens: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_input":   self.raw_input,
            "expression":  self.expression,
            "domain":      self.domain,
            "confidence":  round(self.confidence, 3),
            "graph_ready": self.graph_ready,
            "tokens":      self.tokens,
            "warnings":    self.warnings,
            "entities":    {k: {"quantity": v.quantity, "history": v.history} for k, v in self.entities.items()}
        }

class UniversalParser:
    def __init__(self):
        self.entities: Dict[str, EntityData] = {}
        self.domain = "unknown"

    def _tokenize_sentences(self, text: str) -> List[str]:
        """Split text into sentences handling crude punctuation."""
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    def _extract_number(self, words: List[str], index: int) -> Optional[int]:
        """Attempt to extract a number at the current token index."""
        if index >= len(words): return None
        w = words[index].lower()
        if w in _NUMBER_WORDS: return _NUMBER_WORDS[w]
        # remove punctuation before checking isdigit
        clean_w = re.sub(r'[^\w\s]', '', w)
        if clean_w.isdigit(): return int(clean_w)
        return None

    def _extract_entities(self, words: List[str]) -> None:
        """First Pass: Identify proper nouns (capitalized) as actors/entities."""
        for i, w in enumerate(words):
            if w[0].isupper() and w.lower() not in _QUERY_MARKERS:
                clean_name = re.sub(r'[^\w\s]', '', w)
                
                # Exclude common sentence starters that get capitalized
                if clean_name.lower() in ["how", "what", "if", "then", "the", "a", "an"]:
                    continue
                    
                if clean_name and clean_name not in self.entities:
                    self.entities[clean_name] = EntityData(name=clean_name)

    def _parse_possession(self, words: List[str]):
        """Detect "Actor has X items" patterns."""
        for i, w in enumerate(words):
            if w.lower() in ("has", "starts with", "holds"):
                actor = self._find_actor_before(words, i)
                if actor:
                    num = self._extract_number(words, i + 1)
                    if num is not None:
                        self.entities[actor].quantity = num
                        self.entities[actor].history.append(f"init:{num}")

    def _parse_transfer(self, words: List[str]):
        """Detect "Actor gives/gets X [to ActorB]" patterns."""
        for i, w in enumerate(words):
            w_lower = w.lower()
            if w_lower in _TRANSFER_VERBS_OUT:
                actor_from = self._find_actor_before(words, i)
                num = self._extract_number(words, i + 1)
                actor_to = self._find_actor_after(words, i, "to")
                
                if actor_from and num is not None:
                    self.entities[actor_from].quantity -= num
                    self.entities[actor_from].history.append(f"{w_lower}:-{num}")
                    if actor_to:
                        self.entities[actor_to].quantity += num
                        self.entities[actor_to].history.append(f"receive_from_{actor_from}:+{num}")
                        
            elif w_lower in _TRANSFER_VERBS_IN:
                actor_to = self._find_actor_before(words, i)
                num = self._extract_number(words, i + 1)
                
                if actor_to and num is not None:
                    self.entities[actor_to].quantity += num
                    self.entities[actor_to].history.append(f"{w_lower}:+{num}")


    def _find_target_query(self, words: List[str]) -> Optional[str]:
        """Detect 'How many does Actor have left?' """
        query_text = " ".join([w.lower() for w in words])
        for q in _QUERY_MARKERS:
            if q in query_text:
                # Look for an actor mentioned in the same sentence
                for w in words:
                    clean_name = re.sub(r'[^\w\s]', '', w)
                    # Don't match the query markers themselves if they got added
                    if clean_name in self.entities and clean_name.lower() not in ["how", "what", "find", "does", "do", "have"]:
                        return clean_name
        return None

    def _find_actor_before(self, words: List[str], index: int) -> Optional[str]:
        for i in range(index - 1, -1, -1):
            clean_name = re.sub(r'[^\w\s]', '', words[i])
            if clean_name in self.entities:
                return clean_name
        return None

    def _find_actor_after(self, words: List[str], index: int, prep: str = "") -> Optional[str]:
        found_prep = (prep == "")
        for i in range(index + 1, len(words)):
            if not found_prep and words[i].lower() == prep:
                found_prep = True
                continue
            if found_prep:
                clean_name = re.sub(r'[^\w\s]', '', words[i])
                if clean_name in self.entities:
                    return clean_name
        return None

    def _build_expression(self, target_actor: str) -> str:
        """Compile the history of the target actor into a mathematical expression."""
        if target_actor not in self.entities:
            return ""
        
        hist = self.entities[target_actor].history
        if not hist: return "0"
        
        # history looks like: ["init:3", "gives:-1"]
        expr_parts = []
        for h in hist:
            val = h.split(":")[1]
            if not expr_parts:
                # First item (init) should just be the number without leading operator unless it's truly negative
                expr_parts.append(val)
            else:
                if val.startswith("+") or val.startswith("-"):
                    expr_parts.append(val)
                else:
                    expr_parts.append(f"+{val}")
                
        # "3" "-1" -> "3-1"
        raw_expr = "".join(expr_parts)
        # Add spaces around operators for the AST engine
        raw_expr = raw_expr.replace("+", " + ").replace("-", " - ")
        return raw_expr.strip()

    def parse_multi(self, text: str) -> list:
        """
        Split text into sentences and parse each independently.

        Splits on: '. ' (period+space), '; ' (semicolon+space),
                   ' then ' / ' and then ', 'after that', 'next',
                   newlines.
        Returns a list of ParseResult objects.
        """
        parts = re.split(
            r'(?:\.\s+|\;\s+|\s+then\s+|\s+and\s+then\s+|\s+after\s+that\s+|\s+next\s+|\n+)',
            text,
            flags=re.IGNORECASE
        )
        results = []
        for part in parts:
            part = part.strip().rstrip('.')
            if len(part) > 2:  # skip empty/trivial fragments
                try:
                    result = self.parse(part)
                    results.append(result)
                except Exception:
                    pass
        return results if results else [self.parse(text)]  # fallback: parse whole

    def parse(self, text: str) -> ParseResult:
        """
        Main entry point for Epic 10 Universal Parsing.
        Translates a multi-sentence word problem into a canonical expression.
        P4.1: First tries math NL patterns (simplify/derivative/trig/etc.)
        """
        # P4.1: Try math NL first before word-problem parsing
        math_expr = _try_parse_math_nl(text)
        if math_expr:
            domain = "arithmetic"
            low = math_expr.lower()
            if any(kw in low for kw in ("derivative", "integral", "diff")):
                domain = "calculus"
            elif any(kw in low for kw in ("sin", "cos", "tan", "log", "sqrt")):
                domain = "trigonometry"
            elif "and" in low or "or" in low or "not" in low:
                domain = "logic"
            elif "=" in low:
                domain = "algebra"
            return ParseResult(
                raw_input=text,
                expression=math_expr,
                domain=domain,
                confidence=0.85,
                graph_ready=True,
                entities={},
                tokens=text.split(),
            )

        self.entities = {}
        sentences = self._tokenize_sentences(text)
        
        # Pass 1: Global Entity Extraction
        for sent in sentences:
            words = sent.split()
            self._extract_entities(words)

        # Pass 2: Relational Mapping (Possession & Transfer)
        for sent in sentences:
            words = sent.split()
            self._parse_possession(words)
            self._parse_transfer(words)

        # Pass 3: Query Intent Resolution
        target_actor = None
        for sent in sentences:
            if "?" in sent or any(q in sent.lower() for q in _QUERY_MARKERS):
                target_actor = self._find_target_query(sent.split())
                
        # Fallback to the first entity if no explicit query target is found
        if not target_actor and self.entities:
            target_actor = list(self.entities.keys())[0]

        from sare.interface.nl_parser_v2 import EnhancedNLParser
        fallback_parser = EnhancedNLParser()
        
        # Pass 4: Compilation to Canonical AST
        expression = ""
        confidence = 0.0
        graph_ready = False
        
        if target_actor and target_actor in self.entities and self.entities[target_actor].history:
            expression = self._build_expression(target_actor)
            if expression and expression != "0":
                confidence = 0.9  # High confidence if we cleanly tracked history
                graph_ready = True
                self.domain = "arithmetic"

        if not graph_ready:
            # Fallback to enhanced syntax parser for direct equations
            return fallback_parser.parse(text)
            
        return ParseResult(
            raw_input=text,
            expression=expression,
            domain=self.domain,
            confidence=confidence,
            graph_ready=graph_ready,
            entities=self.entities,
        )

# Simple standalone test
if __name__ == "__main__":
    parser = UniversalParser()
    res = parser.parse("Bob has 3 apples. Bob gives 1 to Alice. How many does Bob have left?")
    print(f"Expression for Bob: {res.expression}") # Expected: 3 - 1
    print(f"Alice's quantity: {res.entities.get('Alice').quantity if 'Alice' in res.entities else 0}") # Expected: 1
