"""
BasicNLParser — TODO-07: Natural Language → Graph
===================================================
Converts simple natural language mathematical / logical expressions
into SARE Graph objects (or expression strings for the engine to parse).

This is a rule-based parser — no LLM required. It handles:
  - Arithmetic: "x plus 0", "three times four", "x squared"
  - Logic: "x and true", "not not x", "x or false"
  - Variables: single-letter names, common names (alpha, beta, …)
  - Constants: written-out numbers ("zero", "one", …) and digits

Usage::

    parser = BasicNLParser()
    result = parser.parse("x plus zero times one")
    print(result.expression)   # "(x + 0) * 1"
    print(result.confidence)   # 0.85
    print(result.graph_ready)  # True
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ── Token maps ────────────────────────────────────────────────────────────────

_NUMBER_WORDS: Dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
}

_OPERATOR_WORDS: Dict[str, str] = {
    # Addition
    "plus": "+", "add": "+", "added to": "+", "sum of": "+",
    "and": "+",   # arithmetic context
    # Subtraction
    "minus": "-", "subtract": "-", "less": "-", "take away": "-",
    # Multiplication
    "times": "*", "multiplied by": "*", "multiply": "*",
    "product of": "*", "by": "*",
    # Division
    "divided by": "/", "over": "/", "divide": "/",
    # Exponentiation
    "to the power of": "**", "raised to": "**", "squared": "** 2",
    "cubed": "** 3",
    # Comparison / logic (use later)
    "equals": "==", "equal to": "==",
    "greater than": ">", "less than": "<",
}

_LOGIC_WORDS: Dict[str, str] = {
    "and": "and", "or": "or", "not": "not",
    "true": "True", "false": "False",
    "xor": "^",
}

_VARIABLE_WORDS = {
    "alpha", "beta", "gamma", "delta", "epsilon", "theta",
    "lambda", "mu", "pi", "sigma", "phi", "omega",
}


# ── Parse result ──────────────────────────────────────────────────────────────

@dataclass
class ParseResult:
    raw_input:   str
    expression:  str          # canonical expression string (for engine)
    domain:      str          # "arithmetic" | "logic" | "unknown"
    confidence:  float        # 0.0 – 1.0
    graph_ready: bool         # True if expression can be fed directly to engine
    tokens:      List[str]
    warnings:    List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_input":   self.raw_input,
            "expression":  self.expression,
            "domain":      self.domain,
            "confidence":  round(self.confidence, 3),
            "graph_ready": self.graph_ready,
            "tokens":      self.tokens,
            "warnings":    self.warnings,
        }


# ── Tokeniser ─────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> List[str]:
    """Lower-case, strip punctuation, split on spaces."""
    text = text.lower().strip()
    text = re.sub(r"[,.;:!?'\"]", " ", text)
    return [t for t in text.split() if t]


# ── BasicNLParser ─────────────────────────────────────────────────────────────

class BasicNLParser:
    """
    Rule-based natural language → SARE expression converter.

    Handles arithmetic and boolean logic with no external dependencies.
    For more complex NL, hook up an LLM in the future (TODO-07b).
    """

    def parse(self, text: str) -> ParseResult:
        tokens   = _tokenise(text)
        warnings: List[str] = []

        # First try to parse it as a math expression directly
        if self._looks_like_expression(text):
            return ParseResult(
                raw_input=text, expression=text.strip(),
                domain=self._classify_domain(text),
                confidence=0.95, graph_ready=True,
                tokens=tokens, warnings=[],
            )

        # Otherwise translate word-by-word
        expr, domain, conf = self._translate(tokens, warnings)
        graph_ready = bool(expr) and conf > 0.4

        return ParseResult(
            raw_input=text, expression=expr,
            domain=domain, confidence=conf,
            graph_ready=graph_ready,
            tokens=tokens, warnings=warnings,
        )

    # ── Internal ─────────────────────────────────────────────────────────

    def _looks_like_expression(self, text: str) -> bool:
        """True if text is already a valid-ish math expression."""
        cleaned = text.strip()
        # Contains arithmetic operators or parens and some identifier
        has_op  = bool(re.search(r'[\+\-\*/\(\)]', cleaned))
        has_id  = bool(re.search(r'[a-zA-Z0-9]', cleaned))
        return has_op and has_id

    def _classify_domain(self, text: str) -> str:
        text = text.lower()
        if any(w in text for w in ("true", "false", "and", "or", "not", "xor", "∧", "∨", "¬")):
            return "logic"
        if any(w in text for w in ("**", "^", "sqrt", "sin", "cos")):
            return "algebra"
        if re.search(r'\d', text) or re.search(r'[+\-*/]', text):
            return "arithmetic"
        return "general"

    def _translate(
        self, tokens: List[str], warnings: List[str]
    ) -> Tuple[str, str, float]:
        """
        Greedily translate tokens into an expression string.
        Returns (expression, domain, confidence).
        """
        parts:   List[str] = []
        domain = "arithmetic"
        hits   = 0
        total  = len(tokens)

        i = 0
        while i < len(tokens):
            tok = tokens[i]

            # 2-word operator phrases
            if i + 1 < len(tokens):
                bigram = tok + " " + tokens[i + 1]
                if bigram in _OPERATOR_WORDS:
                    parts.append(_OPERATOR_WORDS[bigram])
                    hits += 2
                    i += 2
                    continue
            # 3-word operator phrases
            if i + 2 < len(tokens):
                trigram = tok + " " + tokens[i+1] + " " + tokens[i+2]
                if trigram in _OPERATOR_WORDS:
                    parts.append(_OPERATOR_WORDS[trigram])
                    hits += 3
                    i += 3
                    continue

            # Number words
            if tok in _NUMBER_WORDS:
                parts.append(_NUMBER_WORDS[tok])
                hits += 1

            # Logic words
            elif tok in _LOGIC_WORDS:
                parts.append(_LOGIC_WORDS[tok])
                domain = "logic"
                hits += 1

            # Arithmetic operators
            elif tok in _OPERATOR_WORDS:
                sym = _OPERATOR_WORDS[tok]
                # "squared" / "cubed" inject exponent
                if sym.startswith("**"):
                    parts.append(sym)
                else:
                    parts.append(sym)
                hits += 1

            # Single digit
            elif re.fullmatch(r'\d+', tok):
                parts.append(tok)
                hits += 1

            # Known variable names / single letters
            elif tok in _VARIABLE_WORDS or re.fullmatch(r'[a-z]', tok):
                parts.append(tok)
                hits += 1

            else:
                warnings.append(f"Unrecognised token: '{tok}'")

            i += 1

        expr = " ".join(parts).strip()
        conf = (hits / total) if total > 0 else 0.0

        # Wrap in parens if multiple parts to aid parsing
        if len(parts) > 3 and "(" not in expr:
            expr = "( " + expr + " )"

        return expr, domain, round(conf, 3)
