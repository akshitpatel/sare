"""
InternalGrammar — Private Symbolic Thought Language for SARE-HX
================================================================

Human mathematicians think in compressed internal notation, not English.
This module implements a self-growing typed grammar over invented symbols.

NOT token sequences — every thought is a ThoughtGraph (mini graph of Symbol IDs).

Grammar grows by:
  Abstraction:     frequent co-occurring symbols → new composite symbol
  Specialization:  symbol used differently in 2 domains → split into 2
  Invention:       symbol_creator.py invents terminal → grammar adds production rules

Symbol lifecycle:
  sid → assigned on creation
  use_count grows with every ThoughtGraph that references it
  stability grows when the symbol appears in solved (reinforced) steps
  Low stability + low use_count → candidate for pruning

Integration points:
  - symbol_creator.py: calls learn_new_symbol() after promotion
  - hippocampus.py:    calls discover_abstractions() in sleep cycle
  - experiment_runner.py: calls encode_solve_step() per transform applied
  - WorldModel:        stores schemas as ThoughtGraph objects (richer than plain text)
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_MEMORY = Path(__file__).resolve().parents[4] / "data" / "memory"


@dataclass
class Symbol:
    """A single symbol in the internal vocabulary."""
    sid:        str          # unique symbol ID
    meaning:    str          # human-readable meaning (for debugging/introspection)
    domain:     str          # primary domain this symbol was born in
    arity:      int          # how many arguments (0=atom, 1=unary, 2=binary)
    type_sig:   str          # e.g. "Graph→Graph", "Graph,Graph→Graph", "atom"
    use_count:  int   = 0
    stability:  float = 0.5  # [0,1] — higher = more stable in vocabulary
    is_terminal: bool = True  # terminal = leaf symbol; False = composite
    composite_of: List[str] = field(default_factory=list)  # SIDs that compose this

    def to_dict(self) -> dict:
        return {
            "sid": self.sid, "meaning": self.meaning, "domain": self.domain,
            "arity": self.arity, "type_sig": self.type_sig,
            "use_count": self.use_count, "stability": round(self.stability, 3),
            "is_terminal": self.is_terminal, "composite_of": self.composite_of,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Symbol":
        return cls(
            sid=d["sid"], meaning=d.get("meaning", ""), domain=d.get("domain", "general"),
            arity=int(d.get("arity", 0)), type_sig=d.get("type_sig", "atom"),
            use_count=int(d.get("use_count", 0)), stability=float(d.get("stability", 0.5)),
            is_terminal=bool(d.get("is_terminal", True)),
            composite_of=list(d.get("composite_of", [])),
        )


@dataclass
class ProductionRule:
    """Head → Body: a rewrite rule in the internal grammar."""
    rule_id:       str
    head:          str           # SID of the result symbol
    body:          List[str]     # SIDs that compose the head
    semantic_role: str           # "CAUSE", "PART_OF", "TRANSFORMS_TO", "EQUALS"
    domain:        str
    confidence:    float = 0.5
    use_count:     int   = 0

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id, "head": self.head, "body": self.body,
            "semantic_role": self.semantic_role, "domain": self.domain,
            "confidence": round(self.confidence, 3), "use_count": self.use_count,
        }


@dataclass
class ThoughtGraph:
    """
    A thought expressed as a mini-graph of Symbol IDs.
    The unit of internal cognition — NOT a text string.
    """
    thought_id: str
    nodes:      List[str]                     # SIDs
    edges:      List[Tuple[str, str, str]]    # (from_sid, to_sid, role)
    intent:     str = ""                      # "solve", "question", "hypothesis", "recall"
    domain:     str = "general"
    timestamp:  float = field(default_factory=time.time)
    solved:     bool = False

    def to_dict(self) -> dict:
        return {
            "thought_id": self.thought_id,
            "nodes": self.nodes,
            "edges": [(f, t, r) for f, t, r in self.edges],
            "intent": self.intent, "domain": self.domain,
            "timestamp": self.timestamp, "solved": self.solved,
        }

    def __hash__(self):
        return hash(self.thought_id)


class InternalGrammar:
    """
    Self-growing typed hypergraph grammar over invented symbols.

    The grammar grows as the system learns. Frequent co-occurring symbols
    get abstracted into composite symbols. Each transform applied during
    solving generates a ThoughtGraph logged to thought_history.
    """

    PERSIST_PATH_SYMBOLS = _MEMORY / "internal_grammar_symbols.json"
    PERSIST_PATH_RULES = _MEMORY / "internal_grammar_rules.json"
    MAX_THOUGHT_HISTORY = 2000
    MIN_COOCCURRENCE_FOR_ABSTRACTION = 5

    def __init__(self):
        self._symbols: Dict[str, Symbol] = {}         # sid → Symbol
        self._name_to_sid: Dict[str, str] = {}        # meaning → sid (fast lookup)
        self._rules: Dict[str, ProductionRule] = {}   # rule_id → ProductionRule
        self._thought_history: List[ThoughtGraph] = []
        # Co-occurrence counts for abstraction mining
        self._cooccurrence: Counter = Counter()
        self._load()
        # Seed bootstrap symbols for basic cognitive operations
        self._seed_bootstrap_symbols()

    def _seed_bootstrap_symbols(self):
        """Ensure a minimal vocabulary exists from boot."""
        seeds = [
            ("APPLY",    "general", 2, "Graph,Transform→Graph", "TRANSFORMS_TO"),
            ("SIMPLIFY", "general", 1, "Graph→Graph",          "TRANSFORMS_TO"),
            ("EQUAL",    "general", 2, "Graph,Graph→Bool",     "EQUALS"),
            ("CAUSE",    "general", 2, "Event,Event→Event",    "CAUSE"),
            ("PART_OF",  "general", 2, "X,X→Bool",             "PART_OF"),
        ]
        for meaning, domain, arity, type_sig, role in seeds:
            if meaning not in self._name_to_sid:
                self._register_symbol(meaning, domain, arity, type_sig, is_terminal=True)

    def _register_symbol(
        self,
        meaning: str,
        domain: str,
        arity: int,
        type_sig: str,
        is_terminal: bool = True,
        composite_of: Optional[List[str]] = None,
    ) -> Symbol:
        """Internal helper: create and register a new symbol."""
        sid = str(uuid.uuid4())[:8]
        sym = Symbol(
            sid=sid, meaning=meaning, domain=domain,
            arity=arity, type_sig=type_sig,
            is_terminal=is_terminal,
            composite_of=composite_of or [],
        )
        self._symbols[sid] = sym
        self._name_to_sid[meaning] = sid
        return sym

    def learn_new_symbol(self, invented_symbol) -> Optional[Symbol]:
        """
        Called by symbol_creator.py after a symbol is promoted.
        Adds it to the grammar vocabulary and creates production rules.

        invented_symbol: InventedSymbol (from neuro/symbol_creator.py)
        """
        name = getattr(invented_symbol, "name", None) or getattr(invented_symbol, "sym_name", "")
        if not name:
            return None
        if name in self._name_to_sid:
            # Already known — bump stability
            sid = self._name_to_sid[name]
            self._symbols[sid].stability = min(1.0, self._symbols[sid].stability + 0.1)
            return self._symbols[sid]

        domain = getattr(invented_symbol, "domain", "general")
        notation = getattr(invented_symbol, "notation", "")
        description = getattr(invented_symbol, "description", name)

        sym = self._register_symbol(
            meaning=name,
            domain=domain,
            arity=1,
            type_sig="Graph→Graph",
            is_terminal=True,
        )
        sym.stability = 0.6   # newly promoted starts with reasonable stability

        # Create a production rule: APPLY(sym, graph) → SIMPLIFY(graph)
        apply_sid = self._name_to_sid.get("APPLY")
        simplify_sid = self._name_to_sid.get("SIMPLIFY")
        if apply_sid and simplify_sid:
            rule = ProductionRule(
                rule_id=str(uuid.uuid4())[:8],
                head=simplify_sid,
                body=[apply_sid, sym.sid],
                semantic_role="TRANSFORMS_TO",
                domain=domain,
                confidence=0.7,
            )
            self._rules[rule.rule_id] = rule

        log.info("[InternalGrammar] Learned new symbol '%s' (sid=%s, domain=%s)", name, sym.sid, domain)
        self._save()
        return sym

    def encode_solve_step(self, transform, graph_before, graph_after) -> ThoughtGraph:
        """
        Encode a single transform application as a ThoughtGraph.
        Called by experiment_runner after each successful search step.
        """
        t_name = ""
        if transform is not None:
            try:
                t_name = transform.name() if callable(getattr(transform, "name", None)) else str(transform)
            except Exception:
                t_name = str(transform)

        # Get or create symbol for this transform
        if t_name and t_name not in self._name_to_sid:
            domain = getattr(graph_before, "domain", "general") if graph_before else "general"
            self._register_symbol(t_name, domain, 1, "Graph→Graph")
        t_sid = self._name_to_sid.get(t_name, "")

        apply_sid = self._name_to_sid.get("APPLY", "")
        simplify_sid = self._name_to_sid.get("SIMPLIFY", "")

        nodes = [sid for sid in [apply_sid, t_sid, simplify_sid] if sid]
        edges: List[Tuple[str, str, str]] = []
        if apply_sid and t_sid:
            edges.append((apply_sid, t_sid, "APPLIES"))
        if t_sid and simplify_sid:
            edges.append((t_sid, simplify_sid, "PRODUCES"))

        domain = getattr(graph_before, "domain", "general") if graph_before else "general"
        thought = ThoughtGraph(
            thought_id=str(uuid.uuid4())[:8],
            nodes=nodes,
            edges=edges,
            intent="solve",
            domain=domain,
        )

        # Update use counts
        for sid in nodes:
            if sid in self._symbols:
                self._symbols[sid].use_count += 1

        # Update co-occurrence
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                key = tuple(sorted([nodes[i], nodes[j]]))
                self._cooccurrence[key] += 1

        self._thought_history.append(thought)
        if len(self._thought_history) > self.MAX_THOUGHT_HISTORY:
            self._thought_history = self._thought_history[-self.MAX_THOUGHT_HISTORY:]

        return thought

    def encode_question(self, confusion_context: dict) -> ThoughtGraph:
        """Encode a confusion state as a ThoughtGraph (for teacher protocol)."""
        domain = confusion_context.get("domain", "general")
        stuck_expr = confusion_context.get("expression", "")
        question_sid = self._name_to_sid.get("QUESTION", "")
        if not question_sid:
            sym = self._register_symbol("QUESTION", "meta", 1, "Context→ThoughtGraph")
            question_sid = sym.sid

        nodes = [question_sid]
        thought = ThoughtGraph(
            thought_id=str(uuid.uuid4())[:8],
            nodes=nodes,
            edges=[],
            intent="question",
            domain=domain,
        )
        self._thought_history.append(thought)
        return thought

    def discover_abstractions(self, thought_history: Optional[List[ThoughtGraph]] = None) -> List[Symbol]:
        """
        Mine frequent subgraph patterns from thought history and promote
        them to new composite symbols. Called by hippocampus sleep cycle.

        Returns list of newly created abstract symbols.
        """
        history = thought_history or self._thought_history
        if len(history) < 20:
            return []

        new_symbols: List[Symbol] = []

        # Find frequently co-occurring symbol pairs
        frequent_pairs = [
            (pair, count) for pair, count in self._cooccurrence.most_common(20)
            if count >= self.MIN_COOCCURRENCE_FOR_ABSTRACTION
        ]

        for (sid_a, sid_b), count in frequent_pairs:
            sym_a = self._symbols.get(sid_a)
            sym_b = self._symbols.get(sid_b)
            if sym_a is None or sym_b is None:
                continue

            # Don't create composite of bootstrap symbols
            if sym_a.meaning in ("APPLY", "SIMPLIFY", "EQUAL", "CAUSE", "PART_OF"):
                continue

            composite_meaning = f"{sym_a.meaning}_{sym_b.meaning}_COMPOSITE"
            if composite_meaning in self._name_to_sid:
                # Already exists — bump stability
                existing_sid = self._name_to_sid[composite_meaning]
                self._symbols[existing_sid].stability = min(
                    1.0, self._symbols[existing_sid].stability + 0.05
                )
                continue

            # Create composite symbol
            domain = sym_a.domain if sym_a.domain == sym_b.domain else "general"
            composite = self._register_symbol(
                meaning=composite_meaning,
                domain=domain,
                arity=max(sym_a.arity, sym_b.arity),
                type_sig="Graph→Graph",
                is_terminal=False,
                composite_of=[sid_a, sid_b],
            )
            composite.use_count = count
            composite.stability = min(0.8, count / 20.0)

            # Production rule: composite → sym_a + sym_b
            rule = ProductionRule(
                rule_id=str(uuid.uuid4())[:8],
                head=composite.sid,
                body=[sid_a, sid_b],
                semantic_role="PART_OF",
                domain=domain,
                confidence=min(0.9, count / 30.0),
                use_count=count,
            )
            self._rules[rule.rule_id] = rule
            new_symbols.append(composite)
            log.debug("[InternalGrammar] Abstracted '%s' (count=%d)", composite_meaning, count)

        if new_symbols:
            self._save()
        return new_symbols

    def compress(self, thought: ThoughtGraph) -> ThoughtGraph:
        """
        Rewrite a ThoughtGraph using the densest available vocabulary.
        Replaces known subpatterns with composite symbols.
        """
        if not thought.nodes or not self._rules:
            return thought

        # Try to find a production rule that covers a subset of thought.nodes
        node_set = set(thought.nodes)
        compressed_nodes = list(thought.nodes)

        for rule in self._rules.values():
            if rule.confidence < 0.5:
                continue
            body_set = set(rule.body)
            if body_set.issubset(node_set) and rule.head not in node_set:
                # Replace body with head
                compressed_nodes = [n for n in compressed_nodes if n not in body_set]
                compressed_nodes.append(rule.head)
                node_set = set(compressed_nodes)

        compressed = ThoughtGraph(
            thought_id=thought.thought_id,
            nodes=compressed_nodes,
            edges=thought.edges,
            intent=thought.intent,
            domain=thought.domain,
            timestamp=thought.timestamp,
            solved=thought.solved,
        )
        return compressed

    def get_vocabulary_size(self) -> int:
        return len(self._symbols)

    def get_status(self) -> dict:
        total = len(self._symbols)
        composite = sum(1 for s in self._symbols.values() if not s.is_terminal)
        return {
            "total_symbols": total,
            "terminal_symbols": total - composite,
            "composite_symbols": composite,
            "production_rules": len(self._rules),
            "thought_history_len": len(self._thought_history),
            "top_symbols": [
                s.to_dict() for s in sorted(
                    self._symbols.values(), key=lambda x: x.use_count, reverse=True
                )[:10]
            ],
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        try:
            _MEMORY.mkdir(parents=True, exist_ok=True)
            syms_data = [s.to_dict() for s in self._symbols.values()]
            tmp = self.PERSIST_PATH_SYMBOLS.with_suffix(".tmp")
            tmp.write_text(json.dumps(syms_data, indent=2), encoding="utf-8")
            tmp.replace(self.PERSIST_PATH_SYMBOLS)

            rules_data = [r.to_dict() for r in list(self._rules.values())[-500:]]
            tmp2 = self.PERSIST_PATH_RULES.with_suffix(".tmp")
            tmp2.write_text(json.dumps(rules_data, indent=2), encoding="utf-8")
            tmp2.replace(self.PERSIST_PATH_RULES)
        except OSError as e:
            log.debug("[InternalGrammar] Save error: %s", e)

    def _load(self):
        if self.PERSIST_PATH_SYMBOLS.exists():
            try:
                data = json.loads(self.PERSIST_PATH_SYMBOLS.read_text())
                for d in data:
                    sym = Symbol.from_dict(d)
                    self._symbols[sym.sid] = sym
                    self._name_to_sid[sym.meaning] = sym.sid
            except Exception as e:
                log.debug("[InternalGrammar] Load symbols error: %s", e)

        if self.PERSIST_PATH_RULES.exists():
            try:
                data = json.loads(self.PERSIST_PATH_RULES.read_text())
                for d in data:
                    rule = ProductionRule(**{k: v for k, v in d.items()
                                           if k in ProductionRule.__dataclass_fields__})
                    self._rules[rule.rule_id] = rule
            except Exception as e:
                log.debug("[InternalGrammar] Load rules error: %s", e)


# ── Singleton ─────────────────────────────────────────────────────────────────
_instance: Optional[InternalGrammar] = None


def get_internal_grammar() -> InternalGrammar:
    global _instance
    if _instance is None:
        _instance = InternalGrammar()
    return _instance
