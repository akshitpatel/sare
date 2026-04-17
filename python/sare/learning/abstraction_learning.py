from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from sare.meta.macro_registry import MacroSpec

log = logging.getLogger(__name__)

# Cache: hash → LLM-generated name, so we don't re-call for same pattern
_llm_name_cache: dict[str, str] = {}


def _llm_semantic_name(steps: Sequence[str]) -> str:
    """Ask LLM to give a human-readable name for a transform sequence macro."""
    key = "\x1f".join(steps)
    if key in _llm_name_cache:
        return _llm_name_cache[key]
    try:
        from sare.interface.llm_bridge import _call_llm
        steps_str = " → ".join(steps)
        prompt = (
            f"You are a math AI naming discovered transform sequences.\n"
            f"Transform sequence: {steps_str}\n"
            f"Give a short snake_case name (2-4 words, no 'macro' prefix) that describes "
            f"the mathematical operation this sequence performs. "
            f"Reply with ONLY the name, nothing else."
        )
        raw = _call_llm(prompt).strip().lower()
        # Sanitize: keep only alphanumeric + underscore, max 40 chars
        import re
        name = re.sub(r'[^a-z0-9_]', '_', raw)[:40].strip('_')
        if name and len(name) >= 3:
            _llm_name_cache[key] = name
            return name
    except Exception as exc:
        log.debug("LLM macro naming failed: %s", exc)
    return ""


@dataclass(frozen=True)
class TransformPattern:
    steps: tuple[str, ...]
    frequency: int


def mine_frequent_patterns(
    traces: Iterable[Sequence[str]],
    min_frequency: int = 2,
    min_length: int = 2,
    max_length: int = 4,
) -> list[TransformPattern]:
    if min_length < 1:
        raise ValueError("min_length must be >= 1")
    if max_length < min_length:
        raise ValueError("max_length must be >= min_length")
    if min_frequency < 1:
        return []

    counts: dict[tuple[str, ...], int] = defaultdict(int)

    for trace in traces:
        cleaned = [t for t in trace if isinstance(t, str) and t]
        if len(cleaned) < min_length:
            continue

        max_k = min(max_length, len(cleaned))
        for k in range(min_length, max_k + 1):
            seen: set[tuple[str, ...]] = set()
            end = len(cleaned) - k + 1
            for i in range(end):
                gram = tuple(cleaned[i : i + k])
                if gram in seen:
                    continue
                seen.add(gram)
                counts[gram] += 1

    patterns = [
        TransformPattern(steps=s, frequency=f)
        for s, f in counts.items()
        if f >= min_frequency
    ]
    patterns.sort(key=lambda p: (p.frequency, len(p.steps)), reverse=True)
    return patterns


def stable_macro_name(steps: Sequence[str]) -> str:
    # Stable across runs (unlike Python's hash()).
    joined = "\x1f".join(steps).encode("utf-8")
    digest = hashlib.sha1(joined).hexdigest()[:8]

    # Short human hint: first token of each step name.
    hint = "_".join((s.split("_")[0] if s else "x") for s in steps[:3])
    hint = hint[:32] if hint else "macro"
    return f"macro_{hint}_{digest}"


def propose_macros(
    patterns: list[TransformPattern],
    existing_steps: set[tuple[str, ...]],
    max_new: int = 5,
) -> list[MacroSpec]:
    proposed: list[MacroSpec] = []

    for p in patterns:
        if len(proposed) >= max_new:
            break
        if p.steps in existing_steps:
            continue
        if any(s.startswith("macro_") for s in p.steps):
            continue

        llm_name = _llm_semantic_name(p.steps)
        name = llm_name if llm_name else stable_macro_name(p.steps)
        proposed.append(
            MacroSpec(
                name=name,
                steps=list(p.steps),
                frequency=p.frequency,
                enabled=True,
            )
        )
        existing_steps.add(p.steps)

    return proposed