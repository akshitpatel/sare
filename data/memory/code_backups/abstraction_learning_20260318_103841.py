from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from sare.meta.macro_registry import MacroSpec


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
    counts: dict[tuple[str, ...], int] = defaultdict(int)

    for trace in traces:
        trace = [t for t in trace if isinstance(t, str) and t]
        if len(trace) < min_length:
            continue

        for k in range(min_length, max_length + 1):
            if len(trace) < k:
                continue
            seen: set[tuple[str, ...]] = set()
            for i in range(0, len(trace) - k + 1):
                gram = tuple(trace[i : i + k])
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

        proposed.append(
            MacroSpec(
                name=stable_macro_name(p.steps),
                steps=list(p.steps),
                frequency=p.frequency,
                enabled=True,
            )
        )
        existing_steps.add(p.steps)

    return proposed

