from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MACROS_PATH = REPO_ROOT / "configs" / "abstractions.json"


@dataclass(frozen=True)
class MacroSpec:
    name: str
    steps: list[str]
    created_at: str = ""
    frequency: int = 0
    enabled: bool = True


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_macro_file(path: Path = DEFAULT_MACROS_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "macros": []}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "macros": []}

    if not isinstance(data, dict):
        return {"version": 1, "macros": []}

    version = int(data.get("version", 1))
    macros = data.get("macros", [])
    if not isinstance(macros, list):
        macros = []

    return {"version": version, "macros": macros}


def save_macro_file(data: dict[str, Any], path: Path = DEFAULT_MACROS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def list_macros(path: Path = DEFAULT_MACROS_PATH) -> list[MacroSpec]:
    data = load_macro_file(path)
    specs: list[MacroSpec] = []

    for item in data.get("macros", []):
        if not isinstance(item, dict):
            continue
        name = item.get("name", "")
        steps = item.get("steps", [])
        if not name or not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
            continue
        specs.append(
            MacroSpec(
                name=name,
                steps=list(steps),
                created_at=str(item.get("created_at", "")),
                frequency=int(item.get("frequency", 0) or 0),
                enabled=bool(item.get("enabled", True)),
            )
        )

    return specs


def macro_steps_set(macros: Iterable[MacroSpec]) -> set[tuple[str, ...]]:
    return {tuple(m.steps) for m in macros}


def upsert_macros(macros: list[MacroSpec], path: Path = DEFAULT_MACROS_PATH) -> dict[str, Any]:
    data = load_macro_file(path)
    existing = list_macros(path)

    by_name: dict[str, MacroSpec] = {m.name: m for m in existing}
    by_steps: set[tuple[str, ...]] = macro_steps_set(existing)

    for spec in macros:
        steps_key = tuple(spec.steps)
        if steps_key in by_steps:
            continue

        created_at = spec.created_at or _now_iso_utc()
        by_name[spec.name] = MacroSpec(
            name=spec.name,
            steps=list(spec.steps),
            created_at=created_at,
            frequency=int(spec.frequency),
            enabled=bool(spec.enabled),
        )
        by_steps.add(steps_key)

    merged = list(by_name.values())
    merged.sort(key=lambda m: (not m.enabled, m.name))

    data["macros"] = [asdict(m) for m in merged]
    save_macro_file(data, path)
    return data

