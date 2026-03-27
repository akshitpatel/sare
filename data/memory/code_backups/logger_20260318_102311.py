"""
SARE-HX Structured Logger

Outputs JSONL solve logs per specification:
- Problem ID
- Initial/final energy
- Search depth
- Transform sequence
- Compute time
- Abstractions used
- Modules activated
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class SolveLog:
    """Structured log for a single solve operation."""
    problem_id: str = ""
    timestamp: str = ""
    initial_energy: float = 0.0
    final_energy: float = 0.0
    energy_breakdown: dict = field(default_factory=dict)
    search_depth: int = 0
    transform_sequence: list = field(default_factory=list)
    compute_time_seconds: float = 0.0
    total_expansions: int = 0
    energy_trajectory: list = field(default_factory=list)
    abstractions_used: list = field(default_factory=list)
    modules_activated: list = field(default_factory=list)
    node_types: list = field(default_factory=list)
    adjacency: list = field(default_factory=list)
    budget_exhausted: bool = False
    solve_success: bool = False


class SareLogger:
    """JSONL structured logger for solve operations."""

    def __init__(self, output_path: str = "sare_solves.jsonl"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: SolveLog) -> None:
        """Append a solve log entry to the JSONL file."""
        if not entry.timestamp:
            entry.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        with open(self.output_path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def read_all(self) -> list[SolveLog]:
        """Read all log entries from the file."""
        entries = []
        if not self.output_path.exists():
            return entries

        with open(self.output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    # Be tolerant to new fields over time.
                    allowed = {k: v for k, v in data.items() if k in SolveLog.__dataclass_fields__}
                    entries.append(SolveLog(**allowed))
        return entries

    def summary(self) -> dict:
        """Generate summary statistics from all log entries."""
        entries = self.read_all()
        if not entries:
            return {"count": 0}

        return {
            "count": len(entries),
            "avg_initial_energy": sum(e.initial_energy for e in entries) / len(entries),
            "avg_final_energy": sum(e.final_energy for e in entries) / len(entries),
            "avg_search_depth": sum(e.search_depth for e in entries) / len(entries),
            "avg_compute_time": sum(e.compute_time_seconds for e in entries) / len(entries),
            "avg_expansions": sum(e.total_expansions for e in entries) / len(entries),
            "success_rate": sum(1 for e in entries if e.solve_success) / len(entries),
        }
