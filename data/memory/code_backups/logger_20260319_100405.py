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
import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class SolveLog:
    problem_id: str
    timestamp: str = ""
    initial_energy: float = 0.0
    final_energy: float = 0.0
    search_depth: int = 0
    compute_time_seconds: float = 0.0
    total_expansions: int = 0
    solve_success: bool = False
    transform_sequence: List[str] = field(default_factory=list)
    abstractions_used: List[str] = field(default_factory=list)
    modules_activated: List[str] = field(default_factory=list)
    rule_name: Optional[str] = None
    domain: str = "general"
    energy_breakdown: Dict[str, float] = field(default_factory=dict)
    energy_trajectory: List[float] = field(default_factory=list)
    node_types: List[str] = field(default_factory=list)
    adjacency: List[Any] = field(default_factory=list)
    budget_exhausted: bool = False


class SareLogger:
    def __init__(self, output_path: str = "data/memory/solve_log.jsonl"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: SolveLog):
        entry.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(self.output_path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def read_all(self) -> List[SolveLog]:
        entries = []
        if not self.output_path.exists():
            return entries
        with open(self.output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    allowed = {k: v for k, v in data.items()
                               if k in SolveLog.__dataclass_fields__}
                    entries.append(SolveLog(**allowed))
                except Exception:
                    pass
        return entries

    def summary(self) -> dict:
        entries = self.read_all()
        n = len(entries)
        if n == 0:
            return {"total": 0}
        return {
            "total": n,
            "avg_initial_energy": sum(e.initial_energy for e in entries) / n,
            "avg_final_energy": sum(e.final_energy for e in entries) / n,
            "avg_search_depth": sum(e.search_depth for e in entries) / n,
            "avg_compute_time": sum(e.compute_time_seconds for e in entries) / n,
            "avg_expansions": sum(e.total_expansions for e in entries) / n,
            "solve_rate": sum(1 for e in entries if e.solve_success) / n,
        }
