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
        if not self.output_path.exists():
            return {"total": 0}

        total = 0
        sum_initial_energy = 0.0
        sum_final_energy = 0.0
        sum_search_depth = 0.0
        sum_compute_time = 0.0
        sum_expansions = 0.0
        solve_success_count = 0

        allowed_fields = set(SolveLog.__dataclass_fields__.keys())

        with open(self.output_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        continue

                    # Only read fields needed for summary; avoid materializing all entries.
                    initial_energy = data.get("initial_energy", 0.0)
                    final_energy = data.get("final_energy", 0.0)
                    search_depth = data.get("search_depth", 0)
                    compute_time_seconds = data.get("compute_time_seconds", 0.0)
                    total_expansions = data.get("total_expansions", 0)
                    solve_success = data.get("solve_success", False)

                    # Basic type coercion guards
                    try:
                        initial_energy = float(initial_energy)
                    except Exception:
                        initial_energy = 0.0
                    try:
                        final_energy = float(final_energy)
                    except Exception:
                        final_energy = 0.0
                    try:
                        search_depth = int(search_depth)
                    except Exception:
                        search_depth = 0
                    try:
                        compute_time_seconds = float(compute_time_seconds)
                    except Exception:
                        compute_time_seconds = 0.0
                    try:
                        total_expansions = int(total_expansions)
                    except Exception:
                        total_expansions = 0
                    try:
                        solve_success = bool(solve_success)
                    except Exception:
                        solve_success = False

                    total += 1
                    sum_initial_energy += initial_energy
                    sum_final_energy += final_energy
                    sum_search_depth += search_depth
                    sum_compute_time += compute_time_seconds
                    sum_expansions += total_expansions
                    if solve_success:
                        solve_success_count += 1

                except Exception:
                    continue

        if total == 0:
            return {"total": 0}

        return {
            "total": total,
            "avg_initial_energy": sum_initial_energy / total,
            "avg_final_energy": sum_final_energy / total,
            "avg_search_depth": sum_search_depth / total,
            "avg_compute_time": sum_compute_time / total,
            "avg_expansions": sum_expansions / total,
            "solve_rate": solve_success_count / total,
        }