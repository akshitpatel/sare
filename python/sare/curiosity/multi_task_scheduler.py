"""Multi-task batch scheduler — allocates problems across task types by ZPD."""
import json
import logging
from pathlib import Path
from typing import Dict, List

log = logging.getLogger(__name__)

_STATE_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "multi_task_scheduler.json"

_DEFAULT_DOMAINS = ["algebra", "logic", "planning", "language", "coding", "science"]


class MultiTaskScheduler:
    def __init__(self):
        self._domains: List[str] = list(_DEFAULT_DOMAINS)
        self._domain_scores: Dict[str, float] = {}
        self._total_batches: int = 0

    def update_domain_score(self, domain: str, score: float) -> None:
        """Update ZPD score for a domain (higher = more learning opportunity)."""
        self._domain_scores[domain] = max(0.0, float(score))
        if domain not in self._domains:
            self._domains.append(domain)

    def allocate_batch(self, total: int = 20) -> Dict[str, int]:
        """Divide budget across domains proportional to ZPD score."""
        if not self._domain_scores:
            n = max(1, len(self._domains))
            alloc = {d: max(1, total // n) for d in self._domains}
        else:
            total_score = sum(self._domain_scores.values()) or 1.0
            alloc = {d: max(1, int(total * s / total_score))
                     for d, s in self._domain_scores.items()}
        self._total_batches += 1
        log.debug("[MultiTaskScheduler] batch=%d alloc=%s", self._total_batches, alloc)
        return alloc

    def get_status(self) -> dict:
        try:
            if _STATE_PATH.exists():
                return json.loads(_STATE_PATH.read_text())
        except Exception:
            pass
        return {
            "allocations": self.allocate_batch(),
            "task_types": list(self._domains),
            "total_batches": self._total_batches,
            "status": "active" if self._domain_scores else "idle",
        }
