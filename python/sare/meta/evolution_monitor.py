"""
EvolutionMonitor — tracks the rate of self-improvement across all AGI subsystems.

Aggregates signals from self_improver, experiment_runner, causal_induction,
analogy_transfer, and mlx_value_net to produce a single "evolution velocity"
score and per-subsystem health metrics.

Exposed via GET /api/agi/evolution
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_MEMORY = Path(__file__).resolve().parents[3] / "data" / "memory"


@dataclass
class SubsystemHealth:
    name: str
    status: str            # "healthy" | "degraded" | "stalled"
    velocity: float        # improvement events per hour
    last_event: float      # unix timestamp
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "velocity": round(self.velocity, 3),
            "last_event_ago_min": round((time.time() - self.last_event) / 60, 1),
            "details": self.details,
        }


class EvolutionMonitor:
    """
    Aggregates all self-improvement signals into a unified evolution velocity score.

    velocity_score (0-1):
        0.0 = completely stalled
        0.5 = normal improvement rate
        1.0 = rapid self-improvement across all subsystems
    """

    def __init__(self):
        self._cache: Optional[dict] = None
        self._cache_ts: float = 0.0
        self._cache_ttl: float = 30.0  # refresh every 30s

    def get_report(self) -> dict:
        now = time.time()
        if self._cache and (now - self._cache_ts) < self._cache_ttl:
            return self._cache

        subsystems = [
            self._check_self_improver(),
            self._check_learning_loop(),
            self._check_rule_induction(),
            self._check_transfer_learning(),
            self._check_value_net(),
            self._check_world_model(),
        ]

        healthy = sum(1 for s in subsystems if s.status == "healthy")
        degraded = sum(1 for s in subsystems if s.status == "degraded")
        stalled = sum(1 for s in subsystems if s.status == "stalled")

        # Overall velocity: weighted average of subsystem velocities (capped at 1.0)
        weights = [2.0, 3.0, 2.0, 1.5, 1.0, 1.5]  # learning loop most important
        total_w = sum(weights)
        max_velocities = [5.0, 100.0, 10.0, 5.0, 500.0, 20.0]  # per subsystem scale
        velocity_score = sum(
            w * min(1.0, s.velocity / mv)
            for s, w, mv in zip(subsystems, weights, max_velocities)
        ) / total_w

        # Bottlenecks: subsystems with status != healthy
        bottlenecks = [s.name for s in subsystems if s.status != "healthy"]

        report = {
            "velocity_score": round(velocity_score, 3),
            "healthy_subsystems": healthy,
            "degraded_subsystems": degraded,
            "stalled_subsystems": stalled,
            "bottlenecks": bottlenecks,
            "subsystems": [s.to_dict() for s in subsystems],
            "generated_at": now,
            "recommendation": self._recommend(velocity_score, bottlenecks),
        }

        self._cache = report
        self._cache_ts = now
        return report

    # ── Subsystem Checks ──────────────────────────────────────────────────────

    def _check_self_improver(self) -> SubsystemHealth:
        """Self-improver: patches applied per hour."""
        try:
            # Primary source: self_improvements.json (always written by SelfImprover)
            si_path = _MEMORY / "self_improvements.json"
            patches_all: list = []
            last_ts = 0.0
            if si_path.exists():
                try:
                    si_data = json.loads(si_path.read_text())
                    patches_all = si_data.get("patches", [])
                    last_ts = si_data.get("last_updated", 0)
                except Exception as e:
                    log.warning("[evolution_monitor] Failed to read self_improvements.json: %s", e)

            # Supplement with si_stats.json for prescreen/rollback counters
            stats_path = _MEMORY / "si_stats.json"
            stats_data = {}
            if stats_path.exists():
                try:
                    stats_data = json.loads(stats_path.read_text())
                except Exception as e:
                    log.warning("[evolution_monitor] Failed to read si_stats.json: %s", e)
            prescreened_rej = stats_data.get("prescreened_rejected", 0)

            # Also try in-process singleton (most accurate, avoids file races)
            try:
                from sare.meta.self_improver import get_self_improver
                st = get_self_improver().get_status()
                debates = st.get("total_debates", 0)
                patches_applied = st.get("patches_applied", 0)
                last_ts = max(last_ts, st.get("last_active", 0) or 0)
            except Exception as e:
                log.warning("[evolution_monitor] self_improver health check failed: %s", e)
                debates = len(patches_all)
                patches_applied = sum(1 for p in patches_all if not p.get("rolled_back", False))

            # Patches applied in last hour
            cutoff = time.time() - 3600
            patches_1h = sum(
                1 for p in patches_all
                if not p.get("rolled_back", False)
                and p.get("applied_at", 0) > cutoff
            )
            patch_rate = float(patches_1h)

            # If no recent patches, estimate from total rate
            if patch_rate == 0 and patches_applied > 0 and last_ts > 0:
                age_h = (time.time() - last_ts) / 3600
                if age_h < 24:
                    patch_rate = patches_applied / max(1, age_h)

            reject_rate = prescreened_rej / max(1, debates) if debates else 0.0
            status = "healthy" if patch_rate >= 1 else ("degraded" if patches_applied > 0 else "stalled")

            return SubsystemHealth(
                "self_improver", status, patch_rate, last_ts or time.time(),
                {"patches_1h": patches_1h, "reject_rate": round(reject_rate, 2),
                 "total_debates": debates, "total_patches": patches_applied}
            )
        except Exception as e:
            return SubsystemHealth("self_improver", "stalled", 0.0, time.time(), {"error": str(e)})

    def _check_learning_loop(self) -> SubsystemHealth:
        """Learning loop: problems solved per hour."""
        try:
            sm_path = _MEMORY / "self_model.json"
            if not sm_path.exists():
                return SubsystemHealth("learning_loop", "stalled", 0.0, 0.0,
                                       {"reason": "no self_model.json"})

            data = json.loads(sm_path.read_text())
            domains = data.get("domains", {})
            total_recent = sum(
                d.get("recent_attempts", 0) for d in domains.values()
            )
            total_solved = sum(
                d.get("recent_successes", 0) for d in domains.values()
            )

            # Find most recently updated domain
            last_ts = max(
                (d.get("last_updated", 0) for d in domains.values()),
                default=0
            )

            # Estimate: recent window is ~last 30min by convention
            velocity = total_recent * 2.0  # rough problems/hour

            weak_domains = [
                k for k, v in domains.items()
                if v.get("recent_attempts", 0) >= 5
                and (v.get("recent_successes", 0) / v.get("recent_attempts", 1)) < 0.7
            ]

            avg_rate = total_solved / max(1, total_recent)
            status = "healthy" if avg_rate >= 0.7 else ("degraded" if avg_rate >= 0.4 else "stalled")

            return SubsystemHealth(
                "learning_loop", status, velocity, last_ts or time.time(),
                {"solve_rate": round(avg_rate, 2), "recent_attempts": total_recent,
                 "weak_domains": weak_domains, "total_solves": data.get("total_solves", 0)}
            )
        except Exception as e:
            return SubsystemHealth("learning_loop", "stalled", 0.0, 0.0, {"error": str(e)})

    def _check_rule_induction(self) -> SubsystemHealth:
        """Causal induction: rules promoted per hour."""
        try:
            pr_path = _MEMORY / "promoted_rules.json"
            if not pr_path.exists():
                return SubsystemHealth("rule_induction", "stalled", 0.0, 0.0,
                                       {"reason": "no promoted_rules.json"})

            data = json.loads(pr_path.read_text())
            # Format: {"promoted_rules": {"rule_name": count, ...}, "pattern_counts": {...}}
            # OR: list of rule dicts
            if isinstance(data, list):
                rules = data
            else:
                pr = data.get("promoted_rules", {})
                if isinstance(pr, dict):
                    rules = [{"name": k, "count": v} for k, v in pr.items()]
                else:
                    rules = list(pr) if pr else []

            total_rules = len(rules)
            # We can't easily get per-hour promotions from count data; use total as signal
            velocity = float(min(total_rules, 10))  # cap at 10 for display

            last_ts = time.time() - 60  # assume recent if file exists
            status = "healthy" if total_rules >= 5 else ("degraded" if total_rules >= 1 else "stalled")

            return SubsystemHealth(
                "rule_induction", status, velocity, last_ts,
                {"total_rules": total_rules,
                 "rule_names": [r.get("name", r) if isinstance(r, dict) else str(r) for r in rules[:5]]}
            )
        except Exception as e:
            return SubsystemHealth("rule_induction", "stalled", 0.0, 0.0, {"error": str(e)})

    def _check_transfer_learning(self) -> SubsystemHealth:
        """Transfer learning: cross-domain rule applications per hour."""
        try:
            lt_path = _MEMORY / "learned_transfers.json"
            if not lt_path.exists():
                return SubsystemHealth("transfer_learning", "stalled", 0.0, 0.0,
                                       {"reason": "no learned_transfers.json"})

            data = json.loads(lt_path.read_text())
            # transfers can be: list of dicts, or dict with various keys
            if isinstance(data, list):
                raw = data
            else:
                # Try multiple key names used by TransferEngine
                raw = (data.get("transfer_history")
                       or data.get("transfers")
                       or data.get("hypotheses")
                       or {})
                # Also count stats as proxy
                stats = data.get("stats", {})
                _verified = stats.get("hypotheses_verified", 0)
                _observations = stats.get("observations", 0)
            if isinstance(raw, dict):
                transfers = list(raw.values())
            else:
                transfers = list(raw)

            cutoff = time.time() - 3600
            recent = [t for t in transfers if isinstance(t, dict) and t.get("timestamp", 0) > cutoff]
            velocity = float(len(recent))
            ts_vals = [t.get("timestamp", 0) for t in transfers if isinstance(t, dict)]
            last_ts = max(ts_vals) if ts_vals else 0

            # When transfers have no timestamps, use total count or stats as proxy
            total = len(transfers)
            if velocity == 0 and total > 0:
                velocity = min(float(total), 5.0)  # show as degraded-but-present
            # Fall back to stats-based velocity if no transfer records
            if velocity == 0 and isinstance(data, dict):
                _stats = data.get("stats", {})
                _obs = _stats.get("observations", 0)
                _ver = _stats.get("hypotheses_verified", 0)
                if _obs > 0:
                    velocity = min(float(_ver) + _obs / 1000.0, 10.0)
                    total = max(total, _ver)

            # Success rate: prefer explicit field; fall back to successes/failures ratio
            success_rates = []
            for t in transfers:
                if not isinstance(t, dict):
                    continue
                if "success_rate" in t:
                    success_rates.append(t["success_rate"])
                elif "successes" in t or "failures" in t:
                    s = t.get("successes", 0)
                    f = t.get("failures", 0)
                    denom = s + f
                    if denom > 0:
                        success_rates.append(s / denom)
            avg_success = sum(success_rates) / len(success_rates) if success_rates else 0.5

            # If we have any observations (even hypotheses), count as degraded (not stalled)
            _obs_total = total
            if isinstance(data, dict):
                _s = data.get("stats", {})
                _obs_total = max(total, _s.get("observations", 0) // 100)
            status = "healthy" if avg_success >= 0.6 and total >= 5 else (
                "degraded" if _obs_total > 0 else "stalled"
            )

            return SubsystemHealth(
                "transfer_learning", status, float(velocity), last_ts or time.time(),
                {"transfers_1h": velocity, "avg_success_rate": round(avg_success, 2),
                 "total_transfers": total}
            )
        except Exception as e:
            return SubsystemHealth("transfer_learning", "stalled", 0.0, 0.0, {"error": str(e)})

    def _check_value_net(self) -> SubsystemHealth:
        """MLX value network: training updates per hour."""
        try:
            # Try stats file first (written by daemon process every 10 training steps)
            stats_path = _MEMORY / "mlx_value_net_stats.json"
            if stats_path.exists():
                try:
                    file_stats = json.loads(stats_path.read_text())
                    updates = file_stats.get("updates", 0)
                    avg_loss = file_stats.get("avg_loss", 0.0)
                    predictions = file_stats.get("predictions", 0)
                    status = "healthy" if updates > 100 and avg_loss < 0.1 else (
                        "degraded" if updates > 0 else "stalled"
                    )
                    return SubsystemHealth(
                        "value_net_mlx", status, min(float(updates), 500.0), time.time(),
                        {"total_updates": updates, "avg_loss": round(avg_loss, 5),
                         "total_predictions": predictions, "source": "file"}
                    )
                except Exception:
                    pass

            # Try in-process singleton (web server's own instance; useful for device info)
            try:
                from sare.heuristics.mlx_value_net import get_value_net
                vn = get_value_net()
                st = vn.get_stats()
                device = st.get("device", "cpu")
                buf_size = st.get("buffer_size", 0)
                updates = st.get("total_updates", 0)
                avg_loss = st.get("avg_loss", 1.0)
                buf_needed = st.get("buffer_needed", 32)
                if updates > 100 and avg_loss < 0.1:
                    status = "healthy"
                elif updates > 0:
                    status = "degraded"
                elif buf_needed > 0:
                    status = "stalled"
                else:
                    status = "degraded"
                return SubsystemHealth(
                    "value_net_mlx", status, min(float(updates), 500.0), time.time(),
                    {"total_updates": updates, "avg_loss": round(avg_loss, 5),
                     "buffer_size": buf_size, "device": device}
                )
            except Exception:
                pass

            return SubsystemHealth("value_net_mlx", "stalled", 0.0, time.time(),
                                   {"reason": "warming up — needs ≥32 episodes"})
        except Exception as e:
            return SubsystemHealth("value_net_mlx", "stalled", 0.0, time.time(), {"error": str(e)})

    def _check_world_model(self) -> SubsystemHealth:
        """World model: belief updates and hypothesis generation per hour."""
        try:
            # Prefer v2 (active daemon file with full causal links) over v3 (older)
            wm_path = _MEMORY / "world_model_v2.json"
            if not wm_path.exists():
                wm_path = _MEMORY / "world_model_v3.json"
            if not wm_path.exists():
                wm_path = _MEMORY / "world_model.json"
            if not wm_path.exists():
                return SubsystemHealth("world_model", "stalled", 0.0, 0.0,
                                       {"reason": "no world_model file"})

            data = json.loads(wm_path.read_text())
            # v2 uses "facts" dict; v3/legacy uses "beliefs" or "propositions"
            raw_beliefs = (data.get("beliefs")
                           or data.get("facts")
                           or data.get("propositions")
                           or [])
            # v2 "facts" is a dict of domain→[list]; count total items not domains
            if isinstance(raw_beliefs, dict):
                beliefs = sum(len(v) if isinstance(v, list) else 1
                              for v in raw_beliefs.values())
            elif isinstance(raw_beliefs, list):
                beliefs = len(raw_beliefs)
            else:
                beliefs = 0
            raw_links = data.get("causal_links", [])
            causal_links = len(raw_links) if isinstance(raw_links, (list, dict)) else 0
            hypotheses_path = _MEMORY / "world_hypotheses.json"
            hypotheses = 0
            last_hyp_ts = 0.0
            if hypotheses_path.exists():
                hyp_data = json.loads(hypotheses_path.read_text())
                hyp_list = hyp_data if isinstance(hyp_data, list) else hyp_data.get("hypotheses", [])
                hypotheses = len(hyp_list)
                cutoff = time.time() - 3600
                recent_hyp = [h for h in hyp_list if h.get("timestamp", 0) > cutoff]
                velocity = float(len(recent_hyp))
                last_hyp_ts = max((h.get("timestamp", 0) for h in hyp_list), default=0)
            else:
                velocity = 0.0

            # Healthy if we have substantial beliefs OR causal links
            status = ("healthy" if (beliefs > 50 or causal_links > 100)
                      else ("degraded" if (beliefs > 10 or causal_links > 10) else "stalled"))
            # Use causal_links as velocity proxy when no recent hypotheses
            if velocity == 0 and causal_links > 0:
                velocity = min(float(causal_links) / 1000.0, 5.0)

            return SubsystemHealth(
                "world_model", status, velocity, last_hyp_ts or time.time(),
                {"total_beliefs": beliefs, "causal_links": causal_links,
                 "hypotheses": hypotheses}
            )
        except Exception as e:
            return SubsystemHealth("world_model", "stalled", 0.0, 0.0, {"error": str(e)})

    # ── Recommendations ───────────────────────────────────────────────────────

    def _recommend(self, velocity: float, bottlenecks: List[str]) -> str:
        if not bottlenecks:
            if velocity >= 0.7:
                return "System evolving well. Consider harder problem domains."
            return "All subsystems healthy. Increase batch size for faster learning."

        if "learning_loop" in bottlenecks:
            return "Learning loop stalled — check daemon, curriculum generator, and solve rate."
        if "rule_induction" in bottlenecks:
            return "No new rules being promoted — check CausalInduction thresholds or test case coverage."
        if "self_improver" in bottlenecks:
            return "Self-improver stalled — check LLM API keys, prescreen rejection rate, critic scores."
        if "value_net_mlx" in bottlenecks:
            return "MLX value net not training — needs more episodes (>=32) or check MLX install."
        return f"Bottlenecks: {', '.join(bottlenecks)}. Check individual subsystems."


# ── Singleton ──────────────────────────────────────────────────────────────────
_monitor: Optional[EvolutionMonitor] = None


def get_evolution_monitor() -> EvolutionMonitor:
    global _monitor
    if _monitor is None:
        _monitor = EvolutionMonitor()
    return _monitor
