"""
Autobiographical Memory — SARE-HX's learning history as a narrative.

Tracks the system's learning history like a student's academic journal:
significant events, emotional valence, domain mastery trajectory, and
the emerging narrative of "who I am as a learner".

Integrates with:
  - ExperimentRunner (records solve events)
  - CurriculumGenerator (influences next domain selection)
  - IdentityManager (formative experiences shape traits)
  - HomeostaticSystem (emotional state feeds drive levels)

Data stored in: data/memory/autobiographical.json
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

MEMORY_DIR = Path(__file__).resolve().parents[3] / "data" / "memory"
AUTO_PATH = MEMORY_DIR / "autobiographical.json"

# Importance thresholds
_HIGH_IMPORTANCE = 0.7
_STUCK_THRESHOLD = 500   # episodes without improvement = "stuck"


@dataclass
class LearningEpisode:
    """One memorable event in the system's learning history."""
    episode_id: str
    timestamp: float
    event_type: str      # "rule_discovered", "domain_mastered", "analogy_found",
                         # "human_taught", "stuck_period", "breakthrough", "rule_applied",
                         # "social_interaction", "milestone"
    domain: str
    description: str
    importance: float    # 0.0–1.0
    related_rules: List[str]
    emotional_valence: float  # -1.0 (frustrating) to 1.0 (exciting)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "domain": self.domain,
            "description": self.description,
            "importance": round(self.importance, 4),
            "related_rules": self.related_rules,
            "emotional_valence": round(self.emotional_valence, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LearningEpisode":
        return cls(
            episode_id=d.get("episode_id", str(uuid.uuid4())[:8]),
            timestamp=d.get("timestamp", time.time()),
            event_type=d.get("event_type", "milestone"),
            domain=d.get("domain", "general"),
            description=d.get("description", ""),
            importance=d.get("importance", 0.5),
            related_rules=d.get("related_rules", []),
            emotional_valence=d.get("emotional_valence", 0.0),
        )


# Emotional valence by event type
_VALENCE_MAP = {
    "domain_mastered":   0.9,
    "breakthrough":      0.95,
    "rule_discovered":   0.7,
    "analogy_found":     0.7,
    "human_taught":      0.6,
    "rule_applied":     0.3,
    "social_interaction": 0.5,
    "milestone":         0.5,
    "stuck_period":     -0.4,
}

# Importance by event type (base)
_IMPORTANCE_MAP = {
    "domain_mastered":   0.9,
    "breakthrough":      0.9,
    "rule_discovered":   0.7,
    "analogy_found":     0.6,
    "human_taught":      0.6,
    "milestone":         0.5,
    "social_interaction": 0.5,
    "rule_applied":      0.2,
    "stuck_period":     0.4,
}


class AutobiographicalMemory:
    """
    SARE-HX's long-term narrative memory of its own learning journey.

    Answers questions like:
      - What were my most important learning moments?
      - Which domains have I struggled with?
      - What was my emotional journey through learning?
      - What should I explore next based on my history?
    """

    DEFAULT_PATH = AUTO_PATH

    def __init__(self, persist_path: Optional[Path] = None):
        self._path = Path(persist_path or self.DEFAULT_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._episodes: List[LearningEpisode] = []
        self._domain_trajectory: Dict[str, List[Tuple[float, float]]] = {}
        # domain → [(timestamp, mastery_level), ...]

        self._dirty: bool = False
        self.load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load memory state from JSON if present."""
        if not self._path.exists():
            self._episodes = []
            self._domain_trajectory = {}
            self._dirty = False
            return

        try:
            with self._path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            raw_episodes = data.get("episodes", [])
            self._episodes = [LearningEpisode.from_dict(ep) for ep in raw_episodes if isinstance(ep, dict)]

            raw_traj = data.get("domain_trajectory", {})
            traj: Dict[str, List[Tuple[float, float]]] = {}
            if isinstance(raw_traj, dict):
                for domain, points in raw_traj.items():
                    if not isinstance(domain, str) or not isinstance(points, list):
                        continue
                    parsed_points: List[Tuple[float, float]] = []
                    for p in points:
                        if not isinstance(p, (list, tuple)) or len(p) != 2:
                            continue
                        ts, ml = p
                        try:
                            parsed_points.append((float(ts), float(ml)))
                        except Exception:
                            continue
                    traj[domain] = parsed_points
            self._domain_trajectory = traj

            self._dirty = False
        except Exception as e:
            log.exception("AutobiographicalMemory: failed to load %s: %s", self._path, e)
            self._episodes = []
            self._domain_trajectory = {}
            self._dirty = False

    def save(self) -> None:
        """Persist memory state to JSON."""
        if not self._dirty and self._path.exists():
            return

        payload = {
            "version": 1,
            "updated_at": time.time(),
            "episodes": [ep.to_dict() for ep in self._episodes[-5000:]],
            "domain_trajectory": {
                domain: [[ts, ml] for (ts, ml) in points[-2000:]]
                for domain, points in self._domain_trajectory.items()
            },
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_name(self._path.name + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            tmp_path.replace(self._path)
            self._dirty = False
        except Exception as e:
            log.exception("AutobiographicalMemory: failed to save %s: %s", self._path, e)
            self._dirty = True

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(
        self,
        event_type: str,
        domain: str,
        description: str,
        related_rules: Optional[List[str]] = None,
        importance: Optional[float] = None,
        emotional_valence: Optional[float] = None,
    ) -> None:
        """Record a new learning event."""
        if importance is None:
            importance = _IMPORTANCE_MAP.get(event_type, 0.4)
        if emotional_valence is None:
            emotional_valence = _VALENCE_MAP.get(event_type, 0.0)

        try:
            importance = float(importance)
        except Exception:
            importance = _IMPORTANCE_MAP.get(event_type, 0.4)
        importance = max(0.0, min(1.0, importance))

        try:
            emotional_valence = float(emotional_valence)
        except Exception:
            emotional_valence = _VALENCE_MAP.get(event_type, 0.0)
        emotional_valence = max(-1.0, min(1.0, emotional_valence))

        ep = LearningEpisode(
            episode_id=str(uuid.uuid4())[:8],
            timestamp=time.time(),
            event_type=event_type,
            domain=domain,
            description=description,
            importance=importance,
            related_rules=related_rules or [],
            emotional_valence=emotional_valence,
        )
        self._episodes.append(ep)

        # Auto-save every 20 episodes
        if len(self._episodes) % 20 == 0:
            self._dirty = True
            self.save()
        else:
            self._dirty = True

        # Update identity traits
        try:
            from sare.memory.identity import get_identity_manager
            im = get_identity_manager()
            success = emotional_valence > 0
            im.update_from_behavior(event_type, domain, success)
        except Exception:
            pass

        log.debug("AutobiographicalMemory: recorded %s in %s", event_type, domain)

    def record_mastery(self, domain: str, mastery_level: float) -> None:
        """Record a mastery checkpoint for domain trajectory tracking."""
        try:
            ml = float(mastery_level)
        except Exception:
            ml = 0.0
        ml = max(0.0, min(1.0, ml))

        if domain not in self._domain_trajectory:
            self._domain_trajectory[domain] = []
        self._domain_trajectory[domain].append((time.time(), ml))

        self._dirty = True
        if (len(self._domain_trajectory[domain]) % 25) == 0:
            self.save()

    # ── Queries / Narrative generation ───────────────────────────────────────

    def get_narrative(self, last_n: int = 10) -> str:
        """Generate a human-readable story of recent learning."""
        if not self._episodes:
            return "My learning journal is empty. I have not yet recorded any memorable events."

        n = max(1, int(last_n))
        recent = self._episodes[-n:]
        lines: List[str] = []
        lines.append("Recent learning journey:")

        for ep in recent:
            t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ep.timestamp))
            imp = "high" if ep.importance >= _HIGH_IMPORTANCE else "medium/low"
            emo = (
                "exciting" if ep.emotional_valence >= 0.35
                else "neutral"
                if abs(ep.emotional_valence) < 0.2
                else "frustrating"
            )
            rule_bits = f" (rules: {', '.join(ep.related_rules[:3])}{'...' if len(ep.related_rules) > 3 else ''})" if ep.related_rules else ""
            lines.append(
                f"- [{t}] ({ep.domain}) {ep.event_type}: {ep.description} "
                f"[importance={imp}, emotion={emo}]{rule_bits}"
            )

        return "\n".join(lines)

    def retrieve_similar(self, embedding: List[float], top_k: int = 3) -> List["LearningEpisode"]:
        """
        Retrieve episodes similar to a given embedding.

        Note: This system stores no explicit embedding per episode in the current schema.
        We approximate similarity by matching keywords from description/domain and event type.
        """
        if not self._episodes:
            return []

        k = max(1, int(top_k))

        # Best-effort heuristic: use embedding statistics to bias scoring.
        # (Keeps interface stable without requiring stored vectors.)
        emb_sum = float(sum(embedding)) if embedding else 0.0
        emb_sign = 1.0 if emb_sum >= 0 else -1.0

        # Candidate scoring: importance + emotional alignment + recency
        now = time.time()
        scored: List[Tuple[float, LearningEpisode]] = []
        for ep in self._episodes:
            age = max(1.0, now - ep.timestamp)
            recency = 1.0 / age  # smaller age => larger recency
            emo_align = 1.0 - abs(ep.emotional_valence - (0.35 * emb_sign))  # closer => larger
            emo_align = max(0.0, min(1.0, emo_align))
            score = (0.55 * ep.importance) + (0.25 * emo_align) + (0.20 * recency)
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:k]]

    def get_domain_trajectory(self, domain: str) -> List[Tuple[float, float]]:
        """Get (timestamp, mastery_level) points for a domain."""
        return list(self._domain_trajectory.get(domain, []))

    def get_most_important_episodes(self, top_k: int = 5) -> List[LearningEpisode]:
        """Return the most important episodes overall."""
        if not self._episodes:
            return []
        k = max(1, int(top_k))
        eps = sorted(self._episodes, key=lambda e: e.importance, reverse=True)
        return eps[:k]

    def get_hard_episodes(self, n: int = 10) -> List[dict]:
        """
        Return the last N episodes that represent failures or low-confidence events.

        An episode is considered "hard" if:
          - event_type is "stuck_period", OR
          - emotional_valence < -0.2 (frustrating outcome), OR
          - importance < 0.4 (low-value event, proxy for low confidence)

        Returns a list of dicts with keys: expression, domain, episode_id, importance.
        The expression is extracted from the episode description heuristically.
        """
        try:
            n = max(1, int(n))
            hard: List[LearningEpisode] = []
            for ep in reversed(self._episodes):
                if (
                    ep.event_type == "stuck_period"
                    or ep.emotional_valence < -0.2
                    or ep.importance < 0.4
                ):
                    hard.append(ep)
                    if len(hard) >= n:
                        break

            results: List[dict] = []
            for ep in hard:
                # Best-effort: extract an expression from the description.
                # Descriptions often contain the expression after "on " or "for ".
                desc = ep.description or ""
                expression = ""
                for prefix in ("on ", "for ", "expression: ", "solving ", "problem: "):
                    idx = desc.lower().find(prefix)
                    if idx != -1:
                        candidate = desc[idx + len(prefix):].split()[0].strip(".,;:\"'")
                        if candidate:
                            expression = candidate
                            break
                if not expression:
                    expression = desc[:60].strip() or ep.episode_id

                results.append({
                    "expression": expression,
                    "domain": ep.domain or "general",
                    "episode_id": ep.episode_id,
                    "importance": ep.importance,
                })
            return results
        except Exception:
            return []

    def get_domains_with_low_progress(self, min_points: int = 3) -> List[str]:
        """
        Identify domains with limited mastery improvement over recent checkpoints.
        Heuristic: if last mastery - first mastery < 0.1 or too few points.
        """
        weak: List[str] = []
        for domain, points in self._domain_trajectory.items():
            if len(points) < max(1, int(min_points)):
                weak.append(domain)
                continue
            pts = points[-min(10, len(points)):]
            first_ml = pts[0][1]
            last_ml = pts[-1][1]
            if (last_ml - first_ml) < 0.1:
                weak.append(domain)
        return weak

    def get_emotional_arc(self, domain: Optional[str] = None, last_n: int = 30) -> str:
        """Summarize emotional valence trend over time."""
        if not self._episodes:
            return "No emotional arc yet."

        filt = [ep for ep in self._episodes if domain is None or ep.domain == domain]
        if not filt:
            return f"No emotional history recorded for domain '{domain}'."

        n = max(5, int(last_n))
        seq = filt[-n:]
        if not seq:
            return "No emotional history available."

        avg = sum(ep.emotional_valence for ep in seq) / max(1, len(seq))
        trend = seq[-1].emotional_valence - seq[0].emotional_valence

        if trend > 0.2:
            trend_s = "rising toward excitement"
        elif trend < -0.2:
            trend_s = "falling toward frustration"
        else:
            trend_s = "remaining fairly steady"

        return (
            f"Emotional arc{' for ' + domain if domain else ''}: "
            f"avg valence={avg:.2f}, overall tone is {trend_s}."
        )

    # ── Housekeeping ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._episodes)

    def close(self) -> None:
        """Best-effort flush to disk."""
        try:
            if self._dirty:
                self.save()
        except Exception:
            pass


def get_autobiographical_memory() -> AutobiographicalMemory:
    """
    Singleton accessor with lightweight per-process caching.

    Persisted state is stored in data/memory/autobiographical.json.
    """
    # Avoid importing across potential circulars; keep local.
    global _AUTOBIO_MEM  # type: ignore
    try:
        return _AUTOBIO_MEM
    except NameError:
        _AUTOBIO_MEM = AutobiographicalMemory()
        return _AUTOBIO_MEM