"""
ContinuousStreamLearner — S27
Parallel async problem-solving streams with EWC-inspired interference detection.

Streams run as daemon threads, each independently generating and solving problems.
InterferenceShield monitors per-domain confidence EMA and pauses streams that cause
catastrophic forgetting (> _INTERFERENCE_THRESHOLD drop from baseline).
"""
from __future__ import annotations

import threading
import time
import random
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


class StreamType(Enum):
    EXPLORE    = "explore"     # random domain — broad coverage
    EXPLOIT    = "exploit"     # weakest domain — targeted improvement
    IMAGINE    = "imagine"     # problems from GenerativeWorldModel
    CURRICULUM = "curriculum"  # ordered by AutonomousTrainer curriculum


# ── domain problem banks ──────────────────────────────────────────────────────

_DOMAIN_PROBLEMS: Dict[str, List[str]] = {
    "arithmetic": [
        "2 + 3", "10 - 4", "6 * 7", "15 / 3", "2^8", "sqrt(144)",
        "100 / 4 + 5", "3 * (4 + 2)", "2^10 - 1", "7 * 8 - 3",
    ],
    "algebra": [
        "x + 5 = 12", "2*x = 14", "x^2 - 9 = 0",
        "3*x + 2 = 11", "x/4 = 3", "x^2 + 2*x + 1 = 0",
    ],
    "logic": [
        "A and B", "not (A or B)", "A implies B",
        "A xor B", "not not A", "(A and B) or C",
    ],
    "calculus": [
        "d/dx x^2", "d/dx sin(x)", "d/dx x^3 + 2*x",
        "d/dx e^x", "d/dx log(x)", "d/dx x^4 - 3*x",
    ],
    "physics": [
        "F = m * a", "E = m * c^2", "p = m * v",
        "KE = 0.5 * m * v^2", "v = u + a*t",
    ],
    "trig": [
        "sin(0)", "cos(pi)", "sin(pi/2)",
        "tan(pi/4)", "sin^2(x) + cos^2(x)",
    ],
}


# ── InterferenceShield ────────────────────────────────────────────────────────

class InterferenceShield:
    """
    EWC-inspired interference detector.
    Tracks per-domain solve-rate EMA; flags when a domain drops
    more than _INTERFERENCE_THRESHOLD below its recent baseline.
    """

    _DECAY                  = 0.90
    _INTERFERENCE_THRESHOLD = 0.15

    def __init__(self) -> None:
        self._baselines:   Dict[str, float] = {}
        self._current:     Dict[str, float] = {}
        self._events:      List[dict]        = []

    def record(self, domain: str, solved: bool) -> bool:
        """Update EMA for domain; return True if interference detected."""
        new_val = 1.0 if solved else 0.0

        if domain not in self._current:
            self._current[domain]   = new_val
            self._baselines[domain] = new_val
            return False

        self._current[domain] = (
            self._DECAY * self._current[domain]
            + (1 - self._DECAY) * new_val
        )

        baseline = self._baselines[domain]
        drop = baseline - self._current[domain]

        if drop > self._INTERFERENCE_THRESHOLD:
            self._events.append({
                "domain":   domain,
                "baseline": round(baseline, 3),
                "current":  round(self._current[domain], 3),
                "drop":     round(drop, 3),
                "ts":       time.time(),
            })
            self._baselines[domain] = self._current[domain]  # reset
            return True

        # Ratchet baseline upward only
        if self._current[domain] > self._baselines[domain]:
            self._baselines[domain] = self._current[domain]

        return False

    def domain_rates(self) -> Dict[str, float]:
        return {d: round(v, 3) for d, v in self._current.items()}

    def weakest_domain(self) -> str:
        if not self._current:
            return random.choice(list(_DOMAIN_PROBLEMS.keys()))
        return min(self._current, key=self._current.get)

    def summary(self) -> dict:
        return {
            "domain_rates":      self.domain_rates(),
            "total_interference": len(self._events),
            "recent_events":     self._events[-10:],
        }


# ── StreamStats ───────────────────────────────────────────────────────────────

@dataclass
class StreamStats:
    stream_id:   str
    stream_type: StreamType
    domain:      Optional[str]
    solved:      int   = 0
    failed:      int   = 0
    interference_count: int = 0
    active:      bool  = True
    started_at:  float = field(default_factory=time.time)
    last_tick:   float = field(default_factory=time.time)

    @property
    def solve_rate(self) -> float:
        total = self.solved + self.failed
        return self.solved / total if total > 0 else 0.0

    @property
    def uptime_s(self) -> float:
        return round(time.time() - self.started_at, 1)

    def to_dict(self) -> dict:
        return {
            "stream_id":   self.stream_id,
            "type":        self.stream_type.value,
            "domain":      self.domain,
            "solved":      self.solved,
            "failed":      self.failed,
            "solve_rate":  round(self.solve_rate, 3),
            "interference_count": self.interference_count,
            "active":      self.active,
            "uptime_s":    self.uptime_s,
        }


# ── ContinuousStreamLearner ───────────────────────────────────────────────────

class ContinuousStreamLearner:
    """
    Manages N daemon threads, each running an independent problem-solving stream.

    Design:
    - EXPLORE streams: random domain, broad coverage
    - EXPLOIT stream:  always targets weakest domain (via InterferenceShield)
    - IMAGINE stream:  delegates to GenerativeWorldModel for novel problems
    - CURRICULUM stream: pulls from AutonomousTrainer problem source

    Interference management:
    - After every solve attempt, InterferenceShield.record() is called
    - If domain confidence drops > 15%, the stream is auto-paused for 2 s
    - All results are fed back to AffectiveEnergy for curiosity modulation
    """

    _TICK_INTERVAL  = 0.5    # seconds between attempts per stream
    _MAX_STREAMS    = 6
    _HISTORY_LIMIT  = 60

    def __init__(self) -> None:
        self._engine       = None
        self._affective    = None
        self._generative   = None
        self._trainer      = None   # AutonomousTrainer

        self._streams:  List[StreamStats]             = []
        self._threads:  Dict[str, threading.Thread]   = {}
        self._shield    = InterferenceShield()
        self._lock      = threading.Lock()
        self._running   = False

        self._total_solved  = 0
        self._total_failed  = 0
        self._pauses        = 0
        self._recent:       List[dict] = []
        self._started_at    = 0.0

    # ── wiring ────────────────────────────────────────────────────────────────

    def wire(self, engine=None, affective_energy=None,
             generative_world=None, trainer=None) -> None:
        self._engine     = engine
        self._affective  = affective_energy
        self._generative = generative_world
        self._trainer    = trainer

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self, n_streams: int = 4) -> None:
        if self._running:
            return
        self._running   = True
        self._started_at = time.time()

        configs = [
            (StreamType.EXPLORE,    None),
            (StreamType.EXPLOIT,    None),
            (StreamType.IMAGINE,    None),
            (StreamType.CURRICULUM, None),
            (StreamType.EXPLORE,    "arithmetic"),
            (StreamType.EXPLORE,    "algebra"),
        ][:min(n_streams, self._MAX_STREAMS)]

        for i, (stype, domain) in enumerate(configs):
            sid  = f"stream_{i}_{stype.value}"
            stat = StreamStats(sid, stype, domain)
            with self._lock:
                self._streams.append(stat)
            t = threading.Thread(
                target=self._run_stream, args=(stat,),
                daemon=True, name=f"CStream-{i}"
            )
            self._threads[sid] = t
            t.start()

        log.debug(f"ContinuousStreamLearner: {len(self._streams)} streams started")

    def stop(self) -> None:
        self._running = False

    def pause_stream(self, stream_id: str) -> None:
        with self._lock:
            for s in self._streams:
                if s.stream_id == stream_id:
                    s.active = False

    def resume_stream(self, stream_id: str) -> None:
        with self._lock:
            for s in self._streams:
                if s.stream_id == stream_id:
                    s.active = True

    # ── stream worker ─────────────────────────────────────────────────────────

    def _run_stream(self, stat: StreamStats) -> None:
        while self._running:
            if not stat.active:
                time.sleep(self._TICK_INTERVAL)
                continue

            try:
                problem, domain = self._pick_problem(stat)
                solved = self._attempt(problem)

                with self._lock:
                    if solved:
                        stat.solved          += 1
                        self._total_solved   += 1
                    else:
                        stat.failed          += 1
                        self._total_failed   += 1
                    stat.last_tick = time.time()

                    # Affective signal feedback
                    if self._affective and problem:
                        try:
                            self._affective.compute(problem, domain or "general", 0.3)
                        except Exception:
                            pass

                    # Interference check
                    if domain:
                        interfered = self._shield.record(domain, solved)
                        if interfered:
                            stat.interference_count += 1
                            self._pauses            += 1
                            stat.active              = False

                            def _auto_resume(s=stat):
                                time.sleep(2.0)
                                s.active = True

                            threading.Thread(target=_auto_resume, daemon=True).start()

                    self._recent.append({
                        "stream": stat.stream_id,
                        "domain": domain,
                        "expr":   (problem or "")[:45],
                        "solved": solved,
                    })
                    if len(self._recent) > self._HISTORY_LIMIT:
                        self._recent.pop(0)

            except Exception as e:
                log.debug(f"Stream {stat.stream_id} error: {e}")

            time.sleep(self._TICK_INTERVAL)

    # ── problem selection ─────────────────────────────────────────────────────

    def _pick_problem(self, stat: StreamStats):
        """Generate next problem for this stream; return (expression, domain)."""
        if stat.stream_type == StreamType.IMAGINE and self._generative:
            try:
                probs = self._generative.imagine(domain=stat.domain, n=1)
                if probs:
                    p = probs[0]
                    return p.expression, p.domain
            except Exception:
                pass

        if stat.stream_type == StreamType.CURRICULUM and self._trainer:
            try:
                prob = self._trainer.pick_problem()
                if prob:
                    return str(prob), getattr(prob, "domain", "arithmetic")
            except Exception:
                pass

        if stat.stream_type == StreamType.EXPLOIT:
            domain = self._shield.weakest_domain()
        elif stat.domain:
            domain = stat.domain
        else:
            domain = random.choice(list(_DOMAIN_PROBLEMS.keys()))

        problems = _DOMAIN_PROBLEMS.get(domain, _DOMAIN_PROBLEMS["arithmetic"])
        return random.choice(problems), domain

    # ── solve attempt ─────────────────────────────────────────────────────────

    def _attempt(self, problem: str) -> bool:
        """Solve problem; return True on success."""
        if not problem:
            return False
        if not self._engine:
            log.warning("[ContinuousStream] No engine available; skipping solve attempt")
            return False

        try:
            engine = self._engine
            # Brain proxy: delegate to inner engine if present
            if hasattr(engine, '_engine'):
                engine = engine._engine
            if not hasattr(engine, 'solve'):
                log.warning("[ContinuousStream] Engine has no solve() method; skipping")
                return False

            result = engine.solve(problem)
            if result is None:
                return False
            return getattr(result, 'energy', 1.0) < 0.5
        except Exception:
            return False

    # ── throughput ────────────────────────────────────────────────────────────

    @property
    def throughput_per_min(self) -> float:
        elapsed = max(1.0, time.time() - self._started_at) / 60.0
        total   = self._total_solved + self._total_failed
        return round(total / elapsed, 1)

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        with self._lock:
            streams = [s.to_dict() for s in self._streams]
            recent  = list(self._recent[-20:])
        total = self._total_solved + self._total_failed
        return {
            "running":           self._running,
            "n_streams":         len(streams),
            "total_solved":      self._total_solved,
            "total_failed":      self._total_failed,
            "solve_rate":        round(self._total_solved / total, 3) if total else 0.0,
            "throughput_per_min": self.throughput_per_min,
            "interference_pauses": self._pauses,
            "streams":           streams,
            "recent_results":    recent,
            "interference":      self._shield.summary(),
        }


# ── ContinuousLearner ─────────────────────────────────────────────────────────
# T2-3: Episode-driven micro-learning between daemon batches.
# Subscribes to the event bus and triggers learning micro-steps on every k
# solve episodes, rather than waiting for the next N-cycle batch.

_MICRO_STEP_EVERY = 3    # trigger micro-learning after this many episodes
_MAX_BUFFER       = 100  # max replay buffer size
_FLUSH_INTERVAL   = 30.0 # seconds between full buffer flushes


class ContinuousLearner:
    """
    Continuous learning stream wired to the event bus.

    Architecture:
      solve episode → event bus "episode_complete" → _buffer → micro-step

    Micro-step (every _MICRO_STEP_EVERY episodes):
      1. CreditAssigner.assign_credit() per episode
      2. CausalInduction.induce() to check for rule promotion
    """

    def __init__(self, experiment_runner=None, concept_registry=None, world_model=None):
        self._runner = experiment_runner
        self._registry = concept_registry
        self._world_model = world_model
        self._buffer: deque = deque(maxlen=_MAX_BUFFER)
        self._lock = threading.Lock()
        self._episode_count = 0
        self._micro_steps = 0
        self._last_flush = time.time()
        self._active = True

        # Subscribe to episode events
        try:
            from sare.core.event_bus import get_event_bus
            get_event_bus().subscribe("episode_complete", self._on_episode)
            log.info("ContinuousLearner: subscribed to episode_complete events")
        except Exception as e:
            log.debug("ContinuousLearner event bus subscription failed: %s", e)

        # Start flush thread
        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="continuous-flush"
        )
        self._flush_thread.start()

    def record_episode(self, result) -> None:
        """Manually record a solve result (called directly from run_batch)."""
        try:
            episode = {
                "expression": getattr(result, "expression",
                                      getattr(result, "problem_id", "")),
                "domain":       getattr(result, "domain", "general"),
                "solved":       bool(getattr(result, "solved", False)),
                "proof_steps":  list(getattr(result, "proof_steps", None) or []),
                "energy_before": float(getattr(result, "energy_before", 0) or 0),
                "energy_after":  float(getattr(result, "energy_after", 0) or 0),
                "timestamp":    time.time(),
            }
            with self._lock:
                self._buffer.append(episode)
                self._episode_count += 1
                count = self._episode_count

            # Trigger micro-step every _MICRO_STEP_EVERY episodes
            if count % _MICRO_STEP_EVERY == 0:
                threading.Thread(
                    target=self._micro_step, daemon=True, name="micro-learn",
                ).start()
        except Exception as e:
            log.debug("ContinuousLearner.record_episode error: %s", e)

    def _on_episode(self, data: dict) -> None:
        """Handle episode_complete event from event bus."""
        try:
            with self._lock:
                self._buffer.append(data)
                self._episode_count += 1
                count = self._episode_count
            if count % _MICRO_STEP_EVERY == 0:
                threading.Thread(
                    target=self._micro_step, daemon=True, name="micro-learn"
                ).start()
        except Exception as e:
            log.debug("ContinuousLearner._on_episode error: %s", e)

    def _micro_step(self) -> None:
        """Run one micro-learning step on the most recent buffer slice."""
        try:
            with self._lock:
                episodes = list(self._buffer)[-_MICRO_STEP_EVERY:]

            if not episodes:
                return

            solved_count = sum(1 for e in episodes if e.get("solved"))

            # 1. Credit assignment
            try:
                from sare.learning.credit_assignment import CreditAssigner
                ca = CreditAssigner()
                ca.load()
                for ep in episodes:
                    if ep.get("solved") and ep.get("proof_steps"):
                        steps = ep["proof_steps"]
                        n = len(steps)
                        e0 = float(ep.get("energy_before", 0) or 0)
                        e1 = float(ep.get("energy_after", 0) or 0)
                        delta = (e1 - e0) / n if n else 0.0
                        traj = [e0 + i * delta for i in range(n + 1)]
                        ca.assign_credit(steps, traj, domain=ep.get("domain", "general"))
                ca.save()
            except Exception as ca_exc:
                log.debug("ContinuousLearner credit step error: %s", ca_exc)

            # 2. CausalInduction — check for promotable rules
            try:
                from sare.causal.induction import CausalInduction
                ci = CausalInduction()
                for ep in episodes:
                    if ep.get("solved") and ep.get("proof_steps"):
                        # Build minimal problem/result objects for induce()
                        class _P:
                            pass
                        class _R:
                            pass
                        _prob = _P()
                        _prob.expression = ep.get("expression", "")
                        _prob.domain = ep.get("domain", "general")
                        _res = _R()
                        _res.solved = True
                        _res.proof_steps = ep["proof_steps"]
                        _res.domain = ep.get("domain", "general")
                        ci.induce(problem=_prob, result=_res)
            except Exception as ci_exc:
                log.debug("ContinuousLearner induction step error: %s", ci_exc)

            self._micro_steps += 1
            if solved_count > 0:
                log.debug(
                    "ContinuousLearner micro-step %d: %d/%d solved",
                    self._micro_steps, solved_count, len(episodes),
                )
        except Exception as e:
            log.debug("ContinuousLearner._micro_step error: %s", e)

    def _flush_loop(self) -> None:
        """Periodic flush thread — consolidates buffer every _FLUSH_INTERVAL seconds."""
        while self._active:
            time.sleep(_FLUSH_INTERVAL)
            try:
                now = time.time()
                if now - self._last_flush < _FLUSH_INTERVAL * 0.9:
                    continue
                with self._lock:
                    buffer_snapshot = list(self._buffer)
                self._last_flush = now

                if buffer_snapshot:
                    log.debug(
                        "ContinuousLearner flush: %d episodes buffered, %d micro-steps done",
                        len(buffer_snapshot), self._micro_steps,
                    )
                    self._micro_step()
            except Exception as e:
                log.debug("ContinuousLearner flush error: %s", e)

    def stop(self) -> None:
        self._active = False

    @property
    def stats(self) -> dict:
        return {
            "episodes_seen": self._episode_count,
            "micro_steps":   self._micro_steps,
            "buffer_size":   len(self._buffer),
            "active":        self._active,
        }
