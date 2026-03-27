"""
evolver_chat.py — Live monitoring, interactive chat, and feedback loop for SelfImprover.

Provides:
  EvolverLogBuffer  — logging.Handler that captures evolver logs into a circular deque
  EvolverChat       — processes user messages (suggest / apply / interrupt / feedback)
  get_log_buffer()  — singleton log buffer
  get_evolver_chat() — singleton chat engine
"""
from __future__ import annotations

import collections
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

_ROOT   = Path(__file__).resolve().parents[3]
_PYTHON = _ROOT / "python"
_MEMORY = _ROOT / "data" / "memory"

# ── Log buffer ────────────────────────────────────────────────────────────────

# Colour tags fed to the UI (CSS class names)
_LEVEL_TAG = {
    "DEBUG":    "dim",
    "INFO":     "info",
    "WARNING":  "warn",
    "ERROR":    "error",
    "CRITICAL": "error",
}
_TURN_TAG = {
    "PRE-SCREEN":  "prescreen",
    "PROPOSER":    "proposer",
    "PLANNER":     "planner",
    "EXECUTOR":    "executor",
    "CRITIC":      "critic",
    "JUDGE":       "judge",
    "VERIFIER":    "verifier",
    "PATCH":       "patch",
    "ROLLBACK":    "rollback",
}

def _classify(msg: str) -> str:
    """Return a UI colour tag based on message content."""
    upper = msg.upper()
    for kw, tag in _TURN_TAG.items():
        if kw in upper:
            return tag
    return "info"


class EvolverLogBuffer(logging.Handler):
    """Captures log messages from evolver-related loggers into a circular deque."""

    MAX_ENTRIES = 1000

    def __init__(self):
        super().__init__()
        self._buf: collections.deque = collections.deque(maxlen=self.MAX_ENTRIES)
        self._seq   = 0
        self._lock  = threading.Lock()

    def emit(self, record: logging.LogRecord):
        with self._lock:
            self._seq += 1
            msg = record.getMessage()
            entry: dict = {
                "id":    self._seq,
                "ts":    record.created,
                "level": record.levelname,
                "tag":   _classify(msg),
                "msg":   msg,
                "src":   record.name.split(".")[-1],
            }
            # Capture full LLM message payloads when present
            llm_dir = getattr(record, "llm_dir", None)
            if llm_dir == "send":
                entry["tag"]        = "llm_send"
                entry["llm_dir"]    = "send"
                entry["llm_model"]  = getattr(record, "llm_model",  "")
                entry["llm_role"]   = getattr(record, "llm_role",   "")
                entry["llm_system"] = getattr(record, "llm_system", "")
                entry["llm_prompt"] = getattr(record, "llm_prompt", "")
            elif llm_dir == "recv":
                entry["tag"]          = "llm_recv"
                entry["llm_dir"]      = "recv"
                entry["llm_model"]    = getattr(record, "llm_model",    "")
                entry["llm_role"]     = getattr(record, "llm_role",     "")
                entry["llm_response"] = getattr(record, "llm_response", "")
                entry["llm_in_tok"]   = getattr(record, "llm_in_tok",   0)
                entry["llm_out_tok"]  = getattr(record, "llm_out_tok",  0)
                entry["llm_cost"]     = getattr(record, "llm_cost",     0.0)
            self._buf.append(entry)

    def get_since(self, since_id: int) -> List[dict]:
        with self._lock:
            return [e for e in self._buf if e["id"] > since_id]

    def get_all(self) -> List[dict]:
        with self._lock:
            return list(self._buf)


# ── Suggestion ────────────────────────────────────────────────────────────────

class Suggestion:
    TTL = 900   # 15-minute expiry

    def __init__(self, file_path: str, module_name: str, improvement_type: str,
                 description: str, reason: str, score: float):
        self.sid              = uuid.uuid4().hex[:8]
        self.file_path        = file_path
        self.module_name      = module_name
        self.improvement_type = improvement_type
        self.description      = description
        self.reason           = reason
        self.score            = score
        self.created_at       = time.time()
        self.status           = "pending"   # pending | accepted | rejected

    @property
    def expired(self) -> bool:
        return time.time() - self.created_at > self.TTL

    def to_dict(self) -> dict:
        return {
            "sid":              self.sid,
            "file_path":        self.file_path,
            "file_name":        Path(self.file_path).name,
            "module_name":      self.module_name,
            "improvement_type": self.improvement_type,
            "description":      self.description,
            "reason":           self.reason[:200],
            "score":            round(self.score, 2),
            "status":           self.status,
            "created_at":       self.created_at,
        }


# ── Chat message ──────────────────────────────────────────────────────────────

class ChatMessage:
    def __init__(self, role: str, text: str, data: dict = None):
        self.id   = uuid.uuid4().hex[:8]
        self.role = role        # "user" | "assistant" | "system"
        self.text = text
        self.data = data or {}
        self.ts   = time.time()

    def to_dict(self) -> dict:
        return {
            "id":   self.id,
            "role": self.role,
            "text": self.text,
            "data": self.data,
            "ts":   self.ts,
        }


# ── EvolverChat ───────────────────────────────────────────────────────────────

class EvolverChat:
    """Processes user messages, generates improvement suggestions, manages feedback."""

    FEEDBACK_PATH = _MEMORY / "evolver_feedback.json"
    HISTORY_PATH  = _MEMORY / "evolver_chat_history.json"

    def __init__(self):
        self._messages:   List[ChatMessage] = []
        self._suggestions: List[Suggestion] = []
        self._feedback:   List[dict]        = []
        self._interrupt   = threading.Event()
        self._lock        = threading.Lock()
        self._load_feedback()
        self._load_history()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_messages(self, since_id: str = "") -> List[dict]:
        with self._lock:
            msgs = [m.to_dict() for m in self._messages]
        if since_id:
            idx = next((i for i, m in enumerate(msgs) if m["id"] == since_id), -1)
            return msgs[idx + 1:] if idx >= 0 else msgs
        return msgs[-200:]

    def get_pending_suggestions(self) -> List[dict]:
        with self._lock:
            self._suggestions = [s for s in self._suggestions
                                  if not s.expired and s.status == "pending"]
            return [s.to_dict() for s in self._suggestions]

    def is_interrupted(self) -> bool:
        return self._interrupt.is_set()

    # ── Message processor ─────────────────────────────────────────────────────

    def process_message(self, text: str) -> dict:
        """Route user message and return {"message": ..., ...extra}."""
        self._add_message("user", text)
        lower = text.lower().strip()

        # YES / APPLY
        if re.search(r'\byes\b|\bapply\b|\bdo it\b|\bgo ahead\b|\bapprove\b', lower):
            sid_m = re.search(r'\b([a-f0-9]{8})\b', lower)
            return self._handle_apply(sid_m.group(1) if sid_m else None)

        # NO / SKIP
        if re.search(r'\bno\b|\bskip\b|\breject\b|\bdismiss\b|\bnext\b', lower):
            sid_m = re.search(r'\b([a-f0-9]{8})\b', lower)
            return self._handle_reject(sid_m.group(1) if sid_m else None)

        # INTERRUPT
        if re.search(r'\binterrupt\b|\bstop\b|\bpause\b|\bhalt\b', lower):
            return self._handle_interrupt()

        # SUGGEST / SCAN
        if re.search(r'\bsuggest\b|\bfind\b|\bimprove\b|\bscan\b|\bbottleneck\b|\banalyze\b|\bwhat.*improve\b', lower):
            return self._handle_suggest(text)

        # STATUS
        if re.search(r'\bstatus\b|\bwhats happening\b|\bactive\b|\brunning\b|\bshow\b', lower):
            return self._handle_status()

        # FEEDBACK
        if re.search(r'\bgood\b|\bbad\b|\bwrong\b|\bgreat\b|\bterrible\b|\bfeedback\b|\bthumb\b', lower):
            return self._handle_feedback(text)

        # Rollback
        if re.search(r'\brollback\b|\bundo\b|\brevert\b', lower):
            return self._handle_rollback(lower)

        # Fallback → LLM
        return self._handle_llm_chat(text)

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _handle_apply(self, sid: Optional[str]) -> dict:
        from sare.meta.self_improver import get_self_improver
        sug = self._find_suggestion(sid)
        if not sug:
            reply = self._add_message("assistant",
                "No pending suggestions. Type **suggest** to find improvement opportunities.")
            return {"message": reply.to_dict()}

        sug.status = "accepted"
        fname = Path(sug.file_path).name
        reply = self._add_message("assistant",
            f"⚙️ Starting **{sug.improvement_type}** improvement for `{fname}`…\n"
            f"Running the full 7-turn pipeline (3–5 min). Watch the log for progress.",
            {"action": "apply_started", "suggestion": sug.to_dict()})

        def _run():
            try:
                result = get_self_improver().run_once(
                    target_file=sug.file_path,
                    improvement_type=sug.improvement_type,
                )
                outcome = result.get("outcome", "unknown")
                score   = result.get("critic_score", 0)
                icon    = "✅" if outcome == "applied" else "❌"
                self._add_message("system",
                    f"{icon} **{fname}** → `{outcome}` (critic: {score}/10)",
                    {"result": result, "sid": sug.sid})
            except Exception as e:
                self._add_message("system", f"Error: {e}", {"error": str(e)})

        threading.Thread(target=_run, daemon=True, name="EvolverChatApply").start()
        return {"message": reply.to_dict(), "suggestion": sug.to_dict()}

    def _handle_reject(self, sid: Optional[str]) -> dict:
        sug = self._find_suggestion(sid)
        if not sug:
            reply = self._add_message("assistant", "No pending suggestion to dismiss.")
            return {"message": reply.to_dict()}
        sug.status = "rejected"
        # Surface next suggestion if any
        pending = [s for s in self._suggestions if s.status == "pending"]
        extra = ""
        if pending:
            n = pending[0]
            extra = f"\nNext suggestion: **{Path(n.file_path).name}** `{n.improvement_type}` (ID: `{n.sid}`)"
        reply = self._add_message("assistant",
            f"Dismissed `{Path(sug.file_path).name}`.{extra}")
        return {"message": reply.to_dict()}

    def _handle_interrupt(self) -> dict:
        from sare.meta.self_improver import get_self_improver
        si    = get_self_improver()
        active = list(si._active.values())
        if not active:
            reply = self._add_message("assistant", "No active debates to interrupt.")
            return {"message": reply.to_dict()}

        self._interrupt.set()
        files = [a["file"] for a in active]
        reply = self._add_message("system",
            f"⛔ Interrupt signal sent to {len(active)} debate(s): {', '.join(files)}\n"
            "Current turns will finish; next turns will be skipped.",
            {"interrupted": files})

        def _reset():
            time.sleep(5)
            self._interrupt.clear()
        threading.Thread(target=_reset, daemon=True).start()
        return {"message": reply.to_dict(), "interrupted": files}

    def _handle_suggest(self, user_text: str) -> dict:
        """Scan codebase and return prioritised suggestions."""
        thinking = self._add_message("assistant",
            "🔍 Scanning codebase for improvement opportunities…")

        try:
            from sare.meta.bottleneck_analyzer import BottleneckAnalyzer, ImprovementTarget, _path_to_module
            from sare.meta.self_improver import get_self_improver
            si = get_self_improver()
            patched = {p.target_file for p in si._patches}

            try:
                targets = BottleneckAnalyzer().analyze()[:10]
            except Exception:
                targets = []

            # Supplement with codebase sweep if needed
            if len(targets) < 6:
                all_py = sorted(_PYTHON.rglob("*.py"))
                skip   = {"__pycache__", "synthesized_modules", "code_backups"}
                extras = [
                    ImprovementTarget(
                        file_path=str(p),
                        module_name=_path_to_module(str(p)),
                        improvement_type="optimize",
                        score=0.35 + (p.stat().st_size / 1_000_000),
                        reason="codebase sweep",
                    )
                    for p in all_py
                    if not any(d in p.parts for d in skip)
                    and p.stem not in ("__init__",)
                    and p.stat().st_size > 800
                    and str(p) not in patched
                ][:max(0, 8 - len(targets))]
                targets.extend(extras)

            # Expire old suggestions and add new ones
            with self._lock:
                self._suggestions = [s for s in self._suggestions
                                      if not s.expired and s.status == "pending"]

            new_sugs: List[Suggestion] = []
            for t in targets[:6]:
                sug = Suggestion(
                    file_path=t.file_path,
                    module_name=t.module_name,
                    improvement_type=t.improvement_type,
                    description=t.reason or f"{t.improvement_type} {Path(t.file_path).name}",
                    reason=json.dumps(t.evidence or {}, indent=2)[:200] if t.evidence else (t.reason or ""),
                    score=t.score,
                )
                new_sugs.append(sug)
                with self._lock:
                    self._suggestions.insert(0, sug)

            if not new_sugs:
                msg = self._add_message("assistant",
                    "✅ No clear improvement targets found — codebase looks healthy!")
                return {"message": msg.to_dict(), "suggestions": []}

            lines = [f"Found **{len(new_sugs)} improvement opportunities**:\n"]
            for i, s in enumerate(new_sugs, 1):
                icon = {"optimize": "⚡", "extend": "🔧", "fix": "🐛"}.get(s.improvement_type, "🔧")
                lines.append(f"**{i}. {icon} {Path(s.file_path).name}** — `{s.improvement_type}`")
                lines.append(f"   {s.description[:150]}")
                lines.append(f"   Score: **{s.score:.2f}** | ID: `{s.sid}`\n")
            lines.append("Say **yes** (top pick) or **yes `{id}`** for a specific one.")

            msg = self._add_message("assistant", "\n".join(lines),
                {"suggestions": [s.to_dict() for s in new_sugs]})
            return {"message": msg.to_dict(), "suggestions": [s.to_dict() for s in new_sugs]}

        except Exception as e:
            msg = self._add_message("assistant", f"Scan error: {e}")
            return {"message": msg.to_dict(), "suggestions": []}

    def _handle_status(self) -> dict:
        from sare.meta.self_improver import get_self_improver
        st     = get_self_improver().get_status()
        active = st.get("active_debates", [])

        if active:
            active_str = "\n".join(
                f"  • `{a['file']}` → **{a['turn']}** ({a['elapsed_s']}s)"
                for a in active)
        else:
            active_str = "  _(none)_"

        last_target  = Path(st.get("last_run_target") or "").name or "—"
        last_outcome = st.get("last_run_outcome") or "—"

        text = (
            f"**Evolver Status**\n"
            f"Daemon: {'🟢 running' if st.get('running') else '🔴 stopped'} | "
            f"Debates: {st.get('total_debates', 0)} | "
            f"Applied: {st.get('patches_applied', 0)} | "
            f"Rolled back: {st.get('patches_rolled_back', 0)}\n"
            f"Pre-screen rejections: {st.get('prescreened_rejected', 0)} | "
            f"Test rollbacks: {st.get('tests_rolled_back', 0)}\n"
            f"Last: `{last_target}` → `{last_outcome}`\n\n"
            f"**Active debates:**\n{active_str}"
        )
        msg = self._add_message("assistant", text, {"status": st})
        return {"message": msg.to_dict(), "status": st}

    def _handle_feedback(self, text: str) -> dict:
        from sare.meta.self_improver import get_self_improver
        si = get_self_improver()
        if not si._debates:
            reply = self._add_message("assistant", "No debates to give feedback on yet.")
            return {"message": reply.to_dict()}

        last    = si._debates[-1]
        positive = any(w in text.lower() for w in
                       ["good", "great", "correct", "right", "nice", "excellent", "perfect"])
        rating  = 1 if positive else -1
        entry   = {
            "id":          uuid.uuid4().hex[:8],
            "debate_ts":   last.timestamp,
            "target_file": last.target_file,
            "outcome":     last.outcome,
            "rating":      rating,
            "comment":     text,
            "ts":          time.time(),
        }
        with self._lock:
            self._feedback.append(entry)
        self._save_feedback()

        icon  = "👍" if positive else "👎"
        reply = self._add_message("assistant",
            f"{icon} Feedback recorded for `{Path(last.target_file).name}` ({last.outcome}). "
            "This is stored in `data/memory/evolver_feedback.json` and guides future target selection.",
            {"feedback": entry})
        return {"message": reply.to_dict(), "feedback": entry}

    def _handle_rollback(self, text: str) -> dict:
        from sare.meta.self_improver import get_self_improver
        si = get_self_improver()
        # Find patch id in text
        pid_m = re.search(r'\b([a-f0-9]{8})\b', text)
        if pid_m:
            pid = pid_m.group(1)
        elif si._patches:
            pid = si._patches[-1].patch_id
        else:
            reply = self._add_message("assistant", "No patches to roll back.")
            return {"message": reply.to_dict()}
        result = si.rollback(pid)
        icon   = "↩️" if result.get("success") else "❌"
        reply  = self._add_message("assistant",
            f"{icon} Rollback `{pid}`: {result}",
            {"rollback_result": result})
        return {"message": reply.to_dict()}

    def _handle_llm_chat(self, text: str) -> dict:
        """Call LLM in background thread — POST returns immediately, answer arrives via SSE."""
        # Ack message so SSE delivers something quickly (removes typing indicator)
        ack = self._add_message("assistant", "⏳ _Thinking…_")

        def _run():
            try:
                from sare.interface.llm_bridge import _call_model
                from sare.meta.self_improver import get_self_improver
                st = get_self_improver().get_status()
                ctx = json.dumps({
                    "running":         st.get("running"),
                    "active_debates":  st.get("active_debates"),
                    "patches_applied": st.get("patches_applied"),
                    "last_outcome":    st.get("last_run_outcome"),
                    "last_target":     st.get("last_run_target"),
                }, indent=2)
                prompt = (
                    "You are the SARE-HX Evolver assistant. The user is talking to you through "
                    "a chat interface while the self-improvement daemon runs in the background.\n\n"
                    f"CURRENT EVOLVER STATUS:\n{ctx}\n\n"
                    f"USER: {text}\n\n"
                    "Answer concisely (2-4 sentences). Available commands the user can type: "
                    "suggest, yes [id], no [id], interrupt, status, feedback, rollback."
                )
                answer = _call_model(prompt, role="prescreen")
            except Exception:
                answer = (
                    "Commands: **suggest** · **yes [id]** · **no [id]** · "
                    "**interrupt** · **status** · **feedback** · **rollback**"
                )
            self._add_message("assistant", answer)

        threading.Thread(target=_run, daemon=True, name="EvolverChatLLM").start()
        return {"message": ack.to_dict()}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_suggestion(self, sid: Optional[str]) -> Optional[Suggestion]:
        with self._lock:
            pending = [s for s in self._suggestions if s.status == "pending" and not s.expired]
        if sid:
            return next((s for s in pending if s.sid == sid), None)
        return pending[0] if pending else None

    def _add_message(self, role: str, text: str, data: dict = None) -> ChatMessage:
        msg = ChatMessage(role, text, data)
        with self._lock:
            self._messages.append(msg)
            if len(self._messages) > 500:
                self._messages = self._messages[-500:]
        self._save_history()
        return msg

    # ── Persistence ───────────────────────────────────────────────────────────

    def get_feedback(self) -> List[dict]:
        with self._lock:
            return list(self._feedback)

    def _save_feedback(self):
        try:
            _MEMORY.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {"feedback": self._feedback[-500:]}
            self.FEEDBACK_PATH.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _load_feedback(self):
        if not self.FEEDBACK_PATH.exists():
            return
        try:
            self._feedback = json.loads(self.FEEDBACK_PATH.read_text()).get("feedback", [])
        except Exception:
            pass

    def _save_history(self):
        try:
            _MEMORY.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {"messages": [m.to_dict() for m in self._messages[-300:]]}
            self.HISTORY_PATH.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _load_history(self):
        if not self.HISTORY_PATH.exists():
            return
        try:
            for d in json.loads(self.HISTORY_PATH.read_text()).get("messages", []):
                m = ChatMessage(d["role"], d["text"], d.get("data", {}))
                m.id = d.get("id", m.id)
                m.ts = d.get("ts", m.ts)
                self._messages.append(m)
        except Exception:
            pass


# ── Singletons ────────────────────────────────────────────────────────────────

_log_buffer:   Optional[EvolverLogBuffer] = None
_evolver_chat: Optional[EvolverChat]      = None
_singleton_lock = threading.Lock()


def get_log_buffer() -> EvolverLogBuffer:
    global _log_buffer
    if _log_buffer is None:
        with _singleton_lock:
            if _log_buffer is None:
                _log_buffer = EvolverLogBuffer()
                _log_buffer.setLevel(logging.DEBUG)

                # Target loggers — set level to INFO so messages aren't filtered
                # before reaching our handler
                target_loggers = [
                    "sare.meta.self_improver",
                    "sare.interface.llm_bridge",
                    "sare.curiosity.experiment_runner",
                    "sare.curiosity.curriculum_generator",
                    "sare.memory.memory_manager",
                    "sare.meta.evolver_chat",
                    "sare.meta.bottleneck_analyzer",
                    "evolver_runner",
                ]
                for name in target_loggers:
                    lg = logging.getLogger(name)
                    lg.setLevel(logging.INFO)
                    # Don't add duplicate handlers
                    if _log_buffer not in lg.handlers:
                        lg.addHandler(_log_buffer)

                # Also capture all sare.* via parent namespace logger
                sare_root = logging.getLogger("sare")
                sare_root.setLevel(logging.INFO)
                if _log_buffer not in sare_root.handlers:
                    sare_root.addHandler(_log_buffer)

                # Inject a startup marker so user sees the buffer is live
                _log_buffer.emit(_make_system_record("EvolverLogBuffer installed — capturing all sare.* logs"))
    return _log_buffer


def _make_system_record(msg: str) -> logging.LogRecord:
    r = logging.LogRecord(
        name="evolver_chat", level=logging.INFO,
        pathname="", lineno=0, msg=msg, args=(), exc_info=None,
    )
    return r


def get_evolver_chat() -> EvolverChat:
    global _evolver_chat
    if _evolver_chat is None:
        with _singleton_lock:
            if _evolver_chat is None:
                _evolver_chat = EvolverChat()
    return _evolver_chat
