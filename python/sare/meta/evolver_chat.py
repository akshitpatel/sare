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
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]
_MEMORY = _ROOT / "data" / "memory"

# ── Log buffer ────────────────────────────────────────────────────────────────

# Colour tags fed to the UI (CSS class names)
_LEVEL_TAG = {
    "DEBUG": "dim",
    "INFO": "info",
    "WARNING": "warn",
    "ERROR": "error",
    "CRITICAL": "error",
}
_TURN_TAG = {
    "PRE-SCREEN": "prescreen",
    "PROPOSER": "proposer",
    "PLANNER": "planner",
    "EXECUTOR": "executor",
    "CRITIC": "critic",
    "JUDGE": "judge",
    "VERIFIER": "verifier",
    "PATCH": "patch",
    "ROLLBACK": "rollback",
}


def _classify(msg: str) -> str:
    """Return a UI colour tag based on message content."""
    upper = (msg or "").upper()
    for kw, tag in _TURN_TAG.items():
        if kw in upper:
            return tag
    return "info"


class EvolverLogBuffer(logging.Handler):
    """Captures log messages from evolver-related loggers into a circular deque."""

    MAX_ENTRIES = 1000

    def __init__(self):
        super().__init__()
        self._buf: Deque[dict] = collections.deque(maxlen=self.MAX_ENTRIES)
        self._seq = 0
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord):
        with self._lock:
            self._seq += 1
            msg = record.getMessage()
            entry: dict = {
                "id": self._seq,
                "ts": record.created,
                "level": record.levelname,
                "tag": _classify(msg),
                "msg": msg,
                "src": record.name.split(".")[-1],
            }
            # Capture full LLM message payloads when present
            llm_dir = getattr(record, "llm_dir", None)
            if llm_dir == "send":
                entry["tag"] = "llm_send"
                entry["llm_dir"] = "send"
                entry["llm_model"] = getattr(record, "llm_model", "")
                entry["llm_role"] = getattr(record, "llm_role", "")
                entry["llm_system"] = getattr(record, "llm_system", "")
                entry["llm_prompt"] = getattr(record, "llm_prompt", "")
            elif llm_dir == "recv":
                entry["tag"] = "llm_recv"
                entry["llm_dir"] = "recv"
                entry["llm_model"] = getattr(record, "llm_model", "")
                entry["llm_role"] = getattr(record, "llm_role", "")
                entry["llm_response"] = getattr(record, "llm_response", "")
                entry["llm_in_tok"] = getattr(record, "llm_in_tok", 0)
                entry["llm_out_tok"] = getattr(record, "llm_out_tok", 0)
                entry["llm_cost"] = getattr(record, "llm_cost", 0.0)
            self._buf.append(entry)

    def get_since(self, since_id: int) -> List[dict]:
        with self._lock:
            return [e for e in self._buf if e["id"] > since_id]

    def get_all(self) -> List[dict]:
        with self._lock:
            return list(self._buf)


# ── Suggestion ────────────────────────────────────────────────────────────────


class Suggestion:
    TTL = 900  # 15-minute expiry

    def __init__(
        self,
        file_path: str,
        module_name: str,
        improvement_type: str,
        description: str,
        reason: str,
        score: float,
    ):
        self.sid = uuid.uuid4().hex[:8]
        self.file_path = file_path
        self.module_name = module_name
        self.improvement_type = improvement_type
        self.description = description
        self.reason = reason
        self.score = score
        self.created_at = time.time()
        self.status = "pending"  # pending | accepted | rejected

    @property
    def expired(self) -> bool:
        return time.time() - self.created_at > self.TTL

    def to_dict(self) -> dict:
        return {
            "sid": self.sid,
            "file_path": self.file_path,
            "file_name": Path(self.file_path).name,
            "module_name": self.module_name,
            "improvement_type": self.improvement_type,
            "description": self.description,
            "reason": self.reason[:200],
            "score": round(self.score, 2),
            "status": self.status,
            "created_at": self.created_at,
        }


# ── Chat message ──────────────────────────────────────────────────────────────


class ChatMessage:
    def __init__(self, role: str, text: str, data: Optional[dict] = None):
        self.id = uuid.uuid4().hex[:8]
        self.role = role  # "user" | "assistant" | "system"
        self.text = text
        self.data = data or {}
        self.ts = time.time()

    def to_dict(self) -> dict:
        return {"id": self.id, "role": self.role, "text": self.text, "data": self.data, "ts": self.ts}


# ── EvolverChat ───────────────────────────────────────────────────────────────


class EvolverChat:
    """Processes user messages, generates improvement suggestions, manages feedback."""

    FEEDBACK_PATH = _MEMORY / "evolver_feedback.json"
    HISTORY_PATH = _MEMORY / "evolver_chat_history.json"

    def __init__(self):
        self._messages: List[ChatMessage] = []
        self._lock = threading.Lock()

        self._suggestions: List[Suggestion] = []
        self._last_suggestion_sid: Optional[str] = None

        self._load()

    def _load_json_file(self, path: Path, default: Any) -> Any:
        try:
            if not path.exists():
                return default
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def _save_json_file(self, path: Path, obj: Any) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
            tmp.replace(path)
        except Exception:
            log.exception("Failed to save %s", path)

    def _load(self) -> None:
        with self._lock:
            hist = self._load_json_file(self.HISTORY_PATH, default=[])
            if isinstance(hist, dict):
                hist = hist.get("messages", [])
            for item in hist[-500:]:
                role = item.get("role", "assistant")
                text = item.get("text", "")
                data = item.get("data") or {}
                msg = ChatMessage(role=role, text=text, data=data)
                msg.id = item.get("id", msg.id)
                msg.ts = item.get("ts", msg.ts)
                self._messages.append(msg)

            fb = self._load_json_file(self.FEEDBACK_PATH, default=[])
            # feedback array is used as log only; suggestions are transient in this module

            # Attempt to infer last suggestion sid from pending suggestions in feedback history (best-effort)
            pending = None
            for rec in fb[::-1]:
                status = rec.get("status")
                if status == "pending" and rec.get("sid"):
                    pending = rec.get("sid")
                    break
            self._last_suggestion_sid = pending

    def _persist_feedback_event(self, event: dict) -> None:
        with self._lock:
            fb = self._load_json_file(self.FEEDBACK_PATH, default=[])
            fb.append(event)
            self._save_json_file(self.FEEDBACK_PATH, fb)

    def _append_message(self, msg: ChatMessage) -> None:
        with self._lock:
            self._messages.append(msg)
            self._messages = self._messages[-1000:]
            hist = [m.to_dict() for m in self._messages]
            self._save_json_file(self.HISTORY_PATH, hist)

    def get_state(self) -> dict:
        with self._lock:
            return {
                "messages": [m.to_dict() for m in self._messages[-200:]],
                "pending_suggestions": [s.to_dict() for s in self._suggestions if s.status == "pending" and not s.expired],
                "last_suggestion_sid": self._last_suggestion_sid,
            }

    def add_suggestion(
        self,
        file_path: str,
        module_name: str,
        improvement_type: str,
        description: str,
        reason: str,
        score: float,
    ) -> str:
        s = Suggestion(
            file_path=file_path,
            module_name=module_name,
            improvement_type=improvement_type,
            description=description,
            reason=reason,
            score=score,
        )
        with self._lock:
            # expire old pending suggestions
            for sug in self._suggestions:
                if sug.status == "pending" and sug.expired:
                    sug.status = "rejected"
            self._suggestions.append(s)
            self._last_suggestion_sid = s.sid

        # Persist suggestion announcement as a "pending" event so UI can correlate
        self._persist_feedback_event(
            {
                "ts": time.time(),
                "type": "suggestion_created",
                "sid": s.sid,
                "file_path": s.file_path,
                "module_name": s.module_name,
                "improvement_type": s.improvement_type,
                "description": s.description,
                "reason": s.reason,
                "score": s.score,
                "status": "pending",
            }
        )

        msg = ChatMessage(
            role="assistant",
            text=f"Suggestion created: {s.sid} — {Path(s.file_path).name} ({s.improvement_type})",
            data={"sid": s.sid},
        )
        self._append_message(msg)
        return s.sid

    def _find_suggestion_by_sid(self, sid: str) -> Optional[Suggestion]:
        if not sid:
            return None
        with self._lock:
            for s in self._suggestions:
                if s.sid.lower() == sid.lower():
                    return s
        return None

    def _find_active_or_last_suggestion(self) -> Optional[Suggestion]:
        with self._lock:
            # Prefer most recent pending not expired
            pending = [s for s in self._suggestions if s.status == "pending" and not s.expired]
            if pending:
                pending.sort(key=lambda x: x.created_at, reverse=True)
                return pending[0]
            # fallback to last by sid even if expired (we'll treat it as rejected on use)
            if self._last_suggestion_sid:
                for s in self._suggestions:
                    if s.sid == self._last_suggestion_sid:
                        return s
        return None

    def _parse_sid_from_text(self, text: str) -> Optional[str]:
        # Strict 8-hex token for sids
        m = re.search(r"\b([a-fA-F0-9]{8})\b", text or "")
        if not m:
            return None
        sid = m.group(1)
        return sid.lower()[:8]

    def _normalize_text(self, text: str) -> str:
        return (text or "").strip()

    def process_message(self, text: str) -> dict:
        """
        Parse user message and drive feedback signals for SelfImprover.

        Approved change target: EvolverChat.process_message.
        Improvements:
          - More robust command detection (yes/apply/do it/go ahead/approve, etc.)
          - Safer SID extraction (fixed regex, avoid partial matches)
          - Consistent status updates and feedback persistence
          - Handles "interrupt" and "feedback score" with less ambiguity
        """
        raw = text or ""
        user_text = self._normalize_text(raw)
        lower = user_text.lower()

        response_data: dict = {"recognized": False, "action": None}
        action_text = None

        # Expire suggestions in background
        with self._lock:
            for sug in self._suggestions:
                if sug.status == "pending" and sug.expired:
                    sug.status = "rejected"

        yes_pat = re.compile(
            r"\b(?:yes|yep|yeah|apply|do it|go ahead|approve|approved|accept|sure|ok|okay|let's go|let us go)\b",
            re.IGNORECASE,
        )
        no_pat = re.compile(
            r"\b(?:no|nope|nah|reject|rejected|decline|stop|don't|do not|cancel|never)\b",
            re.IGNORECASE,
        )
        interrupt_pat = re.compile(r"\b(?:interrupt|stop now|halt|cancel)\b", re.IGNORECASE)

        # Feedback: "score 0.7", "feedback +1", "good/bad", "rate 5/10"
        # Keep it forgiving but non-invasive.
        good_pat = re.compile(r"\b(?:good|great|excellent|nice|works|success|love it)\b", re.IGNORECASE)
        bad_pat = re.compile(r"\b(?:bad|wrong|broken|fails|failure|revert|not good|doesn't work)\b", re.IGNORECASE)

        score_m = re.search(
            r"\b(?:score|rating|rate)\b\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)\b",
            user_text,
            re.IGNORECASE,
        )
        ratio_m = re.search(
            r"\b(?:rating|rate)\b\s*[:=]?\s*(\d+(?:\.\d+)?)\s*[/]\s*(\d+(?:\.\d+)?)\b",
            user_text,
            re.IGNORECASE,
        )

        # Determine which suggestion SID to apply feedback to
        sid = self._parse_sid_from_text(user_text)
        target = self._find_suggestion_by_sid(sid) if sid else self._find_active_or_last_suggestion()

        # If no suggestions exist, still respond with a helpful message for "yes/apply" commands.
        has_pending = False
        with self._lock:
            has_pending = any(s.status == "pending" and not s.expired for s in self._suggestions)

        def _reject_target(reason: str) -> None:
            nonlocal action_text, response_data, target
            if target and target.status == "pending":
                target.status = "rejected"
                response_data.update(
                    {
                        "recognized": True,
                        "action": "reject",
                        "sid": target.sid,
                        "status": target.status,
                        "reason": reason,
                    }
                )
                self._persist_feedback_event(
                    {
                        "ts": time.time(),
                        "type": "suggestion_feedback",
                        "sid": target.sid,
                        "status": "rejected",
                        "feedback_reason": reason,
                    }
                )
            else:
                response_data.update(
                    {
                        "recognized": True,
                        "action": "reject",
                        "status": "no_target",
                        "reason": reason,
                    }
                )

        def _accept_target(reason: str) -> None:
            nonlocal action_text, response_data, target
            if target:
                if target.status != "pending":
                    # Treat as idempotent accept only if already accepted; otherwise mark accepted anyway
                    prev = target.status
                    if prev == "accepted":
                        response_data.update(
                            {
                                "recognized": True,
                                "action": "accept",
                                "sid": target.sid,
                                "status": target.status,
                                "reason": "Already accepted",
                            }
                        )
                        return
                    target.status = "accepted"
                    response_data.update(
                        {
                            "recognized": True,
                            "action": "accept",
                            "sid": target.sid,
                            "status": target.status,
                            "reason": f"{reason} (was {prev})",
                        }
                    )
                else:
                    target.status = "accepted"
                    response_data.update(
                        {
                            "recognized": True,
                            "action": "accept",
                            "sid": target.sid,
                            "status": target.status,
                            "reason": reason,
                        }
                    )
                self._persist_feedback_event(
                    {
                        "ts": time.time(),
                        "type": "suggestion_feedback",
                        "sid": target.sid,
                        "status": "accepted",
                        "feedback_reason": reason,
                    }
                )
            else:
                response_data.update(
                    {
                        "recognized": True,
                        "action": "accept",
                        "status": "no_target",
                        "reason": reason,
                    }
                )

        # Interrupt handling first
        if interrupt_pat.search(user_text):
            response_data.update({"recognized": True, "action": "interrupt"})
            action_text = "Interrupt requested."
            self._append_message(ChatMessage(role="user", text=user_text))
            self._append_message(ChatMessage(role="assistant", text=action_text, data=response_data))
            return response_data

        # Accept/apply branch
        if yes_pat.search(lower):
            if not has_pending and not target:
                action_text = "No pending suggestion found to apply. Create one or include a valid 8-hex suggestion SID."
                response_data.update({"recognized": True, "action": "accept", "status": "no_target", "reason": action_text})
                self._append_message(ChatMessage(role="user", text=user_text))
                self._append_message(ChatMessage(role="assistant", text=action_text, data=response_data))
                return response_data

            sid_info = f" for {target.sid}" if target and target.sid else ""
            action_text = f"Accepted{sid_info}."
            _accept_target("User approved/apply command")
            self._append_message(ChatMessage(role="user", text=user_text))
            self._append_message(ChatMessage(role="assistant", text=action_text, data=response_data))
            return response_data

        # Reject branch
        if no_pat.search(lower):
            if not has_pending and not target:
                action_text = "No pending suggestion found to reject. If you meant a specific suggestion, provide its 8-hex SID."
                response_data.update({"recognized": True, "action": "reject", "status": "no_target", "reason": action_text})
                self._append_message(ChatMessage(role="user", text=user_text))
                self._append_message(ChatMessage(role="assistant", text=action_text, data=response_data))
                return response_data
            action_text = "Rejected the current suggestion."
            _reject_target("User rejected/stop/no command")
            self._append_message(ChatMessage(role="user", text=user_text))
            self._append_message(ChatMessage(role="assistant", text=action_text, data=response_data))
            return response_data

        # Explicit feedback content (non-accept/reject): record a numeric delta or boolean good/bad
        # If a sid is not provided and there is a pending suggestion, attach to the active one.
        if target and (good_pat.search(user_text) or bad_pat.search(user_text) or score_m or ratio_m):
            computed_score: Optional[float] = None
            note = []

            if score_m:
                try:
                    computed_score = float(score_m.group(1))
                    note.append(f"score={computed_score}")
                except Exception:
                    computed_score = None

            if computed_score is None and ratio_m:
                try:
                    a = float(ratio_m.group(1))
                    b = float(ratio_m.group(2))
                    if b != 0:
                        computed_score = a / b
                        note.append(f"ratio={a}/{b}")
                except Exception:
                    computed_score = None

            if computed_score is None:
                if good_pat.search(user_text):
                    computed_score = 1.0
                    note.append("good=1.0")
                elif bad_pat.search(user_text):
                    computed_score = 0.0
                    note.append("bad=0.0")

            if computed_score is not None:
                response_data.update(
                    {
                        "recognized": True,
                        "action": "feedback",
                        "sid": target.sid,
                        "status": target.status,
                        "score": computed_score,
                        "note": ", ".join(note)[:200],
                    }
                )
                self._persist_feedback_event(
                    {
                        "ts": time.time(),
                        "type": "suggestion_feedback_metric",
                        "sid": target.sid,
                        "status": target.status,
                        "score": computed_score,
                        "note": ", ".join(note)[:200],
                    }
                )
                action_text = f"Recorded feedback for {target.sid}."
            else:
                response_data.update({"recognized": False, "action": None})
                action_text = "Couldn't parse feedback score; try 'score 0.7' or 'good/bad'."

            self._append_message(ChatMessage(role="user", text=user_text))
            self._append_message(ChatMessage(role="assistant", text=action_text, data=response_data))
            return response_data

        # Suggestion listing / help / state queries
        help_pat = re.compile(r"\b(?:help|what can i say|commands|how to use)\b", re.IGNORECASE)
        state_pat = re.compile(r"\b(?:state|status|pending|suggestions)\b", re.IGNORECASE)
        if help_pat.search(user_text):
            action_text = (
                "Commands: 'apply/yes' to accept, 'reject/no' to reject, "
                "'interrupt' to stop. For metrics: 'score 0.7' or 'good/bad'."
            )
            response_data.update({"recognized": True, "action": "help"})
            self._append_message(ChatMessage(role="user", text=user_text))
            self._append_message(ChatMessage(role="assistant", text=action_text, data=response_data))
            return response_data

        if state_pat.search(user_text) and not yes_pat.search(lower) and not no_pat.search(lower):
            st = self.get_state()
            action_text = "Current pending suggestions (see data)."
            response_data.update({"recognized": True, "action": "state", "pending": st.get("pending_suggestions", [])})
            self._append_message(ChatMessage(role="user", text=user_text))
            self._append_message(ChatMessage(role="assistant", text=action_text, data=response_data))
            return response_data

        # Default: unrecognized
        action_text = "Unrecognized command. Try 'apply' or 'reject', or provide an 8-hex suggestion SID."
        response_data.update({"recognized": False, "action": None, "reason": action_text})
        self._append_message(ChatMessage(role="user", text=user_text))
        self._append_message(ChatMessage(role="assistant", text=action_text, data=response_data))
        return response_data


# ── Singletons ────────────────────────────────────────────────────────────────

_LOG_BUFFER: Optional[EvolverLogBuffer] = None
_EVOLVER_CHAT: Optional[EvolverChat] = None
_SINGLETON_LOCK = threading.Lock()


def get_log_buffer() -> EvolverLogBuffer:
    global _LOG_BUFFER
    with _SINGLETON_LOCK:
        if _LOG_BUFFER is None:
            _LOG_BUFFER = EvolverLogBuffer()
        return _LOG_BUFFER


def get_evolver_chat() -> EvolverChat:
    global _EVOLVER_CHAT
    with _SINGLETON_LOCK:
        if _EVOLVER_CHAT is None:
            _EVOLVER_CHAT = EvolverChat()
        return _EVOLVER_CHAT