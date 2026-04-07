"""
CodeExecutor — sandboxed Python execution for SARE.

Uses subprocess for full process isolation. Pre-filters dangerous patterns
via regex before spawning any subprocess. Safe to call from daemon threads.

Usage:
    from sare.execution.code_executor import get_executor
    result = get_executor().execute("print(2 + 2)")
    print(result.stdout)   # "4\\n"
"""
from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

# ── Safety filter ──────────────────────────────────────────────────────────────
# Blocked patterns: OS access, file I/O, network, dynamic code, introspection
_DANGER_RE = re.compile(
    r"\b(os|sys|subprocess|socket|open|eval|exec|compile"
    r"|__import__|importlib|shutil|pathlib|ctypes|signal"
    r"|threading|multiprocessing|builtins|globals|locals|vars)\b"
    r"|__[a-z_]+__\s*\(",          # dunder calls like __class__()
    re.IGNORECASE,
)
# Any import statement (import X / from X import Y)
_IMPORT_RE = re.compile(r"^\s*(import\s+|from\s+\S+\s+import)", re.MULTILINE)

_TIMEOUT_S = 5
_MAX_BYTES  = 32_768   # 32 KB output cap


# ── Result dataclass ───────────────────────────────────────────────────────────
@dataclass
class ExecutionResult:
    stdout:       str
    stderr:       str
    exit_code:    int
    elapsed_ms:   float
    blocked:      bool
    block_reason: str = ""
    timed_out:    bool = False


# ── Executor ───────────────────────────────────────────────────────────────────
class CodeExecutor:
    """Sandboxed Python 3 executor using subprocess for full process isolation."""

    def __init__(self, timeout_s: int = _TIMEOUT_S):
        self._timeout_s = timeout_s

    # ── Public API ─────────────────────────────────────────────────────────────

    def execute(self, code: str) -> ExecutionResult:
        """
        Execute Python code in a subprocess sandbox.
        Blocks until result or timeout. Never raises.
        """
        if not code or not code.strip():
            return ExecutionResult(stdout="", stderr="", exit_code=0,
                                   elapsed_ms=0, blocked=False)
        blocked, reason = self._is_dangerous(code)
        if blocked:
            return ExecutionResult(stdout="", stderr=reason, exit_code=-1,
                                   elapsed_ms=0, blocked=True, block_reason=reason)
        return self._run_subprocess(code)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _is_dangerous(self, code: str) -> tuple[bool, str]:
        if _IMPORT_RE.search(code):
            return True, "imports are not allowed in sandbox"
        m = _DANGER_RE.search(code)
        if m:
            return True, f"blocked: '{m.group()}' is not allowed in sandbox"
        return False, ""

    def _run_subprocess(self, code: str) -> ExecutionResult:
        t0 = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
                # Minimal env — no PATH, no HOME, no secrets
                env={"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"},
            )
            stdout = proc.stdout[:_MAX_BYTES]
            stderr = proc.stderr[:_MAX_BYTES]
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            stdout = ""
            stderr = f"Execution timed out after {self._timeout_s}s"
            exit_code = 124
            timed_out = True
        except Exception as e:
            stdout = ""
            stderr = str(e)
            exit_code = -2
        elapsed_ms = (time.monotonic() - t0) * 1000
        return ExecutionResult(
            stdout=stdout, stderr=stderr, exit_code=exit_code,
            elapsed_ms=elapsed_ms, blocked=False, timed_out=timed_out,
        )


# ── Singleton ──────────────────────────────────────────────────────────────────
_instance: Optional[CodeExecutor] = None
_lock = threading.Lock()


def get_executor() -> CodeExecutor:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = CodeExecutor()
    return _instance
