"""
SARE-HX LLM Bridge (Tier 5 — Epic 19 & 20)

Connects an LLM (Gemini by default) to the SARE-HX reasoning core.

Responsibilities:
  - Parse free-form natural language into a structured problem description
    that SARE-HX can consume (Epic 19: LLM Parser Bridge)
  - Translate a structured proof trace back into plain English explanation
    (Epic 20: LLM Explanation Writer)

Design:
  - Zero hard dependencies: graceful fallback if API key is missing or network fails
  - Provider-agnostic: supports Gemini and OpenAI via the same interface
  - Async-safe: uses urllib (stdlib) to avoid extra dependencies
"""

from __future__ import annotations

import os
import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

# ── Config loader ─────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "llm.json"
_config: Optional[dict] = None


def _load_config(force_reload: bool = False) -> dict:
    global _config
    if _config is not None and not force_reload:
        return _config
    try:
        with open(_CONFIG_PATH) as f:
            _config = json.load(f)
    except Exception:
        _config = {
            "provider": "gemini",
            "model": "gemini-2.5-flash-preview-04-17",
            "api_key_env": "GEMINI_API_KEY",
            "temperature": 0.1,
            "max_tokens": 1024,
        }
    return _config


def _resolve_api_key(cfg: dict) -> str:
    """Resolve API key: prefer explicit `api_key` in config, then fall back to env var."""
    # Direct key in config (user pasted key)
    direct = cfg.get("api_key", "").strip()
    if direct and direct.startswith("AIza"):
        return direct
    # Environment variable fallback
    env_var = cfg.get("api_key_env", "GEMINI_API_KEY")
    return os.environ.get(env_var, "")


# ── Raw API caller ─────────────────────────────────────────────────────────────

def _call_gemini(prompt: str, model: str, api_key: str, temperature: float, max_tokens: int) -> str:
    """Call Gemini generateContent REST endpoint. Returns raw text."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    import logging
    logging.getLogger(__name__).debug(f"RAW API JSON: {json.dumps(body)}")
    try:
        return body["candidates"][0]["content"]["parts"][0]["text"]
    except KeyError:
        return f"Error parsing response: {json.dumps(body)}"


def _call_openai(prompt: str, model: str, api_key: str, temperature: float, max_tokens: int) -> str:
    """Call OpenAI chat completions endpoint. Returns raw text."""
    url = "https://api.openai.com/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    })
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"]


def _call_llm(prompt: str) -> str:
    """Dispatch to the configured LLM. Returns the text response."""
    cfg = _load_config(force_reload=True)  # always re-read so live key changes take effect
    api_key = _resolve_api_key(cfg)
    if not api_key:
        env_var = cfg.get("api_key_env", "GEMINI_API_KEY")
        raise RuntimeError(
            f"LLM API key not set. Either set config 'api_key' or export {env_var}=<your_key>"
        )

    provider = cfg.get("provider", "gemini")
    model = cfg.get("model", "gemini-2.5-flash-preview-04-17")
    temperature = float(cfg.get("temperature", 0.1))
    max_tokens = int(cfg.get("max_tokens", 1024))

    if provider == "gemini":
        return _call_gemini(prompt, model, api_key, temperature, max_tokens)
    elif provider == "openai":
        return _call_openai(prompt, model, api_key, temperature, max_tokens)
    else:
        raise RuntimeError(f"Unknown LLM provider: {provider}")


# ── Epic 19: LLM Parser Bridge ────────────────────────────────────────────────

_DEFAULT_PARSE_PROMPT = (
    "You are a symbolic math/logic parser. Convert the user's problem into a structured "
    "JSON object with fields: {\"expression\": \"<canonical form>\", \"domain\": "
    "\"algebra|logic|arithmetic|code|general\", \"goal\": \"simplify|prove|evaluate|optimize\"}. "
    "Return ONLY valid JSON, no markdown, no explanation.\n\n"
    "Examples:\n"
    "- \"What is 3x + 0?\" → {\"expression\": \"3*x + 0\", \"domain\": \"algebra\", \"goal\": \"simplify\"}\n"
    "- \"Is A AND NOT A true?\" → {\"expression\": \"A & (~A)\", \"domain\": \"logic\", \"goal\": \"evaluate\"}\n"
    "- \"Simplify (x+1)^2 - x^2 - 2x\" → {\"expression\": \"(x+1)^2 - x^2 - 2*x\", \"domain\": \"algebra\", \"goal\": \"simplify\"}\n\n"
    "User problem: {PROBLEM}"
)


class ParsedProblem:
    """Result of LLM-driven natural language parsing."""
    def __init__(self, expression: str, domain: str, goal: str, raw_input: str):
        self.expression = expression
        self.domain = domain
        self.goal = goal
        self.raw_input = raw_input

    def to_dict(self) -> dict:
        return {
            "expression": self.expression,
            "domain": self.domain,
            "goal": self.goal,
            "raw_input": self.raw_input,
        }


def parse_nl_problem(nl_text: str) -> ParsedProblem:
    """
    Convert a free-form natural language problem into a structured ParsedProblem.
    On failure (no API key, network error, bad JSON), returns a best-effort fallback.
    """
    cfg = _load_config()
    prompt_template = cfg.get("parse_prompt", _DEFAULT_PARSE_PROMPT)
    prompt = prompt_template.replace("{{PROBLEM}}", nl_text).replace("{PROBLEM}", nl_text)

    try:
        raw = _call_llm(prompt)
        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip("`").strip()
        data = json.loads(raw)
        return ParsedProblem(
            expression=str(data.get("expression", nl_text)),
            domain=str(data.get("domain", "general")),
            goal=str(data.get("goal", "simplify")),
            raw_input=nl_text,
        )
    except Exception as e:
        print(f"[LLMBridge] parse_nl_problem failed: {e} — falling back to raw input")
        # Heuristic fallback: use the raw text as the expression
        domain = "algebra" if any(c in nl_text for c in "x y z + - * / ^ =") else "general"
        return ParsedProblem(
            expression=nl_text,
            domain=domain,
            goal="simplify",
            raw_input=nl_text,
        )

# ── Epic 24: Metacognitive Loop ───────────────────────────────────────────────

_DEFAULT_PLAN_PROMPT = (
    "You are the pre-frontal cortex of SARE-HX, a neuro-symbolic reasoning engine. "
    "You must break the given user problem down into a sequence of atomic mathematical or logical sub-goals. "
    "The reasoning engine will execute these sub-goals one by one to avoid combinatorial explosion.\n\n"
    "Return ONLY a JSON array of strings, where each string is a clear, actionable directive.\n"
    "Do NOT include markdown formatting or explanations.\n\n"
    "Examples:\n"
    "- \"Solve 2y + 1 = 2x - 3 for y\" → [\"Subtract 1 from both sides of the equation\", \"Divide both sides by 2\"]\n"
    "- \"Prove (A OR B) AND (NOT A) implies B\" → [\"Distribute the AND over the OR\", \"Apply the law of non-contradiction to (A AND NOT A)\", \"Simplify to B\"]\n\n"
    "User problem: {PROBLEM}"
)

def plan_subgoals(problem: str) -> list[str]:
    """Calls the LLM to generate a sequence of logic sub-goals."""
    cfg = _load_config()
    prompt_tpl = cfg.get("plan_prompt", _DEFAULT_PLAN_PROMPT)
    prompt = prompt_tpl.replace("{PROBLEM}", problem)
    
    import logging
    log = logging.getLogger(__name__)
    log.info(f"LLM Planning sub-goals for: {problem[:50]}...")
    
    result_text = _call_llm(prompt)
    
    try:
        import re
        m = re.search(r'\[.*\]', result_text, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
        else:
            data = json.loads(result_text)
            
        if isinstance(data, list):
            return data
        else:
            return [str(problem)] # Fallback
    except Exception as e:
        log.warning(f"LLM Failed to return valid JSON array for plan. Error: {e}\nRaw={result_text}")
        return [str(problem)] # Fallback to a single step plan


# ── Epic 20: LLM Explanation Writer ──────────────────────────────────────────

_DEFAULT_EXPLAIN_PROMPT = (
    "You are a math/logic tutor explaining a specific solved problem to a student.\n"
    "You MUST only describe what is shown below. Do NOT invent steps, algorithms, "
    "or context that are not present. Do NOT mention TSP, sorting, graphs, or anything "
    "not in the problem. If transforms are 'none', say the expression was already in its "
    "simplest form or that the system recognized an identity directly.\n\n"
    "=== FACTS (use ONLY these) ===\n"
    "Original problem: {PROBLEM}\n"
    "Mathematical expression: {EXPRESSION}\n"
    "Domain: {DOMAIN}\n"
    "Goal: {GOAL}\n"
    "Transforms applied: {TRANSFORMS}\n"
    "Complexity score before: {E_BEFORE}\n"
    "Complexity score after:  {E_AFTER}\n"
    "Reduction: {DELTA} ({PCT}% simpler)\n"
    "=== END FACTS ===\n\n"
    "Write exactly 2-3 clear sentences explaining what happened to THIS expression and why. "
    "Use the domain and goal to guide your language. Be concrete and specific."
)


def explain_solve_trace(
    problem: str,
    transforms_applied: list,
    energy_before: float,
    energy_after: float,
    final_expression: str,
    expression: str = "",
    domain: str = "general",
    goal: str = "simplify",
) -> str:
    """
    Translate a SARE-HX solve trace into plain English via LLM.
    Returns the explanation string. Falls back gracefully to a structured summary.
    """
    cfg = _load_config()
    prompt_template = cfg.get("explain_prompt", _DEFAULT_EXPLAIN_PROMPT)

    delta = round(energy_before - energy_after, 3)
    pct = round(delta / energy_before * 100, 1) if energy_before > 0 else 0.0
    transforms_str = " → ".join(transforms_applied) if transforms_applied else "none (identity was recognized directly)"
    expr_str = expression or final_expression or problem

    prompt = (
        prompt_template
        .replace("{PROBLEM}", problem)
        .replace("{EXPRESSION}", expr_str)
        .replace("{DOMAIN}", domain)
        .replace("{GOAL}", goal)
        .replace("{TRANSFORMS}", transforms_str)
        .replace("{E_BEFORE}", str(round(energy_before, 3)))
        .replace("{E_AFTER}", str(round(energy_after, 3)))
        .replace("{DELTA}", str(delta))
        .replace("{PCT}", str(pct))
        .replace("{ANSWER}", final_expression)
        # legacy placeholders
        .replace("{{TRACE}}", transforms_str)
        .replace("{{ANSWER}}", final_expression)
        .replace("{{DELTA}}", str(delta))
        .replace("{{TRANSFORMS}}", transforms_str)
    )

    try:
        explanation = _call_llm(prompt)
        return explanation.strip()
    except Exception as e:
        print(f"[LLMBridge] explain_solve_trace failed: {e} — using structured fallback")
        action = "simplified" if transforms_applied else "recognized as an identity"
        return (
            f"The {domain} expression '{expr_str}' was {action}. "
            f"The complexity score dropped from {round(energy_before, 3)} to {round(energy_after, 3)} "
            f"({pct}% reduction), confirming the expression is now in {'a simpler' if transforms_applied else 'its simplest'} form."
        )


# ── Health check ──────────────────────────────────────────────────────────────

def llm_available() -> bool:
    """Returns True if a valid API key is configured."""
    cfg = _load_config(force_reload=True)
    return bool(_resolve_api_key(cfg))


def llm_status() -> dict:
    """Returns a status dict for the /api/llm-status endpoint."""
    cfg = _load_config(force_reload=True)
    key = _resolve_api_key(cfg)
    available = bool(key)
    return {
        "available": available,
        "provider": cfg.get("provider", "gemini"),
        "model": cfg.get("model", "gemini-2.5-flash-preview-04-17"),
        "api_key_env": cfg.get("api_key_env", "GEMINI_API_KEY"),
        "api_key_set": available,
        "api_key_source": "config" if cfg.get("api_key", "").startswith("AIza") else "env_var",
    }
