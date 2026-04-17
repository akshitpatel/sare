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
            "provider": "lmstudio",
            "lmstudio_url": "http://localhost:1234",
            "model": "local-model",
            "temperature": 0.1,
            "max_tokens": 1024,
        }
    return _config


def _resolve_api_key(cfg: dict) -> str:
    """Resolve API key: prefer explicit `api_key` in config, then fall back to env var."""
    direct = cfg.get("api_key", "").strip()
    if direct:
        return direct
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
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"]


# ── Rate limiter: minimum gap between successive _call_model calls ───────────
# Prevents 429 bursts when the self-improver fires proposer→planner→critic→judge
# in rapid succession. Minimum 3 s between calls (adjustable via config).
_MIN_CALL_GAP_S: float = 3.0
_last_call_ts: float = 0.0
_rate_lock = __import__("threading").Lock()


def _rate_limit_wait() -> None:
    """Block until at least _MIN_CALL_GAP_S has elapsed since the last call."""
    global _last_call_ts
    with _rate_lock:
        now = time.monotonic()
        wait = _MIN_CALL_GAP_S - (now - _last_call_ts)
        if wait > 0:
            time.sleep(wait)
        _last_call_ts = time.monotonic()


_COST_PER_M = {
    # input_$/M, output_$/M
    "stepfun/step-3.5-flash:free":    (0.0,   0.0),
    "openrouter/hunter-alpha":         (0.0,   0.0),
    "openrouter/healer-alpha":         (0.0,   0.0),
    "x-ai/grok-4.1-fast":             (0.0,   0.0),
    "deepseek/deepseek-v3.2":         (0.26,  0.38),
    "x-ai/grok-4.20-multi-agent-beta":(0.0,   0.0),   # price TBD — treating as free until confirmed
    "minimax/minimax-m2.5":           (0.25,  1.20),
    "openai/gpt-5.4-nano":            (0.30,  1.20),
    "google/gemini-3.1-pro-preview":  (2.00, 12.00),
    "anthropic/claude-sonnet-4.6":    (3.00, 15.00),
}

# Global cost ledger — written to by every _call_openrouter call
_cost_ledger: list = []   # [{"model", "role", "in_tok", "out_tok", "cost_usd"}]


def _record_cost(model: str, role: str, in_tok: int, out_tok: int):
    in_price, out_price = _COST_PER_M.get(model, (0.01, 0.03))
    cost = (in_tok * in_price + out_tok * out_price) / 1_000_000
    entry = {"model": model, "role": role, "in_tok": in_tok, "out_tok": out_tok, "cost_usd": round(cost, 6)}
    _cost_ledger.append(entry)
    import logging
    logging.getLogger(__name__).info(
        "[COST] %-40s %-14s in=%6d out=%5d  $%.4f",
        model, role or "—", in_tok, out_tok, cost,
    )
    return entry


def get_cost_summary() -> dict:
    """Return aggregated cost breakdown from the in-memory ledger."""
    total = sum(e["cost_usd"] for e in _cost_ledger)
    by_model: dict = {}
    by_role:  dict = {}
    for e in _cost_ledger:
        by_model.setdefault(e["model"], {"calls": 0, "in_tok": 0, "out_tok": 0, "cost_usd": 0.0})
        by_model[e["model"]]["calls"]    += 1
        by_model[e["model"]]["in_tok"]   += e["in_tok"]
        by_model[e["model"]]["out_tok"]  += e["out_tok"]
        by_model[e["model"]]["cost_usd"] += e["cost_usd"]
        by_role.setdefault(e["role"], {"calls": 0, "cost_usd": 0.0})
        by_role[e["role"]]["calls"]    += 1
        by_role[e["role"]]["cost_usd"] += e["cost_usd"]
    return {"total_usd": round(total, 6), "by_model": by_model, "by_role": by_role,
            "calls": len(_cost_ledger), "ledger": _cost_ledger}


def _call_openrouter(prompt: str, model: str, api_key: str, temperature: float, max_tokens: int,
                     include_reasoning: bool = False, system_prompt: str = "",
                     _role: str = "") -> str:
    """Call OpenRouter (OpenAI-compatible) chat completions endpoint."""
    import logging as _logging
    _llm_log = _logging.getLogger(__name__)

    url = "https://openrouter.ai/api/v1/chat/completions"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    body_dict: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if not include_reasoning:
        body_dict["include_reasoning"] = False

    # ── Log outgoing message ──────────────────────────────────────────────
    short_model = model.split("/")[-1]
    sys_snippet = system_prompt[:120] + "…" if len(system_prompt) > 120 else system_prompt
    # Add caller information for debugging excessive LLM calls
    import traceback
    caller = traceback.extract_stack()[-4].name if len(traceback.extract_stack()) >= 4 else "unknown"
    _llm_log.info(
        "→ SEND [%s / %s]  sys=%d chars  prompt=%d chars  caller=%s",
        short_model, _role or "—", len(system_prompt), len(prompt), caller,
        extra={
            "llm_dir":    "send",
            "llm_model":  model,
            "llm_role":   _role or "—",
            "llm_system": system_prompt,
            "llm_prompt": prompt,
        },
    )

    payload = json.dumps(body_dict).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost:8080",
        "X-Title": "SARE-HX",
    })
    # Exponential backoff on 429 (rate limit) — up to 4 retries
    _MAX_RETRIES = 4
    _backoff = 15  # seconds before first retry
    body = None
    for _attempt in range(_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.loads(resp.read())
            break  # success
        except urllib.error.HTTPError as _he:
            if _he.code == 429 and _attempt < _MAX_RETRIES:
                retry_after = int(_he.headers.get("Retry-After", _backoff))
                wait = max(retry_after, _backoff)
                _llm_log.warning(
                    "429 rate-limited [%s / %s], waiting %ds (attempt %d/%d)",
                    short_model, _role or "—", wait, _attempt + 1, _MAX_RETRIES,
                )
                time.sleep(wait)
                _backoff = min(_backoff * 2, 120)  # cap at 2 min
            else:
                raise
    if body is None:
        raise RuntimeError(f"No response after {_MAX_RETRIES} retries (429)")

    # Record token usage
    usage   = body.get("usage", {})
    in_tok  = usage.get("prompt_tokens", len(prompt) // 4)
    out_tok = usage.get("completion_tokens", 100)
    _record_cost(model, _role, in_tok, out_tok)
    response_text = body["choices"][0]["message"]["content"]

    # ── Log incoming response ─────────────────────────────────────────────
    in_price, out_price = _COST_PER_M.get(model, (0.01, 0.03))
    cost_usd = (in_tok * in_price + out_tok * out_price) / 1_000_000
    _llm_log.info(
        "← RECV [%s / %s]  in=%d out=%d  $%.5f  response=%d chars",
        short_model, _role or "—", in_tok, out_tok, cost_usd, len(response_text),
        extra={
            "llm_dir":      "recv",
            "llm_model":    model,
            "llm_role":     _role or "—",
            "llm_response": response_text,
            "llm_in_tok":   in_tok,
            "llm_out_tok":  out_tok,
            "llm_cost":     round(cost_usd, 6),
        },
    )
    return response_text


def _call_lmstudio(prompt: str, model: str, temperature: float, max_tokens: int, system_prompt: str = "") -> str:
    """Call LM Studio OpenAI-compatible API (default port 1234)."""
    cfg = _load_config()
    base_url = cfg.get("lmstudio_url", "http://localhost:1234").rstrip("/")
    url = f"{base_url}/v1/chat/completions"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def _call_ollama(prompt: str, model: str, temperature: float, max_tokens: int, system_prompt: str = "") -> str:
    """Call local Ollama /api/chat.

    Uses /api/generate (not /api/chat) to avoid message-format issues.
    think:false is only sent for qwen3/deepseek-r1 thinking models.
    Timeout is 300s to allow cold-start model loading.
    """
    url = "http://localhost:11434/api/generate"
    full_prompt = (system_prompt + "\n\n" + prompt).strip() if system_prompt else prompt

    # Only send think:false for reasoning models that support it
    _thinking_models = ("qwen3", "deepseek-r1", "qwq", "marco-o1")
    is_thinking = any(t in model.lower() for t in _thinking_models)

    body: dict = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if is_thinking:
        body["think"] = False

    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    return data.get("response", "")


def _call_model(prompt: str, role: str = "", model_override: str = "",
                system_prompt: str = "") -> str:
    """Call a specific model by role name (reads from evolve_models config) or direct model ID.

    Role names: prescreen, alt_proposer, proposer, critic, judge, judge_fallback, verifier, test_gen
    Falls back to default model if role/model not found.
    """
    _rate_limit_wait()  # enforce min gap to avoid 429 bursts
    cfg = _load_config(force_reload=True)
    provider = cfg.get("provider", "lmstudio")
    api_key = _resolve_api_key(cfg)
    temperature = float(cfg.get("temperature", 0.1))

    if model_override:
        model = model_override
    elif role:
        model = cfg.get("evolve_models", {}).get(role, cfg.get("model", "openrouter/hunter-alpha"))
    else:
        model = cfg.get("model", "openrouter/hunter-alpha")

    # Token budget: free/cheap models get less, expensive ones get more
    _cheap = {"stepfun/step-3.5-flash:free", "openrouter/hunter-alpha", "x-ai/grok-4.1-fast"}
    _large = {"anthropic/claude-sonnet-4.6", "google/gemini-3.1-pro-preview", "openai/gpt-5.4-nano"}
    if model in _cheap:
        max_tokens = min(int(cfg.get("max_tokens", 2048)), 2048)
    elif model in _large:
        max_tokens = int(cfg.get("max_tokens", 8192))
    else:
        max_tokens = 4096

    if provider == "lmstudio":
        return _call_lmstudio(prompt, model, temperature, max_tokens, system_prompt=system_prompt)

    if provider == "ollama":
        return _call_ollama(prompt, model, temperature, max_tokens)

    if not api_key:
        return _call_llm(prompt)  # fallback

    fallback_model = cfg.get("fallback_model", "")

    if provider in ("openrouter", "gemini", "openai"):
        try:
            if provider == "openrouter":
                return _call_openrouter(prompt, model, api_key, temperature, max_tokens,
                                        system_prompt=system_prompt, _role=role or model_override)
            elif provider == "gemini":
                return _call_gemini(prompt, model, api_key, temperature, max_tokens)
            else:
                return _call_openai(prompt, model, api_key, temperature, max_tokens)
        except Exception as _primary_err:
            if fallback_model and fallback_model != model:
                import logging as _log
                _log.getLogger(__name__).warning(
                    "[llm_bridge] Primary model %s failed (%s) — retrying with fallback %s",
                    model, _primary_err, fallback_model,
                )
                time.sleep(2)
                return _call_openrouter(prompt, fallback_model, api_key, temperature,
                                        max_tokens, system_prompt=system_prompt,
                                        _role=f"{role or '?'}[fallback]")
            raise
    return _call_llm(prompt)


def _call_fast_llm(prompt: str) -> str:
    """Call the configured fast/free model (stepfun:free by default).

    Uses _call_model with role='prescreen' as the cheapest routing.
    """
    return _call_model(prompt, role="prescreen")


def _call_llm(prompt: str, use_synthesis_model: bool = False, system_prompt: str = "",
              max_tokens_override: Optional[int] = None, **kwargs) -> str:
    """Dispatch to the configured LLM. Returns the text response.

    use_synthesis_model=True uses the larger `synthesis_model` from config.
    system_prompt is injected as a system-role message when provided.
    max_tokens_override overrides the config value (e.g. for synthesis calls needing more tokens).
    """
    # T2-5: Few-shot adaptation — enrich prompt with relevant solved examples
    try:
        from sare.learning.few_shot_adapter import get_few_shot_adapter
        _adapter = get_few_shot_adapter()
        if _adapter and len(_adapter._examples) >= 5:
            _fsa_domain = kwargs.get("domain", "general")
            # Infer domain from prompt content when not explicitly provided
            if _fsa_domain == "general":
                _kw_map = {
                    "calculus":     ["derivative", "integral", "d/dx", "chain rule"],
                    "trigonometry": ["sin(", "cos(", "tan(", "trig"],
                    "logic":        [" and ", " or ", " not ", "implies", "boolean"],
                    "algebra":      ["equation", "solve", "polynomial", "factor"],
                    "chemistry":    ["mol", "reaction", "compound", "element", "pv=nrt"],
                    "physics":      ["force", "velocity", "momentum", "energy", "f=ma"],
                }
                _pl = prompt.lower()
                for _dk, _kws in _kw_map.items():
                    if any(k in _pl for k in _kws):
                        _fsa_domain = _dk
                        break
            prompt = _adapter.enrich_prompt(prompt, domain=_fsa_domain)
    except Exception:
        pass

    cfg = _load_config(force_reload=True)  # always re-read so live key changes take effect
    provider = cfg.get("provider", "lmstudio")
    default_model = cfg.get("model", "gemini-2.5-flash-preview-04-17")
    synthesis_model = cfg.get("synthesis_model", default_model)
    model = synthesis_model if use_synthesis_model else default_model
    temperature = float(cfg.get("temperature", 0.1))
    max_tokens = max_tokens_override if max_tokens_override else int(cfg.get("max_tokens", 1024))

    if provider == "lmstudio":
        return _call_lmstudio(prompt, model, temperature, max_tokens, system_prompt=system_prompt)

    if provider == "ollama":
        return _call_ollama(prompt, model, temperature, max_tokens, system_prompt=system_prompt)

    api_key = _resolve_api_key(cfg)
    if not api_key:
        env_var = cfg.get("api_key_env", "GEMINI_API_KEY")
        raise RuntimeError(
            f"LLM API key not set. Either set config 'api_key' or export {env_var}=<your_key>"
        )

    fallback_model       = cfg.get("fallback_model", "")         # openai/gpt-oss-120b
    final_fallback_model = cfg.get("final_fallback_model", "")   # deepseek/deepseek-v3.2

    def _dispatch(m: str) -> str:
        if provider == "gemini":
            return _call_gemini(prompt, m, api_key, temperature, max_tokens)
        elif provider == "openai":
            return _call_openai(prompt, m, api_key, temperature, max_tokens)
        elif provider == "openrouter":
            return _call_openrouter(prompt, m, api_key, temperature, max_tokens, system_prompt=system_prompt)
        else:
            raise RuntimeError(f"Unknown LLM provider: {provider}")

    def _is_retryable(err: Exception) -> bool:
        s = str(err)
        return any(code in s for code in ("400", "401", "403", "404", "422", "429", "500", "502", "503"))

    import logging as _log
    _bridge_log = _log.getLogger(__name__)

    # Tier 1: primary (step-3.5-flash)
    try:
        return _dispatch(model)
    except Exception as primary_err:
        if not _is_retryable(primary_err):
            raise

    # Tier 2: fallback (gpt-oss-120b)
    if fallback_model and fallback_model != model:
        _bridge_log.warning("[llm_bridge] %s failed → trying fallback %s", model, fallback_model)
        time.sleep(1)
        try:
            return _dispatch(fallback_model)
        except Exception as fallback_err:
            if not _is_retryable(fallback_err):
                raise

    # Tier 3: final fallback (deepseek-v3.2)
    if final_fallback_model and final_fallback_model not in (model, fallback_model):
        _bridge_log.warning("[llm_bridge] %s failed → using final fallback %s", fallback_model or model, final_fallback_model)
        time.sleep(2)
        return _dispatch(final_fallback_model)

    raise RuntimeError(f"All LLM tiers exhausted for model={model}")


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


# ── Phase A: Language Grounding — Concept Extraction ─────────────────────────

def extract_concepts(text: str) -> list:
    """
    Extract named concepts and relations from natural language text.
    Returns a list of dicts: [{"concept": str, "type": str, "relations": [...]}]
    Falls back to empty list on any failure.

    Used by Phase A (language grounding) to populate ConceptRegistry from text.
    """
    prompt = (
        "Extract mathematical and logical concepts from the following text.\n"
        "Return a JSON array where each element is:\n"
        '{"concept": "concept_name", "type": "operator|rule|identity|property|domain", '
        '"relations": ["related_concept_1", "related_concept_2"]}\n'
        "Focus on: operators, identities, rules, mathematical properties.\n"
        "Return ONLY valid JSON array, no markdown.\n\n"
        f"Text: {text}"
    )
    try:
        raw = _call_llm(prompt)
        raw = raw.strip().strip("`").lstrip("json").strip()
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            concepts = json.loads(m.group(0))
            return [c for c in concepts if isinstance(c, dict) and "concept" in c]
        return []
    except Exception:
        return []


def ingest_text_for_world_model(text: str, domain: str = "general") -> dict:
    """
    Parse free-form text and inject beliefs + causal links into the world model.
    Returns {"facts_added": int, "links_added": int}.
    Used by Phase B (grounded world model) text ingestion.
    """
    prompt = (
        f"Extract causal facts from this {domain} text for a knowledge base.\n"
        "Return a JSON object with:\n"
        '{"facts": ["fact1", "fact2", ...], '
        '"causal_links": [{"cause": "...", "effect": "...", "mechanism": "..."}, ...]}\n'
        "Return ONLY valid JSON.\n\n"
        f"Text: {text[:2000]}"
    )
    try:
        from sare.memory.world_model import get_world_model
        wm = get_world_model()
        raw = _call_llm(prompt)
        raw = raw.strip().strip("`").lstrip("json").strip()
        data = json.loads(raw)

        facts_added = 0
        links_added = 0
        for fact in data.get("facts", []):
            if isinstance(fact, str) and fact.strip():
                wm.add_fact(domain, fact.strip(), 0.7, source="text_ingest")
                facts_added += 1
        for link in data.get("causal_links", []):
            if isinstance(link, dict) and "cause" in link and "effect" in link:
                wm.add_causal_link(
                    link["cause"], link["effect"],
                    link.get("mechanism", "text_ingest"),
                    domain, 0.6,
                )
                links_added += 1
        if facts_added or links_added:
            wm.save()
        return {"facts_added": facts_added, "links_added": links_added}
    except Exception as e:
        return {"facts_added": 0, "links_added": 0, "error": str(e)}


# ── Auto-learning LLM hooks ───────────────────────────────────────────────────

def learn_from_proof(
    expr: str,
    domain: str,
    proof_steps: list,
    energy_delta: float,
) -> dict:
    """
    After a successful solve, ask LLM to extract abstract knowledge from the proof.
    Returns {"facts": [...], "causal_links": [...], "variants": [...]} injected into world model.
    """
    steps_str = " → ".join(proof_steps) if proof_steps else "direct simplification"
    prompt = (
        f"A symbolic AI just solved a {domain} problem.\n"
        f"Expression: {expr}\n"
        f"Proof steps: {steps_str}\n"
        f"Energy reduction: {energy_delta:.2f}\n\n"
        "Extract reusable mathematical knowledge. Return JSON:\n"
        "{\n"
        '  "facts": ["one concise mathematical fact this demonstrates"],\n'
        '  "causal_links": [{"cause": "pattern", "effect": "simplified", "mechanism": "rule name"}],\n'
        '  "variants": ["similar expr to practice", "harder variant"]\n'
        "}\n"
        "Return ONLY valid JSON. Keep facts and links to the most general insight."
    )
    try:
        from sare.memory.world_model import get_world_model
        wm = get_world_model()
        raw = _call_llm(prompt)
        raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`")
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {}
        data = json.loads(m.group())
        facts_added, links_added = 0, 0
        for fact in data.get("facts", [])[:3]:
            if isinstance(fact, str) and fact.strip():
                wm.add_fact(domain, fact.strip(), 0.75, source="llm_proof_learn")
                facts_added += 1
        for link in data.get("causal_links", [])[:3]:
            if isinstance(link, dict) and link.get("cause") and link.get("effect"):
                wm.add_causal_link(link["cause"], link["effect"],
                                   link.get("mechanism", "proof_derived"), domain, 0.75)
                links_added += 1
        if facts_added or links_added:
            wm.save()
        return {
            "facts_added": facts_added,
            "links_added": links_added,
            "variants": [v for v in data.get("variants", []) if isinstance(v, str)][:5],
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_failures(domain: str, failed_exprs: list, known_links: list) -> dict:
    """
    Given a set of unsolved problems, ask LLM what knowledge is missing and
    generate targeted practice problems. Injects new knowledge into world model.
    Returns {"facts_added": int, "links_added": int, "new_seeds": [...], "missing_rule": str}.
    """
    links_summary = ", ".join(f"{l['cause']}→{l['effect']}" for l in known_links[:8])
    prompt = (
        f"A symbolic AI is stuck on {domain} problems it cannot solve.\n"
        f"Unsolved: {failed_exprs[:8]}\n"
        f"Current knowledge: {links_summary or 'none'}\n\n"
        "Diagnose the gap and provide remedial knowledge. Return JSON:\n"
        "{\n"
        '  "missing_rule": "name of the transform that would solve these",\n'
        '  "facts": ["mathematical fact 1", "mathematical fact 2"],\n'
        '  "causal_links": [{"cause": "expr", "effect": "simplified", "mechanism": "why"}],\n'
        '  "practice_problems": ["easier warm-up expr 1", "easier warm-up expr 2", "easier warm-up expr 3"]\n'
        "}\n"
        "Return ONLY valid JSON."
    )
    try:
        from sare.memory.world_model import get_world_model
        wm = get_world_model()
        raw = _call_llm(prompt)
        raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`")
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {}
        data = json.loads(m.group())
        facts_added, links_added = 0, 0
        for fact in data.get("facts", [])[:4]:
            if isinstance(fact, str) and fact.strip():
                wm.add_fact(domain, fact.strip(), 0.7, source="llm_failure_analysis")
                facts_added += 1
        for link in data.get("causal_links", [])[:4]:
            if isinstance(link, dict) and link.get("cause") and link.get("effect"):
                wm.add_causal_link(link["cause"], link["effect"],
                                   link.get("mechanism", "failure_analysis"), domain, 0.68)
                links_added += 1
        if facts_added or links_added:
            wm.save()
        return {
            "missing_rule": data.get("missing_rule", ""),
            "facts_added": facts_added,
            "links_added": links_added,
            "new_seeds": [p for p in data.get("practice_problems", []) if isinstance(p, str)][:5],
        }
    except Exception as e:
        return {"error": str(e)}


def reflect_and_plan(stats: dict, high_surprise_domains: list, recent_rules: list) -> dict:
    """
    Periodic LLM reflection on overall performance. Returns new knowledge and curriculum focus.
    Injects new facts/links into world model.
    """
    prompt = (
        "You are the metacognitive core of a self-learning symbolic AI (SARE-HX).\n\n"
        f"Performance stats:\n"
        f"  Solve rate: {stats.get('solve_rate', 0):.1%}\n"
        f"  Total attempted: {stats.get('total', 0)}\n"
        f"  Rules promoted: {stats.get('rules_promoted', 0)}\n"
        f"  High-surprise domains (struggling): {high_surprise_domains}\n"
        f"  Recently learned rules: {recent_rules[:6]}\n\n"
        "Reflect and provide a learning plan. Return JSON:\n"
        "{\n"
        '  "curriculum_focus": "one sentence: what to practice next",\n'
        '  "priority_domains": ["domain1", "domain2"],\n'
        '  "new_facts": [{"domain": "algebra", "fact": "..."}],\n'
        '  "new_causal_links": [{"cause": "...", "effect": "...", "mechanism": "...", "domain": "..."}],\n'
        '  "knowledge_gaps": ["gap description"]\n'
        "}\n"
        "Return ONLY valid JSON. Focus on actionable knowledge."
    )
    try:
        from sare.memory.world_model import get_world_model
        wm = get_world_model()
        raw = _call_llm(prompt)
        raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`")
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {}
        data = json.loads(m.group())
        facts_added, links_added = 0, 0
        for item in data.get("new_facts", [])[:5]:
            if isinstance(item, dict) and item.get("fact"):
                wm.add_fact(item.get("domain", "general"), item["fact"], 0.7, source="llm_reflection")
                facts_added += 1
        for link in data.get("new_causal_links", [])[:5]:
            if isinstance(link, dict) and link.get("cause") and link.get("effect"):
                wm.add_causal_link(link["cause"], link["effect"],
                                   link.get("mechanism", "reflection"), link.get("domain", "general"), 0.65)
                links_added += 1
        if facts_added or links_added:
            wm.save()
        return {
            "curriculum_focus": data.get("curriculum_focus", ""),
            "priority_domains": data.get("priority_domains", []),
            "facts_added": facts_added,
            "links_added": links_added,
            "knowledge_gaps": data.get("knowledge_gaps", []),
        }
    except Exception as e:
        return {"error": str(e)}


def distill_domain_knowledge(domain: str, causal_links: list, facts: list) -> dict:
    """
    Compress accumulated causal links and facts into higher-level rules.
    Returns {"distilled_rules": [...], "new_schemas": [...]} and enriches world model.
    """
    links_str = "\n".join(f"  {l['cause']} → {l['effect']} ({l.get('mechanism','')})"
                          for l in causal_links[:15])
    facts_str = "\n".join(f"  - {f['fact']}" for f in facts[:10] if isinstance(f, dict) and 'fact' in f)
    prompt = (
        f"Distill the following {domain} knowledge into higher-level mathematical principles.\n\n"
        f"Causal links:\n{links_str or '  (none yet)'}\n\n"
        f"Known facts:\n{facts_str or '  (none yet)'}\n\n"
        "Return JSON:\n"
        "{\n"
        '  "distilled_rules": [{"name": "...", "statement": "...", "confidence": 0.9}],\n'
        '  "new_facts": ["higher-level fact 1", "higher-level fact 2"],\n'
        '  "new_causal_links": [{"cause": "...", "effect": "...", "mechanism": "..."}]\n'
        "}\n"
        "Return ONLY valid JSON. At most 3 distilled rules, 3 facts, 3 links."
    )
    try:
        from sare.memory.world_model import get_world_model
        wm = get_world_model()
        raw = _call_llm(prompt)
        raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`")
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {}
        data = json.loads(m.group())
        facts_added, links_added = 0, 0
        for fact in data.get("new_facts", [])[:3]:
            if isinstance(fact, str) and fact.strip():
                wm.add_fact(domain, fact.strip(), 0.85, source="llm_distilled")
                facts_added += 1
        for link in data.get("new_causal_links", [])[:3]:
            if isinstance(link, dict) and link.get("cause") and link.get("effect"):
                wm.add_causal_link(link["cause"], link["effect"],
                                   link.get("mechanism", "distilled"), domain, 0.82)
                links_added += 1
        if facts_added or links_added:
            wm.save()
        return {
            "distilled_rules": data.get("distilled_rules", []),
            "facts_added": facts_added,
            "links_added": links_added,
        }
    except Exception as e:
        return {"error": str(e)}


# ── Health check ──────────────────────────────────────────────────────────────

def llm_available() -> bool:
    """Returns True if LLM is reachable (LMStudio/Ollama need no key; others need API key)."""
    cfg = _load_config(force_reload=True)
    provider = cfg.get("provider", "lmstudio")
    if provider == "lmstudio":
        try:
            base = cfg.get("lmstudio_url", "http://localhost:1234").rstrip("/")
            urllib.request.urlopen(f"{base}/v1/models", timeout=2)
            return True
        except Exception:
            return False
    if provider == "ollama":
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            return True
        except Exception:
            return False
    if provider == "openrouter":
        key = cfg.get("api_key", "").strip()
        return bool(key and key.startswith("sk-or-"))
    return bool(_resolve_api_key(cfg))


def llm_status() -> dict:
    """Returns a status dict for the /api/llm-status endpoint."""
    cfg = _load_config(force_reload=True)
    provider = cfg.get("provider", "lmstudio")
    if provider == "lmstudio":
        reachable = llm_available()
        return {
            "available": reachable,
            "provider": "lmstudio",
            "model": cfg.get("model", "local-model"),
            "lmstudio_url": cfg.get("lmstudio_url", "http://localhost:1234"),
            "api_key_set": True,
            "api_key_source": "none_required",
        }
    if provider == "ollama":
        reachable = llm_available()
        return {
            "available": reachable,
            "provider": "ollama",
            "model": cfg.get("model", "qwen3.5:2b"),
            "api_key_env": None,
            "api_key_set": True,
            "api_key_source": "none_required",
            "ollama_url": "http://localhost:11434",
            "ollama_reachable": reachable,
        }
    if provider == "openrouter":
        key = cfg.get("api_key", "").strip()
        available = bool(key and key.startswith("sk-or-"))
        return {
            "available": available,
            "provider": "openrouter",
            "model": cfg.get("model", "google/gemini-2.0-flash-001"),
            "synthesis_model": cfg.get("synthesis_model", cfg.get("model", "google/gemini-2.0-flash-001")),
            "api_key_env": None,
            "api_key_set": available,
            "api_key_source": "config",
        }
    key = _resolve_api_key(cfg)
    available = bool(key)
    return {
        "available": available,
        "provider": provider,
        "model": cfg.get("model", "gemini-2.5-flash-preview-04-17"),
        "api_key_env": cfg.get("api_key_env", "GEMINI_API_KEY"),
        "api_key_set": available,
        "api_key_source": "config" if cfg.get("api_key", "").startswith("AIza") else "env_var",
    }
