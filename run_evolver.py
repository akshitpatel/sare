#!/usr/bin/env python3
"""
run_evolver.py — Run one full self-improvement debate and print the complete
                 transcript + per-model token counts and USD cost breakdown.

Usage:
    python3 run_evolver.py                        # auto-pick target
    python3 run_evolver.py sare/memory/hippocampus.py optimize
    python3 run_evolver.py sare/engine.py fix
"""
import sys
import os
import time
import json
import logging

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT   = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(ROOT, "python")
sys.path.insert(0, PYTHON)
os.chdir(PYTHON)

# ── Verbose logging so every LLM call is visible ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evolver_runner")

# Silence noisy sub-loggers but keep cost + SelfImprover visible
for noisy in ["urllib3", "httpx", "httpcore"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

DIVIDER = "═" * 72

def print_section(title, text, color=""):
    RESET = "\033[0m"
    BOLD  = "\033[1m"
    CYAN  = "\033[96m"
    YELLOW= "\033[93m"
    GREEN = "\033[92m"
    RED   = "\033[91m"
    PURPLE= "\033[95m"
    c = {"cyan": CYAN, "yellow": YELLOW, "green": GREEN, "red": RED, "purple": PURPLE}.get(color, "")
    print(f"\n{c}{BOLD}{DIVIDER}")
    print(f"  {title}")
    print(f"{DIVIDER}{RESET}")
    if text:
        print(text)

def fmt_cost(usd: float) -> str:
    if usd == 0:
        return "\033[92m$0.000000 (FREE)\033[0m"
    elif usd < 0.01:
        return f"\033[93m${usd:.6f}\033[0m"
    else:
        return f"\033[91m${usd:.4f}\033[0m"

def main():
    target_file = sys.argv[1] if len(sys.argv) > 1 else None
    improvement_type = sys.argv[2] if len(sys.argv) > 2 else "optimize"

    # ── Import after path setup ────────────────────────────────────────────
    from sare.meta.self_improver import SelfImprover, AGI_SYSTEM_PROMPT, _extract_skeleton
    from sare.interface.llm_bridge import get_cost_summary, _cost_ledger

    print(f"\n\033[1m\033[95m{'═'*72}")
    print("  SARE-HX SELF-EVOLVER — SINGLE DEBATE RUN")
    print(f"{'═'*72}\033[0m")
    print(f"\n  Pipeline (7 turns):")
    print(f"  T0 Dual Pre-screen  : stepfun/step-3.5-flash:free + hunter-alpha  (FREE, AND gate)")
    print(f"  T1 Proposer         : openrouter/hunter-alpha  (FREE)")
    print(f"     Alt Proposers    : stepfun:free + hunter-alpha  (FREE, parallel background)")
    print(f"  T2 Planner          : openai/gpt-5.4  (structured plan from proposal)")
    print(f"  T3 Executor         : stepfun/step-3.5-flash:free  (concrete code spec, FREE)")
    print(f"  T4 Critic           : anthropic/claude-sonnet-4.6 (main, weight×2)")
    print(f"                      + deepseek/deepseek-v3.2  (cheap)")
    print(f"     Critic gate      : claude≥7 OR (claude≥6 AND deepseek≥5)")
    print(f"  T5 Judge            : google/gemini-3.1-pro-preview  (1M token context)")
    print(f"  T5b Judge fallback  : openai/gpt-5.4 → anthropic/claude-opus-4  (chain)")
    print(f"  T6 Verifier         : minimax/minimax-m2.5")
    print(f"  Context             : FULL 129-file codebase + AGI system prompt")
    if target_file:
        print(f"  Target       : {target_file}  [{improvement_type}]")
    else:
        print(f"  Target       : auto-pick (bottleneck analyzer)")

    # Show AGI system prompt summary
    print_section("AGI SYSTEM PROMPT (injected into every LLM call)", AGI_SYSTEM_PROMPT[:800] + "\n...", "purple")

    si = SelfImprover()

    # Monkey-patch _run_debate to capture debate record live
    original_run_debate = si._run_debate
    _debate_ref = [None]

    def _patched_run_debate(target, source_code, progress_key=""):
        debate = original_run_debate(target, source_code, progress_key)
        _debate_ref[0] = debate
        return debate

    si._run_debate = _patched_run_debate

    # Show full codebase context size
    if target_file:
        import os as _os
        abs_path = _os.path.join(PYTHON, target_file.lstrip("/"))
        if _os.path.exists(abs_path):
            src = open(abs_path).read()
            ctx = si._gather_full_codebase_context(abs_path, src)
            print_section("FULL CODEBASE CONTEXT STATS", "", "cyan")
            tier1 = ctx.count("TIER 1")
            tier2 = ctx.count("TIER 2")
            print(f"  Total chars  : {len(ctx):,}")
            print(f"  ~Tokens      : {len(ctx)//4:,} / 1,048,576 ({len(ctx)//4/10485.76:.1f}% of Gemini limit)")
            print(f"  Tier 1 files : full source of target + related files")
            print(f"  Tier 2 files : AST skeletons of all remaining modules")
            print(f"  Tier 3       : run reports, bottleneck, promoted rules")

    # ── Run the debate ─────────────────────────────────────────────────────
    print_section("STARTING DEBATE", "Running... (this takes 2-5 minutes)", "yellow")
    t0 = time.time()
    result = si.run_once(target_file=target_file, improvement_type=improvement_type)
    elapsed = time.time() - t0

    # ── Print full debate transcript ───────────────────────────────────────
    debate = _debate_ref[0]
    if debate:
        print_section(f"TARGET FILE: {debate.target_file.split('/')[-1]}  [{debate.improvement_type}]", "", "cyan")

        print_section("TURN 0 — DUAL PRE-SCREEN  (stepfun:free + hunter-alpha, AND gate)", "", "yellow")
        if "rejected_prescreen" in debate.outcome:
            print(f"  \033[91m✗ PRESCREENED OUT: {debate.outcome}\033[0m")
        else:
            print("  \033[92m✓ PASSED dual pre-screen\033[0m")

        print_section("TURN 1 — PROPOSER  (hunter-alpha FREE)", debate.proposer_text or "[empty]", "yellow")

        planner_text = getattr(debate, "planner_text", "") or ""
        executor_text = getattr(debate, "executor_text", "") or ""
        print_section("TURN 2 — PLANNER  (gpt-5.4)", planner_text[:800] or "[empty]", "cyan")
        print_section("TURN 3 — EXECUTOR  (stepfun:free)", executor_text[:800] or "[empty]", "cyan")

        print_section(f"TURN 4 — CRITICS  (claude-sonnet main ×2 + deepseek cheap, weighted gate)", "", "yellow")
        if debate.panel_scores:
            weights = {"MAIN": 2, "CHEAP": 1}
            for model, score in debate.panel_scores.items():
                bar = "█" * score + "░" * (10 - score)
                color = "\033[92m" if score >= 7 else "\033[93m" if score >= 5 else "\033[91m"
                label = "claude-sonnet-4.6 (main×2)" if model == "MAIN" else "deepseek-v3.2 (cheap)"
                print(f"  {color}{label:35s}  {bar}  {score}/10\033[0m")
            scores = list(debate.panel_scores.values())
            main_s = debate.panel_scores.get("MAIN", 0)
            cheap_s = debate.panel_scores.get("CHEAP", 5)
            weighted = round((main_s * 2 + cheap_s) / 3)
            gate_pass = main_s >= 7 or (main_s >= 6 and cheap_s >= 5)
            verdict = "\033[92m✓ WEIGHTED GATE PASS\033[0m" if gate_pass else "\033[91m✗ GATE FAILED\033[0m"
            print(f"\n  Weighted avg: {weighted}/10   Gate: {verdict}")
        else:
            print(f"  Score: {debate.critic_score}/10")
        print(f"\n{debate.critic_text[:1200] if debate.critic_text else '[no critique]'}")

        if debate.judge_code:
            code_preview = debate.judge_code[:600]
            lines = len(debate.judge_code.splitlines())
            print_section(
                f"TURN 5 — JUDGE  (gemini-3.1-pro 1M ctx)  → {lines} lines written",
                code_preview + f"\n... [{lines} total lines]",
                "green",
            )
        else:
            print_section("TURN 5 — JUDGE  (gemini-3.1-pro 1M ctx)", "[NO CODE RETURNED]", "red")

        if debate.verifier_text:
            print_section("TURN 6 — VERIFIER  (minimax-m2.5)", debate.verifier_text, "cyan")

        outcome_color = "\033[92m" if debate.outcome == "applied" else "\033[91m"
        print_section(f"OUTCOME: {outcome_color}{debate.outcome.upper()}\033[0m", "", "cyan")

    # ── Cost breakdown ─────────────────────────────────────────────────────
    summary = get_cost_summary()
    print_section("COST BREAKDOWN", "", "purple")
    print(f"  {'Model':<42} {'Role':<16} {'In tok':>8} {'Out tok':>8}  {'Cost':>12}")
    print(f"  {'─'*42} {'─'*16} {'─'*8} {'─'*8}  {'─'*12}")
    for entry in summary["ledger"]:
        cost_str = fmt_cost(entry["cost_usd"])
        print(f"  {entry['model']:<42} {entry['role']:<16} {entry['in_tok']:>8,} {entry['out_tok']:>8,}  {cost_str}")
    print(f"\n  {'─'*90}")
    print(f"  {'TOTAL':<60} {summary['total_usd']:>12.6f} USD")
    print(f"\n  Elapsed: {elapsed:.1f}s")

    # By-model summary
    print(f"\n  {'Model summary':}")
    for model, stats in sorted(summary["by_model"].items(), key=lambda x: -x[1]["cost_usd"]):
        print(f"    {model:<42} {stats['calls']:2} calls   in={stats['in_tok']:,}  out={stats['out_tok']:,}   {fmt_cost(stats['cost_usd'])}")

    total = summary["total_usd"]
    if total == 0:
        print(f"\n  \033[92m💰 Total cost: $0.00 (all FREE models this run)\033[0m")
    elif total < 0.05:
        print(f"\n  \033[92m💰 Total cost: ${total:.4f}  (under 5¢)\033[0m")
    else:
        print(f"\n  \033[93m💰 Total cost: ${total:.4f}\033[0m")

    # Save full report
    report_path = os.path.join(ROOT, "data", "memory", "evolver_run_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report = {
        "timestamp": time.time(),
        "elapsed_s": round(elapsed, 1),
        "result": result,
        "debate": debate.to_dict() if debate else None,
        "cost": summary,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report saved → {report_path}\n")


if __name__ == "__main__":
    main()
