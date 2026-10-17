"""End-to-end routing eval with a real LLM: does the interaction agent pick the right agent?

The retrieval eval checks whether the right agent is *shown*. This one checks whether
the model then *uses* it, which is what actually matters. For each test case we build
the exact prompt the interaction agent would get (real system prompt, real tool
schemas, a synthetic roster) and record which agent it routes to.

Variants:
  original   - whole roster as bare names, original tools (OpenPoke today)
  shortlist  - our selector + descriptions + search_agents tool + duplicate guard

Each case is graded as one of:
  correct     - routed to the expected agent (or created a new one when none fit)
  duplicate   - created a NEW agent even though the right one already existed
  wrong_agent - reused an existing agent, but the wrong one
  no_route    - never called send_message_to_agent

Cost: roughly 1-3 requests per case per variant. The default (12 cases x 2 variants)
fits inside OpenRouter's 50 free requests/day. Results are saved after every case, so
hitting the rate limit loses nothing.

Run:  python -m evals.llm_eval --cases 12
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from evals.cases import CASES
from evals.retrieval_eval import original_render
from evals.synthetic_roster import NOW, build_roster
from server.agents.interaction_agent.agent import SYSTEM_PROMPT, render_agent_list
from server.agents.interaction_agent.tools import TOOL_SCHEMAS
from server.config import get_settings
from server.openrouter_client import request_chat_completion
from server.services.execution.agent_selector import rank_agents, select_agents
from server.services.execution.roster import find_duplicate_name

DEFAULT_MODEL = os.getenv("EVAL_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
RESULTS_DIR = Path(__file__).parent / "results"
MAX_STEPS = 4
PAUSE_SECONDS = 3.5  # free models allow ~20 requests/minute

# The original prompt had no search_agents section; strip ours for the baseline.
ORIGINAL_PROMPT = re.sub(r"Search Agents Tool Usage\n.*?(?=Send Message to User Tool Usage)", "",
                         SYSTEM_PROMPT, flags=re.S)
ORIGINAL_TOOLS = [t for t in TOOL_SCHEMAS if t["function"]["name"] != "search_agents"]


def pick_cases(n: int):
    """Round-robin across categories so a small run still covers every kind of case."""
    by_cat = defaultdict(list)
    for c in CASES:
        by_cat[c.category].append(c)
    picked, i = [], 0
    while len(picked) < min(n, len(CASES)):
        for cat in by_cat:
            if i < len(by_cat[cat]) and len(picked) < n:
                picked.append(by_cat[cat][i])
        i += 1
    return picked


def build_user_content(case, agents_block: str) -> str:
    history = case.context.strip() or "None"
    return (f"<conversation_history>\n{history}\n</conversation_history>\n\n{agents_block}\n\n"
            f"<new_user_message>\n{case.message}\n</new_user_message>")


async def run_case(case, variant: str, roster, model: str, api_key: str) -> dict:
    names = [a.name for a in roster]
    if variant == "original":
        system, tools = ORIGINAL_PROMPT, ORIGINAL_TOOLS
        block = original_render(roster)
    else:
        system, tools = SYSTEM_PROMPT, TOOL_SCHEMAS
        sel = select_agents(case.message, roster, context=case.context, now=NOW.timestamp())
        block = render_agent_list(sel)

    messages = [{"role": "user", "content": build_user_content(case, block)}]
    routed, searched, prompt_tokens, calls = None, False, 0, 0
    said = []  # what the model told the user
    started = time.time()

    for _ in range(MAX_STEPS):
        calls += 1
        resp = await request_chat_completion(model=model, messages=messages, system=system,
                                             api_key=api_key, tools=tools)
        prompt_tokens += (resp.get("usage") or {}).get("prompt_tokens", 0)
        msg = (resp.get("choices") or [{}])[0].get("message") or {}
        tool_calls = msg.get("tool_calls") or []
        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         **({"tool_calls": tool_calls} if tool_calls else {})})
        if not tool_calls:
            if msg.get("content"):
                said.append(msg["content"])
            break

        for tc in tool_calls:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            name = fn.get("name")
            if name == "send_message_to_agent" and routed is None:
                routed = args.get("agent_name")
                result = {"status": "submitted"}
            elif name == "search_agents":
                searched = True
                hits = rank_agents(args.get("query", ""), roster, now=NOW.timestamp())[:5]
                result = {"matches": [{"agent_name": r.name, "purpose": r.first_task} for r, _ in hits]}
            else:
                if name == "send_message_to_user":
                    said.append(args.get("message", ""))
                result = {"status": "delivered"}
            messages.append({"role": "tool", "tool_call_id": tc.get("id") or name,
                             "content": json.dumps(result)})
        if routed is not None:
            break
        await asyncio.sleep(PAUSE_SECONDS)

    final = routed
    if routed and variant == "shortlist":
        final = find_duplicate_name(routed, names) or routed  # the guard runs in the real tool

    if final is None:
        # Answered (or tried to) without handing the task to any agent.
        outcome = "answered_without_agent" if said else "no_route"
    elif case.expected is None:
        outcome = "correct" if final not in names else "wrong_agent"
    elif final == case.expected:
        outcome = "correct"
    elif final in names:
        outcome = "wrong_agent"
    else:
        outcome = "duplicate"

    return {"case": case.id, "category": case.category, "variant": variant, "expected": case.expected,
            "routed": routed, "final": final, "outcome": outcome, "searched": searched, "said": " | ".join(said)[:300],
            "prompt_tokens": prompt_tokens, "calls": calls, "seconds": round(time.time() - started, 1)}


def summarize(rows):
    print(f"\n{'variant':<10} {'accuracy':>9} {'duplicates':>11} {'wrong':>6} {'no agent':>9} "
          f"{'avg prompt tok':>15} {'avg sec':>8}")
    for variant in ("original", "shortlist"):
        vr = [r for r in rows if r["variant"] == variant]
        if not vr:
            continue
        c = Counter(r["outcome"] for r in vr)
        n = len(vr)
        print(f"{variant:<10} {100 * c['correct'] / n:>8.0f}% {c['duplicate']:>11} {c['wrong_agent']:>6} "
              f"{c['no_route'] + c['answered_without_agent']:>9} {sum(r['prompt_tokens'] for r in vr) / n:>15.0f} "
              f"{sum(r['seconds'] for r in vr) / n:>8.1f}")
    print("\nPer case (original -> shortlist):")
    by_case = defaultdict(dict)
    for r in rows:
        by_case[r["case"]][r["variant"]] = r
    for cid, v in by_case.items():
        o, s = v.get("original"), v.get("shortlist")
        fmt = lambda r: "-" if not r else f"{r['outcome']}({r['final']})"
        print(f"  {cid:<4} {fmt(o)[:60]:<62} {fmt(s)[:60]}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--size", type=int, default=300, help="roster size")
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--variants", nargs="*", default=["original", "shortlist"])
    parser.add_argument("--ids", nargs="*", help="only run these case ids, e.g. --ids d1 f1")
    parser.add_argument("--tag", default="", help="suffix for the results file, e.g. v2")
    args = parser.parse_args()

    api_key = get_settings().openrouter_api_key
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY in .env first.")

    roster = build_roster(args.size)
    cases = [c for c in CASES if c.id in args.ids] if args.ids else pick_cases(args.cases)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"llm_{re.sub(r'[^a-z0-9]+', '-', args.model.lower())}_{args.size}{('_' + args.tag) if args.tag else ''}.json"
    rows = []

    print(f"Model {args.model} | roster {args.size} agents | {len(cases)} cases x {len(args.variants)} variants")
    try:
        for case in cases:
            for variant in args.variants:
                row = await run_case(case, variant, roster, args.model, api_key)
                rows.append(row)
                print(f"  {case.id:<4} {variant:<10} {row['outcome']:<22} -> {row['final']}")
                if row["said"] and row["final"] is None:
                    print(f"         said: {row['said'][:110]}")
                out.write_text(json.dumps(rows, indent=2))
                await asyncio.sleep(PAUSE_SECONDS)
    except Exception as exc:  # most likely the free-tier rate limit
        print(f"\nStopped early: {exc}\nPartial results kept ({len(rows)} rows).")

    summarize(rows)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    asyncio.run(main())
