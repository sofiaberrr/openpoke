"""Retrieval eval: does the right agent make it into what the interaction agent sees?

No LLM calls, no API key, runs in seconds. For each roster size it compares:

  original      - the whole roster, names only (what OpenPoke does today)
  recent_only   - naive fix: just show the 11 most recently used agents
  shortlist     - our selector (top-8 relevant + 3 recent), keyword matching
  shortlist+sem - same, plus local embeddings (only if `fastembed` is installed)

Metrics (over cases that have an existing correct agent):
  recall   - % of cases where the correct agent is in the list shown to the model.
             "original" is 100% by construction; the question is what it costs.
  top1     - % of cases where the correct agent is ranked first.
  tokens   - approx prompt tokens spent on the agent list per turn (chars / 4).

Run:  python -m evals.retrieval_eval
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from html import escape
from pathlib import Path

from evals.cases import CASES
from evals.synthetic_roster import NOW, build_roster
from server.agents.interaction_agent.agent import render_agent_list
from server.services.execution.agent_selector import _SemanticScorer, select_agents

SIZES = [25, 100, 300, 1000]
RESULTS_DIR = Path(__file__).parent / "results"


def original_render(roster) -> str:
    lines = [f'<agent name="{escape(a.name, quote=True)}" />' for a in roster]
    return "<active_agents>\n" + "\n".join(lines) + "\n</active_agents>"


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def evaluate(size: int, strategy: str) -> dict:
    roster = build_roster(size)
    now = NOW.timestamp()
    found, top1, tokens, shown = 0, 0, 0, 0
    per_cat = defaultdict(lambda: [0, 0])
    misses = []
    cases = [c for c in CASES if c.expected]

    for case in cases:
        if strategy == "original":
            names = [a.name for a in roster]
            rendered = original_render(roster)
            first = None
        elif strategy == "recent_only":
            recent = sorted(roster, key=lambda a: a.last_used_ts(), reverse=True)[:11]
            names = [a.name for a in recent]
            rendered = original_render(recent)
            first = names[0]
        else:
            sel = select_agents(case.message, roster, context=case.context, now=now,
                                use_semantic=(strategy == "shortlist+sem"))
            names = [a.name for a in sel.agents]
            rendered = render_agent_list(sel)
            first = names[0] if names else None

        hit = case.expected in names
        found += hit
        top1 += (first == case.expected) if first is not None else hit
        per_cat[case.category][0] += hit
        per_cat[case.category][1] += 1
        tokens += approx_tokens(rendered)
        shown += len(names)
        if not hit:
            misses.append(case.id)

    n = len(cases)
    return {
        "size": size,
        "strategy": strategy,
        "recall": round(100 * found / n, 1),
        "top1": round(100 * top1 / n, 1),
        "avg_tokens": round(tokens / n),
        "avg_shown": round(shown / n, 1),
        "by_category": {k: f"{v[0]}/{v[1]}" for k, v in per_cat.items()},
        "misses": misses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="*", default=SIZES)
    args = parser.parse_args()

    strategies = ["original", "recent_only", "shortlist"]
    if _SemanticScorer.get().enabled:
        strategies.append("shortlist+sem")
    else:
        print("(fastembed not available: skipping shortlist+sem)\n")

    rows = [evaluate(size, s) for size in args.sizes for s in strategies]

    header = f"{'agents':>6}  {'strategy':<14} {'recall':>7} {'top1':>6} {'tokens/turn':>12} {'shown':>6}  misses"
    print(header)
    print("-" * len(header))
    for r in rows:
        top1 = "n/a" if r["strategy"] == "original" else f"{r['top1']}%"
        print(f"{r['size']:>6}  {r['strategy']:<14} {r['recall']:>6}% {top1:>6} {r['avg_tokens']:>12} "
              f"{r['avg_shown']:>6}  {','.join(r['misses']) or '-'}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "retrieval.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
