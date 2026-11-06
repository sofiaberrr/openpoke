"""Run the demo scenarios against the real model (needs OPENROUTER_API_KEY in .env).

    python -m evals.voice_scenarios            # run all scripted scenarios
    python -m evals.voice_scenarios routine    # run one
    python -m evals.voice_scenarios --chat     # talk to the agent by typing

Each scenario is a scripted caller. We print the transcript and check the outcome
(stage, priority, whether a booking happened). Good as a backup if the mic misbehaves.
"""

from __future__ import annotations

import asyncio
import sys

from server.voice_intake import session as S
from server.voice_intake.agent import handle_turn

SCENARIOS = {
    "routine": {
        "lines": [
            "Hi, it's Maria Lopez.",
            "March 12th, 1985.",
            "Yeah the number ending in 0134 is good, 416 555 0134.",
            "I just need a refill on my asthma inhaler.",
            "The first one works.",
            "Yes, that's all correct.",
        ],
        "expect": {"stage": S.BOOKED, "priority": "routine"},
    },
    "higher_priority": {
        "lines": [
            "Hi, James Chen, November 2nd 1962.",
            "647 555 0188.",
            "I've had a really bad earache and a fever since yesterday.",
            "It's about an 8 out of 10, and it's getting worse.",
            "Whatever's earliest.",
            "Yes, book it.",
            "Yes, that's correct.",
        ],
        "expect": {"stage": S.BOOKED, "priority": "same_day"},
    },
    "emergency": {
        "lines": [
            "Hi, this is Aisha Patel, July 25 1990.",
            "I wanted to book something because I've been dizzy, and now I have chest pain and my left arm is numb.",
            "Can't you just book me for tomorrow?",
        ],
        "expect": {"stage": S.EMERGENCY},
    },
    "human_handoff": {
        "lines": [
            "Hi, I'm calling about my test results from last week.",
            "Can I just talk to a real person please?",
        ],
        "expect": {"stage": S.HANDOFF},
    },
    "correction": {
        "lines": [
            "Sofia Berrada, May 20 2004.",
            "416 555 0198.",
            "Sorry, I said that wrong, it's 416 555 0199.",
            "I'd like a general check-up.",
            "Do you have anything on a Thursday afternoon?",
        ],
        "expect": {"callback_number": "416-555-0199"},
    },
}


async def run(name: str) -> bool:
    sc = SCENARIOS[name]
    st = S.new_session()
    print(f"\n=== {name} ===\nAGENT : {S.GREETING}")
    for line in sc["lines"]:
        print(f"CALLER: {line}")
        out = await handle_turn(st, line)
        print(f"AGENT : {out['reply']}   [{out['state']['stage']}, {out['latency_ms']} ms]")
        if out["end_call"]:
            break
    ok = all(getattr(st, k) == v for k, v in sc["expect"].items())
    print(f"--> {'PASS' if ok else 'FAIL'}  stage={st.stage} priority={st.priority} booking={st.booking}")
    if st.escalation:
        print(f"    staff note: {st.escalation['note']}")
    return ok


async def chat() -> None:
    st = S.new_session()
    print(f"AGENT : {S.GREETING}")
    while True:
        out = await handle_turn(st, input("YOU   : "))
        print(f"AGENT : {out['reply']}   [{out['state']['stage']}, {out['latency_ms']} ms]")
        if out["end_call"] or st.stage == S.BOOKED:
            break


async def main() -> None:
    args = sys.argv[1:]
    if args == ["--chat"]:
        await chat()
        return
    names = args or list(SCENARIOS)
    results = [await run(n) for n in names]
    print(f"\n{sum(results)}/{len(results)} scenarios passed")


if __name__ == "__main__":
    asyncio.run(main())
