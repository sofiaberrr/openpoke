"""One caller turn: safety screen -> (LLM + gated tools) -> reply.

Order of operations every turn:
  1. Terminal stages (emergency / handoff) never go back to booking.
  2. Deterministic safety screen on the raw utterance (emergency, crisis, "get me a person").
  3. LLM turn with only the tools whose preconditions are met.
  4. Any tool failure or LLM error -> human handoff, never a dead end.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Awaitable, Callable, Dict, List

from . import safety, session as S, tools as T
from .prompts import build_system_prompt

MAX_TOOL_ROUNDS = 5
HISTORY_TURNS = 12  # bounded history: the state block carries the facts, not the transcript

HANDOFF_SCRIPT = ("Of course. I'm connecting you with someone at the front desk now, and I'll pass along "
                  "what you've told me so you don't have to repeat yourself.")
FAILURE_SCRIPT = ("Sorry, I'm having trouble with our system right now. I'm going to connect you with "
                  "someone at the front desk who can help.")

# If the model *says* it's transferring but forgot to call the tool, we still transfer.
PROMISED_TRANSFER = re.compile(r"\b(transfer(ring)?|connect(ing)?) you\b|\bput you through\b", re.I)

LLMFn = Callable[..., Awaitable[Dict[str, Any]]]


def _model() -> str:
    return os.getenv("VOICE_AGENT_MODEL", "openai/gpt-4o-mini")


async def _default_llm(**kwargs: Any) -> Dict[str, Any]:
    from ..openrouter_client import request_chat_completion
    return await request_chat_completion(model=_model(), **kwargs)


def _clean(text: str) -> str:
    """Strip things that must never be read aloud: fake tool calls like {calling x}, markdown, tool names."""
    text = re.sub(r"[{\[][^}\]]*[}\]]", "", text)
    text = re.sub(r"\b(" + "|".join(T.SCHEMAS) + r")\b", "", text)
    text = re.sub(r"[*_#`]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _reply(state: S.CallState, text: str, started: float) -> Dict[str, Any]:
    text = _clean(text)
    state.transcript.append({"role": "agent", "text": text})
    T.recompute_stage(state)
    ms = int((time.time() - started) * 1000)
    state.log("reply", ms=ms)
    return {"reply": text, "state": state.public(), "latency_ms": ms,
            "end_call": state.stage in (S.EMERGENCY, S.HANDOFF)}


def _history(state: S.CallState) -> List[Dict[str, str]]:
    msgs = [{"role": "assistant" if m["role"] == "agent" else "user", "content": m["text"]}
            for m in state.transcript[-HISTORY_TURNS:]]
    while msgs and msgs[0]["role"] == "assistant" and len(msgs) > 1 and msgs[1]["role"] == "assistant":
        msgs.pop(0)
    return msgs


async def handle_turn(state: S.CallState, caller_text: str, llm: LLMFn = _default_llm) -> Dict[str, Any]:
    started = time.time()
    caller_text = (caller_text or "").strip()
    state.transcript.append({"role": "caller", "text": caller_text})

    # 1. terminal stages stay terminal
    if state.stage == S.EMERGENCY:
        return _reply(state, safety.EMERGENCY_REPEAT, started)
    if state.stage == S.HANDOFF:
        return _reply(state, "Please hold, someone from the clinic will be right with you.", started)

    # 2. deterministic safety screen (runs before the model ever sees the text)
    check = safety.screen(caller_text)
    if check.kind == "emergency":
        T.flag_emergency(state, "; ".join(check.reasons))
        state.log("safety_screen", result="emergency", reasons=check.reasons)
        return _reply(state, safety.EMERGENCY_SCRIPT, started)
    if check.kind == "crisis":
        T.transfer_to_human(state, "possible mental health crisis")
        state.escalation["type"] = "crisis"
        state.log("safety_screen", result="crisis")
        return _reply(state, safety.CRISIS_SCRIPT, started)
    if check.kind == "handoff":
        T.transfer_to_human(state, "caller asked for a person")
        state.log("safety_screen", result="handoff")
        return _reply(state, HANDOFF_SCRIPT, started)

    # 3. LLM turn with gated tools
    messages: List[Dict[str, Any]] = _history(state)
    try:
        for _ in range(MAX_TOOL_ROUNDS):
            tool_names = T.available_tools(state)
            resp = await llm(messages=messages, system=build_system_prompt(state),
                             tools=[T.SCHEMAS[n] for n in tool_names])
            msg = resp["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if not calls:
                text = (msg.get("content") or "").strip()
                if not text:
                    raise RuntimeError("empty model reply")
                if PROMISED_TRANSFER.search(text):
                    T.transfer_to_human(state, "assistant offered a transfer")
                    state.log("guard", result="model promised transfer without the tool; transferred")
                return _reply(state, text, started)

            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for call in calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if name not in tool_names:
                    result = {"ok": False, "error": f"{name} isn't available yet."}
                else:
                    try:
                        result = T.IMPLS[name](state, **args)
                    except TypeError as exc:  # model sent bad arguments: let it retry
                        result = {"ok": False, "error": f"bad arguments: {exc}"}
                state.log("tool", name=name, args=args, ok=result.get("ok"))
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": json.dumps(result)})

                if name == "flag_emergency":
                    return _reply(state, safety.EMERGENCY_SCRIPT, started)
                if name == "transfer_to_human":
                    return _reply(state, HANDOFF_SCRIPT, started)
            T.recompute_stage(state)
        raise RuntimeError("too many tool rounds")

    except T.ToolFailure as exc:
        state.log("tool_failure", error=str(exc))
        T.transfer_to_human(state, f"tool failure: {exc}")
        return _reply(state, FAILURE_SCRIPT, started)
    except Exception as exc:  # LLM down, bad output, bad tool args...
        state.log("error", error=f"{type(exc).__name__}: {exc}")
        T.transfer_to_human(state, "assistant error")
        return _reply(state, FAILURE_SCRIPT, started)
