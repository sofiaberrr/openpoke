"""Execution Agent implementation."""

import os
from pathlib import Path
from html import escape
from typing import List, Optional, Dict, Any

from ...services.execution import get_agent_roster, get_execution_agent_logs
from ...logging_config import logger

# How many recent requests an execution agent sees in full. Older ones are condensed.
# 0 = old behaviour (whole history, forever).
DEFAULT_FULL_TURNS = int(os.getenv("OPENPOKE_EXEC_HISTORY_TURNS", "6"))
# Older requests are condensed to one line each; beyond this many they are only counted.
DIGEST_MAX_LINES = 20
_REQ_CHARS, _RESP_CHARS = 120, 160


def _short(text: str, limit: int) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def split_into_turns(entries):
    """Group log entries into turns. A turn starts at each agent_request."""
    turns, current = [], []
    for entry in entries:
        if entry[0] == "agent_request" and current:
            turns.append(current)
            current = []
        current.append(entry)
    if current:
        turns.append(current)
    return turns


def condense_turn(turn) -> str:
    """One line per old turn: what it was asked and what it answered. Tool calls are dropped."""
    request = next((e for e in turn if e[0] == "agent_request"), None)
    response = next((e for e in reversed(turn) if e[0] == "agent_response"), None)
    stamp = (request or turn[0])[1][:16].replace("T", " ")
    asked = _short(request[2], _REQ_CHARS) if request else "(no request)"
    answered = _short(response[2], _RESP_CHARS) if response else "(no reply recorded)"
    return f"- [{stamp}] asked: {asked} -> replied: {answered}"


def render_history(entries, full_turns: int) -> str:
    """Bounded history: a digest of older turns + the last `full_turns` turns verbatim.

    The original agent pasted its entire log into the prompt on every call, so a
    long-lived agent ("Nightly email summary") grew without limit, the same overload
    problem as the roster, one level down. Most of that bulk is old tool output.
    No LLM call is used to condense, so it costs nothing and is deterministic.
    """
    turns = split_into_turns(list(entries))
    if full_turns <= 0 or len(turns) <= full_turns:
        older, recent = [], turns
    else:
        older, recent = turns[:-full_turns], turns[-full_turns:]

    parts = []
    if older:
        dropped = max(0, len(older) - DIGEST_MAX_LINES)
        lines = [condense_turn(t) for t in older[-DIGEST_MAX_LINES:]]
        header = f"## Earlier requests (condensed, {len(older)} total"
        header += f", oldest {dropped} not shown)" if dropped else ")"
        parts.append(header + "\n" + "\n".join(lines))
        parts.append("## Recent requests (full detail)")
    for turn in recent:
        for tag, timestamp, payload in turn:
            body = escape(payload, quote=False)
            parts.append(f'<{tag} timestamp="{timestamp}">{body}</{tag}>' if timestamp else f"<{tag}>{body}</{tag}>")
    return "\n".join(parts)


# Load system prompt template from file
_prompt_path = Path(__file__).parent / "system_prompt.md"
if _prompt_path.exists():
    SYSTEM_PROMPT_TEMPLATE = _prompt_path.read_text(encoding="utf-8").strip()
else:
    # Placeholder template - you'll replace this with actual instructions
    SYSTEM_PROMPT_TEMPLATE = """You are an execution agent responsible for completing specific tasks using available tools.

Agent Name: {agent_name}
Purpose: {agent_purpose}

Instructions:
[TO BE FILLED IN BY USER]

You have access to Gmail tools to help complete your tasks. When given instructions:
1. Analyze what needs to be done
2. Use the appropriate tools to complete the task
3. Provide clear status updates on your actions

Be thorough, accurate, and efficient in your execution."""


class ExecutionAgent:
    """Manages state and history for an execution agent."""

    # Initialize execution agent with name, conversation limits, and log store access
    def __init__(
        self,
        name: str,
        conversation_limit: Optional[int] = None
    ):
        """
        Initialize an execution agent.

        Args:
            name: Human-readable agent name (e.g., 'conversation with keith')
            conversation_limit: How many recent requests to include in full; older ones
                are condensed. None = OPENPOKE_EXEC_HISTORY_TURNS (default 6), 0 = all.
        """
        self.name = name
        self.conversation_limit = DEFAULT_FULL_TURNS if conversation_limit is None else conversation_limit
        self._log_store = get_execution_agent_logs()

    # Generate system prompt template with agent name and purpose derived from name
    def build_system_prompt(self) -> str:
        """Build the system prompt for this agent."""
        agent_purpose = f"Handle tasks related to: {self.name}"

        return SYSTEM_PROMPT_TEMPLATE.format(
            agent_name=self.name,
            agent_purpose=agent_purpose
        )

    # Combine base system prompt with a bounded history (recent turns in full, older condensed)
    def build_system_prompt_with_history(self) -> str:
        """
        Build system prompt including agent history.

        Returns:
            System prompt with embedded history transcript
        """
        base_prompt = self.build_system_prompt()
        history = render_history(self._log_store.iter_entries(self.name), self.conversation_limit)
        if history:
            return f"{base_prompt}\n\n# Execution History\n\n{history}"
        return base_prompt

    # Format current instruction as user message for LLM consumption
    def build_messages_for_llm(self, current_instruction: str) -> List[Dict[str, str]]:
        """
        Build message array for LLM call.

        Args:
            current_instruction: Current instruction from interaction agent

        Returns:
            List of messages in OpenRouter format
        """
        return [
            {"role": "user", "content": current_instruction}
        ]

    # Log the agent's final response to the execution log store
    def record_response(self, response: str) -> None:
        """Record agent's response to the log."""
        self._log_store.record_agent_response(self.name, response)
        # keep the roster's "what did this agent last do" snippet fresh for routing
        get_agent_roster().record_result(self.name, response)

    # Log tool invocation and results with truncated content for readability
    def record_tool_execution(self, tool_name: str, arguments: str, result: str) -> None:
        """Record tool execution details."""
        self._log_store.record_action(self.name, f"Calling {tool_name} with: {arguments[:200]}")
        # Record the tool response
        self._log_store.record_tool_response(self.name, tool_name, result[:500])
