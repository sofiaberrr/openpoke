"""Interaction agent helpers for prompt construction."""

from html import escape
from pathlib import Path
from typing import Dict, List

from ...services.execution import get_agent_roster, select_agents
from ...services.execution.agent_selector import humanize_age

_prompt_path = Path(__file__).parent / "system_prompt.md"
SYSTEM_PROMPT = _prompt_path.read_text(encoding="utf-8").strip()

# How much of the recent conversation to use as extra routing context.
_CONTEXT_CHARS = 600


# Load and return the pre-defined system prompt from markdown file
def build_system_prompt() -> str:
    """Return the static system prompt for the interaction agent."""
    return SYSTEM_PROMPT


# Build structured message with conversation history, relevant agents, and current turn
def prepare_message_with_history(
    latest_text: str,
    transcript: str,
    message_type: str = "user",
) -> List[Dict[str, str]]:
    """Compose a message that bundles history, the agent shortlist, and the latest turn."""
    sections: List[str] = []

    sections.append(_render_conversation_history(transcript))
    sections.append(_render_active_agents(latest_text, transcript))
    sections.append(_render_current_turn(latest_text, message_type))

    content = "\n\n".join(sections)
    return [{"role": "user", "content": content}]


# Format conversation transcript into XML tags for LLM context
def _render_conversation_history(transcript: str) -> str:
    history = transcript.strip()
    if not history:
        history = "None"
    return f"<conversation_history>\n{history}\n</conversation_history>"


def render_agent_list(selection) -> str:
    """Render a Selection as the <active_agents> block. Shared with the evals."""
    if not selection.total:
        return "<active_agents>\nNone\n</active_agents>"

    lines: List[str] = []
    for record in selection.agents:
        attrs = [
            f'name="{escape(record.name, quote=True)}"',
            f'last_used="{humanize_age(record)}"',
            f'why_shown="{selection.reasons.get(record.name, "relevant")}"',
        ]
        purpose = escape(record.first_task or "", quote=False)
        last_ask = escape(record.last_task or "", quote=False)
        # Deliberately NOT showing the agent's last result here. In the LLM eval, showing
        # "Alice replied: Thursday works" made the model answer "did she reply?" straight
        # from that snippet instead of asking the agent to check. A "may be outdated"
        # label and a prompt rule did not stop it, so the snippet is now used only for
        # ranking (it is still part of AgentRecord.search_text) and never shown.
        body = purpose
        if last_ask and last_ask != purpose:
            body = f"{purpose} | last asked: {last_ask}" if purpose else f"last asked: {last_ask}"
        lines.append(f"<agent {' '.join(attrs)}>{body}</agent>")

    hidden = selection.total - len(selection.agents)
    header = (
        f"<active_agents shown=\"{len(selection.agents)}\" total=\"{selection.total}\">"
    )
    footer = ""
    if hidden > 0:
        footer = (
            f"\n<!-- {hidden} older or less relevant agents are hidden. If none of the agents "
            "above fit and the task might continue earlier work, call search_agents before "
            "creating a new agent. -->"
        )
    return header + "\n" + "\n".join(lines) + footer + "\n</active_agents>"


# Pick the agents worth showing this turn instead of dumping the whole roster
def _render_active_agents(latest_text: str = "", transcript: str = "") -> str:
    roster = get_agent_roster()
    roster.load()
    selection = select_agents(latest_text, roster.get_records(), context=transcript[-_CONTEXT_CHARS:])
    return render_agent_list(selection)


# Wrap the current message in appropriate XML tags based on sender type
def _render_current_turn(latest_text: str, message_type: str) -> str:
    tag = "new_agent_message" if message_type == "agent" else "new_user_message"
    body = latest_text.strip()
    return f"<{tag}>\n{body}\n</{tag}>"
