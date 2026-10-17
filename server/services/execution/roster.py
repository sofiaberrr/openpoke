"""Agent roster: the list of execution agents the interaction agent can route to.

Originally this was just a JSON list of names. That made "agent overload" unfixable,
because with only a name there is nothing to rank on. Each agent now has a small
record (what it was created for, what it did last, when it was last used) so the
selector can show the interaction agent only the agents that matter right now.

The on-disk format is backwards compatible: an old roster.json containing a plain
list of names is upgraded in place the first time it is loaded.
"""

from __future__ import annotations

import difflib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ...logging_config import logger

_SNIPPET_CHARS = 200

# Words that carry no meaning when comparing agent names ("Email to Alice" == "Email Alice").
_NAME_STOPWORDS = {
    "a", "an", "and", "the", "to", "for", "of", "with", "re", "about", "on", "in", "my", "from",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _snippet(text: Optional[str]) -> str:
    if not text:
        return ""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= _SNIPPET_CHARS else flat[: _SNIPPET_CHARS - 1] + "…"


def normalize_name(name: str) -> str:
    """Lowercase, drop punctuation and filler words, sort tokens.

    "Email to Alice" and "alice email" both become "alice email".
    """
    tokens = re.findall(r"[a-z0-9]+", name.lower())
    kept = sorted(t for t in tokens if t not in _NAME_STOPWORDS)
    return " ".join(kept)


def _same_tokens(a: List[str], b: List[str]) -> bool:
    """True if two sorted token lists match, allowing small typos in long words."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        if min(len(x), len(y)) < 5 or difflib.SequenceMatcher(None, x, y).ratio() < 0.8:
            return False
    return True


def find_duplicate_name(agent_name: str, existing_names) -> Optional[str]:
    """Return the existing name that means the same as `agent_name`, if any."""
    names = list(existing_names)
    if agent_name in names:
        return agent_name
    target = normalize_name(agent_name).split()
    if not target:
        return None
    for existing in names:
        if _same_tokens(normalize_name(existing).split(), target):
            return existing
    return None


@dataclass
class AgentRecord:
    """Everything the router knows about one execution agent."""

    name: str
    first_task: str = ""      # the instructions it was created with (its "purpose")
    last_task: str = ""       # the most recent instructions it received
    last_result: str = ""     # a snippet of its most recent reply
    created_at: str = ""
    last_used_at: str = ""
    use_count: int = 0

    def search_text(self) -> str:
        """Text the selector ranks against."""
        return " ".join(p for p in (self.name, self.first_task, self.last_task, self.last_result) if p)

    def last_used_ts(self) -> float:
        try:
            return datetime.fromisoformat(self.last_used_at).timestamp()
        except (TypeError, ValueError):
            return 0.0


class AgentRoster:
    """Stores agent records in a JSON file."""

    def __init__(self, roster_path: Path):
        self._roster_path = roster_path
        self._agents: Dict[str, AgentRecord] = {}
        self.load()

    # ------------------------------------------------------------------ persistence
    def load(self) -> None:
        """Load records from roster.json (accepts the old list-of-names format)."""
        self._agents = {}
        if not self._roster_path.exists():
            self.save()
            return
        try:
            data = json.loads(self._roster_path.read_text(encoding="utf-8") or "[]")
        except Exception as exc:
            logger.warning(f"Failed to load roster.json: {exc}")
            return

        for item in data if isinstance(data, list) else []:
            if isinstance(item, str):  # legacy format: just a name
                self._agents[item] = AgentRecord(name=item)
            elif isinstance(item, dict) and item.get("name"):
                fields = {k: v for k, v in item.items() if k in AgentRecord.__dataclass_fields__}
                self._agents[item["name"]] = AgentRecord(**fields)

    def save(self) -> None:
        """Save records atomically.

        The original code opened roster.json with "w" (which empties the file) and only
        then took the lock, so a crash or a concurrent reader could see an empty
        roster. Writing to a temp file and renaming it is atomic on POSIX.
        """
        payload = [asdict(record) for record in self._agents.values()]
        try:
            self._roster_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._roster_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, self._roster_path)
        except Exception as exc:
            logger.warning(f"Failed to save roster.json: {exc}")

    # ------------------------------------------------------------------ writes
    def add_agent(self, agent_name: str, first_task: str = "") -> None:
        """Add an agent if it is not already present."""
        if agent_name in self._agents:
            return
        now = _now_iso()
        self._agents[agent_name] = AgentRecord(
            name=agent_name, first_task=_snippet(first_task), created_at=now, last_used_at=now
        )
        self.save()

    def record_request(self, agent_name: str, instructions: str) -> None:
        """Mark an agent as used and remember what it was just asked to do."""
        record = self._agents.get(agent_name)
        if record is None:
            self.add_agent(agent_name, instructions)
            record = self._agents[agent_name]
        if not record.first_task:
            record.first_task = _snippet(instructions)
        record.last_task = _snippet(instructions)
        record.last_used_at = _now_iso()
        record.use_count += 1
        self.save()

    def record_result(self, agent_name: str, response: str) -> None:
        """Remember a snippet of the agent's latest reply."""
        record = self._agents.get(agent_name)
        if record is None:
            return
        record.last_result = _snippet(response)
        self.save()

    def clear(self) -> None:
        """Clear the agent roster."""
        self._agents = {}
        try:
            if self._roster_path.exists():
                self._roster_path.unlink()
            logger.info("Cleared agent roster")
        except Exception as exc:
            logger.warning(f"Failed to clear roster.json: {exc}")

    # ------------------------------------------------------------------ reads
    def get_agents(self) -> List[str]:
        """All agent names (kept for backwards compatibility)."""
        return list(self._agents.keys())

    def get_records(self) -> List[AgentRecord]:
        return list(self._agents.values())

    def get(self, agent_name: str) -> Optional[AgentRecord]:
        return self._agents.get(agent_name)

    def find_duplicate(self, agent_name: str) -> Optional[str]:
        """Return an existing agent whose name means the same thing, if any.

        Catches "Email to Alice" vs "email alice" vs "Email to Alcie" (typo), but not
        "Email to Alice" vs "Email to Bob", and not "Bob" vs "Rob" (short words are
        never treated as typos). Deliberately conservative: merging two genuinely
        different threads is worse than creating one extra agent.
        """
        return find_duplicate_name(agent_name, self._agents.keys())


_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_ROSTER_PATH = _DATA_DIR / "execution_agents" / "roster.json"

_agent_roster = AgentRoster(_ROSTER_PATH)


def get_agent_roster() -> AgentRoster:
    """Get the singleton roster instance."""
    return _agent_roster
