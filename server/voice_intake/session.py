"""Per-call state. The code owns the facts (who, why, priority, slot); the LLM owns the wording."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Stages of a call. "emergency" and "handoff" are terminal: no more booking.
IDENTIFY, INTAKE, SCHEDULE, CONFIRM, BOOKED, EMERGENCY, HANDOFF = (
    "identify", "intake", "schedule", "confirm", "booked", "emergency", "handoff",
)
TERMINAL = {BOOKED, EMERGENCY, HANDOFF}

GREETING = (
    "Hi, thanks for calling Maple Family Clinic. I'm the clinic's virtual assistant. "
    "If this is a medical emergency, please hang up and call 911. "
    "Can I start with your full name?"
)


@dataclass
class CallState:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    stage: str = IDENTIFY

    # identity
    full_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    patient_id: Optional[str] = None
    new_patient: bool = False
    callback_number: Optional[str] = None

    # intake
    reason: Optional[str] = None
    visit_type: Optional[str] = None
    severity: Optional[int] = None
    days_since_onset: Optional[float] = None
    getting_worse: Optional[bool] = None
    priority: Optional[str] = None
    priority_reasons: List[str] = field(default_factory=list)

    # scheduling
    offered_slots: List[Dict[str, Any]] = field(default_factory=list)
    booking: Optional[Dict[str, Any]] = None

    # escalation
    escalation: Optional[Dict[str, Any]] = None

    # demo switch: make tools fail to show the tool-failure handoff
    simulate_tool_failure: bool = False

    transcript: List[Dict[str, str]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def identity_complete(self) -> bool:
        return bool(self.full_name and self.date_of_birth and self.callback_number)

    def log(self, kind: str, **data: Any) -> None:
        self.events.append({"t": round(time.time() - self.created_at, 2), "kind": kind, **data})

    def summary_for_staff(self) -> str:
        """Built by code, not the LLM, so the handoff note is always accurate."""
        parts = [
            f"Caller: {self.full_name or 'unknown'}",
            f"DOB: {self.date_of_birth or 'unknown'}",
            f"Callback: {self.callback_number or 'unknown'}",
            f"Existing patient: {'yes (' + self.patient_id + ')' if self.patient_id else 'no / not found'}",
            f"Reason: {self.reason or 'not collected'}",
        ]
        if self.priority:
            parts.append(f"Priority: {self.priority} ({'; '.join(self.priority_reasons)})")
        if self.booking:
            parts.append(f"Booked: {self.booking['label']}")
        return " | ".join(parts)

    def public(self) -> Dict[str, Any]:
        data = asdict(self)
        data["identity_complete"] = self.identity_complete
        return data


_SESSIONS: Dict[str, CallState] = {}


def new_session() -> CallState:
    state = CallState()
    state.transcript.append({"role": "agent", "text": GREETING})
    _SESSIONS[state.session_id] = state
    return state


def get_session(session_id: str) -> Optional[CallState]:
    return _SESSIONS.get(session_id)
