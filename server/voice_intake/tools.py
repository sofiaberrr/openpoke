"""Mocked clinic tools + which tools the model is allowed to see at each point in the call.

Tool exposure is gated by state (same idea as the agent-overload fix): the model only
sees tools whose preconditions are met. It can't book before the caller is identified,
and it can't check availability before a priority exists. Fewer tools = fewer wrong calls,
shorter prompts, lower latency.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from dateutil import parser as dateparser

from . import session as S
from .triage import ROUTINE, SAME_DAY, assign_priority

TZ = ZoneInfo("America/Toronto")


class ToolFailure(RuntimeError):
    """A backend tool is down. Triggers a human handoff."""


# ----------------------------------------------------------------- mock data
PATIENTS = [
    {"id": "P-1001", "name": "maria lopez", "dob": "1985-03-12", "phone": "4165550134"},
    {"id": "P-1002", "name": "james chen", "dob": "1962-11-02", "phone": "6475550188"},
    {"id": "P-1003", "name": "aisha patel", "dob": "1990-07-25", "phone": "4375550102"},
    {"id": "P-1004", "name": "sofia berrada", "dob": "2004-05-20", "phone": "4165550199"},
]
BOOKED: set[str] = set()  # slots booked during this server run (lets us demo a conflict)


def _now() -> datetime:
    return datetime.now(TZ)


def _is_open(slot: datetime) -> bool:
    """Deterministic fake calendar: about 1 in 3 slots are free."""
    if slot.strftime("%Y-%m-%dT%H:%M") in BOOKED:
        return False
    h = int(hashlib.md5(slot.strftime("%Y-%m-%dT%H:%M").encode()).hexdigest(), 16)
    return h % 3 == 0


def _slots(days: int) -> List[datetime]:
    now = _now()
    out = []
    for d in range(days + 1):
        day = (now + timedelta(days=d)).date()
        if day.weekday() >= 5:
            continue
        for minutes in range(9 * 60, 16 * 60 + 31, 30):
            slot = datetime(day.year, day.month, day.day, minutes // 60, minutes % 60, tzinfo=TZ)
            if slot > now + timedelta(minutes=45) and _is_open(slot):
                out.append(slot)
    return out


def _label(slot: datetime) -> str:
    today = _now().date()
    if slot.date() == today:
        day = "today"
    elif slot.date() == today + timedelta(days=1):
        day = "tomorrow"
    else:
        day = slot.strftime("%A, %B ") + str(slot.day)
    return f"{day} at {slot.strftime('%I:%M %p').lstrip('0')}"


def _slot_dict(slot: datetime) -> Dict[str, str]:
    return {"slot_id": slot.strftime("%Y-%m-%dT%H:%M"), "label": _label(slot)}


def _matches(slot: datetime, preferred_day: Optional[str], time_of_day: Optional[str]) -> bool:
    if time_of_day == "morning" and slot.hour >= 12:
        return False
    if time_of_day == "afternoon" and slot.hour < 12:
        return False
    if preferred_day:
        p = preferred_day.lower().strip()
        today = _now().date()
        if p == "today":
            return slot.date() == today
        if p == "tomorrow":
            return slot.date() == today + timedelta(days=1)
        if p not in slot.strftime("%A").lower():
            return False
    return True


def _normalize_phone(raw: str) -> Optional[str]:
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits if len(digits) == 10 else None


def _normalize_dob(raw: str) -> Optional[str]:
    try:
        return dateparser.parse(raw).date().isoformat()
    except Exception:
        return None


# ----------------------------------------------------------------- tool implementations
def lookup_patient(state: S.CallState, full_name: str, date_of_birth: str) -> Dict[str, Any]:
    if state.simulate_tool_failure:
        raise ToolFailure("patient lookup service unavailable")
    dob = _normalize_dob(date_of_birth)
    if not dob:
        return {"ok": False, "error": "Could not understand the date of birth. Ask again (month, day, year)."}
    state.full_name, state.date_of_birth = full_name.strip().title(), dob
    for p in PATIENTS:
        if p["name"] == full_name.strip().lower() and p["dob"] == dob:
            state.patient_id, state.new_patient = p["id"], False
            return {"ok": True, "found": True, "patient_id": p["id"],
                    "phone_on_file_last4": p["phone"][-4:],
                    "next": "Ask for the best callback number (you may ask if the one ending in "
                            f"{p['phone'][-4:]} is still good)."}
    state.patient_id, state.new_patient = None, True
    return {"ok": True, "found": False,
            "next": "No record found. Don't say this alarmingly: say you'll set them up as a new patient, "
                    "then ask for the best callback number."}


def save_callback_number(state: S.CallState, phone: str) -> Dict[str, Any]:
    number = _normalize_phone(phone)
    if not number:
        return {"ok": False, "error": "That isn't a 10-digit number. Ask the caller to repeat it."}
    state.callback_number = f"{number[:3]}-{number[3:6]}-{number[6:]}"
    return {"ok": True, "saved": state.callback_number}


def record_intake(state: S.CallState, reason: str, visit_type: str, severity: Optional[int] = None,
                  days_since_onset: Optional[float] = None, getting_worse: Optional[bool] = None) -> Dict[str, Any]:
    state.reason, state.visit_type = reason, visit_type
    state.severity, state.days_since_onset, state.getting_worse = severity, days_since_onset, getting_worse
    p = assign_priority(visit_type=visit_type, reason=reason, severity=severity,
                        days_since_onset=days_since_onset, getting_worse=getting_worse)
    state.priority, state.priority_reasons = p.level, p.reasons
    state.offered_slots = []
    return {"ok": True, "priority": p.level, "means": f"should be seen {p.spoken}", "because": p.reasons,
            "next": "Briefly tell the caller how soon they should be seen, in plain words, without any medical "
                    "opinion. Then check availability."}


def check_availability(state: S.CallState, preferred_day: Optional[str] = None,
                       time_of_day: Optional[str] = None) -> Dict[str, Any]:
    if state.simulate_tool_failure:
        raise ToolFailure("scheduling system unavailable")
    window = {SAME_DAY: 0, "soon": 3, ROUTINE: 14}[state.priority]
    candidates = _slots(window)
    note = None
    if state.priority == SAME_DAY and not candidates:
        candidates = _slots(2)[:2]
        note = ("No same-day openings left. Offer the earliest slot below AND offer to transfer them to "
                "a nurse or staff member today.")
    preferred = [s for s in candidates if _matches(s, preferred_day, time_of_day)]
    if (preferred_day or time_of_day) and not preferred:
        note = (note or "") + " The caller's preferred time isn't available within the safe window; say so and offer these instead."
        preferred = candidates
    if state.priority == ROUTINE and not (preferred_day or time_of_day):
        preferred = preferred[1::3] or preferred  # spread routine options over different days
    picks = preferred[:3]
    state.offered_slots = [_slot_dict(s) for s in picks]
    if not picks:
        return {"ok": True, "slots": [], "next": "Nothing is available. Offer to transfer to staff."}
    return {"ok": True, "slots": state.offered_slots, "note": note,
            "next": "Offer at most 2 or 3 of these, naturally. Never invent other times."}


def book_appointment(state: S.CallState, slot_id: str, caller_confirmed: bool) -> Dict[str, Any]:
    if state.simulate_tool_failure:
        raise ToolFailure("booking system unavailable")
    offered = {s["slot_id"]: s for s in state.offered_slots}
    if slot_id not in offered:
        return {"ok": False, "error": "That slot wasn't offered. Only book a slot from check_availability."}
    if not state.identity_complete:
        return {"ok": False, "error": "Missing name, date of birth, or callback number. Collect it first."}
    if not caller_confirmed:
        return {"ok": False, "error": "Read back name, callback number and the time, and get a clear yes first."}
    if slot_id in BOOKED:
        state.offered_slots = []
        alt = check_availability(state)
        return {"ok": False, "conflict": True, "error": "That time was just taken.", "alternatives": alt.get("slots")}
    BOOKED.add(slot_id)
    state.booking = {**offered[slot_id], "confirmation": "MFC-" + slot_id[-5:].replace(":", "") + state.session_id[:3].upper()}
    state.stage = S.BOOKED
    return {"ok": True, "booked": state.booking,
            "next": "Confirm it's booked, say they'll get a text at their callback number, ask if there's anything else."}


def flag_emergency(state: S.CallState, reason: str) -> Dict[str, Any]:
    state.stage = S.EMERGENCY
    state.escalation = {"type": "emergency", "reason": reason, "note": state.summary_for_staff()}
    return {"ok": True}


def transfer_to_human(state: S.CallState, reason: str) -> Dict[str, Any]:
    state.stage = S.HANDOFF
    state.escalation = {"type": "handoff", "reason": reason, "note": state.summary_for_staff()}
    return {"ok": True}


IMPLS = {f.__name__: f for f in [lookup_patient, save_callback_number, record_intake, check_availability,
                                 book_appointment, flag_emergency, transfer_to_human]}


# ----------------------------------------------------------------- schemas + gating
def _fn(name: str, description: str, props: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": required}}}


SCHEMAS = {
    "lookup_patient": _fn("lookup_patient", "Look up the caller once you have their full name and date of birth.",
                          {"full_name": {"type": "string"}, "date_of_birth": {"type": "string", "description": "YYYY-MM-DD"}},
                          ["full_name", "date_of_birth"]),
    "save_callback_number": _fn("save_callback_number", "Save the caller's callback phone number.",
                                {"phone": {"type": "string"}}, ["phone"]),
    "record_intake": _fn("record_intake", "Record the reason for the visit once you have it (and the answers to your 1-2 clarifying questions). Returns the priority. Call again if the caller corrects something.",
                         {"reason": {"type": "string", "description": "Short summary in the caller's words"},
                          "visit_type": {"type": "string", "enum": ["new_problem", "follow_up", "prescription_refill", "checkup", "forms_or_paperwork", "other"]},
                          "severity": {"type": "integer", "description": "0-10 if the caller gave one"},
                          "days_since_onset": {"type": "number"},
                          "getting_worse": {"type": "boolean"}},
                         ["reason", "visit_type"]),
    "check_availability": _fn("check_availability", "Get real open appointment times for this caller's priority.",
                              {"preferred_day": {"type": "string", "description": "e.g. 'tomorrow', 'Thursday'"},
                               "time_of_day": {"type": "string", "enum": ["morning", "afternoon"]}}, []),
    "book_appointment": _fn("book_appointment", "Book an offered slot. Only after reading back name, callback number and time, and the caller saying yes.",
                            {"slot_id": {"type": "string"}, "caller_confirmed": {"type": "boolean"}},
                            ["slot_id", "caller_confirmed"]),
    "flag_emergency": _fn("flag_emergency", "Use if anything sounds like it could be an emergency. Ends booking.",
                          {"reason": {"type": "string"}}, ["reason"]),
    "transfer_to_human": _fn("transfer_to_human", "Transfer to clinic staff: caller asks, you're unsure, the situation is sensitive or upsetting, or something isn't working.",
                             {"reason": {"type": "string"}}, ["reason"]),
}


def available_tools(state: S.CallState) -> List[str]:
    # lookup / callback / intake stay available so the caller can correct things at any point
    names = ["flag_emergency", "transfer_to_human", "lookup_patient", "save_callback_number", "record_intake"]
    if state.priority:
        names.append("check_availability")
    if state.identity_complete and state.offered_slots:
        names.append("book_appointment")
    return names


def recompute_stage(state: S.CallState) -> None:
    if state.stage in S.TERMINAL:
        return
    if not state.identity_complete:
        state.stage = S.IDENTIFY
    elif not state.priority:
        state.stage = S.INTAKE
    elif not state.offered_slots:
        state.stage = S.SCHEDULE
    else:
        state.stage = S.CONFIRM
