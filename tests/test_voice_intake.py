"""Tests for the voice intake agent. No API key needed: the LLM is scripted.

Run:  python -m pytest tests/test_voice_intake.py -q
"""

import asyncio
import json

import pytest

from server.voice_intake import agent, safety, session as S, tools as T
from server.voice_intake.triage import ROUTINE, SAME_DAY, SOON, assign_priority


# ------------------------------------------------------------------ helpers
def call(name, **args):
    return {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": name, "function": {"name": name, "arguments": json.dumps(args)}}]}}]}


def say(text):
    return {"choices": [{"message": {"content": text}}]}


def scripted(*responses):
    """A fake LLM that returns these responses in order and records what tools it was shown."""
    it = iter(responses)
    seen = []

    async def llm(**kwargs):
        seen.append([t["function"]["name"] for t in kwargs["tools"]])
        return next(it)
    llm.seen = seen
    return llm


def turn(state, text, llm):
    return asyncio.run(agent.handle_turn(state, text, llm=llm))


@pytest.fixture(autouse=True)
def fresh_calendar():
    T.BOOKED.clear()


# ------------------------------------------------------------------ safety screen
@pytest.mark.parametrize("text", [
    "I've had chest pain for an hour",
    "my dad can't breathe properly",
    "her face is drooping and her speech is slurred",
    "he passed out in the kitchen",
    "the cut won't stop bleeding",
    "I think my throat is closing up",
])
def test_emergencies_are_caught(text):
    assert safety.screen(text).kind == "emergency"


@pytest.mark.parametrize("text", [
    "no chest pain, just a cough",
    "I don't have trouble breathing",
    "I need a refill on my blood pressure meds",
])
def test_negations_and_normal_requests_are_not_emergencies(text):
    assert safety.screen(text).kind is None


def test_crisis_and_handoff_requests():
    assert safety.screen("honestly I want to end my life").kind == "crisis"
    assert safety.screen("can I just talk to a real person").kind == "handoff"


# ------------------------------------------------------------------ triage rules
def test_priority_rules_are_conservative_and_explained():
    assert assign_priority(visit_type="prescription_refill", reason="refill").level == ROUTINE
    p = assign_priority(visit_type="new_problem", reason="sore knee")  # no severity given
    assert p.level == SOON and any("rounding up" in r for r in p.reasons)
    assert assign_priority(visit_type="new_problem", reason="sore throat", severity=8).level == SAME_DAY
    assert assign_priority(visit_type="new_problem", reason="cough", severity=3,
                           days_since_onset=2, getting_worse=True).level == SAME_DAY
    assert assign_priority(visit_type="follow_up", reason="ear infection follow-up").level == SAME_DAY  # "infection"


# ------------------------------------------------------------------ tool gating (overload fix applied to tools)
def test_booking_tools_hidden_until_preconditions_met():
    st = S.new_session()
    assert "check_availability" not in T.available_tools(st)
    assert "book_appointment" not in T.available_tools(st)
    T.lookup_patient(st, "Maria Lopez", "1985-03-12")
    T.save_callback_number(st, "416 555 0134")
    T.record_intake(st, "refill", "prescription_refill")
    assert "check_availability" in T.available_tools(st)
    assert "book_appointment" not in T.available_tools(st)
    T.check_availability(st)
    assert "book_appointment" in T.available_tools(st)


def test_book_requires_confirmation_and_offered_slot():
    st = S.new_session()
    T.lookup_patient(st, "Maria Lopez", "1985-03-12")
    T.save_callback_number(st, "4165550134")
    T.record_intake(st, "check-up", "checkup")
    slot = T.check_availability(st)["slots"][0]["slot_id"]
    assert T.book_appointment(st, "2099-01-01T09:00", True)["ok"] is False          # never offered
    assert T.book_appointment(st, slot, False)["ok"] is False                        # not confirmed
    assert T.book_appointment(st, slot, True)["ok"] is True
    assert st.stage == S.BOOKED


def test_double_booking_conflict_offers_alternatives():
    a, b = S.new_session(), S.new_session()
    for st in (a, b):
        T.lookup_patient(st, "James Chen", "1962-11-02")
        T.save_callback_number(st, "6475550188")
        T.record_intake(st, "annual physical", "checkup")
        T.check_availability(st)
    slot = a.offered_slots[0]["slot_id"]
    b.offered_slots = list(a.offered_slots)
    assert T.book_appointment(a, slot, True)["ok"]
    res = T.book_appointment(b, slot, True)
    assert res["conflict"] and res["alternatives"]


def test_bad_phone_and_unknown_patient():
    st = S.new_session()
    assert T.save_callback_number(st, "555-12")["ok"] is False
    res = T.lookup_patient(st, "New Person", "March 3 1999")
    assert res["found"] is False and st.new_patient and st.date_of_birth == "1999-03-03"


# ------------------------------------------------------------------ end-to-end turns (scripted LLM)
def test_routine_booking_flow():
    st = S.new_session()
    turn(st, "Maria Lopez, March 12 1985", scripted(
        call("lookup_patient", full_name="Maria Lopez", date_of_birth="1985-03-12"),
        say("Thanks Maria. What's the best number to reach you?")))
    turn(st, "416 555 0134", scripted(
        call("save_callback_number", phone="4165550134"),
        say("Got it. What can we help you with?")))
    turn(st, "I need a refill on my inhaler", scripted(
        call("record_intake", reason="inhaler refill", visit_type="prescription_refill"),
        call("check_availability"),
        say("I can do Tuesday at 10. Does that work?")))
    assert st.priority == ROUTINE and st.stage == S.CONFIRM
    slot = st.offered_slots[0]["slot_id"]
    llm = scripted(call("book_appointment", slot_id=slot, caller_confirmed=True), say("You're all set."))
    out = turn(st, "yes that's right", llm)
    assert st.stage == S.BOOKED and out["reply"] == "You're all set."


def test_emergency_mid_call_stops_booking_without_calling_llm():
    st = S.new_session()
    st.priority, st.offered_slots = SOON, [{"slot_id": "x", "label": "x"}]

    async def must_not_run(**_):
        raise AssertionError("LLM should not be called")
    out = turn(st, "actually now I'm getting chest pain", must_not_run)
    assert st.stage == S.EMERGENCY and "911" in out["reply"] and out["end_call"]
    out = turn(st, "can you still book me for Tuesday?", must_not_run)
    assert "911" in out["reply"] and st.booking is None


def test_llm_can_escalate_what_rules_missed():
    st = S.new_session()
    out = turn(st, "my husband is acting really confused and his lips look blue",
               scripted(call("flag_emergency", reason="confusion and blue lips")))
    assert st.stage == S.EMERGENCY and "911" in out["reply"]


def test_tool_failure_hands_off_to_human_with_summary():
    st = S.new_session()
    st.simulate_tool_failure = True
    out = turn(st, "Aisha Patel, July 25 1990", scripted(
        call("lookup_patient", full_name="Aisha Patel", date_of_birth="1990-07-25")))
    assert st.stage == S.HANDOFF and "front desk" in out["reply"]
    assert "tool failure" in st.escalation["reason"]


def test_llm_outage_hands_off_instead_of_crashing():
    st = S.new_session()

    async def broken(**_):
        raise RuntimeError("OpenRouter 503")
    out = turn(st, "hi I need an appointment", broken)
    assert st.stage == S.HANDOFF and out["end_call"]


def test_model_cannot_call_hidden_tool():
    st = S.new_session()
    llm = scripted(call("book_appointment", slot_id="x", caller_confirmed=True), say("What's your name?"))
    turn(st, "book me anything", llm)
    assert st.booking is None
    assert "book_appointment" not in llm.seen[0]
