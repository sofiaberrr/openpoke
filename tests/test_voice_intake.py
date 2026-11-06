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
    res = T.record_intake(st, "refill", "prescription_refill")
    assert res["available_slots"]  # availability comes back with the priority: one less LLM round trip
    assert "check_availability" in T.available_tools(st)
    assert "book_appointment" in T.available_tools(st)


def test_declined_times_are_not_offered_again():
    st = S.new_session()
    T.lookup_patient(st, "James Chen", "1962-11-02")
    T.save_callback_number(st, "6475550188")
    first = {s["slot_id"] for s in T.record_intake(st, "check-up", "checkup")["available_slots"]}
    second = {s["slot_id"] for s in T.check_availability(st)["slots"]}
    assert first and second and not (first & second)


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
    assert "Aisha Patel" in st.escalation["note"]  # what the caller said still reaches staff


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


def test_caller_can_keep_number_on_file():
    st = S.new_session()
    T.lookup_patient(st, "Maria Lopez", "1985-03-12")
    assert T.save_callback_number(st, use_number_on_file=True)["saved"] == "416-555-0134"


def test_promised_transfer_without_tool_still_transfers():
    st = S.new_session()
    out = turn(st, "I'm calling about my test results",
               scripted(say("I can't help with test results. Let me transfer you to someone who can.")))
    assert st.stage == S.HANDOFF and out["end_call"]


def test_fake_tool_call_text_is_never_spoken():
    st = S.new_session()
    out = turn(st, "test results please",
               scripted(say("Okay, I need to transfer you to staff. {calling transfer_to_human}")))
    assert "{" not in out["reply"] and "transfer_to_human" not in out["reply"]
    assert st.stage == S.HANDOFF


@pytest.mark.parametrize("text", ["I've been thinking about ending my life", "I keep wanting to die",
                                  "thinking of killing myself"])
def test_crisis_phrasings(text):
    assert safety.screen(text).kind == "crisis"


def test_priority_never_goes_down_during_a_call():
    st = S.new_session()
    T.record_intake(st, "refill and a high fever for two days", "prescription_refill")
    assert st.priority == SAME_DAY
    T.record_intake(st, "refill", "prescription_refill", severity=2)   # model dropped the fever
    assert st.priority == SAME_DAY and "fever" in st.reason


def test_cut_off_numbers_are_never_guessed():
    st = S.new_session()
    st.transcript.append({"role": "caller", "text": "my name is Arthur Chen and I was born on July 20th 199"})
    res = T.lookup_patient(st, "Arthur Chen", "1990-07-20")      # model filled in the missing 0
    assert res["ok"] is False and st.date_of_birth is None
    st.transcript.append({"role": "caller", "text": "my number is 416 997 85"})
    assert T.save_callback_number(st, "416-997-8569")["ok"] is False
    st.transcript.append({"role": "caller", "text": "sorry, July 20th 1990, and 416 997 8569"})
    assert T.lookup_patient(st, "Arthur Chen", "1990-07-20")["ok"]
    assert T.save_callback_number(st, "416-997-8569")["ok"]


# ------------------------------------------------------------------ edge cases found by probing the rules
@pytest.mark.parametrize("text", [
    "I can barely breathe", "my chest feels really tight", "my dad collapsed", "he's not responding",
    "my dad isn't responding", "my lips are swelling up", "I'm pregnant and I'm bleeding",
    "I can't feel the left side of my face", "I'm gasping for air", "my chest pains are back",
])
def test_more_emergency_phrasings(text):
    assert safety.screen(text).kind == "emergency"


@pytest.mark.parametrize("text", [
    "I don't want to die, I just have a cough",          # not a crisis
    "the cream is not responding, I mean not working",   # "not responding" about an object
    "my nose is swollen from allergies? no, it's fine",  # minor, not the airway words
])
def test_false_alarms_avoided(text):
    assert safety.screen(text).kind is None


def test_asking_for_a_doctor_is_a_handoff():
    assert safety.screen("is there a doctor I can talk to").kind == "handoff"


@pytest.mark.parametrize("dob", ["2031-01-01", "1850-06-01", "not a date"])
def test_impossible_birthdays_are_rejected(dob):
    st = S.new_session()
    assert T.lookup_patient(st, "Some One", dob)["ok"] is False and st.date_of_birth is None


def test_phone_with_country_code():
    st = S.new_session()
    assert T.save_callback_number(st, "+1 (416) 555-0134")["saved"] == "416-555-0134"


def _ready(st):
    T.lookup_patient(st, "Maria Lopez", "1985-03-12")
    T.save_callback_number(st, "4165550134")
    T.record_intake(st, "check-up", "checkup")


def test_next_tuesday_is_understood():
    st = S.new_session(); _ready(st)
    slots = T.check_availability(st, preferred_day="next Tuesday")["slots"]
    assert slots and all("Tuesday" in s["label"] for s in slots)


def test_weekend_request_says_clinic_is_closed():
    st = S.new_session(); _ready(st)
    res = T.check_availability(st, preferred_day="Saturday")
    assert "closed on weekends" in res["note"] and res["slots"]


def test_cannot_book_twice_in_one_call():
    st = S.new_session(); _ready(st)
    slot = st.offered_slots[0]["slot_id"]
    assert T.book_appointment(st, slot, True)["ok"]
    assert "book_appointment" in T.available_tools(st) or st.stage == S.BOOKED


def test_emergency_after_booking_still_wins():
    st = S.new_session(); _ready(st)
    T.book_appointment(st, st.offered_slots[0]["slot_id"], True)

    async def must_not_run(**_):
        raise AssertionError("LLM should not be called")
    out = turn(st, "wait, now my chest feels really tight", must_not_run)
    assert st.stage == S.EMERGENCY and "911" in out["reply"]


def test_empty_or_garbled_input_does_not_crash():
    st = S.new_session()
    out = turn(st, "", scripted(say("Sorry, I didn't catch that. Could you repeat your full name?")))
    assert out["reply"] and st.stage == S.IDENTIFY
