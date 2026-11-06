# Voice Intake Agent (Maple Family Clinic)

A voice agent for doctor's office intake, urgency routing, scheduling and human handoff, built on OpenPoke.

## Run it

```bash
pip install -r server/requirements.txt          # once
# .env needs OPENROUTER_API_KEY. Model: VOICE_AGENT_MODEL (default openai/gpt-4o-mini, picked for speed)
python -m uvicorn server.voice_intake.app:app --port 8001
```
Open **http://localhost:8001/voice in Chrome**, click **Start call**, and talk. Use headphones so the mic doesn't hear the agent. You can also type in the box.

- Tests (no API key needed): `python -m pytest -q`
- Scripted scenarios against the real model: `python -m evals.voice_scenarios`
- Type to the agent in the terminal: `python -m evals.voice_scenarios --chat`

## The flow

```
caller speaks -> browser speech-to-text -> POST /api/v1/voice/turn
   1. stage is emergency/handoff?  -> fixed reply, never goes back to booking
   2. SAFETY SCREEN (plain code)   -> emergency: "call 911", stop | crisis: 911/988 + human | "real person": human
   3. LLM turn, with only the tools that make sense right now
   4. tool failure / LLM error     -> human handoff with a summary
reply -> browser text-to-speech (the caller can talk over it to interrupt)
```

Stages: `identify -> intake -> schedule -> confirm -> booked`, plus `emergency` and `handoff`, which end the call.

## How it maps to OpenPoke

OpenPoke has an **interaction agent** (talks to the user) that hands work to **execution agents** (do the tasks). Here the voice agent is the interaction agent, and the clinic tools (patient lookup, availability, booking, transfer) are the work it hands off. They're mocked, so they could be swapped for real systems without changing the conversation.

## Key design decisions

| Decision | Why |
|---|---|
| **Emergency screen is code, not the LLM**, and runs before the model on every turn | An emergency should never depend on the model noticing it. The rules over-trigger on purpose. The LLM can *add* an emergency via `flag_emergency` but can never remove one. |
| **Priority is rules** (`triage.py`): same-day / soon (3 days) / routine | Explainable: every decision lists its reasons. Conservative: missing info rounds up. No diagnosis. |
| **Code owns the facts, the LLM owns the wording** | Name, DOB, callback, priority and slots live in `CallState` and get injected into the prompt each turn. The transcript is capped at 12 turns, so the facts can't get lost or made up. |
| **Tools are gated by state** (agent overload, applied to tools) | The model only sees tools whose preconditions are met: no `check_availability` before a priority exists, no `book_appointment` before identity and offered slots. Fewer tools means fewer wrong calls and shorter prompts. |
| **Booking is enforced in code** | `book_appointment` refuses unless the slot was actually offered, identity is complete, and `caller_confirmed=true` (after a read-back). |
| **Every failure ends with a human** | Tool down, LLM error, caller asks, or the model is unsure -> `transfer_to_human`. The staff note is built by code from the state, so it's accurate. |
| **Voice UX** | 1-2 sentences, one question per turn, varied acknowledgements, corrections accepted. A fixed greeting includes the 911 line. Interrupting the agent stops its speech. |

## Demo scenarios

1. **Routine:** Maria Lopez (Mar 12 1985), inhaler refill -> routine -> offered times -> read-back -> booked
2. **Higher priority:** James Chen (Nov 2 1962), earache + fever, 8/10, getting worse -> same-day
3. **Emergency:** "chest pain and my left arm is numb" -> 911 script, booking stopped, even if they ask again
4. **Handoff:** "can I talk to a real person" -> transfer with a summary. Or tick **Simulate tool failure** -> handoff

Mock patients are in `server/voice_intake/tools.py`. Anyone else is treated as a new patient.

## What I'd improve with more time

- **Latency:** stream the LLM output and start speaking at the first sentence. Use a real-time voice stack (e.g. Deepgram + ElevenLabs, or a speech-to-speech model) instead of browser speech. Add a short filler ("one sec, checking") while tools run.
- **Reliability:** better end-of-turn detection than a silence timer. Repeat back phone numbers digit by digit. Retry tools before handing off.
- **Evaluation:** a larger labelled set of caller scripts (accents, rambling, corrections, edge emergencies), with LLM-simulated callers and metrics: emergency recall (must be 100%), over-triage rate, booking success, turns to book, latency p95.
- **Safety:** have clinicians review the emergency phrases and triage rules. Add a second-pass emergency classifier with the rules kept as the floor. Handle minors, third-party callers and language barriers.
- **Privacy:** PHI shouldn't sit in memory or logs in plain text. Use a HIPAA/PHIPA-compliant LLM provider with zero data retention, verify identity before revealing anything on file (we only reveal the last 4 digits of the phone number), and set retention rules for transcripts.
- **Agent/tool overload:** as real integrations get added (pharmacy, insurance, forms), keep gating tools by stage, or route to small sub-agents with their own short tool lists, like the shortlist fix in `SOLUTION.md`.
