"""Labelled routing test cases.

Each case is a user message (plus optional recent conversation) and the agent that
SHOULD handle it. `expected=None` means no existing agent fits and a new one should
be created.

Categories, so results can be broken down by difficulty:
  direct      - names the person/thing ("Did Alice Chen reply?")
  paraphrase  - same thing, different words ("my landlord" for "Mark Ruiz ... lease")
  distractor  - a confusable agent exists (Alice Chen vs Alice Park, Bob vs Rob)
  follow_up   - names nothing; only the recent conversation says what it is about
  new_task    - nothing existing fits; the right move is a new agent
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Case:
    id: str
    message: str
    expected: Optional[str]
    category: str
    context: str = ""  # recent conversation, as the interaction agent would see it


def _ctx(user: str, assistant: str) -> str:
    return f"<user_message>{user}</user_message>\n<poke_reply>{assistant}</poke_reply>"


CASES = [
    # --- direct ----------------------------------------------------------------------
    Case("d1", "Did Alice Chen reply about lunch?", "Email to Alice Chen re lunch", "direct"),
    Case("d2", "What time is my Stripe onsite again?", "Stripe interview scheduling", "direct"),
    Case("d3", "When does the Vercel offer need a response by?", "Vercel job offer", "direct"),
    Case("d4", "Has the Figma invoice been paid?", "Figma invoice", "direct"),
    Case("d5", "Change the nightly email summary to 10pm instead", "Nightly email summary", "direct"),
    Case("d6", "Stop the meditation reminder", "Meditation reminder", "direct"),
    Case("d7", "Did Goodlife confirm the gym freeze?", "Gym membership freeze", "direct"),
    Case("d8", "When is the Amazon headphones return due?", "Amazon headphones return", "direct"),
    Case("d9", "Send the NeurIPS reimbursement email to finance", "Conference travel reimbursement", "direct"),
    Case("d10", "Can you check the ryokan cancellation deadline in Kyoto?", "Kyoto ryokan booking", "direct"),

    # --- paraphrase ------------------------------------------------------------------
    Case("p1", "What did my landlord say about renewing?", "Landlord lease renewal", "paraphrase"),
    Case("p2", "When does my flight to Japan leave?", "Tokyo trip flights", "paraphrase"),
    Case("p3", "How much are they paying me at Vercel?", "Vercel job offer", "paraphrase"),
    Case("p4", "Is my teeth cleaning still on for next week?", "Dentist appointment", "paraphrase"),
    Case("p5", "Where's the package with mom's present?", "Mom birthday gift", "paraphrase"),
    Case("p6", "Did we tell Sam and Jordan we're coming to the wedding?", "Sam and Jordan wedding RSVP", "paraphrase"),
    Case("p7", "Push back the rent reminder to the 3rd of the month", "Rent payment reminder", "paraphrase"),
    Case("p8", "Is my streaming subscription actually cancelled?", "Netflix cancellation", "paraphrase"),
    Case("p9", "Was that leaked key thing on GitHub serious?", "GitHub security alert", "paraphrase"),
    Case("p10", "Ask my landlord if I can sign for two years instead", "Landlord lease renewal", "paraphrase"),

    # --- distractor ------------------------------------------------------------------
    Case("x1", "Reply to Alice Park and ask for the budget spreadsheet again", "Email to Alice Park re Q3 budget", "distractor"),
    Case("x2", "Tell Alice Chen 12:30 at Chez Philippe works", "Email to Alice Chen re lunch", "distractor"),
    Case("x3", "Tell Rob Kim I'll send chapter 4 next week", "Email to Rob Kim re thesis", "distractor"),
    Case("x4", "Ask Bob Martinez to bring the ball on Sunday", "Email to Bob Martinez re soccer", "distractor"),
    Case("x5", "What hotel are we staying at in Kyoto?", "Kyoto ryokan booking", "distractor"),
    Case("x6", "Did Professor Lee confirm Tuesday?", "Professor Lee research meeting", "distractor"),

    # --- follow-up (context carries the meaning) --------------------------------------
    Case("f1", "did she reply yet?", "Email to Alice Chen re lunch", "follow_up",
         _ctx("Can you ask Alice Chen if she's free for lunch Thursday?", "Sent Alice Chen a lunch invite for Thursday.")),
    Case("f2", "ok send it", "Landlord lease renewal", "follow_up",
         _ctx("What did Mark say about the lease?", "Mark says the new lease starts Jan 1 with a 3% increase. I drafted a reply accepting.")),
    Case("f3", "and the return flight?", "Tokyo trip flights", "follow_up",
         _ctx("When do I fly out to Tokyo?", "Your Air Canada flight leaves March 12 at 13:05.")),
    Case("f4", "make it 2 years instead", "Landlord lease renewal", "follow_up",
         _ctx("Reply to the landlord saying I'll renew the lease", "Drafted a reply to Mark accepting the renewal.")),
    Case("f5", "actually move it to 8am", "Meditation reminder", "follow_up",
         _ctx("Is my meditation reminder still on?", "Yep, every morning at 7am.")),

    # --- hard paraphrase: no words in common with the agent at all ----------------------
    # Keyword matching should fail these. They are here on purpose, to measure what
    # embeddings and the search_agents fallback actually buy us.
    Case("h1", "When am I heading to Asia?", "Tokyo trip flights", "hard_paraphrase"),
    Case("h2", "Did the guy who owns my place answer?", "Landlord lease renewal", "hard_paraphrase"),
    Case("h3", "Is my workout spot on hold next month?", "Gym membership freeze", "hard_paraphrase"),
    Case("h4", "Did my supervisor get what I wrote?", "Email to Rob Kim re thesis", "hard_paraphrase"),

    # --- new task (should NOT reuse anything) ------------------------------------------
    Case("n1", "Email Carla Mendes asking for the hackathon slides", None, "new_task"),
    Case("n2", "Remind me to call the bank tomorrow at 2pm", None, "new_task"),
    Case("n3", "Find my Uber receipt from last Friday", None, "new_task"),
    Case("n4", "Ask Alice Chen's manager for a reference letter", None, "new_task"),
    Case("n5", "Look for the email about my Rogers phone bill increase", None, "new_task"),
]
