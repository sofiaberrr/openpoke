"""Build realistic fake rosters so we can test routing at 25 / 100 / 300 / 1000 agents.

Two kinds of agents:
  * ANCHORS: hand-written agents that the test cases point at. Several are deliberate
    traps: two different Alices, Bob vs Rob, Tokyo vs Kyoto.
  * FILLER: generated agents that look like weeks of normal use (emails to people,
    reminders, receipts, trips). Filler reuses common first names, so at large
    sizes there are extra "Alice"s and "Bob"s competing with the anchors.

Everything is seeded, so every run produces the same roster.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import List

from server.services.execution.roster import AgentRecord

NOW = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)


def _ago(days: float) -> str:
    return (NOW - timedelta(days=days)).isoformat(timespec="seconds")


def _agent(name: str, first: str, result: str = "", days_ago: float = 10, uses: int = 2, last: str = "") -> AgentRecord:
    return AgentRecord(
        name=name,
        first_task=first,
        last_task=last or first,
        last_result=result,
        created_at=_ago(days_ago + 5),
        last_used_at=_ago(days_ago),
        use_count=uses,
    )


ANCHORS: List[AgentRecord] = [
    _agent("Email to Alice Chen re lunch",
           "Email Alice Chen (alice.chen@acme.com) asking if she's free for lunch Thursday. Bob is also coming.",
           "Alice replied: Thursday works, she suggested Chez Philippe at 12:30.", 2, 3),
    _agent("Email to Alice Park re Q3 budget",
           "Reply to Alice Park's email about the Q3 marketing budget spreadsheet and ask for the final numbers.",
           "Draft sent to Alice Park asking for final Q3 budget figures.", 40, 2),
    _agent("Landlord lease renewal",
           "Find the email from Mark Ruiz (landlord) about renewing the apartment lease and draft a reply.",
           "Mark says the new lease starts Jan 1 with a 3% rent increase; draft reply ready.", 10, 2),
    _agent("Tokyo trip flights",
           "Find the Air Canada booking confirmation for the Tokyo trip in March and list flight times.",
           "Flights: YYZ to HND March 12 at 13:05, return March 24.", 5, 2),
    _agent("Kyoto ryokan booking",
           "Search for the ryokan reservation email in Kyoto and check the cancellation policy.",
           "Ryokan Yachiyo, March 18-20, free cancellation until March 1.", 60, 1),
    _agent("Vercel job offer",
           "Find the offer letter from Priya, the Vercel recruiter, and summarize the compensation.",
           "Offer: base salary, equity over 4 years, start date in January. Response due Oct 24.", 20, 3),
    _agent("Stripe interview scheduling",
           "Reply to the Stripe recruiting coordinator with my availability for the onsite interview.",
           "Coordinator confirmed onsite for Oct 22 at 10am.", 3, 4),
    _agent("Nightly email summary",
           "Every night at 9pm, summarize today's important emails and send me the summary.",
           "Trigger created: daily at 21:00.", 1, 30),
    _agent("Meditation reminder",
           "Remind me to meditate every morning at 7am.",
           "Trigger created: daily at 07:00.", 30, 1),
    _agent("Dentist appointment",
           "Find the confirmation email from Bloor Dental and remind me the day before the appointment.",
           "Cleaning booked Nov 3 at 4pm. Reminder set for Nov 2.", 15, 1),
    _agent("Figma invoice",
           "Find the latest invoice from Figma and check how much we were charged.",
           "Figma Professional invoice for October found.", 45, 1),
    _agent("Rent payment reminder",
           "Remind me to pay rent on the first of every month.",
           "Trigger created: monthly on day 1 at 09:00.", 25, 1),
    _agent("Email to Bob Martinez re soccer",
           "Email Bob Martinez asking if he's still playing in Sunday's soccer game.",
           "Bob replied he's in, and asked who's bringing the ball.", 7, 2),
    _agent("Email to Rob Kim re thesis",
           "Email Professor Rob Kim with the latest draft of my thesis chapter 3.",
           "Sent chapter 3 draft to Rob Kim.", 50, 1),
    _agent("Mom birthday gift",
           "Search for the order confirmation of the necklace I bought for Mom's birthday and track shipping.",
           "Necklace ships Oct 20, arrives Oct 23.", 12, 2),
    _agent("Sam and Jordan wedding RSVP",
           "Find the wedding invitation from Sam and Jordan and RSVP yes for two.",
           "RSVP sent: 2 guests, chicken entree.", 35, 1),
    _agent("Netflix cancellation",
           "Find the Netflix billing email and figure out how to cancel the subscription.",
           "Cancellation link found; subscription ends Nov 5.", 70, 1),
    _agent("Conference travel reimbursement",
           "Collect receipts from the NeurIPS conference trip and draft the reimbursement email to finance.",
           "Draft reimbursement email with 6 receipts ready.", 22, 2),
    _agent("Professor Lee research meeting",
           "Schedule a research meeting with Professor Lee next week about the quantum project.",
           "Professor Lee proposed Tuesday 2pm.", 8, 2),
    _agent("Amazon headphones return",
           "Start a return for the Sony headphones ordered on Amazon and find the return label.",
           "Return label emailed; drop off by Oct 30.", 4, 2),
    _agent("Gym membership freeze",
           "Email Goodlife Fitness to freeze my gym membership for November.",
           "Goodlife confirmed the freeze from Nov 1 to Nov 30.", 55, 1),
    _agent("GitHub security alert",
           "Check the GitHub security alert email about a leaked token in my repo.",
           "Alert was about a revoked OpenAI key in an old commit.", 90, 1),
]

# --- filler -------------------------------------------------------------------------

_FIRST = ["Alice", "Bob", "Carlos", "Dana", "Ethan", "Fatima", "George", "Hana", "Ivan", "Julia",
          "Kevin", "Lina", "Marco", "Nadia", "Omar", "Paula", "Quinn", "Rosa", "Sanjay", "Tara",
          "Uma", "Victor", "Wei", "Ximena", "Yusuf", "Zoe", "Mark", "Priya", "Sam", "Lee"]
_LAST = ["Nguyen", "Smith", "Garcia", "Patel", "Khan", "Lopez", "Brown", "Wilson", "Ali", "Rossi",
         "Kowalski", "Tanaka", "Okafor", "Dubois", "Silva", "Cohen", "Murphy", "Singh", "Haddad", "Novak"]
_EMAIL_TOPICS = [
    ("coffee chat", "asking if they're free for a coffee chat next week", "They suggested Wednesday at 3pm."),
    ("project update", "with a status update on the group project", "They acknowledged and asked for slides."),
    ("intro request", "asking for an intro to someone on their team", "They agreed to make the intro."),
    ("contract review", "asking them to review the contract draft", "They sent back two comments."),
    ("referral", "asking for a referral for the internship posting", "They submitted the referral."),
    ("party invite", "inviting them to the Saturday house party", "They said they might come late."),
    ("study group", "about the study group before the midterm", "They proposed meeting in the library."),
    ("dinner plans", "about dinner plans on Friday", "They booked a table for 7pm."),
    ("shared doc access", "asking for edit access to the shared doc", "Access granted."),
    ("thank you note", "thanking them for their help last week", "Sent."),
    ("rescheduling", "asking to reschedule our call", "They moved it to Monday."),
    ("photos", "asking them to send the photos from the trip", "They shared an album link."),
]
_ERRANDS = [
    ("{store} order tracking", "Track the {store} order and tell me when it arrives.", "Package arrives in 3 days."),
    ("{store} receipt", "Find the receipt from {store} for my records.", "Receipt found."),
    ("{store} refund", "Check if the {store} refund went through.", "Refund processed."),
]
_STORES = ["Uniqlo", "IKEA", "Best Buy", "Indigo", "Apple", "Lululemon", "Sephora", "Canadian Tire",
           "Costco", "Etsy", "Shoppers", "Walmart", "Steam", "Zara", "Nike"]
_REMINDERS = [
    "water the plants", "call grandma", "submit the timesheet", "take out recycling", "renew passport",
    "back up laptop", "pay the phone bill", "stretch", "review flashcards", "book a haircut",
    "update resume", "check the transit pass balance", "order contact lenses", "file expense report",
]
_NEWSLETTERS = ["Substack digest", "LinkedIn notifications", "university announcements", "course forum",
                "bank alerts", "Slack digest", "Google Scholar alerts", "job board alerts"]
_PLACES = ["Montreal", "Vancouver", "New York", "Chicago", "Lisbon", "Mexico City", "Banff", "Boston",
           "Quebec City", "Ottawa", "Seoul", "London"]


def _filler_agent(rng: random.Random, i: int) -> AgentRecord:
    kind = rng.random()
    days = rng.uniform(0, 120)
    if kind < 0.45:
        person = f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
        topic, ask, result = rng.choice(_EMAIL_TOPICS)
        return _agent(f"Email to {person} re {topic}", f"Email {person} {ask}.",
                      f"{person.split()[0]} replied. {result}", days, rng.randint(1, 4))
    if kind < 0.65:
        tmpl, ask, result = rng.choice(_ERRANDS)
        store = rng.choice(_STORES)
        return _agent(tmpl.format(store=store), ask.format(store=store), result, days, 1)
    if kind < 0.82:
        what = rng.choice(_REMINDERS)
        when = rng.choice(["every Monday", "tomorrow at 9am", "every Sunday night", "on the 15th", "in two weeks"])
        return _agent(f"Reminder: {what} ({when})", f"Remind me to {what} {when}.", "Trigger created.", days, 1)
    if kind < 0.92:
        src = rng.choice(_NEWSLETTERS)
        return _agent(f"Summarize {src}", f"Summarize the latest {src} emails.", "Summary delivered.", days, 1)
    place = rng.choice(_PLACES)
    what = rng.choice(["hotel", "train tickets", "restaurant reservation", "itinerary", "car rental"])
    return _agent(f"{place} {what}", f"Find the {what} confirmation for the {place} trip.",
                  f"{place} {what} confirmation found.", days, 1)


def build_roster(size: int, seed: int = 7) -> List[AgentRecord]:
    """Anchors + enough filler to reach `size` agents (names are unique)."""
    rng = random.Random(seed)
    roster = list(ANCHORS)
    names = {a.name for a in roster}
    i = 0
    while len(roster) < size:
        agent = _filler_agent(rng, i)
        i += 1
        if agent.name in names:
            continue
        names.add(agent.name)
        roster.append(agent)
    rng.shuffle(roster)
    return roster
