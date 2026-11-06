"""Deterministic safety screen. Runs on EVERY caller utterance, before the LLM.

Why plain code and not the LLM: an emergency must never depend on the model
"deciding" to notice it. These rules are conservative (they over-trigger rather
than miss), and the LLM can still escalate things the rules miss via the
flag_emergency tool. The LLM can escalate, but it can never un-escalate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

# Phrases that mean "call 911 now". Kept readable on purpose so the clinic can audit them.
EMERGENCY_PATTERNS = {
    "chest pain": r"chest (pain|pressure|tightness|hurts?)|chest (feels|is) (really |very |so )?(tight|heavy|crushing)|pain in (my|his|her|the) chest|heart attack",
    "trouble breathing": r"(can'?t|cannot|can not|hard to|trouble|difficulty|struggling to|barely|hardly) (breath|catch (my|his|her) breath)|short(ness)? of breath|not breathing|choking|gasping",
    "stroke signs": r"stroke|face (is )?droop|slurred speech|slurring|(one side|left side|right side) (of (my|his|her) (body|face) )?(is )?(numb|weak)|can'?t feel (the |my |his |her )?(left|right|one) side|can'?t move (my|his|her) (arm|leg)",
    "unresponsive": r"unconscious|unresponsive|passed out|won'?t wake up|fainted|blacked out|collapsed|\b(he|she|they|baby|dad|mom|husband|wife|son|daughter|child|kid)('s| is| isn'?t| won'?t)( not)? respond",
    # Fever in a baby under 3 months is an ER visit (standard pediatric guidance).
    "fever in a young infant": r"\b(newborn|\d+[- ]?(day|week)s?[- ]old|(one|two|three|four|five|six|seven|eight|nine|ten|a few|1|2)[- ]?(day|week|month)s?[- ]old)\b.{0,80}\bfever|\bfever\b.{0,80}\b(newborn|\d+[- ]?(day|week)s?[- ]old|(one|two|1|2)[- ]?months?[- ]old)\b",
    "pregnancy bleeding": r"pregnan\w*.{0,40}bleed|bleed\w*.{0,40}pregnan",
    "severe bleeding": r"(won'?t|will not|can'?t) stop bleeding|bleeding (a lot|heavily|badly|everywhere)|severe bleeding|lots of blood",
    "seizure": r"seizure|convuls",
    "overdose or poisoning": r"overdos|took too many|poison",
    "severe allergic reaction": r"anaphyla|(throat|tongue|lips?|face) (is |are )?(swelling|swollen|closing)|epi ?pen",
    "worst headache": r"worst headache",
    "coughing or vomiting blood": r"(cough|vomit|throw)(ing)?( up)? blood",
}

# Mental-health crisis: tell them 988 / 911 and get a human on the line.
CRISIS_PATTERNS = r"suicid|kill(ing)? myself|end(ing)? (my life|it all)|want(ing)? to die|hurt(ing)? myself|self[- ]harm"

# Caller explicitly wants a person.
HANDOFF_PATTERNS = r"(talk|speak) to (a|the) (doctor|nurse)|doctor i can (talk|speak) to|\b(human|real person|a person|someone real|receptionist|front desk|operator|talk to (a|the) (nurse|doctor|staff))\b|\b(speak|talk) to (an? )?agent\b"

# Words right before a match that flip its meaning ("no chest pain").
NEGATIONS = r"\b(no|not|never|without|don'?t have|doesn'?t have|denies|isn'?t|wasn'?t|any)\s+(\w+\s+){0,2}$"


@dataclass
class ScreenResult:
    kind: Optional[str]  # "emergency" | "crisis" | "handoff" | None
    reasons: List[str]


def _negated(text: str, start: int) -> bool:
    window = text[max(0, start - 30):start]
    return re.search(NEGATIONS, window) is not None


def screen(utterance: str) -> ScreenResult:
    text = utterance.lower()

    for m in re.finditer(CRISIS_PATTERNS, text):
        # "I don't want to die, I just have a cough" is not a crisis
        if not re.search(r"\b(don'?t|do not|never)\s+$", text[max(0, m.start() - 12):m.start()]):
            return ScreenResult("crisis", ["possible mental health crisis"])

    hits = []
    for label, pattern in EMERGENCY_PATTERNS.items():
        for m in re.finditer(pattern, text):
            if not _negated(text, m.start()):
                hits.append(label)
                break
    if hits:
        return ScreenResult("emergency", hits)

    if re.search(HANDOFF_PATTERNS, text):
        return ScreenResult("handoff", ["caller asked for a person"])

    return ScreenResult(None, [])


EMERGENCY_SCRIPT = (
    "I'm concerned this could be an emergency. Please hang up and call 911 right now, "
    "or have someone take you to the nearest emergency room. I'm not going to book a regular appointment for this."
)
EMERGENCY_REPEAT = "Please hang up and call 911 now. That's the fastest way to get help."
CRISIS_SCRIPT = (
    "I'm really glad you told me. If you're in immediate danger, please call 911. "
    "You can also call or text 988 any time to talk to someone right now. "
    "I'm connecting you with a member of our staff."
)
