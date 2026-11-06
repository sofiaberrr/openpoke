"""Rule-based priority for NON-emergency calls.

Explainable on purpose: every decision comes with the reasons that produced it,
so staff (and the caller) can see why. Conservative on purpose: when information
is missing, we round UP, never down. This is routing, not diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

SAME_DAY = "same_day"
SOON = "soon"          # within 3 days
ROUTINE = "routine"    # within ~2 weeks

WINDOW_DAYS = {SAME_DAY: 0, SOON: 3, ROUTINE: 14}
SPOKEN = {SAME_DAY: "we'd like to see you today",
          SOON: "we'd like to see you within the next few days",
          ROUTINE: "this can be a regular appointment, no rush"}

# Symptoms that should be seen today even when they don't sound like an emergency.
SAME_DAY_TERMS = [
    "high fever", "fever", "infection", "pus", "can't keep", "keep anything down", "dehydrat",
    "burning when", "pain when peeing", "uti", "ear pain", "spreading rash", "rash is spreading",
    "swollen", "cut", "stitches", "sprain", "pregnan",
]
ROUTINE_TYPES = {"follow_up", "prescription_refill", "checkup", "forms_or_paperwork"}


@dataclass
class Priority:
    level: str
    reasons: List[str] = field(default_factory=list)

    @property
    def window_days(self) -> int:
        return WINDOW_DAYS[self.level]

    @property
    def spoken(self) -> str:
        return SPOKEN[self.level]


def _bump(a: str, b: str) -> str:
    order = [ROUTINE, SOON, SAME_DAY]
    return a if order.index(a) >= order.index(b) else b


def assign_priority(
    *,
    visit_type: str,
    reason: str,
    severity: Optional[int] = None,
    days_since_onset: Optional[float] = None,
    getting_worse: Optional[bool] = None,
) -> Priority:
    level = ROUTINE
    reasons: List[str] = []
    text = (reason or "").lower()

    if visit_type in ROUTINE_TYPES and severity is None and not getting_worse:
        reasons.append(f"visit type is {visit_type.replace('_', ' ')}")
    else:
        level = SOON
        reasons.append("new or changing problem")

    if severity is not None:
        if severity >= 7:
            level = _bump(level, SAME_DAY)
            reasons.append(f"caller rated it {severity}/10")
        elif severity >= 4:
            level = _bump(level, SOON)
            reasons.append(f"caller rated it {severity}/10")
    elif visit_type not in ROUTINE_TYPES:
        reasons.append("severity unknown, so rounding up")

    if getting_worse:
        level = _bump(level, SAME_DAY if (days_since_onset is not None and days_since_onset <= 3) else SOON)
        reasons.append("getting worse")

    hits = [t for t in SAME_DAY_TERMS if t in text]
    if hits:
        level = _bump(level, SAME_DAY)
        reasons.append(f"mentions {hits[0]}")

    return Priority(level, reasons)
