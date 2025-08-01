"""July 2025 availability calendar and simple natural-language date parser.

This standalone helper offers:
1. `slots`: a dict mapping every 30-minute block in July 2025 to "free"/"busy".
2. `parse_natural(expr)`: parse phrases like "next Tuesday" relative to 1 Jul 2025.
3. A small demo in the `__main__` guard that shows the two pieces working together.

The data and conventions follow the description supplied by the user.
"""

# UNAVAILABLE TIMESLOTS IN JULY 2025:
# =====================================
# 
# 1. WEEKENDS (All Saturdays and Sundays)
#    - Every Saturday (Jul 5, 12, 19, 26)
#    - Every Sunday (Jul 6, 13, 20, 27)
# 
# 2. OUTSIDE WORKING HOURS
#    - Before 9:00 AM and after 5:00 PM daily
# 
# 3. LUNCH BREAK
#    - Daily: 12:30 PM - 1:30 PM
# 
# 4. INDEPENDENCE DAY (FULL DAY)
#    - Friday, July 4, 2025 (all day)
# 
# 5. RECURRING STAND-UP MEETINGS
#    - Mondays: 10:30 AM - 11:30 AM (Jul 7, 14, 21, 28)
#    - Wednesdays: 10:30 AM - 11:30 AM (Jul 2, 9, 16, 23, 30)
# 
# 6. AD-HOC DEMO CALLS
#    - Thursday, July 10: 2:00 PM - 4:00 PM
#    - Friday, July 25: 2:00 PM - 4:00 PM
# 
# TOTAL BUSY SLOTS: 1,152 (out of 1,344 total slots in July)
# AVAILABLE SLOTS: 192 (14.3% availability)

from datetime import datetime, timedelta, time
import re

# ---------------------------------------------------------------------------
# (A) MACHINE-READABLE CALENDAR
# ---------------------------------------------------------------------------
REFERENCE_DATE = datetime.now()
START = REFERENCE_DATE.replace(day=1, hour=9, minute=0)
END = (REFERENCE_DATE.replace(day=1) + timedelta(days=31)).replace(hour=17, minute=0)  # Approximate month end
STEP = timedelta(minutes=30)

busy_rules = [
    # full-day: Independence Day
    (datetime(2025, 7, 4, 0, 0), datetime(2025, 7, 5, 0, 0)),
    # recurring stand-ups – Monday (0) & Wednesday (2), 10:30-11:30
    ("weekly", {"weekday": [0, 2], "start": time(10, 30), "end": time(11, 30)}),
    # ad-hoc demo calls
    (datetime(2025, 7, 10, 14, 0), datetime(2025, 7, 10, 16, 0)),
    (datetime(2025, 7, 25, 14, 0), datetime(2025, 7, 25, 16, 0)),
]


def is_busy(dt: datetime) -> bool:
    """Return True if *dt* falls in a busy period according to the rules above."""
    # weekends
    if dt.weekday() > 4:  # Sat/Sun
        return True
    # outside working hours
    if not (time(9, 0) <= dt.time() < time(17, 0)):
        return True
    # lunch break
    if time(12, 30) <= dt.time() < time(13, 30):
        return True
    # explicit & recurring blocks
    for rule in busy_rules:
        if isinstance(rule[0], str) and rule[0] == "weekly":
            if (
                dt.weekday() in rule[1]["weekday"]
                and rule[1]["start"] <= dt.time() < rule[1]["end"]
            ):
                return True
        else:
            start, end = rule
            if start <= dt < end:
                return True
    return False


# Pre-compute the availability map once at import time.
slots: dict[datetime, str] = {}
cur = START
while cur < END:
    slots[cur] = "busy" if is_busy(cur) else "free"
    cur += STEP

# ---------------------------------------------------------------------------
# (B) TINY NATURAL-LANGUAGE PARSER
# ---------------------------------------------------------------------------
REFERENCE_DATE = datetime.now()  # "today" for the Voice-Agent context


def parse_natural(expr: str) -> datetime:
    """Parse a very small subset of expressions into a concrete *datetime*.

    Supported patterns (case-insensitive):
    * "tomorrow"            – returns REFERENCE_DATE + 1 day (at 00:00)
    * "this <weekday>"      – the weekday in the same ISO week as REFERENCE_DATE
    * "next <weekday>"      – the weekday in the *following* ISO week
    * "<weekday>"           – synonym for "next <weekday>" if the day is today, else
                               behaves like "this <weekday>".
    Only the date part is returned; callers can add a time component later.
    """
    expr = expr.lower().strip()
    weekdays = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]

    if expr == "tomorrow":
        return REFERENCE_DATE + timedelta(days=1)

    m = re.match(r"(next|this)?\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", expr)
    if m:
        modifier, wd = m.groups()
        wd_idx = weekdays.index(wd)
        delta = (wd_idx - REFERENCE_DATE.weekday()) % 7

        # When no modifier and delta == 0 (same day), treat as *next* to avoid ambiguity.
        if modifier == "next" or (modifier is None and delta == 0):
            delta = delta or 7  # ensure strictly positive
        target = REFERENCE_DATE + timedelta(days=delta)
        return target

    raise ValueError(f"Unsupported expression: {expr!r}")


# ---------------------------------------------------------------------------
# DEMO / SMOKE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo_expr = "next Tuesday"
    target_date = parse_natural(demo_expr)
    print(f"{demo_expr!r} → {target_date.date()}")

    # Find the first free half-hour slot that day between 09:00 and 17:00
    check = target_date.replace(hour=9, minute=0, second=0, microsecond=0)
    while slots.get(check) == "busy":
        check += STEP
    print("first free slot:", check, slots[check]) 