"""Builds the synthetic demo log used by `millok demo` and by the tests.

Lives inside the package (not in examples/) on purpose: `examples/` is meant
to show how someone would write their OWN generator against real data, and
that file only needs to work when run from a checked-out repo. This one has
to work everywhere the package is installed, including as the pip-installed
`millok` command run from an arbitrary directory — that's a different job.

This is invented data for a generic "assistant with tools" — not drawn from
any real deployment. The failure patterns are modeled on things small local
models genuinely do: guessing a tool name that's close but wrong, using the
right tool with a malformed argument, and needing a rephrase before an
ambiguous request lands. A fixed seed keeps output identical on every run.
"""
from __future__ import annotations

import random

from .types import Turn

TOOLS = ["open_file", "send_message", "set_reminder", "search_web", "adjust_thermostat"]


def _session(name: str, events: list[tuple[str, str, bool, str]]) -> list[Turn]:
    return [Turn(session=name, intent=i, attempt=a, success=s, reason=r)
            for i, a, s, r in events]


def build_demo_log(seed: int = 26) -> list[Turn]:
    rng = random.Random(seed)
    turns: list[Turn] = []

    # Pattern 1: guessed the wrong tool name, error hint lets it self-correct.
    turns += _session("s1", [
        ("open the budget spreadsheet", "open_flie(name='budget')", False,
         "unknown tool 'open_flie' - did you mean 'open_file'?"),
        ("open the budget spreadsheet", "open_file(name='budget')", True, ""),
    ])

    # Pattern 2: right tool, malformed argument, corrected on retry.
    turns += _session("s2", [
        ("remind me at 5pm to call mom", "set_reminder(time='5pm', text='call mom')",
         False, "time must be 24h format, e.g. '17:00'"),
        ("remind me at 5pm to call mom", "set_reminder(time='17:00', text='call mom')",
         True, ""),
    ])

    # Pattern 3: ambiguous phrasing, user rephrases, then it lands. The
    # rejected attempt has to actually LOOK like the failure it claims to
    # be (zone=None) - an earlier draft had it identical to the successful
    # call, which would have made a training pair where "chosen" and
    # "rejected" are the same string. Caught by inspecting the demo's own
    # output before shipping it.
    turns += _session("s3", [
        ("make it warmer", "adjust_thermostat(zone=None, delta=+2)",
         False, "no zone specified and none is currently active"),
        ("make the living room warmer", "adjust_thermostat(zone='living_room', delta=+2)",
         True, ""),
    ])

    # Pattern 4: SAME systematic mistake, five different sessions - this is
    # the "recurring, not a fluke" case that weighting exists for.
    topics = ["weather", "news", "recipes", "flights", "movies"]
    for n, topic in enumerate(topics):
        turns += _session(f"s4_{n}", [
            (f"search for {topic} sites", "search_web(engine='duckduckgo', q=None)",
             False, "missing required argument 'q'"),
            (f"search for {topic} sites", f"search_web(engine='duckduckgo', q='{topic}')",
             True, ""),
        ])

    # Pattern 4b: succeeded, but the agent itself flagged uncertainty, and
    # got confirmed. No wrong attempt anywhere here - success=True from the
    # start, with the hedge kept in `reason`. This is what
    # extract_confirmations() looks for; deliberately unrelated in domain
    # to the other patterns (smart-plug naming, not tools/search) so it
    # doesn't blend into the retry-correction story above it.
    turns += _session("s4b", [
        ("turn off the desk lamp", "smart_plug(id='lamp_2', on=False)", True,
         "two devices are named similarly ('lamp_1', 'lamp_2') - picked the "
         "one used most recently, but wasn't certain which one was meant"),
    ])

    # Pattern 5: a genuine gap - tried, failed, NEVER resolved in-session.
    turns += _session("s5", [
        ("send a message to the whole team at once", "send_message(to='team', text='...')",
         False, "unknown recipient group 'team' - no such contact list exists"),
        ("something unrelated", "search_web(engine='duckduckgo', q='pizza')", True, ""),
    ])

    # Noise: plenty of clean first-try successes, so the demo isn't ONLY
    # failures - a real log is mostly this.
    clean_intents = [
        "open the project notes", "search for python tutorials",
        "remind me to stretch at 09:00", "send a message to sam",
        "set the office temperature to 21", "open the meeting agenda",
        "search for local weather", "remind me about the dentist at 14:30",
    ]
    for i, intent in enumerate(clean_intents):
        tool = rng.choice(TOOLS)
        turns += _session(f"clean_{i}", [(intent, f"{tool}(...)", True, "")])

    return turns
