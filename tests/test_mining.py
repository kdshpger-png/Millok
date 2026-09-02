"""Real tests for the mining algorithm, no framework required.

Run:  python3 tests/test_mining.py [-v]

Style deliberately mirrors the "no dependency, plain asserts, one function
per behavior, print what happened" approach used throughout this codebase's
sibling projects - a test suite should be readable by someone who has never
seen pytest.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from millok.mining import extract_confirmations, extract_gaps, extract_pairs
from millok.types import Turn

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    mark = "OK " if condition else "!! "
    print(f"{mark}{name}")
    if not condition:
        print(f"      {detail}")


# --------------------------------------------------------------- basics

def test_simple_retry_becomes_a_pair():
    turns = [
        Turn("s1", "open budget file", "open_flie(x)", False, "unknown tool"),
        Turn("s1", "open budget file", "open_file(x)", True, ""),
    ]
    pairs = extract_pairs(turns)
    check("simple retry -> exactly one pair", len(pairs) == 1, pairs)
    if pairs:
        check("pair keeps the rejected attempt", pairs[0].rejected == "open_flie(x)")
        check("pair keeps the chosen attempt", pairs[0].chosen == "open_file(x)")
        check("pair keeps the failure reason", pairs[0].reason == "unknown tool")


def test_two_clean_successes_produce_nothing():
    turns = [
        Turn("s1", "open a", "open_file(a)", True, ""),
        Turn("s1", "open b", "open_file(b)", True, ""),
    ]
    check("no failures -> no pairs", extract_pairs(turns) == [])
    check("no failures -> no gaps", extract_gaps(turns) == [])


def test_unrelated_failure_is_not_paired_with_unrelated_success():
    turns = [
        Turn("s1", "open the budget file", "open_flie(x)", False, "unknown tool"),
        Turn("s1", "what's the weather", "search_web(q='weather')", True, ""),
    ]
    check("dissimilar intents stay unpaired", extract_pairs(turns) == [])
    gaps = extract_gaps(turns)
    check("the failed one shows up as a gap instead", len(gaps) == 1, gaps)


# ------------------------------------------------------- session isolation

def test_sessions_never_cross_contaminate():
    turns = [
        Turn("s1", "open budget file", "open_flie(x)", False, "unknown tool"),
        Turn("s2", "open budget file", "open_file(x)", True, ""),   # different session!
    ]
    check("a fix in another session must NOT pair", extract_pairs(turns) == [])
    gaps = extract_gaps(turns)
    check("s1's failure is a real gap, s2 changes nothing about it",
          len(gaps) == 1 and gaps[0].session == "s1", gaps)


# ------------------------------------------------------------------ window

def test_fix_outside_window_is_not_paired_but_counts_as_resolved():
    turns = [Turn("s1", "open budget file", "open_flie(x)", False, "unknown tool")]
    turns += [Turn("s1", f"filler {i}", "noop()", True, "") for i in range(10)]
    turns += [Turn("s1", "open budget file", "open_file(x)", True, "")]

    pairs = extract_pairs(turns, window=5)
    check("fix 11 turns later, window=5 -> not paired (too loose to trust)",
          pairs == [], pairs)

    gaps = extract_gaps(turns)
    check("but session-wide it DID get resolved eventually -> not a gap",
          gaps == [], gaps)

    pairs_wide = extract_pairs(turns, window=15)
    check("same fix, wider window -> now it IS paired", len(pairs_wide) == 1)


# ------------------------------------------------------------- many-to-one

def test_two_different_wrong_attempts_both_pair_with_the_final_success():
    turns = [
        Turn("s1", "remind me at 5pm", "set_reminder(t='5pm')", False, "bad format"),
        Turn("s1", "remind me at 5 pm", "set_reminder(t='5 pm')", False, "bad format"),
        Turn("s1", "remind me at 5pm", "set_reminder(t='17:00')", True, ""),
    ]
    pairs = extract_pairs(turns)
    check("both earlier tries pair with the eventual success", len(pairs) == 2, pairs)


# --------------------------------------------------------------- weighting

def test_recurring_failure_gets_higher_weight_than_a_one_off():
    turns = []
    for i in range(6):
        turns += [
            Turn(f"common_{i}", "search for x", "search_web(q=None)", False,
                 "missing required argument 'q'"),
            Turn(f"common_{i}", "search for x", "search_web(q='x')", True, ""),
        ]
    turns += [
        Turn("rare", "open y", "open_flie(y)", False, "unknown tool 'open_flie'"),
        Turn("rare", "open y", "open_file(y)", True, ""),
    ]
    pairs = extract_pairs(turns)
    common = [p for p in pairs if "missing required" in p.reason]
    rare = [p for p in pairs if "unknown tool" in p.reason]
    check("recurring failure reason -> weight equals its count",
          all(p.weight == 6 for p in common), common)
    check("one-off failure -> weight 1", all(p.weight == 1 for p in rare), rare)


# ------------------------------------------------------------------- gaps

def test_gap_requires_never_being_resolved_in_session():
    turns = [
        Turn("s1", "message the whole team", "send_message(to='team')", False,
             "unknown recipient group"),
    ]
    gaps = extract_gaps(turns)
    check("a failure with no matching success anywhere -> a gap", len(gaps) == 1)

    turns.append(Turn("s1", "message the whole team", "send_message(to='team_all')",
                      True, ""))
    check("...but not once it IS resolved later in the same session",
          extract_gaps(turns) == [])


def test_threshold_controls_how_loosely_intents_are_matched():
    # ratio(a, b) ~= 0.83 - deliberately measured with difflib beforehand,
    # not guessed, so this test asserts a real number instead of a hope.
    turns = [
        Turn("s1", "open the budget spreadsheet", "open_flie(x)", False, "typo"),
        Turn("s1", "please open the budget spreadsheet now", "open_file(x)", True, ""),
    ]
    strict = extract_pairs(turns, threshold=0.95)
    loose = extract_pairs(turns, threshold=0.7)
    check("threshold above the actual similarity rejects the paraphrase",
          strict == [], strict)
    check("threshold below the actual similarity accepts it", len(loose) == 1, loose)


# --------------------------------------------------------- confirmations

def test_successful_but_flagged_turn_is_a_confirmation():
    turns = [
        Turn("s1", "turn off the desk lamp", "smart_plug(id='lamp_2', on=False)",
             True, "two devices are named similarly, picked the closer match"),
    ]
    found = extract_confirmations(turns)
    check("flagged success -> counted as a confirmation", len(found) == 1, found)
    if found:
        check("the turn itself is returned unchanged", found[0].attempt == "smart_plug(id='lamp_2', on=False)")


def test_clean_success_is_not_a_confirmation():
    turns = [Turn("s1", "turn off the desk lamp", "smart_plug(id='lamp_2', on=False)", True, "")]
    check("no reason -> nothing to reinforce, not a confirmation",
          extract_confirmations(turns) == [])


def test_failure_with_a_reason_is_not_a_confirmation():
    # A `reason` on a FAILURE means "why it broke" - a totally different
    # thing from a `reason` on a SUCCESS, which means "why it hedged".
    # extract_confirmations must only ever look at the success side.
    turns = [Turn("s1", "turn off the desk lamp", "smart_plug(id=None)", False,
                  "no device id resolved")]
    check("failure, even with a reason, is never a confirmation",
          extract_confirmations(turns) == [])


def test_confirmations_need_no_pairing_or_window():
    # Deliberately no session relationship at all between these two -
    # extract_confirmations doesn't correlate turns, so that shouldn't matter.
    turns = [
        Turn("a", "mute notifications", "notify(muted=True)", True, "guessed scope=all"),
        Turn("b", "archive the report", "file_move(dst='archive/')", True, "guessed folder"),
    ]
    check("independent sessions both surface, no correlation needed",
          len(extract_confirmations(turns)) == 2)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    gut = sum(1 for _n, ok, _d in RESULTS if ok)
    print("-" * 60)
    print(f"{gut} / {len(RESULTS)} checks passed")
    return 0 if gut == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
