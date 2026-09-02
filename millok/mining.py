"""The core idea, in one sentence:

    A correction is training data you already paid for.

Every agent that retries after a failure — because the user rephrased, or
because an error message told it what went wrong, or because it simply tried
again — produces a labeled example as a side effect of normal operation.
Nobody sat down and wrote "here's the right answer"; the retry loop already
did that, for free, the moment it succeeded.

Millok does not train anything and does not judge quality. It looks for two
shapes of signal in an operational log:

  extract_pairs         a failed attempt, followed by a similar-intent
                         success nearby in the same session -> a
                         (rejected, chosen) pair. "You got it wrong, then
                         got it right."
  extract_confirmations a success the agent itself flagged as uncertain
                         about -> reinforcement, no wrong attempt needed.
                         "You got it right, but weren't sure — be sure."

What you do with either — DPO, SFT, or just reading them — is up to you.

No embeddings, no classifier, no external model. Matching "is this the same
intent, retried" is done with difflib from the standard library — a ratio of
how similar two strings are. That is deliberately the simplest thing that
could possibly work: agent intents that are "the same, retried" are almost
always near-identical text (same command misheard, same request rephrased
slightly), and a fuzzy string match catches that without needing to know
anything about the domain.
"""
from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher

from .types import Gap, Pair, Turn

DEFAULT_WINDOW = 5
DEFAULT_THRESHOLD = 0.6


def _similar(a: str, b: str, threshold: float) -> bool:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold


def _by_session(turns: list[Turn]) -> dict[str, list[Turn]]:
    grouped: dict[str, list[Turn]] = {}
    for t in turns:
        grouped.setdefault(t.session, []).append(t)
    return grouped


def extract_pairs(
    turns: list[Turn],
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[Pair]:
    """Find (rejected, chosen) pairs from retry-then-succeed patterns.

    window     how many turns a failure stays "waiting for its fix" before
               being dropped. Keep this small on purpose: a fix that shows
               up 50 turns later, in a different context, is not a clean
               supervision signal for what to do differently RIGHT NOW —
               it's coincidence. A tight window trades recall for precision,
               and for training data, precision is the one that matters.
    threshold  how similar two intents must be (0..1) to count as "the same
               request, retried". 0.6 catches rephrasing and minor
               correction without also catching two unrelated requests that
               happen to share a few words.

    One failed turn can pair with more than one later success if several
    of them match closely enough — each is a separate, valid lesson.
    """
    pairs: list[Pair] = []

    for session_turns in _by_session(turns).values():
        pending: list[tuple[int, Turn]] = []
        for i, t in enumerate(session_turns):
            pending = [(j, f) for j, f in pending if i - j <= window]
            if t.success:
                for j, f in pending:
                    if _similar(f.intent, t.intent, threshold):
                        pairs.append(Pair(intent=f.intent, rejected=f.attempt,
                                          reason=f.reason, chosen=t.attempt))
            else:
                pending.append((i, t))

    return _weight_by_recurrence(pairs)


def _weight_by_recurrence(pairs: list[Pair]) -> list[Pair]:
    """A failure reason that shows up 40 times across the log is a
    systematic weak spot; one that shows up once is noise. Weighting by how
    often the SAME reason recurs lets a training script upsample the former
    without Millok having to know anything about training scripts."""
    counts = Counter(p.reason or p.rejected for p in pairs)
    for p in pairs:
        p.weight = counts[p.reason or p.rejected]
    return pairs


def extract_confirmations(turns: list[Turn]) -> list[Turn]:
    """Successful attempts the agent itself flagged as uncertain about.

    A different shape of signal than extract_pairs: there's no wrong attempt
    to contrast against here, because the agent didn't get it wrong. It got
    it right, wasn't sure, and asked. This assumes your logging records that
    kind of moment as success=True with the uncertainty kept in `reason` —
    Millok has no opinion on how you detect uncertainty in the first place,
    only that you did.

    No window, no similarity matching, no session-grouping — this is
    deliberately the simplest of the three extraction functions, because it
    doesn't need to correlate two turns at all. One flagged turn is already
    the whole training example: "the model hedged here; it shouldn't have."

    Every one of these has real value even alone, with no downstream
    training step: it's a live list of exactly where an agent still lacks
    confidence, which is often more actionable than a list of outright
    failures — nothing is broken here, something just isn't trusted yet.
    """
    return [t for t in turns if t.success and t.reason.strip()]


def extract_gaps(turns: list[Turn], threshold: float = DEFAULT_THRESHOLD) -> list[Gap]:
    """Intents that were attempted and NEVER resolved in their session.

    Deliberately session-wide, not windowed like extract_pairs: the question
    here isn't "is this a clean training pair" but "did the user ever get
    what they wanted, at all". That's a weaker requirement than pairing, and
    it should be — this list has value even if nobody ever fine-tunes
    anything. It is the shortest path to "what does this agent still not
    understand".

    "Session-wide" still means forward-looking only, though: only successes
    that happen AFTER a given failure count as resolving it. An earlier
    version of this function checked the whole session regardless of order,
    which meant an unrelated success that happened to occur BEFORE a later,
    similarly-worded failure could retroactively mark that failure as
    resolved — backwards, since nothing that already happened can fix
    something that hasn't gone wrong yet. extract_pairs already only ever
    looks forward from a failure; this brings extract_gaps in line with it.
    """
    gaps: list[Gap] = []
    for session_turns in _by_session(turns).values():
        for i, t in enumerate(session_turns):
            if t.success:
                continue
            later_successes = (s.intent for s in session_turns[i + 1:] if s.success)
            if any(_similar(t.intent, r, threshold) for r in later_successes):
                continue
            gaps.append(Gap(session=t.session, intent=t.intent,
                            attempt=t.attempt, reason=t.reason))
    return gaps
