"""Shared data types for Millok.

A Turn is the only thing any application needs to produce to use this
library: one line per attempt an agent made, whether it worked, and why not
if it didn't. Nothing here is specific to any particular agent framework,
tool format, or model provider — that's intentional. If your system can log
"here's what I tried, here's whether it worked", Millok can mine it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Turn:
    """One attempt, in one session.

    session   groups turns that belong together (a conversation, a run).
              Matching NEVER crosses session boundaries — a failure in one
              user's chat must not get "fixed" by an unrelated success in
              someone else's.
    intent    what the agent was trying to satisfy — usually the user's
              request, in whatever text form your system already has it.
    attempt   what the agent actually did in response — a tool call, a
              generated answer, whatever your system considers "the output".
    success   did it work?
    reason    if it didn't: why, in your system's own words. This is what
              makes the mined data useful instead of just "wrong" — a model
              fine-tuned on "wrong" learns nothing; one fine-tuned on
              "wrong because you guessed a tool name that doesn't exist"
              learns something that generalizes.
    """
    session: str
    intent: str
    attempt: str
    success: bool
    reason: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "Turn":
        data = json.loads(line)
        return Turn(session=str(data["session"]), intent=str(data["intent"]),
                    attempt=str(data["attempt"]), success=bool(data["success"]),
                    reason=str(data.get("reason", "")))


@dataclass
class Pair:
    """A mined (rejected, chosen) example for the SAME underlying intent.

    weight counts how often this exact kind of failure (by `reason`) shows
    up across the whole log. A one-off typo and a systematic blind spot both
    produce a Pair — weight is what tells a training script which one
    actually matters.
    """
    intent: str
    rejected: str
    reason: str
    chosen: str
    weight: int = 1


@dataclass
class Gap:
    """An intent that was attempted and never resolved in its session.

    Useful on its own, with no training step at all: it's the honest
    "still doesn't work" list.
    """
    session: str
    intent: str
    attempt: str
    reason: str


def read_turns(path: str | Path) -> list[Turn]:
    turns = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                turns.append(Turn.from_json(line))
    return turns


def write_turns(turns: list[Turn], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for t in turns:
            f.write(t.to_json() + "\n")
