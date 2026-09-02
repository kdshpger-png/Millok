"""Turn mined pairs and confirmations into dataset formats most fine-tuning
tools already accept, so Millok's output can go straight into an existing
trainer (Hugging Face TRL, unsloth, axolotl, ...) without a conversion step.
"""
from __future__ import annotations

import json
from pathlib import Path

from .types import Pair, Turn


def to_dpo_jsonl(pairs: list[Pair], path: str | Path) -> int:
    """Preference-pair format: {"prompt", "chosen", "rejected"}.
    One line per pair, ready for a DPO/ORPO-style trainer."""
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps({
                "prompt": p.intent,
                "chosen": p.chosen,
                "rejected": p.rejected,
                "weight": p.weight,
            }, ensure_ascii=False) + "\n")
    return len(pairs)


def to_sft_jsonl(pairs: list[Pair], path: str | Path) -> int:
    """Plain instruction-tuning format: {"prompt", "completion"} — only the
    corrected side. Repeats a pair `weight` times so a plain (unweighted)
    SFT trainer still sees systematic failures more often than one-offs,
    even without native weight support."""
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            for _ in range(max(1, p.weight)):
                f.write(json.dumps({"prompt": p.intent, "completion": p.chosen},
                                   ensure_ascii=False) + "\n")
    return sum(max(1, p.weight) for p in pairs)


def confirmations_to_sft_jsonl(turns: list[Turn], path: str | Path) -> int:
    """SFT-format examples from extract_confirmations() output.

    Separate from to_sft_jsonl on purpose: a Pair has a rejected side to
    contrast against, a confirmed-uncertain Turn doesn't. Mixing the two
    into one function would mean either faking a "rejected" value for
    something that was never wrong, or silently dropping a field — both
    worse than just keeping the two paths apart."""
    with open(path, "w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps({"prompt": t.intent, "completion": t.attempt},
                               ensure_ascii=False) + "\n")
    return len(turns)
