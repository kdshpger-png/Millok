"""Command-line entry point: `python -m millok <command>`."""
from __future__ import annotations

import argparse
import sys

from . import formats, mining
from .types import read_turns


def _cmd_mine(args: argparse.Namespace) -> int:
    turns = read_turns(args.logfile)
    pairs = mining.extract_pairs(turns, window=args.window, threshold=args.threshold)
    if not pairs:
        print("No retry-correction pairs found in this log.")
        return 0
    writer = formats.to_dpo_jsonl if args.format == "dpo" else formats.to_sft_jsonl
    count = writer(pairs, args.output)
    print(f"{len(pairs)} pairs mined from {len(turns)} turns -> "
          f"{count} lines written to {args.output} ({args.format} format)")
    return 0


def _cmd_gaps(args: argparse.Namespace) -> int:
    turns = read_turns(args.logfile)
    gaps = mining.extract_gaps(turns, threshold=args.threshold)
    if not gaps:
        print("No unresolved intents found.")
        return 0
    print(f"{len(gaps)} intent(s) never resolved:\n")
    for g in gaps:
        print(f"  session={g.session}\n    tried  : {g.intent}\n"
              f"    attempt: {g.attempt}\n    reason : {g.reason}\n")
    return 0


def _cmd_confirmed(args: argparse.Namespace) -> int:
    turns = read_turns(args.logfile)
    confirmed = mining.extract_confirmations(turns)
    if not confirmed:
        print("No confirmed-but-uncertain successes found in this log.")
        return 0
    print(f"{len(confirmed)} confirmed-but-uncertain success(es):\n")
    for t in confirmed:
        print(f"  session={t.session}\n    intent : {t.intent}\n"
              f"    attempt: {t.attempt}\n    reason : {t.reason}\n")
    if args.output:
        count = formats.confirmations_to_sft_jsonl(confirmed, args.output)
        print(f"wrote {count} lines to {args.output} (sft format)")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo_data import build_demo_log

    turns = build_demo_log()
    pairs = mining.extract_pairs(turns)
    gaps = mining.extract_gaps(turns)
    confirmed = mining.extract_confirmations(turns)

    print(f"synthetic log      : {len(turns)} turns across "
          f"{len({t.session for t in turns})} sessions")
    print(f"failed attempts    : {sum(1 for t in turns if not t.success)}")
    print(f"pairs mined        : {len(pairs)}")
    print(f"confirmed-uncertain: {len(confirmed)}")
    print(f"unresolved gaps    : {len(gaps)}")
    print()
    # The demo data is fixed and tested, so pairs/confirmed are never
    # actually empty here - but guarding it costs one line and avoids a
    # raw ValueError/IndexError if someone edits demo_data.py and removes
    # every failure pattern without noticing these depend on it.
    if pairs:
        print("sample pair:")
        p = max(pairs, key=lambda p: p.weight)
        print(f"  intent   : {p.intent}")
        print(f"  rejected : {p.rejected}")
        print(f"  reason   : {p.reason}")
        print(f"  chosen   : {p.chosen}")
        print(f"  weight   : {p.weight}  (this exact failure reason recurred "
              f"{p.weight} times)")
        print()
    if confirmed:
        print("sample confirmation (succeeded, but the agent had hedged):")
        c = confirmed[0]
        print(f"  intent   : {c.intent}")
        print(f"  attempt  : {c.attempt}")
        print(f"  reason   : {c.reason}")
        print()
    print("gap:")
    for g in gaps:
        print(f"  intent   : {g.intent}")
        print(f"  reason   : {g.reason}")

    if args.output:
        count = formats.to_dpo_jsonl(pairs, args.output)
        print(f"\nwrote {count} DPO pairs to {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="millok",
        description="Mine fine-tuning pairs from an agent's own retry-and-correct logs.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_mine = sub.add_parser("mine", help="extract training pairs from a log")
    p_mine.add_argument("logfile")
    p_mine.add_argument("-o", "--output", default="millok_dataset.jsonl")
    p_mine.add_argument("--format", choices=["dpo", "sft"], default="dpo")
    p_mine.add_argument("--window", type=int, default=mining.DEFAULT_WINDOW)
    p_mine.add_argument("--threshold", type=float, default=mining.DEFAULT_THRESHOLD)
    p_mine.set_defaults(func=_cmd_mine)

    p_gaps = sub.add_parser("gaps", help="list intents that were never resolved")
    p_gaps.add_argument("logfile")
    p_gaps.add_argument("--threshold", type=float, default=mining.DEFAULT_THRESHOLD)
    p_gaps.set_defaults(func=_cmd_gaps)

    p_conf = sub.add_parser("confirmed",
        help="list successes the agent itself flagged as uncertain about")
    p_conf.add_argument("logfile")
    p_conf.add_argument("-o", "--output", default=None,
                        help="also write these as SFT-format reinforcement examples")
    p_conf.set_defaults(func=_cmd_confirmed)

    p_demo = sub.add_parser("demo", help="run on a built-in synthetic log")
    p_demo.add_argument("-o", "--output", default=None,
                        help="also write the mined pairs here (DPO format)")
    p_demo.set_defaults(func=_cmd_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
