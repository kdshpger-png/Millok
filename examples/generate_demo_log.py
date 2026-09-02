"""Example: write the built-in demo log to a file, for use with `millok mine`.

This is intentionally thin. The actual log-building logic lives in
`millok/demo_data.py` because it also needs to work as part of the installed
`millok` package (e.g. `millok demo`, run from anywhere) — this script is
just the "how would I write my own generator" example, meant to be read.

    python3 examples/generate_demo_log.py [output_file.jsonl]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from millok.demo_data import build_demo_log
from millok.types import write_turns

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "demo_log.jsonl"
    log = build_demo_log()
    write_turns(log, out)
    print(f"wrote {len(log)} turns to {out}")
