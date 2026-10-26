from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Log a reference reading for a Clinura session")
    parser.add_argument("session", type=Path, help="recorded session directory")
    parser.add_argument("--bpm", type=float, required=True, help="reference heart rate")
    parser.add_argument("--source", default="apple_watch", help="reference source label")
    parser.add_argument("--note", default="", help="optional note")
    parser.add_argument(
        "--timestamp",
        type=float,
        default=None,
        help="Unix wall-clock time of the reading (defaults to now)",
    )
    args = parser.parse_args(argv)
    session = args.session
    if not session.is_dir():
        print(f"Session not found: {session}", file=sys.stderr)
        return 2
    record = {
        "timestamp": float(args.timestamp if args.timestamp is not None else time.time()),
        "bpm": float(args.bpm),
        "source": args.source,
        "note": args.note,
    }
    path = session / "reference.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(f"Logged {record['bpm']:.1f} bpm from {args.source} to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
