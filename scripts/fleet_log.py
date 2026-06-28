#!/usr/bin/env python3
"""Append-only fleet changelog.

Each line in state/fleet-changelog.jsonl is a timestamped JSON event:
- eval_complete: model evaluated, with capability scores
- model_added: new model pulled into fleet
- model_removed: model removed from fleet
- model_upgraded: model replaced another in a fleet tier
- candidate_found: legolas discovered a candidate
- candidate_accepted: user accepted a candidate
- candidate_dismissed: user dismissed a candidate

Usage:
    # From other scripts:
    from scripts.fleet_log import log_event
    log_event("eval_complete", model="qwen3.5:9b", scores={...}, total_time_s=120)

    # CLI:
    python3 scripts/fleet_log.py show           # print all events
    python3 scripts/fleet_log.py show --last 10 # last 10 events
    python3 scripts/fleet_log.py show --model qwen3.5:9b  # filter by model
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parent.parent / "state" / "fleet-changelog.jsonl"


def log_event(event_type: str, model: str = None, **details):
    """Append a timestamped event to the fleet changelog."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
    }
    if model:
        entry["model"] = model
    entry.update(details)
    with open(CHANGELOG, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def read_log(last_n: int = None, model_filter: str = None) -> list[dict]:
    """Read changelog entries, optionally filtered."""
    if not CHANGELOG.exists():
        return []
    entries = []
    for line in open(CHANGELOG):
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        if model_filter and entry.get("model") != model_filter:
            continue
        entries.append(entry)
    if last_n:
        entries = entries[-last_n:]
    return entries


def print_log(entries: list[dict]):
    """Pretty-print changelog entries."""
    if not entries:
        print("  No entries.")
        return

    for e in entries:
        ts = e["timestamp"][:19].replace("T", " ")
        event = e["event"]
        model = e.get("model", "")
        detail = ""

        if event == "eval_complete":
            scores = e.get("scores", {})
            time_s = e.get("total_time_s", 0)
            top = sorted(scores.items(), key=lambda x: -x[1])[:3]
            detail = f"[{time_s:.0f}s] " + ", ".join(f"{k}={v:.2f}" for k, v in top)
        elif event == "model_upgraded":
            detail = f"replaced {e.get('replaced', '?')} in {e.get('tier', '?')} tier"
            if e.get("reason"):
                detail += f" — {e['reason']}"
        elif event == "model_added":
            detail = e.get("tier", "")
            if e.get("reason"):
                detail += f" — {e['reason']}"
        elif event in ("candidate_found", "candidate_accepted", "candidate_dismissed"):
            detail = e.get("tier", "")
            if e.get("reason"):
                detail += f" — {e['reason']}"
        else:
            # Generic: show any extra keys
            extras = {k: v for k, v in e.items() if k not in ("timestamp", "event", "model")}
            if extras:
                detail = json.dumps(extras)

        print(f"  {ts}  {event:<22} {model:<25} {detail}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "show":
        last_n = None
        model_filter = None
        i = 1
        while i < len(args):
            if args[i] == "--last" and i + 1 < len(args):
                last_n = int(args[i + 1])
                i += 2
            elif args[i] == "--model" and i + 1 < len(args):
                model_filter = args[i + 1]
                i += 2
            else:
                i += 1

        entries = read_log(last_n=last_n, model_filter=model_filter)
        print(f"\n  Fleet Changelog ({len(entries)} entries)")
        print(f"  {'─' * 70}")
        print_log(entries)
        print()
    else:
        print(f"Unknown command: {args[0]}")
        print("Usage: python3 scripts/fleet_log.py show [--last N] [--model MODEL]")
        sys.exit(1)
