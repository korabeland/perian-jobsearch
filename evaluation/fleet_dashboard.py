#!/usr/bin/env python3
"""Fleet performance dashboard — shows all evaluated models in a single table.

Usage:
    python3 evaluation/fleet_dashboard.py            # all models, latest eval per model
    python3 evaluation/fleet_dashboard.py --active    # active fleet only
    python3 evaluation/fleet_dashboard.py --md        # markdown output (for docs/copy-paste)
"""

import argparse
import json
from pathlib import Path

EVAL_INDEX = Path(__file__).resolve().parent / "results" / "eval-index.jsonl"
FLEET_PATH = Path(__file__).resolve().parent.parent / "state" / "model-fleet.json"

CAPABILITY_ORDER = [
    "extraction", "classification", "conditional_reasoning",
    "causal_reasoning", "synthesis", "json_output", "nuanced_judgment",
]
CAP_SHORT = ["ext", "cls", "cond", "causal", "synth", "json", "nuance"]

ACTIVE_FLEET_MODELS = set()


def load_active_fleet():
    """Load active fleet model names from model-fleet.json."""
    if not FLEET_PATH.exists():
        return set()
    fleet = json.loads(FLEET_PATH.read_text())
    return {v["model"] for v in fleet.get("fleet", {}).values()}


def load_latest_evals():
    """Load the latest eval per model from eval-index.jsonl."""
    if not EVAL_INDEX.exists():
        return []
    latest = {}
    with open(EVAL_INDEX) as f:
        for line in f:
            entry = json.loads(line.strip())
            model = entry["model"]
            variant = entry.get("prompt_variant", "production")
            if variant != "production":
                continue
            # Keep latest by timestamp
            if model not in latest or entry["timestamp"] > latest[model]["timestamp"]:
                latest[model] = entry
    return list(latest.values())


def format_score(val):
    if val is None:
        return "  — "
    return f"{val:.2f}"


def format_tps(val):
    if val is None:
        return "  — "
    return f"{val:.0f}"


def print_table(evals, active_models, markdown=False):
    """Print a formatted performance table."""
    # Sort: active fleet first, then by total_time
    def sort_key(e):
        is_active = e["model"] in active_models
        return (0 if is_active else 1, e.get("total_time_s") or 99999)

    evals.sort(key=sort_key)

    if markdown:
        print_markdown_table(evals, active_models)
    else:
        print_terminal_table(evals, active_models)


def print_terminal_table(evals, active_models):
    header = f"{'Model':<28} {'tok/s':>5}  {'ext':>4} {'cls':>4} {'cond':>4} {'caus':>4} {'syn':>4} {'json':>4} {'nuan':>4}  {'time':>6}  {'date':>10}"
    sep = "─" * len(header)

    print()
    print("  Fleet Performance Dashboard")
    print(f"  {sep}")
    print(f"  {header}")
    print(f"  {sep}")

    prev_active = True
    for e in evals:
        model = e["model"]
        is_active = model in active_models
        caps = e.get("capabilities", {})
        tps = e.get("mean_gen_tok_s")
        total = e.get("total_time_s")
        date = e["timestamp"][:10]

        # Separator between active and reserve
        if prev_active and not is_active:
            print(f"  {'─' * len(header)}")
            prev_active = False

        marker = "●" if is_active else " "
        time_str = f"{total:.0f}s" if total else "  — "
        scores = " ".join(format_score(caps.get(c)) for c in CAPABILITY_ORDER)
        print(f"  {marker} {model:<26} {format_tps(tps):>5}  {scores}  {time_str:>6}  {date}")

    print(f"  {sep}")
    print(f"  ● = active fleet    tok/s = decode tokens/second    time = full eval duration")
    print()


def print_markdown_table(evals, active_models):
    print("| Model | tok/s | ext | cls | cond | causal | synth | json | nuance | eval time | date |")
    print("|-------|------:|----:|----:|-----:|-------:|------:|-----:|-------:|----------:|------|")
    for e in evals:
        model = e["model"]
        marker = "**" if model in active_models else ""
        caps = e.get("capabilities", {})
        tps = e.get("mean_gen_tok_s")
        total = e.get("total_time_s")
        date = e["timestamp"][:10]

        tps_str = f"{tps:.0f}" if tps else "—"
        time_str = f"{total:.0f}s" if total else "—"
        scores = " | ".join(format_score(caps.get(c)).strip() for c in CAPABILITY_ORDER)
        print(f"| {marker}{model}{marker} | {tps_str} | {scores} | {time_str} | {date} |")


def main():
    parser = argparse.ArgumentParser(description="Fleet performance dashboard")
    parser.add_argument("--active", action="store_true", help="Show active fleet models only")
    parser.add_argument("--md", action="store_true", help="Output as markdown table")
    args = parser.parse_args()

    active_models = load_active_fleet()
    evals = load_latest_evals()

    if not evals:
        print("No eval results found in eval-index.jsonl")
        return

    if args.active:
        evals = [e for e in evals if e["model"] in active_models]

    print_table(evals, active_models, markdown=args.md)


if __name__ == "__main__":
    main()
