#!/usr/bin/env python3
"""CLI interface to autoresearch eval functions.

Thin wrapper so headless Claude Code sessions can call eval/validation
via Bash instead of importing Python directly.

Subcommands:
    evaluate    — Run prompt evaluation against holdout data
    validate    — Structural validation of a candidate mutation
    failures    — Extract worst-performing items from eval results
    safety-gate — Run eval suite L1-L3 regression check
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autoresearch.prompt_loop import (
    evaluate_prompt,
    find_failures,
    load_holdout,
    run_safety_gate,
    split_holdout,
    validate_mutation,
)
from autoresearch.shared import load_autoresearch_config, render_prompt
from evaluation.helpers import load_config, load_plan, load_profile, load_prompt_template


def cmd_evaluate(args):
    """Evaluate a prompt template against holdout items."""
    ar_config = load_autoresearch_config()
    config = load_config()
    plan = load_plan()
    profile = load_profile()
    model = ar_config["loop"]["local_model"]
    n_runs = ar_config["loop"]["runs_per_item"]
    split_ratio = ar_config["loop"]["holdout_split"]
    split_seed = ar_config["loop"]["holdout_seed"]

    all_items = load_holdout(args.template)
    opt_items, val_items = split_holdout(all_items, split_ratio, split_seed)

    if args.items == "opt":
        items = opt_items
    elif args.items == "val":
        items = val_items
    else:
        items = all_items

    # Use candidate prompt if provided, otherwise current production prompt
    if args.candidate:
        template_text = Path(args.candidate).read_text()
    else:
        template_text = load_prompt_template(args.template)

    result = evaluate_prompt(
        template_text, items, args.template, model, config,
        n_runs=n_runs, plan=plan, profile=profile,
    )

    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(f"mean_score={result['mean_score']:.4f} pass_rate={result['pass_rate']:.4f} n={result['n_items']}")


def cmd_validate(args):
    """Structural validation of a candidate prompt mutation."""
    ar_config = load_autoresearch_config()
    candidate = Path(args.candidate).read_text()
    original = load_prompt_template(args.template)

    result = validate_mutation(candidate, original, args.template, ar_config)

    print(json.dumps(result, indent=2))
    if not result["valid"]:
        sys.exit(1)


def cmd_failures(args):
    """Extract worst-performing items from eval results."""
    with open(args.input) as f:
        eval_result = json.load(f)

    failures = find_failures(eval_result, max_n=args.max)

    Path(args.output).write_text(json.dumps(failures, indent=2) + "\n")
    n_failing = sum(1 for f in failures if f["mean_score"] < 0.7)
    print(f"failures={len(failures)} below_0.7={n_failing}")


def cmd_safety_gate(args):
    """Run eval suite L1-L3 as safety regression check."""
    ar_config = load_autoresearch_config()
    model = args.model or ar_config["loop"]["local_model"]

    result = run_safety_gate(model, ar_config)

    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    if result["passed"]:
        print("PASSED")
    else:
        print(f"FAILED: {result.get('error', result.get('regressions', []))}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Autoresearch eval CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate prompt against holdout")
    p_eval.add_argument("--template", choices=["extract", "score"], required=True)
    p_eval.add_argument("--items", choices=["opt", "val", "all"], default="opt")
    p_eval.add_argument("--candidate", help="Path to candidate prompt (default: production)")
    p_eval.add_argument("--output", required=True, help="Path for JSON output")

    # validate
    p_val = sub.add_parser("validate", help="Validate candidate mutation")
    p_val.add_argument("--candidate", required=True, help="Path to candidate prompt")
    p_val.add_argument("--template", choices=["extract", "score"], required=True)

    # failures
    p_fail = sub.add_parser("failures", help="Extract failures from eval results")
    p_fail.add_argument("--input", required=True, help="Path to eval result JSON")
    p_fail.add_argument("--max", type=int, default=8, help="Max failures to return")
    p_fail.add_argument("--output", required=True, help="Path for JSON output")

    # safety-gate
    p_safe = sub.add_parser("safety-gate", help="Run L1-L3 safety check")
    p_safe.add_argument("--model", help="Ollama model (default: from config)")
    p_safe.add_argument("--output", required=True, help="Path for JSON output")

    args = parser.parse_args()

    commands = {
        "evaluate": cmd_evaluate,
        "validate": cmd_validate,
        "failures": cmd_failures,
        "safety-gate": cmd_safety_gate,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
