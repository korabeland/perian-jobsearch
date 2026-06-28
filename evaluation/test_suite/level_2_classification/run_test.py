#!/usr/bin/env python3
"""Level 2 — Classification accuracy test.

Runs each test case through the extraction prompt but scores ONLY on
the three classification fields: seniority, domain, remote_policy.
Test cases are designed to target classification edge cases.

Metrics:
- per_field_accuracy: accuracy for each of seniority, domain, remote_policy
- overall_classification_accuracy: average across all 3 fields
"""

import json
import sys
from pathlib import Path

LEVEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LEVEL_DIR.parent.parent.parent))

from evaluation.helpers import (
    aggregate_token_metrics,
    build_extract_prompt,
    format_listing_adversarial,
    load_config,
    run_n_times,
)

CLASSIFICATION_FIELDS = ["seniority", "domain", "remote_policy"]


def load_test_cases() -> list[dict]:
    with open(LEVEL_DIR / "test_cases.json") as f:
        return json.load(f)


def score_classification(output: dict, test_case: dict) -> dict:
    """Score classification fields only."""
    expected = test_case["expected_output"]
    alt = test_case.get("acceptable_alternatives", {})

    field_results = {}
    for field in CLASSIFICATION_FIELDS:
        actual = output.get(field)
        exp = expected.get(field)
        correct = actual == exp
        if not correct and field in alt:
            correct = actual in alt[field]
        field_results[field] = {
            "actual": actual,
            "expected": exp,
            "alternatives": alt.get(field, []),
            "correct": correct,
        }

    n_correct = sum(1 for r in field_results.values() if r["correct"])
    return {
        "field_results": field_results,
        "accuracy": round(n_correct / len(CLASSIFICATION_FIELDS), 4),
    }


def run(model: str, config: dict = None, n_runs: int = 5,
        prompt_variant: str = "production") -> dict:
    """Run all Level 2 test cases and return results."""
    if config is None:
        config = load_config()

    test_cases = load_test_cases()
    results = []

    for tc in test_cases:
        print(f"  L2 [{tc['id']}] {tc['description'][:50]}...", flush=True)
        listing_text = tc["input"]
        if prompt_variant == "adversarial":
            listing_text = format_listing_adversarial(listing_text)
        prompt = build_extract_prompt(listing_text)
        runs = run_n_times(prompt, model, config, n=n_runs)

        case_scores = []
        for r in runs:
            if r["success"] and r["parsed"]:
                score = score_classification(r["parsed"], tc)
                case_scores.append(score)

        # Per-field accuracy across all runs for this case
        per_field_correct = {f: 0 for f in CLASSIFICATION_FIELDS}
        total_scored = len(case_scores)
        for cs in case_scores:
            for f in CLASSIFICATION_FIELDS:
                if cs["field_results"][f]["correct"]:
                    per_field_correct[f] += 1

        per_field_accuracy = {
            f: round(per_field_correct[f] / total_scored, 4) if total_scored else 0.0
            for f in CLASSIFICATION_FIELDS
        }

        results.append({
            "test_id": tc["id"],
            "description": tc["description"],
            "tags": tc["tags"],
            "n_runs": n_runs,
            "successful_parses": sum(1 for r in runs if r["success"]),
            "per_field_accuracy": per_field_accuracy,
            "mean_accuracy": (
                round(sum(cs["accuracy"] for cs in case_scores) / len(case_scores), 4)
                if case_scores else 0.0
            ),
            "mean_time_s": round(sum(r["time_s"] for r in runs) / len(runs), 2),
            "token_metrics": aggregate_token_metrics(runs),
        })

    # Aggregate metrics
    total_cases = len(results)

    # Per-field accuracy across ALL cases
    aggregate_per_field = {f: [] for f in CLASSIFICATION_FIELDS}
    for r in results:
        for f in CLASSIFICATION_FIELDS:
            aggregate_per_field[f].append(r["per_field_accuracy"][f])

    per_field_overall = {
        f: round(sum(vals) / len(vals), 4) if vals else 0.0
        for f, vals in aggregate_per_field.items()
    }

    overall_accuracy = (
        round(sum(per_field_overall.values()) / len(CLASSIFICATION_FIELDS), 4)
        if per_field_overall else 0.0
    )

    return {
        "level": 2,
        "name": "classification",
        "model": model,
        "n_runs_per_case": n_runs,
        "total_cases": total_cases,
        "metrics": {
            "overall_classification_accuracy": overall_accuracy,
            "per_field_accuracy": per_field_overall,
        },
        "per_case": results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Level 2 — Classification test")
    parser.add_argument("--model", required=True, help="Ollama model name")
    parser.add_argument("--n-runs", type=int, default=5, help="Runs per test case")
    args = parser.parse_args()

    result = run(args.model, n_runs=args.n_runs)
    print(json.dumps(result, indent=2))
