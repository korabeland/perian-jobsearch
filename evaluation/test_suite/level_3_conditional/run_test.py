#!/usr/bin/env python3
"""Level 3 — Conditional reasoning (scoring) test.

Runs each test case through the scoring prompt N times and measures:
- score_deviation: MAE of per-dimension scores vs ground truth
- total_score_in_range: whether total_score falls in the expected range
- bonus_penalty_accuracy: whether bonuses/penalties are correctly applied
- reasoning_present: whether reasoning strings exist for each dimension
"""

import json
import sys
from pathlib import Path

LEVEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LEVEL_DIR.parent.parent.parent))

from evaluation.helpers import (
    aggregate_token_metrics,
    build_score_prompt,
    load_config,
    load_stripped_profile,
    run_n_times,
)

SCORE_DIMENSIONS = ["title_match", "domain_match", "seniority_match", "requirements_fit", "compensation_fit"]


def load_test_cases() -> list[dict]:
    with open(LEVEL_DIR / "test_cases.json") as f:
        return json.load(f)


def score_case(output: dict, test_case: dict) -> dict:
    """Score a single model scoring output against ground truth."""
    expected = test_case["expected_output"]
    exp_scores = expected["scores"]

    # Per-dimension score deviation
    dim_results = {}
    total_deviation = 0
    dims_scored = 0

    scores_section = output.get("scores", {})
    for dim in SCORE_DIMENSIONS:
        exp = exp_scores.get(dim, {})
        exp_score = exp.get("score", 0.5)
        tolerance = exp.get("tolerance", 0.15)

        actual_dim = scores_section.get(dim, {})
        actual_score = actual_dim.get("score") if isinstance(actual_dim, dict) else None

        if actual_score is not None:
            deviation = abs(actual_score - exp_score)
            within_tolerance = deviation <= tolerance
            has_reasoning = bool(actual_dim.get("reasoning"))
        else:
            deviation = 1.0
            within_tolerance = False
            has_reasoning = False

        dim_results[dim] = {
            "actual": actual_score,
            "expected": exp_score,
            "tolerance": tolerance,
            "deviation": round(deviation, 4),
            "within_tolerance": within_tolerance,
            "has_reasoning": has_reasoning,
        }
        total_deviation += deviation
        dims_scored += 1

    mean_deviation = round(total_deviation / dims_scored, 4) if dims_scored else 1.0

    # Total score range check
    total_score = output.get("total_score")
    score_range = expected.get("total_score_range", [0.0, 1.0])
    total_in_range = (
        total_score is not None
        and score_range[0] <= total_score <= score_range[1]
    )

    # Bonus/penalty accuracy (with acceptable_alternatives for ambiguous cases)
    has_bonus = len(output.get("bonuses_applied", [])) > 0
    has_penalty = len(output.get("penalties_applied", [])) > 0

    acceptable_bonus = expected.get("acceptable_bonus")
    if acceptable_bonus is not None:
        bonus_correct = has_bonus in acceptable_bonus
    else:
        bonus_correct = has_bonus == expected.get("expected_bonus", False)

    acceptable_penalty = expected.get("acceptable_penalty")
    if acceptable_penalty is not None:
        penalty_correct = has_penalty in acceptable_penalty
    else:
        penalty_correct = has_penalty == expected.get("expected_penalty", False)

    # Reasoning presence
    reasoning_count = sum(1 for d in dim_results.values() if d["has_reasoning"])

    return {
        "dim_results": dim_results,
        "mean_deviation": mean_deviation,
        "dims_within_tolerance": sum(1 for d in dim_results.values() if d["within_tolerance"]),
        "total_score_actual": total_score,
        "total_score_range": score_range,
        "total_in_range": total_in_range,
        "bonus_correct": bonus_correct,
        "penalty_correct": penalty_correct,
        "reasoning_count": reasoning_count,
    }


def run(model: str, config: dict = None, n_runs: int = 5,
        prompt_variant: str = "production") -> dict:
    """Run all Level 3 test cases and return results."""
    if config is None:
        config = load_config()

    # Load stripped profile once if needed
    stripped_profile = load_stripped_profile() if prompt_variant == "stripped" else None

    test_cases = load_test_cases()
    results = []

    for tc in test_cases:
        print(f"  L3 [{tc['id']}] {tc['description'][:50]}...", flush=True)
        extracted_json = json.dumps(tc["input"], indent=2)
        prompt = build_score_prompt(extracted_json, profile_text=stripped_profile)
        runs = run_n_times(prompt, model, config, n=n_runs)

        case_scores = []
        for r in runs:
            if r["success"] and r["parsed"]:
                score = score_case(r["parsed"], tc)
                case_scores.append(score)

        n_scored = len(case_scores)

        results.append({
            "test_id": tc["id"],
            "description": tc["description"],
            "tags": tc["tags"],
            "n_runs": n_runs,
            "successful_parses": sum(1 for r in runs if r["success"]),
            "mean_deviation": (
                round(sum(s["mean_deviation"] for s in case_scores) / n_scored, 4)
                if n_scored else 1.0
            ),
            "dims_within_tolerance_avg": (
                round(sum(s["dims_within_tolerance"] for s in case_scores) / n_scored, 2)
                if n_scored else 0.0
            ),
            "total_in_range_rate": (
                round(sum(1 for s in case_scores if s["total_in_range"]) / n_scored, 4)
                if n_scored else 0.0
            ),
            "bonus_accuracy": (
                round(sum(1 for s in case_scores if s["bonus_correct"]) / n_scored, 4)
                if n_scored else 0.0
            ),
            "penalty_accuracy": (
                round(sum(1 for s in case_scores if s["penalty_correct"]) / n_scored, 4)
                if n_scored else 0.0
            ),
            "mean_time_s": round(sum(r["time_s"] for r in runs) / len(runs), 2),
            "token_metrics": aggregate_token_metrics(runs),
        })

    # Aggregate metrics
    total_cases = len(results)
    mean_dev = (
        round(sum(r["mean_deviation"] for r in results) / total_cases, 4)
        if total_cases else 1.0
    )
    # Convert deviation to a 0-1 score (lower deviation = higher score)
    # 0.0 deviation = 1.0 score, 0.5 deviation = 0.0 score
    score_correlation = round(max(0, 1.0 - 2 * mean_dev), 4)

    total_in_range = (
        round(sum(r["total_in_range_rate"] for r in results) / total_cases, 4)
        if total_cases else 0.0
    )
    bonus_acc = (
        round(sum(r["bonus_accuracy"] for r in results) / total_cases, 4)
        if total_cases else 0.0
    )
    penalty_acc = (
        round(sum(r["penalty_accuracy"] for r in results) / total_cases, 4)
        if total_cases else 0.0
    )

    return {
        "level": 3,
        "name": "conditional_reasoning",
        "model": model,
        "n_runs_per_case": n_runs,
        "total_cases": total_cases,
        "metrics": {
            "score_correlation": score_correlation,
            "mean_score_deviation": mean_dev,
            "total_in_range_rate": total_in_range,
            "bonus_accuracy": bonus_acc,
            "penalty_accuracy": penalty_acc,
        },
        "per_case": results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Level 3 — Conditional reasoning test")
    parser.add_argument("--model", required=True, help="Ollama model name")
    parser.add_argument("--n-runs", type=int, default=5, help="Runs per test case")
    args = parser.parse_args()

    result = run(args.model, n_runs=args.n_runs)
    print(json.dumps(result, indent=2))
