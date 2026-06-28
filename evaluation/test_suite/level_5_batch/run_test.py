#!/usr/bin/env python3
"""Level 5 — Batch consistency test.

Tests that similar listings produce consistent extraction and scoring results
across repeated runs. For groups of similar listings:

Metrics:
- extraction_consistency: do similar listings get the same seniority/domain/remote?
- score_variance: variance of total_score across similar items (should be low)
- rank_stability: do similar items maintain their relative rank across runs?
"""

import json
import statistics
import sys
from pathlib import Path

LEVEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LEVEL_DIR.parent.parent.parent))

from evaluation.helpers import (
    build_extract_prompt,
    build_score_prompt,
    format_listing_adversarial,
    load_config,
    load_stripped_profile,
    run_n_times,
    parse_json_response,
)


def load_test_cases() -> dict:
    with open(LEVEL_DIR / "test_cases.json") as f:
        return json.load(f)


def check_extraction_consistency(extractions: list[dict], expected: dict) -> dict:
    """Check if extraction results are consistent across similar items."""
    results = {"fields_checked": {}}

    for field in ["seniority", "domain", "remote_policy"]:
        should_match_key = f"{field}_should_match"
        if expected.get(should_match_key, False):
            values = [e.get(field) for e in extractions if e]
            unique = set(v for v in values if v is not None)
            expected_val = expected.get(f"{field}_value")

            results["fields_checked"][field] = {
                "values": values,
                "unique_count": len(unique),
                "all_match": len(unique) <= 1,
                "matches_expected": expected_val in unique if expected_val else None,
            }

    return results


def run(model: str, config: dict = None, n_runs: int = 3,
        prompt_variant: str = "production") -> dict:
    """Run all batch consistency tests."""
    if config is None:
        config = load_config()

    # Load stripped profile once if needed
    stripped_profile = load_stripped_profile() if prompt_variant == "stripped" else None

    test_data = load_test_cases()
    groups = test_data["groups"]
    group_n_runs = test_data.get("n_runs", n_runs)
    results = []

    for group in groups:
        print(f"  L5 [{group['id']}] {group['description'][:50]}...", flush=True)
        expected = group["expected_consistency"]
        items = group["items"]

        # Run extraction for each item, group_n_runs times
        all_extractions = {}  # item_id -> list of parsed extractions
        all_scores = {}  # item_id -> list of total_scores

        for item in items:
            item_id = item["id"]
            all_extractions[item_id] = []
            all_scores[item_id] = []

            for run_idx in range(group_n_runs):
                # Extract
                listing_text = item["input"]
                if prompt_variant == "adversarial":
                    listing_text = format_listing_adversarial(listing_text)
                extract_prompt = build_extract_prompt(listing_text)
                extract_runs = run_n_times(extract_prompt, model, config, n=1)

                if extract_runs and extract_runs[0]["success"]:
                    parsed = extract_runs[0]["parsed"]
                    all_extractions[item_id].append(parsed)

                    # Score
                    extracted_json = json.dumps(parsed, indent=2)
                    score_prompt = build_score_prompt(extracted_json, profile_text=stripped_profile)
                    score_runs = run_n_times(score_prompt, model, config, n=1)

                    if score_runs and score_runs[0]["success"] and score_runs[0]["parsed"]:
                        total = score_runs[0]["parsed"].get("total_score")
                        if total is not None:
                            all_scores[item_id].append(total)

        # Analyze extraction consistency
        # Take the first successful extraction from each item
        first_extractions = [
            all_extractions[item["id"]][0]
            for item in items
            if all_extractions[item["id"]]
        ]
        extraction_consistency = check_extraction_consistency(
            first_extractions,
            expected.get("extraction", {}),
        )

        # Analyze score variance within the group
        # Flatten all scores across items in this group
        all_group_scores = []
        per_item_means = {}
        for item in items:
            item_id = item["id"]
            scores = all_scores[item_id]
            if scores:
                per_item_means[item_id] = round(statistics.mean(scores), 4)
                all_group_scores.extend(scores)

        # Variance across items (should be low for similar items)
        score_variance = (
            round(statistics.variance(list(per_item_means.values())), 6)
            if len(per_item_means) >= 2 else 0.0
        )

        # Rank stability: do items maintain their rank across runs?
        expected_scoring = expected.get("scoring", {})
        max_variance = expected_scoring.get("max_score_variance", 0.05)

        # Check expected rank order if specified
        expected_order = expected_scoring.get("expected_rank_order")
        rank_correct = None
        if expected_order and len(per_item_means) >= 2:
            actual_order = sorted(per_item_means.keys(), key=lambda k: per_item_means[k], reverse=True)
            # Check if the expected order matches (allowing for ties within variance)
            rank_correct = actual_order == expected_order

        results.append({
            "group_id": group["id"],
            "description": group["description"],
            "n_items": len(items),
            "n_runs_per_item": group_n_runs,
            "extraction_consistency": extraction_consistency,
            "per_item_mean_scores": per_item_means,
            "score_variance": score_variance,
            "variance_within_threshold": score_variance <= max_variance,
            "max_variance_threshold": max_variance,
            "rank_order_correct": rank_correct,
        })

    # Aggregate metrics
    n_groups = len(results)

    # Extraction consistency: % of checked fields that all matched
    consistency_checks = []
    for r in results:
        for field, data in r["extraction_consistency"].get("fields_checked", {}).items():
            consistency_checks.append(data.get("all_match", False))
    extraction_consistency_rate = (
        round(sum(consistency_checks) / len(consistency_checks), 4)
        if consistency_checks else 0.0
    )

    # Score variance: average variance across groups
    avg_variance = (
        round(sum(r["score_variance"] for r in results) / n_groups, 6)
        if n_groups else 0.0
    )

    # Variance within threshold rate
    variance_pass_rate = (
        round(sum(1 for r in results if r["variance_within_threshold"]) / n_groups, 4)
        if n_groups else 0.0
    )

    return {
        "level": 5,
        "name": "batch_consistency",
        "model": model,
        "n_runs_per_item": group_n_runs,
        "total_groups": n_groups,
        "metrics": {
            "extraction_consistency": extraction_consistency_rate,
            "mean_score_variance": avg_variance,
            "variance_pass_rate": variance_pass_rate,
        },
        "per_group": results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Level 5 — Batch consistency test")
    parser.add_argument("--model", required=True, help="Ollama model name")
    parser.add_argument("--n-runs", type=int, default=3, help="Runs per item")
    args = parser.parse_args()

    result = run(args.model, n_runs=args.n_runs)
    print(json.dumps(result, indent=2))
