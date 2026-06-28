#!/usr/bin/env python3
"""Level 1 — Extraction accuracy test.

Runs each test case through the extraction prompt N times and measures:
- field_accuracy: % of fields matching ground truth (averaged across runs)
- schema_compliance: % of outputs with valid JSON and all required fields
- key_requirements_overlap: Jaccard similarity of requirement sets
- compensation_accuracy: exact match on compensation sub-fields
"""

import json
import sys
from pathlib import Path

LEVEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LEVEL_DIR.parent.parent.parent))

from evaluation.helpers import (
    EXTRACT_REQUIRED_FIELDS,
    VALID_REMOTE,
    VALID_SENIORITY,
    aggregate_token_metrics,
    build_extract_prompt,
    format_listing_adversarial,
    load_config,
    run_n_times,
)


def load_test_cases() -> list[dict]:
    with open(LEVEL_DIR / "test_cases.json") as f:
        return json.load(f)


def check_schema(output: dict) -> dict:
    """Check if output has all required fields and valid enum values."""
    missing = [f for f in EXTRACT_REQUIRED_FIELDS if f not in output]
    valid_enums = True
    if output.get("seniority") not in VALID_SENIORITY:
        valid_enums = False
    if output.get("remote_policy") not in VALID_REMOTE:
        valid_enums = False
    return {
        "has_all_fields": len(missing) == 0,
        "missing_fields": missing,
        "valid_enums": valid_enums,
        "is_compliant": len(missing) == 0 and valid_enums,
    }


def field_match(actual, expected, alternatives=None) -> bool:
    """Check if actual value matches expected, considering alternatives."""
    if actual == expected:
        return True
    if alternatives and actual in alternatives:
        return True
    return False


def requirements_jaccard(actual: list, expected: list) -> float:
    """Legacy Jaccard similarity — kept for backwards comparison."""
    if not actual and not expected:
        return 1.0
    if not actual or not expected:
        return 0.0
    a = {s.lower() for s in actual}
    e = {s.lower() for s in expected}
    intersection = a & e
    union = a | e
    return len(intersection) / len(union) if union else 0.0


def requirements_semantic(actual: list, expected: list) -> float:
    """Semantic matching score for requirement lists.

    For each model-extracted requirement, finds the best semantic match among
    expected requirements (threshold 0.80). Score = matched / max(expected, actual).
    Lets paraphrases like "5+ years CX management" match "5 years of customer
    experience management".
    """
    if not actual and not expected:
        return 1.0
    if not actual or not expected:
        return 0.0

    from evaluation.semantic import best_semantic_match

    matched = 0
    for req in actual:
        is_match, _score, _ref = best_semantic_match(req, expected, threshold=0.75)
        if is_match:
            matched += 1

    return matched / max(len(expected), len(actual))


def compensation_match(actual: dict, expected: dict) -> dict:
    """Check each compensation sub-field."""
    if not actual or not expected:
        return {"min": False, "max": False, "currency": False, "equity": False, "score": 0.0}

    fields = {
        "min": actual.get("min") == expected.get("min"),
        "max": actual.get("max") == expected.get("max"),
        "currency": actual.get("currency") == expected.get("currency"),
        "equity": actual.get("equity_mentioned") == expected.get("equity_mentioned"),
    }
    fields["score"] = sum(fields.values()) / 4
    return fields


def score_case(output: dict, test_case: dict) -> dict:
    """Score a single model output against the test case ground truth."""
    expected = test_case["expected_output"]
    alt = test_case.get("acceptable_alternatives", {})

    # Schema compliance
    schema = check_schema(output)

    # Field-by-field accuracy
    field_scores = {}

    # Simple string fields
    for field in ["title", "company", "seniority", "remote_policy", "domain"]:
        field_scores[field] = field_match(
            output.get(field),
            expected.get(field),
            alt.get(field),
        )

    # Key requirements — semantic matching (Jaccard kept in output for comparison)
    req_semantic = requirements_semantic(
        output.get("key_requirements", []),
        expected.get("key_requirements", []),
    )
    field_scores["key_requirements"] = req_semantic

    # Compensation — sub-field accuracy
    comp = compensation_match(
        output.get("compensation", {}),
        expected.get("compensation", {}),
    )
    field_scores["compensation"] = comp["score"]

    # Overall accuracy: average of all field scores (booleans count as 0/1)
    values = []
    for v in field_scores.values():
        values.append(float(v) if isinstance(v, (bool, int, float)) else 0.0)
    overall = sum(values) / len(values) if values else 0.0

    return {
        "schema": schema,
        "field_scores": {k: float(v) if isinstance(v, bool) else v for k, v in field_scores.items()},
        "overall_accuracy": round(overall, 4),
    }


def run(model: str, config: dict = None, n_runs: int = 5,
        prompt_variant: str = "production") -> dict:
    """Run all Level 1 test cases and return results."""
    if config is None:
        config = load_config()

    test_cases = load_test_cases()
    results = []

    for tc in test_cases:
        print(f"  L1 [{tc['id']}] {tc['description'][:50]}...", flush=True)
        listing_text = tc["input"]
        if prompt_variant == "adversarial":
            listing_text = format_listing_adversarial(listing_text)
        prompt = build_extract_prompt(listing_text)
        runs = run_n_times(prompt, model, config, n=n_runs)

        case_scores = []
        schema_compliant = 0
        for r in runs:
            if r["success"] and r["parsed"]:
                score = score_case(r["parsed"], tc)
                case_scores.append(score)
                if score["schema"]["is_compliant"]:
                    schema_compliant += 1

        results.append({
            "test_id": tc["id"],
            "description": tc["description"],
            "tags": tc["tags"],
            "n_runs": n_runs,
            "successful_parses": sum(1 for r in runs if r["success"]),
            "schema_compliant": schema_compliant,
            "mean_accuracy": (
                round(sum(s["overall_accuracy"] for s in case_scores) / len(case_scores), 4)
                if case_scores else 0.0
            ),
            "field_scores_avg": _avg_field_scores(case_scores) if case_scores else {},
            "mean_time_s": round(sum(r["time_s"] for r in runs) / len(runs), 2),
            "logprob_stats": [r["logprob_stats"] for r in runs if r["logprob_stats"]],
            "token_metrics": aggregate_token_metrics(runs),
        })

    # Aggregate metrics
    total_cases = len(results)
    mean_accuracy = (
        round(sum(r["mean_accuracy"] for r in results) / total_cases, 4)
        if total_cases else 0.0
    )
    schema_rate = (
        round(sum(r["schema_compliant"] for r in results) / (total_cases * n_runs), 4)
        if total_cases else 0.0
    )
    json_parse_rate = (
        round(sum(r["successful_parses"] for r in results) / (total_cases * n_runs), 4)
        if total_cases else 0.0
    )

    return {
        "level": 1,
        "name": "extraction",
        "model": model,
        "n_runs_per_case": n_runs,
        "total_cases": total_cases,
        "metrics": {
            "field_accuracy": mean_accuracy,
            "schema_compliance": schema_rate,
            "json_parse_rate": json_parse_rate,
        },
        "per_case": results,
    }


def _avg_field_scores(case_scores: list[dict]) -> dict:
    """Average field scores across multiple runs."""
    if not case_scores:
        return {}
    all_fields = case_scores[0]["field_scores"].keys()
    avg = {}
    for field in all_fields:
        vals = [cs["field_scores"][field] for cs in case_scores]
        avg[field] = round(sum(vals) / len(vals), 4)
    return avg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Level 1 — Extraction test")
    parser.add_argument("--model", required=True, help="Ollama model name (e.g. qwen3:32b)")
    parser.add_argument("--n-runs", type=int, default=5, help="Runs per test case")
    args = parser.parse_args()

    result = run(args.model, n_runs=args.n_runs)
    print(json.dumps(result, indent=2))
