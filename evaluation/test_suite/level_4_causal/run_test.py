#!/usr/bin/env python3
"""Level 4 — Causal reasoning test.

Presents multi-step reasoning scenarios and checks whether the model:
1. Reaches the correct conclusion (strong/moderate/weak/no match)
2. Mentions the key causal factors in its reasoning
3. Returns valid JSON

Metrics:
- conclusion_accuracy: % of correct match level conclusions
- reasoning_coverage: % of expected key factors mentioned in reasoning
"""

import json
import sys
from pathlib import Path

LEVEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LEVEL_DIR.parent.parent.parent))

from evaluation.helpers import (
    load_config,
    aggregate_token_metrics,
    call_ollama,
    parse_json_response,
    extract_logprob_stats,
    load_profile,
)
import time

CAUSAL_PROMPT_TEMPLATE = """You are a job match analyst. Given a job listing and a candidate profile summary, analyze whether this is a good match. Think through the reasoning step by step.

## Candidate Profile Summary
{profile_summary}

## Job Listing
{listing_json}

## Context
{context}

## Task
Analyze this match. Consider:
1. Does the title/seniority align with what the candidate is targeting?
2. Is the domain a target domain or adjacent?
3. What fraction of requirements does the candidate meet?
4. Are there any dealbreakers (location, employment type, clearance, company size)?
5. What is the overall match quality considering all factors together?

Return a JSON object:
{{
  "conclusion": "strong_match | moderate_match | weak_match | no_match",
  "confidence": 0.0 to 1.0,
  "key_factors": ["list of the most important factors in your decision"],
  "reasoning": "2-3 sentence explanation of your causal reasoning — why does this combination of factors lead to your conclusion?"
}}

Return ONLY the JSON object. No markdown fences."""

VALID_CONCLUSIONS = {"strong_match", "moderate_match", "weak_match", "no_match"}

STRIPPED_PROFILE_SUMMARY = (
    "Technology leadership professional (10+ years). "
    "Targets product management, strategy, and customer experience roles. "
    "Skills: CRM platforms, analytics tools, SQL, Python, AI/ML tools. "
    "Remote or hybrid preferred."
)

# Map conclusions to numeric scores for distance calculation
CONCLUSION_SCORES = {
    "strong_match": 3,
    "moderate_match": 2,
    "weak_match": 1,
    "no_match": 0,
}


def load_test_cases() -> list[dict]:
    with open(LEVEL_DIR / "test_cases.json") as f:
        return json.load(f)


def build_causal_prompt(test_case: dict, prompt_variant: str = "production") -> str:
    inp = test_case["input"]
    if prompt_variant == "stripped":
        profile_summary = STRIPPED_PROFILE_SUMMARY
    else:
        profile_summary = inp.get("profile_summary", "")
    return CAUSAL_PROMPT_TEMPLATE.format(
        profile_summary=profile_summary,
        listing_json=json.dumps(inp.get("listing", {}), indent=2),
        context=test_case.get("prompt_context", ""),
    )


def score_case(output: dict, test_case: dict) -> dict:
    """Score a causal reasoning output."""
    actual_conclusion = output.get("conclusion", "").lower().strip()
    expected_conclusion = test_case["expected_conclusion"]

    # Exact conclusion match
    conclusion_exact = actual_conclusion == expected_conclusion

    # Distance-based: how far off is the conclusion?
    actual_score = CONCLUSION_SCORES.get(actual_conclusion)
    expected_score = CONCLUSION_SCORES.get(expected_conclusion)
    if actual_score is not None and expected_score is not None:
        conclusion_distance = abs(actual_score - expected_score)
    else:
        conclusion_distance = 3  # maximum distance

    # Reasoning coverage: check if expected factors are mentioned
    reasoning_text = (
        output.get("reasoning", "")
        + " "
        + " ".join(output.get("key_factors", []))
    ).lower()

    expected_mentions = test_case.get("expected_reasoning_mentions", [])
    mentions_found = [
        mention
        for mention in expected_mentions
        if mention.lower() in reasoning_text
    ]
    mentions_coverage = (
        round(len(mentions_found) / len(expected_mentions), 4)
        if expected_mentions else 1.0
    )

    # Valid conclusion enum
    valid_conclusion = actual_conclusion in VALID_CONCLUSIONS

    return {
        "conclusion_exact": conclusion_exact,
        "conclusion_actual": actual_conclusion,
        "conclusion_expected": expected_conclusion,
        "conclusion_distance": conclusion_distance,
        "conclusion_accuracy_distance": round(max(0, 1.0 - conclusion_distance / 3.0), 4),
        "valid_conclusion": valid_conclusion,
        "mentions_coverage": mentions_coverage,
        "mentions_found": mentions_found,
        "mentions_expected": expected_mentions,
        "has_reasoning": bool(output.get("reasoning")),
    }


def run(model: str, config: dict = None, n_runs: int = 5,
        prompt_variant: str = "production") -> dict:
    """Run all Level 4 test cases."""
    if config is None:
        config = load_config()

    test_cases = load_test_cases()
    results = []

    for tc in test_cases:
        print(f"  L4 [{tc['id']}] {tc['description'][:50]}...", flush=True)
        prompt = build_causal_prompt(tc, prompt_variant=prompt_variant)

        run_results = []
        for i in range(n_runs):
            t0 = time.time()
            response_data = call_ollama(prompt, model, config)
            elapsed = time.time() - t0

            if response_data is None:
                run_results.append({"success": False, "time_s": elapsed, "token_metrics": None})
                continue

            raw = response_data.get("response", "")
            parsed = parse_json_response(raw)
            lp = extract_logprob_stats(response_data)
            tok = response_data.get("token_metrics")

            if parsed:
                scored = score_case(parsed, tc)
                run_results.append({
                    "success": True,
                    "scored": scored,
                    "time_s": elapsed,
                    "logprob_stats": lp,
                    "token_metrics": tok,
                })
            else:
                run_results.append({"success": False, "time_s": elapsed, "token_metrics": tok})

        # Aggregate per-case
        scored_runs = [r for r in run_results if r.get("success") and "scored" in r]
        n_scored = len(scored_runs)

        results.append({
            "test_id": tc["id"],
            "description": tc["description"],
            "tags": tc["tags"],
            "n_runs": n_runs,
            "successful_parses": n_scored,
            "conclusion_accuracy": (
                round(sum(1 for r in scored_runs if r["scored"]["conclusion_exact"]) / n_scored, 4)
                if n_scored else 0.0
            ),
            "conclusion_accuracy_distance": (
                round(sum(r["scored"]["conclusion_accuracy_distance"] for r in scored_runs) / n_scored, 4)
                if n_scored else 0.0
            ),
            "mean_conclusion_distance": (
                round(sum(r["scored"]["conclusion_distance"] for r in scored_runs) / n_scored, 4)
                if n_scored else 3.0
            ),
            "reasoning_coverage": (
                round(sum(r["scored"]["mentions_coverage"] for r in scored_runs) / n_scored, 4)
                if n_scored else 0.0
            ),
            "mean_time_s": round(sum(r["time_s"] for r in run_results) / len(run_results), 2),
            "token_metrics": aggregate_token_metrics(run_results),
        })

    # Aggregate metrics
    total_cases = len(results)
    conclusion_acc = (
        round(sum(r["conclusion_accuracy"] for r in results) / total_cases, 4)
        if total_cases else 0.0
    )
    conclusion_acc_dist = (
        round(sum(r["conclusion_accuracy_distance"] for r in results) / total_cases, 4)
        if total_cases else 0.0
    )
    reasoning_cov = (
        round(sum(r["reasoning_coverage"] for r in results) / total_cases, 4)
        if total_cases else 0.0
    )
    mean_distance = (
        round(sum(r["mean_conclusion_distance"] for r in results) / total_cases, 4)
        if total_cases else 3.0
    )

    return {
        "level": 4,
        "name": "causal_reasoning",
        "model": model,
        "n_runs_per_case": n_runs,
        "total_cases": total_cases,
        "metrics": {
            "conclusion_accuracy": conclusion_acc,
            "conclusion_accuracy_distance": conclusion_acc_dist,
            "reasoning_coverage": reasoning_cov,
            "mean_conclusion_distance": mean_distance,
        },
        "per_case": results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Level 4 — Causal reasoning test")
    parser.add_argument("--model", required=True, help="Ollama model name")
    parser.add_argument("--n-runs", type=int, default=5, help="Runs per test case")
    args = parser.parse_args()

    result = run(args.model, n_runs=args.n_runs)
    print(json.dumps(result, indent=2))
