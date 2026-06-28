#!/usr/bin/env python3
"""Level 6 — Synthesis test.

Gives the model multiple related data points and asks it to produce
a coherent synthesis. Grades on:
- factual_coverage: % of expected facts mentioned in the output
- no_hallucination: none of the must_not_hallucinate items appear
- coherence: output is valid JSON with required fields
"""

import json
import sys
import time
from pathlib import Path

LEVEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LEVEL_DIR.parent.parent.parent))

from evaluation.helpers import (
    aggregate_token_metrics,
    call_ollama,
    extract_logprob_stats,
    load_config,
    parse_json_response,
)

STRIPPED_PROFILE_SKILLS = [
    "CRM platforms (proficient)", "Analytics tools (proficient)",
    "SQL (proficient)", "Python (working)", "AI/ML tools (working)"
]

SYNTHESIS_PROMPT_TEMPLATE = """You are a job market analyst. Given a set of job listings data, produce a synthesis as requested.

## Listings Data
{listings_json}

{extra_context}

## Task
{task}

Return a JSON object:
{{
  "summary": "Your synthesis — a clear, concise analysis addressing the task above.",
  "key_findings": ["List of 3-5 key findings or patterns you identified"],
  "data_points_used": {count_of_listings_referenced}
}}

Return ONLY the JSON object. No markdown fences."""


def load_test_cases() -> list[dict]:
    with open(LEVEL_DIR / "test_cases.json") as f:
        return json.load(f)


def build_synthesis_prompt(test_case: dict, prompt_variant: str = "production") -> str:
    inp = test_case["input"]
    listings = inp.get("listings", [])

    extra_parts = []
    if "profile_skills" in inp:
        skills = STRIPPED_PROFILE_SKILLS if prompt_variant == "stripped" else inp["profile_skills"]
        extra_parts.append(f"## Candidate Skills\n{json.dumps(skills)}")

    return SYNTHESIS_PROMPT_TEMPLATE.format(
        listings_json=json.dumps(listings, indent=2),
        extra_context="\n\n".join(extra_parts),
        task=inp.get("task", "Synthesize the data."),
        count_of_listings_referenced=len(listings),
    )


def check_fact_coverage(output_text: str, expected_facts: list[str]) -> dict:
    """Check how many expected facts are semantically present in the output.

    Embeds each expected fact and the full output text via nomic-embed-text,
    then checks cosine similarity. A fact is considered covered if its
    similarity to the output exceeds the threshold.
    """
    if not expected_facts:
        return {"found": [], "missing": [], "coverage": 1.0}

    from evaluation.semantic import embed_texts, cosine_similarity

    all_texts = expected_facts + [output_text]
    embeddings = embed_texts(all_texts)
    fact_embeddings = embeddings[:-1]
    output_embedding = embeddings[-1]

    found = []
    missing = []
    for fact, fact_emb in zip(expected_facts, fact_embeddings):
        score = cosine_similarity(fact_emb, output_embedding)
        if score >= 0.70:
            found.append(fact)
        else:
            missing.append(fact)

    coverage = round(len(found) / len(expected_facts), 4)
    return {"found": found, "missing": missing, "coverage": coverage}


def check_hallucination(output_text: str, must_not_items: list[str]) -> dict:
    """Check that none of the forbidden items are semantically present.

    Uses a higher threshold (0.80) than fact coverage — we want to be sure
    the forbidden concept is actually mentioned before flagging a violation.
    """
    if not must_not_items:
        return {"violations": [], "clean": True}

    from evaluation.semantic import embed_texts, cosine_similarity

    all_texts = must_not_items + [output_text]
    embeddings = embed_texts(all_texts)
    item_embeddings = embeddings[:-1]
    output_embedding = embeddings[-1]

    violations = []
    for item, item_emb in zip(must_not_items, item_embeddings):
        score = cosine_similarity(item_emb, output_embedding)
        if score >= 0.80:
            violations.append(item)

    return {
        "violations": violations,
        "clean": len(violations) == 0,
    }


def score_case(output: dict, test_case: dict) -> dict:
    """Score a synthesis output."""
    # Build full text for checking (summary + key_findings)
    summary = output.get("summary", "")
    key_findings = output.get("key_findings", [])
    full_text = summary + " " + " ".join(key_findings)

    # Factual coverage
    facts = check_fact_coverage(full_text, test_case.get("expected_facts", []))

    # Hallucination check
    hallucination = check_hallucination(
        full_text,
        test_case.get("must_not_hallucinate", []),
    )

    # Structural coherence
    has_summary = bool(summary.strip())
    has_findings = len(key_findings) >= 2
    coherent = has_summary and has_findings

    return {
        "factual_coverage": facts["coverage"],
        "facts_found": facts["found"],
        "facts_missing": facts["missing"],
        "hallucination_clean": hallucination["clean"],
        "hallucination_violations": hallucination["violations"],
        "coherent": coherent,
        "has_summary": has_summary,
        "n_findings": len(key_findings),
    }


def run(model: str, config: dict = None, n_runs: int = 5,
        prompt_variant: str = "production") -> dict:
    """Run all Level 6 test cases."""
    if config is None:
        config = load_config()

    test_cases = load_test_cases()
    results = []

    for tc in test_cases:
        print(f"  L6 [{tc['id']}] {tc['description'][:50]}...", flush=True)
        prompt = build_synthesis_prompt(tc, prompt_variant=prompt_variant)

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

        scored_runs = [r for r in run_results if r.get("success") and "scored" in r]
        n_scored = len(scored_runs)

        results.append({
            "test_id": tc["id"],
            "description": tc["description"],
            "tags": tc["tags"],
            "n_runs": n_runs,
            "successful_parses": n_scored,
            "mean_factual_coverage": (
                round(sum(r["scored"]["factual_coverage"] for r in scored_runs) / n_scored, 4)
                if n_scored else 0.0
            ),
            "hallucination_clean_rate": (
                round(sum(1 for r in scored_runs if r["scored"]["hallucination_clean"]) / n_scored, 4)
                if n_scored else 0.0
            ),
            "coherence_rate": (
                round(sum(1 for r in scored_runs if r["scored"]["coherent"]) / n_scored, 4)
                if n_scored else 0.0
            ),
            "mean_time_s": round(sum(r["time_s"] for r in run_results) / len(run_results), 2),
            "token_metrics": aggregate_token_metrics(run_results),
        })

    # Aggregate metrics
    total_cases = len(results)
    factual_coverage = (
        round(sum(r["mean_factual_coverage"] for r in results) / total_cases, 4)
        if total_cases else 0.0
    )
    hallucination_rate = (
        round(sum(r["hallucination_clean_rate"] for r in results) / total_cases, 4)
        if total_cases else 0.0
    )
    coherence = (
        round(sum(r["coherence_rate"] for r in results) / total_cases, 4)
        if total_cases else 0.0
    )

    return {
        "level": 6,
        "name": "synthesis",
        "model": model,
        "n_runs_per_case": n_runs,
        "total_cases": total_cases,
        "metrics": {
            "factual_coverage": factual_coverage,
            "hallucination_clean_rate": hallucination_rate,
            "coherence": coherence,
        },
        "per_case": results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Level 6 — Synthesis test")
    parser.add_argument("--model", required=True, help="Ollama model name")
    parser.add_argument("--n-runs", type=int, default=5, help="Runs per test case")
    args = parser.parse_args()

    result = run(args.model, n_runs=args.n_runs)
    print(json.dumps(result, indent=2))
