#!/usr/bin/env python3
"""Perian evaluation harness — runs all 6 test levels for a given model.

Usage:
    python3 evaluation/run_all.py --model qwen3:32b
    python3 evaluation/run_all.py --model qwen3:14b --n-runs 3
    python3 evaluation/run_all.py --model qwen3:8b --levels 1 2 3

Outputs:
1. Per-level scores printed to console
2. A capability profile YAML block (paste into fleet.yaml)
3. Detailed results saved to evaluation/results/{model}_{timestamp}.json
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(ROOT))

from evaluation.helpers import load_config, get_system_context
from evaluation.stats import bootstrap_ci, format_ci
from scripts.fleet_log import log_event

# Bump this when test cases, ground truth, or scoring logic change.
# Allows longitudinal comparison: only compare runs with matching versions.
# History:
#   1 — initial test suite (2026-03-22)
#   2 — Phase A metric overhaul: semantic matching, distance-based L4, bootstrap CIs (2026-04-03)
#   3 — Phase C ground truth decontamination: 6 cases decontaminated (2026-04-04)
#   4 — Token metrics + system context logging (2026-04-05)
#   5 — Prompt variant support: production/stripped/adversarial (2026-04-05)
EVAL_SUITE_VERSION = 5
from evaluation.test_suite.level_1_extraction.run_test import run as run_l1
from evaluation.test_suite.level_2_classification.run_test import run as run_l2
from evaluation.test_suite.level_3_conditional.run_test import run as run_l3
from evaluation.test_suite.level_4_causal.run_test import run as run_l4
from evaluation.test_suite.level_5_batch.run_test import run as run_l5
from evaluation.test_suite.level_6_synthesis.run_test import run as run_l6

LEVEL_RUNNERS = {
    1: ("extraction", run_l1),
    2: ("classification", run_l2),
    3: ("conditional_reasoning", run_l3),
    4: ("causal_reasoning", run_l4),
    5: ("batch_consistency", run_l5),
    6: ("synthesis", run_l6),
}


def run_evaluation(model: str, levels: list[int] = None, n_runs: int = 5,
                    tags: list[str] = None, notes: str = None,
                    prompt_variant: str = "production") -> dict:
    """Run specified levels (default: all 6) and return combined results.

    Args:
        prompt_variant: "production" (default), "stripped" (generic profile),
                        or "adversarial" (reformatted inputs).
    """
    if levels is None:
        levels = [1, 2, 3, 4, 5, 6]

    config = load_config()
    all_results = {}
    level_times = {}
    t0_total = time.time()

    print(f"\n{'='*60}")
    print(f"  Perian Evaluation Suite v{EVAL_SUITE_VERSION} — {model}")
    print(f"  Levels: {levels}  |  Runs per case: {n_runs}")
    if prompt_variant != "production":
        print(f"  Prompt variant: {prompt_variant}")
    print(f"{'='*60}\n")

    for level in sorted(levels):
        if level not in LEVEL_RUNNERS:
            print(f"  [!] Unknown level {level}, skipping")
            continue

        name, runner = LEVEL_RUNNERS[level]
        print(f"--- Level {level}: {name} ---")
        t0 = time.time()

        try:
            result = runner(model, config=config, n_runs=n_runs,
                            prompt_variant=prompt_variant)
            elapsed = time.time() - t0

            all_results[f"level_{level}"] = result
            level_times[f"L{level}_{name}"] = round(elapsed, 1)
            print(f"  Completed in {elapsed:.1f}s")
            print(f"  Metrics: {json.dumps(result['metrics'], indent=4)}")
            print()
        except Exception as e:
            elapsed = time.time() - t0
            level_times[f"L{level}_{name}"] = round(elapsed, 1)
            print(f"  FAILED after {elapsed:.1f}s: {e}")
            all_results[f"level_{level}"] = {
                "level": level,
                "name": name,
                "model": model,
                "error": str(e),
            }
            print()

    total_time = time.time() - t0_total

    # Build capability profile
    profile = build_capability_profile(model, all_results)

    # Print summary
    print(f"{'='*60}")
    print(f"  CAPABILITY PROFILE — {model}")
    print(f"  Total evaluation time: {total_time:.0f}s")
    print(f"{'='*60}\n")
    print(profile["yaml_block"])
    print()

    # Save results
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    model_safe = model.replace(":", "-").replace("/", "-")
    results_path = RESULTS_DIR / f"{model_safe}_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Aggregate token metrics across all levels
    performance = _aggregate_performance(all_results, level_times, total_time)

    full_output = {
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_suite_version": EVAL_SUITE_VERSION,
        "prompt_variant": prompt_variant,
        "context": get_system_context(),
        "tags": tags or [],
        "notes": notes,
        "n_runs_per_case": n_runs,
        "levels_run": levels,
        "performance": performance,
        "capability_profile": profile["capabilities"],
        "confidence_intervals": profile.get("confidence_intervals", {}),
        "per_case_scores": profile.get("per_case_scores", {}),
        "level_results": all_results,
    }

    with open(results_path, "w") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"  Results saved to: {results_path}")

    # Append to eval index (one-line summary per run for quick lookup)
    _append_eval_index(model, results_path.name, profile["capabilities"],
                       performance, n_runs, prompt_variant, tags)

    # Log to fleet changelog
    log_event(
        "eval_complete",
        model=model,
        scores=profile["capabilities"],
        n_runs=n_runs,
        levels=levels,
        total_time_s=round(total_time, 1),
        results_file=str(results_path.name),
    )
    print(f"  Logged to fleet changelog.")

    return full_output


def build_capability_profile(model: str, results: dict) -> dict:
    """Derive capability scores from level results.

    Maps level metrics to fleet.yaml capability scores:
    - extraction: L1 field_accuracy
    - classification: L2 overall_classification_accuracy
    - conditional_reasoning: L3 score_correlation
    - causal_reasoning: L4 conclusion_accuracy_distance
    - synthesis: L6 factual_coverage
    - json_output: average JSON parse rate across L1-L4
    - nuanced_judgment: weighted average of L3 and L4
    """
    capabilities = {}

    # L1 → extraction
    l1 = results.get("level_1", {})
    l1_metrics = l1.get("metrics", {})
    capabilities["extraction"] = l1_metrics.get("field_accuracy", 0.0)

    # L2 → classification
    l2 = results.get("level_2", {})
    l2_metrics = l2.get("metrics", {})
    capabilities["classification"] = l2_metrics.get("overall_classification_accuracy", 0.0)

    # L3 → conditional_reasoning
    l3 = results.get("level_3", {})
    l3_metrics = l3.get("metrics", {})
    capabilities["conditional_reasoning"] = l3_metrics.get("score_correlation", 0.0)

    # L4 → causal_reasoning
    l4 = results.get("level_4", {})
    l4_metrics = l4.get("metrics", {})
    capabilities["causal_reasoning"] = l4_metrics.get("conclusion_accuracy_distance", 0.0)

    # L6 → synthesis
    l6 = results.get("level_6", {})
    l6_metrics = l6.get("metrics", {})
    capabilities["synthesis"] = l6_metrics.get("factual_coverage", 0.0)

    # json_output: average parse rate across levels with JSON output
    json_rates = []
    for key in ["level_1", "level_2"]:
        level_data = results.get(key, {})
        metrics = level_data.get("metrics", {})
        if "json_parse_rate" in metrics:
            json_rates.append(metrics["json_parse_rate"])
        elif "overall_classification_accuracy" in metrics:
            # L2 parse rate derived from successful parses
            per_case = level_data.get("per_case", [])
            if per_case:
                n_runs = per_case[0].get("n_runs", 5)
                total_parses = sum(c.get("successful_parses", 0) for c in per_case)
                total_attempts = len(per_case) * n_runs
                if total_attempts > 0:
                    json_rates.append(total_parses / total_attempts)

    # Also check L3 and L4 parse rates
    for key in ["level_3", "level_4"]:
        level_data = results.get(key, {})
        per_case = level_data.get("per_case", [])
        if per_case:
            n_runs = per_case[0].get("n_runs", 5)
            total_parses = sum(c.get("successful_parses", 0) for c in per_case)
            total_attempts = len(per_case) * n_runs
            if total_attempts > 0:
                json_rates.append(total_parses / total_attempts)

    capabilities["json_output"] = (
        round(sum(json_rates) / len(json_rates), 4) if json_rates else 0.0
    )

    # nuanced_judgment: weighted average of L3 (0.6) and L4 (0.4)
    l3_score = capabilities.get("conditional_reasoning", 0.0)
    l4_score = capabilities.get("causal_reasoning", 0.0)
    capabilities["nuanced_judgment"] = round(l3_score * 0.6 + l4_score * 0.4, 4)

    # Round all capabilities
    capabilities = {k: round(v, 2) for k, v in capabilities.items()}

    # Compute confidence intervals from per-case scores
    confidence_intervals = {}
    per_case_scores = {}

    # Extract per-case score lists for each capability
    l1_per_case = l1.get("per_case", [])
    if l1_per_case:
        per_case_scores["extraction"] = [c.get("mean_accuracy", 0.0) for c in l1_per_case]

    l2_per_case = l2.get("per_case", [])
    if l2_per_case:
        per_case_scores["classification"] = [c.get("mean_accuracy", 0.0) for c in l2_per_case]

    l3_per_case = l3.get("per_case", [])
    if l3_per_case:
        # Match score_correlation formula: max(0, 1.0 - 2 * deviation)
        per_case_scores["conditional_reasoning"] = [
            max(0, 1.0 - 2 * c.get("mean_deviation", 1.0)) for c in l3_per_case
        ]

    l4_per_case = l4.get("per_case", [])
    if l4_per_case:
        per_case_scores["causal_reasoning"] = [
            c.get("conclusion_accuracy_distance", 0.0) for c in l4_per_case
        ]

    l6_per_case = l6.get("per_case", [])
    if l6_per_case:
        per_case_scores["synthesis"] = [
            c.get("mean_factual_coverage", 0.0) for c in l6_per_case
        ]

    for cap, scores in per_case_scores.items():
        mean, ci_lo, ci_hi = bootstrap_ci(scores, seed=42)
        confidence_intervals[cap] = {
            "mean": mean,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "n_cases": len(scores),
        }

    # Build YAML block
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yaml_lines = [f"# Generated by run_all.py on {timestamp} for {model}"]
    yaml_lines.append("capabilities:")

    # Level labels for comments
    comments = {
        "extraction": f"Level 1: {capabilities['extraction']*100:.0f}% field accuracy",
        "classification": f"Level 2: {capabilities['classification']*100:.0f}% accuracy",
        "conditional_reasoning": f"Level 3: {capabilities['conditional_reasoning']:.2f} score correlation",
        "causal_reasoning": f"Level 4: {capabilities['causal_reasoning']*100:.0f}% conclusion accuracy (distance-based)",
        "synthesis": f"Level 6: {capabilities['synthesis']*100:.0f}% factual coverage",
        "json_output": f"Cross-cutting: {capabilities['json_output']*100:.0f}% valid JSON",
        "nuanced_judgment": f"Derived from L3+L4 weighted average",
    }

    for cap in ["extraction", "classification", "conditional_reasoning",
                "causal_reasoning", "synthesis", "json_output", "nuanced_judgment"]:
        score = capabilities[cap]
        comment = comments.get(cap, "")
        yaml_lines.append(f"  {cap}: {score:.2f}        # {comment}")

    yaml_block = "\n".join(yaml_lines)

    return {
        "capabilities": capabilities,
        "confidence_intervals": confidence_intervals,
        "per_case_scores": per_case_scores,
        "yaml_block": yaml_block,
    }


def _aggregate_performance(all_results: dict, level_times: dict, total_time: float) -> dict:
    """Aggregate timing and token metrics across all levels into a performance summary."""
    perf = {
        "total_time_s": round(total_time, 1),
        "level_times_s": level_times,
    }

    # Collect token metrics from all per-case results across levels
    all_gen_tok_s = []
    all_prompt_tok_s = []
    total_gen_tokens = 0
    total_prompt_tokens = 0

    for level_key, level_data in all_results.items():
        per_case = level_data.get("per_case", [])
        if not per_case:
            # L5 uses per_group instead
            per_case = level_data.get("per_group", [])

        for case in per_case:
            tm = case.get("token_metrics")
            if not tm:
                continue
            if "mean_gen_tok_s" in tm:
                all_gen_tok_s.append(tm["mean_gen_tok_s"])
            if "mean_prompt_tok_s" in tm:
                all_prompt_tok_s.append(tm["mean_prompt_tok_s"])
            if "mean_gen_tokens" in tm:
                n = tm.get("n_samples", 1)
                total_gen_tokens += int(tm["mean_gen_tokens"] * n)
            if "mean_prompt_tokens" in tm:
                n = tm.get("n_samples", 1)
                total_prompt_tokens += int(tm["mean_prompt_tokens"] * n)

    if all_gen_tok_s:
        perf["mean_gen_tok_s"] = round(statistics.mean(all_gen_tok_s), 2)
    if all_prompt_tok_s:
        perf["mean_prompt_tok_s"] = round(statistics.mean(all_prompt_tok_s), 2)
    if total_gen_tokens:
        perf["total_gen_tokens"] = total_gen_tokens
    if total_prompt_tokens:
        perf["total_prompt_tokens"] = total_prompt_tokens

    return perf


EVAL_INDEX_PATH = RESULTS_DIR / "eval-index.jsonl"


def _append_eval_index(model: str, results_file: str, capabilities: dict,
                       performance: dict, n_runs: int, prompt_variant: str,
                       tags: list[str] | None) -> None:
    """Append a one-line summary to eval-index.jsonl for quick lookup."""
    entry = {
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_suite_version": EVAL_SUITE_VERSION,
        "prompt_variant": prompt_variant,
        "n_runs": n_runs,
        "results_file": results_file,
        "total_time_s": performance.get("total_time_s"),
        "mean_gen_tok_s": performance.get("mean_gen_tok_s"),
        "capabilities": capabilities,
        "tags": tags or [],
    }
    with open(EVAL_INDEX_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perian evaluation suite — run all test levels for a model"
    )
    parser.add_argument("--model", required=True, help="Ollama model name (e.g. qwen3:32b)")
    parser.add_argument("--n-runs", type=int, default=5, help="Runs per test case (default: 5)")
    parser.add_argument(
        "--levels", type=int, nargs="+", default=None,
        help="Which levels to run (default: all). Example: --levels 1 2 3"
    )
    parser.add_argument(
        "--tags", nargs="+", default=None,
        help="Tags for this run (e.g. --tags baseline kv-q8 post-decontamination)"
    )
    parser.add_argument(
        "--notes", type=str, default=None,
        help="Free-text note for this run (e.g. --notes 'First run with KV cache q8_0')"
    )
    parser.add_argument(
        "--prompt-variant", choices=["production", "stripped", "adversarial"],
        default="production",
        help="Prompt variant: production (default), stripped (generic profile), adversarial (reformatted inputs)"
    )
    args = parser.parse_args()

    run_evaluation(args.model, levels=args.levels, n_runs=args.n_runs,
                   tags=args.tags, notes=args.notes,
                   prompt_variant=args.prompt_variant)
