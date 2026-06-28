#!/usr/bin/env python3
"""Compare capability profiles of two models side-by-side.

Usage:
    python3 evaluation/compare.py --baseline qwen3:32b --candidate qwen3:14b
    python3 evaluation/compare.py --baseline results/qwen3-32b_*.json --candidate results/qwen3-14b_*.json

Loads the most recent results file for each model (or uses explicit paths)
and prints a comparison table.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.stats import bootstrap_ci, format_ci, scores_significantly_different

RESULTS_DIR = Path(__file__).resolve().parent / "results"

DEFAULT_MIN_RELIABILITY = 0.80

CAPABILITY_ORDER = [
    "extraction",
    "classification",
    "conditional_reasoning",
    "causal_reasoning",
    "synthesis",
    "json_output",
    "nuanced_judgment",
]


def find_latest_results(model: str, variant: str = None) -> dict | None:
    """Find the most recent results file for a model.

    Args:
        variant: If set, only return results matching this prompt_variant.
                 If None, returns the latest result regardless of variant.
    """
    model_safe = model.replace(":", "-").replace("/", "-")
    pattern = str(RESULTS_DIR / f"{model_safe}_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None

    # Search from newest to oldest
    for f_path in reversed(files):
        with open(f_path) as f:
            data = json.load(f)
        if variant is None:
            return data
        if data.get("prompt_variant", "production") == variant:
            return data

    return None


def load_results(model_or_path: str, variant: str = None) -> dict:
    """Load results by model name or explicit file path.

    Args:
        variant: If set, filter by prompt_variant when searching by model name.
    """
    path = Path(model_or_path)
    if path.exists() and path.suffix == ".json":
        with open(path) as f:
            return json.load(f)

    results = find_latest_results(model_or_path, variant=variant)
    if results is None:
        msg = f"Error: No results found for '{model_or_path}'"
        if variant:
            msg += f" with variant '{variant}'"
        print(msg)
        print(f"Run: python3 evaluation/run_all.py --model {model_or_path}")
        sys.exit(1)
    return results


def compare(baseline_data: dict, candidate_data: dict, min_reliability: float) -> None:
    """Print a comparison table with CIs and significance tests."""
    baseline_model = baseline_data.get("model", "baseline")
    candidate_model = candidate_data.get("model", "candidate")
    baseline_caps = baseline_data.get("capability_profile", {})
    candidate_caps = candidate_data.get("capability_profile", {})
    baseline_ci = baseline_data.get("confidence_intervals", {})
    candidate_ci = candidate_data.get("confidence_intervals", {})
    baseline_per_case = baseline_data.get("per_case_scores", {})
    candidate_per_case = candidate_data.get("per_case_scores", {})

    has_ci = bool(baseline_ci or candidate_ci)

    # Header
    b_label = baseline_model[:18]
    c_label = candidate_model[:18]

    print()
    if has_ci:
        print(f"{'Capability':<25} {'Baseline':<20} {'Candidate':<20} {'Significant?':<16}")
        print(f"{'':─<25} {'(' + b_label + ')':<20} {'(' + c_label + ')':<20} {'':─<16}")
    else:
        print(f"{'Capability':<25} {'Baseline':>10} {'Candidate':>10} {'Delta':>8} {'Min':>6} {'Pass?':>6}")
        print(f"{'':─<25} {'(' + b_label + ')':>10} {'(' + c_label + ')':>10} {'':─>8} {min_reliability:>6.2f} {'':─>6}")

    all_pass = True
    for cap in CAPABILITY_ORDER:
        b_score = baseline_caps.get(cap, 0.0)
        c_score = candidate_caps.get(cap, 0.0)
        passes = c_score >= min_reliability
        if not passes:
            all_pass = False

        if has_ci:
            # Format with CIs if available
            b_ci_data = baseline_ci.get(cap)
            c_ci_data = candidate_ci.get(cap)
            b_str = format_ci(b_ci_data["mean"], b_ci_data["ci_lower"], b_ci_data["ci_upper"]) if b_ci_data else f"{b_score:.2f}"
            c_str = format_ci(c_ci_data["mean"], c_ci_data["ci_lower"], c_ci_data["ci_upper"]) if c_ci_data else f"{c_score:.2f}"

            # Significance test if we have per-case scores for both
            b_scores = baseline_per_case.get(cap, [])
            c_scores = candidate_per_case.get(cap, [])
            if b_scores and c_scores and len(b_scores) == len(c_scores):
                is_sig, p_val = scores_significantly_different(b_scores, c_scores, seed=42)
                sig_str = f"Yes (p={p_val:.2f})" if is_sig else f"No (p={p_val:.2f})"
            else:
                sig_str = "n/a"

            marker = " !" if not passes else ""
            print(f"  {cap:<23} {b_str:<20} {c_str:<20} {sig_str:<16}{marker}")
        else:
            delta = c_score - b_score
            delta_str = f"{delta:+.2f}"
            if delta > 0.05:
                delta_str = f"{delta_str} ^"
            elif delta < -0.05:
                delta_str = f"{delta_str} v"
            pass_str = "YES" if passes else "NO !"
            print(f"  {cap:<23} {b_score:>10.2f} {c_score:>10.2f} {delta_str:>8} {min_reliability:>6.2f} {pass_str:>6}")

    print()

    # Summary
    b_avg = sum(baseline_caps.get(c, 0) for c in CAPABILITY_ORDER) / len(CAPABILITY_ORDER)
    c_avg = sum(candidate_caps.get(c, 0) for c in CAPABILITY_ORDER) / len(CAPABILITY_ORDER)
    print(f"  {'AVERAGE':<23} {b_avg:>10.2f} {c_avg:>10.2f} {c_avg - b_avg:>+8.2f}")
    print()

    if not all_pass:
        below = [
            cap for cap in CAPABILITY_ORDER
            if candidate_caps.get(cap, 0) < min_reliability
        ]
        print(f"  WARNING: {candidate_model} falls below {min_reliability} threshold on:")
        for cap in below:
            print(f"    - {cap}: {candidate_caps.get(cap, 0):.2f}")
        print(f"  These tasks MUST stay on {baseline_model}.")
    else:
        print(f"  {candidate_model} meets threshold on all capabilities.")

    print()

    # Timing comparison
    b_time = baseline_data.get("total_time_s", 0)
    c_time = candidate_data.get("total_time_s", 0)
    if b_time and c_time:
        speedup = b_time / c_time if c_time > 0 else 0
        print(f"  Eval time: {baseline_model} {b_time:.0f}s / {candidate_model} {c_time:.0f}s ({speedup:.1f}x)")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two model capability profiles")
    parser.add_argument("--baseline", required=True, help="Baseline model name or results JSON path")
    parser.add_argument("--candidate", required=True, help="Candidate model name or results JSON path")
    parser.add_argument(
        "--min-reliability", type=float, default=DEFAULT_MIN_RELIABILITY,
        help=f"Minimum reliability threshold (default: {DEFAULT_MIN_RELIABILITY})"
    )
    parser.add_argument(
        "--variant", choices=["production", "stripped", "adversarial"],
        default=None,
        help="Filter results by prompt variant (when loading by model name)"
    )
    args = parser.parse_args()

    baseline_data = load_results(args.baseline, variant=args.variant)
    candidate_data = load_results(args.candidate, variant=args.variant)
    compare(baseline_data, candidate_data, args.min_reliability)
