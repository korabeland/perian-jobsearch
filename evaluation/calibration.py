#!/usr/bin/env python3
"""Calibration analysis for Perian eval results.

Computes Expected Calibration Error (ECE) using consistency-based confidence
(agreement rate across N runs per case) as a proxy for model confidence.

Logprob-based ECE is blocked: Ollama 0.20.0 does not return logprobs from
/api/chat. Stub functions are included for when support lands.

Usage:
    python3 evaluation/calibration.py evaluation/results/qwen3-8b_20260404T230112.json
    python3 evaluation/calibration.py evaluation/results/qwen3-8b_*.json --n-bins 5
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Consistency-based confidence (works now)
# ---------------------------------------------------------------------------

def compute_consistency_confidence(per_case: list[dict], level: int) -> list[dict]:
    """Compute confidence and accuracy per case from eval results.

    Uses agreement rate across runs as a confidence proxy:
    - L1/L2: successful_parses / n_runs as confidence, mean_accuracy as accuracy
    - L3: 1 - mean_deviation as confidence, total_in_range_rate as accuracy
    - L4: successful_parses / n_runs as confidence, conclusion_accuracy_distance as accuracy
    - L5: skipped (groups, not comparable cases)
    - L6: successful_parses / n_runs as confidence, mean_factual_coverage as accuracy

    Returns list of {"confidence": float, "accuracy": float, "case_id": str}.
    """
    if level == 5:
        return []

    results = []
    for case in per_case:
        n_runs = case.get("n_runs", 5)
        successful = case.get("successful_parses", 0)
        case_id = case.get("test_id", "unknown")

        if level in (1, 2):
            confidence = successful / n_runs if n_runs > 0 else 0.0
            accuracy = case.get("mean_accuracy", 0.0)
        elif level == 3:
            confidence = max(0.0, 1.0 - case.get("mean_deviation", 1.0))
            accuracy = case.get("total_in_range_rate", 0.0)
        elif level == 4:
            confidence = successful / n_runs if n_runs > 0 else 0.0
            accuracy = case.get("conclusion_accuracy_distance", 0.0)
        elif level == 6:
            confidence = successful / n_runs if n_runs > 0 else 0.0
            accuracy = case.get("mean_factual_coverage", 0.0)
        else:
            continue

        results.append({
            "confidence": round(confidence, 4),
            "accuracy": round(accuracy, 4),
            "case_id": case_id,
        })

    return results


def compute_ece(confidences: list[float], accuracies: list[float],
                n_bins: int = 5) -> float:
    """Compute Expected Calibration Error.

    Bins samples by confidence, measures |mean_conf - mean_acc| per bin,
    weighted by bin size. Uses 5 bins by default (small eval suites have
    7-10 cases per level, so 10 bins would be mostly empty).
    """
    if not confidences or not accuracies:
        return 0.0

    n = len(confidences)
    bin_width = 1.0 / n_bins

    total_ece = 0.0
    for i in range(n_bins):
        bin_lower = i * bin_width
        bin_upper = (i + 1) * bin_width

        # Collect samples in this bin
        bin_confs = []
        bin_accs = []
        for conf, acc in zip(confidences, accuracies):
            if bin_lower <= conf < bin_upper or (i == n_bins - 1 and conf == bin_upper):
                bin_confs.append(conf)
                bin_accs.append(acc)

        if not bin_confs:
            continue

        mean_conf = sum(bin_confs) / len(bin_confs)
        mean_acc = sum(bin_accs) / len(bin_accs)
        total_ece += (len(bin_confs) / n) * abs(mean_conf - mean_acc)

    return round(total_ece, 4)


def reliability_diagram_data(confidences: list[float], accuracies: list[float],
                              n_bins: int = 5) -> list[dict]:
    """Compute reliability diagram bin data.

    Returns list of {"bin_lower", "bin_upper", "mean_confidence",
    "mean_accuracy", "count"} for each non-empty bin.
    """
    if not confidences or not accuracies:
        return []

    bin_width = 1.0 / n_bins
    bins = []

    for i in range(n_bins):
        bin_lower = round(i * bin_width, 4)
        bin_upper = round((i + 1) * bin_width, 4)

        bin_confs = []
        bin_accs = []
        for conf, acc in zip(confidences, accuracies):
            if bin_lower <= conf < bin_upper or (i == n_bins - 1 and conf == bin_upper):
                bin_confs.append(conf)
                bin_accs.append(acc)

        if bin_confs:
            bins.append({
                "bin_lower": bin_lower,
                "bin_upper": bin_upper,
                "mean_confidence": round(sum(bin_confs) / len(bin_confs), 4),
                "mean_accuracy": round(sum(bin_accs) / len(bin_accs), 4),
                "count": len(bin_confs),
            })

    return bins


def calibration_report(results_path: str, n_bins: int = 5) -> dict:
    """Load a result JSON and compute per-level and overall calibration.

    Returns a structured report with ECE per level and overall.
    """
    with open(results_path) as f:
        data = json.load(f)

    model = data.get("model", "unknown")
    level_results = data.get("level_results", {})

    all_confidences = []
    all_accuracies = []
    per_level = {}

    for level_key, level_data in sorted(level_results.items()):
        level_num = level_data.get("level")
        if level_num is None or level_num == 5:
            continue

        per_case = level_data.get("per_case", [])
        cases = compute_consistency_confidence(per_case, level_num)

        if not cases:
            continue

        confs = [c["confidence"] for c in cases]
        accs = [c["accuracy"] for c in cases]

        level_ece = compute_ece(confs, accs, n_bins=n_bins)
        diagram = reliability_diagram_data(confs, accs, n_bins=n_bins)

        n_cases = len(cases)
        per_level[f"L{level_num}"] = {
            "ece": level_ece,
            "n_cases": n_cases,
            "mean_confidence": round(sum(confs) / n_cases, 4),
            "mean_accuracy": round(sum(accs) / n_cases, 4),
            "reliability_diagram": diagram,
            "sparse_warning": n_cases < 20,
        }

        all_confidences.extend(confs)
        all_accuracies.extend(accs)

    overall_ece = compute_ece(all_confidences, all_accuracies, n_bins=n_bins)
    overall_diagram = reliability_diagram_data(all_confidences, all_accuracies, n_bins=n_bins)

    return {
        "model": model,
        "results_file": str(results_path),
        "n_bins": n_bins,
        "overall_ece": overall_ece,
        "overall_n_cases": len(all_confidences),
        "overall_mean_confidence": (
            round(sum(all_confidences) / len(all_confidences), 4)
            if all_confidences else 0.0
        ),
        "overall_mean_accuracy": (
            round(sum(all_accuracies) / len(all_accuracies), 4)
            if all_accuracies else 0.0
        ),
        "overall_reliability_diagram": overall_diagram,
        "per_level": per_level,
        "sparse_warning": len(all_confidences) < 20,
    }


# ---------------------------------------------------------------------------
# Logprob-based confidence (stubs — blocked by Ollama 0.20.0)
# ---------------------------------------------------------------------------

def compute_logprob_confidence(logprob_stats: dict) -> float | None:
    """Compute confidence from token logprobs.

    Blocked: Ollama 0.20.0 does not return logprobs from /api/chat.
    When support lands, convert mean_logprob to a probability:
        confidence = exp(mean_logprob)
    """
    # Ollama 0.20.0 does not return logprobs — return None
    return None


def compute_logprob_ece(results_path: str) -> float | None:
    """Compute ECE using logprob-derived confidence.

    Blocked: Ollama 0.20.0 does not return logprobs from /api/chat.
    """
    # Ollama 0.20.0 does not return logprobs — return None
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(report: dict) -> None:
    """Pretty-print a calibration report."""
    model = report["model"]
    n_cases = report["overall_n_cases"]
    ece = report["overall_ece"]

    print(f"\n{'='*55}")
    print(f"  Calibration Report — {model}")
    print(f"  {n_cases} cases across {len(report['per_level'])} levels, {report['n_bins']} bins")
    if report["sparse_warning"]:
        print(f"  WARNING: Only {n_cases} cases — ECE may be unreliable (need 20+)")
    print(f"{'='*55}\n")

    print(f"  Overall ECE: {ece:.4f}")
    print(f"  Mean confidence: {report['overall_mean_confidence']:.4f}")
    print(f"  Mean accuracy:   {report['overall_mean_accuracy']:.4f}")
    print()

    # Per-level
    print(f"  {'Level':<8} {'ECE':<8} {'Cases':<7} {'Conf':<8} {'Acc':<8} {'Note'}")
    print(f"  {'─'*8} {'─'*8} {'─'*7} {'─'*8} {'─'*8} {'─'*10}")
    for level_key in sorted(report["per_level"]):
        ldata = report["per_level"][level_key]
        note = "sparse" if ldata["sparse_warning"] else ""
        print(f"  {level_key:<8} {ldata['ece']:<8.4f} {ldata['n_cases']:<7} "
              f"{ldata['mean_confidence']:<8.4f} {ldata['mean_accuracy']:<8.4f} {note}")

    print()

    # Reliability diagram
    print("  Reliability diagram:")
    print(f"  {'Bin':<12} {'Conf':<8} {'Acc':<8} {'Count':<6} {'Gap'}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*6} {'─'*10}")
    for b in report["overall_reliability_diagram"]:
        gap = b["mean_confidence"] - b["mean_accuracy"]
        bar = "+" * int(abs(gap) * 20)
        direction = "over" if gap > 0 else "under"
        print(f"  [{b['bin_lower']:.1f}-{b['bin_upper']:.1f})"
              f"  {b['mean_confidence']:<8.3f} {b['mean_accuracy']:<8.3f} {b['count']:<6}"
              f" {gap:+.3f} {bar} ({direction})")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibration analysis for Perian eval results"
    )
    parser.add_argument(
        "results", nargs="+",
        help="Result JSON file(s) — supports glob patterns"
    )
    parser.add_argument(
        "--n-bins", type=int, default=5,
        help="Number of bins for ECE (default: 5)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of formatted report"
    )
    args = parser.parse_args()

    # Expand glob patterns
    paths = []
    for pattern in args.results:
        expanded = sorted(glob.glob(pattern))
        if expanded:
            paths.extend(expanded)
        else:
            paths.append(pattern)

    for path in paths:
        if not Path(path).exists():
            print(f"Error: {path} not found", file=sys.stderr)
            continue

        report = calibration_report(path, n_bins=args.n_bins)

        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_report(report)
