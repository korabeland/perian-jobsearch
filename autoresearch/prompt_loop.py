"""Autoresearch eval and validation library.

Provides deterministic evaluation, structural validation, failure analysis,
and safety gate functions. Used by eval_cli.py (headless sessions) and
can be imported directly for testing.

The optimization loop itself is driven by a headless Claude Code session —
see headless_prompt.md for the protocol.
"""

import json
import logging
import random
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.helpers import run_n_times

from autoresearch.shared import (
    load_autoresearch_config,
    render_prompt,
)

logger = logging.getLogger("perian.autoresearch.loop")

# Import scoring functions from eval suite
from evaluation.test_suite.level_1_extraction.run_test import (
    score_case as l1_score_case,
)
from evaluation.test_suite.level_3_conditional.run_test import (
    score_case as l3_score_case,
)


# ---------------------------------------------------------------------------
# Holdout loading and splitting
# ---------------------------------------------------------------------------

HOLDOUT_DIR = Path(__file__).resolve().parent / "holdout"


def load_holdout(template_type: str) -> list[dict]:
    """Load holdout data from JSON file."""
    path = HOLDOUT_DIR / f"{template_type}_holdout.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Holdout data not found: {path}\n"
            f"Run: python3 autoresearch/generate_holdout.py --template {template_type}"
        )
    with open(path) as f:
        return json.load(f)


def split_holdout(
    items: list[dict], ratio: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Deterministic split into optimization (ratio) and validation (1-ratio) sets."""
    rng = random.Random(seed)
    shuffled = items.copy()
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_prompt(
    template_text: str,
    items: list[dict],
    template_type: str,
    model: str,
    config: dict,
    n_runs: int = 3,
    plan: dict = None,
    profile: str = None,
) -> dict:
    """Evaluate a prompt template against holdout items.

    Returns aggregate metrics and per-item details.
    """
    per_item = []
    total_score = 0
    total_pass = 0

    for i, item in enumerate(items):
        prompt = render_prompt(
            template_text, item["input"], template_type,
            plan=plan, profile=profile,
        )
        runs = run_n_times(prompt, model, config, n=n_runs)

        item_scores = []
        for r in runs:
            if not r["success"] or not r["parsed"]:
                continue

            if template_type == "extract":
                # Build a test_case-like dict for l1_score_case
                test_case = {
                    "expected_output": item.get("expected_output", {}),
                    "acceptable_alternatives": item.get("acceptable_alternatives", {}),
                }
                score = l1_score_case(r["parsed"], test_case)
                item_scores.append(score["overall_accuracy"])
            else:
                test_case = {
                    "expected_output": item.get("expected_output", {}),
                }
                score = l3_score_case(r["parsed"], test_case)
                # Convert deviation to accuracy (lower deviation = higher score)
                accuracy = max(0, 1.0 - 2 * score["mean_deviation"])
                item_scores.append(accuracy)

        if item_scores:
            mean_score = sum(item_scores) / len(item_scores)
            passed = mean_score > 0.7
        else:
            mean_score = 0.0
            passed = False

        total_score += mean_score
        total_pass += int(passed)

        per_item.append({
            "id": item.get("id", f"item_{i}"),
            "description": item.get("description", ""),
            "mean_score": round(mean_score, 4),
            "passed": passed,
            "n_successful": len(item_scores),
            "n_runs": n_runs,
            "input": item["input"],
            "expected_output": item.get("expected_output"),
        })

    n_items = len(items)
    return {
        "mean_score": round(total_score / n_items, 4) if n_items else 0.0,
        "pass_rate": round(total_pass / n_items, 4) if n_items else 0.0,
        "n_items": n_items,
        "per_item": per_item,
    }


# ---------------------------------------------------------------------------
# Failure analysis
# ---------------------------------------------------------------------------


def find_failures(eval_result: dict, max_n: int = 10) -> list[dict]:
    """Extract worst-performing items sorted by score ascending."""
    items = eval_result["per_item"]
    sorted_items = sorted(items, key=lambda x: x["mean_score"])
    return sorted_items[:max_n]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_mutation(
    candidate: str, original: str, template_type: str, ar_config: dict
) -> dict:
    """Structural validation of a candidate prompt mutation."""
    reasons = []

    # Delimiter check
    if "--- ITEM TO PROCESS ---" not in candidate:
        reasons.append("Missing delimiter: --- ITEM TO PROCESS ---")

    # Placeholder checks
    if template_type == "extract":
        if "{{listing}}" not in candidate:
            reasons.append("Missing placeholder: {{listing}}")
    elif template_type == "score":
        for ph in ["{{extracted}}", "{{profile}}"]:
            if ph not in candidate:
                reasons.append(f"Missing placeholder: {ph}")
        # Check for at least one scoring variable
        if "{{scoring" not in candidate:
            reasons.append("Missing scoring template variables ({{scoring.*}})")

    # Length check
    max_increase = ar_config["mutation"]["max_prompt_length_increase"]
    if len(candidate) > len(original) * (1 + max_increase):
        reasons.append(
            f"Prompt too long: {len(candidate)} chars "
            f"(max {int(len(original) * (1 + max_increase))} = "
            f"original {len(original)} + {int(max_increase*100)}%)"
        )

    return {
        "valid": len(reasons) == 0,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Safety gate
# ---------------------------------------------------------------------------


def run_safety_gate(model: str, ar_config: dict) -> dict:
    """Run eval suite L1-L3 as subprocess, compare against baseline capabilities."""
    safety = ar_config["safety"]
    levels = safety["eval_levels"]
    n_runs = safety["eval_n_runs"]

    levels_str = " ".join(str(l) for l in levels)
    cmd = [
        sys.executable,
        str(ROOT / "evaluation" / "run_all.py"),
        "--model", model,
        "--n-runs", str(n_runs),
        "--levels", *[str(l) for l in levels],
    ]

    logger.info(f"Safety gate: running eval L{levels_str} with {n_runs} runs...")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, cwd=str(ROOT)
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "Safety gate timed out (600s)"}

    if result.returncode != 0:
        logger.error(f"Safety gate failed: {result.stderr[:500]}")
        return {"passed": False, "error": result.stderr[:500]}

    # Find the most recent results file
    results_dir = ROOT / "evaluation" / "results"
    if not results_dir.exists():
        return {"passed": False, "error": "No results directory found"}

    model_safe = model.replace(":", "-").replace("/", "-")
    result_files = sorted(
        results_dir.glob(f"{model_safe}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not result_files:
        return {"passed": False, "error": "No results file found after eval run"}

    with open(result_files[0]) as f:
        eval_results = json.load(f)

    capabilities = eval_results.get("capability_profile", {})
    max_drop = safety["max_capability_drop"]

    # Load baseline capabilities from fleet.yaml
    fleet_path = ROOT / "config" / "fleet.yaml"
    with open(fleet_path) as f:
        fleet = yaml.safe_load(f)

    # Find matching model in fleet (fleet.yaml uses "id" and "ollama_name")
    baseline_caps = {}
    for m in fleet.get("fleet", {}).get("models", []):
        if m.get("ollama_name") == model or m.get("id") == model.replace(":", "-"):
            baseline_caps = m.get("capabilities", {})
            break

    if not baseline_caps:
        logger.warning(f"No baseline capabilities found for {model} in fleet.yaml")
        return {"passed": True, "capabilities": capabilities, "regressions": [],
                "warning": "No baseline to compare against"}

    regressions = []
    for cap, new_val in capabilities.items():
        baseline_val = baseline_caps.get(cap)
        if baseline_val is not None and (baseline_val - new_val) > max_drop:
            regressions.append({
                "capability": cap,
                "baseline": baseline_val,
                "new": new_val,
                "drop": round(baseline_val - new_val, 4),
            })

    return {
        "passed": len(regressions) == 0,
        "capabilities": capabilities,
        "regressions": regressions,
    }


# ---------------------------------------------------------------------------
# Git operations
# ---------------------------------------------------------------------------


def git_commit_prompt(template_type: str, message: str, ar_config: dict) -> bool:
    """Commit the prompt file with an autoresearch prefix."""
    prefix = ar_config["git"]["commit_prefix"]
    prompt_path = f"prompts/{template_type}.md"
    full_message = f"{prefix} {message}"

    try:
        subprocess.run(
            ["git", "add", prompt_path],
            check=True, capture_output=True, cwd=str(ROOT),
        )
        subprocess.run(
            ["git", "commit", "-m", full_message],
            check=True, capture_output=True, cwd=str(ROOT),
        )
        logger.info(f"Committed: {full_message}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Git commit failed: {e.stderr}")
        return False


def git_revert_last() -> bool:
    """Revert the last commit."""
    try:
        subprocess.run(
            ["git", "revert", "--no-edit", "HEAD"],
            check=True, capture_output=True, cwd=str(ROOT),
        )
        logger.info("Reverted last commit")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Git revert failed: {e.stderr}")
        return False


