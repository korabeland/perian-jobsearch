#!/usr/bin/env python3
"""Generate synthetic holdout data for autoresearch prompt optimization.

Creates labeled test items via Claude API, then runs them through Ollama
to capture baseline model outputs. The holdout data is used by prompt_loop.py
to evaluate candidate prompt mutations.

TODO(real-data): This generates SYNTHETIC data as scaffolding. When a real
labeled data pipeline exists (e.g., from feedback.py ratings or manual review),
add a --from-file flag that loads real items in the same format and skips the
Claude generation step. The baseline step (running items through Ollama) still
applies. Synthetic holdout should be cleared and replaced entirely at that point
— don't mix synthetic and real data in the same holdout set.

Usage:
    python3 autoresearch/generate_holdout.py --template extract
    python3 autoresearch/generate_holdout.py --template score
    python3 autoresearch/generate_holdout.py --template both
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.helpers import (
    call_ollama,
    load_config,
    load_plan,
    load_profile,
    load_prompt_template,
    parse_json_response,
    run_n_times,
)

from autoresearch.shared import (
    load_autoresearch_config,
    render_prompt,
)


# ---------------------------------------------------------------------------
# Claude API helpers (inlined — only used for one-time holdout generation)
# ---------------------------------------------------------------------------

# Sonnet 4 pricing (per million tokens)
_PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
}
_DEFAULT_PRICING = {"input": 3.0, "output": 15.0}


def get_claude_client() -> anthropic.Anthropic:
    """Create an Anthropic client from the ANTHROPIC_API_KEY env var."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Export it before running holdout generation."
        )
    return anthropic.Anthropic(api_key=api_key)


def call_claude(
    client: anthropic.Anthropic,
    system: str,
    prompt: str,
    model: str,
    max_tokens: int = 4096,
) -> tuple[str, anthropic.types.Message]:
    """Call Claude API and return (text, full_response)."""
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text
    return text, response


def track_cost(response: anthropic.types.Message, cost_entries: list) -> float:
    """Append a cost entry from response.usage. Returns the cost of this call."""
    usage = response.usage
    model = response.model
    pricing = _PRICING.get(model, _DEFAULT_PRICING)
    input_cost = (usage.input_tokens / 1_000_000) * pricing["input"]
    output_cost = (usage.output_tokens / 1_000_000) * pricing["output"]
    total = input_cost + output_cost

    cost_entries.append({
        "model": model,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": round(total, 6),
    })
    return total


def total_cost(cost_entries: list) -> float:
    """Sum all cost entries."""
    return sum(e["cost_usd"] for e in cost_entries)

logger = logging.getLogger("perian.autoresearch.holdout")
HOLDOUT_DIR = Path(__file__).resolve().parent / "holdout"

# ---------------------------------------------------------------------------
# Normalization for Claude's synthetic data output
# ---------------------------------------------------------------------------

SENIORITY_NORMALIZE = {
    "head": "principal", "head of": "principal",
    "director": "principal", "vp": "executive",
    "vice president": "executive", "chief": "executive",
    "manager": "senior", "senior manager": "staff",
    "lead": "staff", "coordinator": "junior",
    "analyst": "junior", "associate": "junior",
    "specialist": "mid", "intern": "junior",
    "entry-level": "junior", "entry level": "junior",
}

REMOTE_NORMALIZE = {
    "fully remote": "remote", "full remote": "remote",
    "100% remote": "remote", "work from home": "remote",
    "in-office": "onsite", "in office": "onsite",
    "on-site": "onsite", "on site": "onsite",
    "in-person": "onsite",
    "not specified": "unknown",
    "flexible": "hybrid",
}

VALID_SENIORITY = {"junior", "mid", "senior", "staff", "principal", "executive"}
VALID_REMOTE = {"remote", "hybrid", "onsite", "unknown"}


def _repair_truncated_json(text: str) -> str:
    """Attempt to close a truncated JSON array so it can be parsed.

    If Claude hit max_tokens mid-array, the text starts with '[' but
    lacks the closing ']'. We find the last complete object and close
    the array there.
    """
    text = text.strip()
    if text.startswith("[") and not text.rstrip().endswith("]"):
        last_brace = text.rfind("}")
        if last_brace > 0:
            text = text[: last_brace + 1] + "\n]"
            logger.info("Repaired truncated JSON array")
    return text


def _normalize_enum(value: str, valid: set, mapping: dict) -> str:
    """Normalize a string value to a valid enum, or return it unchanged."""
    if not isinstance(value, str):
        return value
    low = value.lower().strip()
    if low in valid:
        return low
    return mapping.get(low, value)


def _normalize_holdout_items(items: list[dict]) -> list[dict]:
    """Normalize enum values in Claude-generated holdout items."""
    fixed = 0
    for item in items:
        for section_key in ("expected_output", "input"):
            section = item.get(section_key, {})
            if not isinstance(section, dict):
                continue

            sen = section.get("seniority", "")
            new_sen = _normalize_enum(sen, VALID_SENIORITY, SENIORITY_NORMALIZE)
            if new_sen != sen:
                section["seniority"] = new_sen
                fixed += 1

            rp = section.get("remote_policy", "")
            new_rp = _normalize_enum(rp, VALID_REMOTE, REMOTE_NORMALIZE)
            if new_rp != rp:
                section["remote_policy"] = new_rp
                fixed += 1

    if fixed:
        logger.info(f"Normalized {fixed} enum values in holdout items")
    return items

# ---------------------------------------------------------------------------
# Extraction holdout generation
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = """\
You are a synthetic data generator for testing a job listing extraction system.
Generate realistic job listings with ground truth extracted fields.

Each listing should be a JSON object with:
- "input": the raw job listing text (realistic, varied lengths and formats)
- "expected_output": the correct extraction result with fields: title, company, seniority, remote_policy, domain, key_requirements, compensation
- "tags": list of tags describing what this tests (e.g., "ambiguous_seniority", "missing_compensation")
- "description": one-line description of what makes this case interesting

Rules for ground truth:
- seniority: one of junior, mid, senior, staff, principal, executive
  IC track: Coordinator/Analyst/Associate=junior, Specialist/individual contributor 2-4yrs=mid, Senior [Role]/5+years=senior, Staff/Lead=staff, Principal=principal
  Management track: Manager=senior, Senior Manager=staff, Director/Head of=principal, VP/Chief=executive
  Special: Product Manager defaults to senior (ownership scope). Lead without Manager=staff.
  Fantasy title normalization: Guru→Manager, Ninja→Specialist, Rockstar→Developer
- remote_policy: one of remote, hybrid, onsite, unknown
- key_requirements: array of 3-8 specific technologies or skills
- compensation: object with min (int|null), max (int|null), currency (string|null), equity_mentioned (bool)
  Normalize hourly to annual (hourly × 2080). Convert EUR/GBP to USD estimate.
- domain: concise label like "SaaS", "AI/ML", "B2B technology", "developer tools", "martech", "fintech", "healthcare"

Return a JSON array of listings. No markdown fences."""


def generate_extract_items(
    client, model: str, cost_entries: list, n: int = 35
) -> list[dict]:
    """Generate synthetic extraction test items via Claude API."""
    items = []
    batch_size = 10
    batches_needed = (n + batch_size - 1) // batch_size

    edge_case_batches = [
        "Generate 10 Customer Experience and Product job listings that test EDGE CASES: ambiguous seniority (CX Manager vs Senior CX Manager), missing compensation, vague remote policy, fantasy titles (Customer Happiness Guru, CX Ninja), management-track seniority mapping (Manager=senior, Director=principal, VP=executive). Include Product Manager (defaults to senior).",
        "Generate 10 Customer Experience and Product job listings that test VARIETY: SaaS, AI/ML, B2B technology, developer tools, martech, fintech, healthcare CX roles. Include listings with equity but no base salary, listings in EUR/GBP, listings with hourly rates that need annual normalization (e.g., $65/hr CX consultant), and listings with long requirement lists (15+ items to test the 8-item cap).",
        "Generate 10 Customer Experience and Product job listings that test TRICKY CASES: Lead titles (CX Lead = staff or senior?), Head of vs Director (both principal), PM default-to-senior, hybrid with specific days (2d vs 4d/week), contract-to-hire CX roles, listings where seniority must be inferred from years of experience, and compensation embedded in prose.",
        "Generate 5 Customer Experience and Product job listings that test RARE CASES: VP/executive level CX roles, CX roles at AI companies (domain ambiguity: is it AI-native or the company's primary domain?), multi-location listings, listings entirely in bullet points, and listings where the company name appears only in the URL.",
    ]

    for i, prompt in enumerate(edge_case_batches[:batches_needed]):
        logger.info(f"Generating extract batch {i+1}/{batches_needed}...")
        text, response = call_claude(
            client, EXTRACT_SYSTEM, prompt, model, max_tokens=8192,
        )
        track_cost(response, cost_entries)

        text = _repair_truncated_json(text)
        parsed = parse_json_response(text)
        if parsed is None:
            logger.error(f"Failed to parse batch {i+1} response")
            continue

        if isinstance(parsed, list):
            items.extend(parsed)
        elif isinstance(parsed, dict) and "listings" in parsed:
            items.extend(parsed["listings"])
        else:
            logger.warning(f"Unexpected batch {i+1} format: {type(parsed)}")

    items = _normalize_holdout_items(items)

    # Add IDs
    for idx, item in enumerate(items):
        item["id"] = f"extract_{idx+1:03d}"

    logger.info(f"Generated {len(items)} extraction items")
    return items[:n]


# ---------------------------------------------------------------------------
# Score holdout generation
# ---------------------------------------------------------------------------

SCORE_SYSTEM_TEMPLATE = """\
You are a synthetic data generator for testing a job listing scoring system.
Generate pre-extracted job listing JSON objects with ground truth scores.

The scoring system evaluates listings against this candidate profile:
{profile}

Scoring dimensions and weights:
{scoring}

Each item should be a JSON object with:
- "input": a pre-extracted job listing (the JSON object that would come out of the extraction step, with fields: title, company, seniority, remote_policy, domain, key_requirements, compensation)
- "expected_output": the correct scoring result with:
  - "scores": object with title_match, domain_match, seniority_match, requirements_fit, compensation_fit — each having "score" (0.0-1.0), "tolerance" (0.10-0.20), and "reasoning"
  - "total_score_range": [min, max] expected range for total_score
  - "expected_bonus": boolean (should a bonus be applied?)
  - "expected_penalty": boolean (should a penalty be applied?)
- "tags": list of tags
- "description": one-line description

Be precise with scores. Use the dimension definitions and weights to calculate expected weighted totals.

Return a JSON array. No markdown fences."""


def generate_score_items(
    client, model: str, cost_entries: list, plan: dict, profile: str, n: int = 25
) -> list[dict]:
    """Generate synthetic scoring test items via Claude API."""
    scoring_text = json.dumps(plan["scoring"], indent=2)
    system = SCORE_SYSTEM_TEMPLATE.format(profile=profile, scoring=scoring_text)

    items = []
    batch_size = 10
    batches_needed = (n + batch_size - 1) // batch_size

    score_batches = [
        "Generate 10 pre-extracted Customer Experience and Product job listings that span the FULL SCORE RANGE against this profile (targets Customer Experience Manager, Customer Success Manager, Product Manager, Customer Operations Lead, Director of Customer Experience; domains: SaaS, B2B technology, AI/ML, developer tools; remote preferred): 2 high-match Customer Experience and Product roles (total_score > 0.8), 3 medium-match adjacent roles (0.4-0.7), 3 low-match engineering/non-CX roles (< 0.3), and 2 that should trigger bonuses (remote) or penalties (onsite + large enterprise).",
        "Generate 10 pre-extracted Customer Experience and Product job listings that test SCORING EDGE CASES: unknown compensation (score at 0.5), seniority one level off from target (mid or executive, score 0.5), adjacent domains like martech or digital transformation (0.3-0.7), listings where CX title matches but requirements are engineering, and listings where title doesn't match (e.g., Operations Manager) but CX requirements are strong.",
        "Generate 5 pre-extracted Customer Experience and Product job listings that test BONUS/PENALTY LOGIC: remote CX Manager roles at SaaS companies (bonus), onsite large-enterprise CX roles (penalty), junior + onsite (stacked penalties), and a remote role at a non-target domain (remote bonus but domain penalty).",
    ]

    for i, prompt in enumerate(score_batches[:batches_needed]):
        logger.info(f"Generating score batch {i+1}/{batches_needed}...")
        text, response = call_claude(
            client, system, prompt, model, max_tokens=8192,
        )
        track_cost(response, cost_entries)

        text = _repair_truncated_json(text)
        parsed = parse_json_response(text)
        if parsed is None:
            logger.error(f"Failed to parse batch {i+1} response")
            continue

        if isinstance(parsed, list):
            items.extend(parsed)
        elif isinstance(parsed, dict) and "listings" in parsed:
            items.extend(parsed["listings"])
        else:
            logger.warning(f"Unexpected batch {i+1} format: {type(parsed)}")

    items = _normalize_holdout_items(items)

    for idx, item in enumerate(items):
        item["id"] = f"score_{idx+1:03d}"

    logger.info(f"Generated {len(items)} scoring items")
    return items[:n]


# ---------------------------------------------------------------------------
# Baseline execution
# ---------------------------------------------------------------------------


def run_baseline(
    items: list[dict],
    template_type: str,
    model: str,
    config: dict,
    ar_config: dict,
    plan: dict = None,
    profile: str = None,
) -> list[dict]:
    """Run each holdout item through the current prompt template via Ollama.

    Adds baseline_output, baseline_correct, and failure_type to each item.
    """
    template_text = load_prompt_template(template_type)
    n_runs = ar_config["loop"].get("runs_per_item", 3)

    for i, item in enumerate(items):
        logger.info(
            f"  Baseline [{i+1}/{len(items)}] {item.get('id', '?')}..."
        )

        prompt = render_prompt(
            template_text, item["input"], template_type,
            plan=plan, profile=profile,
        )

        runs = run_n_times(prompt, model, config, n=n_runs)

        # Take best run (most fields correct for extract, closest score for score)
        best_run = None
        best_score = -1

        for r in runs:
            if not r["success"] or not r["parsed"]:
                continue

            if template_type == "extract":
                score = _quick_extract_score(r["parsed"], item.get("expected_output", {}))
            else:
                score = _quick_score_score(r["parsed"], item.get("expected_output", {}))

            if score > best_score:
                best_score = score
                best_run = r

        if best_run:
            item["baseline_output"] = best_run["parsed"]
            item["baseline_correct"] = best_score > 0.7
            item["baseline_score"] = round(best_score, 4)
            item["failure_type"] = None if best_score > 0.7 else _classify_failure(
                best_run["parsed"], item.get("expected_output", {}), template_type
            )
        else:
            item["baseline_output"] = None
            item["baseline_correct"] = False
            item["baseline_score"] = 0.0
            item["failure_type"] = "parse_failure"

    return items


def _quick_extract_score(output: dict, expected: dict) -> float:
    """Quick accuracy score for extraction output vs ground truth."""
    if not expected:
        return 0.5

    matches = 0
    total = 0

    for field in ["title", "company", "seniority", "remote_policy", "domain"]:
        total += 1
        if output.get(field) == expected.get(field):
            matches += 1

    # Key requirements — Jaccard
    total += 1
    actual_reqs = {s.lower() for s in output.get("key_requirements", [])}
    expected_reqs = {s.lower() for s in expected.get("key_requirements", [])}
    if actual_reqs or expected_reqs:
        jaccard = len(actual_reqs & expected_reqs) / len(actual_reqs | expected_reqs)
        matches += jaccard
    else:
        matches += 1

    # Compensation
    total += 1
    actual_comp = output.get("compensation", {}) or {}
    expected_comp = expected.get("compensation", {}) or {}
    comp_fields = ["min", "max", "currency", "equity_mentioned"]
    if expected_comp:
        comp_match = sum(
            1 for f in comp_fields if actual_comp.get(f) == expected_comp.get(f)
        ) / len(comp_fields)
        matches += comp_match
    else:
        matches += 0.5

    return matches / total if total else 0.0


def _quick_score_score(output: dict, expected: dict) -> float:
    """Quick accuracy score for scoring output vs ground truth."""
    if not expected:
        return 0.5

    dims = ["title_match", "domain_match", "seniority_match",
            "requirements_fit", "compensation_fit"]

    exp_scores = expected.get("scores", {})
    out_scores = output.get("scores", {})

    total_deviation = 0
    counted = 0

    for dim in dims:
        exp = exp_scores.get(dim, {})
        exp_val = exp.get("score", 0.5) if isinstance(exp, dict) else 0.5
        tolerance = exp.get("tolerance", 0.15) if isinstance(exp, dict) else 0.15

        out_dim = out_scores.get(dim, {})
        out_val = out_dim.get("score") if isinstance(out_dim, dict) else None

        if out_val is not None:
            deviation = abs(out_val - exp_val)
            total_deviation += deviation
            counted += 1
        else:
            total_deviation += 1.0
            counted += 1

    # Total score range check
    total_score = output.get("total_score")
    score_range = expected.get("total_score_range", [0.0, 1.0])
    in_range = (
        total_score is not None
        and score_range[0] <= total_score <= score_range[1]
    )

    mean_dev = total_deviation / counted if counted else 1.0
    accuracy = max(0, 1.0 - 2 * mean_dev)

    if in_range:
        accuracy = min(1.0, accuracy + 0.1)

    return accuracy


def _classify_failure(output: dict, expected: dict, template_type: str) -> str:
    """Classify the type of failure for debugging."""
    if not output:
        return "empty_output"

    if template_type == "extract":
        from evaluation.helpers import EXTRACT_REQUIRED_FIELDS, VALID_SENIORITY, VALID_REMOTE

        missing = [f for f in EXTRACT_REQUIRED_FIELDS if f not in output]
        if missing:
            return f"missing_fields:{','.join(missing)}"

        if output.get("seniority") not in VALID_SENIORITY:
            return "invalid_seniority"
        if output.get("remote_policy") not in VALID_REMOTE:
            return "invalid_remote_policy"

        return "field_accuracy"

    elif template_type == "score":
        scores = output.get("scores", {})
        dims = ["title_match", "domain_match", "seniority_match",
                "requirements_fit", "compensation_fit"]
        missing = [d for d in dims if d not in scores]
        if missing:
            return f"missing_dimensions:{','.join(missing)}"

        if output.get("total_score") is None:
            return "missing_total_score"

        return "score_deviation"

    return "unknown"


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_holdout(items: list[dict], template_type: str) -> Path:
    """Save holdout data to JSON file."""
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    path = HOLDOUT_DIR / f"{template_type}_holdout.json"
    with open(path, "w") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
        f.write("\n")
    logger.info(f"Saved {len(items)} items to {path}")
    return path


def save_cost_log(cost_entries: list) -> Path:
    """Save cost log to JSON file."""
    path = Path(__file__).resolve().parent / "cost_log.json"

    # Append to existing log if present
    existing = []
    if path.exists():
        with open(path) as f:
            existing = json.load(f)

    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "operation": "generate_holdout",
        "calls": cost_entries,
        "total_cost_usd": round(total_cost(cost_entries), 4),
    }
    existing.append(entry)

    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic holdout data for autoresearch"
    )
    parser.add_argument(
        "--template",
        choices=["extract", "score", "both"],
        required=True,
        help="Which template to generate holdout data for",
    )
    parser.add_argument(
        "--n-extract", type=int, default=35,
        help="Number of extraction items to generate (default: 35)",
    )
    parser.add_argument(
        "--n-score", type=int, default=25,
        help="Number of score items to generate (default: 25)",
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip running baseline through Ollama (just generate via Claude)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    ar_config = load_autoresearch_config()
    cloud_model = os.environ.get("HOLDOUT_CLOUD_MODEL", "claude-sonnet-4-20250514")
    local_model = ar_config["loop"]["local_model"]

    client = get_claude_client()
    config = load_config()
    plan = load_plan()
    profile = load_profile()
    cost_entries = []

    templates = []
    if args.template in ("extract", "both"):
        templates.append("extract")
    if args.template in ("score", "both"):
        templates.append("score")

    for template_type in templates:
        print(f"\n{'='*50}")
        print(f"  Generating {template_type} holdout data")
        print(f"{'='*50}\n")

        # Generate synthetic data via Claude
        if template_type == "extract":
            items = generate_extract_items(
                client, cloud_model, cost_entries, n=args.n_extract
            )
        else:
            items = generate_score_items(
                client, cloud_model, cost_entries, plan, profile, n=args.n_score
            )

        if not items:
            print(f"  ERROR: No items generated for {template_type}")
            continue

        print(f"  Generated {len(items)} synthetic items")

        # Run baseline through Ollama
        if not args.skip_baseline:
            print(f"  Running baseline through {local_model}...")
            items = run_baseline(
                items, template_type, local_model, config, ar_config,
                plan=plan, profile=profile,
            )
            correct = sum(1 for i in items if i.get("baseline_correct"))
            print(f"  Baseline: {correct}/{len(items)} correct")
        else:
            print("  Skipping baseline (--skip-baseline)")

        # Save
        path = save_holdout(items, template_type)
        print(f"  Saved to: {path}")

    # Save cost log
    cost_path = save_cost_log(cost_entries)
    print(f"\nTotal cost: ${total_cost(cost_entries):.4f}")
    print(f"Cost log: {cost_path}")


if __name__ == "__main__":
    main()
