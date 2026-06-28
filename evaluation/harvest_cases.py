#!/usr/bin/env python3
"""Harvest test cases from real pipeline data for eval suite expansion.

Reads extract_prompts.json from a run-log, generates candidate test cases
using qwen3.5:27b as teacher, and produces candidates for human review.

L1/L2 candidates get teacher-generated ground truth (consensus from N runs).
L3/L4 candidates produce shells with input populated but expected_output null
(manual ground truth needed).

Usage:
    # Generate L1/L2 with teacher
    python3 evaluation/harvest_cases.py --levels 1 2 --teacher qwen3.5:27b \\
        --source run-log/20260326-223656/extract_prompts.json \\
        --output evaluation/candidates/

    # Generate L3/L4 shells
    python3 evaluation/harvest_cases.py --levels 3 4 \\
        --source run-log/20260326-223656/extract_prompts.json \\
        --output evaluation/candidates/

    # Review candidates
    python3 evaluation/harvest_cases.py review \\
        --candidates evaluation/candidates/l1_candidates.json

    # Merge approved into test suite
    python3 evaluation/harvest_cases.py merge \\
        --candidates-dir evaluation/candidates/ \\
        --test-suite-dir evaluation/test_suite/

    # Limit candidate count (for testing)
    python3 evaluation/harvest_cases.py --levels 1 --teacher qwen3.5:27b \\
        --source run-log/20260326-223656/extract_prompts.json \\
        --output evaluation/candidates/ --max-candidates 5
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.helpers import (
    EXTRACT_REQUIRED_FIELDS,
    build_extract_prompt,
    build_score_prompt,
    call_ollama,
    load_config,
    parse_json_response,
)


# ---------------------------------------------------------------------------
# Source data loading
# ---------------------------------------------------------------------------

def load_source_data(source_path: str) -> list[dict]:
    """Parse extract_prompts.json and extract listing text per entry.

    Each entry's prompt contains the listing text after the
    "--- ITEM TO PROCESS ---" separator.
    """
    with open(source_path) as f:
        data = json.load(f)

    entries = []
    for item in data:
        prompt = item.get("prompt", "")
        if "--- ITEM TO PROCESS ---" not in prompt:
            continue

        parts = prompt.split("--- ITEM TO PROCESS ---", 1)
        listing_text = parts[1].strip()

        # Strip the instruction prefix if present
        if listing_text.startswith("Extract structured fields"):
            lines = listing_text.split("\n", 2)
            if len(lines) > 2:
                listing_text = lines[2].strip()
            else:
                listing_text = "\n".join(lines[1:]).strip()

        entries.append({
            "item_id": item.get("item_id", ""),
            "listing_text": listing_text,
            "model_response": item.get("raw_response", ""),
        })

    return entries


def _select_diverse(entries: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Select a diverse subset of entries prioritizing domain/title variety.

    Extracts domain signals from listing text (SaaS, AI, consulting, etc.)
    and samples to maximize coverage.
    """
    if len(entries) <= n:
        return entries

    rng = random.Random(seed)

    # Simple domain detection from listing text
    domain_keywords = {
        "saas": ["saas", "b2b", "subscription"],
        "ai_ml": ["ai", "machine learning", "ml", "artificial intelligence"],
        "consulting": ["consulting", "advisory", "strategy"],
        "fintech": ["fintech", "financial", "banking", "payments"],
        "healthcare": ["health", "medical", "pharma", "biotech"],
        "edtech": ["education", "edtech", "learning"],
        "ecommerce": ["ecommerce", "e-commerce", "retail", "marketplace"],
        "cybersecurity": ["security", "cyber", "infosec"],
    }

    # Assign domains to entries
    bucketed = {k: [] for k in domain_keywords}
    bucketed["other"] = []

    for entry in entries:
        text_lower = entry["listing_text"].lower()
        assigned = False
        for domain, keywords in domain_keywords.items():
            if any(kw in text_lower for kw in keywords):
                bucketed[domain].append(entry)
                assigned = True
                break
        if not assigned:
            bucketed["other"].append(entry)

    # Round-robin sample from each bucket
    selected = []
    buckets = [v for v in bucketed.values() if v]
    for bucket in buckets:
        rng.shuffle(bucket)

    idx = 0
    while len(selected) < n and any(buckets):
        for bucket in buckets:
            if idx < len(bucket) and len(selected) < n:
                selected.append(bucket[idx])
        idx += 1
        buckets = [b for b in buckets if idx < len(b)]

    return selected


# ---------------------------------------------------------------------------
# L1/L2 candidate generation (teacher-based ground truth)
# ---------------------------------------------------------------------------

def generate_l1_candidates(entries: list[dict], teacher_model: str,
                            config: dict, n_teacher_runs: int = 3,
                            max_candidates: int = None) -> list[dict]:
    """Generate L1 extraction test case candidates using a teacher model.

    Runs the teacher N times per entry and takes consensus as ground truth.
    """
    selected = _select_diverse(entries, max_candidates or 60)
    candidates = []

    for i, entry in enumerate(selected):
        print(f"  L1 candidate {i+1}/{len(selected)}: {entry['item_id'][:30]}...", flush=True)
        prompt = build_extract_prompt(entry["listing_text"])

        # Run teacher N times
        results = []
        for _ in range(n_teacher_runs):
            resp = call_ollama(prompt, teacher_model, config)
            if resp:
                parsed = parse_json_response(resp.get("response", ""))
                if parsed:
                    results.append(parsed)

        if not results:
            continue

        # Take consensus: for each field, use the most common value
        consensus = _build_consensus(results)

        # Extract a concise description from the listing
        title = consensus.get("title", "Unknown")
        company = consensus.get("company", "Unknown")
        desc = f"{title} at {company}"

        candidates.append({
            "id": f"harvest_l1_{i+1:03d}",
            "description": desc,
            "input": entry["listing_text"],
            "expected_output": consensus,
            "tags": ["harvested", "teacher_generated"],
            "notes": f"Teacher: {teacher_model}, {n_teacher_runs} runs. Source: {entry['item_id']}",
            "synthetic": False,
            "review_status": "pending",
            "teacher_agreement": _compute_agreement(results, consensus),
        })

    return candidates


def generate_l2_candidates(entries: list[dict], teacher_model: str,
                            config: dict, max_candidates: int = None) -> list[dict]:
    """Generate L2 classification test case candidates.

    Same as L1 but only scores classification fields.
    """
    selected = _select_diverse(entries, max_candidates or 60)
    candidates = []

    for i, entry in enumerate(selected):
        print(f"  L2 candidate {i+1}/{len(selected)}: {entry['item_id'][:30]}...", flush=True)
        prompt = build_extract_prompt(entry["listing_text"])

        results = []
        for _ in range(3):
            resp = call_ollama(prompt, teacher_model, config)
            if resp:
                parsed = parse_json_response(resp.get("response", ""))
                if parsed:
                    results.append(parsed)

        if not results:
            continue

        consensus = _build_consensus(results)

        title = consensus.get("title", "Unknown")
        company = consensus.get("company", "Unknown")

        # L2 expected output is just classification fields
        expected = {
            f: consensus.get(f) for f in ["seniority", "domain", "remote_policy"]
        }
        # Include acceptable alternatives from teacher disagreements
        alternatives = {}
        for field in ["seniority", "domain", "remote_policy"]:
            values = set(r.get(field) for r in results if r.get(field))
            if len(values) > 1:
                alternatives[field] = list(values - {consensus.get(field)})

        candidates.append({
            "id": f"harvest_l2_{i+1:03d}",
            "description": f"Classification: {title} at {company}",
            "input": entry["listing_text"],
            "expected_output": expected,
            "acceptable_alternatives": alternatives,
            "tags": ["harvested", "classification_edge"],
            "notes": f"Teacher: {teacher_model}. Source: {entry['item_id']}",
            "synthetic": False,
            "review_status": "pending",
        })

    return candidates


# ---------------------------------------------------------------------------
# L3/L4 candidate generation (shells — manual ground truth needed)
# ---------------------------------------------------------------------------

def generate_l3_candidates(entries: list[dict],
                            max_candidates: int = None) -> list[dict]:
    """Generate L3 scoring test case shells.

    Produces input (extracted JSON) but expected_output is null —
    ground truth scores need manual annotation.
    """
    selected = _select_diverse(entries, max_candidates or 60)
    candidates = []

    for i, entry in enumerate(selected):
        # Try to parse the model's original extraction response
        parsed = parse_json_response(entry.get("model_response", ""))
        if not parsed:
            continue

        title = parsed.get("title", "Unknown")
        company = parsed.get("company", "Unknown")

        candidates.append({
            "id": f"harvest_l3_{i+1:03d}",
            "description": f"Score: {title} at {company}",
            "input": parsed,
            "expected_output": None,
            "tags": ["harvested", "needs_ground_truth"],
            "notes": f"Shell only — manual scoring ground truth needed. Source: {entry['item_id']}",
            "review_status": "needs_ground_truth",
        })

    return candidates


def generate_l4_candidates(entries: list[dict],
                            max_candidates: int = None) -> list[dict]:
    """Generate L4 causal reasoning test case shells.

    Produces input (listing + profile) but expected_conclusion is null.
    """
    selected = _select_diverse(entries, max_candidates or 60)
    candidates = []

    for i, entry in enumerate(selected):
        parsed = parse_json_response(entry.get("model_response", ""))
        if not parsed:
            continue

        title = parsed.get("title", "Unknown")
        company = parsed.get("company", "Unknown")

        candidates.append({
            "id": f"harvest_l4_{i+1:03d}",
            "description": f"Causal: {title} at {company}",
            "input": {
                "profile_summary": None,
                "listing": parsed,
            },
            "expected_conclusion": None,
            "expected_reasoning_mentions": [],
            "prompt_context": "",
            "tags": ["harvested", "needs_ground_truth"],
            "notes": f"Shell only — manual conclusion + reasoning needed. Source: {entry['item_id']}",
            "review_status": "needs_ground_truth",
        })

    return candidates


# ---------------------------------------------------------------------------
# Hard negatives
# ---------------------------------------------------------------------------

def generate_hard_negatives(entries: list[dict], level: int,
                             n: int = 20) -> list[dict]:
    """Generate edge-case test candidates per level.

    Produces cases designed to trip up models: garbled formatting,
    ambiguous titles, contradictory signals.
    """
    selected = _select_diverse(entries, min(n * 2, len(entries)))
    negatives = []

    for i, entry in enumerate(selected[:n]):
        listing = entry["listing_text"]
        case_id = f"harvest_neg_l{level}_{i+1:03d}"

        if level in (1, 2):
            # Garble formatting: remove colons, shuffle lines
            lines = listing.strip().splitlines()
            random.Random(i).shuffle(lines)
            garbled = "\n".join(lines)
            negatives.append({
                "id": case_id,
                "description": f"Hard negative L{level}: garbled format",
                "input": garbled,
                "expected_output": None,
                "tags": ["harvested", "hard_negative", "garbled"],
                "notes": "Shuffled lines to test format robustness.",
                "review_status": "needs_ground_truth",
            })
        elif level == 3:
            # Contradictory signals in extracted JSON
            parsed = parse_json_response(entry.get("model_response", ""))
            if parsed:
                # Inject contradictory seniority
                modified = dict(parsed)
                modified["seniority"] = "junior"  # Force mismatch with likely senior listing
                negatives.append({
                    "id": case_id,
                    "description": f"Hard negative L3: contradictory seniority",
                    "input": modified,
                    "expected_output": None,
                    "tags": ["harvested", "hard_negative", "contradictory"],
                    "notes": "Seniority forced to junior for senior-looking listing.",
                    "review_status": "needs_ground_truth",
                })
        elif level == 4:
            parsed = parse_json_response(entry.get("model_response", ""))
            if parsed:
                negatives.append({
                    "id": case_id,
                    "description": f"Hard negative L4: ambiguous match",
                    "input": {
                        "profile_summary": None,
                        "listing": parsed,
                    },
                    "expected_conclusion": None,
                    "tags": ["harvested", "hard_negative", "ambiguous"],
                    "notes": "Ambiguous edge case for causal reasoning.",
                    "review_status": "needs_ground_truth",
                })

    return negatives


# ---------------------------------------------------------------------------
# Review workflow
# ---------------------------------------------------------------------------

def review_candidates(candidates_path: str) -> None:
    """Interactive review: approve/edit/reject/skip per candidate.

    Saves progress incrementally so you can quit and resume.
    """
    path = Path(candidates_path)
    with open(path) as f:
        candidates = json.load(f)

    reviewed = 0
    total = len(candidates)

    for i, c in enumerate(candidates):
        if c.get("review_status") in ("approved", "rejected"):
            continue

        print(f"\n{'='*55}")
        print(f"  Candidate {i+1}/{total}: {c['id']}")
        print(f"  {c['description']}")
        print(f"  Tags: {', '.join(c.get('tags', []))}")
        print(f"  Notes: {c.get('notes', '')}")
        print(f"{'='*55}")

        if c.get("expected_output"):
            print(f"\n  Expected output:")
            print(json.dumps(c["expected_output"], indent=4)[:500])

        if isinstance(c.get("input"), str):
            print(f"\n  Input (first 200 chars):")
            print(f"  {c['input'][:200]}...")
        elif isinstance(c.get("input"), dict):
            print(f"\n  Input keys: {list(c['input'].keys())}")

        print(f"\n  [a]pprove  [r]eject  [s]kip  [q]uit")
        choice = input("  > ").strip().lower()

        if choice == "a":
            c["review_status"] = "approved"
            reviewed += 1
        elif choice == "r":
            c["review_status"] = "rejected"
            reviewed += 1
        elif choice == "q":
            break
        # else: skip, leave status unchanged

        # Save progress incrementally
        with open(path, "w") as f:
            json.dump(candidates, f, indent=2, ensure_ascii=False)
            f.write("\n")

    print(f"\nReviewed {reviewed} candidates. Progress saved to {path}")


# ---------------------------------------------------------------------------
# Merge approved candidates into test suite
# ---------------------------------------------------------------------------

def merge_approved(candidates_dir: str, test_suite_dir: str) -> dict:
    """Merge approved candidates into test suite JSON files.

    Assigns sequential IDs continuing from existing test cases.
    """
    candidates_path = Path(candidates_dir)
    suite_path = Path(test_suite_dir)
    stats = {}

    level_map = {
        "l1": ("level_1_extraction", "extract"),
        "l2": ("level_2_classification", "class"),
        "l3": ("level_3_conditional", "score"),
        "l4": ("level_4_causal", "causal"),
    }

    for prefix, (level_dir, id_prefix) in level_map.items():
        candidates_file = candidates_path / f"{prefix}_candidates.json"
        if not candidates_file.exists():
            continue

        with open(candidates_file) as f:
            candidates = json.load(f)

        approved = [c for c in candidates if c.get("review_status") == "approved"]
        if not approved:
            stats[prefix] = 0
            continue

        # Load existing test cases
        tc_file = suite_path / level_dir / "test_cases.json"
        with open(tc_file) as f:
            existing = json.load(f)

        # Find max existing ID number
        max_num = 0
        for tc in existing if isinstance(existing, list) else existing.get("groups", []):
            tc_id = tc.get("id", "")
            parts = tc_id.split("_")
            try:
                num = int(parts[-1])
                max_num = max(max_num, num)
            except (ValueError, IndexError):
                pass

        # Assign new IDs and add
        for i, candidate in enumerate(approved):
            new_num = max_num + i + 1
            candidate["id"] = f"{id_prefix}_{new_num:02d}"
            # Remove review metadata
            candidate.pop("review_status", None)
            candidate.pop("teacher_agreement", None)

            if isinstance(existing, list):
                existing.append(candidate)

        with open(tc_file, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
            f.write("\n")

        stats[prefix] = len(approved)

    return stats


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _build_consensus(results: list[dict]) -> dict:
    """Build consensus from multiple teacher runs.

    For each field, takes the most common value across runs.
    """
    if not results:
        return {}

    consensus = {}
    all_fields = set()
    for r in results:
        all_fields.update(r.keys())

    for field in all_fields:
        values = [r.get(field) for r in results if field in r]
        if not values:
            continue

        # For list/dict fields, take the longest (most complete)
        if isinstance(values[0], list):
            consensus[field] = max(values, key=len)
        elif isinstance(values[0], dict):
            consensus[field] = max(values, key=lambda d: len(str(d)))
        else:
            # Most common value
            counts = Counter(str(v) for v in values)
            most_common_str = counts.most_common(1)[0][0]
            # Find original typed value
            for v in values:
                if str(v) == most_common_str:
                    consensus[field] = v
                    break

    return consensus


def _compute_agreement(results: list[dict], consensus: dict) -> float:
    """Compute agreement rate between teacher runs and consensus."""
    if not results:
        return 0.0

    fields = [f for f in EXTRACT_REQUIRED_FIELDS if f in consensus]
    if not fields:
        return 0.0

    total_agree = 0
    total_checks = 0
    for r in results:
        for field in fields:
            if field in r:
                if str(r[field]) == str(consensus.get(field)):
                    total_agree += 1
                total_checks += 1

    return round(total_agree / total_checks, 4) if total_checks else 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Harvest test cases from real pipeline data"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Default: generate candidates
    parser.add_argument("--levels", type=int, nargs="+", help="Levels to generate (1-4)")
    parser.add_argument("--teacher", type=str, default="qwen3.5:27b", help="Teacher model for L1/L2")
    parser.add_argument("--source", type=str, help="Path to extract_prompts.json")
    parser.add_argument("--output", type=str, help="Output directory for candidates")
    parser.add_argument("--max-candidates", type=int, default=None, help="Max candidates per level")
    parser.add_argument("--n-teacher-runs", type=int, default=3, help="Teacher runs for consensus")

    # Review subcommand
    review_parser = subparsers.add_parser("review", help="Review candidates interactively")
    review_parser.add_argument("--candidates", required=True, help="Candidates JSON file")

    # Merge subcommand
    merge_parser = subparsers.add_parser("merge", help="Merge approved candidates into test suite")
    merge_parser.add_argument("--candidates-dir", required=True, help="Directory with candidate files")
    merge_parser.add_argument("--test-suite-dir", required=True, help="Test suite directory")

    args = parser.parse_args()

    if args.command == "review":
        review_candidates(args.candidates)
        return

    if args.command == "merge":
        stats = merge_approved(args.candidates_dir, args.test_suite_dir)
        print(f"Merged: {stats}")
        return

    # Generate candidates
    if not args.levels or not args.source:
        parser.error("--levels and --source required for candidate generation")

    entries = load_source_data(args.source)
    print(f"Loaded {len(entries)} entries from {args.source}")

    config = load_config()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for level in args.levels:
        print(f"\n--- Generating L{level} candidates ---")

        if level == 1:
            candidates = generate_l1_candidates(
                entries, args.teacher, config,
                n_teacher_runs=args.n_teacher_runs,
                max_candidates=args.max_candidates,
            )
        elif level == 2:
            candidates = generate_l2_candidates(
                entries, args.teacher, config,
                max_candidates=args.max_candidates,
            )
        elif level == 3:
            candidates = generate_l3_candidates(
                entries, max_candidates=args.max_candidates,
            )
        elif level == 4:
            candidates = generate_l4_candidates(
                entries, max_candidates=args.max_candidates,
            )
        else:
            print(f"  Skipping unsupported level {level}")
            continue

        out_file = output_dir / f"l{level}_candidates.json"
        with open(out_file, "w") as f:
            json.dump(candidates, f, indent=2, ensure_ascii=False)
            f.write("\n")

        print(f"  Generated {len(candidates)} candidates → {out_file}")


if __name__ == "__main__":
    main()
