#!/usr/bin/env python3
"""Perian — feedback collection + Aragorn review workflow.

Usage:
    python3 scripts/feedback.py pending [--json]             # Show unreviewed Aragorn disagreements
    python3 scripts/feedback.py review [--limit N] [--all] [--item ID]  # Interactive review
    python3 scripts/feedback.py record-verdict --item ID --agree dim1 --disagree dim2  # Non-interactive
    python3 scripts/feedback.py job-search --item ID --rating good|bad|corrected [--correction "..."]

Subcommands:
    pending        — Show count and table of unreviewed Aragorn disagreements
    review         — Walk through each disagreement interactively (a/d/s/q)
    record-verdict — Non-interactive verdict recording (used by in-chat review)
    job-search     — Legacy feedback recording (backward compatible)
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
FEEDBACK_PATH = STATE_DIR / "feedback.json"
ARAGORN_PATH = STATE_DIR / "aragorn-challenges.json"
RUN_LOG_DIR = ROOT / "run-log"
OUTPUT_DIR = ROOT / "output"

# Standard scoring weights (must match prompts/score.md)
SCORE_WEIGHTS = {
    "title_match": 0.25,
    "domain_match": 0.20,
    "seniority_match": 0.20,
    "requirements_fit": 0.20,
    "compensation_fit": 0.15,
}

AGENT_NAMES = {"job-search"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> list | dict:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def parse_json_response(text: str) -> dict | None:
    """Extract JSON from an LLM response, handling common quirks."""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass
    return None


def find_latest_run_log() -> Path | None:
    """Find the most recent run-log directory that has prompt logs."""
    run_dirs = sorted(
        [d for d in RUN_LOG_DIR.iterdir() if d.is_dir() and d.name != ".gitkeep"],
        reverse=True,
    )
    for d in run_dirs:
        if (d / "extract_prompts.json").exists() or (d / "score_prompts.json").exists():
            return d
    return run_dirs[0] if run_dirs else None


def find_run_log_by_name(name: str) -> Path | None:
    """Find a specific run-log directory by name."""
    p = RUN_LOG_DIR / name
    return p if p.is_dir() else None


def lookup_item_in_log(item_id: str, run_log_dir: Path) -> dict:
    """Find prompt and response for an item across extract and score logs."""
    result = {"extract": None, "score": None}
    for step_id in ("extract", "score"):
        log_file = run_log_dir / f"{step_id}_prompts.json"
        if not log_file.exists():
            continue
        entries = load_json(log_file)
        for entry in entries:
            if entry.get("item_id") == item_id:
                result[step_id] = entry
                break
    return result


# ---------------------------------------------------------------------------
# Pending subcommand
# ---------------------------------------------------------------------------
def get_pending_items() -> list[dict]:
    """Return unreviewed Aragorn disagreements not already in feedback."""
    challenges = load_json(ARAGORN_PATH)
    if not challenges or not isinstance(challenges, dict):
        return []

    feedback = load_json(FEEDBACK_PATH)
    feedback_ids = set()
    if isinstance(feedback, list):
        feedback_ids = {e.get("item_id") for e in feedback if e.get("step_id") == "score"}

    pending = []
    for item in challenges.get("items", []):
        if not item.get("is_disagreement"):
            continue
        if item.get("reviewed"):
            continue
        if item.get("triaged_out"):
            continue
        if item.get("item_id") in feedback_ids:
            continue
        pending.append(item)
    return pending


def cmd_pending(use_json: bool = False):
    """Show unreviewed Aragorn disagreements."""
    challenges = load_json(ARAGORN_PATH)
    if not challenges or not isinstance(challenges, dict):
        if use_json:
            print(json.dumps({"pending_count": 0, "total_disagreements": 0, "items": []}))
        else:
            print("No Aragorn challenges found. Run the pipeline + aragorn_challenge.py first.")
        return

    all_items = challenges.get("items", [])
    disagreements = [i for i in all_items if i.get("is_disagreement")]
    pending = get_pending_items()

    if use_json:
        out = {
            "pending_count": len(pending),
            "total_disagreements": len(disagreements),
            "items": [
                {
                    "item_id": item.get("item_id"),
                    "title": item.get("title", "?"),
                    "company": item.get("company", "?"),
                    "original_score": item.get("original_score", 0),
                    "aragorn_score": item.get("aragorn_adjusted_score", 0),
                    "delta": item.get("delta", 0),
                    "challenged_dimensions": [
                        c.get("dimension") for c in item.get("challenges", [])
                    ],
                    "overall_assessment": item.get("overall_assessment", ""),
                }
                for item in pending
            ],
        }
        print(json.dumps(out, indent=2))
        return

    if not pending:
        reviewed = sum(1 for i in disagreements if i.get("reviewed"))
        print(f"No pending reviews. {reviewed}/{len(disagreements)} disagreements reviewed.")
        return

    print(f"\nPending reviews: {len(pending)} of {len(disagreements)} Aragorn disagreements\n")
    print(f"  {'#':>3}  {'Item ID':<14} {'Title':<42} {'Orig':>5}  {'Aragorn':>7}  {'Delta':>5}")
    print(f"  {'─'*3}  {'─'*14} {'─'*42} {'─'*5}  {'─'*7}  {'─'*5}")

    for i, item in enumerate(pending, 1):
        title = item.get("title", "?")
        if len(title) > 40:
            title = title[:37] + "..."
        print(
            f"  {i:>3}  {item['item_id']:<14} {title:<42} "
            f"{item.get('original_score', 0):>5.3f}  "
            f"{item.get('aragorn_adjusted_score', 0):>7.3f}  "
            f"{item.get('delta', 0):>5.3f}"
        )

    print(f"\nRun: python3 scripts/feedback.py review")


# ---------------------------------------------------------------------------
# Review subcommand
# ---------------------------------------------------------------------------
def build_corrected_scoring(item: dict, run_log_dir: Path | None,
                            dimension_verdicts: dict | None = None) -> str | None:
    """Build corrected scoring JSON by applying Aragorn's suggested scores.

    Looks up the original score_prompts.json entry, parses the raw response,
    replaces challenged dimensions with Aragorn's suggested scores, and
    recalculates weighted_total and total_score.

    If dimension_verdicts is provided, only applies corrections for dimensions
    where the user agreed with Aragorn (verdict == "agree").
    """
    if not run_log_dir:
        return None

    log_data = lookup_item_in_log(item["item_id"], run_log_dir)
    score_entry = log_data.get("score")
    if not score_entry:
        return None

    scoring = parse_json_response(score_entry.get("raw_response", ""))
    if not scoring or "scores" not in scoring:
        return None

    # Apply Aragorn's dimension overrides (only for agreed dimensions)
    for challenge in item.get("challenges", []):
        dim = challenge.get("dimension")
        suggested = challenge.get("suggested_score")
        # Skip dimensions the user disagreed with or skipped
        if dimension_verdicts and dim in dimension_verdicts:
            if dimension_verdicts[dim].get("verdict") != "agree":
                continue
        if dim and suggested is not None and dim in scoring["scores"]:
            scoring["scores"][dim]["score"] = suggested
            scoring["scores"][dim]["reasoning"] += f" [Aragorn corrected from {challenge.get('original_score', '?')}]"

    # Recalculate weighted_total
    weighted_total = 0.0
    for dim, weight in SCORE_WEIGHTS.items():
        dim_data = scoring["scores"].get(dim)
        if dim_data:
            weighted_total += dim_data["score"] * weight

    # Add bonuses/penalties
    bonuses = sum(b.get("bonus", 0) for b in scoring.get("bonuses_applied", []))
    penalties = sum(p.get("penalty", 0) for p in scoring.get("penalties_applied", []))

    scoring["weighted_total"] = round(weighted_total, 3)
    scoring["total_score"] = round(max(0.0, min(1.0, weighted_total + bonuses + penalties)), 3)

    return json.dumps(scoring, ensure_ascii=False)


def lookup_extracted_data(item_id: str, run_log_dir: Path | None) -> dict | None:
    """Look up the extracted listing data from the run-log."""
    if not run_log_dir:
        return None
    extract_path = run_log_dir / "extract_prompts.json"
    if not extract_path.exists():
        return None
    entries = load_json(extract_path)
    for e in entries:
        if e.get("item_id") == item_id:
            return parse_json_response(e.get("raw_response", ""))
    return None


def display_review_item(item: dict, index: int, total: int, extracted: dict | None = None):
    """Print a review card for one Aragorn disagreement."""
    print()
    print("═" * 70)
    title = item.get("title", "Unknown")
    company = item.get("company", "Unknown")
    orig = item.get("original_score", 0)
    adjusted = item.get("aragorn_adjusted_score", 0)
    delta = item.get("delta", 0)
    print(f"  Item {index} of {total}: {title}")
    print(f"  Company: {company} | Score: {orig:.3f} → Aragorn: {adjusted:.3f} (delta: {delta:.3f})")
    print("═" * 70)

    # Listing context from extraction
    if extracted:
        comp = extracted.get("compensation", {})
        comp_min = comp.get("min")
        comp_max = comp.get("max")
        if comp_min and comp_max:
            comp_str = f"${comp_min:,.0f}–${comp_max:,.0f} {comp.get('currency', '')}"
        elif comp_min or comp_max:
            comp_str = f"${(comp_min or comp_max):,.0f} {comp.get('currency', '')}"
        else:
            comp_str = "unknown"
        equity = " + equity" if comp.get("equity_mentioned") else ""

        seniority = extracted.get("seniority", "?")
        remote = extracted.get("remote_policy", "?")
        domain = extracted.get("domain", "?")
        company_size = extracted.get("company_size", "unknown")
        reqs = extracted.get("key_requirements", [])
        reqs_str = ", ".join(reqs[:6])
        if len(reqs) > 6:
            reqs_str += f" (+{len(reqs)-6})"

        print()
        size_flag = " ⚠ ENTERPRISE" if company_size == "enterprise" else ""
        print(f"  Listing: {seniority} | {remote} | {domain} | {company_size}{size_flag}")
        print(f"  Comp: {comp_str}{equity}  (your target: $125k–$175k)")
        if reqs_str:
            print(f"  Reqs: {reqs_str}")

    challenges = item.get("challenges", [])
    if challenges:
        print()
        print("  Aragorn's challenges:")
        for c in challenges:
            dim = c.get("dimension", "?")
            orig_s = c.get("original_score", "?")
            sugg_s = c.get("suggested_score", "?")
            issue = c.get("issue", "")
            # Format: dimension with scores, then wrapped issue text
            orig_str = f"{orig_s}" if isinstance(orig_s, (int, float)) else orig_s
            sugg_str = f"{sugg_s}" if isinstance(sugg_s, (int, float)) else sugg_s
            print(f"    {dim:<20} {orig_str} → {sugg_str}")
            # Wrap issue text
            if issue:
                words = issue.split()
                line = "      "
                for word in words:
                    if len(line) + len(word) + 1 > 68:
                        print(line)
                        line = "      " + word
                    else:
                        line += " " + word if line.strip() else "      " + word
                if line.strip():
                    print(line)

    overall = item.get("overall_assessment", "")
    if overall:
        print()
        print(f"  Overall: \"{overall}\"")

    print()
    print("  Review each challenged dimension, then overall verdict.")
    print("  [q] Quit at any time")


def record_review_verdict(item: dict, verdict: str, feedback: list,
                          corrected_output: str | None = None,
                          correction_text: str | None = None,
                          run_log_dir: Path | None = None,
                          review_notes: str | None = None,
                          dimension_verdicts: dict | None = None):
    """Record a review verdict to both feedback.json and aragorn-challenges.json."""
    now = datetime.now(timezone.utc).isoformat()

    # Update aragorn-challenges.json in-place
    challenges = load_json(ARAGORN_PATH)
    if isinstance(challenges, dict):
        for ci in challenges.get("items", []):
            if ci.get("item_id") == item["item_id"] and not ci.get("reviewed"):
                ci["reviewed"] = True
                ci["review_verdict"] = verdict
                ci["review_timestamp"] = now
                if review_notes:
                    ci["review_notes"] = review_notes
                if dimension_verdicts:
                    ci["dimension_verdicts"] = dimension_verdicts
                break
        save_json(ARAGORN_PATH, challenges)

    # Skip feedback recording for "skipped" verdicts
    if verdict == "skipped":
        return

    # Determine rating and corrected output
    if verdict == "agree_aragorn":
        rating = "corrected"
        output = corrected_output
    elif verdict == "agree_pipeline":
        rating = "good"
        output = None
    elif verdict == "partial":
        rating = "partial"
        output = corrected_output
    elif verdict == "corrected":
        rating = "corrected"
        output = correction_text
    else:
        return

    # Look up the original prompt from run-log
    prompt = ""
    model_output = ""
    model_used = "qwen3.5:9b-nvfp4"
    if run_log_dir:
        log_data = lookup_item_in_log(item["item_id"], run_log_dir)
        score_entry = log_data.get("score")
        if score_entry:
            prompt = score_entry.get("prompt", "")
            model_output = score_entry.get("raw_response", "")
            model_used = score_entry.get("model", "qwen3.5:9b-nvfp4")

    feedback_entry = {
        "item_id": item["item_id"],
        "step_id": "score",
        "input_prompt": prompt,
        "model_output": model_output,
        "user_rating": rating,
        "corrected_output": output,
        "review_notes": review_notes,
        "dimension_verdicts": dimension_verdicts,
        "timestamp": now,
        "model_used": model_used,
    }
    feedback.append(feedback_entry)
    save_json(FEEDBACK_PATH, feedback)


def _ask_review_notes() -> str | None:
    """Prompt for optional review notes. Returns None if empty/skipped."""
    print("  Notes? (optional — why this verdict, hit Enter to skip):")
    try:
        notes = input("  notes> ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    return notes or None


def cmd_review(args):
    """Interactive review of Aragorn disagreements."""
    challenges = load_json(ARAGORN_PATH)
    if not challenges or not isinstance(challenges, dict):
        print("No Aragorn challenges found.")
        return

    all_challenge_items = challenges.get("items", [])

    if args.item:
        # Review a single item
        items = [i for i in all_challenge_items if i["item_id"] == args.item]
        if not items:
            print(f"Item {args.item} not found in challenges.")
            return
    elif args.all:
        items = [i for i in all_challenge_items if i.get("is_disagreement")]
    else:
        items = get_pending_items()

    if not items:
        print("No items to review.")
        return

    if args.limit and args.limit > 0:
        items = items[:args.limit]

    # Default run-log directory (fallback for items without per-item run_log_dir)
    default_run_log_name = challenges.get("run_log_dir", "")
    default_run_log_dir = (find_run_log_by_name(default_run_log_name)
                           if default_run_log_name and default_run_log_name != "batch"
                           else find_latest_run_log())

    feedback = load_json(FEEDBACK_PATH)
    if not isinstance(feedback, list):
        feedback = []

    reviewed_count = 0
    verdicts = {"agree_aragorn": 0, "agree_pipeline": 0, "corrected": 0, "skipped": 0}

    for idx, item in enumerate(items, 1):
        # Per-item run_log_dir resolution (supports batch items from different runs)
        item_run_log_name = item.get("run_log_dir")
        if item_run_log_name:
            run_log_dir = find_run_log_by_name(item_run_log_name) or default_run_log_dir
        else:
            run_log_dir = default_run_log_dir

        extracted = lookup_extracted_data(item["item_id"], run_log_dir)
        display_review_item(item, idx, len(items), extracted=extracted)

        challenges = item.get("challenges", [])
        quit_requested = False

        if not challenges:
            # No per-dimension challenges — fall back to overall verdict
            dim_verdicts = {}
        else:
            # --- Per-dimension review ---
            dim_verdicts = {}
            print()
            for c in challenges:
                dim = c.get("dimension", "?")
                orig_s = c.get("original_score", "?")
                sugg_s = c.get("suggested_score", "?")
                issue = c.get("issue", "")
                print(f"  {dim}: {orig_s} → {sugg_s}")
                if issue:
                    print(f"    \"{issue}\"")
                print("  [a] Agree with Aragorn  [d] Disagree (pipeline right)  [s] Skip")

                while True:
                    try:
                        choice = input("  > ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        print("\n\nReview interrupted. Progress saved.")
                        return
                    if choice == "q":
                        quit_requested = True
                        break
                    elif choice in ("a", "d"):
                        print("  Why? (Enter to skip):")
                        try:
                            reason = input("  notes> ").strip() or None
                        except (EOFError, KeyboardInterrupt):
                            print("\n\nReview interrupted. Progress saved.")
                            return
                        verdict_str = "agree" if choice == "a" else "disagree"
                        dim_verdicts[dim] = {
                            "verdict": verdict_str,
                            "original_score": orig_s,
                            "aragorn_score": sugg_s,
                            "notes": reason,
                        }
                        print(f"  ✓ {dim}: {verdict_str}")
                        break
                    elif choice == "s":
                        dim_verdicts[dim] = {"verdict": "skipped"}
                        print(f"  → {dim}: skipped")
                        break
                    else:
                        print("  Invalid choice. Use a/d/s/q.")
                print()

                if quit_requested:
                    break

        if quit_requested:
            print(f"\nQuit. Reviewed {reviewed_count} items this session.")
            _print_verdict_summary(verdicts)
            return

        # --- Overall verdict (computed from dimension verdicts) ---
        agree_count = sum(1 for v in dim_verdicts.values() if v.get("verdict") == "agree")
        disagree_count = sum(1 for v in dim_verdicts.values() if v.get("verdict") == "disagree")
        non_skip = agree_count + disagree_count

        if non_skip == 0:
            # All skipped or no challenges — ask for overall
            print("  Overall: [y] Agree with Aragorn  [n] Pipeline was right  [s] Skip")
            while True:
                try:
                    choice = input("  > ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\n\nReview interrupted. Progress saved.")
                    return
                if choice == "q":
                    print(f"\nQuit. Reviewed {reviewed_count} items this session.")
                    _print_verdict_summary(verdicts)
                    return
                elif choice == "y":
                    overall_verdict = "agree_aragorn"
                    break
                elif choice == "n":
                    overall_verdict = "agree_pipeline"
                    break
                elif choice == "s":
                    overall_verdict = "skipped"
                    break
                else:
                    print("  Invalid choice. Use y/n/s/q.")
        elif disagree_count == 0:
            overall_verdict = "agree_aragorn"
            print(f"  → Overall: agree with Aragorn ({agree_count}/{non_skip} dimensions)")
        elif agree_count == 0:
            overall_verdict = "agree_pipeline"
            print(f"  → Overall: pipeline was right ({disagree_count}/{non_skip} dimensions)")
        else:
            overall_verdict = "partial"
            print(f"  → Overall: mixed ({agree_count} agree, {disagree_count} disagree)")

        # Overall notes
        overall_notes = _ask_review_notes()

        # Record
        if overall_verdict == "skipped":
            record_review_verdict(item, "skipped", feedback, run_log_dir=run_log_dir)
            verdicts["skipped"] += 1
        elif overall_verdict == "agree_aragorn":
            corrected = build_corrected_scoring(item, run_log_dir, dim_verdicts)
            record_review_verdict(item, "agree_aragorn", feedback,
                                  corrected_output=corrected,
                                  run_log_dir=run_log_dir,
                                  review_notes=overall_notes,
                                  dimension_verdicts=dim_verdicts)
            verdicts["agree_aragorn"] += 1
        elif overall_verdict == "agree_pipeline":
            record_review_verdict(item, "agree_pipeline", feedback,
                                  run_log_dir=run_log_dir,
                                  review_notes=overall_notes,
                                  dimension_verdicts=dim_verdicts)
            verdicts["agree_pipeline"] += 1
        elif overall_verdict == "partial":
            corrected = build_corrected_scoring(item, run_log_dir, dim_verdicts)
            record_review_verdict(item, "partial", feedback,
                                  corrected_output=corrected,
                                  run_log_dir=run_log_dir,
                                  review_notes=overall_notes,
                                  dimension_verdicts=dim_verdicts)
            if "partial" not in verdicts:
                verdicts["partial"] = 0
            verdicts["partial"] += 1

        reviewed_count += 1
        print(f"  ✓ Recorded: {overall_verdict}")

    print(f"\nDone. Reviewed {reviewed_count} items.")
    _print_verdict_summary(verdicts)


def _print_verdict_summary(verdicts: dict):
    """Print a summary of review verdicts."""
    parts = []
    if verdicts.get("agree_aragorn"):
        parts.append(f"{verdicts['agree_aragorn']} agreed w/ Aragorn")
    if verdicts.get("agree_pipeline"):
        parts.append(f"{verdicts['agree_pipeline']} agreed w/ pipeline")
    if verdicts.get("partial"):
        parts.append(f"{verdicts['partial']} mixed/partial")
    if verdicts.get("corrected"):
        parts.append(f"{verdicts['corrected']} manually corrected")
    if verdicts.get("skipped"):
        parts.append(f"{verdicts['skipped']} skipped")
    if parts:
        print(f"  Summary: {', '.join(parts)}")


# ---------------------------------------------------------------------------
# Legacy record subcommand (backward compatible: feedback.py job-search ...)
# ---------------------------------------------------------------------------
def cmd_record(args):
    """Legacy feedback recording — unchanged from original."""
    if args.rating == "corrected" and not args.correction:
        print("Error: --correction is required when --rating is 'corrected'", file=sys.stderr)
        sys.exit(1)

    run_log_dir = find_latest_run_log()
    if run_log_dir is None:
        print("Error: No run logs found. Run the pipeline first.", file=sys.stderr)
        sys.exit(1)

    print(f"Using run log: {run_log_dir.name}")

    log_data = lookup_item_in_log(args.item, run_log_dir)

    if not log_data["extract"] and not log_data["score"]:
        print(f"Error: Item '{args.item}' not found in run log {run_log_dir.name}", file=sys.stderr)
        for step_id in ("extract", "score"):
            log_file = run_log_dir / f"{step_id}_prompts.json"
            if log_file.exists():
                entries = load_json(log_file)
                ids = [e["item_id"] for e in entries]
                if ids:
                    print(f"  Available item IDs in {step_id}: {', '.join(ids)}", file=sys.stderr)
        sys.exit(1)

    steps_to_record = []
    if args.step:
        if log_data[args.step]:
            steps_to_record.append(args.step)
        else:
            print(f"Warning: No {args.step} log found for item {args.item}", file=sys.stderr)
    else:
        for step_id in ("extract", "score"):
            if log_data[step_id]:
                steps_to_record.append(step_id)

    if not steps_to_record:
        print("Error: No log entries found for this item", file=sys.stderr)
        sys.exit(1)

    feedback = load_json(FEEDBACK_PATH)

    for step_id in steps_to_record:
        entry_data = log_data[step_id]
        feedback_entry = {
            "item_id": args.item,
            "step_id": step_id,
            "input_prompt": entry_data["prompt"],
            "model_output": entry_data["raw_response"],
            "user_rating": args.rating,
            "corrected_output": args.correction,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_used": entry_data.get("model", "qwen3.5:27b-int4"),
        }
        feedback.append(feedback_entry)
        print(f"Recorded {args.rating} feedback for step '{step_id}' on item {args.item}")

    save_json(FEEDBACK_PATH, feedback)
    print(f"Feedback saved to {FEEDBACK_PATH} ({len(feedback)} total entries)")


# ---------------------------------------------------------------------------
# Record-verdict subcommand (non-interactive, for in-chat review)
# ---------------------------------------------------------------------------
def cmd_record_verdict(args):
    """Non-interactive verdict recording — called by Claude during in-chat review."""
    challenges = load_json(ARAGORN_PATH)
    if not isinstance(challenges, dict):
        print(json.dumps({"status": "error", "message": "No aragorn-challenges.json found"}))
        sys.exit(1)

    # Find the item
    item = None
    for ci in challenges.get("items", []):
        if ci.get("item_id") == args.item:
            item = ci
            break

    if not item:
        print(json.dumps({"status": "error", "message": f"Item {args.item} not found"}))
        sys.exit(1)

    if item.get("reviewed"):
        print(json.dumps({"status": "error", "message": f"Item {args.item} already reviewed"}))
        sys.exit(1)

    # Build dimension_verdicts from --agree/--disagree/--skip flags
    agree_dims = args.agree or []
    disagree_dims = args.disagree or []
    skip_dims = args.skip or []

    dim_verdicts = {}
    challenge_lookup = {c["dimension"]: c for c in item.get("challenges", []) if "dimension" in c}

    for dim in agree_dims:
        c = challenge_lookup.get(dim, {})
        dim_verdicts[dim] = {
            "verdict": "agree",
            "original_score": c.get("original_score"),
            "aragorn_score": c.get("suggested_score"),
        }
    for dim in disagree_dims:
        c = challenge_lookup.get(dim, {})
        dim_verdicts[dim] = {
            "verdict": "disagree",
            "original_score": c.get("original_score"),
            "aragorn_score": c.get("suggested_score"),
        }
    for dim in skip_dims:
        dim_verdicts[dim] = {"verdict": "skipped"}

    # Auto-compute overall verdict
    agree_count = len(agree_dims)
    disagree_count = len(disagree_dims)
    non_skip = agree_count + disagree_count

    if non_skip == 0:
        overall_verdict = "skipped"
    elif disagree_count == 0:
        overall_verdict = "agree_aragorn"
    elif agree_count == 0:
        overall_verdict = "agree_pipeline"
    else:
        overall_verdict = "partial"

    # Find run-log directory (per-item first, then top-level fallback)
    run_log_name = item.get("run_log_dir") or challenges.get("run_log_dir", "")
    if run_log_name and run_log_name != "batch":
        run_log_dir = find_run_log_by_name(run_log_name)
    else:
        run_log_dir = find_latest_run_log()

    run_log_warning = None
    if not run_log_dir:
        run_log_warning = "Run-log directory not found — corrected_output will be null"

    # Build corrected output if needed
    corrected = None
    if overall_verdict in ("agree_aragorn", "partial") and run_log_dir:
        corrected = build_corrected_scoring(item, run_log_dir, dim_verdicts)

    # Record
    feedback = load_json(FEEDBACK_PATH)
    if not isinstance(feedback, list):
        feedback = []

    record_review_verdict(
        item, overall_verdict, feedback,
        corrected_output=corrected,
        run_log_dir=run_log_dir,
        review_notes=args.notes,
        dimension_verdicts=dim_verdicts,
    )

    result = {
        "status": "ok",
        "item_id": args.item,
        "verdict": overall_verdict,
        "dimensions_applied": agree_count,
    }
    if run_log_warning:
        result["warning"] = run_log_warning
    print(json.dumps(result))


# ---------------------------------------------------------------------------
# Main — route to subcommand or legacy path
# ---------------------------------------------------------------------------
def main():
    # Backward compatibility: if first arg is an agent name, use legacy path
    if len(sys.argv) > 1 and sys.argv[1] in AGENT_NAMES:
        parser = argparse.ArgumentParser(description="Perian feedback collection")
        parser.add_argument("agent", choices=sorted(AGENT_NAMES), help="Agent name")
        parser.add_argument("--item", required=True, help="Item ID from the briefing")
        parser.add_argument(
            "--rating", required=True, choices=["good", "bad", "corrected"],
            help="Rating for the item",
        )
        parser.add_argument("--correction", default=None, help="Corrected output")
        parser.add_argument(
            "--step", choices=["extract", "score"], default=None,
            help="Which step to give feedback on",
        )
        args = parser.parse_args()
        cmd_record(args)
        return

    # Subcommand routing
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    subcmd = sys.argv[1]

    if subcmd == "pending":
        use_json = "--json" in sys.argv
        cmd_pending(use_json=use_json)
    elif subcmd == "record-verdict":
        parser = argparse.ArgumentParser(description="Record verdict non-interactively")
        parser.add_argument("_subcmd", help=argparse.SUPPRESS)
        parser.add_argument("--item", required=True, help="Item ID")
        parser.add_argument("--agree", nargs="*", default=[],
                            help="Dimensions to agree with Aragorn on")
        parser.add_argument("--disagree", nargs="*", default=[],
                            help="Dimensions to disagree on (pipeline was right)")
        parser.add_argument("--skip", nargs="*", default=[],
                            help="Dimensions to skip")
        parser.add_argument("--notes", default=None,
                            help="Overall review notes")
        args = parser.parse_args()
        cmd_record_verdict(args)
    elif subcmd == "review":
        parser = argparse.ArgumentParser(description="Review Aragorn disagreements")
        parser.add_argument("_subcmd", help=argparse.SUPPRESS)
        parser.add_argument("--all", action="store_true",
                            help="Include previously reviewed items")
        parser.add_argument("--item", default=None,
                            help="Review a single item by ID")
        parser.add_argument("--limit", type=int, default=0,
                            help="Review at most N items (default: all pending)")
        args = parser.parse_args()
        cmd_review(args)
    else:
        print(f"Unknown command: {subcmd}", file=sys.stderr)
        print("Available commands: pending, review, record-verdict, job-search", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
