#!/usr/bin/env python3
"""Enrich queue consumer — fetch full descriptions for low-signal items.

Reads state/enrich-queue.json, fetches full job descriptions from URLs,
re-runs extraction+scoring, and updates the queue with results. Items that
score above threshold after enrichment are written to a supplemental briefing.

LinkedIn URLs require authentication and will be skipped (marked as
fetch_failed). Non-LinkedIn career page URLs are fetched directly.

Usage:
    python3 scripts/enrich_queue.py              # Process queue
    python3 scripts/enrich_queue.py --dry-run    # Show what would be processed
    python3 scripts/enrich_queue.py --limit 5    # Process at most 5 items
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml

from ollama_utils import warm_up_model, cool_down_model
from runner import inject_variables, load_prompt_template
from schemas import EXTRACTION_SCHEMA, SCORING_SCHEMA

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
QUEUE_PATH = ROOT / "state" / "enrich-queue.json"
CONFIG_PATH = ROOT / "config" / "env.yaml"
PLAN_PATH = ROOT / "plan.yaml"
PROFILE_PATH = ROOT / "PROFILE.md"
OUTPUT_DIR = ROOT / "output"
STATE_DIR = ROOT / "state"
RUN_LOG_DIR = ROOT / "run-log"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("enrich")

log = setup_logging()

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


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_json_response(text: str) -> dict | None:
    """Extract JSON from an LLM response."""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()
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


# ---------------------------------------------------------------------------
# Fetch descriptions
# ---------------------------------------------------------------------------

# LinkedIn URLs require auth — we can't scrape them with plain requests.
LINKEDIN_DOMAINS = {"linkedin.com", "www.linkedin.com"}

# Selectors for extracting job description text from HTML
_DESCRIPTION_PATTERNS = [
    # Common job description containers
    r'<div[^>]*class="[^"]*(?:job-description|description|posting-body|content-body)[^"]*"[^>]*>(.*?)</div>',
    # Broader: any large text block
    r'<article[^>]*>(.*?)</article>',
]

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _html_to_text(html_str: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    text = _TAG_RE.sub(" ", html_str)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def fetch_description(url: str, logger: logging.Logger) -> dict:
    """Fetch a job description from a URL.

    Returns dict with keys:
        success: bool
        description: str (if success)
        reason: str (if not success)
    """
    domain = urlparse(url).netloc.lower()

    if domain in LINKEDIN_DOMAINS:
        return {"success": False, "reason": "linkedin_auth_required"}

    try:
        resp = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": "Perian/1.0"},
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Fetch failed for {url}: {e}")
        return {"success": False, "reason": f"http_error: {e}"}

    html_content = resp.text

    # Try extracting from known containers
    for pattern in _DESCRIPTION_PATTERNS:
        match = re.search(pattern, html_content, re.DOTALL | re.IGNORECASE)
        if match:
            text = _html_to_text(match.group(1))
            if len(text) > 100:  # meaningful content threshold
                logger.debug(f"Extracted {len(text)} chars from {url}")
                return {"success": True, "description": text[:8000]}

    # Fallback: extract all visible text from body
    body_match = re.search(r"<body[^>]*>(.*?)</body>", html_content, re.DOTALL | re.IGNORECASE)
    if body_match:
        # Remove script and style blocks
        body = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", body_match.group(1), flags=re.DOTALL | re.IGNORECASE)
        text = _html_to_text(body)
        if len(text) > 200:
            logger.debug(f"Fallback extraction: {len(text)} chars from {url}")
            return {"success": True, "description": text[:8000]}

    return {"success": False, "reason": "no_description_found"}


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def call_ollama(prompt: str, model: str, config: dict,
                format_schema=None) -> dict | None:
    """Call Ollama /api/chat."""
    ollama_cfg = config.get("ollama", {})
    base_url = ollama_cfg.get("base_url", "http://localhost:11434")
    num_ctx = ollama_cfg.get("num_ctx", 8192)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "format": format_schema if format_schema is not None else "json",
        "options": {"num_ctx": num_ctx},
    }
    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        if "message" in data and "response" not in data:
            data["response"] = data["message"].get("content", "")
        return data
    except requests.RequestException as e:
        log.error(f"Ollama call failed: {e}")
        return None


def extract_and_score(item: dict, config: dict, plan: dict,
                      profile: str, model: str,
                      logger: logging.Logger) -> dict | None:
    """Run extraction + scoring on an enriched item.

    Returns dict with extracted, scoring, and total_score, or None on failure.
    """
    delimiter = "--- ITEM TO PROCESS ---"

    # --- Extraction ---
    extract_step = next(s for s in plan["steps"] if s["id"] == "extract")
    extract_template = load_prompt_template(extract_step["prompt_template"])

    listing_text = (
        f"Title: {item['title']}\n"
        f"Company: {item['company']}\n"
        f"URL: {item['url']}\n\n"
        f"{item['description']}"
    )

    parts = extract_template.split(delimiter, 1)
    static_prefix = parts[0]
    per_item = parts[1] if len(parts) > 1 else "\n\n{{listing}}"
    prompt = static_prefix + delimiter + per_item.replace("{{listing}}", listing_text)

    t0 = time.time()
    resp = call_ollama(prompt, model, config, format_schema=EXTRACTION_SCHEMA)
    extract_time = time.time() - t0
    if not resp:
        return None

    extracted = parse_json_response(resp.get("response", ""))
    if not extracted:
        logger.warning(f"Failed to parse extraction for {item['title']}")
        return None

    # --- Scoring ---
    score_step = next(s for s in plan["steps"] if s["id"] == "score")
    score_template = load_prompt_template(score_step["prompt_template"])
    score_template = inject_variables(score_template, plan)
    score_template = score_template.replace("{{profile}}", profile)

    parts = score_template.split(delimiter, 1)
    static_prefix = parts[0]
    per_item = parts[1] if len(parts) > 1 else "\n\n{{extracted}}"
    extracted_json = json.dumps(extracted, indent=2)
    prompt = static_prefix + delimiter + per_item.replace("{{extracted}}", extracted_json)

    t0 = time.time()
    resp = call_ollama(prompt, model, config, format_schema=SCORING_SCHEMA)
    score_time = time.time() - t0
    if not resp:
        return None

    scoring = parse_json_response(resp.get("response", ""))
    if not scoring:
        logger.warning(f"Failed to parse scoring for {item['title']}")
        return None

    total = scoring.get("total_score", 0)
    logger.info(
        f"  Enriched: {item['title']} @ {item['company']} → {total:.3f} "
        f"(extract: {extract_time:.1f}s, score: {score_time:.1f}s)"
    )

    return {
        "extracted": extracted,
        "scoring": scoring,
        "total_score": total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_queue(dry_run: bool = False, limit: int | None = None,
                  model: str = "qwen3.5:9b-nvfp4") -> None:
    """Process the enrich queue."""
    queue = load_json(QUEUE_PATH)
    if not isinstance(queue, list):
        log.warning("Enrich queue is not a list, skipping")
        return

    # Filter to items not yet processed
    pending = [item for item in queue if item.get("status") not in ("enriched", "fetch_failed")]
    if not pending:
        log.info("Enrich queue: nothing to process")
        return

    if limit:
        pending = pending[:limit]

    log.info(f"Enrich queue: {len(pending)} item(s) to process ({len(queue)} total in queue)")

    if dry_run:
        for item in pending:
            domain = urlparse(item["url"]).netloc
            fetchable = domain not in LINKEDIN_DOMAINS
            log.info(f"  {'✓' if fetchable else '✗'} {item['title']} @ {item['company']} ({domain})")
        fetchable_count = sum(1 for i in pending if urlparse(i["url"]).netloc not in LINKEDIN_DOMAINS)
        log.info(f"  {fetchable_count}/{len(pending)} fetchable (non-LinkedIn)")
        return

    config = load_config()
    plan_data = yaml.safe_load(open(PLAN_PATH))
    profile = PROFILE_PATH.read_text() if PROFILE_PATH.exists() else ""

    # Warm up model
    warm_up_model(model, config, log)

    enriched_count = 0
    failed_count = 0
    skipped_count = 0

    for i, item in enumerate(pending):
        log.info(f"Processing [{i+1}/{len(pending)}]: {item['title']} @ {item['company']}")

        # Fetch description
        result = fetch_description(item["url"], log)

        if not result["success"]:
            log.info(f"  Fetch failed: {result['reason']}")
            item["status"] = "fetch_failed"
            item["fetch_reason"] = result["reason"]
            item["processed_at"] = datetime.now(timezone.utc).isoformat()
            failed_count += 1
            continue

        item["description"] = result["description"]

        # Re-extract and re-score with full description
        llm_result = extract_and_score(item, config, plan_data, profile, model, log)

        if llm_result is None:
            log.warning(f"  LLM processing failed")
            item["status"] = "fetch_failed"
            item["fetch_reason"] = "llm_error"
            item["processed_at"] = datetime.now(timezone.utc).isoformat()
            failed_count += 1
            continue

        item["status"] = "enriched"
        item["enriched_score"] = llm_result["total_score"]
        item["original_score"] = item.get("score", 0)
        item["score_delta"] = round(llm_result["total_score"] - item.get("score", 0), 3)
        item["extracted"] = llm_result["extracted"]
        item["scoring"] = llm_result["scoring"]
        item["processed_at"] = datetime.now(timezone.utc).isoformat()
        enriched_count += 1

        # Remove description from queue (it's large and we have extracted data)
        del item["description"]

    # Cool down model
    cool_down_model(model, config, log)

    # Save updated queue
    save_json(QUEUE_PATH, queue)

    # Summary
    log.info(
        f"Enrich complete: {enriched_count} enriched, {failed_count} failed, "
        f"{skipped_count} skipped"
    )

    # Report score changes for enriched items
    improved = [
        item for item in queue
        if item.get("status") == "enriched"
        and item.get("score_delta", 0) > 0.05
        and item.get("processed_at", "").startswith(datetime.now().strftime("%Y-%m-%d"))
    ]
    if improved:
        log.info(f"Score improvements from enrichment:")
        for item in sorted(improved, key=lambda x: x.get("score_delta", 0), reverse=True):
            log.info(
                f"  {item['title']} @ {item['company']}: "
                f"{item['original_score']:.3f} → {item['enriched_score']:.3f} "
                f"(+{item['score_delta']:.3f})"
            )


def main():
    parser = argparse.ArgumentParser(description="Enrich queue consumer")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    parser.add_argument("--limit", type=int, help="Max items to process")
    parser.add_argument("--model", default="qwen3.5:9b-nvfp4", help="Ollama model for extraction/scoring")
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("enrich").setLevel(logging.DEBUG)

    process_queue(dry_run=args.dry_run, limit=args.limit, model=args.model)


if __name__ == "__main__":
    main()
