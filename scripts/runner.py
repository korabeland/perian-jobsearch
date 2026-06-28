#!/usr/bin/env python3
"""Perian Phase 2 — multi-model router with confidence escalation.

Reads plan.yaml, processes job listings through:
  ingest → dedup → extract (LLM) → score (LLM) → rank/filter → briefing → update ledger

Phase 1: prefix caching, logprob collection, timing, keep-alive management.
Phase 2: router picks cheapest qualifying model per step, escalates to 32B
         when logprobs signal low confidence. Circuit breaker at 25%.
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

from router import Router
from ollama_utils import warm_up_model, unload_model as _unload_model, cool_down_model
from schemas import EXTRACTION_SCHEMA, SCORING_SCHEMA, TRIAGE_SCHEMA

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PLAN_PATH = ROOT / "plan.yaml"
PROFILE_PATH = ROOT / "profile.md"
CONFIG_PATH = ROOT / "config" / "env.yaml"
STATE_DIR = ROOT / "state"
SEEN_PATH = STATE_DIR / "seen.json"
LOGPROBS_PATH = STATE_DIR / "logprobs.json"
METRICS_PATH = STATE_DIR / "metrics.json"
INPUT_DIR = ROOT / "inputs"
OUTPUT_DIR = ROOT / "output"
RUN_LOG_DIR = ROOT / "run-log"
PROMPTS_DIR = ROOT / "prompts"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(run_id: str) -> logging.Logger:
    """Configure logging to both console and run-log file."""
    log_dir = RUN_LOG_DIR / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("perian")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Console handler — INFO level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    # File handler — DEBUG level
    fh = logging.FileHandler(log_dir / "run.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_item_id(url: str, company: str, title: str) -> str:
    """Stable item ID: SHA-256 of url+company+title, truncated to 12 hex chars."""
    raw = f"{url.strip()}|{company.strip().lower()}|{title.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def load_json(path: Path) -> list | dict:
    if not path.exists():
        return [] if path.name != "metrics.json" else {}
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_plan() -> dict:
    with open(PLAN_PATH) as f:
        return yaml.safe_load(f)


def load_profile() -> str:
    return PROFILE_PATH.read_text()


def load_prompt_template(path: str) -> str:
    p = ROOT / path
    return p.read_text()


def load_config() -> dict:
    """Load config/env.yaml, falling back to defaults."""
    defaults = {
        "ollama": {
            "base_url": "http://localhost:11434",
            "keep_alive": -1,
            "num_ctx": 8192,
            "default_keep_alive_after": "5m",
        }
    }
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        for section, vals in defaults.items():
            if section not in cfg:
                cfg[section] = vals
            else:
                for k, v in vals.items():
                    if k not in cfg[section]:
                        cfg[section][k] = v
        return cfg
    return defaults


def inject_variables(template: str, variables: dict) -> str:
    """Replace {{key}} and {{nested.key}} placeholders with values.

    Also handles simple {{#each}} / {{/each}} blocks for arrays.
    """
    # Handle {{#each scoring.bonuses}} ... {{/each}} blocks
    each_pattern = re.compile(
        r"\{\{#each\s+([\w.]+)\}\}(.*?)\{\{/each\}\}", re.DOTALL
    )

    def expand_each(match):
        key_path = match.group(1)
        block_template = match.group(2)
        items = _resolve_path(variables, key_path)
        if not isinstance(items, list):
            return ""
        lines = []
        for item in items:
            line = block_template
            for k, v in item.items():
                line = line.replace("{{" + k + "}}", str(v))
            lines.append(line.strip())
        return "\n".join(lines)

    result = each_pattern.sub(expand_each, template)

    # Handle simple {{key.path}} variables
    def replace_var(match):
        key_path = match.group(1)
        val = _resolve_path(variables, key_path)
        if val is None:
            return match.group(0)  # leave unresolved
        return str(val)

    result = re.sub(r"\{\{([\w.]+)\}\}", replace_var, result)
    return result


def _resolve_path(data: dict, path: str):
    """Resolve a dotted path like 'scoring.dimensions.title_match' in a dict."""
    parts = path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_ollama(prompt: str, model: str, logger: logging.Logger, config: dict,
                keep_alive=None, format_schema=None) -> dict | None:
    """Call Ollama /api/chat with think=false and return the full response dict.

    Uses /api/chat instead of /api/generate so we can disable qwen3's thinking
    mode (think=false). Without this, qwen3 generates thousands of hidden
    reasoning tokens before the actual output, making extraction 10-100x slower.

    Args:
        format_schema: JSON Schema dict for constrained decoding, or None for
            basic "json" mode. Ollama v0.5+ converts schemas to GBNF grammars,
            guaranteeing structurally valid output.
    """
    ollama_cfg = config.get("ollama", {})
    base_url = ollama_cfg.get("base_url", "http://localhost:11434")
    num_ctx = ollama_cfg.get("num_ctx", 8192)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "format": format_schema if format_schema is not None else "json",
        # Logprobs must be requested top-level (Ollama 0.24+); placed under
        # "options" they are silently dropped from the response.
        "logprobs": True,
        "top_logprobs": 1,
        "options": {
            "num_ctx": num_ctx,
        },
    }
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive

    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        # Normalize chat response to look like generate response for callers
        # /api/chat puts text in data["message"]["content"] instead of data["response"]
        if "message" in data and "response" not in data:
            data["response"] = data["message"].get("content", "")
        return data
    except requests.RequestException as e:
        logger.error(f"Ollama call failed: {e}")
        return None


def _fix_json_math_expressions(text: str) -> str:
    """Replace inline math expressions in JSON values with the final number.

    LLMs sometimes 'show their work' in numeric fields, e.g.:
      "weighted_total": 1.0 * 0.25 + 0.8 * 0.2 = 0.41
    This extracts the number after the last '=' (the model's own computed result).
    Only touches unquoted values — string values in quotes are left alone.
    """
    lines = text.split("\n")
    fixed = []
    for line in lines:
        # Pattern: "key": <unquoted math with => <final number> <optional comma/brace>
        # [^"]* ensures we don't touch quoted string values
        fixed.append(re.sub(
            r'("[^"]+":\s*)[^"]*=\s*([\d.]+)(\s*[,}]?\s*)$',
            r"\1\2\3",
            line,
        ))
    return "\n".join(fixed)


def parse_json_response(text: str, logger: logging.Logger) -> dict | None:
    """Extract JSON from an LLM response, handling common failure modes."""
    if not text:
        return None

    # Strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try fixing inline math expressions (e.g. "total": 0.9 + 0.10 = 1.0)
    fixed = _fix_json_math_expressions(cleaned)
    if fixed != cleaned:
        try:
            result = json.loads(fixed)
            logger.debug("Parsed JSON after fixing inline math expressions")
            return result
        except json.JSONDecodeError:
            pass

    # Try to find first { ... } block
    brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if brace_match:
        candidate = brace_match.group()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        fixed_candidate = _fix_json_math_expressions(candidate)
        if fixed_candidate != candidate:
            try:
                result = json.loads(fixed_candidate)
                logger.debug("Parsed JSON after fixing math in extracted block")
                return result
            except json.JSONDecodeError:
                pass

    logger.error(f"Failed to parse JSON from LLM response: {text[:200]}...")
    return None


def extract_logprob_stats(response_data: dict, logger: logging.Logger) -> dict | None:
    """Extract logprob statistics from an Ollama /api/chat response.

    Ollama 0.24+ returns a top-level "logprobs" list of per-token dicts:
      [{"token": "{", "logprob": -0.0025, "bytes": [...], "top_logprobs": [...]}, ...]
    We average the chosen-token logprobs into the confidence signal the router
    uses (mean_logprob vs router.yaml threshold). Works on ggml and MLX runners.
    """
    if not response_data:
        return None

    logprobs_data = response_data.get("logprobs")
    if not isinstance(logprobs_data, list) or not logprobs_data:
        return None

    token_logprobs = [
        t["logprob"] for t in logprobs_data
        if isinstance(t, dict) and isinstance(t.get("logprob"), (int, float))
    ]
    if not token_logprobs:
        return None

    return {
        "mean_logprob": round(statistics.mean(token_logprobs), 4),
        "min_logprob": round(min(token_logprobs), 4),
        "token_count": len(token_logprobs),
    }


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_ingest(plan: dict, logger: logging.Logger) -> list[dict]:
    """Load all CSV files from inputs/ and normalize to common schema."""
    source_cfg = plan["sources"][0]
    fields = source_cfg["fields"]
    input_path = ROOT / source_cfg["path"].strip("./")

    items = []
    csv_files = sorted(input_path.glob("*.csv"))
    if not csv_files:
        logger.info(f"No CSV files found in {input_path}")

    for csv_file in csv_files:
        logger.info(f"Loading {csv_file.name}")
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row_num, row in enumerate(reader, 1):
                if not row or all(cell.strip() == "" for cell in row):
                    continue
                # Skip header rows
                if row_num == 1 and row[fields["title"]].strip().lower() in ("title", "job title"):
                    continue
                try:
                    item = {
                        "title": row[fields["title"]].strip(),
                        "company": row[fields["company"]].strip(),
                        "url": row[fields["url"]].strip(),
                        "description": row[fields["description"]].strip(),
                        "source_file": csv_file.name,
                    }
                    item["item_id"] = make_item_id(item["url"], item["company"], item["title"])
                    items.append(item)
                except IndexError:
                    logger.warning(f"Skipping malformed row {row_num} in {csv_file.name}: {row}")

    # --- JSONL loading (from collect.py output) ---
    jsonl_files = sorted(input_path.glob("*.jsonl"))
    for jsonl_file in jsonl_files:
        logger.info(f"Loading {jsonl_file.name}")
        with open(jsonl_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    item = {
                        "title": record["title"].strip(),
                        "company": record["company"].strip(),
                        "url": record["url"].strip(),
                        "description": record["description"].strip(),
                        "source_file": jsonl_file.name,
                    }
                    # Carry through signal_level for enrichment queue
                    if record.get("signal_level"):
                        item["signal_level"] = record["signal_level"]
                    item["item_id"] = make_item_id(item["url"], item["company"], item["title"])
                    items.append(item)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping malformed line {line_num} in {jsonl_file.name}: {e}")

    total_files = len(csv_files) + len(jsonl_files)
    logger.info(f"Ingested {len(items)} listings from {total_files} file(s)")
    return items


def step_dedup(items: list[dict], logger: logging.Logger) -> list[dict]:
    """Filter out items already in seen.json."""
    seen = load_json(SEEN_PATH)
    seen_ids = {entry["item_id"] for entry in seen if isinstance(entry, dict)}
    seen_urls = {entry.get("url", "") for entry in seen if isinstance(entry, dict)}

    new_items = []
    for item in items:
        if item["item_id"] in seen_ids:
            logger.debug(f"Dedup (id): skipping {item['title']} @ {item['company']}")
            continue
        if item["url"] in seen_urls:
            logger.debug(f"Dedup (url): skipping {item['title']} @ {item['company']}")
            continue
        # Check title+company combo
        key = f"{item['title'].lower()}|{item['company'].lower()}"
        seen_keys = {
            f"{e.get('title', '').lower()}|{e.get('company', '').lower()}"
            for e in seen
            if isinstance(e, dict)
        }
        if key in seen_keys:
            logger.debug(f"Dedup (title+company): skipping {item['title']} @ {item['company']}")
            continue
        new_items.append(item)

    skipped = len(items) - len(new_items)
    if skipped:
        logger.info(f"Dedup: skipped {skipped} already-seen item(s)")
    logger.info(f"Dedup: {len(new_items)} new item(s) to process")
    return new_items


# ---------------------------------------------------------------------------
# Pre-filter: reject obviously irrelevant titles before LLM extraction
# ---------------------------------------------------------------------------

# Patterns that disqualify a title outright (case-insensitive).
# Conservative — only things that are clearly outside the CX/Product/AI Strategy
# profile. When in doubt, let the LLM handle it.
_EXCLUDE_PATTERNS = re.compile(r"""
    engineer|developer|architect|devops|
    \bsre\b|
    designer|                          # overridden by _PROTECT below
    account\s+executive|account\s+manager|
    sales\s+development|business\s+development|
    \bsdr\b|\bbdr\b|
    ai\s+trainer|                      # data-labeling gigs
    \bintern\b|internship|
    nurse|physician|medical\s+director|clinical|
    therapist|pharmacist|social\s+worker|
    technologist|radiolog|
    teacher|professor|
    accountant|\bcontroller\b|bookkeeper|\bclerk\b|
    attorney|counsel|paralegal|
    recruiter|talent\s+acquisition|
    mechanic|electrician|plumber|welder|
    \bdriver\b|cashier|janitor|custodian|
    warehouse|forklift|construction\s+worker|
    \bcook\b|bartender|
    real\s+estate|insurance|procurement|
    \bhr\s|\bhris\b|human\s+resources|
    \bfinance\b|\bfinancial\b|\btreasury\b|\baudit\b
""", re.IGNORECASE | re.VERBOSE)

# Titles matching an exclude pattern are still kept if they match a protect pattern.
_PROTECT_PATTERNS = re.compile(
    r"""learning\s+designer|instructional\s+designer|learning\s+experience\s+designer|
    customer\s+success|customer\s+experience|
    people\s+analytics|customer\s+account|
    \binsurtech\b|insurance\s+tech""",
    re.IGNORECASE | re.VERBOSE,
)


def step_prefilter(items: list[dict], run_log_dir: Path,
                   logger: logging.Logger) -> list[dict]:
    """Reject titles that are obviously outside the target profile.

    Runs before LLM extraction to avoid spending ~3s/item on listings like
    'Vibration Engineer' or 'AI Trainer'. Only excludes clear mismatches;
    ambiguous titles pass through for the LLM to judge.

    Saves a structured log to run_log_dir/prefilter.json for auditing.
    """
    kept = []
    filtered_log = []
    for item in items:
        title = item["title"]
        match = _EXCLUDE_PATTERNS.search(title)
        if match and not _PROTECT_PATTERNS.search(title):
            logger.debug(f"Prefilter: skipping {title} @ {item['company']}")
            filtered_log.append({
                "item_id": item["item_id"],
                "title": title,
                "company": item["company"],
                "matched_pattern": match.group(),
            })
            continue
        kept.append(item)

    if filtered_log:
        logger.info(f"Prefilter: skipped {len(filtered_log)} obviously irrelevant title(s)")
    logger.info(f"Prefilter: {len(kept)} item(s) to extract")

    save_json(run_log_dir / "prefilter.json", {
        "total_input": len(items),
        "kept": len(kept),
        "filtered": len(filtered_log),
        "filtered_items": filtered_log,
    })

    return kept


_TRIAGE_SYSTEM_PROMPT = """You are a job listing classifier. Given a job listing, decide if it is relevant to this profile:

Target roles: Customer Experience, Customer Success, Product Management, Customer Operations, Analytics
Target seniority: Manager through Director level
Target domains: SaaS, B2B technology, AI/ML, developer tools

Respond with ONLY a JSON object: {"relevant": true} or {"relevant": false}
Do not explain. Do not add any other text."""

_TRIAGE_MODEL = "qwen3.5:2b"


def step_triage(items: list[dict], config: dict, run_log_dir: Path,
                logger: logging.Logger) -> tuple[list[dict], list[dict]]:
    """Cheap 2B binary gate: reject obviously irrelevant items before extraction.

    Returns (kept_items, rejected_items). Both lists are logged for training data.
    """
    base_url = config.get("ollama", {}).get("base_url", "http://localhost:11434")
    kept = []
    rejected = []
    triage_log = []

    logger.info(f"Triage: classifying {len(items)} item(s) with {_TRIAGE_MODEL}")

    for i, item in enumerate(items):
        prompt = f"Title: {item['title']}\nCompany: {item['company']}"
        payload = {
            "model": _TRIAGE_MODEL,
            "messages": [
                {"role": "system", "content": _TRIAGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "format": TRIAGE_SCHEMA,
            "stream": False,
            "think": False,
            "options": {"temperature": 0},
        }

        start = time.time()
        try:
            resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=30)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            parsed = json.loads(content)
            relevant = parsed.get("relevant", True)  # default to relevant on parse error
        except Exception as e:
            logger.warning(f"Triage: error on {item['title']}, keeping item: {e}")
            relevant = True  # fail-open: let extraction handle it

        elapsed = time.time() - start

        entry = {
            "item_id": item["item_id"],
            "title": item["title"],
            "company": item["company"],
            "relevant": relevant,
            "latency_s": round(elapsed, 2),
        }
        triage_log.append(entry)

        if relevant:
            kept.append(item)
        else:
            rejected.append(item)
            logger.debug(f"Triage: rejected {item['title']} @ {item['company']} ({elapsed:.1f}s)")

    logger.info(
        f"Triage: {len(kept)} relevant, {len(rejected)} rejected "
        f"({len(rejected)*100//len(items)}% filtered)"
    )

    save_json(run_log_dir / "triage.json", {
        "model": _TRIAGE_MODEL,
        "total_input": len(items),
        "kept": len(kept),
        "rejected": len(rejected),
        "items": triage_log,
    })

    return kept, rejected


def step_extract(
    items: list[dict], plan: dict, config: dict, run_log_dir: Path,
    logger: logging.Logger, router: Router, model_config: dict
) -> tuple[list[dict], list[dict], list[float]]:
    """Run LLM extraction with prefix caching + confidence escalation.

    Returns (results, logprob_entries, item_times).
    """
    extract_step = next(s for s in plan["steps"] if s["id"] == "extract")
    template = load_prompt_template(extract_step["prompt_template"])
    model = model_config["ollama_name"]

    # Split template on delimiter for prefix caching
    delimiter = "--- ITEM TO PROCESS ---"
    parts = template.split(delimiter, 1)
    static_prefix = parts[0]
    per_item_template = parts[1] if len(parts) > 1 else "\n\n{{listing}}"

    logger.info(f"Extract: model={model_config['id']} | static prefix = {len(static_prefix)} chars")

    results = []
    prompts_log = []
    logprob_entries = []
    item_times = []

    for i, item in enumerate(items):
        logger.info(f"Extracting [{i+1}/{len(items)}]: {item['title']} @ {item['company']}")

        listing_text = (
            f"Title: {item['title']}\n"
            f"Company: {item['company']}\n"
            f"URL: {item['url']}\n\n"
            f"{item['description']}"
        )

        item_content = per_item_template.replace("{{listing}}", listing_text)
        full_prompt = static_prefix + delimiter + item_content

        t0 = time.time()
        response_data = call_ollama(full_prompt, model, logger, config,
                                    format_schema=EXTRACTION_SCHEMA)
        elapsed = time.time() - t0
        item_times.append(elapsed)

        raw_response = response_data.get("response", "") if response_data else None

        # Collect logprobs
        lp_stats = None
        if response_data:
            lp_stats = extract_logprob_stats(response_data, logger)
            if lp_stats:
                logprob_entries.append({
                    "item_id": item["item_id"],
                    "step_id": "extract",
                    "model": model,
                    **lp_stats,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            elif i == 0:
                logger.warning("Logprobs not available in Ollama response (may need newer Ollama version)")

        # --- Confidence escalation check ---
        escalation = router.should_escalate("extract", lp_stats, model_config)
        escalated = False
        escalation_response = None

        if escalation["escalate"]:
            if escalation["mode"] == "active":
                # Actually re-run on the stronger model
                target = router.get_escalation_target()
                logger.info(
                    f"  ESCALATING to {target['id']}: {escalation['reason']}"
                )
                t0_esc = time.time()
                escalation_response = call_ollama(
                    full_prompt, target["ollama_name"], logger, config,
                    format_schema=EXTRACTION_SCHEMA,
                )
                elapsed_esc = time.time() - t0_esc
                item_times[-1] += elapsed_esc  # total time includes both passes

                if escalation_response:
                    raw_response = escalation_response.get("response", "")
                    model_used = target["ollama_name"]
                    escalated = True
                    logger.info(f"  Escalation complete ({elapsed_esc:.1f}s)")
                else:
                    logger.warning("  Escalation call failed, using original output")
            else:
                # logging_only mode: just log what would have happened
                logger.info(
                    f"  WOULD ESCALATE (logging_only): {escalation['reason']}"
                )

        router.record_escalation_decision(
            "extract",
            escalated=escalation["escalate"],
            circuit_breaker_prevented=(not escalation["escalate"] and "circuit breaker" in escalation["reason"]),
        )

        # Log prompt and response (include escalation info)
        log_entry = {
            "item_id": item["item_id"],
            "step_id": "extract",
            "prompt": full_prompt,
            "raw_response": raw_response,
            "model": model,
            "model_id": model_config["id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "llm_time_s": round(elapsed, 2),
        }
        if escalation["escalate"]:
            log_entry["escalation"] = {
                "triggered": True,
                "reason": escalation["reason"],
                "mode": escalation["mode"],
                "metric_value": escalation["metric_value"],
                "threshold": escalation["threshold"],
            }
            if escalated and router.escalation_cfg.get("log_both_outputs"):
                log_entry["escalation"]["cheap_model_output"] = (
                    response_data.get("response", "") if response_data else None
                )
                log_entry["escalation"]["escalated_model"] = router.get_escalation_target()["id"]
                log_entry["escalation"]["escalated_output"] = raw_response
        prompts_log.append(log_entry)

        if raw_response is None:
            logger.error(f"  LLM call failed, skipping item")
            continue

        extracted = parse_json_response(raw_response, logger)
        if extracted is None:
            logger.error(f"  Failed to parse extraction response, skipping item")
            continue

        item["extracted"] = extracted
        results.append(item)
        model_tag = f" [escalated→{router.get_escalation_target()['id']}]" if escalated else ""
        logger.info(
            f"  Extracted: {extracted.get('seniority', '?')} / "
            f"{extracted.get('domain', '?')} / "
            f"{extracted.get('remote_policy', '?')} ({elapsed:.1f}s){model_tag}"
        )

    # Save extraction logs
    save_json(run_log_dir / "extract_prompts.json", prompts_log)

    if item_times:
        logger.info(
            f"Extraction: {len(results)}/{len(items)} succeeded | "
            f"avg {statistics.mean(item_times):.1f}s/item | "
            f"total {sum(item_times):.1f}s"
        )
    else:
        logger.info(f"Extraction: {len(results)}/{len(items)} succeeded")

    return results, logprob_entries, item_times


DIMENSION_ANCHORS = {
    "title_match": [0.0, 0.4, 0.7, 1.0],
    "domain_match": [0.0, 0.3, 0.7, 1.0],
    "seniority_match": [0.0, 0.5, 1.0],
    "compensation_fit": [0.2, 0.5, 0.7, 1.0],
}


def snap_dimension_scores(scoring: dict, logger: logging.Logger) -> dict:
    """Snap dimension scores to nearest valid rubric anchor.

    requirements_fit is continuous (fraction) and not snapped.
    Recalculates total_score from snapped dimensions + bonuses/penalties.
    """
    scores = scoring.get("scores", {})
    snapped_any = False

    for dim, anchors in DIMENSION_ANCHORS.items():
        dim_info = scores.get(dim)
        if dim_info is None:
            continue
        raw = dim_info.get("score") if isinstance(dim_info, dict) else dim_info
        if raw is None:
            continue
        nearest = min(anchors, key=lambda a: abs(a - raw))
        if nearest != raw:
            logger.info(f"  Snap {dim}: {raw} -> {nearest}")
            if isinstance(dim_info, dict):
                dim_info["score"] = nearest
                dim_info["pre_snap"] = raw
            else:
                scores[dim] = nearest
            snapped_any = True

    if snapped_any:
        scoring["scores"] = scores
        scoring["total_score"] = _recalc_total(scoring)

    return scoring


def _recalc_total(scoring: dict) -> float:
    """Recalculate total_score from dimension scores, bonuses, and penalties."""
    weights = {
        "title_match": 0.25, "domain_match": 0.15, "seniority_match": 0.25,
        "requirements_fit": 0.20, "compensation_fit": 0.15,
    }
    scores = scoring.get("scores", {})
    weighted = 0.0
    for dim, w in weights.items():
        dim_info = scores.get(dim)
        if dim_info is None:
            continue
        val = dim_info.get("score") if isinstance(dim_info, dict) else dim_info
        if val is not None:
            weighted += val * w

    for bonus in scoring.get("bonuses_applied", []):
        weighted += bonus.get("bonus", 0)
    for penalty in scoring.get("penalties_applied", []):
        weighted += penalty.get("penalty", 0)

    return round(max(0.0, min(1.0, weighted)), 4)


def step_score(
    items: list[dict], plan: dict, profile: str, config: dict,
    run_log_dir: Path, logger: logging.Logger, router: Router, model_config: dict
) -> tuple[list[dict], list[dict], list[float]]:
    """Run LLM scoring with prefix caching + confidence escalation.

    Returns (results, logprob_entries, item_times).
    """
    score_step = next(s for s in plan["steps"] if s["id"] == "score")
    template = load_prompt_template(score_step["prompt_template"])
    model = model_config["ollama_name"]

    # Inject static variables: scoring config and profile (all part of cached prefix)
    template = inject_variables(template, plan)
    template = template.replace("{{profile}}", profile)

    # Split on delimiter for prefix caching
    delimiter = "--- ITEM TO PROCESS ---"
    parts = template.split(delimiter, 1)
    static_prefix = parts[0]
    per_item_template = parts[1] if len(parts) > 1 else "\n\n{{extracted}}"

    logger.info(f"Score: model={model_config['id']} | static prefix = {len(static_prefix)} chars")

    results = []
    prompts_log = []
    logprob_entries = []
    item_times = []

    for i, item in enumerate(items):
        logger.info(f"Scoring [{i+1}/{len(items)}]: {item['title']} @ {item['company']}")

        extracted_json = json.dumps(item["extracted"], indent=2)
        item_content = per_item_template.replace("{{extracted}}", extracted_json)
        full_prompt = static_prefix + delimiter + item_content

        t0 = time.time()
        response_data = call_ollama(full_prompt, model, logger, config,
                                    format_schema=SCORING_SCHEMA)
        elapsed = time.time() - t0
        item_times.append(elapsed)

        raw_response = response_data.get("response", "") if response_data else None

        # Collect logprobs
        lp_stats = None
        if response_data:
            lp_stats = extract_logprob_stats(response_data, logger)
            if lp_stats:
                logprob_entries.append({
                    "item_id": item["item_id"],
                    "step_id": "score",
                    "model": model,
                    **lp_stats,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            elif i == 0:
                logger.warning("Logprobs not available in Ollama response for scoring step")

        # --- Confidence escalation check ---
        escalation = router.should_escalate("score", lp_stats, model_config)
        escalated = False

        if escalation["escalate"]:
            if escalation["mode"] == "active":
                target = router.get_escalation_target()
                logger.info(
                    f"  ESCALATING to {target['id']}: {escalation['reason']}"
                )
                t0_esc = time.time()
                escalation_response = call_ollama(
                    full_prompt, target["ollama_name"], logger, config,
                    format_schema=SCORING_SCHEMA,
                )
                elapsed_esc = time.time() - t0_esc
                item_times[-1] += elapsed_esc

                if escalation_response:
                    raw_response = escalation_response.get("response", "")
                    escalated = True
                    logger.info(f"  Escalation complete ({elapsed_esc:.1f}s)")
                else:
                    logger.warning("  Escalation call failed, using original output")
            else:
                logger.info(
                    f"  WOULD ESCALATE (logging_only): {escalation['reason']}"
                )

        router.record_escalation_decision(
            "score",
            escalated=escalation["escalate"],
            circuit_breaker_prevented=(not escalation["escalate"] and "circuit breaker" in escalation["reason"]),
        )

        # Log prompt and response
        log_entry = {
            "item_id": item["item_id"],
            "step_id": "score",
            "prompt": full_prompt,
            "raw_response": raw_response,
            "model": model,
            "model_id": model_config["id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "llm_time_s": round(elapsed, 2),
        }
        if escalation["escalate"]:
            log_entry["escalation"] = {
                "triggered": True,
                "reason": escalation["reason"],
                "mode": escalation["mode"],
                "metric_value": escalation["metric_value"],
                "threshold": escalation["threshold"],
            }
            if escalated and router.escalation_cfg.get("log_both_outputs"):
                log_entry["escalation"]["cheap_model_output"] = (
                    response_data.get("response", "") if response_data else None
                )
                log_entry["escalation"]["escalated_model"] = router.get_escalation_target()["id"]
                log_entry["escalation"]["escalated_output"] = raw_response
        prompts_log.append(log_entry)

        if raw_response is None:
            logger.error(f"  LLM call failed, skipping item")
            continue

        scoring = parse_json_response(raw_response, logger)
        if scoring is None:
            logger.error(f"  Failed to parse scoring response, skipping item")
            continue

        # Validate total_score exists
        if "total_score" not in scoring:
            logger.error(f"  Scoring response missing total_score, skipping item")
            continue

        # Snap dimension scores to valid rubric anchors
        scoring = snap_dimension_scores(scoring, logger)

        item["scoring"] = scoring
        model_tag = f" [escalated→{router.get_escalation_target()['id']}]" if escalated else ""
        results.append(item)
        logger.info(f"  Score: {scoring['total_score']:.3f} ({elapsed:.1f}s){model_tag}")

    save_json(run_log_dir / "score_prompts.json", prompts_log)

    if item_times:
        logger.info(
            f"Scoring: {len(results)}/{len(items)} succeeded | "
            f"avg {statistics.mean(item_times):.1f}s/item | "
            f"total {sum(item_times):.1f}s"
        )
    else:
        logger.info(f"Scoring: {len(results)}/{len(items)} succeeded")

    return results, logprob_entries, item_times


FLOOR_PENALTIES_PATH = STATE_DIR / "floor_penalties.json"


def apply_floor_penalties(items: list[dict], plan: dict, logger: logging.Logger) -> list[dict]:
    """Apply floor penalties after LLM scoring.

    Two types of penalties:
    1. Dimension floors: if a scoring dimension falls below its threshold,
       subtract the penalty from total_score.
    2. Extraction floors: if an extracted field matches a specific value,
       subtract the penalty from total_score.

    Logs all activations to state/floor_penalties.json for review.
    """
    dim_floors = plan.get("scoring", {}).get("dimension_floors", [])
    ext_floors = plan.get("scoring", {}).get("extraction_floors", [])
    if not dim_floors and not ext_floors:
        return items

    activations = []

    for item in items:
        scoring = item.get("scoring", {})
        scores = scoring.get("scores", {})
        extracted = item.get("extracted", {})
        original_total = scoring.get("total_score", 0)
        cumulative_penalty = 0.0

        for floor in dim_floors:
            dim = floor["dimension"]
            threshold = floor["threshold"]
            penalty = floor["penalty"]

            dim_info = scores.get(dim, {})
            dim_score = dim_info.get("score") if isinstance(dim_info, dict) else dim_info
            if dim_score is None:
                continue

            if dim_score < threshold:
                cumulative_penalty += penalty
                activations.append({
                    "item_id": item["item_id"],
                    "title": item.get("title", ""),
                    "company": item.get("company", ""),
                    "dimension": dim,
                    "dimension_score": dim_score,
                    "threshold": threshold,
                    "penalty": penalty,
                    "original_total": round(original_total, 3),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        for floor in ext_floors:
            field = floor["field"]
            match_val = floor["match"]
            penalty = floor["penalty"]
            field_val = extracted.get(field)
            if field_val is None:
                continue
            if str(field_val).lower() == str(match_val).lower():
                cumulative_penalty += penalty
                activations.append({
                    "item_id": item["item_id"],
                    "title": item.get("title", ""),
                    "company": item.get("company", ""),
                    "dimension": f"{field}={match_val}",
                    "dimension_score": field_val,
                    "threshold": match_val,
                    "penalty": penalty,
                    "original_total": round(original_total, 3),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        if cumulative_penalty != 0.0:
            new_total = max(0.0, min(1.0, original_total + cumulative_penalty))
            scoring["total_score"] = round(new_total, 3)
            scoring["floor_penalties_applied"] = round(cumulative_penalty, 3)
            logger.info(
                f"  Floor penalty: {item['title']} @ {item['company']} "
                f"{original_total:.3f} → {new_total:.3f} ({cumulative_penalty:+.3f})"
            )

    if activations:
        # Append to existing log
        existing = load_json(FLOOR_PENALTIES_PATH)
        if not isinstance(existing, list):
            existing = []
        existing.extend(activations)
        save_json(FLOOR_PENALTIES_PATH, existing)
        logger.info(
            f"Floor penalties: {len(activations)} activation(s) on "
            f"{len({a['item_id'] for a in activations})} item(s), "
            f"logged to {FLOOR_PENALTIES_PATH.name}"
        )
    else:
        logger.info("Floor penalties: no activations (all dimensions above thresholds)")

    return items


def step_rank_and_filter(items: list[dict], plan: dict, logger: logging.Logger) -> list[dict]:
    """Filter by min_score, sort descending, cap at max_items."""
    rank_step = next(s for s in plan["steps"] if s["id"] == "rank_and_filter")
    min_score = rank_step.get("min_score", 0.6)
    max_items = rank_step.get("max_items", 15)

    filtered = [it for it in items if it["scoring"]["total_score"] >= min_score]
    filtered.sort(key=lambda x: x["scoring"]["total_score"], reverse=True)
    filtered = filtered[:max_items]

    logger.info(
        f"Rank & filter: {len(filtered)}/{len(items)} passed "
        f"(min_score={min_score}, max_items={max_items})"
    )
    return filtered


def step_output(items: list[dict], run_id: str, logger: logging.Logger) -> tuple[Path, list[str]]:
    """Write markdown briefing. Returns (path, list of briefed item_ids).

    Skips items already briefed in a previous run (have briefed_at in seen.json).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    briefing_path = OUTPUT_DIR / f"briefing-{date_str}.md"

    # Filter out items already briefed in previous runs
    seen = load_json(SEEN_PATH)
    already_briefed = {
        entry["item_id"] for entry in seen
        if isinstance(entry, dict) and entry.get("briefed_at")
    }
    new_items = [it for it in items if it["item_id"] not in already_briefed]
    skipped = len(items) - len(new_items)
    if skipped:
        logger.info(f"Briefing dedup: skipped {skipped} previously-briefed item(s)")

    lines = [
        f"# Job Briefing — {date_str}",
        f"",
        f"**Run ID**: `{run_id}`  ",
        f"**Items scored**: {len(new_items)}  ",
        f"**Generated**: {datetime.now(timezone.utc).isoformat()}",
        f"",
        f"---",
        f"",
    ]

    briefed_ids = []
    items = new_items

    if not items:
        lines.append("No listings passed the score threshold today.")
    else:
        for rank, item in enumerate(items, 1):
            briefed_ids.append(item["item_id"])
            scoring = item["scoring"]
            extracted = item["extracted"]
            total = scoring["total_score"]
            comp = extracted.get("compensation", {})

            lines.append(f"## #{rank} — {extracted.get('title', item['title'])} @ {extracted.get('company', item['company'])}")
            lines.append(f"")
            lines.append(f"**Item ID**: `{item['item_id']}`  ")
            lines.append(f"**Score**: {total:.3f}  ")
            lines.append(f"**URL**: {item['url']}  ")
            lines.append(f"**Seniority**: {extracted.get('seniority', 'unknown')}  ")
            lines.append(f"**Remote**: {extracted.get('remote_policy', 'unknown')}  ")
            lines.append(f"**Domain**: {extracted.get('domain', 'unknown')}  ")
            lines.append(f"")

            # Compensation
            if comp.get("min") or comp.get("max"):
                comp_str = ""
                if comp.get("min") and comp.get("max"):
                    comp_str = f"${comp['min']:,.0f} – ${comp['max']:,.0f}"
                elif comp.get("min"):
                    comp_str = f"${comp['min']:,.0f}+"
                elif comp.get("max"):
                    comp_str = f"Up to ${comp['max']:,.0f}"
                if comp.get("currency") and comp["currency"] != "USD":
                    comp_str += f" {comp['currency']}"
                if comp.get("equity_mentioned"):
                    comp_str += " + equity"
                lines.append(f"**Compensation**: {comp_str}  ")
                lines.append(f"")

            # Key requirements
            reqs = extracted.get("key_requirements", [])
            if reqs:
                lines.append(f"**Key Requirements**: {', '.join(reqs)}  ")
                lines.append(f"")

            # Score breakdown
            lines.append(f"<details><summary>Score breakdown</summary>")
            lines.append(f"")
            scores = scoring.get("scores", {})
            for dim, info in scores.items():
                if isinstance(info, dict):
                    lines.append(f"- **{dim}**: {info.get('score', 'N/A')} — {info.get('reasoning', '')}")
                else:
                    lines.append(f"- **{dim}**: {info}")
            bonuses = scoring.get("bonuses_applied", [])
            if bonuses:
                lines.append(f"- **Bonuses**: {', '.join(b['condition'] + ' (+' + str(b['bonus']) + ')' for b in bonuses)}")
            penalties = scoring.get("penalties_applied", [])
            if penalties:
                lines.append(f"- **Penalties**: {', '.join(p['condition'] + ' (' + str(p['penalty']) + ')' for p in penalties)}")
            lines.append(f"")
            lines.append(f"</details>")
            lines.append(f"")
            lines.append(f"---")
            lines.append(f"")

    briefing_path.write_text("\n".join(lines))
    logger.info(f"Briefing written to {briefing_path}")
    return briefing_path, briefed_ids


def step_enrich_queue(all_scored: list[dict], logger: logging.Logger) -> Path | None:
    """Write enrich-queue.json for low-signal items scored 0.4–0.7.

    These are LinkedIn alert listings (signal_level='low') that landed in an
    uncertain scoring range — worth enriching with full descriptions.
    """
    queue = []
    for item in all_scored:
        if item.get("signal_level") != "low":
            continue
        scoring = item.get("scoring", {})
        total = scoring.get("total_score", 0)
        if 0.4 <= total <= 0.7:
            queue.append({
                "item_id": item["item_id"],
                "title": item["title"],
                "company": item["company"],
                "url": item["url"],
                "score": round(total, 3),
                "source_file": item.get("source_file", ""),
            })

    if not queue:
        logger.info("Enrich queue: no low-signal items in 0.4–0.7 range")
        return None

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    queue_path = STATE_DIR / "enrich-queue.json"
    # Merge with existing queue (don't lose items from previous runs)
    existing = load_json(queue_path)
    if not isinstance(existing, list):
        existing = []
    existing_ids = {e["item_id"] for e in existing}
    for entry in queue:
        if entry["item_id"] not in existing_ids:
            existing.append(entry)
    save_json(queue_path, existing)
    logger.info(f"Enrich queue: {len(queue)} new item(s) added ({len(existing)} total in queue)")
    return queue_path


def step_update_ledger(items: list[dict], all_processed: list[dict],
                       briefed_ids: list[str], logger: logging.Logger) -> None:
    """Append all processed items (not just filtered ones) to seen.json.

    Items whose item_id is in briefed_ids get a briefed_at timestamp so they
    won't appear in future briefings (dedup across days).
    """
    briefed_set = set(briefed_ids)
    now = datetime.now(timezone.utc).isoformat()
    seen = load_json(SEEN_PATH)
    for item in all_processed:
        entry = {
            "item_id": item["item_id"],
            "title": item["title"],
            "company": item["company"],
            "url": item["url"],
            "processed_at": now,
        }
        if item["item_id"] in briefed_set:
            entry["briefed_at"] = now
        seen.append(entry)
    save_json(SEEN_PATH, seen)
    logger.info(f"Ledger updated: {len(all_processed)} item(s) added to seen.json ({len(briefed_set)} briefed)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Perian pipeline runner")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max items to process per run (0 = unlimited). Used for chunked re-runs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_start = time.time()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger = setup_logging(run_id)
    run_log_dir = RUN_LOG_DIR / run_id

    logger.info(f"=== Perian run {run_id} ===")

    # Load config
    plan = load_plan()
    profile = load_profile()
    config = load_config()
    logger.info(f"Plan loaded: {len(plan['steps'])} steps")

    # Initialize router — reads fleet.yaml + router.yaml
    router = Router(logger)
    logger.info(
        f"Router: {len(router.fleet.get('models', []))} models, "
        f"escalation mode={router.escalation_cfg['mode']}, "
        f"threshold={router.escalation_cfg['threshold']}"
    )

    # Step 1: Ingest
    items = step_ingest(plan, logger)
    if not items:
        logger.info("No items to process. Exiting.")
        save_json(run_log_dir / "summary.json", {
            "run_id": run_id,
            "status": "empty",
            "items_ingested": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return

    # Step 2: Dedup
    deduped_items = step_dedup(items, logger)
    if not deduped_items:
        logger.info("All items already seen. Exiting.")
        save_json(run_log_dir / "summary.json", {
            "run_id": run_id,
            "status": "all_deduped",
            "items_ingested": len(items),
            "items_new": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return

    # Step 2b: Pre-filter obviously irrelevant titles
    new_items = step_prefilter(deduped_items, run_log_dir, logger)
    if not new_items:
        logger.info("All items filtered by prefilter. Exiting.")
        save_json(run_log_dir / "summary.json", {
            "run_id": run_id,
            "status": "all_prefiltered",
            "items_ingested": len(items),
            "items_deduped": len(deduped_items),
            "items_prefiltered": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return

    # Apply --limit: cap items for chunked re-runs
    if args.limit and len(new_items) > args.limit:
        logger.info(f"Limit: processing {args.limit} of {len(new_items)} items (--limit {args.limit})")
        new_items = new_items[:args.limit]

    # Step 2c: Triage — cheap 2B binary gate before expensive extraction
    new_items, triage_rejected = step_triage(new_items, config, run_log_dir, logger)
    if not new_items:
        logger.info("All items rejected by triage. Exiting.")
        save_json(run_log_dir / "summary.json", {
            "run_id": run_id,
            "status": "all_triaged",
            "items_ingested": len(items),
            "items_deduped": len(deduped_items),
            "items_triaged_out": len(triage_rejected),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return

    # --- Route LLM steps ---
    # The router picks the cheapest qualifying model for each step based on
    # its declared capability and min_reliability. This replaces the hardcoded
    # model field from Phase 1.
    llm_steps = [s for s in plan["steps"] if s["type"] == "llm_local"]
    batches = router.plan_model_batches(llm_steps)

    logger.info(f"Model batches: {' → '.join(b['model']['id'] for b in batches)}")

    # Process each model batch with proper keep-alive management
    extract_logprobs = []
    extract_times = []
    extracted = new_items  # starts as new_items, refined by extract step
    score_logprobs = []
    score_times = []
    scored = []

    for batch_idx, batch in enumerate(batches):
        batch_model = batch["model"]
        batch_steps = batch["steps"]

        # Warm up this batch's model (pin in memory)
        warm_up_model(batch_model["ollama_name"], config, logger)

        for step in batch_steps:
            if step["id"] == "extract":
                extracted, extract_logprobs, extract_times = step_extract(
                    extracted, plan, config, run_log_dir, logger,
                    router, batch_model,
                )
            elif step["id"] == "score":
                scored, score_logprobs, score_times = step_score(
                    extracted, plan, profile, config, run_log_dir, logger,
                    router, batch_model,
                )

        # Cool down: unload immediately if next batch uses a different model,
        # or set a timeout if this is the last batch
        if batch["is_last"]:
            cool_down_model(batch_model["ollama_name"], config, logger)
        else:
            next_model = batches[batch_idx + 1]["model"]
            if next_model["id"] != batch_model["id"]:
                # Unload immediately to free VRAM for next model
                logger.info(
                    f"Unloading {batch_model['id']} before switching to {next_model['id']}"
                )
                _unload_model(batch_model["ollama_name"], config, logger)

    # Step 4b: Apply dimension floor penalties (post-LLM guardrail)
    apply_floor_penalties(scored, plan, logger)

    # Step 5: Rank and filter
    ranked = step_rank_and_filter(scored, plan, logger)

    # Step 6: Output briefing
    briefing_path, briefed_ids = step_output(ranked, run_id, logger)

    # Step 6b: Enrich queue (low-signal items needing full descriptions)
    step_enrich_queue(scored, logger)

    # Step 7: Update ledger (all scored items + triage-rejected items)
    step_update_ledger(ranked, scored + triage_rejected, briefed_ids, logger)

    # ---------------------------------------------------------------------------
    # Logprob collection
    # ---------------------------------------------------------------------------
    all_logprobs = extract_logprobs + score_logprobs
    if all_logprobs:
        existing = load_json(LOGPROBS_PATH)
        if not isinstance(existing, list):
            existing = []
        existing.extend(all_logprobs)
        save_json(LOGPROBS_PATH, existing)
        logger.info(f"Logprobs: {len(all_logprobs)} entries appended to logprobs.json")

    # ---------------------------------------------------------------------------
    # Metrics summary
    # ---------------------------------------------------------------------------
    all_times = extract_times + score_times
    total_time = time.time() - run_start

    metrics = load_json(METRICS_PATH)
    if not isinstance(metrics, dict):
        metrics = {}
    metrics["last_run"] = datetime.now(timezone.utc).isoformat()

    # Logprob summary
    if all_logprobs:
        means = [e["mean_logprob"] for e in all_logprobs]
        sorted_means = sorted(means)

        if len(sorted_means) >= 2:
            p25 = statistics.quantiles(sorted_means, n=4)[0]
        else:
            p25 = sorted_means[0]

        metrics["logprob_summary"] = {
            "mean_confidence": round(statistics.mean(means), 4),
            "p25_confidence": round(p25, 4),
            "items_below_neg3": sum(1 for e in all_logprobs if e["min_logprob"] < -3.0),
            "total_items": len(all_logprobs),
            "by_step": {},
        }
        for step_id in ("extract", "score"):
            step_entries = [e for e in all_logprobs if e["step_id"] == step_id]
            if step_entries:
                metrics["logprob_summary"]["by_step"][step_id] = {
                    "mean": round(statistics.mean(e["mean_logprob"] for e in step_entries), 4),
                    "min": round(min(e["min_logprob"] for e in step_entries), 4),
                    "count": len(step_entries),
                }

    # Escalation metrics (Phase 2)
    metrics["escalation"] = router.get_escalation_metrics()

    # Routing summary — which model was used per step
    metrics["routing"] = {}
    for step in llm_steps:
        model_cfg = router.get_model(step)
        metrics["routing"][step["id"]] = {
            "model": model_cfg["id"],
            "ollama_name": model_cfg["ollama_name"],
            "capability": step.get("capability", "none"),
            "min_reliability": step.get("min_reliability"),
        }

    # Timing summary
    metrics["timing"] = {
        "total_run_s": round(total_time, 1),
        "extract_total_s": round(sum(extract_times), 1) if extract_times else 0,
        "extract_avg_s": round(statistics.mean(extract_times), 1) if extract_times else 0,
        "score_total_s": round(sum(score_times), 1) if score_times else 0,
        "score_avg_s": round(statistics.mean(score_times), 1) if score_times else 0,
        "llm_total_s": round(sum(all_times), 1) if all_times else 0,
        "llm_avg_s": round(statistics.mean(all_times), 1) if all_times else 0,
        "llm_calls": len(all_times),
    }

    save_json(METRICS_PATH, metrics)

    # Save run summary
    summary = {
        "run_id": run_id,
        "status": "complete",
        "items_ingested": len(items),
        "items_deduped": len(deduped_items),
        "items_prefiltered": len(deduped_items) - len(new_items),
        "items_new": len(new_items),
        "items_extracted": len(extracted),
        "items_scored": len(scored),
        "items_output": len(ranked),
        "briefing_path": str(briefing_path),
        "routing": metrics["routing"],
        "escalation": metrics["escalation"],
        "timing": metrics["timing"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_json(run_log_dir / "summary.json", summary)

    # Final summary log
    if all_times:
        esc = metrics["escalation"]
        esc_info = ""
        if esc["total_items"]:
            esc_info = (
                f" | escalation: {esc['escalated_count']}/{esc['total_items']} "
                f"({esc['escalation_rate']:.1%}, mode={esc['mode']})"
            )
        logger.info(
            f"Run complete: {len(ranked)} listings in briefing | "
            f"total {total_time:.1f}s | "
            f"LLM {sum(all_times):.1f}s ({len(all_times)} calls, avg {statistics.mean(all_times):.1f}s/call)"
            f"{esc_info}"
        )
    else:
        logger.info(f"Run complete: {len(ranked)} listings in briefing | total {total_time:.1f}s")


if __name__ == "__main__":
    main()
