"""Shared utilities for evaluation test runners.

Provides Ollama client, JSON parsing, prompt building, and logprob extraction.
Self-contained — duplicates the small helper functions from runner.py to avoid
fragile cross-directory imports.
"""

import json
import logging
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "prompts"
CONFIG_PATH = ROOT / "config" / "env.yaml"
PLAN_PATH = ROOT / "plan.yaml"
PROFILE_PATH = ROOT / "profile.md"

logger = logging.getLogger("perian.eval")


# ---------------------------------------------------------------------------
# Config / data loading
# ---------------------------------------------------------------------------

def load_config() -> dict:
    defaults = {
        "ollama": {
            "base_url": "http://localhost:11434",
            "num_ctx": 8192,
        }
    }
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        ollama = cfg.setdefault("ollama", {})
        for k, v in defaults["ollama"].items():
            ollama.setdefault(k, v)
        return cfg
    return defaults


def load_plan() -> dict:
    with open(PLAN_PATH) as f:
        return yaml.safe_load(f)


def load_profile() -> str:
    return PROFILE_PATH.read_text()


def load_prompt_template(name: str) -> str:
    """Load a prompt template by name (e.g., 'extract' -> prompts/extract.md)."""
    return (PROMPTS_DIR / f"{name}.md").read_text()


# ---------------------------------------------------------------------------
# Template rendering (mirrors runner.py's inject_variables)
# ---------------------------------------------------------------------------

def inject_variables(template: str, variables: dict) -> str:
    """Replace {{key}}, {{nested.key}}, and {{#each}} blocks."""
    each_pattern = re.compile(
        r"\{\{#each\s+([\w.]+)\}\}(.*?)\{\{/each\}\}", re.DOTALL
    )

    def expand_each(match):
        key_path = match.group(1)
        block = match.group(2)
        items = _resolve_path(variables, key_path)
        if not isinstance(items, list):
            return ""
        lines = []
        for item in items:
            line = block
            for k, v in item.items():
                line = line.replace("{{" + k + "}}", str(v))
            lines.append(line.strip())
        return "\n".join(lines)

    result = each_pattern.sub(expand_each, template)

    def replace_var(match):
        key_path = match.group(1)
        val = _resolve_path(variables, key_path)
        if val is None:
            return match.group(0)
        return str(val)

    result = re.sub(r"\{\{([\w.]+)\}\}", replace_var, result)
    return result


def _resolve_path(data: dict, path: str):
    parts = path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

def call_ollama(prompt: str, model: str, config: dict, think: bool = False) -> dict | None:
    """Call Ollama /api/chat with logprobs=true.

    Args:
        think: Enable internal chain-of-thought reasoning. Only sent to models
               that support it (Qwen, DeepSeek). Ignored for other families
               (Gemma, Mistral, etc.).

    Returns the full response dict with data["response"] normalized to contain
    the text output (Ollama /api/chat puts it in data["message"]["content"]).
    """
    ollama_cfg = config.get("ollama", {})
    base_url = ollama_cfg.get("base_url", "http://localhost:11434")
    num_ctx = ollama_cfg.get("num_ctx", 8192)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "logprobs": True,
        },
    }

    # Send think parameter for models that support it — omit for models that don't
    model_lower = model.lower()
    if model_lower.startswith(("qwen", "deepseek", "gemma")):
        payload["think"] = think

    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        if "message" in data and "response" not in data:
            data["response"] = data["message"].get("content", "")

        # Extract token metrics from Ollama response (nanoseconds → seconds)
        token_metrics = {}
        eval_count = data.get("eval_count")
        eval_duration = data.get("eval_duration")  # nanoseconds
        prompt_eval_count = data.get("prompt_eval_count")
        prompt_eval_duration = data.get("prompt_eval_duration")  # nanoseconds

        if eval_count is not None:
            token_metrics["gen_tokens"] = eval_count
        if eval_duration and eval_duration > 0:
            token_metrics["gen_duration_s"] = round(eval_duration / 1e9, 4)
            if eval_count:
                token_metrics["gen_tok_s"] = round(eval_count / (eval_duration / 1e9), 2)
        if prompt_eval_count is not None:
            token_metrics["prompt_tokens"] = prompt_eval_count
        if prompt_eval_duration and prompt_eval_duration > 0:
            token_metrics["prompt_duration_s"] = round(prompt_eval_duration / 1e9, 4)
            if prompt_eval_count:
                token_metrics["prompt_tok_s"] = round(prompt_eval_count / (prompt_eval_duration / 1e9), 2)

        if token_metrics:
            data["token_metrics"] = token_metrics

        return data
    except requests.RequestException as e:
        logger.error(f"Ollama call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _fix_json_math(text: str) -> str:
    """Replace inline math expressions in JSON values with the final number."""
    lines = text.split("\n")
    fixed = []
    for line in lines:
        fixed.append(re.sub(
            r'("[^"]+":\s*)[^"]*=\s*([\d.]+)(\s*[,}]?\s*)$',
            r"\1\2\3",
            line,
        ))
    return "\n".join(fixed)


def parse_json_response(text: str) -> dict | None:
    """Extract JSON from an LLM response, handling fences and math expressions."""
    if not text:
        return None

    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    fixed = _fix_json_math(cleaned)
    if fixed != cleaned:
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if brace_match:
        candidate = brace_match.group()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        fixed = _fix_json_math(candidate)
        if fixed != candidate:
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

    logger.error(f"Failed to parse JSON: {text[:200]}...")
    return None


def extract_logprob_stats(response_data: dict) -> dict | None:
    """Extract logprob statistics from Ollama response."""
    if not response_data:
        return None

    logprobs_data = response_data.get("logprobs")
    if not logprobs_data:
        return None

    token_logprobs = None
    if isinstance(logprobs_data, dict):
        token_logprobs = (
            logprobs_data.get("token_logprobs")
            or logprobs_data.get("logprobs")
        )
    elif isinstance(logprobs_data, list):
        token_logprobs = logprobs_data

    if not token_logprobs or not isinstance(token_logprobs, list):
        return None

    valid = [lp for lp in token_logprobs if isinstance(lp, (int, float))]
    if not valid:
        return None

    return {
        "mean_logprob": round(statistics.mean(valid), 4),
        "min_logprob": round(min(valid), 4),
        "token_count": len(valid),
    }


# ---------------------------------------------------------------------------
# Prompt builders (use the actual pipeline prompt templates)
# ---------------------------------------------------------------------------

def build_extract_prompt(listing_text: str) -> str:
    """Build a full extraction prompt from raw listing text."""
    template = load_prompt_template("extract")
    parts = template.split("--- ITEM TO PROCESS ---")
    prefix = parts[0]
    suffix = parts[1] if len(parts) > 1 else "\n\n{{listing}}"
    item_content = suffix.replace("{{listing}}", listing_text)
    return prefix + "--- ITEM TO PROCESS ---" + item_content


def build_score_prompt(extracted_json: str, profile_text: str = None) -> str:
    """Build a full scoring prompt from extracted listing JSON.

    Args:
        profile_text: Override profile content. If None, loads the production
                      profile from profile.md.
    """
    template = load_prompt_template("score")
    plan = load_plan()
    profile = profile_text if profile_text is not None else load_profile()

    # Inject static variables into the template
    template = inject_variables(template, {"scoring": plan["scoring"]})
    template = template.replace("{{profile}}", profile)

    parts = template.split("--- ITEM TO PROCESS ---")
    prefix = parts[0]
    suffix = parts[1] if len(parts) > 1 else "\n\n{{extracted}}"
    item_content = suffix.replace("{{extracted}}", extracted_json)
    return prefix + "--- ITEM TO PROCESS ---" + item_content


# ---------------------------------------------------------------------------
# Adversarial input formatting
# ---------------------------------------------------------------------------

STRIPPED_PROFILE_PATH = PROMPTS_DIR / "profile-stripped.md"


def load_stripped_profile() -> str:
    """Load the generic (profile-stripped) profile text."""
    return STRIPPED_PROFILE_PATH.read_text()


def format_listing_adversarial(listing_text: str, format_type: str = "yaml") -> str:
    """Reformat a plain-text listing into an adversarial format.

    Tests whether the model handles varied input formats gracefully.

    Args:
        format_type: "yaml" (key-value), "markdown_table", or "sparse" (minimal).
    """
    lines = [l.strip() for l in listing_text.strip().splitlines() if l.strip()]

    if format_type == "yaml":
        yaml_lines = []
        for line in lines:
            if ":" in line and not line.startswith("http"):
                key, _, val = line.partition(":")
                yaml_lines.append(f"{key.strip().lower().replace(' ', '_')}: {val.strip()}")
            else:
                yaml_lines.append(f"content: {line}")
        return "\n".join(yaml_lines)

    elif format_type == "markdown_table":
        table_lines = ["| Field | Value |", "| --- | --- |"]
        for line in lines:
            if ":" in line and not line.startswith("http"):
                key, _, val = line.partition(":")
                table_lines.append(f"| {key.strip()} | {val.strip()} |")
            else:
                table_lines.append(f"| Info | {line} |")
        return "\n".join(table_lines)

    elif format_type == "sparse":
        # Strip headers and section markers, keep only data-bearing lines
        sparse = []
        for line in lines:
            stripped = line.lstrip("-•*# ")
            if stripped and len(stripped) > 5:
                sparse.append(stripped)
        return "\n".join(sparse)

    else:
        return listing_text


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def run_n_times(prompt: str, model: str, config: dict, n: int = 5,
                think: bool = False) -> list[dict]:
    """Run a prompt N times and collect results with timing and logprobs.

    Args:
        think: Enable qwen3's chain-of-thought reasoning for this run.

    Returns a list of result dicts, each containing:
      - parsed: the parsed JSON output (or None)
      - raw: the raw text response
      - logprob_stats: logprob statistics (or None)
      - time_s: wall clock time for the call
      - success: whether JSON parsing succeeded
    """
    results = []
    for i in range(n):
        t0 = time.time()
        response_data = call_ollama(prompt, model, config, think=think)
        elapsed = time.time() - t0

        if response_data is None:
            results.append({
                "parsed": None,
                "raw": None,
                "logprob_stats": None,
                "token_metrics": None,
                "time_s": elapsed,
                "success": False,
            })
            continue

        raw_text = response_data.get("response", "")
        parsed = parse_json_response(raw_text)
        lp_stats = extract_logprob_stats(response_data)
        tok_metrics = response_data.get("token_metrics")

        results.append({
            "parsed": parsed,
            "raw": raw_text,
            "logprob_stats": lp_stats,
            "token_metrics": tok_metrics,
            "time_s": elapsed,
            "success": parsed is not None,
        })

    return results


# Required fields for extraction output
EXTRACT_REQUIRED_FIELDS = [
    "title", "company", "seniority", "remote_policy",
    "domain", "key_requirements", "compensation",
]

VALID_SENIORITY = {"junior", "mid", "senior", "staff", "principal", "executive"}
VALID_REMOTE = {"remote", "hybrid", "onsite", "unknown"}


# ---------------------------------------------------------------------------
# Token metric aggregation
# ---------------------------------------------------------------------------

def aggregate_token_metrics(runs: list[dict]) -> dict | None:
    """Aggregate token metrics across multiple runs into mean values.

    Works with results from either run_n_times() or manual call_ollama() loops.
    Expects each run dict to have a 'token_metrics' key (or None).
    """
    metrics_list = [r.get("token_metrics") for r in runs if r.get("token_metrics")]
    if not metrics_list:
        return None

    agg = {}
    for key in ["gen_tokens", "gen_tok_s", "prompt_tokens", "prompt_tok_s",
                 "gen_duration_s", "prompt_duration_s"]:
        vals = [m[key] for m in metrics_list if key in m]
        if vals:
            agg[f"mean_{key}"] = round(statistics.mean(vals), 2)

    agg["n_samples"] = len(metrics_list)
    return agg


# ---------------------------------------------------------------------------
# System context for eval provenance
# ---------------------------------------------------------------------------

def get_system_context() -> dict:
    """Capture system context for eval run provenance."""
    context = {}

    # Ollama version
    try:
        result = subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0:
            # "ollama version is 0.20.0" → "0.20.0"
            version_text = result.stdout.strip()
            parts = version_text.split()
            context["ollama_version"] = parts[-1] if parts else version_text
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        context["ollama_version"] = "unknown"

    # KV cache settings from environment
    context["kv_cache_type"] = os.environ.get("OLLAMA_KV_CACHE_TYPE", "default")
    context["flash_attention"] = os.environ.get("OLLAMA_FLASH_ATTENTION", "0") == "1"

    # Eval code git hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
            cwd=str(ROOT),
        )
        if result.returncode == 0:
            context["eval_git_hash"] = result.stdout.strip()

        # Check if working tree is dirty
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", "evaluation/"],
            capture_output=True, text=True, timeout=3,
            cwd=str(ROOT),
        )
        if result.returncode == 0:
            context["eval_dirty"] = bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return context
