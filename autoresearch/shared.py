"""Shared utilities for autoresearch modules."""

import json
import sys
from pathlib import Path

import yaml

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.helpers import inject_variables


def load_autoresearch_config() -> dict:
    """Load autoresearch/config.yaml."""
    config_path = Path(__file__).resolve().parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def render_prompt(
    template_text: str,
    item_input,
    template_type: str,
    plan: dict = None,
    profile: str = None,
) -> str:
    """Render a prompt from template text + item input.

    Unlike build_extract_prompt()/build_score_prompt() in helpers.py,
    this takes the template as a string parameter — needed for evaluating
    candidate mutations without writing to disk.
    """
    if template_type == "extract":
        parts = template_text.split("--- ITEM TO PROCESS ---")
        prefix = parts[0]
        suffix = parts[1] if len(parts) > 1 else "\n\n{{listing}}"
        item_content = suffix.replace("{{listing}}", item_input)
        return prefix + "--- ITEM TO PROCESS ---" + item_content

    elif template_type == "score":
        rendered = inject_variables(template_text, {"scoring": plan["scoring"]})
        rendered = rendered.replace("{{profile}}", profile)
        parts = rendered.split("--- ITEM TO PROCESS ---")
        prefix = parts[0]
        suffix = parts[1] if len(parts) > 1 else "\n\n{{extracted}}"
        extracted_json = (
            json.dumps(item_input, indent=2)
            if isinstance(item_input, dict)
            else item_input
        )
        item_content = suffix.replace("{{extracted}}", extracted_json)
        return prefix + "--- ITEM TO PROCESS ---" + item_content

    else:
        raise ValueError(f"Unknown template type: {template_type}")
