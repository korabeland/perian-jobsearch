"""JSON Schema definitions for Ollama constrained decoding.

Ollama v0.5+ accepts a JSON Schema object in the `format` parameter,
enabling grammar-guided decoding that guarantees structurally valid output.
These schemas match the output specs in prompts/extract.md, prompts/score.md,
prompts/challenge.md, and the triage system prompt.
"""

# ---------------------------------------------------------------------------
# Triage: simple binary classification
# ---------------------------------------------------------------------------
TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {"type": "boolean"},
    },
    "required": ["relevant"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Extraction: structured job listing fields
# ---------------------------------------------------------------------------
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "company": {"type": "string"},
        "seniority": {
            "type": "string",
            "enum": ["junior", "mid", "senior", "staff", "principal", "executive"],
        },
        "remote_policy": {
            "type": "string",
            "enum": ["remote", "hybrid", "onsite", "unknown"],
        },
        "domain": {"type": "string"},
        "company_size": {
            "type": "string",
            "enum": ["startup", "smb", "midmarket", "enterprise", "unknown"],
        },
        "key_requirements": {
            "type": "array",
            "items": {"type": "string"},
        },
        "compensation": {
            "type": "object",
            "properties": {
                "min": {"type": ["number", "null"]},
                "max": {"type": ["number", "null"]},
                "currency": {"type": ["string", "null"]},
                "equity_mentioned": {"type": "boolean"},
            },
            "required": ["min", "max", "currency", "equity_mentioned"],
            "additionalProperties": False,
        },
    },
    "required": [
        "title", "company", "seniority", "remote_policy",
        "domain", "company_size", "key_requirements", "compensation",
    ],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Scoring: per-dimension scores with reasoning + totals
# ---------------------------------------------------------------------------
_DIMENSION_SCORE = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}

SCORING_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {
                "title_match": _DIMENSION_SCORE,
                "domain_match": _DIMENSION_SCORE,
                "seniority_match": _DIMENSION_SCORE,
                "requirements_fit": _DIMENSION_SCORE,
                "compensation_fit": _DIMENSION_SCORE,
            },
            "required": [
                "title_match", "domain_match", "seniority_match",
                "requirements_fit", "compensation_fit",
            ],
            "additionalProperties": False,
        },
        "bonuses_applied": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string"},
                    "bonus": {"type": "number"},
                },
                "required": ["condition", "bonus"],
                "additionalProperties": False,
            },
        },
        "penalties_applied": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string"},
                    "penalty": {"type": "number"},
                },
                "required": ["condition", "penalty"],
                "additionalProperties": False,
            },
        },
        "weighted_total": {"type": "number"},
        "total_score": {"type": "number"},
    },
    "required": [
        "scores", "bonuses_applied", "penalties_applied",
        "weighted_total", "total_score",
    ],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Aragorn challenge: adversarial review
# ---------------------------------------------------------------------------
CHALLENGE_SCHEMA = {
    "type": "object",
    "properties": {
        "agree": {"type": "boolean"},
        "adjusted_score": {"type": "number"},
        "challenges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dimension": {"type": "string"},
                    "issue": {"type": "string"},
                    "original_score": {"type": "number"},
                    "suggested_score": {"type": "number"},
                },
                "required": ["dimension", "issue", "original_score", "suggested_score"],
                "additionalProperties": False,
            },
        },
        "overall_assessment": {"type": "string"},
    },
    "required": ["agree", "adjusted_score", "challenges", "overall_assessment"],
    "additionalProperties": False,
}
