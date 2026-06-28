# Evaluation Criteria for Prompt Optimization

> **This file is LOCKED.** It is the ground truth rubric for the autoresearch loop.
> Never modify this file programmatically. Changes require human review.

---

## extract.md Criteria

### Hard Pass/Fail

1. **Valid JSON**: Output must parse as valid JSON. No markdown fences, no commentary.
2. **Required fields present**: All 7 fields must exist: `title`, `company`, `seniority`, `remote_policy`, `domain`, `key_requirements`, `compensation`.
3. **Valid enums**: `seniority` must be one of: junior, mid, senior, staff, principal, executive. `remote_policy` must be one of: remote, hybrid, onsite, unknown.
4. **Correct types**: `key_requirements` must be an array. `compensation` must be an object with keys: min, max, currency, equity_mentioned.

### Soft Quality (scored 0.0–1.0)

1. **Seniority inference**: Correctly infers seniority from title keywords and years-of-experience mentions. Supports both IC and management tracks: "Senior" for 5+ years, "Senior" in title, or "Manager" title. "Staff" for "Staff" in title, 8+ years with broad scope, or "Senior Manager". "Principal" for "Principal" in title, "Director", or "Head of" (interchangeable). "Executive" for "VP", "Chief", or C-level adjacent. Default to "mid" when ambiguous.
2. **Domain classification**: Uses concise, consistent labels (e.g., "SaaS", "AI/ML", "B2B technology", "digital transformation", "martech", "fintech", "healthcare", "developer tools"). Does not hallucinate domains not implied by the listing.
3. **Compensation normalization**: Extracts exact numbers. Normalizes to annual figures. Sets null when not explicitly stated — never guesses.
4. **Requirement specificity**: Extracts specific tools, technologies, and competencies over vague terms. Caps at 8 items. Prioritizes hard requirements over nice-to-haves.
5. **Null handling**: Uses null/unknown/empty values appropriately. Never fabricates data not present in the listing.

---

## score.md Criteria

### Hard Pass/Fail

1. **Valid JSON**: Output must parse as valid JSON. No markdown fences, no commentary.
2. **All 5 dimensions present**: `scores` object must contain: `title_match`, `domain_match`, `seniority_match`, `requirements_fit`, `compensation_fit`. Each must have `score` (number) and `reasoning` (string).
3. **Score range**: Each dimension score must be in [0.0, 1.0].
4. **Total score**: `total_score` must be present and in [0.0, 1.0].
5. **Weighted total**: `weighted_total` must be present and approximately equal to sum of (score * weight) for each dimension.

### Soft Quality (scored 0.0–1.0)

1. **Dimension independence**: Each dimension scored on its own merits. A low title_match should not drag down requirements_fit.
2. **Arithmetic correctness**: `weighted_total` = sum of (dimension_score * weight). `total_score` = clamp(weighted_total + bonuses + penalties, 0.0, 1.0). Arithmetic errors are penalized.
3. **Bonus/penalty logic**: Bonuses applied only when conditions are met (e.g., remote_policy matches preference). Penalties applied only when dealbreaker conditions from plan.yaml are met (e.g., full-time on-site, junior seniority). Never applies both bonus and penalty for the same condition.
4. **Reasoning quality**: Each dimension has a concise, specific reasoning string. Reasoning references actual data from the extracted listing. Vague reasoning ("seems good") is penalized.
5. **Score calibration**: Scores are distributed across the range, not clustered at extremes. A perfect match is 1.0, a complete mismatch is 0.0, partial matches fall between. Scores are consistent across similar inputs.
