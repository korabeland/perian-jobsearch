# Evaluation Methodology

> Living reference for how Perian evaluates, compares, and selects local LLMs.
> Last updated: 2026-03-27.

---

## 1. Evaluation Philosophy

Perian's eval system answers one question: **which model produces the best structured output for this pipeline, on this hardware, at this cost?**

Standard benchmarks (MMLU, HumanEval, GPQA) measure general capability. We need something narrower — task-specific accuracy on structured JSON extraction, scoring, and reasoning for job listings. A model that scores 5 points higher on MMLU but produces inconsistent JSON is worse for us than a smaller model that nails the schema every time.

We measure three dimensions, following the CMU "Beyond Capable" framework:

| Dimension | What it means | Why it matters |
|-----------|---------------|----------------|
| **Accuracy** | Does the model get the right answer? | Core correctness — wrong extractions waste human review time |
| **Calibration** | Does the model's confidence match its correctness? | Drives the router: if a model says "90% sure" but is right 60% of the time, escalation thresholds are meaningless |
| **Robustness** | Is the model stable across runs, rephrasings, and edge cases? | Production reliability — a model that scores 0.95 on 3 runs but 0.50 on the 4th is unusable |

> **★ Insight — Capability vs. Prompt-Fit**
> A model's eval score is always the product of two things: the model's capability *and*
> how well the prompt fits the model. Perian's prompts were optimized for qwen3 via
> autoresearch (Phase 4). When evaluating a new model family, a lower score might mean
> "the prompt doesn't suit this model" rather than "the model is worse." The upgrade
> decision framework (§5) accounts for this by distinguishing capability evals from
> prompt-fit evals.

### What we deliberately do NOT measure

- **General knowledge** — tested by MMLU/GPQA; we trust published benchmarks for this
- **Code generation** — not part of the pipeline
- **Creative writing / style** — irrelevant for structured output
- **Multilingual** — all pipeline inputs are English

### Sources

- CMU SEI. "Beyond Capable: Accuracy, Calibration, and Robustness in LLMs." [[link]](https://sei.cmu.edu)
- "Calibration Is Not Enough: Evaluating Confidence Estimation Under Language Variations." arXiv:2601.08064

---

## 2. Metric Definitions

### 2.1 Syntactic Metrics

These are pass/fail gates. A model that can't produce valid JSON reliably is disqualified before any accuracy measurement.

**JSON Parse Rate** — % of runs where the output parses as valid JSON.
- Computed per level, aggregated as `json_output` in the capability profile (average of L1–L4 parse rates).
- Target: ≥ 0.98. Below 0.95 is disqualifying.
- The parse pipeline (`helpers.py:parse_json_response`) strips markdown fences, fixes inline math expressions (`r'"[^"]+":\s*[^"]*=\s*([\d.]+)'`), and falls back to innermost `{...}` extraction.

**Schema Compliance** — % of parsed outputs that have all required fields with valid enum values.
- Required fields: `title`, `company`, `seniority`, `remote_policy`, `domain`, `key_requirements`, `compensation`.
- Valid enums: seniority ∈ {junior, mid, senior, staff, principal, executive}; remote_policy ∈ {remote, hybrid, onsite, unknown}.
- Measured at L1 only. L2–L6 have implicit schema checks via field access.

### 2.2 Field Accuracy Metrics

**Exact Match** — binary 1/0 for string fields (title, company, seniority, remote_policy, domain).
- Case-sensitive by default.
- `acceptable_alternatives` list per test case provides known-valid alternate values.
- Currently used at L1 and L2.

**Jaccard Similarity** — for list fields (`key_requirements`).
```
J(A, E) = |A ∩ E| / |A ∪ E|, where A and E are lowercased sets
```
- `requirements_jaccard()` in `level_1_extraction/run_test.py:58–68`.
- **Known limitation:** Exact string comparison means "Salesforce Service Cloud" vs "Salesforce" = zero overlap. This penalizes valid paraphrases.
- **Planned improvement:** Semantic matching via embedding similarity (see §2.4).

**Compensation Sub-field Match** — for the `compensation` object.
- Four binary sub-fields: `min`, `max`, `currency`, `equity_mentioned`.
- Score = correct_subfields / 4.

**Overall Field Accuracy (L1)** — mean of all 7 field scores per case, averaged across cases.
- Maps directly to `capabilities.extraction` in the profile.

**Per-Field Classification Accuracy (L2)** — for each of seniority, domain, remote_policy: % of runs matching expected (or acceptable alternative).
- `overall_classification_accuracy` = mean of the 3 per-field accuracies.
- Maps to `capabilities.classification`.

### 2.3 Reasoning & Scoring Metrics

**Score Correlation (L3)** — how closely the model's dimension scores match ground truth.
```
score_correlation = max(0, 1.0 - 2 × mean_deviation)
```
Where `mean_deviation` is the MAE across 5 scoring dimensions (title_match, domain_match, seniority_match, requirements_fit, compensation_fit), each 0.0–1.0. A 0.0 deviation = perfect 1.0 correlation; a 0.5 deviation = 0.0.
- Maps to `capabilities.conditional_reasoning`.

**Dims Within Tolerance** — count of dimensions where |actual − expected| ≤ tolerance (typically 0.10–0.20).
- Diagnostic metric, not used in the capability profile.

**Total In Range Rate** — % of cases where `total_score` falls within the expected range.
- **Known issue:** Can contradict `dims_within_tolerance` — a case with 5/5 dims in tolerance can still have total_in_range=0% if the range is tightly set and bonus/penalty logic shifts the total.

**Bonus/Penalty Accuracy** — % of cases where bonus/penalty application matches expected.
- **Known issue:** Opacity — the ground truth defines whether a bonus *should* apply, but the criteria can be ambiguous (e.g., "hybrid acceptable" vs "remote bonus eligible").

**Conclusion Accuracy (L4)** — % of exact matches on conclusion enum {strong_match, moderate_match, weak_match, no_match}.
- Maps to `capabilities.causal_reasoning`.
- **Known limitation:** Binary scoring creates artificial cliff for small models. A model answering "moderate_match" when "weak_match" is expected gets the same 0.0 as one answering "strong_match."

**Conclusion Distance (L4)** — numeric distance between actual and expected conclusions.
```
strong=3, moderate=2, weak=1, no=0
distance = |actual_score − expected_score|  (range: 0–3)
```
- More informative than binary accuracy for tracking improvement.
- **Proposed improvement:** Use `max(0, 1.0 - distance / 3.0)` as a continuous capability score instead of binary accuracy.

**Reasoning Coverage (L4)** — fraction of `expected_reasoning_mentions` found via case-insensitive substring match in the model's `reasoning` + `key_factors` text.
- Diagnostic — measures whether the model considered the right factors, independent of reaching the right conclusion.

**Nuanced Judgment (composite)** — weighted blend of L3 and L4:
```
nuanced_judgment = 0.6 × conditional_reasoning + 0.4 × causal_reasoning
```
- Why 60/40: L3 (scoring) is the core numerical judgment that drives routing; L4 (reasoning) is more subjective and less stable.

### 2.4 Semantic Matching (Planned)

Exact string matching penalizes valid paraphrases ("CX Manager" vs "Customer Experience Manager", "SaaS" vs "Software-as-a-Service"). Semantic matching resolves this.

**BERTScore** — contextual embedding metric that computes token-level cosine similarity between candidate and reference texts, producing precision, recall, and F1.
- arXiv:1904.09675.
- Handles synonyms and rephrasings that exact match cannot.

**Planned implementation for Perian:** Use `nomic-embed-text` (already installed in Ollama, 768-dim embeddings) instead of BERT to stay local-first. Threshold: 0.80 cosine similarity for a "match."
- Primary targets: `key_requirements` (replace Jaccard) and `domain` (allow "observability" ≈ "monitoring").
- Proposed module: `evaluation/semantic.py` with `embed_text()`, `cosine_similarity()`, `semantic_match(threshold=0.80)`.

### 2.5 Synthesis Metrics (L6)

**Factual Coverage** — fraction of expected facts found in the model's output (summary + key_findings).
```
Short facts (≤4 words): exact phrase match OR ≥70% word overlap
Long facts (>4 words): ≥60% keyword match (excluding stopwords)
```
- See `level_6_synthesis/run_test.py:67–102`.
- Maps to `capabilities.synthesis`.
- **Known limitation:** Word-overlap matching is crude. A model that correctly synthesizes the same information using different phrasing scores low. All fleet models cluster at 0.46–0.55 — the rubric is the bottleneck, not model capability.
- **Planned improvement:** Replace keyword overlap with embedding similarity per fact.

**Hallucination Clean Rate** — % of outputs containing none of the `must_not_hallucinate` items (case-insensitive substring match).
- Binary per output, aggregated as a rate.

**Coherence** — % of outputs with non-empty `summary` AND ≥ 2 `key_findings`.
- Structural check only, not semantic coherence.

### 2.6 Consistency Metrics (L5)

**Extraction Consistency** — for groups of similar items, do classification fields (seniority, domain, remote_policy) agree across items where they should match?
- Per group: each `should_match` field checked for unique_count ≤ 1.

**Score Variance** — variance of `total_score` within each group.
- `variance_within_threshold = variance ≤ max_score_variance` (typically 0.05).

**Rank Stability** — does the rank order of items match expected?
- Checks `actual_order == expected_rank_order`.

### 2.7 Calibration Metrics (Planned)

**Expected Calibration Error (ECE)** — bins items by logprob confidence, computes accuracy per bin, measures divergence.
- Items where the model says "90% sure" should actually be correct ~90% of the time.
- Critical for routing: the escalation threshold in `router.yaml` (`mean_logprob < -1.5`) was set without calibration data.
- arXiv:2406.03441v1 (Cycles of Thought) describes both white-box (logprob) and black-box (re-answer consistency) confidence estimation.

**Logprob-to-Confidence Conversion** — map `mean_logprob` to a 0–1 confidence score via sigmoid or min-max scaling.
- Proposed module: `evaluation/calibration.py` with `compute_ece()`, `logprob_to_confidence()`, reliability diagram data.

### 2.8 Statistical Metrics (Planned)

**Bootstrap Confidence Intervals** — resample scores with replacement (n_bootstrap=1000), compute 95% CI.
- With n=7 cases per level, intervals will be wide. That is the correct answer — it tells us we can't distinguish 0.88 from 0.91 with current sample sizes.
- Returns `{mean, ci_lower, ci_upper, n}`.

**Significance Testing** — `scores_significantly_different(scores_a, scores_b, alpha=0.05)`.
- Returns `{different: bool, p_value, effect_size}`.
- Prevents false confidence from comparing point estimates without intervals.

**Cost-per-Correct-Extraction** — `inference_time × model_size / accuracy`.
- Normalizes across model tiers: a 2x-faster model with 5% lower accuracy may still be more efficient.

### Sources

- Zhang et al. "BERTScore: Evaluating Text Generation with BERT." arXiv:1904.09675. 2020.
- Bao et al. "Cycles of Thought: Measuring LLM Confidence through Stable Explanations." arXiv:2406.03441v1.
- Bhatt et al. "Towards Provably Unbiased LLM Judges via Bias-Bounded Evaluation." arXiv:2603.05485.

---

## 3. Scoring Methodology — Per Level

### Level 1: Extraction

| Metric | How it's computed | What it means |
|--------|-------------------|---------------|
| `field_accuracy` | Mean of 7 field scores across all cases | Core extraction quality |
| `schema_compliance` | % outputs with valid JSON + all fields + valid enums | Structural reliability |
| `json_parse_rate` | % runs where JSON parsed at all | Basic output format |

**What works well:**
- Compensation sub-field matching is granular (4 separate checks).
- Schema compliance catches enum drift early.
- Per-case breakdown reveals which listing types are hardest.

**Known issues and planned changes:**
1. **key_requirements uses Jaccard on exact strings.** "Salesforce Service Cloud" vs "Salesforce" = zero overlap. Fix: semantic match with 0.80 cosine threshold.
2. **No acceptable_alternatives for some fields.** Domain is particularly ambiguous for multi-domain companies (e.g., Datadog = observability + security + infrastructure). Fix: expand alternatives, then migrate to semantic matching.
3. **Only 7 test cases.** A 2–3 point swing from variance moves field_accuracy by 20–30%. Fix: expand to 40 cases (15 standard + 15 edge + 10 adversarial).

### Level 2: Classification

| Metric | How it's computed | What it means |
|--------|-------------------|---------------|
| `per_field_accuracy` | Per field: % runs matching expected/alternatives | Accuracy per classification type |
| `overall_classification_accuracy` | Mean of 3 per-field accuracies | Aggregate classification quality |

**What works well:**
- Separate per-field scoring reveals which classification is weakest (domain is consistently hardest).
- Acceptable alternatives handle known ambiguities.

**Known issues:**
1. **Domain classification is ambiguous by nature.** Multi-domain companies have no single correct answer without alternatives. Fix: expand acceptable_alternatives, then semantic matching.
2. **Only 7 test cases.** Same statistical power issue as L1.

### Level 3: Conditional Reasoning (Scoring)

| Metric | How it's computed | What it means |
|--------|-------------------|---------------|
| `score_correlation` | `max(0, 1 − 2 × MAE)` across 5 dimensions | How close dimension scores are to truth |
| `total_in_range_rate` | % outputs where total_score ∈ expected range | End-to-end scoring accuracy |
| `bonus_accuracy` | % cases with correct bonus application | Logic correctness |
| `penalty_accuracy` | % cases with correct penalty application | Logic correctness |

**What works well:**
- 5-dimensional scoring gives fine-grained signal vs binary right/wrong.
- Tolerance bands allow near-misses to count.

**Known issues:**
1. **total_in_range can contradict dims_within_tolerance.** A case with 5/5 dims in tolerance but total_in_range=0% (score_03). Root cause: bonus/penalty logic shifts the total outside the range even when dimensions are correct. Fix: widen range or decouple dimensional accuracy from total accuracy.
2. **Bonus/penalty ground truth is opaque.** Score_07: model correctly doesn't apply remote bonus for "hybrid acceptable location" but gets marked wrong. Fix: review and recalibrate bonus/penalty expectations.
3. **43% synthetic test cases.** Edge cases by design, but this over-represents unusual scenarios vs real-world distribution. Fix: rebalance to ≤ 25% synthetic.

### Level 4: Causal Reasoning

| Metric | How it's computed | What it means |
|--------|-------------------|---------------|
| `conclusion_accuracy` | % exact conclusion matches | Hard accuracy |
| `mean_conclusion_distance` | Mean |actual − expected| on 0–3 scale | Soft accuracy (how wrong) |
| `reasoning_coverage` | % of expected reasoning mentions found | Does it consider the right factors? |

**What works well:**
- Conclusion distance captures partial credit (moderate vs strong = 1, not 3).
- Reasoning coverage is independent of conclusion — a model can fail the conclusion but still demonstrate correct reasoning.

**Known issues:**
1. **Binary conclusion_accuracy creates artificial cliff for small models.** qwen3.5:2b scores 0.10 despite reasonable reasoning coverage. Fix: replace binary accuracy with distance-based scoring: `max(0, 1.0 − distance / 3.0)`.
2. **Test cases conflate domain knowledge with causal reasoning.** causal_01 (gaming company mismatch) tests whether the model knows gaming is wrong for the candidate's profile, not whether it can reason causally. Fix: add profile-independent causal reasoning cases.
3. **Substring matching for reasoning coverage is brittle.** "domain mismatch" won't match "the domain doesn't align." Fix: semantic matching.
4. **10 test cases is better than L1–L3 but still insufficient** for 95% CI on conclusion_accuracy.

### Level 5: Batch Consistency

| Metric | How it's computed | What it means |
|--------|-------------------|---------------|
| `extraction_consistency` | % of should_match fields that are uniform | Classification stability |
| `mean_score_variance` | Variance of total_scores within groups | Scoring stability |
| `rank_stability` | Correct rank order? | Relative judgment stability |

**What works well:**
- Tests a different dimension than accuracy — two models with identical L1 scores can have very different L5 scores.
- Variance threshold (0.05) is appropriately tight.

**Known issues:**
1. **Only 3 groups.** Not enough for reliable consistency measurement. Fix: expand to 6 groups.
2. **Not included in capability profile.** L5 doesn't map to any capability score. Fix: consider adding a `stability` capability.

### Level 6: Synthesis

| Metric | How it's computed | What it means |
|--------|-------------------|---------------|
| `factual_coverage` | % expected facts found via word overlap | Content accuracy |
| `hallucination_clean_rate` | % outputs with no forbidden content | Safety |
| `coherence` | Has summary + ≥2 findings? | Structure |

**What works well:**
- Hallucination check is well-designed — specific, verifiable.
- Structural coherence gate is minimal but useful.

**Known issues:**
1. **Factual coverage is capped by the rubric, not model capability.** All models cluster at 0.46–0.55. Crude word overlap with hardcoded stopwords penalizes valid rephrasings. Fix: embedding-based fact matching.
2. **100% synthetic test cases.** No real-world synthesis tasks. Fix: generate cases from actual collected listings.
3. **No partial credit** for facts that are semantically correct but phrased differently. Fix: semantic matching with 0.70 cosine threshold.

---

## 4. Test Suite Requirements

### Current State

| Level | Cases | Min for 95% CI | Gap | Synthetic % |
|-------|-------|-----------------|-----|-------------|
| L1 Extraction | 7 | ~40 | 33 | 14% |
| L2 Classification | 7 | ~40 | 33 | 0% |
| L3 Conditional | 7 | ~40 | 33 | 43% |
| L4 Causal | 10 | ~40 | 30 | 0% |
| L5 Batch | 3 groups | ~6 groups | 3 | 0% |
| L6 Synthesis | 8 | ~15 | 7 | 100% |

The 95% CI target comes from binomial power analysis: detecting a 10% accuracy difference between two models at α=0.05 requires ~40 cases per level.

### Coverage Matrix (Target)

| Level | Standard | Edge/Adversarial | Total | Max Synthetic |
|-------|----------|-------------------|-------|---------------|
| L1 | 15 standard listings | 15 edge + 10 adversarial | 40 | 25% |
| L2 | 15 seniority + 10 domain | 15 remote_policy edge | 40 | 25% |
| L3 | 12 strong match | 12 weak/negative + 16 mixed | 40 | 25% |
| L4 | 12 dealbreaker scenarios | 16 nuanced + 12 profile-independent | 40 | 25% |
| L5 | 3 same-role groups | 3 cross-role groups | 6 groups | 0% |
| L6 | 10 multi-listing synthesis | 5 trend analysis | 15 | 50% |

### Test Case Sources

1. **Real listings** from `inputs/collected-*.jsonl` — cleaned, anonymized where needed.
2. **Edge cases** crafted manually for specific failure modes (missing compensation, ambiguous seniority, multi-domain companies).
3. **Adversarial cases** designed to trigger known weaknesses (math in JSON, extremely long requirements lists, contradictory information).
4. **Profile-independent cases** for L4 — causal reasoning that doesn't depend on profile-specific domain knowledge.

### Rotation Policy

- **Never modify test cases to make a model score higher.** Ground truth is independent of any model's behavior.
- **Add cases when new failure modes are discovered.** Tag with `added_date` and `reason`.
- **Retire cases only when the scenario is no longer relevant** (e.g., a job board changes its format).
- **Holdout split:** 80% optimization / 20% validation (seed=42). Autoresearch only sees the optimization split.

### Contamination Safeguards

Ground truth validated against qwen3:8b behavior introduces evaluation leakage — the test implicitly rewards models that behave like qwen3:8b. Autoresearch tuned prompts against qwen3, compounding the effect.

Mitigations:
1. **Profile-independent test cases** that don't depend on domain knowledge.
2. **Multi-model ground truth calibration** — validate expected answers against ≥ 2 model families.
3. **Temporal rotation** — refresh 10% of test cases quarterly from newly collected real listings.
4. **Track prompt-fit gap** — run new models with both Perian prompts (prompt-fit eval) and generic prompts (capability eval). If the gap is > 0.10, the prompt needs tuning for that model before drawing conclusions.

### Sources

- "All Leaks Count, Some Count More: Interpretable Temporal Contamination Detection." arXiv:2602.17234
- "Towards Provably Unbiased LLM Judges via Bias-Bounded Evaluation." arXiv:2603.05485

---

## 5. Model Comparison & Upgrade Decisions

### When to Compare

| Trigger | What runs | Decision |
|---------|-----------|----------|
| New Ollama model released (Legolas candidate) | Full L1–L6 eval | Fleet upgrade? |
| Prompt change via autoresearch | L1–L3 safety gate + holdout | Accept mutation? |
| Monthly maintenance | Full L1–L6 eval on current fleet | Drift detection |
| Post-fine-tune (Phase 5) | Full L1–L6 + general regression (GSM8K, SQuAD subset) | Accept fine-tune? |

### Comparison Protocol

1. **Same conditions.** Same n_runs (≥5), same test cases, same hardware state (close background apps for timing).
2. **Bootstrap CIs.** Never compare point estimates. Compute 95% bootstrap CIs for each capability.
3. **Significance test.** `scores_significantly_different()` at α=0.05. Only claim a difference if p < 0.05.
4. **Effect size.** Report Cohen's d alongside p-value. Statistical significance with tiny effect size doesn't justify the cost of switching.

### Upgrade Decision Framework

Model B replaces Model A if ALL of the following hold:

```
1. B.capability.ci_lower ≥ A.capability.score
   → B's worst case beats A's average

2. B.production.score ≥ A.production.score × 0.95
   → At most 5% drop on production (Perian prompt) eval

3. B.prompt_fit_gap > −0.05
   → Prompts work reasonably with B (gap = production − capability)

4. B.inference_time ≤ A.inference_time × 1.5  OR  B.size_gb ≤ A.size_gb
   → Not dramatically slower or larger
```

**Exception:** If B.capability >> A.capability (e.g., +0.15 across 3+ capabilities) AND B.prompt_fit_gap < −0.05, run autoresearch for B's prompts before deciding. The capability is there; the prompts just need tuning.

### Dual Scoring: Production vs Capability

Every model comparison runs two eval passes:

| Pass | Prompts used | What it measures |
|------|-------------|------------------|
| **Production eval** | Current Perian prompts (extract.md, score.md) | How well B works *right now* with our prompts |
| **Capability eval** | Generic prompts (no profile-specific rules) | B's raw capability, independent of prompt fit |

The gap between these two scores is the **prompt-fit gap**. A large negative gap means the model is capable but our prompts don't suit it — a signal to run autoresearch, not to reject the model.

### Sources

- "The Ultimate Guide to Fine-Tuning LLMs." arXiv:2408.13296v1 — regression testing methodology post-fine-tune.
- Cleanlab. "LLM Structured Output Benchmarks are Riddled with Mistakes." — conflation of format and semantic correctness.

---

## 6. Eval Cadence

| Frequency | What runs | Purpose |
|-----------|-----------|---------|
| **Per commit** (autoresearch) | L1–L3 safety gate on holdout | Prevent prompt regression |
| **Per candidate** (Legolas) | Full L1–L6 on standard suite | Model comparison |
| **Weekly** | BERTScore + logprob calibration on holdout (planned) | Track drift |
| **Monthly** | Full L1–L6 on current fleet | Detect degradation (Ollama updates, quantization changes) |
| **Quarterly** | Human review of N=30–50 hardest cases + test case refresh | Ground truth maintenance |

### What Triggers an Eval

1. Legolas surfaces a candidate → full eval.
2. Autoresearch accepts a prompt mutation → safety gate (L1–L3, 3 runs).
3. Ollama version update → spot-check eval (L1 + L3, current fleet primary).
4. Test case expansion → re-eval current fleet to rebaseline.
5. Fleet configuration change (model added/removed/tier changed) → full eval on changed model.

### Eval Budget

Each full 6-level eval with n_runs=5 takes ~20 minutes on the M5 Pro (varies by model size). A fleet comparison of 5 candidates = ~100 minutes. Budget accordingly — don't run during active work sessions where Ollama contention affects results.

---

## 7. Known Limitations

### Eval Can't Measure

| Gap | Why | Mitigation |
|-----|-----|------------|
| **Real-world distribution fidelity** | Test cases are hand-picked or synthetic; production listings have different distributions | Quarterly refresh from real collected data |
| **Long-tail failures** | 7–10 cases per level can't surface rare failure modes | Expand to 40+ cases; monitor production failures |
| **Human judgment alignment** | No human annotations for subjective tasks (L4 reasoning quality, L6 synthesis insight) | Quarterly human review of disagreements |
| **Cross-profile generalization** | All tests use a single candidate profile; no evidence of performance on other profiles | Add multi-profile test cases when Phase 6 begins |
| **Prompt-model interaction effects** | Prompts were optimized for qwen3; other families may underperform due to prompt fit, not capability | Dual scoring (§5) separates prompt-fit from capability |
| **Hardware-specific performance** | Timing benchmarks are M5 Pro–specific; MoE models may behave differently on other hardware | Document hardware in eval results; never extrapolate |

### Where Human Review is Needed

1. **L4 conclusion disagreements** — when model and ground truth disagree by 1 level (moderate vs weak), a human should judge which is more defensible.
2. **L6 synthesis quality** — factual coverage doesn't measure insight quality. A model that covers all facts but draws no conclusions is less useful than one that misses a fact but identifies a real trend.
3. **New failure modes** — production monitoring should feed novel failures back into the test suite.
4. **Acceptable alternatives** — when a model produces a valid answer not in the alternatives list, a human must decide whether to add it.

---

## 8. References

### Papers

| Citation | Topic | How we use it |
|----------|-------|---------------|
| Zhang et al. "BERTScore." arXiv:1904.09675 | Semantic text similarity | Planned: replace exact match with embedding similarity for fields and facts |
| CMU SEI. "Beyond Capable." | Accuracy + calibration + robustness | Framework: three evaluation dimensions |
| "Calibration Is Not Enough." arXiv:2601.08064 | Confidence under variation | Motivation for testing across input phrasings |
| Bao et al. "Cycles of Thought." arXiv:2406.03441v1 | LLM confidence estimation | Logprob-based and consistency-based confidence |
| "Bias-Bounded Evaluation." arXiv:2603.05485 | 12 bias types in LLM-as-judge | Awareness of biases in automated scoring |
| "Temporal Contamination Detection." arXiv:2602.17234 | Benchmark leakage | Contamination safeguards for test suite |
| "Fine-Tuning LLMs Guide." arXiv:2408.13296v1 | Fine-tuning eval pipeline | Regression testing methodology (Phase 5) |

### Frameworks & Tools

| Framework | What it is | Relevance |
|-----------|-----------|-----------|
| LM Evaluation Harness (EleutherAI) | Open-source benchmark framework, 60+ benchmarks | Most applicable external framework; YAML task definitions could complement Perian's custom suite |
| HELM (Stanford) | Holistic multi-metric evaluation | Design principles (transparency, incompleteness as feature) |
| StructEval (arXiv:2505.20139v1) | 21 output formats, 44 task types | Splits format correctness from semantic correctness — directly relevant |
| LLMStructBench (arXiv:2602.14743) | Structured extraction from text | Most directly applicable published benchmark |
| EnviroLLM (arXiv:2512.12004v1) | Energy + latency + accuracy holistic framework | Model for cost-per-correct-extraction metric |
| aidatatools/ollama-benchmark | Ollama throughput benchmarking | Reference for timing methodology |

### Perian Eval Infrastructure

| File | Purpose |
|------|---------|
| `evaluation/run_all.py` | Orchestrator: runs levels, builds profile, saves results, logs to changelog |
| `evaluation/helpers.py` | Ollama client, JSON parsing, prompt building, logprob extraction |
| `evaluation/compare.py` | Side-by-side capability comparison tool |
| `evaluation/test_suite/level_N_*/run_test.py` | Per-level test runner + scoring |
| `evaluation/test_suite/level_N_*/test_cases.json` | Ground truth data |
| `autoresearch/eval_criteria.md` | Locked evaluation rubric (never auto-modified) |
| `autoresearch/config.yaml` | Loop params: iterations, cost cap, safety thresholds |
| `config/fleet.yaml` | Fleet definition with capability scores |
| `config/router.yaml` | Escalation config and thresholds |

---

## Appendix A: Capability Profile Reference

The 7 capability scores derived from eval results, as computed in `run_all.py:build_capability_profile()`:

| Capability | Source | Formula |
|------------|--------|---------|
| `extraction` | L1 | `field_accuracy` |
| `classification` | L2 | `overall_classification_accuracy` |
| `conditional_reasoning` | L3 | `max(0, 1 − 2 × mean_deviation)` |
| `causal_reasoning` | L4 | `conclusion_accuracy` (binary) |
| `synthesis` | L6 | `factual_coverage` |
| `json_output` | L1–L4 | Mean JSON parse rate across levels |
| `nuanced_judgment` | L3 + L4 | `0.6 × conditional + 0.4 × causal` |

## Appendix B: Current Fleet Eval Results (2026-03-27)

| Capability | qwen3:8b | qwen3:14b | qwen3:32b | qwen3.5:2b | qwen3.5:4b | qwen3.5:9b | qwen3.5:27b | qwen3.5:35b-a3b |
|---|---|---|---|---|---|---|---|---|
| extraction | 0.90 | — | — | 0.83 | 0.89 | 0.91 | **0.94** | 0.93 |
| classification | 0.97 | — | — | 0.84 | 0.97 | 0.97 | **1.00** | 0.98 |
| conditional | 0.88 | **0.91** | 0.89 | 0.74 | 0.80 | 0.84 | 0.88 | 0.88 |
| causal | 0.60 | **0.70** | 0.60 | 0.10 | 0.47 | 0.57 | 0.57 | 0.57 |
| synthesis | 0.55 | 0.52 | 0.49 | 0.46 | 0.52 | 0.55 | **0.67** | 0.63 |
| json_output | 1.00 | 1.00 | 1.00 | 0.98 | 0.99 | 0.99 | 1.00 | 1.00 |
| nuanced | 0.65 | **0.83** | 0.77 | 0.48 | 0.67 | 0.73 | 0.76 | 0.76 |
| Size | 5.2 GB | 9.3 GB | 20 GB | 2.7 GB | 3.4 GB | 6.6 GB | 17 GB | 24 GB |
| Eval time | — | — | — | — | — | — | 5732s | 1490s |
| n_runs | 3 | 5 | 5 | 3 | 3 | 3 | 3 | 3 |

Dashes indicate missing data (early runs before full test suite). **Bold** = fleet best for that capability. qwen3.5:35b-a3b is MoE (256 experts, 3B active) — 3.8x faster than 27b dense but slightly lower accuracy across all capabilities.
