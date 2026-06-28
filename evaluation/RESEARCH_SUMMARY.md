# Eval Research Summary

> What we investigated, what we found, and what it means for Perian's local LLM fleet.
> Research completed: 2026-03-27.

---

## 1. Overview

### What Was Researched

How should a local-first pipeline evaluate, compare, and select LLMs for structured JSON extraction tasks — when standard benchmarks (MMLU, HumanEval) don't measure what matters?

### Why

Perian's fleet is expanding. Five qwen3.5 candidates arrived alongside three existing qwen3 models. We need a rigorous process for deciding which models to keep, replace, or specialize — and the current eval system has gaps that could lead to wrong decisions.

### Three Research Tracks

| Track | Method | What it produced |
|-------|--------|------------------|
| **Infrastructure audit** | Code reading across 15 files (run_all.py, 6 level runners, helpers.py, router.py, fleet.yaml, autoresearch config) | Full map of the scoring pipeline, capability profile derivation, and router integration |
| **External literature** | Web search + paper review (7 papers, 6 frameworks, 3 blog posts) | Evaluation best practices, semantic matching alternatives, calibration methods, contamination detection |
| **Failure analysis** | Cross-referencing eval results across 8 models against test case definitions and scoring functions | 7 systemic issues in current eval that distort model comparison |

---

## 2. External Research — Papers & Frameworks

### Papers

#### BERTScore — Semantic Text Similarity
**Zhang et al.** "BERTScore: Evaluating Text Generation with BERT." arXiv:1904.09675. 2020.

Computes token-level cosine similarity between contextual embeddings of candidate and reference text, producing precision, recall, and F1. Handles synonyms and rephrasings that exact string matching cannot — "staff engineer" matches "engineer level 4", "Customer Experience" matches "CX."

**Perian application:** Replace exact-string Jaccard similarity on `key_requirements` (L1) and word-overlap fact matching (L6) with embedding similarity via nomic-embed-text (already installed in Ollama, 768-dim). Threshold: 0.80 cosine for field matching, 0.70 for synthesis facts.

---

#### CMU "Beyond Capable" — The Three-Dimension Framework
**CMU Software Engineering Institute.** "Beyond Capable: Accuracy, Calibration, and Robustness in LLMs."

Argues that accuracy alone is insufficient for evaluating LLMs in production systems. Three dimensions matter:

| Dimension | What it measures | Why it matters for Perian |
|-----------|------------------|--------------------------|
| **Accuracy** | Does the model get the right answer? | Core correctness — wrong extractions waste human review time |
| **Calibration** | Does confidence match correctness? | Drives the router — if 90% confidence = 60% accuracy, escalation thresholds are meaningless |
| **Robustness** | Stable across runs and rephrasings? | A model scoring 0.95 on 3 runs and 0.50 on the 4th is unusable in production |

**Perian application:** Adopted as the evaluation philosophy. Currently we only measure accuracy. Calibration (ECE from logprobs) and robustness (test-retest reliability) are planned additions.

---

#### "Calibration Is Not Enough"
arXiv:2601.08064.

Calibration measured on a single prompt phrasing is misleading. A model can be well-calibrated on one phrasing and wildly miscalibrated on a paraphrase. True reliability requires testing across input variations.

**Perian application:** When expanding the test suite, include rephrasings of the same listing (different paragraph structure, different detail ordering) and verify calibration holds across variations.

---

#### Cycles of Thought — Confidence Estimation
**Bao et al.** "Cycles of Thought: Measuring LLM Confidence through Stable Explanations." arXiv:2406.03441v1.

Two approaches to estimating LLM confidence:
- **White-box (logprobs):** Use token-level log probabilities as a direct confidence signal. Fast, no extra calls.
- **Black-box (consistency):** Ask the model the same question multiple times; confidence = agreement rate. Slower but works without logprob access.

**Perian application:** We already collect logprobs via `options.logprobs: true` in every Ollama call. The router's escalation threshold (`mean_logprob < -1.5`) uses logprobs as a proxy for confidence — but this threshold was set without calibration data. ECE computation would validate whether the threshold is meaningful.

---

#### Bias-Bounded Evaluation
**"Towards Provably Unbiased LLM Judges via Bias-Bounded Evaluation."** arXiv:2603.05485.

Identifies and quantifies 12 distinct bias types in LLM-as-judge systems: position bias, verbosity bias, self-enhancement bias, and others. Even automated scoring functions can embed biases through their design choices.

**Perian application:** Awareness item. Our scoring functions aren't LLM judges, but they embed design biases — e.g., the L6 word-overlap rubric favors verbose outputs, and ground truth calibrated to one model's behavior biases toward that model's style.

---

#### Temporal Contamination Detection
**"All Leaks Count, Some Count More: Interpretable Temporal Contamination Detection."** arXiv:2602.17234.

Training data that overlaps with test data inflates benchmark scores. The problem is worse than commonly assumed — even partial overlap (similar patterns, not exact matches) can leak signal.

**Perian application:** Three contamination vectors discovered in our own eval (see §4, Finding 7). Mitigations: temporal rotation of test cases, multi-model ground truth calibration, capping synthetic test case percentage.

---

#### Fine-Tuning Evaluation Guide
**"The Ultimate Guide to Fine-Tuning LLMs."** arXiv:2408.13296v1.

7-stage fine-tuning pipeline. Critical eval sub-step: after fine-tuning, run regression tests on general benchmarks (GSM8K, SQuAD, GLUE) to verify the model didn't lose general capability while gaining task-specific performance.

**Perian application:** Phase 5 (fine-tuning pipeline) will need regression tests. The eval suite should include a small general-capability subset alongside the task-specific levels.

---

### Frameworks

| Framework | What it is | Key finding | Perian relevance |
|-----------|-----------|-------------|------------------|
| **LM Evaluation Harness** (EleutherAI) | Open-source, 60+ benchmarks, powers HF Open LLM Leaderboard | Extensible via YAML task definitions | Most applicable external framework; could complement Perian's custom suite with standard benchmarks |
| **HELM** (Stanford) | Holistic multi-metric evaluation | Recognizes incompleteness as a design principle — no eval captures everything | Validates our "Known Limitations" section approach |
| **StructEval** (arXiv:2505.20139v1) | 21 output formats, 44 task types | Splits into format correctness (StructEval-T) and semantic correctness — the two are independent | Directly relevant: Perian's `json_parse_rate` measures format, `field_accuracy` measures semantics. Good — we already separate them |
| **LLMStructBench** (arXiv:2602.14743) | Structured extraction from unstructured text | Benchmarks specifically for extraction tasks, not generation | Most directly applicable published benchmark to Perian's L1 extraction |
| **EnviroLLM** (arXiv:2512.12004v1) | Energy + latency + accuracy holistic framework | Models should be compared on efficiency, not just accuracy | Inspired the cost-per-correct-extraction metric |
| **aidatatools/ollama-benchmark** | Ollama throughput benchmarking | Community tool for Ollama-specific perf testing | Reference for timing methodology; confirms: test on actual hardware, q4_K_M quant, average 3+ runs |

### Structured Output Research — Key Takeaway

**Cleanlab** published "LLM Structured Output Benchmarks are Riddled with Mistakes" — arguing that most benchmarks conflate format correctness (valid JSON, correct schema) with semantic correctness (right values in the fields). A model that produces perfect JSON with wrong answers scores well. A model that produces the right answers with a minor formatting issue scores poorly.

This directly validates Perian's design of separating `json_parse_rate` / `schema_compliance` (format) from `field_accuracy` (semantic). But it also highlights a gap: we don't have a pure semantic accuracy measure that's independent of format — if JSON parsing fails, we score 0 on accuracy even though the content might be correct.

---

## 3. Infrastructure Audit — Current Eval System

### The 6-Level Pipeline

| Level | What it tests | Primary metric | How it scores |
|-------|---------------|----------------|---------------|
| **L1 Extraction** | Raw structured data extraction from listings | `field_accuracy` | Per-field: exact match (strings), Jaccard (requirements), sub-field match (compensation). Average across 7 fields and all cases. |
| **L2 Classification** | Categorical field accuracy (seniority, domain, remote_policy) | `overall_classification_accuracy` | Per-field % correct across runs, averaged. Supports `acceptable_alternatives`. |
| **L3 Conditional** | Multi-dimensional scoring against ground truth | `score_correlation` | `max(0, 1 − 2 × MAE)` across 5 scoring dimensions. Also tracks bonus/penalty accuracy and total-in-range rate. |
| **L4 Causal** | Multi-step reasoning on match quality | `conclusion_accuracy` | Binary exact match on conclusion enum (strong/moderate/weak/no_match). Also tracks conclusion distance (0-3) and reasoning coverage (substring match). |
| **L5 Batch** | Consistency across similar items | `extraction_consistency` | Check classification fields match within groups where they should. Score variance within threshold (0.05). Rank order stability. |
| **L6 Synthesis** | Multi-document summarization | `factual_coverage` | Word overlap: short facts need phrase/70% word match, long facts need 60% keyword match (excluding stopwords). Also: hallucination check, structural coherence. |

### Capability Profile Derivation

Seven scores fed to the router, derived from level metrics:

```
extraction         = L1.field_accuracy
classification     = L2.overall_classification_accuracy
conditional_reasoning = L3.score_correlation
causal_reasoning   = L4.conclusion_accuracy
synthesis          = L6.factual_coverage
json_output        = mean(L1-L4 parse rates)
nuanced_judgment   = 0.6 × L3 + 0.4 × L4
```

L5 (batch consistency) is not mapped to any capability score — it's diagnostic only.

### Router Integration

The router selects the cheapest model where `model.capabilities[step] ≥ min_reliability` (default 0.80). One specialist override: `nuanced_judgment → qwen3:14b` (scores 0.83 vs 0.77 for 32b). Escalation threshold: `mean_logprob < -1.5` (currently in logging_only mode — logs both outputs but doesn't actually escalate).

### Autoresearch Safety Gate

The autoresearch loop (Phase 4) proposes prompt mutations via Claude API, evaluates them locally on Ollama, and accepts improvements. The safety gate runs **only L1–L3** after each accepted mutation, with a 3% max capability drop threshold.

**L4–L6 are deliberately excluded** because their rubrics are unreliable (binary scoring cliff on L4, word-overlap ceiling on L6, only 3 groups on L5). This means autoresearch can accept prompt mutations that regress on reasoning and synthesis without detection.

---

## 4. Failure Analysis — What's Broken

### Finding 1: Exact String Matching Penalizes Valid Paraphrases

`key_requirements` uses Jaccard similarity on exact lowercased strings. "Salesforce Service Cloud" vs "Salesforce" = zero set overlap. "5+ years CX management" vs "5 years of customer experience management experience" = near-zero overlap.

**Impact:** L1 field_accuracy is systematically depressed for models that paraphrase well. A model extracting the right skills in different words scores lower than one copying verbatim from the listing.

**Fix:** Semantic matching via embedding similarity (nomic-embed-text, threshold 0.80).

---

### Finding 2: acceptable_alternatives Missing for L3–L6

L1 and L2 test cases define `acceptable_alternatives` — known-valid alternate values. L3 through L6 have zero. Any valid answer that doesn't exactly match the single expected value is marked wrong.

**Example:** L3 score_07 — model correctly doesn't apply a remote bonus for "hybrid acceptable location" but the ground truth expects the bonus to apply. The model's interpretation is at least as defensible as the ground truth's.

**Impact:** Inflates failure rates on L3–L6, particularly on subjective or ambiguous cases.

**Fix:** Add acceptable_alternatives to L3–L6 test cases. For L4 conclusions, allow ±1 level tolerance as partial credit.

---

### Finding 3: L6 Synthesis Capped by Rubric, Not Model Capability

All models tested cluster between 0.46 and 0.55 on L6 factual_coverage (with one outlier: qwen3.5:27b at 0.67). The scoring function uses crude word overlap with a hardcoded stopword list. A model that correctly synthesizes the same information using different phrasing scores low.

**Specific case:** synth_02 — qwen3:32b scores 0.33 coverage. synth_05 — qwen3:32b scores 0.27 coverage. These are large models producing substantive synthesis that happens to not use the expected keywords.

**Impact:** L6 scores measure rubric-matching ability, not synthesis quality. The `capabilities.synthesis` score in the router is essentially noise.

**Fix:** Replace keyword overlap with embedding similarity per expected fact. Threshold: 0.70 cosine.

---

### Finding 4: L4 Binary Scoring Creates Artificial Cliff for Small Models

`conclusion_accuracy` is binary: exact match on the conclusion enum (strong/moderate/weak/no_match) or zero. No partial credit. A model answering "moderate_match" when "weak_match" is expected (1 step off) gets the same 0.0 as one answering "strong_match" (2 steps off).

**Specific case:** qwen3.5:2b scores 0.10 on conclusion_accuracy despite reasonable reasoning_coverage. It's getting the reasoning right but landing on an adjacent conclusion — which binary scoring treats as completely wrong.

**Impact:** Small models appear catastrophically worse than they are. The step from 2b to 4b looks like 0.10 → 0.47, suggesting a massive capability jump. In reality, conclusion_distance tells a more nuanced story.

**Fix:** Replace binary accuracy with distance-based scoring: `max(0, 1.0 − distance / 3.0)`. A 1-step miss scores 0.67, a 2-step miss scores 0.33, exact match scores 1.0.

---

### Finding 5: Bonus/Penalty Ground Truth Opacity

L3 tracks bonus_accuracy and penalty_accuracy, but the criteria for whether a bonus or penalty *should* apply are ambiguous in edge cases.

**Specific case:** L3 score_07 — `mean_deviation=0.003` (excellent), `dims_within_tolerance=5.0` (perfect), `total_in_range=1.0` — but `bonus_accuracy=0.0`. The model correctly assessed all 5 dimensions nearly perfectly but got the bonus call "wrong" on an ambiguous criterion.

**Impact:** Bonus/penalty accuracy metrics are unreliable for model comparison. A model can be penalized for a defensible interpretation.

**Fix:** Review and recalibrate bonus/penalty expectations. Consider making bonus/penalty a soft metric (informational) rather than scored.

---

### Finding 6: Statistical Underpowering

| Level | Current cases | Min for 95% CI on 10% difference |
|-------|---------------|-----------------------------------|
| L1 | 7 | ~40 |
| L2 | 7 | ~40 |
| L3 | 7 | ~40 |
| L4 | 10 | ~40 |
| L5 | 3 groups | ~6 groups |
| L6 | 8 | ~15 |

With 7 cases, a 2–3 case swing from random variance moves field_accuracy by 20–30%. We currently compare point estimates (e.g., "0.88 vs 0.91") without confidence intervals, creating false confidence in differences that may be noise.

**Impact:** Model comparisons are unreliable. The difference between qwen3:8b (0.88 conditional) and qwen3.5:27b (0.88 conditional) might be real or might be noise — we can't tell.

**Fix:** Expand to 40 cases per level. Compute bootstrap CIs. Use significance testing (α=0.05) before claiming a model is better or worse.

---

### Finding 7: Triple Contamination

Three independent contamination vectors stack:

1. **Ground truth bias:** Expected answers were validated against qwen3:8b behavior. If qwen3:8b extracts "SaaS" as the domain, that becomes ground truth — rewarding models that think like qwen3:8b.

2. **Prompt optimization bias:** Autoresearch (Phase 4) tuned prompts against qwen3. The prompts now contain patterns, phrasings, and instruction styles that work well for qwen3. Other model families may parse these prompts differently.

3. **Synthetic distribution bias:** L3 is 43% synthetic (edge cases by design). Real-world job listing distributions don't have 43% edge cases. Models optimized for real distributions may underperform on this synthetic-heavy test.

**Impact:** qwen3.5 models are being evaluated on a test designed around qwen3 behavior. Lower scores might mean "doesn't think like qwen3" rather than "worse at the task."

**Fix:** Multi-model ground truth calibration (validate against ≥2 model families). Temporal rotation (refresh 10% quarterly from real listings). Cap synthetic at 25%. Dual scoring (production prompts vs generic prompts) to separate prompt-fit from capability.

---

## 5. Fleet Evaluation Results

### Full Fleet Comparison (2026-03-27)

All 8 models evaluated on the same test suite. n_runs=3 for qwen3.5 candidates, n_runs=5 for qwen3 baselines.

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

**Bold** = fleet best for that capability. Dashes = missing data (early runs before full test suite). qwen3.5:35b-a3b is MoE (256 experts, 3B active).

### Key Findings

**1. qwen3.5:27b is the new extraction/classification leader.**
Best-in-fleet on extraction (0.94), classification (1.00), and synthesis (0.67). 3 GB smaller than qwen3:32b (17 GB vs 20 GB). Perfect JSON output. But causal reasoning (0.57) is below qwen3:8b (0.60) and well below qwen3:14b (0.70).

**2. qwen3:14b remains the causal reasoning champion.**
At 0.70, no qwen3.5 model has matched it — not even the 27b. Combined with the highest conditional_reasoning (0.91) and nuanced_judgment (0.83), it's the best model for subjective judgment tasks. This may be a prompt-fit advantage (prompts tuned for qwen3) or genuine capability.

**3. The MoE speed story is real, but accuracy isn't.**
qwen3.5:35b-a3b (256 experts, 3B active) evaluates 3.8x faster than qwen3.5:27b dense (1490s vs 5732s). But it scores lower on every capability metric. Extraction consistency dropped to 0.80 (vs 1.00 for 27b), suggesting sparse expert routing introduces minor inconsistencies where every token matters in JSON schemas.

**4. qwen3.5:4b is the surprise performer.**
Nearly matches qwen3:8b on extraction (0.89 vs 0.90) and classification (0.97 = tied) at 35% less memory (3.4 GB vs 5.2 GB). Viable fast-tier candidate for simple extraction tasks.

**5. The synthesis ceiling was broken — partially.**
qwen3.5:27b hit 0.67, well above the 0.46–0.55 cluster. This complicates the "rubric is the bottleneck" hypothesis from Finding 3. The 27b may be better at matching the expected keywords, or it may genuinely synthesize closer to what the test expects. Further investigation needed.

**6. JSON reliability is a solved problem.**
All models score ≥0.98 on json_output. The qwen3/qwen3.5 families with `think: false` produce valid JSON essentially every time. This can be treated as a gate (pass/fail at 0.95) rather than a differentiating metric.

**7. Causal reasoning shows the sharpest model differentiation.**
Range: 0.10 (qwen3.5:2b) to 0.70 (qwen3:14b). This is where model selection matters most — and where the binary scoring cliff (Finding 4) distorts the picture most severely. Fixing L4 scoring is the highest-leverage improvement.

---

## 6. Key Insights & Conclusions

### Capability vs. Prompt-Fit

A model's eval score = model capability × prompt fit. Perian's prompts were optimized for qwen3. When qwen3.5:27b scores 0.57 on causal reasoning vs qwen3:14b at 0.70, we can't tell if 27b is genuinely worse at reasoning or if our prompts just don't suit it. **Dual scoring** (production prompts + generic prompts) separates these two factors.

### MoE Is Not a Free Lunch for Structured Output

The qwen3.5:35b-a3b MoE model delivered on its speed promise (3.8x faster) but not on accuracy. For structured JSON extraction — where every token in a schema matters and consistency across runs is critical — dense models have an edge. MoE may be better positioned as a fast-tier model for simple tasks rather than a heavy-tier replacement.

### The Rubric Is Often the Bottleneck

Three of seven failure findings (Findings 1, 3, 5) trace back to the scoring rubric measuring the wrong thing — not the model failing. Before concluding a model is worse, check whether the eval is capable of measuring the difference. The L6 synthesis ceiling is the clearest example: 6 models spanning 2B to 32B all scoring 0.46–0.55 is a rubric signal, not a model signal.

### Statistical Rigor Gaps Undermine Confidence

We're making fleet decisions based on point estimate comparisons without confidence intervals. The difference between 0.88 and 0.91 with 7 test cases is statistically meaningless. Until the test suite expands to ~40 cases per level with bootstrap CIs and significance testing, model comparisons should be treated as directional, not definitive.

### Contamination Is Systemic, Not Incidental

The three contamination vectors (ground truth bias, prompt optimization bias, synthetic distribution bias) don't just add noise — they systematically favor the incumbent model family. This makes fleet upgrades appear riskier than they are and creates inertia toward the status quo.

---

## 7. Recommendations

### Immediate (Before Fleet Decision)

1. **Run generic prompt capability evals** on qwen3.5:27b and qwen3:14b to separate prompt-fit from raw capability, especially on L4 causal reasoning.
2. **Don't force a wholesale fleet swap.** The data supports a specialist split: qwen3.5:27b for extraction/classification (where it leads) + qwen3:14b for causal/nuanced judgment (where it leads).

### Short-Term (PR 1 — Scoring Fixes)

3. **Create `evaluation/semantic.py`** — embedding-based matching via nomic-embed-text. Replace Jaccard on L1 requirements and word-overlap on L6 facts.
4. **Create `evaluation/stats.py`** — bootstrap CIs and significance testing. Add intervals to every capability comparison.
5. **Fix L4 scoring** — replace binary conclusion_accuracy with distance-based: `max(0, 1.0 − distance / 3.0)`.

### Medium-Term (PR 2 — Calibration + Test Expansion)

6. **Create `evaluation/calibration.py`** — ECE computation from logprobs, logprob-to-confidence conversion. Validate the router's -1.5 escalation threshold.
7. **Expand test suite to 40 cases per level** from real collected listings. Cap synthetic at 25%. Add acceptable_alternatives to L3–L6.

### Longer-Term (PR 3 — Decouple + Process)

8. **Implement dual scoring** — production prompts (prompt-fit eval) + generic prompts (capability eval) for every model comparison.
9. **Multi-model ground truth calibration** — validate expected answers against ≥2 model families.
10. **Quarterly test case rotation** — refresh 10% from newly collected real listings to combat temporal contamination.

### Fleet Configuration (Pending Generic Prompt Evals)

Proposed specialist fleet:

| Tier | Model | Role | Why |
|------|-------|------|-----|
| Fast | qwen3.5:4b (3.4 GB) | Simple extraction, first pass | Matches qwen3:8b extraction at 35% less memory |
| Medium | qwen3.5:27b (17 GB) | Extraction, classification, synthesis | Fleet-best on L1/L2/L6, 3 GB smaller than qwen3:32b |
| Specialist | qwen3:14b (9.3 GB) | Causal reasoning, nuanced judgment | Fleet-best on L4 and nuanced_judgment; no qwen3.5 model has matched it |

This replaces the current 3-model fleet (qwen3 8b/14b/32b) with a more capable configuration that uses 2 model families, each where they're strongest. Total memory footprint: 29.7 GB (vs 34.3 GB current).

---

## References

### Papers
- Zhang et al. "BERTScore: Evaluating Text Generation with BERT." arXiv:1904.09675. 2020.
- CMU SEI. "Beyond Capable: Accuracy, Calibration, and Robustness in LLMs."
- "Calibration Is Not Enough: Evaluating Confidence Estimation Under Language Variations." arXiv:2601.08064.
- Bao et al. "Cycles of Thought: Measuring LLM Confidence through Stable Explanations." arXiv:2406.03441v1.
- "Towards Provably Unbiased LLM Judges via Bias-Bounded Evaluation." arXiv:2603.05485.
- "All Leaks Count, Some Count More: Interpretable Temporal Contamination Detection." arXiv:2602.17234.
- "The Ultimate Guide to Fine-Tuning LLMs." arXiv:2408.13296v1.

### Frameworks
- LM Evaluation Harness (EleutherAI) — github.com/EleutherAI/lm-evaluation-harness
- HELM (Stanford) — crfm.stanford.edu/helm/latest/
- StructEval — arXiv:2505.20139v1
- LLMStructBench — arXiv:2602.14743
- EnviroLLM — arXiv:2512.12004v1
- aidatatools/ollama-benchmark — github.com/aidatatools/ollama-benchmark

### Companion Document
- [EVAL_METHODOLOGY.md](EVAL_METHODOLOGY.md) — the process/how-to reference that codifies this research into actionable procedures
