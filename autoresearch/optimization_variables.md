# Autoresearch Optimization Variables

Variables that affect output quality in the Perian pipeline, ranked by impact and implementation effort.

## Currently Optimized (Phase 4)

### Prompt Template Text
- **What:** Instructions, rules, few-shot examples, output schema description in `prompts/extract.md` and `prompts/score.md`
- **How:** Claude API proposes mutations, local model evaluates, improvements committed
- **Impact:** High — prompt wording directly controls what the model extracts and how it scores

## High-Value Extensions (post Phase 4)

### Inference Parameters (temperature, top_p, top_k, repeat_penalty)
- **What:** Ollama model generation parameters. Currently using defaults (temperature ~0.7).
- **Why it matters:** Temperature controls consistency vs. creativity tradeoff. A prompt optimized at temp=0.3 may fail at temp=0.7. Lower temperature = more deterministic extraction, which is usually what we want for structured JSON output.
- **How to add:** Grid search over a few values (e.g., temp=[0.1, 0.3, 0.5, 0.7]) for each template. `evaluate_prompt()` already passes config to `call_ollama()`, so adding `{"temperature": X}` to options is a one-line change.
- **Effort:** Small — config addition + loop extension
- **Priority:** First extension after the loop is proven

### think Parameter (per-step)
- **What:** qwen3's internal chain-of-thought reasoning. Currently hardcoded to `False` for speed.
- **Why it matters:** Phase 3 measured that think=True helps small models significantly (8B gained +0.17 on L4 causal reasoning) but costs 3-7x slowdown. For extraction (structured task), think=False is clearly better. For scoring (reasoning task), think=True might improve quality enough to justify the cost.
- **How to add:** Test think=True vs think=False per template. Could also enable think selectively for items where the model's logprob confidence is low (complementing the escalation system).
- **Effort:** Small — already supported in `call_ollama(think=bool)`
- **Priority:** High for scoring template, low for extraction template

### Few-Shot Example Selection
- **What:** Which examples are embedded in the prompt templates. Currently 2 fixed examples per template.
- **Why it matters:** Examples teach the model output format AND edge case handling. Different example sets may help with different failure patterns. The loop currently treats examples as part of the prompt text — Claude can modify them. A dedicated example optimization would be more systematic.
- **How to add:** Maintain an example bank (10-15 examples per template). Loop tests different subsets of 2-4 examples. Measure which combination maximizes accuracy.
- **Effort:** Medium — needs example bank creation + selection logic
- **Priority:** Medium — try after prompt text + temperature are optimized

### num_ctx (Context Window Size)
- **What:** Ollama context window. Currently 8192 tokens.
- **Why it matters:** Long job listings (~500+ words) plus the prompt prefix may approach the limit. Truncation would silently degrade quality. Larger context = more VRAM + slower inference.
- **How to add:** Test 8192 vs 16384. Measure whether any holdout items improve with larger context.
- **Effort:** Tiny — config change
- **Priority:** Low unless we see truncation-related failures

## Architectural Extensions (future phases)

### Per-Item Model Routing Thresholds
- **What:** The logprob threshold (-1.5) that triggers confidence escalation from 8B to 32B.
- **Why it matters:** Too aggressive = expensive, too conservative = quality loss on hard items. Phase 2 built the escalation system; autoresearch could optimize the threshold.
- **How to add:** Test different thresholds against holdout items where ground truth is known. Find the threshold that maximizes accuracy per dollar.
- **Effort:** Medium — needs cost-aware evaluation metric
- **Priority:** After real data collection is in place

### Scoring Dimension Weights
- **What:** The weights in plan.yaml (title_match: 0.25, domain_match: 0.20, etc.) that determine final score.
- **Why it matters:** Hand-tuned weights may not match actual user preferences. A "domain_match" might matter more than "title_match" in practice.
- **How to add:** Needs labeled preference data — pairs of listings where the user says "I prefer A over B." Optimize weights to match human ranking. This is more of a Phase 5 (fine-tuning) concern.
- **Effort:** High — needs preference collection + optimization framework
- **Priority:** After accumulating real user feedback

### Post-Processing Heuristics
- **What:** Rules applied after model output (inline math fixer, JSON fence stripper). Currently in `parse_json_response()`.
- **Why it matters:** Models sometimes produce systematically broken output that a simple regex can fix. The loop could discover new patterns from failure cases.
- **How to add:** Analyze failure cases for systematic patterns. Add targeted fixers.
- **Effort:** Low per heuristic, but discovery is manual
- **Priority:** Opportunistic — add when patterns emerge from failure analysis
