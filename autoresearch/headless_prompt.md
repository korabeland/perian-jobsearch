# Autoresearch — Headless Prompt Optimization

You are running the Perian autoresearch prompt optimization loop. Your job is to
iteratively improve a prompt template used by a local LLM (qwen3.5:9b-nvfp4) for structured
extraction or scoring of job listings.

## How it works

You propose prompt mutations. A deterministic eval pipeline (Python + Ollama) scores
them against holdout data with ground truth labels. Mutations land only when metrics
improve. You do NOT judge your own output — the numbers decide.

## Tools available

All eval logic runs through `autoresearch/eval_cli.py` subcommands:

```
python3 autoresearch/eval_cli.py evaluate --template {T} --items opt --output /tmp/ar_baseline.json
python3 autoresearch/eval_cli.py evaluate --template {T} --items opt --candidate /tmp/ar_candidate.md --output /tmp/ar_candidate_eval.json
python3 autoresearch/eval_cli.py evaluate --template {T} --items val --output /tmp/ar_val_baseline.json
python3 autoresearch/eval_cli.py evaluate --template {T} --items val --candidate /tmp/ar_candidate.md --output /tmp/ar_val_eval.json
python3 autoresearch/eval_cli.py failures --input /tmp/ar_baseline.json --max 8 --output /tmp/ar_failures.json
python3 autoresearch/eval_cli.py validate --candidate /tmp/ar_candidate.md --template {T}
python3 autoresearch/eval_cli.py safety-gate --model qwen3.5:9b-nvfp4 --output /tmp/ar_safety.json
```

Replace `{T}` with `extract` or `score` based on what you're optimizing (specified in your initial prompt).

## Protocol — repeat for each iteration

### Step 1: Baseline

Run eval on the current production prompt:
```
python3 autoresearch/eval_cli.py evaluate --template {T} --items opt --output /tmp/ar_baseline.json
```
Read `/tmp/ar_baseline.json`. Record `mean_score` and `pass_rate`.

If `mean_score > 0.95`: stop — prompt is at ceiling.

### Step 2: Analyze failures

```
python3 autoresearch/eval_cli.py failures --input /tmp/ar_baseline.json --max 8 --output /tmp/ar_failures.json
```
Read `/tmp/ar_failures.json`. Study the failure patterns — what is the model getting wrong?

Also read `autoresearch/eval_criteria.md` (the LOCKED rubric) to understand exactly what
"correct" means for each dimension.

### Step 3: Propose mutation

Based on the failure patterns, propose a targeted improvement to the prompt. Write the
complete new prompt to `/tmp/ar_candidate.md`.

Rules for mutations:
- Read the current prompt from `prompts/{T}.md` first
- Make the SMALLEST change that addresses the failures
- The prompt MUST contain the delimiter: `--- ITEM TO PROCESS ---`
- Preserve all placeholders: `{{listing}}` for extract, `{{extracted}}` + `{{profile}}` + `{{scoring.*}}` for score
- Prompt length must not increase by more than 25% from the original
- Prefer adding examples or clarifying existing rules over rewriting
- Do not remove existing examples unless replacing with better ones

### Step 4: Validate structure

```
python3 autoresearch/eval_cli.py validate --candidate /tmp/ar_candidate.md --template {T}
```
If validation fails, fix the candidate and re-validate. Do not proceed with an invalid mutation.

### Step 5: Evaluate candidate

```
python3 autoresearch/eval_cli.py evaluate --template {T} --items opt --candidate /tmp/ar_candidate.md --output /tmp/ar_candidate_eval.json
```
Read results. Compare `mean_score` against baseline.

### Step 6: Accept or reject

**Accept ONLY if candidate mean_score > baseline mean_score.**

No tie-breaking. No subjective judgment. If the numbers don't improve, reject.

If rejected: increment the consecutive rejection counter and go to next iteration.

### Step 7: Validation set check (accepted mutations only)

Evaluate on the validation holdout to catch overfitting:
```
python3 autoresearch/eval_cli.py evaluate --template {T} --items val --output /tmp/ar_val_baseline.json
python3 autoresearch/eval_cli.py evaluate --template {T} --items val --candidate /tmp/ar_candidate.md --output /tmp/ar_val_eval.json
```
If validation `mean_score` regresses by more than 0.01: reject the mutation.

### Step 8: Land the mutation (accepted + validated only)

1. Copy the candidate to the production prompt:
   ```
   cp /tmp/ar_candidate.md prompts/{T}.md
   ```
2. Git commit:
   ```
   git add prompts/{T}.md && git commit -m "autoresearch: {T} prompt — mean {old}→{new} (+{delta})"
   ```
3. Reset consecutive rejection counter.

### Step 9: Safety gate

```
python3 autoresearch/eval_cli.py safety-gate --model qwen3.5:9b-nvfp4 --output /tmp/ar_safety.json
```
Read results. If safety gate FAILS (capability regression > 3%):
```
git revert --no-edit HEAD
```
Report the regression and stop the loop.

## Termination conditions

Stop the loop when ANY of these is true:
- Completed the maximum number of iterations (specified in your initial prompt)
- 3 consecutive rejected mutations (prompt has plateaued)
- Baseline `mean_score > 0.95` (at ceiling)
- Safety gate failure (stop immediately, do not continue)

## Output

At the end, print a summary:
```
Autoresearch complete.
Iterations: N
Accepted: N
Rejected: N
Final mean_score: X.XXXX (was Y.YYYY)
```

## Important constraints

- Never modify `autoresearch/eval_criteria.md` — it is LOCKED
- Never skip the validation set check for accepted mutations
- Never skip the safety gate after landing a mutation
- Do not add markdown fences to prompt output instructions
- Each accepted mutation is independently committed — partial progress is safe
