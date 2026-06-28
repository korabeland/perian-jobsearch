# Perian — Job-Search Reference

A runnable, sanitized reference implementation of the job-search workflow that the
[Perian framework](https://github.com/korabeland/perian) was first built around.

Everything here runs **locally** against [Ollama](https://ollama.com) on **placeholder
data** — a fictional candidate profile and example job listings. There is no personal
data and no cloud API in the daily path. Swap in your own `profile.md` and
`config/watchlist.yaml` to use it for real.

> This repo is the *example*; the reusable engine (model router, evaluation
> methodology, knowledge base) lives in the [Perian framework repo](https://github.com/korabeland/perian).

## The pipeline

```
inputs/*.jsonl ─► ingest ─► dedup ─► prefilter ─► triage (2B) ─► extract (LLM) ─► score (LLM) ─► rank ─► output/briefing-{date}.md
```

- **ingest / dedup** — load listings from `inputs/`, drop ones already seen (`state/seen.json`).
- **prefilter** — cheap regex drop of clearly off-target titles (no model call).
- **triage** — a small 2B model makes a binary relevant/not-relevant call.
- **extract** — an LLM pulls structured fields (`prompts/extract.md`).
- **score** — an LLM scores each listing 0–1 against `profile.md` using the weights in
  `plan.yaml` (`prompts/score.md`); deterministic floor penalties are applied after.
- **rank / output** — filter by `min_score`, sort, and write a markdown briefing.

## Quickstart

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Make sure Ollama is running and the models exist
ollama pull qwen3.5:2b          # triage gate
ollama pull qwen3.5:9b          # extraction + scoring (or edit config/fleet.yaml)

# 3. Run the pipeline on the bundled placeholder listings
python3 scripts/runner.py --limit 2
```

The run writes a briefing to `output/briefing-{date}.md` and per-step logs to
`run-log/`. `state/`, `output/`, and `run-log/` are runtime artifacts and are
gitignored.

> **Models:** `config/fleet.yaml` lists the model fleet and the triage model is set in
> `scripts/runner.py`. Edit the `ollama_name` values to match the models you have
> installed locally.

## Evaluation

The `evaluation/` suite measures local-model capability with a 6-level battery
(extraction → classification → conditional scoring → causal reasoning → batch
consistency → synthesis), bootstrap confidence intervals, and a production-vs-capability
"prompt-fit gap". Ground truth lives in `test_cases.json` per level — the data here is
**placeholder** (fictional listings aligned to the placeholder profile).

```bash
# All 6 levels for one model (writes evaluation/results/{model}_{timestamp}.json)
python3 evaluation/run_all.py --model qwen3.5:9b --n-runs 3

# A subset of levels
python3 evaluation/run_all.py --model qwen3.5:9b --levels 2 3 4
```

Levels 1 and 6 use embedding-based semantic matching — `ollama pull nomic-embed-text`
first (levels 2–5 are pure string/numeric comparison). See
`evaluation/EVAL_METHODOLOGY.md` for the metrics, statistical design, and the model
upgrade decision framework.

## Autoresearch (prompt optimization)

`autoresearch/` is a closed loop that improves the `extract.md` / `score.md` prompts
against held-out data with ground-truth labels. A headless reasoning session proposes
prompt mutations; a deterministic eval gate accepts one **only if the numbers improve**,
then a safety gate guards against capability regressions. The rubric in
`autoresearch/eval_criteria.md` is locked — the loop never edits its own grading.

```bash
# One-time holdout generation (uses the Claude API; see autoresearch/readiness.md)
python3 autoresearch/generate_holdout.py --template both

# Run the optimization loop
python3 autoresearch/prompt_loop.py --template extract --max-iterations 5
```

## Layout

```
profile.md            ← candidate profile (placeholder — replace with your own)
plan.yaml             ← pipeline steps + scoring weights, bonuses, penalties, floors
prompts/              ← extract / score / challenge / stripped-profile templates
config/
  watchlist.yaml      ← companies to poll (placeholder)
  sources.yaml        ← job-board / ATS source config
  fleet.yaml          ← model fleet + capability profiles
  router.yaml         ← confidence-escalation config
  env.yaml            ← Ollama connection settings
scripts/
  runner.py           ← the pipeline (ingest → … → briefing)
  collect.py          ← fetch listings from job boards / ATS watchlists
  router.py           ← picks the cheapest qualifying model; escalates on low confidence
  ollama_utils.py     ← keep-alive / warm-up helpers
  schemas.py          ← JSON schemas for constrained decoding
  triage_eval.py      ← triage-gate evaluation
  feedback.py         ← record human verdicts on scored items
  enrich_queue.py     ← re-fetch + re-score low-signal items
  fleet_log.py        ← append-only fleet changelog (used by the eval harness)
inputs/sample.jsonl   ← placeholder listings for the smoke run
evaluation/
  run_all.py          ← 6-level capability eval harness
  test_suite/         ← levels 1–6: run_test.py + placeholder test_cases.json
  helpers.py          ← Ollama client, prompt builders, logprob stats
  stats.py            ← bootstrap CIs + significance tests
  semantic.py         ← embedding-based matching (nomic-embed-text)
  EVAL_METHODOLOGY.md ← metrics, test design, upgrade framework
autoresearch/
  prompt_loop.py      ← optimization loop (propose → eval → accept/reject → safety gate)
  generate_holdout.py ← one-time holdout generation (Claude API)
  eval_criteria.md    ← LOCKED grading rubric
```

## License

MIT © 2026 Korab Eland
