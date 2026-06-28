# Autoresearch Pre-Flight Checklist

Before running the optimization loop, verify each item:

## Environment

- [ ] **ANTHROPIC_API_KEY** set in environment: `echo $ANTHROPIC_API_KEY`
- [ ] **Ollama running**: `curl -s http://localhost:11434/api/tags | jq '.models | length'`
- [ ] **Local model loaded**: `ollama list | grep qwen3.5:9b-nvfp4`
- [ ] **Python venv active**: `python3 -c "import anthropic; print('OK')"`

## Git State

- [ ] **Clean working tree**: `git status` shows no uncommitted changes
- [ ] **On main branch** (or a feature branch): `git branch --show-current`

## Data

- [ ] **Holdout data generated**: `ls autoresearch/holdout/extract_holdout.json`
- [ ] **Holdout has baseline**: `python3 -c "import json; d=json.load(open('autoresearch/holdout/extract_holdout.json')); print(f'{len(d)} items, {sum(1 for i in d if i.get(\"baseline_correct\"))} correct')"`

## Resources

- [ ] **Power adapter connected** (loop can run 30+ minutes with Ollama)
- [ ] **Cost bounded**: The loop uses a headless Claude Code subscription (no per-call API cost); only one-time holdout generation hits the API (~$0.20). Bound iterations via `loop.max_iterations`.
- [ ] **Sufficient disk**: Holdout + cost log are small, but eval results can accumulate

## Running

```bash
# Generate holdout data (one-time, costs ~$0.20)
export ANTHROPIC_API_KEY="sk-..."
python3 autoresearch/generate_holdout.py --template both

# Run optimization loop
python3 autoresearch/prompt_loop.py --template extract --max-iterations 5 --verbose

# Evaluate-only (no mutations, no cost)
python3 autoresearch/prompt_loop.py --template extract --max-iterations 0 --evaluate-only
```
