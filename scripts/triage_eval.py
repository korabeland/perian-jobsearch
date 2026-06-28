"""Triage gate evaluation — 2x2 matrix: (2B vs 4B) × (title-only vs title+snippet).

Measures whether a cheap model can reliably classify job listings as relevant
before sending them to the expensive extraction step.

Usage: python3 scripts/triage_eval.py
"""

import json
import time
import requests
from pathlib import Path

from schemas import TRIAGE_SCHEMA

ROOT = Path(__file__).resolve().parent.parent
EVAL_SET = ROOT / "state" / "triage-eval-set.json"
OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT_V1 = """You are a job listing classifier. Given a job listing, decide if it is relevant to this profile:

Target roles: Customer Experience, Customer Success, Product Management, Customer Operations, Analytics
Target seniority: Manager through Director level
Target domains: SaaS, B2B technology, AI/ML, developer tools

Respond with ONLY a JSON object: {"relevant": true} or {"relevant": false}
Do not explain. Do not add any other text."""

SYSTEM_PROMPT_V2 = """You are a job listing pre-screener. Your job is to reject only the OBVIOUS mismatches. When in doubt, say relevant.

Reject these: trades, medical, food service, legal, accounting, pure engineering/development, retail, logistics, administrative assistants, recruitment.

Keep anything that MIGHT relate to: customer success, customer experience, product management, operations, strategy, AI, data, analytics, marketing, or leadership at a tech company.

If the title is ambiguous, say relevant. It is better to let a bad one through than to block a good one.

Respond with ONLY a JSON object: {"relevant": true} or {"relevant": false}"""

SYSTEM_PROMPT = SYSTEM_PROMPT_V2

TITLE_ONLY_TEMPLATE = "Title: {title}\nCompany: {company}"

SNIPPET_TEMPLATE = """Title: {title}
Company: {company}

Description excerpt:
{snippet}"""


def call_ollama(model: str, prompt: str) -> tuple[dict, float]:
    """Call Ollama and return parsed response + latency."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "format": TRIAGE_SCHEMA,
        "stream": False,
        "options": {"temperature": 0},
        "think": False,
    }
    start = time.time()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
    elapsed = time.time() - start
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"relevant": None, "error": content}
    return parsed, elapsed


def run_condition(items: list[dict], model: str, use_snippet: bool) -> list[dict]:
    """Run one condition of the eval matrix."""
    results = []
    label = f"{model} / {'snippet' if use_snippet else 'title-only'}"
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"{'='*60}")

    for i, item in enumerate(items):
        if use_snippet:
            prompt = SNIPPET_TEMPLATE.format(
                title=item["title"],
                company=item["company"],
                snippet=item["description_snippet"][:500],  # cap snippet length
            )
        else:
            prompt = TITLE_ONLY_TEMPLATE.format(
                title=item["title"],
                company=item["company"],
            )

        resp, elapsed = call_ollama(model, prompt)
        predicted = resp.get("relevant")
        actual = item["label"] == "relevant"
        correct = predicted == actual

        marker = "✓" if correct else "✗"
        print(
            f"  [{i+1:2d}/40] {marker} {elapsed:.1f}s | "
            f"pred={'Y' if predicted else 'N'} actual={'Y' if actual else 'N'} | "
            f"{item['title'][:45]} @ {item['company'][:15]}"
        )

        results.append({
            "item_id": item["item_id"],
            "title": item["title"],
            "company": item["company"],
            "actual": item["label"],
            "actual_score": item["total_score"],
            "predicted_relevant": predicted,
            "correct": correct,
            "latency_s": round(elapsed, 2),
        })

    return results


def score_results(results: list[dict], label: str) -> dict:
    """Compute precision, recall, accuracy, and timing."""
    tp = sum(1 for r in results if r["predicted_relevant"] and r["actual"] == "relevant")
    fp = sum(1 for r in results if r["predicted_relevant"] and r["actual"] == "irrelevant")
    fn = sum(1 for r in results if not r["predicted_relevant"] and r["actual"] == "relevant")
    tn = sum(1 for r in results if not r["predicted_relevant"] and r["actual"] == "irrelevant")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / len(results) if results else 0
    avg_latency = sum(r["latency_s"] for r in results) / len(results) if results else 0

    # False negatives are the critical failures — relevant items the gate would block
    false_negatives = [r for r in results if not r["predicted_relevant"] and r["actual"] == "relevant"]

    return {
        "label": label,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "accuracy": round(accuracy, 3),
        "avg_latency_s": round(avg_latency, 2),
        "total_time_s": round(sum(r["latency_s"] for r in results), 1),
        "false_negatives": [
            f"{r['title']} @ {r['company']} (score={r['actual_score']})"
            for r in false_negatives
        ],
    }


def main():
    items = json.load(open(EVAL_SET))
    print(f"Loaded {len(items)} eval items")
    print(f"  Relevant: {sum(1 for i in items if i['label'] == 'relevant')}")
    print(f"  Irrelevant: {sum(1 for i in items if i['label'] == 'irrelevant')}")

    conditions = [
        ("qwen3.5:2b", False, "2B / title-only"),
        ("qwen3.5:2b", True,  "2B / title+snippet"),
        ("qwen3.5:4b", False, "4B / title-only"),
        ("qwen3.5:4b", True,  "4B / title+snippet"),
    ]

    all_results = {}
    for model, use_snippet, label in conditions:
        results = run_condition(items, model, use_snippet)
        all_results[label] = score_results(results, label)

    # Print summary table
    print(f"\n{'='*60}")
    print("RESULTS MATRIX")
    print(f"{'='*60}")
    print(f"{'Condition':<25} {'Acc':>5} {'Prec':>5} {'Rec':>5} {'FN':>3} {'Avg(s)':>6} {'Total':>6}")
    print("-" * 60)
    for label, s in all_results.items():
        print(
            f"{label:<25} {s['accuracy']:>5.1%} {s['precision']:>5.1%} "
            f"{s['recall']:>5.1%} {s['fn']:>3d} {s['avg_latency_s']:>6.2f} {s['total_time_s']:>6.1f}"
        )

    # Show false negatives (the critical failures)
    print(f"\n{'='*60}")
    print("FALSE NEGATIVES (relevant items the gate would block)")
    print(f"{'='*60}")
    for label, s in all_results.items():
        if s["false_negatives"]:
            print(f"\n{label}:")
            for fn in s["false_negatives"]:
                print(f"  ✗ {fn}")
        else:
            print(f"\n{label}: none — perfect recall")

    # Save full results
    output_path = ROOT / "state" / "triage-eval-results.json"
    json.dump(all_results, open(output_path, "w"), indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
