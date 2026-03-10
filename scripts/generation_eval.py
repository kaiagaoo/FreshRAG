"""
Generation Stage Evaluation Pipeline
======================================
Stage 4 of the FreshRAG experiment pipeline.

Takes generation-ready payloads from Stage 3c (context assembly) and
prompts Gemini to produce a final answer conditioned on the assembled
context window.

Metrics tracked:
- Input tokens: prompt token count reported by the API
- Output tokens: completion token count reported by the API
- Generation latency (ms): wall-clock time per API call
- API cost estimate: output_tokens × price_per_token
- Response length (words): whitespace-split word count of the generated answer

Usage (run from repo root):
  python scripts/generation_eval.py --corpus_dir ./freshrag_experiment
  python scripts/generation_eval.py --corpus_dir ./freshrag_experiment --model gemini-2.5-flash
"""

import json
import os
import time
import argparse
import numpy as np
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEFAULT_MODEL = "gemini-2.5-flash"
CONDITIONS = ["fresh", "stale_10", "stale_30", "stale_50"]

# Gemini pricing per 1M tokens (as of 2025 — adjust if needed)
# https://ai.google.dev/pricing
PRICING = {
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
}

GENERATION_PROMPT = """You are a helpful assistant. Answer the user's question based ONLY on the provided context. If the context does not contain enough information to answer, say so.

Context:
{context}

Question: {question}

Answer:"""


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def call_gemini(prompt, api_key, model):
    """Call Gemini API and return generated text + usage metadata."""
    import requests

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1024,
        },
    }

    t0 = time.perf_counter()
    response = requests.post(url, json=payload, timeout=120)
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000

    response.raise_for_status()
    result = response.json()

    # Extract generated text
    try:
        text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        text = ""

    # Extract token usage from usageMetadata
    usage = result.get("usageMetadata", {})
    input_tokens = usage.get("promptTokenCount", 0)
    output_tokens = usage.get("candidatesTokenCount", 0)

    return {
        "generated_answer": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
    }


def estimate_cost(input_tokens, output_tokens, model):
    """Estimate API cost in USD."""
    prices = PRICING.get(model, PRICING[DEFAULT_MODEL])
    input_cost = input_tokens * prices["input"] / 1_000_000
    output_cost = output_tokens * prices["output"] / 1_000_000
    return input_cost + output_cost


# ─────────────────────────────────────────────
# AGGREGATION
# ─────────────────────────────────────────────
def _agg(results):
    """Compute mean and std for all numeric metrics."""
    if not results:
        return {}

    metric_keys = [
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "cost_usd",
        "response_length_words",
    ]

    agg = {"n": len(results)}
    for key in metric_keys:
        values = [r[key] for r in results if key in r]
        if values:
            agg[f"{key}_mean"] = np.mean(values)
            agg[f"{key}_std"] = np.std(values)
            agg[f"{key}_sum"] = np.sum(values)
    return agg


def print_summary_table(all_results, conditions, model):
    """Print a formatted summary table across conditions."""
    print("\n" + "=" * 100)
    print(f"GENERATION EVALUATION SUMMARY (model={model})")
    print("=" * 100)

    header = f"{'Metric':<35}"
    for cond in conditions:
        header += f"  {cond:>14}"
    print(f"\n{header}")
    print("-" * (35 + 16 * len(conditions)))

    metrics_to_show = [
        ("Input tokens (mean)", "input_tokens_mean", ".1f"),
        ("Output tokens (mean)", "output_tokens_mean", ".1f"),
        ("Generation latency ms (mean)", "latency_ms_mean", ".1f"),
        ("Response length words (mean)", "response_length_words_mean", ".1f"),
        ("Cost per query USD (mean)", "cost_usd_mean", ".6f"),
        ("Total cost USD", "cost_usd_sum", ".4f"),
    ]

    cond_aggs = {}
    for cond in conditions:
        cond_results = [r for r in all_results if r["condition"] == cond]
        cond_aggs[cond] = _agg(cond_results)

    for label, key, fmt in metrics_to_show:
        row = f"{label:<35}"
        for cond in conditions:
            val = cond_aggs[cond].get(key, 0)
            row += f"  {val:>{14}{fmt}}"
        print(row)

    # ── By time sensitivity ──
    print(f"\n{'─' * 100}")
    print("BREAKDOWN: Time-sensitive vs. Time-insensitive queries")
    print(f"{'─' * 100}")

    for ts_label, ts_val in [("Time-sensitive", True), ("Time-insensitive", False)]:
        print(f"\n  {ts_label}:")
        header = f"  {'Metric':<33}"
        for cond in conditions:
            header += f"  {cond:>14}"
        print(header)
        print("  " + "-" * (33 + 16 * len(conditions)))

        for label, key, fmt in [
            ("Input tokens (mean)", "input_tokens_mean", ".1f"),
            ("Output tokens (mean)", "output_tokens_mean", ".1f"),
            ("Latency ms (mean)", "latency_ms_mean", ".1f"),
            ("Response length words", "response_length_words_mean", ".1f"),
            ("Cost per query USD", "cost_usd_mean", ".6f"),
        ]:
            row = f"  {label:<33}"
            for cond in conditions:
                cond_ts = [
                    r for r in all_results
                    if r["condition"] == cond and r["time_sensitive"] == ts_val
                ]
                agg = _agg(cond_ts)
                val = agg.get(key, 0)
                row += f"  {val:>{14}{fmt}}"
            print(row)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generation stage evaluation")
    parser.add_argument(
        "--corpus_dir",
        type=str,
        required=True,
        help="Directory containing freshrag_experiment results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=CONDITIONS,
        help="Which conditions to evaluate (default: all four)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: corpus_dir/results)",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Max queries per condition (for testing/debugging)",
    )
    args = parser.parse_args()

    corpus_dir = args.corpus_dir
    model = args.model
    conditions = args.conditions
    output_dir = args.output_dir or os.path.join(corpus_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    # ── Check API key ──
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        # Try loading from .env
        env_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
        )
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GOOGLE_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip("'\"")
                        break

    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY environment variable.")
        print("  Example: export GOOGLE_API_KEY='your-key-here'")
        return

    print("=" * 60)
    print("GENERATION STAGE EVALUATION")
    print(f"  Model: {model}")
    print(f"  Conditions: {conditions}")
    print(f"  Corpus dir: {corpus_dir}")
    print(f"  Output dir: {output_dir}")
    if args.max_queries:
        print(f"  Max queries per condition: {args.max_queries}")
    print("=" * 60)

    # ── Load generation payloads ──
    payloads_path = os.path.join(output_dir, "generation_payloads.jsonl")
    if not os.path.exists(payloads_path):
        print(f"\n  ERROR: {payloads_path} not found.")
        print("  Run context_assembly_eval.py first to generate payloads.")
        return

    payloads = load_jsonl(payloads_path)
    print(f"\n  Loaded {len(payloads)} generation payloads")

    # ── Check for existing progress (resume support) ──
    results_path = os.path.join(output_dir, "generation_results_detailed.jsonl")
    existing = {}
    if os.path.exists(results_path):
        existing_data = load_jsonl(results_path)
        existing = {(r["query_id"], r["condition"]): r for r in existing_data}
        print(f"  Found {len(existing)} already generated (resuming)")

    # ── Generate answers ──
    all_results = list(existing.values())
    total_cost = sum(r.get("cost_usd", 0) for r in all_results)

    for cond in conditions:
        cond_payloads = [p for p in payloads if p["condition"] == cond]
        if args.max_queries:
            cond_payloads = cond_payloads[: args.max_queries]

        # Filter out already-completed
        to_process = [
            p for p in cond_payloads
            if (p["query_id"], p["condition"]) not in existing
        ]

        print(f"\n{'─' * 60}")
        print(f"  CONDITION: {cond}")
        print(f"  Total payloads: {len(cond_payloads)}, remaining: {len(to_process)}")
        print(f"{'─' * 60}")

        for i, payload in enumerate(to_process):
            query_id = payload["query_id"]
            question = payload["question"]
            context = payload["context"]

            prompt = GENERATION_PROMPT.format(context=context, question=question)

            try:
                gen_result = call_gemini(prompt, api_key, model)
            except Exception as e:
                print(f"    [{i+1}/{len(to_process)}] {query_id} — ERROR: {e}")
                # Rate limit back-off
                if "429" in str(e) or "Resource" in str(e):
                    print("    Rate limited, waiting 30s...")
                    time.sleep(30)
                    try:
                        gen_result = call_gemini(prompt, api_key, model)
                    except Exception as e2:
                        print(f"    Retry failed: {e2}")
                        continue
                else:
                    continue

            generated_answer = gen_result["generated_answer"]
            input_tokens = gen_result["input_tokens"]
            output_tokens = gen_result["output_tokens"]
            latency_ms = gen_result["latency_ms"]
            cost_usd = estimate_cost(input_tokens, output_tokens, model)
            response_length_words = len(generated_answer.split())
            total_cost += cost_usd

            result = {
                "query_id": query_id,
                "condition": cond,
                "domain": payload["domain"],
                "time_sensitive": payload["time_sensitive"],
                "model": model,
                "question": question,
                "ground_truth": payload["ground_truth"],
                "generated_answer": generated_answer,
                "context_tokens": payload["context_tokens"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
                "response_length_words": response_length_words,
                # Carry forward context assembly metadata
                "assembled_doc_ids": payload["assembled_doc_ids"],
                "answer_bearing_in_context": payload["answer_bearing_in_context"],
                "num_chunks": payload["num_chunks"],
                "contradiction_density": payload["contradiction_density"],
                "redundancy_score": payload["redundancy_score"],
                "stale_token_ratio": payload["stale_token_ratio"],
            }
            all_results.append(result)
            existing[(query_id, cond)] = result

            if (i + 1) % 10 == 0 or (i + 1) == len(to_process):
                print(
                    f"    [{i+1}/{len(to_process)}] "
                    f"in={input_tokens} out={output_tokens} "
                    f"lat={latency_ms:.0f}ms words={response_length_words} "
                    f"cost=${cost_usd:.5f}  (total=${total_cost:.4f})"
                )

            # Incremental save every 20 queries
            if (i + 1) % 20 == 0:
                save_jsonl(all_results, results_path)

            # Small delay to avoid rate limits
            time.sleep(0.5)

        # Save after each condition
        save_jsonl(all_results, results_path)

        # Quick summary
        cond_results = [r for r in all_results if r["condition"] == cond]
        agg = _agg(cond_results)
        print(f"\n  Quick summary for {cond} ({agg.get('n', 0)} queries):")
        print(f"    Input tokens (mean):     {agg.get('input_tokens_mean', 0):.1f}")
        print(f"    Output tokens (mean):    {agg.get('output_tokens_mean', 0):.1f}")
        print(f"    Latency ms (mean):       {agg.get('latency_ms_mean', 0):.1f}")
        print(f"    Response words (mean):   {agg.get('response_length_words_mean', 0):.1f}")
        print(f"    Cost (total):            ${agg.get('cost_usd_sum', 0):.4f}")

    # ── Print full summary ──
    evaluated_conditions = [c for c in conditions if any(r["condition"] == c for r in all_results)]
    print_summary_table(all_results, evaluated_conditions, model)

    # ── Save detailed results (final) ──
    save_jsonl(all_results, results_path)
    print(f"\n  Detailed results saved to {results_path}")

    # ── Save aggregated results ──
    agg_results = {
        "overall": {},
        "by_time_sensitivity": {},
        "by_domain": {},
    }

    for cond in evaluated_conditions:
        cond_results = [r for r in all_results if r["condition"] == cond]
        agg_results["overall"][cond] = _agg(cond_results)

        for ts_val, ts_label in [(True, "time_sensitive"), (False, "time_insensitive")]:
            ts_results = [r for r in cond_results if r["time_sensitive"] == ts_val]
            key = f"{cond}__{ts_label}"
            agg_results["by_time_sensitivity"][key] = _agg(ts_results)

        domains = set(r["domain"] for r in cond_results)
        for domain in domains:
            domain_results = [r for r in cond_results if r["domain"] == domain]
            key = f"{cond}__{domain}"
            agg_results["by_domain"][key] = _agg(domain_results)

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        return obj

    agg_results = convert_numpy(agg_results)

    agg_path = os.path.join(output_dir, "generation_results_aggregated.json")
    with open(agg_path, "w") as f:
        json.dump(agg_results, f, indent=2)
    print(f"  Aggregated results saved to {agg_path}")

    # ── Total cost ──
    print(f"\n  TOTAL API COST: ${total_cost:.4f}")

    print(f"\n{'=' * 60}")
    print("GENERATION EVALUATION COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
