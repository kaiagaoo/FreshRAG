"""
Temporal-Aware Verification Stage Evaluation Pipeline
=======================================================
Temporal variant of verification_eval.py.

Takes temporal generation results and verifies factual consistency.
Reads from results_temporal/, writes to results_temporal/.

Usage (run from repo root):
  python scripts/verification_temporal_eval.py --corpus_dir ./freshrag_experiment
  python scripts/verification_temporal_eval.py --corpus_dir ./freshrag_experiment --max_regen_attempts 2
"""

import json
import os
import re
import time
import argparse
import numpy as np

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
NLI_LABELS = ["contradiction", "entailment", "neutral"]
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
CONDITIONS = ["fresh", "stale_10", "stale_30", "stale_50"]
MAX_REGEN_ATTEMPTS = 1  # default: one regeneration attempt per failed answer

# Gemini pricing per 1M tokens
PRICING = {
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
}

REGEN_PROMPT = """You are a helpful assistant. Answer the user's question based ONLY on the provided context. Do NOT add any information that is not directly stated in the context. If the context does not contain enough information, say "I cannot answer this based on the provided context."

Your previous answer was flagged as potentially inconsistent with the context. Please provide a revised answer that is strictly grounded in the context below.

Context:
{context}

Question: {question}

Previous answer (flagged): {previous_answer}

Revised answer:"""


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


def split_sentences(text):
    """Simple sentence splitter."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


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
            "temperature": 0.2,
            "maxOutputTokens": 1024,
        },
    }

    t0 = time.perf_counter()
    response = requests.post(url, json=payload, timeout=120)
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000

    response.raise_for_status()
    result = response.json()

    try:
        text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        text = ""

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
    prices = PRICING.get(model, PRICING[DEFAULT_GEMINI_MODEL])
    input_cost = input_tokens * prices["input"] / 1_000_000
    output_cost = output_tokens * prices["output"] / 1_000_000
    return input_cost + output_cost


# ─────────────────────────────────────────────
# NLI VERIFIER
# ─────────────────────────────────────────────
class NLIVerifier:
    """Verify factual consistency of an answer against context using NLI."""

    def __init__(self, model_name=NLI_MODEL_NAME):
        from sentence_transformers import CrossEncoder
        import torch

        device = "cpu" if not torch.cuda.is_available() else "cuda"
        print(f"  Loading NLI verifier: {model_name} (device={device})")
        self.model = CrossEncoder(model_name, device=device)

    def verify(self, context, answer):
        """
        Check if the answer is entailed by the context.

        For short answers (single sentence), checks the whole answer.
        For longer answers, splits into sentences and checks each one.
        The answer-level verdict is the worst-case: one contradiction
        or neutral sentence fails the whole answer.

        Returns:
            entailed: bool — True if answer passes verification
            verdict: str — 'entailment', 'contradiction', or 'neutral'
            sentence_results: list of per-sentence dicts
        """
        sentences = split_sentences(answer)
        if not sentences:
            # Fallback: treat entire answer as one unit
            sentences = [answer.strip()]

        sentence_results = []
        worst_label = "entailment"
        worst_priority = {"entailment": 0, "neutral": 1, "contradiction": 2}

        # Truncate context to avoid exceeding model max length (512 tokens for DeBERTa)
        # Keep first ~400 words as representative
        context_words = context.split()
        if len(context_words) > 400:
            context_truncated = " ".join(context_words[:400])
        else:
            context_truncated = context

        pairs = [[context_truncated, sent] for sent in sentences]
        scores = self.model.predict(pairs)

        for sent, score_arr in zip(sentences, scores):
            label_idx = int(np.argmax(score_arr))
            label = NLI_LABELS[label_idx]
            sentence_results.append({
                "sentence": sent,
                "label": label,
                "scores": score_arr.tolist(),
            })
            if worst_priority.get(label, 0) > worst_priority.get(worst_label, 0):
                worst_label = label

        entailed = worst_label == "entailment"
        return entailed, worst_label, sentence_results


# ─────────────────────────────────────────────
# AGGREGATION
# ─────────────────────────────────────────────
def _agg(results):
    """Compute mean and std for all numeric metrics."""
    if not results:
        return {}

    metric_keys = [
        "entailment_failure",
        "regeneration_triggered",
        "total_cost_usd",
        "generation_cost_usd",
        "regen_cost_usd",
        "regen_attempts",
        "verification_latency_ms",
        "regen_latency_ms",
        "final_response_length_words",
        "entailed_after_regen",
    ]

    agg = {"n": len(results)}
    for key in metric_keys:
        values = [r[key] for r in results if key in r]
        if values:
            agg[f"{key}_mean"] = np.mean(values)
            agg[f"{key}_std"] = np.std(values)
            agg[f"{key}_sum"] = np.sum(values)
    return agg


def print_summary_table(all_results, conditions):
    """Print a formatted summary table across conditions."""
    print("\n" + "=" * 100)
    print("VERIFICATION EVALUATION SUMMARY")
    print("=" * 100)

    header = f"{'Metric':<40}"
    for cond in conditions:
        header += f"  {cond:>12}"
    print(f"\n{header}")
    print("-" * (40 + 14 * len(conditions)))

    metrics_to_show = [
        ("Entailment failure rate", "entailment_failure_mean", ".4f"),
        ("Regeneration trigger rate", "regeneration_triggered_mean", ".4f"),
        ("Entailed after regen", "entailed_after_regen_mean", ".4f"),
        ("Generation cost USD (mean)", "generation_cost_usd_mean", ".6f"),
        ("Regen cost USD (mean)", "regen_cost_usd_mean", ".6f"),
        ("Total cost/answer USD (mean)", "total_cost_usd_mean", ".6f"),
        ("Total cost USD (sum)", "total_cost_usd_sum", ".4f"),
        ("Verification latency ms (mean)", "verification_latency_ms_mean", ".1f"),
        ("Regen latency ms (mean)", "regen_latency_ms_mean", ".1f"),
        ("Final response words (mean)", "final_response_length_words_mean", ".1f"),
    ]

    cond_aggs = {}
    for cond in conditions:
        cond_results = [r for r in all_results if r["condition"] == cond]
        cond_aggs[cond] = _agg(cond_results)

    for label, key, fmt in metrics_to_show:
        row = f"{label:<40}"
        for cond in conditions:
            val = cond_aggs[cond].get(key, 0)
            row += f"  {val:>{12}{fmt}}"
        print(row)

    # ── By time sensitivity ──
    print(f"\n{'─' * 100}")
    print("BREAKDOWN: Time-sensitive vs. Time-insensitive queries")
    print(f"{'─' * 100}")

    for ts_label, ts_val in [("Time-sensitive", True), ("Time-insensitive", False)]:
        print(f"\n  {ts_label}:")
        header = f"  {'Metric':<38}"
        for cond in conditions:
            header += f"  {cond:>12}"
        print(header)
        print("  " + "-" * (38 + 14 * len(conditions)))

        for label, key, fmt in [
            ("Entailment failure rate", "entailment_failure_mean", ".4f"),
            ("Regeneration trigger rate", "regeneration_triggered_mean", ".4f"),
            ("Total cost/answer USD", "total_cost_usd_mean", ".6f"),
            ("Final response words", "final_response_length_words_mean", ".1f"),
        ]:
            row = f"  {label:<38}"
            for cond in conditions:
                cond_ts = [
                    r for r in all_results
                    if r["condition"] == cond and r["time_sensitive"] == ts_val
                ]
                agg = _agg(cond_ts)
                val = agg.get(key, 0)
                row += f"  {val:>{12}{fmt}}"
            print(row)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Verification stage evaluation")
    parser.add_argument(
        "--corpus_dir",
        type=str,
        required=True,
        help="Directory containing freshrag_experiment results",
    )
    parser.add_argument(
        "--gemini_model",
        type=str,
        default=DEFAULT_GEMINI_MODEL,
        help=f"Gemini model for regeneration (default: {DEFAULT_GEMINI_MODEL})",
    )
    parser.add_argument(
        "--max_regen_attempts",
        type=int,
        default=MAX_REGEN_ATTEMPTS,
        help=f"Max regeneration attempts per failed answer (default: {MAX_REGEN_ATTEMPTS})",
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
        help="Output directory for results (default: corpus_dir/results_temporal)",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Max queries per condition (for testing/debugging)",
    )
    args = parser.parse_args()

    corpus_dir = args.corpus_dir
    gemini_model = args.gemini_model
    max_regen = args.max_regen_attempts
    conditions = args.conditions
    output_dir = args.output_dir or os.path.join(corpus_dir, "results_temporal")
    os.makedirs(output_dir, exist_ok=True)

    # ── Check API key (needed for regeneration) ──
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
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
        print("  Needed for regeneration of failed answers.")
        return

    print("=" * 60)
    print("TEMPORAL-AWARE VERIFICATION STAGE EVALUATION")
    print(f"  NLI model: {NLI_MODEL_NAME}")
    print(f"  Gemini model (regen): {gemini_model}")
    print(f"  Max regen attempts: {max_regen}")
    print(f"  Conditions: {conditions}")
    print(f"  Corpus dir: {corpus_dir}")
    print(f"  Output dir: {output_dir}")
    if args.max_queries:
        print(f"  Max queries per condition: {args.max_queries}")
    print("=" * 60)

    # ��─ Load temporal generation results ──
    gen_path = os.path.join(output_dir, "generation_temporal_results_detailed.jsonl")
    if not os.path.exists(gen_path):
        print(f"\n  ERROR: {gen_path} not found.")
        print("  Run generation_temporal_eval.py first.")
        return

    gen_results = load_jsonl(gen_path)
    print(f"\n  Loaded {len(gen_results)} generation results")

    # ── Load generation payloads for context text ──
    payloads_path = os.path.join(output_dir, "generation_payloads.jsonl")
    if not os.path.exists(payloads_path):
        print(f"\n  ERROR: {payloads_path} not found.")
        print("  Run context_assembly_eval.py first.")
        return

    payloads = load_jsonl(payloads_path)
    payload_lookup = {(p["query_id"], p["condition"]): p for p in payloads}
    print(f"  Loaded {len(payloads)} generation payloads (for context)")

    # ── Initialize NLI verifier ──
    verifier = NLIVerifier(NLI_MODEL_NAME)

    # ── Check for existing progress (resume support) ──
    results_path = os.path.join(output_dir, "verification_temporal_results_detailed.jsonl")
    existing = {}
    if os.path.exists(results_path):
        existing_data = load_jsonl(results_path)
        existing = {(r["query_id"], r["condition"]): r for r in existing_data}
        print(f"  Found {len(existing)} already verified (resuming)")

    # ── Run verification ──
    all_results = list(existing.values())
    total_regen_cost = sum(r.get("regen_cost_usd", 0) for r in all_results)

    for cond in conditions:
        cond_gen = [g for g in gen_results if g["condition"] == cond]
        if args.max_queries:
            cond_gen = cond_gen[: args.max_queries]

        to_process = [
            g for g in cond_gen
            if (g["query_id"], g["condition"]) not in existing
        ]

        print(f"\n{'─' * 60}")
        print(f"  CONDITION: {cond}")
        print(f"  Total: {len(cond_gen)}, remaining: {len(to_process)}")
        print(f"{'─' * 60}")

        fail_count = 0
        regen_count = 0

        for i, gen in enumerate(to_process):
            query_id = gen["query_id"]
            condition = gen["condition"]
            generated_answer = gen["generated_answer"]
            question = gen["question"]
            generation_cost = gen["cost_usd"]

            # Get context from payloads
            payload = payload_lookup.get((query_id, condition))
            if not payload:
                continue
            context = payload["context"]

            # ── NLI verification ──
            t0 = time.perf_counter()
            entailed, verdict, sentence_results = verifier.verify(
                context, generated_answer
            )
            t1 = time.perf_counter()
            verification_latency_ms = (t1 - t0) * 1000

            entailment_failure = 0 if entailed else 1
            if not entailed:
                fail_count += 1

            # ── Regeneration if needed ──
            regen_attempts = 0
            regen_cost = 0.0
            regen_latency = 0.0
            regen_input_tokens = 0
            regen_output_tokens = 0
            final_answer = generated_answer
            final_verdict = verdict
            entailed_after_regen = 1 if entailed else 0

            if not entailed and max_regen > 0:
                current_answer = generated_answer
                for attempt in range(max_regen):
                    regen_attempts += 1
                    regen_count += 1

                    prompt = REGEN_PROMPT.format(
                        context=context,
                        question=question,
                        previous_answer=current_answer,
                    )

                    try:
                        regen_result = call_gemini(prompt, api_key, gemini_model)
                    except Exception as e:
                        print(f"    [{i+1}] {query_id} regen ERROR: {e}")
                        if "429" in str(e) or "Resource" in str(e):
                            print("    Rate limited, waiting 30s...")
                            time.sleep(30)
                            try:
                                regen_result = call_gemini(prompt, api_key, gemini_model)
                            except Exception as e2:
                                print(f"    Retry failed: {e2}")
                                break
                        else:
                            break

                    r_cost = estimate_cost(
                        regen_result["input_tokens"],
                        regen_result["output_tokens"],
                        gemini_model,
                    )
                    regen_cost += r_cost
                    regen_latency += regen_result["latency_ms"]
                    regen_input_tokens += regen_result["input_tokens"]
                    regen_output_tokens += regen_result["output_tokens"]
                    total_regen_cost += r_cost

                    current_answer = regen_result["generated_answer"]

                    # Re-verify
                    entailed_now, verdict_now, _ = verifier.verify(
                        context, current_answer
                    )
                    final_answer = current_answer
                    final_verdict = verdict_now

                    if entailed_now:
                        entailed_after_regen = 1
                        break

                    time.sleep(0.5)

            total_cost = generation_cost + regen_cost

            result = {
                "query_id": query_id,
                "condition": condition,
                "domain": gen["domain"],
                "time_sensitive": gen["time_sensitive"],
                "question": question,
                "ground_truth": gen["ground_truth"],
                "original_answer": generated_answer,
                "final_answer": final_answer,
                "initial_verdict": verdict,
                "final_verdict": final_verdict,
                "entailment_failure": entailment_failure,
                "regeneration_triggered": 1 if regen_attempts > 0 else 0,
                "regen_attempts": regen_attempts,
                "entailed_after_regen": entailed_after_regen,
                "generation_cost_usd": generation_cost,
                "regen_cost_usd": regen_cost,
                "total_cost_usd": total_cost,
                "regen_input_tokens": regen_input_tokens,
                "regen_output_tokens": regen_output_tokens,
                "verification_latency_ms": verification_latency_ms,
                "regen_latency_ms": regen_latency,
                "final_response_length_words": len(final_answer.split()),
                # Carry forward upstream metadata
                "stale_token_ratio": gen.get("stale_token_ratio", 0),
                "contradiction_density": gen.get("contradiction_density", 0),
                "redundancy_score": gen.get("redundancy_score", 0),
                "answer_bearing_in_context": gen.get("answer_bearing_in_context", False),
            }
            all_results.append(result)
            existing[(query_id, condition)] = result

            if (i + 1) % 10 == 0 or (i + 1) == len(to_process):
                print(
                    f"    [{i+1}/{len(to_process)}] "
                    f"fails={fail_count} regens={regen_count} "
                    f"regen_cost=${total_regen_cost:.4f}"
                )

            # Incremental save every 20 queries
            if (i + 1) % 20 == 0:
                save_jsonl(all_results, results_path)

            time.sleep(0.3)

        # Save after each condition
        save_jsonl(all_results, results_path)

        # Quick summary
        cond_results = [r for r in all_results if r["condition"] == cond]
        agg = _agg(cond_results)
        n = agg.get("n", 0)
        print(f"\n  Quick summary for {cond} ({n} queries):")
        print(f"    Entailment failure rate:  {agg.get('entailment_failure_mean', 0):.4f}")
        print(f"    Regeneration trigger rate:{agg.get('regeneration_triggered_mean', 0):.4f}")
        print(f"    Entailed after regen:     {agg.get('entailed_after_regen_mean', 0):.4f}")
        print(f"    Total cost/answer (mean): ${agg.get('total_cost_usd_mean', 0):.6f}")
        print(f"    Regen cost (total):       ${agg.get('regen_cost_usd_sum', 0):.4f}")

    # ── Print full summary ──
    evaluated = [c for c in conditions if any(r["condition"] == c for r in all_results)]
    print_summary_table(all_results, evaluated)

    # ── Save detailed results (final) ──
    save_jsonl(all_results, results_path)
    print(f"\n  Detailed results saved to {results_path}")

    # ── Save aggregated results ──
    agg_results = {
        "overall": {},
        "by_time_sensitivity": {},
        "by_domain": {},
    }

    for cond in evaluated:
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

    def convert_numpy(obj):
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        return obj

    agg_results = convert_numpy(agg_results)

    agg_path = os.path.join(output_dir, "verification_temporal_results_aggregated.json")
    with open(agg_path, "w") as f:
        json.dump(agg_results, f, indent=2)
    print(f"  Aggregated results saved to {agg_path}")

    # ── Grand totals ──
    grand_gen = sum(data["overall"][c].get("generation_cost_usd_sum", 0) for c in evaluated for data in [agg_results])
    grand_regen = sum(data["overall"][c].get("regen_cost_usd_sum", 0) for c in evaluated for data in [agg_results])
    grand_total = sum(data["overall"][c].get("total_cost_usd_sum", 0) for c in evaluated for data in [agg_results])
    print(f"\n  GRAND TOTALS:")
    print(f"    Generation cost: ${grand_gen:.4f}")
    print(f"    Regen cost:      ${grand_regen:.4f}")
    print(f"    Total cost:      ${grand_total:.4f}")

    print(f"\n{'=' * 60}")
    print("TEMPORAL-AWARE VERIFICATION EVALUATION COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
