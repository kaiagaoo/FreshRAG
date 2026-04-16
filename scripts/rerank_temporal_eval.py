"""
Temporal-Aware Reranking Stage Evaluation Pipeline
====================================================
Temporal variant of rerank_eval.py.

Takes temporal retrieval results and reranks using a cross-encoder model,
then applies a freshness bonus/penalty to cross-encoder scores before
re-sorting. Fresh documents receive +alpha, stale documents receive -alpha.

Model: cross-encoder/ms-marco-MiniLM-L6-v2

Usage (run from repo root):
  python scripts/rerank_temporal_eval.py --corpus_dir ./freshrag_experiment --k 5
  python scripts/rerank_temporal_eval.py --corpus_dir ./freshrag_experiment --k 5 --alpha 0.3
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
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_K = 5
DEFAULT_ALPHA = 0.2
CONDITIONS = ["fresh", "stale_10", "stale_30", "stale_50"]


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


# ─────────────────────────────────────────────
# RERANKER
# ─────────────────────────────────────────────
class CrossEncoderReranker:
    """Cross-encoder reranker with temporal freshness bonus."""

    def __init__(self, model_name=RERANKER_MODEL_NAME, alpha=DEFAULT_ALPHA):
        from sentence_transformers import CrossEncoder

        print(f"  Loading cross-encoder: {model_name}")
        print(f"  Temporal alpha: {alpha}")
        self.model = CrossEncoder(model_name)
        self.alpha = alpha

    def rerank(self, query, doc_texts, doc_ids, doc_metadata=None):
        """
        Score all (query, doc) pairs, apply temporal bonus, and return reranked order.

        Fresh docs receive +alpha bonus, stale docs receive -alpha penalty
        on their cross-encoder scores before re-sorting.

        Returns:
            reranked_doc_ids: doc IDs sorted by adjusted score (desc)
            reranked_scores: corresponding adjusted scores
            inference_time_ms: time for cross-encoder inference
        """
        pairs = [[query, text] for text in doc_texts]

        start = time.time()
        scores = self.model.predict(pairs)
        inference_time_ms = (time.time() - start) * 1000

        # Apply temporal bonus/penalty
        adjusted_scores = []
        for doc_id, score in zip(doc_ids, scores.tolist()):
            if doc_metadata:
                is_fresh = doc_metadata.get(doc_id, {}).get("is_fresh", True)
                bonus = self.alpha if is_fresh else -self.alpha
            else:
                bonus = 0.0
            adjusted_scores.append(score + bonus)

        # Sort by adjusted score descending
        scored = list(zip(doc_ids, adjusted_scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked_ids = [s[0] for s in scored]
        reranked_scores = [s[1] for s in scored]

        return reranked_ids, reranked_scores, inference_time_ms


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_rerank_metrics(
    original_doc_ids,
    reranked_doc_ids,
    reranked_scores,
    inference_time_ms,
    answer_bearing_doc_ids,
    doc_metadata,
    k,
):
    """
    Compute reranking-specific metrics for a single query.

    Returns dict with:
    - mean_rank_shift: average position change of answer-bearing docs
      (negative = promoted upward, positive = demoted)
    - inference_time_ms: cross-encoder scoring time
    - semantic_ambiguity_score: std dev of reranker scores across top-k
    - relevant_promoted: count of answer-bearing docs that moved up
    - relevant_demoted: count of answer-bearing docs that moved down
    - stale_promoted: count of stale docs that moved up after reranking
    - reranked_precision_at_k: precision after reranking
    - reranked_stale_intrusion_rate: stale intrusion after reranking
    """
    ab_set = set(answer_bearing_doc_ids)

    # Build position maps (0-indexed)
    original_rank = {doc_id: i for i, doc_id in enumerate(original_doc_ids)}
    reranked_rank = {doc_id: i for i, doc_id in enumerate(reranked_doc_ids)}

    # ── Mean rank shift of relevant documents ──
    rank_shifts = []
    promoted = 0
    demoted = 0
    for doc_id in ab_set:
        if doc_id in original_rank and doc_id in reranked_rank:
            shift = reranked_rank[doc_id] - original_rank[doc_id]
            rank_shifts.append(shift)
            if shift < 0:
                promoted += 1
            elif shift > 0:
                demoted += 1

    mean_rank_shift = np.mean(rank_shifts) if rank_shifts else 0.0

    # ── Semantic ambiguity score ──
    # Low std = reranker can't differentiate docs = high ambiguity
    semantic_ambiguity = np.std(reranked_scores) if len(reranked_scores) > 1 else 0.0

    # ── Stale doc promotion ──
    stale_promoted = 0
    for doc_id in reranked_doc_ids:
        meta = doc_metadata.get(doc_id, {})
        is_fresh = meta.get("is_fresh", True)
        if not is_fresh and doc_id in original_rank and doc_id in reranked_rank:
            if reranked_rank[doc_id] < original_rank[doc_id]:
                stale_promoted += 1

    # ── Reranked precision@k ──
    reranked_top_k = set(reranked_doc_ids[:k])
    reranked_relevant = reranked_top_k & ab_set
    reranked_precision = len(reranked_relevant) / k if k > 0 else 0

    # ── Reranked stale intrusion ──
    stale_in_reranked = 0
    for doc_id in reranked_doc_ids[:k]:
        meta = doc_metadata.get(doc_id, {})
        if not meta.get("is_fresh", True):
            stale_in_reranked += 1
    reranked_stale_intrusion = stale_in_reranked / k if k > 0 else 0

    return {
        "mean_rank_shift": float(mean_rank_shift),
        "inference_time_ms": inference_time_ms,
        "semantic_ambiguity_score": float(semantic_ambiguity),
        "relevant_promoted": promoted,
        "relevant_demoted": demoted,
        "stale_promoted": stale_promoted,
        "reranked_precision_at_k": reranked_precision,
        "reranked_stale_intrusion_rate": reranked_stale_intrusion,
    }


# ─────────────────────────────────────────────
# EVALUATION LOOP
# ─────────────────────────────────────────────
def evaluate_condition(reranker, retrieval_results, doc_metadata, k, condition_name):
    """Rerank all queries for one condition and collect metrics."""
    results = []

    cond_results = [r for r in retrieval_results if r["condition"] == condition_name]
    print(f"    Reranking {len(cond_results)} queries...")

    for i, ret in enumerate(cond_results):
        query_id = ret["query_id"]
        domain = ret["domain"]
        time_sensitive = ret["time_sensitive"]
        original_doc_ids = ret["retrieved_doc_ids"]
        ab_doc_ids = ret["answer_bearing_doc_ids"]

        # Get doc texts for reranking
        doc_texts = []
        valid_doc_ids = []
        for doc_id in original_doc_ids:
            meta = doc_metadata.get(doc_id)
            if meta:
                doc_texts.append(meta["text"])
                valid_doc_ids.append(doc_id)

        if not doc_texts:
            continue

        # Look up question from queries (stored in retrieval results via query_id)
        # We need the question text — get it from the queries file
        question = ret.get("question", "")

        # Rerank with temporal bonus (measure total latency including doc lookup overhead)
        total_start = time.time()
        reranked_ids, reranked_scores, inference_ms = reranker.rerank(
            question, doc_texts, valid_doc_ids, doc_metadata=doc_metadata
        )
        total_rerank_ms = (time.time() - total_start) * 1000

        # Compute metrics
        metrics = compute_rerank_metrics(
            original_doc_ids=valid_doc_ids,
            reranked_doc_ids=reranked_ids,
            reranked_scores=reranked_scores,
            inference_time_ms=inference_ms,
            answer_bearing_doc_ids=ab_doc_ids,
            doc_metadata=doc_metadata,
            k=k,
        )

        result = {
            "condition": condition_name,
            "query_id": query_id,
            "domain": domain,
            "time_sensitive": time_sensitive,
            "k": k,
            "original_doc_ids": valid_doc_ids,
            "reranked_doc_ids": reranked_ids,
            "reranked_scores": reranked_scores,
            "answer_bearing_doc_ids": ab_doc_ids,
            "original_precision_at_k": ret.get("precision_at_k", 0),
            "original_stale_intrusion_rate": ret.get("stale_intrusion_rate", 0),
            "total_rerank_latency_ms": total_rerank_ms,
            "retrieval_latency_ms": ret.get("latency_ms", 0),
            **metrics,
        }
        results.append(result)

        if (i + 1) % 50 == 0:
            print(f"      Processed {i+1}/{len(cond_results)} queries")

    return results


# ─────────────────────────────────────────────
# AGGREGATION
# ─────────────────────────────────────────────
def _agg(results):
    """Compute mean and std for all numeric metrics."""
    if not results:
        return {}

    metric_keys = [
        "mean_rank_shift",
        "inference_time_ms",
        "total_rerank_latency_ms",
        "retrieval_latency_ms",
        "semantic_ambiguity_score",
        "relevant_promoted",
        "relevant_demoted",
        "stale_promoted",
        "reranked_precision_at_k",
        "reranked_stale_intrusion_rate",
        "original_precision_at_k",
        "original_stale_intrusion_rate",
    ]

    agg = {"n": len(results)}
    for key in metric_keys:
        values = [r[key] for r in results if key in r]
        if values:
            agg[f"{key}_mean"] = np.mean(values)
            agg[f"{key}_std"] = np.std(values)
    return agg


def print_summary_table(all_results, conditions, k):
    """Print a formatted summary table across conditions."""
    print("\n" + "=" * 100)
    print(f"TEMPORAL-AWARE RERANKING EVALUATION SUMMARY (k={k})")
    print(f"Model: {RERANKER_MODEL_NAME}")
    print("=" * 100)

    header = f"{'Metric':<35}"
    for cond in conditions:
        header += f"  {cond:>12}"
    print(f"\n{header}")
    print("-" * (35 + 14 * len(conditions)))

    metrics_to_show = [
        ("Mean rank shift (relevant)", "mean_rank_shift_mean"),
        ("Semantic ambiguity score", "semantic_ambiguity_score_mean"),
        ("Reranker inference time (ms)", "inference_time_ms_mean"),
        ("Total rerank latency (ms)", "total_rerank_latency_ms_mean"),
        ("Retrieval latency (ms)", "retrieval_latency_ms_mean"),
        ("Precision@k (before)", "original_precision_at_k_mean"),
        ("Precision@k (after)", "reranked_precision_at_k_mean"),
        ("Stale intrusion (before)", "original_stale_intrusion_rate_mean"),
        ("Stale intrusion (after)", "reranked_stale_intrusion_rate_mean"),
        ("Relevant docs promoted", "relevant_promoted_mean"),
        ("Relevant docs demoted", "relevant_demoted_mean"),
        ("Stale docs promoted", "stale_promoted_mean"),
    ]

    cond_aggs = {}
    for cond in conditions:
        cond_results = [r for r in all_results if r["condition"] == cond]
        cond_aggs[cond] = _agg(cond_results)

    for label, key in metrics_to_show:
        row = f"{label:<35}"
        for cond in conditions:
            val = cond_aggs[cond].get(key, 0)
            if "time" in key:
                row += f"  {val:>12.2f}"
            else:
                row += f"  {val:>12.4f}"
        print(row)

    # ── By time sensitivity ──
    print(f"\n{'─' * 100}")
    print("BREAKDOWN: Time-sensitive vs. Time-insensitive queries")
    print(f"{'─' * 100}")

    for ts_label, ts_val in [("Time-sensitive", True), ("Time-insensitive", False)]:
        print(f"\n  {ts_label}:")
        header = f"  {'Metric':<33}"
        for cond in conditions:
            header += f"  {cond:>12}"
        print(header)
        print("  " + "-" * (33 + 14 * len(conditions)))

        for label, key in [
            ("Mean rank shift", "mean_rank_shift_mean"),
            ("Semantic ambiguity", "semantic_ambiguity_score_mean"),
            ("Precision@k (after)", "reranked_precision_at_k_mean"),
            ("Stale intrusion (after)", "reranked_stale_intrusion_rate_mean"),
        ]:
            row = f"  {label:<33}"
            for cond in conditions:
                cond_ts_results = [
                    r for r in all_results
                    if r["condition"] == cond and r["time_sensitive"] == ts_val
                ]
                agg = _agg(cond_ts_results)
                val = agg.get(key, 0)
                row += f"  {val:>12.4f}"
            print(row)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Temporal-aware reranking stage evaluation")
    parser.add_argument(
        "--corpus_dir",
        type=str,
        required=True,
        help="Directory containing corpus_*.jsonl and queries_clean.jsonl",
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K, help="Top-k for evaluation (default: 5)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Temporal signal weight (default: 0.2). Added to fresh doc scores, subtracted from stale.",
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
    args = parser.parse_args()

    corpus_dir = args.corpus_dir
    k = args.k
    alpha = args.alpha
    conditions = args.conditions
    output_dir = args.output_dir or os.path.join(corpus_dir, "results_temporal")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("TEMPORAL-AWARE RERANKING STAGE EVALUATION")
    print(f"  Reranker: {RERANKER_MODEL_NAME}")
    print(f"  k: {k}")
    print(f"  Alpha (temporal weight): {alpha}")
    print(f"  Conditions: {conditions}")
    print(f"  Corpus dir: {corpus_dir}")
    print(f"  Output dir: {output_dir}")
    print("=" * 60)

    # ── Load temporal retrieval results ──
    retrieval_path = os.path.join(output_dir, "retrieval_temporal_results_detailed.jsonl")
    if not os.path.exists(retrieval_path):
        print(f"\n  ERROR: {retrieval_path} not found.")
        print("  Run retrieval_temporal_eval.py first to generate temporal retrieval results.")
        return

    retrieval_results = load_jsonl(retrieval_path)
    print(f"\n  Loaded {len(retrieval_results)} retrieval results")

    # ── Load queries (to get question text) ──
    queries = load_jsonl(os.path.join(corpus_dir, "queries_clean.jsonl"))
    query_lookup = {q["query_id"]: q["question"] for q in queries}

    # Attach question text to retrieval results
    for r in retrieval_results:
        r["question"] = query_lookup.get(r["query_id"], "")

    # ── Initialize reranker with temporal bonus ──
    reranker = CrossEncoderReranker(RERANKER_MODEL_NAME, alpha=alpha)

    # ── Evaluate each condition ──
    all_results = []

    for cond in conditions:
        corpus_path = os.path.join(corpus_dir, f"corpus_{cond}.jsonl")
        if not os.path.exists(corpus_path):
            print(f"\n  WARNING: {corpus_path} not found, skipping {cond}")
            continue

        print(f"\n{'─' * 60}")
        print(f"  CONDITION: {cond}")
        print(f"{'─' * 60}")

        # Load corpus for doc texts
        corpus = load_jsonl(corpus_path)
        doc_metadata = {d["doc_id"]: d for d in corpus}
        print(f"  Loaded {len(corpus)} docs")

        stale_count = sum(1 for d in corpus if not d["is_fresh"])
        print(f"  Stale docs in corpus: {stale_count}")

        # Run reranking evaluation
        results = evaluate_condition(reranker, retrieval_results, doc_metadata, k, cond)
        all_results.extend(results)

        # Quick summary
        agg = _agg(results)
        print(f"\n  Quick summary for {cond}:")
        print(f"    Mean rank shift:      {agg.get('mean_rank_shift_mean', 0):+.4f}")
        print(f"    Semantic ambiguity:   {agg.get('semantic_ambiguity_score_mean', 0):.4f}")
        print(f"    Inference time:       {agg.get('inference_time_ms_mean', 0):.2f}ms")
        print(f"    Precision@k before:   {agg.get('original_precision_at_k_mean', 0):.4f}")
        print(f"    Precision@k after:    {agg.get('reranked_precision_at_k_mean', 0):.4f}")
        print(f"    Stale intrusion before: {agg.get('original_stale_intrusion_rate_mean', 0):.4f}")
        print(f"    Stale intrusion after:  {agg.get('reranked_stale_intrusion_rate_mean', 0):.4f}")

        del corpus, doc_metadata
        import gc
        gc.collect()

    # ── Print full summary ──
    print_summary_table(all_results, conditions, k)

    # ── Save detailed results ──
    save_jsonl(all_results, os.path.join(output_dir, "rerank_temporal_results_detailed.jsonl"))
    print(f"\n  Detailed results saved to {output_dir}/rerank_temporal_results_detailed.jsonl")

    # ── Save aggregated results ──
    agg_results = {
        "config": {
            "alpha": alpha,
            "k": k,
            "reranker_model": RERANKER_MODEL_NAME,
        },
        "overall": {},
        "by_time_sensitivity": {},
        "by_domain": {},
    }

    for cond in conditions:
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

    with open(os.path.join(output_dir, "rerank_temporal_results_aggregated.json"), "w") as f:
        json.dump(agg_results, f, indent=2)
    print(f"  Aggregated results saved to {output_dir}/rerank_temporal_results_aggregated.json")

    print(f"\n{'=' * 60}")
    print("TEMPORAL-AWARE RERANKING EVALUATION COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
