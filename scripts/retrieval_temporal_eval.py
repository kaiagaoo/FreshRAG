"""
Temporal-Aware Retrieval Stage Evaluation Pipeline
====================================================
Variant of retrieval_eval.py that augments embeddings with a temporal signal.

Each document embedding is concatenated with a temporal freshness dimension:
  - Fresh documents (is_fresh=True):  +alpha
  - Stale documents (is_fresh=False): -alpha

Query embeddings are concatenated with +alpha (preference for fresh content).

The weight `alpha` controls the strength of the temporal signal relative to
semantic similarity. With normalized embeddings in dim D, the temporal
dimension shifts cosine similarity by approximately ±alpha²/(D+1).

Results are saved to a separate folder (default: freshrag_experiment/results_temporal/).

Usage (run from repo root):
  python scripts/retrieval_temporal_eval.py --corpus_dir ./freshrag_experiment --k 5
  python scripts/retrieval_temporal_eval.py --corpus_dir ./freshrag_experiment --k 5 --alpha 0.3
  python scripts/retrieval_temporal_eval.py --corpus_dir ./freshrag_experiment --k 5 --conditions fresh stale_30
"""

import json
import os
import time
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
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


def get_domain(doc_id):
    parts = doc_id.split("_")
    for i, part in enumerate(parts):
        if part.isdigit():
            return "_".join(parts[:i])
    return parts[0]


# ─────────────────────────────────────────────
# TEMPORAL EMBEDDING + INDEX
# ─────────────────────────────────────────────
class TemporalRetrievalIndex:
    """FAISS-based retrieval index with temporal-augmented embeddings."""

    def __init__(self, model_name=EMBEDDING_MODEL_NAME, alpha=DEFAULT_ALPHA):
        from sentence_transformers import SentenceTransformer
        import faiss

        print(f"  Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.aug_dim = self.dim + 1  # +1 for temporal signal
        self.alpha = alpha
        self.faiss = faiss
        self.index = None
        self.doc_ids = []
        self.doc_metadata = {}

    def build_index(self, corpus):
        """Embed all documents, append temporal signal, and build FAISS index."""
        print(f"  Embedding {len(corpus)} documents...", end=" ")

        texts = [doc["text"] for doc in corpus]
        freshness = [doc.get("is_fresh", True) for doc in corpus]
        self.doc_ids = [doc["doc_id"] for doc in corpus]
        self.doc_metadata = {doc["doc_id"]: doc for doc in corpus}

        start = time.time()
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            batch_size=64,
            normalize_embeddings=True,
        )
        embed_time = time.time() - start
        print(f"done ({embed_time:.1f}s)")

        # Append temporal signal: +alpha for fresh, -alpha for stale
        temporal_signal = np.array(
            [self.alpha if f else -self.alpha for f in freshness],
            dtype=np.float32,
        ).reshape(-1, 1)

        augmented = np.hstack([embeddings.astype(np.float32), temporal_signal])

        # Re-normalize augmented vectors for cosine similarity via inner product
        norms = np.linalg.norm(augmented, axis=1, keepdims=True)
        augmented = augmented / norms

        # Build FAISS index (inner product = cosine sim for normalized vectors)
        self.index = self.faiss.IndexFlatIP(self.aug_dim)
        self.index.add(augmented)
        print(f"  FAISS index built: {self.index.ntotal} vectors, dim={self.aug_dim} (base {self.dim} + 1 temporal)")

        return embed_time

    def search(self, query_text, k=DEFAULT_K):
        """Search the index for top-k documents. Query gets +alpha temporal signal (prefer fresh)."""
        start = time.time()

        query_embedding = self.model.encode(
            [query_text],
            normalize_embeddings=True,
        ).astype(np.float32)

        # Append +alpha to query (express preference for fresh content)
        temporal_signal = np.array([[self.alpha]], dtype=np.float32)
        augmented_query = np.hstack([query_embedding, temporal_signal])

        # Re-normalize
        norm = np.linalg.norm(augmented_query, axis=1, keepdims=True)
        augmented_query = augmented_query / norm

        scores, indices = self.index.search(augmented_query, k)
        latency_ms = (time.time() - start) * 1000

        result_doc_ids = [self.doc_ids[idx] for idx in indices[0]]
        result_scores = scores[0].tolist()

        return result_doc_ids, result_scores, latency_ms


# ─────────────────────────────────────────────
# METRICS COMPUTATION
# ─────────────────────────────────────────────
def compute_retrieval_metrics(
    retrieved_doc_ids,
    retrieved_scores,
    latency_ms,
    answer_bearing_doc_ids,
    doc_metadata,
    k,
):
    """
    Compute all retrieval metrics for a single query.

    Returns dict with:
    - precision_at_k: fraction of retrieved docs that are answer-bearing
    - recall_at_k: fraction of answer-bearing docs that were retrieved
    - stale_intrusion_rate: fraction of retrieved docs that are stale (is_fresh=False)
    - stale_in_top_k: count of stale docs in top-k
    - fresh_answer_bearing_retrieved: count of fresh answer-bearing docs retrieved
    - stale_answer_bearing_retrieved: count of stale answer-bearing docs retrieved
    - latency_ms: retrieval latency
    - avg_score: average similarity score of retrieved docs
    - top_score: highest similarity score
    """
    ab_set = set(answer_bearing_doc_ids)
    retrieved_set = set(retrieved_doc_ids)

    # Precision@k
    relevant_retrieved = retrieved_set & ab_set
    precision_at_k = len(relevant_retrieved) / k if k > 0 else 0

    # Recall@k
    recall_at_k = (
        len(relevant_retrieved) / len(ab_set) if len(ab_set) > 0 else 0
    )

    # Stale intrusion rate
    stale_count = 0
    stale_ab_count = 0
    fresh_ab_count = 0
    for doc_id in retrieved_doc_ids:
        meta = doc_metadata.get(doc_id, {})
        is_fresh = meta.get("is_fresh", True)
        is_ab = doc_id in ab_set

        if not is_fresh:
            stale_count += 1
            if is_ab:
                stale_ab_count += 1
        elif is_ab:
            fresh_ab_count += 1

    stale_intrusion_rate = stale_count / k if k > 0 else 0

    # Retrieval scores
    avg_score = np.mean(retrieved_scores) if retrieved_scores else 0
    top_score = max(retrieved_scores) if retrieved_scores else 0

    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "stale_intrusion_rate": stale_intrusion_rate,
        "stale_in_top_k": stale_count,
        "fresh_answer_bearing_retrieved": fresh_ab_count,
        "stale_answer_bearing_retrieved": stale_ab_count,
        "latency_ms": latency_ms,
        "avg_similarity_score": avg_score,
        "top_similarity_score": top_score,
    }


# ─────────────────────────────────────────────
# MAIN EVALUATION LOOP
# ─────────────────────────────────────────────
def evaluate_condition(index, queries, doc_metadata, k, condition_name):
    """Run all queries against one index and collect metrics."""
    results = []

    for i, query in enumerate(queries):
        query_id = query["query_id"]
        question = query["question"]
        ab_doc_ids = query["answer_bearing_doc_ids"]
        domain = query["domain"]
        time_sensitive = query["time_sensitive"]

        # Retrieve
        retrieved_ids, retrieved_scores, latency_ms = index.search(question, k=k)

        # Compute metrics
        metrics = compute_retrieval_metrics(
            retrieved_doc_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            latency_ms=latency_ms,
            answer_bearing_doc_ids=ab_doc_ids,
            doc_metadata=doc_metadata,
            k=k,
        )

        # Combine into result record
        result = {
            "condition": condition_name,
            "query_id": query_id,
            "domain": domain,
            "time_sensitive": time_sensitive,
            "k": k,
            "retrieved_doc_ids": retrieved_ids,
            "answer_bearing_doc_ids": ab_doc_ids,
            **metrics,
        }
        results.append(result)

        if (i + 1) % 50 == 0:
            print(f"    Processed {i+1}/{len(queries)} queries")

    return results


def aggregate_metrics(results, group_by=None):
    """
    Aggregate metrics across results.
    If group_by is specified, returns per-group aggregations.
    """
    if group_by:
        groups = defaultdict(list)
        for r in results:
            key = r[group_by]
            groups[key].append(r)
        return {key: _agg(group) for key, group in groups.items()}
    else:
        return _agg(results)


def _agg(results):
    """Compute mean and std for all numeric metrics."""
    if not results:
        return {}

    metric_keys = [
        "precision_at_k",
        "recall_at_k",
        "stale_intrusion_rate",
        "stale_in_top_k",
        "fresh_answer_bearing_retrieved",
        "stale_answer_bearing_retrieved",
        "latency_ms",
        "avg_similarity_score",
        "top_similarity_score",
    ]

    agg = {"n": len(results)}
    for key in metric_keys:
        values = [r[key] for r in results if key in r]
        if values:
            agg[f"{key}_mean"] = np.mean(values)
            agg[f"{key}_std"] = np.std(values)
    return agg


def print_summary_table(all_results, conditions, k, alpha):
    """Print a formatted summary table across conditions."""
    print("\n" + "=" * 90)
    print(f"TEMPORAL-AWARE RETRIEVAL EVALUATION SUMMARY (k={k}, alpha={alpha})")
    print("=" * 90)

    # ── Overall comparison ──
    header = f"{'Metric':<30}"
    for cond in conditions:
        header += f"  {cond:>12}"
    print(f"\n{header}")
    print("-" * (30 + 14 * len(conditions)))

    metrics_to_show = [
        ("Precision@k", "precision_at_k_mean"),
        ("Recall@k", "recall_at_k_mean"),
        ("Stale intrusion rate", "stale_intrusion_rate_mean"),
        ("Stale docs in top-k", "stale_in_top_k_mean"),
        ("Latency (ms)", "latency_ms_mean"),
        ("Avg similarity score", "avg_similarity_score_mean"),
    ]

    cond_aggs = {}
    for cond in conditions:
        cond_results = [r for r in all_results if r["condition"] == cond]
        cond_aggs[cond] = _agg(cond_results)

    for label, key in metrics_to_show:
        row = f"{label:<30}"
        for cond in conditions:
            val = cond_aggs[cond].get(key, 0)
            if "latency" in key:
                row += f"  {val:>12.2f}"
            else:
                row += f"  {val:>12.4f}"
        print(row)

    # ── By time sensitivity ──
    print(f"\n{'─' * 90}")
    print("BREAKDOWN: Time-sensitive vs. Time-insensitive queries")
    print(f"{'─' * 90}")

    for ts_label, ts_val in [("Time-sensitive", True), ("Time-insensitive", False)]:
        print(f"\n  {ts_label}:")
        header = f"  {'Metric':<28}"
        for cond in conditions:
            header += f"  {cond:>12}"
        print(header)
        print("  " + "-" * (28 + 14 * len(conditions)))

        for label, key in metrics_to_show:
            row = f"  {label:<28}"
            for cond in conditions:
                cond_ts_results = [
                    r for r in all_results
                    if r["condition"] == cond and r["time_sensitive"] == ts_val
                ]
                agg = _agg(cond_ts_results)
                val = agg.get(key, 0)
                if "latency" in key:
                    row += f"  {val:>12.2f}"
                else:
                    row += f"  {val:>12.4f}"
            print(row)

    # ── By domain ──
    print(f"\n{'─' * 90}")
    print("BREAKDOWN: By domain")
    print(f"{'─' * 90}")

    domains = sorted(set(r["domain"] for r in all_results))
    for domain in domains:
        print(f"\n  {domain}:")
        header = f"  {'Metric':<28}"
        for cond in conditions:
            header += f"  {cond:>12}"
        print(header)
        print("  " + "-" * (28 + 14 * len(conditions)))

        for label, key in [
            ("Precision@k", "precision_at_k_mean"),
            ("Stale intrusion rate", "stale_intrusion_rate_mean"),
            ("Recall@k", "recall_at_k_mean"),
        ]:
            row = f"  {label:<28}"
            for cond in conditions:
                cond_domain_results = [
                    r for r in all_results
                    if r["condition"] == cond and r["domain"] == domain
                ]
                agg = _agg(cond_domain_results)
                val = agg.get(key, 0)
                row += f"  {val:>12.4f}"
            print(row)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Temporal-aware retrieval stage evaluation")
    parser.add_argument(
        "--corpus_dir",
        type=str,
        required=True,
        help="Directory containing corpus_*.jsonl and queries_clean.jsonl",
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K, help="Top-k for retrieval (default: 5)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Temporal signal weight (default: 0.2). Higher values penalize stale docs more.",
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
    print(f"TEMPORAL-AWARE RETRIEVAL STAGE EVALUATION")
    print(f"  Model: {EMBEDDING_MODEL_NAME}")
    print(f"  k: {k}")
    print(f"  Alpha (temporal weight): {alpha}")
    print(f"  Conditions: {conditions}")
    print(f"  Corpus dir: {corpus_dir}")
    print(f"  Output dir: {output_dir}")
    print("=" * 60)

    # ── Load queries ──
    queries = load_jsonl(os.path.join(corpus_dir, "queries_clean.jsonl"))
    print(f"\n  Loaded {len(queries)} queries")

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

        # Load corpus for this condition
        corpus = load_jsonl(corpus_path)
        print(f"  Loaded {len(corpus)} docs from {corpus_path}")

        # Count stale docs
        stale_count = sum(1 for d in corpus if not d["is_fresh"])
        print(f"  Stale docs in corpus: {stale_count}")

        # Build temporal-augmented index
        index = TemporalRetrievalIndex(EMBEDDING_MODEL_NAME, alpha=alpha)
        index_time = index.build_index(corpus)
        print(f"  Index build time: {index_time:.1f}s")

        # Run evaluation
        doc_metadata = {d["doc_id"]: d for d in corpus}
        results = evaluate_condition(index, queries, doc_metadata, k, cond)
        all_results.extend(results)

        # Quick summary for this condition
        agg = _agg(results)
        print(f"\n  Quick summary for {cond}:")
        print(f"    Precision@{k}: {agg['precision_at_k_mean']:.4f} ± {agg['precision_at_k_std']:.4f}")
        print(f"    Recall@{k}:    {agg['recall_at_k_mean']:.4f} ± {agg['recall_at_k_std']:.4f}")
        print(f"    Stale intrusion: {agg['stale_intrusion_rate_mean']:.4f}")
        print(f"    Avg latency:  {agg['latency_ms_mean']:.2f}ms")

        # Free memory
        del index, corpus
        import gc
        gc.collect()

    # ── Print full summary ──
    print_summary_table(all_results, conditions, k, alpha)

    # ── Save detailed results ──
    save_jsonl(all_results, os.path.join(output_dir, "retrieval_temporal_results_detailed.jsonl"))
    print(f"\n  Detailed results saved to {output_dir}/retrieval_temporal_results_detailed.jsonl")

    # ── Save aggregated results ──
    agg_results = {
        "config": {
            "alpha": alpha,
            "k": k,
            "model": EMBEDDING_MODEL_NAME,
        },
        "overall": {},
        "by_time_sensitivity": {},
        "by_domain": {},
        "by_condition_x_domain": {},
    }

    for cond in conditions:
        cond_results = [r for r in all_results if r["condition"] == cond]
        agg_results["overall"][cond] = _agg(cond_results)

        # By time sensitivity
        for ts_val, ts_label in [(True, "time_sensitive"), (False, "time_insensitive")]:
            ts_results = [r for r in cond_results if r["time_sensitive"] == ts_val]
            key = f"{cond}__{ts_label}"
            agg_results["by_time_sensitivity"][key] = _agg(ts_results)

        # By domain
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

    with open(os.path.join(output_dir, "retrieval_temporal_results_aggregated.json"), "w") as f:
        json.dump(agg_results, f, indent=2)
    print(f"  Aggregated results saved to {output_dir}/retrieval_temporal_results_aggregated.json")

    print(f"\n{'=' * 60}")
    print("TEMPORAL-AWARE RETRIEVAL EVALUATION COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
