"""
Context Assembly Stage Evaluation Pipeline
============================================
Stage 4 of the FreshRAG experiment pipeline.

Takes reranked results from Stage 3b and assembles context windows using
deterministic rule-based concatenation with a fixed token budget.

Logic:
- Start with top-k reranked docs, concatenate in rank order
- If top-k falls short of the token budget, append additional docs from
  the retrieval pool (beyond top-k) until the budget is reached
- Truncate the last chunk if it would exceed the budget

Metrics tracked:
- Contradiction density (NLI): fraction of adjacent doc pairs with
  contradiction relationship (using cross-encoder NLI model)
- Redundancy score: mean pairwise cosine similarity among assembled docs
- Num chunks: number of document chunks needed to fill the token budget
- Assembly latency (ms): wall-clock time for assembly + scoring per query
- Token utilisation: fraction of budget actually filled
- Context freshness ratio: fraction of context tokens from fresh docs
- Stale token ratio: fraction of context tokens from stale docs
- Answer coverage: whether any answer-bearing doc made it into the context

This stage also prepares generation-ready payloads: each query's assembled
context, question text, and ground-truth answer are saved so the next
generation stage can load and send directly to an LLM.

Model: cross-encoder/nli-MiniLM2-L6-H768 (for NLI contradiction detection)
Embeddings: all-MiniLM-L6-v2 (for redundancy via cosine similarity)

Usage (run from repo root):
  python scripts/context_assembly_eval.py --corpus_dir ./freshrag_experiment --k 5
  python scripts/context_assembly_eval.py --corpus_dir ./freshrag_experiment --k 5 --token_budget 2000
"""

import json
import os
import time
import argparse
import re
import numpy as np
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEFAULT_K = 5
DEFAULT_TOKEN_BUDGET = 2000
CONDITIONS = ["fresh", "stale_10", "stale_30", "stale_50"]
NLI_MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# NLI label mapping for the cross-encoder model
NLI_LABELS = ["contradiction", "entailment", "neutral"]


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


def count_tokens(text):
    """
    Approximate token count using whitespace splitting.
    This is a simple heuristic; ~1.3 words per token for English text,
    but whitespace splitting is deterministic and reproducible.
    """
    return len(text.split())


# ─────────────────────────────────────────────
# CONTEXT ASSEMBLY
# ─────────────────────────────────────────────
def assemble_context(reranked_doc_ids, doc_metadata, token_budget, corpus_doc_ids=None):
    """
    Assemble a context window from reranked docs within a token budget.

    1. Concatenate top-k docs in reranked order
    2. If budget not filled, append additional corpus docs by their
       original corpus order (simulating a fallback pool)
    3. Truncate the last doc if it would exceed the budget

    Returns:
        assembled_docs: list of dicts with doc_id, text, token_count, is_fresh, is_answer_bearing
        total_tokens: total tokens in assembled context
        context_text: the full concatenated context string
    """
    assembled_docs = []
    total_tokens = 0

    # Phase 1: Add reranked docs in order
    for doc_id in reranked_doc_ids:
        if total_tokens >= token_budget:
            break

        meta = doc_metadata.get(doc_id)
        if not meta:
            continue

        text = meta["text"]
        doc_tokens = count_tokens(text)

        if total_tokens + doc_tokens > token_budget:
            # Truncate to fit budget
            remaining = token_budget - total_tokens
            words = text.split()
            text = " ".join(words[:remaining])
            doc_tokens = remaining

        assembled_docs.append({
            "doc_id": doc_id,
            "text": text,
            "token_count": doc_tokens,
            "is_fresh": meta.get("is_fresh", True),
            "is_answer_bearing": meta.get("is_answer_bearing", False),
        })
        total_tokens += doc_tokens

    # Phase 2: If budget not filled, append additional docs from corpus
    if total_tokens < token_budget and corpus_doc_ids:
        used_ids = set(d["doc_id"] for d in assembled_docs)
        for doc_id in corpus_doc_ids:
            if doc_id in used_ids:
                continue
            if total_tokens >= token_budget:
                break

            meta = doc_metadata.get(doc_id)
            if not meta:
                continue

            text = meta["text"]
            doc_tokens = count_tokens(text)

            if total_tokens + doc_tokens > token_budget:
                remaining = token_budget - total_tokens
                words = text.split()
                text = " ".join(words[:remaining])
                doc_tokens = remaining

            assembled_docs.append({
                "doc_id": doc_id,
                "text": text,
                "token_count": doc_tokens,
                "is_fresh": meta.get("is_fresh", True),
                "is_answer_bearing": meta.get("is_answer_bearing", False),
            })
            total_tokens += doc_tokens

    context_text = "\n\n".join(d["text"] for d in assembled_docs)
    return assembled_docs, total_tokens, context_text


# ─────────────────────────────────────────────
# NLI CONTRADICTION SCORER
# ─────────────────────────────────────────────
class NLIScorer:
    """Detect contradictions between adjacent document pairs using NLI."""

    def __init__(self, model_name=NLI_MODEL_NAME):
        from sentence_transformers import CrossEncoder

        print(f"  Loading NLI model: {model_name}")
        import torch
        device = "cpu" if not torch.cuda.is_available() else "cuda"
        self.model = CrossEncoder(model_name, device=device)

    def score_pairs(self, texts):
        """
        Score all adjacent pairs of texts for contradiction.

        Returns:
            contradiction_density: fraction of pairs classified as contradiction
            pair_labels: list of predicted labels for each pair
            pair_scores: list of score arrays for each pair
        """
        if len(texts) < 2:
            return 0.0, [], []

        pairs = [[texts[i], texts[i + 1]] for i in range(len(texts) - 1)]
        scores = self.model.predict(pairs)

        pair_labels = []
        pair_scores_list = []
        contradiction_count = 0

        for score_arr in scores:
            label_idx = int(np.argmax(score_arr))
            label = NLI_LABELS[label_idx]
            pair_labels.append(label)
            pair_scores_list.append(score_arr.tolist())
            if label == "contradiction":
                contradiction_count += 1

        contradiction_density = contradiction_count / len(pairs)
        return contradiction_density, pair_labels, pair_scores_list


# ─────────────────────────────────────────────
# REDUNDANCY SCORER
# ─────────────────────────────────────────────
class RedundancyScorer:
    """Measure redundancy via mean pairwise cosine similarity."""

    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        from sentence_transformers import SentenceTransformer

        print(f"  Loading embedding model for redundancy: {model_name}")
        import torch
        device = "cpu" if not torch.cuda.is_available() else "cuda"
        self.model = SentenceTransformer(model_name, device=device)

    def score(self, texts):
        """
        Compute mean pairwise cosine similarity among texts.

        Returns:
            redundancy_score: mean of all pairwise cosine similarities (0 to 1)
            pairwise_sims: list of pairwise similarity values
        """
        if len(texts) < 2:
            return 0.0, []

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Compute pairwise cosine similarities
        sim_matrix = np.dot(embeddings, embeddings.T)
        n = len(texts)
        pairwise_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                pairwise_sims.append(float(sim_matrix[i, j]))

        redundancy_score = np.mean(pairwise_sims) if pairwise_sims else 0.0
        return float(redundancy_score), pairwise_sims


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_context_metrics(
    assembled_docs,
    total_tokens,
    token_budget,
    contradiction_density,
    redundancy_score,
    assembly_latency_ms,
    answer_bearing_doc_ids,
):
    """
    Compute context assembly metrics for a single query.

    Returns dict with:
    - contradiction_density: fraction of adjacent pairs that contradict
    - redundancy_score: mean pairwise cosine similarity
    - num_chunks: how many doc chunks were assembled to fill budget
    - assembly_latency_ms: wall-clock time for assembly + scoring
    - token_utilisation: fraction of budget filled
    - context_freshness_ratio: fraction of context tokens from fresh docs
    - stale_token_ratio: fraction of context tokens from stale docs
    - answer_coverage: 1 if any answer-bearing doc is in context, else 0
    - answer_bearing_token_ratio: fraction of context tokens from answer-bearing docs
    """
    ab_set = set(answer_bearing_doc_ids)

    # Token utilisation
    token_utilisation = total_tokens / token_budget if token_budget > 0 else 0

    # Freshness and staleness by token count
    fresh_tokens = sum(d["token_count"] for d in assembled_docs if d["is_fresh"])
    stale_tokens = sum(d["token_count"] for d in assembled_docs if not d["is_fresh"])
    context_freshness_ratio = fresh_tokens / total_tokens if total_tokens > 0 else 0
    stale_token_ratio = stale_tokens / total_tokens if total_tokens > 0 else 0

    # Answer coverage
    context_doc_ids = set(d["doc_id"] for d in assembled_docs)
    answer_in_context = context_doc_ids & ab_set
    answer_coverage = 1 if answer_in_context else 0

    # Answer-bearing token ratio
    ab_tokens = sum(
        d["token_count"] for d in assembled_docs if d["doc_id"] in ab_set
    )
    answer_bearing_token_ratio = ab_tokens / total_tokens if total_tokens > 0 else 0

    return {
        "contradiction_density": contradiction_density,
        "redundancy_score": redundancy_score,
        "num_chunks": len(assembled_docs),
        "assembly_latency_ms": assembly_latency_ms,
        "token_utilisation": token_utilisation,
        "total_tokens": total_tokens,
        "context_freshness_ratio": context_freshness_ratio,
        "stale_token_ratio": stale_token_ratio,
        "answer_coverage": answer_coverage,
        "answer_bearing_token_ratio": answer_bearing_token_ratio,
    }


# ─────────────────────────────────────────────
# EVALUATION LOOP
# ─────────────────────────────────────────────
def evaluate_condition(
    nli_scorer,
    redundancy_scorer,
    rerank_results,
    doc_metadata,
    corpus_doc_ids,
    token_budget,
    k,
    condition_name,
    query_lookup=None,
):
    """Assemble context for all queries in one condition and collect metrics.

    Also builds generation-ready payloads when query_lookup is provided.
    """
    results = []
    gen_payloads = []

    cond_results = [r for r in rerank_results if r["condition"] == condition_name]
    print(f"    Assembling context for {len(cond_results)} queries...")

    for i, ret in enumerate(cond_results):
        query_id = ret["query_id"]
        domain = ret["domain"]
        time_sensitive = ret["time_sensitive"]
        reranked_doc_ids = ret["reranked_doc_ids"]
        ab_doc_ids = ret["answer_bearing_doc_ids"]

        # ── Timed assembly + scoring ──
        t0 = time.perf_counter()

        # Assemble context
        assembled_docs, total_tokens, context_text = assemble_context(
            reranked_doc_ids=reranked_doc_ids,
            doc_metadata=doc_metadata,
            token_budget=token_budget,
            corpus_doc_ids=corpus_doc_ids,
        )

        if not assembled_docs:
            continue

        # NLI contradiction density
        doc_texts = [d["text"] for d in assembled_docs]
        contradiction_density, nli_labels, _ = nli_scorer.score_pairs(doc_texts)

        # Redundancy score
        redundancy, _ = redundancy_scorer.score(doc_texts)

        t1 = time.perf_counter()
        assembly_latency_ms = (t1 - t0) * 1000

        # Compute metrics
        metrics = compute_context_metrics(
            assembled_docs=assembled_docs,
            total_tokens=total_tokens,
            token_budget=token_budget,
            contradiction_density=contradiction_density,
            redundancy_score=redundancy,
            assembly_latency_ms=assembly_latency_ms,
            answer_bearing_doc_ids=ab_doc_ids,
        )

        result = {
            "condition": condition_name,
            "query_id": query_id,
            "domain": domain,
            "time_sensitive": time_sensitive,
            "k": k,
            "token_budget": token_budget,
            "assembled_doc_ids": [d["doc_id"] for d in assembled_docs],
            "assembled_doc_tokens": [d["token_count"] for d in assembled_docs],
            "answer_bearing_doc_ids": ab_doc_ids,
            "reranked_doc_ids": reranked_doc_ids,
            "nli_pair_labels": nli_labels,
            "reranked_precision_at_k": ret.get("reranked_precision_at_k", 0),
            "reranked_stale_intrusion_rate": ret.get("reranked_stale_intrusion_rate", 0),
            **metrics,
        }
        results.append(result)

        # ── Generation-ready payload ──
        if query_lookup is not None:
            query_info = query_lookup.get(query_id, {})
            gen_payloads.append({
                "query_id": query_id,
                "condition": condition_name,
                "domain": domain,
                "time_sensitive": time_sensitive,
                "question": query_info.get("question", ""),
                "ground_truth": query_info.get("ground_truth", ""),
                "context": context_text,
                "context_tokens": total_tokens,
                "assembled_doc_ids": [d["doc_id"] for d in assembled_docs],
                "answer_bearing_in_context": bool(
                    set(d["doc_id"] for d in assembled_docs) & set(ab_doc_ids)
                ),
                "num_chunks": len(assembled_docs),
                "contradiction_density": contradiction_density,
                "redundancy_score": redundancy,
                "stale_token_ratio": metrics["stale_token_ratio"],
            })

        if (i + 1) % 50 == 0:
            print(f"      Processed {i+1}/{len(cond_results)} queries")

    return results, gen_payloads


# ─────────────────────────────────────────────
# AGGREGATION
# ─────────────────────────────────────────────
def _agg(results):
    """Compute mean and std for all numeric metrics."""
    if not results:
        return {}

    metric_keys = [
        "contradiction_density",
        "redundancy_score",
        "num_chunks",
        "assembly_latency_ms",
        "token_utilisation",
        "total_tokens",
        "context_freshness_ratio",
        "stale_token_ratio",
        "answer_coverage",
        "answer_bearing_token_ratio",
        "reranked_precision_at_k",
        "reranked_stale_intrusion_rate",
    ]

    agg = {"n": len(results)}
    for key in metric_keys:
        values = [r[key] for r in results if key in r]
        if values:
            agg[f"{key}_mean"] = np.mean(values)
            agg[f"{key}_std"] = np.std(values)
    return agg


def print_summary_table(all_results, conditions, k, token_budget):
    """Print a formatted summary table across conditions."""
    print("\n" + "=" * 110)
    print(f"CONTEXT ASSEMBLY EVALUATION SUMMARY (k={k}, token_budget={token_budget})")
    print("=" * 110)

    header = f"{'Metric':<40}"
    for cond in conditions:
        header += f"  {cond:>12}"
    print(f"\n{header}")
    print("-" * (40 + 14 * len(conditions)))

    metrics_to_show = [
        ("Contradiction density (NLI)", "contradiction_density_mean"),
        ("Redundancy score (cosine sim)", "redundancy_score_mean"),
        ("Num chunks to fill budget", "num_chunks_mean"),
        ("Assembly latency (ms)", "assembly_latency_ms_mean"),
        ("Token utilisation", "token_utilisation_mean"),
        ("Avg tokens used", "total_tokens_mean"),
        ("Context freshness ratio", "context_freshness_ratio_mean"),
        ("Stale token ratio", "stale_token_ratio_mean"),
        ("Answer coverage", "answer_coverage_mean"),
        ("Answer-bearing token ratio", "answer_bearing_token_ratio_mean"),
        ("Precision@k (from rerank)", "reranked_precision_at_k_mean"),
        ("Stale intrusion (from rerank)", "reranked_stale_intrusion_rate_mean"),
    ]

    cond_aggs = {}
    for cond in conditions:
        cond_results = [r for r in all_results if r["condition"] == cond]
        cond_aggs[cond] = _agg(cond_results)

    for label, key in metrics_to_show:
        row = f"{label:<40}"
        for cond in conditions:
            val = cond_aggs[cond].get(key, 0)
            if "tokens" in key and "ratio" not in key:
                row += f"  {val:>12.1f}"
            elif "latency" in key:
                row += f"  {val:>12.1f}"
            elif "num_chunks" in key:
                row += f"  {val:>12.2f}"
            else:
                row += f"  {val:>12.4f}"
        print(row)

    # ── By time sensitivity ──
    print(f"\n{'─' * 110}")
    print("BREAKDOWN: Time-sensitive vs. Time-insensitive queries")
    print(f"{'─' * 110}")

    for ts_label, ts_val in [("Time-sensitive", True), ("Time-insensitive", False)]:
        print(f"\n  {ts_label}:")
        header = f"  {'Metric':<38}"
        for cond in conditions:
            header += f"  {cond:>12}"
        print(header)
        print("  " + "-" * (38 + 14 * len(conditions)))

        for label, key in [
            ("Contradiction density", "contradiction_density_mean"),
            ("Redundancy score", "redundancy_score_mean"),
            ("Num chunks", "num_chunks_mean"),
            ("Assembly latency (ms)", "assembly_latency_ms_mean"),
            ("Stale token ratio", "stale_token_ratio_mean"),
            ("Answer coverage", "answer_coverage_mean"),
        ]:
            row = f"  {label:<38}"
            for cond in conditions:
                cond_ts_results = [
                    r for r in all_results
                    if r["condition"] == cond and r["time_sensitive"] == ts_val
                ]
                agg = _agg(cond_ts_results)
                val = agg.get(key, 0)
                if "latency" in key:
                    row += f"  {val:>12.1f}"
                elif "num_chunks" in key:
                    row += f"  {val:>12.2f}"
                else:
                    row += f"  {val:>12.4f}"
            print(row)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Context assembly stage evaluation")
    parser.add_argument(
        "--corpus_dir",
        type=str,
        required=True,
        help="Directory containing corpus_*.jsonl and queries_clean.jsonl",
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K, help="Top-k from reranking (default: 5)"
    )
    parser.add_argument(
        "--token_budget",
        type=int,
        default=DEFAULT_TOKEN_BUDGET,
        help="Max tokens for assembled context (default: 2000)",
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
    args = parser.parse_args()

    corpus_dir = args.corpus_dir
    k = args.k
    token_budget = args.token_budget
    conditions = args.conditions
    output_dir = args.output_dir or os.path.join(corpus_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("CONTEXT ASSEMBLY STAGE EVALUATION")
    print(f"  Token budget: {token_budget}")
    print(f"  k: {k}")
    print(f"  NLI model: {NLI_MODEL_NAME}")
    print(f"  Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"  Conditions: {conditions}")
    print(f"  Corpus dir: {corpus_dir}")
    print(f"  Output dir: {output_dir}")
    print("=" * 60)

    # ── Load rerank results ──
    rerank_path = os.path.join(output_dir, "rerank_results_detailed.jsonl")
    if not os.path.exists(rerank_path):
        print(f"\n  ERROR: {rerank_path} not found.")
        print("  Run rerank_eval.py first to generate reranking results.")
        return

    rerank_results = load_jsonl(rerank_path)
    print(f"\n  Loaded {len(rerank_results)} rerank results")

    # ── Load queries (needed for generation payloads) ──
    queries_path = os.path.join(corpus_dir, "queries_clean.jsonl")
    if not os.path.exists(queries_path):
        print(f"\n  WARNING: {queries_path} not found. Generation payloads will lack question/ground_truth.")
        query_lookup = None
    else:
        queries = load_jsonl(queries_path)
        query_lookup = {q["query_id"]: q for q in queries}
        print(f"  Loaded {len(queries)} queries for generation prep")

    # ── Initialize scorers ──
    nli_scorer = NLIScorer(NLI_MODEL_NAME)
    redundancy_scorer = RedundancyScorer(EMBEDDING_MODEL_NAME)

    # ── Evaluate each condition ──
    all_results = []
    all_gen_payloads = []

    for cond in conditions:
        corpus_path = os.path.join(corpus_dir, f"corpus_{cond}.jsonl")
        if not os.path.exists(corpus_path):
            print(f"\n  WARNING: {corpus_path} not found, skipping {cond}")
            continue

        print(f"\n{'─' * 60}")
        print(f"  CONDITION: {cond}")
        print(f"{'─' * 60}")

        # Load corpus
        corpus = load_jsonl(corpus_path)
        doc_metadata = {d["doc_id"]: d for d in corpus}
        corpus_doc_ids = [d["doc_id"] for d in corpus]
        print(f"  Loaded {len(corpus)} docs")

        stale_count = sum(1 for d in corpus if not d["is_fresh"])
        print(f"  Stale docs in corpus: {stale_count}")

        # Run context assembly evaluation
        results, gen_payloads = evaluate_condition(
            nli_scorer=nli_scorer,
            redundancy_scorer=redundancy_scorer,
            rerank_results=rerank_results,
            doc_metadata=doc_metadata,
            corpus_doc_ids=corpus_doc_ids,
            token_budget=token_budget,
            k=k,
            condition_name=cond,
            query_lookup=query_lookup,
        )
        all_results.extend(results)
        all_gen_payloads.extend(gen_payloads)

        # Quick summary
        agg = _agg(results)
        print(f"\n  Quick summary for {cond}:")
        print(f"    Contradiction density: {agg.get('contradiction_density_mean', 0):.4f}")
        print(f"    Redundancy score:      {agg.get('redundancy_score_mean', 0):.4f}")
        print(f"    Num chunks (avg):      {agg.get('num_chunks_mean', 0):.2f}")
        print(f"    Assembly latency (ms): {agg.get('assembly_latency_ms_mean', 0):.1f}")
        print(f"    Answer coverage:       {agg.get('answer_coverage_mean', 0):.4f}")

        del corpus, doc_metadata
        import gc
        gc.collect()

    # ── Print full summary ──
    print_summary_table(all_results, conditions, k, token_budget)

    # ── Save detailed results ──
    save_jsonl(
        all_results,
        os.path.join(output_dir, "context_assembly_results_detailed.jsonl"),
    )
    print(f"\n  Detailed results saved to {output_dir}/context_assembly_results_detailed.jsonl")

    # ── Save generation-ready payloads ──
    if all_gen_payloads:
        gen_path = os.path.join(output_dir, "generation_payloads.jsonl")
        save_jsonl(all_gen_payloads, gen_path)
        print(f"  Generation payloads saved to {gen_path}")
        print(f"  ({len(all_gen_payloads)} payloads across {len(conditions)} conditions)")

    # ── Save aggregated results ──
    agg_results = {
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

    with open(os.path.join(output_dir, "context_assembly_results_aggregated.json"), "w") as f:
        json.dump(agg_results, f, indent=2)
    print(f"  Aggregated results saved to {output_dir}/context_assembly_results_aggregated.json")

    print(f"\n{'=' * 60}")
    print("CONTEXT ASSEMBLY EVALUATION COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
