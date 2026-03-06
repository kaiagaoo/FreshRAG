"""
Stale Corpus Preparation Pipeline
==================================
Steps:
1. Filter out time-sensitive queries with zero answer-bearing docs
2. Cap doc text at 1000 words for stale generation
3. Generate all stale variants via Gemini API
4. Build four corpus conditions (Fresh, Stale-10%, 30%, 50%) stratified by domain
5. Save corpus and query JSONL files

Usage:
  python stale_pipeline.py --step prep       # Steps 1-2: prepare stale candidates
  python stale_pipeline.py --step generate   # Step 3: call Gemini API
  python stale_pipeline.py --step build      # Steps 4-5: build four corpus conditions
  python stale_pipeline.py --step all        # Run everything
"""

import json
import os
import random
import time
import argparse
from collections import Counter, defaultdict
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_QUERIES = "queries.jsonl"
INPUT_CORPUS = "corpus.jsonl"
OUTPUT_DIR = "freshrag_experiment"
MAX_WORDS_FOR_STALE = 1000  # Truncate long docs before sending to Gemini
RANDOM_SEED = 42
GEMINI_MODEL = "gemini-2.5-flash"  # Change if needed

# Staleness ratios for each condition
STALE_RATIOS = {
    "fresh": 0.0,
    "stale_10": 0.10,
    "stale_30": 0.30,
    "stale_50": 0.50,
}


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
    print(f"  Saved {len(data)} records to {path}")


def truncate_text(text, max_words=MAX_WORDS_FOR_STALE):
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " [TRUNCATED]"


def get_domain(doc_id):
    """Extract domain from doc_id like 'covidqa_01_doc_0' -> 'covidqa'"""
    # Handle domains with underscores in name (e.g., 'expertqa')
    parts = doc_id.split("_")
    # Find where the numeric part starts
    for i, part in enumerate(parts):
        if part.isdigit():
            return "_".join(parts[:i])
    return parts[0]


# ─────────────────────────────────────────────
# STEP 1-2: PREP — Filter queries, identify stale candidates
# ─────────────────────────────────────────────
def step_prep():
    print("=" * 60)
    print("STEP 1-2: Preparing stale candidates")
    print("=" * 60)

    queries = load_jsonl(INPUT_QUERIES)
    corpus = load_jsonl(INPUT_CORPUS)
    corpus_lookup = {d["doc_id"]: d for d in corpus}

    # ── Filter out time-sensitive queries with 0 answer-bearing docs ──
    ts_queries = [q for q in queries if q["time_sensitive"]]
    nts_queries = [q for q in queries if not q["time_sensitive"]]

    valid_ts = [q for q in ts_queries if len(q["answer_bearing_doc_ids"]) > 0]
    dropped = [q for q in ts_queries if len(q["answer_bearing_doc_ids"]) == 0]

    print(f"\n  Time-sensitive queries: {len(ts_queries)}")
    print(f"  Dropped (0 answer-bearing docs): {len(dropped)}")
    if dropped:
        for q in dropped:
            print(f"    - {q['query_id']}: {q['question'][:60]}...")
    print(f"  Valid time-sensitive queries: {len(valid_ts)}")
    print(f"  Time-insensitive queries (unchanged): {len(nts_queries)}")

    # ── Identify stale candidates ──
    # Answer-bearing docs from time-sensitive queries
    stale_candidate_ids = set()
    for q in valid_ts:
        for doc_id in q["answer_bearing_doc_ids"]:
            stale_candidate_ids.add(doc_id)

    stale_candidates = []
    for doc_id in stale_candidate_ids:
        if doc_id in corpus_lookup:
            doc = corpus_lookup[doc_id].copy()
            doc["text_for_stale_gen"] = truncate_text(doc["text"])
            doc["word_count"] = len(doc["text"].split())
            doc["truncated"] = len(doc["text"].split()) > MAX_WORDS_FOR_STALE
            stale_candidates.append(doc)

    # Domain breakdown
    domain_counts = Counter(get_domain(d["doc_id"]) for d in stale_candidates)
    print(f"\n  Total stale candidates: {len(stale_candidates)}")
    print(f"  By domain:")
    for domain, count in domain_counts.most_common():
        print(f"    {domain}: {count}")

    # Length stats
    truncated = sum(1 for d in stale_candidates if d["truncated"])
    print(f"\n  Docs truncated to {MAX_WORDS_FOR_STALE} words: {truncated}")

    # ── Save outputs ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save cleaned queries (drop the 0-answer-bearing ones)
    all_valid_queries = valid_ts + nts_queries
    save_jsonl(all_valid_queries, os.path.join(OUTPUT_DIR, "queries_clean.jsonl"))

    # Save stale candidates (for Gemini generation)
    save_jsonl(stale_candidates, os.path.join(OUTPUT_DIR, "stale_candidates.jsonl"))

    # Save metadata for later steps
    metadata = {
        "total_queries": len(all_valid_queries),
        "time_sensitive_queries": len(valid_ts),
        "time_insensitive_queries": len(nts_queries),
        "dropped_queries": len(dropped),
        "total_corpus_docs": len(corpus),
        "stale_candidates": len(stale_candidates),
        "domains": dict(domain_counts),
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata saved to {OUTPUT_DIR}/metadata.json")

    return stale_candidates


# ─────────────────────────────────────────────
# STEP 3: GENERATE — Call Gemini to create stale versions
# ─────────────────────────────────────────────
STALE_GENERATION_PROMPT = """You are helping create a research dataset to study how outdated content affects AI systems.

Given the following document passage, rewrite it so that it reflects plausible but OUTDATED information from 6-12 months ago.

RULES:
1. Keep the same writing style, structure, and approximate length.
2. Only change TIME-SENSITIVE FACTS: statistics, numbers, rankings, dates, holders of positions, event outcomes, prices, versions, status updates.
3. The rewritten passage should still read naturally and be semantically similar to the original.
4. Do NOT add disclaimers or notes about it being outdated.
5. Return ONLY the rewritten passage, nothing else.

ORIGINAL PASSAGE:
{text}

OUTDATED VERSION:"""


def generate_stale_version_gemini(text, api_key):
    """Call Gemini API to generate a stale version of the text."""
    import requests

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": STALE_GENERATION_PROMPT.format(text=text)}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 2048,
        },
    }

    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()

    # Extract text from Gemini response
    try:
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as e:
        print(f"    WARNING: Failed to parse Gemini response: {e}")
        return None


def step_generate():
    print("=" * 60)
    print("STEP 3: Generating stale versions via Gemini")
    print("=" * 60)

    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n  ERROR: Set GOOGLE_API_KEY or GOOGLE_API_KEY environment variable.")
        print("  Example: export GOOGLE_API_KEY='your-key-here'")
        print("  Then re-run: python stale_pipeline.py --step generate")
        return

    candidates_path = os.path.join(OUTPUT_DIR, "stale_candidates.jsonl")
    output_path = os.path.join(OUTPUT_DIR, "stale_variants.jsonl")

    candidates = load_jsonl(candidates_path)
    print(f"\n  Loaded {len(candidates)} stale candidates")

    # Check for existing progress (resume support)
    existing = {}
    if os.path.exists(output_path):
        existing_data = load_jsonl(output_path)
        existing = {d["doc_id"]: d for d in existing_data}
        print(f"  Found {len(existing)} already generated (resuming)")

    results = list(existing.values())
    to_generate = [c for c in candidates if c["doc_id"] not in existing]
    print(f"  Remaining to generate: {len(to_generate)}")

    for i, candidate in enumerate(to_generate):
        doc_id = candidate["doc_id"]
        text = candidate["text_for_stale_gen"]

        print(f"\n  [{i+1}/{len(to_generate)}] Generating stale for {doc_id} "
              f"({len(text.split())} words)...", end=" ")

        try:
            stale_text = generate_stale_version_gemini(text, api_key)

            if stale_text:
                result = {
                    "doc_id": doc_id,
                    "original_text": candidate["text"],
                    "stale_text": stale_text,
                    "domain": get_domain(doc_id),
                    "word_count_original": len(candidate["text"].split()),
                    "word_count_stale": len(stale_text.split()),
                    "was_truncated": candidate["truncated"],
                }
                results.append(result)
                print("OK")
            else:
                print("FAILED (empty response)")

        except Exception as e:
            print(f"ERROR: {e}")

        # Save progress after every 10 docs (resume-friendly)
        if (i + 1) % 10 == 0:
            save_jsonl(results, output_path)
            print(f"  Progress saved ({len(results)} total)")

        # Rate limiting: Gemini free tier = 15 RPM
        time.sleep(4)

    # Final save
    save_jsonl(results, output_path)
    print(f"\n  DONE. Generated {len(results)} stale variants.")
    print(f"  Saved to {output_path}")


# ─────────────────────────────────────────────
# STEP 4-5: BUILD — Create four corpus conditions
# ─────────────────────────────────────────────
def step_build():
    print("=" * 60)
    print("STEP 4-5: Building four corpus conditions")
    print("=" * 60)

    random.seed(RANDOM_SEED)

    corpus = load_jsonl(INPUT_CORPUS)
    stale_variants = load_jsonl(os.path.join(OUTPUT_DIR, "stale_variants.jsonl"))
    queries = load_jsonl(os.path.join(OUTPUT_DIR, "queries_clean.jsonl"))

    # Build lookup: doc_id -> stale_text
    stale_lookup = {v["doc_id"]: v["stale_text"] for v in stale_variants}
    print(f"\n  Corpus docs: {len(corpus)}")
    print(f"  Stale variants available: {len(stale_lookup)}")

    # ── Group stale candidates by domain for stratified sampling ──
    stale_ids_by_domain = defaultdict(list)
    for doc_id in stale_lookup:
        domain = get_domain(doc_id)
        stale_ids_by_domain[domain].append(doc_id)

    # Shuffle within each domain
    for domain in stale_ids_by_domain:
        random.shuffle(stale_ids_by_domain[domain])

    total_stale = len(stale_lookup)
    print(f"\n  Stale candidates by domain:")
    for domain, ids in sorted(stale_ids_by_domain.items()):
        print(f"    {domain}: {len(ids)}")

    # ── Build each condition ──
    for condition_name, ratio in STALE_RATIOS.items():
        n_to_swap = round(total_stale * ratio)
        print(f"\n  ── Building {condition_name} (swap {n_to_swap}/{total_stale}) ──")

        # Stratified sampling: proportional per domain
        swap_ids = set()
        for domain, ids in stale_ids_by_domain.items():
            domain_n = round(len(ids) * ratio)
            swap_ids.update(ids[:domain_n])

        # Adjust if rounding caused slight mismatch
        all_remaining = [did for domain_ids in stale_ids_by_domain.values()
                         for did in domain_ids if did not in swap_ids]
        while len(swap_ids) < n_to_swap and all_remaining:
            swap_ids.add(all_remaining.pop())
        while len(swap_ids) > n_to_swap and swap_ids:
            swap_ids.pop()

        print(f"    Actual swaps: {len(swap_ids)}")

        # Domain breakdown of swaps
        swap_domains = Counter(get_domain(did) for did in swap_ids)
        for domain, count in swap_domains.most_common():
            print(f"      {domain}: {count}")

        # Build the corpus for this condition
        condition_corpus = []
        swapped_count = 0
        for doc in corpus:
            new_doc = doc.copy()
            if doc["doc_id"] in swap_ids:
                new_doc["text"] = stale_lookup[doc["doc_id"]]
                new_doc["is_fresh"] = False
                swapped_count += 1
            else:
                new_doc["is_fresh"] = True
            condition_corpus.append(new_doc)

        assert len(condition_corpus) == len(corpus), "Corpus size mismatch!"
        print(f"    Verified: {len(condition_corpus)} docs, {swapped_count} swapped")

        # Save
        save_jsonl(
            condition_corpus,
            os.path.join(OUTPUT_DIR, f"corpus_{condition_name}.jsonl"),
        )

    # ── Save swap mapping for analysis ──
    swap_mapping = {}
    for condition_name, ratio in STALE_RATIOS.items():
        n_to_swap = round(total_stale * ratio)
        swap_ids = set()
        for domain, ids in stale_ids_by_domain.items():
            domain_n = round(len(ids) * ratio)
            swap_ids.update(ids[:domain_n])
        all_remaining = [did for domain_ids in stale_ids_by_domain.values()
                         for did in domain_ids if did not in swap_ids]
        while len(swap_ids) < n_to_swap and all_remaining:
            swap_ids.add(all_remaining.pop())
        while len(swap_ids) > n_to_swap and swap_ids:
            swap_ids.pop()
        swap_mapping[condition_name] = sorted(swap_ids)

    with open(os.path.join(OUTPUT_DIR, "swap_mapping.json"), "w") as f:
        json.dump(swap_mapping, f, indent=2)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  Output directory: {OUTPUT_DIR}")
    print(f"  Files created:")
    print(f"    queries_clean.jsonl        — {len(queries)} queries")
    for condition_name in STALE_RATIOS:
        print(f"    corpus_{condition_name}.jsonl  — {len(corpus)} docs")
    print(f"    stale_variants.jsonl       — {len(stale_variants)} stale versions")
    print(f"    swap_mapping.json          — which docs swapped per condition")
    print(f"    metadata.json              — experiment metadata")
    print(f"\n  Ready for pipeline execution!")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stale corpus preparation pipeline")
    parser.add_argument(
        "--step",
        choices=["prep", "generate", "build", "all"],
        required=True,
        help="Which step to run",
    )
    args = parser.parse_args()

    if args.step == "prep":
        step_prep()
    elif args.step == "generate":
        step_generate()
    elif args.step == "build":
        step_build()
    elif args.step == "all":
        step_prep()
        step_generate()
        step_build()
