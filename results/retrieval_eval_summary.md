# Retrieval Evaluation Summary

## Bottom Line

When outdated documents contaminate an AI search corpus, the system **silently serves wrong answers** — with no drop in speed or confidence scores to alert anyone.

## Key Numbers

| Metric | Fresh Corpus | 50% Stale | Impact |
|--------|-------------|-----------|--------|
| Fresh docs retrieved per query | 1.20 | 0.91 | **−24% fewer current answers** |
| Stale docs infiltrating results | 0.00 | 0.77 | **0.8 outdated docs per query** |
| Retrieval speed | 18ms | 12ms | No degradation |
| Confidence scores | 0.488 | 0.488 | Identical — system can't tell the difference |

## What This Means

1. **The system looks healthy while serving stale content.** Speed and relevance scores stay flat. Standard monitoring won't catch it.

2. **Time-sensitive queries are hit hardest.** Fresh docs retrieved dropped 51% for time-sensitive queries vs. <1% for static ones. Healthcare and IT domains saw the steepest quality declines (−2.4% precision).

3. **Contamination spreads beyond the target.** Even queries unrelated to the outdated content picked up stale documents (9.2% intrusion rate), because outdated and current versions look identical to the search engine.

4. **Legal documents are most vulnerable.** Highest stale intrusion rate (26%) — formulaic language makes old and new versions indistinguishable in search.

## Implication

Current embedding-based retrieval cannot distinguish fresh from stale content. Fixing this requires either injecting temporal metadata into the retrieval pipeline or adding a freshness filter at the generation stage — tuning the search engine alone will not solve it.
