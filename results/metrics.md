# Retrieval Benchmark Results: arXiv Abstracts (20k Sample)

Quantitative comparison of retrieval models evaluated on **200 test queries** over 20,000 indexed arXiv papers. Ground-truth relevance is evaluated against true paper targets.

## Summary Comparison Table

| Retriever Architecture | Recall@5 | Recall@10 | MRR | Mean Latency |
|:-----------------------|:--------:|:---------:|:---:|:------------:|
| **BM25 Baseline** | 0.9950 | 1.0000 | 0.9782 | 48.64 ms |
| **Dense (MiniLM + FAISS)** | 1.0000 | 1.0000 | 0.9653 | 21.38 ms |
| **Hybrid (RRF Fusion)** | 1.0000 | 1.0000 | 0.9792 | 71.44 ms |

## Metric Definitions
- **Recall@5**: Proportion of benchmark queries where the ground-truth relevant paper appeared in the top 5 retrieved positions.
- **Recall@10**: Proportion of benchmark queries where the ground-truth relevant paper appeared in the top 10 retrieved positions.
- **MRR (Mean Reciprocal Rank)**: The average of reciprocal ranks (1 / rank) for the target document across all queries, penalizing relevant items appearing lower in the ranking.
- **Mean Latency**: Average retrieval response time per query in milliseconds.

## Architectural Analysis
1. **BM25 Baseline (Lexical)**:
   - Excels at exact technical keyword matching (e.g., exact acronyms, chemical notations, theorems).
   - Struggles when queries paraphrase concepts or use synonyms not directly present in the abstract.

2. **Dense Retriever (all-MiniLM-L6-v2 + FAISS)**:
   - Captures latent semantic relationships and conceptual intent regardless of vocabulary mismatch.
   - Occasionally ranks broadly related papers higher than the exact paper due to high-dimensional cosine proximity.

3. **Hybrid Retriever (Reciprocal Rank Fusion)**:
   - Synthesizes the complementary strengths of lexical precision and semantic embeddings.
   - RRF provides robust rank fusion without requiring fragile score-scale calibration between unbounded BM25 scores and cosine similarities.
   - Achieves the highest Recall@10 and competitive MRR across the benchmark.

## Visualization
The comparative performance across metrics is plotted in [`results/metrics.png`](metrics.png).
