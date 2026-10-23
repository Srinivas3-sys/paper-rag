# Retrieval Benchmark Results: arXiv Abstracts (20k Sample)

Quantitative comparison of retrieval models evaluated on a **360-query multi-regime benchmark** (120 dev queries for tuning, 222 held-out test queries for final reporting). Ground truth relevance is strictly separated between tuning and testing.

All metrics are reported on the **held-out test split** with **95% bootstrap confidence intervals** (1,000 resamples).

---

## 1. Overall Held-Out Test Set Performance (N=222)

| Retriever Architecture | Recall@1 [95% CI] | Recall@5 [95% CI] | Recall@10 [95% CI] | Recall@20 [95% CI] | MRR [95% CI] | nDCG@10 [95% CI] | Latency (Mean, p95) |
|:-----------------------|:-----------------:|:-----------------:|:------------------:|:------------------:|:------------:|:----------------:|:-------------------:|
| **BM25 Baseline** | 0.586 [0.523, 0.649] | 0.694 [0.631, 0.752] | 0.716 [0.653, 0.770] | 0.766 [0.707, 0.820] | 0.636 [0.578, 0.695] | 0.653 [0.595, 0.710] | 25.7 ms (p95: 38.5) |
| **Dense (MiniLM+FAISS)** | 0.401 [0.333, 0.464] | 0.464 [0.396, 0.527] | 0.500 [0.437, 0.568] | 0.532 [0.464, 0.595] | 0.435 [0.372, 0.497] | 0.448 [0.385, 0.510] | 9.4 ms (p95: 10.1) |
| **Hybrid (RRF Optimal)** | 0.563 [0.495, 0.631] | 0.703 [0.640, 0.761] | 0.730 [0.667, 0.784] | 0.775 [0.721, 0.829] | 0.628 [0.568, 0.686] | 0.649 [0.591, 0.705] | 36.0 ms (p95: 49.3) |

### Paired Significance Testing (Hybrid vs. Best Single Retriever)
- **Comparison**: Hybrid (RRF Optimal) vs. BM25 Baseline on test MRR.
- **Mean Pairwise Difference**: -0.0089
- **Wilcoxon Signed-Rank Test**: W = 3458.5, **p = 0.2373**
- **Interpretation**: The difference is NOT statistically significant at p < 0.05; hybrid performance is statistically comparable within margin of error.

---

## 2. Regime-by-Regime Performance Breakdown

Evaluating performance across distinct query regimes demonstrates where each retrieval paradigm excels or degrades:

| Query Regime | Retriever Architecture | Recall@5 | Recall@10 | MRR | nDCG@10 |
|:-------------|:-----------------------|:--------:|:---------:|:---:|:-------:|
| Lexical | BM25 Baseline | 1.000 | 1.000 | 0.978 | 0.983 |
| Lexical | Dense (MiniLM+FAISS) | 0.963 | 0.988 | 0.927 | 0.941 |
| Lexical | Hybrid (RRF Optimal) | 1.000 | 1.000 | 0.981 | 0.986 |
| Paraphrase | BM25 Baseline | 0.194 | 0.226 | 0.158 | 0.169 |
| Paraphrase | Dense (MiniLM+FAISS) | 0.145 | 0.194 | 0.132 | 0.143 |
| Paraphrase | Hybrid (RRF Optimal) | 0.226 | 0.274 | 0.179 | 0.196 |
| Acronym | BM25 Baseline | 0.775 | 0.812 | 0.667 | 0.697 |
| Acronym | Dense (MiniLM+FAISS) | 0.212 | 0.250 | 0.178 | 0.193 |
| Acronym | Hybrid (RRF Optimal) | 0.775 | 0.812 | 0.622 | 0.664 |

### Key Regime Insights
1. **Lexical Queries (Easy Baseline)**:
   - BM25 achieves high recall because search keywords directly overlap with document tokens.
   - Dense retrieval performs competitively but BM25 leads on exact token intersection.
2. **Paraphrase Queries (Strict Zero-Leakage Conceptual Search)**:
   - Evaluated on queries with zero bigram overlap and zero title keyword overlap against the target paper.
   - Both single retrievers struggle when lexical overlap is absent (BM25 Recall@10 = 0.226, Dense Recall@10 = 0.194).
   - Hybrid RRF achieves the highest Recall@10 (0.274) and MRR (0.179) by combining complementary weak signals from both sparse and dense rankings.
3. **Acronym / Exact-Term Queries**:
   - Queries centered on distinctive mathematical theorems, algorithmic acronyms (e.g. ADMM, MIMO, QEC, GAN), and named entities.
   - BM25 captures rare token signals without embedding collision, matching or exceeding dense search.

---

## 3. RRF Hyperparameter Tuning (Dev Split Ablation)

Hyperparameter search was executed exclusively on the **dev split (N=120)** to guard against test-set leakage:

| RRF Constant (k) | Alpha (Dense Weight) | Dev MRR | Dev nDCG@10 | Dev Recall@10 |
|:-----------------|:--------------------:|:-------:|:-----------:|:-------------:|
| k=10 | alpha=0.3 | 0.6033 | 0.6355 | 0.7417 |
| k=10 | alpha=0.5 | 0.5562 | 0.5952 | 0.7250 |
| k=10 | alpha=0.7 | 0.4812 | 0.4990 | 0.5833 |
| k=30 | alpha=0.3 | 0.5911 | 0.6218 | 0.7250 |
| k=30 | alpha=0.5 | 0.5487 | 0.5825 | 0.7000 |
| k=30 | alpha=0.7 | 0.5067 | 0.5228 | 0.5833 |
| k=60 | alpha=0.3 | 0.5869 | 0.6163 | 0.7167 |
| k=60 | alpha=0.5 | 0.5512 | 0.5821 | 0.6917 |
| k=60 | alpha=0.7 | 0.5040 | 0.5232 | 0.5833 |
| k=100 | alpha=0.3 | 0.5721 | 0.6052 | 0.7167 |
| k=100 | alpha=0.5 | 0.5491 | 0.5804 | 0.6917 |
| k=100 | alpha=0.7 | 0.5040 | 0.5232 | 0.5833 |

**Selected Configuration**: `k = 10`, `alpha = 0.3`. This chosen setting was frozen and evaluated on the held-out test split.

---

## 4. Visual Comparison

The comparative performance across metrics and regimes is visualized in [`results/metrics.png`](metrics.png).
