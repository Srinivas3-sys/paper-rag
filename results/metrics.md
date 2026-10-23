# Retrieval Benchmark Results: arXiv Abstracts (20k Sample)

Quantitative comparison of retrieval models evaluated on an overhauled **450-query multi-regime benchmark** (150 dev queries for tuning, 300 held-out test queries with n=100 per regime). Ground truth relevance is strictly separated between tuning and testing.

All metrics are reported on the **held-out test split** with **95% bootstrap confidence intervals** (1,000 resamples).
- **Strict Relevance**: Evaluates exact retrieval of the designated target document.
- **Lenient Relevance**: Treats a document as relevant if it is the target OR shares the target's arXiv category with dense cosine similarity >= 0.3000 (calibrated on DEV split).

---

## 1. Overall Held-Out Test Set Performance (N=300, Strict Target Ground Truth)

| Retriever Architecture | Recall@1 [95% CI] | Recall@5 [95% CI] | Recall@10 [95% CI] | Recall@20 [95% CI] | MRR [95% CI] | nDCG@10 [95% CI] | Latency (Mean, p95) |
|:-----------------------|:-----------------:|:-----------------:|:------------------:|:------------------:|:------------:|:----------------:|:-------------------:|
| **BM25 Baseline** | 0.563 [0.507, 0.620] | 0.700 [0.647, 0.750] | 0.747 [0.700, 0.793] | 0.777 [0.727, 0.823] | 0.624 [0.574, 0.675] | 0.651 [0.602, 0.700] | 3.7 ms (p95: 4.1) |
| **Dense (MiniLM+FAISS)** | 0.407 [0.350, 0.463] | 0.510 [0.450, 0.567] | 0.543 [0.480, 0.597] | 0.583 [0.527, 0.633] | 0.454 [0.401, 0.506] | 0.472 [0.418, 0.525] | 9.6 ms (p95: 12.3) |
| **Hybrid (RRF Optimal)** | 0.547 [0.490, 0.600] | 0.723 [0.673, 0.773] | 0.757 [0.707, 0.807] | 0.787 [0.737, 0.830] | 0.623 [0.572, 0.672] | 0.653 [0.604, 0.700] | 19.7 ms (p95: 33.4) |

### Overall Paired Significance Testing (Hybrid vs. Best Single Retriever: BM25 Baseline)
- **Mean Pairwise Difference**: -0.0012 [-0.0180, +0.0167]
- **Wilcoxon Signed-Rank Test**: W = 8721.0, **p = 0.9221**

---

## 2. Overall Held-Out Test Set Performance (N=300, Lenient Category Multi-Relevant)

| Retriever Architecture | Lenient Recall@1 [95% CI] | Lenient Recall@5 [95% CI] | Lenient Recall@10 [95% CI] | Lenient Recall@20 [95% CI] | Lenient MRR [95% CI] | Lenient nDCG@10 [95% CI] |
|:-----------------------|:-------------------------:|:-------------------------:|:--------------------------:|:--------------------------:|:--------------------:|:------------------------:|
| **BM25 Baseline** | 0.663 [0.607, 0.717] | 0.823 [0.777, 0.860] | 0.873 [0.837, 0.910] | 0.897 [0.860, 0.930] | 0.733 [0.686, 0.776] | 0.589 [0.551, 0.627] |
| **Dense (MiniLM+FAISS)** | 0.603 [0.547, 0.660] | 0.753 [0.700, 0.803] | 0.817 [0.770, 0.863] | 0.860 [0.820, 0.900] | 0.675 [0.628, 0.725] | 0.524 [0.482, 0.565] |
| **Hybrid (RRF Optimal)** | 0.683 [0.630, 0.733] | 0.857 [0.813, 0.893] | 0.893 [0.857, 0.927] | 0.917 [0.883, 0.947] | 0.757 [0.714, 0.799] | 0.579 [0.542, 0.615] |

---

## 3. Regime-by-Regime Breakdown (Strict vs Lenient, N=300)

| Query Regime | Retriever Architecture | Strict Recall@10 [95% CI] | Strict MRR [95% CI] | Lenient Recall@10 [95% CI] | Lenient MRR [95% CI] |
|:-------------|:-----------------------|:-------------------------:|:-------------------:|:--------------------------:|:--------------------:|
| Lexical (n=100) | BM25 Baseline | 1.000 [1.000, 1.000] | 0.990 [0.975, 1.000] | 1.000 [1.000, 1.000] | 0.995 [0.985, 1.000] |
| Lexical (n=100) | Dense (MiniLM+FAISS) | 0.980 [0.950, 1.000] | 0.915 [0.864, 0.958] | 1.000 [1.000, 1.000] | 0.983 [0.960, 1.000] |
| Lexical (n=100) | Hybrid (RRF Optimal) | 1.000 [1.000, 1.000] | 0.985 [0.965, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| Paraphrase (n=100) | BM25 Baseline | 0.380 [0.290, 0.480] | 0.222 [0.153, 0.292] | 0.650 [0.560, 0.740] | 0.398 [0.315, 0.481] |
| Paraphrase (n=100) | Dense (MiniLM+FAISS) | 0.340 [0.250, 0.430] | 0.216 [0.151, 0.289] | 0.760 [0.670, 0.840] | 0.526 [0.440, 0.608] |
| Paraphrase (n=100) | Hybrid (RRF Optimal) | 0.420 [0.320, 0.520] | 0.250 [0.181, 0.317] | 0.710 [0.620, 0.800] | 0.471 [0.385, 0.555] |
| Acronym (n=100) | BM25 Baseline | 0.860 [0.790, 0.920] | 0.660 [0.584, 0.739] | 0.970 [0.930, 1.000] | 0.805 [0.745, 0.867] |
| Acronym (n=100) | Dense (MiniLM+FAISS) | 0.310 [0.220, 0.400] | 0.232 [0.159, 0.314] | 0.690 [0.600, 0.780] | 0.518 [0.428, 0.606] |
| Acronym (n=100) | Hybrid (RRF Optimal) | 0.850 [0.780, 0.910] | 0.633 [0.556, 0.713] | 0.970 [0.930, 1.000] | 0.800 [0.736, 0.863] |

---

## 4. Paired Significance Testing by Regime (Hybrid vs Best Single Retriever)

| Query Regime | Best Single Baseline | Mean MRR Diff (`Hybrid - Baseline`) [95% CI] | Wilcoxon W | p-value |
|:---|:---|:---:|:---:|:---:|
| **Lexical** (n=100) | BM25 Baseline | -0.0050 [-0.0250, +0.0100] | W = 99.0 | p = 0.5637 |
| **Paraphrase** (n=100) | BM25 Baseline | +0.0282 [-0.0083, +0.0632] | W = 791.0 | p = 0.0054 |
| **Acronym** (n=100) | BM25 Baseline | -0.0268 [-0.0599, +0.0067] | W = 536.0 | p = 0.0049 |

---

## 5. RRF Hyperparameter Tuning (Dev Split Ablation, N=150)

Grid search across RRF rank smoothing constant $k$ and dense weighting factor $\alpha$ conducted strictly on the dev split:

| RRF Parameter k | Weight Ratio $\alpha$ | Dev MRR | Dev nDCG@10 | Dev Recall@10 |
|:---|:---:|:---:|:---:|:---:|
| k=10 | alpha=0.3 | 0.6227 | 0.6484 | 0.7400 |
| k=10 | alpha=0.5 | 0.5629 | 0.6069 | 0.7533 |
| k=10 | alpha=0.7 | 0.4744 | 0.4942 | 0.5933 |
| k=30 | alpha=0.3 | 0.6110 | 0.6419 | 0.7467 |
| k=30 | alpha=0.5 | 0.5474 | 0.5942 | 0.7533 |
| k=30 | alpha=0.7 | 0.4779 | 0.5035 | 0.6000 |
| k=60 | alpha=0.3 | 0.5945 | 0.6271 | 0.7400 |
| k=60 | alpha=0.5 | 0.5516 | 0.5975 | 0.7533 |
| k=60 | alpha=0.7 | 0.4807 | 0.5103 | 0.6067 |
| k=100 | alpha=0.3 | 0.5802 | 0.6182 | 0.7467 |
| k=100 | alpha=0.5 | 0.5538 | 0.5991 | 0.7533 |
| k=100 | alpha=0.7 | 0.4832 | 0.5124 | 0.6067 |

**Optimal Dev Setting**: $k=10$, $\alpha=0.3$.
