# Dense Retrieval Diagnostic Report

Empirical investigation into why dense semantic retrieval (all-MiniLM-L6-v2) underperformed BM25 on the initial multi-regime benchmark.

---

## 1. Self-Retrieval Test (Verification of Indexing Correctness)

To verify that the FAISS index has no ID misalignment, shuffled ordering, metric mismatches, or missing normalization, 500 random documents were queried against the index.

| Query Formulation | Evaluated Documents | Rank-1 Recall | Rank-5 Recall | Mean Rank |
|:------------------|:-------------------:|:-------------:|:-------------:|:---------:|
| Exact Document Representation (`combined_text`) | 500 | **1.0000** (500/500) | 1.0000 | 1.00 |
| Abstract Alone (without Title prefix) | 500 | **0.9980** (499/500) | 0.9980 | 1.00 |

**Conclusion**: The FAISS index vector ordering and SQLite/Parquet document store are **100.0% aligned**. When an indexed paper's text is queried, dense search retrieves the exact document at Rank 1 in 500 out of 500 cases. There is no ID misalignment bug.

---

## 2. Encoder, Embedding, and Index Alignment Audit

- **Embedding Model**: `all-MiniLM-L6-v2` (Max sequence length: `256`)
- **Total Vectors in FAISS**: `20,000`
- **Total Rows in Docstore**: `20,000`
- **Counts Match**: `True`
- **FAISS Metric**: `IndexFlatIP` (Cosine similarity via inner product on normalized embeddings)
- **Prefix Requirement**: `None` (MiniLM is a symmetric bi-encoder)
- **Data Type**: `float32`
- **Reconstructed vs Fresh Embeddings Cosine Similarity**: Mean = `1.000000`, Min = `1.000000`
- **Max Vector Element Difference**: `1.19e-07` (at floating-point machine epsilon)

---

## 3. Abstract Length and Token Truncation Analysis

Analysis across all 20,000 corpus abstracts:

| Representation | Word Count (p50 / p95 / max) | MiniLM Tokens (p50 / p95 / max) | Truncation at 256 tokens | Truncation at 512 tokens |
|:---|:---:|:---:|:---:|:---:|
| **Abstract Only** | 109 / 247 / 342 | — | — | — |
| **Combined (`Title + Abstract`)** | 120 / 262 / 355 | 178 / 386 / 666 | **23.86%** (4,771 docs) | **0.38%** (76 docs) |

**Finding**: `all-MiniLM-L6-v2` hard-truncates at 256 tokens. Nearly a quarter (**23.86%**) of the arXiv corpus has concluding sentences cut off during embedding, impairing dense representation of longer papers. A 512-token model (such as BGE-small or E5-small) reduces truncation to 0.38%.

---

## 4. Query Specificity & Generic Collapse Audit

Analysis across all 102 paraphrase queries:

| Metric | Dense (MiniLM) | BM25 |
|:-------|:--------------:|:----:|
| **Corpus docs scoring above target (p25)** | 18.2 | 7.0 |
| **Corpus docs scoring above target (p50 / median)** | **244.5** | **130.0** |
| **Corpus docs scoring above target (p75)** | 4143.0 | 1564.5 |
| **Corpus docs scoring above target (Mean)** | 2914.8 | 2460.4 |
| **Mean Target Score** | 0.2593 | — |
| **Mean Rank-1 Score** | 0.4709 | — |
| **Mean Score Gap (`Top1 - Target`)** | 0.2116 | — |
| **Target in Top 10** | 19.6% | 27.5% |
| **Target in Top 100** | 37.3% | 49.0% |

### Root Cause: Boilerplate Prompt Collapse in Synthetic Paraphrase Generation
Inspection revealed that the query generation prompt allowed Flan-T5-base to parrot the prompt instructions whenever the title words were banned, producing degenerate queries such as:
1. `"What is the main problem"` (Dense Rank: 19,987; 19,986 docs above target)
2. `"What is the core problem of this paper"` (Dense Rank: 19,971; 19,970 docs above target)
3. `"What is the basic idea of the problem"` (Dense Rank: 18,548; 18,547 docs above target)
4. `"What is the best solution to this problem"` (Dense Rank: 16,256; 16,255 docs above target)

Because these boilerplate strings had zero n-gram overlap with the target abstract, the leakage filter accepted them, injecting contentless queries that destroyed dense retrieval MRR.

---

## 5. Under-Specified Queries & True Semantic Relevance

In qualitative manual inspection of 20 random paraphrase queries:
- For specific queries (e.g., `"How do we measure the turbulence of the atmosphere"`), dense retrieval found papers directly on point (`Accurate seeing measurements with MASS and DIMM`, `Improved detection of atmospheric turbulence with SLODAR`), while the ground-truth target paper was only tangentially related (`Tracking granules at the Sun's surface and reconstructing velocity fields`).
- In a corpus of 20,000 papers, single-target ground truth unfairly penalizes dense retrieval when a query matches multiple genuinely relevant preprints.
