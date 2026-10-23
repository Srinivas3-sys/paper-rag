# Diagnostics: Why Dense Retrieval Underperformed

This document details the investigation into why dense semantic retrieval (`all-MiniLM-L6-v2` + FAISS) underperformed BM25 on the initial multi-regime benchmark.

---

## 1. What Was Checked and Ruled Out (Indexing Integrity)

Before modifying any retrieval logic, we wrote a test suite in `scripts/diagnostics/` to rule out common vector indexing failure modes.

### Self-Retrieval Test (`scripts/diagnostics/self_retrieval.py`)
- We sampled 500 random documents from the 20,000-paper corpus and queried their exact indexed representations (`Title: ...\nAbstract: ...`) against the FAISS index.
- **Result**: 500 / 500 documents (100.0%) returned their own ID at Rank 1.
- When querying using only the raw abstract text (omitting the "Title: " prefix), 499 / 500 documents (99.8%) returned at Rank 1, with a mean rank of 1.00.

### Index Alignment & Metric Audit (`scripts/diagnostics/index_alignment.py`)
- The index uses `IndexFlatIP` (exact inner product) over L2-normalized float32 vectors, which is mathematically identical to cosine similarity.
- Reconstructing vectors from FAISS and comparing them against freshly encoded texts yields a mean cosine similarity of `1.000000` with a maximum element-wise difference of `1.19e-07` (floating-point epsilon).
- Docstore parquet rows and FAISS index indices map 1-to-1 without row shuffling or key offsets.

**Conclusion**: The low retrieval metrics are not caused by an indexing bug, normalization failure, or ID misalignment.

---

## 2. What Was Broken (Root Causes)

### Root Cause 1: Synthetic Paraphrase Prompt Collapse
When generating zero-leakage paraphrase queries, the prompt instructed Flan-T5-base:
> "Generate a concise search question describing the core problem below. You MUST NOT use these words: [title words]. Do not copy exact phrases."

When title words were strictly forbidden, Flan-T5 frequently defaulted to generic prompt boilerplate rather than synthesizing the abstract:
- `"What is the main problem"` (Dense Rank: 19,987 out of 20,000)
- `"What is the core problem of this paper"` (Dense Rank: 19,971 out of 20,000)
- `"What is the basic idea of the problem"` (Dense Rank: 18,548 out of 20,000)
- `"What is the best solution to this problem"` (Dense Rank: 16,256 out of 20,000)

Because these generic strings shared zero n-grams with the target paper's abstract, the leakage filter marked them as "valid zero-leakage paraphrases." Across all 102 paraphrase queries, the median number of corpus documents scoring above the target was 244.5. These non-informative queries severely distorted dense retrieval MRR.

### Root Cause 2: Under-Specified Queries Under Single-Target Ground Truth
Even for non-boilerplate queries, the prompt often produced broad conceptual queries. In a 20,000-paper corpus, multiple preprints cover the same subject:
- Query: `"How do we measure the turbulence of the atmosphere"`
- Dense Rank 1: `Accurate seeing measurements with MASS and DIMM` (Directly describes measuring atmospheric turbulence).
- Dense Rank 2: `Improved detection of atmospheric turbulence with SLODAR` (Directly describes atmospheric turbulence detection).
- Ground Truth Target (Rank 34): `Tracking granules at the Sun's surface and reconstructing velocity fields` (A solar physics paper that mentions atmospheric turbulence in passing).

Dense semantic search successfully retrieved the two most relevant papers in the entire corpus, but was penalized with a score of zero because the benchmark assumed only one target paper existed.

### Root Cause 3: Silent Ingestion Loss of arXiv Categories
In `src/data.py`, `clean_text(raw.get("categories", ""))` expected a string. However, HuggingFace's `gfissore/arxiv-abstracts-2021` dataset delivers categories as a Python list (`['hep-ph']`, `['cs.CV']`). Because `isinstance(text, str)` evaluated to `False`, category metadata was silently wiped to empty string `""` across all 20,000 records.

---

## 3. Properties of the Data and Models

### Token Truncation at 256 Tokens
`sentence-transformers/all-MiniLM-L6-v2` has a hard sequence limit of 256 tokens.
- Analysis across the 20,000 documents revealed that 4,771 documents (**23.86%**) exceed 256 tokens when combining title and abstract.
- For nearly a quarter of all papers, concluding sentences containing experimental results and domain conclusions are discarded during embedding.
- Migrating to a 512-token model (such as `BAAI/bge-small-en-v1.5`) reduces truncation to 0.38% (76 documents).

### Acronym and Out-of-Vocabulary Collisions
Academic search heavily features rare domain acronyms (e.g., ADMM, MIMO, QEC, H-infinity).
- WordPiece and BPE tokenizers split unfamiliar technical acronyms into generic subwords, diffusing dense semantic representations.
- BM25 operates directly on exact lexical tokens with inverted index lookups, giving it an inherent advantage on acronym and exact theorem queries (BM25 MRR = 0.667 vs Dense MRR = 0.178).
