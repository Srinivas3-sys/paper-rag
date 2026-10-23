# Retrieval Performance & Engineering Benchmark

Comprehensive empirical evaluation of sparse (BM25s), dense (MiniLM FAISS), hybrid Reciprocal Rank Fusion, parallelization, cross-encoder reranking, and sequence length truncation.

## Hardware Specification
- **CPU**: 13th Gen Intel(R) Core(TM) i7-13620H (10 physical cores / 16 threads, up to 4.90 GHz)
- **RAM**: 16 GB DDR5
- **GPU**: NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 13.2)
- **OS**: Linux x86_64

---

## 1. Retrieval Accuracy: Hybrid RRF vs RRF + Cross-Encoder Reranker

Evaluated on the held-out TEST benchmark ($N=222$):
- **Base Hybrid**: RRF with optimal parameters ($k=10, \alpha=0.3$), top-50 sparse + dense candidates.
- **Reranked Hybrid**: Cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranking top-30 fused candidates.

| Method | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid RRF (Base)** | 0.545 | 0.703 | 0.739 | 0.784 | 0.617 | 0.645 |
| **Hybrid + Cross-Encoder Rerank** | 0.599 | 0.716 | 0.770 | 0.797 | 0.656 | 0.682 |
| **Absolute Delta** | +0.054 | +0.014 | +0.032 | +0.014 | +0.038 | +0.037 |

---

## 2. Granular Latency Breakdown (Warm runs, N=100)

Timed across 100 queries after 50 warmup iterations:

| Stage | Mean (ms) | p50 / Median (ms) | p95 (ms) | p99 (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **Dense Query Encoding** (`all-MiniLM-L6-v2` GPU) | 3.95 | 3.82 | 4.72 | 6.05 |
| **FAISS Dense Search** (`IndexFlatIP`) | 2.11 | 2.06 | 2.40 | 2.52 |
| **Sparse BM25 Search** (`bm25s` mmap) | 5.09 | 4.92 | 6.26 | 7.91 |
| **Hybrid Sequential** (BM25 then Dense) | 14.51 | 14.07 | 17.66 | 20.61 |
| **Hybrid Parallel** (`ThreadPoolExecutor(2)`) | 15.83 | 15.32 | 20.33 | 23.35 |
| **Cross-Encoder Reranking** (30 candidates GPU) | 23.01 | 21.49 | 34.16 | 41.09 |
| **Full Pipeline** (Parallel Hybrid + Reranker) | 38.84 | 37.26 | 53.38 | 57.37 |

---

## 3. BM25 Implementation Benchmark: `bm25s` vs `rank-bm25`

- **`rank-bm25`**: Pure Python inverted index. Serialized pickle size: 16 MB. Query latency: ~22.4 ms.
- **`bm25s`**: Memory-mapped CSC sparse matrix. Index size: 12 MB. Query latency: ~1.0 ms.
- **Latency Improvement**: **21.8x faster** lexical retrieval with identical top-k candidates.

---

## 4. MiniLM Sequence Length Truncation Analysis

- **Model Context Window**: 256 tokens (`sentence-transformers/all-MiniLM-L6-v2`)
- **Corpus Texts Evaluated**: 20,000 arXiv records (`Title: ...\nAbstract: ...`)
- **Token Length Statistics**:
  - Min: 24 tokens
  - Median (p50): 178.0 tokens
  - Mean: 197.3 tokens
  - p90: 337.0 tokens
  - p95: 386.0 tokens
  - Max: 666 tokens
- **Truncation Rate (> 256 tokens)**: **23.86%** (4771 of 20,000 papers).
- **Comparison to 512-Token Context (`BAAI/bge-small-en-v1.5`)**:
  - Truncation Rate (> 512 tokens): **0.38%** (76 of 20,000 papers).
  - Indexing Throughput: MiniLM encodes 20k abstracts in ~18s; BGE takes ~137s (~7.6x longer forward pass).

---

## 5. Corpus Sampling Methodology

The 20,000 corpus records in `data/arxiv_20000.parquet` were drawn chronologically from `gfissore/arxiv-abstracts-2021` starting with id `0704.0001` (April 2007) through `0705.4678` (May 2007).
- It is **not** a uniformly random sample across all arXiv disciplines or decades.
- This sample captures real STEM preprints across condensed matter, high energy physics, math, and computer science during the first two months of the modern arXiv ID format.
