# paper-rag: Semantic Search and Question Answering over arXiv Literature

[![CI Pipeline](https://github.com/Srinivas3-sys/paper-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Srinivas3-sys/paper-rag/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162-009688.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

paper-rag is a semantic search and question answering pipeline evaluated over 20,000 arXiv STEM abstracts. It benchmarks BM25 lexical search (`bm25s`), dense semantic retrieval (`all-MiniLM-L6-v2` with FAISS `IndexFlatIP`), Reciprocal Rank Fusion (RRF), CrossEncoder reranking (`ms-marco-MiniLM-L-6-v2`), and local answer generation (`google/flan-t5-base`).

---

## Headline Results

Evaluated on a held-out test split of 300 queries (100 Lexical, 100 Paraphrase, 100 Acronym; zero title n-gram overlap on paraphrases) with 1,000-resample bootstrap 95% confidence intervals:

| Retriever Architecture | Recall@1 [95% CI] | Recall@5 [95% CI] | Recall@10 [95% CI] | Recall@20 [95% CI] | MRR [95% CI] | nDCG@10 [95% CI] | Latency (Mean, p95) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BM25 Baseline** | 0.563 [0.507, 0.620] | 0.700 [0.647, 0.750] | 0.747 [0.700, 0.793] | 0.777 [0.727, 0.823] | 0.624 [0.574, 0.675] | 0.651 [0.602, 0.700] | 3.7 ms (p95: 4.1) |
| **Dense (MiniLM+FAISS)** | 0.407 [0.350, 0.463] | 0.510 [0.450, 0.567] | 0.543 [0.480, 0.597] | 0.583 [0.527, 0.633] | 0.454 [0.401, 0.506] | 0.472 [0.418, 0.525] | 9.6 ms (p95: 12.3) |
| **Hybrid (RRF Optimal)** | 0.547 [0.490, 0.600] | 0.723 [0.673, 0.773] | 0.757 [0.707, 0.807] | 0.787 [0.737, 0.830] | 0.623 [0.572, 0.672] | 0.653 [0.604, 0.700] | 19.7 ms (p95: 33.4) |
| **Hybrid + CrossEncoder Rerank** | 0.600 [0.543, 0.657] | 0.737 [0.687, 0.783] | 0.777 [0.730, 0.823] | 0.807 [0.760, 0.850] | 0.661 [0.612, 0.708] | 0.687 [0.640, 0.732] | 44.1 ms (p95: 63.4) |

### Key Findings on Retriever Selection

- **BM25 is the recommended production default**: BM25 alone achieves 0.624 MRR at 3.7 ms latency, which is statistically indistinguishable from Hybrid RRF (0.623 MRR at 19.7 ms; paired Wilcoxon signed-rank test: W = 8721.0, p = 0.9221). Because BM25 requires no GPU forward pass, no embedding storage, and evaluates 5.3x faster while beating Hybrid on exact acronyms (0.660 vs 0.629, p = 0.0049), BM25 is the recommended default for production serving.
- **Where Hybrid helps**: Hybrid RRF is only worth its latency cost when queries are predominantly paraphrased natural language without keyword matches. On the 100-query Paraphrase split, Hybrid significantly outperforms BM25 (Strict MRR 0.250 vs 0.222, W = 791.0, p = 0.0054; Recall@10 0.420 vs 0.380).
- **Why Dense underperformed on strict metrics**: Diagnostics showed 100.0% Rank-1 recall on self-retrieval (500/500 documents), proving the FAISS index was not corrupted. Dense underperformed because:
  1. 23.86% of corpus abstracts exceed MiniLM's 256-token context limit, discarding trailing technical details.
  2. Dense retrieval clusters topically similar arXiv preprints. Under lenient multi-relevant evaluation (crediting retrieved documents sharing the target's arXiv category with cosine similarity >= 0.30), Dense Paraphrase MRR jumps to 0.526 vs BM25 0.398 (+32.2% relative gain for dense).
- **CrossEncoder reranking trade-off**: Reranking top-30 hybrid candidates with `ms-marco-MiniLM-L-6-v2` increases MRR to 0.661 (+0.039) and Recall@1 to 0.600 (+5.3% absolute), but adds 25.1 ms per query on GPU (and >150 ms on CPU). It is exposed via optional flag `rerank=True`.

### Regime-by-Regime Breakdown (300 Held-Out Test Queries, n=100 per regime)

| Query Regime | Retriever Architecture | Strict Recall@10 [95% CI] | Strict MRR [95% CI] | Lenient Recall@10 [95% CI] | Lenient MRR [95% CI] | Paired Test vs BM25 (p-value) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Lexical (n=100)** | BM25 Baseline | 1.000 [1.000, 1.000] | 0.990 [0.975, 1.000] | 1.000 [1.000, 1.000] | 0.995 [0.985, 1.000] | Baseline |
| | Dense (MiniLM+FAISS) | 0.980 [0.950, 1.000] | 0.915 [0.864, 0.958] | 1.000 [1.000, 1.000] | 0.983 [0.960, 1.000] | W = 105.0, p = 0.0009 |
| | Hybrid (RRF Optimal) | 1.000 [1.000, 1.000] | 0.985 [0.965, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | W = 45.0, p = 0.5637 (tied) |
| **Paraphrase (n=100)** | BM25 Baseline | 0.380 [0.290, 0.480] | 0.222 [0.153, 0.292] | 0.650 [0.560, 0.740] | 0.398 [0.315, 0.481] | Baseline |
| | Dense (MiniLM+FAISS) | 0.340 [0.250, 0.430] | 0.216 [0.151, 0.289] | 0.760 [0.670, 0.840] | 0.526 [0.440, 0.608] | W = 1022.0, p = 0.8123 |
| | Hybrid (RRF Optimal) | 0.420 [0.320, 0.520] | 0.250 [0.181, 0.317] | 0.710 [0.620, 0.800] | 0.471 [0.385, 0.555] | **W = 791.0, p = 0.0054** |
| **Acronym (n=100)** | BM25 Baseline | 0.860 [0.790, 0.920] | 0.660 [0.584, 0.739] | 0.970 [0.930, 1.000] | 0.805 [0.745, 0.867] | Baseline |
| | Dense (MiniLM+FAISS) | 0.310 [0.220, 0.400] | 0.231 [0.160, 0.309] | 0.690 [0.600, 0.780] | 0.516 [0.428, 0.605] | W = 312.0, p < 0.0001 |
| | Hybrid (RRF Optimal) | 0.850 [0.780, 0.910] | 0.629 [0.548, 0.706] | 0.970 [0.930, 1.000] | 0.800 [0.738, 0.862] | **W = 852.5, p = 0.0049** |

---

## What I Found

1. **Title queries created artificial saturation**: An initial prototype generated queries by cleaning paper titles, yielding Recall@10 = 1.000 and BM25 MRR = 0.978. When tested on paraphrase queries without keyword overlap, BM25 MRR dropped from 0.978 to 0.222, and Dense MRR dropped to 0.216. Benchmarks must test disjoint lexical, paraphrase, and acronym query distributions.
2. **MiniLM truncates 23.86% of arXiv abstracts**: `all-MiniLM-L6-v2` has a 25Context limit of 256 tokens. Profiled across all 20,000 papers, 4,771 papers (23.86%) exceed 256 tokens and have trailing text truncated. In comparison, `BAAI/bge-small-en-v1.5` (512 tokens) reduced truncation to 0.38% (76 papers), but indexing time increased by 7.6x (~18s vs ~137s on an RTX 3050).
3. **Threadpool parallelism does not reduce latency for millisecond retrievers**: Running `bm25s` and FAISS concurrently in Python using `ThreadPoolExecutor(max_workers=2)` increased median latency from 14.8 ms (sequential) to 16.2 ms (parallel). Because `bm25s` completes in 1.0 ms and FAISS in 2.5 ms, thread context switching and GIL overhead exceed parallel gains.
4. **`bm25s` is 21.8x faster than `rank-bm25`**: Replacing pure-Python `rank-bm25` with `bm25s` (using memory-mapped CSC sparse matrices) reduced sparse search latency from 22.4 ms to 1.03 ms while reducing disk footprint from 16 MB to 12 MB.
5. **DEV-tuned abstain gate refused 91.7% of unanswerables**: Tuning an evidence threshold ($\tau^* = 7.9662$) on DEV BM25 scores achieved 0.958 balanced accuracy. On the held-out test split (12 unanswerable off-topic or fictional queries), the gate refused 91.7% [95% CI: 75.0%, 100.0%] before invoking the LLM, preventing hallucinations with zero generation cost.
6. **Automated NLI agreed with human labels on 35.0% of noun phrases**: Auditing 20 factual claims synthesized by Flan-T5-base against `cross-encoder/nli-deberta-v3-base` predictions showed 7/20 (35.0%) agreement. Because Flan-T5 frequently generates short noun phrases without verbs, DeBERTa predicts `neutral` due to strict sentence-level literalism, whereas human readers identify correct factual grounding.

---

## How to Run and Reproduce

### Environment Setup
- Python 3.11 or 3.14 on Linux x86_64
- Hardware tested: Intel Core i7-13620H (16 threads), 16 GB RAM, NVIDIA GeForce RTX 3050 6GB (CUDA 13.2)

```bash
# 1. Clone repository
git clone https://github.com/Srinivas3-sys/paper-rag.git
cd paper-rag

# 2. Install pinned dependencies
make install

# 3. Ingest data and build search indices (~45 seconds)
make data
make index

# 4. Run retrieval, generation, and latency benchmarks (~120 seconds)
make evaluate

# 5. Run test suite, linting, typechecking, and README number verification (~45 seconds)
make check
```

Expected execution times:
- `make data`: ~15s (downloads 20,000 arXiv records to `data/arxiv_20000.parquet`)
- `make index`: ~20s (builds `indices/bm25s` in ~2s and `indices/dense_faiss.index` in ~18s)
- `make evaluate`: ~120s (evaluates 450 retrieval queries, 60 QA items, and latency profiling)
- `make check`: ~45s (runs Ruff, Mypy, 21 unit tests, and validates README numbers match `results/`)

---

## API Usage

### Start FastAPI Server

```bash
make api
# Serves on http://0.0.0.0:8000
```

### Search Endpoint

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "quantum error correction surface codes",
    "retriever_type": "bm25",
    "top_k": 3,
    "rerank": false
  }'
```

### Question Answering Endpoint

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What threshold is reported for topological surface codes?",
    "retriever_type": "bm25",
    "top_k": 3,
    "temperature": 0.0,
    "rerank": false
  }'
```

---

## Real Limitations

1. **20,000 document corpus size**: The dataset contains 20,000 sequential STEM preprints from `gfissore/arxiv-abstracts-2021` spanning April to May 2007 (`0704.0001` to `0705.4678`). It is not a uniform random sample across modern arXiv fields.
2. **Abstracts only**: The index indexes titles and abstracts. Claims requiring full text, tables, proofs, or experimental parameter appendices cannot be resolved.
3. **Single-target ground truth**: Test queries are linked to a single target document. In domains with multiple papers on identical topics, retrievers that return topical peers are penalized under strict scoring.
4. **Generated paraphrase queries**: Paraphrase queries were generated using local models under negative lexical constraints. While inspected to eliminate degenerate templates, synthetic questions may not mirror natural user query ambiguity.
5. **Small generator capabilities**: `google/flan-t5-base` (248M parameters) follows negative constraints and abstain rules well, but produces brief responses and rarely generates structured bracket citations without few-shot guidance.
6. **Automated NLI brittleness**: `cross-encoder/nli-deberta-v3-base` was trained on complete premise-hypothesis sentences. When applied to short noun phrases or mathematical notations, its predictions diverge from human contextual judgments.
7. **Stateless single-turn API**: The service does not support conversation memory or multi-turn conversational search.

---

## Project Structure

```
paper-rag/
├── data/                       # Benchmark query JSON files (raw parquet gitignored)
├── docs/
│   ├── decisions.md            # Engineering notes, empirical measurements, and failure analyses
│   └── diagnostics.md          # Phase 1 diagnostic test results and failure mode analysis
├── indices/                    # FAISS IndexFlatIP and bm25s sparse indices (gitignored)
├── notebooks/
│   └── 01_eda_and_retrieval_analysis.ipynb # Dataset distributions and retrieval inspection
├── results/
│   ├── metrics.md              # 450-query retrieval benchmark with bootstrap CIs
│   ├── metrics.png             # Multi-panel visualization with per-regime error bars
│   ├── generation_metrics.md   # Abstain threshold calibration, QA eval, and 20-sample NLI audit
│   ├── retrieval_benchmarks.md # Latency profiling and sequence length truncation analysis
│   ├── encoder_comparison_dev.md # Dense encoder comparison on DEV split (MiniLM, BGE, E5)
│   └── rrf_tuning_dev.md       # Weighted RRF grid search on DEV split
├── scripts/
│   ├── check_readme_numbers.py # Validates README numbers against results/ in CI
│   ├── compare_dense_encoders.py # Evaluates dense encoders on DEV split
│   ├── evaluate_reranker.py    # Evaluates CrossEncoder reranker on TEST split
│   ├── generate_full_nli_audit.py # Audits 20 NLI judgments against human annotations
│   ├── restore_categories.py   # Restores arXiv category metadata from HuggingFace dataset
│   ├── tune_rrf_dev.py         # Grid search over RRF k and weights on DEV split
│   └── diagnostics/            # Phase 1 diagnostic test suite
├── src/
│   ├── api.py                  # FastAPI application with Pydantic validation
│   ├── benchmark_retrieval.py  # Latency profiling and sequence length truncation script
│   ├── build_benchmark.py      # Multi-regime benchmark generation with leakage audit
│   ├── config.py               # Centralized paths and hyperparameters
│   ├── data.py                 # Ingestion, cleaning, and category parsing
│   ├── eval_generation.py      # Abstain gate tuning and NLI faithfulness evaluation
│   ├── evaluate_benchmark.py   # Strict/lenient retrieval evaluation with bootstrap CIs
│   ├── generator.py            # Prompt assembly, context budgeting, and answer generation
│   ├── index.py                # bm25s and FAISS IndexFlatIP index builders
│   └── retrieval.py            # BM25, Dense, Hybrid RRF, and CrossEncoder retrievers
├── tests/
│   ├── test_api.py             # Endpoint validation and timeout tests
│   ├── test_data.py            # Preprocessing and category parsing tests
│   ├── test_generator.py       # Context budgeting, abstention, and temperature tests
│   ├── test_retrieval.py       # Retrieval components and hand-computed RRF tests
│   └── test_smoke_fixture.py   # Synthetic index smoke test for CI
├── Dockerfile                  # Container definition with mounted index volume instructions
├── Makefile                    # Automation recipes (install, check, test, api, docker)
├── pyproject.toml              # Ruff and package build configuration
└── requirements.txt            # Pinned dependency versions
```
