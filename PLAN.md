# Implementation Plan: paper-rag

A portfolio-grade AI/ML semantic search and retrieval-augmented generation (RAG) system over arXiv abstracts (~20k sample).

## Architecture & System Overview
- **Data Ingestion**: Hugging Face arXiv abstracts dataset (`gfissore/arxiv-abstracts-2021`), sampled to 20,000 papers across computer science, physics, mathematics, and related fields.
- **Retrievers**:
  - **BM25 Baseline**: Lexical keyword matching with custom tokenization and normalization via `rank-bm25`.
  - **Dense Retriever**: Semantic search with `sentence-transformers/all-MiniLM-L6-v2` and FAISS `IndexFlatIP` (cosine similarity with normalized L2 vectors).
  - **Hybrid Retriever**: Reciprocal Rank Fusion (RRF) and weighted linear combination of dense and sparse scores with rank cutoffs.
- **Evaluation**:
  - Standardized benchmark evaluating Recall@5, Recall@10, and Mean Reciprocal Rank (MRR).
  - Comparative metrics table saved to `results/metrics.md` and visualizations plotted to `results/metrics.png`.
- **Grounded Answer Generator**:
  - Local Seq2Seq LLM (`google/flan-t5-base`) utilizing top-k retrieved contexts with structured citations (`[1]`, `[2]`, arXiv IDs).
- **FastAPI Service**:
  - `/health`: Service health, loaded document count, device status.
  - `/search`: Multi-mode retrieval (`bm25`, `dense`, `hybrid`).
  - `/ask`: Retrieval-augmented Q&A with inline citations.
- **Engineering Quality**:
  - Unit tests with `pytest`.
  - Modular structure (`src/`, `tests/`, `notebooks/`, `results/`).
  - Reproducible pipeline via `Makefile`, `Dockerfile`, and GitHub Actions CI (`.github/workflows/ci.yml`).

---

## Checklist

- [x] **Milestone 1: Project Scaffolding & Verification Loop**
  - [x] Create `.gitignore`, `requirements.txt`
  - [x] Create `Makefile`, `PLAN.md`, `LOOP_LOG.md`
  - [x] Implement `verify.sh` self-correction test harness
  - [x] Initialize git repo, commit scaffolding

- [ ] **Milestone 2: Data Ingestion & Preprocessing**
  - [ ] Implement `src/config.py` for centralized paths and hyperparams
  - [ ] Implement `src/data.py` (download ~20k arXiv abstracts, clean, preprocess, create evaluation queries)
  - [ ] Implement `tests/test_data.py`
  - [ ] Run `./verify.sh` and ensure passing
  - [ ] Commit milestone (`feat: ...`)

- [ ] **Milestone 3: Indexing & Retrieval Engine**
  - [ ] Implement `src/index.py` (build and persist BM25 and FAISS indices)
  - [ ] Implement `src/retrieval.py` (`BM25Retriever`, `DenseRetriever`, `HybridRetriever`)
  - [ ] Implement `tests/test_retrieval.py`
  - [ ] Run `./verify.sh` and ensure passing
  - [ ] Commit milestone (`feat: ...`)

- [ ] **Milestone 4: Evaluation Benchmark & Metrics**
  - [ ] Implement `src/evaluate.py` (Recall@5, Recall@10, MRR calculation)
  - [ ] Run evaluation on real indexed dataset
  - [ ] Generate `results/metrics.md` and `results/metrics.png`
  - [ ] Run `./verify.sh` and verify metrics are real and non-empty
  - [ ] Commit milestone (`feat: ...`)

- [ ] **Milestone 5: Answer Generation & Citations**
  - [ ] Implement `src/generator.py` with `google/flan-t5-base`
  - [ ] Context formatting with `[1]`, `[2]` citation markers and source metadata
  - [ ] Implement `tests/test_generator.py`
  - [ ] Run `./verify.sh` and ensure passing
  - [ ] Commit milestone (`feat: ...`)

- [ ] **Milestone 6: FastAPI Service**
  - [ ] Implement `src/api.py` with `/health`, `/search`, `/ask`
  - [ ] Implement `tests/test_api.py`
  - [ ] Run `./verify.sh` and ensure passing
  - [ ] Commit milestone (`feat: ...`)

- [ ] **Milestone 7: EDA Notebook, Dockerfile, CI, and Documentation**
  - [ ] Create `notebooks/01_eda_and_retrieval_analysis.ipynb`
  - [ ] Create `Dockerfile`
  - [ ] Create `.github/workflows/ci.yml`
  - [ ] Create portfolio-grade `README.md` with Mermaid diagram, real metric results, and architecture
  - [ ] Run `./verify.sh` until output: `ALL CHECKS PASSED`
  - [ ] Commit milestone (`docs: ...`)

- [ ] **Milestone 8: Final Review & Git Push**
  - [ ] Check git status and clean working tree
  - [ ] Verify git remote origin and push
