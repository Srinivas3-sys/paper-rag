"""Configuration settings for paper-rag semantic search and RAG system."""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDICES_DIR = BASE_DIR / "indices"
RESULTS_DIR = BASE_DIR / "results"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

DATA_DIR.mkdir(parents=True, exist_ok=True)
INDICES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_NAME = os.getenv("ARXIV_DATASET_NAME", "gfissore/arxiv-abstracts-2021")
DATASET_SAMPLE_SIZE = int(os.getenv("ARXIV_SAMPLE_SIZE", "20000"))
RAW_DATA_PATH = DATA_DIR / f"arxiv_{DATASET_SAMPLE_SIZE}.parquet"
EVAL_DATA_PATH = DATA_DIR / "eval_queries.json"
BENCHMARK_PATH = DATA_DIR / "eval_benchmark_450.json"
LEAKAGE_REPORT_PATH = DATA_DIR / "leakage_report.json"

# Retrieval & Embedding configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
FAISS_INDEX_PATH = INDICES_DIR / "dense_faiss.index"
DOCSTORE_PATH = INDICES_DIR / "docstore.parquet"
BM25_INDEX_PATH = INDICES_DIR / "bm25_index.pkl"
BM25S_INDEX_DIR = INDICES_DIR / "bm25s"
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

DEFAULT_TOP_K = 5
HYBRID_ALPHA = 0.3  # Tuned on DEV split: w_bm25=3.0, w_dense=1.0 -> alpha = 1 / (1 + 3) = 0.25 ~ 0.3
RRF_K = 10          # Tuned on DEV split (k=10 achieved highest dev MRR 0.6451)

# Generation configuration
GENERATOR_MODEL_NAME = os.getenv("GENERATOR_MODEL_NAME", "google/flan-t5-base")
MAX_GENERATION_LENGTH = 256
MAX_INPUT_CONTEXT_LENGTH = 1024

# Service configuration
API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8000"))
