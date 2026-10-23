"""Retrieval implementations for BM25, Dense FAISS, and Hybrid Reciprocal Rank Fusion."""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
import torch

from src.config import (
    BM25_INDEX_PATH,
    BM25S_INDEX_DIR,
    DOCSTORE_PATH,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    HYBRID_ALPHA,
    RERANKER_MODEL_NAME,
    RRF_K,
)
from src.index import tokenize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Abstract base class for paper retrieval."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> List[Dict[str, Any]]:
        """Retrieve top_k documents for a search query."""
        pass


class BM25Retriever(BaseRetriever):
    """BM25 sparse keyword-based retriever supporting bm25s and rank-bm25 backends."""

    def __init__(
        self,
        index_path: Path = BM25_INDEX_PATH,
        bm25s_dir: Path = BM25S_INDEX_DIR,
        docstore_path: Path = DOCSTORE_PATH,
        bm25: Optional[BM25Okapi] = None,
        bm25s_model: Optional[Any] = None,
        docstore: Optional[pd.DataFrame] = None,
        prefer_bm25s: bool = True,
    ):
        self.docstore = docstore
        self.bm25 = bm25
        self.bm25s_model = bm25s_model
        self.backend = "rank_bm25"

        if bm25s_model is not None and docstore is not None:
            self.bm25s_model = bm25s_model
            self.backend = "bm25s"
            self.doc_ids = docstore["id"].tolist()
        elif bm25 is not None and docstore is not None:
            self.bm25 = bm25
            self.backend = "rank_bm25"
            self.doc_ids = docstore["id"].tolist()
        else:
            if docstore is None:
                logger.info(f"Loading docstore from {docstore_path}...")
                self.docstore = pd.read_parquet(docstore_path)
            assert self.docstore is not None, "docstore cannot be None"
            self.doc_ids = self.docstore["id"].tolist()

            # Attempt bm25s loading first if preferred and available
            loaded_bm25s = False
            if prefer_bm25s and bm25s_dir.exists() and (bm25s_dir / "vocab.index.json").exists():
                try:
                    import bm25s

                    logger.info(f"Loading fast bm25s index from {bm25s_dir}...")
                    self.bm25s_model = bm25s.BM25.load(str(bm25s_dir), mmap=True)
                    self.backend = "bm25s"
                    loaded_bm25s = True
                except Exception as e:
                    logger.warning(f"Failed to load bm25s index ({e}), falling back to rank-bm25...")

            if not loaded_bm25s:
                logger.info(f"Loading rank-bm25 index from {index_path}...")
                with open(index_path, "rb") as f:
                    data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.backend = "rank_bm25"

        assert self.docstore is not None, "docstore cannot be None"
        self._id_to_idx = {pid: i for i, pid in enumerate(self.docstore["id"])}

    def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> List[Dict[str, Any]]:
        if not query.strip():
            return []

        assert self.docstore is not None, "docstore cannot be None"

        if self.backend == "bm25s" and self.bm25s_model is not None:
            try:
                import bm25s

                tokens = bm25s.tokenize([query], stopwords="en")
                doc_indices, scores = self.bm25s_model.retrieve(tokens, k=min(top_k, len(self.docstore)))
                top_indices = doc_indices[0]
                top_scores = scores[0]

                results = []
                for rank, (idx, score) in enumerate(zip(top_indices, top_scores)):
                    if idx < 0 or idx >= len(self.docstore):
                        continue
                    row = self.docstore.iloc[idx]
                    results.append(
                        {
                            "id": str(row["id"]),
                            "title": str(row["title"]),
                            "abstract": str(row["abstract"]),
                            "authors": str(row["authors"]),
                            "categories": str(row["categories"]),
                            "score": float(score),
                            "rank": rank + 1,
                            "retriever": "bm25",
                        }
                    )
                return results
            except Exception as e:
                logger.warning(f"bm25s retrieval error ({e}), falling back to rank-bm25...")

        tokens = tokenize_text(query)
        if not tokens:
            return []

        assert self.bm25 is not None, "BM25 model is uninitialized"
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            score = float(scores[idx])
            row = self.docstore.iloc[idx]
            results.append(
                {
                    "id": str(row["id"]),
                    "title": str(row["title"]),
                    "abstract": str(row["abstract"]),
                    "authors": str(row["authors"]),
                    "categories": str(row["categories"]),
                    "score": score,
                    "rank": rank + 1,
                    "retriever": "bm25",
                }
            )
        return results


class DenseRetriever(BaseRetriever):
    """Dense semantic retriever using SentenceTransformers and FAISS IndexFlatIP."""

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        docstore_path: Path = DOCSTORE_PATH,
        model_name: str = EMBEDDING_MODEL_NAME,
        index: Optional[faiss.Index] = None,
        docstore: Optional[pd.DataFrame] = None,
        model: Optional[SentenceTransformer] = None,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model or SentenceTransformer(model_name, device=device)

        if index is not None and docstore is not None:
            self.index = index
            self.docstore = docstore
        else:
            logger.info(f"Loading FAISS index from {index_path}...")
            self.index = faiss.read_index(str(index_path))

            logger.info(f"Loading docstore from {docstore_path}...")
            self.docstore = pd.read_parquet(docstore_path)

    def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> List[Dict[str, Any]]:
        if not query.strip():
            return []

        assert self.docstore is not None, "docstore cannot be None"

        # Generate query vector with unit normalization for cosine similarity
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        distances, indices = self.index.search(query_vec, top_k)
        top_indices = indices[0]
        top_scores = distances[0]

        results = []
        for rank, (idx, score) in enumerate(zip(top_indices, top_scores)):
            if idx < 0 or idx >= len(self.docstore):
                continue
            row = self.docstore.iloc[idx]
            results.append(
                {
                    "id": str(row["id"]),
                    "title": str(row["title"]),
                    "abstract": str(row["abstract"]),
                    "authors": str(row["authors"]),
                    "categories": str(row["categories"]),
                    "score": float(score),
                    "rank": rank + 1,
                    "retriever": "dense",
                }
            )
        return results


class CrossEncoderReranker:
    """Cross-Encoder reranker using a pretrained transformer (ms-marco-MiniLM-L-6-v2)."""

    def __init__(
        self,
        model_name: str = RERANKER_MODEL_NAME,
        device: Optional[str] = None,
        model: Optional[CrossEncoder] = None,
    ):
        if model is not None:
            self.model = model
        else:
            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Loading CrossEncoder reranker '{model_name}' on {dev}...")
            self.model = CrossEncoder(model_name, device=dev)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using cross-encoder relevance scores."""
        if not candidates or not query.strip():
            return candidates[:top_k]

        pairs = [(query, f"Title: {c.get('title', '')}\nAbstract: {c.get('abstract', '')}") for c in candidates]
        scores = self.model.predict(pairs)

        reranked = []
        for item, score in zip(candidates, scores):
            c_copy = dict(item)
            c_copy["score"] = float(score)
            c_copy["retriever"] = "cross-encoder"
            reranked.append(c_copy)

        reranked.sort(key=lambda x: x["score"], reverse=True)
        for rank, item in enumerate(reranked[:top_k]):
            item["rank"] = rank + 1
        return reranked[:top_k]


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining Dense semantic search and BM25 lexical search with RRF."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        alpha: float = HYBRID_ALPHA,
        rrf_k: int = RRF_K,
        reranker: Optional[CrossEncoderReranker] = None,
        parallel: bool = True,
    ):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.reranker = reranker
        self.parallel = parallel
        self.docstore = dense_retriever.docstore
        self._executor = ThreadPoolExecutor(max_workers=2) if parallel else None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 50,
        rerank: bool = False,
        rerank_candidates: int = 30,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Retrieve using Reciprocal Rank Fusion (RRF) over dense and sparse candidates."""
        if self.parallel and self._executor is not None:
            future_dense = self._executor.submit(self.dense_retriever.retrieve, query, candidate_k)
            future_bm25 = self._executor.submit(self.bm25_retriever.retrieve, query, candidate_k)
            dense_results = future_dense.result()
            bm25_results = future_bm25.result()
        else:
            dense_results = self.dense_retriever.retrieve(query, top_k=candidate_k)
            bm25_results = self.bm25_retriever.retrieve(query, top_k=candidate_k)

        # Build candidate map
        doc_map: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}

        # Add dense ranks
        for item in dense_results:
            pid = item["id"]
            rank = item["rank"]
            doc_map[pid] = item
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (self.alpha / (self.rrf_k + rank))

        # Add BM25 ranks
        for item in bm25_results:
            pid = item["id"]
            rank = item["rank"]
            if pid not in doc_map:
                doc_map[pid] = item
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + ((1.0 - self.alpha) / (self.rrf_k + rank))

        # Determine how many fused candidates to pull
        need_candidates = rerank_candidates if (rerank and self.reranker is not None) else top_k
        sorted_pids = sorted(rrf_scores.keys(), key=lambda p: rrf_scores[p], reverse=True)[:need_candidates]

        fused_results = []
        for rank, pid in enumerate(sorted_pids):
            item = dict(doc_map[pid])
            item["score"] = float(rrf_scores[pid])
            item["rank"] = rank + 1
            item["retriever"] = "hybrid"
            fused_results.append(item)

        if rerank and self.reranker is not None:
            return self.reranker.rerank(query, fused_results, top_k=top_k)

        return fused_results


def get_retriever(
    retriever_type: str = "hybrid",
    bm25_retriever: Optional[BM25Retriever] = None,
    dense_retriever: Optional[DenseRetriever] = None,
    reranker: Optional[CrossEncoderReranker] = None,
) -> BaseRetriever:
    """Factory function to instantiate or return configured retriever."""
    retriever_type = retriever_type.lower()
    if retriever_type == "bm25":
        return bm25_retriever or BM25Retriever()
    elif retriever_type == "dense":
        return dense_retriever or DenseRetriever()
    elif retriever_type == "hybrid":
        b_ret = bm25_retriever or BM25Retriever()
        d_ret = dense_retriever or DenseRetriever()
        return HybridRetriever(bm25_retriever=b_ret, dense_retriever=d_ret, reranker=reranker)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}. Expected 'bm25', 'dense', or 'hybrid'.")
