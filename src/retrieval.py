"""Retrieval implementations for BM25, Dense FAISS, and Hybrid Reciprocal Rank Fusion."""

from abc import ABC, abstractmethod
import logging
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import torch

from src.config import (
    BM25_INDEX_PATH,
    DOCSTORE_PATH,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    HYBRID_ALPHA,
    RRF_K,
)
from src.index import tokenize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Abstract base class for paper retrieval."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top_k documents for a search query."""
        pass


class BM25Retriever(BaseRetriever):
    """BM25 sparse keyword-based retriever."""

    def __init__(
        self,
        index_path: Path = BM25_INDEX_PATH,
        docstore_path: Path = DOCSTORE_PATH,
        bm25: Optional[BM25Okapi] = None,
        docstore: Optional[pd.DataFrame] = None,
    ):
        if bm25 is not None and docstore is not None:
            self.bm25 = bm25
            self.docstore = docstore
            self.doc_ids = docstore["id"].tolist()
        else:
            logger.info(f"Loading BM25 index from {index_path}...")
            with open(index_path, "rb") as f:
                data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.doc_ids = data["doc_ids"]

            logger.info(f"Loading docstore from {docstore_path}...")
            self.docstore = pd.read_parquet(docstore_path)

        self._id_to_idx = {pid: i for i, pid in enumerate(self.docstore["id"])}

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        tokens = tokenize_text(query)
        if not tokens:
            return []

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

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query.strip():
            return []

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


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining Dense semantic search and BM25 lexical search with RRF."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        alpha: float = HYBRID_ALPHA,
        rrf_k: int = RRF_K,
    ):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.docstore = dense_retriever.docstore

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieve using Reciprocal Rank Fusion (RRF) over dense and sparse candidates."""
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

        # Sort by final fused RRF score
        sorted_pids = sorted(rrf_scores.keys(), key=lambda p: rrf_scores[p], reverse=True)[:top_k]

        fused_results = []
        for rank, pid in enumerate(sorted_pids):
            item = dict(doc_map[pid])
            item["score"] = float(rrf_scores[pid])
            item["rank"] = rank + 1
            item["retriever"] = "hybrid"
            fused_results.append(item)

        return fused_results


def get_retriever(
    retriever_type: str = "hybrid",
    bm25_retriever: Optional[BM25Retriever] = None,
    dense_retriever: Optional[DenseRetriever] = None,
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
        return HybridRetriever(bm25_retriever=b_ret, dense_retriever=d_ret)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}. Expected 'bm25', 'dense', or 'hybrid'.")
