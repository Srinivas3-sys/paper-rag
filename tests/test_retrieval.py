"""Unit tests for retrieval implementations (BM25, Dense, Hybrid)."""

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from src.index import tokenize_text
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever


def test_tokenize_text():
    tokens = tokenize_text("Deep Learning in 2024: A Novel Method for Transformers.")
    assert "deep" in tokens
    assert "learning" in tokens
    assert "transformers" in tokens
    # Stopwords like 'in', 'a', 'for' should be stripped
    assert "in" not in tokens
    assert "a" not in tokens
    assert "for" not in tokens


class MockSentenceTransformer:
    def __init__(self, dim=4):
        self.dim = dim

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, **kwargs):
        vecs = []
        for t in texts:
            # Deterministic hash to vector
            val = float(len(t) % 10)
            vec = np.array([val, 1.0, 0.5, 0.2], dtype=np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                vec = vec / (norm + 1e-9)
            vecs.append(vec)
        return np.array(vecs, dtype=np.float32)


def test_retrieval_components():
    # Setup miniature test corpus
    records = [
        {
            "id": "p1",
            "title": "Quantum Computing Algorithms",
            "abstract": "Quantum circuits and algorithmic speedup in computation.",
            "authors": "Physicist A",
            "categories": "quant-ph",
            "combined_text": "Title: Quantum Computing Algorithms\nAbstract: Quantum circuits and algorithmic speedup in computation.",
        },
        {
            "id": "p2",
            "title": "Deep Neural Networks for Vision",
            "abstract": "Convolutional neural networks and vision transformers for image classification.",
            "authors": "Computer Scientist B",
            "categories": "cs.CV",
            "combined_text": "Title: Deep Neural Networks for Vision\nAbstract: Convolutional neural networks and vision transformers.",
        },
        {
            "id": "p3",
            "title": "Reinforcement Learning with Transformers",
            "abstract": "Decision transformers in offline reinforcement learning settings.",
            "authors": "AI Researcher C",
            "categories": "cs.LG",
            "combined_text": "Title: Reinforcement Learning with Transformers\nAbstract: Decision transformers in offline reinforcement learning.",
        },
    ]
    df = pd.DataFrame(records)

    # 1. BM25 test
    tokenized_corpus = [tokenize_text(text) for text in df["combined_text"]]
    bm25_model = BM25Okapi(tokenized_corpus)
    bm25_retriever = BM25Retriever(bm25=bm25_model, docstore=df)

    res_bm25 = bm25_retriever.retrieve("quantum circuits", top_k=2)
    assert len(res_bm25) > 0
    assert res_bm25[0]["id"] == "p1"
    assert "score" in res_bm25[0]
    assert res_bm25[0]["retriever"] == "bm25"

    # 2. Dense test with mock model & FAISS
    mock_model = MockSentenceTransformer(dim=4)
    embeddings = mock_model.encode(df["combined_text"].tolist(), normalize_embeddings=True)
    index = faiss.IndexFlatIP(4)
    index.add(embeddings)

    dense_retriever = DenseRetriever(index=index, docstore=df, model=mock_model)
    res_dense = dense_retriever.retrieve("quantum computing", top_k=2)
    assert len(res_dense) == 2
    assert res_dense[0]["retriever"] == "dense"

    # 3. Hybrid test
    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        alpha=0.6,
    )
    res_hybrid = hybrid_retriever.retrieve("quantum computation", top_k=2)
    assert len(res_hybrid) <= 2
    assert res_hybrid[0]["retriever"] == "hybrid"
    assert res_hybrid[0]["score"] > 0


class MockCrossEncoder:
    def predict(self, pairs):
        # Assign higher score if 'Quantum' is in doc text
        scores = []
        for q, doc in pairs:
            if "Quantum" in doc:
                scores.append(3.5)
            elif "Decision" in doc:
                scores.append(1.2)
            else:
                scores.append(-0.5)
        return scores


def test_cross_encoder_reranker_and_hybrid():
    records = [
        {"id": "p1", "title": "Quantum Computing", "abstract": "Quantum physics and algorithms.", "authors": "A", "categories": "quant-ph", "combined_text": "Quantum physics and algorithms."},
        {"id": "p2", "title": "Computer Vision", "abstract": "Image recognition with CNNs.", "authors": "B", "categories": "cs.CV", "combined_text": "Image recognition with CNNs."},
        {"id": "p3", "title": "Decision Transformers", "abstract": "Decision transformers in RL.", "authors": "C", "categories": "cs.LG", "combined_text": "Decision transformers in RL."},
    ]
    df = pd.DataFrame(records)

    tokenized_corpus = [tokenize_text(text) for text in df["combined_text"]]
    bm25_model = BM25Okapi(tokenized_corpus)
    bm25_ret = BM25Retriever(bm25=bm25_model, docstore=df)

    mock_model = MockSentenceTransformer(dim=4)
    embeddings = mock_model.encode(df["combined_text"].tolist(), normalize_embeddings=True)
    index = faiss.IndexFlatIP(4)
    index.add(embeddings)
    dense_ret = DenseRetriever(index=index, docstore=df, model=mock_model)

    from src.retrieval import CrossEncoderReranker

    mock_ce = MockCrossEncoder()
    reranker = CrossEncoderReranker(model=mock_ce)

    hybrid = HybridRetriever(
        bm25_retriever=bm25_ret,
        dense_retriever=dense_ret,
        reranker=reranker,
        parallel=True,
    )

    # Test parallel vs sequential determinism
    res_parallel = hybrid.retrieve("quantum", top_k=3, rerank=False)
    hybrid_seq = HybridRetriever(
        bm25_retriever=bm25_ret,
        dense_retriever=dense_ret,
        reranker=reranker,
        parallel=False,
    )
    res_seq = hybrid_seq.retrieve("quantum", top_k=3, rerank=False)

    assert [r["id"] for r in res_parallel] == [r["id"] for r in res_seq]
    assert [r["score"] for r in res_parallel] == [r["score"] for r in res_seq]

    # Test reranking enabled
    res_reranked = hybrid.retrieve("quantum", top_k=2, rerank=True, rerank_candidates=3)
    assert len(res_reranked) == 2
    assert res_reranked[0]["id"] == "p1"
    assert res_reranked[0]["retriever"] == "cross-encoder"
    assert res_reranked[0]["score"] == 3.5

