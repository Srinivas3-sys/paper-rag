"""Smoke test on a lightweight synthetic fixture index (no network or large downloads)."""

import tempfile
import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from src.generator import ArxivRAGGenerator
from src.index import tokenize_text
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever


class MockEmbedder:
    def __init__(self, dim: int = 8):
        self.dim = dim

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, **kwargs):
        vecs = []
        for i, text in enumerate(texts):
            v = np.zeros(self.dim, dtype=np.float32)
            v[i % self.dim] = 1.0
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)


class MockGeneratorModel:
    def __init__(self):
        class Config:
            is_encoder_decoder = True
        self.config = Config()

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        import torch
        return torch.tensor([[10, 20, 30]])


class MockGeneratorTokenizer:
    def __call__(self, text, **kwargs):
        import torch
        class Tokens(dict):
            def to(self, device):
                return self
        return Tokens({"input_ids": torch.tensor([[1, 2, 3]])})

    def encode(self, text, **kwargs):
        return list(range(max(1, len(text) // 4)))

    def decode(self, token_ids, **kwargs):
        return "Quantum error mitigation improves circuit fidelity [1]."


def test_fixture_pipeline_smoke():
    with tempfile.TemporaryDirectory():
        records = [
            {"id": "fix_01", "title": "Quantum Error Correction", "abstract": "Surface codes and stabilizers.", "authors": "Physicist A", "categories": "quant-ph", "combined_text": "Quantum Error Correction. Surface codes and stabilizers."},
            {"id": "fix_02", "title": "Transformer Attention", "abstract": "Multi-head attention in language models.", "authors": "Scientist B", "categories": "cs.CL", "combined_text": "Transformer Attention. Multi-head attention in language models."},
            {"id": "fix_03", "title": "Convex Optimization", "abstract": "Proximal gradient descent algorithms.", "authors": "Mathematician C", "categories": "math.OC", "combined_text": "Convex Optimization. Proximal gradient descent algorithms."},
        ]
        df = pd.DataFrame(records)

        # 1. Build fixture BM25
        tokenized_corpus = [tokenize_text(t) for t in df["combined_text"]]
        bm25_model = BM25Okapi(tokenized_corpus)
        bm25_retriever = BM25Retriever(bm25=bm25_model, docstore=df)

        # 2. Build fixture FAISS index
        embedder = MockEmbedder(dim=8)
        embeddings = embedder.encode(df["combined_text"].tolist(), normalize_embeddings=True)
        faiss_index = faiss.IndexFlatIP(8)
        faiss_index.add(embeddings)
        dense_retriever = DenseRetriever(index=faiss_index, docstore=df, model=embedder)

        # 3. Build fixture Hybrid
        hybrid_retriever = HybridRetriever(
            bm25_retriever=bm25_retriever,
            dense_retriever=dense_retriever,
            alpha=0.5,
            rrf_k=10,
            parallel=False,
        )

        results = hybrid_retriever.retrieve("quantum error", top_k=2)
        assert len(results) == 2
        assert results[0]["id"] == "fix_01"
        assert results[0]["retriever"] == "hybrid"

        # 4. Generate grounded answer
        generator = ArxivRAGGenerator(
            device="cpu",
            tokenizer=MockGeneratorTokenizer(),
            model=MockGeneratorModel(),
        )
        answer_result = generator.generate_answer(
            question="How does quantum error correction work?",
            retrieved_docs=results,
            temperature=0.0,
        )
        assert answer_result["abstained"] is False
        assert len(answer_result["citations"]) == 1
        assert answer_result["citations"][0]["paper_id"] == "fix_01"
