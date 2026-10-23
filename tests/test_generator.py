"""Unit tests for answer generation and citation parsing."""

import torch

from src.generator import ArxivRAGGenerator


def test_format_context_prompt():
    docs = [
        {
            "id": "2104.0001",
            "title": "Quantum Circuit Optimization",
            "abstract": "We present an efficient method for quantum gate optimization.",
            "authors": "Physicist X",
            "categories": "quant-ph",
        },
        {
            "id": "2104.0002",
            "title": "Graph Representation Learning",
            "abstract": "A review of message-passing neural networks on graphs.",
            "authors": "Scientist Y",
            "categories": "cs.LG",
        },
    ]

    # Initialize generator with dummy/None model for pure prompt & citation testing
    gen = ArxivRAGGenerator(device="cpu", tokenizer=object(), model=object())
    prompt = gen.format_context_prompt("How to optimize quantum gates?", docs)

    assert "[1] (arXiv:2104.0001) Quantum Circuit Optimization" in prompt
    assert "[2] (arXiv:2104.0002) Graph Representation Learning" in prompt
    assert "Question: How to optimize quantum gates?" in prompt


def test_extract_citations():
    docs = [
        {
            "id": "doc_1",
            "title": "Attention Mechanisms",
            "authors": "A. Vaswani",
            "categories": "cs.CL",
        },
        {
            "id": "doc_2",
            "title": "BERT Language Representations",
            "authors": "J. Devlin",
            "categories": "cs.CL",
        },
    ]

    gen = ArxivRAGGenerator(device="cpu", tokenizer=object(), model=object())

    # Case 1: Answer with explicit citation [1]
    ans1 = "Transformers rely on multi-head attention [1] for self-attention."
    cits1 = gen.extract_citations(ans1, docs)
    assert len(cits1) == 1
    assert cits1[0]["citation_id"] == "[1]"
    assert cits1[0]["paper_id"] == "doc_1"

    # Case 2: Answer with multiple citations [1] and [2]
    ans2 = "Both self-attention [1] and masked language modeling [2] revolutionized NLP."
    cits2 = gen.extract_citations(ans2, docs)
    assert len(cits2) == 2
    assert cits2[0]["paper_id"] == "doc_1"
    assert cits2[1]["paper_id"] == "doc_2"

    # Case 3: Answer without citation tag -> fallback to top document
    ans3 = "Transformers are neural network architectures."
    cits3 = gen.extract_citations(ans3, docs)
    assert len(cits3) == 1
    assert cits3[0]["paper_id"] == "doc_1"


class MockTokenizer:
    def __call__(self, text, **kwargs):
        class Tokens(dict):
            def to(self, device):
                return self
        return Tokens({"input_ids": torch.tensor([[1, 2, 3]])})

    def decode(self, token_ids, **kwargs):
        return "Quantum circuits can be simplified using gate cancellation [1]."


class MockModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        return torch.tensor([[10, 20, 30]])


def test_generate_answer_mock():
    mock_tok = MockTokenizer()
    mock_mod = MockModel()

    gen = ArxivRAGGenerator(device="cpu", tokenizer=mock_tok, model=mock_mod)

    docs = [
        {
            "id": "0704.1234",
            "title": "Quantum Gate Simplification",
            "abstract": "We present gate cancellation rules.",
            "authors": "Alice Researcher",
            "categories": "quant-ph",
        }
    ]

    res = gen.generate_answer("How to simplify quantum circuits?", docs)
    assert "Quantum circuits can be simplified" in res["answer"]
    assert len(res["citations"]) == 1
    assert res["citations"][0]["paper_id"] == "0704.1234"
