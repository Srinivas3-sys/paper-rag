"""Unit tests for answer generation, token budgeting, and citation parsing."""

import torch

from src.generator import ABSTAIN_MESSAGE, ArxivRAGGenerator


class DummyTokenizer:
    def __call__(self, text, **kwargs):
        class Tokens(dict):
            def to(self, device):
                return self
        return Tokens({"input_ids": torch.tensor([[1, 2, 3]])})

    def encode(self, text, **kwargs):
        # Approximate 1 token per 4 chars for mock budgeting
        return list(range(max(1, len(text) // 4)))

    def decode(self, token_ids, **kwargs):
        return "Quantum circuits can be simplified using gate cancellation [1]."


class DummyModel:
    def __init__(self):
        class Config:
            is_encoder_decoder = True
        self.config = Config()
        self.last_kwargs = {}

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.last_kwargs = kwargs
        return torch.tensor([[10, 20, 30]])


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

    gen = ArxivRAGGenerator(device="cpu", tokenizer=DummyTokenizer(), model=DummyModel())
    prompt, retained = gen.format_context_prompt("How to optimize quantum gates?", docs)

    assert "[1] (arXiv:2104.0001) Quantum Circuit Optimization" in prompt
    assert "[2] (arXiv:2104.0002) Graph Representation Learning" in prompt
    assert "Question: How to optimize quantum gates?" in prompt
    assert len(retained) == 2


def test_context_budget_truncation():
    docs = [
        {"id": f"doc_{i}", "title": f"Paper {i}", "abstract": "Detailed abstract " * 40}
        for i in range(10)
    ]
    gen = ArxivRAGGenerator(device="cpu", tokenizer=DummyTokenizer(), model=DummyModel())
    prompt, retained = gen.format_context_prompt("Question about papers?", docs, max_tokens=300)

    # Context should drop lowest-ranked docs to stay within 150 token budget
    assert len(retained) < len(docs)
    assert len(retained) >= 1
    assert "Question about papers?" in prompt


def test_extract_citations_strict_no_fallbacks():
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

    gen = ArxivRAGGenerator(device="cpu", tokenizer=DummyTokenizer(), model=DummyModel())

    # Case 1: Answer with explicit citation [1]
    ans1 = "Transformers rely on multi-head attention [1] for self-attention."
    cits1 = gen.extract_citations(ans1, docs)
    assert len(cits1) == 1
    assert cits1[0]["citation_id"] == "[1]"
    assert cits1[0]["paper_id"] == "doc_1"

    # Case 2: Multiple citations [1] and [2]
    ans2 = "Both self-attention [1] and masked language modeling [2] revolutionized NLP."
    cits2 = gen.extract_citations(ans2, docs)
    assert len(cits2) == 2
    assert cits2[0]["paper_id"] == "doc_1"
    assert cits2[1]["paper_id"] == "doc_2"

    # Case 3: Answer without citation tag -> STRICT: empty list, NO fake fallback
    ans3 = "Transformers are neural network architectures."
    cits3 = gen.extract_citations(ans3, docs)
    assert len(cits3) == 0


def test_abstain_path():
    docs = [
        {
            "id": "doc_1",
            "title": "Unrelated Physics Paper",
            "abstract": "Astrophysical spectroscopy of white dwarfs.",
            "score": 0.12,  # Low score
        }
    ]
    gen = ArxivRAGGenerator(
        device="cpu",
        tokenizer=DummyTokenizer(),
        model=DummyModel(),
        abstain_threshold=0.50,
    )

    # When top retrieval score is below abstain_threshold (0.12 < 0.50)
    res = gen.generate_answer("How to bake sourdough bread?", docs)
    assert res["abstained"] is True
    assert res["answer"] == ABSTAIN_MESSAGE
    assert res["citations"] == []


def test_generate_answer_mock():
    mock_tok = DummyTokenizer()
    mock_mod = DummyModel()

    gen = ArxivRAGGenerator(device="cpu", tokenizer=mock_tok, model=mock_mod)
    docs = [
        {
            "id": "doc_101",
            "title": "Quantum Error Mitigation",
            "abstract": "We evaluate zero-noise extrapolation techniques.",
            "authors": "Physicist Z",
            "categories": "quant-ph",
            "score": 0.85,
        }
    ]

    response = gen.generate_answer(
        question="How does quantum error mitigation work?",
        retrieved_docs=docs,
        temperature=0.0,
    )

    assert response["question"] == "How does quantum error mitigation work?"
    assert "Quantum circuits can be simplified" in response["answer"]
    assert len(response["citations"]) == 1
    assert response["citations"][0]["paper_id"] == "doc_101"
    assert response["abstained"] is False


def test_temperature_handling():
    mock_tok = DummyTokenizer()
    mock_mod = DummyModel()
    gen = ArxivRAGGenerator(device="cpu", tokenizer=mock_tok, model=mock_mod)
    docs = [{"id": "d1", "title": "T", "abstract": "A", "score": 0.9}]

    # 1. Greedy decoding with temperature=0.0
    gen.generate_answer("Query", docs, temperature=0.0)
    assert mock_mod.last_kwargs["do_sample"] is False
    assert "temperature" not in mock_mod.last_kwargs

    # 2. Stochastic sampling with temperature=0.7
    gen.generate_answer("Query", docs, temperature=0.7)
    assert mock_mod.last_kwargs["do_sample"] is True
    assert mock_mod.last_kwargs["temperature"] == 0.7
    assert mock_mod.last_kwargs["top_p"] == 0.9

