"""Unit tests for data preprocessing and evaluation benchmark creation."""

import tempfile
from pathlib import Path
import pandas as pd

from src.data import (
    clean_text,
    create_evaluation_benchmark,
    parse_arxiv_record,
)


def test_clean_text():
    assert clean_text("  hello   world  \n\n\t") == "hello world"
    assert clean_text(None) == ""
    assert clean_text(123) == ""
    assert clean_text("Line 1.\nLine 2.") == "Line 1. Line 2."


def test_parse_arxiv_record_valid():
    raw = {
        "id": "2104.00001",
        "title": "Quantum   Entanglement in \n Many-Body Systems",
        "abstract": "This is a detailed abstract discussing quantum entanglement in many-body physics systems over several sentences.",
        "authors": "Alice Smith, Bob Jones",
        "categories": "quant-ph cond-mat",
    }
    parsed = parse_arxiv_record(raw)
    assert parsed is not None
    assert parsed["id"] == "2104.00001"
    assert parsed["title"] == "Quantum Entanglement in Many-Body Systems"
    assert "Title: Quantum Entanglement" in parsed["combined_text"]
    assert "Alice Smith, Bob Jones" == parsed["authors"]


def test_parse_arxiv_record_invalid():
    # Missing required id or title
    assert parse_arxiv_record({"id": "", "title": "Test", "abstract": "Valid abstract words"}) is None
    assert parse_arxiv_record({"id": "123", "title": "", "abstract": "Valid abstract words"}) is None
    # Abstract too short
    assert parse_arxiv_record({"id": "123", "title": "Test", "abstract": "Too short"}) is None


def test_create_evaluation_benchmark():
    dummy_data = [
        {
            "id": f"paper_{i}",
            "title": f"Deep Learning Theory Part {i}",
            "abstract": f"Abstract for deep learning paper number {i} with sufficient words for testing.",
            "authors": f"Author {i}",
            "categories": "cs.LG",
            "combined_text": f"Title: Deep Learning Theory Part {i}\nAbstract: ...",
        }
        for i in range(10)
    ]
    df = pd.DataFrame(dummy_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        eval_path = Path(tmpdir) / "test_eval.json"
        eval_set = create_evaluation_benchmark(df, num_queries=5, output_path=eval_path, seed=42)

        assert len(eval_set) == 5
        assert eval_path.exists()
        first = eval_set[0]
        assert "query" in first
        assert "target_doc_id" in first
        assert first["target_doc_id"].startswith("paper_")
