"""Analyze token and word length distribution of abstracts in the 20,000 corpus:
- Measure word count distribution
- Tokenize with all-MiniLM-L6-v2 (max_seq_length=256)
- Tokenize with bge-small-en-v1.5 (max_seq_length=512)
- Calculate truncation percentages for abstract alone and combined_text (Title + Abstract).
"""

import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.config import DOCSTORE_PATH


def analyze_length_distribution():
    print(f"Loading docstore from {DOCSTORE_PATH}...")
    df = pd.read_parquet(DOCSTORE_PATH)

    print("Loading tokenizers...")
    minilm_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    bge_tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

    abstract_words = [len(str(a).split()) for a in df["abstract"]]
    combined_words = [len(str(t).split()) for t in df["combined_text"]]

    print("Tokenizing combined_text with MiniLM...")
    minilm_tokens = [len(tokens) for tokens in minilm_tok(df["combined_text"].tolist(), truncation=False)["input_ids"]]

    print("Tokenizing combined_text with BGE...")
    bge_tokens = [len(tokens) for tokens in bge_tok(df["combined_text"].tolist(), truncation=False)["input_ids"]]

    def get_stats(arr):
        return {
            "min": int(np.min(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": int(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    n = len(df)
    minilm_trunc_256 = sum(1 for t in minilm_tokens if t > 256) / n
    sum(1 for t in minilm_tokens if t > 128) / n
    bge_trunc_512 = sum(1 for t in bge_tokens if t > 512) / n
    sum(1 for t in bge_tokens if t > 256) / n

    results = {
        "num_documents": n,
        "abstract_word_stats": get_stats(abstract_words),
        "combined_word_stats": get_stats(combined_words),
        "minilm_token_stats": get_stats(minilm_tokens),
        "bge_token_stats": get_stats(bge_tokens),
        "minilm_max_seq_length": 256,
        "minilm_truncation_rate_at_256": minilm_trunc_256,
        "minilm_count_truncated_at_256": int(sum(1 for t in minilm_tokens if t > 256)),
        "bge_max_seq_length": 512,
        "bge_truncation_rate_at_512": bge_trunc_512,
        "bge_count_truncated_at_512": int(sum(1 for t in bge_tokens if t > 512)),
    }
    return results


if __name__ == "__main__":
    res = analyze_length_distribution()
    print("\n=== Length Truncation Analysis ===")
    print(json.dumps(res, indent=2))
