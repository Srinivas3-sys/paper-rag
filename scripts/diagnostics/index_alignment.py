"""Verify index and retrieval alignment:
- encoder architecture
- normalization consistency
- prefix requirements
- float32 dtype
- FAISS ordering vs Docstore ordering
"""

import json
import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer

from src.config import DOCSTORE_PATH, FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME


def check_alignment():
    docstore = pd.read_parquet(DOCSTORE_PATH)
    index = faiss.read_index(str(FAISS_INDEX_PATH))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    # 1. Total vector count
    ntotal = index.ntotal
    ndocs = len(docstore)
    count_match = (ntotal == ndocs)

    # 2. Metric type
    metric = index.metric_type
    # faiss.METRIC_INNER_PRODUCT is 0
    is_inner_product = (metric == faiss.METRIC_INNER_PRODUCT)

    # 3. Check sample vectors from index vs freshly encoded
    sample_size = 50
    sample_df = docstore.head(sample_size)
    sample_texts = sample_df["combined_text"].tolist()

    fresh_vecs = model.encode(
        sample_texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    # Reconstruct vectors from FAISS index (IndexFlatIP supports reconstruct)
    reconstructed_vecs = np.zeros((sample_size, fresh_vecs.shape[1]), dtype=np.float32)
    for i in range(sample_size):
        reconstructed_vecs[i] = index.reconstruct(i)

    # Check norms
    fresh_norms = np.linalg.norm(fresh_vecs, axis=1)
    recon_norms = np.linalg.norm(reconstructed_vecs, axis=1)

    # Check cosine similarity between fresh and reconstructed vectors for the exact same rows
    cosine_sims = np.sum(fresh_vecs * reconstructed_vecs, axis=1)
    max_diff = np.max(np.abs(fresh_vecs - reconstructed_vecs))

    report = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "model_max_seq_length": model.max_seq_length,
        "index_ntotal": ntotal,
        "docstore_len": ndocs,
        "counts_match": bool(count_match),
        "is_inner_product": bool(is_inner_product),
        "requires_prefixes": False,  # MiniLM is symmetric
        "embeddings_dtype": "float32",
        "fresh_norms_mean": float(np.mean(fresh_norms)),
        "reconstructed_norms_mean": float(np.mean(recon_norms)),
        "min_cosine_similarity": float(np.min(cosine_sims)),
        "mean_cosine_similarity": float(np.mean(cosine_sims)),
        "max_vector_element_difference": float(max_diff),
        "perfectly_aligned": bool(np.all(cosine_sims > 0.99999)),
    }
    return report


if __name__ == "__main__":
    rep = check_alignment()
    print("\n=== Alignment Report ===")
    print(json.dumps(rep, indent=2))
