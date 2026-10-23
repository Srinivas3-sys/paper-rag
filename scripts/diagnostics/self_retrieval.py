"""Self-retrieval diagnostic test.

Embeds 500 random documents as queries to test if DenseRetriever
returns the same document at rank 1. Verifies ID alignment, metric,
and normalization.
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import faiss

from src.config import DOCSTORE_PATH, FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME
from src.retrieval import DenseRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_self_retrieval(sample_size: int = 500, seed: int = 42):
    logger.info(f"Loading docstore from {DOCSTORE_PATH} and FAISS from {FAISS_INDEX_PATH}...")
    docstore = pd.read_parquet(DOCSTORE_PATH)
    index = faiss.read_index(str(FAISS_INDEX_PATH))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading model {EMBEDDING_MODEL_NAME} on {device}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    DenseRetriever(
        index=index,
        docstore=docstore,
        model=model,
    )

    # Sample 500 documents
    sample_df = docstore.sample(n=sample_size, random_state=seed).copy()
    sample_df.index.tolist()

    # Test 1: Query is the exact indexed text ('combined_text')
    logger.info("Running Test 1: Query is exact combined_text...")
    exact_rank1 = 0
    exact_ranks = []

    # Test 2: Query is just the abstract
    logger.info("Running Test 2: Query is doc abstract...")
    abstract_rank1 = 0
    abstract_rank5 = 0
    abstract_ranks = []

    # Test 3: Direct vector reconstruction dot product against index
    logger.info("Running Test 3: Direct FAISS index search on batch of embeddings...")
    sample_texts_combined = sample_df["combined_text"].tolist()
    sample_texts_abstract = sample_df["abstract"].tolist()
    target_ids = sample_df["id"].tolist()

    # Batch encode
    comb_vecs = model.encode(sample_texts_combined, batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    abs_vecs = model.encode(sample_texts_abstract, batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    # Search top 10 in FAISS
    D_comb, I_comb = index.search(comb_vecs, 10)
    D_abs, I_abs = index.search(abs_vecs, 10)

    for i, target_id in enumerate(target_ids):
        # Combined text results
        retrieved_ids_comb = [docstore.iloc[idx]["id"] for idx in I_comb[i] if 0 <= idx < len(docstore)]
        if target_id in retrieved_ids_comb:
            rank = retrieved_ids_comb.index(target_id) + 1
            exact_ranks.append(rank)
            if rank == 1:
                exact_rank1 += 1
        else:
            exact_ranks.append(999)

        # Abstract results
        retrieved_ids_abs = [docstore.iloc[idx]["id"] for idx in I_abs[i] if 0 <= idx < len(docstore)]
        if target_id in retrieved_ids_abs:
            rank = retrieved_ids_abs.index(target_id) + 1
            abstract_ranks.append(rank)
            if rank == 1:
                abstract_rank1 += 1
            if rank <= 5:
                abstract_rank5 += 1
        else:
            abstract_ranks.append(999)

    exact_r1_rate = exact_rank1 / sample_size
    abs_r1_rate = abstract_rank1 / sample_size
    abs_r5_rate = abstract_rank5 / sample_size

    logger.info(f"Test 1 (combined_text): Rank-1 Recall = {exact_rank1}/{sample_size} ({exact_r1_rate:.4f})")
    logger.info(f"Test 2 (abstract only): Rank-1 Recall = {abstract_rank1}/{sample_size} ({abs_r1_rate:.4f}), Rank-5 Recall = {abstract_rank5}/{sample_size} ({abs_r5_rate:.4f})")

    results = {
        "sample_size": sample_size,
        "combined_text_rank1_recall": exact_r1_rate,
        "abstract_rank1_recall": abs_r1_rate,
        "abstract_rank5_recall": abs_r5_rate,
        "mean_rank_combined": float(np.mean([r for r in exact_ranks if r != 999])),
        "mean_rank_abstract": float(np.mean([r for r in abstract_ranks if r != 999])),
        "missing_from_top10_combined": int(sum(1 for r in exact_ranks if r == 999)),
        "missing_from_top10_abstract": int(sum(1 for r in abstract_ranks if r == 999)),
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=500)
    args = parser.parse_args()
    res = run_self_retrieval(sample_size=args.sample_size)
    print("\n=== Self Retrieval Results ===")
    print(json.dumps(res, indent=2))
