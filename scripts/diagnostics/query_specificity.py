"""Diagnostic 5: Check Paraphrase query specificity.
Quantifies how many corpus documents score above the target document for
dense (MiniLM) and BM25 across all paraphrase queries in the benchmark.
"""

import json
import logging
import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer

from src.config import DATA_DIR, DOCSTORE_PATH, FAISS_INDEX_PATH, BM25_INDEX_PATH, BM25S_INDEX_DIR, EMBEDDING_MODEL_NAME
from src.retrieval import BM25Retriever, DenseRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_PATH = DATA_DIR / "eval_benchmark_360.json"


def check_query_specificity():
    logger.info("Loading docstore and benchmark...")
    docstore = pd.read_parquet(DOCSTORE_PATH)
    with open(BENCHMARK_PATH) as f:
        bench = json.load(f)

    para_queries = [q for q in bench if q.get("regime") == "paraphrase"]
    logger.info(f"Loaded {len(para_queries)} paraphrase queries.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Initializing retrievers on {device}...")
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    DenseRetriever(index=index, docstore=docstore, model=model)
    bm25_retriever = BM25Retriever(index_path=BM25_INDEX_PATH, bm25s_dir=BM25S_INDEX_DIR, docstore=docstore)

    doc_id_to_idx = {str(pid): i for i, pid in enumerate(docstore["id"])}

    dense_above_counts = []
    bm25_above_counts = []

    dense_target_ranks = []
    bm25_target_ranks = []

    dense_target_scores = []
    dense_top1_scores = []

    query_details = []

    # We can retrieve top 1000 for each query to compute rank and count above target accurately
    for q_idx, item in enumerate(para_queries):
        query = item["query"]
        target_id = item["target_doc_id"]
        target_idx = doc_id_to_idx.get(target_id)

        # Dense evaluation
        query_vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        # Search all 20,000 for exact ranking
        all_dists, all_indices = index.search(query_vec, len(docstore))
        dists = all_dists[0]
        indices = all_indices[0]

        # Find where target_idx appears
        target_pos = np.where(indices == target_idx)[0]
        if len(target_pos) > 0:
            dense_rank = int(target_pos[0]) + 1
            dense_target_score = float(dists[target_pos[0]])
        else:
            dense_rank = len(docstore)
            dense_target_score = 0.0

        dense_top1_score = float(dists[0])
        dense_above = dense_rank - 1

        dense_above_counts.append(dense_above)
        dense_target_ranks.append(dense_rank)
        dense_target_scores.append(dense_target_score)
        dense_top1_scores.append(dense_top1_score)

        # BM25 evaluation: compute full scores across corpus
        if bm25_retriever.bm25s_model is not None:
            import bm25s
            tokens = bm25s.tokenize([query], stopwords="en")
            bm_indices, bm_scores = bm25_retriever.bm25s_model.retrieve(tokens, k=len(docstore))
            bm_indices = bm_indices[0]
            bm_scores = bm_scores[0]

            target_pos_bm = np.where(bm_indices == target_idx)[0]
            if len(target_pos_bm) > 0:
                bm25_rank = int(target_pos_bm[0]) + 1
                float(bm_scores[target_pos_bm[0]])
            else:
                bm25_rank = len(docstore)
        else:
            bm25_rank = len(docstore)

        bm25_above = bm25_rank - 1
        bm25_above_counts.append(bm25_above)
        bm25_target_ranks.append(bm25_rank)

        query_details.append({
            "query_id": item["query_id"],
            "query": query,
            "target_id": target_id,
            "target_title": item["title"],
            "dense_rank": dense_rank,
            "dense_above": dense_above,
            "dense_target_score": dense_target_score,
            "dense_top1_score": dense_top1_score,
            "dense_score_gap": dense_top1_score - dense_target_score,
            "bm25_rank": bm25_rank,
            "bm25_above": bm25_above,
        })

    def stats(arr):
        return {
            "min": int(np.min(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "mean": float(np.mean(arr)),
            "max": int(np.max(arr)),
        }

    summary = {
        "num_paraphrase_queries": len(para_queries),
        "dense_docs_above_target": stats(dense_above_counts),
        "bm25_docs_above_target": stats(bm25_above_counts),
        "dense_mean_target_score": float(np.mean(dense_target_scores)),
        "dense_mean_top1_score": float(np.mean(dense_top1_scores)),
        "dense_mean_score_gap": float(np.mean(np.array(dense_top1_scores) - np.array(dense_target_scores))),
        "fraction_dense_target_in_top10": float(np.mean([1 if r <= 10 else 0 for r in dense_target_ranks])),
        "fraction_bm25_target_in_top10": float(np.mean([1 if r <= 10 else 0 for r in bm25_target_ranks])),
        "fraction_dense_target_in_top100": float(np.mean([1 if r <= 100 else 0 for r in dense_target_ranks])),
        "fraction_bm25_target_in_top100": float(np.mean([1 if r <= 100 else 0 for r in bm25_target_ranks])),
    }

    # Sort worst queries by dense rank
    worst_dense = sorted(query_details, key=lambda x: x["dense_rank"], reverse=True)[:5]
    best_dense = sorted(query_details, key=lambda x: x["dense_rank"])[:5]

    return summary, worst_dense, best_dense


if __name__ == "__main__":
    summary, worst, best = check_query_specificity()
    print("\n=== Paraphrase Specificity Summary ===")
    print(json.dumps(summary, indent=2))

    print("\n=== 5 Queries with Best Dense Ranks ===")
    for q in best:
        print(f"Rank {q['dense_rank']} (Gap {q['dense_score_gap']:.4f}) | Query: \"{q['query']}\" | Target: {q['target_title'][:50]}")

    print("\n=== 5 Queries with Worst Dense Ranks ===")
    for q in worst:
        print(f"Rank {q['dense_rank']} (Docs above: {q['dense_above']}) | Query: \"{q['query']}\" | Target: {q['target_title'][:50]}")
