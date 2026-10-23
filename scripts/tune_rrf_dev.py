"""Tune Weighted RRF parameters strictly on the DEV split (N=150).

Grid search:
- Dense Encoders: all-MiniLM-L6-v2, intfloat/e5-small-v2
- k in {10, 30, 60}
- bm25_weight in {1.0, 1.5, 2.0, 3.0} (dense_weight fixed at 1.0)

Saves results to results/rrf_tuning_dev.md
"""

import json
import logging
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from src.config import BENCHMARK_PATH, DOCSTORE_PATH, FAISS_INDEX_PATH, RESULTS_DIR
from src.retrieval import BM25Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_rrf_tuning():
    logger.info("Loading DEV benchmark queries...")
    with open(BENCHMARK_PATH) as f:
        all_queries = json.load(f)
    dev_queries = [q for q in all_queries if q.get("split") == "dev"]
    logger.info(f"Loaded {len(dev_queries)} DEV queries.")

    logger.info("Loading docstore...")
    docstore = pd.read_parquet(DOCSTORE_PATH)
    doc_idx_to_id = {i: str(pid) for i, pid in enumerate(docstore["id"])}

    logger.info("Loading BM25 retriever...")
    bm25 = BM25Retriever(docstore_path=DOCSTORE_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Precompute BM25 top 50 rankings for DEV queries
    logger.info("Precomputing BM25 top-50 results for DEV queries...")
    bm25_top50 = []
    for q in dev_queries:
        res = bm25.retrieve(q["query"], top_k=50)
        bm25_top50.append([(r["id"], r["rank"]) for r in res])

    # Encoders to evaluate
    encoders_to_test = [
        {
            "name": "all-MiniLM-L6-v2",
            "model_id": "sentence-transformers/all-MiniLM-L6-v2",
            "query_prefix": "",
            "faiss_path": FAISS_INDEX_PATH,
            "embs_path": None,
        },
        {
            "name": "intfloat/e5-small-v2",
            "model_id": "intfloat/e5-small-v2",
            "query_prefix": "query: ",
            "faiss_path": None,
            "embs_path": Path(".scratch/e5_small_embs.npy"),
        },
    ]

    grid_results = []
    best_config = None
    best_dev_mrr = -1.0

    k_values = [10, 30, 60]
    bm25_weights = [1.0, 1.5, 2.0, 3.0]

    for enc_info in encoders_to_test:
        enc_name = enc_info["name"]
        logger.info(f"Setting up dense index for {enc_name}...")

        if enc_info["faiss_path"] and Path(enc_info["faiss_path"]).exists():
            index = faiss.read_index(str(enc_info["faiss_path"]))
        else:
            embs = np.load(enc_info["embs_path"])
            index = faiss.IndexFlatIP(embs.shape[1])
            index.add(embs)

        model = SentenceTransformer(enc_info["model_id"], device=device)
        q_texts = [f"{enc_info['query_prefix']}{q['query']}" for q in dev_queries]
        q_vecs = model.encode(q_texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

        # Precompute dense top 50 rankings
        dists, idxs = index.search(q_vecs, 50)
        dense_top50 = []
        for i in range(len(dev_queries)):
            row_idxs = idxs[i]
            dense_top50.append([(doc_idx_to_id[idx], rank + 1) for rank, idx in enumerate(row_idxs) if idx in doc_idx_to_id])

        # Grid search over k and bm25_weight
        for k in k_values:
            for w_bm25 in bm25_weights:
                overall_ranks = []
                reg_ranks = {"lexical": [], "paraphrase": [], "acronym": []}
                reg_r10 = {"lexical": [], "paraphrase": [], "acronym": []}

                for i, q in enumerate(dev_queries):
                    target_id = q["target_doc_id"]
                    reg = q["regime"]

                    # Compute fused scores
                    fused_scores = {}
                    # Dense contribution: 1.0 / (k + rank)
                    for pid, rank in dense_top50[i]:
                        fused_scores[pid] = fused_scores.get(pid, 0.0) + 1.0 / (k + rank)
                    # BM25 contribution: w_bm25 / (k + rank)
                    for pid, rank in bm25_top50[i]:
                        fused_scores[pid] = fused_scores.get(pid, 0.0) + w_bm25 / (k + rank)

                    # Sort top candidates
                    sorted_pids = sorted(fused_scores.keys(), key=lambda p: fused_scores[p], reverse=True)
                    rank = (sorted_pids.index(target_id) + 1) if target_id in sorted_pids else 0

                    rr = 1.0 / rank if rank > 0 else 0.0
                    r10 = 1.0 if (rank > 0 and rank <= 10) else 0.0

                    overall_ranks.append(rr)
                    reg_ranks[reg].append(rr)
                    reg_r10[reg].append(r10)

                mean_mrr = float(np.mean(overall_ranks))
                para_mrr = float(np.mean(reg_ranks["paraphrase"]))
                lex_mrr = float(np.mean(reg_ranks["lexical"]))
                acro_mrr = float(np.mean(reg_ranks["acronym"]))
                overall_r10 = float(np.mean([np.mean(reg_r10[r]) for r in reg_r10]))

                entry = {
                    "dense_encoder": enc_name,
                    "k": k,
                    "w_bm25": w_bm25,
                    "w_dense": 1.0,
                    "overall_mrr": mean_mrr,
                    "lexical_mrr": lex_mrr,
                    "paraphrase_mrr": para_mrr,
                    "acronym_mrr": acro_mrr,
                    "overall_recall_at_10": overall_r10,
                }
                grid_results.append(entry)

                if mean_mrr > best_dev_mrr:
                    best_dev_mrr = mean_mrr
                    best_config = entry

    # Sort results by overall MRR descending
    grid_results.sort(key=lambda x: x["overall_mrr"], reverse=True)

    # Generate markdown report
    md_lines = [
        "# RRF Hyperparameter Grid Ablation on DEV Split (N=150)",
        "",
        "Evaluation strictly on the 150 DEV queries (50 Lexical, 50 Paraphrase, 50 Acronym).",
        "Dense weight fixed at 1.0; BM25 weight tuned over {1.0, 1.5, 2.0, 3.0}; k tuned over {10, 30, 60}.",
        "",
        "| Rank | Dense Encoder | k | BM25 Weight | Overall MRR | Lexical MRR | Paraphrase MRR | Acronym MRR | Overall Recall@10 |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for rank, r in enumerate(grid_results, 1):
        md_lines.append(
            f"| {rank} | {r['dense_encoder']} | {r['k']} | {r['w_bm25']:.1f} | "
            f"**{r['overall_mrr']:.4f}** | {r['lexical_mrr']:.4f} | {r['paraphrase_mrr']:.4f} | {r['acronym_mrr']:.4f} | {r['overall_recall_at_10']:.4f} |"
        )

    md_lines.extend([
        "",
        "## Selected Configuration on Dev",
        f"- **Best Dense Encoder**: `{best_config['dense_encoder']}`",
        f"- **Best RRF k**: `{best_config['k']}`",
        f"- **Best BM25 Weight**: `{best_config['w_bm25']:.1f}` (Dense Weight: 1.0)",
        f"- **Dev Overall MRR**: `{best_config['overall_mrr']:.4f}` (vs BM25 DEV MRR: 0.6321)",
        f"- **Dev Paraphrase MRR**: `{best_config['paraphrase_mrr']:.4f}` (vs BM25 DEV MRR: 0.2284)",
        "",
    ])

    out_path = RESULTS_DIR / "rrf_tuning_dev.md"
    out_path.write_text("\n".join(md_lines))
    logger.info(f"Saved RRF tuning report to {out_path}")
    print("\n".join(md_lines[:25]))
    return best_config


if __name__ == "__main__":
    run_rrf_tuning()
