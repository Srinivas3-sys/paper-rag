"""Compare dense encoders on the DEV split:
- sentence-transformers/all-MiniLM-L6-v2
- BAAI/bge-small-en-v1.5 (with instruction prefix)
- intfloat/e5-small-v2 (with 'query: ' and 'passage: ' prefixes)

Picks the best dense encoder on DEV and reports numbers.
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from src.config import BENCHMARK_PATH, DOCSTORE_PATH, FAISS_INDEX_PATH, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path(".scratch")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_encoder(
    model_name: str,
    index: faiss.Index,
    dev_queries: List[Dict[str, Any]],
    docstore: pd.DataFrame,
    query_prefix: str = "",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 64,
) -> Dict[str, Any]:
    logger.info(f"Evaluating {model_name} on {len(dev_queries)} DEV queries...")
    model = SentenceTransformer(model_name, device=device)

    doc_id_to_idx = {str(pid): i for i, pid in enumerate(docstore["id"])}

    q_texts = [f"{query_prefix}{q['query']}" for q in dev_queries]
    q_vecs = model.encode(q_texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    t0 = time.perf_counter()
    dists, indices = index.search(q_vecs, 20)
    dur_ms = (time.perf_counter() - t0) * 1000.0 / len(dev_queries)

    # Compute per-regime and overall strict metrics
    regimes = ["lexical", "paraphrase", "acronym"]
    results: Dict[str, Any] = {"overall": {}, "by_regime": {}}

    all_ranks = []
    all_r1 = []
    all_r10 = []

    reg_ranks = {r: [] for r in regimes}
    reg_r10 = {r: [] for r in regimes}

    for i, q in enumerate(dev_queries):
        reg = q["regime"]
        target_id = q["target_doc_id"]
        target_idx = doc_id_to_idx.get(target_id)

        retrieved_indices = indices[i]
        rank = 0
        if target_idx is not None and target_idx in retrieved_indices:
            rank = int(np.where(retrieved_indices == target_idx)[0][0]) + 1

        rr = 1.0 / rank if rank > 0 else 0.0
        r1 = 1.0 if rank == 1 else 0.0
        r10 = 1.0 if (rank > 0 and rank <= 10) else 0.0

        all_ranks.append(rr)
        all_r1.append(r1)
        all_r10.append(r10)

        if reg in reg_ranks:
            reg_ranks[reg].append(rr)
            reg_r10[reg].append(r10)

    results["overall"] = {
        "mrr": float(np.mean(all_ranks)),
        "recall_at_1": float(np.mean(all_r1)),
        "recall_at_10": float(np.mean(all_r10)),
        "latency_ms": dur_ms,
    }

    for r in regimes:
        results["by_regime"][r] = {
            "mrr": float(np.mean(reg_ranks[r])) if reg_ranks[r] else 0.0,
            "recall_at_10": float(np.mean(reg_r10[r])) if reg_r10[r] else 0.0,
            "n": len(reg_ranks[r]),
        }

    logger.info(f"{model_name} DEV MRR: Overall={results['overall']['mrr']:.4f}, Para={results['by_regime']['paraphrase']['mrr']:.4f}, Lex={results['by_regime']['lexical']['mrr']:.4f}, Acro={results['by_regime']['acronym']['mrr']:.4f}")
    return results


def get_or_build_index(
    model_name: str,
    docstore: pd.DataFrame,
    passage_prefix: str = "",
    cache_name: str = "",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 128,
) -> faiss.Index:
    cache_path = CACHE_DIR / f"{cache_name}.npy"
    if cache_path.exists():
        logger.info(f"Loading cached embeddings from {cache_path}...")
        embeddings = np.load(cache_path)
    else:
        logger.info(f"Encoding 20,000 documents with {model_name} (batch_size={batch_size})...")
        model = SentenceTransformer(model_name, device=device)
        texts = [f"{passage_prefix}{t}" for t in docstore["combined_text"]]
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        np.save(cache_path, embeddings)
        logger.info(f"Saved embeddings to {cache_path}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def main():
    logger.info("Starting dense encoder comparison on DEV split...")
    with open(BENCHMARK_PATH) as f:
        bench = json.load(f)

    dev_queries = [q for q in bench if q["split"] == "dev"]
    docstore = pd.read_parquet(DOCSTORE_PATH)
    logger.info(f"Loaded {len(dev_queries)} DEV queries and {len(docstore)} documents.")

    results = {}

    # 1. all-MiniLM-L6-v2 (Current)
    logger.info("Loading existing FAISS index for MiniLM...")
    minilm_index = faiss.read_index(str(FAISS_INDEX_PATH))
    results["all-MiniLM-L6-v2"] = evaluate_encoder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        index=minilm_index,
        dev_queries=dev_queries,
        docstore=docstore,
        query_prefix="",
    )

    # 2. BAAI/bge-small-en-v1.5 (512 tokens)
    bge_index = get_or_build_index(
        model_name="BAAI/bge-small-en-v1.5",
        docstore=docstore,
        passage_prefix="",
        cache_name="bge_small_embs",
    )
    results["BAAI/bge-small-en-v1.5"] = evaluate_encoder(
        model_name="BAAI/bge-small-en-v1.5",
        index=bge_index,
        dev_queries=dev_queries,
        docstore=docstore,
        query_prefix="Represent this sentence for searching relevant passages: ",
    )

    # 3. intfloat/e5-small-v2 (512 tokens with required prefixes)
    e5_index = get_or_build_index(
        model_name="intfloat/e5-small-v2",
        docstore=docstore,
        passage_prefix="passage: ",
        cache_name="e5_small_embs",
    )
    results["intfloat/e5-small-v2"] = evaluate_encoder(
        model_name="intfloat/e5-small-v2",
        index=e5_index,
        dev_queries=dev_queries,
        docstore=docstore,
        query_prefix="query: ",
    )

    # Generate markdown report
    md = [
        "# Dense Encoder Comparison on Dev Split (N=150)",
        "",
        "Empirical comparison of dense embedding architectures evaluated on the DEV split across 3 regimes (50 Lexical, 50 Paraphrase, 50 Acronym).",
        "",
        "| Dense Architecture | Parameters | Max Seq Len | Overall MRR | Lexical MRR | Paraphrase MRR | Acronym MRR | Overall Recall@10 |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    meta = {
        "all-MiniLM-L6-v2": ("22M", "256"),
        "BAAI/bge-small-en-v1.5": ("33M", "512"),
        "intfloat/e5-small-v2": ("33M", "512"),
    }

    for m, d in results.items():
        params, seq_len = meta.get(m, ("—", "—"))
        ov = d["overall"]
        reg = d["by_regime"]
        md.append(
            f"| **{m}** | {params} | {seq_len} | **{ov['mrr']:.4f}** | {reg['lexical']['mrr']:.4f} | **{reg['paraphrase']['mrr']:.4f}** | {reg['acronym']['mrr']:.4f} | {ov['recall_at_10']:.4f} |"
        )

    output_path = RESULTS_DIR / "encoder_comparison_dev.md"
    with open(output_path, "w") as f:
        f.write("\n".join(md) + "\n")

    print("\n" + "\n".join(md))
    logger.info(f"Saved encoder comparison to {output_path}")


if __name__ == "__main__":
    main()
