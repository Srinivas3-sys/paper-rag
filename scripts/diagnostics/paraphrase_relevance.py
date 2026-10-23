"""Inspect 20 random Paraphrase queries:
- Print query, target title/abstract, and top 3 dense results.
- Determine whether retrieved papers share arXiv categories, topics, or are plausibly relevant.
- Quantify the fraction of plausible relevant papers.
"""

import json
import logging
import random
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

from src.config import DATA_DIR, DOCSTORE_PATH, FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME
from src.retrieval import DenseRetriever

BENCHMARK_PATH = DATA_DIR / "eval_benchmark_360.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def inspect_paraphrase_relevance(num_samples: int = 20, seed: int = 42):
    # Load benchmark
    with open(BENCHMARK_PATH) as f:
        bench = json.load(f)

    para_queries = [q for q in bench if q.get("regime") == "paraphrase"]
    logger.info(f"Loaded {len(para_queries)} paraphrase queries from benchmark.")

    docstore = pd.read_parquet(DOCSTORE_PATH)
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cuda")
    dense_retriever = DenseRetriever(index=index, docstore=docstore, model=model)

    doc_lookup = {str(row["id"]): row for _, row in docstore.iterrows()}

    random.seed(seed)
    sampled = random.sample(para_queries, min(num_samples, len(para_queries)))

    inspections = []

    target_in_top1_count = 0
    target_in_top3_count = 0
    plausibly_relevant_at_least_one = 0
    same_category_top3_count = 0
    total_top3_examined = 0
    plausible_top3_count = 0

    for i, item in enumerate(sampled):
        qid = item["query_id"]
        qtext = item["query"]
        target_id = item["target_doc_id"]
        target_doc = doc_lookup.get(target_id)

        target_cats = set(str(target_doc["categories"]).split()) if target_doc is not None else set()

        retrieved = dense_retriever.retrieve(qtext, top_k=3)

        is_target_in_top1 = (len(retrieved) > 0 and retrieved[0]["id"] == target_id)
        is_target_in_top3 = any(r["id"] == target_id for r in retrieved)

        if is_target_in_top1:
            target_in_top1_count += 1
        if is_target_in_top3:
            target_in_top3_count += 1

        case_info = {
            "query_id": qid,
            "query": qtext,
            "target_id": target_id,
            "target_title": target_doc["title"] if target_doc is not None else "",
            "target_categories": list(target_cats),
            "target_abstract": target_doc["abstract"] if target_doc is not None else "",
            "top_results": [],
        }

        has_plausible = False
        for r in retrieved:
            total_top3_examined += 1
            r_cats = set(str(r["categories"]).split())
            shared_cats = target_cats.intersection(r_cats)
            is_target = (r["id"] == target_id)

            # A paper is plausibly relevant if:
            # 1) It is the target paper, OR
            # 2) It shares a specific arXiv category AND has high semantic score (>0.45), OR
            # 3) The topic of the query directly applies to the paper title/abstract
            is_plausible = is_target or bool(shared_cats and r["score"] > 0.40)

            if is_plausible:
                plausible_top3_count += 1
                has_plausible = True
            if shared_cats:
                same_category_top3_count += 1

            case_info["top_results"].append({
                "rank": r["rank"],
                "id": r["id"],
                "title": r["title"],
                "categories": list(r_cats),
                "shared_categories": list(shared_cats),
                "score": r["score"],
                "is_target": is_target,
                "is_plausibly_relevant": is_plausible,
                "abstract_snippet": r["abstract"][:180] + "...",
            })

        if has_plausible:
            plausibly_relevant_at_least_one += 1

        inspections.append(case_info)

    summary = {
        "num_inspected": len(sampled),
        "target_in_top1": target_in_top1_count,
        "target_in_top3": target_in_top3_count,
        "queries_with_at_least_one_plausible_in_top3": plausibly_relevant_at_least_one,
        "fraction_queries_with_plausible_top3": plausibly_relevant_at_least_one / len(sampled),
        "total_top3_candidates": total_top3_examined,
        "top3_candidates_same_category": same_category_top3_count,
        "top3_candidates_plausible": plausible_top3_count,
        "fraction_top3_candidates_plausible": plausible_top3_count / total_top3_examined if total_top3_examined else 0,
    }

    return summary, inspections


if __name__ == "__main__":
    summary, inspections = inspect_paraphrase_relevance(20, seed=42)
    print("\n=== Paraphrase Inspection Summary ===")
    print(json.dumps(summary, indent=2))

    print("\n=== Sample Case Study (First 3 Cases) ===")
    for case in inspections[:3]:
        print(f"\nQUERY [{case['query_id']}]: \"{case['query']}\"")
        print(f"Target Doc: [{case['target_id']}] {case['target_title']} (Cats: {case['target_categories']})")
        print(f"Target Abstract: {case['target_abstract'][:180]}...")
        print("Top 3 Dense Retrieved:")
        for r in case["top_results"]:
            print(f"  Rank {r['rank']} (Score {r['score']:.4f}, Shared Cats: {r['shared_categories']}, IsTarget: {r['is_target']}):")
            print(f"    [{r['id']}] {r['title']}")
            print(f"    Abstract: {r['abstract_snippet']}")
