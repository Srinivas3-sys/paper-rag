"""Rigorous multi-regime retrieval benchmark evaluation.

Computes:
- Strict metrics (single target doc): Recall@1/5/10/20, MRR, nDCG@10
- Lenient metrics (target OR shared category with cosine >= dev threshold): Recall@1/5/10/20, MRR, nDCG@10
- 95% Bootstrap Confidence Intervals (1,000 resamples) for both Overall AND Per-Regime
- Paired significance test (Wilcoxon signed-rank and bootstrap paired difference) PER REGIME and OVERALL
- RRF hyperparameter grid ablation (k in {10, 30, 60, 100}, alpha in {0.3, 0.5, 0.7}) tuned strictly on dev split
- Per-regime visualization with bootstrap error bars that do not obscure value labels
"""

import argparse
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Set, Tuple

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer

from src.config import (
    BENCHMARK_PATH,
    DOCSTORE_PATH,
    FAISS_INDEX_PATH,
    RESULTS_DIR,
)
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_ndcg_at_k(binary_relevance: List[bool], k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank K for binary relevance."""
    dcg = 0.0
    for rank_idx, is_rel in enumerate(binary_relevance[:k]):
        if is_rel:
            dcg += 1.0 / math.log2(rank_idx + 2)
    # Binary relevance ideal DCG for at least one relevant document is 1.0 (at rank 1)
    # If multiple relevant documents exist, ideal DCG is sum of discounts for top-m ranks
    num_rel = sum(1 for r in binary_relevance if r)
    if num_rel == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(r + 2) for r in range(min(num_rel, k)))
    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics_for_query(
    retrieved_ids: List[str],
    target_id: str,
    retrieved_sims: List[float],
    doc_categories_map: Dict[str, Set[str]],
    target_categories: Set[str],
    similarity_threshold: float,
) -> Dict[str, float]:
    """Compute strict and lenient IR metrics for a single query."""
    # Strict binary relevance: exactly matches target_id
    strict_rels = [doc_id == target_id for doc_id in retrieved_ids]

    # Lenient binary relevance: matches target_id OR (shares category AND cosine_sim >= threshold)
    lenient_rels = []
    for doc_id, sim in zip(retrieved_ids, retrieved_sims):
        if doc_id == target_id:
            lenient_rels.append(True)
        else:
            cand_cats = doc_categories_map.get(doc_id, set())
            is_cat_match = bool(cand_cats.intersection(target_categories))
            lenient_rels.append(bool(is_cat_match and sim >= similarity_threshold))

    # Strict MRR
    strict_rank = (strict_rels.index(True) + 1) if True in strict_rels else 0
    strict_rr = 1.0 / strict_rank if strict_rank > 0 else 0.0

    # Lenient MRR
    lenient_rank = (lenient_rels.index(True) + 1) if True in lenient_rels else 0
    lenient_rr = 1.0 / lenient_rank if lenient_rank > 0 else 0.0

    return {
        # Strict metrics
        "strict_recall_at_1": 1.0 if any(strict_rels[:1]) else 0.0,
        "strict_recall_at_5": 1.0 if any(strict_rels[:5]) else 0.0,
        "strict_recall_at_10": 1.0 if any(strict_rels[:10]) else 0.0,
        "strict_recall_at_20": 1.0 if any(strict_rels[:20]) else 0.0,
        "strict_mrr": strict_rr,
        "strict_ndcg_at_10": compute_ndcg_at_k(strict_rels, k=10),
        # Lenient metrics
        "lenient_recall_at_1": 1.0 if any(lenient_rels[:1]) else 0.0,
        "lenient_recall_at_5": 1.0 if any(lenient_rels[:5]) else 0.0,
        "lenient_recall_at_10": 1.0 if any(lenient_rels[:10]) else 0.0,
        "lenient_recall_at_20": 1.0 if any(lenient_rels[:20]) else 0.0,
        "lenient_mrr": lenient_rr,
        "lenient_ndcg_at_10": compute_ndcg_at_k(lenient_rels, k=10),
    }


def bootstrap_metric_ci(
    values: List[float], n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42
) -> Tuple[float, float, float]:
    """Calculate mean and bootstrap percentile confidence interval."""
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values)
    mean_val = float(np.mean(arr))
    if len(arr) == 1:
        return mean_val, mean_val, mean_val

    rng = np.random.default_rng(seed)
    boot_means = []
    n = len(arr)
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return mean_val, lower, upper


def paired_significance_test(
    sample_a: List[float], sample_b: List[float], n_bootstrap: int = 1000, seed: int = 42
) -> Dict[str, Any]:
    """Perform paired Wilcoxon signed-rank test and paired bootstrap difference with 95% CI."""
    arr_a = np.array(sample_a)
    arr_b = np.array(sample_b)
    diff = arr_a - arr_b
    mean_diff = float(np.mean(diff))

    rng = np.random.default_rng(seed)
    boot_diffs = [float(np.mean(rng.choice(diff, size=len(diff), replace=True))) for _ in range(n_bootstrap)]
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    if np.all(diff == 0):
        return {
            "mean_diff": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "statistic": 0.0,
            "p_value": 1.0,
        }

    try:
        res = stats.wilcoxon(arr_a, arr_b, zero_method="pratt")
        p_val = float(res.pvalue)
        stat = float(res.statistic)
    except Exception:
        p_val = 1.0
        stat = 0.0

    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "statistic": stat,
        "p_value": p_val,
    }


def calibrate_dev_similarity_threshold(
    dev_queries: List[Dict[str, Any]],
    model: SentenceTransformer,
    docstore: pd.DataFrame,
    percentile: float = 25.0,
) -> float:
    """Calibrate category similarity threshold theta_cat on the dev split only."""
    logger.info(f"Calibrating similarity threshold theta_cat strictly on {len(dev_queries)} DEV queries...")
    id_to_text = {str(r["id"]): str(r["combined_text"]) for _, r in docstore.iterrows()}

    valid_queries = [q for q in dev_queries if q["target_doc_id"] in id_to_text]
    q_texts = [q["query"] for q in valid_queries]
    t_texts = [id_to_text[q["target_doc_id"]] for q in valid_queries]

    q_emb = model.encode(q_texts, normalize_embeddings=True, convert_to_numpy=True)
    t_emb = model.encode(t_texts, normalize_embeddings=True, convert_to_numpy=True)

    sims = np.sum(q_emb * t_emb, axis=1)
    threshold = float(np.percentile(sims, percentile))
    logger.info(f"Dev calibration: target similarity p{percentile:.0f} = {threshold:.4f} (median: {np.median(sims):.4f})")
    # Clamp to reasonable range [0.30, 0.50]
    threshold = max(0.30, min(0.50, threshold))
    return threshold


def run_evaluation(
    retriever_fn: Callable[[str, int], List[Dict[str, Any]]],
    queries: List[Dict[str, Any]],
    faiss_index: faiss.Index,
    doc_id_to_idx: Dict[str, int],
    doc_categories_map: Dict[str, Set[str]],
    model: SentenceTransformer,
    similarity_threshold: float,
    top_k: int = 50,
) -> Dict[str, Any]:
    """Execute queries through a retriever and compute strict and lenient IR metrics."""
    per_query_metrics: List[Dict[str, float]] = []
    latencies: List[float] = []

    # Batch encode queries for fast cosine similarity against retrieved items
    query_texts = [q["query"] for q in queries]
    query_vecs = model.encode(query_texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    for i, item in enumerate(queries):
        query_text = item["query"]
        target_id = str(item["target_doc_id"]).strip()
        target_cats = set(str(item.get("categories", "")).split())
        q_vec = query_vecs[i]

        t0 = time.perf_counter()
        results = retriever_fn(query_text, top_k)
        t_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_ms)

        retrieved_ids = [str(r["id"]).strip() for r in results]

        # Calculate cosine similarity for retrieved items using FAISS vector reconstruction
        retrieved_sims = []
        for pid in retrieved_ids:
            idx = doc_id_to_idx.get(pid)
            if idx is not None and 0 <= idx < faiss_index.ntotal:
                cand_vec = faiss_index.reconstruct(idx)
                sim = float(np.dot(q_vec, cand_vec))
            else:
                sim = 0.0
            retrieved_sims.append(sim)

        m = compute_metrics_for_query(
            retrieved_ids=retrieved_ids,
            target_id=target_id,
            retrieved_sims=retrieved_sims,
            doc_categories_map=doc_categories_map,
            target_categories=target_cats,
            similarity_threshold=similarity_threshold,
        )
        per_query_metrics.append(m)

    aggregated: Dict[str, Any] = {}
    metric_keys = [
        "strict_recall_at_1", "strict_recall_at_5", "strict_recall_at_10", "strict_recall_at_20", "strict_mrr", "strict_ndcg_at_10",
        "lenient_recall_at_1", "lenient_recall_at_5", "lenient_recall_at_10", "lenient_recall_at_20", "lenient_mrr", "lenient_ndcg_at_10",
    ]
    for k in metric_keys:
        vals = [q[k] for q in per_query_metrics]
        mean_val, low_ci, high_ci = bootstrap_metric_ci(vals)
        aggregated[k] = {
            "mean": mean_val,
            "ci_lower": low_ci,
            "ci_upper": high_ci,
            "values": vals,
        }

    aggregated["latency_ms"] = {
        "mean": float(np.mean(latencies)),
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
    }
    return aggregated


def ablate_rrf_parameters(
    bm25: BM25Retriever,
    dense: DenseRetriever,
    dev_queries: List[Dict[str, Any]],
    faiss_index: faiss.Index,
    doc_id_to_idx: Dict[str, int],
    doc_categories_map: Dict[str, Set[str]],
    model: SentenceTransformer,
    similarity_threshold: float,
) -> Dict[str, Any]:
    """Ablate RRF k and alpha strictly on the dev split."""
    logger.info(f"Executing RRF hyperparameter grid search on {len(dev_queries)} DEV queries...")
    k_candidates = [10, 30, 60, 100]
    alpha_candidates = [0.3, 0.5, 0.7]

    results_grid: List[Dict[str, Any]] = []
    best_setting = None
    best_mrr = -1.0

    for k in k_candidates:
        for alpha in alpha_candidates:
            hybrid = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense, alpha=alpha, rrf_k=k)
            eval_res = run_evaluation(
                lambda q, top_k: hybrid.retrieve(q, top_k=top_k),
                dev_queries,
                faiss_index=faiss_index,
                doc_id_to_idx=doc_id_to_idx,
                doc_categories_map=doc_categories_map,
                model=model,
                similarity_threshold=similarity_threshold,
            )
            dev_mrr = eval_res["strict_mrr"]["mean"]
            dev_ndcg = eval_res["strict_ndcg_at_10"]["mean"]

            entry = {
                "rrf_k": k,
                "alpha": alpha,
                "dev_mrr": dev_mrr,
                "dev_ndcg": dev_ndcg,
                "dev_recall_at_10": eval_res["strict_recall_at_10"]["mean"],
            }
            results_grid.append(entry)
            logger.info(f"Dev RRF (k={k}, alpha={alpha:.1f}): Strict MRR={dev_mrr:.4f}, nDCG@10={dev_ndcg:.4f}")

            if dev_mrr > best_mrr:
                best_mrr = dev_mrr
                best_setting = entry

    assert best_setting is not None, "best_setting cannot be None"
    logger.info(f"Optimal RRF configuration on dev split: k={best_setting['rrf_k']}, alpha={best_setting['alpha']:.1f} (MRR={best_mrr:.4f})")
    return {
        "grid": results_grid,
        "best": best_setting,
    }


def generate_benchmark_markdown(
    results_by_regime: Dict[str, Dict[str, Any]],
    overall_test_results: Dict[str, Any],
    overall_significance: Dict[str, Any],
    regime_significance: Dict[str, Dict[str, Any]],
    rrf_ablation: Dict[str, Any],
    similarity_threshold: float,
    dev_count: int,
    test_count: int,
    regime_counts: Dict[str, int],
    output_path: Path = RESULTS_DIR / "metrics.md",
):
    """Write comprehensive metrics report with bootstrap CIs, regime breakdown, and significance."""
    models = ["BM25 Baseline", "Dense (MiniLM+FAISS)", "Hybrid (RRF Optimal)"]

    # Table 1: Strict Overall Held-out Test Set
    t1_rows = []
    for m in models:
        d = overall_test_results[m]
        r1 = f"{d['strict_recall_at_1']['mean']:.3f} [{d['strict_recall_at_1']['ci_lower']:.3f}, {d['strict_recall_at_1']['ci_upper']:.3f}]"
        r5 = f"{d['strict_recall_at_5']['mean']:.3f} [{d['strict_recall_at_5']['ci_lower']:.3f}, {d['strict_recall_at_5']['ci_upper']:.3f}]"
        r10 = f"{d['strict_recall_at_10']['mean']:.3f} [{d['strict_recall_at_10']['ci_lower']:.3f}, {d['strict_recall_at_10']['ci_upper']:.3f}]"
        r20 = f"{d['strict_recall_at_20']['mean']:.3f} [{d['strict_recall_at_20']['ci_lower']:.3f}, {d['strict_recall_at_20']['ci_upper']:.3f}]"
        mrr = f"{d['strict_mrr']['mean']:.3f} [{d['strict_mrr']['ci_lower']:.3f}, {d['strict_mrr']['ci_upper']:.3f}]"
        ndcg = f"{d['strict_ndcg_at_10']['mean']:.3f} [{d['strict_ndcg_at_10']['ci_lower']:.3f}, {d['strict_ndcg_at_10']['ci_upper']:.3f}]"
        lat = f"{d['latency_ms']['mean']:.1f} ms (p95: {d['latency_ms']['p95']:.1f})"
        t1_rows.append(f"| **{m}** | {r1} | {r5} | {r10} | {r20} | {mrr} | {ndcg} | {lat} |")
    table1_strict = "\n".join(t1_rows)

    # Table 2: Lenient Overall Held-out Test Set
    t2_rows = []
    for m in models:
        d = overall_test_results[m]
        r1 = f"{d['lenient_recall_at_1']['mean']:.3f} [{d['lenient_recall_at_1']['ci_lower']:.3f}, {d['lenient_recall_at_1']['ci_upper']:.3f}]"
        r5 = f"{d['lenient_recall_at_5']['mean']:.3f} [{d['lenient_recall_at_5']['ci_lower']:.3f}, {d['lenient_recall_at_5']['ci_upper']:.3f}]"
        r10 = f"{d['lenient_recall_at_10']['mean']:.3f} [{d['lenient_recall_at_10']['ci_lower']:.3f}, {d['lenient_recall_at_10']['ci_upper']:.3f}]"
        r20 = f"{d['lenient_recall_at_20']['mean']:.3f} [{d['lenient_recall_at_20']['ci_lower']:.3f}, {d['lenient_recall_at_20']['ci_upper']:.3f}]"
        mrr = f"{d['lenient_mrr']['mean']:.3f} [{d['lenient_mrr']['ci_lower']:.3f}, {d['lenient_mrr']['ci_upper']:.3f}]"
        ndcg = f"{d['lenient_ndcg_at_10']['mean']:.3f} [{d['lenient_ndcg_at_10']['ci_lower']:.3f}, {d['lenient_ndcg_at_10']['ci_upper']:.3f}]"
        t2_rows.append(f"| **{m}** | {r1} | {r5} | {r10} | {r20} | {mrr} | {ndcg} |")
    table2_lenient = "\n".join(t2_rows)

    # Table 3: Regime-by-Regime Strict & Lenient
    regime_rows = []
    for reg, reg_data in results_by_regime.items():
        n_reg = regime_counts.get(reg, 0)
        for m in models:
            d = reg_data[m]
            s_r10 = f"{d['strict_recall_at_10']['mean']:.3f} [{d['strict_recall_at_10']['ci_lower']:.3f}, {d['strict_recall_at_10']['ci_upper']:.3f}]"
            s_mrr = f"{d['strict_mrr']['mean']:.3f} [{d['strict_mrr']['ci_lower']:.3f}, {d['strict_mrr']['ci_upper']:.3f}]"
            l_r10 = f"{d['lenient_recall_at_10']['mean']:.3f} [{d['lenient_recall_at_10']['ci_lower']:.3f}, {d['lenient_recall_at_10']['ci_upper']:.3f}]"
            l_mrr = f"{d['lenient_mrr']['mean']:.3f} [{d['lenient_mrr']['ci_lower']:.3f}, {d['lenient_mrr']['ci_upper']:.3f}]"
            regime_rows.append(f"| {reg.capitalize()} (n={n_reg}) | {m} | {s_r10} | {s_mrr} | {l_r10} | {l_mrr} |")
    table3_regimes = "\n".join(regime_rows)

    # Table 4: Per-Regime Paired Significance
    sig_rows = []
    for reg, s in regime_significance.items():
        n_reg = regime_counts.get(reg, 0)
        sig_rows.append(
            f"| **{reg.capitalize()}** (n={n_reg}) | {s['best_single']} | {s['mean_diff']:+.4f} [{s['ci_lower']:+.4f}, {s['ci_upper']:+.4f}] | W = {s['statistic']:.1f} | p = {s['p_value']:.4f} |"
        )
    table4_sig = "\n".join(sig_rows)

    # Table 5: RRF Dev Ablation
    ablation_rows = []
    for row in rrf_ablation["grid"]:
        ablation_rows.append(
            f"| k={row['rrf_k']} | alpha={row['alpha']:.1f} | {row['dev_mrr']:.4f} | {row['dev_ndcg']:.4f} | {row['dev_recall_at_10']:.4f} |"
        )
    table5_ablation = "\n".join(ablation_rows)

    best_k = rrf_ablation["best"]["rrf_k"]
    best_alpha = rrf_ablation["best"]["alpha"]

    content = f"""# Retrieval Benchmark Results: arXiv Abstracts (20k Sample)

Quantitative comparison of retrieval models evaluated on an overhauled **450-query multi-regime benchmark** ({dev_count} dev queries for tuning, {test_count} held-out test queries with n={regime_counts['lexical']} per regime). Ground truth relevance is strictly separated between tuning and testing.

All metrics are reported on the **held-out test split** with **95% bootstrap confidence intervals** (1,000 resamples).
- **Strict Relevance**: Evaluates exact retrieval of the designated target document.
- **Lenient Relevance**: Treats a document as relevant if it is the target OR shares the target's arXiv category with dense cosine similarity >= {similarity_threshold:.4f} (calibrated on DEV split).

---

## 1. Overall Held-Out Test Set Performance (N={test_count}, Strict Target Ground Truth)

| Retriever Architecture | Recall@1 [95% CI] | Recall@5 [95% CI] | Recall@10 [95% CI] | Recall@20 [95% CI] | MRR [95% CI] | nDCG@10 [95% CI] | Latency (Mean, p95) |
|:-----------------------|:-----------------:|:-----------------:|:------------------:|:------------------:|:------------:|:----------------:|:-------------------:|
{table1_strict}

### Overall Paired Significance Testing (Hybrid vs. Best Single Retriever: {overall_significance['best_single']})
- **Mean Pairwise Difference**: {overall_significance['mean_diff']:+.4f} [{overall_significance['ci_lower']:+.4f}, {overall_significance['ci_upper']:+.4f}]
- **Wilcoxon Signed-Rank Test**: W = {overall_significance['statistic']:.1f}, **p = {overall_significance['p_value']:.4f}**

---

## 2. Overall Held-Out Test Set Performance (N={test_count}, Lenient Category Multi-Relevant)

| Retriever Architecture | Lenient Recall@1 [95% CI] | Lenient Recall@5 [95% CI] | Lenient Recall@10 [95% CI] | Lenient Recall@20 [95% CI] | Lenient MRR [95% CI] | Lenient nDCG@10 [95% CI] |
|:-----------------------|:-------------------------:|:-------------------------:|:--------------------------:|:--------------------------:|:--------------------:|:------------------------:|
{table2_lenient}

---

## 3. Regime-by-Regime Breakdown (Strict vs Lenient, N={test_count})

| Query Regime | Retriever Architecture | Strict Recall@10 [95% CI] | Strict MRR [95% CI] | Lenient Recall@10 [95% CI] | Lenient MRR [95% CI] |
|:-------------|:-----------------------|:-------------------------:|:-------------------:|:--------------------------:|:--------------------:|
{table3_regimes}

---

## 4. Paired Significance Testing by Regime (Hybrid vs Best Single Retriever)

| Query Regime | Best Single Baseline | Mean MRR Diff (`Hybrid - Baseline`) [95% CI] | Wilcoxon W | p-value |
|:---|:---|:---:|:---:|:---:|
{table4_sig}

---

## 5. RRF Hyperparameter Tuning (Dev Split Ablation, N={dev_count})

Grid search across RRF rank smoothing constant $k$ and dense weighting factor $\\alpha$ conducted strictly on the dev split:

| RRF Parameter k | Weight Ratio $\\alpha$ | Dev MRR | Dev nDCG@10 | Dev Recall@10 |
|:---|:---:|:---:|:---:|:---:|
{table5_ablation}

**Optimal Dev Setting**: $k={best_k}$, $\\alpha={best_alpha:.1f}$.
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Saved benchmark metrics report to {output_path}")


def plot_benchmark_charts(
    overall_test_results: Dict[str, Any],
    results_by_regime: Dict[str, Dict[str, Any]],
    regime_counts: Dict[str, int],
    test_count: int,
    output_path: Path = RESULTS_DIR / "metrics.png",
):
    """Generate side-by-side visualization with bootstrap error bars that do not cover labels."""
    models = ["BM25 Baseline", "Dense (MiniLM+FAISS)", "Hybrid (RRF Optimal)"]
    short_models = ["BM25", "Dense (MiniLM)", "Hybrid RRF"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Overall Metrics with Bootstrap CIs
    metrics = ["strict_recall_at_1", "strict_recall_at_5", "strict_recall_at_10", "strict_recall_at_20", "strict_mrr", "strict_ndcg_at_10"]
    metric_labels = ["R@1", "R@5", "R@10", "R@20", "MRR", "nDCG@10"]
    x = np.arange(len(metric_labels))
    width = 0.25

    for idx, (m_name, s_name) in enumerate(zip(models, short_models)):
        means = [overall_test_results[m_name][m]["mean"] for m in metrics]
        lows = [overall_test_results[m_name][m]["ci_lower"] for m in metrics]
        highs = [overall_test_results[m_name][m]["ci_upper"] for m in metrics]
        yerr = [
            np.array(means) - np.array(lows),
            np.array(highs) - np.array(means),
        ]

        rects = axes[0].bar(
            x + (idx - 1) * width,
            means,
            width,
            yerr=yerr,
            capsize=3,
            label=s_name,
            color=colors[idx],
            edgecolor="black",
            linewidth=0.6,
        )
        for i, rect in enumerate(rects):
            val = rect.get_height()
            err_top = highs[i]
            # Place label cleanly above error bar
            axes[0].annotate(
                f"{val:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, err_top),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
            )

    axes[0].set_title(f"Overall Held-Out Test Set (N={test_count}, 95% Bootstrap CIs)", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_labels, fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Score", fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, 1.15)
    axes[0].legend(loc="upper left", frameon=True, fontsize=9)

    # Panel 2: MRR by Query Regime with Bootstrap CIs
    regimes = ["lexical", "paraphrase", "acronym"]
    regime_labels = [f"Lexical\n(n={regime_counts.get('lexical', 100)})", f"Paraphrase\n(n={regime_counts.get('paraphrase', 100)})", f"Acronym\n(n={regime_counts.get('acronym', 100)})"]
    rx = np.arange(len(regimes))

    for idx, (m_name, s_name) in enumerate(zip(models, short_models)):
        mrr_means = [results_by_regime[r][m_name]["strict_mrr"]["mean"] for r in regimes]
        mrr_lows = [results_by_regime[r][m_name]["strict_mrr"]["ci_lower"] for r in regimes]
        mrr_highs = [results_by_regime[r][m_name]["strict_mrr"]["ci_upper"] for r in regimes]
        yerr = [
            np.array(mrr_means) - np.array(mrr_lows),
            np.array(mrr_highs) - np.array(mrr_means),
        ]

        r_rects = axes[1].bar(
            rx + (idx - 1) * width,
            mrr_means,
            width,
            yerr=yerr,
            capsize=3,
            label=s_name,
            color=colors[idx],
            edgecolor="black",
            linewidth=0.6,
        )
        for i, rect in enumerate(r_rects):
            val = rect.get_height()
            err_top = mrr_highs[i]
            axes[1].annotate(
                f"{val:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, err_top),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
            )

    axes[1].set_title("MRR by Query Regime (Held-Out Test, 95% Bootstrap CIs)", fontsize=11, fontweight="bold")
    axes[1].set_xticks(rx)
    axes[1].set_xticklabels(regime_labels, fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Strict MRR", fontsize=10, fontweight="bold")
    axes[1].set_ylim(0, 1.15)
    axes[1].legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved benchmark figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-regime retrieval benchmark.")
    parser.add_argument("--benchmark-file", type=str, default=str(BENCHMARK_PATH), help="Benchmark JSON path")
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark_file)
    logger.info(f"Loading benchmark from {benchmark_path}...")
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_queries = json.load(f)

    dev_queries = [q for q in benchmark_queries if q.get("split") == "dev"]
    test_queries = [q for q in benchmark_queries if q.get("split") == "test"]
    logger.info(f"Loaded {len(benchmark_queries)} queries: {len(dev_queries)} DEV, {len(test_queries)} TEST")

    # Load corpus docstore and FAISS index
    logger.info(f"Loading docstore from {DOCSTORE_PATH}...")
    docstore = pd.read_parquet(DOCSTORE_PATH)
    logger.info(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
    faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))

    doc_id_to_idx = {str(pid): i for i, pid in enumerate(docstore["id"])}
    doc_categories_map = {str(r["id"]): set(str(r["categories"]).split()) for _, r in docstore.iterrows()}

    # Initialize retrievers
    logger.info("Initializing retrievers...")
    bm25 = BM25Retriever(docstore=docstore)
    dense = DenseRetriever(index=faiss_index, docstore=docstore)
    model = dense.model

    # Step 1: Calibrate similarity threshold theta_cat strictly on DEV split
    similarity_threshold = calibrate_dev_similarity_threshold(dev_queries, model, docstore, percentile=25.0)

    # Step 2: Ablate RRF on DEV split
    rrf_ablation = ablate_rrf_parameters(
        bm25=bm25,
        dense=dense,
        dev_queries=dev_queries,
        faiss_index=faiss_index,
        doc_id_to_idx=doc_id_to_idx,
        doc_categories_map=doc_categories_map,
        model=model,
        similarity_threshold=similarity_threshold,
    )
    best_k = rrf_ablation["best"]["rrf_k"]
    best_alpha = rrf_ablation["best"]["alpha"]

    optimal_hybrid = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense, alpha=best_alpha, rrf_k=best_k)

    # Helper evaluation closure
    def eval_model(retriever_call, q_list):
        return run_evaluation(
            retriever_call,
            q_list,
            faiss_index=faiss_index,
            doc_id_to_idx=doc_id_to_idx,
            doc_categories_map=doc_categories_map,
            model=model,
            similarity_threshold=similarity_threshold,
        )

    # Step 3: Evaluate on HELD-OUT TEST SPLIT (N=300)
    logger.info(f"Evaluating BM25 on held-out test split (N={len(test_queries)})...")
    bm25_test = eval_model(lambda q, k: bm25.retrieve(q, top_k=k), test_queries)

    logger.info(f"Evaluating Dense on held-out test split (N={len(test_queries)})...")
    dense_test = eval_model(lambda q, k: dense.retrieve(q, top_k=k), test_queries)

    logger.info(f"Evaluating Hybrid (RRF Optimal) on held-out test split (N={len(test_queries)})...")
    hybrid_test = eval_model(lambda q, k: optimal_hybrid.retrieve(q, top_k=k), test_queries)

    overall_test_results = {
        "BM25 Baseline": bm25_test,
        "Dense (MiniLM+FAISS)": dense_test,
        "Hybrid (RRF Optimal)": hybrid_test,
    }

    # Step 4: Overall Significance Test
    best_single_name = "BM25 Baseline" if bm25_test["strict_mrr"]["mean"] >= dense_test["strict_mrr"]["mean"] else "Dense (MiniLM+FAISS)"
    best_single_vals = bm25_test["strict_mrr"]["values"] if best_single_name == "BM25 Baseline" else dense_test["strict_mrr"]["values"]
    overall_sig = paired_significance_test(hybrid_test["strict_mrr"]["values"], best_single_vals)
    overall_sig["best_single"] = best_single_name

    # Step 5: Regime Breakdown and Paired Significance per Regime
    results_by_regime: Dict[str, Dict[str, Any]] = {}
    regime_significance: Dict[str, Dict[str, Any]] = {}
    regime_counts: Dict[str, int] = {}

    for regime in ["lexical", "paraphrase", "acronym"]:
        reg_queries = [q for q in test_queries if q.get("regime") == regime]
        regime_counts[regime] = len(reg_queries)
        logger.info(f"Evaluating regime '{regime}' on test set (N={len(reg_queries)})...")

        bm25_r = eval_model(lambda q, k: bm25.retrieve(q, top_k=k), reg_queries)
        dense_r = eval_model(lambda q, k: dense.retrieve(q, top_k=k), reg_queries)
        hybrid_r = eval_model(lambda q, k: optimal_hybrid.retrieve(q, top_k=k), reg_queries)

        results_by_regime[regime] = {
            "BM25 Baseline": bm25_r,
            "Dense (MiniLM+FAISS)": dense_r,
            "Hybrid (RRF Optimal)": hybrid_r,
        }

        # Per-regime significance
        best_single_reg = "BM25 Baseline" if bm25_r["strict_mrr"]["mean"] >= dense_r["strict_mrr"]["mean"] else "Dense (MiniLM+FAISS)"
        best_single_reg_vals = bm25_r["strict_mrr"]["values"] if best_single_reg == "BM25 Baseline" else dense_r["strict_mrr"]["values"]
        sig_r = paired_significance_test(hybrid_r["strict_mrr"]["values"], best_single_reg_vals)
        sig_r["best_single"] = best_single_reg
        regime_significance[regime] = sig_r
        logger.info(f"Regime '{regime}' Paired test (Hybrid vs {best_single_reg}): mean_diff={sig_r['mean_diff']:+.4f}, p={sig_r['p_value']:.4f}")

    # Step 6: Save reports and generate charts
    generate_benchmark_markdown(
        results_by_regime=results_by_regime,
        overall_test_results=overall_test_results,
        overall_significance=overall_sig,
        regime_significance=regime_significance,
        rrf_ablation=rrf_ablation,
        similarity_threshold=similarity_threshold,
        dev_count=len(dev_queries),
        test_count=len(test_queries),
        regime_counts=regime_counts,
    )
    plot_benchmark_charts(
        overall_test_results=overall_test_results,
        results_by_regime=results_by_regime,
        regime_counts=regime_counts,
        test_count=len(test_queries),
    )
    logger.info("Evaluation complete! Benchmark reports and plots generated in results/")


if __name__ == "__main__":
    main()
