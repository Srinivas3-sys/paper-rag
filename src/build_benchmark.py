"""Construct a rigorous, leakage-free 450-query benchmark across three distinct query regimes:
1. Lexical queries (natural term overlap from titles and key abstract phrases)
2. Paraphrase queries (specific technical queries with zero title words and zero n-gram leakage)
3. Acronym / exact-term queries (domain-specific acronyms, named algorithms, and theorems)

Includes programmatic n-gram leakage auditing and 50 dev / 100 test split partitioning per regime.
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import BENCHMARK_PATH, LEAKAGE_REPORT_PATH, RAW_DATA_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
    "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves", "study", "paper", "author", "authors", "propose", "show", "investigate",
}

DEGENERATE_PATTERNS = [
    r"\bmain problem\b",
    r"\bcore problem\b",
    r"\bbasic idea\b",
    r"\bsolution to\b",
    r"\bthe problem\b",
    r"\bthis paper\b",
    r"\bwhat is the problem\b",
    r"\bhow to solve the problem\b",
    r"\bwhat is the core\b",
    r"\bwhat is the main\b",
]


def tokenize_words(text: str) -> List[str]:
    """Extract lowercased alphanumeric words."""
    return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", text)]


def get_ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    """Compute n-grams from a list of tokens."""
    if len(tokens) < n:
        return set()
    return set(zip(*[tokens[i:] for i in range(n)]))


def calculate_leakage(query: str, target_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Compute quantitative overlap metrics between a query and its target document."""
    query_tokens = [w for w in tokenize_words(query) if w not in STOP_WORDS]
    title_tokens = [w for w in tokenize_words(target_doc["title"]) if w not in STOP_WORDS]

    first_sentence = target_doc["abstract"].split(".")[0] if "." in target_doc["abstract"] else target_doc["abstract"]
    first_sent_tokens = [w for w in tokenize_words(first_sentence) if w not in STOP_WORDS]

    abstract_tokens = [w for w in tokenize_words(target_doc["abstract"]) if w not in STOP_WORDS]
    all_doc_tokens = [w for w in tokenize_words(target_doc["combined_text"]) if w not in STOP_WORDS]

    title_overlap = set(query_tokens).intersection(set(title_tokens))
    first_sent_overlap = set(query_tokens).intersection(set(first_sent_tokens))

    q_bigrams = get_ngrams(query_tokens, 2)
    doc_bigrams = get_ngrams(all_doc_tokens, 2)
    bigram_overlap = q_bigrams.intersection(doc_bigrams)

    q_trigrams = get_ngrams(query_tokens, 3)
    doc_trigrams = get_ngrams(all_doc_tokens, 3)
    trigram_overlap = q_trigrams.intersection(doc_trigrams)

    unigram_doc_overlap = set(query_tokens).intersection(set(abstract_tokens))

    return {
        "query_token_count": len(query_tokens),
        "title_overlap_words": list(title_overlap),
        "first_sent_overlap_words": list(first_sent_overlap),
        "bigram_overlap_count": len(bigram_overlap),
        "bigram_overlap": [" ".join(bg) for bg in bigram_overlap],
        "trigram_overlap_count": len(trigram_overlap),
        "unigram_overlap_fraction": len(unigram_doc_overlap) / max(len(query_tokens), 1),
    }


def build_lexical_queries(df: pd.DataFrame, n: int = 150, dev_count: int = 50, seed: int = 42) -> List[Dict[str, Any]]:
    """Build Set A: Lexical queries with natural term overlap from paper titles and abstracts."""
    logger.info(f"Building {n} Lexical queries ({dev_count} dev, {n - dev_count} test)...")
    sample = df.sample(n=n, random_state=seed)
    queries = []
    for i, row in enumerate(sample.itertuples()):
        title = row.title
        words = [w for w in tokenize_words(title) if w not in STOP_WORDS]
        query_text = " ".join(words[:7])

        split = "dev" if i < dev_count else "test"
        queries.append({
            "query_id": f"lex_{i:04d}",
            "regime": "lexical",
            "split": split,
            "query": query_text,
            "target_doc_id": str(row.id),
            "title": title,
            "abstract_snippet": row.abstract[:200],
            "categories": str(row.categories),
        })
    return queries


def build_acronym_queries(df: pd.DataFrame, n: int = 150, dev_count: int = 50, seed: int = 43) -> List[Dict[str, Any]]:
    """Build Set C: Acronym / exact-term queries centered on rare identifiers, acronyms, and theorems."""
    logger.info(f"Scanning corpus for {n} papers with distinctive acronyms and named theorems...")
    candidates = []

    noise_caps = {
        "THE", "AND", "FOR", "WITH", "NOT", "ALL", "CAN", "ARE", "BUT", "FROM", "THIS", "THAT",
        "INTO", "OVER", "NEW", "TWO", "ONE", "VIA", "HOW", "NON", "WELL", "SOME", "MORE", "LESS",
    }

    for row in df.itertuples():
        found = set(re.findall(r"\b[A-Z]{2,6}\b", row.title + " " + row.abstract))
        acrs = [a for a in found if a not in noise_caps]
        if len(acrs) >= 1:
            candidates.append((row, acrs))

    logger.info(f"Found {len(candidates)} candidate papers with technical acronyms.")

    import random
    random.seed(seed)
    selected_indices = random.sample(range(len(candidates)), min(n, len(candidates)))

    queries = []
    for i, idx in enumerate(selected_indices):
        row, acrs = candidates[idx]
        primary_acronym = acrs[0]

        title_words = [w for w in tokenize_words(row.title) if w not in STOP_WORDS and w.lower() != primary_acronym.lower()]
        context_word = title_words[0] if title_words else "analysis"

        query_text = f"{primary_acronym} {context_word} methods and properties"
        split = "dev" if i < dev_count else "test"

        queries.append({
            "query_id": f"acro_{i:04d}",
            "regime": "acronym",
            "split": split,
            "query": query_text,
            "target_doc_id": str(row.id),
            "title": row.title,
            "abstract_snippet": row.abstract[:200],
            "acronym": primary_acronym,
            "categories": str(row.categories),
        })
    return queries


def clean_paraphrase_query(text: str) -> str:
    """Strip conversational conversational boilerplate from model output."""
    clean_q = re.sub(
        r"^(we\s+study|we\s+investigate|we\s+have\s+studied|we\s+present|the\s+purpose\s+of\s+this\s+paper\s+is\s+to|in\s+this\s+paper\s+we)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    clean_q = re.sub(
        r"^(a\s+description\s+of|an\s+investigation\s+into|a\s+study\s+of|an\s+analysis\s+of)\s+",
        "",
        clean_q,
        flags=re.IGNORECASE,
    )
    clean_q = re.sub(r"[\"?]", "", clean_q)
    clean_q = re.sub(r"\s+", " ", clean_q).strip()
    return clean_q


def build_paraphrase_queries(
    df: pd.DataFrame,
    n: int = 150,
    dev_count: int = 50,
    seed: int = 44,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """Build Set B: Specific technical paraphrase queries with strictly zero n-gram leakage."""
    logger.info(f"Loading local generator model for paraphrase generation on {device}...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)
    model.eval()

    sample_pool = df.sample(n=min(n * 20, len(df)), random_state=seed).copy()
    queries: List[Dict[str, Any]] = []

    logger.info(f"Generating {n} specific paraphrase queries in batches of {batch_size}...")
    pbar = tqdm(total=n, desc="Generating Paraphrase Queries")

    rows = list(sample_pool.itertuples())
    for batch_start in range(0, len(rows), batch_size):
        if len(queries) >= n:
            break
        batch_rows = rows[batch_start: batch_start + batch_size]
        prompts = []
        valid_batch_items = []

        for row in batch_rows:
            title = row.title
            abstract = row.abstract

            title_words = [w for w in tokenize_words(title) if w not in STOP_WORDS and len(w) > 3]
            forbidden_str = ", ".join(title_words)

            sentences = [s.strip() for s in abstract.split(".") if len(s.strip()) > 20]
            context_snippet = ". ".join(sentences[1:3]) if len(sentences) >= 3 else abstract[:350]

            prompt = (
                f"Generate a technical search question for the following research topic. "
                f"Do not use these words: {forbidden_str}.\n\n"
                f"Topic: {context_snippet[:350]}\n\n"
                f"Question:"
            )
            prompts.append(prompt)
            valid_batch_items.append(row)

        inputs = tokenizer(prompts, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for row, raw_query in zip(valid_batch_items, decoded):
            if len(queries) >= n:
                break
            clean_q = clean_paraphrase_query(raw_query)

            words = clean_q.split()
            if len(words) < 5 or len(words) > 30:
                continue

            # Reject prompt boilerplate parroting
            if any(re.search(pat, clean_q, re.I) for pat in DEGENERATE_PATTERNS):
                continue

            leakage = calculate_leakage(clean_q, {"title": row.title, "abstract": row.abstract, "combined_text": row.combined_text})
            if leakage["bigram_overlap_count"] == 0 and len(leakage["title_overlap_words"]) == 0:
                split = "dev" if len(queries) < dev_count else "test"
                queries.append({
                    "query_id": f"para_{len(queries):04d}",
                    "regime": "paraphrase",
                    "split": split,
                    "query": clean_q,
                    "target_doc_id": str(row.id),
                    "title": row.title,
                    "abstract_snippet": row.abstract[:200],
                    "categories": str(row.categories),
                    "leakage_audit": leakage,
                })
                pbar.update(1)

    pbar.close()
    logger.info(f"Successfully generated {len(queries)} specific zero-leakage paraphrase queries.")
    return queries


def main():
    parser = argparse.ArgumentParser(description="Build 450-query multi-regime evaluation benchmark.")
    parser.add_argument("--queries-per-set", type=int, default=150, help="Total queries per regime (50 dev, 100 test)")
    parser.add_argument("--dev-count", type=int, default=50, help="Dev queries per regime")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=str(BENCHMARK_PATH), help="Output benchmark path")
    args = parser.parse_args()

    logger.info(f"Loading raw arXiv dataset from {RAW_DATA_PATH}...")
    df = pd.read_parquet(RAW_DATA_PATH)

    n_per_set = args.queries_per_set
    dev_count = args.dev_count
    lexical_queries = build_lexical_queries(df, n=n_per_set, dev_count=dev_count, seed=args.seed)
    acronym_queries = build_acronym_queries(df, n=n_per_set, dev_count=dev_count, seed=args.seed + 1)
    paraphrase_queries = build_paraphrase_queries(df, n=n_per_set, dev_count=dev_count, seed=args.seed + 2)

    all_queries = lexical_queries + paraphrase_queries + acronym_queries
    logger.info(f"Total benchmark queries created: {len(all_queries)}")

    leakage_records = []
    for item in all_queries:
        target_row = df[df["id"] == item["target_doc_id"]].iloc[0]
        audit = calculate_leakage(
            item["query"],
            {"title": target_row["title"], "abstract": target_row["abstract"], "combined_text": target_row["combined_text"]}
        )
        leakage_records.append({
            "query_id": item["query_id"],
            "regime": item["regime"],
            "split": item["split"],
            "query": item["query"],
            "audit": audit,
        })

    summary = {
        "total_queries": len(all_queries),
        "regimes": {
            "lexical": len(lexical_queries),
            "paraphrase": len(paraphrase_queries),
            "acronym": len(acronym_queries),
        },
        "dev_count": sum(1 for q in all_queries if q["split"] == "dev"),
        "test_count": sum(1 for q in all_queries if q["split"] == "test"),
        "paraphrase_mean_bigram_overlap": float(pd.Series([
            r["audit"]["bigram_overlap_count"] for r in leakage_records if r["regime"] == "paraphrase"
        ]).mean()),
        "paraphrase_mean_title_overlap": float(pd.Series([
            len(r["audit"]["title_overlap_words"]) for r in leakage_records if r["regime"] == "paraphrase"
        ]).mean()),
        "records": leakage_records,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_queries, f, indent=2)

    with open(LEAKAGE_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Benchmark saved to {output_path}")
    logger.info(f"Leakage audit report saved to {LEAKAGE_REPORT_PATH}")
    logger.info(f"Summary: dev={summary['dev_count']}, test={summary['test_count']}")
    logger.info(f"Paraphrase Bigram Overlap Mean: {summary['paraphrase_mean_bigram_overlap']:.4f}")
    logger.info(f"Paraphrase Title Overlap Mean: {summary['paraphrase_mean_title_overlap']:.4f}")


if __name__ == "__main__":
    main()
